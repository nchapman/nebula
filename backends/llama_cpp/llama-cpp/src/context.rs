//! Safe wrapper around `llama_context`.

use std::fmt::{Debug, Formatter};
use std::num::NonZeroI32;
use std::ops::RangeBounds;
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;

use crate::clip::ImageEmbed;
use crate::llama_batch::LlamaBatch;
use crate::model::{AddBos, LlamaModel};
use crate::timing::LlamaTimings;
use crate::token::data::LlamaTokenData;
use crate::token::LlamaToken;
use crate::{DecodeError, EmbeddingsError, PredictError};

pub mod kv_cache;
pub mod params;
pub mod sample;
pub mod session;

#[allow(clippy::module_name_repetitions)]
pub struct LlamaContextInternal {
    pub(crate) context: NonNull<llama_cpp_sys::llama_context>,
}

unsafe impl Send for LlamaContextInternal {}
unsafe impl Sync for LlamaContextInternal {}

impl Drop for LlamaContextInternal {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys::llama_free(self.context.as_ptr()) }
    }
}

/// Safe wrapper around `llama_context`.
#[allow(clippy::module_name_repetitions)]
pub struct LlamaContext {
    pub(crate) context: Arc<LlamaContextInternal>,
    /// a reference to the contexts model.
    pub model: LlamaModel,
    initialized_logits: Vec<i32>,
    embeddings_enabled: bool,
}

impl Debug for LlamaContext {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaContext")
            .field("context", &self.context.context)
            .finish()
    }
}

impl LlamaContext {
    pub(crate) fn new(
        llama_model: &LlamaModel,
        llama_context: NonNull<llama_cpp_sys::llama_context>,
        embeddings_enabled: bool,
    ) -> Self {
        Self {
            context: Arc::new(LlamaContextInternal {
                context: llama_context,
            }),
            model: llama_model.clone(),
            initialized_logits: Vec::new(),
            embeddings_enabled,
        }
    }

    /// Gets the max number of tokens in a batch.
    #[must_use]
    pub fn n_batch(&self) -> u32 {
        unsafe { llama_cpp_sys::llama_n_batch(self.context.context.as_ptr()) }
    }

    /// Gets the size of the context.
    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_cpp_sys::llama_n_ctx(self.context.context.as_ptr()) }
    }

    /// Decodes the batch.
    ///
    /// # Errors
    ///
    /// - `DecodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), DecodeError> {
        let result = unsafe {
            llama_cpp_sys::llama_decode(self.context.context.as_ptr(), batch.llama_batch)
        };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits = batch.initialized_logits.clone();
                Ok(())
            }
            Some(error) => Err(DecodeError::from(error)),
        }
    }

    /// Get the embeddings for the `i`th sequence in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch.
    /// The size corresponds to the `n_embd` parameter of the context's model.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - If the current model had a pooling type of [`llama_cpp_sys::LLAMA_POOLING_TYPE_NONE`]
    /// - If the given sequence index exceeds the max sequence id.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn embeddings_seq_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let embedding =
                llama_cpp_sys::llama_get_embeddings_seq(self.context.context.as_ptr(), i);

            // Technically also possible whenever `i >= max(batch.n_seq)`, but can't check that here.
            if embedding.is_null() {
                Err(EmbeddingsError::NonePoolType)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// Get the embeddings for the `i`th token in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch of the given token.
    /// The size corresponds to the `n_embd` parameter of the context's model.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - When the given token didn't have logits enabled when it was passed.
    /// - If the given token index exceeds the max token id.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn embeddings_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let embedding =
                llama_cpp_sys::llama_get_embeddings_ith(self.context.context.as_ptr(), i);
            // Technically also possible whenever `i >= batch.n_tokens`, but no good way of checking `n_tokens` here.
            if embedding.is_null() {
                Err(EmbeddingsError::LogitsNotEnabled)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - logit `i` is not initialized.
    pub fn candidates_ith(&self, i: i32) -> impl Iterator<Item = LlamaTokenData> + '_ {
        (0_i32..).zip(self.get_logits_ith(i)).map(|(i, logit)| {
            let token = LlamaToken::new(i);
            LlamaTokenData::new(token, *logit, 0_f32)
        })
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - `i` is greater than `n_ctx`
    /// - `n_vocab` does not fit into a usize
    /// - logit `i` is not initialized.
    #[must_use]
    pub fn get_logits_ith(&self, i: i32) -> &[f32] {
        assert!(
            self.initialized_logits.contains(&i),
            "logit {i} is not initialized. only {:?} is",
            self.initialized_logits
        );
        assert!(
            self.n_ctx() > u32::try_from(i).expect("i does not fit into a u32"),
            "n_ctx ({}) must be greater than i ({})",
            self.n_ctx(),
            i
        );

        let data = unsafe { llama_cpp_sys::llama_get_logits_ith(self.context.context.as_ptr(), i) };
        let len = usize::try_from(self.model.n_vocab()).expect("n_vocab does not fit into a usize");

        unsafe { slice::from_raw_parts(data, len) }
    }

    /// Reset the timings for the context.
    pub fn reset_timings(&mut self) {
        unsafe { llama_cpp_sys::llama_reset_timings(self.context.context.as_ptr()) }
    }

    /// Returns the timings for the context.
    pub fn timings(&mut self) -> LlamaTimings {
        let timings = unsafe { llama_cpp_sys::llama_get_timings(self.context.context.as_ptr()) };
        LlamaTimings { timings }
    }

    pub fn eval_string(
        &mut self,
        string: &str,
        batch: usize,
        add_bos: AddBos,
        n_curr: &mut i32,
    ) -> Result<i32, DecodeError> {
        let tokens = self.model.str_to_token(string, add_bos)?;
        tokens.chunks(batch).into_iter().try_fold(0, |_acc, ch| {
            let mut batch = LlamaBatch::new(batch, 1);
            let last_index = ch.len() - 1;
            ch.into_iter().enumerate().try_for_each(|(i, t)| {
                batch.add(t.clone(), *n_curr, &[0], i == last_index)?;
                *n_curr += 1;
                Ok::<(), DecodeError>(())
            })?;
            self.decode(&mut batch)?;
            Ok::<_, DecodeError>(batch.n_tokens() - 1)
        })
    }

    pub fn eval_embed_image(
        &mut self,
        tokens: ImageEmbed,
        batch: usize,
        n_curr: &mut i32,
    ) -> Result<i32, DecodeError> {
        let res = unsafe {
            llama_cpp_sys::llava_eval_image_embed(
                self.context.context.as_ptr(),
                tokens.embed.as_ptr(),
                batch as i32,
                n_curr,
            )
        };
        if !res {
            Err(DecodeError::EvalEmbedImage)
        } else {
            Ok(0)
        }
    }

    pub fn pedict(
        &mut self,
        mut logit: i32,
        n_curr: &mut i32,
        n_len: Option<usize>,
        stop_tokens: &[String],
        token_callback: Box<dyn Fn(String) -> bool + Send + 'static>,
    ) -> Result<i32, PredictError> {
        let mut batch = LlamaBatch::new(2048, 1);
        let max_stop_len = stop_tokens.iter().map(|s| s.len()).max().unwrap();
        log::trace!("{}", max_stop_len);
        let mut buffer = TokenBuf::new();
        let mut count = 0;
        loop {
            if let Some(nn_len) = n_len {
                if count >= nn_len {
                    break;
                }
            }
            {
                let candidates = self.candidates_ith(logit);
                let candidates_p =
                    crate::token::data_array::LlamaTokenDataArray::from_iter(candidates, false);
                let new_token_id = self.sample_token_greedy(candidates_p);
                if new_token_id == self.model.token_eos() {
                    for t in buffer.drain(..).into_iter() {
                        *n_curr = t.1;
                        if !token_callback(t.0) {
                            return Ok(0);
                        }
                    }
                    return Ok(0);
                }
                let ntr = self.model.token_to_str(new_token_id)?;
                log::trace!("{:?} ", ntr);
                buffer.add(ntr, *n_curr);
                if let Some(s) = buffer.find(stop_tokens) {
                    for t in buffer.drain(..s).into_iter() {
                        *n_curr = t.1;
                        if !token_callback(t.0) {
                            return Ok(0);
                        }
                    }
                    return Ok(0);
                }
                if buffer.len() >= max_stop_len {
                    for t in buffer.drain(..1).into_iter() {
                        if !token_callback(t.0) {
                            *n_curr = t.1;
                            return Ok(0);
                        }
                    }
                }
                batch.clear();
                batch.add(new_token_id, *n_curr, &[0], true)?;
            }
            *n_curr += 1;
            self.decode(&mut batch)?;
            logit = 0;
            count += 1;
        }
        Ok(logit)
    }
}

struct TokenBuf {
    tokens: Vec<(String, i32)>,
    len: usize,
}

impl TokenBuf {
    pub fn new() -> Self {
        Self {
            tokens: vec![],
            len: 0,
        }
    }

    pub fn add(&mut self, token: String, n: i32) {
        self.len += token.len();
        self.tokens.push((token, n));
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn find(&self, data: &[String]) -> Option<usize> {
        let mut acc = "".to_string();
        for i in (0..self.tokens.len()).rev() {
            acc = self.tokens[i].0.clone() + &acc;
            if data.iter().any(|d| acc.contains(d)) {
                return Some(i);
            }
        }
        None
    }

    pub fn drain(&mut self, range: impl RangeBounds<usize>) -> Vec<(String, i32)> {
        let res: Vec<(String, i32)> = self.tokens.drain(range).collect();
        let res_len = res.iter().fold(0, |mut acc, (s, _)| {
            acc += s.len();
            acc
        });
        self.len -= res_len;
        res
    }
}
