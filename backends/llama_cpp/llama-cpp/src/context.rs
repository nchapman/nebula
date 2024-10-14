//! Safe wrapper around `llama_context`.

use std::fmt::{Debug, Formatter};
use std::num::NonZeroI32;
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;

use crate::clip::ImageEmbed;
use crate::context::params::LlamaContextParams;
use crate::llama_batch::LlamaBatch;
use crate::model::{AddBos, LlamaModel};
use crate::token::data::LlamaTokenData;
use crate::token::LlamaToken;
use crate::LlamaContextLoadError;
use crate::{DecodeError, EmbeddingsError};

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

#[derive(Debug)]
pub struct ClipImageU8 {
    img: NonNull<llama_cpp_sys::clip_image_u8>,
}

impl Drop for ClipImageU8 {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys::clip_image_u8_free(self.img.as_ptr()) }
    }
}

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
    pub(crate) fn new(llama_model: &LlamaModel, params: LlamaContextParams) -> crate::Result<Self> {
        let context_params = params.context_params;
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        let guard = stdio_override::StderrOverride::from_file("/dev/null").unwrap();
        #[cfg(target_os = "windows")]
        let guard = stdio_override::StderrOverride::from_file("nul").unwrap();
        let context = unsafe {
            llama_cpp_sys::llama_new_context_with_model(
                llama_model.model.model.as_ptr(),
                context_params,
            )
        };
        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
        drop(guard);
        let context = NonNull::new(context).ok_or(LlamaContextLoadError::NullReturn)?;
        Ok(Self {
            context: Arc::new(LlamaContextInternal { context }),
            model: llama_model.clone(),
            initialized_logits: Vec::new(),
            embeddings_enabled: params.embeddings(),
        })
    }

    pub fn token_to_piece_with_special(
        &self,
        token: &LlamaToken,
        special: bool,
    ) -> crate::Result<String> {
        Ok(self.model.token_to_str_with_special(token, special)?)
    }
    pub fn token_to_piece(&self, token: &LlamaToken) -> crate::Result<String> {
        self.token_to_piece_with_special(token, false)
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
        //        assert!(
        //            self.initialized_logits.contains(&i),
        //            "logit {i} is not initialized. only {:?} is",
        //            self.initialized_logits
        //        );
        //        assert!(
        //            self.n_ctx() > u32::try_from(i).expect("i does not fit into a u32"),
        //            "n_ctx ({}) must be greater than i ({})",
        //            self.n_ctx(),
        //            i
        //        );

        let data = unsafe { llama_cpp_sys::llama_get_logits_ith(self.context.context.as_ptr(), i) };
        let len = usize::try_from(self.model.n_vocab()).expect("n_vocab does not fit into a usize");

        unsafe { slice::from_raw_parts(data, len) }
    }

    pub fn eval_tokens(
        &mut self,
        tokens: Vec<LlamaToken>,
        batch: usize,
        n_curr: &mut i32,
    ) -> Result<i32, DecodeError> {
        let mut rr = 0;
        for chunk in tokens.chunks(batch).into_iter() {
            let mut batch = LlamaBatch::new(batch, 1);
            let last_index = chunk.len() - 1;
            chunk.into_iter().enumerate().try_for_each(|(i, t)| {
                batch.add(t.clone(), *n_curr, &[0], i == last_index)?;
                *n_curr += 1;
                Ok::<(), DecodeError>(())
            })?;
            self.decode(&mut batch)?;
            rr = batch.n_tokens() - 1;
        }
        Ok(rr)
    }

    pub fn eval_id(&mut self, token: LlamaToken, n_curr: &mut i32) -> Result<i32, DecodeError> {
        self.eval_tokens(vec![token], 1, n_curr)
    }

    pub fn eval_string(
        &mut self,
        string: &str,
        batch: usize,
        add_bos: AddBos,
        n_curr: &mut i32,
    ) -> Result<i32, DecodeError> {
        let tokens = self.model.str_to_token(string, add_bos)?;
        self.eval_tokens(tokens, batch, n_curr)
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
}
