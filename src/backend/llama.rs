use std::{
    num::NonZeroU32,
    path::{Path, PathBuf},
    pin::Pin,
};

use llama_cpp::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    token::data_array::LlamaTokenDataArray,
};
use log::debug;

use crate::{
    error::Error,
    options::{ModelOptions, PredictOptions},
    Result,
};

use super::Backend;

impl Into<LlamaModelParams> for ModelOptions {
    fn into(self) -> LlamaModelParams {
        let lmp = LlamaModelParams::default();
        if !self.cpu {
            lmp.with_n_gpu_layers(self.n_gpu_layers as u32)
        } else {
            lmp
        }
    }
}

impl Into<LlamaContextParams> for PredictOptions {
    fn into(self) -> LlamaContextParams {
        LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx as u32))
            .with_seed(self.seed)
    }
}

pub struct Llama {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl Llama {
    pub fn new(model: impl Into<PathBuf>, options: ModelOptions) -> Result<Self> {
        let backend = LlamaBackend::init()?;
        let model_params: Pin<Box<LlamaModelParams>> = Box::pin(options.into());
        let model = LlamaModel::load_from_file(&backend, Path::new(&model.into()), &model_params)?;
        Ok(Self { backend, model })
    }
}

impl Backend for Llama {
    fn predict(
        &mut self,
        prompt: &str,
        options: PredictOptions,
        token_callback: Box<dyn Fn(String) -> bool + Send + 'static>,
    ) -> Result<()> {
        let n_len = options.n_len;
        let ctx_params: LlamaContextParams = options.into();
        let mut ctx = self.model.new_context(&self.backend, ctx_params)?;
        let tokens_list = self.model.str_to_token(prompt, AddBos::Always)?;
        let n_cxt = ctx.n_ctx() as usize;
        let n_kv_req = tokens_list.len() + (n_len - tokens_list.len());
        debug!("n_len = {}, n_ctx = {n_cxt}, k_kv_req = {n_kv_req}", n_len);
        if n_kv_req > n_cxt {
            return Err(Error::KVCacheNotBigEnough(n_kv_req, n_cxt));
        }
        if tokens_list.len() >= n_len {
            return Err(Error::PromtTooLong);
        }
        debug!(
            "prompt tokens: {}",
            tokens_list
                .iter()
                .map(|t| self.model.token_to_str(*t).unwrap_or_default())
                .collect::<Vec<String>>()
                .join(" ")
        );
        let mut batch = LlamaBatch::new(512, 1);
        let last_index = tokens_list.len() - 1;
        tokens_list
            .into_iter()
            .enumerate()
            .try_for_each(|(i, t)| batch.add(t, i as i32, &[0], i == last_index))?;

        ctx.decode(&mut batch)?;

        let mut n_cur = batch.n_tokens() as usize;

        while n_cur <= n_len {
            {
                let candidates = ctx.candidates_ith(batch.n_tokens() - 1);
                let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);
                let new_token_id = ctx.sample_token_greedy(candidates_p);
                if new_token_id == self.model.token_eos() {
                    break;
                }
                if !token_callback(self.model.token_to_str(new_token_id)?) {
                    break;
                }
                batch.clear();
                batch.add(new_token_id, n_cur as i32, &[0], true)?;
            }
            n_cur += 1;
            ctx.decode(&mut batch)?;
        }
        Ok(())
    }
}
