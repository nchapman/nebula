use std::{
    num::NonZeroU32,
    path::{Path, PathBuf},
    pin::Pin,
};

use crate::{
    options::{ModelOptions, PredictOptions},
    Result,
};
use llama_cpp::{
    clip::ClipContext,
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
};

use super::Backend;

impl From<ModelOptions> for LlamaModelParams {
    fn from(val: ModelOptions) -> Self {
        let lmp = Self::default();
        if !val.cpu {
            lmp.with_n_gpu_layers(val.n_gpu_layers as u32)
        } else {
            lmp
        }
    }
}

impl From<PredictOptions> for LlamaContextParams {
    fn from(val: PredictOptions) -> Self {
        Self::default()
            .with_n_ctx(NonZeroU32::new(val.n_ctx as u32))
            .with_seed(val.seed)
    }
}

pub struct Llama {
    backend: LlamaBackend,
    model: LlamaModel,
    mmproj: Option<ClipContext>,
}

impl Llama {
    pub fn new(model: impl Into<PathBuf>, options: ModelOptions) -> Result<Self> {
        let backend = LlamaBackend::init()?;
        let model_params: Pin<Box<LlamaModelParams>> = Box::pin(options.into());
        let model = LlamaModel::load_from_file(&backend, Path::new(&model.into()), &model_params)?;
        Ok(Self {
            backend,
            model,
            mmproj: None,
        })
    }

    pub fn with_mmproj(mut self, model: impl Into<PathBuf>) -> Result<Self> {
        let clip_context = ClipContext::load(Path::new(&model.into()))?;
        self.mmproj = Some(clip_context);
        Ok(self)
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
        let mut n_curr = 0;
        let mut _logit = ctx.eval_string(prompt, 512, AddBos::Always, &mut n_curr)?;
        _logit = ctx.pedict(_logit, &mut n_curr, n_len as i32, token_callback)?;
        Ok(())
    }

    fn predict_with_image(
        &mut self,
        image: Vec<u8>,
        prompt: &str,
        options: PredictOptions,
        token_callback: Box<dyn Fn(String) -> bool + Send + 'static>,
    ) -> Result<()> {
        let n_len = options.n_len;
        let ctx_params: LlamaContextParams = options.into();
        let embedded_image = if let Some(clip_context) = &self.mmproj {
            clip_context.embed_image(ctx_params.n_threads() as usize, &image)?
        } else {
            return Err(crate::error::Error::MmprojNotDefined);
        };
        let (system_prompt, user_prompt) = if let Some(s) = prompt.split_once("<image>") {
            s
        } else {
            return Err(crate::error::Error::MmprojNotDefined);
        };
        eprintln!("{system_prompt}, {user_prompt}");
        let mut ctx = self.model.new_context(&self.backend, ctx_params)?;
        let mut n_curr = 0;
        ctx.eval_string(system_prompt, 2048, AddBos::Always, &mut n_curr)?;
        ctx.eval_embed_image(embedded_image, 2048, &mut n_curr)?;
        let mut _logit = ctx.eval_string(user_prompt, 2048, AddBos::Always, &mut n_curr)?;
        _logit = ctx.pedict(_logit, &mut n_curr, n_len as i32, token_callback)?;
        Ok(())
    }
}
