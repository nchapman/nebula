#![allow(clippy::too_many_arguments)]
use std::{
    num::NonZeroU32,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, Mutex},
};

use crate::{
    options::{ContextOptions, ModelOptions},
    Result,
};
use llama_cpp::{
    clip::ClipContext,
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
};

use super::{Context, Model};

lazy_static::lazy_static! {
    static ref LLAMA_BACKEND: Arc<LlamaBackend> = Arc::new(LlamaBackend::init().unwrap());
}

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

impl From<ContextOptions> for LlamaContextParams {
    fn from(val: ContextOptions) -> Self {
        Self::default()
            .with_n_ctx(NonZeroU32::new(val.n_ctx as u32))
            .with_seed(val.seed)
            .with_n_threads(val.n_threads as u32)
            .with_n_batch(2048)
    }
}

#[derive(Clone)]
pub struct Llama {
    model: LlamaModel,
    mmproj: Option<ClipContext>,
}

impl Llama {
    pub fn new(
        model: impl Into<PathBuf>,
        options: ModelOptions,
        callback: Option<impl FnMut(f32) -> bool + 'static>,
    ) -> Result<Self> {
        let mut lmp: LlamaModelParams = options.into();
        if let Some(cb) = callback {
            lmp = lmp.with_load_process_callback(cb);
        }
        let model_params = Box::pin(lmp);
        let model =
            LlamaModel::load_from_file(&LLAMA_BACKEND, Path::new(&model.into()), &model_params)?;
        Ok(Self {
            model,
            mmproj: None,
        })
    }
}

impl Model for Llama {
    fn with_mmproj(&mut self, mmproj: PathBuf) -> Result<()> {
        let clip_context = ClipContext::load(Path::new(&mmproj))?;
        self.mmproj = Some(clip_context);
        Ok(())
    }
    fn new_context(&self, options: ContextOptions) -> Result<Pin<Box<Mutex<dyn Context>>>> {
        Ok(Box::pin(Mutex::new(LlamaContext::new(self, options)?)))
    }
}

pub struct LlamaContext {
    logit: i32,
    n_curr: i32,
    n_threads: usize,
    ctx: Pin<Box<llama_cpp::context::LlamaContext>>,
    model: Arc<Llama>,
}

impl<'a> LlamaContext {
    pub fn new(model: &'a Llama, options: ContextOptions) -> Result<Self> {
        let ctx_params: LlamaContextParams = options.into();
        let n_threads = ctx_params.n_threads() as usize;
        let ctx = Self {
            ctx: Box::pin(model.model.new_context(&LLAMA_BACKEND, ctx_params)?),
            n_curr: 0,
            logit: 0,
            n_threads,
            model: Arc::new(model.clone()),
        };
        Ok(ctx)
    }
}

impl Context for LlamaContext {
    fn eval_str(&mut self, prompt: &str, add_bos: bool) -> Result<()> {
        self.logit = self.ctx.eval_string(
            prompt,
            2048,
            if add_bos {
                AddBos::Always
            } else {
                AddBos::Never
            },
            &mut self.n_curr,
        )?;
        Ok(())
    }
    fn eval_image(&mut self, image: Vec<u8>) -> Result<()> {
        let embedded_image = if let Some(clip_context) = &self.model.mmproj {
            clip_context.embed_image(self.n_threads, &image)?
        } else {
            return Err(crate::error::Error::MmprojNotDefined);
        };
        log::debug!("image embedding created: {} tokens", embedded_image.len());
        self.ctx
            .eval_embed_image(embedded_image, 2048, &mut self.n_curr)?;
        Ok(())
    }

    fn predict(
        &mut self,
        max_len: Option<usize>,
        top_k: Option<i32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        temperature: Option<f32>,
        stop_tokens: &[String],
    ) -> Result<String> {
        let res = Arc::new(Mutex::new("".to_string()));
        let rres = res.clone();
        self.predict_with_callback(
            std::sync::Arc::new(Box::new(move |token| {
                rres.lock().unwrap().push_str(&token);
                true
            })),
            max_len,
            top_k,
            top_p,
            min_p,
            temperature,
            stop_tokens,
        )?;
        let rres = res.lock().unwrap();
        Ok(rres.clone())
    }

    fn predict_with_callback(
        &mut self,
        token_callback: std::sync::Arc<Box<dyn Fn(String) -> bool + Send + 'static>>,
        max_len: Option<usize>,
        top_k: Option<i32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        temperature: Option<f32>,
        stop_tokens: &[String],
    ) -> Result<()> {
        let dst = crate::options::default_stop_tokens();
        self.logit = self.ctx.pedict(
            self.logit,
            &mut self.n_curr,
            max_len,
            top_k,
            top_p,
            min_p,
            temperature,
            if stop_tokens.is_empty() {
                &dst
            } else {
                stop_tokens
            },
            token_callback,
        )?;
        Ok(())
    }
}
