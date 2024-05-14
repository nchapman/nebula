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
    backend: Arc<LlamaBackend>,
    model: LlamaModel,
    mmproj: Option<ClipContext>,
}

impl Llama {
    pub fn new(model: impl Into<PathBuf>, options: ModelOptions) -> Result<Self> {
        let backend = Arc::new(LlamaBackend::init()?);
        let model_params: Pin<Box<LlamaModelParams>> = Box::pin(options.into());
        let model = LlamaModel::load_from_file(&backend, Path::new(&model.into()), &model_params)?;
        Ok(Self {
            backend,
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
            ctx: Box::pin(model.model.new_context(&model.backend, ctx_params)?),
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

    fn predict(&mut self, max_len: Option<usize>, stop_tokens: &[String]) -> Result<String> {
        let res = Arc::new(Mutex::new("".to_string()));
        let rres = res.clone();
        self.predict_with_callback(
            Box::new(move |token| {
                rres.lock().unwrap().push_str(&token);
                true
            }),
            max_len,
            stop_tokens,
        )?;
        let rres = res.lock().unwrap();
        Ok(rres.clone())
    }

    fn predict_with_callback(
        &mut self,
        token_callback: Box<dyn Fn(String) -> bool + Send + 'static>,
        max_len: Option<usize>,
        stop_tokens: &[String],
    ) -> Result<()> {
        self.logit = self.ctx.pedict(
            self.logit,
            &mut self.n_curr,
            max_len,
            stop_tokens,
            token_callback,
        )?;
        Ok(())
    }
}
