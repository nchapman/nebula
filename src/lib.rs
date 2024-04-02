use crate::backend::Model as _;
use std::{path::PathBuf, pin::Pin, sync::Mutex};

pub mod error;
pub mod options;
pub type Result<T> = std::result::Result<T, error::Error>;

mod backend;

pub struct Model {
    backend: Pin<Box<dyn backend::Model>>,
}

impl Model {
    pub fn new(
        model: impl Into<PathBuf> + 'static,
        options: options::ModelOptions,
    ) -> Result<Self> {
        let backend = backend::init(model, options)?;
        Ok(Self {
            backend: Box::pin(backend),
        })
    }

    pub fn new_with_mmproj(
        model: impl Into<PathBuf> + 'static,
        mmproj: impl Into<PathBuf> + 'static,
        options: options::ModelOptions,
    ) -> Result<Self> {
        let mut backend = backend::init(model, options)?;
        backend.with_mmproj(mmproj.into())?;
        Ok(Self {
            backend: Box::pin(backend),
        })
    }

    pub fn context(&self, options: options::ContextOptions) -> Result<Context> {
        Ok(Context {
            backend: self.backend.new_context(options)?,
        })
    }
}

pub struct Context<'a> {
    backend: Pin<Box<Mutex<dyn backend::Context + 'a>>>,
}

impl Context<'_> {
    pub fn eval_str(&mut self, prompt: &str, add_bos: bool) -> Result<()> {
        self.backend.lock().unwrap().eval_str(prompt, add_bos)?;
        Ok(())
    }

    pub fn eval_image(&mut self, image: Vec<u8>) -> Result<()> {
        self.backend.lock().unwrap().eval_image(image)?;
        Ok(())
    }

    pub fn predict(&mut self, max_len: usize) -> Result<String> {
        Ok(self.backend.lock().unwrap().predict(max_len)?)
    }

    pub fn predict_with_callback(
        &mut self,
        token_callback: Box<dyn Fn(String) -> bool + Send + 'static>,
        max_len: usize,
    ) -> Result<()> {
        Ok(self
            .backend
            .lock()
            .unwrap()
            .predict_with_callback(token_callback, max_len)?)
    }
}

impl Drop for Model {
    fn drop(&mut self) {}
}
