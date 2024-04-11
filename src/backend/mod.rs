use std::{path::PathBuf, pin::Pin, sync::Mutex};

use crate::{
    options::{ContextOptions, ModelOptions},
    Result,
};

#[cfg(feature = "llama")]
pub mod llama;

pub trait Context: Send {
    fn eval_str(&mut self, prompt: &str, add_bos: bool) -> Result<()>;
    fn eval_image(&mut self, image: Vec<u8>) -> Result<()>;
    fn predict(&mut self, max_len: usize, stop_tokens: &[String]) -> Result<String>;
    fn predict_with_callback(
        &mut self,
        token_callback: Box<dyn Fn(String) -> bool + Send + 'static>,
        max_len: usize,
        stop_tokens: &[String],
    ) -> Result<()>;
}

pub trait Model: Send {
    fn with_mmproj(&mut self, mmproj: PathBuf) -> Result<()>;
    fn new_context(&self, opions: ContextOptions) -> Result<Pin<Box<Mutex<dyn Context>>>>;
}

pub fn init(model: impl Into<PathBuf>, options: ModelOptions) -> Result<impl Model> {
    #[cfg(feature = "llama")]
    llama::Llama::new(model, options)
}
