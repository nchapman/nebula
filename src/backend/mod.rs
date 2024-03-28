use std::path::PathBuf;

use crate::{
    options::{ModelOptions, PredictOptions},
    Result,
};

#[cfg(feature = "llama")]
pub mod llama;

pub trait Backend {
    fn predict(
        &mut self,
        prompt: &str,
        options: PredictOptions,
        token_callback: Box<dyn Fn(String) -> bool + Send + 'static>,
    ) -> Result<()>;

    fn predict_with_image(
        &mut self,
        image: Vec<u8>,
        prompt: &str,
        options: PredictOptions,
        token_callback: Box<dyn Fn(String) -> bool + Send + 'static>,
    ) -> Result<()>;
}

pub fn init(model: impl Into<PathBuf>, options: ModelOptions) -> Result<impl Backend> {
    #[cfg(feature = "llama")]
    Ok(llama::Llama::new(model, options)?)
}

pub fn init_with_mmproj(
    model: impl Into<PathBuf>,
    mmproj: impl Into<PathBuf>,
    options: ModelOptions,
) -> Result<impl Backend> {
    #[cfg(feature = "llama")]
    Ok(llama::Llama::new(model, options)?.with_mmproj(mmproj)?)
}
