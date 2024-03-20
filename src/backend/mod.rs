use std::path::PathBuf;

use crate::{
    options::{ModelOptions, PredictOptions},
    Result,
};

#[cfg(feature = "llama")]
pub mod llama;
#[cfg(feature = "whisper")]
pub mod whisper;

pub trait Backend {
    fn predict(
        &mut self,
        prompt: &str,
        options: PredictOptions,
        token_callback: Box<dyn Fn(String) -> bool + Send + 'static>,
    ) -> Result<()>;
}

pub fn init(model: impl Into<PathBuf>, options: ModelOptions) -> Result<impl Backend> {
    #[cfg(feature = "llama")]
    Ok(llama::Llama::new(model, options)?);
    #[cfg(feature = "whisper")]
    Ok(whisper::Whisper::new(model, options)?)
}
