use std::path::PathBuf;

use crate::{
    options::{ModelOptions, PredictOptions, AutomaticSpeechRecognitionOptions},
    Result,
};

#[cfg(feature = "llama")]
pub mod llama;

pub mod whisper;

pub trait Backend {
    fn predict(
        &mut self,
        prompt: &str,
        options: PredictOptions,
        token_callback: Box<dyn Fn(String) -> bool + Send + 'static>,
    ) -> Result<()>;
}

#[cfg(feature = "llama")]
pub fn init(model: impl Into<PathBuf>, options: ModelOptions) -> Result<impl Backend> {
    Ok(llama::Llama::new(model, options)?)
}

pub trait AutomaticSpeechRecognitionBackend {
    fn predict(
        &mut self,
        samples: &[f32],
        out_file_path: &str,
        options: AutomaticSpeechRecognitionOptions,
    ) -> Result<()>;
}

pub fn init_automatic_speech_recognition_backend(model: impl Into<PathBuf>) -> Result<impl AutomaticSpeechRecognitionBackend> {
    Ok(whisper::Whisper::new(model)?)
}
