use std::{path::PathBuf, pin::Pin, sync::Mutex};

#[cfg(feature = "whisper")]
use crate::{options::AutomaticSpeechRecognitionOptions, Result};

#[cfg(feature = "llama")]
use crate::{
    options::{ContextOptions, ModelOptions},
    Result,
};

#[cfg(feature = "llama")]
pub mod llama;

#[cfg(feature = "whisper")]
pub mod whisper;

#[cfg(feature = "llama")]
pub trait Context: Send {
    fn eval_str(&mut self, prompt: &str, add_bos: bool) -> Result<()>;
    fn eval_image(&mut self, image: Vec<u8>) -> Result<()>;
    fn predict(&mut self, max_len: Option<usize>, stop_tokens: &[String]) -> Result<String>;
    fn predict_with_callback(
        &mut self,
        token_callback: Box<dyn Fn(String) -> bool + Send + 'static>,
        max_len: Option<usize>,
        stop_tokens: &[String],
    ) -> Result<()>;
}

#[cfg(feature = "llama")]
pub trait Model: Send {
    fn with_mmproj(&mut self, mmproj: PathBuf) -> Result<()>;
    fn new_context(&self, opions: ContextOptions) -> Result<Pin<Box<Mutex<dyn Context>>>>;
}

#[cfg(feature = "llama")]
pub fn init(model: impl Into<PathBuf>, options: ModelOptions) -> Result<impl Model> {
    llama::Llama::new(model, options)
}

#[cfg(feature = "whisper")]
pub trait AutomaticSpeechRecognitionBackend {
    fn predict(
        &mut self,
        samples: &[f32],
        options: AutomaticSpeechRecognitionOptions,
    ) -> Result<String>;
}

#[cfg(feature = "whisper")]
pub fn init_automatic_speech_recognition_backend(
    model: impl Into<PathBuf>,
) -> Result<impl AutomaticSpeechRecognitionBackend> {
    Ok(whisper::Whisper::new(model)?)
}
