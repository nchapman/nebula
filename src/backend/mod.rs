#![allow(clippy::too_many_arguments)]
#[cfg(feature = "llama")]
use std::{path::PathBuf, pin::Pin, sync::Mutex};

#[cfg(feature = "whisper")]
use crate::{options::AutomaticSpeechRecognitionOptions, Result};
#[cfg(feature = "whisper")]
use std::path::PathBuf;

#[cfg(feature = "tts")]
use crate::options::{TTSOptions, TTSModelType};

#[cfg(feature = "llama")]
use crate::{
    options::{ContextOptions, ModelOptions},
    Result,
};

#[cfg(feature = "embeddings")]
use crate::{
    options::{EmbeddingsModelType, EmbeddingsOptions},
    Result,
};

#[cfg(feature = "llama")]
pub mod llama;

#[cfg(feature = "whisper")]
pub mod whisper;

#[cfg(feature = "embeddings")]
pub mod embeddings;

#[cfg(feature = "tts")]
pub mod tts;

#[cfg(feature = "llama")]
pub trait Context: Send {
    fn eval_str(&mut self, prompt: &str, add_bos: bool) -> Result<()>;
    fn eval_image(&mut self, image: Vec<u8>) -> Result<()>;
    fn predict(
        &mut self,
        max_len: Option<usize>,
        top_k: Option<i32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        temperature: Option<f32>,
        stop_tokens: &[String],
    ) -> Result<String>;
    fn predict_with_callback(
        &mut self,
        token_callback: std::sync::Arc<Box<dyn Fn(String) -> bool + Send + 'static>>,
        max_len: Option<usize>,
        top_k: Option<i32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        temperature: Option<f32>,
        stop_tokens: &[String],
    ) -> Result<()>;
}

#[cfg(feature = "llama")]
pub trait Model: Send {
    fn with_mmproj(&mut self, mmproj: PathBuf) -> Result<()>;
    fn new_context(&self, opions: ContextOptions) -> Result<Pin<Box<Mutex<dyn Context>>>>;
}

#[cfg(feature = "llama")]
pub fn init(
    model: impl Into<PathBuf>,
    options: ModelOptions,
    callback: Option<impl FnMut(f32) -> bool + 'static>,
) -> Result<impl Model> {
    llama::Llama::new(model, options, callback)
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

#[cfg(feature = "embeddings")]
pub trait EmbeddingsBackend {
    fn encode(&mut self, text: String) -> Result<Vec<f32>>;
}

#[cfg(feature = "embeddings")]
pub fn init_embeddings_backend(options: EmbeddingsOptions) -> Result<Box<dyn EmbeddingsBackend>> {
    match options.model_type {
        EmbeddingsModelType::JinaBert => Ok(Box::new(embeddings::JinaBertBackend::new(options)?)),
        EmbeddingsModelType::T5 => Ok(Box::new(embeddings::T5Backend::new(options)?)),
        EmbeddingsModelType::Bert => Ok(Box::new(embeddings::BertBackend::new(options)?)),
        _ => {
            panic!("This model type is not implemented yet!")
        }
    }
}

#[cfg(feature = "tts")]
pub trait TextToSpeechBackend {
    fn train(&mut self, ref_samples: Vec<f32>) -> anyhow::Result<()>;

    fn predict(&mut self, text: String) -> anyhow::Result<Vec<f32>>;
}

#[cfg(feature = "tts")]
pub fn init_text_to_speech_backend(
    options: TTSOptions,
) -> anyhow::Result<Box<dyn TextToSpeechBackend>> {
    match options.model_type {
        TTSModelType::Style => anyhow::Ok(Box::new(tts::StyleTTSBackend::new(options)?)),
        TTSModelType::ParlerMini => anyhow::Ok(Box::new(tts::ParlerBackend::new(options)?)),
        TTSModelType::ParlerLarge => anyhow::Ok(Box::new(tts::ParlerBackend::new(options)?)),
        _ => { panic!("This model type is not implemented yet!") }
    }
}
