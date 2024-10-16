#![allow(clippy::too_many_arguments)]
#[cfg(feature = "llama")]
use std::{path::PathBuf, pin::Pin, sync::Mutex};

use crate::options::{Message, PredictOptions};
#[cfg(feature = "whisper")]
use crate::{options::AutomaticSpeechRecognitionOptions, Result};
#[cfg(feature = "whisper")]
use std::path::PathBuf;

#[cfg(feature = "tts")]
use crate::options::TTSOptions;

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

#[cfg(feature = "text-to-image")]
use candle_core::Tensor as CandleTensor;
#[cfg(feature = "text-to-image")]
use crate::options::{TextToImageOptions, TextToImageModelType};
#[cfg(feature = "text-to-image")]
pub mod text_to_image;

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
    fn eval(&mut self, msg: Vec<Message>) -> Result<()>;
    fn predict(&mut self, params: &PredictOptions) -> Result<String>;
    fn predict_with_callback(
        &mut self,
        params: &PredictOptions,
        token_callback: std::sync::Arc<Box<dyn Fn(String) -> bool + Send + Sync + Sync + 'static>>,
    ) -> Result<()>;
}

#[cfg(feature = "llama")]
pub trait Model: Send + Sync {
    fn name(&self) -> Result<&str>;
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
) -> anyhow::Result<impl TextToSpeechBackend> {
    anyhow::Ok(tts::StyleTTSBackend::new(options)?)
}

#[cfg(feature = "text-to-image")]
pub trait TextToImageBackend {
    fn generate(&mut self, prompt: String) -> anyhow::Result<CandleTensor>;
}

#[cfg(feature = "text-to-image")]
pub fn init_text_to_image_backend(
        options: TextToImageOptions
) -> anyhow::Result<Box<dyn TextToImageBackend>> {
    match options.model_type {
        TextToImageModelType::StableDiffusion => anyhow::Ok(Box::new(text_to_image::StableDiffusionBackend::new(options)?)),
        TextToImageModelType::Wuerstchen => anyhow::Ok(Box::new(text_to_image::WuerstchenBackend::new(options)?)),
        _ => { panic!("This model type is not implemented yet!") }
    }
}
