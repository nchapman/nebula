use std::{
    io::{BufWriter, IntoInnerError},
    string::FromUtf8Error,
};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LLamaCpp(#[from] llama_cpp::LLamaCppError),
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LLamaModelLoad(#[from] llama_cpp::LlamaModelLoadError),
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LlamaContextLoad(#[from] llama_cpp::LlamaContextLoadError),
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LlamaStringToToken(#[from] llama_cpp::StringToTokenError),
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LlamaDecode(#[from] llama_cpp::DecodeError),
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LlamaClip(#[from] llama_cpp::ClipError),
    #[cfg(feature = "llama")]
    #[error("{0}")]
    Predict(#[from] llama_cpp::PredictError),
    #[error("not mmproj models not support images")]
    ModelNotMmproj,
    #[error("{0} > {1}: the required kv cache size is not big enough either reduce n_len or increase n_ctx")]
    KVCacheNotBigEnough(usize, usize),
    #[error("the prompt is too long, it has more tokens than n_len")]
    PromtTooLong,
    #[error("for image processing mmproj model should be defined")]
    MmprojNotDefined,
    #[error("{0}")]
    Unknown(String),
    #[error("{0}")]
    UnsupportedTemplate(String),
    #[error("{0}")]
    FromUtf8(#[from] FromUtf8Error),
    #[error("{0}")]
    IntoInner(#[from] IntoInnerError<BufWriter<Vec<u8>>>),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(feature = "llama-http")]
impl actix_web::error::ResponseError for Error {}
