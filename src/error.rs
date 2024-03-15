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
    LlamaBatchAdd(#[from] llama_cpp::llama_batch::BatchAddError),
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LlamaDecode(#[from] llama_cpp::DecodeError),
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LlamaTokenToString(#[from] llama_cpp::TokenToStringError),

    #[error("{0} > {1}: the required kv cache size is not big enough either reduce n_len or increase n_ctx")]
    KVCacheNotBigEnough(usize, usize),
    #[error("the prompt is too long, it has more tokens than n_len")]
    PromtTooLong,

    #[error("{0}")]
    Unknown(String),
}
