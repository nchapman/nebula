use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LLamaCpp(#[from] llama_cpp_2::LLamaCppError),
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LLamaModelLoad(#[from] llama_cpp_2::LlamaModelLoadError),
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LlamaContextLoad(#[from] llama_cpp_2::LlamaContextLoadError),
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LlamaStringToToken(#[from] llama_cpp_2::StringToTokenError),
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LlamaBatchAdd(#[from] llama_cpp_2::llama_batch::BatchAddError),
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LlamaDecode(#[from] llama_cpp_2::DecodeError),
    #[cfg(feature = "llama")]
    #[error("{0}")]
    LlamaTokenToString(#[from] llama_cpp_2::TokenToStringError),

    #[error("{0} > {1}: the required kv cache size is not big enough either reduce n_len or increase n_ctx")]
    KVCacheNotBigEnough(usize, usize),
    #[error("the prompt is too long, it has more tokens than n_len")]
    PromtTooLong,

    #[error("{0}")]
    Unknown(String),
}
