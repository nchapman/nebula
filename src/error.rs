use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    SslStack(#[from] openssl::error::ErrorStack),
    #[error("{0}")]
    ListenerType(String),
    #[error("{0}")]
    RequiredParameter(&'static str),
    #[error("{0}")]
    Unknown(String),
}

impl actix_web::ResponseError for Error {}
