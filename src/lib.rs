pub mod error;
use error::Error;

pub mod server;

pub type Result<T> = std::result::Result<T, Error>;

pub trait Completions {
    fn get(&self, promt: &str, sample_len: usize) -> Result<impl Iterator<Item = &str>>;
}
