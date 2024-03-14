use std::path::PathBuf;

pub mod error;
pub mod options;
pub type Result<T> = std::result::Result<T, error::Error>;

mod backend;

pub struct Model {
    backend: Box<dyn backend::Backend>,
}

impl Model {
    pub fn new(
        model: impl Into<PathBuf> + 'static,
        options: options::ModelOptions,
    ) -> Result<Self> {
        let backend = backend::init(model, options)?;
        Ok(Self {
            backend: Box::new(backend),
        })
    }

    pub fn predict(
        &mut self,
        prompt: &str,
        options: options::PredictOptions,
        token_callback: Box<dyn Fn(String) -> bool + Send + 'static>,
    ) -> Result<()> {
        self.backend.predict(prompt, options, token_callback)?;
        Ok(())
    }
}

impl Drop for Model {
    fn drop(&mut self) {}
}
