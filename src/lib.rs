use strfmt::strfmt;

use crate::backend::Model as _;
use std::{collections::HashMap, path::PathBuf, pin::Pin, sync::Mutex};

pub mod error;
pub mod options;
pub type Result<T> = std::result::Result<T, error::Error>;

mod backend;

pub struct Model {
    backend: Pin<Box<dyn backend::Model>>,
}

impl Model {
    pub fn new(
        model: impl Into<PathBuf> + 'static,
        options: options::ModelOptions,
    ) -> Result<Self> {
        let backend = backend::init(model, options)?;
        Ok(Self {
            backend: Box::pin(backend),
        })
    }

    pub fn new_with_mmproj(
        model: impl Into<PathBuf> + 'static,
        mmproj: impl Into<PathBuf> + 'static,
        options: options::ModelOptions,
    ) -> Result<Self> {
        let mut backend = backend::init(model, options)?;
        backend.with_mmproj(mmproj.into())?;
        Ok(Self {
            backend: Box::pin(backend),
        })
    }

    pub fn context(&self, options: options::ContextOptions) -> Result<Context> {
        let mut ctx = Context {
            options: options.clone(),
            backend: self.backend.new_context(options.clone())?,
        };
        options.ctx.into_iter().try_for_each(|m| {
            let (prompt, bos) = match m.is_user {
                true => {
                    let mut vars = HashMap::new();
                    vars.insert("prompt".to_string(), m.message);
                    (strfmt(&options.user_format, &vars).unwrap(), true)
                }
                false => {
                    let mut vars = HashMap::new();
                    vars.insert("prompt".to_string(), m.message);
                    (strfmt(&options.assistant_format, &vars).unwrap(), true)
                }
            };
            eprintln!("{}", prompt);
            ctx.eval_str(&prompt, bos)?;
            Ok::<(), error::Error>(())
        })?;
        Ok(ctx)
    }
}

pub struct Context {
    options: options::ContextOptions,
    backend: Pin<Box<Mutex<dyn backend::Context>>>,
}

impl Context {
    pub fn eval_str(&mut self, prompt: &str, add_bos: bool) -> Result<()> {
        let mut vars = HashMap::new();
        vars.insert("prompt".to_string(), prompt);
        let prompt = strfmt(&self.options.prompt_format, &vars).unwrap();
        self.backend.lock().unwrap().eval_str(&prompt, add_bos)?;
        Ok(())
    }

    pub fn eval_image(&mut self, image: Vec<u8>, prompt: &str) -> Result<()> {
        let mut vars = HashMap::new();
        vars.insert("prompt".to_string(), prompt);
        let prompt = strfmt(&self.options.prompt_format_with_image, &vars).unwrap();
        if let Some((s1, s2)) = prompt.split_once("{image}") {
            let mut bb = self.backend.lock().unwrap();
            bb.eval_str(s1, false)?;
            bb.eval_image(image)?;
            bb.eval_str(s2, true)?;
        } else {
            let mut bb = self.backend.lock().unwrap();
            bb.eval_image(image)?;
            bb.eval_str(&prompt, true)?;
        };
        Ok(())
    }

    pub fn predict(&mut self, max_len: usize) -> Result<String> {
        self.backend
            .lock()
            .unwrap()
            .predict(max_len, &self.options.stop_tokens)
    }

    pub fn predict_with_callback(
        &mut self,
        token_callback: Box<dyn Fn(String) -> bool + Send + 'static>,
        max_len: usize,
    ) -> Result<()> {
        self.backend.lock().unwrap().predict_with_callback(
            token_callback,
            max_len,
            &self.options.stop_tokens,
        )
    }
}

impl Drop for Model {
    fn drop(&mut self) {}
}
