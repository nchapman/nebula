use strfmt::strfmt;

use crate::backend::Model as _;
use std::{collections::HashMap, path::PathBuf, pin::Pin, sync::Mutex};

pub mod error;
pub mod options;
pub type Result<T> = std::result::Result<T, error::Error>;
pub mod utils;

mod backend;

pub struct Model {
    backend: Pin<Box<dyn backend::Model>>,
}

#[cfg(feature = "llama")]
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
        if let Some((s1, s2)) = &self.options.prompt_format_with_image.split_once("{image}") {
            //            eprintln!("{s1}: {s2}");
            let mut vars = HashMap::new();
            vars.insert("prompt".to_string(), prompt);
            let prompt = strfmt(s2, &vars).unwrap();
            let mut bb = self.backend.lock().unwrap();
            //            eprintln!("{prompt}");
            bb.eval_str(s1, true)?;
            bb.eval_image(image)?;
            bb.eval_str(&prompt, false)?;
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

#[cfg(test)]
mod test {
    use std::{io::Read, path::PathBuf};

    struct TestModel {
        pub _repo: String,
        pub filename: PathBuf,
        pub _mmproj: Option<PathBuf>,
    }

    impl TestModel {
        pub fn new(repo: &str, model_file_name: &str) -> Self {
            let api = hf_hub::api::sync::Api::new();
            assert!(api.is_ok());
            let api = api.unwrap();
            let filename = api.model(repo.to_string()).get(model_file_name);
            assert!(filename.is_ok());
            let filename = filename.unwrap();
            Self {
                _repo: repo.to_string(),
                filename,
                _mmproj: None,
            }
        }

        pub fn _with_mmproj(mut self, mmproj: &str) -> Self {
            let api = hf_hub::api::sync::Api::new();
            assert!(api.is_ok());
            let api = api.unwrap();
            let filename = api.model(self._repo.clone()).get(mmproj);
            assert!(filename.is_ok());
            let filename = filename.unwrap();
            self._mmproj = Some(filename);
            self
        }
    }

    impl Drop for TestModel {
        fn drop(&mut self) {
            let res = std::fs::remove_file(&self.filename);
            assert!(res.is_ok())
        }
    }

    fn main_with_model(model_repo: &str, model_file_name: &str) {
        let test_model = TestModel::new(model_repo, model_file_name);
        let model_options = super::options::ModelOptions::default();
        let prompt = "Write simple Rust programm.";
        let model = super::Model::new(test_model.filename.clone(), model_options);
        assert!(model.is_ok());
        let model = model.unwrap();
        let ctx_options = super::options::ContextOptions::default();
        let ctx = model.context(ctx_options);
        assert!(ctx.is_ok());
        let mut ctx = ctx.unwrap();
        let eval_res = ctx.eval_str(&prompt, true);
        assert!(eval_res.is_ok());
        let answer = ctx.predict(100);
        assert!(answer.is_ok());
        let answer = answer.unwrap();
        println!("{answer}");
    }

    macro_rules! models_tests {
        ($($name:ident: ($value:expr, $value2:expr),)*) => {
            $(
                #[test]
                fn $name() {
                    main_with_model($value, $value2);
                }
            )*
        }
    }

    models_tests! {
    //        model_test_llava_1_6_mistral_7b_gguf:
    //        ("cjpais/llava-1.6-mistral-7b-gguf",
        //        "llava-v1.6-mistral-7b.Q4_K_M.gguf"),
        test: ("stabilityai/stable-code-instruct-3b","stable-code-3b-q4_k_m.gguf"),
        }

    fn _main_with_model_and_mmproj(
        model_repo: &str,
        model_file_name: &str,
        mmproj_file_name: &str,
    ) {
        let image_path = std::env::var("NEBULA_IMAGE").unwrap();
        let model_options = super::options::ModelOptions::default();
        let prompt = "Write simple Rust programm.";
        let test_model = TestModel::new(model_repo, model_file_name)._with_mmproj(mmproj_file_name);
        let model = super::Model::new_with_mmproj(
            test_model.filename.clone(),
            test_model._mmproj.clone().unwrap(),
            model_options,
        );
        assert!(model.is_ok());
        let model = model.unwrap();
        let ctx_options = super::options::ContextOptions::default();
        let ctx = model.context(ctx_options);
        assert!(ctx.is_ok());
        let mut ctx = ctx.unwrap();

        let mut image_bytes = vec![];
        let f = std::fs::File::open(&image_path);
        assert!(f.is_ok());
        let mut f = f.unwrap();
        let read_res = f.read_to_end(&mut image_bytes);
        assert!(read_res.is_ok());
        let eval_res = ctx.eval_image(image_bytes, &prompt);
        assert!(eval_res.is_ok());
        let answer = ctx.predict(100);
        assert!(answer.is_ok());
        let answer = answer.unwrap();
        println!("{answer}");
    }

    macro_rules! models_with_mmproj_tests {
        ($($name:ident: ($value:expr, $value2:expr, $value3:expr),)*) => {
            $(
                #[test]
                fn $name() {
                    main_with_model_and_mmproj($value, $value2, $value3);
                }
            )*
        }
    }

    models_with_mmproj_tests! {
    //        model_test_llava_1_6_mistral_7b_gguf_with_mmproj: ("cjpais/llava-1.6-mistral-7b-gguf", "llava-v1.6-mistral-7b.Q4_K_M.gguf", "mmproj-model-f16.gguf"),
        }
}

pub struct AutomaticSpeechRecognitionModel {
    backend: Box<dyn backend::AutomaticSpeechRecognitionBackend>,
}

impl AutomaticSpeechRecognitionModel {
    pub fn new(
        model: impl Into<PathBuf> + 'static
    ) -> Result<Self> {
        let backend = backend::init_automatic_speech_recognition_backend(model)?;
        Ok(Self {
            backend: Box::new(backend),
        })
    }

    pub fn predict(
        &mut self,
        samples: &[f32],
        options: options::AutomaticSpeechRecognitionOptions,
    ) -> Result<String> {
        Ok(self.backend.predict(samples, options)?)
    }
}

impl Drop for AutomaticSpeechRecognitionModel {
    fn drop(&mut self) {}
}
