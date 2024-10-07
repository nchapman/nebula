#[cfg(feature = "llama-http")]
use actix_web::{App, HttpResponse, HttpServer, Responder};
use options::{ContextOptions, Message, PredictOptions, TokenCallback};
use serde::Deserialize;
#[cfg(feature = "llama-http")]
use tokio::sync::RwLock;

//#![allow(clippy::type_complexity)]
//#![allow(clippy::arc_with_non_send_sync)]
#[cfg(feature = "llama")]
use crate::backend::Model as _;

use std::{net::IpAddr, sync::Arc};
#[cfg(feature = "llama")]
use std::{path::PathBuf, pin::Pin, sync::Mutex};

#[cfg(feature = "whisper")]
use std::path::PathBuf;

pub mod error;
pub mod options;
pub type Result<T> = std::result::Result<T, error::Error>;
pub mod utils;

mod backend;

pub fn init(resource_path: std::path::PathBuf) -> Result<()> {
    resource_path::set(resource_path).map_err(error::Error::Unknown)
}

#[cfg(feature = "llama")]
#[derive(Clone)]
pub struct Model {
    backend: Arc<Pin<Box<dyn backend::Model>>>,
}

#[cfg(feature = "llama")]
impl Model {
    pub fn new(
        model: impl Into<PathBuf> + 'static,
        options: options::ModelOptions,
    ) -> Result<Self> {
        let backend = backend::init(
            model,
            options,
            None::<Box<dyn FnMut(f32) -> bool + 'static>>,
        )?;
        Ok(Self {
            backend: Arc::new(Box::pin(backend)),
        })
    }

    pub fn new_with_progress_callback(
        model: impl Into<PathBuf> + 'static,
        options: options::ModelOptions,
        callback: impl FnMut(f32) -> bool + 'static,
    ) -> Result<Self> {
        let backend = backend::init(model, options, Some(Box::new(callback)))?;
        Ok(Self {
            backend: Arc::new(Box::pin(backend)),
        })
    }

    pub fn new_with_mmproj(
        model: impl Into<PathBuf> + 'static,
        mmproj: impl Into<PathBuf> + 'static,
        options: options::ModelOptions,
    ) -> Result<Self> {
        let mut backend = backend::init(
            model,
            options,
            None::<Box<dyn FnMut(f32) -> bool + 'static>>,
        )?;
        backend.with_mmproj(mmproj.into())?;
        Ok(Self {
            backend: Arc::new(Box::pin(backend)),
        })
    }

    pub fn new_with_mmproj_with_callback(
        model: impl Into<PathBuf> + 'static,
        mmproj: impl Into<PathBuf> + 'static,
        options: options::ModelOptions,
        callback: impl FnMut(f32) -> bool + 'static,
    ) -> Result<Self> {
        let mut backend = backend::init(model, options, Some(Box::new(callback)))?;
        backend.with_mmproj(mmproj.into())?;
        Ok(Self {
            backend: Arc::new(Box::pin(backend)),
        })
    }

    pub fn context(&self, options: options::ContextOptions) -> Result<Context> {
        let ctx = Context {
            _options: options.clone(),
            backend: self.backend.new_context(options)?,
        };
        Ok(ctx)
    }
}

#[cfg(feature = "llama")]
pub struct Context {
    _options: options::ContextOptions,
    backend: Pin<Box<Mutex<dyn backend::Context>>>,
}

#[cfg(feature = "llama")]
#[derive(bon::Builder)]
pub struct Predict<'a> {
    context: &'a Context,
    options: options::PredictOptions,
    token_callback: Option<Box<TokenCallback>>,
}

#[cfg(feature = "llama")]
impl<'a> Predict<'a> {
    pub fn new(context: &Context, options: options::PredictOptions) -> Predict {
        Predict {
            context,
            options,
            token_callback: None,
        }
    }
    pub fn with_token_callback(
        mut self,
        token_callback: Box<dyn Fn(String) -> bool + Send + Sync + 'static>,
    ) -> Self {
        self.token_callback = Some(token_callback);
        self
    }

    pub fn predict(&mut self) -> Result<String> {
        if let Some(callback) = self.options.token_callback.clone() {
            self.context
                .backend
                .lock()
                .unwrap()
                .predict_with_callback(&self.options, callback)?;
            Ok("".to_string())
        } else {
            self.context.backend.lock().unwrap().predict(&self.options)
        }
    }
}

#[cfg(feature = "llama")]
impl Context {
    pub fn eval(&mut self, msgs: Vec<Message>) -> Result<()> {
        self.backend.lock().unwrap().eval(msgs)?;
        Ok(())
    }

    pub fn predict(&mut self, options: options::PredictOptions) -> Predict {
        Predict::new(self, options)
    }
}

#[cfg(feature = "llama")]
impl Drop for Model {
    fn drop(&mut self) {}
}

fn default_f32_1() -> f32{
    1.0
}

#[cfg(feature = "llama-http")]
#[derive(Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    frequency_penalty: f32,
    max_completion_tokens: Option<i32>,
    #[serde(default)]
    presence_penalty: f32,
    seed: Option<u32>,
    #[serde(default = "default_f32_1")]
    temperature: f32,
    #[serde(default = "default_f32_1")]
    top_p: f32,
    _model: String,
    messages: Vec<Message>,
}


#[cfg(feature = "llama-http")]
#[actix_web::post("/v1/chat/completions")]
async fn complitions(state: actix_web::web::Data<AppState>, json: actix_web::web::Json<CompletionRequest>) -> Result<impl Responder> {
    let data = json.into_inner();
    let mut ctx = state.model.read().await.context(state.context_options.clone())?;
    ctx.eval(data.messages)?;
    let mut predict_options = PredictOptions::builder()
        .penalty_freq(data.frequency_penalty)
        .penalty_present(data.presence_penalty)
        .temp(data.temperature)
        .top_p(data.top_p)
        .build();
    if let Some(ss) = data.seed{
        predict_options.seed = ss;
    }
    predict_options.max_len = data.max_completion_tokens;
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "id": "chatcmpl",
        "object": "chat.completion",
        "created": 1677652288,
        "model": state.model.read().await.backend.name()?,
        "system_fingerprint": "fp",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ctx.predict(predict_options).predict()?
            },
            "logprobs": null,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "completion_tokens_details": {
                "reasoning_tokens": 0
            }
        }
    })))
}

#[cfg(feature = "llama-http")]
struct AppState {
    context_options: ContextOptions,
    model: Arc<RwLock<Model>>,
}

#[cfg(feature = "llama-http")]
pub struct Server {
    host: IpAddr,
    port: u16,
    model: Model,
    context_options: ContextOptions
}

#[cfg(feature = "llama-http")]
impl Server {
    pub fn new(host: impl Into<IpAddr>, port: u16, model: Model, context_options: ContextOptions) -> Self{
        Self{
            host: host.into(),
            port,
            model,
            context_options
        }

    }

    pub fn run(&self) -> Result<()>{
        let mm = Arc::new(RwLock::new(self.model.clone()));
        let context_options = self.context_options.clone();
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(async move{
                HttpServer::new(move || {
                    App::new()
                        .app_data(actix_web::web::Data::new(AppState {
                            context_options: context_options.clone(),
                            model: mm.clone(),
                        }))
                        .service(complitions)
                })
                    .bind((self.host, self.port))?
                    .run()
                    .await
            })?;
            Ok(())
    }
}



#[cfg(test)]
#[cfg(feature = "llama")]
mod test {
    use std::{io::Write, path::PathBuf};

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
        simple_logger::SimpleLogger::new()
            .with_level(log::LevelFilter::Debug)
            .init()
            .unwrap();
        super::init(std::path::PathBuf::from(
            "backends/llama_cpp/llama-cpp-sys/dist",
        ))
        .unwrap();
        let test_model = TestModel::new(model_repo, model_file_name);
        eprintln!("{}", test_model.filename.display());
        let model_options = super::options::ModelOptions::default();
        let prompt = r###"{"role": "user", "content": "Write simple Rust programm."}"###;
        let model = super::Model::new(test_model.filename.clone(), model_options);
        assert!(model.is_ok());
        let model = model.unwrap();
        let ctx_options = super::options::ContextOptions::builder().build();
        let ctx = model.context(ctx_options);
        assert!(ctx.is_ok());
        let mut ctx = ctx.unwrap();
        let eval_result = ctx.eval(vec![prompt.try_into().unwrap()]);
        assert!(eval_result.is_ok());
        let answer = ctx
            .predict(super::options::PredictOptions::builder().token_callback(std::sync::Arc::new(Box::new(|token| {
            print!("{}", token);
            std::io::stdout().flush().unwrap();
            true
        }))).build())
            .predict();
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
        //test: ("stabilityai/stable-code-instruct-3b","stable-code-3b-q4_k_m.gguf"),
        test: ("TheBloke/evolvedSeeker_1_3-GGUF","evolvedseeker_1_3.Q2_K.gguf"),
        }

    //     fn _main_with_model_and_mmproj(
    //         model_repo: &str,
    //         model_file_name: &str,
    //         mmproj_file_name: &str,
    //     ) {
    //         let image_path = std::env::var("NEBULA_IMAGE").unwrap();
    //         let model_options = super::options::ModelOptions::default();
    //         let prompt = "Write simple Rust programm.";
    //         let test_model = TestModel::new(model_repo, model_file_name)._with_mmproj(mmproj_file_name);
    //         let model = super::Model::new_with_mmproj(
    //             test_model.filename.clone(),
    //             test_model._mmproj.clone().unwrap(),
    //             model_options,
    //         );
    //         assert!(model.is_ok());
    //         let model = model.unwrap();
    //         let ctx_options = super::options::ContextOptions::default();
    //         let ctx = model.context(ctx_options);

    //         assert!(ctx.is_ok());
    //         let mut ctx = ctx.unwrap();

    //         let mut image_bytes = vec![];
    //         let f = std::fs::File::open(&image_path);
    //         assert!(f.is_ok());
    //         let mut f = f.unwrap();
    // //        let read_res = f.read_to_end(&mut image_bytes);
    // //        assert!(read_res.is_ok());
    // //        let eval_res = ctx.eval_image(image_bytes, &prompt);
    // //        assert!(eval_res.is_ok());
    //         let answer = ctx.predict().predict();
    //         assert!(answer.is_ok());
    //         let answer = answer.unwrap();
    //         println!("{answer}");
    //     }

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

#[cfg(feature = "whisper")]
pub struct AutomaticSpeechRecognitionModel {
    backend: Box<dyn backend::AutomaticSpeechRecognitionBackend>,
}

#[cfg(feature = "whisper")]
impl AutomaticSpeechRecognitionModel {
    pub fn new(model: impl Into<PathBuf> + 'static) -> Result<Self> {
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

#[cfg(feature = "embeddings")]
pub struct EmbeddingsModel {
    backend: Box<dyn backend::EmbeddingsBackend>,
}

#[cfg(feature = "embeddings")]
impl EmbeddingsModel {
    pub fn new(options: options::EmbeddingsOptions) -> Result<Self> {
        let backend = backend::init_embeddings_backend(options)?;
        Ok(Self { backend: backend })
    }

    pub fn encode(&mut self, text: String) -> Result<Vec<f32>> {
        Ok(self.backend.encode(text)?)
    }
}

#[cfg(feature = "tts")]
pub struct TextToSpeechModel {
    backend: Box<dyn backend::TextToSpeechBackend>,
}

#[cfg(feature = "tts")]
impl TextToSpeechModel {
    pub fn new(options: options::TTSOptions) -> anyhow::Result<Self> {
        let backend = backend::init_text_to_speech_backend(options)?;
        anyhow::Ok(Self {backend: backend})
    }

    pub fn train(&mut self, ref_samples: Vec<f32>) -> anyhow::Result<()> {
        anyhow::Ok(self.backend.train(ref_samples)?)
    }

    pub fn predict(&mut self, text: String) -> anyhow::Result<Vec<f32>> {
        anyhow::Ok(self.backend.predict(text)?)
    }
}
