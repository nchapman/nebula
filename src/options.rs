use base64::prelude::*;
use llama_cpp::sample::SamplingParams;
use serde::{de::Visitor, Deserialize, Deserializer};
use serde_json::Value;
use std::{fmt::Display, io::Read};

fn default_i32_minus_1() -> i32 {
    -1
}

fn default_i32_64() -> i32 {
    64
}

fn default_i32_40() -> i32 {
    40
}

fn default_f32_0_95() -> f32 {
    0.95
}

fn default_f32_0_05() -> f32 {
    0.05
}

fn default_f32_1_0() -> f32 {
    1.0
}

fn default_f32_5_0() -> f32 {
    5.0
}

fn default_f32_0_8() -> f32 {
    0.8
}

fn default_f32_0_1() -> f32 {
    0.1
}

fn default_usize_2048() -> usize {
    2048
}

fn default_samplers() -> Vec<SamplerType> {
    vec![
        SamplerType::TopK,
        SamplerType::TfsZ,
        SamplerType::TypicalP,
        SamplerType::TopP,
        SamplerType::MinP,
        SamplerType::Temperature,
    ]
}

#[derive(bon::Builder, serde::Deserialize)]
pub struct ModelOptions {
    #[builder(default)]
    #[serde(default)]
    pub cpu: bool,
    #[builder(default = -1)]
    #[serde(default = "default_i32_minus_1")]
    pub n_gpu_layers: i32,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self::builder().build()
    }
}

#[derive(Clone, Debug, serde::Deserialize, PartialEq)]
pub enum Role {
    #[serde(alias = "system")]
    System,
    #[serde(alias = "assistant")]
    Assistant,
    #[serde(alias = "user")]
    User,
}

impl Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Image(pub Vec<u8>);

struct ImageVisitor;

impl<'de> Visitor<'de> for ImageVisitor {
    type Value = Image;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("string with either file name or base64 encoded image")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        log::debug!("{v}");
        match std::fs::File::open(v) {
            Ok(mut f) => {
                let mut image_bytes = vec![];
                match f.read_to_end(&mut image_bytes) {
                    Ok(_ib) => return Ok(Image(image_bytes)),
                    Err(e) => log::warn!("{e}"),
                }
            }
            Err(e) => log::warn!("{e}"),
        }
        match BASE64_STANDARD.decode(v) {
            Ok(ss) => return Ok(Image(ss)),
            Err(e) => log::warn!("{e}"),
        }
        Err(E::custom("can`t parse image`"))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        log::debug!("{value}");
        match std::fs::File::open(&value) {
            Ok(mut f) => {
                let mut image_bytes = vec![];
                match f.read_to_end(&mut image_bytes) {
                    Ok(_ib) => return Ok(Image(image_bytes)),
                    Err(e) => log::warn!("{e}"),
                }
            }
            Err(e) => log::warn!("{e}"),
        }
        match BASE64_STANDARD.decode(&value) {
            Ok(ss) => return Ok(Image(ss)),
            Err(e) => log::warn!("{e}"),
        }
        Err(E::custom("can`t parse image`"))
    }
}

impl<'de> Deserialize<'de> for Image {
    fn deserialize<D>(deserializer: D) -> Result<Image, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(ImageVisitor)
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Message {
    pub content: String,
    pub role: Role,
    #[serde(default)]
    pub images: Vec<Image>,
}

impl TryFrom<&str> for Message {
    type Error = crate::error::Error;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(serde_json::from_str::<Message>(value)?)
    }
}

impl TryFrom<Value> for Message {
    type Error = crate::error::Error;
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Ok(serde_json::from_value::<Message>(value)?)
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub enum SamplerType {
    None = 0,
    TopK = 1,
    TopP = 2,
    MinP = 3,
    TfsZ = 4,
    TypicalP = 5,
    Temperature = 6,
}

impl From<SamplerType> for llama_cpp::sample::SamplerType {
    fn from(val: SamplerType) -> Self {
        match val {
            SamplerType::None => llama_cpp::sample::SamplerType::None,
            SamplerType::TopK => llama_cpp::sample::SamplerType::TopK,
            SamplerType::TopP => llama_cpp::sample::SamplerType::TopP,
            SamplerType::MinP => llama_cpp::sample::SamplerType::MinP,
            SamplerType::TfsZ => llama_cpp::sample::SamplerType::TfsZ,
            SamplerType::TypicalP => llama_cpp::sample::SamplerType::TypicalP,
            SamplerType::Temperature => llama_cpp::sample::SamplerType::Temperature,
        }
    }
}

pub type TokenCallback = dyn Fn(String) -> bool + Send + 'static;

#[derive(Clone, bon::Builder, serde::Deserialize)]
pub struct PredictOptions {
    #[builder(default)]
    #[serde(default)]
    pub seed: u32,
    #[builder(default = default_i32_64())]
    #[serde(default = "default_i32_64")]
    pub n_prev: i32,
    #[builder(default)]
    #[serde(default)]
    pub n_probs: i32,
    #[builder(default)]
    #[serde(default)]
    pub min_keep: i32,
    #[builder(default = default_i32_40())]
    #[serde(default = "default_i32_40")]
    pub top_k: i32,
    #[builder(default = default_f32_0_95())]
    #[serde(default = "default_f32_0_95")]
    pub top_p: f32,
    #[builder(default = default_f32_0_05())]
    #[serde(default = "default_f32_0_05")]
    pub min_p: f32,
    #[builder(default = default_f32_1_0())]
    #[serde(default = "default_f32_1_0")]
    pub tfs_z: f32,
    #[builder(default = default_f32_1_0())]
    #[serde(default = "default_f32_1_0")]
    pub typ_p: f32,
    #[builder(default = default_f32_0_8())]
    #[serde(default = "default_f32_0_8")]
    pub temp: f32,
    #[builder(default)]
    #[serde(default)]
    pub dynatemp_range: f32,
    #[builder(default = default_f32_1_0())]
    #[serde(default = "default_f32_1_0")]
    pub dynatemp_exponent: f32,
    #[builder(default = default_i32_64())]
    #[serde(default = "default_i32_64")]
    pub penalty_last_n: i32,
    #[builder(default = default_f32_1_0())]
    #[serde(default = "default_f32_1_0")]
    pub penalty_repeat: f32,
    #[builder(default)]
    #[serde(default)]
    pub penalty_freq: f32,
    #[builder(default)]
    #[serde(default)]
    pub penalty_present: f32,
    #[builder(default)]
    #[serde(default)]
    pub mirostat: i32,
    #[builder(default = default_f32_5_0())]
    #[serde(default = "default_f32_5_0")]
    pub mirostat_tau: f32,
    #[builder(default = default_f32_0_1())]
    #[serde(default = "default_f32_0_1")]
    pub mirostat_eta: f32,
    #[builder(default)]
    #[serde(default)]
    pub penalize_nl: bool,
    #[builder(default)]
    #[serde(default)]
    pub ignore_eos: bool,
    #[builder(default = default_samplers())]
    #[serde(default = "default_samplers")]
    pub samplers: Vec<SamplerType>,
    #[builder(default)]
    #[serde(default)]
    pub grammar: String,
    #[serde(skip_deserializing)]
    pub token_callback: Option<std::sync::Arc<Box<TokenCallback>>>,
    pub max_len: Option<i32>,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl From<PredictOptions> for SamplingParams {
    fn from(val: PredictOptions) -> SamplingParams {
        SamplingParams {
            seed: val.seed,
            n_prev: val.n_prev,
            n_probs: val.n_probs,
            min_keep: val.min_keep,
            top_k: val.top_k,
            top_p: val.top_p,
            min_p: val.min_p,
            tfs_z: val.tfs_z,
            typ_p: val.typ_p,
            temp: val.temp,
            dynatemp_range: val.dynatemp_range,
            dynatemp_exponent: val.dynatemp_exponent,
            penalty_last_n: val.penalty_last_n,
            penalty_repeat: val.penalty_repeat,
            penalty_freq: val.penalty_freq,
            penalty_present: val.penalty_present,
            mirostat: val.mirostat,
            mirostat_tau: val.mirostat_tau,
            mirostat_eta: val.mirostat_eta,
            penalize_nl: val.penalize_nl,
            ignore_eos: val.ignore_eos,
            samplers: val.samplers.into_iter().map(|s| s.into()).collect(),
            grammar: val.grammar,
            logit_bias: vec![],
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize, bon::Builder)]
pub struct ContextOptions {
    #[builder(default)]
    #[serde(default)]
    pub seed: u32,
    #[builder(default = default_usize_2048())]
    #[serde(default = "default_usize_2048")]
    pub n_ctx: usize,
    #[builder(default = num_cpus::get())]
    #[serde(default = "num_cpus::get")]
    pub n_threads: usize,
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self::builder().build()
    }
}

#[derive(bon::Builder)]
pub struct NebulaOptions {
    #[builder(default = -1)]
    pub seed: i32,
    #[builder(default = num_cpus::get() as i32)]
    pub n_threads: i32,
    #[builder(default = -1)]
    pub n_threads_draft: i32,
    #[builder(default = -1)]
    pub n_threads_batch: i32,
    #[builder(default = -1)]
    pub n_threads_batch_draft: i32,
    #[builder(default = -1)]
    pub n_predict: i32,
    #[builder(default = 512)]
    pub n_ctx: i32,
    #[builder(default = 2048)]
    pub n_batch: i32,
    #[builder(default = 512)]
    pub n_ubatch: i32,
    #[builder(default)]
    pub n_keep: i32,
    #[builder(default = 5)]
    pub n_draft: i32,
    #[builder(default = -1)]
    pub n_chunks: i32,
    #[builder(default = 1)]
    pub n_parallel: i32,
    #[builder(default = 1)]
    pub n_sequences: i32,
    #[builder(default = 0.1)]
    pub p_split: f32,
    #[builder(default = -1)]
    pub n_gpu_layers: i32,
    #[builder(default = -1)]
    pub n_gpu_layers_draft: i32,
    //pub split_mode: SplitMode, //SplitMode::Layer
    #[builder(default = 0)]
    pub main_gpu: i32,
    #[builder(default = [0.0; 128])]
    pub tensor_split: [f32; 128],
    #[builder(default = 0)]
    pub n_beams: i32,
    #[builder(default = 1)]
    pub grp_attn_n: i32,
    #[builder(default = 512)]
    pub grp_attn_w: i32,
    #[builder(default = -1)]
    pub n_print: i32,
    #[builder(default = 0.0)]
    pub rope_freq_base: f32,
    #[builder(default = 0.0)]
    pub rope_freq_scale: f32,
    #[builder(default = -1.0)]
    pub yarn_ext_factor: f32,
    #[builder(default = 1.0)]
    pub yarn_attn_factor: f32,
    #[builder(default = 32.0)]
    pub yarn_beta_fast: f32,
    #[builder(default = 1.0)]
    pub yarn_beta_slow: f32,
    #[builder(default)]
    pub yarn_orig_ctx: i32,
    #[builder(default = 1.0)]
    pub defrag_thold: f32,
    //pub numa: NumaStrategy, //NumaStrategy::Disabled,
    //pub rope_scaling_type: RopeScalingType, //RopeScalingType::Unspecified,
    //pub pooling_type: PoolingType, //PoolingType::Unspecified,
    #[builder(default = PredictOptions::default())]
    pub sparams: PredictOptions,
}

#[cfg(feature = "whisper")]
pub struct AutomaticSpeechRecognitionOptions<'a> {
    pub n_threads: i32,
    pub translate: bool,
    pub language: &'a str,
    pub print_special: bool,
    pub print_progress: bool,
    pub print_realtime: bool,
    pub print_timestamps: bool,
}

#[cfg(feature = "whisper")]
impl<'a> AutomaticSpeechRecognitionOptions<'a> {
    pub fn with_n_threads(mut self, n_threads: i32) -> Self {
        self.n_threads = n_threads;
        self
    }

    pub fn with_translate(mut self, translate: bool) -> Self {
        self.translate = translate;
        self
    }

    pub fn with_language(mut self, language: &'a str) -> Self {
        self.language = language;
        self
    }

    pub fn with_print_special(mut self, print_special: bool) -> Self {
        self.print_special = print_special;
        self
    }

    pub fn with_print_progress(mut self, print_progress: bool) -> Self {
        self.print_progress = print_progress;
        self
    }

    pub fn with_print_realtime(mut self, print_realtime: bool) -> Self {
        self.print_realtime = print_realtime;
        self
    }

    pub fn with_print_timestamps(mut self, print_timestamps: bool) -> Self {
        self.print_timestamps = print_timestamps;
        self
    }
}

#[cfg(feature = "whisper")]
impl<'a> Default for AutomaticSpeechRecognitionOptions<'a> {
    fn default() -> Self {
        Self {
            n_threads: 1,
            translate: true,
            language: "en",
            print_special: false,
            print_progress: false,
            print_realtime: false,
            print_timestamps: false,
        }
    }
}

#[cfg(feature = "embeddings")]
#[derive(Debug)]
pub enum EmbeddingsModelType {
    T5,
    JinaBert,
    Bert,
}

#[cfg(feature = "embeddings")]
pub struct EmbeddingsOptions {
    pub cpu: bool,
    pub model_type: EmbeddingsModelType,
    pub tokenizer: Option<String>,
    pub model: Option<String>,
    pub revision: Option<String>,
    pub config: Option<String>,
}

#[cfg(feature = "embeddings")]
impl EmbeddingsOptions {
    pub fn with_cpu(mut self, cpu: bool) -> Self {
        self.cpu = cpu;
        self
    }

    pub fn with_model_type(mut self, model_type: EmbeddingsModelType) -> Self {
        self.model_type = model_type;
        self
    }

    pub fn with_tokenizer(mut self, tokenizer: Option<String>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    pub fn with_model(mut self, model: Option<String>) -> Self {
        self.model = model;
        self
    }

    pub fn with_revision(mut self, revision: Option<String>) -> Self {
        self.revision = revision;
        self
    }

    pub fn with_config(mut self, config: Option<String>) -> Self {
        self.config = config;
        self
    }
}

#[cfg(feature = "tts")]
pub enum TTSDevice {
    Cpu,
    Cuda,
}

#[cfg(feature = "tts")]
pub struct TTSOptions {
    pub device: TTSDevice,
}

#[cfg(feature = "tts")]
impl TTSOptions {
    pub fn with_device(mut self, device: TTSDevice) -> Self {
        self.device = device;
        self
    }
}

#[cfg(feature = "embeddings")]
impl Default for EmbeddingsOptions {
    fn default() -> Self {
        Self {
            cpu: true,
            model_type: EmbeddingsModelType::JinaBert,
            tokenizer: None,
            model: None,
            revision: None,
            config: None,
        }
    }
}

#[cfg(feature = "tts")]
impl Default for TTSOptions {
    fn default() -> Self {
        Self {
            device: TTSDevice::Cpu,
        }
    }
}
