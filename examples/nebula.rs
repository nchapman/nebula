#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod model;
mod text_generation;
mod token_output_stream;

use anyhow::{bail, Error as E, Ok, Result};
use clap::Parser;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::utils::{cuda_is_available, metal_is_available};

use model::Model;
use text_generation::TextGeneration;

use candle_transformers::models::mistral::{Config as MistralConfig, Model as Mistral};
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::models::quantized_mistral::Model as QMistral;

/// Loads the safetensors files for a model from the hub based on a json index file.
pub fn hub_load_safetensors<P: AsRef<std::path::Path>>(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: P,
) -> Result<Vec<std::path::PathBuf>> {
    let json_file = std::fs::File::open(json_file.as_ref())?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(safetensors_files)
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    model_path: Option<String>,

    #[arg(long)]
    quantized: bool,

    #[arg(long)]
    llama: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

impl Args {
    fn tokenizer(&self) -> Result<Tokenizer> {
        let tokenizer_path = match &self.tokenizer_file {
            Some(config) => std::path::PathBuf::from(config),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let repo = "mistralai/Mistral-7B-v0.1";
                let api = api.model(repo.to_string());
                api.get("tokenizer.json")?
            }
        };
        Tokenizer::from_file(tokenizer_path).map_err(E::msg)
    }

    fn device(&self) -> Result<Device> {
        if self.cpu {
            Ok(Device::Cpu)
        } else if cuda_is_available() {
            Ok(Device::new_cuda(0)?)
        } else if metal_is_available() {
            Ok(Device::new_metal(0)?)
        } else {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                println!(
                    "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
                );
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                println!(
                    "Running on CPU, to run on GPU, build this example with `--features cuda`"
                );
            }
            Ok(Device::Cpu)
        }
    }

    fn model(&self, device: &Device) -> Result<Model> {
        let api = Api::new()?;
        let model_id = match &self.model_id {
            Some(model_id) => model_id,
            None => {
                if self.quantized {
                    "DanielClough/Candle_Mistral-7B-Instruct-v0.2"
                } else {
                    "mistralai/Mistral-7B-v0.1"
                }
            }
        };
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            self.revision.clone(),
        ));

        let model_path = match &self.model_path {
            Some(file) => repo.get(file)?,
            None => {
                if self.quantized {
                    repo.get("Candle_Mistral-7B-Instruct-v0.2_q4k.gguf")?
                } else {
                    repo.get("model.safetensors.index.json")?
                }
            }
        };

        let config: MistralConfig = MistralConfig::config_7b_v0_1(self.use_flash_attn);
        let model = if self.quantized {
            if self.llama {
                let mut file = std::fs::File::open(&model_path)?;
                let model = match model_path.extension().and_then(|v| v.to_str()) {
                    Some("gguf") => {
                        let model = gguf_file::Content::read(&mut file)
                            .map_err(|e| e.with_path(model_path))?;
                        ModelWeights::from_gguf(model, &mut file, device)?
                    }
                    Some("ggml" | "bin") | Some(_) | None => {
                        let model = ggml_file::Content::read(&mut file, device)
                            .map_err(|e| e.with_path(model_path))?;
                        ModelWeights::from_ggml(model, 8)?
                    }
                };
                Model::LlmaMistral(model)
            } else {
                let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                    &model_path,
                    device,
                )?;
                let model = QMistral::new(&config, vb)?;
                Model::QuantizedMistral(model)
            }
        } else {
            let dtype = if device.is_cuda() {
                DType::BF16
            } else {
                DType::F32
            };
            let safetensors = hub_load_safetensors(&repo, &model_path)?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensors, dtype, device)? };
            let model = Mistral::new(&config, vb)?;
            Model::Mistral(model)
        };
        Ok(model)
    }
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "cuda: {}, avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::cuda_is_available(),
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let tokenizer = args.tokenizer().map_err(E::msg)?;
    println!("retrieved the files in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let device = args.device()?;
    let model = args.model(&device)?;
    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    pipeline.run(args.prompt, args.sample_len)?;
    Ok(())
}
