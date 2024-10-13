use crate::options::TextToImageOptions;

use super::TextToImageBackend;

use candle_core::{
    Device as CandleDevice,
    Tensor as CandleTensor,
    DType as CandleDType,
    Module, D,
};
use candle_transformers::models::stable_diffusion;
use stable_diffusion::{vae::AutoEncoderKL, unet_2d::UNet2DConditionModel, StableDiffusionConfig};
use tokenizers::Tokenizer;
use hf_hub::api::sync::Api;
use anyhow::Error as E;

pub struct StableDiffusionBackend {
    n_steps: usize,
    sd_config: StableDiffusionConfig,
    vae: AutoEncoderKL,
    vae_scale: f64,
    unet: UNet2DConditionModel,
    tokenizer: Option<String>,
    device: CandleDevice,
    samples_count: usize,
}

impl StableDiffusionBackend {
    pub fn new(options: TextToImageOptions) -> anyhow::Result<Self> {
        let n_steps = 1;
        let sd_config = StableDiffusionConfig::sdxl_turbo(
            None,
            None,
            None,
        );
        let device = if options.cpu {
            CandleDevice::Cpu
        } else {
            CandleDevice::cuda_if_available(0).unwrap()
        };

        let vae_weights = ModelFile::Vae.get(None)?;
        let vae = sd_config.build_vae(
            vae_weights, &device, CandleDType::F32
        )?;
        let vae_scale = 0.13025;

        let unet_weights = ModelFile::Unet.get(None)?;
        let unet = sd_config.build_unet(
            unet_weights, &device, 4, false, CandleDType::F32)?;

        Ok(
            Self {
                n_steps: n_steps,
                sd_config: sd_config,
                vae: vae,
                vae_scale: vae_scale,
                unet: unet,
                tokenizer: options.tokenizer.clone(),
                device: device,
                samples_count: options.samples_count,
            }
        )
    }
}

impl TextToImageBackend for StableDiffusionBackend {
    fn generate(&mut self, prompt: String) -> anyhow::Result<CandleTensor> {
        let which =  vec![true, false];
        let text_embeddings = which
            .iter()
            .map(|first| {
                text_embeddings(
                    &prompt,
                    self.tokenizer.clone(),
                    &self.sd_config,
                    &self.device,
                    *first,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let text_embeddings = CandleTensor::cat(&text_embeddings, D::Minus1)?;
        let text_embeddings = text_embeddings.repeat((self.samples_count, 1, 1))?;
        println!("{text_embeddings:?}");

        let t_start = 0;
        let scheduler = self.sd_config.build_scheduler(self.n_steps)?;

        let timesteps = scheduler.timesteps();
        let latents = CandleTensor::randn(
            0f32,
            1f32,
            (self.samples_count, 4, self.sd_config.height / 8, self.sd_config.width / 8),
            &self.device,
        )?;
        // scale the initial noise by the standard deviation required by the scheduler
        let latents = (latents * scheduler.init_noise_sigma())?;
        let mut latents = latents.to_dtype(CandleDType::F32)?;

        println!("starting sampling");
        for (timestep_index, &timestep) in timesteps.iter().enumerate() {
            if timestep_index < t_start {
                continue;
            }
            let start_time = std::time::Instant::now();
            let latent_model_input = latents.clone();

            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;
            let noise_pred =
                self.unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

            latents = scheduler.step(&noise_pred, timestep, &latents)?;
            let dt = start_time.elapsed().as_secs_f32();
            println!("step {}/{} done, {:.2}s", timestep_index + 1, self.n_steps, dt);
        }

        println!("Generating the final image for sample");
        let images = self.vae.decode(&(latents / self.vae_scale)?)?;
        let images = ((images / 2.)? + 0.5)?.to_device(&CandleDevice::Cpu)?;
        let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(CandleDType::U8)?;
        Ok(images)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    Tokenizer,
    Tokenizer2,
    Clip,
    Clip2,
    Unet,
    Vae,
}


impl ModelFile {
    fn get(&self, filename: Option<String>) -> anyhow::Result<std::path::PathBuf> {
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let (repo, path) = match self {
                    Self::Tokenizer => {
                        ("openai/clip-vit-large-patch14", "tokenizer.json")
                    }
                    Self::Tokenizer2 => {
                        ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json")
                    }
                    Self::Clip => ("stabilityai/sdxl-turbo", "text_encoder/model.safetensors"),
                    Self::Clip2 => ("stabilityai/sdxl-turbo", "text_encoder_2/model.safetensors"),
                    Self::Unet => ("stabilityai/sdxl-turbo", "unet/diffusion_pytorch_model.safetensors"),
                    Self::Vae => {
                        ("stabilityai/sdxl-turbo", "vae/diffusion_pytorch_model.safetensors")
                    }
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn text_embeddings(
    prompt: &str,
    tokenizer: Option<String>,
    sd_config: &stable_diffusion::StableDiffusionConfig,
    device: &CandleDevice,
    first: bool,
) -> anyhow::Result<CandleTensor> {
    let tokenizer_file = if first {
        ModelFile::Tokenizer
    } else {
        ModelFile::Tokenizer2
    };
    let tokenizer = tokenizer_file.get(tokenizer)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let pad_id = match &sd_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    println!("Running with prompt \"{prompt}\".");
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    if tokens.len() > sd_config.clip.max_position_embeddings {
        anyhow::bail!(
            "the prompt is too long, {} > max-tokens ({})",
            tokens.len(),
            sd_config.clip.max_position_embeddings
        )
    }
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = CandleTensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    println!("Building the Clip transformer.");
    let clip_weights_file = if first {
        ModelFile::Clip
    } else {
        ModelFile::Clip2
    };
    let clip_weights = clip_weights_file.get(None)?;
    let clip_config = if first {
        &sd_config.clip
    } else {
        sd_config.clip2.as_ref().unwrap()
    };
    let text_model =
        stable_diffusion::build_clip_transformer(clip_config, clip_weights, device, CandleDType::F32)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let text_embeddings = text_embeddings.to_dtype(CandleDType::F32)?;
    Ok(text_embeddings)
}
