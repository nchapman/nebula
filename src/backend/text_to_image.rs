use crate::options::TextToImageOptions;

use super::TextToImageBackend;

use candle_core::{
    Device as CandleDevice,
    Tensor as CandleTensor,
    DType as CandleDType,
    Module, D,
};
use candle_transformers::models::stable_diffusion;
use candle_transformers::models::wuerstchen;
use stable_diffusion::{vae::AutoEncoderKL, unet_2d::UNet2DConditionModel, StableDiffusionConfig};
use tokenizers::Tokenizer;
use hf_hub::api::sync::Api;
use anyhow::Error as E;

const PRIOR_GUIDANCE_SCALE: f64 = 4.0;
const RESOLUTION_MULTIPLE: f64 = 42.67;
const LATENT_DIM_SCALE: f64 = 10.67;
const PRIOR_CIN: usize = 16;
const DECODER_CIN: usize = 4;

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

        let vae_weights = StableDiffusionModelFile::Vae.get(None)?;
        let vae = sd_config.build_vae(
            vae_weights, &device, CandleDType::F32
        )?;
        let vae_scale = 0.13025;

        let unet_weights = StableDiffusionModelFile::Unet.get(None)?;
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
                text_embeddings_stable_diffusion(
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
enum StableDiffusionModelFile {
    Tokenizer,
    Tokenizer2,
    Clip,
    Clip2,
    Unet,
    Vae,
}


impl StableDiffusionModelFile {
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
fn text_embeddings_stable_diffusion(
    prompt: &str,
    tokenizer: Option<String>,
    sd_config: &stable_diffusion::StableDiffusionConfig,
    device: &CandleDevice,
    first: bool,
) -> anyhow::Result<CandleTensor> {
    let tokenizer_file = if first {
        StableDiffusionModelFile::Tokenizer
    } else {
        StableDiffusionModelFile::Tokenizer2
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
        StableDiffusionModelFile::Clip
    } else {
        StableDiffusionModelFile::Clip2
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

pub struct WuerstchenBackend {
    tokenizer: Option<String>,
    device: CandleDevice,
    height: i32,
    width: i32,
    prior: wuerstchen::prior::WPrior,
    vqgan: wuerstchen::paella_vq::PaellaVQ,
    decoder: wuerstchen::diffnext::WDiffNeXt,
    samples_count: usize,
}

impl WuerstchenBackend {
    pub fn new(options: TextToImageOptions) -> anyhow::Result<Self> {
        let device = if options.cpu {
            CandleDevice::Cpu
        } else {
            CandleDevice::cuda_if_available(0).unwrap()
        };
        let height = 1024;
        let width = 1024;

        println!("Building the prior.");
        let prior = {
            let file = WuerstchenModelFile::Prior.get(None)?;
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[file], CandleDType::F32, &device)?
            };
            wuerstchen::prior::WPrior::new(
                /* c_in */ PRIOR_CIN,
                /* c */ 1536,
                /* c_cond */ 1280,
                /* c_r */ 64,
                /* depth */ 32,
                /* nhead */ 24,
                false,
                vb,
            )?
        };

        println!("Building the vqgan.");
        let vqgan = {
            let file = WuerstchenModelFile::VqGan.get(None)?;
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[file], CandleDType::F32, &device)?
            };
            wuerstchen::paella_vq::PaellaVQ::new(vb)?
        };

        println!("Building the decoder.");

        // https://huggingface.co/warp-ai/wuerstchen/blob/main/decoder/config.json
        let decoder = {
            let file = WuerstchenModelFile::Decoder.get(None)?;
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[file], CandleDType::F32, &device)?
            };
            wuerstchen::diffnext::WDiffNeXt::new(
                /* c_in */ DECODER_CIN,
                /* c_out */ DECODER_CIN,
                /* c_r */ 64,
                /* c_cond */ 1024,
                /* clip_embd */ 1024,
                /* patch_size */ 2,
                false,
                vb,
            )?
        };

        Ok(
            Self {
                tokenizer: options.tokenizer,
                device: device,
                height: height,
                width: width,
                prior: prior,
                vqgan: vqgan,
                decoder: decoder,
                samples_count: 1,
            }
        )
    }
}

impl TextToImageBackend for WuerstchenBackend {
    fn generate(&mut self, prompt: String) -> anyhow::Result<CandleTensor> {
        let prior_text_embeddings = {
            let tokenizer = WuerstchenModelFile::PriorTokenizer.get(None)?;
            let weights = WuerstchenModelFile::PriorClip.get(None)?;
            text_embeddings_wuerstchen(
                &prompt,
                Some(""),
                tokenizer.clone(),
                weights,
                stable_diffusion::clip::Config::wuerstchen_prior(),
                &self.device,
            )?
        };
        println!("generated prior text embeddings {prior_text_embeddings:?}");

        let text_embeddings = {
            let tokenizer = WuerstchenModelFile::Tokenizer.get(self.tokenizer.clone())?;
            let weights = WuerstchenModelFile::Clip.get(None)?;
            text_embeddings_wuerstchen(
                &prompt,
                None,
                tokenizer.clone(),
                weights,
                stable_diffusion::clip::Config::wuerstchen(),
                &self.device,
            )?
        };
        println!("generated text embeddings {text_embeddings:?}");

        let image_embeddings = {
            // https://huggingface.co/warp-ai/wuerstchen-prior/blob/main/prior/config.json
            let latent_height = (self.height as f64 / RESOLUTION_MULTIPLE).ceil() as usize;
            let latent_width = (self.width as f64 / RESOLUTION_MULTIPLE).ceil() as usize;
            let mut latents = CandleTensor::randn(
                0f32,
                1f32,
                (self.samples_count, PRIOR_CIN, latent_height, latent_width),
                &self.device,
            )?;
            let prior_scheduler = wuerstchen::ddpm::DDPMWScheduler::new(60, Default::default())?;
            let timesteps = prior_scheduler.timesteps();
            let timesteps = &timesteps[..timesteps.len() - 1];
            println!("prior denoising");
            for (index, &t) in timesteps.iter().enumerate() {
                let start_time = std::time::Instant::now();
                let latent_model_input = CandleTensor::cat(&[&latents, &latents], 0)?;
                let ratio = (CandleTensor::ones(2, CandleDType::F32, &self.device)? * t)?;
                let noise_pred = self.prior.forward(&latent_model_input, &ratio, &prior_text_embeddings)?;
                let noise_pred = noise_pred.chunk(2, 0)?;
                let (noise_pred_text, noise_pred_uncond) = (&noise_pred[0], &noise_pred[1]);
                let noise_pred = (noise_pred_uncond
                    + ((noise_pred_text - noise_pred_uncond)? * PRIOR_GUIDANCE_SCALE)?)?;
                latents = prior_scheduler.step(&noise_pred, t, &latents)?;
                let dt = start_time.elapsed().as_secs_f32();
                println!("step {}/{} done, {:.2}s", index + 1, timesteps.len(), dt);
            }
            ((latents * 42.)? - 1.)?
        };

        // https://huggingface.co/warp-ai/wuerstchen/blob/main/model_index.json
        let latent_height = (image_embeddings.dim(2)? as f64 * LATENT_DIM_SCALE) as usize;
        let latent_width = (image_embeddings.dim(3)? as f64 * LATENT_DIM_SCALE) as usize;

        let mut latents = CandleTensor::randn(
            0f32,
            1f32,
            (self.samples_count, DECODER_CIN, latent_height, latent_width),
            &self.device,
        )?;

        println!("diffusion process with prior {image_embeddings:?}");
        let scheduler = wuerstchen::ddpm::DDPMWScheduler::new(12, Default::default())?;
        let timesteps = scheduler.timesteps();
        let timesteps = &timesteps[..timesteps.len() - 1];
        for (index, &t) in timesteps.iter().enumerate() {
            let start_time = std::time::Instant::now();
            let ratio = (CandleTensor::ones(1, CandleDType::F32, &self.device)? * t)?;
            let noise_pred =
                self.decoder.forward(&latents, &ratio, &image_embeddings, Some(&text_embeddings))?;
            latents = scheduler.step(&noise_pred, t, &latents)?;
            let dt = start_time.elapsed().as_secs_f32();
            println!("step {}/{} done, {:.2}s", index + 1, timesteps.len(), dt);
        }
        println!("Generating the final image for sample");
        let images = self.vqgan.decode(&(&latents * 0.3764)?)?;
        let images = (images.clamp(0f32, 1f32)? * 255.)?
            .to_dtype(CandleDType::U8)?;
        Ok(images)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WuerstchenModelFile {
    Tokenizer,
    PriorTokenizer,
    Clip,
    PriorClip,
    Decoder,
    VqGan,
    Prior,
}

impl WuerstchenModelFile {
    fn get(&self, filename: Option<String>) -> anyhow::Result<std::path::PathBuf> {
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let repo_main = "warp-ai/wuerstchen";
                let repo_prior = "warp-ai/wuerstchen-prior";
                let (repo, path) = match self {
                    Self::Tokenizer => (repo_main, "tokenizer/tokenizer.json"),
                    Self::PriorTokenizer => (repo_prior, "tokenizer/tokenizer.json"),
                    Self::Clip => (repo_main, "text_encoder/model.safetensors"),
                    Self::PriorClip => (repo_prior, "text_encoder/model.safetensors"),
                    Self::Decoder => (repo_main, "decoder/diffusion_pytorch_model.safetensors"),
                    Self::VqGan => (repo_main, "vqgan/diffusion_pytorch_model.safetensors"),
                    Self::Prior => (repo_prior, "prior/diffusion_pytorch_model.safetensors"),
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}

fn text_embeddings_wuerstchen(
    prompt: &str,
    uncond_prompt: Option<&str>,
    tokenizer: std::path::PathBuf,
    clip_weights: std::path::PathBuf,
    clip_config: stable_diffusion::clip::Config,
    device: &CandleDevice,
) -> anyhow::Result<CandleTensor> {
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let pad_id = match &clip_config.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    println!("Running with prompt \"{prompt}\".");
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let tokens_len = tokens.len();
    while tokens.len() < clip_config.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = CandleTensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    println!("Building the clip transformer.");
    let text_model =
        stable_diffusion::build_clip_transformer(&clip_config, clip_weights, device, CandleDType::F32)?;
    let text_embeddings = text_model.forward_with_mask(&tokens, tokens_len - 1)?;
    match uncond_prompt {
        None => Ok(text_embeddings),
        Some(uncond_prompt) => {
            let mut uncond_tokens = tokenizer
                .encode(uncond_prompt, true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            let uncond_tokens_len = uncond_tokens.len();
            while uncond_tokens.len() < clip_config.max_position_embeddings {
                uncond_tokens.push(pad_id)
            }
            let uncond_tokens = CandleTensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;

            let uncond_embeddings =
                text_model.forward_with_mask(&uncond_tokens, uncond_tokens_len - 1)?;
            let text_embeddings = CandleTensor::cat(&[text_embeddings, uncond_embeddings], 0)?;
            Ok(text_embeddings)
        }
    }
}
