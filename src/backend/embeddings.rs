use crate::{options::EmbeddingsOptions, Result};

use super::EmbeddingsBackend;

use anyhow::Error as E;
use std::path::{Path, PathBuf};
use hf_hub::{api::sync::Api, Repo, RepoType};
use candle_transformers::models::jina_bert::{BertModel, Config as JinaBertConfig};
use candle_transformers::models::t5::{T5EncoderModel, Config as T5Config};
use candle_core::{DType, Device, Tensor, Result as CandleResult};
use candle_nn::{Module, VarBuilder};
use candle_examples::hub_load_safetensors;
use tokenizers::Tokenizer;

pub struct JinaBertBackend {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl JinaBertBackend {
    pub fn new(options: EmbeddingsOptions) -> Result<Self> {
        let model_source = match options.model {
            Some(model_string) => {
                if Path::new(&model_string).exists() {
                    PathBuf::from(model_string)
                } else {
                    get_jina_bert_model_source_from_hugging_face_repo(model_string)
                }
            },
            None => {
                get_jina_bert_model_source_from_hugging_face_repo(
                    "jinaai/jina-embeddings-v2-base-en".to_string()
                )
            }
        };
        let tokenizer_source = match options.tokenizer {
            Some(tokenizer_string) => {
                if Path::new(&tokenizer_string).exists() {
                    PathBuf::from(tokenizer_string)
                } else {
                    get_jina_bert_tokenizer_source_from_hugging_face_repo(tokenizer_string)
                }
             },
             None => {
                get_jina_bert_tokenizer_source_from_hugging_face_repo(
                    "sentence-transformers/all-MiniLM-L6-v2".to_string()
                )
             }
        };
        let device = if options.cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap()
        };
        let config = JinaBertConfig::v2_base();
        let tokenizer = Tokenizer::from_file(tokenizer_source)
            .map_err(E::msg)
            .unwrap();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_source], DType::F32, &device
            ).unwrap() 
        };
        let model = BertModel::new(vb, &config).unwrap();
        Ok(Self { model, tokenizer, device })
    }
}

fn get_jina_bert_model_source_from_hugging_face_repo(key: String) -> PathBuf {
    let model_source = Api::new()
        .unwrap()
        .repo(Repo::new(
            key,
            RepoType::Model,
        ))
        .get("model.safetensors")
        .unwrap();
    model_source
}

fn get_jina_bert_tokenizer_source_from_hugging_face_repo(key: String) -> PathBuf {
    let tokenizer_source = Api::new()
        .unwrap()
        .repo(Repo::new(
            key,
            RepoType::Model,
        ))
        .get("tokenizer.json")
        .unwrap();
    tokenizer_source
}

impl EmbeddingsBackend for JinaBertBackend {
    fn predict(
        &mut self,
        text: String
    ) -> Result<Vec<f32>> {
        let pseudo_sentences = vec![text];
        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            self.tokenizer.with_padding(Some(pp));
        }
        let tokens = self.tokenizer
            .encode_batch(pseudo_sentences, true)
            .map_err(E::msg)
            .unwrap();
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Tensor::new(tokens.as_slice(), &self.device)
            })
            .collect::<CandleResult<Vec<_>>>()
            .unwrap();

        let token_ids = Tensor::stack(&token_ids, 0).unwrap();
        println!("running inference on batch {:?}", token_ids.shape());
        let embedding = self.model.forward(&token_ids).unwrap();
        println!("generated embeddings {:?}", embedding.shape());
        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embedding
            .dims3()
            .unwrap();
        let embedding = (embedding.sum(1).unwrap() / (n_tokens as f64))
            .unwrap();
        println!("pooled embeddings {:?}", embedding.shape());

        let flat_embedding = embedding.flatten_all().unwrap();
        let embedding_size = flat_embedding.dims()[0];

        let mut embedding_vec: Vec<f32> = Vec::new();
        for i in 0..embedding_size {
            embedding_vec.push(
                flat_embedding
                    .get(i)
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
                    as f32
            )
        }
        Ok(embedding_vec)
    }
}

pub struct T5Backend {
    model: T5EncoderModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl T5Backend {
    pub fn new(options: EmbeddingsOptions) -> Result<Self> {
        let revision = match &options.revision {
            Some(user_revision) => user_revision,
            None => "main"
        };
        let model_id = match &options.model {
            Some(model_string) => model_string,
            None => "t5-small"
        };
        let revision = match &options.model {
            Some(_m) => revision,
            None => "refs/pr/15"
        };
        let model_source = match &options.model {
            Some(model_string) => {
                if Path::new(model_string).exists() {
                    model_string
                        .to_string()
                        .split(',')
                        .map(|v| v.into())
                        .collect::<Vec<_>>()
                } else {
                    get_t5_model_source_from_hugging_face_repo(
                        model_string.to_string(), revision.to_string()
                    )
                }
            },
            None => {
                get_t5_model_source_from_hugging_face_repo(
                    model_id.to_string(), revision.to_string()
                )
            }
        };
        let tokenizer_source = match &options.tokenizer {
            Some(tokenizer_string) => {
                if Path::new(tokenizer_string).exists() {
                    PathBuf::from(tokenizer_string)
                } else {
                    get_t5_tokenizer_source_from_hugging_face_repo(
                        tokenizer_string.to_string(), revision.to_string()
                    )
                }
            }
            None => {
                get_t5_tokenizer_source_from_hugging_face_repo(
                    model_id.to_string(), revision.to_string()
                )
            }
        };
        let config_source = match &options.config {
            Some(config_string) => {
                if Path::new(config_string).exists() {
                    PathBuf::from(config_string)
                } else {
                    get_t5_config_source_from_hugging_face_repo(
                        config_string.to_string(), revision.to_string()
                    )
                }
            },
            None => {
                get_t5_config_source_from_hugging_face_repo(
                    model_id.to_string(), revision.to_string()
                )
            }
        };
        let config = std::fs::read_to_string(config_source).unwrap();
        let config: T5Config = serde_json::from_str(&config).unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_source)
            .map_err(E::msg)
            .unwrap();
        let device = if options.cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap()
        };
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&model_source, DType::F32, &device).unwrap()
        };
        Ok(
            Self {
                model: T5EncoderModel::load(vb, &config).unwrap(),
                tokenizer: tokenizer,
                device: device
            }
        )
    }
}

fn get_t5_model_source_from_hugging_face_repo(key: String, revision: String) -> Vec<PathBuf> {
    let repo = Repo::with_revision(key.clone(), RepoType::Model, revision);
    let api = Api::new().unwrap();
    let repo = api.repo(repo);
    if key == "google/flan-t5-xxl".to_string() || key == "google/flan-ul2".to_string() {
        hub_load_safetensors(&repo, "model.safetensors.index.json").unwrap()
    } else {
        vec![repo.get("model.safetensors").unwrap()]
    }
}

fn get_t5_tokenizer_source_from_hugging_face_repo(key: String, revision: String) -> PathBuf {
    let repo = Repo::with_revision(key.clone(), RepoType::Model, revision);
    let api = Api::new().unwrap();
    let repo = api.repo(repo);
    if key == "google/mt5-base".to_string() {
        api
            .model("lmz/mt5-tokenizers".into())
            .get("mt5-base.tokenizer.json")
            .unwrap()
    } else if key == "google/mt5-small".to_string() {
        api
            .model("lmz/mt5-tokenizers".into())
            .get("mt5-small.tokenizer.json")
            .unwrap()
    } else if key == "google/mt5-large".to_string() {
        api
            .model("lmz/mt5-tokenizers".into())
            .get("mt5-large.tokenizer.json")
            .unwrap()
    } else {
        repo.get("tokenizer.json").unwrap()
    }
}

fn get_t5_config_source_from_hugging_face_repo(key: String, revision: String) -> PathBuf {
    println!("Config...");
    println!("Key: {}", key);
    println!("Revision: {}", revision);
    let repo = Repo::with_revision(key.clone(), RepoType::Model, revision);
    let api = Api::new().unwrap();
    let repo = api.repo(repo);
    repo.get("config.json").unwrap()
}

impl EmbeddingsBackend for T5Backend {
    fn predict(
            &mut self,
            text: String
    ) -> Result<Vec<f32>> {
        let tokenizer = self.tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)
            .unwrap();
        let tokens = tokenizer
            .encode(text, true)
            .map_err(E::msg)
            .unwrap()
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], &self.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let embedding = self.model
            .forward(&token_ids)
            .unwrap();
        let (_n_sentence, n_tokens, _hidden_size) = embedding
            .dims3()
            .unwrap();
        let embedding = (embedding.sum(1).unwrap() / (n_tokens as f64))
            .unwrap();

        let flat_embedding = embedding.flatten_all().unwrap();
        let embedding_size = flat_embedding.dims()[0];

        let mut embedding_vec: Vec<f32> = Vec::new();
        for i in 0..embedding_size {
            embedding_vec.push(
                flat_embedding
                    .get(i)
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
                    as f32
            )
        }
        Ok(embedding_vec)
    }
}