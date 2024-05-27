use crate::{options::EmbeddingsOptions, Result};

use super::EmbeddingsBackend;

use anyhow::Error as E;
use std::path::{Path, PathBuf};
use hf_hub::{api::sync::Api, Repo, RepoType};
use candle_transformers::models::jina_bert::{BertModel, Config};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub struct JinaBertBackend {
    model: BertModel,
    tokenizer: tokenizers::Tokenizer,
}

impl JinaBertBackend {
    pub fn new(options: EmbeddingsOptions) -> Result<Self> {
        let model_source = match options.model {
            Some(model_string) => {
                if Path::new(&model_string).exists() {
                    PathBuf::from(model_string)
                } else {
                    get_model_source_from_hugging_face_repo(model_string)
                }
            },
            None => {
                get_model_source_from_hugging_face_repo(
                    "jinaai/jina-embeddings-v2-base-en".to_string()
                )
            }
        };
        let tokenizer_source = match options.tokenizer {
            Some(tokenizer_string) => {
                if Path::new(&tokenizer_string).exists() {
                    PathBuf::from(tokenizer_string)
                } else {
                    get_tokenizer_source_from_hugging_face_repo(tokenizer_string)
                }
             },
             None => {
                get_tokenizer_source_from_hugging_face_repo(
                    "sentence-transformers/all-MiniLM-L6-v2".to_string()
                )
             }
        };
        let device = if options.cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap()
        };
        let config = Config::v2_base();
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_source)
            .map_err(E::msg)
            .unwrap();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_source], DType::F32, &device
            ).unwrap() 
        };
        let model = BertModel::new(vb, &config).unwrap();
        Ok(Self { model, tokenizer })
    }
}

fn get_model_source_from_hugging_face_repo(key: String) -> PathBuf {
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

fn get_tokenizer_source_from_hugging_face_repo(key: String) -> PathBuf {
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
        let result: Vec<f32> = Vec::new();
        Ok(result)
    }
}
