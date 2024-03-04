
use candle_core::{Result, Tensor};

use candle_transformers::models::mistral::Model as Mistral;
use candle_transformers::models::quantized_mistral::Model as QMistral;
use candle_transformers::models::quantized_llama::ModelWeights;

pub enum Model {
    Mistral(Mistral),
    QuantizedMistral(QMistral),
    LlmaMistral(ModelWeights),
}

impl Model {
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        match self {
            Model::Mistral(m) => m.forward(x, index_pos),
            Model::QuantizedMistral(m) => m.forward(x, index_pos),
            Model::LlmaMistral(m) => m.forward(x, index_pos)        
        }
    }    
    pub fn prompt(&self, input: String) -> String {
        match  self {
            Model::LlmaMistral(_) => format!("[INST] {input} [/INST]"),
            _ => input
        }
    
    }
}

