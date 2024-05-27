use crate::{options::EmbeddingsOptions, Result};

use super::EmbeddingsBackend;

pub struct JinaBertBackend {
    options: EmbeddingsOptions,
}

impl JinaBertBackend {
    pub fn new(options: EmbeddingsOptions) -> Result<Self> {
        Ok(Self { options })
    }
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
