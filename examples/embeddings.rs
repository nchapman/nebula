use nebula::{
    options::{EmbeddingsOptions, EmbeddingsModelType},
    EmbeddingsModel
};
use anyhow::Result;
use itertools::Itertools;


fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_type = args[1].clone();
    let model_type = match model_type.as_str() {
        "jina" => EmbeddingsModelType::JinaBert,
        "t5" => EmbeddingsModelType::T5,
        "bert" => EmbeddingsModelType::Bert,
        _ => EmbeddingsModelType::JinaBert,
    };

    let options = EmbeddingsOptions::default().with_model_type(model_type);
    let mut model = EmbeddingsModel::new(options)?;

    let text = "Hi! My name is Nick Chapman! Nice to meet you and all the best! I build an amazing Rust project called nebula. Would you like to participate?";
    let text = text.to_string();

    let out = model.predict(text)?;
    println!(
        "Calculated embedding: {}",
        out[..10].iter().join(", ") + " ... " + &out.last().unwrap().to_string()
    );

    Ok(())
}
