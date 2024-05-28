use nebula::{
    options::EmbeddingsOptions,
    EmbeddingsModel
};
use anyhow::Result;
use itertools::Itertools;


fn main() -> Result<()> {
    let options = EmbeddingsOptions::default();
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
