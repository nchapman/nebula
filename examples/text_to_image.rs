use nebula::{
    options::{TextToImageModelType, TextToImageOptions},
    TextToImageModel,
};
use candle_core::IndexOp;
use candle_examples::save_image;

const SAMPLES_COUNT: usize = 8 as usize;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_type = if args.len() < 2 {
        "".to_string()
    } else {
        args[1].clone()
    };
    let model_type = match model_type.as_str() {
        "stable-diffusion" => TextToImageModelType::StableDiffusion,
        "wuerstchen" => TextToImageModelType::Wuerstchen,
        _ => TextToImageModelType::StableDiffusion,
    };
    println!("Model type: {:?}", model_type);

    let options = TextToImageOptions::default()
        .with_model_type(model_type)
        .with_samples_count(SAMPLES_COUNT);
    let mut model = TextToImageModel::new(options)?;

    let prompt = "A happy dog running in a forest"
        .to_string();

    let samples = model.generate(prompt)?;
    for sample_index in 0..samples.shape().dims()[0] {
        let image = samples.i(sample_index)?;
        let image_filename = format!("test_{}.png", sample_index);
        save_image(&image, image_filename)?;
    }

    Ok(())
}
