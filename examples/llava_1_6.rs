use nebula::{
    options::{ModelOptions, PredictOptions},
    Model,
};

fn main() {
    let model_options = ModelOptions::default().with_n_gpu_layers(10);

    let mut model = Model::new("models/llava-v1.6-mistral-7b.Q4_K_M.gguf", model_options).unwrap();

    let predict_options = PredictOptions::default().with_n_len(150);

    model
        .predict(
            "Write helloworld code in Rust.".into(),
            predict_options,
            Box::new(|token| {
                print!("{}", token);
                true
            }),
        )
        .unwrap();
    println!("");
}