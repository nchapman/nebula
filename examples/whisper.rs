use nebula::{
    options::{ModelOptions, PredictOptions},
    Model,
};


fn main() {
    let model_options = ModelOptions::default().with_n_gpu_layers(10);

    let mut model = Model::new(
        "/home/ivan/Projects/nebula/models/ggml-base.en.bin", model_options)
        .unwrap();

    let predict_options = PredictOptions::default().with_n_len(150);

    model
        .predict(
            "/home/ivan/Projects/nebula/samples/nata_1.wav".into(),
            predict_options,
            Box::new(|token| {
                print!("{}", token);
                true
            }),
        )
        .unwrap();
    println!("");
}