use nebula::{
    options::{ModelOptions, PredictOptions},
    Model,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        println!(
            "usage:\n\t{} <model_file_path> \"<promt>\" <n_len = 150>",
            args[0]
        );
        return;
    }
    let model_file_name = args[1].clone();
    let prompt = args[2].clone();
    let mut n_len = 150;
    if args.len() > 3 {
        if let Ok(ss) = str::parse(&args[3]) {
            n_len = ss;
        }
    }
    let model_options = ModelOptions::default().with_n_gpu_layers(10);

    let mut model = Model::new(model_file_name, model_options).unwrap();

    let predict_options = PredictOptions::default().with_n_len(n_len);

    model
        .predict(
            &prompt,
            predict_options,
            Box::new(|token| {
                print!("{}", token);
                true
            }),
        )
        .unwrap();
    println!("");
}
