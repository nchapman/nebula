use std::io::Read;

use nebula::{
    options::{ModelOptions, PredictOptions},
    Model,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        println!(
            "usage:\n\t{} <model_file_path> <mmproj_model_file_path> <image_file_name> \"<promt = <image>\\nUSER:\\nProvide a full description.\\nASSISTANT:\\n>\" <n_len = 6000>",
            args[0]
        );
        return;
    }
    let model_file_name = args[1].clone();
    let mmproj_model_file_name = args[2].clone();
    let image_file_name = args[3].clone();
    let prompt = if args.len() > 4 && !args[4].is_empty() {
        args[4].clone()
    } else {
        "<image>\nUSER:\nProvide a full description.\nASSISTANT:\n".to_string()
    };
    let mut n_len = 6000;
    if args.len() > 5 {
        if let Ok(ss) = str::parse(&args[5]) {
            n_len = ss;
        }
    }
    let model_options = ModelOptions::default().with_n_gpu_layers(10);

    let mut model =
        Model::new_with_mmproj(model_file_name, mmproj_model_file_name, model_options).unwrap();

    let predict_options = PredictOptions::default().with_n_ctx(6000).with_n_len(n_len);
    //read image
    let mut image_bytes = vec![];
    let mut f = std::fs::File::open(&image_file_name).unwrap();
    f.read_to_end(&mut image_bytes).unwrap();

    model
        .predict_with_image(
            image_bytes,
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
