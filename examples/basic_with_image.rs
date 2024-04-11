use std::io::Read;

use nebula::{
    options::{ContextOptions, ModelOptions},
    Model,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        println!(
            "usage:\n\t{} <model_file_path> <mmproj_model_file_path> <image_file_name> \"<promt = Provide a full description.>\" <n_len = 6000>",
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
        "Provide a full description.".to_string()
    };
    let mut n_len = 6000;
    if args.len() > 5 {
        if let Ok(ss) = str::parse(&args[5]) {
            n_len = ss;
        }
    }
    let model_options = ModelOptions::default().with_n_gpu_layers(10);
    let model =
        Model::new_with_mmproj(model_file_name, mmproj_model_file_name, model_options).unwrap();

    let context_options = ContextOptions::default().with_n_ctx(6000);
    let mut ctx = model.context(context_options).unwrap();

    //read image
    let mut image_bytes = vec![];
    let mut f = std::fs::File::open(&image_file_name).unwrap();
    f.read_to_end(&mut image_bytes).unwrap();

    //eval data
    ctx.eval_str(&"", true).unwrap();
    ctx.eval_image(image_bytes, &prompt).unwrap();

    //generate predict

    //    let answer = ctx.predict(n_len).unwrap();
    //    println!("{answer}");

    //or
    ctx.predict_with_callback(
        Box::new(|token| {
            print!("{}", token);
            true
        }),
        n_len,
    )
    .unwrap();

    println!("");
}
