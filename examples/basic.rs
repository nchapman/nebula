use std::io::Write;

use nebula::{
    options::{ContextOptions, ModelOptions},
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

    let model = Model::new(model_file_name, model_options).unwrap();

    let ctx_options = ContextOptions::default();

    let mut ctx = model.context(ctx_options).unwrap();

    ctx.eval_str(&prompt, true).unwrap();

    ctx.predict_with_callback(
        Box::new(|token| {
            print!("{}", token);
            std::io::stdout().flush().unwrap();
            true
        }),
        n_len,
    )
    .unwrap();

    println!("");
}
