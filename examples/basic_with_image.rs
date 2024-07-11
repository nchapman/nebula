use std::io::{Read, Write};

use nebula::{
    options::{ContextOptions, ModelOptions},
    Model,
};

fn main() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Error)
        .init()
        .unwrap();
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
    println!("model loading...");
    let total_size = 1000;
    let pb = indicatif::ProgressBar::new(total_size);
    pb.set_draw_target(indicatif::ProgressDrawTarget::stdout());
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}]",
        )
        .unwrap()
        .progress_chars("#>-"),
    );
    pb.set_position(0);
    let pbb = pb.clone();
    let model_options = ModelOptions::default()
        .with_n_gpu_layers(10)
        .with_load_progress_callback(move |a| {
            pbb.set_position((a * 1000.0) as u64);
            std::thread::sleep(std::time::Duration::from_millis(12));
            true
        });
    let model =
        Model::new_with_mmproj(model_file_name, mmproj_model_file_name, model_options).unwrap();
    pb.finish_with_message("done");

    let context_options = ContextOptions::default().with_n_ctx(6000);
    let mut ctx = model.context(context_options).unwrap();

    //read image
    let mut image_bytes = vec![];
    let mut f = std::fs::File::open(&image_file_name).unwrap();
    f.read_to_end(&mut image_bytes).unwrap();

    //eval data
    ctx.eval_image(image_bytes, &prompt).unwrap();

    //generate predict

    //    let answer = ctx.predict(n_len).unwrap();
    //    println!("{answer}");

    //or
    ctx.predict()
        .with_token_callback(Box::new(|token| {
            print!("{}", token);
            std::io::stdout().flush().unwrap();
            true
        }))
        .with_max_len(n_len)
        .predict()
        .unwrap();
    println!("");
}
