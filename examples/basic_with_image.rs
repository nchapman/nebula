use std::{io::Write, sync::Arc};

use nebula::{
    options::{ContextOptions, ModelOptions},
    Model,
};

fn main() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Debug)
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
    nebula::init(std::path::PathBuf::from(
        "backends/llama_cpp/llama-cpp-sys/dist",
    ))
    .unwrap();
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
    let model_options = ModelOptions::builder().n_gpu_layers(10).build();
    let model = Model::new_with_mmproj_with_callback(
        model_file_name,
        mmproj_model_file_name,
        model_options,
        move |a| {
            pbb.set_position((a * 1000.0) as u64);
            std::thread::sleep(std::time::Duration::from_millis(12));
            true
        },
    )
    .unwrap();
    pb.finish_with_message("done");

    let context_options = ContextOptions::builder().n_ctx(6000).build();
    let mut ctx = model.context(context_options).unwrap();
    eprintln!("{image_file_name}");
    //eval data
    ctx.eval(vec![serde_json::json!({
        "role": "user",
        "images": [image_file_name],
        "content": prompt
    })
    .try_into()
    .unwrap()])
        .unwrap();
    ctx.predict(
        nebula::options::PredictOptions::builder()
            .token_callback(Arc::new(Box::new(|token| {
                print!("{}", token);
                std::io::stdout().flush().unwrap();
                true
            })))
            .max_len(n_len)
            .build(),
    )
    .predict()
    .unwrap();
    println!("");
}
