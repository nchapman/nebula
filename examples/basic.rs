use std::io::Write;

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
    if args.len() < 3 {
        println!(
            "usage:\n\t{} <model_file_path> \"<promt>\" <n_len = 150>",
            args[0]
        );
        return;
    }
    let model_file_name = args[1].clone();
    let prompt = args[2].clone();
    let mut n_len = None;
    if args.len() > 3 {
        if let Ok(ss) = str::parse(&args[3]) {
            n_len = Some(ss);
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
    let model_options = ModelOptions::default().with_n_gpu_layers(10);
    let model = Model::new_with_progress_callback(model_file_name, model_options, move |a| {
        pbb.set_position((a * 1000.0) as u64);
        std::thread::sleep(std::time::Duration::from_millis(12));
        true
    })
    .unwrap();
    pb.finish_with_message("done");

    let ctx_options = ContextOptions::default();

    let mut ctx = model.context(ctx_options).unwrap();

    ctx.eval_str(&prompt, true).unwrap();

    let mut p = ctx
        .predict()
        .with_token_callback(Box::new(|token| {
            print!("{}", token);
            std::io::stdout().flush().unwrap();
            true
        }))
        .with_temp(0.8);
    if let Some(n_len) = n_len {
        p = p.with_max_len(n_len);
    }
    p.predict().unwrap();

    println!("");
}
