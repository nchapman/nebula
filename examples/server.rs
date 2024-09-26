use std::net::IpAddr;

use nebula::{
    options::{ContextOptions, ModelOptions},
    Model, Server,
};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model: String,
    #[arg(long)]
    mmproj: Option<String>,
}

fn main() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Debug)
        .init()
        .unwrap();
    let args = Args::parse();
    //init nebula
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
    let model = if let Some(mmp) = args.mmproj {
        Model::new_with_mmproj_with_callback(args.model, mmp, model_options, move |a| {
            pbb.set_position((a * 1000.0) as u64);
            std::thread::sleep(std::time::Duration::from_millis(12));
            true
        })
    } else {
        Model::new_with_progress_callback(args.model, model_options, move |a| {
            pbb.set_position((a * 1000.0) as u64);
            std::thread::sleep(std::time::Duration::from_millis(12));
            true
        })
    }
    .unwrap();

    pb.finish_with_message("done");

    let ctx_options = ContextOptions::default();

    let server = Server::new(
        "0.0.0.0".parse::<IpAddr>().expect("parse failed"),
        8081,
        model,
        ctx_options,
    );
    server.run().unwrap();
}
