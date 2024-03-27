use nebula::{
    options::AutomaticSpeechRecognitionOptions,
    AutomaticSpeechRecognitionModel,
    utils,
};

use std::{
    fs::File,
    io::Write,
};

fn main() {
    let mut model = AutomaticSpeechRecognitionModel::new(
        "models/ggml-base.en.bin")
        .unwrap();

    let wav_file_path = "samples/ivan_2.wav";
    let samples = utils::convert_wav_to_samples(wav_file_path);
    let options = AutomaticSpeechRecognitionOptions::default().with_n_threads(1);

    let out = model
        .predict(
            &samples[..],
            options,
        )
        .unwrap();
    println!("Text extracted from wav: {}", out);

    let out_file_path = "samples/ivan_2.txt";
    let mut out_file = File::create(out_file_path).expect("failed to create file");
    out_file.write_all(out.as_bytes()).expect("failed to write to file");
    print!("");
}
