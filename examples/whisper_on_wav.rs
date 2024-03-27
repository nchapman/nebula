use hound;
use nebula::{
    options::AutomaticSpeechRecognitionOptions,
    AutomaticSpeechRecognitionModel,
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
    let samples = convert_wav_to_samples(wav_file_path);
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


fn convert_wav_to_samples(wav_file_path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(
        wav_file_path)
        .expect("failed to open file");
    #[allow(unused_variables)]
    let hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        ..
    } = reader.spec();

    let mut samples = whisper_rs::convert_integer_to_float_audio(
        &reader
            .samples::<i16>()
            .map(|s| s.expect("invalid sample"))
            .collect::<Vec<_>>(),
    );

    if channels == 2 {
        samples = whisper_rs::convert_stereo_to_mono_audio(&samples).unwrap();
    } else if channels != 1 {
        panic!(">2 channels unsupported");
    }
    if sample_rate != 16000 {
        panic!("sample rate must be 16KHz");
    }
    samples
}
