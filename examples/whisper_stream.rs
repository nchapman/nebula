extern crate cpal;
extern crate hound;
extern crate rubato;

use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{WavSpec, WavWriter};

use nebula::{
    options::AutomaticSpeechRecognitionOptions,
    AutomaticSpeechRecognitionModel,
    utils,
};

use std::{env, path::PathBuf};

fn main() {
    let host = cpal::default_host();

    let input_device = host.default_input_device().expect("no input device available");
    let config: cpal::StreamConfig = input_device.default_input_config().unwrap().into();

    println!("Config channels: {}", config.channels);
    println!("Config sample rate: {}", config.sample_rate.0);
    let input_sample_rate = config.sample_rate.0;
    let input_channels = config.channels;
    let target_channels = 1;
    let target_sample_rate = 16000;
    let spec = WavSpec {
        channels: target_channels,
        sample_rate: target_sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let path_to_tmp_wav: PathBuf = [env::temp_dir(), "tmp.wav".into()].iter().collect();
    let mut writer = WavWriter::create(path_to_tmp_wav, spec)
        .expect("failed to create WAV file");

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.99,
        interpolation: SincInterpolationType::Nearest,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f64>::new(
        target_sample_rate as f64 / input_sample_rate as f64,
        2.0,
        params,
        1024,
        target_channels as usize,
    ).unwrap();

    let stream = input_device.build_input_stream(
        &config,
        move |data: &[i16], _: &_| {
            let data_f64 = vec![
                data
                    .iter()
                    .step_by(input_channels as usize)
                    .map(|&x| x as f64)
                    .collect::<Vec<f64>>()
            ];
            resampler
                .process(&data_f64, None)
                .unwrap()[0]
                .iter()
                .for_each(|&value| {writer.write_sample(value as i32).unwrap()});
        },
        |err| eprintln!("an error occurred on the input stream: {}", err),
        None
    ).expect("failed to build input stream");

    stream.play().expect("failed to start audio stream");

    println!("Capturing audio, press Enter to stop...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).expect("failed to read line");

    stream.pause().expect("failed to pause audio stream");

    let mut model = AutomaticSpeechRecognitionModel::new(
        "models/ggml-base.en.bin")
        .unwrap();
    let options = AutomaticSpeechRecognitionOptions::default().with_n_threads(1);

    let path_to_tmp_wav: PathBuf = [env::temp_dir(), "tmp.wav".into()].iter().collect();
    let samples = utils::convert_wav_to_samples(
        path_to_tmp_wav.into_os_string().into_string().unwrap().as_str()
    );

    let out = model.predict(&samples[..], options).unwrap();
    println!("Text that I understood from your speech: {}", out)
}
