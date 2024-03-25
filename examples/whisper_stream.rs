extern crate anyhow;
extern crate cpal;
extern crate ringbuf;


use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{LocalRb, Rb, SharedRb};
use std::{
    time::{Duration, Instant},
    thread,
};

use nebula::{
    options::AutomaticSpeechRecognitionOptions,
    AutomaticSpeechRecognitionModel,
};

const LATENCY_MS: f32 = 5000.0;
const NUM_ITERS: usize = 2;
const NUM_ITERS_SAVED: usize = 2;

fn main() {
    let mut model = AutomaticSpeechRecognitionModel::new(
        "models/ggml-base.en.bin")
        .unwrap();

    let host = cpal::default_host();

    let input_device = host
        .default_input_device()
        .expect("failed to get default input device");
    println!("Using default input device: \"{}\"", input_device.name().unwrap());

    // Top level variables
    let config: cpal::StreamConfig = input_device.default_input_config().unwrap().into();
    let latency_frames = (LATENCY_MS / 1_000.0) * config.sample_rate.0 as f32;
    let latency_samples = latency_frames as usize * config.channels as usize;
    let sampling_freq = config.sample_rate.0 as f32 / 2.0; // TODO: JPB: Divide by 2 because of stereo to mono

    // The buffer to share samples
    let ring = SharedRb::new(latency_samples * 2);
    let (mut producer, mut consumer) = ring.split();

    // Setup microphone callback
    let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let mut output_fell_behind = false;
        for &sample in data {
            if producer.push(sample).is_err() {
                output_fell_behind = true;
            }
        }
        if output_fell_behind {
            eprintln!("output stream fell behind: try increasing latency");
        }
    };

    let out_file_path = "samples/stream.txt";

    // Variables used across loop iterations
    let mut iter_samples = LocalRb::new(latency_samples * NUM_ITERS * 2);
    let mut iter_num_samples = LocalRb::new(NUM_ITERS);
    for _ in 0..NUM_ITERS {
        iter_num_samples
            .push(0)
            .expect("Error initailizing iter_num_samples");
    }

    // Build streams.
    println!(
        "Attempting to build both streams with f32 samples and `{:?}`.",
        config
    );
    println!("Setup input stream");
    let input_stream = input_device.build_input_stream(
        &config, input_data_fn, err_fn, None)
        .unwrap();
    println!("Successfully built streams.");

    // Play the streams.
    println!(
        "Starting the input and output streams with `{}` milliseconds of latency.",
        LATENCY_MS
    );
    input_stream.play().unwrap();
    //output_stream.play()?;

    // Remove the initial samples
    consumer.pop_iter().count();
    let mut start_time = Instant::now();

    // Main loop
    // TODO: JPB: Make this it's own function (And the lines above it)
    let mut num_chars_to_delete = 0;
    let mut loop_num = 0;
    let mut words = "".to_owned();

    loop {
        loop_num += 1;

        // Only run every LATENCY_MS
        let duration = start_time.elapsed();
        let latency = Duration::from_millis(LATENCY_MS as u64);
        if duration < latency {
            let sleep_time = latency - duration;
            thread::sleep(sleep_time);
        } else {
            panic!("Classification got behind. It took to long. Try using a smaller model and/or more threads");
        }
        start_time = Instant::now();

        // Collect the samples
        let samples: Vec<_> = consumer.pop_iter().collect();
        let samples = whisper_rs::convert_stereo_to_mono_audio(&samples).unwrap();
        //let samples = make_audio_louder(&samples, 1.0);
        let num_samples_to_delete = iter_num_samples
            .push_overwrite(samples.len())
            .expect("Error num samples to delete is off");
        for _ in 0..num_samples_to_delete {
            iter_samples.pop();
        }
        iter_samples.push_iter(&mut samples.into_iter());
        let (head, tail) = iter_samples.as_slices();
        let current_samples = [head, tail].concat();

        let options = AutomaticSpeechRecognitionOptions::default().with_n_threads(1);
        model
            .predict(
                &current_samples[..],
                out_file_path,
                options,
            )
            .unwrap();
    }
}

fn err_fn(err: cpal::StreamError) {
    eprintln!("an error occurred on stream: {}", err);
}
