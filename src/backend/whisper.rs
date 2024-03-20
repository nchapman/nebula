extern crate anyhow;
extern crate cpal;
extern crate ringbuf;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{LocalRb, Rb, SharedRb};
use hound;
use whisper_rs::{
    FullParams,
    SamplingStrategy,
    WhisperContext,
    WhisperContextParameters,
    WhisperState,
    WhisperToken,
};
use std::{
    num::NonZeroU32,
    path::{Path, PathBuf},
    pin::Pin,
    fs::File,
    io::Write,
    time::{Duration, Instant},
    cmp,
    thread,
};

use crate::{
    options::{ModelOptions, PredictOptions},
    Result,
};

const LATENCY_MS: f32 = 5000.0;
const NUM_ITERS: usize = 2;
const NUM_ITERS_SAVED: usize = 2;

use super::Backend;

pub struct Whisper {
    model_str: String,
}

impl Whisper {
    pub fn new(model: impl Into<PathBuf>, options: ModelOptions) -> Result<Self> {
        let model_str = model.into().into_os_string().into_string().unwrap();
        Ok(Self { model_str })
    }
}

impl Backend for Whisper {
    fn predict(
        &mut self,
        prompt: &str,
        options: PredictOptions,
        token_callback: Box<dyn Fn(String) -> bool + Send + 'static>,
    ) -> Result<()> {
        if prompt == "stream" {
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
        
            // Setup whisper
            let ctx = WhisperContext::new_with_params(
                &self.model_str,
                WhisperContextParameters::default()
            )
            .expect("failed to load model");
            let mut state = ctx.create_state().expect("failed to create key");
        
            // Variables used across loop iterations
            let mut iter_samples = LocalRb::new(latency_samples * NUM_ITERS * 2);
            let mut iter_num_samples = LocalRb::new(NUM_ITERS);
            let mut iter_tokens = LocalRb::new(NUM_ITERS_SAVED);
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
        
                // Get tokens to be deleted
                if loop_num > 1 {
                    let num_tokens = state.full_n_tokens(0).unwrap();
                    let token_time_end = state.full_get_segment_t1(0).unwrap();
                    let token_time_per_ms =
                        token_time_end as f32 / (LATENCY_MS * cmp::min(loop_num, NUM_ITERS) as f32); // token times are not a value in ms, they're 150 per second
                    let ms_per_token_time = 1.0 / token_time_per_ms;
        
                    let mut tokens_saved = vec![];
                    // Skip beginning and end token
                    for i in 1..num_tokens - 1 {
                        let token = state.full_get_token_data(0, i).unwrap();
                        let token_t0_ms = token.t0 as f32 * ms_per_token_time;
                        let ms_to_delete = num_samples_to_delete as f32 / (sampling_freq / 1000.0);
        
                        // Save tokens for whisper context
                        if (loop_num > NUM_ITERS) && token_t0_ms < ms_to_delete {
                            tokens_saved.push(token.id);
                        }
                    }
                    num_chars_to_delete = words.chars().count();
                    if loop_num > NUM_ITERS {
                        num_chars_to_delete -= tokens_saved
                            .iter()
                            .map(|x| ctx.token_to_str(*x).expect("Error"))
                            .collect::<String>()
                            .chars()
                            .count();
                    }
                    iter_tokens.push_overwrite(tokens_saved);
                }
        
                // Make the model params
                let (head, tail) = iter_tokens.as_slices();
                let tokens = [head, tail]
                    .concat()
                    .into_iter()
                    .flatten()
                    .collect::<Vec<WhisperToken>>();
                let mut params = gen_whisper_params();
                params.set_tokens(&tokens);
        
                // Run the model
                state
                    .full(params, &current_samples)
                    .expect("failed to convert samples");
        
                // Update the words on screen
                if num_chars_to_delete != 0 {
                    // TODO: JPB: Potentially unneeded if statement
                    print!(
                        "\x1B[{}D{}\x1B[{}D",
                        num_chars_to_delete,
                        " ".repeat(num_chars_to_delete),
                        num_chars_to_delete
                    );
                }
                let num_tokens = state.full_n_tokens(0).unwrap();
                words = (1..num_tokens - 1)
                    .map(|i| state.full_get_token_text(0, i).expect("Error"))
                    .collect::<String>();
                print!("{}", words);
                std::io::stdout().flush().unwrap();
            }
        }
        else {
            let ctx = WhisperContext::new_with_params(
                &self.model_str,
                WhisperContextParameters::default()
            )
            .expect("failed to load model");
            let mut state = ctx.create_state().expect("failed to create key");
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
    
            // TODO Later include it into ModelOptions? Or into separate struct?
            params.set_n_threads(1);
            params.set_translate(true);
            params.set_language(Some("en"));
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);

            let mut reader = hound::WavReader::open(prompt).expect("failed to open file");
            #[allow(unused_variables)]
            let hound::WavSpec {
                channels,
                sample_rate,
                bits_per_sample,
                ..
            } = reader.spec();
        
            let mut audio = whisper_rs::convert_integer_to_float_audio(
                &reader
                    .samples::<i16>()
                    .map(|s| s.expect("invalid sample"))
                    .collect::<Vec<_>>(),
            );
        
            if channels == 2 {
                audio = whisper_rs::convert_stereo_to_mono_audio(&audio).unwrap();
            } else if channels != 1 {
                panic!(">2 channels unsupported");
            }
        
            if sample_rate != 16000 {
                panic!("sample rate must be 16KHz");
            }
        
            state.full(params, &audio[..]).expect("failed to run model");
        
            let mut file = File::create("transcript.txt").expect("failed to create file");
        
            let num_segments = state
                .full_n_segments()
                .expect("failed to get number of segments");
            for i in 0..num_segments {
                let segment = state
                    .full_get_segment_text(i)
                    .expect("failed to get segment");
                let start_timestamp = state
                    .full_get_segment_t0(i)
                    .expect("failed to get start timestamp");
                let end_timestamp = state
                    .full_get_segment_t1(i)
                    .expect("failed to get end timestamp");
        
                println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
        
                let line = format!("[{} - {}]: {}\n", start_timestamp, end_timestamp, segment);
        
                file.write_all(line.as_bytes())
                    .expect("failed to write to file");
            }
        }
        Ok(())
    }
}


fn gen_whisper_params<'a>() -> FullParams<'a, 'a> {
    let mut params = FullParams::new(SamplingStrategy::default());
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    params.set_language(Some("en"));
    params.set_token_timestamps(true);
    params.set_duration_ms(LATENCY_MS as i32);
    params.set_no_context(true);
    //params.set_n_threads(4);

    //params.set_no_speech_thold(0.3);
    //params.set_split_on_word(true);

    // This impacts token times, don't use
    //params.set_single_segment(true);

    params
}


fn err_fn(err: cpal::StreamError) {
    eprintln!("an error occurred on stream: {}", err);
}
