use hound;
use whisper_rs::{
    FullParams,
    SamplingStrategy,
    WhisperContext,
    WhisperContextParameters,
    WhisperState,
};
use std::{
    num::NonZeroU32,
    path::{Path, PathBuf},
    pin::Pin,
    fs::File,
    io::Write
};

use crate::{
    options::{ModelOptions, PredictOptions},
    Result,
};

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
        Ok(())
    }
}
