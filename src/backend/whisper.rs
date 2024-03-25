extern crate anyhow;
extern crate cpal;
extern crate ringbuf;

use whisper_rs::{
    FullParams,
    SamplingStrategy,
    WhisperContext,
    WhisperContextParameters,
};
use std::{
    path::PathBuf,
    fs::File,
    io::Write,
};

use crate::{
    options::{AutomaticSpeechRecognitionOptions},
    Result,
};

use super::AutomaticSpeechRecognitionBackend;

pub struct Whisper {
    model_str: String,
}

impl Whisper {
    pub fn new(model: impl Into<PathBuf>) -> Result<Self> {
        let model_str = model.into().into_os_string().into_string().unwrap();
        Ok(Self { model_str })
    }
}

impl AutomaticSpeechRecognitionBackend for Whisper {
    fn predict(
        &mut self,
        samples: &[f32],
        out_file_path: &str,
        options: AutomaticSpeechRecognitionOptions,
    ) -> Result<()> {
        let ctx = WhisperContext::new_with_params(
            &self.model_str,
            WhisperContextParameters::default()
        )
            .expect("failed to load model");
        let mut state = ctx.create_state().expect("failed to create key");
        let mut params = FullParams::new(
            SamplingStrategy::Greedy { best_of: 0 }
        );

        params.set_n_threads(options.n_threads);
        params.set_translate(options.translate);
        params.set_language(Some(options.language));
        params.set_print_special(options.print_special);
        params.set_print_progress(options.print_progress);
        params.set_print_realtime(options.print_realtime);
        params.set_print_timestamps(options.print_timestamps);
    
        state.full(params, &samples[..]).expect("failed to run model");

        let mut out_file = File::create(out_file_path)
            .expect("failed to create file");

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

            out_file.write_all(line.as_bytes())
                .expect("failed to write to file");
        }
        Ok(())
    }
}
