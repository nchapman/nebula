use whisper_rs::{
    FullParams,
    SamplingStrategy,
    WhisperContext,
    WhisperContextParameters,
};
use std::path::PathBuf;

use crate::{
    options::AutomaticSpeechRecognitionOptions,
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
        options: AutomaticSpeechRecognitionOptions,
    ) -> Result<String> {
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

        let num_segments = state
            .full_n_segments()
            .expect("failed to get number of segments");
        let mut out = String::new();
        for i in 0..num_segments {
            let segment = state
                .full_get_segment_text(i)
                .expect("failed to get segment");
            out.push_str(&segment);
            out.push_str(" ");
        }
        Ok(out)
    }
}
