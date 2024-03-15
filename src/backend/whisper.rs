use whisper_rs::{
    FullParams,
    SamplingStrategy,
    WhisperContext,
    WhisperContextParameters,
    WhisperState,
};
use crate::Result;

use super::Backend;

pub struct Whisper {
    ctx: WhisperContext,
    state: WhisperState<'_>,
    params: FullParams<'_, '_>,
}

impl Whisper {
    pub fn new(model: impl Into<PathBuf>) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(
            model, WhisperContextParameters::default()
        )
        .expect("failed to load model");
        let mut state = ctx.create_state().expect("failed to create key");
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });

        // Edit params as needed.
        // Set the number of threads to use to 1.
        params.set_n_threads(1);
        // Enable translation.
        params.set_translate(true);
        // Set the language to translate to to English.
        params.set_language(Some("en"));
        // Disable anything that prints to stdout.
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        Ok(Self {ctx, state, params})
    }
}
