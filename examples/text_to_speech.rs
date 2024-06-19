use nebula::{
    options::TTSOptions,
    TextToSpeechModel,
    utils,
};
use std::path::{Path, PathBuf};

fn main() -> anyhow::Result<()> {
    let ref_audio_path = Path::new("samples").join("resampled_ref.wav");
    let ref_samples = utils::get_tts_samples_from_audio_path(ref_audio_path)?;

    let tts_options = TTSOptions::default();
    let mut model = TextToSpeechModel::new(tts_options)?;
    model.train(ref_samples)?;

    let text = String::from("Hi! My name is Nick Chapman! Nice to meet you and all the best! I build an amazing Rust project called nebula. Would you like to participate?");
    let generated_audio_sample = model.predict(text)?;
    let test_file_path = PathBuf::from("test.wav");
    utils::write_tts_samples_to_test_file(test_file_path, generated_audio_sample)?;

    Ok(())
}
