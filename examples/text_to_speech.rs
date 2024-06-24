use nebula::{
    options::TTSOptions,
    TextToSpeechModel,
};
use hound::{WavReader, WavSpec, SampleFormat, WavWriter};
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction, ResampleError};
use std::path::{Path, PathBuf};

fn main() -> anyhow::Result<()> {
    let ref_audio_path = Path::new("samples").join("resampled_ref.wav");
    let ref_samples = get_samples_from_audio_path(ref_audio_path)?;

    let tts_options = TTSOptions::default();
    let mut model = TextToSpeechModel::new(tts_options)?;
    model.train(ref_samples)?;

    let text = String::from("Hi! My name is Nick Chapman! Nice to meet you and all the best! I build an amazing Rust project called nebula. Would you like to participate?");
    let generated_audio_sample = model.predict(text)?;
    write_samples_to_test_file(generated_audio_sample)?;

    Ok(())
}


fn get_samples_from_audio_path(audio_path: PathBuf) -> anyhow::Result<Vec<f32>> {
    let (wave, channels, sample_rate, bits_per_sample) = load_audio(audio_path);
    let mut resampled_wave: Vec<f32>;
    if sample_rate != 24000 {
        resampled_wave = resample_audio(&wave, channels, sample_rate as u32, 24000)?;
    }
    else {
        resampled_wave = wave;
    }
    Ok(resampled_wave)
}

fn load_audio(wav_file_path: PathBuf) -> (Vec<f32>, u16, u32, u16) {
    let mut reader = WavReader::open(
        wav_file_path)
        .expect("failed to open file");
    #[allow(unused_variables)]
    let WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        ..
    } = reader.spec();

    let mut samples = convert_integer_to_float_audio(
        &reader
            .samples::<i16>()
            .map(|s| s.expect("invalid sample"))
            .collect::<Vec<_>>(),
    );

    if channels == 2 {
        samples = convert_stereo_to_mono_audio(&samples).unwrap();
    } else if channels != 1 {
        panic!(">2 channels unsupported");
    }
    (samples, channels, sample_rate, bits_per_sample)
}


fn convert_integer_to_float_audio(samples: &[i16]) -> Vec<f32> {
    let mut floats = Vec::with_capacity(samples.len());
    for sample in samples {
        floats.push(*sample as f32 / 32768.0);
    }
    floats
}

fn convert_stereo_to_mono_audio(samples: &[f32]) -> Result<Vec<f32>, &'static str> {
    if samples.len() & 1 != 0 {
        return Err("The stereo audio vector has an odd number of samples. \
            This means a half-sample is missing somewhere");
    }

    Ok(samples
        .chunks_exact(2)
        .map(|x| (x[0] + x[1]) / 2.0)
        .collect())
}

fn resample_audio(audio: &[f32], original_channels: u16, original_samplerate: u32, target_samplerate: u32) -> Result<Vec<f32>, ResampleError> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.99,
        interpolation: SincInterpolationType::Nearest,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f64>::new(
        target_samplerate as f64 / original_samplerate as f64,
        2.0,
        params,
        1024,
        1,
    ).unwrap();
    let audio_f64 = vec![
        audio
        .iter()
        .step_by(original_channels as usize)
        .map(|&x| x as f64)
        .collect::<Vec<f64>>()
    ];
    let resampled_audio = resampler.process(&audio_f64, None).unwrap()[0].iter().map(|&x| x as f32).collect();

    Ok(resampled_audio)
}

fn write_samples_to_test_file(samples: Vec<f32>) -> anyhow::Result<()> {
    let target_channels = 1;
    let target_sample_rate = 24000;
    let spec = WavSpec {
        channels: target_channels,
        sample_rate: target_sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create("test.wav", spec)?;
    for single_sample in samples {
        writer.write_sample(single_sample)?;
    }
    writer.finalize()?;
    Ok(())
}
