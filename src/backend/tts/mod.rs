use tch::{self, IndexOp, Tensor};
use std::path::{Path, PathBuf};
use punkt::{SentenceTokenizer, TrainingData};
use punkt::params::Standard;
use fancy_regex::Regex;


mod treebank_word_tokenizer;
mod phonemizer;
mod text_cleaner;
use treebank_word_tokenizer::TreebankWordTokenizer;
use phonemizer::text_to_phonemes;
use text_cleaner::TextCleaner;

use crate::options::{TTSOptions, TTSDevice};
use super::TextToSpeechBackend;

const MEAN: f64 = -4.0;
const STD: f64 = 4.0;
const STD_TOKEN_COUNT: usize = 60;
const STD_SAMPLE_LENGTH: usize = 72000;
const ALPHA: f64 = 0.3;
const BETA: f64 = 0.9;
const MU: f64 = 0.7;

pub struct StyleTTSBackend {
    device: tch::Device,
    style_ref: tch::Tensor,
}

impl StyleTTSBackend {
    pub fn new(options: TTSOptions) -> anyhow::Result<Self> {
        let device = match options.device {
            TTSDevice::Cpu => tch::Device::Cpu,
            TTSDevice::Cuda => tch::Device::Cuda(0)
        };
        let style_ref = Tensor::new();
        Ok(Self { device, style_ref })
    }
    
}

impl TextToSpeechBackend for StyleTTSBackend {
    fn train(
        &mut self,
        ref_samples: Vec<f32>,
    ) -> anyhow::Result<()> {
        self.style_ref = self.compute_style(ref_samples)?;
        Ok(())
    }

    fn predict(
        &mut self,
        text: String,
    ) -> anyhow::Result<Vec<f32>> {
        if self.style_ref.size().len() == 1 && self.style_ref.size()[0] == 0 {
            panic!("Error: reference style is not initialized! Run train before running predict!");
        }

        let mut sentences: Vec<String> = self.split_by_any_sep(text);

        let mut i: i32 = 0;
        while i < sentences.len() as i32 - 1 {
            if sentences[i as usize].len() + sentences[i as usize + 1].len() < 51 {
                sentences[i as usize] = sentences[i as usize].clone() + " " + &sentences[i as usize + 1];
                sentences.remove(i as usize + 1);
                i -= 1;
            }
            i += 1;
        }
        println!("{:?}", sentences);

        let treebank_word_tokenizer = TreebankWordTokenizer::new();

        let mut all_out_vecs: Vec<Vec<f32>> = Vec::new();
        let mut s_prev: Option<Tensor> = None;
        for sent in sentences {
            let mut s_ref_1 = tch::Tensor::ones(self.style_ref.size(), (self.style_ref.kind(), self.style_ref.device()));
            s_ref_1.copy_(&self.style_ref);

            let mut single_text = sent.trim().to_string();
            single_text = single_text.replace("\"", "");
            let english = TrainingData::english();
            let punkt_tokenizer = SentenceTokenizer::<Standard>::new(&single_text, &english);
            let mut tokens_vec: Vec<String> = Vec::new();
            for sub_s in punkt_tokenizer {
                let phonemized_sentence = text_to_phonemes(sub_s, "en-us", None, false, false)?;
                let mut phoneme = phonemized_sentence[0].clone();
                let single_tokens = treebank_word_tokenizer.tokenize(phoneme, false);
                tokens_vec.extend(single_tokens);
            }

            let merged_tokens = tokens_vec.join(" ");
            let text_cleaner = TextCleaner::new();
            let mut tokens_from_string = text_cleaner.clean_text(&merged_tokens);
            tokens_from_string.insert(0, 0);
            if tokens_from_string.len() <= STD_TOKEN_COUNT {
                while tokens_from_string.len() < STD_TOKEN_COUNT {
                    tokens_from_string.push(1 as i64);
                }
            }
            else {
                panic!("Error: tokens count should be smaller or equal to 50. The threshold shoul be adjusted. Received: {}", tokens_from_string.len());
            }
            let tokens = Tensor::of_slice(&tokens_from_string).to_device(self.device).to_kind(tch::Kind::Int64).unsqueeze(0);

            let mut tokens_1 = tch::Tensor::ones(&[1, STD_TOKEN_COUNT as i64], (tch::Kind::Int64, self.device));
            tokens_1.copy_(&tokens);
            let input_lengths = tch::Tensor::of_slice(&[*tokens.size().last().unwrap() as i64]).to(self.device);
            let mut input_lengths_1 = tch::Tensor::ones(input_lengths.size(), (input_lengths.kind(), input_lengths.device()));
            input_lengths_1.copy_(&input_lengths);
            let mut input_lengths_2 = tch::Tensor::ones(input_lengths.size(), (input_lengths.kind(), input_lengths.device()));
            input_lengths_2.copy_(&input_lengths);
            let text_mask = self.length_to_mask(&input_lengths);
            let mut text_mask_1 = tch::Tensor::ones(text_mask.size(), (text_mask.kind(), text_mask.device()));
            text_mask_1.copy_(&text_mask);
            let attention_mask = tch::Tensor::bitwise_not(&text_mask).to_kind(tch::Kind::Int64);

            let text_encoder = tch::CModule::load(self.resolve_path_to_model("text_encoder.pt"))?;
            let t_en = text_encoder.forward_ts(&[tokens, input_lengths, text_mask])?;

            let bert = tch::CModule::load(self.resolve_path_to_model("bert_dur.pt"))?;
            let bert_dur = bert.forward_ts(&[tokens_1, attention_mask])?;
            let mut bert_dur_1 = tch::Tensor::ones(bert_dur.size(), (bert_dur.kind(), bert_dur.device()));
            bert_dur_1.copy_(&bert_dur);

            let bert_encoder = tch::CModule::load(self.resolve_path_to_model("bert_encoder.pt"))?;
            let d_en = bert_encoder.forward_ts(&[bert_dur])?.transpose(-1, -2);

            let noise = tch::Tensor::randn(&[1, 1, 256], (tch::Kind::Float, self.device));
            let sampler_embd = bert_dur_1.i(0).unsqueeze(0);
            let sampler = tch::CModule::load(self.resolve_path_to_model("sampler.pt"))?;
            let mut s_pred = sampler.forward_ts(&[noise, sampler_embd, s_ref_1])?.squeeze_dim(0);

            if s_prev.is_some() {
                s_pred = MU * s_prev.unwrap() + (1.0 - MU) * s_pred;
            }

            let mut s = s_pred.i((.., 128..));
            s = BETA * s + (1.0 - BETA) * self.style_ref.i((.., 128..));
            let mut s_1 = tch::Tensor::ones(s.size(), (s.kind(), s.device()));
            s_1.copy_(&s);
            let mut s_vec_1: Vec<tch::Tensor> = Vec::new();
            let mut s_vec_2: Vec<tch::Tensor> = Vec::new();
            for i in 0..3 {
                s_vec_1.push(tch::Tensor::ones(s.size(), (s.kind(), s.device())));
                s_vec_1.last_mut().unwrap().copy_(&s);
                s_vec_2.push(tch::Tensor::ones(s.size(), (s.kind(), s.device())));
                s_vec_2.last_mut().unwrap().copy_(&s);
            }
            s_vec_1.reverse();
            s_vec_2.reverse();
            let mut ref_ = s_pred.i((.., ..128));
            ref_ = ALPHA * ref_ + (1.0 - ALPHA) * self.style_ref.i((.., ..128));
            let mut ref_1 = tch::Tensor::ones(ref_.size(), (ref_.kind(), ref_.device()));
            ref_1.copy_(&ref_);

            s_pred = Tensor::cat(&[ref_1, s_1], -1);


            let predictor_text_encoder = tch::CModule::load(self.resolve_path_to_model("predictor_text_encoder.pt"))?;
            let d = predictor_text_encoder.forward_ts(&[d_en, s, input_lengths_1, text_mask_1])?;
            let mut d_1 = tch::Tensor::ones(d.size(), (d.kind(), d.device()));
            d_1.copy_(&d);

            let predictor_lstm = tch::CModule::load(self.resolve_path_to_model("predictor_lstm.pt"))?;
            let x = predictor_lstm.forward_ts(&[d])?;
            let mut x_1 = tch::Tensor::ones(x.size(), (x.kind(), x.device()));
            x_1.copy_(&x);

            let predictor_duration_proj = tch::CModule::load(self.resolve_path_to_model("predictor_duration_proj.pt"))?;
            let duration = predictor_duration_proj.forward_ts(&[x])?;
            let dim: Vec<i64> = vec![-1];
            let duration = duration.sigmoid().sum_dim_intlist(dim, false, tch::Kind::Float);
            let pred_dur = duration.squeeze().round().clamp_min(1.0);
            // let pred_dur = tch::Tensor::concatenate(&[pred_dur.i(..pred_dur.size()[0]-1), pred_dur.i(pred_dur.size()[0]-1..) + 5.0], 0);
            let dim0 = input_lengths_2.int64_value(&[0]);
            let dim1 = pred_dur.sum_dim_intlist(0, true, pred_dur.kind()).to_kind(tch::Kind::Int64).int64_value(&[0]);
            let mut pred_aln_trg = tch::Tensor::zeros(&[dim0, dim1], (tch::Kind::Float, pred_dur.device()));

            let mut c_frame = 0;
            for i in 0..pred_aln_trg.size()[0] {
                let pred_dur_i = pred_dur.i(i..i+1).to_kind(tch::Kind::Int64).int64_value(&[0]);
                pred_aln_trg
                    .get(i)
                    .slice(0, c_frame, c_frame + pred_dur_i, 1)
                    .fill_(1.0);
                c_frame += pred_dur_i;
            }

            let mut en = d_1.transpose(-1, -2).matmul(&pred_aln_trg.unsqueeze(0));
            en = tch::Tensor::concatenate(&[en.i((.., .., 0..1)), en.i((.., .., 0..en.size()[2]-1))], 2);

            let predictor_shared = tch::CModule::load(self.resolve_path_to_model("predictor_shared.pt"))?;
            let x_1 = predictor_shared.forward_ts(&[en.transpose_(-1, -2)])?;

            let mut F0 = x_1.transpose(-1, -2);
            for i in 0..3 {
                let F0_block = tch::CModule::load(self.resolve_path_to_model(&format!("F0_block_{}.pt", i)))?;
                F0 = F0_block.forward_ts(&[F0, s_vec_1.pop().unwrap()])?;
            }

            let F0_proj = tch::CModule::load(self.resolve_path_to_model("F0_proj.pt"))?;
            F0 = F0_proj.forward_ts(&[F0])?.squeeze_dim(1);

            let mut N = x_1.transpose(-1, -2);
            for i in 0..3 {
                let N_block = tch::CModule::load(self.resolve_path_to_model(&format!("N_block_{}.pt", i)))?;
                N = N_block.forward_ts(&[N, s_vec_2.pop().unwrap()])?;
            }

            let N_proj = tch::CModule::load(self.resolve_path_to_model("N_proj.pt"))?;
            N = N_proj.forward_ts(&[N])?.squeeze_dim(1);

            let mut asr = t_en.matmul(&pred_aln_trg.unsqueeze(0));
            asr = tch::Tensor::concatenate(&[asr.i((.., .., 0..1)), asr.i((.., .., 0..asr.size()[2]-1))], 2);

            let decoder = tch::CModule::load(self.resolve_path_to_model("decoder.pt"))?;
            let mut out = decoder.forward_ts(&[asr, F0, N, ref_.squeeze().unsqueeze(0)])?.squeeze();
            out = out.i(0..out.size()[0]-100);

            println!("Output successfully calculated! Output shape: {:?}", out.size());
            println!("Output kind: {:?}", out.kind());

            let mut out_float_vec: Vec<f32> = Vec::new();
            for i in 0..out.size()[0] {
                let tensor_slice = out.i(i..i+1).to_kind(tch::Kind::Float).double_value(&[0]) as f32;
                out_float_vec.push(tensor_slice);
            }
            all_out_vecs.push(out_float_vec);
            let mut s_prev_tensor = tch::Tensor::ones(s_pred.size(), (s_pred.kind(), s_pred.device()));
            s_prev_tensor.copy_(&s_pred);
            s_prev = Some(s_prev_tensor);
        }

        let mut output_samples: Vec<f32> = Vec::new();
        for out_float_vec in all_out_vecs {
            for sample in out_float_vec {
                output_samples.push(sample);
            }
        }

        Ok(output_samples)
    }
}

impl StyleTTSBackend {

    fn resolve_path_to_model(&mut self, model_filename: &str) -> PathBuf {
        self.resolve_path_to_models_folder().join(model_filename)
    }

    fn resolve_path_to_models_folder(&mut self) -> PathBuf {
        match self.device {
            tch::Device::Cpu => Path::new("models").join("torch-scripts"),
            tch::Device::Cuda(_i) => Path::new("models").join("torch-scripts-cuda"),
            _ => panic!("Device not supported!")
        }
    }

    fn compute_style(&mut self, ref_samples: Vec<f32>) -> anyhow::Result<Tensor> {
        println!("{:?}", ref_samples.len());
        let mut trimmed_ref_samples: Vec<f32> = Vec::new();
        if ref_samples.len() >= STD_SAMPLE_LENGTH {
            trimmed_ref_samples.extend(ref_samples.iter().take(STD_SAMPLE_LENGTH).cloned().collect::<Vec<f32>>());
        }
        else {
            panic!("Error: reference input audio is too short. Received: {}", ref_samples.len());
        }
        let mel_tensor = self.preprocess(trimmed_ref_samples)?;
        let mut mel_tensor_1 = tch::Tensor::ones(mel_tensor.size(), (mel_tensor.kind(), mel_tensor.device()));
        mel_tensor_1.copy_(&mel_tensor);

        let style_encoder = tch::CModule::load(self.resolve_path_to_model("style_encoder.pt"))?;
        let ref_s = style_encoder.forward_ts(&[mel_tensor.unsqueeze(1)])?;
        let predictor_encoder = tch::CModule::load(self.resolve_path_to_model("predictor_encoder.pt"))?;
        let ref_p = predictor_encoder.forward_ts(&[mel_tensor_1.unsqueeze(1)])?;

        Ok(Tensor::cat(&[ref_s, ref_p], 1))
    }

    fn preprocess(&mut self, wave: Vec<f32>) -> anyhow::Result<Tensor> {
        let wave_tensor = Tensor::of_slice(&wave).to_device(self.device);
        let to_mel = tch::CModule::load(self.resolve_path_to_model("to_mel.pt"))?;
        let mel_tensor = to_mel.forward_ts(&[wave_tensor])?;
        let adjusted_tensor: Tensor = 1e-5 + mel_tensor.unsqueeze(0);
        let log_tensor = adjusted_tensor.log();
        let normalized_tensor = (log_tensor - MEAN) / STD;
        Ok(normalized_tensor)
    }

    fn split_by_any_sep(&mut self, str_to_split: String) -> Vec<String> {
        // Regular expression pattern to split by ',', '.', '!', ';', or '?'
        // The capturing group ensures that the delimiters are included in the result
        let pattern = r"([,.!?;])";
        let re = Regex::new(pattern).unwrap();

        // Split the string using the pattern
        let mut split_result: Vec<String> = Vec::new();
        let mut last_end = 0;

        for mat in re.find_iter(&str_to_split) {
            if let Ok(m) = mat {
                if m.start() != last_end {
                    split_result.push(str_to_split[last_end..m.start()].to_string());
                }
                split_result.push(m.as_str().to_string());
                last_end = m.end();
            }
        }

        if last_end < str_to_split.len() {
            split_result.push(str_to_split[last_end..].to_string());
        }

        // Combine the substrings and their delimiters
        let mut combined_result: Vec<String> = Vec::new();
        let mut i = 0;
        while i < split_result.len() - 1 {
            combined_result.push(format!("{}{}", split_result[i], split_result[i + 1]));
            i += 2;
        }
        if split_result.len() % 2 == 1 {
            combined_result.push(split_result[split_result.len() - 1].clone());
        }

        combined_result
    }


    fn length_to_mask(&mut self, lengths: &Tensor) -> Tensor {
        let max_length = lengths.max().to_kind(tch::Kind::Int64).unsqueeze(0).int64_value(&[0]);
        let expand_dim: Vec<i64> = vec![*lengths.size().first().unwrap(), -1];
        let mask = Tensor::arange(max_length, (lengths.kind(), lengths.device()))
            .unsqueeze(0)
            .expand(expand_dim, true);


        let mask = mask + 1;
        let mask_gt = mask.gt_tensor(&lengths.unsqueeze(1));

        mask_gt
    }
}


pub struct ParlerBackend {}

impl ParlerBackend {
    pub fn new(options: TTSOptions) -> anyhow::Result<Self> {
        todo!()
    }
}

impl TextToSpeechBackend for ParlerBackend {
    fn train(&mut self, ref_samples: Vec<f32>) -> anyhow::Result<()> {
        todo!()
    }

    fn predict(&mut self, text: String) -> anyhow::Result<Vec<f32>> {
        todo!()
    }
}