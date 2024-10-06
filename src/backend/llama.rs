#![allow(clippy::too_many_arguments)]
use std::{
    borrow::Cow,
    io::{BufWriter, Write},
    num::NonZeroU32,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, Mutex},
};

use crate::{
    options::{ContextOptions, Message, ModelOptions, PredictOptions, Role},
    Result,
};
use llama_cpp::{
    clip::ClipContext,
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sample::Sampler,
    token::LlamaToken,
};

use super::{Context, Model};

lazy_static::lazy_static! {
    static ref LLAMA_BACKEND: Arc<LlamaBackend> = Arc::new(LlamaBackend::init().unwrap());
}

impl From<ModelOptions> for LlamaModelParams {
    fn from(val: ModelOptions) -> Self {
        let lmp = Self::default();
        if !val.cpu {
            lmp.with_n_gpu_layers(val.n_gpu_layers as u32)
        } else {
            lmp
        }
    }
}

impl From<&ContextOptions> for LlamaContextParams {
    fn from(val: &ContextOptions) -> Self {
        Self::default()
            .with_n_ctx(NonZeroU32::new(val.n_ctx as u32))
            .with_n_threads(val.n_threads as i32)
            .with_n_batch(2048)
    }
}

#[derive(Debug)]
pub enum Templated {
    Str(String),
    Image(Vec<u8>),
}

#[derive(Clone)]
pub struct Llama {
    name: String,
    model: LlamaModel,
    mmproj: Option<ClipContext>,
}

impl Llama {
    pub fn new(
        model_path: impl Into<PathBuf>,
        options: ModelOptions,
        callback: Option<impl FnMut(f32) -> bool + 'static>,
    ) -> Result<Self> {
        let mut lmp: LlamaModelParams = options.into();
        if let Some(cb) = callback {
            lmp = lmp.with_load_process_callback(cb);
        }
        let model_params = Box::pin(lmp);
        let mm: PathBuf = model_path.into();
        let model = LlamaModel::load_from_file(&LLAMA_BACKEND, Path::new(&mm), &model_params)?;
        Ok(Self {
            name: mm.to_str().unwrap().to_string(),
            model,
            mmproj: None,
        })
    }

    pub fn token_is_eog(&self, id: LlamaToken) -> Result<bool> {
        Ok(self.model.token_is_eog(id))
    }

    pub fn template_stops(&self, template: Option<String>) -> Result<Vec<&'static str>> {
        let template: Cow<str> = if let Some(tt) = template {
            tt.into()
        } else if let Some(tt) = self.model.meta_val_str("tokenizer.chat_template")? {
            tt.into()
        } else {
            "chatml".into()
        };
        if template == "chatml" || template.contains("<|im_start|>") {
            Ok(vec!["<|im_start|>", "<|im_end|>"])
        } else if template == "llama2" || template == "mistral" || template.contains("[INST]") {
            Ok(vec!["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"])
        } else if template == "phi3"
            || (template.contains("<|assistant|>") && template.contains("<|end|>"))
        {
            Ok(vec!["<|end|>", "<|system|>", "<|user|>", "<|assistant|>"])
        } else if template == "zephyr" || template.contains("<|user|>") {
            Ok(vec!["<|system|>", "</s>", "<|user|>", "<|assistant|>"])
        } else if template == "monarch" || template.contains("bos_token + message['role']") {
            Ok(vec![])
        } else if template == "gemma"
            || template == "gemma2"
            || template.contains("<start_of_turn>")
        {
            Ok(vec!["<start_of_turn>", "<end_of_turn>"])
        } else if template == "orion" || template.contains("'\\n\\nAssistant: ' + eos_token") {
            Ok(vec![])
        } else if template == "openchat" || template.contains("GPT4 Correct ") {
            Ok(vec!["<|end_of_turn|>"])
        } else if template == "vicuna"
            || template == "vicuna-orca"
            || (template.contains("USER: ") && template.contains("ASSISTANT: "))
        {
            Ok(vec!["USER:", "ASSISTANT:"])
        } else if false
            && (template == "deepseek"
                || (template.contains("### Instruction:") && template.contains("<|EOT|>")))
        {
            Ok(vec!["<|EOT|>"])
        } else if template == "command-r"
            || (template.contains("<|START_OF_TURN_TOKEN|>") && template.contains("<|USER_TOKEN|>"))
        {
            Ok(vec![])
        } else if template == "llama3"
            || (template.contains("<|start_header_id|>") && template.contains("<|end_header_id|>"))
        {
            Ok(vec![
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ])
        } else if template == "chatglm3" || template.contains("[gMASK]sop") {
            Ok(vec![])
        } else if template == "chatglm4" || template.contains("[gMASK]<sop>") {
            Ok(vec![])
        } else if template == "minicpm" || template.contains("<用户>") {
            Ok(vec![])
        } else if template == "deepseek2"
            || template.contains("'Assistant: ' + message['content'] + eos_token")
        {
            Ok(vec![])
        } else if template == "exaone3"
            || (template.contains("[|system|]")
                && template.contains("[|assistant|]")
                && template.contains("[|endofturn|]"))
        {
            Ok(vec![])
        } else {
            self.template_stops(Some("chatml".to_string()))
        }
    }

    pub fn apply_template(
        &self,
        msgs: Vec<Message>,
        template: Option<String>,
        add_ass: bool,
    ) -> Result<Vec<Templated>> {
        let template: Cow<str> = if let Some(tt) = template {
            tt.into()
        } else if let Some(tt) = self.model.meta_val_str("tokenizer.chat_template")? {
            tt.into()
        } else {
            "chatml".into()
        };
        let mut res = vec![];
        if template == "chatml" || template.contains("<|im_start|>") {
            let mut buf =
                msgs.into_iter()
                    .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                        if msg.images.is_empty() {
                            writeln!(buf, "<|im_start|>{}\n{}<|im_end|>", msg.role, msg.content)?;
                        } else {
                            writeln!(buf, "<|im_start|>{}", msg.role)?;
                            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                            msg.images.into_iter().for_each(|im| {
                                res.push(Templated::Image(im.0));
                            });
                            buf = BufWriter::new(Vec::new());
                            writeln!(buf, "{}<|im_end|>", msg.content)?;
                        }
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                writeln!(buf, "<|im_start|>assistant")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "llama2" || template == "mistral" || template.contains("[INST]") {
            // llama2 template and its variants
            // [variant] support system message
            let support_system_message = template.contains("<<SYS>>") || template == "mistral";
            // [variant] space before + after response
            let space_around_response = template.contains("' ' + eos_token");
            // [variant] add BOS inside history
            let add_bos_inside_history = template.contains("bos_token + '[INST]");
            // [variant] trim spaces from the input message
            let strip_message = template.contains("content.strip()");
            // construct the prompt
            let mut is_inside_turn = true; // skip BOS at the beginning
            let mut buf = BufWriter::new(Vec::new());
            write!(buf, "[INST]")?;
            let buf = msgs.into_iter().try_fold(buf, |mut buf, msg| {
                let content = if strip_message {
                    msg.content.trim().to_string()
                } else {
                    msg.content
                };
                if !is_inside_turn {
                    is_inside_turn = true;
                    if add_bos_inside_history {
                        write!(buf, "<s>[INST]")?;
                    } else {
                        write!(buf, "[INST]")?;
                    }
                }
                if msg.role == Role::System {
                    if support_system_message {
                        write!(buf, "<<SYS>>\n{}\n<</SYS>>\n\n", content)?;
                    } else {
                        // if the model does not support system message, we still include it in the first message, but without <<SYS>>
                        writeln!(buf)?;
                    }
                } else if msg.role == Role::User {
                    if !msg.images.is_empty() {
                        res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                        msg.images.into_iter().for_each(|im| {
                            res.push(Templated::Image(im.0));
                        });
                        buf = BufWriter::new(Vec::new());
                    }
                    write!(buf, "{} [/INST]", content)?;
                } else {
                    write!(
                        buf,
                        "{}{}{}</s>",
                        if space_around_response { " " } else { "" },
                        content,
                        if space_around_response { " " } else { "" }
                    )?;
                    is_inside_turn = false;
                }
                Ok::<_, crate::error::Error>(buf)
            })?;
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            // llama2 templates seem to not care about "add_generation_prompt"
            Ok(res)
        } else if template == "phi3"
            || (template.contains("<|assistant|>") && template.contains("<|end|>"))
        {
            // Phi 3
            let mut buf =
                msgs.into_iter()
                    .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                        writeln!(buf, "<|{}|>", msg.role)?;
                        if !msg.images.is_empty() {
                            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                            msg.images.into_iter().for_each(|im| {
                                res.push(Templated::Image(im.0));
                            });
                            buf = BufWriter::new(Vec::new());
                        }
                        writeln!(buf, "{}<|end|>", msg.content)?;
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                writeln!(buf, "<|assistant|>")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "zephyr" || template.contains("<|user|>") {
            // zephyr template
            let mut buf =
                msgs.into_iter()
                    .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                        writeln!(buf, "<|{}|>", msg.role)?;
                        if !msg.images.is_empty() {
                            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                            msg.images.into_iter().for_each(|im| {
                                res.push(Templated::Image(im.0));
                            });
                            buf = BufWriter::new(Vec::new());
                        }
                        writeln!(buf, "{}<|endoftext|>", msg.content)?;
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                writeln!(buf, "<|assistant|>")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "monarch" || template.contains("bos_token + message['role']") {
            // mlabonne/AlphaMonarch-7B template (the <s> is included inside history)
            let mut buf = msgs.into_iter().enumerate().try_fold(
                BufWriter::new(Vec::new()),
                |mut buf, (i, msg)| {
                    writeln!(buf, "{}{}", if i == 0 { "" } else { "<s>" }, msg.role)?;
                    if !msg.images.is_empty() {
                        res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                        msg.images.into_iter().for_each(|im| {
                            res.push(Templated::Image(im.0));
                        });
                        buf = BufWriter::new(Vec::new());
                    }
                    writeln!(buf, "{}</s>", msg.content)?;
                    Ok::<_, crate::error::Error>(buf)
                },
            )?;
            if add_ass {
                writeln!(buf, "<s>assistant")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "gemma"
            || template == "gemma2"
            || template.contains("<start_of_turn>")
        {
            // google/gemma-7b-it
            let mut system_prompt = "".to_string();
            let mut buf =
                msgs.into_iter()
                    .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                        if msg.role == Role::System {
                            // there is no system message for gemma, but we will merge it with user prompt, so nothing is broken
                            system_prompt = msg.content.trim().to_string();
                        } else {
                            writeln!(
                                buf,
                                "<start_of_turn>{}",
                                if msg.role == Role::Assistant {
                                    "model".to_string()
                                } else {
                                    msg.role.to_string()
                                }
                            )?;
                            if !system_prompt.is_empty() && msg.role == Role::User {
                                write!(buf, "{}\n\n", system_prompt)?;
                                system_prompt = "".to_string();
                            }
                            if !msg.images.is_empty() {
                                res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                                msg.images.into_iter().for_each(|im| {
                                    res.push(Templated::Image(im.0));
                                });
                                buf = BufWriter::new(Vec::new());
                            }
                            writeln!(buf, "{}<end_of_turn>", msg.content.trim())?;
                        }
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                writeln!(buf, "<start_of_turn>model")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "orion" || template.contains("'\\n\\nAssistant: ' + eos_token") {
            // OrionStarAI/Orion-14B-Chat
            let mut system_prompt = "".to_string();
            let buf = msgs
                .into_iter()
                .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                    if msg.role == Role::System {
                        // there is no system message support, we will merge it with user prompt
                        system_prompt = msg.content;
                    } else if msg.role == Role::User {
                        write!(buf, "Human: ")?;
                        if !system_prompt.is_empty() && msg.role == Role::User {
                            write!(buf, "{}\n\n", system_prompt)?;
                            system_prompt = "".to_string();
                        }
                        if !msg.images.is_empty() {
                            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                            msg.images.into_iter().for_each(|im| {
                                res.push(Templated::Image(im.0));
                            });
                            buf = BufWriter::new(Vec::new());
                        }
                        write!(buf, "{}\n\nAssistant: </s>", msg.content)?;
                    } else {
                        write!(buf, "{}</s>", msg.content)?;
                    }
                    Ok::<_, crate::error::Error>(buf)
                })?;
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "openchat" || template.contains("GPT4 Correct ") {
            // openchat/openchat-3.5-0106,
            let mut buf =
                msgs.into_iter()
                    .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                        if msg.role == Role::System {
                            write!(buf, "{}<|end_of_turn|>", msg.content)?;
                        } else {
                            write!(
                                buf,
                                "GPT4 Correct {}: ",
                                msg.role.to_string().to_uppercase()
                            )?;
                            if !msg.images.is_empty() {
                                res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                                msg.images.into_iter().for_each(|im| {
                                    res.push(Templated::Image(im.0));
                                });
                                buf = BufWriter::new(Vec::new());
                            }
                            write!(buf, "{}<|end_of_turn|>", msg.content)?;
                        }
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                write!(buf, "GPT4 Correct Assistant:")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "vicuna"
            || template == "vicuna-orca"
            || (template.contains("USER: ") && template.contains("ASSISTANT: "))
        {
            // eachadea/vicuna-13b-1.1 (and Orca variant)
            let mut buf =
                msgs.into_iter()
                    .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                        if msg.role == Role::System {
                            // Orca-Vicuna variant uses a system prefix
                            if template == "vicuna-orca" || template.contains("SYSTEM: ") {
                                writeln!(buf, "SYSTEM: {}", msg.content)?;
                            } else {
                                write!(buf, "{}\n\n", msg.content)?;
                            }
                        } else if msg.role == Role::User {
                            write!(buf, "USER: ")?;
                            if !msg.images.is_empty() {
                                res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                                msg.images.into_iter().for_each(|im| {
                                    res.push(Templated::Image(im.0));
                                });
                                buf = BufWriter::new(Vec::new());
                            }
                            writeln!(buf, "{}", msg.content)?;
                        } else {
                            writeln!(buf, "ASSISTANT: {}</s>", msg.content)?;
                        }
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                write!(buf, "ASSISTANT:")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if false
            && (template == "deepseek"
                || (template.contains("### Instruction:") && template.contains("<|EOT|>")))
        {
            // deepseek-ai/deepseek-coder-33b-instruct
            let mut buf =
                msgs.into_iter()
                    .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                        if msg.role == Role::System {
                            write!(buf, "{}", msg.content)?;
                        } else if msg.role == Role::User {
                            writeln!(buf, "### Instruction:")?;
                            if !msg.images.is_empty() {
                                res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                                msg.images.into_iter().for_each(|im| {
                                    res.push(Templated::Image(im.0));
                                });
                                buf = BufWriter::new(Vec::new());
                            }
                            writeln!(buf, "{}", msg.content)?;
                        } else {
                            write!(buf, "### Response:\n{}\n<|EOT|>\n", msg.content)?;
                        }
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                writeln!(buf, "### Response:")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "command-r"
            || (template.contains("<|START_OF_TURN_TOKEN|>") && template.contains("<|USER_TOKEN|>"))
        {
            // CohereForAI/c4ai-command-r-plus
            let mut buf =
                msgs.into_iter()
                    .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                        if msg.role == Role::System {
                            write!(
                                buf,
                                "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{}<|END_OF_TURN_TOKEN|>",
                                msg.content.trim(),
                            )?;
                        } else if msg.role == Role::User {
                            write!(buf, "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>")?;
                            if !msg.images.is_empty() {
                                res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                                msg.images.into_iter().for_each(|im| {
                                    res.push(Templated::Image(im.0));
                                });
                                buf = BufWriter::new(Vec::new());
                            }
                            write!(buf, "{}<|END_OF_TURN_TOKEN|>", msg.content.trim())?;
                        } else {
                            write!(
                                buf,
                                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{}<|END_OF_TURN_TOKEN|>",
                                msg.content
                            )?;
                        }
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                write!(buf, "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "llama3"
            || (template.contains("<|start_header_id|>") && template.contains("<|end_header_id|>"))
        {
            // Llama 3
            let mut buf =
                msgs.into_iter()
                    .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                        if msg.role == Role::User {
                            write!(buf, "<|start_header_id|>{}<|end_header_id|>\n\n", msg.role)?;
                            if !msg.images.is_empty() {
                                res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                                msg.images.into_iter().for_each(|im| {
                                    res.push(Templated::Image(im.0));
                                });
                                buf = BufWriter::new(Vec::new());
                            }
                            write!(buf, "{}<|eot_id|>", msg.content.trim())?;
                        } else {
                            write!(
                                buf,
                                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                                msg.role, msg.content
                            )?;
                        }
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                write!(buf, "<|start_header_id|>assistant<|end_header_id|>\n\n")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "chatglm3" || template.contains("[gMASK]sop") {
            // chatglm3-6b
            let mut buf = BufWriter::new(Vec::new());
            write!(buf, "[gMASK]sop")?;
            let mut buf = msgs.into_iter().try_fold(buf, |mut buf, msg| {
                if msg.role == Role::User {
                    write!(buf, "<|{}|>\n ", msg.role)?;
                    if !msg.images.is_empty() {
                        res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                        msg.images.into_iter().for_each(|im| {
                            res.push(Templated::Image(im.0));
                        });
                        buf = BufWriter::new(Vec::new());
                    }
                    write!(buf, "{}", msg.content)?;
                } else {
                    write!(buf, "<|{}|>\n {}", msg.role, msg.content)?;
                }
                Ok::<_, crate::error::Error>(buf)
            })?;
            if add_ass {
                write!(buf, "<|assistant|>")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "chatglm4" || template.contains("[gMASK]<sop>") {
            // chatglm3-6b
            let mut buf = BufWriter::new(Vec::new());
            write!(buf, "[gMASK]<sop>")?;
            let mut buf = msgs.into_iter().try_fold(buf, |mut buf, msg| {
                if msg.role == Role::User {
                    write!(buf, "<|{}|>\n ", msg.role)?;
                    if !msg.images.is_empty() {
                        res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                        msg.images.into_iter().for_each(|im| {
                            res.push(Templated::Image(im.0));
                        });
                        buf = BufWriter::new(Vec::new());
                    }
                    write!(buf, "{}", msg.content)?;
                } else {
                    write!(buf, "<|{}|>\n {}", msg.role, msg.content)?;
                }
                Ok::<_, crate::error::Error>(buf)
            })?;
            if add_ass {
                write!(buf, "<|assistant|>")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "minicpm" || template.contains("<用户>") {
            // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
            let buf = msgs
                .into_iter()
                .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                    if msg.role == Role::User {
                        write!(buf, "<用户>")?;
                        if !msg.images.is_empty() {
                            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                            msg.images.into_iter().for_each(|im| {
                                res.push(Templated::Image(im.0));
                            });
                            buf = BufWriter::new(Vec::new());
                        }
                        write!(buf, "{}<AI>", msg.content.trim())?;
                    } else {
                        write!(buf, "{}", msg.content.trim())?;
                    }
                    Ok::<_, crate::error::Error>(buf)
                })?;
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "deepseek2"
            || template.contains("'Assistant: ' + message['content'] + eos_token")
        {
            // DeepSeek-V2
            let mut buf =
                msgs.into_iter()
                    .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                        if msg.role == Role::System {
                            write!(buf, "{}\n\n", msg.content)?;
                        } else if msg.role == Role::User {
                            write!(buf, "User: :")?;
                            if !msg.images.is_empty() {
                                res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                                msg.images.into_iter().for_each(|im| {
                                    res.push(Templated::Image(im.0));
                                });
                                buf = BufWriter::new(Vec::new());
                            }
                            write!(buf, "{}\n\n", msg.content)?;
                        } else {
                            write!(buf, "Assistant: {}<｜end▁of▁sentence｜>", msg.content)?;
                        }
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                write!(buf, "Assistant:")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "exaone3"
            || (template.contains("[|system|]")
                && template.contains("[|assistant|]")
                && template.contains("[|endofturn|]"))
        {
            // ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct/discussions/8#66bae61b1893d14ee8ed85bb
            // EXAONE-3.0-7.8B-Instruct
            let mut buf =
                msgs.into_iter()
                    .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                        if msg.role == Role::System {
                            writeln!(buf, "[|system|]{}[|endofturn|]", msg.content.trim())?;
                        } else if msg.role == Role::User {
                            write!(buf, "[|user|]")?;
                            if !msg.images.is_empty() {
                                res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                                msg.images.into_iter().for_each(|im| {
                                    res.push(Templated::Image(im.0));
                                });
                                buf = BufWriter::new(Vec::new());
                            }
                            writeln!(buf, "{}", msg.content.trim())?;
                        } else {
                            writeln!(buf, "[|assistant|]{}[|endofturn|]", msg.content.trim())?;
                        }
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                write!(buf, "[|assistant|]")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else {
            //            log::info!("unsupported template {}, used chatml instead", template);
            //            Err(crate::error::Error::UnsupportedTemplate(template.into()))
            self.apply_template(msgs, Some("chatml".to_string()), add_ass)
        }
    }
}

impl Model for Llama {
    fn name(&self) -> Result<&str> {
        Ok(&self.name)
    }
    fn with_mmproj(&mut self, mmproj: PathBuf) -> Result<()> {
        let clip_context = ClipContext::load(Path::new(&mmproj))?;
        self.mmproj = Some(clip_context);
        Ok(())
    }
    fn new_context(&self, options: ContextOptions) -> Result<Pin<Box<Mutex<dyn Context>>>> {
        Ok(Box::pin(Mutex::new(LlamaContext::new(self, options)?)))
    }
}

pub struct LlamaContext {
    options: ContextOptions,
    logit: i32,
    n_curr: i32,
    ctx: Pin<Box<llama_cpp::context::LlamaContext>>,
    //    sampler: Pin<Box<Sampler>>,
    model: Arc<Llama>,
}

impl<'a> LlamaContext {
    pub fn new(model: &'a Llama, options: ContextOptions) -> Result<Self> {
        let ctx_params: LlamaContextParams = (&options).into();
        let ctx = Self {
            options,
            logit: 0,
            n_curr: 0,
            ctx: Box::pin(model.model.new_context(&LLAMA_BACKEND, ctx_params)?),
            model: Arc::new(model.clone()),
        };
        Ok(ctx)
    }

    fn eval_str(&mut self, prompt: &str, add_bos: bool) -> Result<()> {
        self.logit = self.ctx.eval_string(
            prompt,
            2048,
            if add_bos {
                AddBos::Always
            } else {
                AddBos::Never
            },
            &mut self.n_curr,
        )?;
        Ok(())
    }

    fn eval_image(&mut self, image: &[u8]) -> Result<()> {
        let embedded_image = if let Some(clip_context) = &self.model.mmproj {
            clip_context.embed_image(self.options.n_threads, image)?
        } else {
            return Err(crate::error::Error::MmprojNotDefined);
        };
        log::debug!("image embedding created: {} tokens", embedded_image.len());
        self.ctx
            .eval_embed_image(embedded_image, 2048, &mut self.n_curr)?;
        Ok(())
    }

    fn find_partial_stop_pos(&self, stop: &str, text: &str) -> Result<Option<usize>> {
        if !text.is_empty() && !stop.is_empty() {
            let text_last_char = text.chars().last().unwrap();
            Ok(stop.char_indices().rev().find_map(|(i, ch)| {
                if ch == text_last_char {
                    let current_partial = stop[..i + 1].to_string();
                    if text.ends_with(&current_partial) {
                        return Some(text.len() - i - 1);
                    }
                }
                None
            }))
        } else {
            Ok(None)
        }
    }

    fn find_stopping_string(
        &self,
        text: &str,
        last_token_size: usize,
        is_stop_type_full: bool,
    ) -> Result<(bool, Option<usize>)> {
        let mut has_stop_token = true;
        let mut stop_pos = None;
        for w in self.model.template_stops(None)? {
            let mut pos = None;
            if is_stop_type_full {
                let tmp = w.len() + last_token_size;
                let from_pos = if text.len() > tmp {
                    text.len() - tmp
                } else {
                    0
                };
                if let Some(p) = text[from_pos..].find(w) {
                    pos = Some(from_pos + p);
                }
            } else {
                pos = self.find_partial_stop_pos(w, text)?;
            }
            if pos != stop_pos && (stop_pos.is_none() || pos < stop_pos) {
                if is_stop_type_full {
                    has_stop_token = false;
                }
                stop_pos = pos;
            }
        }
        Ok((has_stop_token, stop_pos))
    }

    fn process_token(
        &self,
        mut n_sent_text: usize,
        mut generated_string: String,
        token: LlamaToken,
        callback: std::sync::Arc<Box<dyn Fn(String) -> bool + Send + 'static>>,
    ) -> Result<(bool, String, usize)> {
        let mut text_to_send = "".to_string();
        let token_str = self.ctx.token_to_piece(&token)?;
        if !self.model.token_is_eog(token)? {
            generated_string += &token_str;
        }
        let mut has_next_token = true;
        let mut incomplete = false;
        for i in 1..std::cmp::min(5, generated_string.len() + 1) {
            let c = generated_string.as_bytes()[generated_string.len() - i];
            if (c & 0xC0) == 0x80 {
                // continuation byte: 10xxxxxx
                continue;
            }
            if (c & 0xE0) == 0xC0 {
                // 2-byte character: 110xxxxx ...
                incomplete = i < 2;
            } else if (c & 0xF0) == 0xE0 {
                // 3-byte character: 1110xxxx ...
                incomplete = i < 3;
            } else if (c & 0xF8) == 0xF0 {
                // 4-byte character: 11110xxx ...
                incomplete = i < 4;
            }
            // else 1-byte character or invalid byte
            break;
        }
        if !incomplete {
            let mut pos = std::cmp::min(n_sent_text, generated_string.len());
            if !self.model.token_is_eog(token)? {
                let str_test = generated_string[pos..].to_string();
                let is_stop_full;
                let (h, mut stop_pos) =
                    self.find_stopping_string(&str_test, token_str.len(), true)?;
                has_next_token = h;
                if let Some(sp) = &stop_pos {
                    is_stop_full = true;
                    generated_string = generated_string[pos + sp..].to_string();
                    pos = std::cmp::min(n_sent_text, generated_string.len());
                } else {
                    is_stop_full = false;
                    (has_next_token, stop_pos) =
                        self.find_stopping_string(&str_test, token_str.len(), false)?;
                }
                if stop_pos.is_none()
                    || (!has_next_token
                        && !is_stop_full
                        && stop_pos.is_some()
                        && stop_pos.unwrap() > 0)
                {
                    text_to_send = generated_string[pos..].to_string();
                    n_sent_text += text_to_send.len();
                }
            } else {
                text_to_send = generated_string[pos..].to_string();
                n_sent_text += text_to_send.len();
            }
            if !callback(text_to_send) {
                return Ok((false, generated_string, n_sent_text));
            }
        }
        if incomplete {
            has_next_token = true;
        }
        if self.model.token_is_eog(token)? {
            has_next_token = false;
        }
        Ok((has_next_token, generated_string, n_sent_text))
    }
}

impl Context for LlamaContext {
    fn eval(&mut self, messages: Vec<Message>) -> Result<()> {
        let templated_message = self.model.apply_template(messages, None, true)?;
        templated_message.into_iter().try_for_each(|m| {
            match m {
                Templated::Str(st) => self.eval_str(&st, false)?,
                Templated::Image(st) => self.eval_image(&st)?,
            }
            Ok::<_, crate::error::Error>(())
        })?;
        Ok(())
    }

    fn predict(&mut self, params: &PredictOptions) -> Result<String> {
        let res = Arc::new(Mutex::new("".to_string()));
        let rres = res.clone();
        self.predict_with_callback(
            params,
            std::sync::Arc::new(Box::new(move |token| {
                rres.lock().unwrap().push_str(&token);
                true
            })),
        )?;
        let rres = res.lock().unwrap();
        Ok(rres.clone())
    }

    fn predict_with_callback(
        &mut self,
        params: &PredictOptions,
        token_callback: std::sync::Arc<Box<dyn Fn(String) -> bool + Send + 'static>>,
    ) -> Result<()> {
        let mut generated_text = "".to_string();
        let mut n_sent_text = 0;
        let mut sampler = Sampler::new(&self.model.model, params.clone().into())?;
        let stop = if let Some(mm) = params.max_len {
            mm as usize
        } else {
            usize::MAX
        };
        for _ in 0..stop {
            let token_id = sampler.sample(&self.ctx, -1, false)?;
            sampler.accept(token_id, true)?;
            self.ctx.eval_id(token_id, &mut self.n_curr)?;
            let (has_next_token, g, n) = self.process_token(
                n_sent_text,
                generated_text,
                token_id,
                token_callback.clone(),
            )?;
            generated_text = g;
            n_sent_text = n;
            if !has_next_token {
                break;
            }
            //            let token_str = self.ctx.token_to_piece(&token_id)?;
            //            token_callback(token_str);
        }
        Ok(())
    }
}
