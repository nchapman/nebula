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
                            write!(buf, "<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content)?;
                        } else {
                            write!(buf, "<|im_start|>{}\n", msg.role)?;
                            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                            msg.images.into_iter().for_each(|im| {
                                res.push(Templated::Image(im.0));
                            });
                            buf = BufWriter::new(Vec::new());
                            write!(buf, "{}<|im_end|>\n", msg.content)?;
                        }
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                write!(buf, "<|im_start|>assistant\n")?;
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
                        write!(buf, "\n")?;
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
                        write!(buf, "<|{}|>\n", msg.role)?;
                        if !msg.images.is_empty() {
                            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                            msg.images.into_iter().for_each(|im| {
                                res.push(Templated::Image(im.0));
                            });
                            buf = BufWriter::new(Vec::new());
                        }
                        write!(buf, "{}<|end|>\n", msg.content)?;
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                write!(buf, "<|assistant|>\n")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "zephyr" || template.contains("<|user|>") {
            // zephyr template
            let mut buf =
                msgs.into_iter()
                    .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                        write!(buf, "<|{}|>\n", msg.role)?;
                        if !msg.images.is_empty() {
                            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                            msg.images.into_iter().for_each(|im| {
                                res.push(Templated::Image(im.0));
                            });
                            buf = BufWriter::new(Vec::new());
                        }
                        write!(buf, "{}<|endoftext|>\n", msg.content)?;
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                write!(buf, "<|assistant|>\n")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "monarch" || template.contains("bos_token + message['role']") {
            // mlabonne/AlphaMonarch-7B template (the <s> is included inside history)
            let mut buf = msgs.into_iter().enumerate().try_fold(
                BufWriter::new(Vec::new()),
                |mut buf, (i, msg)| {
                    write!(buf, "{}{}\n", if i == 0 { "" } else { "<s>" }, msg.role)?;
                    if !msg.images.is_empty() {
                        res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                        msg.images.into_iter().for_each(|im| {
                            res.push(Templated::Image(im.0));
                        });
                        buf = BufWriter::new(Vec::new());
                    }
                    write!(buf, "{}</s>\n", msg.content)?;
                    Ok::<_, crate::error::Error>(buf)
                },
            )?;
            if add_ass {
                write!(buf, "<s>assistant\n")?;
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
                            write!(
                                buf,
                                "<start_of_turn>{}\n",
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
                            write!(buf, "{}<end_of_turn>\n", msg.content.trim())?;
                        }
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                write!(buf, "<start_of_turn>model\n")?;
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
                                write!(buf, "SYSTEM: {}\n", msg.content)?;
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
                            write!(buf, "{}\n", msg.content)?;
                        } else {
                            write!(buf, "ASSISTANT: {}</s>\n", msg.content)?;
                        }
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                write!(buf, "ASSISTANT:")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else if template == "deepseek"
            || (template.contains("### Instruction:") && template.contains("<|EOT|>"))
        {
            // deepseek-ai/deepseek-coder-33b-instruct
            let mut buf =
                msgs.into_iter()
                    .try_fold(BufWriter::new(Vec::new()), |mut buf, msg| {
                        if msg.role == Role::System {
                            write!(buf, "{}", msg.content)?;
                        } else if msg.role == Role::User {
                            write!(buf, "### Instruction:\n")?;
                            if !msg.images.is_empty() {
                                res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                                msg.images.into_iter().for_each(|im| {
                                    res.push(Templated::Image(im.0));
                                });
                                buf = BufWriter::new(Vec::new());
                            }
                            write!(buf, "{}\n", msg.content)?;
                        } else {
                            write!(buf, "### Response:\n{}\n<|EOT|>\n", msg.content)?;
                        }
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                write!(buf, "### Response:\n")?;
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
                            write!(buf, "[|system|]{}[|endofturn|]\n", msg.content.trim())?;
                        } else if msg.role == Role::User {
                            write!(buf, "[|user|]")?;
                            if !msg.images.is_empty() {
                                res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
                                msg.images.into_iter().for_each(|im| {
                                    res.push(Templated::Image(im.0));
                                });
                                buf = BufWriter::new(Vec::new());
                            }
                            write!(buf, "{}\n", msg.content.trim())?;
                        } else {
                            write!(buf, "[|assistant|]{}[|endofturn|]\n", msg.content.trim())?;
                        }
                        Ok::<_, crate::error::Error>(buf)
                    })?;
            if add_ass {
                write!(buf, "[|assistant|]")?;
            }
            res.push(Templated::Str(String::from_utf8(buf.into_inner()?)?));
            Ok(res)
        } else {
            Err(crate::error::Error::UnsupportedTemplate(template.into()))
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
            clip_context.embed_image(self.options.n_threads, &image)?
        } else {
            return Err(crate::error::Error::MmprojNotDefined);
        };
        log::debug!("image embedding created: {} tokens", embedded_image.len());
        self.ctx
            .eval_embed_image(embedded_image, 2048, &mut self.n_curr)?;
        Ok(())
    }
}

impl Context for LlamaContext {
    fn eval(&mut self, messages: Vec<Message>) -> Result<()> {
        let templated_message = self.model.apply_template(messages, None, true)?;
        templated_message.into_iter().try_for_each(|m| {
            Ok::<_, crate::error::Error>(match m {
                Templated::Str(st) => self.eval_str(&st, false)?,
                Templated::Image(st) => self.eval_image(&st)?,
            })
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
            if self.model.token_is_eog(token_id)? {
                break;
            }
            let token_str = self.ctx.token_to_piece(&token_id)?;
            //eprint!("{token_str}");
            token_callback(token_str);
        }
        Ok(())
    }
}
