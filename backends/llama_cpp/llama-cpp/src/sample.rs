use crate::{
    context::LlamaContext,
    model::LlamaModel,
    token::{data::LlamaTokenData, data_array::LlamaTokenDataArray, LlamaToken},
};
use ringbuffer::{AllocRingBuffer, RingBuffer};
use std::{ffi::CString, ptr::NonNull};

#[derive(Debug, Clone)]
pub enum SamplerType {
    None = 0,
    TopK = 1,
    TopP = 2,
    MinP = 3,
    TfsZ = 4,
    TypicalP = 5,
    Temperature = 6,
}

#[derive(Debug, Clone)]
#[bon::builder]
pub struct SamplingParams {
    #[builder(default = llama_cpp_sys::LLAMA_DEFAULT_SEED)]
    pub seed: u32,
    #[builder(default = 64)]
    pub n_prev: i32,
    #[builder(default = 0)]
    pub n_probs: i32,
    #[builder(default = 0)]
    pub min_keep: i32,
    #[builder(default = 40)]
    pub top_k: i32,
    #[builder(default = 0.95)]
    pub top_p: f32,
    #[builder(default = 0.05)]
    pub min_p: f32,
    #[builder(default = 1.0)]
    pub tfs_z: f32,
    #[builder(default = 1.0)]
    pub typ_p: f32,
    #[builder(default = 0.8)]
    pub temp: f32,
    #[builder(default = 0.0)]
    pub dynatemp_range: f32,
    #[builder(default = 1.0)]
    pub dynatemp_exponent: f32,
    #[builder(default = 64)]
    pub penalty_last_n: i32,
    #[builder(default = 1.0)]
    pub penalty_repeat: f32,
    #[builder(default = 0.0)]
    pub penalty_freq: f32,
    #[builder(default = 0.0)]
    pub penalty_present: f32,
    #[builder(default = 0)]
    pub mirostat: i32,
    #[builder(default = 5.0)]
    pub mirostat_tau: f32,
    #[builder(default = 0.1)]
    pub mirostat_eta: f32,
    #[builder(default = false)]
    pub penalize_nl: bool,
    #[builder(default = false)]
    pub ignore_eos: bool,
    #[builder(default = vec![
        SamplerType::TopK,
        SamplerType::TfsZ,
        SamplerType::TypicalP,
        SamplerType::TopP,
        SamplerType::MinP,
        SamplerType::Temperature,
    ])]
    pub samplers: Vec<SamplerType>,
    #[builder(default = String::from(""))]
    pub grammar: String,
    #[builder(default)]
    pub logit_bias: Vec<llama_cpp_sys::llama_logit_bias>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::builder().build()
    }
}

#[derive(Debug)]
pub struct Sampler {
    params: SamplingParams,
    grmr: NonNull<llama_cpp_sys::llama_sampler>,
    chain: NonNull<llama_cpp_sys::llama_sampler>,
    prev: AllocRingBuffer<LlamaToken>,
    //    cur: Vec<LlamaTokenData>,
    cur_p: LlamaTokenDataArray,
}

unsafe impl Send for Sampler {}
unsafe impl Sync for Sampler {}

impl Clone for Sampler {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            grmr: NonNull::new(self.grmr.as_ptr()).unwrap(),
            chain: NonNull::new(self.chain.as_ptr()).unwrap(),
            prev: self.prev.clone(),
            //            cur: self.cur.clone(),
            cur_p: self.cur_p.clone(),
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys::llama_sampler_free(self.grmr.as_mut()) };
        unsafe { llama_cpp_sys::llama_sampler_free(self.chain.as_mut()) };
    }
}

impl Sampler {
    pub fn new(model: &LlamaModel, params: SamplingParams) -> crate::Result<Self> {
        let mut lparams = unsafe { llama_cpp_sys::llama_sampler_chain_default_params() };
        lparams.no_perf = false;
        let n_prev = params.n_prev;
        let gramm_str = params.grammar.clone();
        let gstr = CString::new(&gramm_str[..]).unwrap();
        let rootstr = CString::new("root").unwrap();
        let mut res = Self {
            params,
            grmr: NonNull::new(unsafe {
                llama_cpp_sys::llama_sampler_init_grammar(
                    model.model.model.as_ptr(),
                    gstr.as_ptr() as *const char,
                    rootstr.as_ptr() as *const char,
                )
            })
            .ok_or(crate::LLamaCppError::SamplerInitGramar)?,
            chain: NonNull::new(unsafe { llama_cpp_sys::llama_sampler_chain_init(lparams) })
                .ok_or(crate::LLamaCppError::SamplerInitChain)?,
            prev: AllocRingBuffer::new(std::cmp::max(32, n_prev as usize)),
            //cur: vec![],
            cur_p: LlamaTokenDataArray::new(vec![], -1, false),
        };
        unsafe {
            llama_cpp_sys::llama_sampler_chain_add(
                res.chain.as_mut(),
                llama_cpp_sys::llama_sampler_init_logit_bias(
                    llama_cpp_sys::llama_n_vocab(model.model.model.as_ptr()),
                    res.params.logit_bias.len() as i32,
                    res.params.logit_bias.as_ptr(),
                ),
            )
        };

        unsafe {
            llama_cpp_sys::llama_sampler_chain_add(
                res.chain.as_mut(),
                llama_cpp_sys::llama_sampler_init_penalties(
                    llama_cpp_sys::llama_n_vocab(model.model.model.as_ptr()),
                    llama_cpp_sys::llama_token_eos(model.model.model.as_ptr()),
                    llama_cpp_sys::llama_token_nl(model.model.model.as_ptr()),
                    res.params.penalty_last_n,
                    res.params.penalty_repeat,
                    res.params.penalty_freq,
                    res.params.penalty_present,
                    res.params.penalize_nl,
                    res.params.ignore_eos,
                ),
            )
        };

        if res.params.temp > 0.0 {
            match res.params.mirostat {
                0 => {
                    for cnstr in &res.params.samplers {
                        match cnstr {
                            SamplerType::TopK => unsafe {
                                llama_cpp_sys::llama_sampler_chain_add(
                                    res.chain.as_mut(),
                                    llama_cpp_sys::llama_sampler_init_top_k(res.params.top_k),
                                )
                            },
                            SamplerType::TopP => unsafe {
                                llama_cpp_sys::llama_sampler_chain_add(
                                    res.chain.as_mut(),
                                    llama_cpp_sys::llama_sampler_init_top_p(
                                        res.params.top_p,
                                        res.params.min_keep as usize,
                                    ),
                                )
                            },
                            SamplerType::MinP => unsafe {
                                llama_cpp_sys::llama_sampler_chain_add(
                                    res.chain.as_mut(),
                                    llama_cpp_sys::llama_sampler_init_min_p(
                                        res.params.min_p,
                                        res.params.min_keep as usize,
                                    ),
                                )
                            },
                            SamplerType::TfsZ => unsafe {
                                llama_cpp_sys::llama_sampler_chain_add(
                                    res.chain.as_mut(),
                                    llama_cpp_sys::llama_sampler_init_tail_free(
                                        res.params.tfs_z,
                                        res.params.min_keep as usize,
                                    ),
                                )
                            },
                            SamplerType::TypicalP => unsafe {
                                llama_cpp_sys::llama_sampler_chain_add(
                                    res.chain.as_mut(),
                                    llama_cpp_sys::llama_sampler_init_typical(
                                        res.params.typ_p,
                                        res.params.min_keep as usize,
                                    ),
                                )
                            },
                            SamplerType::Temperature => unsafe {
                                llama_cpp_sys::llama_sampler_chain_add(
                                    res.chain.as_mut(),
                                    llama_cpp_sys::llama_sampler_init_temp_ext(
                                        res.params.temp,
                                        res.params.dynatemp_range,
                                        res.params.dynatemp_exponent,
                                    ),
                                )
                            },
                            _ => panic!("unknown sampler type"),
                        }
                    }
                    unsafe {
                        llama_cpp_sys::llama_sampler_chain_add(
                            res.chain.as_mut(),
                            llama_cpp_sys::llama_sampler_init_softmax(),
                        )
                    };
                    unsafe {
                        llama_cpp_sys::llama_sampler_chain_add(
                            res.chain.as_mut(),
                            llama_cpp_sys::llama_sampler_init_dist(res.params.seed),
                        )
                    };
                }
                1 => {
                    unsafe {
                        llama_cpp_sys::llama_sampler_chain_add(
                            res.chain.as_mut(),
                            llama_cpp_sys::llama_sampler_init_temp(res.params.temp),
                        )
                    };
                    unsafe {
                        llama_cpp_sys::llama_sampler_chain_add(
                            res.chain.as_mut(),
                            llama_cpp_sys::llama_sampler_init_mirostat(
                                llama_cpp_sys::llama_n_vocab(model.model.model.as_ptr()),
                                res.params.seed,
                                res.params.mirostat_tau,
                                res.params.mirostat_eta,
                                100,
                            ),
                        )
                    };
                }
                2 => {
                    unsafe {
                        llama_cpp_sys::llama_sampler_chain_add(
                            res.chain.as_mut(),
                            llama_cpp_sys::llama_sampler_init_temp(res.params.temp),
                        )
                    };
                    unsafe {
                        llama_cpp_sys::llama_sampler_chain_add(
                            res.chain.as_mut(),
                            llama_cpp_sys::llama_sampler_init_mirostat_v2(
                                res.params.seed,
                                res.params.mirostat_tau,
                                res.params.mirostat_eta,
                            ),
                        )
                    };
                }
                _ => panic!("unknown mirostat version"),
            }
        } else {
            unsafe {
                llama_cpp_sys::llama_sampler_chain_add(
                    res.chain.as_mut(),
                    llama_cpp_sys::llama_sampler_init_softmax(),
                )
            };
            unsafe {
                llama_cpp_sys::llama_sampler_chain_add(
                    res.chain.as_mut(),
                    llama_cpp_sys::llama_sampler_init_greedy(),
                )
            };
        }
        Ok(res)
    }

    pub fn accept(&mut self, token: LlamaToken, accept_grammar: bool) -> crate::Result<()> {
        if accept_grammar {
            unsafe { llama_cpp_sys::llama_sampler_accept(self.grmr.as_mut(), token.0) };
        }
        unsafe { llama_cpp_sys::llama_sampler_accept(self.chain.as_mut(), token.0) };
        self.prev.enqueue(token);
        Ok(())
    }

    pub fn reset(&mut self) -> crate::Result<()> {
        unsafe { llama_cpp_sys::llama_sampler_reset(self.grmr.as_mut()) };
        unsafe { llama_cpp_sys::llama_sampler_reset(self.chain.as_mut()) };
        Ok(())
    }

    fn set_logits(&mut self, ctx: &LlamaContext, index: i32) -> crate::Result<()> {
        let cur = (0_i32..)
            .zip(ctx.get_logits_ith(index))
            .map(|(i, logit)| {
                let token = LlamaToken::new(i);
                LlamaTokenData::new(token, *logit, 0_f32)
            })
            .collect();
        self.cur_p = LlamaTokenDataArray::new(cur, -1, false);
        Ok(())
    }

    pub fn sample(
        &mut self,
        ctx: &LlamaContext,
        index: i32,
        grammar_first: bool,
    ) -> crate::Result<LlamaToken> {
        self.set_logits(ctx, index)?;
        if grammar_first {
            unsafe {
                self.cur_p.modify_as_c_llama_token_data_array(|t| {
                    llama_cpp_sys::llama_sampler_apply(self.grmr.as_mut(), t)
                });
            };
        }
        unsafe {
            self.cur_p.modify_as_c_llama_token_data_array(|t| {
                llama_cpp_sys::llama_sampler_apply(self.chain.as_mut(), t)
            });
        };
        assert!(self.cur_p.selected != -1); // "no selected token during sampling - check your sampling configuration");
        let id = self.cur_p.data[self.cur_p.selected as usize].id();

        if grammar_first {
            return Ok(id);
        }

        {
            let single_token_data = LlamaTokenData::new(id, 1.0, 0.0);
            let mut single_token_data_array =
                LlamaTokenDataArray::new(vec![single_token_data], -1, false);
            unsafe {
                single_token_data_array.modify_as_c_llama_token_data_array(|t| {
                    llama_cpp_sys::llama_sampler_apply(self.grmr.as_mut(), t)
                });
            };
            let is_valid = single_token_data_array.data[0].logit() != -std::f32::INFINITY;
            if is_valid {
                return Ok(id);
            }
        }

        self.set_logits(ctx, index)?;
        unsafe {
            self.cur_p.modify_as_c_llama_token_data_array(|t| {
                llama_cpp_sys::llama_sampler_apply(self.grmr.as_mut(), t)
            });
        };
        unsafe {
            self.cur_p.modify_as_c_llama_token_data_array(|t| {
                llama_cpp_sys::llama_sampler_apply(self.grmr.as_mut(), t)
            });
        };
        assert!(self.cur_p.selected != -1); // "no selected token during sampling - check your sampling configuration");
        Ok(self.cur_p.data[self.cur_p.selected as usize].id())
    }

    pub fn get_candidates(&self) -> crate::Result<&LlamaTokenDataArray> {
        Ok(&self.cur_p)
    }

    pub fn last(&self) -> Option<&LlamaToken> {
        self.prev.back()
    }

    pub fn prev_str(&self, ctx_main: &LlamaContext, n: i32) -> crate::Result<String> {
        let n = std::cmp::min(n, self.prev.len() as i32);
        if n <= 0 {
            return Ok("".to_string());
        }

        let mut result = "".to_string();
        for i in &self.prev.to_vec()[self.prev.len() - n as usize..] {
            assert!(i != &LlamaToken::new(-1));
            result.push_str(&ctx_main.token_to_piece(i)?);
        }
        Ok(result)
    }
}
