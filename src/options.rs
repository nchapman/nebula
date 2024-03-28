pub struct ModelOptions {
    pub cpu: bool,
    pub n_gpu_layers: i32,
}

impl ModelOptions {
    pub fn use_cpu(mut self) -> Self {
        self.cpu = true;
        self
    }

    pub fn with_n_gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            cpu: false,
            n_gpu_layers: -1,
        }
    }
}

pub struct PredictOptions {
    pub seed: u32,
    pub n_ctx: usize,
    pub n_len: usize,
    pub n_threads: usize,
}

impl PredictOptions {
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_n_ctx(mut self, n_ctx: usize) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    pub fn with_n_len(mut self, n_len: usize) -> Self {
        self.n_len = n_len;
        self
    }
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            n_ctx: 2048,
            n_len: 150,
            n_threads: 10,
        }
    }
}

pub struct NebulaOptions {
    seed: i32, // RNG seed
    n_threads: u32,
    n_threads_draft: i32,
    n_threads_batch: i32,
    n_threads_batch_draft: i32,
    n_predict: i32,
    n_ctx: i32,
    n_batch: i32,
    n_ubatch: i32,
    n_keep: i32,
    n_draft: i32,
    n_chunks: i32,
    n_parallel: i32,
    n_sequences: i32,
    p_split: f32,
    n_gpu_layers: i32,
    n_gpu_layers_draft: i32,
    //    split_mode: SplitMode,
    main_gpu: i32,
    tensor_split: [f32; 128],
    n_beams: i32,
    grp_attn_n: i32,
    grp_attn_w: i32,
    n_print: i32,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: i32,
    defrag_thold: f32,
    //    numa: NumaStrategy,
    //    rope_scaling_type: RopeScalingType,
    //    pooling_type: PoolingType,
    //  sparams: SamplingParams;
}
