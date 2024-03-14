pub struct ModelOptions {
    pub cpu: bool,
    pub n_gpu_layers: usize,
}

impl ModelOptions {
    pub fn use_cpu(mut self) -> Self {
        self.cpu = true;
        self
    }

    pub fn with_n_gpu_layers(mut self, n_gpu_layers: usize) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            cpu: false,
            n_gpu_layers: 0,
        }
    }
}

pub struct PredictOptions {
    pub seed: u32,
    pub n_ctx: usize,
    pub n_len: usize,
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
        }
    }
}
