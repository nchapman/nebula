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

pub struct AutomaticSpeechRecognitionOptions<'a> {
    pub n_threads: i32,
    pub translate: bool,
    pub language: &'a str,
    pub print_special: bool,
    pub print_progress: bool,
    pub print_realtime: bool,
    pub print_timestamps: bool,
}

impl<'a> AutomaticSpeechRecognitionOptions<'a> {
    pub fn with_n_threads(mut self, n_threads: i32) -> Self {
        self.n_threads = n_threads;
        self
    }

    pub fn with_translate(mut self, translate: bool) -> Self {
        self.translate = translate;
        self
    }

    pub fn with_language(mut self, language: &'a str) -> Self {
        self.language = language;
        self
    }

    pub fn with_print_special(mut self, print_special: bool) -> Self {
        self.print_special = print_special;
        self
    }

    pub fn with_print_progress(mut self, print_progress: bool) -> Self {
        self.print_progress = print_progress;
        self
    }

    pub fn with_print_realtime(mut self, print_realtime: bool) -> Self {
        self.print_realtime = print_realtime;
        self
    }

    pub fn with_print_timestamps(mut self, print_timestamps: bool) -> Self {
        self.print_timestamps = print_timestamps;
        self
    }
}

impl<'a> Default for AutomaticSpeechRecognitionOptions<'a> {
    fn default() -> Self {
        Self {
            n_threads: 1,
            translate: true,
            language: "en",
            print_special: false,
            print_progress: false,
            print_realtime: false,
            print_timestamps: false,
        }
    }
}
