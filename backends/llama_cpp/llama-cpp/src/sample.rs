use std::{ffi::CString, path::Path, ptr::NonNull};

use crate::ClipError;

pub struct SamplingParams {
    pub(crate) params: NonNull<llama_cpp_sys::llama_sampling_params>,
}

#[allow(clippy::module_name_repetitions)]
pub struct SampleContext {
    pub(crate) context: NonNull<llama_cpp_sys::llama_sampling_context>,
}

impl SampleContext {
    pub fn init(params: SamplingParams) -> Result<Self, SampleError> {
        let clip = unsafe { llama_cpp_sys::clip_model_load(SamplingParams.params.as_ptr()) };
        let context = NonNull::new(clip).ok_or(ClipError::NullReturn)?;
        Ok(Self { context })
    }
}

impl Drop for SampleContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys::llama_sampling_free(self.context.as_ptr()) }
    }
}
