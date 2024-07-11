use std::{ffi::CString, path::Path, ptr::NonNull};

use crate::LlavaError;

#[allow(clippy::module_name_repetitions)]
pub struct LlavaContext {
    pub(crate) context: NonNull<llama_cpp_sys::clip_ctx>,
}

impl ClipContext {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ClipError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");
        let path = path
            .to_str()
            .ok_or(ClipError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        let guard = stdio_override::StderrOverride::override_file("/dev/null").unwrap();
        let clip = unsafe { llama_cpp_sys::clip_model_load(cstr.as_ptr(), 1) };
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        drop(guard);
        let context = NonNull::new(clip).ok_or(ClipError::NullReturn)?;

        tracing::debug!(?path, "Loaded model");
        Ok(Self { context })
    }
}

impl Drop for ClipContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys::clip_free(self.context.as_ptr()) }
    }
}
