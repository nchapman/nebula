use crate::ClipError;
use std::{ffi::CString, path::Path, ptr::NonNull, sync::Arc};

pub struct ImageEmbed {
    pub(crate) embed: NonNull<llama_cpp_sys::llava_image_embed>,
}

impl ImageEmbed {
    pub fn len(&self) -> usize {
        unsafe { self.embed.as_ref().n_image_pos as usize }
    }
}

impl Drop for ImageEmbed {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys::llava_image_embed_free(self.embed.as_ptr()) }
    }
}

#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct ClipContextInternal {
    pub(crate) context: NonNull<llama_cpp_sys::clip_ctx>,
}
unsafe impl Send for ClipContextInternal {}
unsafe impl Sync for ClipContextInternal {}

#[derive(Clone, Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct ClipContext {
    pub(crate) context: Arc<ClipContextInternal>,
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
        let guard = stdio_override::StderrOverride::from_file("/dev/null").unwrap();
        #[cfg(target_os = "windows")]
        let guard = stdio_override::StderrOverride::from_file("nul").unwrap();
        #[cfg(debug_assertions)]
        let clip = unsafe { llama_cpp_sys::clip_model_load(cstr.as_ptr(), 0) };
        #[cfg(not(debug_assertions))]
        let clip = unsafe { llama_cpp_sys::clip_model_load(cstr.as_ptr(), 0) };
        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
        drop(guard);
        let context = NonNull::new(clip).ok_or(ClipError::NullReturn)?;

        tracing::debug!(?path, "Loaded model");
        Ok(Self {
            context: Arc::new(ClipContextInternal { context }),
        })
    }

    pub fn embed_image(&self, n_threads: usize, image: &[u8]) -> Result<ImageEmbed, ClipError> {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        let guard = stdio_override::StderrOverride::from_file("/dev/null").unwrap();
        #[cfg(target_os = "windows")]
        let guard = stdio_override::StdoutOverride::from_file("nul").unwrap();
        let embed = unsafe {
            llama_cpp_sys::llava_image_embed_make_with_bytes(
                self.context.context.as_ptr(),
                n_threads as i32,
                image.as_ptr(),
                image.len() as i32,
            )
        };
        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
        drop(guard);
        let embed = NonNull::new(embed).ok_or(ClipError::NullReturn)?;
        Ok(ImageEmbed { embed })
    }
}

impl Drop for ClipContextInternal {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys::clip_free(self.context.as_ptr()) }
    }
}
