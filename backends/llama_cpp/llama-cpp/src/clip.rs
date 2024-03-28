use std::{ffi::CString, path::Path, ptr::NonNull};

use crate::ClipError;

pub struct ImageEmbed {
    pub(crate) embed: NonNull<llama_cpp_sys::llava_image_embed>,
}

impl Drop for ImageEmbed {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys::llava_image_embed_free(self.embed.as_ptr()) }
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct ClipContext {
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
        let clip = unsafe { llama_cpp_sys::clip_model_load(cstr.as_ptr(), 1) };

        let context = NonNull::new(clip).ok_or(ClipError::NullReturn)?;

        tracing::debug!(?path, "Loaded model");
        Ok(Self { context })
    }

    pub fn embed_image(&self, n_threads: usize, image: &[u8]) -> Result<ImageEmbed, ClipError> {
        let embed = unsafe {
            llama_cpp_sys::llava_image_embed_make_with_bytes(
                self.context.as_ptr(),
                n_threads as i32,
                image.as_ptr(),
                image.len() as i32,
            )
        };
        let embed = NonNull::new(embed).ok_or(ClipError::NullReturn)?;
        Ok(ImageEmbed { embed })
    }
}

impl Drop for ClipContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys::clip_free(self.context.as_ptr()) }
    }
}
