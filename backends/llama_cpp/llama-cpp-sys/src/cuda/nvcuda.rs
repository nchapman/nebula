pub struct NvCudaHandle {}

impl NvCudaHandle {
    pub fn new() -> crate::Result<(usize, Self)> {
        Err(crate::Error::Unimplemented(file!(), line!()))
    }
}
