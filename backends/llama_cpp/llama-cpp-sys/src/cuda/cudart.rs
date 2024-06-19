pub struct CudartHandle {}

impl CudartHandle {
    pub fn new() -> crate::Result<(usize, Self)> {
        Err(crate::Error::Unimplemented(file!(), line!()))
    }
}
