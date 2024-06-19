use std::ffi::c_int;

pub struct CudartHandle {
    handler: libloading::Library,
    path: std::path::PathBuf,
}

#[cfg(windows)]
const CUDART_GLOBS: &'static [&'static str] =
    &["c:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\bin\\cudart64_*.dll"];
#[cfg(unix)]
const CUDART_GLOBS: &'static [&'static str] = &[
    "/usr/local/cuda/lib64/libcudart.so*",
    "/usr/lib/x86_64-linux-gnu/nvidia/current/libcudart.so*",
    "/usr/lib/x86_64-linux-gnu/libcudart.so*",
    "/usr/lib/wsl/lib/libcudart.so*",
    "/usr/lib/wsl/drivers/*/libcudart.so*",
    "/opt/cuda/lib64/libcudart.so*",
    "/usr/local/cuda*/targets/aarch64-linux/lib/libcudart.so*",
    "/usr/lib/aarch64-linux-gnu/nvidia/current/libcudart.so*",
    "/usr/lib/aarch64-linux-gnu/libcudart.so*",
    "/usr/local/cuda/lib*/libcudart.so*",
    "/usr/lib*/libcudart.so*",
    "/usr/local/lib*/libcudart.so*",
];

#[cfg(windows)]
const CUDART_MGMT_NAME: &'static str = "cudart64_*.dll";
#[cfg(unix)]
const CUDART_MGMT_NAME: &'static str = "libcudart.so*";

impl CudartHandle {
    pub fn new() -> crate::Result<(usize, Self)> {
        let pp = super::find_libs(CUDART_MGMT_NAME, CUDART_GLOBS);
        for p in pp.iter() {
            if let Ok((dc, m)) = Self::load(p) {
                log::debug!("cudart loaded {}", p.display());
                return Ok((
                    dc,
                    Self {
                        handler: m,
                        path: p.clone(),
                    },
                ));
            }
        }
        Err(crate::Error::NvCudaLoad)
    }

    pub fn load(path: &std::path::PathBuf) -> crate::Result<(usize, libloading::Library)> {
        let lib = unsafe { libloading::Library::new(path.clone())? };
        if [
            "cudaSetDevice",
            "cudaDeviceSynchronize",
            "cudaDeviceReset",
            "cudaMemGetInfo",
            "cudaGetDeviceCount",
            "cudaDeviceGetAttribute",
            "cudaDriverGetVersion",
            "cudaGetDeviceProperties",
        ]
        .iter()
        .all(|s| unsafe {
            lib.get::<libloading::Symbol<unsafe extern "C" fn() -> c_int>>(s.as_bytes())
                .is_ok()
        }) {
            let cu_set_device: libloading::Symbol<unsafe extern "C" fn(c_int) -> c_int> =
                unsafe { lib.get(b"cudaSetDevice").unwrap() };
            let res = unsafe { cu_set_device(0) };
            if res != 0 {
                return Err(crate::Error::CudartCall("cudaSetDevice", res));
            }
            let cuda_driver_get_version: libloading::Symbol<
                unsafe extern "C" fn(*mut c_int) -> c_int,
            > = unsafe { lib.get(b"cudaDriverGetVersion").unwrap() };
            let mut version = 0;
            let res = unsafe { cuda_driver_get_version(&mut version) };
            if res != 0 {
                return Err(crate::Error::CudartCall("cudaDriverGetVersion", res));
            }
            log::info!(
                "CUDA driver version: {}.{}",
                version / 1000,
                (version - version / 1000 * 1000) / 10,
            );
            let cuda_get_device_count: libloading::Symbol<
                unsafe extern "C" fn(*mut c_int) -> c_int,
            > = unsafe { lib.get(b"cudaGetDeviceCount").unwrap() };
            let mut device_count = 0;
            let res = unsafe { cuda_get_device_count(&mut device_count) };
            if res != 0 {
                return Err(crate::Error::CudartCall("cudaGetDeviceCount", res));
            }
            Ok((device_count as usize, lib))
        } else {
            Err(crate::Error::NvCudaLoad)
        }
    }
}
