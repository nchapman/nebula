use std::ffi::c_int;

pub struct NvCudaHandle {
    handler: libloading::Library,
    path: std::path::PathBuf,
}

#[cfg(windows)]
const NVCUDA_GLOBS: &'static [&'static str] = &["c:\\windows\\system*\\nvcuda.dll"];
#[cfg(unix)]
const NVCUDA_GLOBS: &'static [&'static str] = &[
    "/usr/local/cuda*/targets/*/lib/libcuda.so*",
    "/usr/lib/*-linux-gnu/nvidia/current/libcuda.so*",
    "/usr/lib/*-linux-gnu/libcuda.so*",
    "/usr/lib/wsl/lib/libcuda.so*",
    "/usr/lib/wsl/drivers/*/libcuda.so*",
    "/opt/cuda/lib*/libcuda.so*",
    "/usr/local/cuda/lib*/libcuda.so*",
    "/usr/lib*/libcuda.so*",
    "/usr/local/lib*/libcuda.so*",
];

#[cfg(windows)]
const NVCUDA_MGMT_NAME: &'static str = "nvcuda.dll";
#[cfg(unix)]
const NVCUDA_MGMT_NAME: &'static str = "libcuda.so*";

impl NvCudaHandle {
    pub fn new() -> crate::Result<(usize, Self)> {
        let pp = super::find_libs(NVCUDA_MGMT_NAME, NVCUDA_GLOBS);
        for p in pp.iter() {
            if let Ok((dc, m)) = Self::load(p) {
                log::debug!("nvidia-cuda loaded {}", p.display());
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
            "cuInit",
            "cuDriverGetVersion",
            "cuDeviceGetCount",
            "cuDeviceGet",
            "cuDeviceGetAttribute",
            "cuDeviceGetUuid",
            "cuDeviceGetName",
            "cuCtxCreate_v3",
            "cuMemGetInfo_v2",
            "cuCtxDestroy",
        ]
        .iter()
        .all(|s| unsafe {
            lib.get::<libloading::Symbol<unsafe extern "C" fn() -> c_int>>(s.as_bytes())
                .is_ok()
        }) {
            let cu_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> c_int> =
                unsafe { lib.get(b"cuInit").unwrap() };
            let res = unsafe { cu_init(0) };
            if res != 0 {
                return Err(crate::Error::NvCudaCall("cuInit", res));
            }
            let cu_driver_get_version: libloading::Symbol<
                unsafe extern "C" fn(*mut c_int) -> c_int,
            > = unsafe { lib.get(b"cuDriverGetVersion").unwrap() };
            let mut version = 0;
            let res = unsafe { cu_driver_get_version(&mut version) };
            if res != 0 {
                return Err(crate::Error::NvCudaCall("cuDriverGetVersion", res));
            }
            log::info!(
                "CUDA driver version: {}.{}",
                version / 1000,
                (version - version / 1000 * 1000) / 10,
            );
            let cu_device_get_count: libloading::Symbol<unsafe extern "C" fn(*mut c_int) -> c_int> =
                unsafe { lib.get(b"cuDriverGetVersion").unwrap() };
            let mut device_count = 0;
            let res = unsafe { cu_device_get_count(&mut device_count) };
            if res != 0 {
                return Err(crate::Error::NvCudaCall("cuDeviceGetCount", res));
            }
            Ok((device_count as usize, lib))
        } else {
            Err(crate::Error::NvCudaLoad)
        }
    }
}
