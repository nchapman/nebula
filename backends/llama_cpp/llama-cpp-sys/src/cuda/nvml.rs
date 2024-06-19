use std::ffi::{c_int, c_ulonglong, c_void};

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Memory {
    total: c_ulonglong,
    free: c_ulonglong,
    used: c_ulonglong,
}

pub struct NvMlHandle {
    handler: libloading::Library,
    path: std::path::PathBuf,
}

const NVML_GLOBS: &'static [&'static str] = &["c:\\Windows\\System32\\nvml.dll"];
const NVML_MGMT_NAME: &'static str = "nvml.dll";

impl NvMlHandle {
    pub fn new() -> crate::Result<Self> {
        let pp = super::find_libs(NVML_MGMT_NAME, NVML_GLOBS);
        for p in pp.iter() {
            if let Ok(m) = Self::load(p) {
                log::debug!("nvidia-ml loaded {}", p.display());
                return Ok(Self {
                    handler: m,
                    path: p.clone(),
                });
            }
        }
        Err(crate::Error::NvMlLoad)
    }

    pub fn load(path: &std::path::PathBuf) -> crate::Result<libloading::Library> {
        let lib = unsafe { libloading::Library::new(path.clone())? };
        let nvml_init_v2: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
            unsafe { lib.get(b"nvmlInit_v2")? };
        let _: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
            unsafe { lib.get(b"nvmlShutdown")? };
        let _: libloading::Symbol<unsafe extern "C" fn(c_int, *mut c_void) -> c_int> =
            unsafe { lib.get(b"nvmlDeviceGetHandleByIndex")? };
        let _: libloading::Symbol<unsafe extern "C" fn(*const c_void, *mut Memory) -> c_int> =
            unsafe { lib.get(b"nvmlDeviceGetMemoryInfo")? };
        let res = unsafe { nvml_init_v2() };
        if res == 0 {
            Ok(lib)
        } else {
            Err(crate::Error::NvMlInit_v2(res))
        }
    }
}
