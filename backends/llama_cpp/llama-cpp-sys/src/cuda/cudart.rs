use std::ffi::{c_int, c_uchar, c_uint};

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

#[repr(C)]
#[derive(Default, Debug)]
struct CudaUUID {
    bytes: [c_uchar; 16],
}

#[repr(C)]
#[derive(Debug)]
struct CudaDeviceProp {
    name: [c_uchar; 256],
    uuid: CudaUUID,
    luid: [c_uchar; 8],
    luid_device_node_mask: c_uint,
    total_global_mem: usize,
    shared_mem_per_block: usize,
    regs_per_block: c_int,
    warp_size: c_int,
    mem_pitch: usize,
    max_threads_per_block: c_int,
    max_threads_dim: [c_int; 3],
    max_grid_size: [c_int; 3],
    clockRate: c_int,
    total_const_mem: usize,
    major: c_int,
    minor: c_int,
    texture_alignment: usize,
    texture_pitch_alignment: usize,
    device_overlap: c_int,
    multi_processor_count: c_int,
    kernel_exec_timeout_enabled: c_int,
    integrated: c_int,
    can_map_host_memory: c_int,
    compute_mode: c_int,
    max_texture_id: c_int,
    max_texture1d_mipmap: c_int,
    max_texture1d_linear: c_int,
    max_texture2d: [c_int; 2],
    max_texture2d_mipmap: [c_int; 2],
    max_texture2d_linear: [c_int; 3],
    max_texture2d_gather: [c_int; 2],
    max_texture3d: [c_int; 3],
    max_texture3d_alt: [c_int; 3],
    max_texture_cubemap: c_int,
    max_texture1d_layered: [c_int; 2],
    max_texture_2d_layered: [c_int; 3],
    max_texture_cubemap_layered: [c_int; 2],
    max_surface1d: c_int,
    max_surface2d: [c_int; 2],
    max_surface3d: [c_int; 3],
    max_surface1d_layered: [c_int; 2],
    max_surface2d_layered: [c_int; 3],
    max_surface_cubemap: c_int,
    max_surface_cubemap_layered: [c_int; 2],
    surface_alignment: usize,
    concurrent_kernels: c_int,
    ecc_enabled: c_int,
    pci_bus_id: c_int,
    pci_device_id: c_int,
    pci_domain_id: c_int,
    tcc_driver: c_int,
    async_engine_count: c_int,
    unified_addressing: c_int,
    memory_clock_rate: c_int,
    memory_bus_width: c_int,
    l2_cache_size: c_int,
    persisting_l2_cache_max_size: c_int,
    max_threads_per_multi_processor: c_int,
    stream_priorities_supported: c_int,
    global_l1_cache_supported: c_int,
    local_l1_cache_cupported: c_int,
    shared_mem_per_multiprocessor: usize,
    regs_per_multiprocessor: c_int,
    managed_memory: c_int,
    is_multi_gpu_board: c_int,
    multi_gpu_board_group_id: c_int,
    host_native_atomic_supported: c_int,
    single_to_double_precision_perf_ratio: c_int,
    pageable_memory_access: c_int,
    concurrent_managed_access: c_int,
    compute_preemption_supported: c_int,
    can_use_host_pointer_for_registered_mem: c_int,
    cooperative_launch: c_int,
    cooperative_multi_device_launch: c_int,
    shared_mem_per_block_optin: c_int,
    pageable_memory_access_uses_host_page_tables: c_int,
    direct_managed_mem_access_from_host: c_int,
    max_blocks_per_multi_processor: c_int,
    access_policy_max_window_size: c_int,
    reserved_shared_mem_per_block: usize,
}

impl Default for CudaDeviceProp {
    fn default() -> Self {
        Self {
            name: [0; 256],
            uuid: CudaUUID::default(),
            luid: [0; 8],
            luid_device_node_mask: 0,
            total_global_mem: 0,
            shared_mem_per_block: 0,
            regs_per_block: 0,
            warp_size: 0,
            mem_pitch: 0,
            max_threads_per_block: 0,
            max_threads_dim: [0; 3],
            max_grid_size: [0; 3],
            clockRate: 0,
            total_const_mem: 0,
            major: 0,
            minor: 0,
            texture_alignment: 0,
            texture_pitch_alignment: 0,
            device_overlap: 0,
            multi_processor_count: 0,
            kernel_exec_timeout_enabled: 0,
            integrated: 0,
            can_map_host_memory: 0,
            compute_mode: 0,
            max_texture_id: 0,
            max_texture1d_mipmap: 0,
            max_texture1d_linear: 0,
            max_texture2d: [0; 2],
            max_texture2d_mipmap: [0; 2],
            max_texture2d_linear: [0; 3],
            max_texture2d_gather: [0; 2],
            max_texture3d: [0; 3],
            max_texture3d_alt: [0; 3],
            max_texture_cubemap: 0,
            max_texture1d_layered: [0; 2],
            max_texture_2d_layered: [0; 3],
            max_texture_cubemap_layered: [0; 2],
            max_surface1d: 0,
            max_surface2d: [0; 2],
            max_surface3d: [0; 3],
            max_surface1d_layered: [0; 2],
            max_surface2d_layered: [0; 3],
            max_surface_cubemap: 0,
            max_surface_cubemap_layered: [0; 2],
            surface_alignment: 0,
            concurrent_kernels: 0,
            ecc_enabled: 0,
            pci_bus_id: 0,
            pci_device_id: 0,
            pci_domain_id: 0,
            tcc_driver: 0,
            async_engine_count: 0,
            unified_addressing: 0,
            memory_clock_rate: 0,
            memory_bus_width: 0,
            l2_cache_size: 0,
            persisting_l2_cache_max_size: 0,
            max_threads_per_multi_processor: 0,
            stream_priorities_supported: 0,
            global_l1_cache_supported: 0,
            local_l1_cache_cupported: 0,
            shared_mem_per_multiprocessor: 0,
            regs_per_multiprocessor: 0,
            managed_memory: 0,
            is_multi_gpu_board: 0,
            multi_gpu_board_group_id: 0,
            host_native_atomic_supported: 0,
            single_to_double_precision_perf_ratio: 0,
            pageable_memory_access: 0,
            concurrent_managed_access: 0,
            compute_preemption_supported: 0,
            can_use_host_pointer_for_registered_mem: 0,
            cooperative_launch: 0,
            cooperative_multi_device_launch: 0,
            shared_mem_per_block_optin: 0,
            pageable_memory_access_uses_host_page_tables: 0,
            direct_managed_mem_access_from_host: 0,
            max_blocks_per_multi_processor: 0,
            access_policy_max_window_size: 0,
            reserved_shared_mem_per_block: 0,
        }
    }
}

impl Into<crate::DeviceInfo> for CudaDeviceProp {
    fn into(self) -> crate::DeviceInfo {
        crate::DeviceInfo {
            memInfo: crate::MemInfo::default(),
            library: "cuda",
            variant: crate::CPUCapability::None,
            minimum_memory: 457*1024*1024,
            #[cfg(windows)]
            dependency_paths: vec![PathBuf::from(format!("{}/dist/windows/{}", crate::TMP_DIR, std::env::consts::ARCH))],
            #[cfg(not(windows))]
            dependency_paths: vec![],
            env_workarounds: vec![],
            id: format!("GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                        self.uuid.bytes[0],
                        self.uuid.bytes[1],
                        self.uuid.bytes[2],
                        self.uuid.bytes[3],
                        self.uuid.bytes[4],
                        self.uuid.bytes[5],
                        self.uuid.bytes[6],
                        self.uuid.bytes[7],
                        self.uuid.bytes[8],
                        self.uuid.bytes[9],
                        self.uuid.bytes[10],
                        self.uuid.bytes[11],
                        self.uuid.bytes[12],
                        self.uuid.bytes[13],
                        self.uuid.bytes[14],
                        self.uuid.bytes[15],
            ),
            name: String::from_utf8_lossy(&self.name[..])
                .trim_end_matches('\0')
                .to_string(),
            compute: format!("{}.{}", self.major, self.minor),
            driver_version: crate::DriverVersion::default(),
        }
    }
}

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

    pub fn set_device(&self, device: usize) -> crate::Result<()> {
        let cu_set_device: libloading::Symbol<unsafe extern "C" fn(c_int) -> c_int> =
            unsafe { self.handler.get(b"cudaSetDevice").unwrap() };
        let res = unsafe { cu_set_device(device as i32) };
        if res != 0 {
            Err(crate::Error::CudartCall("cudaSetDevice", res))
        } else {
            Ok(())
        }
    }

    pub fn get_device_properties(&self, device: usize) -> crate::Result<CudaDeviceProp> {
        let mut props = CudaDeviceProp::default();
        let get_device_props: libloading::Symbol<
            unsafe extern "C" fn(*mut CudaDeviceProp, c_int) -> c_int,
        > = unsafe { self.handler.get(b"cudaGetDeviceProperties").unwrap() };
        let res = unsafe { get_device_props(&mut props, device as i32) };
        if res != 0 {
            Err(crate::Error::CudartCall("cudaGetDeviceProperties)", res))
        } else {
            Ok(props)
        }
    }

    pub fn get_mem_info(&self) -> crate::Result<(usize, usize)> {
        let mut mem_free = 0;
        let mut mem_total = 0;
        let get_mem_info: libloading::Symbol<
            unsafe extern "C" fn(*mut usize, *mut usize) -> c_int,
        > = unsafe { self.handler.get(b"cudaMemGetInfo").unwrap() };
        let res = unsafe { get_mem_info(&mut mem_free, &mut mem_total) };
        if res != 0 {
            Err(crate::Error::CudartCall("cudaGetDeviceProperties)", res))
        } else {
            Ok((mem_free, mem_total))
        }
    }

    pub fn bootstrap(&self, device: usize) -> crate::Result<crate::DeviceInfo> {
        self.set_device(device)?;
        let props = self.get_device_properties(device)?;
        let mut props: crate::DeviceInfo = props.into();
        let (mem_free, mem_total) = self.get_mem_info()?;
        props.memInfo.free = mem_free as u64;
        props.memInfo.total = mem_total as u64;
        Ok(props)
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
