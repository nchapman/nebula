#[cfg(windows)]
mod windows {
    #[repr(C)]
    #[derive(Default)]
    pub struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    impl Into<crate::MemInfo> for MemoryStatusEx {
        fn into(self) -> crate::MemInfo {
            crate::MemInfo {
                total: self.total_phys,
                free: self.avail_phys,
            }
        }
    }
}

#[cfg(windows)]
pub fn get_mem() -> crate::Result<crate::MemInfo> {
    use std::ffi::c_int;
    let mut mem_status = windows::MemoryStatusEx::default();
    let lib = unsafe { libloading::Library::new("kernel32.dll")? };
    let global_memory_status_ex: libloading::Symbol<
        unsafe extern "C" fn(*mut MemoryStatusEx) -> c_int,
    > = unsafe { lib.get(b"GlobalMemoryStatusEx")? };
    let res = unsafe { global_memory_status_ex(&mut mem_status) };
    if res == 0 {
        return Err(crate::Error::SystemCall("GlobalMemoryStatusEx", 1));
    } else {
        Ok(mem_status.into())
    }
}

#[cfg(macos)]
pub fn get_mem() -> crate::Result<crate::MemInfo> {
    let fm = objc2_foundation::NSProcessInfo::procsessInfo().physicalMemory();
    Ok(crate::MemInfo { total: fm, free: 0 })
}

#[cfg(unix)]
pub fn get_mem() -> crate::Result<crate::MemInfo> {
    use procfs::Current;
    let meminfo = procfs::Meminfo::current()?;
    if let (mt, Some(ma)) = (meminfo.mem_total, meminfo.mem_available) {
        if mt > 0 && ma > 0 {
            return Ok(crate::MemInfo {
                total: mt,
                free: ma,
            });
        }
    }
    Ok(crate::MemInfo {
        total: meminfo.mem_total,
        free: meminfo.mem_free + meminfo.buffers + meminfo.cached,
    })
}
