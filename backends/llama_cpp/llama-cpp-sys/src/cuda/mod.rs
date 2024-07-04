pub mod cudart;
pub mod nvcuda;
pub mod nvml;

pub fn find_libs(name: &str, patterns: &[&str]) -> Vec<std::path::PathBuf> {
    log::debug!("Searching for GPU library {}", name);
    #[cfg(target_os = "windows")]
    return std::env::var("PATH")
        .unwrap_or_default()
        .split(";")
        .map(|p| {
            let pp = p.to_string() + name + "*";
            pp
        })
        .chain(patterns.iter().map(|s| s.to_string()))
        .filter(|p| !p.contains("PhysX"))
        .filter_map(|p| glob::glob(&p).ok())
        .flatten()
        .filter_map(|p| p.ok())
        .collect();
    #[cfg(target_os = "linux")]
    return std::env::var("LD_LIBRARY_PATH")
        .unwrap_or_default()
        .split(":")
        .map(|p| {
            let pp = p.to_string() + name + "*";
            pp
        })
        .chain(patterns.iter().map(|s| s.to_string()))
        .filter(|p| !p.contains("PhysX"))
        .filter_map(|p| glob::glob(&p).ok())
        .flatten()
        .filter_map(|p| p.ok())
        .collect();
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    return patterns
        .iter()
        .map(|s| s.to_string())
        .filter(|p| !p.contains("PhysX"))
        .filter_map(|p| glob::glob(&p).ok())
        .flatten()
        .filter_map(|p| p.ok())
        .collect();
}
