lazy_static::lazy_static! {

    static ref RESOURCE_PATH: std::sync::Arc<std::sync::RwLock<Option<std::path::PathBuf>>> = std::sync::Arc::new(std::sync::RwLock::new(None));
}

pub fn get() -> Result<std::path::PathBuf, String> {
    let p = RESOURCE_PATH.read().map_err(|s| s.to_string())?;
    match &*p {
        Some(p) => Ok(p.clone()),
        None => Err("resources-path should be initialized".into()),
    }
}

pub fn set(path: std::path::PathBuf) -> Result<(), String> {
    let mut p = RESOURCE_PATH.write().map_err(|s| s.to_string())?;
    *p = Some(path);
    Ok(())
}
