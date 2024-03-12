use serde::{Deserialize, Deserializer};
use std::collections::HashMap;

fn deserialize_loglevel<'de, D>(deserializer: D) -> std::result::Result<log::Level, D::Error>
where
    D: Deserializer<'de>,
{
    let ll = String::deserialize(deserializer)?;
    match ll.as_str() {
        "error" => Ok(log::Level::Error),
        "warn" => Ok(log::Level::Warn),
        "info" => Ok(log::Level::Info),
        "debug" => Ok(log::Level::Debug),
        "trace" => Ok(log::Level::Trace),
        _ => Err(serde::de::Error::custom(format!(
            "undefined loglevel: {}",
            ll
        ))),
    }
}

fn default_loglevel() -> log::Level {
    log::Level::Info
}

#[derive(Deserialize, Clone, Debug)]
pub struct Settings {
    #[serde(default)]
    pub logging: LoggingSettings,
    pub listeners: Vec<SpecificSettings>,
    pub models: Vec<SpecificSettings>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct LoggingSettings {
    #[serde(
        deserialize_with = "deserialize_loglevel",
        default = "default_loglevel"
    )]
    pub level: log::Level,
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            level: log::Level::Info,
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct SpecificSettings {
    pub r#type: String,
    #[serde(flatten)]
    pub extra: serde_yaml::Value,
}

impl Settings {
    pub fn from_file(path: &str) -> crate::Result<Self> {
        let conf_str = std::fs::read_to_string(path)?;
        Self::from_str_buffer(&conf_str)
    }

    pub fn from_str_buffer(s: &str) -> crate::Result<Self> {
        Ok(serde_yaml::from_str(s)?)
    }
}
