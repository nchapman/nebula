#[path = "platforms/nix/mod.rs"]
pub mod platform;
pub mod config;
pub mod httplistener;
use config::Settings;
use crate::Result;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use clap::Parser;

#[derive(Parser)]
#[clap(version, author, about)]
struct CliOpts {
    #[clap(short = 'c', long)]
    config: String,
}

fn parse_settings_from_cli() -> Result<Settings> {
    let cli_opts = CliOpts::parse();
    let cfg_file = &cli_opts.config;
    Settings::from_file(cfg_file)
}

pub struct Server {
    settings: config::Settings,
    stopped: Arc<AtomicBool>
}

impl Server {
    pub fn new() -> Result<Self> {
        let settings = parse_settings_from_cli()?;
        let ll = settings.logging.level;
        Ok(Self {
            settings,
            stopped: Arc::new(AtomicBool::new(false))
        })
    }

    pub fn run(&self) -> Result<()> {
        let listeners = self
            .settings
            .listeners
            .iter()
            .try_fold(vec![], |mut r, listener| {
                match listener.r#type.as_str(){
                    "http" => {
                        r.push(httplistener::HttpListener::new(serde_yaml::from_value::<httplistener::HttpListenerSettings>(listener.extra.clone())?));
                        Ok(r)
                    }
                    _ => Err(crate::error::Error::ListenerType(listener.r#type.clone()))
                }
            });
        eprintln!("{:?}", listeners);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let _ = rt.block_on(async move{
            //let threads = vec![];
            for listener in listeners{
//                threads.push(tokio::spawn(async move{
//                    listener.run().await;
//                }));
            }
        });
        Ok(())
    }

    pub fn stop(&self) -> Result<()> {
        self.stopped.store(true, Ordering::Relaxed);
        Ok(())
    }
}
