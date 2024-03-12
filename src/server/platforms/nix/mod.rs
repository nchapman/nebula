use crate::{server::Server, Result};
use signal_hook::{
    consts::{SIGINT, SIGTERM},
    iterator::Signals,
};

pub fn start_server() -> Result<()> {
    let server = std::sync::Arc::new(Server::new()?);
    let sserver = server.clone();
    let mut signals = Signals::new(&[SIGINT, SIGTERM])?;
    std::thread::spawn(move || {
        for _ in signals.forever() {
            sserver.stop().unwrap();
        }
    });
    server.run()?;
    Ok(())
}
