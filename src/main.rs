use nebula::{server::platform, Result};

fn main() -> Result<()> {
    match platform::start_server() {
        Ok(_server) => Ok(()),
        Err(e) => {
            log::error!("{:?}", e);
            Err(e)
        }
    }
}
