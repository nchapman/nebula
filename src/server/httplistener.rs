use crate::Result;
use actix_web::{web, App, HttpServer, Responder};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    }};
use openssl::ssl::{SslAcceptor, SslFiletype, SslMethod};

#[derive(Clone)]
struct State {
}

#[derive(Debug, serde::Deserialize)]
struct Request {
    model: String,
    prompt: String,
}

#[derive(Debug, serde::Serialize)]
struct Response {
    text: String,
}

async fn index(data: web::Json<Request>) -> Result<impl Responder> {
    log::trace!("{:?}", data);
    Ok(web::Json(Response{text: "".to_string()}))
}

#[derive(Debug, serde::Deserialize)]
pub struct HttpListenerSettings{
    port: u16,
    tls_certs: Option<String>,
    tls_key: Option<String>
}

#[derive(Debug)]
pub struct HttpListener{
    settings: HttpListenerSettings,
    stopped: Arc<AtomicBool>
}

impl HttpListener {
    pub fn new(
        settings: HttpListenerSettings
    ) -> Self {
        Self{
            settings,
            stopped: Arc::new(AtomicBool::new(false))
        }
    }

    pub async fn run(&self) -> Result<()>{
        let http_server = if let (Some(c), Some(k)) = (&self.settings.tls_certs, &self.settings.tls_key) {
            let mut builder = SslAcceptor::mozilla_intermediate(SslMethod::tls())?;
            builder
                .set_private_key_file(k, SslFiletype::PEM)?;
            builder.set_certificate_chain_file(c)?;
            HttpServer::new(move || App::new()
                            .app_data(web::Data::new(State{
                            }))
                            .route("/ingest1", web::post().to(index))
            )
                .bind_openssl(&format!("0.0.0.0:{}", self.settings.port), builder)?
                .run()
        } else {
            HttpServer::new(move || App::new()
                            .app_data(web::Data::new(State{
                            }))
                            .route("/ingest1", web::post().to(index))
            )
                .keep_alive(std::time::Duration::from_secs(75))
                .bind(&format!("0.0.0.0:{}", self.settings.port))?
                .run()
        };
        tokio::spawn(http_server);
        Ok(())
    }

    pub async fn stop(&self) -> Result<()>{
        self.stopped.store(true, Ordering::Relaxed);
        Ok(())
    }
}
