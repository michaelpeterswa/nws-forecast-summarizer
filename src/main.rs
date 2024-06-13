use axum::{routing::get, Router};
use tracing::info;

mod config;
mod log;
mod metrics;
mod routes;

#[tokio::main]
async fn main() {
    // load config
    let app_config = config::load();

    // init log
    log::init(app_config.log_level);

    info!("welcome to rust-start!");

    let app = Router::new()
        .route("/", get(routes::root))
        .route("/count", get(routes::count));

    tokio::spawn(async move {
        metrics::start_metrics_server(app_config.metrics_host, app_config.metrics_port).await;
    });

    let listener =
        tokio::net::TcpListener::bind(format!("{}:{}", app_config.api_host, app_config.api_port))
            .await
            .unwrap();
    axum::serve(listener, app).await.unwrap();
}
