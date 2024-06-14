use std::sync::Arc;

use axum::{routing::get, Router};
use tracing::info;

mod config;
mod log;
mod metrics;
mod ollama;
mod routes;

#[tokio::main]
async fn main() {
    // load config
    let app_config = config::load();

    // init log
    log::init(app_config.log_level);

    // connect to ollama
    let ollama_client = ollama::connect(app_config.ollama_host, app_config.ollama_port);

    let forecast_state = Arc::new(routes::ForecastState {
        client: reqwest::Client::new(),
        ollama_connection: ollama_client,
        ollama_model: app_config.ollama_model,
    });

    info!("welcome to rust-start!");

    let app = Router::new().route("/", get(routes::root)).route(
        "/api/v1/forecast",
        get(routes::forecast).with_state(forecast_state),
    );

    tokio::spawn(async move {
        metrics::start_metrics_server(app_config.metrics_host, app_config.metrics_port).await;
    });

    let listener =
        tokio::net::TcpListener::bind(format!("{}:{}", app_config.api_host, app_config.api_port))
            .await
            .unwrap();
    axum::serve(listener, app).await.unwrap();
}
