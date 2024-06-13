use axum::{routing::get, Router};
use prometheus::{Encoder, TextEncoder};

pub async fn start_metrics_server(host: String, port: u16) {
    let app = Router::new()
        .route("/metrics", get(metrics))
        .route("/healthcheck", get(healthcheck));

    let listener = tokio::net::TcpListener::bind(format!("{}:{}", host, port))
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn metrics() -> String {
    let mut buffer = vec![];
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    encoder.encode(&metric_families, &mut buffer).unwrap();

    String::from_utf8(buffer).unwrap()
}

async fn healthcheck() -> String {
    gethostname::gethostname().into_string().unwrap()
}
