use lazy_static::lazy_static;
use prometheus::{opts, register_counter, Counter};

lazy_static! {
    pub static ref DEMO_COUNTER: Counter =
        register_counter!(opts!("count_total", "times the /count endpoint was called")).unwrap();
}

pub async fn root() -> &'static str {
    "Hello, World!"
}

pub async fn count() -> &'static str {
    DEMO_COUNTER.inc();

    "count"
}
