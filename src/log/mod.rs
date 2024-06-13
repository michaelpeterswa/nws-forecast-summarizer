pub fn init(s: String) {
    let log_level = s
        .parse::<tracing::Level>()
        .unwrap_or_else(|_| panic!("{} is not a valid log level", s));

    tracing_subscriber::fmt()
        .json()
        .with_max_level(log_level)
        .with_current_span(false)
        .with_span_list(false)
        .flatten_event(true)
        .with_target(false)
        .init();
}
