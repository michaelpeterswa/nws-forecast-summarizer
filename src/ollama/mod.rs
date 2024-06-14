use ollama_rs::Ollama;

pub fn connect(host: String, port: u16) -> Ollama {
    let client = Ollama::new(host, port);

    client
}
