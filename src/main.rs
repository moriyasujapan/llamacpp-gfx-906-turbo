use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::num::NonZeroU32;

fn main() {
    let backend = LlamaBackend::init().expect("Failed to initialize llama backend");

    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .expect("Usage: llamacpp-turbo <model.gguf> [prompt]");
    let prompt = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("Hello, world!");

    let model_params = LlamaModelParams::default().with_n_gpu_layers(999);
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .expect("Failed to load model");

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048));

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .expect("Failed to create context");

    let tokens = model
        .str_to_token(prompt, llama_cpp_2::model::AddBos::Always)
        .expect("Failed to tokenize");

    println!("Prompt tokens: {}", tokens.len());

    let mut batch = LlamaBatch::new(2048, 1);
    for (i, &token) in tokens.iter().enumerate() {
        batch
            .add(token, i as i32, &[0], i == tokens.len() - 1)
            .expect("Failed to add token to batch");
    }

    ctx.decode(&mut batch).expect("Failed to decode prompt");

    let mut n_decoded = tokens.len();
    let max_tokens = 128;

    for _ in 0..max_tokens {
        let mut candidates = ctx.token_data_array_ith(batch.n_tokens() - 1);
        let token = candidates.sample_token_greedy();

        if model.is_eog_token(token) {
            break;
        }

        let piece = model
            .token_to_str(token, llama_cpp_2::model::Special::Tokenize)
            .unwrap_or_default();
        print!("{piece}");

        batch.clear();
        batch
            .add(token, n_decoded as i32, &[0], true)
            .expect("Failed to add token");

        ctx.decode(&mut batch).expect("Failed to decode");
        n_decoded += 1;
    }
    println!();
    println!("Generated {} tokens", n_decoded - tokens.len());
}
