//! Trainer Server Binary
//! 
//! Runs the trainer as a gRPC server.

use sloughgpt_grpc::{TrainerGrpcService, GradientAggregator};
use sloughgpt_trainer::TrainConfig;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting SloughGPT Trainer gRPC Server");

    let config = TrainConfig::default();
    let _service = TrainerGrpcService::new(config.clone());
    let _aggregator = GradientAggregator::new(1);

    tracing::info!("Trainer initialized");
    tracing::info!("Config: vocab_size={}, layers={}, dim={}", 
        config.vocab_size, config.num_layers, config.embedding_dim);

    // Note: Full gRPC server with tonic requires proto compilation
    // Run `cargo build` after proto files are generated
    
    // Example usage:
    // let loss = service.step(vec![1, 2, 3], vec![2, 3, 4]).await;
    // tracing::info!("Step loss: {}", loss);

    tracing::info!("Server ready. Press Ctrl+C to stop.");

    tokio::signal::ctrl_c().await?;

    tracing::info!("Shutting down...");
    Ok(())
}
