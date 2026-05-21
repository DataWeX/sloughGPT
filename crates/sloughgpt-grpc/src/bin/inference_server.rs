//! Inference Server Binary
//! 
//! Runs the inference engine as a gRPC server.

use sloughgpt_grpc::InferenceGrpcService;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting SloughGPT Inference gRPC Server");

    let service = InferenceGrpcService::default();

    let health = service.health_check().await;
    tracing::info!(
        "Health: healthy={}, version={}, status={}",
        health.healthy,
        health.version,
        health.status
    );

    tracing::info!("Server ready. Press Ctrl+C to stop.");

    tokio::signal::ctrl_c().await?;

    tracing::info!("Shutting down...");
    Ok(())
}
