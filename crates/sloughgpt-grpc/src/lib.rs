//! SloughGPT gRPC Services
//! 
//! Async gRPC services for distributed training and inference.
//! 
//! Architecture:
//! - Local: Direct PyO3 calls (fastest)
//! - Distributed: gRPC over network
//! - Streaming: gRPC streaming for federated learning

pub mod trainer_service;
pub mod inference_service;

pub use trainer_service::{TrainerGrpcService, GradientAggregator, GradientUpdate};
pub use inference_service::InferenceGrpcService;

use std::collections::HashMap;

#[allow(dead_code)]
pub fn weights_to_proto(weights: HashMap<String, Vec<f32>>) -> Vec<(String, Vec<f32>)> {
    weights.into_iter().collect()
}

#[allow(dead_code)]
pub fn weights_from_proto(weights: Vec<(String, Vec<f32>)>) -> HashMap<String, Vec<f32>> {
    weights.into_iter().collect()
}
