//! Trainer gRPC Service
//! 
//! Async service for distributed training coordination.

use sloughgpt_trainer::{TrainConfig, Trainer};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Trainer service state
pub struct TrainerGrpcService {
    trainer: Arc<RwLock<Trainer>>,
}

impl TrainerGrpcService {
    pub fn new(config: TrainConfig) -> Self {
        Self {
            trainer: Arc::new(RwLock::new(Trainer::new(config))),
        }
    }

    pub async fn step(&self, batch: Vec<i32>, targets: Vec<i32>) -> TrainingResult {
        let mut trainer = self.trainer.write().await;
        let loss = trainer.step(&batch, &targets);
        let config = trainer.get_config();
        
        TrainingResult {
            loss,
            learning_rate: config.learning_rate,
            step: 0, // TODO: get actual step
        }
    }

    pub async fn forward(&self, tokens: Vec<i32>) -> Vec<f32> {
        let trainer = self.trainer.read().await;
        trainer.forward(&tokens)
    }

    pub async fn get_weights(&self) -> HashMap<String, Vec<f32>> {
        let trainer = self.trainer.read().await;
        trainer.get_weights()
    }

    pub async fn load_weights(&self, weights: HashMap<String, Vec<f32>>) {
        let mut trainer = self.trainer.write().await;
        trainer.load_weights(weights);
    }

    pub async fn save_checkpoint(&self, path: &str) -> Result<(), String> {
        let trainer = self.trainer.read().await;
        trainer.save_checkpoint(path).map_err(|e| e.to_string())
    }

    pub async fn load_checkpoint(&self, path: &str) -> Result<(), String> {
        let mut trainer = self.trainer.write().await;
        trainer.load_checkpoint(path).map_err(|e| e.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub loss: f32,
    pub learning_rate: f32,
    pub step: usize,
}

/// Gradient aggregation for federated learning
pub struct GradientAggregator {
    pending_updates: Arc<RwLock<Vec<GradientUpdate>>>,
    num_nodes: usize,
}

#[derive(Debug, Clone)]
pub struct GradientUpdate {
    pub node_id: String,
    pub gradients: HashMap<String, Vec<f32>>,
    pub num_samples: usize,
}

impl GradientAggregator {
    pub fn new(num_nodes: usize) -> Self {
        Self {
            pending_updates: Arc::new(RwLock::new(Vec::new())),
            num_nodes,
        }
    }

    pub async fn add_update(&self, update: GradientUpdate) {
        let mut pending = self.pending_updates.write().await;
        pending.push(update);
    }

    pub async fn aggregate(&self) -> Option<HashMap<String, Vec<f32>>> {
        let pending = self.pending_updates.read().await;
        
        if pending.len() < self.num_nodes {
            return None;
        }

        // Clear pending and aggregate
        drop(pending);
        let mut pending = self.pending_updates.write().await;
        let updates: Vec<_> = pending.drain(..).collect();
        
        if updates.is_empty() {
            return None;
        }

        let mut aggregated: HashMap<String, Vec<f32>> = HashMap::new();
        let total_samples: usize = updates.iter().map(|u| u.num_samples).sum();

        for update in &updates {
            for (name, grad) in &update.gradients {
                let weight = update.num_samples as f32 / total_samples as f32;
                let entry = aggregated.entry(name.clone()).or_insert_with(|| vec![0.0; grad.len()]);
                for (i, &g) in grad.iter().enumerate() {
                    entry[i] += g * weight;
                }
            }
        }

        Some(aggregated)
    }
}
