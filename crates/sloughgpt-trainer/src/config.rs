use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyo3::pyclass]
pub struct TrainConfig {
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub seq_len: usize,
    #[pyo3(get, set)]
    pub vocab_size: usize,
    #[pyo3(get, set)]
    pub embedding_dim: usize,
    #[pyo3(get, set)]
    pub hidden_dim: usize,
    #[pyo3(get, set)]
    pub num_layers: usize,
    #[pyo3(get, set)]
    pub num_heads: usize,
    #[pyo3(get, set)]
    pub head_dim: usize,
    #[pyo3(get, set)]
    pub dropout: f32,
    #[pyo3(get, set)]
    pub learning_rate: f32,
    #[pyo3(get, set)]
    pub weight_decay: f32,
    #[pyo3(get, set)]
    pub beta1: f32,
    #[pyo3(get, set)]
    pub beta2: f32,
    #[pyo3(get, set)]
    pub eps: f32,
    #[pyo3(get, set)]
    pub grad_clip: f32,
    #[pyo3(get, set)]
    pub warmup_steps: usize,
    #[pyo3(get, set)]
    pub total_steps: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            seq_len: 512,
            vocab_size: 32000,
            embedding_dim: 768,
            hidden_dim: 3072,
            num_layers: 12,
            num_heads: 12,
            head_dim: 64,
            dropout: 0.1,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            grad_clip: 1.0,
            warmup_steps: 100,
            total_steps: 10000,
        }
    }
}

impl TrainConfig {
    pub fn from_inference_config(
        vocab_size: usize,
        embedding_dim: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
    ) -> Self {
        Self {
            vocab_size,
            embedding_dim,
            num_layers,
            num_heads,
            head_dim,
            hidden_dim,
            ..Default::default()
        }
    }
}
