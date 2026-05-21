//! IPC trait definitions

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone)]
pub struct IpcConfig {
    pub name: String,
    pub capacity_bytes: usize,
}

impl IpcConfig {
    pub fn new(name: impl Into<String>, capacity_bytes: usize) -> Self {
        Self {
            name: name.into(),
            capacity_bytes,
        }
    }
}

pub trait IpcChannel: Send + Sync {
    fn send<T: Serialize>(&self, data: &T) -> Result<(), super::error::IpcError>;
    fn recv<T: for<'a> Deserialize<'a>>(&self) -> Result<T, super::error::IpcError>;
    fn name(&self) -> &str;
    fn capacity(&self) -> usize;
}

pub type IpcResult<T> = Result<T, super::error::IpcError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IpcMessage {
    Float32(Vec<f32>),
    Int32(Vec<i32>),
    String(String),
    Tensor {
        name: String,
        data: Vec<f32>,
        shape: Vec<usize>,
    },
    Weights(serde_json::Map<String, serde_json::Value>),
    GradientUpdate {
        node_id: String,
        gradients: serde_json::Map<String, serde_json::Value>,
        num_samples: usize,
    },
    TrainingStep {
        batch: Vec<i32>,
        targets: Vec<i32>,
    },
    InferenceRequest {
        tokens: Vec<i32>,
        max_length: usize,
    },
    Shutdown,
}

impl fmt::Display for IpcMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IpcMessage::Float32(v) => write!(f, "Float32({} elements)", v.len()),
            IpcMessage::Int32(v) => write!(f, "Int32({} elements)", v.len()),
            IpcMessage::String(s) => write!(f, "String({} chars)", s.len()),
            IpcMessage::Tensor { name, shape, .. } => write!(f, "Tensor({:?}, {:?})", name, shape),
            IpcMessage::Weights(m) => write!(f, "Weights({} keys)", m.len()),
            IpcMessage::GradientUpdate { node_id, .. } => write!(f, "GradientUpdate({})", node_id),
            IpcMessage::TrainingStep { batch, .. } => {
                write!(f, "TrainingStep({} tokens)", batch.len())
            }
            IpcMessage::InferenceRequest { tokens, .. } => {
                write!(f, "InferenceRequest({} tokens)", tokens.len())
            }
            IpcMessage::Shutdown => write!(f, "Shutdown"),
        }
    }
}
