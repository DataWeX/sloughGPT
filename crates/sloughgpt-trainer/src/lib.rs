//! SloughGPT Training Core
//! High-performance Rust training engine with Python protocol layer.

pub mod config;
pub mod model;
pub mod optimizer;

pub use config::TrainConfig;
pub use model::{matvec, rms_norm, silu, softmax, ModelWeights, Trainer};

#[cfg(feature = "pyo3")]
mod pyo3_bindings;

#[cfg(feature = "pyo3")]
pub use pyo3_bindings::*;
