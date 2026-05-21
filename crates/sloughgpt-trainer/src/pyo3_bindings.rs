use crate::{TrainConfig, Trainer};
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct PyTrainer {
    inner: Trainer,
}

#[pymethods]
impl PyTrainer {
    #[new]
    fn new(config: TrainConfig) -> Self {
        Self {
            inner: Trainer::new(config),
        }
    }

    fn step(&mut self, batch: Vec<i32>, targets: Vec<i32>) -> f32 {
        self.inner.step(&batch, &targets)
    }

    fn forward(&self, tokens: Vec<i32>) -> Vec<f32> {
        self.inner.forward(&tokens)
    }

    fn get_weights(&self) -> HashMap<String, Vec<f32>> {
        self.inner.get_weights()
    }

    fn load_weights(&mut self, weights: HashMap<String, Vec<f32>>) {
        self.inner.load_weights(weights)
    }

    fn save_checkpoint(&self, path: &str) -> PyResult<()> {
        self.inner
            .save_checkpoint(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    fn load_checkpoint(&mut self, path: &str) -> PyResult<()> {
        self.inner
            .load_checkpoint(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    fn get_config(&self) -> TrainConfig {
        self.inner.get_config()
    }
}

pub fn into_py_module(py: Python<'_>) -> pyo3::PyResult<Bound<'_, pyo3::types::PyModule>> {
    let m = pyo3::types::PyModule::new(py, "sloughgpt_trainer")?;
    m.add_class::<PyTrainer>()?;
    m.add_class::<TrainConfig>()?;
    Ok(m)
}
