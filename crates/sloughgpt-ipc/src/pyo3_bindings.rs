//! PyO3 bindings for SloughGPT IPC module
//!
//! Provides Python-friendly interface to the IPC layer.

use crate::{traits::IpcMessage, IpcChannel, IpcConfig};
use parking_lot::RwLock;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
pub struct PyIpcConfig {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub capacity_bytes: usize,
}

#[pymethods]
impl PyIpcConfig {
    #[new]
    pub fn new(name: String, capacity_bytes: usize) -> Self {
        Self {
            name,
            capacity_bytes,
        }
    }
}

impl From<PyIpcConfig> for IpcConfig {
    fn from(py_config: PyIpcConfig) -> Self {
        IpcConfig::new(py_config.name, py_config.capacity_bytes)
    }
}

#[pyclass]
pub struct PyIpcChannel {
    inner: Arc<RwLock<Option<IpcChannel>>>,
    config: PyIpcConfig,
}

#[pymethods]
impl PyIpcChannel {
    #[new]
    pub fn new(config: PyIpcConfig) -> PyResult<Self> {
        let inner = IpcChannel::new(config.clone().into())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(RwLock::new(Some(inner))),
            config,
        })
    }

    #[staticmethod]
    pub fn connect(config: PyIpcConfig) -> PyResult<Self> {
        let inner = IpcChannel::connect(config.clone().into())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(RwLock::new(Some(inner))),
            config,
        })
    }

    pub fn send(&self, data: &Bound<PyAny>) -> PyResult<()> {
        let mut inner = self.inner.write();
        let channel = inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Channel not connected"))?;

        let bytes_obj = data.call_method0("__bytes__")?;
        let bytes_slice = bytes_obj.downcast::<PyBytes>()?.as_bytes();
        let owned_bytes = bytes_slice.to_vec();

        channel
            .send::<Vec<u8>>(&owned_bytes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(())
    }

    pub fn recv(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut inner = self.inner.write();
        let channel = inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Channel not connected"))?;

        let data: Vec<u8> = channel
            .recv()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let pickle = PyModule::import(py, "pickle")?;
        let pickled = pickle.getattr("loads")?.call1((PyBytes::new(py, &data),))?;

        Ok(pickled.into())
    }

    pub fn send_float32(&self, data: Vec<f32>) -> PyResult<()> {
        let mut inner = self.inner.write();
        let channel = inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Channel not connected"))?;

        channel
            .send(&IpcMessage::Float32(data))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(())
    }

    pub fn recv_float32(&self) -> PyResult<Vec<f32>> {
        let mut inner = self.inner.write();
        let channel = inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Channel not connected"))?;

        let msg: IpcMessage = channel
            .recv()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        match msg {
            IpcMessage::Float32(data) => Ok(data),
            _ => Err(PyRuntimeError::new_err("Expected Float32 message")),
        }
    }

    pub fn send_weights(&self, weights: Vec<(String, Vec<f32>)>) -> PyResult<()> {
        let mut inner = self.inner.write();
        let channel = inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Channel not connected"))?;

        let map: serde_json::Map<String, serde_json::Value> = weights
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    serde_json::Value::Array(v.into_iter().map(serde_json::Value::from).collect()),
                )
            })
            .collect();

        channel
            .send(&IpcMessage::Weights(map))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(())
    }

    pub fn recv_weights(&self) -> PyResult<Vec<(String, Vec<f32>)>> {
        let mut inner = self.inner.write();
        let channel = inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Channel not connected"))?;

        let msg: IpcMessage = channel
            .recv()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        match msg {
            IpcMessage::Weights(map) => {
                let result: Vec<(String, Vec<f32>)> = map
                    .into_iter()
                    .filter_map(|(k, v)| match v {
                        serde_json::Value::Array(arr) => {
                            let floats: Vec<f32> = arr
                                .into_iter()
                                .filter_map(|x| x.as_f64().map(|f| f as f32))
                                .collect();
                            Some((k, floats))
                        }
                        _ => None,
                    })
                    .collect();
                Ok(result)
            }
            _ => Err(PyRuntimeError::new_err("Expected Weights message")),
        }
    }

    pub fn send_training_step(&self, batch: Vec<i32>, targets: Vec<i32>) -> PyResult<()> {
        let mut inner = self.inner.write();
        let channel = inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Channel not connected"))?;

        channel
            .send(&IpcMessage::TrainingStep { batch, targets })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(())
    }

    pub fn recv_training_step(&self) -> PyResult<(Vec<i32>, Vec<i32>)> {
        let mut inner = self.inner.write();
        let channel = inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Channel not connected"))?;

        let msg: IpcMessage = channel
            .recv()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        match msg {
            IpcMessage::TrainingStep { batch, targets } => Ok((batch, targets)),
            _ => Err(PyRuntimeError::new_err("Expected TrainingStep message")),
        }
    }

    #[getter]
    pub fn name(&self) -> &str {
        &self.config.name
    }

    #[getter]
    pub fn capacity(&self) -> usize {
        self.config.capacity_bytes
    }
}

#[pymodule]
fn sloughgpt_ipc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIpcConfig>()?;
    m.add_class::<PyIpcChannel>()?;
    Ok(())
}
