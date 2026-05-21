//! IPC Error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum IpcError {
    #[error("Failed to create shared memory: {0}")]
    CreationFailed(String),

    #[error("Failed to connect: {0}")]
    ConnectFailed(String),

    #[error("Not connected to IPC channel")]
    NotConnected,

    #[error("Buffer overflow: attempted to write {attempted} bytes, capacity is {capacity}")]
    BufferOverflow { attempted: usize, capacity: usize },

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[cfg(target_os = "macos")]
    #[error("Mmap error: {0}")]
    Mmap(String),

    #[cfg(target_os = "linux")]
    #[error("Shm error: {0}")]
    Shm(String),

    #[cfg(target_os = "windows")]
    #[error("Named pipe error: {0}")]
    NamedPipe(String),
}

impl From<serde_json::Error> for IpcError {
    fn from(e: serde_json::Error) -> Self {
        IpcError::Serialization(e.to_string())
    }
}
