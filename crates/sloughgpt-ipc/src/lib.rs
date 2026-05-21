//! SloughGPT IPC Layer
//!
//! Platform-agnostic inter-process communication for local Python <-> Rust communication.
//!
//! Architecture:
//! - macOS: Memory-mapped files (mmap)
//! - Linux: System V shared memory
//! - Windows: Named pipes

pub mod error;
pub mod traits;

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(feature = "pyo3")]
pub mod pyo3_bindings;

pub use error::IpcError;
pub use traits::{IpcConfig, IpcResult};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub struct IpcChannel {
    inner: Arc<RwLock<Option<SharedBuffer>>>,
    config: IpcConfig,
}

struct SharedBuffer {
    #[cfg(target_os = "macos")]
    buf: macos::MmapBuffer,
    #[cfg(target_os = "linux")]
    buf: linux::ShmBuffer,
    #[cfg(target_os = "windows")]
    buf: windows::NamedPipeBuffer,
}

impl IpcChannel {
    pub fn new(config: IpcConfig) -> Result<Self, IpcError> {
        let inner = Arc::new(RwLock::new(None));

        #[cfg(target_os = "macos")]
        let buf = macos::MmapBuffer::new(&config.name, config.capacity_bytes)?;

        #[cfg(target_os = "linux")]
        let buf = linux::ShmBuffer::new(&config.name, config.capacity_bytes)?;

        #[cfg(target_os = "windows")]
        let buf = windows::NamedPipeBuffer::new(&config.name, config.capacity_bytes)?;

        *inner.write() = Some(SharedBuffer { buf });

        Ok(Self { inner, config })
    }

    pub fn connect(config: IpcConfig) -> Result<Self, IpcError> {
        let inner = Arc::new(RwLock::new(None));

        #[cfg(target_os = "macos")]
        let buf = macos::MmapBuffer::connect(&config.name)?;

        #[cfg(target_os = "linux")]
        let buf = linux::ShmBuffer::connect(&config.name)?;

        #[cfg(target_os = "windows")]
        let buf = windows::NamedPipeBuffer::connect(&config.name)?;

        *inner.write() = Some(SharedBuffer { buf });

        Ok(Self { inner, config })
    }

    pub fn send<T: Serialize>(&self, data: &T) -> Result<(), IpcError> {
        let mut inner = self.inner.write();
        let buffer = inner.as_mut().ok_or(IpcError::NotConnected)?;

        let bytes = serde_json::to_vec(data)?;

        #[cfg(target_os = "macos")]
        buffer.buf.write(&bytes)?;

        #[cfg(target_os = "linux")]
        buffer.buf.write(&bytes)?;

        #[cfg(target_os = "windows")]
        buffer.buf.write(&bytes)?;

        Ok(())
    }

    pub fn recv<T: for<'de> Deserialize<'de>>(&self) -> Result<T, IpcError> {
        let mut inner = self.inner.write();
        let buffer = inner.as_mut().ok_or(IpcError::NotConnected)?;

        #[cfg(target_os = "macos")]
        let bytes = buffer.buf.read()?;

        #[cfg(target_os = "linux")]
        let bytes = buffer.buf.read()?;

        #[cfg(target_os = "windows")]
        let bytes = buffer.buf.read()?;

        let data: T = serde_json::from_slice(&bytes)?;
        Ok(data)
    }

    pub fn name(&self) -> &str {
        &self.config.name
    }

    pub fn capacity(&self) -> usize {
        self.config.capacity_bytes
    }
}

#[cfg(target_os = "macos")]
pub type PlatformBuffer = macos::MmapBuffer;
#[cfg(target_os = "linux")]
pub type PlatformBuffer = linux::ShmBuffer;
#[cfg(target_os = "windows")]
pub type PlatformBuffer = windows::NamedPipeBuffer;
