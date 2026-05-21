//! macOS-specific IPC using memory-mapped files
//!
//! Uses mmap with a backing file in /tmp for inter-process communication.
//! This provides a simple and reliable mechanism for sharing memory between
//! Python and Rust processes on macOS.

use crate::error::IpcError;
use libc::{flock, LOCK_EX, LOCK_UN};
use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::path::PathBuf;

const HEADER_SIZE: usize = 64;

pub struct MmapBuffer {
    data_file: File,
    mmap: MmapMut,
    name: String,
    capacity: usize,
    lock_file: File,
}

impl MmapBuffer {
    pub fn new(name: &str, capacity: usize) -> Result<Self, IpcError> {
        let base_path = Self::get_base_path(name)?;

        let data_path = base_path.with_extension("data");
        let lock_path = base_path.with_extension("lock");

        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&data_path)?;

        data_file.set_len((HEADER_SIZE + capacity) as u64)?;

        let file_size = HEADER_SIZE + capacity;

        let lock_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&lock_path)?;

        let mut mmap = unsafe { MmapOptions::new().len(file_size).map_mut(&data_file)? };

        mmap[..4].copy_from_slice(&(0u32).to_le_bytes());

        Ok(Self {
            data_file,
            mmap,
            name: name.to_string(),
            capacity,
            lock_file,
        })
    }

    pub fn connect(name: &str) -> Result<Self, IpcError> {
        let base_path = Self::get_base_path(name)?;

        let data_path = base_path.with_extension("data");
        let lock_path = base_path.with_extension("lock");

        let data_file = OpenOptions::new().read(true).write(true).open(&data_path)?;

        let lock_file = OpenOptions::new().read(true).write(true).open(&lock_path)?;

        let file_size = data_file.metadata()?.len() as usize;
        let capacity = file_size.saturating_sub(HEADER_SIZE);

        let mmap = unsafe { MmapOptions::new().len(file_size).map_mut(&data_file)? };

        Ok(Self {
            data_file,
            mmap,
            name: name.to_string(),
            capacity,
            lock_file,
        })
    }

    fn get_base_path(name: &str) -> Result<PathBuf, IpcError> {
        let tmp_dir = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
        let sanitized_name = name.replace(['/', '\\', ':'], "_");
        Ok(PathBuf::from(tmp_dir).join(format!("sloughgpt_ipc_{}", sanitized_name)))
    }

    fn acquire_lock(&self) -> Result<(), IpcError> {
        unsafe {
            let ret = flock(self.lock_file.as_raw_fd(), LOCK_EX);
            if ret != 0 {
                return Err(IpcError::Mmap("Failed to acquire lock".into()));
            }
        }
        Ok(())
    }

    fn release_lock(&self) -> Result<(), IpcError> {
        unsafe {
            let ret = flock(self.lock_file.as_raw_fd(), LOCK_UN);
            if ret != 0 {
                return Err(IpcError::Mmap("Failed to release lock".into()));
            }
        }
        Ok(())
    }

    pub fn write(&mut self, data: &[u8]) -> Result<(), IpcError> {
        if data.len() > self.capacity {
            return Err(IpcError::BufferOverflow {
                attempted: data.len(),
                capacity: self.capacity,
            });
        }

        self.acquire_lock()?;

        self.mmap[..4].copy_from_slice(&(data.len() as u32).to_le_bytes());
        self.mmap[HEADER_SIZE..HEADER_SIZE + data.len()].copy_from_slice(data);
        self.mmap[4..8].copy_from_slice(&(1u32).to_le_bytes());

        self.release_lock()?;

        self.mmap.flush()?;

        Ok(())
    }

    pub fn read(&mut self) -> Result<Vec<u8>, IpcError> {
        self.acquire_lock()?;

        let ready: u32 = u32::from_le_bytes(
            self.mmap[4..8]
                .try_into()
                .map_err(|_| IpcError::Mmap("Invalid ready flag".into()))?,
        );

        if ready == 0 {
            self.release_lock()?;
            return Err(IpcError::Mmap("No data available".into()));
        }

        let len = u32::from_le_bytes(
            self.mmap[..4]
                .try_into()
                .map_err(|_| IpcError::Mmap("Invalid length".into()))?,
        ) as usize;

        if len > self.capacity {
            self.release_lock()?;
            return Err(IpcError::Mmap("Corrupt length in header".into()));
        }

        let data = self.mmap[HEADER_SIZE..HEADER_SIZE + len].to_vec();

        self.mmap[4..8].copy_from_slice(&(0u32).to_le_bytes());

        self.release_lock()?;

        Ok(data)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

use std::os::fd::AsRawFd;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_buffer_creation() {
        let buffer = MmapBuffer::new("test_buffer", 1024).unwrap();
        assert_eq!(buffer.name(), "test_buffer");
        assert_eq!(buffer.capacity(), 1024);
    }

    #[test]
    fn test_write_read() {
        let mut buffer = MmapBuffer::new("test_write_read", 1024).unwrap();

        let data = b"Hello, IPC!";
        buffer.write(data).unwrap();

        let read = buffer.read().unwrap();
        assert_eq!(read, data);
    }
}
