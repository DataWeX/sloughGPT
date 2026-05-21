//! Linux-specific IPC using System V shared memory
//!
//! Uses shmget/shmat/shmdt/shmctl for inter-process communication.
//! This is the traditional and efficient mechanism for sharing memory
//! between processes on Linux.

use crate::error::IpcError;
use libc::{c_int, c_void, shmat, shmctl, shmdt, shmget, shmid_ds, size_t, IPC_CREAT, IPC_EXCL};
use std::ptr;

const HEADER_SIZE: usize = 64;

pub struct ShmBuffer {
    shm_id: c_int,
    ptr: *mut c_void,
    name: String,
    capacity: usize,
    size: usize,
}

impl ShmBuffer {
    pub fn new(name: &str, capacity: usize) -> Result<Self, IpcError> {
        let sanitized_key = Self::get_shm_key(name)?;
        let size = HEADER_SIZE + capacity;

        let shm_id = unsafe { shmget(sanitized_key, size, IPC_CREAT | IPC_EXCL | 0o666) };

        if shm_id < 0 {
            let existing_id = unsafe { shmget(sanitized_key, 0, 0) };
            if existing_id >= 0 {
                unsafe {
                    shmctl(existing_id, libc::IPC_RMID, ptr::null_mut());
                }
            }

            let shm_id = unsafe { shmget(sanitized_key, size, IPC_CREAT | IPC_EXCL | 0o666) };
            if shm_id < 0 {
                return Err(IpcError::Shm(format!(
                    "Failed to create shared memory: {}",
                    std::io::Error::last_os_error()
                )));
            }
            shm_id
        } else {
            shm_id
        };

        let ptr = unsafe { shmat(shm_id, ptr::null(), 0) };
        if ptr == ptr::null_mut() {
            return Err(IpcError::Shm("Failed to attach shared memory".into()));
        }

        let mut buffer = Self {
            shm_id,
            ptr,
            name: name.to_string(),
            capacity,
            size,
        };

        buffer.write_length(0);
        buffer.write_ready(0);

        Ok(buffer)
    }

    pub fn connect(name: &str) -> Result<Self, IpcError> {
        let sanitized_key = Self::get_shm_key(name)?;

        let shm_id = unsafe { shmget(sanitized_key, 0, 0) };
        if shm_id < 0 {
            return Err(IpcError::ConnectFailed(format!(
                "Shared memory '{}' not found",
                name
            )));
        }

        let mut stats: shmid_ds = unsafe { std::mem::zeroed() };
        unsafe {
            shmctl(shm_id, libc::IPC_STAT, &mut stats);
        }
        let size = stats.shm_segsz as usize;
        let capacity = size.saturating_sub(HEADER_SIZE);

        let ptr = unsafe { shmat(shm_id, ptr::null(), 0) };
        if ptr == ptr::null_mut() {
            return Err(IpcError::Shm("Failed to attach to shared memory".into()));
        }

        Ok(Self {
            shm_id,
            ptr,
            name: name.to_string(),
            capacity,
            size,
        })
    }

    fn get_shm_key(name: &str) -> Result<c_int, IpcError> {
        let hash = simple_hash(name);
        let key = (0xC0C0_0000 | (hash & 0xFFFF)) as c_int;
        Ok(key)
    }

    fn write_at(&self, offset: usize, data: &[u8]) {
        unsafe {
            let dest = self.ptr.cast::<u8>().add(offset);
            dest.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }
    }

    fn read_at(&self, offset: usize, len: usize) -> Vec<u8> {
        unsafe {
            let src = self.ptr.cast::<u8>().add(offset);
            std::slice::from_raw_parts(src, len).to_vec()
        }
    }

    fn write_length(&self, len: u32) {
        self.write_at(0, &len.to_le_bytes());
    }

    fn read_length(&self) -> usize {
        let bytes = self.read_at(0, 4);
        u32::from_le_bytes(bytes.try_into().unwrap()) as usize
    }

    fn write_ready(&self, ready: u32) {
        self.write_at(4, &ready.to_le_bytes());
    }

    fn read_ready(&self) -> u32 {
        let bytes = self.read_at(4, 4);
        u32::from_le_bytes(bytes.try_into().unwrap())
    }

    pub fn write(&self, data: &[u8]) -> Result<(), IpcError> {
        if data.len() > self.capacity {
            return Err(IpcError::BufferOverflow {
                attempted: data.len(),
                capacity: self.capacity,
            });
        }

        self.write_length(data.len() as u32);
        self.write_at(HEADER_SIZE, data);
        self.write_ready(1);

        Ok(())
    }

    pub fn read(&self) -> Result<Vec<u8>, IpcError> {
        if self.read_ready() == 0 {
            return Err(IpcError::Shm("No data available".into()));
        }

        let len = self.read_length();
        if len > self.capacity {
            return Err(IpcError::Shm("Corrupt length in header".into()));
        }

        let data = self.read_at(HEADER_SIZE, len);
        self.write_ready(0);

        Ok(data)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Drop for ShmBuffer {
    fn drop(&mut self) {
        unsafe {
            shmdt(self.ptr);
            shmctl(self.shm_id, libc::IPC_RMID, ptr::null_mut());
        }
    }
}

fn simple_hash(s: &str) -> u32 {
    let mut hash: u32 = 5381;
    for c in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(c as u32);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shm_key_generation() {
        let key1 = simple_hash("test");
        let key2 = simple_hash("test");
        assert_eq!(key1, key2);

        let key3 = simple_hash("different");
        assert_ne!(key1, key3);
    }
}
