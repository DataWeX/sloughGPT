//! Windows-specific IPC using Named Pipes
//!
//! Uses CreateNamedPipe/ConnectNamedPipe for inter-process communication.
//! This is the standard mechanism for sharing data between processes on Windows.

use crate::error::IpcError;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

const HEADER_SIZE: usize = 64;
const PIPE_BUFFER_SIZE: usize = 64 * 1024;

pub struct NamedPipeBuffer {
    #[cfg(windows)]
    pipe: WindowsPipe,
    name: String,
    capacity: usize,
}

#[cfg(windows)]
struct WindowsPipe {
    handle: std::os::windows::io::RawHandle,
    connected: AtomicBool,
}

#[cfg(windows)]
impl NamedPipeBuffer {
    pub fn new(name: &str, capacity: usize) -> Result<Self, IpcError> {
        use std::os::windows::io::{AsRawHandle, FromRawHandle};
        use winapi::shared::winerror::ERROR_PIPE_BUSY;
        use winapi::um::namedpipeapi::CreateNamedPipeW;
        use winapi::um::winbase::INVALID_HANDLE_VALUE;
        use winapi::um::winbase::PIPE_UNLIMITED_INSTANCES;
        use winapi::um::winbase::{
            PIPE_ACCESS_DUPLEX, PIPE_READMODE_MESSAGE, PIPE_TYPE_MESSAGE, PIPE_WAIT,
        };

        let pipe_name = Self::format_pipe_name(name);
        let wide_name: Vec<u16> = pipe_name.encode_utf16().chain(std::iter::once(0)).collect();

        let handle = unsafe {
            CreateNamedPipeW(
                wide_name.as_ptr(),
                PIPE_ACCESS_DUPLEX,
                PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
                PIPE_UNLIMITED_INSTANCES,
                PIPE_BUFFER_SIZE as u32,
                PIPE_BUFFER_SIZE as u32,
                0,
                ptr::null_mut(),
            )
        };

        if handle == INVALID_HANDLE_VALUE {
            return Err(IpcError::NamedPipe(format!(
                "Failed to create pipe: {}",
                std::io::Error::last_os_error()
            )));
        }

        Ok(Self {
            pipe: WindowsPipe {
                handle: handle as std::os::windows::io::RawHandle,
                connected: AtomicBool::new(false),
            },
            name: name.to_string(),
            capacity,
        })
    }

    pub fn connect(name: &str) -> Result<Self, IpcError> {
        use std::os::windows::io::{AsRawHandle, IntoRawHandle};
        use winapi::shared::winerror::{ERROR_FILE_NOT_FOUND, ERROR_PIPE_BUSY};
        use winapi::um::winbase::OPEN_EXISTING;
        use winapi::um::winbase::{CreateFileW, WaitNamedPipeW};
        use winapi::um::winerror::ERROR_SEM_TIMEOUT;

        let pipe_name = Self::format_pipe_name(name);
        let wide_name: Vec<u16> = pipe_name.encode_utf16().chain(std::iter::once(0)).collect();

        unsafe {
            let timeout_ms: u32 = 5000;
            let ret = WaitNamedPipeW(wide_name.as_ptr(), timeout_ms);
            if ret == 0 {
                let err = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                if err == ERROR_FILE_NOT_FOUND as i32 || err == ERROR_SEM_TIMEOUT as i32 {
                    return Err(IpcError::ConnectFailed(format!(
                        "Pipe '{}' not found",
                        name
                    )));
                }
                return Err(IpcError::ConnectFailed(format!(
                    "Timeout waiting for pipe: {}",
                    name
                )));
            }
        }

        let handle = unsafe {
            CreateFileW(
                wide_name.as_ptr(),
                winapi::um::winbase::GENERIC_READ | winapi::um::winbase::GENERIC_WRITE,
                0,
                ptr::null_mut(),
                OPEN_EXISTING,
                0,
                ptr::null_mut(),
            )
        };

        if handle == winapi::um::winbase::INVALID_HANDLE_VALUE {
            return Err(IpcError::NamedPipe(format!(
                "Failed to connect to pipe: {}",
                std::io::Error::last_os_error()
            )));
        }

        Ok(Self {
            pipe: WindowsPipe {
                handle: handle as std::os::windows::io::RawHandle,
                connected: AtomicBool::new(true),
            },
            name: name.to_string(),
            capacity: PIPE_BUFFER_SIZE - HEADER_SIZE,
        })
    }

    fn format_pipe_name(name: &str) -> String {
        format!("\\\\.\\pipe\\sloughgpt_ipc_{}", name)
    }

    pub fn write(&self, data: &[u8]) -> Result<(), IpcError> {
        use std::os::windows::io::AsRawHandle;
        use winapi::um::fileapi::{ReadFile, WriteFile};
        use winapi::um::winbase::FlushFileBuffers;

        if data.len() > self.capacity {
            return Err(IpcError::BufferOverflow {
                attempted: data.len(),
                capacity: self.capacity,
            });
        }

        let mut header = vec![0u8; HEADER_SIZE];
        header[..4].copy_from_slice(&(data.len() as u32).to_le_bytes());

        let handle = self.pipe.handle;

        unsafe {
            let mut bytes_written: u32 = 0;
            let ret = WriteFile(
                handle as *mut _,
                header.as_ptr(),
                HEADER_SIZE as u32,
                &mut bytes_written,
                ptr::null_mut(),
            );

            if ret == 0 || bytes_written != HEADER_SIZE as u32 {
                return Err(IpcError::NamedPipe("Failed to write header".into()));
            }

            let mut bytes_written: u32 = 0;
            let ret = WriteFile(
                handle as *mut _,
                data.as_ptr(),
                data.len() as u32,
                &mut bytes_written,
                ptr::null_mut(),
            );

            if ret == 0 || bytes_written != data.len() as u32 {
                return Err(IpcError::NamedPipe("Failed to write data".into()));
            }
        }

        Ok(())
    }

    pub fn read(&self) -> Result<Vec<u8>, IpcError> {
        use std::os::windows::io::AsRawHandle;
        use winapi::um::fileapi::ReadFile;

        let handle = self.pipe.handle;

        let mut header = vec![0u8; HEADER_SIZE];
        unsafe {
            let mut bytes_read: u32 = 0;
            let ret = ReadFile(
                handle as *mut _,
                header.as_mut_ptr(),
                HEADER_SIZE as u32,
                &mut bytes_read,
                ptr::null_mut(),
            );

            if ret == 0 || bytes_read != HEADER_SIZE as u32 {
                return Err(IpcError::NamedPipe("Failed to read header".into()));
            }
        }

        let len = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
        if len > self.capacity {
            return Err(IpcError::NamedPipe("Corrupt length in header".into()));
        }

        let mut data = vec![0u8; len];
        unsafe {
            let mut bytes_read: u32 = 0;
            let ret = ReadFile(
                handle as *mut _,
                data.as_mut_ptr(),
                len as u32,
                &mut bytes_read,
                ptr::null_mut(),
            );

            if ret == 0 || bytes_read != len as u32 {
                return Err(IpcError::NamedPipe("Failed to read data".into()));
            }
        }

        Ok(data)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
