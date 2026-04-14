//! File-backed `mmap` on Unix (including macOS).

use std::fs::{self, OpenOptions};
use std::io;
use std::path::PathBuf;

use crate::error::OpenError;
use crate::options::QueueOptions;
use crate::semaphore::Semaphore;

/// File-backed queue: keeps the `.qu` file open alongside a writable [`memmap2::MmapMut`].
pub(super) struct UnixMapping {
    /// Open file handle; must outlive `mmap`.
    _file: std::fs::File,
    /// Writable mapping of the entire file.
    mmap: memmap2::MmapMut,
    /// Path passed to [`crate::QueueOptions::file_path`].
    file_path: PathBuf,
    /// Byte length of the mapping (header plus ring).
    len: usize,
}

impl UnixMapping {
    /// Returns the start of the mapped file.
    pub(super) fn as_ptr(&self) -> *const u8 {
        self.mmap.as_ptr()
    }

    /// Returns the start of the mapped file for writes.
    pub(super) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.mmap.as_mut_ptr()
    }

    /// Length of the mapping in bytes.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Path to the backing `.qu` file (always [`Some`] on Unix).
    pub(super) fn backing_file_path(&self) -> Option<&PathBuf> {
        Some(&self.file_path)
    }
}

/// Opens or creates the `.qu` file, sets its length, maps it read/write, and opens the POSIX semaphore.
pub(super) fn open_queue(options: &QueueOptions) -> Result<(UnixMapping, Semaphore), OpenError> {
    let path = options.file_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(OpenError)?;
    }

    let storage_size = options.actual_storage_size() as u64;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        // Do not truncate: additional participants must retain existing queue contents.
        .truncate(false)
        .open(&path)
        .map_err(OpenError)?;

    file.set_len(storage_size).map_err(OpenError)?;

    let mmap = unsafe {
        memmap2::MmapMut::map_mut(&file)
            .map_err(|e| OpenError(io::Error::other(format!("mmap failed: {e}"))))?
    };

    let sem = Semaphore::open(options.memory_view_name.as_str()).map_err(OpenError)?;

    let len = storage_size as usize;
    Ok((
        UnixMapping {
            _file: file,
            mmap,
            file_path: path,
            len,
        },
        sem,
    ))
}
