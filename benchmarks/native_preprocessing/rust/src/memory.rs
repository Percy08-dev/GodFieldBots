//! Same coarse policies as native json.rs/control_memory.rs; no per-node charging.
//! Managed estimates are admission limits, not an allocator/RSS hard limit.
use std::os::unix::fs::MetadataExt;
use std::{
    fs::{self, File},
    io::Read,
    path::Path,
};

pub const MIB: usize = 1024 * 1024;
pub fn control_bytes(request: usize) -> Result<usize, String> {
    request
        .checked_mul(64)
        .and_then(|v| v.checked_add(32 * MIB))
        .ok_or("control estimate overflow".into())
}
pub fn json_bytes(raw: usize) -> Result<usize, String> {
    raw.checked_mul(32)
        .and_then(|v| v.checked_add(65536))
        .ok_or("JSON estimate overflow".into())
}
pub fn raw_bytes(raw: usize) -> Result<usize, String> {
    raw.checked_add(1).ok_or("raw capacity overflow".into())
}
#[derive(Debug)]
pub struct Memory {
    pub limit: usize,
    pub used: usize,
    pub peak: usize,
}
impl Memory {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            used: 0,
            peak: 0,
        }
    }
    pub fn reserve(&mut self, bytes: usize) -> Result<(), String> {
        let used = self
            .used
            .checked_add(bytes)
            .ok_or("managed memory overflow")?;
        if used > self.limit {
            return Err(format!(
                "managed memory admission exceeds budget: requested={bytes},used={},limit={}",
                self.used, self.limit
            ));
        }
        self.used = used;
        self.peak = self.peak.max(used);
        Ok(())
    }
    pub fn release(&mut self, bytes: usize) -> Result<(), String> {
        self.used = self
            .used
            .checked_sub(bytes)
            .ok_or("managed memory release underflow")?;
        Ok(())
    }
    pub fn admit_file(&mut self, length: usize) -> Result<(usize, usize), String> {
        let raw = raw_bytes(length)?;
        let json = json_bytes(length)?;
        self.reserve(raw.checked_add(json).ok_or("file admission overflow")?)?;
        Ok((raw, json))
    }
    pub fn assert_capacity(&self, capacity: usize, admitted: usize) -> Result<(), String> {
        if capacity > admitted || admitted > self.used || self.used > self.limit {
            return Err("raw allocation exceeded pre-admitted capacity".into());
        }
        Ok(())
    }
}
fn identity(meta: &fs::Metadata) -> (u64, u64, u64, i64, i64, i64, i64) {
    (
        meta.dev(),
        meta.ino(),
        meta.len(),
        meta.mtime(),
        meta.mtime_nsec(),
        meta.ctime(),
        meta.ctime_nsec(),
    )
}
/// Read at most declared+1 bytes with preallocated, already charged capacity.
/// Growth, replacement and size changes fail; never use unbounded read_to_end.
pub fn read_admitted(
    path: &Path,
    length: usize,
    capacity: usize,
    memory: &Memory,
) -> Result<Vec<u8>, String> {
    if capacity != raw_bytes(length)? {
        return Err("raw read admission mismatch".into());
    }
    memory.assert_capacity(capacity, capacity)?;
    let before = fs::metadata(path).map_err(|e| e.to_string())?;
    if before.len() != length as u64 {
        return Err("declared source size changed".into());
    }
    let mut input = File::open(path).map_err(|e| e.to_string())?;
    if identity(&input.metadata().map_err(|e| e.to_string())?) != identity(&before) {
        return Err("source replaced before read".into());
    }
    let mut raw = Vec::new();
    raw.try_reserve_exact(capacity).map_err(|e| e.to_string())?;
    memory.assert_capacity(raw.capacity(), capacity)?;
    let mut block = [0u8; 32768];
    loop {
        let left = capacity
            .checked_sub(raw.len())
            .ok_or("read capacity exhausted")?;
        if left == 0 {
            return Err("source grew during bounded read".into());
        }
        let count = input
            .read(&mut block[..left.min(32768)])
            .map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        raw.extend_from_slice(&block[..count]);
        if raw.len() > length {
            return Err("source grew during bounded read".into());
        }
    }
    if raw.len() != length
        || identity(&input.metadata().map_err(|e| e.to_string())?) != identity(&before)
        || identity(&fs::metadata(path).map_err(|e| e.to_string())?) != identity(&before)
    {
        return Err("source changed during bounded read".into());
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn estimates_overflow_and_admission_are_atomic() {
        assert_eq!(json_bytes(100).unwrap(), 68736);
        assert_eq!(control_bytes(100).unwrap(), 32 * MIB + 6400);
        assert!(json_bytes(usize::MAX).is_err());
        assert!(raw_bytes(usize::MAX).is_err());
        assert!(control_bytes(usize::MAX).is_err());
        let mut memory = Memory::new(65536);
        assert!(memory.admit_file(1).is_err());
        assert_eq!(memory.used, 0);
        assert_eq!(memory.peak, 0);
        memory.reserve(65535).unwrap();
        assert!(memory.reserve(2).is_err());
        assert_eq!(memory.used, 65535);
        assert!(memory.assert_capacity(2, 1).is_err());
        memory.release(65535).unwrap();
        assert_eq!(memory.used, 0);
    }
    #[test]
    fn file_refused_before_read_or_allocation() {
        let mut memory = Memory::new(1024);
        // No file is opened or heap buffer allocated if this preflight fails.
        let result = memory.admit_file(crate::common::MAX_FILE);
        assert!(result.is_err());
        assert_eq!(memory.peak, 0);
        let error = read_admitted(
            Path::new("/path-that-must-never-be-opened"),
            100,
            101,
            &memory,
        )
        .unwrap_err();
        assert!(error.contains("pre-admitted capacity"));
    }
}
