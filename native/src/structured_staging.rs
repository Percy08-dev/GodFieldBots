//! Private typed raw-input spool. No rewards, GAE, models or policy execution.
use crate::{
    budget::{Budget, BudgetVec, Control, Lease},
    error::{Error, Result},
    json::{self, Value, ValueExt},
};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::unix::{
        fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
        io::AsRawFd,
    },
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
const LIMIT: usize = 64 * 1024 * 1024;
static NEXT: AtomicU64 = AtomicU64::new(0);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    U8,
    I64,
    F32,
}
impl Dtype {
    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "uint8",
            Self::I64 => "int64",
            Self::F32 => "float32",
        }
    }
    fn width(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::I64 => 8,
            Self::F32 => 4,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Snapshot(u64, u64, u64, i64, i64, i64, i64);
impl Snapshot {
    fn of(m: &fs::Metadata) -> Self {
        Self(
            m.dev(),
            m.ino(),
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
        )
    }
}
struct Bank {
    name: String,
    file: File,
    snapshot: Snapshot,
    dtype: Dtype,
    type_index: usize,
    hash: Sha256,
    _scratch: Lease,
    _metadata: Lease,
    deleted: bool,
}
pub struct Stage {
    budget: Arc<Budget>,
    scratch: Arc<Budget>,
    control: Control,
    directory: PathBuf,
    dir: File,
    identity: (u64, u64),
    target: usize,
    banks: BudgetVec<Bank>,
    descriptors: BudgetVec<Value>,
    keys: std::collections::BTreeSet<String>,
    metadata: BudgetVec<Lease>,
    closed: bool,
    failed: bool,
    exported: bool,
    _owner: Lease,
}
struct NewDirectory {
    path: PathBuf,
    identity: Option<(u64, u64)>,
    armed: bool,
}
impl Drop for NewDirectory {
    fn drop(&mut self) {
        if self.armed
            && let Some(identity) = self.identity
            && let Ok(actual) = fs::symlink_metadata(&self.path)
            && actual.is_dir()
            && !actual.file_type().is_symlink()
            && (actual.dev(), actual.ino()) == identity
        {
            // rmdir is nonrecursive: unexpected contents are preserved.
            let _ = fs::remove_dir(&self.path);
        }
    }
}
pub struct Banks {
    pub bytes: Vec<BudgetVec<u8>>,
    pub integers: Vec<BudgetVec<i64>>,
    pub floats: Vec<BudgetVec<f32>>,
    pub(crate) _metadata: Lease,
}
fn path_identity(file: &File) -> Result<(u64, u64)> {
    let m = file.metadata()?;
    Ok((m.dev(), m.ino()))
}
fn owned_name(name: &str) -> Result<std::ffi::CString> {
    std::ffi::CString::new(name).map_err(|_| Error::contract("invalid private spool basename"))
}
fn stat_at(dir: &File, name: &str) -> Result<fs::Metadata> {
    let name = owned_name(name)?;
    let fd = unsafe {
        libc::openat(
            dir.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    use std::os::fd::FromRawFd;
    let file = unsafe { File::from_raw_fd(fd) };
    Ok(file.metadata()?)
}
impl Stage {
    pub fn new(
        budget: Arc<Budget>,
        scratch_bytes: usize,
        target: usize,
        control: Control,
    ) -> Result<Self> {
        Self::new_open(budget, scratch_bytes, target, control, |path| {
            OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(path)
        })
    }
    fn new_open(
        budget: Arc<Budget>,
        scratch_bytes: usize,
        target: usize,
        control: Control,
        open: impl FnOnce(&std::path::Path) -> std::io::Result<File>,
    ) -> Result<Self> {
        control.check()?;
        if scratch_bytes == 0 || target == 0 || target > LIMIT {
            return Err(Error::contract("typed spool bounds invalid"));
        }
        let root = std::env::temp_dir();
        let owner = budget.reserve(8192 + root.as_os_str().len() * 4)?;
        for parent in root.ancestors() {
            let m = fs::symlink_metadata(parent)?;
            if !m.is_dir() || m.file_type().is_symlink() {
                return Err(Error::contract("private scratch requires real ancestors"));
            }
        }
        let name = format!(
            "gf-structured-stage-{}-{}-{}",
            std::process::id(),
            crate::budget::monotonic().to_bits(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let directory = root.join(name);
        fs::DirBuilder::new().mode(0o700).create(&directory)?;
        let mut guard = NewDirectory {
            path: directory,
            identity: None,
            armed: true,
        };
        let created = fs::symlink_metadata(&guard.path)?;
        guard.identity = Some((created.dev(), created.ino()));
        let dir = open(&guard.path)?;
        let identity = path_identity(&dir)?;
        if Some(identity) != guard.identity {
            return Err(Error::contract(
                "new private spool directory replaced before ownership",
            ));
        }
        let result = Self {
            banks: BudgetVec::new(&budget),
            descriptors: BudgetVec::new(&budget),
            keys: std::collections::BTreeSet::new(),
            metadata: BudgetVec::new(&budget),
            budget,
            scratch: Budget::new(scratch_bytes),
            control,
            directory: guard.path.clone(),
            dir,
            identity,
            target,
            closed: false,
            failed: false,
            exported: false,
            _owner: owner,
        };
        result.current_directory()?;
        guard.armed = false;
        Ok(result)
    }
    fn work(&self) -> Result<()> {
        self.control.check()?;
        if self.closed || self.failed || self.exported {
            return Err(Error::Closed);
        }
        self.current_directory()
    }
    fn current_directory(&self) -> Result<()> {
        let m = fs::symlink_metadata(&self.directory)?;
        if !m.is_dir()
            || m.file_type().is_symlink()
            || (m.dev(), m.ino()) != self.identity
            || path_identity(&self.dir)? != self.identity
        {
            return Err(Error::contract("private spool directory replaced"));
        }
        Ok(())
    }
    fn current_bank(&self, index: usize) -> Result<()> {
        self.current_directory()?;
        let b = &self.banks[index];
        let actual = stat_at(&self.dir, &b.name)?;
        if !actual.is_file()
            || Snapshot::of(&actual) != b.snapshot
            || Snapshot::of(&b.file.metadata()?) != b.snapshot
        {
            return Err(Error::contract("private spool bank snapshot changed"));
        }
        Ok(())
    }
    fn bank(&mut self, dtype: Dtype, bytes: usize) -> Result<usize> {
        if let Some(index) = self.banks.iter().rposition(|b| b.dtype == dtype) {
            let b = &self.banks[index];
            if b.snapshot.2 as usize + bytes <= self.target {
                return Ok(index);
            }
        }
        let type_index = self.banks.iter().filter(|b| b.dtype == dtype).count();
        let name = format!("{}-{type_index:06}.bin", dtype.name());
        let cname = owned_name(&name)?;
        let metadata = self.budget.reserve(2048 + name.len() * 4)?;
        self.banks.reserve(1)?;
        let fd = unsafe {
            libc::openat(
                self.dir.as_raw_fd(),
                cname.as_ptr(),
                libc::O_RDWR
                    | libc::O_CREAT
                    | libc::O_EXCL
                    | libc::O_NOFOLLOW
                    | libc::O_NONBLOCK
                    | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        use std::os::fd::FromRawFd;
        let file = unsafe { File::from_raw_fd(fd) };
        let snapshot = Snapshot::of(&file.metadata()?);
        let scratch = self.scratch.reserve(0)?;
        self.banks.push(Bank {
            name,
            file,
            snapshot,
            dtype,
            type_index,
            hash: Sha256::new(),
            _scratch: scratch,
            _metadata: metadata,
            deleted: false,
        })?;
        Ok(self.banks.len() - 1)
    }
    fn add_encoded(
        &mut self,
        key: &str,
        dtype: Dtype,
        shape: &[usize],
        bytes: &[u8],
    ) -> Result<()> {
        self.work()?;
        if bytes.len() > LIMIT || shape.len() > 8 || bytes.len() % dtype.width() != 0 {
            return Err(Error::contract("typed array exceeds64MiB/alignment"));
        }
        let count = shape.iter().try_fold(1usize, |a, &b| {
            a.checked_mul(b)
                .ok_or_else(|| Error::resource("typed shape overflow"))
        })?;
        if count.checked_mul(dtype.width()) != Some(bytes.len())
            || key.is_empty()
            || !key
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
            || self.keys.contains(key)
        {
            return Err(Error::contract("typed array key/shape ownership mismatch"));
        }
        let meta = self
            .budget
            .reserve(4096 + key.len() * 4 + shape.len() * 64)?;
        self.descriptors.reserve(1)?;
        self.metadata.reserve(1)?;
        let index = self.bank(dtype, bytes.len())?;
        self.current_bank(index)?;
        let bank = &mut self.banks[index];
        let offset = bank.snapshot.2 as usize;
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| Error::resource("typed bank overflow"))?;
        if end > LIMIT {
            return Err(Error::contract("typed bank exceeds64MiB"));
        }
        bank._scratch.resize(end)?;
        bank.file.seek(SeekFrom::Start(offset as u64))?;
        let result = (|| {
            for block in bytes.chunks(32768) {
                self.control.check()?;
                bank.file.write_all(block)?;
                bank.hash.update(block);
            }
            self.control.check()
        })();
        bank.snapshot = Snapshot::of(&bank.file.metadata()?);
        if let Err(error) = result {
            self.failed = true;
            return Err(error);
        }
        self.current_bank(index)?;
        let bank = &self.banks[index];
        self.descriptors.push(json::object([
            ("key", json::text(key)),
            ("dtype", json::text(dtype.name())),
            (
                "shape",
                json::array(shape.iter().map(|&v| json::integer(v as i128))),
            ),
            ("bank_index", json::integer(bank.type_index as i128)),
            (
                "element_offset",
                json::integer((offset / dtype.width()) as i128),
            ),
            ("element_count", json::integer(count as i128)),
        ]))?;
        self.metadata.push(meta)?;
        self.keys.insert(key.to_owned());
        Ok(())
    }
    pub fn bytes(&mut self, key: &str, bytes: &[u8]) -> Result<()> {
        self.add_encoded(key, Dtype::U8, &[bytes.len()], bytes)
    }
    pub fn array(&mut self, key: &str, dtype: Dtype, value: &Value) -> Result<()> {
        self.work()?;
        fn shape(value: &Value) -> Result<Vec<usize>> {
            match value {
                Value::Array(rows) => {
                    let child = if rows.is_empty() {
                        vec![]
                    } else {
                        shape(&rows[0])?
                    };
                    for row in rows {
                        if shape(row)? != child {
                            return Err(Error::contract("ragged raw token shape"));
                        }
                    }
                    let mut result = vec![rows.len()];
                    result.extend(child);
                    if result.len() > 8 {
                        return Err(Error::contract("raw tensor dimensions"));
                    }
                    Ok(result)
                }
                Value::Number(_) | Value::Bool(_) => Ok(vec![]),
                _ => Err(Error::contract("raw tensor scalar required")),
            }
        }
        let shape = shape(value)?;
        let count = shape.iter().try_fold(1usize, |a, &b| {
            a.checked_mul(b)
                .ok_or_else(|| Error::resource("raw token shape overflow"))
        })?;
        let length = count
            .checked_mul(dtype.width())
            .ok_or_else(|| Error::resource("raw token bytes overflow"))?;
        if length > LIMIT {
            return Err(Error::contract(
                "raw token tensor exceeds64MiB before allocation",
            ));
        }
        let mut encoded = BudgetVec::with_capacity(&self.budget, length)?;
        fn append(
            value: &Value,
            dtype: Dtype,
            out: &mut BudgetVec<u8>,
            control: &Control,
        ) -> Result<()> {
            control.check()?;
            if let Value::Array(rows) = value {
                for row in rows {
                    append(row, dtype, out, control)?;
                }
                return Ok(());
            }
            match dtype {
                Dtype::U8 => {
                    let v = if let Value::Bool(v) = value {
                        u8::from(*v)
                    } else {
                        value.integer(0, 255)? as u8
                    };
                    out.push(v)?;
                }
                Dtype::I64 => out.extend_from_slice(
                    &(value.integer(i64::MIN as i128, i64::MAX as i128)? as i64).to_le_bytes(),
                )?,
                Dtype::F32 => {
                    let v = value.number()? as f32;
                    if !v.is_finite() {
                        return Err(Error::contract("nonfinite/overflow FP32 policy statistic"));
                    }
                    out.extend_from_slice(&v.to_le_bytes())?;
                }
            }
            Ok(())
        }
        append(value, dtype, &mut encoded, &self.control)?;
        self.add_encoded(key, dtype, &shape, &encoded)
    }
    pub fn descriptors(&self) -> Result<Value> {
        self.control.check()?;
        json::copy(
            &json::array(self.descriptors.iter().cloned()),
            &self.control,
        )
    }
    pub fn decoded_bytes(&self) -> usize {
        self.banks.iter().map(|b| b.snapshot.2 as usize).sum()
    }
    pub fn scratch_bytes(&self) -> usize {
        self.scratch.used()
    }
    pub fn export(&mut self) -> Result<Banks> {
        self.work()?;
        let metadata = self.budget.reserve(1024 * (1 + self.banks.len()))?;
        let mut escrow = self.budget.reserve(self.decoded_bytes())?;
        let _scratch = self.budget.reserve(32768)?;
        // The full cohort is admitted before the first buffer. Transfer its
        // reservation into each actual BudgetVec instead of charging it twice.
        self.exported = true;
        let mut result = Banks {
            bytes: Vec::new(),
            integers: Vec::new(),
            floats: Vec::new(),
            _metadata: metadata,
        };
        struct Hashed<'a> {
            file: &'a mut File,
            hash: Sha256,
        }
        impl Read for Hashed<'_> {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                let n = self.file.read(out)?;
                self.hash.update(&out[..n]);
                Ok(n)
            }
        }
        for index in 0..self.banks.len() {
            self.control.check()?;
            self.current_bank(index)?;
            let bank = &mut self.banks[index];
            let length = bank.snapshot.2 as usize;
            bank.file.seek(SeekFrom::Start(0))?;
            escrow.resize(escrow.bytes() - length)?;
            let mut source = Hashed {
                file: &mut bank.file,
                hash: Sha256::new(),
            };
            match bank.dtype {
                Dtype::U8 => {
                    let mut values = BudgetVec::with_capacity(&self.budget, length)?;
                    values.resize(length, 0)?;
                    for block in values.chunks_mut(32768) {
                        self.control.check()?;
                        source.read_exact(block)?;
                    }
                    result.bytes.push(values);
                }
                Dtype::I64 => {
                    let mut values = BudgetVec::new(&self.budget);
                    crate::numeric_io::read_i64(
                        &mut source,
                        &mut values,
                        length / 8,
                        &self.control,
                    )?;
                    result.integers.push(values);
                }
                Dtype::F32 => {
                    let mut values = BudgetVec::new(&self.budget);
                    crate::numeric_io::read_f32(
                        &mut source,
                        &mut values,
                        length / 4,
                        &self.control,
                    )?;
                    result.floats.push(values);
                }
            }
            if source.hash.finalize()[..] != bank.hash.clone().finalize()[..] {
                return Err(Error::contract("typed spool bank hash changed"));
            }
            self.current_bank(index)?;
        }
        drop(escrow);
        Ok(result)
    }
    pub fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.current_directory()?;
        let expected = self
            .banks
            .iter()
            .filter(|b| !b.deleted)
            .map(|b| b.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            let name = entry.file_name();
            if !expected.contains(name.to_str().unwrap_or("")) {
                return Err(Error::contract("foreign private spool evidence preserved"));
            }
        }
        for index in 0..self.banks.len() {
            self.current_directory()?;
            if self.banks[index].deleted {
                continue;
            }
            let bank = &self.banks[index];
            let meta = stat_at(&self.dir, &bank.name)?;
            let held = bank.file.metadata()?;
            if !meta.is_file() || (meta.dev(), meta.ino()) != (held.dev(), held.ino()) {
                return Err(Error::contract(
                    "foreign private spool replacement preserved",
                ));
            }
            let name = owned_name(&bank.name)?;
            if unsafe { libc::unlinkat(self.dir.as_raw_fd(), name.as_ptr(), 0) } != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            self.banks[index].deleted = true;
        }
        self.current_directory()?;
        fs::remove_dir(&self.directory)?;
        self.closed = true;
        self.banks.clear();
        self.descriptors.clear();
        self.metadata.clear();
        self.keys.clear();
        Ok(())
    }
    pub(crate) fn export_retiring_disk(&mut self) -> Result<Banks> {
        let banks = self.export()?;
        self.close()?;
        Ok(banks)
    }
}
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn stage(memory: usize, scratch: usize, target: usize) -> (Arc<Budget>, Stage) {
        let budget = Budget::new(memory);
        let stage = Stage::new(
            budget.clone(),
            scratch,
            target,
            Control::new(crate::budget::monotonic() + 30.0).unwrap(),
        )
        .unwrap();
        (budget, stage)
    }
    #[test]
    fn directory_open_failure_cleans_owned_creation_but_preserves_replacement() {
        let budget = Budget::new(1024 * 1024);
        let control = Control::new(crate::budget::monotonic() + 30.0).unwrap();
        let mut captured = None;
        let error = Stage::new_open(budget.clone(), 1024, 64, control.clone(), |path| {
            captured = Some(path.to_owned());
            Err(std::io::Error::from_raw_os_error(libc::EMFILE))
        })
        .err()
        .unwrap();
        assert!(matches!(error,Error::Io(message) if message.contains("open files")));
        assert!(!captured.unwrap().exists());
        assert_eq!(budget.used(), 0);
        let mut paths = None;
        assert!(
            Stage::new_open(budget.clone(), 1024, 64, control, |path| {
                let original = path.with_extension("original");
                fs::rename(path, &original)?;
                fs::create_dir(path)?;
                paths = Some((path.to_owned(), original));
                Err(std::io::Error::from_raw_os_error(libc::EMFILE))
            })
            .is_err()
        );
        let (foreign, original) = paths.unwrap();
        assert!(foreign.is_dir() && original.is_dir());
        fs::remove_dir(foreign).unwrap();
        fs::remove_dir(original).unwrap();
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn typed_banks_preserve_widths_offsets_empty_shapes_and_negative_zero() {
        let (budget, mut stage) = stage(1024 * 1024, 1024 * 1024, 8);
        stage.bytes("e.0.payload", b"abc").unwrap();
        stage
            .array("e.0.empty", Dtype::I64, &json::array([]))
            .unwrap();
        stage
            .array(
                "e.0.integer",
                Dtype::I64,
                &json::array([
                    json::integer(i64::MIN as i128),
                    json::integer(i64::MAX as i128),
                ]),
            )
            .unwrap();
        stage
            .array(
                "e.0.value",
                Dtype::F32,
                &json::array([json::float(-0.0).unwrap(), json::float(1.25).unwrap()]),
            )
            .unwrap();
        let meta = stage.descriptors().unwrap();
        assert_eq!(
            meta.array().unwrap()[1].require("shape").unwrap(),
            &json::array([json::integer(0)])
        );
        assert_eq!(
            meta.array().unwrap()[2]
                .require("dtype")
                .unwrap()
                .string()
                .unwrap(),
            "int64"
        );
        assert_eq!(stage.decoded_bytes(), 27);
        let banks = stage.export().unwrap();
        assert_eq!(&*banks.bytes[0], b"abc");
        assert!(banks.integers[0].is_empty());
        assert_eq!(&*banks.integers[1], &[i64::MIN, i64::MAX]);
        assert_eq!(banks.floats[0][0].to_bits(), 0x80000000);
        assert_eq!(banks.floats[0][1].to_bits(), 0x3fa00000);
        assert!(stage.export().is_err());
        let directory = stage.directory.clone();
        stage.close().unwrap();
        drop(stage);
        assert!(!directory.exists());
        assert!(budget.used() >= 27);
        drop(banks);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn scratch_and_float_overflow_fail_before_array_publication() {
        let (_budget, mut stage) = stage(1024 * 1024, 3, 64);
        assert!(matches!(
            stage.bytes("too_big", b"1234"),
            Err(Error::ResourceLimit(_))
        ));
        assert_eq!(stage.scratch_bytes(), 0);
        assert!(stage.descriptors().unwrap().array().unwrap().is_empty());
        assert!(
            stage
                .array(
                    "overflow",
                    Dtype::F32,
                    &json::array([json::float(f64::MAX).unwrap()])
                )
                .is_err()
        );
        assert!(stage.descriptors().unwrap().array().unwrap().is_empty());
        stage.close().unwrap();
    }
    #[test]
    fn exported_whole_cohort_is_admitted_before_buffers() {
        let (budget, mut stage) = stage(1024 * 1024, 1024 * 1024, 65536);
        stage.bytes("raw", &[1; 65536]).unwrap();
        let held = budget
            .reserve(budget.limit() - budget.used() - 65535)
            .unwrap();
        let before = budget.used();
        assert!(matches!(stage.export(), Err(Error::ResourceLimit(_))));
        assert_eq!(budget.used(), before);
        drop(held);
        stage.close().unwrap();
    }
    #[test]
    fn disk_retirement_precedes_exposure_but_ram_banks_remain_owned() {
        let (budget, mut stage) = stage(1024 * 1024, 12, 64);
        stage.bytes("raw", b"owned").unwrap();
        stage
            .array(
                "floats",
                Dtype::F32,
                &json::array([json::float(-0.0).unwrap()]),
            )
            .unwrap();
        let directory = stage.directory.clone();
        let banks = stage.export_retiring_disk().unwrap();
        assert!(!directory.exists());
        assert_eq!(stage.scratch_bytes(), 0);
        assert_eq!(&*banks.bytes[0], b"owned");
        assert_eq!(banks.floats[0][0].to_bits(), 0x80000000);
        stage.close().unwrap(); // later logical retirement remains idempotent
        drop(stage);
        assert!(budget.used() >= 9);
        drop(banks);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn disk_retirement_failure_returns_no_payload_and_keeps_retry_owner() {
        let (budget, mut stage) = stage(1024 * 1024, 1024, 64);
        stage.bytes("raw", b"owned").unwrap();
        let foreign = stage.directory.join("foreign");
        fs::write(&foreign, b"preserve").unwrap();
        let before = budget.used();
        assert!(stage.export_retiring_disk().is_err());
        assert_eq!(budget.used(), before); // copied buffers dropped, spool still owned
        assert_eq!(stage.scratch_bytes(), 5);
        assert!(!stage.closed);
        assert_eq!(fs::read(&foreign).unwrap(), b"preserve");
        fs::remove_file(&foreign).unwrap();
        stage.close().unwrap();
        drop(stage);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn foreign_private_evidence_and_same_bytes_replacement_are_preserved() {
        let (_budget, mut stage) = stage(1024 * 1024, 1024 * 1024, 65536);
        stage.bytes("raw", b"original").unwrap();
        let foreign = stage.directory.join("foreign");
        fs::write(&foreign, b"evidence").unwrap();
        assert!(stage.close().is_err());
        assert_eq!(fs::read(&foreign).unwrap(), b"evidence");
        fs::remove_file(foreign).unwrap();
        let path = stage.directory.join(&stage.banks[0].name);
        let moved = stage.directory.with_extension("held-original");
        fs::rename(&path, &moved).unwrap();
        fs::write(&path, b"original").unwrap();
        assert!(stage.export().is_err());
        assert!(stage.close().is_err());
        assert_eq!(fs::read(&path).unwrap(), b"original");
        fs::remove_file(&path).unwrap();
        fs::rename(&moved, &path).unwrap();
        stage.close().unwrap();
    }
    #[test]
    fn cancellation_keeps_cleanup_owned_and_joins_without_exposure() {
        let (budget, mut stage) = stage(1024 * 1024, 1024 * 1024, 65536);
        stage.bytes("raw", b"owned").unwrap();
        stage.control.cancel();
        assert_eq!(stage.export().err(), Some(Error::Cancelled));
        stage.close().unwrap();
        drop(stage);
        assert_eq!(budget.used(), 0);
    }
}

impl Stage {
    pub(crate) fn frame(
        &mut self,
        episode_index: usize,
        position: usize,
        frame: &Value,
        table: &Value,
        events: &[Value],
        recipe: &crate::structured_tokens::Recipe,
    ) -> Result<Value> {
        self.work()?;
        if table.require("synthetic_capacity")?.boolean()?
            || table.require("purpose")?.string()? != "component"
        {
            return Err(Error::contract(
                "native gameplay payload rejects synthetic capacity",
            ));
        }
        let fp = format!("e.{episode_index}.f.{position}");
        let ep = format!("e.{episode_index}.event-tokens.{position}");
        let payload = json::canonical(frame.require("payload")?, &self.budget, &self.control)?;
        self.bytes(&(fp.clone() + ".payload"), &payload)?;
        drop(payload);
        let tokens = crate::structured_tokens::frame_tokens(
            frame,
            events,
            recipe,
            &self.budget,
            &self.control,
        )?;
        for name in [
            "state_numerics",
            "state_present",
            "state_categories",
            "state_category_present",
            "card_numerics",
            "card_present",
            "card_categories",
            "card_mask",
            "opponent_learned",
            "history",
        ] {
            self.array(
                &(fp.clone() + "." + name),
                if name.contains("present") || name.contains("mask") {
                    Dtype::U8
                } else {
                    Dtype::I64
                },
                tokens.value.require(name)?,
            )?;
        }
        for name in [
            "interval_numerics",
            "interval_present",
            "interval_categories",
        ] {
            self.array(
                &(ep.clone() + "." + name),
                if name.contains("present") {
                    Dtype::U8
                } else {
                    Dtype::I64
                },
                tokens.value.require(name)?,
            )?;
        }
        drop(tokens);
        let candidates =
            crate::structured_tokens::candidate_tokens(table, recipe, &self.budget, &self.control)?;
        for name in [
            "kinds", "targets", "numerics", "present", "offsets", "slots",
        ] {
            self.array(
                &(fp.clone() + ".candidate." + name),
                if name == "present" {
                    Dtype::U8
                } else {
                    Dtype::I64
                },
                candidates.value.require(name)?,
            )?;
        }
        let mut result = json::object([]);
        for key in [
            "actor",
            "episode",
            "revision",
            "interval_start",
            "boundary",
            "interval_start_hash",
            "frame_hash",
        ] {
            result
                .as_object_mut()
                .unwrap()
                .insert(key.into(), json::copy(frame.require(key)?, &self.control)?);
        }
        for key in [
            "action_contract",
            "purpose",
            "synthetic_capacity",
            "table_hash",
        ] {
            result
                .as_object_mut()
                .unwrap()
                .insert(key.into(), json::copy(table.require(key)?, &self.control)?);
        }
        for (key, value) in [
            ("payload_key", fp.clone() + ".payload"),
            ("token_prefix", fp),
            ("event_token_prefix", ep),
        ] {
            result
                .as_object_mut()
                .unwrap()
                .insert(key.into(), json::text(&value));
        }
        Ok(result)
    }
    pub(crate) fn finish_episode_arrays(
        &mut self,
        index: usize,
        events: &[Value],
        selections: &[Value],
    ) -> Result<Value> {
        self.work()?;
        let prefix = format!("e.{index}");
        let mut blocks = Vec::new();
        for start in (0..events.len()).step_by(1024) {
            self.control.check()?;
            let end = (start + 1024).min(events.len());
            let key = format!("{prefix}.raw-events.{start}");
            let mut raw_size = 2usize;
            for (index, event) in events[start..end].iter().enumerate() {
                self.control.check()?;
                raw_size = raw_size
                    .checked_add(
                        json::canonical(event, &self.budget, &self.control)?.len()
                            + usize::from(index > 0),
                    )
                    .ok_or_else(|| Error::resource("raw event block size overflow"))?;
            }
            if raw_size > LIMIT {
                return Err(Error::contract("raw event block exceeds64MiB before copy"));
            }
            let lease = json::admit(raw_size, &self.budget)?;
            let values = json::array(
                events[start..end]
                    .iter()
                    .map(|v| json::copy(v, &self.control))
                    .collect::<Result<Vec<_>>>()?,
            );
            let bytes = json::canonical(&values, &self.budget, &self.control)?;
            self.bytes(&key, &bytes)?;
            drop(bytes);
            drop(values);
            drop(lease);
            blocks.push(json::object([
                ("start", json::integer(start as i128)),
                ("end", json::integer(end as i128)),
                ("key", json::text(&key)),
            ]));
        }
        for (name, field, dtype) in [
            ("selected", "index", Dtype::I64),
            ("old_logprob", "logprob", Dtype::F32),
            ("old_value", "value", Dtype::F32),
        ] {
            let values = json::array(
                selections
                    .iter()
                    .map(|v| json::copy(v.require(field)?, &self.control))
                    .collect::<Result<Vec<_>>>()?,
            );
            self.array(&(prefix.clone() + "." + name), dtype, &values)?;
        }
        Ok(json::array(blocks))
    }
}
