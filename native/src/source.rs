//! Bounded immutable reads; metadata and digest remain owned through publication.
use crate::budget::{Budget, BudgetVec, Control, JsonAdmission, Lease};
use crate::error::{Error, Result};
use crate::json::{self, Document};
use crate::telemetry::{Phase, Telemetry};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File, Metadata};
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    device: u64,
    inode: u64,
    size: u64,
    mtime: i64,
    mtime_ns: i64,
    ctime: i64,
    ctime_ns: i64,
}
impl Identity {
    pub fn of(m: &Metadata) -> Self {
        Self {
            device: m.dev(),
            inode: m.ino(),
            size: m.len(),
            mtime: m.mtime(),
            mtime_ns: m.mtime_nsec(),
            ctime: m.ctime(),
            ctime_ns: m.ctime_nsec(),
        }
    }
}
pub struct Evidence {
    pub path: PathBuf,
    pub identity: Identity,
    pub digest: String,
    _lease: Lease,
}
#[derive(Clone, Copy)]
pub enum InventoryKind {
    JsonFiles,
    ArtifactLeaves,
    ParticipantJournals,
    ParticipantReplays,
}
struct InventoryPath {
    path: PathBuf,
    _lease: Lease,
}
struct Inventory {
    directory: PathBuf,
    kind: InventoryKind,
    paths: BudgetVec<InventoryPath>,
    _lease: Lease,
}
pub struct Sources {
    pub entries: BudgetVec<Evidence>,
    budget: Arc<Budget>,
    max_file: usize,
    max_record: usize,
    inventories: BudgetVec<Inventory>,
    lookup: HashMap<PathBuf, usize>,
    telemetry: Arc<Telemetry>,
}
/// Worker-owned bytes/admission stay stable while Journal DTOs borrow them.
/// DTOs cannot outlive this external raw owner.
pub struct RawJournal {
    bytes: BudgetVec<u8>,
    admission: Arc<JsonAdmission>,
    max_record: usize,
}
impl RawJournal {
    pub fn parse<'a>(
        &'a self,
        control: &Control,
        telemetry: &Telemetry,
    ) -> Result<crate::typed_journal::journal::Records<'a>> {
        let _timer = telemetry.phase_timer(Phase::SourceDecode);
        let mut kind =
            telemetry.decode_timer(crate::telemetry::DecodeKind::Journal, self.bytes.len());
        let records = crate::typed_journal::journal::parse_admitted(
            &self.bytes,
            self.max_record,
            &self.admission,
            control,
        )?;
        telemetry.typed_journal(records.records.len());
        kind.success();
        Ok(records)
    }
}

impl Sources {
    pub fn new(budget: Arc<Budget>, max_file: usize, max_record: usize) -> Self {
        Self::with_telemetry(
            budget,
            max_file,
            max_record,
            &Arc::new(Telemetry::default()),
        )
    }
    pub fn with_telemetry(
        budget: Arc<Budget>,
        max_file: usize,
        max_record: usize,
        telemetry: &Arc<Telemetry>,
    ) -> Self {
        Self {
            entries: BudgetVec::new(&budget),
            inventories: BudgetVec::new(&budget),
            lookup: HashMap::new(),
            budget,
            max_file,
            max_record,
            telemetry: telemetry.clone(),
        }
    }
    pub fn telemetry(&self) -> &Arc<Telemetry> {
        &self.telemetry
    }
    pub fn read(&mut self, path: &Path, control: &Control) -> Result<BudgetVec<u8>> {
        // Includes bounded metadata/read/hash/evidence work once per file attempt.
        let _timer = self.telemetry.phase_timer(Phase::SourceReadHash);
        control.check()?;
        let path = fs::canonicalize(path)?;
        let before = Identity::of(&fs::metadata(&path)?);
        let length =
            usize::try_from(before.size).map_err(|_| Error::resource("source size overflow"))?;
        if length > self.max_file {
            return Err(Error::resource("source exceeds max_source_file_mib"));
        }
        let mut input = File::open(&path)?;
        if Identity::of(&input.metadata()?) != before {
            return Err(Error::contract("source replaced before bounded read"));
        }
        let mut bytes = BudgetVec::with_capacity(
            &self.budget,
            length
                .checked_add(1)
                .ok_or_else(|| Error::resource("source size overflow"))?,
        )?;
        let mut hasher = Sha256::new();
        let mut block = [0u8; 32768];
        loop {
            control.check()?;
            let left = length.saturating_sub(bytes.len()) + 1;
            let count = input.read(&mut block[..left.min(32768)])?;
            if count == 0 {
                break;
            }
            bytes.extend_from_slice(&block[..count])?;
            hasher.update(&block[..count]);
            if bytes.len() > length {
                return Err(Error::contract("source grew during bounded read"));
            }
        }
        if bytes.len() != length
            || Identity::of(&input.metadata()?) != before
            || Identity::of(&fs::metadata(&path)?) != before
        {
            return Err(Error::contract("source changed during bounded read"));
        }
        let digest = format!("{:x}", hasher.finalize());
        if let Some(index) = self.lookup.get(&path).copied() {
            let prior = &self.entries[index];
            if prior.identity != before || prior.digest != digest {
                return Err(Error::contract("source changed during preparation"));
            }
        } else {
            // Path and digest buffers are charged before creating retained copies.
            let lease = self.budget.reserve(path.as_os_str().len() + 64 + 256)?;
            self.lookup.insert(path.clone(), self.entries.len());
            self.entries.push(Evidence {
                path,
                identity: before,
                digest,
                _lease: lease,
            })?;
        }
        control.check()?;
        Ok(bytes)
    }
    pub fn json(&mut self, path: &Path, replay: bool, control: &Control) -> Result<Document> {
        let bytes = self.read(path, control)?;
        let _timer = self.telemetry.phase_timer(Phase::SourceDecode);
        let mut kind = self.telemetry.decode_timer(
            if replay {
                crate::telemetry::DecodeKind::Replay
            } else {
                crate::telemetry::DecodeKind::Json
            },
            bytes.len(),
        );
        let admission = json::admit(bytes.len(), &self.budget)?;
        let result = if replay {
            parse_replay(&bytes, &admission, control)
        } else {
            json::parse_admitted(&bytes, &admission, control)
        };
        if result.is_ok() {
            kind.success();
        }
        result
    }
    pub fn journal(&mut self, path: &Path, control: &Control) -> Result<Vec<Document>> {
        let bytes = self.read(path, control)?;
        let _timer = self.telemetry.phase_timer(Phase::SourceDecode);
        let mut kind = self
            .telemetry
            .decode_timer(crate::telemetry::DecodeKind::Journal, bytes.len());
        let admission = json::admit(bytes.len(), &self.budget)?;
        std::str::from_utf8(&bytes).map_err(|_| Error::contract("invalid UTF-8 journal"))?;
        let mut records = Vec::new();
        let mut start = 0;
        while start < bytes.len() {
            control.check()?;
            let mut end = start;
            while end < bytes.len() && bytes[end] != b'\r' && bytes[end] != b'\n' {
                end += 1;
            }
            if end - start > self.max_record {
                return Err(Error::resource("journal exceeds max_record_mib"));
            }
            records.push(json::parse_admitted(
                &bytes[start..end],
                &admission,
                control,
            )?);
            start = end;
            if bytes.get(start) == Some(&b'\r') {
                start += 1;
                if bytes.get(start) == Some(&b'\n') {
                    start += 1;
                }
            } else if bytes.get(start) == Some(&b'\n') {
                start += 1;
            }
        }
        control.check()?;
        kind.success();
        self.telemetry.legacy_journal();
        Ok(records)
    }
    pub fn raw_journal(&mut self, path: &Path, control: &Control) -> Result<RawJournal> {
        let bytes = self.read(path, control)?;
        let admission = json::admit(bytes.len(), &self.budget)?;
        Ok(RawJournal {
            bytes,
            admission,
            max_record: self.max_record,
        })
    }
    pub fn verify(&self, control: &Control) -> Result<()> {
        for entry in self.entries.iter() {
            control.check()?;
            if Identity::of(&fs::metadata(&entry.path)?) != entry.identity {
                return Err(Error::contract("source invalidated before publication"));
            }
            if entry.path.file_name().is_some_and(|name| {
                ["run.json", "latest.json", "artifact-registry.json"]
                    .iter()
                    .any(|expected| name == *expected)
            }) {
                let _timer = self.telemetry.phase_timer(Phase::SourceReadHash);
                let mut file = File::open(&entry.path)?;
                let mut hasher = Sha256::new();
                let mut bytes = [0u8; 32768];
                loop {
                    control.check()?;
                    let count = file.read(&mut bytes)?;
                    if count == 0 {
                        break;
                    }
                    hasher.update(&bytes[..count]);
                }
                if format!("{:x}", hasher.finalize()) != entry.digest {
                    return Err(Error::contract(
                        "mutable parent control changed before publication",
                    ));
                }
            }
        }
        for inventory in self.inventories.iter() {
            let actual =
                inventory_paths(&inventory.directory, inventory.kind, &self.budget, control)?;
            if actual.len() != inventory.paths.len()
                || actual
                    .iter()
                    .zip(inventory.paths.iter())
                    .any(|(a, b)| a.path != b.path)
            {
                return Err(Error::contract(
                    "source inventory changed before publication",
                ));
            }
        }
        Ok(())
    }
    pub fn inventory(
        &mut self,
        directory: &Path,
        kind: InventoryKind,
        control: &Control,
    ) -> Result<()> {
        let directory = fs::canonicalize(directory)?;
        let lease = self.budget.reserve(directory.as_os_str().len() + 128)?;
        let paths = inventory_paths(&directory, kind, &self.budget, control)?;
        self.inventories.push(Inventory {
            directory,
            kind,
            paths,
            _lease: lease,
        })
    }
    pub fn digest(&self, path: &Path, control: &Control) -> Result<&str> {
        control.check()?;
        let path = fs::canonicalize(path)?;
        self.lookup
            .get(&path)
            .map(|index| self.entries[*index].digest.as_str())
            .ok_or_else(|| Error::contract("source was not read"))
    }
    pub fn checked_digest(&self, path: &Path, control: &Control) -> Result<Option<&str>> {
        control.check()?;
        let path = fs::canonicalize(path)?;
        if let Some(index) = self.lookup.get(&path) {
            let entry = &self.entries[*index];
            if Identity::of(&fs::metadata(&path)?) != entry.identity {
                return Err(Error::contract(
                    "source identity changed during preparation",
                ));
            }
            return Ok(Some(&entry.digest));
        }
        Ok(None)
    }
    pub fn merge(&mut self, other: Sources, control: &Control) -> Result<()> {
        let (entries, _capacity) = other.entries.into_parts();
        for entry in entries {
            control.check()?;
            if let Some(index) = self.lookup.get(&entry.path).copied() {
                let previous = &self.entries[index];
                if previous.identity != entry.identity || previous.digest != entry.digest {
                    return Err(Error::contract(
                        "source changed during parallel preparation",
                    ));
                }
            } else {
                self.entries.reserve(1)?;
                self.lookup.insert(entry.path.clone(), self.entries.len());
                self.entries.push(entry)?;
            }
        }
        Ok(())
    }
}
fn inventory_paths(
    directory: &Path,
    kind: InventoryKind,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<BudgetVec<InventoryPath>> {
    let mut paths = BudgetVec::new(budget);
    let mut add = |path: PathBuf| -> Result<()> {
        control.check()?;
        let path = fs::canonicalize(&path).unwrap_or(path);
        let lease = budget.reserve(path.as_os_str().len() + 128)?;
        paths.push(InventoryPath {
            path,
            _lease: lease,
        })
    };
    for entry in fs::read_dir(directory)? {
        control.check()?;
        let entry = entry?;
        if matches!(kind, InventoryKind::JsonFiles) {
            if entry.path().extension().is_some_and(|ext| ext == "json") {
                add(entry.path())?;
            }
            continue;
        }
        if !entry.path().is_dir() {
            continue;
        }
        for child in fs::read_dir(entry.path())? {
            control.check()?;
            let child = child?;
            let name = child.file_name();
            let name = name.as_encoded_bytes();
            let included = match kind {
                InventoryKind::ArtifactLeaves => true,
                InventoryKind::ParticipantJournals => {
                    name.starts_with(b"player-") && name.ends_with(b".jsonl")
                }
                InventoryKind::ParticipantReplays => {
                    name.starts_with(b"replay-player-") && name.ends_with(b".json")
                }
                InventoryKind::JsonFiles => false,
            };
            if included {
                add(child.path())?;
            }
        }
    }
    crate::budget::sort_unstable_by(
        &mut paths,
        |a, b| {
            a.path
                .as_os_str()
                .as_encoded_bytes()
                .cmp(b.path.as_os_str().as_encoded_bytes())
        },
        control,
    )?;
    Ok(paths)
}

fn parse_replay(
    bytes: &[u8],
    admission: &Arc<crate::budget::JsonAdmission>,
    control: &Control,
) -> Result<Document> {
    control.check()?;
    if let Some(bytes) = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]) {
        return json::parse_admitted(bytes, admission, control);
    }
    let (unit, little, skip) = if bytes.starts_with(&[0xff, 0xfe, 0, 0]) {
        (4, true, 4)
    } else if bytes.starts_with(&[0, 0, 0xfe, 0xff]) {
        (4, false, 4)
    } else if bytes.starts_with(&[0xff, 0xfe]) {
        (2, true, 2)
    } else if bytes.starts_with(&[0xfe, 0xff]) {
        (2, false, 2)
    } else if bytes.len() >= 4 && bytes[..3] == [0, 0, 0] {
        (4, false, 0)
    } else if bytes.len() >= 4 && bytes[1..4] == [0, 0, 0] {
        (4, true, 0)
    } else if bytes.len() >= 2 && bytes[0] == 0 {
        (2, false, 0)
    } else if bytes.len() >= 2 && bytes[1] == 0 {
        (2, true, 0)
    } else {
        return json::parse_admitted(bytes, admission, control);
    };
    if (bytes.len() - skip) % unit != 0 {
        return Err(Error::contract("truncated Unicode source"));
    }
    let decoded = if unit == 2 {
        let units = bytes[skip..]
            .chunks_exact(2)
            .map(|b| {
                if little {
                    u16::from_le_bytes([b[0], b[1]])
                } else {
                    u16::from_be_bytes([b[0], b[1]])
                }
            })
            .collect::<Vec<_>>();
        String::from_utf16(&units).map_err(|e| Error::contract(e.to_string()))?
    } else {
        let mut text = String::new();
        for (index, b) in bytes[skip..].chunks_exact(4).enumerate() {
            if index % 4096 == 0 {
                control.check()?;
            }
            let value = if little {
                u32::from_le_bytes(b.try_into().expect("four bytes"))
            } else {
                u32::from_be_bytes(b.try_into().expect("four bytes"))
            };
            text.push(
                char::from_u32(value)
                    .ok_or_else(|| Error::contract("invalid Unicode codepoint"))?,
            );
        }
        text
    };
    control.check()?;
    json::parse_admitted(decoded.as_bytes(), admission, control)
}
