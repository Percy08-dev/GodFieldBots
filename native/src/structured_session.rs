//! Additive diagnostic ownership for original structured parent controls.
//! No Journal5, ZIP/NPY, authorization or numerical export claims.
use crate::{
    budget::{Budget, BudgetVec, Control, Lease},
    error::{Error, Result},
    json::{self, Document, Value, ValueExt},
};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::Read,
    os::unix::{fs::MetadataExt, fs::OpenOptionsExt},
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity(u64, u64, u64, i64, i64, i64, i64);
impl Identity {
    fn of(value: &fs::Metadata) -> Self {
        Self(
            value.dev(),
            value.ino(),
            value.len(),
            value.mtime(),
            value.mtime_nsec(),
            value.ctime(),
            value.ctime_nsec(),
        )
    }
}
struct Evidence {
    path: PathBuf,
    identity: Identity,
    digest: String,
    _lease: Lease,
}
struct Inventory {
    path: PathBuf,
    names: Vec<String>,
    identity: Identity,
    directories: bool,
    pattern: Option<SourcePattern>,
    append: Option<LiveAppend>,
    _lease: Lease,
}
struct LiveAppend {
    name: String,
    observed: std::cell::RefCell<Option<Identity>>,
    _lease: Lease,
}
#[derive(Clone, Copy)]
pub(crate) enum SourcePattern {
    Journals,
    Partials,
    Columns,
    Replays,
    Results,
}
impl SourcePattern {
    fn matches(self, name: &str) -> bool {
        match self {
            Self::Journals => name.starts_with("player-") && name.ends_with(".jsonl"),
            Self::Partials => name.starts_with("player-") && name.ends_with(".partial.jsonl"),
            Self::Columns => name.ends_with(".columns"),
            Self::Replays => name.starts_with("replay-player-") && name.ends_with(".json"),
            Self::Results => name == "result.json",
        }
    }
}

pub struct Sources {
    pub budget: Arc<Budget>,
    pub control: Control,
    pub root: PathBuf,
    max_json: usize,
    root_identity: Identity,
    files: BudgetVec<Evidence>,
    inventories: BudgetVec<Inventory>,
}
impl Sources {
    pub fn new(
        directory: &Path,
        budget: Arc<Budget>,
        control: Control,
        max_json: usize,
    ) -> Result<Self> {
        control.check()?;
        let directory = std::path::absolute(directory)?;
        for path in directory.ancestors() {
            let info = fs::symlink_metadata(path)?;
            if !info.is_dir() || info.file_type().is_symlink() {
                return Err(Error::contract(
                    "structured parent requires nonsymlink directory ancestors",
                ));
            }
        }
        let root_identity = Identity::of(&fs::symlink_metadata(&directory)?);
        Ok(Self {
            root: fs::canonicalize(directory)?,
            root_identity,
            files: BudgetVec::new(&budget),
            inventories: BudgetVec::new(&budget),
            budget,
            control,
            max_json,
        })
    }
    fn safe(&self, path: &Path) -> Result<()> {
        self.control.check()?;
        if Identity::of(&fs::symlink_metadata(&self.root)?) != self.root_identity {
            return Err(Error::contract(
                "structured original root directory changed",
            ));
        }
        if !path.starts_with(&self.root) {
            return Err(Error::contract("foreign structured parent path"));
        }
        for parent in path
            .parent()
            .ok_or_else(|| Error::contract("parent required"))?
            .ancestors()
        {
            if !fs::symlink_metadata(parent)?.is_dir() {
                return Err(Error::contract(
                    "structured path ancestor is not a real directory",
                ));
            }
            if parent == self.root {
                break;
            }
        }
        Ok(())
    }
    fn open(&self, path: &Path) -> Result<(File, Identity)> {
        self.safe(path)?;
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)?;
        let info = file.metadata()?;
        if !info.is_file() {
            return Err(Error::contract("structured control input must be regular"));
        }
        let identity = Identity::of(&info);
        self.current(path, &file, &identity)?;
        Ok((file, identity))
    }
    fn current(&self, path: &Path, file: &File, original: &Identity) -> Result<()> {
        self.control.check()?;
        let path_info = fs::symlink_metadata(path)?;
        if !path_info.is_file()
            || Identity::of(&file.metadata()?) != *original
            || Identity::of(&path_info) != *original
        {
            return Err(Error::contract(
                "structured control original FD/path snapshot changed",
            ));
        }
        Ok(())
    }
    fn remember(&mut self, path: &Path, identity: Identity, digest: String) -> Result<()> {
        if let Some(old) = self.files.iter().find(|item| item.path == path) {
            if old.identity != identity || old.digest != digest {
                return Err(Error::contract(
                    "structured control input changed; no fresh baseline",
                ));
            }
            return Ok(());
        }
        let lease = self.budget.reserve(1024 + path.as_os_str().len() * 4)?;
        self.files.push(Evidence {
            path: path.into(),
            identity,
            digest,
            _lease: lease,
        })
    }
    pub fn hash(&mut self, path: &Path) -> Result<String> {
        let (mut file, identity) = self.open(path)?;
        let _scratch = self.budget.reserve(32768)?;
        let mut hasher = Sha256::new();
        let mut block = [0u8; 32768];
        loop {
            self.control.check()?;
            let count = file.read(&mut block)?;
            if count == 0 {
                break;
            }
            hasher.update(&block[..count]);
        }
        self.current(path, &file, &identity)?;
        let digest = format!("{:x}", hasher.finalize());
        self.remember(path, identity, digest.clone())?;
        Ok(digest)
    }
    pub fn bytes(&mut self, path: &Path, max_bytes: usize) -> Result<BudgetVec<u8>> {
        let (mut file, identity) = self.open(path)?;
        let length =
            usize::try_from(identity.2).map_err(|_| Error::resource("input size overflow"))?;
        if length > max_bytes {
            return Err(Error::resource("structured raw file cap exceeded"));
        }
        let mut bytes = BudgetVec::with_capacity(
            &self.budget,
            length
                .checked_add(1)
                .ok_or_else(|| Error::resource("input size overflow"))?,
        )?;
        let _scratch = self.budget.reserve(32768)?;
        let mut block = [0u8; 32768];
        let mut hasher = Sha256::new();
        loop {
            self.control.check()?;
            let left = length.saturating_sub(bytes.len()) + 1;
            let count = file.read(&mut block[..left.min(32768)])?;
            if count == 0 {
                break;
            }
            if bytes.len() + count > length {
                return Err(Error::contract("structured raw input grew"));
            }
            bytes.extend_from_slice(&block[..count])?;
            hasher.update(&block[..count]);
        }
        self.current(path, &file, &identity)?;
        if bytes.len() != length {
            return Err(Error::contract("structured raw input size changed"));
        }
        self.remember(path, identity, format!("{:x}", hasher.finalize()))?;
        Ok(bytes)
    }
    pub fn json(&mut self, path: &Path) -> Result<Document> {
        let (mut file, identity) = self.open(path)?;
        let length =
            usize::try_from(identity.2).map_err(|_| Error::resource("JSON size overflow"))?;
        if length > self.max_json {
            return Err(Error::resource("structured JSON file cap exceeded"));
        }
        let admission = json::admit(length, &self.budget)?;
        let mut bytes = BudgetVec::with_capacity(
            &self.budget,
            length
                .checked_add(1)
                .ok_or_else(|| Error::resource("JSON size overflow"))?,
        )?;
        let _scratch = self.budget.reserve(32768)?;
        let mut block = [0u8; 32768];
        let mut hasher = Sha256::new();
        loop {
            self.control.check()?;
            let left = length.saturating_sub(bytes.len()) + 1;
            let count = file.read(&mut block[..left.min(32768)])?;
            if count == 0 {
                break;
            }
            if bytes.len() + count > length {
                return Err(Error::contract("structured JSON grew while reading"));
            }
            bytes.extend_from_slice(&block[..count])?;
            hasher.update(&block[..count]);
        }
        self.current(path, &file, &identity)?;
        if bytes.len() != length {
            return Err(Error::contract("structured JSON size changed"));
        }
        let document = json::parse_admitted(&bytes, &admission, &self.control)?;
        self.remember(path, identity, format!("{:x}", hasher.finalize()))?;
        Ok(document)
    }
    pub fn checked_json(&mut self, path: &Path, expected: &Value) -> Result<Document> {
        let value = self.json(path)?;
        let actual = &self
            .files
            .iter()
            .find(|item| item.path == path)
            .expect("remembered JSON")
            .digest;
        if actual != expected.string()? {
            return Err(Error::contract("structured metadata SHA256 mismatch"));
        }
        Ok(value)
    }
    pub fn file_map(&mut self, path: &Path, files: &Value, retained: bool) -> Result<()> {
        for (name, expected) in files.object()? {
            self.control.check()?;
            super::structured_controls::basename(name)?;
            super::structured_controls::checksum(expected)?;
            if retained && self.hash(&path.join(name))? != expected.string()? {
                return Err(Error::contract("structured artifact file hash mismatch"));
            }
        }
        Ok(())
    }
    fn names(
        &self,
        path: &Path,
        expected: &[String],
        directories: bool,
    ) -> Result<(Vec<String>, Identity, Lease)> {
        self.safe(&path.join("placeholder"))?;
        let before = fs::symlink_metadata(path)?;
        if !before.is_dir() {
            return Err(Error::contract(
                "structured inventory real directory required",
            ));
        }
        let mut lease = self.budget.reserve(512)?;
        let mut names = Vec::new();
        for item in fs::read_dir(path)? {
            self.control.check()?;
            let item = item?;
            let raw = item.file_name();
            let name = raw
                .to_str()
                .ok_or_else(|| Error::contract("non UTF8 structured inventory"))?;
            if !expected.iter().any(|value| value == name)
                || names.iter().any(|value| value == name)
            {
                return Err(Error::contract(
                    "missing/foreign structured control inventory",
                ));
            }
            let kind = item.file_type()?;
            if if directories {
                !kind.is_dir()
            } else {
                !kind.is_file()
            } {
                return Err(Error::contract("nonregular structured inventory entry"));
            }
            lease.resize(lease.bytes() + 1024 + 4 * name.len())?;
            names
                .try_reserve_exact(1)
                .map_err(|_| Error::resource("inventory allocation failed"))?;
            names.push(name.to_owned());
        }
        self.control.check()?;
        if names.len() != expected.len()
            || Identity::of(&fs::symlink_metadata(path)?) != Identity::of(&before)
        {
            return Err(Error::contract(
                "structured control inventory changed/missing",
            ));
        }
        names.sort();
        Ok((names, Identity::of(&before), lease))
    }
    pub fn inventory(&mut self, path: &Path, names: Vec<String>, directories: bool) -> Result<()> {
        let (names, identity, lease) = self.names(path, &names, directories)?;
        self.inventories.push(Inventory {
            path: path.into(),
            names,
            identity,
            directories,
            pattern: None,
            append: None,
            _lease: lease,
        })
    }
    pub fn recheck(&self) -> Result<()> {
        self.control.check()?;
        for item in self.files.iter() {
            let (mut file, identity) = self.open(&item.path)?;
            if identity != item.identity {
                return Err(Error::contract("structured original input changed"));
            }
            let _scratch = self.budget.reserve(32768)?;
            let mut block = [0u8; 32768];
            let mut hasher = Sha256::new();
            loop {
                self.control.check()?;
                let count = file.read(&mut block)?;
                if count == 0 {
                    break;
                }
                hasher.update(&block[..count]);
            }
            self.current(&item.path, &file, &item.identity)?;
            if format!("{:x}", hasher.finalize()) != item.digest {
                return Err(Error::contract("structured original input hash changed"));
            }
        }
        for item in self.inventories.iter() {
            if let Some(append) = &item.append {
                self.live_inventory(item, append)?;
                continue;
            }
            let (names, identity, _) = if let Some(pattern) = item.pattern {
                self.source_names(pattern, &item.names)?
            } else {
                self.names(&item.path, &item.names, item.directories)?
            };
            if names != item.names || identity != item.identity {
                return Err(Error::contract("structured original inventory changed"));
            }
        }
        for item in self.files.iter() {
            self.control.check()?;
            if Identity::of(&fs::symlink_metadata(&item.path)?) != item.identity {
                return Err(Error::contract(
                    "structured input changed after final hash sweep",
                ));
            }
        }
        self.current_directories()
    }
    fn current_directories(&self) -> Result<()> {
        self.control.check()?;
        if Identity::of(&fs::symlink_metadata(&self.root)?) != self.root_identity {
            return Err(Error::contract(
                "structured original root directory changed",
            ));
        }
        for item in self.inventories.iter() {
            self.control.check()?;
            let current = Identity::of(&fs::symlink_metadata(&item.path)?);
            let expected = if let Some(append) = &item.append {
                append
                    .observed
                    .borrow()
                    .clone()
                    .unwrap_or_else(|| item.identity.clone())
            } else {
                item.identity.clone()
            };
            if current != expected || (current.0, current.1) != (item.identity.0, item.identity.1) {
                return Err(Error::contract(
                    "structured original inventory changed after sweep",
                ));
            }
        }
        self.control.check()
    }
    fn live_inventory(&self, item: &Inventory, append: &LiveAppend) -> Result<()> {
        self.control.check()?;
        let current = Identity::of(&fs::symlink_metadata(&item.path)?);
        if (current.0, current.1) != (item.identity.0, item.identity.1) {
            return Err(Error::contract("original live artifact directory replaced"));
        }
        let present = match fs::symlink_metadata(item.path.join(&append.name)) {
            Ok(info) => {
                if !info.is_dir() || info.file_type().is_symlink() {
                    return Err(Error::contract(
                        "derived next-generation output must be a real directory",
                    ));
                }
                true
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(error.into()),
        };
        let seen = append.observed.borrow();
        if let Some(original_append) = seen.as_ref() {
            if !present || current != *original_append {
                return Err(Error::contract(
                    "live artifact namespace changed after its permitted append",
                ));
            }
        } else if !present && current != item.identity {
            return Err(Error::contract(
                "original artifact namespace changed before its derived append",
            ));
        }
        drop(seen);
        let _scratch = self.budget.reserve(1024 + item.names.len() * 256)?;
        let mut expected = item.names.clone();
        if present {
            expected.push(append.name.clone());
            expected.sort();
        }
        let (names, identity, _) = self.names(&item.path, &expected, true)?;
        if names != expected || identity != current {
            return Err(Error::contract(
                "live artifact namespace changed during final sweep",
            ));
        }
        if present && append.observed.borrow().is_none() {
            append.observed.replace(Some(identity));
        }
        Ok(())
    }
    pub(crate) fn begin_live_output(&mut self) -> Result<()> {
        self.recheck()?;
        let path = self.root.join("artifacts");
        let index = self
            .inventories
            .iter()
            .position(|i| i.path == path && i.directories && i.pattern.is_none())
            .ok_or_else(|| Error::contract("original parent artifact inventory required"))?;
        let item = &self.inventories[index];
        if item.append.is_some()
            || item.names.is_empty()
            || item.names.len() > 1000
            || item
                .names
                .iter()
                .enumerate()
                .any(|(i, name)| *name != format!("{i:06}"))
        {
            return Err(Error::contract(
                "one derived live output phase requires exact original generation inventory",
            ));
        }
        let name = format!("{:06}", item.names.len());
        match fs::symlink_metadata(path.join(&name)) {
            Ok(_) => {
                return Err(Error::contract(
                    "derived next-generation output already exists",
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let lease = self.budget.reserve(1024)?;
        self.inventories[index].append = Some(LiveAppend {
            name,
            observed: std::cell::RefCell::new(None),
            _lease: lease,
        });
        Ok(())
    }
    pub(crate) fn final_snapshots(&self) -> Result<()> {
        for item in self.files.iter() {
            self.control.check()?;
            if Identity::of(&fs::symlink_metadata(&item.path)?) != item.identity {
                return Err(Error::contract(
                    "original Source changed after combined final sweep",
                ));
            }
        }
        self.current_directories()
    }
    fn source_names(
        &self,
        pattern: SourcePattern,
        expected: &[String],
    ) -> Result<(Vec<String>, Identity, Lease)> {
        self.safe(&self.root.join("placeholder"))?;
        let before = Identity::of(&fs::symlink_metadata(&self.root)?);
        let _lookup = self.budget.reserve(1024 + expected.len() * 128)?;
        let lookup = expected
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let mut lease = self.budget.reserve(1024)?;
        let mut names = Vec::new();
        for directory in fs::read_dir(&self.root)? {
            self.control.check()?;
            let directory = directory?;
            let kind = directory.file_type()?;
            if kind.is_symlink() {
                return Err(Error::contract("symlink in original Source inventory"));
            }
            if !kind.is_dir() {
                continue;
            }
            let dirname = directory.file_name();
            let dirname = dirname
                .to_str()
                .ok_or_else(|| Error::contract("nonUTF8 original directory"))?;
            for entry in fs::read_dir(directory.path())? {
                self.control.check()?;
                let entry = entry?;
                let raw = entry.file_name();
                let name = raw
                    .to_str()
                    .ok_or_else(|| Error::contract("nonUTF8 original Source entry"))?;
                if !pattern.matches(name) {
                    continue;
                }
                if !entry.file_type()?.is_file() {
                    return Err(Error::contract(
                        "nonregular original Source inventory entry",
                    ));
                }
                lease.resize(lease.bytes() + 1024 + 4 * (dirname.len() + name.len() + 1))?;
                let relative = format!("{dirname}/{name}");
                if !lookup.contains(relative.as_str()) {
                    return Err(Error::contract("missing/foreign original Source inventory"));
                }
                names
                    .try_reserve_exact(1)
                    .map_err(|_| Error::resource("Source inventory allocation failed"))?;
                names.push(relative);
            }
        }
        self.control.check()?;
        names.sort();
        if names.len() != expected.len()
            || names.windows(2).any(|v| v[0] == v[1])
            || Identity::of(&fs::symlink_metadata(&self.root)?) != before
        {
            return Err(Error::contract("original Source inventory changed/missing"));
        }
        Ok((names, before, lease))
    }
    pub(crate) fn source_inventory(
        &mut self,
        pattern: SourcePattern,
        expected: Vec<String>,
    ) -> Result<()> {
        let (names, identity, lease) = self.source_names(pattern, &expected)?;
        self.inventories.push(Inventory {
            path: self.root.clone(),
            names,
            identity,
            directories: false,
            pattern: Some(pattern),
            append: None,
            _lease: lease,
        })
    }
    pub(crate) fn proof_files(&self, root: &Path) -> Result<Document> {
        self.control.check()?;
        let _copy = self.budget.reserve(65536 + self.files.len() * 4096)?;
        let mut values = json::object([]);
        for item in self.files.iter() {
            self.control.check()?;
            let relative = item
                .path
                .strip_prefix(root)
                .map_err(|_| Error::contract("foreign Source evidence root"))?
                .to_str()
                .ok_or_else(|| Error::contract("nonUTF8 Source evidence path"))?;
            let i = &item.identity;
            let identity = json::array(
                [
                    i.0 as i128,
                    i.1 as i128,
                    i.2 as i128,
                    i.3 as i128 * 1000000000 + i.4 as i128,
                    i.5 as i128 * 1000000000 + i.6 as i128,
                ]
                .into_iter()
                .map(json::integer),
            );
            values.as_object_mut().unwrap().insert(
                relative.into(),
                json::object([
                    ("sha256", json::text(&item.digest)),
                    ("bytes", json::integer(i.2 as i128)),
                    ("identity", identity),
                ]),
            );
        }
        let raw = json::canonical(&values, &self.budget, &self.control)?;
        json::parse(&raw, &self.budget, &self.control)
    }
    pub(crate) fn verified_sha(&self, path: &Path) -> Result<&str> {
        self.control.check()?;
        self.files
            .iter()
            .find(|entry| entry.path == path)
            .map(|entry| entry.digest.as_str())
            .ok_or_else(|| Error::contract("original verified file identity required"))
    }
    pub fn fingerprint(&self) -> Result<String> {
        let mut hasher = Sha256::new();
        hasher.update(b"structured-parent-controls-v1\0");
        for item in self.files.iter() {
            self.control.check()?;
            hasher.update(item.path.as_os_str().as_encoded_bytes());
            hasher.update([0]);
            hasher.update(item.digest.as_bytes());
            hasher.update([0]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }
}

pub struct ParentControls {
    pub sources: Sources,
    pub summary: Document,
}
impl ParentControls {
    pub fn open(
        directory: &Path,
        budget: Arc<Budget>,
        control: Control,
        max_json: usize,
    ) -> Result<Self> {
        let mut sources = Sources::new(directory, budget, control, max_json)?;
        let summary = crate::structured_controls::parent(&mut sources)?;
        sources.recheck()?;
        Ok(Self { sources, summary })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn late_earlier_inventory_mutation_is_checked_against_original_after_sweep() {
        let root = std::env::temp_dir().join(format!(
            "gfb-structured-final-inventory-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let first = root.join("generations");
        let last = root.join("artifacts");
        fs::create_dir(&first).unwrap();
        fs::create_dir(&last).unwrap();
        let mut sources = Sources::new(
            &root,
            Budget::new(1024 * 1024),
            Control::new(crate::budget::monotonic() + 30.0).unwrap(),
            1024,
        )
        .unwrap();
        sources.inventory(&first, vec![], false).unwrap();
        sources.inventory(&last, vec![], false).unwrap();
        // Both inventories passed. A later directory's read can trigger a
        // mutation in an earlier one; neither file evidence nor another
        // fresh snapshot may stand in for the ORIGINAL directory identity.
        for item in sources.inventories.iter() {
            let (_, identity, _) = sources
                .names(&item.path, &item.names, item.directories)
                .unwrap();
            assert_eq!(identity, item.identity);
        }
        fs::write(first.join("late.json"), b"{}").unwrap();
        // Filesystems may coalesce adjacent directory timestamp updates.
        // Make the mutation observable deterministically without sleeps.
        File::open(&first)
            .unwrap()
            .set_modified(std::time::UNIX_EPOCH)
            .unwrap();
        let checked = sources.current_directories();
        assert!(
            matches!(&checked, Err(Error::Contract(message))
            if message.contains("inventory changed after sweep")),
            "{checked:?}"
        );
        assert!(sources.recheck().is_err());
        fs::remove_file(first.join("late.json")).unwrap();
        File::open(&first)
            .unwrap()
            .set_modified(std::time::UNIX_EPOCH)
            .unwrap();
        // Restoring names does not reset the original directory snapshot.
        assert!(sources.current_directories().is_err());
        fs::write(root.join("late-root.json"), b"{}").unwrap();
        File::open(&root)
            .unwrap()
            .set_modified(std::time::UNIX_EPOCH)
            .unwrap();
        assert!(
            matches!(sources.current_directories(), Err(Error::Contract(message))
            if message.contains("root directory changed"))
        );
        fs::remove_dir_all(root).unwrap();
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Scanned,
    Authorized,
    Exported,
    PayloadRetired,
    Updating,
    Finalized,
    Closed,
    Failed,
}
pub(crate) struct CohortSession {
    pub inputs: crate::structured_source::OriginalInputs,
    pub controls: crate::structured_source::Controls,
    phase: Phase,
}
impl CohortSession {
    pub(crate) fn is_closed(&self) -> bool {
        self.phase == Phase::Closed
    }
    pub(crate) fn fail_export(&mut self) {
        self.phase = Phase::Failed;
    }
    pub(crate) fn open(
        plan: &[u8],
        budget: Arc<Budget>,
        control: Control,
        source_bytes: usize,
        record_bytes: usize,
        scratch_bytes: usize,
        chunk_bytes: usize,
    ) -> Result<Self> {
        let mut inputs = crate::structured_source::OriginalInputs::open(
            plan,
            budget,
            control,
            source_bytes,
            record_bytes,
            scratch_bytes,
            chunk_bytes,
        )?;
        let controls = inputs.controls()?;
        Ok(Self {
            inputs,
            controls,
            phase: Phase::Scanned,
        })
    }
    pub(crate) fn check(&self) -> Result<()> {
        self.inputs.control.check()?;
        if matches!(self.phase, Phase::Closed | Phase::Finalized | Phase::Failed) {
            return Err(Error::Closed);
        }
        Ok(())
    }
    pub(crate) fn recheck(&self) -> Result<()> {
        self.check()?;
        self.inputs.recheck()
    }
    pub(crate) fn original_config_inputs(&self, parent_index: usize) -> Result<BudgetVec<u8>> {
        self.recheck()?;
        let parent = self
            .inputs
            .parents
            .get(parent_index)
            .ok_or_else(|| Error::contract("original input parent index out of range"))?;
        let b = &self.inputs.budget;
        let c = &self.inputs.control;
        let original = parent.state.root.require("config_inputs")?;
        let _copy = crate::json::admit(crate::json::canonical(original, b, c)?.len() + 65536, b)?;
        let value = crate::json::object([
            ("schema_version", crate::json::integer(2)),
            (
                "kind",
                crate::json::text("structured-original-config-inputs"),
            ),
            (
                "control_fingerprint",
                crate::json::text(&self.controls.fingerprint),
            ),
            ("parent_index", crate::json::integer(parent_index as i128)),
            (
                "run_sha256",
                crate::json::text(
                    parent
                        .sources
                        .verified_sha(&parent.directory.join("run.json"))?,
                ),
            ),
            ("config_inputs", crate::json::copy(original, c)?),
        ]);
        let bytes = crate::json::canonical(&value, b, c)?;
        if bytes.len() > 64 * 1024 * 1024 {
            return Err(Error::resource("original input query wire exceeds64MiB"));
        }
        self.recheck()?;
        Ok(bytes)
    }
    pub(crate) fn summary(&self) -> Result<BudgetVec<u8>> {
        self.recheck()?;
        let bytes = json::canonical(
            &self.controls.summary.root,
            &self.inputs.budget,
            &self.inputs.control,
        )?;
        if bytes.len() > 64 * 1024 * 1024 {
            return Err(Error::resource("structured control wire exceeds64MiB"));
        }
        Ok(bytes)
    }
    pub(crate) fn authorize(&mut self, encoded: &[u8]) -> Result<()> {
        self.check()?;
        if self.phase != Phase::Scanned {
            return Err(Error::contract(
                "one structured authorization in scanned state required",
            ));
        }
        let max = self
            .controls
            .summary
            .root
            .require("authorization_max_bytes")?
            .integer(1, 64i128 << 20)? as usize;
        if encoded.len() > max {
            return Err(Error::resource(
                "structured authorization exceeds admitted wire",
            ));
        }
        let value = json::parse(encoded, &self.inputs.budget, &self.inputs.control)?;
        if json::canonical(&value.root, &self.inputs.budget, &self.inputs.control)?[..]
            != json::canonical(
                &self.controls.authorization.root,
                &self.inputs.budget,
                &self.inputs.control,
            )?[..]
        {
            return Err(Error::contract(
                "closed structured authorization differs from ORIGINAL controls/sealed/schedule links",
            ));
        }
        self.inputs.recheck()?;
        self.phase = Phase::Authorized;
        Ok(())
    }
    pub(crate) fn export(&mut self) -> Result<(BudgetVec<u8>, crate::structured_staging::Banks)> {
        self.check()?;
        if self.phase != Phase::Authorized {
            return Err(Error::contract(
                "one live cohort export after authorization required",
            ));
        }
        let collection = match self.inputs.mode {
            crate::structured_source::ScanMode::Live { collection, .. } => collection,
            _ => return Err(Error::contract("historical parent mode cannot export")),
        };
        self.inputs.recheck()?;
        let source = &self.inputs.collections[collection];
        let model = source
            .model
            .as_ref()
            .ok_or_else(|| Error::contract("selected neural model descriptor required"))?;
        let declaration = self.controls.declarations[collection]
            .as_ref()
            .ok_or_else(|| Error::contract("selected recorded semantics required"))?;
        let c = &self.inputs.control;
        let b = &self.inputs.budget;
        let _copy = json::admit(
            json::canonical(&source.episodes.root, b, c)?.len()
                + json::canonical(&source.proof.root, b, c)?.len()
                + 65536,
            b,
        )?;
        let mut metadata = json::object([
            (
                "architecture_identity",
                json::text(&crate::structured_controls::architecture(
                    &model.root,
                    b,
                    c,
                )?),
            ),
            (
                "action_config_hash",
                json::text(&crate::structured_controls::digest(
                    source
                        .manifest
                        .root
                        .require("effective_config")?
                        .require("actions")?,
                    b,
                    c,
                )?),
            ),
            (
                "policy_version",
                source
                    .learner_label
                    .as_ref()
                    .map(|v| json::text(v))
                    .unwrap_or(Value::Null),
            ),
            (
                "decision_count",
                json::integer(source.accepted_learner as i128),
            ),
            ("episodes", json::copy(&source.episodes.root, c)?),
        ]);
        for (key, value) in declaration.root.object()? {
            metadata
                .as_object_mut()
                .unwrap()
                .insert(key.clone(), json::copy(value, c)?);
        }
        for key in ["action_contract", "purpose", "synthetic_capacity"] {
            metadata.as_object_mut().unwrap().insert(
                key.into(),
                json::copy(source.manifest.root.require(key)?, c)?,
            );
        }
        let stage = self.inputs.stage.as_mut().ok_or(Error::Closed)?;
        let payload = json::object([
            ("schema_version", json::integer(1)),
            ("kind", json::text("structured-cohort-payload")),
            (
                "control_fingerprint",
                json::text(&self.controls.fingerprint),
            ),
            ("collection_index", json::integer(collection as i128)),
            ("metadata", metadata),
            ("arrays", stage.descriptors()?),
            ("source_proof", json::copy(&source.proof.root, c)?),
            (
                "correspondence",
                json::copy(
                    &self.controls.correspondences[collection]
                        .as_ref()
                        .ok_or(Error::Closed)?
                        .root,
                    c,
                )?,
            ),
        ]);
        let encoded = json::canonical(&payload, b, c)?;
        if encoded.len() > 64 * 1024 * 1024 {
            return Err(Error::resource("structured cohort metadata exceeds64MiB"));
        }
        self.phase = Phase::Failed;
        // These independently owned RAM banks have been hash-verified. Retire
        // the disk spool before exposing them to C1 serialization, so its files
        // cannot overlap the numeric output on disk. Keep the Stage reachable
        // on failure for confirmed cleanup; no payload or Exported phase escapes.
        let banks = stage.export_retiring_disk()?;
        self.phase = Phase::Exported;
        Ok((encoded, banks))
    }
    pub(crate) fn retire_payload(&mut self) -> Result<()> {
        self.check()?;
        if self.phase != Phase::Exported {
            return Err(Error::contract("exported payload must retire exactly once"));
        }
        self.inputs.recheck()?;
        let mut stage = self.inputs.stage.take().ok_or(Error::Closed)?;
        if let Err(error) = stage.close() {
            self.inputs.stage = Some(stage);
            return Err(error);
        }
        drop(stage);
        self.phase = Phase::PayloadRetired;
        Ok(())
    }
    pub(crate) fn begin_update(&mut self) -> Result<()> {
        self.check()?;
        if self.phase != Phase::PayloadRetired {
            return Err(Error::contract(
                "live update requires retired payload after actual Python correspondence",
            ));
        }
        let parent = match self.inputs.mode {
            crate::structured_source::ScanMode::Live { parent, .. } => parent,
            _ => return Err(Error::contract("historical parent cannot begin an update")),
        };
        self.inputs.recheck()?;
        self.inputs.parents[parent].sources.begin_live_output()?;
        self.phase = Phase::Updating;
        Ok(())
    }
    pub(crate) fn finalize(&mut self) -> Result<BudgetVec<u8>> {
        self.check()?;
        match self.inputs.mode {
            crate::structured_source::ScanMode::Live {
                parent, generation, ..
            } => {
                if self.phase != Phase::Updating {
                    return Err(Error::contract(
                        "live finalize requires the explicit update phase",
                    ));
                }
                let output = self.inputs.parents[parent]
                    .directory
                    .join("artifacts")
                    .join(format!("{:06}", generation + 1));
                let info = fs::symlink_metadata(output)?;
                if !info.is_dir() || info.file_type().is_symlink() {
                    return Err(Error::contract(
                        "successful live finalization lacks derived learner output directory",
                    ));
                }
            }
            crate::structured_source::ScanMode::Historical => {
                if self.phase != Phase::Authorized {
                    return Err(Error::contract(
                        "historical finalize requires one authorization after ALL actual Python numeric checks",
                    ));
                }
            }
        }
        self.inputs.recheck()?;
        let proof = json::object([
            ("schema_version", json::integer(2)),
            ("kind", json::text("structured-source-finalized")),
            (
                "control_fingerprint",
                json::text(&self.controls.fingerprint),
            ),
            ("qualified", Value::Bool(false)),
            ("parent_qualification", json::text("not_performed")),
        ]);
        let bytes = json::canonical(&proof, &self.inputs.budget, &self.inputs.control)?;
        self.phase = Phase::Finalized;
        Ok(bytes)
    }
    pub(crate) fn close(&mut self) -> Result<()> {
        if self.phase == Phase::Closed {
            return Ok(());
        }
        let check = self.inputs.recheck();
        if let Some(stage) = self.inputs.stage.as_mut() {
            stage.close()?;
        }
        self.inputs.stage.take();
        self.phase = Phase::Closed;
        check
    }
}
