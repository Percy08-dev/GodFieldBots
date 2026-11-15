//! Additive diagnostic ownership for original structured parent controls.
//! No Journal5, ZIP/NPY, authorization or numerical export claims.
#[path = "card_instance_final_validation.rs"]
pub(crate) mod final_validation;
#[path = "card_instance_original_episode.rs"]
mod original_episode;
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
#[cfg(test)]
thread_local! { static SOURCE_IO: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[inline]
fn source_io() {
    #[cfg(test)]
    SOURCE_IO.with(|count| count.set(count.get() + 1));
}

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
    mixed: bool,
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
    access: SourceAccess,
    pub(crate) registry: Option<Arc<crate::logical_artifacts::Registry>>,
    logical_files: BudgetVec<(PathBuf, Lease)>,
    files: BudgetVec<Evidence>,
    inventories: BudgetVec<Inventory>,
    absent_controls: BudgetVec<(PathBuf, Lease)>,
}
enum SourceAccess {
    Physical(Identity),
    Logical,
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
            access: SourceAccess::Physical(root_identity),
            registry: None,
            logical_files: BudgetVec::new(&budget),
            files: BudgetVec::new(&budget),
            inventories: BudgetVec::new(&budget),
            absent_controls: BudgetVec::new(&budget),
            budget,
            control,
            max_json,
        })
    }
    pub(crate) fn from_registry(
        directory: &Path,
        budget: Arc<Budget>,
        control: Control,
        max_json: usize,
        registry: Option<Arc<crate::logical_artifacts::Registry>>,
    ) -> Result<Self> {
        control.check()?;
        if registry
            .as_ref()
            .is_some_and(|r| r.locate(directory).is_some())
        {
            let directory = std::path::absolute(directory)?;
            let (_, _, prefix) = registry.as_ref().unwrap().locate(&directory).unwrap();
            if !prefix.is_empty() {
                crate::logical_artifacts::member_name(&prefix)?;
            }
            Ok(Self {
                root: directory,
                access: SourceAccess::Logical,
                registry,
                logical_files: BudgetVec::new(&budget),
                files: BudgetVec::new(&budget),
                inventories: BudgetVec::new(&budget),
            absent_controls: BudgetVec::new(&budget),
                budget,
                control,
                max_json,
            })
        } else {
            let mut source = Self::new(directory, budget, control, max_json)?;
            source.registry = registry;
            Ok(source)
        }
    }
    pub(crate) fn is_logical(&self, path: &Path) -> bool {
        self.registry
            .as_ref()
            .is_some_and(|r| r.locate(path).is_some())
    }
    fn logical_file(&self, path: &Path) -> Result<Option<&Value>> {
        self.control.check()?;
        if !path.starts_with(&self.root) {
            return Err(Error::contract("foreign original logical path"));
        }
        if let Some(registry) = &self.registry {
            if let Some((_, source, name)) = registry.locate(path) {
                return Ok(Some(registry.file(source, &name)?));
            }
        }
        Ok(None)
    }
    fn remember_logical(&mut self, path: &Path) -> Result<()> {
        if !self.logical_files.iter().any(|(p, _)| p == path) {
            let lease = self.budget.reserve(1024 + 4 * path.as_os_str().len())?;
            self.logical_files.push((path.into(), lease))?;
        }
        Ok(())
    }
    pub(crate) fn file_len(&self, path: &Path) -> Result<usize> {
        if let Some(file) = self.logical_file(path)? {
            return Ok(file.require("bytes")?.integer(0, usize::MAX as i128)? as usize);
        }
        let (_, identity) = self.open(path)?;
        usize::try_from(identity.2).map_err(|_| Error::resource("original file length overflow"))
    }
    pub(crate) fn child_names(&self, path: &Path) -> Result<(Vec<(String, bool)>, Lease)> {
        self.control.check()?;
        let mut lease = self.budget.reserve(65536)?;
        let mut children = std::collections::BTreeMap::new();
        if let Some(registry) = &self.registry {
            if let Some((_, source, prefix)) = registry.locate(path) {
                for row in source.description.root.require("files")?.array()? {
                    lease
                        .resize(lease.bytes() + 2048 + 16 * row.require("name")?.string()?.len())?;
                }
                let prefix = if prefix.is_empty() {
                    String::new()
                } else {
                    format!("{prefix}/")
                };
                for row in source.description.root.require("files")?.array()? {
                    if let Some(relative) = row.require("name")?.string()?.strip_prefix(&prefix) {
                        let (name, directory) = relative
                            .split_once('/')
                            .map_or((relative, false), |(name, _)| (name, true));
                        if let Some(old) = children.insert(name.to_owned(), directory) {
                            if old != directory {
                                return Err(Error::contract(
                                    "logical file/directory name collision",
                                ));
                            }
                        }
                    }
                }
                return Ok((children.into_iter().collect(), lease));
            }
        }
        self.safe(&path.join("placeholder"))?;
        for item in fs::read_dir(path)? {
            let item = item?;
            let kind = item.file_type()?;
            if !kind.is_dir() && !kind.is_file() {
                return Err(Error::contract("nonregular original child"));
            }
            lease.resize(lease.bytes() + 2048 + 16 * item.file_name().len())?;
            children.insert(
                item.file_name()
                    .into_string()
                    .map_err(|_| Error::contract("nonUTF8 original child"))?,
                kind.is_dir(),
            );
        }
        Ok((children.into_iter().collect(), lease))
    }
    pub(crate) fn packed_context(&self) -> Result<(&Value, &Value)> {
        let (_, source, _) = self
            .registry
            .as_ref()
            .and_then(|r| r.locate(&self.root))
            .ok_or_else(|| Error::contract("packed original root required"))?;
        Ok((
            source.description.root.require("storage")?,
            source.description.root.require("custody")?,
        ))
    }
    fn safe(&self, path: &Path) -> Result<()> {
        self.control.check()?;
        source_io();
        let SourceAccess::Physical(root_identity) = &self.access else {
            return Err(Error::contract(
                "logical originals cannot be opened as physical files",
            ));
        };
        if Identity::of(&fs::symlink_metadata(&self.root)?) != *root_identity {
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
        source_io();
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
    /// Cached original byte bound for preallocation; never establishes new evidence.
    pub(crate) fn original_len(&self, path: &Path) -> Result<usize> {
        self.control.check()?;
        if self.logical_files.iter().any(|(p, _)| p == path) {
            return self.file_len(path);
        }
        let file = self
            .files
            .iter()
            .find(|e| e.path == path)
            .ok_or_else(|| Error::contract("retained original file required for admission"))?;
        usize::try_from(file.identity.2)
            .map_err(|_| Error::resource("original byte bound overflow"))
    }
    pub fn hash(&mut self, path: &Path) -> Result<String> {
        if let Some(file) = self.logical_file(path)? {
            let digest = file.require("sha256")?.string()?.to_owned();
            self.remember_logical(path)?;
            return Ok(digest);
        }
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
        if self.is_logical(path) {
            let raw = self.registry.as_ref().unwrap().read(
                path,
                max_bytes.min(self.max_json),
                &self.budget,
                &self.control,
            )?;
            self.remember_logical(path)?;
            return Ok(raw);
        }
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
        if self.is_logical(path) {
            let length = self.file_len(path)?;
            let admission = json::admit(length, &self.budget)?;
            let raw = self.bytes(path, self.max_json)?;
            return crate::logical_artifacts::parse_admitted(&raw, &admission, &self.control);
        }
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
        let document = if self.registry.is_some() {
            crate::logical_artifacts::parse_admitted(&bytes, &admission, &self.control)?
        } else {
            json::parse_admitted(&bytes, &admission, &self.control)?
        };
        self.remember(path, identity, format!("{:x}", hasher.finalize()))?;
        Ok(document)
    }
    pub(crate) fn parse_json(&self, bytes: &[u8]) -> Result<Document> {
        if self.registry.is_some() {
            crate::logical_artifacts::parse(bytes, &self.budget, &self.control)
        } else {
            json::parse(bytes, &self.budget, &self.control)
        }
    }
    pub fn checked_json(&mut self, path: &Path, expected: &Value) -> Result<Document> {
        let value = self.json(path)?;
        let actual = self.verified_sha(path)?;
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
        self.names_kind(path, expected, Some(directories))
    }
    fn names_kind(
        &self,
        path: &Path,
        expected: &[String],
        kind: Option<bool>,
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
            let entry_kind = item.file_type()?;
            if match kind {
                Some(true) => !entry_kind.is_dir(),
                Some(false) => !entry_kind.is_file(),
                None => !entry_kind.is_dir() && !entry_kind.is_file(),
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
        if self.is_logical(path) {
            let (actual, _admission) = self.child_names(path)?;
            let expected = names
                .iter()
                .map(|n| (n.clone(), directories))
                .collect::<std::collections::BTreeMap<_, _>>();
            if expected.len() != names.len()
                || actual
                    .into_iter()
                    .collect::<std::collections::BTreeMap<_, _>>()
                    != expected
            {
                return Err(Error::contract("logical original inventory differs"));
            }
            return Ok(());
        }
        let (names, identity, lease) = self.names(path, &names, directories)?;
        self.inventories.push(Inventory {
            path: path.into(),
            names,
            identity,
            directories,
            mixed: false,
            pattern: None,
            append: None,
            _lease: lease,
        })
    }
    /// Snapshot a complete public filesystem subtree without parsing opaque
    /// content. Uses the same file/inventory evidence and final-sweep owner.
    pub(crate) fn owns_original(&self, path: &Path) -> bool {
        self.files.iter().any(|e| e.path == path)
            || self.logical_files.iter().any(|(p, _)| p == path)
    }
    pub(crate) fn pin_tree(&mut self, path: &Path) -> Result<()> {
        if self.is_logical(path) {
            let (children, _admission) = self.child_names(path)?;
            for (name, directory) in children {
                let child = path.join(name);
                if directory {
                    self.pin_tree(&child)?;
                } else {
                    self.hash(&child)?;
                }
            }
            return Ok(());
        }
        self.safe(&path.join("placeholder"))?;
        let mut names = Vec::new();
        let mut lease = self.budget.reserve(1024)?;
        for entry in fs::read_dir(path)? {
            self.control.check()?;
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| Error::contract("nonUTF8 original subtree"))?;
            let kind = entry.file_type()?;
            if !kind.is_file() && !kind.is_dir() {
                return Err(Error::contract("foreign original subtree entry"));
            }
            lease.resize(lease.bytes() + 2048 + 4 * name.len())?;
            names.push((name, kind.is_dir()));
        }
        names.sort();
        let expected = names
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        let (actual, identity, retained) = self.names_kind(path, &expected, None)?;
        self.inventories.push(Inventory {
            path: path.into(),
            names: actual,
            identity,
            directories: false,
            mixed: true,
            pattern: None,
            append: None,
            _lease: retained,
        })?;
        for (name, directory) in names {
            let child = path.join(name);
            if directory {
                self.pin_tree(&child)?;
            } else {
                self.hash(&child)?;
            }
        }
        Ok(())
    }
    /// Exact one-target numeric append. Prior subtrees are fully snapshotted
    /// before any target can be published; this never creates the numeric root.
    pub(crate) fn begin_numeric_output(&mut self, generation: usize) -> Result<usize> {
        self.recheck()?;
        let root = self.root.join("numeric");
        let prior = (0..generation)
            .map(|i| format!("batch-{i:06}"))
            .collect::<Vec<_>>();
        let (names, identity, lease) = self.names(&root, &prior, true)?;
        let name = format!("batch-{generation:06}");
        match fs::symlink_metadata(root.join(&name)) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            _ => return Err(Error::contract("numeric append target must be absent")),
        };
        for child in &names {
            self.pin_tree(&root.join(child))?;
        }
        let index = self.inventories.len();
        let append_lease = self.budget.reserve(1024)?;
        self.inventories.push(Inventory {
            path: root,
            names,
            identity,
            directories: true,
            mixed: false,
            pattern: None,
            append: Some(LiveAppend {
                name,
                observed: std::cell::RefCell::new(None),
                _lease: append_lease,
            }),
            _lease: lease,
        })?;
        self.recheck()?;
        Ok(index)
    }
    pub(crate) fn bind_numeric_output(&mut self, index: usize) -> Result<PathBuf> {
        self.recheck()?;
        let item = self
            .inventories
            .get(index)
            .ok_or_else(|| Error::contract("numeric append permit missing"))?;
        let append = item
            .append
            .as_ref()
            .ok_or_else(|| Error::contract("numeric append permit already consumed"))?;
        let target = item.path.join(&append.name);
        let info = fs::symlink_metadata(&target)?;
        if !info.is_dir() || info.file_type().is_symlink() {
            return Err(Error::contract(
                "numeric append target is not a real directory",
            ));
        }
        let mut expected = item.names.clone();
        expected.push(append.name.clone());
        expected.sort();
        let (names, identity, lease) = self.names(&item.path, &expected, true)?;
        self.inventories[index].names = names;
        self.inventories[index].identity = identity;
        self.inventories[index]._lease = lease;
        self.inventories[index].append = None;
        Ok(target)
    }
    pub fn recheck(&self) -> Result<()> {
        self.control.check()?;
        if let Some(registry) = &self.registry {
            registry.recheck(&self.root, &self.control)?;
        }
        if matches!(self.access, SourceAccess::Logical) {
            return Ok(());
        }
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
                self.names_kind(
                    &item.path,
                    &item.names,
                    if item.mixed {
                        None
                    } else {
                        Some(item.directories)
                    },
                )?
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
    pub(crate) fn absent_control(&mut self, path: &Path) -> Result<()> {
        self.safe(path)?;
        if path.exists() || path.is_symlink() { return Err(Error::contract("resident absent control appeared")); }
        let lease = self.budget.reserve(1024 + path.as_os_str().len() * 4)?;
        self.absent_controls.push((path.into(),lease))
    }
    fn current_directories(&self) -> Result<()> {
        for (path,_) in self.absent_controls.iter() {
            if path.exists() || path.is_symlink() { return Err(Error::contract("resident absent control appeared after snapshot")); }
        }
        self.control.check()?;
        source_io();
        let SourceAccess::Physical(root_identity) = &self.access else {
            return self
                .registry
                .as_ref()
                .unwrap()
                .recheck(&self.root, &self.control);
        };
        if Identity::of(&fs::symlink_metadata(&self.root)?) != *root_identity {
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
    pub(crate) fn begin_live_output(
        &mut self,
        format: crate::structured_controls::SourceFormat,
    ) -> Result<()> {
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
            || !live_artifact_count_allowed(format, item.names.len())
            || item
                .names
                .iter()
                .enumerate()
                .any(|(i, name)| !live_artifact_name_allowed(format, name, i, item.names.len()))
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
        if self.is_logical(&self.root) {
            let mut actual = Vec::new();
            let (children, _admission) = self.child_names(&self.root)?;
            for (episode, directory) in children {
                if directory {
                    let (members, _member_admission) =
                        self.child_names(&self.root.join(&episode))?;
                    for (name, child_dir) in members {
                        if pattern.matches(&name) {
                            if child_dir {
                                return Err(Error::contract(
                                    "logical original member must be file",
                                ));
                            }
                            actual.push(format!("{episode}/{name}"));
                        }
                    }
                }
            }
            actual.sort();
            let mut expected = expected;
            expected.sort();
            if actual != expected || expected.windows(2).any(|v| v[0] == v[1]) {
                return Err(Error::contract("logical Source inventory differs"));
            }
            return Ok(());
        }
        let (names, identity, lease) = self.source_names(pattern, &expected)?;
        self.inventories.push(Inventory {
            path: self.root.clone(),
            names,
            identity,
            directories: false,
            mixed: false,
            pattern: Some(pattern),
            append: None,
            _lease: lease,
        })
    }
    pub(crate) fn proof_files(&self, root: &Path) -> Result<Document> {
        self.control.check()?;
        let _copy = self.budget.reserve(
            65536
                + (self.files.len() + self.logical_files.len()) * 4096
                + self
                    .logical_files
                    .iter()
                    .map(|(p, _)| p.as_os_str().len() * 8)
                    .sum::<usize>(),
        )?;
        let mut values = json::object([]);
        for (path, _) in self.logical_files.iter() {
            let relative = path
                .strip_prefix(root)
                .map_err(|_| Error::contract("foreign logical proof root"))?
                .to_str()
                .ok_or_else(|| Error::contract("nonUTF8 logical proof name"))?;
            let file = self
                .logical_file(path)?
                .ok_or_else(|| Error::contract("logical proof member missing"))?;
            values.as_object_mut().unwrap().insert(
                relative.into(),
                json::object([
                    (
                        "sha256",
                        json::copy(file.require("sha256")?, &self.control)?,
                    ),
                    ("bytes", json::copy(file.require("bytes")?, &self.control)?),
                    (
                        "identity",
                        json::copy(file.require("identity")?, &self.control)?,
                    ),
                ]),
            );
        }
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
        if self.logical_files.iter().any(|(p, _)| p == path) {
            return self
                .logical_file(path)?
                .unwrap()
                .require("sha256")?
                .string();
        }
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
        for (path, _) in self.logical_files.iter() {
            self.control.check()?;
            hasher.update(path.as_os_str().as_encoded_bytes());
            hasher.update([0]);
            hasher.update(self.verified_sha(path)?.as_bytes());
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
    RecordedExported,
    PayloadRetired,
    NumericBound,
    Updating,
    Prepared,
    Finalized,
    Closed,
    Failed,
}
pub(crate) struct CohortSession {
    pub inputs: crate::structured_source::OriginalInputs,
    pub controls: crate::structured_source::Controls,
    phase: Phase,
    numeric_permit: Option<usize>,
    numeric: Option<crate::json::Document>,
    expected_arrays: Option<crate::json::Document>,
    recorded_episodes: Option<crate::json::Document>,
    numeric_guard: Option<Arc<crate::budget::NumericGuard>>,
}
impl CohortSession {
    fn ensure_numeric_guard(&mut self) -> Result<()> {
        if self.inputs.format.hpo() && self.numeric_guard.is_none() {
            let limit = self.inputs.parents[0]
                .state
                .root
                .require("effective_config")?
                .require("hpo_spec")?
                .require("operating")?
                .require("numeric_budget_bytes")?
                .integer(1, i64::MAX as i128)? as usize;
            self.numeric_guard = Some(self.inputs.budget.numeric_guard(limit)?);
        }
        Ok(())
    }
    fn abort_empty_numeric_guard(&mut self) {
        if self.inputs.stage.is_none() && self.inputs.budget.numeric_used() == Some(0) {
            self.numeric_guard.take();
        }
    }
    fn used_decisions(&self, index: usize) -> Result<usize> {
        let source = self.inputs.collections.get(index).ok_or(Error::Closed)?;
        if self.inputs.format.hpo() {
            Ok(source
                .proof
                .root
                .require("selection")?
                .require("used_decisions")?
                .integer(0, source.accepted_learner as i128)? as usize)
        } else {
            Ok(source.accepted_learner)
        }
    }
    pub(crate) fn open5(
        plan: &[u8],
        provider: Option<Arc<dyn crate::logical_artifacts::LogicalArtifactProvider>>,
        budget: Arc<Budget>,
        control: Control,
        source_bytes: usize,
        record_bytes: usize,
        scratch_bytes: usize,
        chunk_bytes: usize,
    ) -> Result<Self> {
        let mut inputs = crate::structured_source::OriginalInputs::open5(
            plan,
            provider,
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
            numeric_permit: None,
            numeric: None,
            expected_arrays: None,
            recorded_episodes: None,
            numeric_guard: None,
        })
    }
    pub(crate) fn open4(
        plan: &[u8],
        provider: Arc<dyn crate::logical_artifacts::LogicalArtifactProvider>,
        budget: Arc<Budget>,
        control: Control,
        source_bytes: usize,
        record_bytes: usize,
        scratch_bytes: usize,
        chunk_bytes: usize,
    ) -> Result<Self> {
        let mut inputs = crate::structured_source::OriginalInputs::open4(
            plan,
            provider,
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
            numeric_permit: None,
            numeric: None,
            expected_arrays: None,
            recorded_episodes: None,
            numeric_guard: None,
        })
    }
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
        Self::open_context(
            plan,
            budget,
            control,
            source_bytes,
            record_bytes,
            scratch_bytes,
            chunk_bytes,
            false,
        )
    }
    pub(crate) fn open3(
        plan: &[u8],
        budget: Arc<Budget>,
        control: Control,
        source_bytes: usize,
        record_bytes: usize,
        scratch_bytes: usize,
        chunk_bytes: usize,
    ) -> Result<Self> {
        Self::open_context(
            plan,
            budget,
            control,
            source_bytes,
            record_bytes,
            scratch_bytes,
            chunk_bytes,
            true,
        )
    }
    fn open_context(
        plan: &[u8],
        budget: Arc<Budget>,
        control: Control,
        source_bytes: usize,
        record_bytes: usize,
        scratch_bytes: usize,
        chunk_bytes: usize,
        corrected: bool,
    ) -> Result<Self> {
        let mut inputs = (if corrected {
            crate::structured_source::OriginalInputs::open3
        } else {
            crate::structured_source::OriginalInputs::open
        })(
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
            numeric_permit: None,
            numeric: None,
            expected_arrays: None,
            recorded_episodes: None,
            numeric_guard: None,
        })
    }
    pub(crate) fn check(&self) -> Result<()> {
        self.inputs.control.check()?;
        if matches!(
            self.phase,
            Phase::Closed | Phase::Finalized | Phase::Failed | Phase::Prepared
        ) {
            return Err(Error::Closed);
        }
        Ok(())
    }
    // Corrected transitions first perform only owner/phase/mode/view checks.
    // Once admitted, every reportable failure is terminal for operations; only
    // cleanup may retry. Old component checks retain their original ordering.
    fn transition_owner_check(&self) -> Result<()> {
        if self.inputs.format.corrected()
            && !matches!(
                self.phase,
                Phase::Closed | Phase::Finalized | Phase::Failed | Phase::Prepared
            )
        {
            Ok(())
        } else {
            self.check()
        }
    }
    fn admitted_corrected_work(&mut self) -> Result<()> {
        if self.inputs.format.corrected() {
            self.phase = Phase::Failed;
            self.inputs.control.check()?;
        }
        Ok(())
    }
    fn export_preconditions(&self) -> Result<usize> {
        self.transition_owner_check()?;
        if self.phase != Phase::Authorized {
            if self.inputs.format.corrected() {
                self.check()?;
            }
            return Err(Error::contract(
                "one live cohort export after authorization required",
            ));
        }
        match self.inputs.mode {
            crate::structured_source::ScanMode::Live { collection, .. } => Ok(collection),
            _ => {
                if self.inputs.format.corrected() {
                    self.check()?;
                }
                Err(Error::contract("historical parent mode cannot export"))
            }
        }
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
            (
                "schema_version",
                crate::json::integer(if self.inputs.format.corrected() { 3 } else { 2 }),
            ),
            (
                "kind",
                crate::json::text(if self.inputs.format.corrected() {
                    "card-instance-original-config-inputs"
                } else {
                    "structured-original-config-inputs"
                }),
            ),
            (
                if self.inputs.format.corrected() {
                    "summary_sha256"
                } else {
                    "control_fingerprint"
                },
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
        self.transition_owner_check()?;
        if self.phase != Phase::Scanned {
            if self.inputs.format.corrected() {
                self.check()?;
            }
            return Err(Error::contract(
                "one structured authorization in scanned state required",
            ));
        }
        self.admitted_corrected_work()?;
        if self.inputs.format.hpo() {
            return Err(Error::contract(
                "HPO pending facts require additive authorize_hpo",
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
    pub(crate) fn authorize_hpo(&mut self, encoded: &[u8]) -> Result<BudgetVec<u8>> {
        self.transition_owner_check()?;
        if !self.inputs.format.hpo() || self.phase != Phase::Scanned {
            self.check()?;
            return Err(Error::contract(
                "one HPO pending authorization in the SAME scanned session required",
            ));
        }
        self.admitted_corrected_work()?;
        let maximum = self
            .controls
            .summary
            .root
            .require("authorization_max_bytes")?
            .integer(1, 64i128 << 20)? as usize;
        if encoded.len() > maximum {
            return Err(Error::resource(
                "HPO authorization exceeds bounded original wire",
            ));
        }
        let request = json::parse(encoded, &self.inputs.budget, &self.inputs.control)?;
        let (controls, proofs) = self.inputs.authorize_hpo(&self.controls, &request.root)?;
        let result = json::canonical(
            &controls.summary.root,
            &self.inputs.budget,
            &self.inputs.control,
        )?;
        if result.len() > 64 * 1024 * 1024 {
            return Err(Error::resource(
                "HPO final control summary exceeds wire limit",
            ));
        }
        self.inputs.recheck()?;
        for (source, proof) in self.inputs.collections.iter_mut().zip(proofs) {
            source.proof = proof;
        }
        self.controls = controls;
        self.phase = Phase::Authorized;
        Ok(result)
    }
    pub(crate) fn export(&mut self) -> Result<(BudgetVec<u8>, crate::structured_staging::Banks)> {
        let collection = self.export_preconditions()?;
        let corrected = self.inputs.format.corrected();
        self.admitted_corrected_work()?;
        self.ensure_numeric_guard()?;
        self.inputs.recheck()?;
        if self.inputs.format.hpo() {
            self.inputs.stage = Some(crate::structured_staging::Stage::new(
                self.inputs.budget.clone(),
                self.inputs.scratch_bytes,
                self.inputs.chunk_bytes,
                self.inputs.control.clone(),
            )?);
            let episodes = self.inputs.restage_recorded3(collection)?;
            self.inputs.collections[collection].episodes = episodes;
        }
        self.export_selected(collection, false)
    }
    fn export_selected(
        &mut self,
        collection: usize,
        recorded: bool,
    ) -> Result<(BudgetVec<u8>, crate::structured_staging::Banks)> {
        let corrected = self.inputs.format.corrected();
        let hpo = self.inputs.format.hpo();
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
            json::canonical(
                if recorded {
                    &self.recorded_episodes.as_ref().ok_or(Error::Closed)?.root
                } else {
                    &source.episodes.root
                },
                b,
                c,
            )?
            .len()
                + json::canonical(&source.proof.root, b, c)?.len()
                + 65536,
            b,
        )?;
        let mut metadata = json::object([
            (
                "architecture_identity",
                json::text(&(if hpo {
                    crate::structured_controls::hpo::architecture
                } else if corrected {
                    crate::structured_controls::architecture3
                } else {
                    crate::structured_controls::architecture
                })(&model.root, b, c)?),
            ),
            (
                "action_config_hash",
                json::text(&crate::structured_controls::digest(
                    source
                        .manifest
                        .root
                        .require(if corrected {
                            "config"
                        } else {
                            "effective_config"
                        })?
                        .require("actions")?,
                    b,
                    c,
                )?),
            ),
            (
                "policy_version",
                if corrected && source.accepted_learner == 0 {
                    Value::Null
                } else {
                    source
                        .learner_label
                        .as_ref()
                        .map(|v| json::text(v))
                        .unwrap_or(Value::Null)
                },
            ),
            (
                "decision_count",
                json::integer(if hpo {
                    source
                        .proof
                        .root
                        .require("selection")?
                        .require("used_decisions")?
                        .integer(0, i64::MAX as i128)?
                } else {
                    source.accepted_learner as i128
                }),
            ),
            (
                "episodes",
                json::copy(
                    if recorded {
                        &self.recorded_episodes.as_ref().ok_or(Error::Closed)?.root
                    } else {
                        &source.episodes.root
                    },
                    c,
                )?,
            ),
        ]);
        if hpo {
            metadata.as_object_mut().unwrap().insert(
                "training_recipe".into(),
                self.inputs.parents[0]
                    .state
                    .root
                    .require("effective_config")?
                    .require("hpo_spec")?
                    .require("training_recipe")?
                    .clone(),
            );
            metadata.as_object_mut().unwrap().insert(
                "selection".into(),
                source.proof.root.require("selection")?.clone(),
            );
            metadata.as_object_mut().unwrap().insert(
                "expected_optimizer_step_delta".into(),
                source
                    .proof
                    .root
                    .require("expected_optimizer_step_delta")?
                    .clone(),
            );
        }
        for (key, value) in declaration.root.object()? {
            metadata
                .as_object_mut()
                .unwrap()
                .insert(key.clone(), json::copy(value, c)?);
        }
        for key in if corrected {
            &[
                "action_contract",
                "purpose",
                "synthetic_capacity",
                "rule_signature",
                "layout_signature",
            ][..]
        } else {
            &["action_contract", "purpose", "synthetic_capacity"][..]
        } {
            metadata.as_object_mut().unwrap().insert(
                (*key).into(),
                json::copy(source.manifest.root.require(key)?, c)?,
            );
        }
        if corrected && !recorded {
            let (parent, generation) = match self.inputs.mode {
                crate::structured_source::ScanMode::Live {
                    parent, generation, ..
                } => (parent, generation),
                _ => return Err(Error::contract("historical export prohibited")),
            };
            self.numeric_permit = Some(
                self.inputs.parents[parent]
                    .sources
                    .begin_numeric_output(generation)?,
            );
        }
        let stage = self.inputs.stage.as_mut().ok_or(Error::Closed)?;
        let mut payload = json::object([
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
        if corrected {
            payload
                .as_object_mut()
                .unwrap()
                .remove("control_fingerprint");
            payload["schema_version"] = json::integer(if hpo { 3 } else { 2 });
            payload["kind"] = json::text(if hpo {
                if recorded {
                    "card-instance-hpo-recorded-cohort-payload"
                } else {
                    "card-instance-hpo-cohort-payload"
                }
            } else if recorded {
                "card-instance-recorded-cohort-payload"
            } else {
                "card-instance-cohort-payload"
            });
            if !recorded {
                self.expected_arrays = Some(crate::structured_source::doc(
                    payload.require("arrays")?,
                    b,
                    c,
                )?);
            }
        }
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
        self.phase = if recorded {
            Phase::RecordedExported
        } else {
            Phase::Exported
        };
        Ok((encoded, banks))
    }
    pub(crate) fn recorded_export_preconditions(&self, index: usize) -> Result<()> {
        self.transition_owner_check()?;
        if !self.inputs.format.corrected()
            || self.phase != Phase::Authorized
            || !matches!(
                self.inputs.mode,
                crate::structured_source::ScanMode::Historical
            )
            || index >= self.inputs.collections.len()
            || !self
                .inputs
                .committed
                .iter()
                .any(|v| v.collection_index == index)
        {
            self.check()?;
            return Err(Error::contract(
                "recorded export requires authorized historical committed index",
            ));
        }
        Ok(())
    }
    pub(crate) fn export_recorded_cohort(
        &mut self,
        index: usize,
    ) -> Result<(BudgetVec<u8>, crate::structured_staging::Banks)> {
        self.recorded_export_preconditions(index)?;
        self.admitted_corrected_work()?;
        self.ensure_numeric_guard()?;
        self.inputs.recheck()?;
        self.inputs.stage = Some(crate::structured_staging::Stage::new(
            self.inputs.budget.clone(),
            self.inputs.scratch_bytes,
            self.inputs.chunk_bytes,
            self.inputs.control.clone(),
        )?);
        self.recorded_episodes = Some(self.inputs.restage_recorded3(index)?);
        // Original proof/declaration/correspondence remain the acknowledged values;
        // all held path/inode/hash snapshots were rechecked around the raw walk.
        self.export_selected(index, true)
    }
    pub(crate) fn retire_payload(&mut self) -> Result<()> {
        self.transition_owner_check()?;
        if self.phase != Phase::Exported
            && !(self.inputs.format.corrected() && self.phase == Phase::RecordedExported)
        {
            if self.inputs.format.corrected() {
                self.check()?;
            }
            return Err(Error::contract("exported payload must retire exactly once"));
        }
        let recorded = self.phase == Phase::RecordedExported;
        self.admitted_corrected_work()?;
        self.inputs.recheck()?;
        let mut stage = self.inputs.stage.take().ok_or(Error::Closed)?;
        if let Err(error) = stage.close() {
            self.inputs.stage = Some(stage);
            return Err(error);
        }
        drop(stage);
        self.recorded_episodes.take();
        self.numeric_guard.take();
        self.phase = if recorded {
            Phase::Authorized
        } else {
            Phase::PayloadRetired
        };
        Ok(())
    }
    pub(crate) fn begin_update(&mut self) -> Result<()> {
        self.transition_owner_check()?;
        if self.phase
            != if self.inputs.format.corrected() {
                Phase::NumericBound
            } else {
                Phase::PayloadRetired
            }
        {
            if self.inputs.format.corrected() {
                self.check()?;
            }
            return Err(Error::contract(
                "live update requires retired payload after actual Python correspondence",
            ));
        }
        let parent = match self.inputs.mode {
            crate::structured_source::ScanMode::Live { parent, .. } => parent,
            _ => {
                if self.inputs.format.corrected() {
                    self.check()?;
                }
                return Err(Error::contract("historical parent cannot begin an update"));
            }
        };
        if self.inputs.format.corrected() {
            let collection = match self.inputs.mode {
                crate::structured_source::ScanMode::Live { collection, .. } => collection,
                _ => return Err(Error::contract("historical update prohibited")),
            };
            if self.used_decisions(collection)? == 0 {
                self.check()?;
                return Err(Error::contract("empty cohort cannot update"));
            }
        }
        self.admitted_corrected_work()?;
        self.inputs.recheck()?;
        self.inputs.parents[parent]
            .sources
            .begin_live_output(self.inputs.format)?;
        self.phase = Phase::Updating;
        Ok(())
    }
    pub(crate) fn finalize(&mut self) -> Result<BudgetVec<u8>> {
        if self.inputs.format.corrected() {
            return Err(Error::contract(
                "corrected sessions require prepare_commit/finalize_read",
            ));
        }
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
            if let Err(cleanup) = stage.close() {
                if self.inputs.format.corrected() {
                    return match check {
                        Err(primary) => Err(Error::Cleanup {
                            primary: Box::new(primary),
                            cleanup: cleanup.to_string(),
                        }),
                        Ok(()) => Err(cleanup),
                    };
                }
                return Err(cleanup);
            }
        }
        self.inputs.stage.take();
        self.phase = Phase::Closed;
        check
    }
}

impl CohortSession {
    pub(crate) fn bind_numeric(&mut self, encoded: &[u8]) -> Result<()> {
        self.transition_owner_check()?;
        if !self.inputs.format.corrected() || self.phase != Phase::PayloadRetired {
            if self.inputs.format.corrected() {
                self.check()?;
            }
            return Err(Error::contract(
                "numeric4 binding requires corrected retired payload",
            ));
        }
        let (collection, parent, generation) = match self.inputs.mode {
            crate::structured_source::ScanMode::Live {
                collection,
                parent,
                generation,
            } => (collection, parent, generation),
            _ => {
                self.check()?;
                return Err(Error::contract("historical numeric append prohibited"));
            }
        };
        self.admitted_corrected_work()?;
        if encoded.len() > 64 * 1024 * 1024 {
            return Err(Error::resource("numeric binding wire cap"));
        }
        let b = self.inputs.budget.clone();
        let c = self.inputs.control.clone();
        let request = json::parse(encoded, &b, &c)?;
        request
            .root
            .closed(&["directory", "manifest_sha256", "files"], &[])?;
        let expected = self.inputs.parents[parent]
            .directory
            .join("numeric")
            .join(format!("batch-{generation:06}"));
        if std::path::absolute(request.root.require("directory")?.string()?)? != expected {
            return Err(Error::contract(
                "numeric binding differs from derived target",
            ));
        }
        crate::structured_controls::checksum(request.root.require("manifest_sha256")?)?;
        self.inputs.recheck()?;
        self.phase = Phase::Failed;
        let permit = self
            .numeric_permit
            .take()
            .ok_or_else(|| Error::contract("one-use numeric append permit absent"))?;
        let hpo = self.inputs.format.hpo();
        let recipe = if hpo {
            Some(
                self.inputs.parents[parent]
                    .state
                    .root
                    .require("effective_config")?
                    .require("hpo_spec")?
                    .require("training_recipe")?
                    .clone(),
            )
        } else {
            None
        };
        let used = self.used_decisions(collection)?;
        let source = &self.inputs.collections[collection];
        let model = &source.model.as_ref().ok_or(Error::Closed)?.root;
        let declaration = &self.controls.declarations[collection]
            .as_ref()
            .ok_or(Error::Closed)?
            .root;
        let owner = &mut self.inputs.parents[parent].sources;
        if owner.bind_numeric_output(permit)? != expected {
            return Err(Error::contract("numeric permit target mismatch"));
        }
        let manifest = owner.checked_json(
            &expected.join("rollout.json"),
            request.root.require("manifest_sha256")?,
        )?;
        let v = &manifest.root;
        let mut required = vec![
            "schema_version",
            "kind",
            "architecture_identity",
            "action_config_hash",
            "policy_version",
            "decision_count",
            "episodes",
            "data",
            "action_contract",
            "purpose",
            "synthetic_capacity",
            "rule_signature",
            "layout_signature",
        ];
        required.extend(declaration.object()?.keys().map(String::as_str));
        if hpo {
            required.extend([
                "training_recipe",
                "selection",
                "expected_optimizer_step_delta",
            ]);
        }
        v.closed(&required, &[])?;
        v.require("schema_version")?
            .integer(if hpo { 5 } else { 4 }, if hpo { 5 } else { 4 })?;
        if v.require("kind")?.string()? != "structured-rollout"
            || v.require("architecture_identity")?.string()?
                != (if hpo {
                    crate::structured_controls::hpo::architecture
                } else {
                    crate::structured_controls::architecture3
                })(model, &b, &c)?
            || v.require("action_config_hash")?
                != source.manifest.root.require("action_config_hash")?
            || v.require("decision_count")?.integer(0, i32::MAX as i128)? != used as i128
        {
            return Err(Error::contract("numeric4 model/config/decision mismatch"));
        }
        if hpo {
            if v.require("training_recipe")? != recipe.as_ref().ok_or(Error::Closed)?
                || v.require("selection")? != source.proof.root.require("selection")?
                || v.require("expected_optimizer_step_delta")?
                    != source.proof.root.require("expected_optimizer_step_delta")?
            {
                return Err(Error::contract(
                    "numeric5 full recipe/selection/delta differs from SAME authorized originals",
                ));
            }
        }
        crate::structured_controls::game_binding3(v, &b, &c)?;
        for (key, value) in declaration.object()? {
            if v.require(key)? != value {
                return Err(Error::contract(
                    "numeric4 full semantic declaration differs",
                ));
            }
        }
        let label = if source.accepted_learner == 0 {
            Value::Null
        } else {
            json::text(source.learner_label.as_deref().ok_or(Error::Closed)?)
        };
        if v.require("policy_version")? != &label {
            return Err(Error::contract("numeric4 original learner label differs"));
        }
        let mut episodes = json::copy(&source.episodes.root, &c)?;
        for ep in episodes
            .as_array_mut()
            .ok_or_else(|| Error::contract("original staged episode metadata missing"))?
        {
            ep["has_targets"] = Value::Bool(!ep.require("frames")?.array()?.is_empty());
        }
        if v.require("episodes")? != &episodes {
            return Err(Error::contract(
                "numeric4 original episode/frame/selection/closure metadata differs",
            ));
        }
        let arrays = crate::structured_controls::bundle_metadata(v.require("data")?)?;
        let mut expected_arrays = std::collections::BTreeMap::new();
        for a in self
            .expected_arrays
            .as_ref()
            .ok_or(Error::Closed)?
            .root
            .array()?
        {
            expected_arrays.insert(
                a.require("key")?.string()?.to_owned(),
                json::object([
                    ("dtype", json::copy(a.require("dtype")?, &c)?),
                    ("shape", json::copy(a.require("shape")?, &c)?),
                ]),
            );
        }
        for (index, ep) in episodes.array()?.iter().enumerate() {
            let count = ep.require("frames")?.array()?.len();
            if count > 0 {
                for field in ["advantage", "target"] {
                    expected_arrays.insert(
                        format!("e.{index}.{field}"),
                        json::object([
                            ("dtype", json::text("float32")),
                            ("shape", json::array([json::integer(count as i128)])),
                        ]),
                    );
                }
            }
        }
        if arrays.len() != expected_arrays.len() {
            return Err(Error::contract("numeric4 exact original array set differs"));
        }
        for (key, expected_array) in expected_arrays {
            let actual = arrays
                .get(&key)
                .ok_or_else(|| Error::contract("numeric4 original array absent"))?;
            for field in ["dtype", "shape"] {
                if actual.require(field)? != expected_array.require(field)? {
                    return Err(Error::contract(
                        "numeric4 original array dtype/shape differs",
                    ));
                }
            }
        }
        let mut files = v.require("data")?.require("files")?.object()?.clone();
        files.insert(
            "rollout.json".into(),
            json::copy(request.root.require("manifest_sha256")?, &c)?,
        );
        if request.root.require("files")? != &Value::Object(files.clone()) {
            return Err(Error::contract("numeric4 exact persisted file map differs"));
        }
        owner.file_map(&expected, request.root.require("files")?, true)?;
        owner.inventory(&expected, files.keys().cloned().collect(), false)?;
        for shard in v.require("data")?.require("shards")?.array()? {
            let name = shard.require("file")?.string()?.replace(".npz", ".json");
            let side = owner.checked_json(
                &expected.join(&name),
                v.require("data")?.require("files")?.require(&name)?,
            )?;
            if &side.root != shard.require("metadata")? {
                return Err(Error::contract("numeric4 shard sidecar differs"));
            }
        }
        self.numeric = Some(request);
        self.inputs.recheck()?;
        self.phase = Phase::NumericBound;
        Ok(())
    }
    fn prepare_commit3(&mut self) -> Result<BudgetVec<u8>> {
        self.transition_owner_check()?;
        if !self.inputs.format.corrected() || self.phase != Phase::Updating {
            if self.inputs.format.corrected() {
                self.check()?;
            }
            return Err(Error::contract(
                "corrected precommit requires one completed update phase",
            ));
        }
        let (collection, parent, generation) = match self.inputs.mode {
            crate::structured_source::ScanMode::Live {
                collection,
                parent,
                generation,
            } => (collection, parent, generation),
            _ => {
                self.check()?;
                return Err(Error::contract("historical precommit prohibited"));
            }
        };
        self.admitted_corrected_work()?;
        self.inputs.recheck()?;
        let b = self.inputs.budget.clone();
        let c = self.inputs.control.clone();
        let hpo = self.inputs.format.hpo();
        let count = self.used_decisions(collection)?;
        let before = self.inputs.parents[parent]
            .state
            .root
            .require("optimizer_steps")?
            .integer(0, i32::MAX as i128)? as usize;
        let delta = if hpo {
            self.inputs.collections[collection]
                .proof
                .root
                .require("expected_optimizer_step_delta")?
                .integer(1, i64::MAX as i128)? as usize
        } else {
            4 * ((count + 127) / 128)
        };
        let after = before
            .checked_add(delta)
            .ok_or_else(|| Error::resource("optimizer count overflow"))?;
        {
            let original_parent = &mut self.inputs.parents[parent];
            (if hpo {
                crate::structured_controls::hpo::verify_derived_artifact
            } else {
                crate::structured_controls::verify_derived_artifact3
            })(
                &mut original_parent.sources,
                &original_parent.state.root,
                generation,
                after,
            )?;
        }
        if let Some(stage) = self.inputs.stage.as_mut() {
            stage.close()?;
        }
        self.inputs.stage.take();
        self.inputs.recheck()?;
        let numeric = self.numeric.as_ref().ok_or(Error::Closed)?;
        let mut receipt = json::object([
            (
                "schema_version",
                json::integer(if hpo {
                    3
                } else if self.inputs.format.packed() {
                    2
                } else {
                    1
                }),
            ),
            (
                "kind",
                json::text(if hpo {
                    if self.inputs.format.packed() {
                        "card-instance-hpo-packed-source-prepared"
                    } else {
                        "card-instance-hpo-source-prepared"
                    }
                } else if self.inputs.format.packed() {
                    "card-instance-packed-source-prepared"
                } else {
                    "card-instance-source-prepared"
                }),
            ),
            ("summary_sha256", json::text(&self.controls.fingerprint)),
            (
                "collection_manifest_sha256",
                json::copy(
                    self.inputs.collections[collection]
                        .proof
                        .root
                        .require("manifest_sha256")?,
                    &c,
                )?,
            ),
            (
                "source_correspondence",
                json::copy(
                    &self.controls.correspondences[collection]
                        .as_ref()
                        .ok_or(Error::Closed)?
                        .root,
                    &c,
                )?,
            ),
            (
                "numeric_manifest_sha256",
                json::copy(numeric.root.require("manifest_sha256")?, &c)?,
            ),
            (
                "numeric_files_sha256",
                json::text(&crate::structured_controls::digest(
                    numeric.root.require("files")?,
                    &b,
                    &c,
                )?),
            ),
            ("optimizer_steps", json::integer(after as i128)),
        ]);
        if hpo {
            for (key, value) in [
                (
                    "collected_decisions",
                    json::integer(self.inputs.collections[collection].accepted_learner as i128),
                ),
                ("used_decisions", json::integer(count as i128)),
                (
                    "expected_optimizer_step_delta",
                    json::integer(delta as i128),
                ),
                (
                    "selection_sha256",
                    json::text(&crate::structured_controls::digest(
                        self.inputs.collections[collection]
                            .proof
                            .root
                            .require("selection")?,
                        &b,
                        &c,
                    )?),
                ),
            ] {
                receipt.as_object_mut().unwrap().insert(key.into(), value);
            }
        }
        let bytes = json::canonical(&receipt, &b, &c)?;
        self.phase = Phase::Prepared;
        Ok(bytes)
    }
    fn finalize_read3(&mut self) -> Result<BudgetVec<u8>> {
        self.transition_owner_check()?;
        if !self.inputs.format.corrected()
            || !matches!(
                self.inputs.mode,
                crate::structured_source::ScanMode::Historical
            )
            || self.phase != Phase::Authorized
        {
            if self.inputs.format.corrected() {
                self.check()?;
            }
            return Err(Error::contract(
                "historical finalization requires one corrected authorization",
            ));
        }
        self.admitted_corrected_work()?;
        self.inputs.recheck()?;
        let hpo = self.inputs.format.hpo();
        let receipt = json::object([
            (
                "schema_version",
                json::integer(if hpo {
                    3
                } else if self.inputs.format.packed() {
                    2
                } else {
                    1
                }),
            ),
            (
                "kind",
                json::text(if hpo {
                    if self.inputs.format.packed() {
                        "card-instance-hpo-packed-source-checked"
                    } else {
                        "card-instance-hpo-source-checked"
                    }
                } else if self.inputs.format.packed() {
                    "card-instance-packed-source-read-finalized"
                } else {
                    "card-instance-source-read-finalized"
                }),
            ),
            ("summary_sha256", json::text(&self.controls.fingerprint)),
            ("qualified", Value::Bool(false)),
            ("parent_qualification", json::text("not_performed")),
        ]);
        let bytes = json::canonical(&receipt, &self.inputs.budget, &self.inputs.control)?;
        self.phase = Phase::Finalized;
        Ok(bytes)
    }
}

/// Internal exported packet. Its private guard cannot be dropped separately
/// from owned banks; the future FFI bridge must retain this packet/guard through
/// the existing exported-array base-owner leases. No Python ABI is exposed here.
pub(crate) struct CardInstancePayload {
    metadata: BudgetVec<u8>,
    banks: crate::structured_staging::Banks,
    _guard: Arc<()>,
    _lease: crate::budget::Lease,
}
impl CardInstancePayload {
    pub(crate) fn metadata(&self) -> &[u8] {
        &self.metadata
    }
    pub(crate) fn banks(&self) -> &crate::structured_staging::Banks {
        &self.banks
    }
}
pub(crate) struct CardInstanceNativeSession {
    core: Option<CohortSession>,
    retirement: Option<crate::retirement::Retirement<CohortSession>>,
    budget: Arc<Budget>,
    payload: std::sync::Weak<()>,
    receipt: Option<BudgetVec<u8>>,
    final_slot: Option<final_validation::Slot>,
    original_episode: original_episode::State,
}
impl CardInstanceNativeSession {
    pub(crate) fn open(
        plan: &[u8],
        budget: Arc<Budget>,
        control: Control,
        source_bytes: usize,
        record_bytes: usize,
        scratch_bytes: usize,
        chunk_bytes: usize,
    ) -> Result<Self> {
        Self::open_context(
            plan,
            None,
            budget,
            control,
            source_bytes,
            record_bytes,
            scratch_bytes,
            chunk_bytes,
        )
    }
    pub(crate) fn open4(
        plan: &[u8],
        provider: Arc<dyn crate::logical_artifacts::LogicalArtifactProvider>,
        budget: Arc<Budget>,
        control: Control,
        source_bytes: usize,
        record_bytes: usize,
        scratch_bytes: usize,
        chunk_bytes: usize,
    ) -> Result<Self> {
        Self::open_context(
            plan,
            Some(provider),
            budget,
            control,
            source_bytes,
            record_bytes,
            scratch_bytes,
            chunk_bytes,
        )
    }
    fn open_context(
        plan: &[u8],
        provider: Option<Arc<dyn crate::logical_artifacts::LogicalArtifactProvider>>,
        budget: Arc<Budget>,
        control: Control,
        source_bytes: usize,
        record_bytes: usize,
        scratch_bytes: usize,
        chunk_bytes: usize,
    ) -> Result<Self> {
        if plan.len() > 65536 {
            return Err(Error::resource("corrected plan bounded wire required"));
        }
        let hpo = {
            let parsed = json::parse(plan, &budget, &control)?;
            matches!(parsed.root.get("schema_version").and_then(Value::as_i64), Some(5 | 6))
        };
        if (!hpo && budget.limit() > 12usize * 1024 * 1024 * 1024)
            || source_bytes == 0
            || source_bytes > 64 * 1024 * 1024
            || record_bytes == 0
            || record_bytes > source_bytes
            || scratch_bytes == 0
            || (!hpo && scratch_bytes > 12usize * 1024 * 1024 * 1024)
            || chunk_bytes == 0
            || chunk_bytes > 64 * 1024 * 1024
        {
            return Err(Error::resource(
                "corrected native settings bounds before Source I/O",
            ));
        }
        let retirement = crate::retirement::Retirement::start(&budget, |core: CohortSession| {
            drop(core);
            Ok(())
        })?;
        let core = if hpo {
            CohortSession::open5(
                plan,
                provider,
                budget.clone(),
                control,
                source_bytes,
                record_bytes,
                scratch_bytes,
                chunk_bytes,
            )?
        } else if let Some(provider) = provider {
            CohortSession::open4(
                plan,
                provider,
                budget.clone(),
                control,
                source_bytes,
                record_bytes,
                scratch_bytes,
                chunk_bytes,
            )?
        } else {
            CohortSession::open3(
                plan,
                budget.clone(),
                control,
                source_bytes,
                record_bytes,
                scratch_bytes,
                chunk_bytes,
            )?
        };
        Ok(Self {
            core: Some(core),
            retirement: Some(retirement),
            budget,
            payload: std::sync::Weak::new(),
            receipt: None,
            final_slot: None,
            original_episode: original_episode::State::new(),
        })
    }
    fn core(&self) -> Result<&CohortSession> {
        self.core.as_ref().ok_or(Error::Closed)
    }
    fn core_mut(&mut self) -> Result<&mut CohortSession> {
        self.core.as_mut().ok_or(Error::Closed)
    }
    pub(crate) fn control_summary(&self) -> Result<BudgetVec<u8>> {
        self.core()?.summary()
    }
    pub(crate) fn is_hpo(&self) -> bool {
        self.core
            .as_ref()
            .is_some_and(|core| core.inputs.format.hpo())
    }
    pub(crate) fn original_config_inputs(&self, parent: usize) -> Result<BudgetVec<u8>> {
        self.core()?.original_config_inputs(parent)
    }
    pub(crate) fn authorize(&mut self, encoded: &[u8]) -> Result<()> {
        self.core_mut()?.authorize(encoded)
    }
    pub(crate) fn authorize_hpo(&mut self, encoded: &[u8]) -> Result<BudgetVec<u8>> {
        self.core_mut()?.authorize_hpo(encoded)
    }
    pub(crate) fn recheck(&self) -> Result<()> {
        self.core()?.recheck()
    }
    pub(crate) fn export_cohort(&mut self) -> Result<CardInstancePayload> {
        self.core()?.export_preconditions()?;
        if let Err(error) = self.core()?.inputs.control.check() {
            self.core_mut()?.phase = Phase::Failed;
            return Err(error);
        }
        self.core_mut()?.ensure_numeric_guard()?;
        let mut lease = match self.budget.reserve(1024) {
            Ok(lease) => lease,
            Err(error) => {
                self.core_mut()?.phase = Phase::Failed;
                self.core_mut()?.abort_empty_numeric_guard();
                return Err(error);
            }
        };
        let (metadata, banks) = match self.core_mut()?.export() {
            Ok(value) => value,
            Err(error) => {
                drop(lease);
                self.core_mut()?.abort_empty_numeric_guard();
                return Err(error);
            }
        };
        if let Some(guard) = self.core()?.numeric_guard.as_ref() {
            lease.retain_numeric_guard(guard.clone());
        }
        let guard = Arc::new(());
        self.payload = Arc::downgrade(&guard);
        Ok(CardInstancePayload {
            metadata,
            banks,
            _guard: guard,
            _lease: lease,
        })
    }
    pub(crate) fn export_recorded_cohort(&mut self, index: usize) -> Result<CardInstancePayload> {
        self.final_slot_operation(CardInstanceOperation::Recorded(index))?;
        self.original_episode_idle()?;
        self.core()?.recorded_export_preconditions(index)?;
        if self.payload.upgrade().is_some() {
            return Err(Error::contract("outstanding exported packet"));
        }
        if let Err(error) = self.core()?.inputs.control.check() {
            self.core_mut()?.phase = Phase::Failed;
            return Err(error);
        }
        self.core_mut()?.ensure_numeric_guard()?;
        let mut lease = match self.budget.reserve(1024) {
            Ok(lease) => lease,
            Err(error) => {
                self.core_mut()?.phase = Phase::Failed;
                self.core_mut()?.abort_empty_numeric_guard();
                return Err(error);
            }
        };
        let (metadata, banks) = match self.core_mut()?.export_recorded_cohort(index) {
            Ok(value) => value,
            Err(error) => {
                drop(lease);
                self.core_mut()?.abort_empty_numeric_guard();
                return Err(error);
            }
        };
        if let Some(guard) = self.core()?.numeric_guard.as_ref() {
            lease.retain_numeric_guard(guard.clone());
        }
        let guard = Arc::new(());
        self.payload = Arc::downgrade(&guard);
        Ok(CardInstancePayload {
            metadata,
            banks,
            _guard: guard,
            _lease: lease,
        })
    }
    pub(crate) fn retire_payload(&mut self) -> Result<()> {
        if self.payload.upgrade().is_some() {
            return Err(Error::contract(
                "exported packet views must retire before decoder reconstruction",
            ));
        }
        self.core_mut()?.retire_payload()
    }
    pub(crate) fn bind_numeric(&mut self, encoded: &[u8]) -> Result<()> {
        self.core_mut()?.bind_numeric(encoded)
    }
    pub(crate) fn begin_update(&mut self) -> Result<()> {
        self.core_mut()?.begin_update()
    }
    pub(crate) fn used_bytes(&self) -> usize {
        self.budget.used()
    }
    pub(crate) fn peak_bytes(&self) -> usize {
        self.budget.peak()
    }
    fn retire_core(&mut self) -> Result<()> {
        let core = self.core.take();
        if let Some(retirement) = self.retirement.take() {
            retirement.retire(core)
        } else {
            Ok(())
        }
    }
    pub(crate) fn prepare_commit(&mut self) -> Result<&[u8]> {
        let receipt = self.core_mut()?.prepare_commit3()?;
        // No receipt is exposed before the reportable cleanup worker has joined.
        self.retire_core()?;
        self.receipt = Some(receipt);
        Ok(self.receipt.as_ref().unwrap())
    }
    pub(crate) fn finalize_read(&mut self) -> Result<&[u8]> {
        if !self.final_slot_retired() {
            return Err(Error::contract(
                "final64 loan must retire before parent finalize",
            ));
        }
        let receipt = self.core_mut()?.finalize_read3()?;
        self.retire_core()?;
        self.receipt = Some(receipt);
        Ok(self.receipt.as_ref().unwrap())
    }
    pub(crate) fn close(&mut self) -> Result<()> {
        let episode_result = self.close_original_episode();
        if !self.original_episode_retired() {
            return episode_result;
        }
        let final_result = episode_result.and(self.close_final_validation());
        if !self.final_slot_retired() {
            return final_result.and(Err(Error::contract(
                "final64 collection retirement unconfirmed",
            )));
        }
        if self.core.is_none() {
            return final_result;
        }
        let result = final_result.and(self.core_mut()?.close());
        if self.core.as_ref().is_some_and(CohortSession::is_closed) {
            let retired = self.retire_core();
            return result.and(retired);
        }
        result
    }
}
impl Drop for CardInstanceNativeSession {
    fn drop(&mut self) {
        let _ = self.close();
        self.retire_original_on_drop();
        if self.core.is_some() {
            let _ = self.retire_core();
        }
    }
}

fn live_artifact_count_allowed(
    format: crate::structured_controls::SourceFormat,
    count: usize,
) -> bool {
    count > 0
        && count
            <= if format.hpo() {
                i64::MAX as usize
            } else {
                1000
            }
}
fn live_artifact_name_allowed(
    format: crate::structured_controls::SourceFormat,
    name: &str,
    index: usize,
    count: usize,
) -> bool {
    if !format.hpo() {
        return name == format!("{index:06}");
    }
    // Physical inventories are unique and lexically sorted. Above six digits
    // that order differs from generation order; canonical range membership
    // still proves the complete 0..count prefix without another allocation.
    name.parse::<usize>()
        .is_ok_and(|generation| generation < count && name == format!("{generation:06}"))
}

#[cfg(test)]
mod card_instance_permit_tests {
    use super::*;
    fn fixture() -> (PathBuf, Arc<Budget>, Control) {
        let root = std::env::temp_dir().join(format!(
            "gf-card-permit-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("numeric/batch-000000/nested")).unwrap();
        fs::write(
            root.join("numeric/batch-000000/nested/canonical-opaque.bin"),
            b"not JSON, hashed only",
        )
        .unwrap();
        (
            root,
            Budget::new(64 * 1024 * 1024),
            Control::new(crate::budget::monotonic() + 30.).unwrap(),
        )
    }
    #[test]
    fn hpo_owned_artifact_append_crosses_legacy_thousand_without_a_new_abi() {
        let (root, b, c) = fixture();
        let artifacts = root.join("artifacts");
        fs::create_dir(&artifacts).unwrap();
        let names = (0..1001).map(|g| format!("{g:06}")).collect::<Vec<_>>();
        for name in &names {
            fs::create_dir(artifacts.join(name)).unwrap();
        }
        let mut source = Sources::new(&root, b, c, 1024 * 1024).unwrap();
        source.inventory(&artifacts, names, true).unwrap();
        use crate::structured_controls::SourceFormat;
        assert!(!live_artifact_count_allowed(SourceFormat::Corrected, 1001));
        assert!(live_artifact_count_allowed(
            SourceFormat::Hpo,
            i64::MAX as usize
        ));
        assert!(!live_artifact_count_allowed(
            SourceFormat::PackedHpo,
            i64::MAX as usize + 1
        ));
        assert!(source.begin_live_output(SourceFormat::Corrected).is_err());
        assert!(source.begin_live_output(SourceFormat::Hpo).is_ok());
        assert!(live_artifact_name_allowed(
            SourceFormat::Hpo,
            "1000000",
            100001,
            1000001
        ));
        assert!(!live_artifact_name_allowed(
            SourceFormat::Corrected,
            "1000000",
            100001,
            1000001
        ));
        assert!(!live_artifact_name_allowed(
            SourceFormat::Hpo,
            "01000000",
            100001,
            1000001
        ));
        assert!(!live_artifact_name_allowed(
            SourceFormat::PackedHpo,
            "1000001",
            100001,
            1000001
        ));
        fs::create_dir(artifacts.join("001001")).unwrap();
        source.recheck().unwrap();
        fs::create_dir(artifacts.join("001002")).unwrap();
        assert!(source.recheck().is_err());
        drop(source);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn card_instance_numeric_append_pins_complete_prior_subtrees_and_exact_target() {
        let (root, b, c) = fixture();
        let mut source = Sources::new(&root, b, c, 1024 * 1024).unwrap();
        let permit = source.begin_numeric_output(1).unwrap();
        let target = root.join("numeric/batch-000001");
        fs::create_dir(&target).unwrap();
        fs::write(target.join("raw.bin"), b"new").unwrap();
        source.recheck().unwrap();
        assert_eq!(source.bind_numeric_output(permit).unwrap(), target);
        source.pin_tree(&target).unwrap();
        source.recheck().unwrap();
        assert!(source.bind_numeric_output(permit).is_err());
        fs::write(target.join("raw.bin"), b"late").unwrap();
        assert!(source.recheck().is_err());
        drop(source);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn card_instance_numeric_append_rejects_preexisting_foreign_replaced_and_changed_prior() {
        for mutation in 0..5 {
            let (root, b, c) = fixture();
            let mut source = Sources::new(&root, b, c, 1024 * 1024).unwrap();
            if mutation == 0 {
                fs::create_dir(root.join("numeric/batch-000001")).unwrap();
                assert!(source.begin_numeric_output(1).is_err());
            } else {
                source.begin_numeric_output(1).unwrap();
                match mutation {
                    1 => {
                        fs::write(root.join("numeric/foreign"), b"preserve").unwrap();
                    }
                    2 => {
                        fs::write(
                            root.join("numeric/batch-000000/nested/canonical-opaque.bin"),
                            b"changed",
                        )
                        .unwrap();
                    }
                    3 => {
                        fs::rename(root.join("numeric"), root.join("held-numeric")).unwrap();
                        fs::create_dir(root.join("numeric")).unwrap();
                    }
                    _ => {
                        fs::write(root.join("numeric/batch-000000/nested/late"), b"foreign")
                            .unwrap();
                    }
                }
                assert!(source.recheck().is_err());
            }
            drop(source);
            fs::remove_dir_all(root).unwrap();
        }
    }
    #[test]
    fn card_instance_numeric_append_admits_before_allocation_and_cancel_preserves_owner() {
        let (root, _, c) = fixture();
        let mut tiny = Sources::new(&root, Budget::new(1), c.clone(), 1024 * 1024).unwrap();
        assert!(matches!(
            tiny.begin_numeric_output(1),
            Err(Error::ResourceLimit(_))
        ));
        drop(tiny);
        let mut source =
            Sources::new(&root, Budget::new(64 * 1024 * 1024), c.clone(), 1024 * 1024).unwrap();
        c.cancel();
        assert_eq!(source.begin_numeric_output(1).err(), Some(Error::Cancelled));
        assert!(!root.join("numeric/batch-000001").exists());
        drop(source);
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod card_instance_actual_tests {
    use super::*;
    use sha2::Digest;
    #[test]
    #[ignore = "opt-in genuine approved immutable MLP parent; NOT RUN when originals unavailable"]
    fn actual_card_instance_summary3_correspondence_and_all_staged_arrays_match_python() {
        let b = Budget::new(8usize * 1024 * 1024 * 1024);
        let c = Control::new(crate::budget::monotonic() + 360.).unwrap();
        let reference = crate::json::parse(
            include_bytes!("../tests/fixtures/card_instance_actual_mlp_reference.json"),
            &b,
            &c,
        )
        .unwrap();
        let root = PathBuf::from(reference.root["parent"].string().unwrap());
        assert!(
            root.is_dir(),
            "approved genuine originals absent: opt-in check cannot pass"
        );
        let plan = json::object([
            ("schema_version", json::integer(3)),
            ("mode", json::text("historical_parent")),
            (
                "parents",
                json::array([json::object([
                    ("directory", json::text(root.to_str().unwrap())),
                    ("require_complete", Value::Bool(true)),
                ])]),
            ),
            ("collections", json::array([])),
        ]);
        let encoded = json::canonical(&plan, &b, &c).unwrap();
        let mut session = CardInstanceNativeSession::open(
            &encoded,
            b.clone(),
            c.clone(),
            64 * 1024 * 1024,
            8 * 1024 * 1024,
            256 * 1024 * 1024,
            16 * 1024 * 1024,
        )
        .unwrap();
        let summary = session.control_summary().unwrap();
        let summary = json::parse(&summary, &b, &c).unwrap();
        let summary_sha = crate::structured_controls::digest(&summary.root, &b, &c).unwrap();
        assert_eq!(summary.root["schema_version"], json::integer(3));
        assert_eq!(
            summary.root["parent"]["run_sha256"],
            reference.root["run_sha256"]
        );
        assert!(summary.root.get("binding").is_none());
        assert!(summary.root["parent"].get("journal5_validated").is_none());
        assert_eq!(
            summary.root["parent"]["journal6_validated"],
            Value::Bool(false)
        );
        for (source, expected) in summary.root["collections"]
            .array()
            .unwrap()
            .iter()
            .zip(reference.root["collections"].array().unwrap())
        {
            assert_eq!(source["source_proof"], expected["proof"]);
            assert_eq!(source["correspondence"], expected["correspondence"]);
        }
        assert_eq!(summary.root["collections"].array().unwrap().len(), 2);
        let ack = json::object([
            ("schema_version", json::integer(3)),
            ("summary_sha256", json::text(&summary_sha)),
            ("parent", reference.root["parent_ack"].clone()),
            (
                "collections",
                json::array(
                    reference.root["collections"]
                        .array()
                        .unwrap()
                        .iter()
                        .map(|v| v["ack"].clone()),
                ),
            ),
        ]);
        let encoded_ack = json::canonical(&ack, &b, &c).unwrap();
        session.authorize(&encoded_ack).unwrap();
        assert!(session.authorize(&encoded_ack).is_err());
        assert!(session.export_cohort().is_err());
        assert!(session.begin_update().is_err());
        assert!(session.prepare_commit().is_err());
        let inputs = session.original_config_inputs(0).unwrap();
        let query = json::parse(&inputs, &b, &c).unwrap();
        assert_eq!(query.root["summary_sha256"], json::text(&summary_sha));
        drop(query);
        drop(inputs);
        // Pure raw staging over the SAME approved original parent is a
        // diagnostic; the historical session still cannot export/update.
        let state = session.core().unwrap().inputs.parents[0].state.root.clone();
        let mut stage = crate::structured_staging::Stage::new(
            b.clone(),
            256 * 1024 * 1024,
            16 * 1024 * 1024,
            c.clone(),
        )
        .unwrap();
        let source = crate::structured_source::inspect_card_instance_collection(
            &root.join("collections/batch-000000"),
            (&root, &state),
            &crate::structured_source::CollectionLimits {
                source_bytes: 64 * 1024 * 1024,
                record_bytes: 8 * 1024 * 1024,
            },
            Some(&mut stage),
            b.clone(),
            c.clone(),
        )
        .unwrap();
        let expected = &reference.root["collections"][0];
        assert_eq!(source.proof.root, expected["proof"]);
        assert_eq!(
            source.digests,
            expected["correspondence"]["episode_digests"]
                .array()
                .unwrap()
                .iter()
                .map(|v| v.string().unwrap().to_owned())
                .collect::<Vec<_>>()
        );
        let mut episodes = source.episodes.root.clone();
        for ep in episodes.as_array_mut().unwrap() {
            ep["has_targets"] = Value::Bool(!ep["frames"].array().unwrap().is_empty());
        }
        assert_eq!(episodes, expected["episodes"]);
        let descriptors = stage.descriptors().unwrap();
        let banks = stage.export_retiring_disk().unwrap();
        assert_eq!(
            descriptors.array().unwrap().len(),
            expected["array_hashes"].object().unwrap().len()
        );
        for descriptor in descriptors.array().unwrap() {
            let key = descriptor["key"].string().unwrap();
            let oracle = &expected["array_hashes"][key];
            assert_eq!(descriptor["dtype"], oracle["dtype"]);
            assert_eq!(descriptor["shape"], oracle["shape"]);
            let index = descriptor["bank_index"]
                .integer(0, i32::MAX as i128)
                .unwrap() as usize;
            let first = descriptor["element_offset"]
                .integer(0, i64::MAX as i128)
                .unwrap() as usize;
            let count = descriptor["element_count"]
                .integer(0, i64::MAX as i128)
                .unwrap() as usize;
            let mut hash = sha2::Sha256::new();
            match descriptor["dtype"].string().unwrap() {
                "uint8" => hash.update(&banks.bytes[index][first..first + count]),
                "int64" => {
                    for v in &banks.integers[index][first..first + count] {
                        hash.update(v.to_le_bytes());
                    }
                }
                "float32" => {
                    for v in &banks.floats[index][first..first + count] {
                        hash.update(v.to_bits().to_le_bytes());
                    }
                }
                _ => panic!("unknown dtype"),
            }
            assert_eq!(
                format!("{:x}", hash.finalize()),
                oracle["sha256"].string().unwrap(),
                "array {key}"
            );
        }
        for owner in source.sources.iter() {
            owner.recheck().unwrap();
        }
        drop(source);
        drop(banks);
        stage.close().unwrap();
        drop(stage);
        session.recheck().unwrap();
        let receipt = session.finalize_read().unwrap();
        assert!(!receipt.is_empty());
        assert!(session.recheck().is_err());
        session.close().unwrap(); // memory-only after joined preparation
        assert!(session.peak_bytes() >= session.used_bytes());
    }
}

#[cfg(test)]
mod card_instance_lifecycle_primitives {
    use super::*;
    fn setup() -> (PathBuf, CardInstanceNativeSession, Arc<Budget>, Control) {
        let root = std::env::temp_dir().join(format!(
            "gf-unadmitted-card-kernel-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("numeric")).unwrap();
        fs::create_dir_all(root.join("artifacts/000000")).unwrap();
        fs::create_dir(root.join("unadmitted-originals")).unwrap();
        fs::write(
            root.join("original-control.bin"),
            b"private control primitive, not Run3",
        )
        .unwrap();
        fs::write(
            root.join("unadmitted-originals/original.bin"),
            b"owned bytes, not C5/J6",
        )
        .unwrap();
        let b = Budget::new(512 * 1024 * 1024);
        let c = Control::new(crate::budget::monotonic() + 60.).unwrap();
        let model_fixture = json::parse(
            include_bytes!("../tests/fixtures/card_instance_models3.json"),
            &b,
            &c,
        )
        .unwrap();
        let example = &model_fixture.root["cases"][0];
        let mut declaration = example["manifest"].clone();
        for key in [
            "schema_version",
            "kind",
            "model",
            "architecture_identity",
            "action_config_hash",
            "producer",
            "weights",
            "action_contract",
            "purpose",
            "synthetic_capacity",
            "layout_signature",
        ] {
            declaration.as_object_mut().unwrap().remove(key);
        }
        let mut manifest =
            crate::structured_controls::binding3_value(&example["manifest"], &c).unwrap();
        manifest["scope"] = json::text("UNADMITTED private primitive, not Collection5");
        manifest["config"] = json::object([("actions", example["actions"].clone())]);
        manifest["action_config_hash"] = example["manifest"]["action_config_hash"].clone();
        let state = json::object([
            (
                "kind",
                json::text("UNADMITTED, not Run3 or learning evidence"),
            ),
            ("optimizer_steps", json::integer(0)),
        ]);
        let episodes = json::array([json::object([
            ("actor", json::integer(1)),
            ("episode", json::text("not an original episode")),
            (
                "frames",
                json::array([json::object([(
                    "scope",
                    json::text("not an admitted original frame"),
                )])]),
            ),
            ("end_kind", json::text("unadmitted-component")),
            ("outcome", Value::Null),
            ("reason", json::text("not a natural game")),
        ])]);
        let mut stage =
            crate::structured_staging::Stage::new(b.clone(), 1024 * 1024, 1024 * 1024, c.clone())
                .unwrap();
        for (key, dtype, value) in [
            (
                "e.0.selected",
                crate::structured_staging::Dtype::I64,
                json::array([json::integer(0)]),
            ),
            (
                "e.0.old_logprob",
                crate::structured_staging::Dtype::F32,
                json::array([json::float(-1.).unwrap()]),
            ),
            (
                "e.0.old_value",
                crate::structured_staging::Dtype::F32,
                json::array([json::float(0.).unwrap()]),
            ),
        ] {
            stage.array(key, dtype, &value).unwrap();
        }
        let inputs = crate::structured_source::OriginalInputs::unadmitted_component_lifecycle(
            &root,
            &state,
            &manifest,
            &example["model"],
            &declaration,
            &episodes,
            stage,
            b.clone(),
            c.clone(),
        )
        .unwrap();
        let controls = crate::structured_source::Controls {
            summary: crate::structured_source::doc(
                &json::object([("scope", json::text("not an authorization summary"))]),
                &b,
                &c,
            )
            .unwrap(),
            authorization: crate::structured_source::doc(
                &json::object([("scope", json::text("cannot authorize a Source"))]),
                &b,
                &c,
            )
            .unwrap(),
            declarations: vec![Some(
                crate::structured_source::doc(&declaration, &b, &c).unwrap(),
            )],
            correspondences: vec![Some(
                crate::structured_source::doc(
                    &json::object([("scope", json::text("not Source correspondence"))]),
                    &b,
                    &c,
                )
                .unwrap(),
            )],
            fingerprint: "not Source authority".into(),
        };
        let core = CohortSession {
            inputs,
            controls,
            phase: Phase::Authorized,
            numeric_permit: None,
            numeric: None,
            expected_arrays: None,
            recorded_episodes: None,
            numeric_guard: None,
        };
        let retirement = crate::retirement::Retirement::start(&b, |core: CohortSession| {
            drop(core);
            Ok(())
        })
        .unwrap();
        let session = CardInstanceNativeSession {
            core: Some(core),
            retirement: Some(retirement),
            budget: b.clone(),
            payload: std::sync::Weak::new(),
            receipt: None,
            final_slot: None,
            original_episode: original_episode::State::new(),
        };
        assert!(!root.join("run.json").exists());
        assert!(!root.join("generations").exists());
        (root, session, b, c)
    }
    fn publish(
        root: &Path,
        payload: &CardInstancePayload,
        b: &Arc<Budget>,
        c: &Control,
        mutation: usize,
    ) -> BudgetVec<u8> {
        let exported = json::parse(payload.metadata(), b, c).unwrap();
        let io = json::parse(
            include_bytes!("../tests/fixtures/card_instance_unadmitted_numeric_bundle.json"),
            b,
            c,
        )
        .unwrap();
        let target = root.join("numeric/batch-000000");
        fs::create_dir(&target).unwrap();
        for (name, hex) in io.root["files_hex"].object().unwrap() {
            let hex = hex.string().unwrap();
            let data = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect::<Vec<_>>();
            fs::write(target.join(name), data).unwrap();
        }
        let mut manifest = exported.root["metadata"].clone();
        manifest["schema_version"] = json::integer(4);
        manifest["kind"] = json::text("structured-rollout");
        manifest["episodes"][0]["has_targets"] = Value::Bool(true);
        manifest["data"] = io.root["bundle"].clone();
        if mutation == 1 {
            manifest["policy_version"] = json::text("foreign");
        }
        if mutation == 2 {
            manifest["token_signature"] = json::text(&"0".repeat(64));
        }
        if mutation == 3 {
            manifest["episodes"][0]["frames"][0]["foreign"] = Value::Bool(true);
        }
        let raw = json::canonical(&manifest, b, c).unwrap();
        let mut data = raw.to_vec();
        data.push(b'\n');
        fs::write(target.join("rollout.json"), &data).unwrap();
        use sha2::Digest;
        let hash = format!("{:x}", sha2::Sha256::digest(&data));
        let mut files = io.root["bundle"]["files"].clone();
        files["rollout.json"] = json::text(&hash);
        if mutation == 4 {
            fs::write(target.join("foreign"), b"preserved").unwrap();
        }
        let directory = if mutation == 5 {
            root.join("foreign-target")
        } else {
            target
        };
        json::canonical(
            &json::object([
                ("directory", json::text(directory.to_str().unwrap())),
                ("manifest_sha256", json::text(&hash)),
                ("files", files),
            ]),
            b,
            c,
        )
        .unwrap()
    }
    #[test]
    fn card_instance_unadmitted_numeric_kernel_snapshots_through_update_and_blocks_false_precommit()
    {
        let (root, mut session, b, c) = setup();
        let payload = session.export_cohort().unwrap();
        assert!(!root.join("numeric/batch-000000").exists());
        assert!(session.retire_payload().is_err());
        let binding = publish(&root, &payload, &b, &c, 0);
        drop(payload);
        session.retire_payload().unwrap();
        session.bind_numeric(&binding).unwrap();
        assert!(session.bind_numeric(&binding).is_err());
        session.begin_update().unwrap();
        session.recheck().unwrap();
        // No derived artifact/learning exists; the primitive must not publish
        // a prepared receipt merely because numeric files and phases are valid.
        assert!(session.prepare_commit().is_err());
        assert!(session.receipt.is_none());
        fs::write(
            root.join("numeric/batch-000000/rollout.json"),
            b"late mutation",
        )
        .unwrap();
        assert!(session.recheck().is_err());
        assert!(session.prepare_commit().is_err());
        assert!(session.close().is_err());
        assert!(session.core.is_none());
        drop(session);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn card_instance_unadmitted_numeric_kernel_refuses_wrong_context_inventory_and_target() {
        for mutation in 1..=5 {
            let (root, mut session, b, c) = setup();
            let payload = session.export_cohort().unwrap();
            let binding = publish(&root, &payload, &b, &c, mutation);
            drop(payload);
            session.retire_payload().unwrap();
            assert!(
                session.bind_numeric(&binding).is_err(),
                "mutation{mutation}"
            );
            assert!(session.begin_update().is_err());
            assert!(session.receipt.is_none());
            let _ = session.close();
            drop(session);
            fs::remove_dir_all(root).unwrap();
        }
    }
    #[test]
    fn card_instance_unadmitted_packet_lifetime_and_cancel_keep_owned_buffers_and_cleanup() {
        let (root, mut session, b, c) = setup();
        let payload = session.export_cohort().unwrap();
        assert!(session.retire_payload().is_err());
        assert!(session.export_cohort().is_err());
        assert_eq!(payload.banks().integers[0][0], 0);
        c.cancel();
        assert_eq!(session.recheck().err(), Some(Error::Cancelled));
        assert!(session.close().is_err());
        assert!(session.core.is_none());
        // Core/spool retirement cannot invalidate the independent owned bank.
        assert_eq!(payload.banks().floats[0][0], -1.);
        let retained = b.used();
        drop(payload);
        assert!(b.used() < retained);
        drop(session);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn card_instance_unadmitted_export_failure_is_not_retryable_after_source_guard() {
        let (root, mut session, _, _) = setup();
        fs::write(root.join("original-control.bin"), b"mutated original").unwrap();
        assert!(session.export_cohort().is_err());
        assert!(session.export_cohort().is_err());
        assert!(session.receipt.is_none());
        let _ = session.close();
        drop(session);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn cancelled_cleanup_closes_actual_native_core_without_any_original_source_io() {
        let (root, mut session, budget, control) = setup();
        SOURCE_IO.with(|count| count.set(0));
        control.cancel();
        assert!(matches!(session.close(), Err(Error::Cancelled)));
        assert!(session.core.is_none());
        assert_eq!(SOURCE_IO.with(|count| count.get()), 0);
        session.close().unwrap();
        assert_eq!(budget.used(), 0);
        drop(session);
        fs::remove_dir_all(root).unwrap();
    }
    fn assert_failed(session: &CardInstanceNativeSession) {
        assert!(
            session
                .core
                .as_ref()
                .is_some_and(|core| core.phase == Phase::Failed)
        );
        assert!(session.receipt.is_none());
    }
    fn ready_numeric(
        root: &Path,
        session: &mut CardInstanceNativeSession,
        b: &Arc<Budget>,
        c: &Control,
    ) {
        let payload = session.export_cohort().unwrap();
        let binding = publish(root, &payload, b, c, 0);
        drop(payload);
        session.retire_payload().unwrap();
        session.bind_numeric(&binding).unwrap();
    }
    #[test]
    fn card_instance_authorize_and_packet_admission_failures_cannot_retry_after_repair() {
        let (root, mut session, b, c) = setup();
        {
            let core = session.core_mut().unwrap();
            core.phase = Phase::Scanned;
            core.controls.summary.root["authorization_max_bytes"] = json::integer(4096);
        }
        let valid =
            json::canonical(&session.core().unwrap().controls.authorization.root, &b, &c).unwrap();
        assert!(
            session
                .authorize(br#"{"scope":"wrong acknowledgement"}"#)
                .is_err()
        );
        assert_failed(&session);
        assert_eq!(session.authorize(&valid).err(), Some(Error::Closed));
        session.close().unwrap();
        fs::remove_dir_all(root).unwrap();
        let (root, mut session, b, _) = setup();
        let held = b.reserve(b.limit() - b.used()).unwrap();
        assert!(matches!(
            session.export_cohort(),
            Err(Error::ResourceLimit(_))
        ));
        assert_failed(&session);
        drop(held);
        assert_eq!(session.export_cohort().err(), Some(Error::Closed));
        assert!(!root.join("numeric/batch-000000").exists());
        session.close().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn card_instance_bind_and_begin_failures_cannot_retry_after_budget_repair() {
        let (root, mut session, b, c) = setup();
        let payload = session.export_cohort().unwrap();
        let binding = publish(&root, &payload, &b, &c, 0);
        drop(payload);
        session.retire_payload().unwrap();
        let held = b.reserve(b.limit() - b.used()).unwrap();
        assert!(matches!(
            session.bind_numeric(&binding),
            Err(Error::ResourceLimit(_))
        ));
        assert_failed(&session);
        drop(held);
        assert_eq!(session.bind_numeric(&binding).err(), Some(Error::Closed));
        session.close().unwrap();
        fs::remove_dir_all(root).unwrap();
        let (root, mut session, b, c) = setup();
        ready_numeric(&root, &mut session, &b, &c);
        let held = b.reserve(b.limit() - b.used()).unwrap();
        assert!(matches!(
            session.begin_update(),
            Err(Error::ResourceLimit(_))
        ));
        assert_failed(&session);
        drop(held);
        assert_eq!(session.begin_update().err(), Some(Error::Closed));
        session.close().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn card_instance_retire_cleanup_failure_keeps_owner_and_only_cleanup_can_retry() {
        let (root, mut session, b, c) = setup();
        let payload = session.export_cohort().unwrap();
        drop(payload);
        let replacement =
            crate::structured_staging::Stage::new(b.clone(), 1024 * 1024, 1024 * 1024, c).unwrap();
        let spool = replacement.unadmitted_test_directory().to_owned();
        let foreign = spool.join("foreign");
        fs::write(&foreign, b"preserve").unwrap();
        session.core_mut().unwrap().inputs.stage = Some(replacement); // unadmitted cleanup primitive only
        assert!(session.retire_payload().is_err());
        assert_failed(&session);
        assert!(session.core().unwrap().inputs.stage.is_some());
        assert_eq!(fs::read(&foreign).unwrap(), b"preserve");
        fs::remove_file(&foreign).unwrap();
        assert_eq!(session.retire_payload().err(), Some(Error::Closed));
        session.close().unwrap();
        assert!(!spool.exists());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn card_instance_retire_cancel_error_keeps_priority_when_cleanup_also_fails() {
        let (root, mut session, b, c) = setup();
        let payload = session.export_cohort().unwrap();
        drop(payload);
        let replacement =
            crate::structured_staging::Stage::new(b, 1024 * 1024, 1024 * 1024, c.clone()).unwrap();
        let spool = replacement.unadmitted_test_directory().to_owned();
        let foreign = spool.join("foreign");
        fs::write(&foreign, b"preserve").unwrap();
        session.core_mut().unwrap().inputs.stage = Some(replacement);
        c.cancel();
        assert_eq!(session.retire_payload().err(), Some(Error::Cancelled));
        assert_failed(&session);
        match session.close().unwrap_err() {
            Error::Cleanup { primary, cleanup } => {
                assert_eq!(*primary, Error::Cancelled);
                assert!(cleanup.contains("foreign"));
            }
            other => panic!("lost initiating cancellation: {other:?}"),
        }
        assert!(session.core.is_some());
        assert!(foreign.exists());
        fs::remove_file(foreign).unwrap();
        assert_eq!(session.close().err(), Some(Error::Cancelled));
        assert!(session.core.is_none());
        assert!(!spool.exists());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn card_instance_historical_final_guard_failure_cannot_finalize_after_budget_repair() {
        let (root, mut session, b, _) = setup();
        session.core_mut().unwrap().inputs.mode = crate::structured_source::ScanMode::Historical;
        let held = b.reserve(b.limit() - b.used()).unwrap();
        assert!(matches!(
            session.finalize_read(),
            Err(Error::ResourceLimit(_))
        ));
        assert_failed(&session);
        drop(held);
        assert_eq!(session.finalize_read().err(), Some(Error::Closed));
        session.close().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
    fn write_json_fixture(path: &Path, value: &Value, b: &Arc<Budget>, c: &Control) -> String {
        use sha2::Digest;
        let mut raw = json::canonical(value, b, c).unwrap().to_vec();
        raw.push(b'\n');
        fs::write(path, &raw).unwrap();
        format!("{:x}", sha2::Sha256::digest(&raw))
    }
    fn materialize_metadata_bundle(path: &Path, bundle: &mut Value, b: &Arc<Budget>, c: &Control) {
        use sha2::Digest;
        // Metadata/file-ownership primitive, deliberately not NPZ/learner state.
        // There is no Run3/C5/J6/issuer here and a sealed NumPy reader would reject it.
        for index in 0..bundle["shards"].array().unwrap().len() {
            let name = bundle["shards"][index]["file"].string().unwrap().to_owned();
            let raw = b"unadmitted ownership bytes, not tensors";
            let sha = format!("{:x}", sha2::Sha256::digest(raw));
            fs::write(path.join(&name), raw).unwrap();
            bundle["files"][&name] = json::text(&sha);
            bundle["shards"][index]["metadata"]["sha256"] = json::text(&sha);
            let side = name.replace(".npz", ".json");
            let hash = write_json_fixture(
                &path.join(&side),
                &bundle["shards"][index]["metadata"],
                b,
                c,
            );
            bundle["files"][&side] = json::text(&hash);
        }
    }
    fn late_metadata_pair(
        root: &Path,
        session: &mut CardInstanceNativeSession,
        b: &Arc<Budget>,
        c: &Control,
    ) {
        let all = json::parse(
            include_bytes!("../tests/fixtures/card_instance_models3.json"),
            b,
            c,
        )
        .unwrap();
        let example = &all.root["cases"][0];
        let mut configured = example["model"].clone();
        configured["checkpoint"] = Value::Null;
        configured["inference"] = example["inference"].clone();
        configured["cuda_visible_devices"] = json::text("");
        let state = &mut session.core_mut().unwrap().inputs.parents[0].state.root;
        state["effective_config"] = json::object([
            ("actions", example["actions"].clone()),
            ("model", configured),
        ]);
        state["device"] = json::text("cpu");
        state["resume_identity"] = json::text(&"0".repeat(64));
        state["execution"] = json::object([(
            "learner",
            json::object([
                ("binding", example["inference"].clone()),
                ("cuda_visible_devices", json::text("")),
            ]),
        )]);
        state["action_contract"] = example["manifest"]["action_contract"].clone();
        let path = root.join("artifacts/000001");
        fs::create_dir(&path).unwrap();
        let mut inference = example["manifest"].clone();
        materialize_metadata_bundle(&path, &mut inference["weights"], b, c);
        let inference_hash = write_json_fixture(&path.join("inference.json"), &inference, b, c);
        let mut resume = session.core().unwrap().controls.declarations[0]
            .as_ref()
            .unwrap()
            .root
            .clone();
        for key in [
            "action_contract",
            "rule_signature",
            "layout_signature",
            "purpose",
            "synthetic_capacity",
            "action_config_hash",
            "producer",
        ] {
            resume[key] = inference[key].clone();
        }
        resume["schema_version"] = json::integer(4);
        resume["kind"] = json::text("structured-resume");
        resume["inference_digest"] = json::text(&inference_hash);
        resume["model"] = example["model"].clone();
        resume["binding"] = example["inference"].clone();
        resume["resume_identity"] = json::text(&"0".repeat(64));
        resume["optimizer_steps"] = json::integer(4);
        resume["optimizer_recipe"]=json::parse(br#"{"name":"Adam","lr":0.0003,"betas":[0.9,0.999],"eps":1e-8,"weight_decay":0,"amsgrad":false,"maximize":false,"foreach":null,"capturable":false,"differentiable":false,"fused":null,"decoupled_weight_decay":false}"#,b,c).unwrap().root.clone();
        resume["parameter_states"] = Value::Object(
            example["shapes"]
                .object()
                .unwrap()
                .keys()
                .map(|k| (k.clone(), json::object([("present", Value::Bool(false))])))
                .collect(),
        );
        resume["cuda_rng_count"] = json::integer(0);
        resume["shuffle_rng"] = json::parse(
            br#"{"bit_generator":"PCG64","state":{"state":1,"inc":1},"has_uint32":0,"uinteger":0}"#,
            b,
            c,
        )
        .unwrap()
        .root
        .clone();
        resume["state"]=json::parse(br#"{"schema_version":1,"prefix":"resume","shards":[{"file":"resume-000000.npz","metadata":{"sha256":"0000000000000000000000000000000000000000000000000000000000000000","arrays":{"rng.torch_cpu":{"dtype":"uint8","shape":[1],"bytes":1}}}}],"files":{"resume-000000.npz":"0000000000000000000000000000000000000000000000000000000000000000","resume-000000.json":"0000000000000000000000000000000000000000000000000000000000000000"},"tensor_count":1,"decoded_bytes":1}"#,b,c).unwrap().root.clone();
        materialize_metadata_bundle(&path, &mut resume["state"], b, c);
        write_json_fixture(&path.join("resume.json"), &resume, b, c);
    }
    #[test]
    fn card_instance_precommit_failure_cannot_accept_late_native_valid_metadata_pair() {
        let (root, mut session, b, c) = setup();
        ready_numeric(&root, &mut session, &b, &c);
        session.begin_update().unwrap();
        assert!(matches!(session.prepare_commit(), Err(Error::Io(_))));
        assert_failed(&session);
        late_metadata_pair(&root, &mut session, &b, &c);
        {
            let parent = &mut session.core_mut().unwrap().inputs.parents[0];
            crate::structured_controls::verify_derived_artifact3(
                &mut parent.sources,
                &parent.state.root,
                0,
                4,
            )
            .unwrap();
        }
        assert_eq!(session.prepare_commit().err(), Some(Error::Closed));
        assert!(session.receipt.is_none());
        session.close().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod card_instance_plan_tests {
    use super::*;
    #[test]
    fn card_instance_plan3_cardinalities_versions_and_limits_fail_before_source_io() {
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        for raw in [
            br#"{"schema_version":2,"mode":"historical_parent","parents":[{"directory":"/not-an-original","require_complete":false}],"collections":[]}"#.as_slice(),
            br#"{"schema_version":3,"mode":"historical_parent","parents":[],"collections":[]}"#.as_slice(),
            br#"{"schema_version":3,"mode":"historical_parent","parents":[{"directory":"/not-an-original","require_complete":false}],"collections":[{"directory":"/not-a-collection","parent_index":0}]}"#.as_slice(),
            br#"{"schema_version":3,"mode":"live_cohort","parents":[{"directory":"/not-an-original","require_complete":false}],"collections":[{"directory":"/not-a-collection","parent_index":0}],"cohort_collection_index":1}"#.as_slice(),
        ] {
            let b=Budget::new(64*1024*1024);
            let result=CardInstanceNativeSession::open(raw,b.clone(),c.clone(),64*1024*1024,8*1024*1024,1024*1024,1024*1024);
            assert!(matches!(result,Err(Error::Contract(_))));assert_eq!(b.used(),0,"failed constructor must join empty retirement owner");
        }
        let b = Budget::new(64 * 1024 * 1024);
        assert!(matches!(
            CardInstanceNativeSession::open(b"{}", b.clone(), c, 0, 0, 0, 0),
            Err(Error::ResourceLimit(_))
        ));
        assert_eq!(b.used(), 0);
    }
}

/// Closed FFI accounting admission; never supplied by a Python recipe flag.
#[derive(Clone, Copy)]
pub(crate) enum CardInstanceOperation {
    Authorize,
    Export,
    Recorded(usize),
    Retire,
    Bind,
    Begin,
    Prepare,
    FinalizeRead,
}
impl CardInstanceNativeSession {
    pub(crate) fn preflight(&self, op: CardInstanceOperation) -> Result<()> {
        self.final_slot_operation(op)?;
        let core = self.core()?;
        match op {
            CardInstanceOperation::Export => {
                core.export_preconditions()?;
            }
            CardInstanceOperation::Recorded(index) => core.recorded_export_preconditions(index)?,
            _ => {
                core.transition_owner_check()?;
                let valid = match op {
                    CardInstanceOperation::Authorize => core.phase == Phase::Scanned,
                    CardInstanceOperation::Retire => {
                        matches!(core.phase, Phase::Exported | Phase::RecordedExported)
                    }
                    CardInstanceOperation::Bind => {
                        core.phase == Phase::PayloadRetired
                            && matches!(
                                core.inputs.mode,
                                crate::structured_source::ScanMode::Live { .. }
                            )
                    }
                    CardInstanceOperation::Begin => {
                        core.phase == Phase::NumericBound
                            && match core.inputs.mode {
                                crate::structured_source::ScanMode::Live { collection, .. } => {
                                    core.used_decisions(collection)? > 0
                                }
                                _ => false,
                            }
                    }
                    CardInstanceOperation::Prepare => {
                        core.phase == Phase::Updating
                            && matches!(
                                core.inputs.mode,
                                crate::structured_source::ScanMode::Live { .. }
                            )
                    }
                    CardInstanceOperation::FinalizeRead => {
                        core.phase == Phase::Authorized
                            && matches!(
                                core.inputs.mode,
                                crate::structured_source::ScanMode::Historical
                            )
                    }
                    _ => false,
                };
                if !valid {
                    core.check()?;
                    return Err(Error::contract(
                        "corrected FFI operation mode/phase differs",
                    ));
                }
            }
        }
        Ok(())
    }
    pub(crate) fn fail_admitted(&mut self) {
        if let Some(core) = self.core.as_mut() {
            core.phase = Phase::Failed;
        }
    }
    /// FFI garbage collection cannot report I/O; use the already-started owned
    /// retirement worker rather than dropping a deep core while holding the GIL.
    pub(crate) fn retire_on_drop(&mut self) {
        self.retire_original_on_drop();
        let _ = self.retire_core();
    }
}
impl CardInstancePayload {
    /// The private FFI bridge moves this guard/lease into its existing shared
    /// NumPy base-owner ExportLease; it must not release either before views.
    pub(crate) fn into_parts(
        self,
    ) -> (
        BudgetVec<u8>,
        crate::structured_staging::Banks,
        Arc<()>,
        crate::budget::Lease,
    ) {
        (self.metadata, self.banks, self._guard, self._lease)
    }
}
impl CardInstanceNativeSession {
    pub(crate) fn authorization_max_bytes(&self) -> Result<usize> {
        self.core()?
            .controls
            .summary
            .root
            .require("authorization_max_bytes")?
            .integer(1, 64i128 << 20)
            .map(|v| v as usize)
    }
}
impl CardInstanceNativeSession {
    pub(crate) fn is_closed(&self) -> bool {
        self.core.is_none()
    }
    pub(crate) fn take_receipt(&mut self) -> Result<BudgetVec<u8>> {
        self.receipt.take().ok_or(Error::Closed)
    }
}
