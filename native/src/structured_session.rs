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
    _lease: Lease,
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
            let (names, identity, _) = self.names(&item.path, &item.names, item.directories)?;
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
            if Identity::of(&fs::symlink_metadata(&item.path)?) != item.identity {
                return Err(Error::contract(
                    "structured original inventory changed after sweep",
                ));
            }
        }
        self.control.check()
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
