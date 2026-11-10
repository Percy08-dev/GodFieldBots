//! Original logical bytes, independent of Python and physical file metadata.
use crate::{
    budget::{Budget, BudgetVec, Control},
    error::{Error, Result},
    json::{self, Document, Value, ValueExt},
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

pub(crate) const FORMAT: &str = "python-original-bytes-v1";
pub(crate) const METADATA_LIMIT: usize = 8 * 1024 * 1024;
#[path = "logical_name_ranges.rs"]
mod names;
fn category_c(value: char) -> bool {
    let point = value as u32;
    let index = names::C_RANGES.partition_point(|&(start, _)| start <= point);
    index > 0 && point <= names::C_RANGES[index - 1].1
}

/// Packed originals use strict JSON; the legacy parser's last-key behavior is
/// intentionally preserved. The existing admission covers this nonretained key
/// scan before the ordinary Rust value parser builds its owned document.
pub(crate) fn parse(bytes: &[u8], budget: &Arc<Budget>, control: &Control) -> Result<Document> {
    let admission = json::admit(bytes.len(), budget)?;
    parse_admitted(bytes, &admission, control)
}
pub(crate) fn parse_admitted(
    bytes: &[u8],
    admission: &Arc<crate::budget::JsonAdmission>,
    control: &Control,
) -> Result<Document> {
    use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
    struct Unique<'a>(&'a Control);
    impl<'de> DeserializeSeed<'de> for Unique<'_> {
        type Value = ();
        fn deserialize<D: serde::Deserializer<'de>>(
            self,
            decoder: D,
        ) -> std::result::Result<(), D::Error> {
            self.0.check().map_err(serde::de::Error::custom)?;
            decoder.deserialize_any(self)
        }
    }
    impl<'de> Visitor<'de> for Unique<'_> {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("strict original JSON")
        }
        fn visit_bool<E: serde::de::Error>(self, _: bool) -> std::result::Result<(), E> {
            Ok(())
        }
        fn visit_i64<E: serde::de::Error>(self, _: i64) -> std::result::Result<(), E> {
            Ok(())
        }
        fn visit_u64<E: serde::de::Error>(self, _: u64) -> std::result::Result<(), E> {
            Ok(())
        }
        fn visit_f64<E: serde::de::Error>(self, v: f64) -> std::result::Result<(), E> {
            if v.is_finite() {
                Ok(())
            } else {
                Err(E::custom("nonfinite original JSON"))
            }
        }
        fn visit_str<E: serde::de::Error>(self, _: &str) -> std::result::Result<(), E> {
            Ok(())
        }
        fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<(), E> {
            Ok(())
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<(), A::Error> {
            while seq.next_element_seed(Unique(self.0))?.is_some() {}
            Ok(())
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<(), A::Error> {
            let mut keys = std::collections::BTreeSet::new();
            while let Some(key) = map.next_key::<String>()? {
                self.0.check().map_err(serde::de::Error::custom)?;
                // serde's arbitrary-precision number adapter is internal only;
                // actual raw members with this name were rejected beforehand.
                if key == "$serde_json::private::Number" {
                    let token = map.next_value::<String>()?;
                    if token.bytes().any(|c| matches!(c, b'.' | b'e' | b'E'))
                        && !token
                            .parse::<f64>()
                            .map_err(serde::de::Error::custom)?
                            .is_finite()
                    {
                        return Err(serde::de::Error::custom("nonfinite original JSON"));
                    }
                } else {
                    if !keys.insert(key) {
                        return Err(serde::de::Error::custom("duplicate original JSON key"));
                    }
                    map.next_value_seed(Unique(self.0))?;
                }
            }
            Ok(())
        }
    }
    json::reject_reserved_number_members(bytes, control)?;
    {
        let mut decoder = serde_json::Deserializer::from_slice(bytes);
        Unique(control)
            .deserialize(&mut decoder)
            .map_err(|e| Error::contract(e.to_string()))?;
        decoder.end().map_err(|e| Error::contract(e.to_string()))?;
    }
    json::parse_admitted(bytes, admission, control)
}

pub(crate) trait LogicalArtifactProvider: Send + Sync {
    fn registry(&self, budget: &Arc<Budget>, control: &Control) -> Result<BudgetVec<u8>>;
    fn describe(
        &self,
        root: &str,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<BudgetVec<u8>>;
    fn read(
        &self,
        root: &str,
        name: &str,
        length: usize,
        sha256: &str,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<BudgetVec<u8>>;
    fn recheck(&self, root: &str, control: &Control) -> Result<()>;
}

pub(crate) fn binding() -> Value {
    json::object([
        ("format", json::text("sqlite-group-pack-v1")),
        ("pack_format", json::text("group-pack-v1")),
        ("database_schema", json::integer(1)),
        ("member_identity", json::text("packed-member-v1")),
        ("provider", json::text(FORMAT)),
    ])
}
pub(crate) fn member_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.chars().any(category_c)
        || name.contains('\\')
        || name.contains('\0')
        || Path::new(name)
            .components()
            .any(|v| !matches!(v, Component::Normal(_)))
        || Path::new(name).as_os_str().as_encoded_bytes() != name.as_bytes()
        || name
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(Error::contract(
            "safe logical original member name required",
        ));
    }
    Ok(())
}

pub(crate) struct Registry {
    provider: Arc<dyn LogicalArtifactProvider>,
    pub(crate) digest: String,
    pub(crate) roots: BTreeMap<PathBuf, LogicalSource>,
    _registry: Document,
    _metadata: crate::budget::Lease,
}
pub(crate) struct LogicalSource {
    pub(crate) description: Document,
}
impl Registry {
    pub(crate) fn open(
        provider: Arc<dyn LogicalArtifactProvider>,
        expected: &str,
        budget: &Arc<Budget>,
        control: &Control,
        file_limit: usize,
    ) -> Result<Arc<Self>> {
        let raw = provider.registry(budget, control)?;
        if raw.len() > METADATA_LIMIT || format!("{:x}", Sha256::digest(&*raw)) != expected {
            return Err(Error::contract(
                "plan/provider registry digest or metadata limit differs",
            ));
        }
        let registry = parse(&raw, budget, control)?;
        registry.root.closed(&["format", "roots"], &[])?;
        if registry.root.require("format")?.string()? != FORMAT {
            return Err(Error::contract("original provider registry format differs"));
        }
        let mut metadata = budget.reserve(65536)?;
        let mut roots = BTreeMap::new();
        let mut previous: Option<String> = None;
        for row in registry.root.require("roots")?.array()? {
            control.check()?;
            row.closed(&["root", "storage", "custody"], &[])?;
            let root = row.require("root")?.string()?;
            let path = Path::new(root);
            if !path.is_absolute()
                || path
                    .components()
                    .any(|p| matches!(p, Component::ParentDir | Component::CurDir))
                || previous.as_deref().is_some_and(|p| p >= root)
                || roots
                    .keys()
                    .any(|p: &PathBuf| path.starts_with(p) || p.starts_with(path))
            {
                return Err(Error::contract("duplicate/foreign logical provider root"));
            }
            metadata.resize(metadata.bytes() + 4096 + 8 * root.len())?;
            let bytes = provider.describe(root, budget, control)?;
            if bytes.len() > METADATA_LIMIT {
                return Err(Error::resource("logical description metadata cap"));
            }
            let description = parse(&bytes, budget, control)?;
            let d = &description.root;
            d.closed(
                &["format", "root", "storage", "custody", "files", "index"],
                &[],
            )?;
            if d.require("format")?.string()? != FORMAT
                || d.require("root")? != row.require("root")?
                || d.require("storage")? != row.require("storage")?
                || d.require("custody")? != row.require("custody")?
            {
                return Err(Error::contract(
                    "logical description differs from original registry",
                ));
            }
            validate_custody(d.require("custody")?, budget, control)?;
            validate_index_shape(d.require("index")?, control)?;
            let storage = d.require("storage")?;
            storage.closed(
                &[
                    "format",
                    "locator_sha256",
                    "descriptor_sha256",
                    "database_sha256",
                    "seal_id",
                    "logical_sha256",
                ],
                &[],
            )?;
            if storage.require("format")?.string()? != "sqlite-group-pack-v1" {
                return Err(Error::contract("logical original storage format differs"));
            }
            for key in [
                "locator_sha256",
                "descriptor_sha256",
                "database_sha256",
                "logical_sha256",
            ] {
                crate::structured_controls::checksum(storage.require(key)?)?;
            }
            storage.require("seal_id")?.string()?;
            for (role, key) in [
                ("locator", "locator_sha256"),
                ("descriptor", "descriptor_sha256"),
                ("database", "database_sha256"),
            ] {
                let original = d
                    .require("custody")?
                    .require("files")?
                    .array()?
                    .iter()
                    .find(|f| f.get("role").and_then(Value::as_str) == Some(role))
                    .ok_or_else(|| Error::contract("missing original custody role"))?;
                if original.require("sha256")? != storage.require(key)? {
                    return Err(Error::contract("original custody/storage checksum differs"));
                }
            }
            let mut last: Option<&str> = None;
            for file in d.require("files")?.array()? {
                file.closed(&["name", "bytes", "sha256", "identity"], &[])?;
                let name = file.require("name")?.string()?;
                member_name(name)?;
                if last.is_some_and(|v| v >= name) {
                    return Err(Error::contract(
                        "duplicate/unsorted logical member inventory",
                    ));
                }
                let length = file.require("bytes")?.integer(0, file_limit as i128)?;
                crate::structured_controls::checksum(file.require("sha256")?)?;
                let identity = file.require("identity")?;
                identity.closed(
                    &[
                        "format",
                        "database_sha256",
                        "group_id",
                        "pack_sha256",
                        "name",
                        "offset",
                        "length",
                        "sha256",
                    ],
                    &[],
                )?;
                if identity.require("format")?.string()? != "packed-member-v1"
                    || identity.require("database_sha256")? != storage.require("database_sha256")?
                    || identity.require("name")? != file.require("name")?
                    || identity.require("sha256")? != file.require("sha256")?
                    || identity.require("length")?.integer(0, file_limit as i128)? != length
                {
                    return Err(Error::contract("logical original member identity differs"));
                }
                if identity.require("group_id")?.is_null() {
                    if !identity.require("pack_sha256")?.is_null()
                        || !identity.require("offset")?.is_null()
                    {
                        return Err(Error::contract("logical original control identity differs"));
                    }
                } else {
                    member_name(identity.require("group_id")?.string()?)?;
                    crate::structured_controls::checksum(identity.require("pack_sha256")?)?;
                    identity
                        .require("offset")?
                        .integer(0, u64::MAX as i128 - length)?;
                }
                last = Some(name);
            }
            previous = Some(root.to_owned());
            roots.insert(path.to_owned(), LogicalSource { description });
        }
        Ok(Arc::new(Self {
            provider,
            digest: expected.to_owned(),
            roots,
            _registry: registry,
            _metadata: metadata,
        }))
    }
    pub(crate) fn locate<'a, 'p>(
        &'a self,
        path: &'p Path,
    ) -> Option<(&'a Path, &'a LogicalSource, &'p str)> {
        self.roots.iter().find_map(|(root, source)| {
            path.strip_prefix(root)
                .ok()
                .and_then(|part| part.to_str())
                .map(|name| (root.as_path(), source, name))
        })
    }
    pub(crate) fn file<'a>(&self, source: &'a LogicalSource, name: &str) -> Result<&'a Value> {
        member_name(name)?;
        source
            .description
            .root
            .require("files")?
            .array()?
            .iter()
            .find(|v| v.get("name").and_then(Value::as_str) == Some(name))
            .ok_or_else(|| Error::contract("unregistered original logical member"))
    }
    pub(crate) fn read(
        &self,
        path: &Path,
        maximum: usize,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<BudgetVec<u8>> {
        let (root, source, name) = self
            .locate(path)
            .ok_or_else(|| Error::contract("unregistered logical root"))?;
        let file = self.file(source, &name)?;
        let length = file.require("bytes")?.integer(0, maximum as i128)? as usize;
        let sha = file.require("sha256")?.string()?;
        let raw =
            self.provider
                .read(root.to_str().unwrap(), &name, length, sha, budget, control)?;
        if raw.len() != length || format!("{:x}", Sha256::digest(&*raw)) != sha {
            return Err(Error::contract("original logical byte length/hash differs"));
        }
        Ok(raw)
    }
    pub(crate) fn recheck(&self, root: &Path, control: &Control) -> Result<()> {
        for path in self
            .roots
            .keys()
            .filter(|path| path.starts_with(root) || root.starts_with(path))
        {
            self.provider.recheck(path.to_str().unwrap(), control)?;
        }
        Ok(())
    }
}

fn validate_custody(value: &Value, budget: &Arc<Budget>, control: &Control) -> Result<()> {
    value.closed(&["recipe", "files", "directories", "fingerprint"], &[])?;
    if value.require("recipe")?.string()? != "packed-original-custody-v1" {
        return Err(Error::contract("original custody recipe differs"));
    }
    let mut unsigned = json::copy(value, control)?;
    unsigned.as_object_mut().unwrap().remove("fingerprint");
    if crate::structured_controls::digest(&unsigned, budget, control)?
        != value.require("fingerprint")?.string()?
    {
        return Err(Error::contract("original custody fingerprint differs"));
    }
    let mut previous: Option<&str> = None;
    if value.require("directories")?.array()?.len() != 2 {
        return Err(Error::contract(
            "two actual original custody directories required",
        ));
    }
    let mut roles = std::collections::BTreeSet::new();
    for file in value.require("files")?.array()? {
        file.closed(&["role", "path", "sha256", "identity"], &[])?;
        let role = file.require("role")?.string()?;
        let path = file.require("path")?.string()?;
        if !["locator", "descriptor", "database", "predecessor"].contains(&role)
            || !roles.insert(role)
            || !Path::new(path).is_absolute()
            || previous.is_some_and(|p| p >= path)
        {
            return Err(Error::contract(
                "original custody file ordering/role differs",
            ));
        }
        crate::structured_controls::checksum(file.require("sha256")?)?;
        let identity = file.require("identity")?.array()?;
        if identity.len() != 6 {
            return Err(Error::contract("original custody stat6 required"));
        }
        for item in identity {
            item.integer(0, u64::MAX as i128)?;
        }
        previous = Some(path);
    }
    if !["locator", "descriptor", "database"]
        .into_iter()
        .all(|r| roles.contains(r))
    {
        return Err(Error::contract("complete original custody required"));
    }
    let mut previous: Option<&str> = None;
    for directory in value.require("directories")?.array()? {
        directory.closed(&["path", "device", "inode"], &[])?;
        let path = directory.require("path")?.string()?;
        if !Path::new(path).is_absolute() || previous.is_some_and(|p| p >= path) {
            return Err(Error::contract(
                "original custody directory ordering differs",
            ));
        }
        directory.require("device")?.integer(0, u64::MAX as i128)?;
        directory.require("inode")?.integer(0, u64::MAX as i128)?;
        previous = Some(path);
    }
    Ok(())
}

fn validate_index_shape(index: &Value, control: &Control) -> Result<()> {
    index.closed(&["groups", "episodes"], &[])?;
    let mut previous: Option<&str> = None;
    for group in index.require("groups")?.array()? {
        control.check()?;
        group.closed(&["group_id", "record"], &[])?;
        let id = group.require("group_id")?.string()?;
        member_name(id)?;
        if previous.is_some_and(|p| p >= id) {
            return Err(Error::contract("duplicate/unsorted original index group"));
        }
        group.require("record")?.object()?;
        previous = Some(id);
    }
    let mut previous: Option<&str> = None;
    for episode in index.require("episodes")?.array()? {
        control.check()?;
        episode.closed(&["episode_id", "group_id", "game_id", "record"], &[])?;
        let id = episode.require("episode_id")?.string()?;
        member_name(id)?;
        member_name(episode.require("group_id")?.string()?)?;
        if previous.is_some_and(|p| p >= id) {
            return Err(Error::contract("duplicate/unsorted original index episode"));
        }
        if !episode.require("game_id")?.is_null() {
            episode.require("game_id")?.string()?;
        }
        episode.require("record")?.object()?;
        previous = Some(id);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[test]
    fn frozen_unicode_predicate_matches_every_scalar_and_preserves_valid_names() {
        assert_eq!(names::C_RANGES.len(), 736);
        assert!(names::C_RANGES.windows(2).all(|v| v[0].1 + 1 < v[1].0));
        let mut index = 0;
        for point in 0..=0x10ffff {
            while index < names::C_RANGES.len() && names::C_RANGES[index].1 < point {
                index += 1;
            }
            if let Some(value) = char::from_u32(point) {
                let expected = index < names::C_RANGES.len() && names::C_RANGES[index].0 <= point;
                assert_eq!(category_c(value), expected, "{point:x}");
            }
        }
        for invalid in [
            "a/\n.json",
            "a/\u{200d}.json",
            "a/\u{e000}.json",
            "a/\u{378}.json",
            "a/../b",
            "a//b",
        ] {
            assert!(member_name(invalid).is_err());
        }
        member_name("診断/ログ.json").unwrap();
        let b = Budget::new(1 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        assert!(parse(b"{\"name\":\"\xff\"}", &b, &c).is_err());
        assert!(parse(br#"{"name":"\ud800"}"#, &b, &c).is_err());
    }
    struct Memory {
        registry: Vec<u8>,
        description: Vec<u8>,
        raw: Vec<u8>,
        reads: AtomicUsize,
    }
    fn copy(raw: &[u8], budget: &Arc<Budget>) -> Result<BudgetVec<u8>> {
        let mut value = BudgetVec::with_capacity(budget, raw.len())?;
        value.extend_from_slice(raw)?;
        Ok(value)
    }
    impl LogicalArtifactProvider for Memory {
        fn registry(&self, b: &Arc<Budget>, _: &Control) -> Result<BudgetVec<u8>> {
            copy(&self.registry, b)
        }
        fn describe(&self, _: &str, b: &Arc<Budget>, _: &Control) -> Result<BudgetVec<u8>> {
            copy(&self.description, b)
        }
        fn read(
            &self,
            _: &str,
            _: &str,
            _: usize,
            _: &str,
            b: &Arc<Budget>,
            c: &Control,
        ) -> Result<BudgetVec<u8>> {
            c.check()?;
            self.reads.fetch_add(1, Ordering::Relaxed);
            copy(&self.raw, b)
        }
        fn recheck(&self, _: &str, c: &Control) -> Result<()> {
            c.check()
        }
    }
    fn fixture(b: &Arc<Budget>, c: &Control) -> Arc<Memory> {
        let root = "/virtual-parent/collections/batch-000000";
        let sha = "a".repeat(64);
        let raw = b"{\"original\":true}\n";
        let hash = format!("{:x}", Sha256::digest(raw));
        let storage = json::object([
            ("format", json::text("sqlite-group-pack-v1")),
            ("locator_sha256", json::text(&sha)),
            ("descriptor_sha256", json::text(&sha)),
            ("database_sha256", json::text(&sha)),
            ("seal_id", json::text("seal")),
            ("logical_sha256", json::text(&sha)),
        ]);
        let files = ["database", "descriptor", "locator"]
            .into_iter()
            .map(|role| {
                json::object([
                    ("role", json::text(role)),
                    ("path", json::text(&format!("/physical/{role}"))),
                    ("sha256", json::text(&sha)),
                    (
                        "identity",
                        json::array([1, 2, 3, 4, 5, 6].into_iter().map(json::integer)),
                    ),
                ])
            });
        let mut custody = json::object([
            ("recipe", json::text("packed-original-custody-v1")),
            ("files", json::array(files)),
            (
                "directories",
                json::array(["/physical", root].into_iter().map(|path| {
                    json::object([
                        ("path", json::text(path)),
                        ("device", json::integer(1)),
                        ("inode", json::integer(2)),
                    ])
                })),
            ),
        ]);
        let fingerprint = crate::structured_controls::digest(&custody, b, c).unwrap();
        custody
            .as_object_mut()
            .unwrap()
            .insert("fingerprint".into(), json::text(&fingerprint));
        let registry = json::object([
            ("format", json::text(FORMAT)),
            (
                "roots",
                json::array([json::object([
                    ("root", json::text(root)),
                    ("storage", storage.clone()),
                    ("custody", custody.clone()),
                ])]),
            ),
        ]);
        let identity = json::object([
            ("format", json::text("packed-member-v1")),
            ("database_sha256", json::text(&sha)),
            ("group_id", Value::Null),
            ("pack_sha256", Value::Null),
            ("name", json::text("e/player-1.jsonl")),
            ("offset", Value::Null),
            ("length", json::integer(raw.len() as i128)),
            ("sha256", json::text(&hash)),
        ]);
        let description = json::object([
            ("format", json::text(FORMAT)),
            ("root", json::text(root)),
            ("storage", storage),
            ("custody", custody),
            (
                "index",
                json::object([("groups", json::array([])), ("episodes", json::array([]))]),
            ),
            (
                "files",
                json::array([json::object([
                    ("name", json::text("e/player-1.jsonl")),
                    ("bytes", json::integer(raw.len() as i128)),
                    ("sha256", json::text(&hash)),
                    ("identity", identity),
                ])]),
            ),
        ]);
        Arc::new(Memory {
            registry: json::canonical(&registry, b, c).unwrap().to_vec(),
            description: json::canonical(&description, b, c).unwrap().to_vec(),
            raw: raw.to_vec(),
            reads: AtomicUsize::new(0),
        })
    }
    #[test]
    fn virtual_sources_use_original_member_identity_and_budgeted_bytes() {
        let b = Budget::new(64 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let memory = fixture(&b, &c);
        let hash = format!("{:x}", Sha256::digest(&memory.registry));
        let registry = Registry::open(memory.clone(), &hash, &b, &c, 1024).unwrap();
        let root = Path::new("/virtual-parent/collections/batch-000000");
        let mut source = crate::structured_session::Sources::from_registry(
            &root.join("e"),
            b.clone(),
            c.clone(),
            1024,
            Some(registry),
        )
        .unwrap();
        let path = source.root.join("player-1.jsonl");
        source.hash(&path).unwrap();
        assert_eq!(memory.reads.load(Ordering::Relaxed), 0);
        assert_eq!(&*source.bytes(&path, 1024).unwrap(), &memory.raw);
        source
            .inventory(&source.root.clone(), vec!["player-1.jsonl".into()], false)
            .unwrap();
        assert!(source.bytes(&path, 1).is_err());
        assert!(source.bytes(&source.root.join("unknown"), 1024).is_err());
        let proof = source.proof_files(root).unwrap();
        assert_eq!(
            proof.root["e/player-1.jsonl"]["identity"]["format"],
            json::text("packed-member-v1")
        );
        source.recheck().unwrap();
        drop(proof);
        drop(source);
        assert_eq!(b.used(), 0);
    }
    #[test]
    fn registry_mismatch_caps_and_cancel_reject_before_original_read() {
        let b = Budget::new(64 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let memory = fixture(&b, &c);
        let hash = format!("{:x}", Sha256::digest(&memory.registry));
        assert!(Registry::open(memory.clone(), &"0".repeat(64), &b, &c, 1024).is_err());
        assert!(Registry::open(memory.clone(), &hash, &b, &c, 1).is_err());
        c.cancel();
        assert!(Registry::open(memory.clone(), &hash, &b, &c, 1024).is_err());
        assert_eq!(memory.reads.load(Ordering::Relaxed), 0);
        assert_eq!(b.used(), 0);
    }
    #[test]
    fn original_index_shape_refuses_missing_extra_duplicate_and_unsorted_rows() {
        let b = Budget::new(64 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let good = json::object([
            (
                "groups",
                json::array(["a", "b"].into_iter().map(|id| {
                    json::object([("group_id", json::text(id)), ("record", json::object([]))])
                })),
            ),
            (
                "episodes",
                json::array(["a", "b"].into_iter().map(|id| {
                    json::object([
                        ("episode_id", json::text(id)),
                        ("group_id", json::text(id)),
                        ("game_id", Value::Null),
                        ("record", json::object([])),
                    ])
                })),
            ),
        ]);
        validate_index_shape(&good, &c).unwrap();
        for case in 0..7 {
            let mut value = good.clone();
            match case {
                0 => {
                    value.as_object_mut().unwrap().remove("groups");
                }
                1 => {
                    value["groups"][0]
                        .as_object_mut()
                        .unwrap()
                        .insert("unknown".into(), Value::Null);
                }
                2 => value["groups"][1]["group_id"] = json::text("a"),
                3 => value["groups"].as_array_mut().unwrap().reverse(),
                4 => value["episodes"][1]["episode_id"] = json::text("a"),
                5 => value["episodes"].as_array_mut().unwrap().reverse(),
                _ => value["episodes"][0]["game_id"] = Value::Bool(false),
            }
            assert!(validate_index_shape(&value, &c).is_err(), "case{case}");
        }
        assert!(parse(br#"{"index":{},"index":{}}"#, &b, &c).is_err());
        assert!(parse(br#"{"nested":{"a":1,"\u0061":2}}"#, &b, &c).is_err());
        assert!(parse(br#"{"number":NaN}"#, &b, &c).is_err());
        assert!(parse(br#"{"number":1e400}"#, &b, &c).is_err());
        let legacy = json::parse(br#"{"value":1,"value":2}"#, &b, &c).unwrap();
        assert_eq!(legacy.root["value"], json::integer(2));
        drop(legacy);
        assert_eq!(b.used(), 0);
    }
    #[test]
    fn whole_describe_eight_mib_cap_includes_index_before_parser_admission() {
        let setup = Budget::new(64 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let base = fixture(&setup, &c);
        let mut description = json::parse(&base.description, &setup, &c).unwrap();
        description.root["index"]["groups"] = json::array([json::object([
            ("group_id", json::text("group")),
            ("record", json::object([("padding", json::text(""))])),
        ])]);
        let fixed = serde_json::to_vec(&description.root).unwrap().len();
        for extra in [0, 1] {
            description.root["index"]["groups"][0]["record"]["padding"] =
                json::text(&"x".repeat(METADATA_LIMIT - fixed + extra));
            let bytes = serde_json::to_vec(&description.root).unwrap();
            assert_eq!(bytes.len(), METADATA_LIMIT + extra);
            let provider = Arc::new(Memory {
                registry: base.registry.clone(),
                description: bytes,
                raw: base.raw.clone(),
                reads: AtomicUsize::new(0),
            });
            let hash = format!("{:x}", Sha256::digest(&provider.registry));
            let budget = Budget::new(512 << 20);
            let result = Registry::open(provider, &hash, &budget, &c, 1024);
            if extra == 0 {
                assert!(result.is_ok());
            } else {
                assert!(result.is_err());
                assert!(budget.json_estimated_peak() < (1 << 20));
            }
            drop(result);
            assert_eq!(budget.used(), 0);
        }
    }
}
