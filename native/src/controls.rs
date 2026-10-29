//! Structural parent chains. Python authorizes schedules and lifetime ownership.
use crate::json::{Map, ValueExt};
use crate::{
    budget::{Budget, BudgetVec, Control},
    error::{Error, Result},
    journal::{equal, string},
    json::{self, Document, Value},
    source::Sources,
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

pub fn digest(value: &Value, budget: &Arc<Budget>, control: &Control) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(&*json::canonical(value, budget, control)?)
    ))
}
pub fn hash(sources: &mut Sources, path: &Path, control: &Control) -> Result<String> {
    if let Some(checksum) = sources.checked_digest(path, control)? {
        return Ok(checksum.to_owned());
    }
    let bytes = sources.read(path, control)?;
    drop(bytes);
    Ok(sources.digest(path, control)?.to_owned())
}
fn same_hash(
    sources: &mut Sources,
    path: &Path,
    expected: &Value,
    control: &Control,
) -> Result<()> {
    if hash(sources, path, control)? != expected.string()? {
        return Err(Error::contract("parent artifact/source hash mismatch"));
    }
    Ok(())
}
pub fn truthy(value: &Value) -> bool {
    crate::json::truthy(value)
}
pub fn field_truthy(value: &Value, key: &str) -> bool {
    value.get(key).is_some_and(truthy)
}
fn numeric_equal(value: &Value, number: i128) -> bool {
    value.eq_python(&Value::Number(
        crate::json::Number::from_i128(number).expect("i128 number"),
    ))
}
pub fn numeric_field(value: &Value, key: &str, number: i128) -> bool {
    value.get(key).is_some_and(|v| numeric_equal(v, number))
}

pub struct Parent {
    pub directory: PathBuf,
    pub state: Document,
    pub generations: BudgetVec<Document>,
    pub updates: BudgetVec<Document>,
    pub registry: Document,
}
impl Parent {
    pub fn open(
        directory: &Path,
        complete: bool,
        sources: &mut Sources,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<Self> {
        let directory = fs::canonicalize(directory)?;
        sources.inventory(
            &directory.join("generations"),
            crate::source::InventoryKind::JsonFiles,
            control,
        )?;
        sources.inventory(
            &directory.join("updates"),
            crate::source::InventoryKind::JsonFiles,
            control,
        )?;
        sources.inventory(
            &directory.join("artifacts"),
            crate::source::InventoryKind::ArtifactLeaves,
            control,
        )?;
        let state = sources.json(&directory.join("run.json"), false, control)?;
        let s = &state.root;
        if !numeric_field(s, "schema_version", 1)
            || string(s, "algorithm")? != "ppo"
            || !["running", "paused", "complete"].contains(&string(s, "status")?)
            || complete && string(s, "status")? != "complete"
        {
            return Err(Error::contract("healthy completed PPO parent required"));
        }
        for key in [
            "failure",
            "quarantine",
            "cleanup_failure",
            "budget_overrun_seconds",
        ] {
            if field_truthy(s, key) {
                return Err(Error::contract("PPO parent contains failure evidence"));
            }
        }
        let config = s.require("effective_config")?;
        if digest(config, budget, control)? != string(s, "config_sha256")?
            || !["cpu", "cuda:0"].contains(&string(s, "device")?)
        {
            return Err(Error::contract(
                "PPO parent config/device identity mismatch",
            ));
        }
        let mut times = [0.; 5];
        for (i, key) in [
            "created_unix",
            "last_unix",
            "wall_deadline_unix",
            "admission_deadline_unix",
            "execution_deadline_unix",
        ]
        .iter()
        .enumerate()
        {
            times[i] = s.require(key)?.number()?;
            if !times[i].is_finite() {
                return Err(Error::contract("invalid parent clock provenance"));
            }
        }
        let [created, last, wall, admission, execution] = times;
        if !(created <= last
            && last <= wall
            && created < admission
            && admission <= execution
            && execution < wall)
            || wall - created > 7200.
            || admission - created > 5400.
            || execution - created > 7140.
            || ((wall - created) - config.require("run")?.require("wall_seconds")?.number()?).abs()
                > 0.001
        {
            return Err(Error::contract(
                "parent original deadline/elapsed provenance invalid",
            ));
        }
        if ["paused", "complete"].contains(&string(s, "status")?) {
            let elapsed = s.require("elapsed_seconds")?.number()?;
            if !elapsed.is_finite()
                || elapsed < 0.
                || elapsed > wall - created
                || (elapsed - (last - created)).abs() > 0.001
            {
                return Err(Error::contract(
                    "parent completion elapsed exceeds original budget",
                ));
            }
            if let Some(invocations) = s.get("invocations") {
                for invocation in invocations.array()? {
                    control.check()?;
                    let health = invocation.require("child_health")?;
                    health.closed(&["exit_code"], &["stderr_bytes"])?;
                    if !numeric_field(health, "exit_code", 0) {
                        return Err(Error::contract("parent learner invocation unhealthy"));
                    }
                }
            }
        }
        let latest = sources.json(&directory.join("latest.json"), false, control)?;
        latest.root.closed(&["generation", "sha256"], &[])?;
        let count = s.require("updates")?.integer(0, 1000)? as usize;
        if !numeric_field(&latest.root, "generation", count as i128) {
            return Err(Error::contract("incomplete latest generation commit"));
        }
        same_hash(
            sources,
            &generation_path(&directory, count),
            latest.root.require("sha256")?,
            control,
        )?;
        equal(
            s.require("generation_sha256")?,
            latest.root.require("sha256")?,
            "run/latest commit mismatch",
        )?;
        check_inventory(&directory.join("generations"), 0, count, budget, control)?;
        check_inventory(&directory.join("updates"), 1, count, budget, control)?;
        let registry = sources.json(&directory.join("artifact-registry.json"), false, control)?;
        let mut generations = BudgetVec::with_capacity(budget, count + 1)?;
        let mut updates = BudgetVec::with_capacity(budget, count)?;
        let mut previous = None::<String>;
        let mut cumulative = 0i128;
        let mut steps = 0i128;
        for number in 0..=count {
            control.check()?;
            let record = sources.json(&generation_path(&directory, number), false, control)?;
            let r = &record.root;
            r.closed(
                &[
                    "schema_version",
                    "run_id",
                    "generation",
                    "parent_sha256",
                    "inference_sha256",
                    "weights_sha256",
                    "artifact_hashes",
                    "accepted_decisions",
                    "optimizer_steps",
                    "update_sha256",
                    "parent_controls",
                ],
                &[],
            )?;
            if !numeric_field(r, "schema_version", 1)
                || !numeric_field(r, "generation", number as i128)
                || !r.require("run_id")?.eq_python(s.require("run_id")?)
                || match &previous {
                    Some(hash) => string(r, "parent_sha256")? != hash,
                    None => !r.require("parent_sha256")?.is_null(),
                }
            {
                return Err(Error::contract("broken generation chain"));
            }
            let path = directory.join("artifacts").join(format!("{number:06}"));
            same_hash(
                sources,
                &path.join("inference.json"),
                r.require("inference_sha256")?,
                control,
            )?;
            let inference = sources.json(&path.join("inference.json"), false, control)?;
            equal(
                inference.root.require("weights")?.require("sha256")?,
                r.require("weights_sha256")?,
                "generation weights identity mismatch",
            )?;
            let receipt = registry.root.require(&number.to_string())?;
            equal(
                receipt.require("inference_sha256")?,
                r.require("inference_sha256")?,
                "missing artifact registry identity",
            )?;
            let retained = number == 0 || number % 10 == 0 || number == count;
            let status = string(receipt, "status")?;
            if retained || status == "retained" {
                if status != "retained" {
                    return Err(Error::contract("retained generation artifacts mismatch"));
                }
                let hashes = r.require("artifact_hashes")?;
                hashes.closed(
                    &[
                        "inference.json",
                        "weights.npz",
                        "weights.json",
                        "resume.json",
                        "resume.npz",
                    ],
                    &[],
                )?;
                for name in [
                    "inference.json",
                    "weights.npz",
                    "weights.json",
                    "resume.json",
                    "resume.npz",
                ] {
                    same_hash(sources, &path.join(name), hashes.require(name)?, control)?;
                }
            } else if status == "deleted" {
                if receipt.require("deletion_intent")?.boolean()? != true
                    || [
                        "weights.npz",
                        "resume.npz",
                        "resume.json",
                        "resume.json.tmp",
                    ]
                    .iter()
                    .any(|name| path.join(name).exists())
                {
                    return Err(Error::contract("artifact deletion receipt mismatch"));
                }
            } else {
                return Err(Error::contract("unrecognized generation artifact state"));
            }
            if number > 0 {
                let update_path = directory.join("updates").join(format!("{number:06}.json"));
                let update = sources.json(&update_path, false, control)?;
                let u = &update.root;
                same_hash(sources, &update_path, r.require("update_sha256")?, control)?;
                if !u.require("run_id")?.eq_python(s.require("run_id")?)
                    || !numeric_field(u, "generation", number as i128)
                    || string(u, "parent_sha256")? != previous.as_deref().unwrap_or("")
                    || string(u, "status")? != "complete"
                {
                    return Err(Error::contract("broken immutable update linkage"));
                }
                let collection = directory.join(string(u, "collection")?);
                if !fs::canonicalize(&collection)?.starts_with(&directory) {
                    return Err(Error::contract("update collection escapes parent"));
                }
                same_hash(
                    sources,
                    &collection.join("manifest.json"),
                    u.require("collection_manifest_sha256")?,
                    control,
                )?;
                same_hash(
                    sources,
                    &directory.join(string(u, "numeric")?).join("rollout.json"),
                    u.require("numeric_sha256")?,
                    control,
                )?;
                cumulative += u
                    .require("accepted_decisions")?
                    .integer(1, u64::MAX as i128)?;
                if !numeric_field(r, "accepted_decisions", cumulative)
                    || r.require("optimizer_steps")?.integer(1, u64::MAX as i128)? <= steps
                {
                    return Err(Error::contract("generation counters mismatch"));
                }
                updates.push(update)?;
            } else if !numeric_field(r, "accepted_decisions", 0)
                || !numeric_field(r, "optimizer_steps", 0)
                || !r.require("update_sha256")?.is_null()
            {
                return Err(Error::contract("generation zero counters mismatch"));
            }
            steps = if number == 0 {
                0
            } else {
                r.require("optimizer_steps")?.integer(1, u64::MAX as i128)?
            };
            previous = Some(hash(
                sources,
                &generation_path(&directory, number),
                control,
            )?);
            generations.push(record)?;
        }
        if complete && (count == 0 || !numeric_field(s, "completed_updates", count as i128)) {
            return Err(Error::contract(
                "zero-update/unfinalized PPO run ineligible",
            ));
        }
        Ok(Self {
            directory,
            state,
            generations,
            updates,
            registry,
        })
    }
    pub fn summary(&self, control: &Control) -> Result<Value> {
        let mut generations = Vec::new();
        for record in self.generations.iter() {
            let r = &record.root;
            generations.push(json::object([
                ("generation", json::copy(r.require("generation")?, control)?),
                (
                    "inference_sha256",
                    json::copy(r.require("inference_sha256")?, control)?,
                ),
                (
                    "parent_controls",
                    json::copy(r.require("parent_controls")?, control)?,
                ),
            ]));
        }
        let mut fields = Map::new();
        for key in [
            "run_id",
            "seed",
            "status",
            "updates",
            "accepted_decisions",
            "elapsed_seconds",
        ] {
            let value = match self.state.root.get(key) {
                Some(v) => json::copy(v, control)?,
                None => Value::Null,
            };
            fields.insert(key.to_owned(), value);
        }
        let state = Value::Object(fields);
        Ok(json::object([
            (
                "directory",
                json::text(
                    self.directory
                        .to_str()
                        .ok_or_else(|| Error::contract("non-UTF8 parent path"))?,
                ),
            ),
            ("state", state),
            ("generations", Value::Array(generations)),
        ]))
    }
}
pub fn generation_path(directory: &Path, number: usize) -> PathBuf {
    directory
        .join("generations")
        .join(format!("{number:06}.json"))
}
fn check_inventory(
    directory: &Path,
    first: usize,
    last: usize,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<()> {
    let length = if last < first { 0 } else { last - first + 1 };
    let mut seen = BudgetVec::with_capacity(budget, length)?;
    for _ in 0..length {
        seen.push(false)?;
    }
    for entry in fs::read_dir(directory)? {
        control.check()?;
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.ends_with(".json") {
            continue;
        }
        let number = name
            .strip_suffix(".json")
            .and_then(|n| n.parse::<usize>().ok())
            .ok_or_else(|| Error::contract("unexpected immutable record"))?;
        if number < first
            || number > last
            || name != format!("{number:06}.json")
            || seen[number - first]
        {
            return Err(Error::contract("missing/extra immutable records"));
        }
        seen[number - first] = true;
    }
    if seen.iter().any(|v| !*v) {
        return Err(Error::contract("missing/extra immutable records"));
    }
    Ok(())
}
