//! Native lifecycle and compact authorization handshake. Learning never lives here.
use crate::json::ValueExt;
use crate::{
    budget::{Budget, BudgetVec, Control, Lease},
    controls::{self, Parent},
    error::{Error, Result},
    feature::FeatureConfig,
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

pub struct Settings {
    pub memory: usize,
    pub file: usize,
    pub record: usize,
    pub chunk: usize,
    pub scratch: usize,
    pub workers: usize,
    pub in_flight: usize,
    pub json_depth: usize,
}
impl Settings {
    pub fn decode(encoded: &[u8], control: &Control) -> Result<Self> {
        if encoded.len() > 65536 {
            return Err(Error::resource("oversized native settings"));
        }
        let initial = Budget::new(1048576);
        let settings = json::parse(encoded, &initial, control)?;
        Self::parse(&settings.root)
    }
    fn parse(value: &Value) -> Result<Self> {
        value.closed(
            &[
                "memory_budget_mib",
                "cache_budget_mib",
                "chunk_target_mib",
                "max_source_file_mib",
                "max_record_mib",
                "max_in_flight",
                "workers",
                "backend",
                "scratch_budget_mib",
                "wall_seconds",
                "cleanup_seconds",
            ],
            &["native_json_depth"],
        )?;
        let mib = |key| -> Result<usize> {
            let n = value
                .require(key)?
                .integer(1, (usize::MAX / 1048576) as i128)?;
            Ok(n as usize * 1048576)
        };
        let memory = mib("memory_budget_mib")?;
        let file = mib("max_source_file_mib")?;
        let record = mib("max_record_mib")?;
        let chunk = mib("chunk_target_mib")?;
        let scratch = mib("scratch_budget_mib")?;
        let cache = value
            .require("cache_budget_mib")?
            .integer(0, (usize::MAX / 1048576) as i128)? as usize
            * 1048576;
        let workers = value.require("workers")?.integer(1, 4)? as usize;
        let in_flight = value
            .require("max_in_flight")?
            .integer(1, usize::MAX as i128)? as usize;
        if ![1, 2, 4].contains(&workers)
            || string(value, "backend")? != "rust"
            || cache > memory
            || chunk > memory
            || record > file
        {
            return Err(Error::contract("inconsistent native preparation settings"));
        }
        let wall = value.require("wall_seconds")?.number()?;
        let cleanup = value.require("cleanup_seconds")?.number()?;
        if !wall.is_finite() || !cleanup.is_finite() || cleanup <= 0. || wall <= cleanup {
            return Err(Error::contract("invalid native preparation time limits"));
        }
        let json_depth = match value.get("native_json_depth") {
            Some(value) => value.integer(1, 990)? as usize,
            None => 990,
        };
        Ok(Self {
            memory,
            file,
            record,
            chunk,
            scratch,
            workers,
            in_flight,
            json_depth,
        })
    }
}

pub struct Collection {
    pub directory: PathBuf,
    pub manifest: Document,
    pub report: Document,
    pub parent: usize,
    pub expected_source: Option<String>,
    _path_lease: Lease,
}
pub struct Session {
    pub budget: Arc<Budget>,
    pub control: Control,
    pub settings: Settings,
    pub sources: Sources,
    pub cleanup_control: Control,
    pub parents: BudgetVec<Parent>,
    pub collections: BudgetVec<Collection>,
    pub online: bool,
    pub config: FeatureConfig,
    pub fingerprint: String,
    pub authorized: bool,
    pub closed: bool,
    pub operation: String,
    pub dataset: Option<Document>,
    _overhead: Lease,
    pub stage: Option<crate::pipeline::Stage>,
    pub retained_python: Option<Lease>,
    pub final_consumer: Option<Lease>,
    pub proof_delivered: bool,
    pub telemetry: Arc<crate::telemetry::Telemetry>,
}
impl Session {
    #[cfg(test)]
    pub fn test_core(budget: &Arc<Budget>, depth: usize, fail_close: bool) -> Self {
        let control = Control::new(crate::budget::monotonic() + 60.).unwrap();
        let encoded = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        let document = json::parse(encoded.as_bytes(), budget, &control).unwrap();
        Self {
            budget: budget.clone(),
            control: control.clone(),
            cleanup_control: control.clone(),
            settings: Settings {
                memory: budget.limit(),
                file: 1024 * 1024,
                record: 1024 * 1024,
                chunk: 1024 * 1024,
                scratch: 1024 * 1024,
                workers: 1,
                in_flight: 1,
                json_depth: 990,
            },
            sources: Sources::new(budget.clone(), 1024 * 1024, 1024 * 1024),
            parents: BudgetVec::new(budget),
            collections: BudgetVec::new(budget),
            online: false,
            config: FeatureConfig {
                history: "none".to_owned(),
                max_elements: 3,
                hash_buckets: 32,
                history_window: 128,
            },
            fingerprint: String::new(),
            authorized: false,
            closed: false,
            operation: "summary".to_owned(),
            dataset: Some(document),
            _overhead: budget.reserve(65536).unwrap(),
            stage: if fail_close {
                Some(crate::pipeline::Stage::test_failed_close(budget))
            } else {
                None
            },
            retained_python: None,
            final_consumer: None,
            proof_delivered: false,
            telemetry: Arc::new(crate::telemetry::Telemetry::default()),
        }
    }
    pub fn open(
        plan: &[u8],
        settings: Settings,
        budget: Arc<Budget>,
        control: Control,
        wall_deadline: f64,
        telemetry: Arc<crate::telemetry::Telemetry>,
    ) -> Result<Self> {
        let control = control.with_json_depth(settings.json_depth)?;
        let cleanup_control = control.at_deadline(wall_deadline)?;
        // Bounded runtime/path/hash metadata and fixed native stack work are
        // reserved up front; variable buffers have their own capacity leases.
        let overhead = budget.reserve(65536)?;
        let plan = json::parse(plan, &budget, &control)?;
        let p = &plan.root;
        let kind = string(p, "kind")?;
        let online = kind == "online";
        if !["offline", "build", "online"].contains(&kind) {
            return Err(Error::contract("unsupported native preparation plan"));
        }
        p.closed(
            &["kind", "encoder"],
            &[
                "ppo_runs",
                "validation_runs",
                "dataset_path",
                "collection_path",
                "parent_path",
                "operation",
            ],
        )?;
        let operation = p
            .get("operation")
            .map(Value::string)
            .transpose()?
            .unwrap_or(if online { "ppo" } else { "prepare" });
        if !["summary", "index", "prepare", "ppo"].contains(&operation)
            || online != (operation == "ppo")
        {
            return Err(Error::contract("unsupported native operation/layout"));
        }
        let config = FeatureConfig::from_value(p.require("encoder")?)?;
        let mut sources =
            Sources::with_telemetry(budget.clone(), settings.file, settings.record, &telemetry);
        let dataset = if let Some(path) = p.get("dataset_path").filter(|v| !v.is_null()) {
            Some(sources.json(Path::new(path.string()?), false, &control)?)
        } else {
            None
        };
        let roots = dataset.as_ref().map(|d| &d.root).unwrap_or(p);
        let mut parents = BudgetVec::new(&budget);
        if online {
            parents.push(Parent::open(
                Path::new(string(p, "parent_path")?),
                false,
                &mut sources,
                &budget,
                &control,
            )?)?;
        } else {
            let runs = roots.require("ppo_runs")?.array()?;
            let validations = roots.require("validation_runs")?.array()?;
            if runs.len() != 3 || validations.len() != 3 {
                return Err(Error::contract(
                    "fixed dataset requires three ordered parents",
                ));
            }
            for (index, path) in runs.iter().enumerate() {
                let parent = Parent::open(
                    Path::new(path.string()?),
                    true,
                    &mut sources,
                    &budget,
                    &control,
                )?;
                if !controls::numeric_field(&parent.state.root, "seed", [17, 29, 43][index]) {
                    return Err(Error::contract(
                        "fixed ordered PPO learning seeds must be 17,29,43",
                    ));
                }
                if parents
                    .iter()
                    .any(|prior: &Parent| prior.directory == parent.directory)
                {
                    return Err(Error::contract("duplicate PPO parent"));
                }
                parents.push(parent)?;
            }
        }
        let mut collections = BudgetVec::new(&budget);
        if online {
            collections.push(open_collection(
                Path::new(string(p, "collection_path")?),
                0,
                None,
                &parents[0],
                true,
                &mut sources,
                &budget,
                &control,
            )?)?;
        } else {
            let validations = roots.require("validation_runs")?.array()?;
            for (index, parent) in parents.iter().enumerate() {
                for update in parent.updates.iter() {
                    let u = &update.root;
                    collections.push(open_collection(
                        &parent.directory.join(string(u, "collection")?),
                        index,
                        Some(string(u.require("source_proof")?, "source_set_sha256")?.to_owned()),
                        parent,
                        false,
                        &mut sources,
                        &budget,
                        &control,
                    )?)?;
                }
                collections.push(open_collection(
                    Path::new(validations[index].string()?),
                    index,
                    None,
                    parent,
                    false,
                    &mut sources,
                    &budget,
                    &control,
                )?)?;
            }
            for (i, left) in collections.iter().enumerate() {
                if parents.iter().any(|p| p.directory == left.directory)
                    || collections[..i].iter().any(|right| {
                        left.directory == right.directory
                            || left.manifest.root.require("run_id").ok()
                                == right.manifest.root.require("run_id").ok()
                    })
                {
                    return Err(Error::contract("duplicate source run identity"));
                }
            }
            crate::budget::sort_by(
                &mut collections,
                |left, right| {
                    left.directory
                        .as_os_str()
                        .as_encoded_bytes()
                        .iter()
                        .chain(b"/")
                        .cmp(
                            right
                                .directory
                                .as_os_str()
                                .as_encoded_bytes()
                                .iter()
                                .chain(b"/"),
                        )
                },
                &control,
            )?;
        }
        let mut hasher = Sha256::new();
        for entry in sources.entries.iter() {
            hasher.update(entry.path.as_os_str().as_encoded_bytes());
            hasher.update([0]);
            hasher.update(entry.digest.as_bytes());
            hasher.update([0]);
        }
        let fingerprint = format!("{:x}", hasher.finalize());
        Ok(Self {
            budget,
            control,
            cleanup_control,
            settings,
            sources,
            parents,
            collections,
            online,
            operation: operation.to_owned(),
            config,
            fingerprint,
            authorized: false,
            closed: false,
            dataset,
            _overhead: overhead,
            stage: None,
            retained_python: None,
            final_consumer: None,
            proof_delivered: false,
            telemetry,
        })
    }
    /// Full escaped UTF8 authorization envelope bound, before Python constructs it.
    pub fn authorization_bound(&self) -> Result<usize> {
        let mut total = 256usize; // Outer fingerprint, keys, separators and brackets.
        for collection in self.collections.iter() {
            self.control.check()?;
            let m = &collection.manifest.root;
            let parent = &self.parents[collection.parent];
            let quoted_path = json::canonical(
                &json::text(path_text(&parent.directory)?, &self.budget)?,
                &self.budget,
                &self.control,
            )?
            .len();
            let quoted_directory = json::canonical(
                &json::text(path_text(&collection.directory)?, &self.budget)?,
                &self.budget,
                &self.control,
            )?
            .len();
            let training = string(m, "collection_kind")? == "ppo_batch";
            let models = if training {
                let generation = m
                    .require("ppo")?
                    .require("collection_generation")?
                    .integer(0, 1000)? as usize;
                let pool = parent.generations[generation]
                    .root
                    .require("parent_controls")?
                    .require("snapshot_pool")?
                    .array()?
                    .len();
                parent.generations.len().min(
                    pool.checked_add(1)
                        .ok_or_else(|| Error::resource("authorization size overflow"))?,
                )
            } else {
                1
            };
            // Recipe UUIDs/labels/IDs are bounded ASCII; each group is two
            // episodes with uint64 sampling seeds. Paths use escaped UTF8 size.
            let schedule = (if training { 8usize } else { 32 })
                .checked_mul(1024)
                .ok_or_else(|| Error::resource("authorization size overflow"))?;
            let bindings = quoted_path
                .checked_add(320)
                .and_then(|n| n.checked_mul(models))
                .and_then(|n| n.checked_add(256))
                .ok_or_else(|| Error::resource("authorization size overflow"))?;
            total = total
                .checked_add(schedule)
                .and_then(|n| n.checked_add(bindings))
                .and_then(|n| n.checked_add(quoted_directory))
                .and_then(|n| n.checked_add(128))
                .ok_or_else(|| Error::resource("authorization size overflow"))?;
        }
        if total > self.settings.file {
            return Err(Error::resource(
                "authorization bound exceeds max_source_file_mib",
            ));
        }
        Ok(total)
    }
    pub fn summary(&self) -> Result<(BudgetVec<u8>, usize)> {
        self.check()?;
        let authorization_bound = self.authorization_bound()?;
        let mut parents = Vec::new();
        for parent in self.parents.iter() {
            parents.push(parent.summary(&self.budget, &self.control)?);
        }
        let mut collections = Vec::new();
        for collection in self.collections.iter() {
            let m = &collection.manifest.root;
            let link = m.require("ppo")?;
            collections.push(json::object(
                [
                    (
                        "directory",
                        json::text(path_text(&collection.directory)?, &self.budget)?,
                    ),
                    (
                        "split",
                        json::copy(m.require("split")?, &self.budget, &self.control)?,
                    ),
                    (
                        "collection_kind",
                        json::copy(m.require("collection_kind")?, &self.budget, &self.control)?,
                    ),
                    (
                        "collection_generation",
                        json::copy(
                            link.require("collection_generation")?,
                            &self.budget,
                            &self.control,
                        )?,
                    ),
                    (
                        "batch_id",
                        json::copy(link.require("batch_id")?, &self.budget, &self.control)?,
                    ),
                    (
                        "parent_run_id",
                        json::copy(link.require("parent_run_id")?, &self.budget, &self.control)?,
                    ),
                    (
                        "parent_index",
                        json::integer(collection.parent as i128, &self.budget)?,
                    ),
                    (
                        "recorded_schedule_sha256",
                        json::text(
                            &controls::digest(m.require("schedule")?, &self.budget, &self.control)?,
                            &self.budget,
                        )?,
                    ),
                    (
                        "recorded_bindings_sha256",
                        json::text(
                            &controls::digest(
                                m.require("policy_bindings")?,
                                &self.budget,
                                &self.control,
                            )?,
                            &self.budget,
                        )?,
                    ),
                ],
                &self.budget,
            )?);
        }
        let summary = json::object(
            [
                ("parents", Value::Array(parents)),
                ("collections", Value::Array(collections)),
                (
                    "authorization_wire_bound_bytes",
                    json::integer(authorization_bound as i128, &self.budget)?,
                ),
                (
                    "control_fingerprint",
                    json::text(&self.fingerprint, &self.budget)?,
                ),
                (
                    "scope",
                    json::text(
                        if self.online {
                            "online-current-batch"
                        } else {
                            "offline-complete-parents"
                        },
                        &self.budget,
                    )?,
                ),
            ],
            &self.budget,
        )?;
        Ok((
            json::canonical(&summary, &self.budget, &self.control)?,
            authorization_bound,
        ))
    }
    pub fn authorize(&mut self, encoded: &[u8]) -> Result<()> {
        self.check()?;
        if encoded.len() > self.settings.file {
            return Err(Error::resource("authorization exceeds max_source_file_mib"));
        }
        if self.authorized {
            return Err(Error::contract("native session already authorized"));
        }
        let authorization = json::parse(encoded, &self.budget, &self.control)?;
        let a = &authorization.root;
        a.closed(&["control_fingerprint", "collections"], &[])?;
        if string(a, "control_fingerprint")? != self.fingerprint {
            return Err(Error::contract("stale native authorization controls"));
        }
        let expected = a.require("collections")?.array()?;
        if expected.len() != self.collections.len() {
            return Err(Error::contract(
                "incomplete collection schedule authorization",
            ));
        }
        for (collection, expected) in self.collections.iter().zip(expected) {
            self.control.check()?;
            expected.closed(&["directory", "schedule", "bindings"], &[])?;
            if Path::new(string(expected, "directory")?) != collection.directory
                || !collection
                    .manifest
                    .root
                    .require("schedule")?
                    .eq_python(expected.require("schedule")?)
                || !collection
                    .manifest
                    .root
                    .require("policy_bindings")?
                    .eq_python(expected.require("bindings")?)
            {
                return Err(Error::contract(
                    "PPO source schedule/frozen policy bindings mismatch",
                ));
            }
        }
        self.sources.verify(&self.control)?;
        self.authorized = true;
        Ok(())
    }
    pub fn check(&self) -> Result<()> {
        if self.closed {
            return Err(Error::Closed);
        }
        self.control.check()
    }
    pub fn next_batch(
        &mut self,
        retained_python: usize,
        layout: crate::pipeline::ConsumerLayout,
    ) -> Result<Option<crate::pipeline::Batch>> {
        self.check()?;
        self.retained_python.take();
        self.retained_python = Some(self.budget.reserve(retained_python)?);
        if self.stage.is_none() {
            self.stage = Some(crate::pipeline::scan(self)?);
        }
        let stage = self.stage.as_mut().expect("staged sources");
        if stage.cursor == stage.episodes.len() {
            let mut decisions = 0usize;
            let mut candidates = 0usize;
            for episode in stage.episodes.iter() {
                self.control.check()?;
                decisions += episode.decisions;
                candidates += episode.candidates;
            }
            self.final_consumer.take();
            self.final_consumer = Some(self.budget.reserve(layout.bytes(
                decisions,
                candidates,
                stage.episodes.len(),
            )?)?);
            return Ok(None);
        }
        let width = if ["prepare", "ppo"].contains(&self.operation.as_str()) {
            self.config.width()
        } else {
            0
        };
        stage.next(
            &self.budget,
            &self.control,
            width,
            self.settings.chunk,
            layout,
            self.operation == "index",
            &self.telemetry,
        )
    }
    pub fn proof(&self) -> Result<BudgetVec<u8>> {
        self.check()?;
        let stage = self
            .stage
            .as_ref()
            .ok_or_else(|| Error::contract("native sources have not passed global validation"))?;
        json::canonical(&stage.proof, &self.budget, &self.control)
    }
    pub fn finalize(&mut self) -> Result<BudgetVec<u8>> {
        if self.closed {
            return Err(Error::Closed);
        }
        self.cleanup_control.check()?;
        let stage = self
            .stage
            .as_ref()
            .ok_or_else(|| Error::contract("native sources were not consumed"))?;
        if stage.cursor != stage.episodes.len() {
            return Err(Error::contract(
                "native complete-episode batches were not consumed",
            ));
        }
        if self.final_consumer.is_none() {
            return Err(Error::contract(
                "native EOF consumer admission was not completed",
            ));
        }
        self.sources.verify(&self.cleanup_control)?;
        let proof = json::canonical(&stage.proof, &self.budget, &self.cleanup_control)?;
        self.close()?;
        self.cleanup_control.check()?;
        Ok(proof)
    }
    pub fn close(&mut self) -> Result<()> {
        self.closed = true;
        if let Some(stage) = &mut self.stage {
            self.telemetry.scratch_peak(stage.scratch.peak());
            stage.close()?;
        }
        Ok(())
    }
}
pub fn dataset_manifest(
    session: &Session,
    source_hash: &str,
    counts: [usize; 2],
    action_hash: &str,
) -> Result<Value> {
    let budget = &session.budget;
    let control = &session.control;
    let mut parents = Vec::new();
    let mut parent_paths = Vec::new();
    let mut validation_paths = Vec::new();
    let mut parent_ids = Vec::new();
    let mut validation_ids = Vec::new();
    let mut collections = Vec::new();
    let mut cost = 0.;
    for (index, parent) in session.parents.iter().enumerate() {
        control.check()?;
        let state = &parent.state.root;
        parent_paths.push(json::text(path_text(&parent.directory)?, budget)?);
        parent_ids.push(json::copy(state.require("run_id")?, budget, control)?);
        let generation_cost = state.require("elapsed_seconds")?.number()?;
        let mut generations = Vec::new();
        for (number, record) in parent.generations.iter().enumerate() {
            let r = &record.root;
            generations.push(json::object(
                [
                    ("generation", json::integer(number as i128, budget)?),
                    (
                        "record_sha256",
                        json::text(
                            session.sources.digest(
                                &controls::generation_path(&parent.directory, number),
                                control,
                            )?,
                            budget,
                        )?,
                    ),
                    (
                        "inference_sha256",
                        json::copy(r.require("inference_sha256")?, budget, control)?,
                    ),
                    (
                        "weights_sha256",
                        json::copy(r.require("weights_sha256")?, budget, control)?,
                    ),
                ],
                budget,
            )?);
        }
        parents.push(json::object(
            [
                (
                    "directory",
                    json::text(path_text(&parent.directory)?, budget)?,
                ),
                (
                    "run_id",
                    json::copy(state.require("run_id")?, budget, control)?,
                ),
                (
                    "run_sha256",
                    json::text(
                        session
                            .sources
                            .digest(&parent.directory.join("run.json"), control)?,
                        budget,
                    )?,
                ),
                ("seed", json::copy(state.require("seed")?, budget, control)?),
                ("generations", Value::Array(generations)),
                (
                    "accepted_decisions",
                    json::copy(state.require("accepted_decisions")?, budget, control)?,
                ),
                (
                    "generation_cost_seconds",
                    json::copy(state.require("elapsed_seconds")?, budget, control)?,
                ),
                (
                    "artifact_registry_sha256",
                    json::text(
                        &controls::digest(&parent.registry.root, budget, control)?,
                        budget,
                    )?,
                ),
            ],
            budget,
        )?);
        let mut ordered = BudgetVec::new(budget);
        for collection in session.collections.iter().filter(|c| c.parent == index) {
            ordered.push(collection)?;
        }
        let key = |c: &&Collection| {
            let m = &c.manifest.root;
            let kind = string(m, "collection_kind").unwrap_or("");
            (
                kind == "ppo_validation",
                m.require("ppo")
                    .and_then(|v| v.require("collection_generation"))
                    .and_then(|v| v.integer(0, 1000))
                    .unwrap_or(0),
            )
        };
        crate::budget::sort_by(&mut ordered, |a, b| key(a).cmp(&key(b)), control)?;
        for collection in ordered.iter() {
            control.check()?;
            let m = &collection.manifest.root;
            if string(m, "split")? == "validation" {
                validation_paths.push(json::text(path_text(&collection.directory)?, budget)?);
                validation_ids.push(json::copy(m.require("run_id")?, budget, control)?);
                cost += generation_cost + m.require("elapsed_seconds")?.number()?;
            }
            let actions = m.require("effective_config")?.require("actions")?;
            let encoded = json::canonical(actions, budget, control)?;
            collections.push(json::object(
                [
                    (
                        "directory",
                        json::text(path_text(&collection.directory)?, budget)?,
                    ),
                    ("run_id", json::copy(m.require("run_id")?, budget, control)?),
                    (
                        "manifest_sha256",
                        json::text(
                            session
                                .sources
                                .digest(&collection.directory.join("manifest.json"), control)?,
                            budget,
                        )?,
                    ),
                    (
                        "collection_seconds",
                        json::float(m.require("elapsed_seconds")?.number()?)?,
                    ),
                    ("split", json::copy(m.require("split")?, budget, control)?),
                    (
                        "config_sha256",
                        json::text(
                            &controls::digest(m.require("effective_config")?, budget, control)?,
                            budget,
                        )?,
                    ),
                    (
                        "semantic_config",
                        json::text(
                            std::str::from_utf8(&encoded)
                                .map_err(|_| Error::contract("non-UTF8 semantic config"))?,
                            budget,
                        )?,
                    ),
                ],
                budget,
            )?);
        }
    }
    let key = json::object(
        [
            ("parent_run_ids", Value::Array(parent_ids)),
            ("validation_run_ids", Value::Array(validation_ids)),
        ],
        budget,
    )?;
    let cost_key = controls::digest(&key, budget, control)?;
    let namespaces = json::object(
        [
            ("train", json::text("ppo-train-v1", budget)?),
            ("validation", json::text("fixed-existing-32", budget)?),
            ("test", json::text("reserved-not-run", budget)?),
        ],
        budget,
    )?;
    let mut manifest = json::object(
        [
            ("schema_version", json::integer(1, budget)?),
            (
                "selection",
                json::text("all-explicit-learner-accepted-natural-returns-v1", budget)?,
            ),
            ("ppo_runs", Value::Array(parent_paths)),
            ("validation_runs", Value::Array(validation_paths)),
            ("parents", Value::Array(parents)),
            ("collections", Value::Array(collections)),
            ("source_set_sha256", json::text(source_hash, budget)?),
            ("train_decisions", json::integer(counts[0] as i128, budget)?),
            (
                "validation_decisions",
                json::integer(counts[1] as i128, budget)?,
            ),
            ("validation_games", json::integer(192, budget)?),
            ("shared_generation_seconds", json::float(cost)?),
            ("shared_generation_cost_key", json::text(&cost_key, budget)?),
            ("action_config_hash", json::text(action_hash, budget)?),
            (
                "label_recipe",
                json::text("behavior-selected-natural-return-gamma1-v1", budget)?,
            ),
            ("split_namespaces", namespaces),
        ],
        budget,
    )?;
    let content_hash = controls::digest(&manifest, budget, control)?;
    if let Value::Object(entries) = &mut manifest {
        entries.insert(
            "content_sha256".to_owned(),
            json::text(&content_hash, budget)?,
        );
    }
    Ok(manifest)
}
pub fn path_text(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| Error::contract("non-UTF8 source path"))
}

fn open_collection(
    directory: &Path,
    parent_index: usize,
    expected_source: Option<String>,
    parent: &Parent,
    online: bool,
    sources: &mut Sources,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<Collection> {
    let directory = fs::canonicalize(directory)?;
    sources.inventory(
        &directory,
        crate::source::InventoryKind::ParticipantJournals,
        control,
    )?;
    sources.inventory(
        &directory,
        crate::source::InventoryKind::ParticipantReplays,
        control,
    )?;
    let path_lease = budget.reserve(directory.as_os_str().len() + 256)?;
    let manifest = sources.json(&directory.join("manifest.json"), false, control)?;
    let m = &manifest.root;
    m.require("schema_version")?.integer(4, 4)?;
    if string(m, "status")? != "complete" {
        return Err(Error::contract(
            "training requires a finalized complete healthy run",
        ));
    }
    let kind = string(m, "collection_kind")?;
    if !["ppo_batch", "ppo_validation"].contains(&kind) {
        return Err(Error::contract("unsupported schema-4 collection kind"));
    }
    let link = m.require("ppo")?;
    link.closed(
        &[
            "parent_run_id",
            "parent_run_path",
            "batch_id",
            "collection_generation",
            "learner_seats",
        ],
        &[],
    )?;
    for key in ["parent_run_id", "parent_run_path", "batch_id"] {
        link.require(key)?.string()?;
    }
    let generation = link.require("collection_generation")?.integer(0, 1000)? as usize;
    if generation >= parent.generations.len()
        || fs::canonicalize(string(link, "parent_run_path")?)? != parent.directory
    {
        return Err(Error::contract("collection parent/generation mismatch"));
    }
    equal(
        link.require("parent_run_id")?,
        parent.state.root.require("run_id")?,
        "collection parent identity mismatch",
    )?;
    let integration = parent
        .state
        .root
        .require("identity")?
        .require("integration")?;
    for (a, b) in [
        ("binary_sha256", "binary_sha256"),
        ("server_config_sha256", "server_config_sha256"),
    ] {
        equal(
            m.require(a)?,
            integration.require(b)?,
            "collection Simulator integration differs from frozen parent identity",
        )?;
    }
    for (a, b) in [
        ("revision", "source_revision"),
        ("spec_sha256", "source_spec_sha256"),
    ] {
        equal(
            m.require("simulator")?.require(a)?,
            integration.require(b)?,
            "collection Simulator integration differs from frozen parent identity",
        )?;
    }
    let updates = parent.state.root.require("updates")?.integer(0, 1000)? as usize;
    if online {
        if string(&parent.state.root, "status")? != "running"
            || kind != "ppo_batch"
            || generation != updates
        {
            return Err(Error::contract(
                "owned online context requires exact current frozen batch",
            ));
        }
    } else if kind == "ppo_batch" {
        if generation >= updates {
            return Err(Error::contract("offline batch has no committed update"));
        }
        let update = &parent.updates[generation].root;
        if fs::canonicalize(parent.directory.join(string(update, "collection")?))? != directory {
            return Err(Error::contract(
                "offline batch differs from immutable update source",
            ));
        }
        equal(
            update.require("batch_id")?,
            link.require("batch_id")?,
            "offline batch differs from immutable update source",
        )?;
        equal(
            update.require("policy_bindings")?,
            m.require("policy_bindings")?,
            "collection frozen binding evidence mismatch",
        )?;
    } else if generation != updates {
        return Err(Error::contract(
            "validation must use completed final generation",
        ));
    }
    let split = string(m, "split")?;
    if split
        != if kind == "ppo_batch" {
            "train"
        } else {
            "validation"
        }
    {
        return Err(Error::contract(
            "source split is not permitted for training role",
        ));
    }
    let config = m.require("effective_config")?;
    if controls::digest(config, budget, control)? != string(m, "config_sha256")?
        || string(config.require("run")?, "split")? != split
    {
        return Err(Error::contract("run config provenance mismatch"));
    }
    let elapsed = m.require("elapsed_seconds")?.number()?;
    let wall = config.require("run")?.require("wall_seconds")?.number()?;
    if !elapsed.is_finite() || !wall.is_finite() || elapsed < 0. || elapsed > wall {
        return Err(Error::contract(
            "source collection exceeded/misses original budget",
        ));
    }
    let schedule = m.require("schedule")?.array()?;
    let episodes = m.require("episodes")?.array()?;
    let mut seen = BudgetVec::with_capacity(budget, episodes.len())?;
    for episode in episodes {
        control.check()?;
        let id = string(episode, "episode_id")?;
        if id.is_empty() || Path::new(id).file_name().and_then(|v| v.to_str()) != Some(id) {
            return Err(Error::contract("unsafe episode source path"));
        }
        if seen.contains(&id) {
            return Err(Error::contract("complete original schedule required"));
        }
        seen.push(id)?;
        if string(episode, "end_kind")? != "natural"
            || episode.require("outcome")?.is_null()
            || controls::field_truthy(episode, "failure")
            || episode
                .require("health")?
                .object()?
                .iter()
                .any(|(_, v)| controls::truthy(v))
            || !matches!(episode.get("replay_reconciled"), Some(Value::Bool(true)))
        {
            return Err(Error::contract(
                "all original episodes must be healthy natural outcomes",
            ));
        }
        let mut matches = 0;
        for group in schedule {
            for spec in group.require("episodes")?.array()? {
                control.check()?;
                if string(spec, "episode_id")? == id {
                    matches += 1;
                    let viewer = spec.require("learner_viewer")?.integer(1, 2)?;
                    equal(
                        link.require("learner_seats")?.require(id)?,
                        spec.require("learner_viewer")?,
                        "PPO schedule/learner linkage mismatch",
                    )?;
                    let _ = viewer;
                }
            }
        }
        if matches != 1 {
            return Err(Error::contract("complete original schedule required"));
        }
    }
    let mut scheduled_count = 0usize;
    for group in schedule {
        control.check()?;
        scheduled_count += group.require("episodes")?.array()?.len();
    }
    if episodes.is_empty()
        || scheduled_count != episodes.len()
        || link.require("learner_seats")?.object()?.len() != episodes.len()
        || kind == "ppo_batch" && (episodes.len() != 16 || schedule.len() != 8)
    {
        return Err(Error::contract("complete original schedule required"));
    }
    let report = sources.json(&directory.join("report.json"), false, control)?;
    if !controls::numeric_field(&report.root, "planned_games", episodes.len() as i128)
        || !controls::numeric_field(&report.root, "natural_denominator", episodes.len() as i128)
    {
        return Err(Error::contract("source report schedule count mismatch"));
    }
    Ok(Collection {
        directory,
        manifest,
        report,
        parent: parent_index,
        expected_source,
        _path_lease: path_lease,
    })
}
