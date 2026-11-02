//! One frozen TreeDataset2 worklist; never a fit or qualified-source authority.
use crate::{
    budget::{Budget, BudgetVec, Control, Lease},
    error::{Error, Result},
    json::{self, Document, Value, ValueExt},
    structured_controls::{checksum, digest},
    structured_session::Sources,
    structured_source::{self, CollectionLimits, CollectionScan, ParentScan, TreeSourceKind},
    structured_staging::{Banks, Stage},
};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    sync::Arc,
};
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::contract(message))
    }
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value.require(key)?.string()
}
fn doc(value: &Value, b: &Arc<Budget>, c: &Control) -> Result<Document> {
    let bytes = json::canonical(value, b, c)?;
    json::parse(&bytes, b, c)
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Controls,
    Ready,
    Exhausted,
    Staged,
    Authorized,
    Exported,
    Finalized,
    Closed,
    Failed,
}
pub(crate) struct Work {
    pub parent: usize,
    pub generation: usize,
    pub directory: PathBuf,
    pub validation: bool,
    pub item: Document,
    // The original validation manifest is frozen before any train scan. This
    // is a snapshot owner, not a retained full collection or fresh baseline.
    pub validation_control: Option<Sources>,
}
struct Current {
    scan: CollectionScan,
    summary: Document,
    authorization: Document,
}
pub(crate) struct TreeDataset {
    pub budget: Arc<Budget>,
    pub control: Control,
    pub parents: BudgetVec<ParentScan>,
    pub work: BudgetVec<Work>,
    pub projection: Document,
    pub summary: Document,
    authorization: Document,
    phase: Phase,
    pub position: usize,
    source_bytes: usize,
    record_bytes: usize,
    scratch_bytes: usize,
    chunk_bytes: usize,
    stage: Option<Stage>,
    current: Option<Current>,
    retired: BudgetVec<Sources>,
    evidence: BudgetVec<Document>,
    _plan: Document,
    _metadata: BudgetVec<Lease>,
}
impl TreeDataset {
    pub(crate) fn open(
        encoded: &[u8],
        budget: Arc<Budget>,
        control: Control,
        source_bytes: usize,
        record_bytes: usize,
        scratch_bytes: usize,
        chunk_bytes: usize,
    ) -> Result<Self> {
        control.check()?;
        require(
            encoded.len() <= 65536,
            "TreeDataset plan exceeds bounded wire",
        )?;
        let plan = json::parse(encoded, &budget, &control)?;
        let p = &plan.root;
        p.closed(
            &[
                "schema_version",
                "kind",
                "parents",
                "encoder",
                "action_contract",
                "purpose",
                "synthetic_capacity",
            ],
            &[],
        )?;
        p.require("schema_version")?.integer(2, 2)?;
        require(
            text(p, "kind")? == "structured-tree-dataset-plan",
            "TreeDataset exact plan kind required",
        )?;
        require(
            p.require("parents")?.array()?.len() == 3,
            "TreeDataset requires exactly3 ordered parents",
        )?;
        require(
            text(p, "purpose")? == "component" && !p.require("synthetic_capacity")?.boolean()?,
            "TreeDataset engineering component/false-capacity input required",
        )?;
        crate::structured_controls::contract(p.require("action_contract")?)?;
        let layout =
            crate::structured_controls::tree_layout(p.require("encoder")?, &budget, &control)?;
        let mut projection_value = json::copy(&layout.root, &control)?;
        projection_value.as_object_mut().unwrap().insert(
            "input_recipe".into(),
            json::text("participant-structured-v1"),
        );
        let projection = doc(&projection_value, &budget, &control)?;
        let mut parents = BudgetVec::new(&budget);
        let mut work = BudgetVec::new(&budget);
        let mut metadata = BudgetVec::new(&budget);
        metadata.push(budget.reserve(65536)?)?;
        let mut parent_paths = BTreeSet::new();
        let mut validation_paths = Vec::new();
        let mut all_paths = BTreeSet::new();
        let mut homogeneous: Option<String> = None;
        let mut copy_leases = BudgetVec::new(&budget);
        let mut parent_items = Vec::new();
        let mut parent_acks = Vec::new();
        for (pi, entry) in p.require("parents")?.array()?.iter().enumerate() {
            control.check()?;
            entry.closed(&["directory", "validation_directory"], &[])?;
            for key in ["directory", "validation_directory"] {
                require(
                    !text(entry, key)?.is_empty(),
                    "TreeDataset nonempty source directory required",
                )?;
            }
            metadata.push(budget.reserve(
                2048 + 4
                    * (text(entry, "directory")?.len()
                        + text(entry, "validation_directory")?.len()),
            )?)?;
            let mut sources = Sources::new(
                Path::new(text(entry, "directory")?),
                budget.clone(),
                control.clone(),
                source_bytes,
            )?;
            let directory = sources.root.clone();
            require(
                parent_paths.insert(directory.clone()),
                "duplicate TreeDataset parent directory",
            )?;
            metadata.push(budget.reserve(2048 + 4 * directory.as_os_str().len())?)?;
            let summary = crate::structured_controls::parent(&mut sources)?;
            let state = sources.json(&directory.join("run.json"))?;
            require(
                text(&state.root, "status")? == "complete",
                "TreeDataset requires complete healthy original parents",
            )?;
            require(
                state.root.require("seed")?.integer(0, i32::MAX as i128)? == [17, 29, 43][pi],
                "TreeDataset parent order must be17/29/43",
            )?;
            let config = state.root.require("effective_config")?;
            let model = summary.root.require("model")?;
            require(
                ["mlp", "gru", "transformer"].contains(&text(model, "model_id")?),
                "TreeDataset source must be neural2",
            )?;
            require(
                model
                    .require("event")?
                    .require("aggregate_recipe")?
                    .string()?
                    == "public-exact-v2",
                "TreeDataset cannot reinterpret old62 sources",
            )?;
            structured_source::input_classification(
                p,
                config.require("actions")?.require("action_contract")?,
            )?;
            let max_positions = p
                .require("encoder")?
                .require("max_positions")?
                .integer(1, 128)?;
            require(
                max_positions
                    >= config
                        .require("actions")?
                        .require("max_sequence_length")?
                        .integer(1, 128)?,
                "Tree encoder does not cover original ordered action capacity",
            )?;
            let cohort = json::object([
                ("model", json::copy(model, &control)?),
                (
                    "semantics",
                    json::copy(summary.root.require("semantics")?, &control)?,
                ),
                (
                    "actions_sha256",
                    json::text(&digest(config.require("actions")?, &budget, &control)?),
                ),
                (
                    "execution",
                    json::copy(state.root.require("execution")?, &control)?,
                ),
                (
                    "action_contract",
                    json::copy(p.require("action_contract")?, &control)?,
                ),
            ]);
            let cohort_sha = digest(&cohort, &budget, &control)?;
            if let Some(expected) = &homogeneous {
                require(
                    *expected == cohort_sha,
                    "TreeDataset heterogeneous neural/action/execution cohort",
                )?;
            } else {
                homogeneous = Some(cohort_sha);
            }
            let parent = ParentScan {
                directory: directory.clone(),
                sources,
                summary,
                state,
            };
            let copied = json::canonical(&parent.summary.root, &budget, &control)?.len();
            copy_leases.push(
                budget.reserve(
                    copied
                        .checked_mul(32)
                        .and_then(|n| n.checked_add(65536))
                        .ok_or_else(|| Error::resource("TreeDataset parent metadata overflow"))?,
                )?,
            )?;
            let (mut item, ack) =
                structured_source::compact_parent(pi, &parent, &budget, &control)?;
            item.as_object_mut().unwrap().insert(
                "elapsed_seconds".into(),
                json::copy(parent.state.root.require("elapsed_seconds")?, &control)?,
            );
            item.as_object_mut().unwrap().insert(
                "registry_sha256".into(),
                json::text(
                    parent
                        .sources
                        .verified_sha(&parent.directory.join("artifact-registry.json"))?,
                ),
            );
            let mut short_ack = json::object([]);
            for key in [
                "parent_index",
                "config_sha256",
                "resume_identity",
                "controls_sha256",
            ] {
                short_ack
                    .as_object_mut()
                    .unwrap()
                    .insert(key.into(), json::copy(ack.require(key)?, &control)?);
            }
            short_ack.as_object_mut().unwrap().insert(
                "sealed_sha256".into(),
                json::text(&digest(ack.require("sealed")?, &budget, &control)?),
            );
            parent_items.push(item);
            parent_acks.push(short_ack);
            validation_paths.push(PathBuf::from(text(entry, "validation_directory")?));
            parents.push(parent)?;
        }
        for pi in 0..parents.len() {
            let parent = &mut parents[pi];
            for generation in 1..=parent.state.root.require("updates")?.integer(1, 1000)? as usize {
                control.check()?;
                let record = &parent.summary.root.require("generations")?.array()?[generation];
                let update = parent.sources.checked_json(
                    &parent
                        .directory
                        .join("updates")
                        .join(format!("{generation:06}.json")),
                    record.require("update_sha256")?,
                )?;
                let path = structured_source::owned_relative(
                    &parent.directory,
                    update.root.require("collection")?,
                )?;
                require(
                    path == parent
                        .directory
                        .join("collections")
                        .join(format!("batch-{:06}", generation - 1)),
                    "TreeDataset committed train batch path differs",
                )?;
                require(
                    all_paths.insert(path.clone()),
                    "TreeDataset duplicate committed collection ownership",
                )?;
                let numeric = structured_source::owned_relative(
                    &parent.directory,
                    update.root.require("numeric")?,
                )?;
                let item =
                    json::object([
                        ("work_index", json::integer(work.len() as i128)),
                        ("parent_index", json::integer(pi as i128)),
                        ("collection_kind", json::text("ppo_batch")),
                        ("generation", json::integer((generation - 1) as i128)),
                        (
                            "directory",
                            json::text(
                                path.to_str()
                                    .ok_or_else(|| Error::contract("nonUTF8 train directory"))?,
                            ),
                        ),
                        (
                            "manifest_sha256",
                            json::text(checksum(
                                update.root.require("collection_manifest_sha256")?,
                            )?),
                        ),
                        (
                            "numeric",
                            json::object([
                                (
                                    "directory",
                                    json::text(numeric.to_str().ok_or_else(|| {
                                        Error::contract("nonUTF8 numeric directory")
                                    })?),
                                ),
                                (
                                    "sha256",
                                    json::text(checksum(update.root.require("numeric_sha256")?)?),
                                ),
                                (
                                    "files",
                                    json::copy(update.root.require("numeric_files")?, &control)?,
                                ),
                            ]),
                        ),
                    ]);
                metadata.push(budget.reserve(2048 + 4 * path.as_os_str().len())?)?;
                work.push(Work {
                    parent: pi,
                    generation: generation - 1,
                    directory: path,
                    validation: false,
                    item: doc(&item, &budget, &control)?,
                    validation_control: None,
                })?;
            }
        }
        for (pi, path) in validation_paths.iter().enumerate() {
            control.check()?;
            let mut owner = Sources::new(path, budget.clone(), control.clone(), source_bytes)?;
            let path = owner.root.clone();
            require(
                !parents
                    .iter()
                    .any(|parent| path.starts_with(&parent.directory)),
                "final validation output must be outside all parents",
            )?;
            require(
                all_paths.insert(path.clone()),
                "TreeDataset duplicate final validation directory",
            )?;
            let manifest = owner.json(&path.join("manifest.json"))?;
            require(
                manifest.root.require("schema_version")?.integer(4, 4)? == 4
                    && text(&manifest.root, "collection_kind")? == "ppo_validation",
                "TreeDataset final validation Collection4 required",
            )?;
            let item = json::object([
                ("work_index", json::integer(work.len() as i128)),
                ("parent_index", json::integer(pi as i128)),
                ("collection_kind", json::text("ppo_validation")),
                (
                    "generation",
                    json::copy(parents[pi].state.root.require("updates")?, &control)?,
                ),
                (
                    "directory",
                    json::text(
                        path.to_str()
                            .ok_or_else(|| Error::contract("nonUTF8 validation directory"))?,
                    ),
                ),
                (
                    "manifest_sha256",
                    json::text(owner.verified_sha(&path.join("manifest.json"))?),
                ),
                ("numeric", Value::Null),
            ]);
            metadata.push(budget.reserve(2048 + 4 * path.as_os_str().len())?)?;
            work.push(Work {
                parent: pi,
                generation: parents[pi]
                    .state
                    .root
                    .require("updates")?
                    .integer(1, 1000)? as usize,
                directory: path,
                validation: true,
                item: doc(&item, &budget, &control)?,
                validation_control: Some(owner),
            })?;
        }
        let mut copied = 65536usize;
        for item in work.iter() {
            copied = copied
                .checked_add(json::canonical(&item.item.root, &budget, &control)?.len())
                .ok_or_else(|| Error::resource("TreeDataset worklist metadata overflow"))?;
        }
        let _summary_copy = json::admit(copied, &budget)?;
        let items = json::array(
            work.iter()
                .map(|v| json::copy(&v.item.root, &control))
                .collect::<Result<Vec<_>>>()?,
        );
        let projection_sha = digest(&projection.root, &budget, &control)?;
        let fingerprint = digest(
            &json::object([
                ("recipe", json::text("structured-tree-dataset-controls-v2")),
                ("plan", json::copy(p, &control)?),
                (
                    "parents",
                    json::array(
                        parent_items
                            .iter()
                            .map(|item| json::copy(item.require("control_fingerprint")?, &control))
                            .collect::<Result<Vec<_>>>()?,
                    ),
                ),
                ("worklist", json::copy(&items, &control)?),
            ]),
            &budget,
            &control,
        )?;
        let summary = doc(
            &json::object([
                ("schema_version", json::integer(2)),
                ("kind", json::text("structured-tree-dataset-controls")),
                ("scope", json::text("engineering-component-originals")),
                ("qualified", Value::Bool(false)),
                ("parent_qualification", json::text("not_performed")),
                ("control_fingerprint", json::text(&fingerprint)),
                (
                    "input_binding",
                    json::object([
                        (
                            "action_contract",
                            json::copy(p.require("action_contract")?, &control)?,
                        ),
                        ("purpose", json::text("component")),
                        ("synthetic_capacity", Value::Bool(false)),
                    ]),
                ),
                ("tree_projection", json::copy(&projection.root, &control)?),
                ("parents", json::array(parent_items)),
                ("worklist", items),
            ]),
            &budget,
            &control,
        )?;
        let authorization = doc(
            &json::object([
                ("schema_version", json::integer(2)),
                (
                    "kind",
                    json::text("structured-tree-dataset-controls-authorization"),
                ),
                ("control_fingerprint", json::text(&fingerprint)),
                ("tree_projection_sha256", json::text(&projection_sha)),
                ("parents", json::array(parent_acks)),
                (
                    "worklist_sha256",
                    json::text(&digest(
                        summary.root.require("worklist")?,
                        &budget,
                        &control,
                    )?),
                ),
            ]),
            &budget,
            &control,
        )?;
        // Exactly one initial full control sweep, then current-work checks and
        // one final full sweep. Never rewalk every old Source on each cohort.
        for parent in parents.iter() {
            parent.sources.recheck()?;
            parent.sources.final_snapshots()?;
        }
        for item in work.iter() {
            if let Some(owner) = &item.validation_control {
                owner.recheck()?;
                owner.final_snapshots()?;
            }
        }
        Ok(Self {
            budget: budget.clone(),
            control,
            parents,
            work,
            projection,
            summary,
            authorization,
            phase: Phase::Controls,
            position: 0,
            source_bytes,
            record_bytes,
            scratch_bytes,
            chunk_bytes,
            stage: None,
            current: None,
            retired: BudgetVec::new(&budget),
            evidence: BudgetVec::new(&budget),
            _plan: plan,
            _metadata: metadata,
        })
    }
    pub(crate) fn control_summary(&self) -> Result<crate::budget::BudgetVec<u8>> {
        self.control.check()?;
        require(
            self.phase == Phase::Controls,
            "TreeDataset controls available only before authorization",
        )?;
        json::canonical(&self.summary.root, &self.budget, &self.control)
    }
    pub(crate) fn authorize_controls(&mut self, bytes: &[u8]) -> Result<()> {
        self.control.check()?;
        require(
            self.phase == Phase::Controls,
            "one TreeDataset control authorization required",
        )?;
        let value = json::parse(bytes, &self.budget, &self.control)?;
        require(
            json::canonical(&value.root, &self.budget, &self.control)?[..]
                == json::canonical(&self.authorization.root, &self.budget, &self.control)?[..],
            "TreeDataset closed control authorization mismatch",
        )?;
        self.phase = Phase::Ready;
        Ok(())
    }
}

#[cfg(test)]
mod selected_tests {
    use super::*;
    use crate::{
        structured_journal::{Journal, OriginalHeader},
        structured_staging::Stage,
    };
    use sha2::{Digest, Sha256};
    #[test]
    fn original_sixteen_selected_roles_match_python_selected_tokens_at_different_tree_window() {
        let budget = Budget::new(512 << 20);
        let control = Control::new(crate::budget::monotonic() + 60.0).unwrap();
        let fixture = json::parse(
            include_bytes!("../tests/fixtures/structured_tree_selected_inputs.json"),
            &budget,
            &control,
        )
        .unwrap();
        assert_eq!(
            fixture
                .root
                .require("episodes")
                .unwrap()
                .array()
                .unwrap()
                .len(),
            16
        );
        assert_eq!(
            fixture
                .root
                .require("row_sha256")
                .unwrap()
                .array()
                .unwrap()
                .len(),
            32
        );
        assert_eq!(
            fixture
                .root
                .require("different_history_frames")
                .unwrap()
                .integer(1, 1000)
                .unwrap(),
            32
        );
        let root = std::env::temp_dir().join(format!(
            "gfb-tree-selected-{}-{}",
            std::process::id(),
            crate::budget::monotonic().to_bits()
        ));
        std::fs::create_dir(&root).unwrap();
        let mut stage = Stage::new(budget.clone(), 64 << 20, 65536, control.clone()).unwrap();
        let mut originals = Vec::new();
        for (index, episode) in fixture
            .root
            .require("episodes")
            .unwrap()
            .array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let directory = root.join(format!("e-{index}"));
            std::fs::create_dir(&directory).unwrap();
            let name = episode.require("journal_file").unwrap().string().unwrap();
            let raw = episode.require("journal").unwrap().string().unwrap();
            std::fs::write(directory.join(name), raw).unwrap();
            for (name, hex) in episode.require("columns").unwrap().object().unwrap() {
                let hex = hex.string().unwrap();
                let bytes = (0..hex.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                    .collect::<Vec<_>>();
                std::fs::write(directory.join(name), bytes).unwrap();
            }
            let h = json::parse(raw.lines().next().unwrap().as_bytes(), &budget, &control).unwrap();
            let mut identity = json::object([]);
            for key in [
                "run_id",
                "game_id",
                "episode_id",
                "viewer",
                "policy",
                "model_id",
                "model_version",
                "generation_schema",
                "action_config_hash",
                "sampling_seed",
                "caps",
                "aggregate_window",
                "token_signature",
                "action_contract",
                "purpose",
                "synthetic_capacity",
            ] {
                identity
                    .as_object_mut()
                    .unwrap()
                    .insert(key.into(), h.root.require(key).unwrap().clone());
            }
            let declaration = OriginalHeader::new(
                Some(fixture.root.require("model").unwrap()),
                &identity,
                &budget,
                &control,
            )
            .unwrap();
            let journal = Journal::original_tree(
                &directory,
                name,
                budget.clone(),
                control.clone(),
                64 << 20,
                8 << 20,
                &declaration,
                Some((&mut stage, index)),
                1,
            )
            .unwrap();
            let metadata = journal
                .staged_tree_episode(&mut stage, index, &Value::Null)
                .unwrap();
            assert_eq!(
                metadata.require("frames").unwrap().array().unwrap().len(),
                2
            );
            assert!(metadata.get("selections").is_none());
            for (position, frame) in metadata
                .require("frames")
                .unwrap()
                .array()
                .unwrap()
                .iter()
                .enumerate()
            {
                assert_eq!(
                    frame
                        .require("selected_index")
                        .unwrap()
                        .integer(0, 0)
                        .unwrap(),
                    0
                );
                assert_eq!(
                    frame.require("original_selection_index").unwrap(),
                    &fixture
                        .root
                        .require("selected_indices")
                        .unwrap()
                        .array()
                        .unwrap()[index]
                        .array()
                        .unwrap()[position]
                );
            }
            originals.push(journal.into_sources());
        }
        let descriptions = stage.descriptors().unwrap();
        let banks = stage.export_retiring_disk().unwrap();
        assert!(banks.floats.is_empty());
        let expected = fixture
            .root
            .require("expected_arrays")
            .unwrap()
            .object()
            .unwrap();
        assert_eq!(descriptions.array().unwrap().len(), expected.len());
        for description in descriptions.array().unwrap() {
            let name = description.require("key").unwrap().string().unwrap();
            let original = &expected[name];
            assert_eq!(
                description.require("dtype").unwrap(),
                original.require("dtype").unwrap()
            );
            assert_eq!(
                description.require("shape").unwrap(),
                original.require("shape").unwrap()
            );
            assert!(
                !name.ends_with("old_logprob")
                    && !name.ends_with("old_value")
                    && !name.ends_with("selected")
            );
            let bank = description
                .require("bank_index")
                .unwrap()
                .integer(0, 10000)
                .unwrap() as usize;
            let start = description
                .require("element_offset")
                .unwrap()
                .integer(0, i64::MAX as i128)
                .unwrap() as usize;
            let count = description
                .require("element_count")
                .unwrap()
                .integer(0, i64::MAX as i128)
                .unwrap() as usize;
            let bytes = match description.require("dtype").unwrap().string().unwrap() {
                "uint8" => banks.bytes[bank][start..start + count].to_vec(),
                "int64" => banks.integers[bank][start..start + count]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
                _ => panic!("TreeDataset raw selected inputs must not emit float policy values"),
            };
            assert_eq!(
                format!("{:x}", Sha256::digest(bytes)),
                original.require("sha256").unwrap().string().unwrap(),
                "{name}"
            );
        }
        for owner in originals {
            owner.recheck().unwrap();
            owner.final_snapshots().unwrap();
        }
        stage.close().unwrap();
        drop(banks);
        drop(stage);
        drop(fixture);
        std::fs::remove_dir_all(root).unwrap();
    }
}

impl TreeDataset {
    fn fingerprint(&self) -> Result<&str> {
        text(&self.summary.root, "control_fingerprint")
    }
    fn current_checks(&mut self) -> Result<()> {
        self.control.check()?;
        require(
            !matches!(self.phase, Phase::Closed | Phase::Failed | Phase::Finalized),
            "TreeDataset owner is not active",
        )?;
        if self.position < self.work.len() {
            let work = &self.work[self.position];
            let parent = &mut self.parents[work.parent];
            let run = parent.directory.join("run.json");
            require(
                parent.sources.hash(&run)?
                    == text(
                        &self.summary.root.require("parents")?.array()?[work.parent],
                        "run_sha256",
                    )?,
                "original TreeDataset parent control changed",
            )?;
            if let Some(owner) = &work.validation_control {
                owner.recheck()?;
                owner.final_snapshots()?;
            }
        }
        if let Some(current) = &self.current {
            for owner in current.scan.sources.iter() {
                owner.recheck()?;
            }
            for owner in current.scan.sources.iter() {
                owner.final_snapshots()?;
            }
        }
        Ok(())
    }
    pub(crate) fn stage_next(&mut self) -> Result<bool> {
        self.control.check()?;
        require(
            self.phase == Phase::Ready,
            "TreeDataset must retire current payload before advance",
        )?;
        if self.position == self.work.len() {
            self.phase = Phase::Exhausted;
            return Ok(false);
        }
        self.current_checks()?;
        self.phase = Phase::Failed;
        self.stage = Some(Stage::new(
            self.budget.clone(),
            self.scratch_bytes,
            self.chunk_bytes,
            self.control.clone(),
        )?);
        let work = &self.work[self.position];
        let parent = &mut self.parents[work.parent];
        let scan = structured_source::inspect_tree_collection(
            &work.directory,
            (&parent.directory, &parent.state.root),
            &CollectionLimits {
                source_bytes: self.source_bytes,
                record_bytes: self.record_bytes,
            },
            self.stage.as_mut().unwrap(),
            self.budget.clone(),
            self.control.clone(),
            if work.validation {
                TreeSourceKind::FinalValidation
            } else {
                TreeSourceKind::Train
            },
            self.projection
                .root
                .require("encoder")?
                .require("aggregate_window")?
                .integer(1, 65536)? as usize,
        )?;
        require(
            scan.proof.root.require("manifest_sha256")?
                == work.item.root.require("manifest_sha256")?,
            "TreeDataset original manifest changed from frozen worklist",
        )?;
        require(
            scan.manifest
                .root
                .require("ppo")?
                .require("collection_generation")?
                .integer(0, 1000)?
                == work.generation as i128,
            "TreeDataset original generation context differs",
        )?;
        if !work.validation {
            let record = &parent.summary.root.require("generations")?.array()?[work.generation + 1];
            let update = parent.sources.checked_json(
                &parent
                    .directory
                    .join("updates")
                    .join(format!("{:06}.json", work.generation + 1)),
                record.require("update_sha256")?,
            )?;
            require(
                update.root.require("source_proof")? == &scan.proof.root,
                "TreeDataset original source proof differs from immutable update",
            )?;
        }
        let generation = &parent.summary.root.require("generations")?.array()?[work.generation];
        let inference = parent.sources.checked_json(
            &parent
                .directory
                .join("artifacts")
                .join(format!("{:06}", work.generation))
                .join("inference.json"),
            generation.require("inference_sha256")?,
        )?;
        let declaration = structured_source::semantic_declaration(&inference.root, &self.control)?;
        require(
            declaration == *parent.summary.root.require("semantics")?,
            "TreeDataset original numerical semantics drift within cohort",
        )?;
        let correspondence = structured_source::cohort_binding(
            self.fingerprint()?,
            &declaration,
            &scan.digests,
            &self.budget,
            &self.control,
        )?;
        let b = &self.budget;
        let c = &self.control;
        let _copy = json::admit(
            json::canonical(&scan.episodes.root, b, c)?.len()
                + json::canonical(&self.projection.root, b, c)?.len()
                + json::canonical(&work.item.root, b, c)?.len()
                + 65536,
            b,
        )?;
        let mut roles = Vec::new();
        let mut index = 0;
        let mut zero = 0;
        for group in scan.manifest.root.require("schedule")?.array()? {
            for spec in group.require("episodes")?.array()? {
                c.check()?;
                let episode = &scan.episodes.root.array()?[index];
                index += 1;
                let count = episode.require("frames")?.array()?.len();
                zero += usize::from(count == 0);
                require(
                    episode.require("actor")? == spec.require("learner_viewer")?
                        && text(episode, "end_kind")? == "natural",
                    "TreeDataset selected actor/closure mismatch",
                )?;
                roles.push(json::object([
                    ("episode_id", json::copy(spec.require("episode_id")?, c)?),
                    ("viewer", json::copy(episode.require("actor")?, c)?),
                    ("accepted_count", json::integer(count as i128)),
                    ("end_kind", json::text("natural")),
                    ("outcome", json::copy(episode.require("outcome")?, c)?),
                ]));
            }
        }
        let counts = json::object([
            ("games", json::integer(index as i128)),
            ("participants", json::integer(2 * index as i128)),
            (
                "selected_decisions",
                json::integer(scan.accepted_learner as i128),
            ),
            ("zero_endpoint_games", json::integer(zero as i128)),
        ]);
        let summary = doc(
            &json::object([
                ("schema_version", json::integer(2)),
                ("kind", json::text("structured-tree-dataset-collection")),
                ("control_fingerprint", json::text(self.fingerprint()?)),
                ("work_index", json::integer(self.position as i128)),
                ("parent_index", json::integer(work.parent as i128)),
                (
                    "split",
                    json::copy(scan.manifest.root.require("split")?, c)?,
                ),
                (
                    "collection_kind",
                    json::copy(work.item.root.require("collection_kind")?, c)?,
                ),
                ("generation", json::integer(work.generation as i128)),
                (
                    "manifest_sha256",
                    json::copy(scan.proof.root.require("manifest_sha256")?, c)?,
                ),
                (
                    "source_set_sha256",
                    json::copy(scan.proof.root.require("source_set_sha256")?, c)?,
                ),
                ("schedule_sha256", json::text(&scan.schedule_sha256)),
                ("bindings_sha256", json::text(&scan.bindings_sha256)),
                ("original_correspondence", correspondence),
                (
                    "numeric",
                    json::copy(work.item.root.require("numeric")?, c)?,
                ),
                ("selected_roles", json::array(roles)),
                ("counts", counts),
                ("tree_projection", json::copy(&self.projection.root, c)?),
                (
                    "elapsed_seconds",
                    json::copy(scan.manifest.root.require("elapsed_seconds")?, c)?,
                ),
            ]),
            b,
            c,
        )?;
        let mut auth = json::object([]);
        for key in [
            "schema_version",
            "control_fingerprint",
            "work_index",
            "parent_index",
            "manifest_sha256",
            "source_set_sha256",
            "schedule_sha256",
            "bindings_sha256",
        ] {
            auth.as_object_mut()
                .unwrap()
                .insert(key.into(), json::copy(summary.root.require(key)?, c)?);
        }
        auth.as_object_mut().unwrap().insert(
            "kind".into(),
            json::text("structured-tree-dataset-collection-authorization"),
        );
        for (key, value) in [
            (
                "selected_roles_sha256",
                json::text(&digest(summary.root.require("selected_roles")?, b, c)?),
            ),
            (
                "correspondence_sha256",
                json::copy(
                    summary
                        .root
                        .require("original_correspondence")?
                        .require("binding_sha256")?,
                    c,
                )?,
            ),
            (
                "numeric_sha256",
                if work.validation {
                    Value::Null
                } else {
                    json::copy(work.item.root.require("numeric")?.require("sha256")?, c)?
                },
            ),
            (
                "tree_projection_sha256",
                json::text(&digest(&self.projection.root, b, c)?),
            ),
        ] {
            auth.as_object_mut().unwrap().insert(key.into(), value);
        }
        let authorization = doc(&auth, b, c)?;
        self.current = Some(Current {
            scan,
            summary,
            authorization,
        });
        self.phase = Phase::Staged;
        self.current_checks()?;
        Ok(true)
    }
    pub(crate) fn collection_summary(&mut self) -> Result<BudgetVec<u8>> {
        require(
            self.phase == Phase::Staged,
            "one staged TreeDataset collection summary required",
        )?;
        self.current_checks()?;
        json::canonical(
            &self.current.as_ref().ok_or(Error::Closed)?.summary.root,
            &self.budget,
            &self.control,
        )
    }
    pub(crate) fn authorize_collection(&mut self, bytes: &[u8]) -> Result<()> {
        require(
            self.phase == Phase::Staged,
            "one staged TreeDataset collection authorization required",
        )?;
        self.current_checks()?;
        let value = json::parse(bytes, &self.budget, &self.control)?;
        require(
            json::canonical(&value.root, &self.budget, &self.control)?[..]
                == json::canonical(
                    &self
                        .current
                        .as_ref()
                        .ok_or(Error::Closed)?
                        .authorization
                        .root,
                    &self.budget,
                    &self.control,
                )?[..],
            "TreeDataset closed collection authorization mismatch",
        )?;
        self.phase = Phase::Authorized;
        Ok(())
    }
    pub(crate) fn export_current(&mut self) -> Result<(BudgetVec<u8>, Banks)> {
        require(
            self.phase == Phase::Authorized,
            "one authorized TreeDataset selected-input export required",
        )?;
        self.current_checks()?;
        let current = self.current.as_ref().ok_or(Error::Closed)?;
        let _copy = json::admit(
            json::canonical(&current.scan.episodes.root, &self.budget, &self.control)?.len()
                + 65536,
            &self.budget,
        )?;
        let stage = self.stage.as_mut().ok_or(Error::Closed)?;
        let payload = json::object([
            ("schema_version", json::integer(1)),
            ("kind", json::text("structured-tree-selected-inputs")),
            (
                "control_fingerprint",
                json::copy(
                    self.summary.root.require("control_fingerprint")?,
                    &self.control,
                )?,
            ),
            ("work_index", json::integer(self.position as i128)),
            (
                "tree_projection",
                json::copy(&self.projection.root, &self.control)?,
            ),
            (
                "episodes",
                json::copy(&current.scan.episodes.root, &self.control)?,
            ),
            ("arrays", stage.descriptors()?),
        ]);
        let metadata = json::canonical(&payload, &self.budget, &self.control)?;
        require(
            metadata.len() <= 64 << 20,
            "TreeDataset selected-input metadata exceeds64MiB",
        )?;
        self.phase = Phase::Failed;
        let banks = stage.export_retiring_disk()?;
        require(
            banks.floats.is_empty(),
            "TreeDataset selected inputs cannot export policy-value FP32 banks",
        )?;
        self.phase = Phase::Exported;
        Ok((metadata, banks))
    }
    pub(crate) fn retire_current(&mut self) -> Result<()> {
        require(
            self.phase == Phase::Exported,
            "TreeDataset exported inputs must retire once",
        )?;
        self.current_checks()?;
        self.retired.reserve(
            self.current
                .as_ref()
                .ok_or(Error::Closed)?
                .scan
                .sources
                .len(),
        )?;
        self.evidence.reserve(1)?;
        let current = self.current.as_ref().ok_or(Error::Closed)?;
        let _copy = json::admit(
            json::canonical(&current.summary.root, &self.budget, &self.control)?.len() + 65536,
            &self.budget,
        )?;
        let mut receipt = json::object([]);
        for (key, value) in current.summary.root.object()? {
            if key != "tree_projection" {
                receipt
                    .as_object_mut()
                    .unwrap()
                    .insert(key.clone(), json::copy(value, &self.control)?);
            }
        }
        receipt.as_object_mut().unwrap().insert(
            "tree_projection_sha256".into(),
            json::text(&digest(&self.projection.root, &self.budget, &self.control)?),
        );
        let receipt = doc(&receipt, &self.budget, &self.control)?;
        self.stage.as_mut().ok_or(Error::Closed)?.close()?;
        self.stage.take();
        let mut current = self.current.take().ok_or(Error::Closed)?;
        self.evidence.push(receipt)?;
        while let Some(owner) = current.scan.sources.pop() {
            self.retired.push(owner)?;
        }
        self.position += 1;
        self.phase = Phase::Ready;
        Ok(())
    }
    pub(crate) fn recheck(&mut self) -> Result<()> {
        self.current_checks()
    }
    pub(crate) fn verify_complete(&mut self) -> Result<BudgetVec<u8>> {
        self.control.check()?;
        require(
            self.phase == Phase::Exhausted
                && self.position == self.work.len()
                && self.current.is_none()
                && self.stage.is_none(),
            "TreeDataset exact worklist exhaustion/retirement required before finalization",
        )?;
        for parent in self.parents.iter() {
            parent.sources.recheck()?;
        }
        for work in self.work.iter() {
            if let Some(owner) = &work.validation_control {
                owner.recheck()?;
            }
        }
        for owner in self.retired.iter() {
            owner.recheck()?;
        }
        for parent in self.parents.iter() {
            parent.sources.final_snapshots()?;
        }
        for work in self.work.iter() {
            if let Some(owner) = &work.validation_control {
                owner.final_snapshots()?;
            }
        }
        for owner in self.retired.iter() {
            owner.final_snapshots()?;
        }
        let mut copied = 65536usize;
        for item in self.evidence.iter() {
            copied = copied
                .checked_add(json::canonical(&item.root, &self.budget, &self.control)?.len())
                .ok_or_else(|| Error::resource("TreeDataset final evidence metadata overflow"))?;
        }
        let _copy = json::admit(copied, &self.budget)?;
        let result = json::object([
            ("schema_version", json::integer(2)),
            ("kind", json::text("structured-tree-dataset-complete")),
            ("control_fingerprint", json::text(self.fingerprint()?)),
            ("qualified", Value::Bool(false)),
            ("parent_qualification", json::text("not_performed")),
            (
                "collections",
                json::array(
                    self.evidence
                        .iter()
                        .map(|item| json::copy(&item.root, &self.control))
                        .collect::<Result<Vec<_>>>()?,
                ),
            ),
        ]);
        let bytes = json::canonical(&result, &self.budget, &self.control)?;
        Ok(bytes)
    }
    pub(crate) fn finalize(&mut self) -> Result<BudgetVec<u8>> {
        let value = self.verify_complete()?;
        self.phase = Phase::Finalized;
        Ok(value)
    }
    pub(crate) fn control_authorization_limit(&self) -> Result<usize> {
        Ok(
            json::canonical(&self.authorization.root, &self.budget, &self.control)?
                .len()
                .max(65536),
        )
    }
    pub(crate) fn collection_authorization_limit(&self) -> Result<usize> {
        Ok(json::canonical(
            &self
                .current
                .as_ref()
                .ok_or(Error::Closed)?
                .authorization
                .root,
            &self.budget,
            &self.control,
        )?
        .len()
        .max(65536))
    }
    pub(crate) fn fail_export(&mut self) {
        self.phase = Phase::Failed;
    }
    pub(crate) fn close(&mut self) -> Result<()> {
        if self.phase == Phase::Closed {
            return Ok(());
        }
        if let Some(stage) = self.stage.as_mut() {
            stage.close()?;
        }
        self.stage.take();
        self.current.take();
        self.retired.clear();
        self.evidence.clear();
        self.work.clear();
        self.parents.clear();
        self.phase = Phase::Closed;
        Ok(())
    }
    pub(crate) fn is_closed(&self) -> bool {
        self.phase == Phase::Closed
    }
}
