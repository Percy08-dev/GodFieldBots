//! Original structured Source composition. No numeric/model runtime or eligibility.
use crate::{
    budget::{Budget, Control},
    error::Result,
    json::{self, Value, ValueExt},
    structured_controls::digest,
};
use std::sync::Arc;

pub(crate) fn episode_digest(
    classification: &Value,
    actor: &Value,
    episode: &Value,
    boundary: &Value,
    frames: &[Value],
    end_kind: &Value,
    outcome: &Value,
    reason: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<String> {
    control.check()?;
    let mut raw = 65536usize;
    for value in frames.iter().chain([
        classification,
        actor,
        episode,
        boundary,
        end_kind,
        outcome,
        reason,
    ]) {
        control.check()?;
        raw = raw
            .checked_add(json::canonical(value, budget, control)?.len())
            .ok_or_else(|| {
                crate::error::Error::resource("episode correspondence metadata overflow")
            })?;
    }
    let _copy = json::admit(raw, budget)?;
    let mut value = json::object([
        (
            "recipe",
            json::text("structured-source-episode-correspondence-v1"),
        ),
        ("actor", json::copy(actor, control)?),
        ("episode", json::copy(episode, control)?),
        ("boundary", json::copy(boundary, control)?),
        (
            "frames",
            json::array(
                frames
                    .iter()
                    .map(|v| json::copy(v, control))
                    .collect::<Result<Vec<_>>>()?,
            ),
        ),
        ("end_kind", json::copy(end_kind, control)?),
        ("outcome", json::copy(outcome, control)?),
        ("reason", json::copy(reason, control)?),
    ]);
    for key in ["action_contract", "purpose", "synthetic_capacity"] {
        value.as_object_mut().unwrap().insert(
            key.into(),
            json::copy(classification.require(key)?, control)?,
        );
    }
    digest(&value, budget, control)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        structured_journal::{Journal, OriginalHeader},
        structured_staging::Stage,
    };
    use sha2::{Digest, Sha256};
    use std::{fs, path::PathBuf};
    fn temp() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "gf-raw-cohort-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        root
    }
    #[test]
    fn original_sixteen_role_journals_stage_exact_python_c1_named_arrays_metadata_and_digests() {
        // A frozen Python Source reference component, not whole-parent/Source
        // authorization. No NumPy, Python callback, weights or model run here.
        let budget = Budget::new(512 * 1024 * 1024);
        let control = Control::new(crate::budget::monotonic() + 60.0).unwrap();
        let fixture = json::parse(
            include_bytes!("../tests/fixtures/structured_cohort16_inputs.json"),
            &budget,
            &control,
        )
        .unwrap();
        let root = temp();
        let mut stage =
            Stage::new(budget.clone(), 8 * 1024 * 1024, 65536, control.clone()).unwrap();
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
            fs::create_dir(&directory).unwrap();
            let journal_file = episode.require("journal_file").unwrap().string().unwrap();
            let raw = episode.require("journal").unwrap().string().unwrap();
            fs::write(directory.join(journal_file), raw).unwrap();
            for (name, hex) in episode.require("columns").unwrap().object().unwrap() {
                let hex = hex.string().unwrap();
                let bytes = (0..hex.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                    .collect::<Vec<_>>();
                fs::write(directory.join(name), bytes).unwrap();
            }
            let header =
                json::parse(raw.lines().next().unwrap().as_bytes(), &budget, &control).unwrap();
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
                    .insert(key.into(), header.root.require(key).unwrap().clone());
            }
            let declaration = OriginalHeader::new(
                Some(fixture.root.require("model").unwrap()),
                &identity,
                &budget,
                &control,
            )
            .unwrap();
            let journal = Journal::original(
                &directory,
                journal_file,
                budget.clone(),
                control.clone(),
                64 * 1024 * 1024,
                8 * 1024 * 1024,
                &declaration,
                Some((&mut stage, index)),
            )
            .unwrap();
            let expected = episode.require("expected_metadata").unwrap();
            let metadata = journal
                .staged_episode(&mut stage, index, expected.require("reason").unwrap())
                .unwrap();
            assert_eq!(metadata, *expected, "episode metadata {index}");
            let frames = journal.correspondence_frames().unwrap();
            let h = journal.header_value();
            let end = journal.end_value();
            let digest = episode_digest(
                h,
                h.require("viewer").unwrap(),
                h.require("bank_episode").unwrap(),
                &journal.boundary_value(),
                frames.root.array().unwrap(),
                end.require("end_kind").unwrap(),
                end.require("outcome").unwrap(),
                expected.require("reason").unwrap(),
                &budget,
                &control,
            )
            .unwrap();
            assert_eq!(
                digest,
                episode
                    .require("expected_digest")
                    .unwrap()
                    .string()
                    .unwrap()
            );
            originals.push(journal.into_sources());
        }
        let descriptors = stage.descriptors().unwrap();
        let banks = stage.export().unwrap();
        let expected = fixture
            .root
            .require("expected_arrays")
            .unwrap()
            .object()
            .unwrap();
        assert_eq!(descriptors.array().unwrap().len(), expected.len());
        for descriptor in descriptors.array().unwrap() {
            let key = descriptor.require("key").unwrap().string().unwrap();
            let gold = expected.get(key).expect("exact C1 array name");
            assert_eq!(
                descriptor.require("dtype").unwrap(),
                gold.require("dtype").unwrap()
            );
            assert_eq!(
                descriptor.require("shape").unwrap(),
                gold.require("shape").unwrap()
            );
            let bank = descriptor
                .require("bank_index")
                .unwrap()
                .integer(0, 1000000)
                .unwrap() as usize;
            let offset = descriptor
                .require("element_offset")
                .unwrap()
                .integer(0, 1 << 30)
                .unwrap() as usize;
            let count = descriptor
                .require("element_count")
                .unwrap()
                .integer(0, 1 << 30)
                .unwrap() as usize;
            let mut hash = Sha256::new();
            match descriptor.require("dtype").unwrap().string().unwrap() {
                "uint8" => hash.update(&banks.bytes[bank][offset..offset + count]),
                "int64" => {
                    for &v in &banks.integers[bank][offset..offset + count] {
                        hash.update(v.to_le_bytes());
                    }
                }
                "float32" => {
                    for &v in &banks.floats[bank][offset..offset + count] {
                        hash.update(v.to_le_bytes());
                    }
                }
                _ => panic!("unknown dtype"),
            }
            assert_eq!(
                format!("{:x}", hash.finalize()),
                gold.require("sha256").unwrap().string().unwrap(),
                "array {key}"
            );
        }
        for source in &originals {
            source.recheck().unwrap();
        }
        drop(banks);
        stage.close().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}

use crate::{
    budget::{BudgetVec, Lease},
    error::Error,
    json::Document,
    structured_journal::{Journal, OriginalHeader},
    structured_session::{SourcePattern, Sources},
    structured_staging::Stage,
    validation,
};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

pub(crate) struct CollectionLimits {
    pub source_bytes: usize,
    pub record_bytes: usize,
}
pub(crate) struct CollectionScan {
    pub directory: PathBuf,
    pub manifest: Document,
    pub sources: BudgetVec<Sources>,
    pub proof: Document,
    pub schedule_sha256: String,
    pub bindings_sha256: String,
    pub episodes: Document,
    pub digests: Vec<String>,
    pub learner_label: Option<String>,
    pub model: Option<Document>,
    pub accepted_learner: usize,
    _metadata: BudgetVec<Lease>,
}
// Preserve S1's explicit nullable management-field semantics, not descriptor defaults.
fn nullable<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or(&Value::Null)
}
fn field<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value.require(key)?.string()
}
fn integer(value: &Value, key: &str, lo: i128, hi: i128) -> Result<usize> {
    Ok(value.require(key)?.integer(lo, hi)? as usize)
}
fn require(value: bool, message: &str) -> Result<()> {
    if value {
        Ok(())
    } else {
        Err(Error::contract(message))
    }
}
pub(crate) fn doc(value: &Value, budget: &Arc<Budget>, control: &Control) -> Result<Document> {
    let bytes = json::canonical(value, budget, control)?;
    json::parse(&bytes, budget, control)
}
pub(crate) fn input_classification(value: &Value, contract: &Value) -> Result<()> {
    crate::structured_controls::contract(contract)?;
    require(
        value.require("action_contract")? == contract
            && field(value, "purpose")? == "component"
            && !value.require("synthetic_capacity")?.boolean()?,
        "original Source input classification mismatch",
    )
}
fn safe_name(value: &Value) -> Result<&str> {
    let value = value.string()?;
    require(
        !value.is_empty()
            && value.len() <= 200
            && value.as_bytes()[0].is_ascii_alphanumeric()
            && value
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
            && ![".", ".."].contains(&value),
        "unsafe original episode basename",
    )?;
    Ok(value)
}
fn healthy_group(value: &Value) -> Result<bool> {
    for key in [
        "failure",
        "client_close_failure",
        "budget_overrun_seconds",
        "finalization_cancelled",
        "cleanup_failure",
    ] {
        if value.get(key).is_some_and(json::truthy) {
            return Ok(false);
        }
    }
    if let Some(process) = value.get("process").filter(|v| !v.is_null()) {
        if process.get("exit_code").and_then(Value::as_i64) != Some(0)
            || process.get("shutdown").and_then(Value::as_str) != Some("stdin_eof")
        {
            return Ok(false);
        }
        for key in ["log_error", "reap_failure", "budget_overrun_seconds"] {
            if process.get(key).is_some_and(json::truthy) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
fn role_identity<'a>(
    header: &Value,
    manifest: &'a Value,
    episode: &Value,
    spec: &Value,
    viewer: i128,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<(Value, Option<&'a Value>)> {
    let config = manifest.require("effective_config")?;
    let actions = config.require("actions")?;
    require(
        header.require("run_id")? == manifest.require("run_id")?
            && header.require("episode_id")? == spec.require("episode_id")?
            && header.require("viewer")?.integer(1, 2)? == viewer
            && header.require("game_id")? == episode.require("game_id")?
            && header.require("policy")?
                == &spec.require("policies")?.array()?[(viewer - 1) as usize],
        "original journal/manifest participant identity mismatch",
    )?;
    let policy = field(header, "policy")?;
    let binding = manifest.require("policy_bindings")?.require(policy)?;
    binding.closed(&["model_id", "policy_options"], &[])?;
    require(
        binding.require("model_id")? == header.require("model_id")?,
        "original policy binding model differs",
    )?;
    let mut caps = json::object([
        ("max_interval_events", json::integer(4096)),
        ("max_episode_events", json::integer(262144)),
        ("max_decisions", Value::Null),
        (
            "max_positions",
            json::copy(actions.require("max_sequence_length")?, control)?,
        ),
        (
            "max_candidates",
            json::copy(actions.require("max_candidates")?, control)?,
        ),
    ]);
    let model = if field(header, "model_version")? == "1" {
        require(
            binding.require("policy_options")?.is_null()
                && policy == format!("{}-structured-v1", field(header, "model_id")?),
            "original baseline execution identity differs",
        )?;
        None
    } else if field(header, "model_id")? == "lightgbm" {
        let options = binding.require("policy_options")?;
        input_classification(options, actions.require("action_contract")?)?;
        crate::structured_controls::tree_execution(options)?;
        let model = options.require("model")?;
        let encoder = model.require("encoder")?;
        require(
            field(options, "feature_signature")?
                == crate::structured_controls::tree_feature_signature(model, budget, control)?,
            "original Tree2 exact feature/category signature mismatch",
        )?;
        let execution = json::object([
            (
                "inference",
                json::copy(options.require("inference")?, control)?,
            ),
            (
                "resources",
                json::copy(options.require("resources")?, control)?,
            ),
            (
                "tree_descriptor",
                json::copy(options.require("tree_descriptor")?, control)?,
            ),
        ]);
        let label = format!(
            "lightgbm-tree-v2-{}-{}",
            field(options, "expected_manifest_digest")?,
            digest(&execution, budget, control)?
        );
        require(
            policy == label
                && header.require("aggregate_window")? == encoder.require("aggregate_window")?
                && integer(encoder, "max_positions", 1, 128)?
                    >= integer(actions, "max_sequence_length", 1, 128)?
                && integer(options.require("inference")?, "max_candidates", 1, 65536)?
                    >= integer(actions, "max_candidates", 1, 65536)?,
            "original Tree2 CPU execution/window/capacity mismatch",
        )?;
        Some(model)
    } else {
        let options = binding.require("policy_options")?;
        options.closed(
            &[
                "version",
                "checkpoint",
                "expected_manifest_digest",
                "model",
                "inference",
                "cuda_visible_devices",
                "action_contract",
                "purpose",
                "synthetic_capacity",
            ],
            &["allow_toy"],
        )?;
        input_classification(options, actions.require("action_contract")?)?;
        require(
            field(options, "version")? == "2" && !field(options, "checkpoint")?.is_empty(),
            "original neural options required",
        )?;
        if let Some(toy) = options.get("allow_toy") {
            toy.boolean()?;
        }
        let model = options.require("model")?;
        crate::structured_controls::model(model, budget, control)?;
        crate::structured_controls::binding(options.require("inference")?)?;
        crate::structured_controls::visibility(options.require("cuda_visible_devices")?)?;
        let digest_value =
            crate::structured_controls::checksum(options.require("expected_manifest_digest")?)?;
        let execution = json::object([
            (
                "inference",
                json::copy(options.require("inference")?, control)?,
            ),
            (
                "resources",
                json::copy(model.require("resources")?, control)?,
            ),
            (
                "cuda_visible_devices",
                json::copy(options.require("cuda_visible_devices")?, control)?,
            ),
        ]);
        let label = format!(
            "{}-ppo-v2-{}-{}",
            field(model, "model_id")?,
            digest_value,
            digest(&execution, budget, control)?
        );
        require(
            policy == label
                && header.require("model_id")? == model.require("model_id")?
                && header.require("aggregate_window")?
                    == model.require("event")?.require("aggregate_window")?,
            "original neural execution/window identity differs",
        )?;
        require(
            integer(model, "max_positions", 1, 128)?
                >= integer(actions, "max_sequence_length", 1, 128)?
                && integer(options.require("inference")?, "max_candidates", 1, 65536)?
                    >= integer(actions, "max_candidates", 1, 65536)?,
            "original model capacity does not cover actions",
        )?;
        for (key, field) in [
            ("max_interval_events", "max_interval"),
            ("max_episode_events", "max_episode"),
        ] {
            caps.as_object_mut().unwrap().insert(
                key.into(),
                json::copy(model.require("event")?.require(field)?, control)?,
            );
        }
        caps.as_object_mut().unwrap().insert(
            "max_decisions".into(),
            json::copy(
                model.require("temporal")?.require("max_decisions")?,
                control,
            )?,
        );
        Some(model)
    };
    require(
        header.require("caps")? == &caps,
        "original journal caps differ from policy/actions",
    )?;
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
        "aggregate_window",
        "token_signature",
    ] {
        identity
            .as_object_mut()
            .unwrap()
            .insert(key.into(), json::copy(header.require(key)?, control)?);
    }
    identity
        .as_object_mut()
        .unwrap()
        .insert("caps".into(), caps);
    identity.as_object_mut().unwrap().insert(
        "action_config_hash".into(),
        json::text(&digest(actions, budget, control)?),
    );
    let seed = &spec.require("sampling_seeds")?.array()?[(viewer - 1) as usize];
    seed.integer(0, u64::MAX as i128)?;
    identity
        .as_object_mut()
        .unwrap()
        .insert("sampling_seed".into(), json::copy(seed, control)?);
    for key in ["action_contract", "purpose", "synthetic_capacity"] {
        identity.as_object_mut().unwrap().insert(
            key.into(),
            if key == "action_contract" {
                json::copy(actions.require(key)?, control)?
            } else if key == "purpose" {
                json::text("component")
            } else {
                Value::Bool(false)
            },
        );
    }
    Ok((identity, model))
}

pub(crate) fn inspect_collection(
    directory: &Path,
    parent: Option<(&Path, &Value)>,
    limits: &CollectionLimits,
    stage: Option<&mut Stage>,
    budget: Arc<Budget>,
    control: Control,
) -> Result<CollectionScan> {
    inspect_collection_inner(directory, parent, limits, stage, budget, control, None)
}
#[derive(Clone, Copy)]
pub(crate) enum TreeSourceKind {
    Train,
    FinalValidation,
}
pub(crate) fn inspect_tree_collection(
    directory: &Path,
    parent: (&Path, &Value),
    limits: &CollectionLimits,
    stage: &mut Stage,
    budget: Arc<Budget>,
    control: Control,
    kind: TreeSourceKind,
    window: usize,
) -> Result<CollectionScan> {
    inspect_collection_inner(
        directory,
        Some(parent),
        limits,
        Some(stage),
        budget,
        control,
        Some((kind, window)),
    )
}
fn inspect_collection_inner(
    directory: &Path,
    parent: Option<(&Path, &Value)>,
    limits: &CollectionLimits,
    mut stage: Option<&mut Stage>,
    budget: Arc<Budget>,
    control: Control,
    tree: Option<(TreeSourceKind, usize)>,
) -> Result<CollectionScan> {
    control.check()?;
    let mut sources = Sources::new(
        directory,
        budget.clone(),
        control.clone(),
        limits.source_bytes,
    )?;
    let root = sources.root.clone();
    let manifest = sources.json(&root.join("manifest.json"))?;
    let m = &manifest.root;
    let schema = m.require("schema_version")?.integer(3, 4)?;
    require(
        ["complete", "incomplete", "interrupted"].contains(&field(m, "status")?),
        "finalized original collection required",
    )?;
    if let Some((kind, window)) = tree {
        require(
            (1..=65536).contains(&window),
            "TreeDataset projection window invalid",
        )?;
        require(
            schema == 4 && field(m, "status")? == "complete",
            "TreeDataset requires complete original Collection4",
        )?;
        let (expected_kind, expected_split) = match kind {
            TreeSourceKind::Train => ("ppo_batch", "train"),
            TreeSourceKind::FinalValidation => ("ppo_validation", "validation"),
        };
        require(
            field(m, "collection_kind")? == expected_kind && field(m, "split")? == expected_split,
            "TreeDataset collection kind/split mismatch",
        )?;
    }
    let cfg = m.require("effective_config")?;
    crate::structured_controls::actions(cfg)?;
    let actions = cfg.require("actions")?;
    input_classification(m, actions.require("action_contract")?)?;
    for (key, expected) in [
        ("feature_version", "participant-structured-v1"),
        ("input_recipe", "participant-structured-v1"),
        ("reward_policy_version", "natural-terminal-v1"),
        ("end_projection_version", "simulator-closure-v1"),
        ("policy_input_version", "player-atomic-options-v1"),
        (
            "simulator_contract_revision",
            "ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6",
        ),
        ("observation_schema_version", "1"),
    ] {
        require(
            field(m, key)? == expected,
            "original collection semantic identity differs",
        )?;
    }
    m.require("journal_schema_version")?.integer(5, 5)?;
    m.require("action_schema_version")?.integer(1, 1)?;
    crate::journal::catalog_meta(m.require("catalog")?)?;
    require(
        digest(
            m.require("reviewed_simulator_spec_sha256")?,
            &budget,
            &control,
        )? == "17dc86ba14c8fac378e352acd5c7c55429e910b37482954f74ca3673f8b1767d",
        "original collection reviewed specification differs",
    )?;
    require(
        digest(cfg, &budget, &control)?
            == crate::structured_controls::checksum(m.require("config_sha256")?)?,
        "original collection Config hash differs",
    )?;
    let split = field(m, "split")?;
    require(
        ["train", "validation", "test"].contains(&split)
            && cfg.require("run")?.require("split")? == m.require("split")?,
        "original collection split differs",
    )?;
    let elapsed = m.require("elapsed_seconds")?.number()?;
    let wall = cfg.require("run")?.require("wall_seconds")?.number()?;
    require(
        elapsed.is_finite() && wall.is_finite() && elapsed >= 0.0 && elapsed <= wall,
        "original collection clock provenance invalid",
    )?;
    if let Some((parent_directory, state)) = parent {
        let projected;
        let expected_config = if matches!(tree, Some((TreeSourceKind::FinalValidation, _))) {
            projected = {
                let mut value = json::copy(state.require("effective_config")?, &control)?;
                value
                    .get_mut("run")
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .insert("split".into(), json::text("validation"));
                value
            };
            &projected
        } else {
            state.require("effective_config")?
        };
        require(
            m.require("config_sha256")?.string()? == digest(expected_config, &budget, &control)?
                && cfg == expected_config
                && m.require("config_inputs")? == state.require("config_inputs")?,
            "original collection full frozen Config/input origins differ",
        )?;
        let integration = state.require("identity")?.require("integration")?;
        for key in ["binary_sha256", "server_config_sha256"] {
            require(
                m.require(key)? == integration.require(key)?,
                "recorded original binary/server provenance differs",
            )?;
        }
        require(
            m.require("simulator_contract_revision")?
                == integration.require("declared_simulator_revision")?
                && m.require("code")? == state.require("identity")?.require("code")?
                && m.require("simulator")?.require("revision")?
                    == integration.require("source_revision")?
                && m.require("simulator")?.require("spec_sha256")?
                    == integration.require("source_spec_sha256")?,
            "recorded code/Simulator provenance differs",
        )?;
        if schema == 4 {
            let link = m.require("ppo")?;
            require(
                link.require("parent_run_id")? == state.require("run_id")?
                    && field(link, "parent_run_path")?
                        == parent_directory
                            .to_str()
                            .ok_or_else(|| Error::contract("nonUTF8 parent directory"))?,
                "original parent linkage differs",
            )?;
        }
    } else {
        require(
            schema == 3,
            "PPO collection requires its original parent context",
        )?;
    }
    let mut _metadata = BudgetVec::new(&budget);
    _metadata.push(budget.reserve(65536)?)?;
    let mut specs = Vec::new();
    let mut names = BTreeSet::new();
    for group in m.require("schedule")?.array()? {
        for spec in group.require("episodes")?.array()? {
            control.check()?;
            let name = safe_name(spec.require("episode_id")?)?;
            _metadata.push(budget.reserve(1024 + 4 * name.len())?)?;
            require(names.insert(name), "duplicate original scheduled episode")?;
            let policies = spec.require("policies")?.array()?;
            require(
                policies.len() == 2,
                "exact two original policy roles required",
            )?;
            for policy in policies {
                validation::text_value(policy)?;
            }
            let seeds = spec.require("sampling_seeds")?.array()?;
            require(
                seeds.len() == 2,
                "exact two scheduled participant seeds required",
            )?;
            for seed in seeds {
                seed.integer(0, u64::MAX as i128)?;
            }
            specs.push((group, spec));
        }
    }
    let episodes = m.require("episodes")?.array()?;
    require(
        !specs.is_empty() && episodes.len() == specs.len(),
        "entire original cohort required",
    )?;
    let mut gameids = BTreeSet::new();
    let mut episode_names = BTreeSet::new();
    for ep in episodes {
        _metadata.push(budget.reserve(1024)?)?;
        require(
            episode_names.insert(safe_name(ep.require("episode_id")?)?)
                && gameids.insert(field(ep, "game_id")?)
                && !field(ep, "game_id")?.is_empty(),
            "duplicate original game/participant identity",
        )?;
    }
    require(
        names == episode_names,
        "original result/schedule population differs",
    )?;
    if schema == 4 {
        require(
            ["ppo_batch", "ppo_validation"].contains(&field(m, "collection_kind")?),
            "unknown PPO collection kind",
        )?;
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
        link.require("collection_generation")?.integer(0, 1000)?;
        for key in ["parent_run_id", "parent_run_path", "batch_id"] {
            link.require(key)?.string()?;
        }
        if field(m, "collection_kind")? == "ppo_batch" {
            require(
                specs.len() == 16 && m.require("schedule")?.array()?.len() == 8,
                "PPO batch requires all eight original swapped pairs",
            )?;
        }
        if matches!(tree, Some((TreeSourceKind::FinalValidation, _))) {
            require(
                specs.len() == 64 && m.require("schedule")?.array()?.len() == 32,
                "TreeDataset final validation requires all32 swapped pairs/64 games",
            )?;
        }
        let mut seats = json::object([]);
        for (_, spec) in &specs {
            spec.require("learner_viewer")?.integer(1, 2)?;
            seats.as_object_mut().unwrap().insert(
                field(spec, "episode_id")?.into(),
                json::copy(spec.require("learner_viewer")?, &control)?,
            );
        }
        require(
            link.require("learner_seats")? == &seats,
            "original explicit learner roles differ",
        )?;
    } else {
        require(
            stage.is_none(),
            "baseline/nonPPO collection cannot export a neural PPO cohort",
        )?;
    }
    let report = sources.checked_json(&root.join("report.json"), m.require("report_sha256")?)?;
    let r = &report.root;
    r.require("schema_version")?.integer(1, 1)?;
    let planned = integer(r, "planned_games", 1, u64::MAX as i128)?;
    let natural = episodes
        .iter()
        .filter(|ep| ep.get("end_kind").and_then(Value::as_str) == Some("natural"))
        .count();
    require(
        planned == specs.len()
            && integer(r, "natural_denominator", 0, u64::MAX as i128)? == natural,
        "original report cohort/natural ledger differs",
    )?;
    for key in ["failures", "health_failures", "group_health_failures"] {
        r.require(key)?.object()?;
    }
    let report_healthy = r.get("comparison_valid") == Some(&Value::Bool(true))
        && ["failures", "health_failures", "group_health_failures"]
            .iter()
            .all(|&key| {
                r.get(key)
                    .is_some_and(|v| v.is_object() && !json::truthy(v))
            });
    if field(m, "status")? == "complete" || r.get("comparison_valid") == Some(&Value::Bool(true)) {
        require(
            natural == specs.len(),
            "complete/successful original cohort is not all natural",
        )?;
    }
    if field(m, "status")? == "complete" || natural == specs.len() {
        require(
            report_healthy,
            "complete/all-natural original report unhealthy",
        )?;
    }
    let expected_journals = names
        .iter()
        .flat_map(|name| {
            [
                format!("{name}/player-1.jsonl"),
                format!("{name}/player-2.jsonl"),
            ]
        })
        .collect();
    sources.source_inventory(SourcePattern::Journals, expected_journals)?;
    sources.source_inventory(SourcePattern::Partials, Vec::new())?;
    sources.source_inventory(
        SourcePattern::Results,
        names
            .iter()
            .map(|name| format!("{name}/result.json"))
            .collect(),
    )?;
    // Read matched original names incrementally before parsing any journal table.
    let mut all_columns = BTreeSet::new();
    let mut replays = BTreeSet::new();
    for name in &names {
        for item in std::fs::read_dir(root.join(name))? {
            control.check()?;
            let item = item?;
            let file = item.file_name();
            let file = file
                .to_str()
                .ok_or_else(|| Error::contract("nonUTF8 Source child"))?;
            if file.ends_with(".columns") {
                _metadata.push(budget.reserve(1024 + 4 * (name.len() + file.len() + 1))?)?;
                all_columns.insert(format!("{name}/{file}"));
            }
            if file.starts_with("replay-player-") && file.ends_with(".json") {
                _metadata.push(budget.reserve(1024 + 4 * (name.len() + file.len() + 1))?)?;
                require(
                    ["replay-player-1.json", "replay-player-2.json"].contains(&file),
                    "extra participant Replay",
                )?;
                replays.insert(format!("{name}/{file}"));
            }
        }
    }
    sources.source_inventory(
        SourcePattern::Columns,
        all_columns.iter().cloned().collect(),
    )?;
    sources.source_inventory(SourcePattern::Replays, replays.iter().cloned().collect())?;
    let mut original_sources = BudgetVec::new(&budget);
    let mut referenced = BTreeSet::new();
    let mut staged = Vec::new();
    let mut digests = Vec::new();
    let mut learner_label = None;
    let mut learner_model = None;
    let mut accepted_learner = 0;
    let mut accepted_total = 0;
    if tree.is_some() {
        require(
            episodes
                .iter()
                .all(|ep| ep.get("end_kind").and_then(Value::as_str) == Some("natural")),
            "TreeDataset rejects the whole interrupted cohort",
        )?;
    }
    let mut closure_counts = json::object([]);
    let mut missing_replays = Vec::new();
    for (group, spec) in specs {
        control.check()?;
        let name = field(spec, "episode_id")?;
        let ep = episodes
            .iter()
            .find(|e| e.get("episode_id").and_then(Value::as_str) == Some(name))
            .unwrap();
        require(
            ep.require("policies")? == spec.require("policies")?
                && ep.require("seed")? == group.require("seed")?
                && ep.get("pair_id") == spec.get("pair_id"),
            "original schedule/result identity differs",
        )?;
        let kind = field(ep, "end_kind")?;
        require(
            ["natural", "truncated", "aborted", "cancelled", "incomplete"].contains(&kind),
            "unknown original collection closure",
        )?;
        let result = sources.json(&root.join(name).join("result.json"))?;
        for (key, value) in ep.object()? {
            require(
                nullable(&result.root, key) == value,
                "original saved result differs from manifest",
            )?;
        }
        let groups = m
            .require("groups")?
            .array()?
            .iter()
            .filter(|g| {
                g.get("episode_ids")
                    .and_then(Value::as_array)
                    .is_some_and(|a| a.iter().any(|v| v.as_str() == Some(name)))
            })
            .collect::<Vec<_>>();
        require(
            groups.len() == 1,
            "original group health missing/duplicated",
        )?;
        if kind == "natural" {
            require(
                !ep.get("failure").is_some_and(json::truthy)
                    && !ep.require("health")?.object()?.values().any(json::truthy)
                    && healthy_group(groups[0])?,
                "natural original episode has failure evidence",
            )?;
        }
        let mut closures = Vec::new();
        for viewer in 1..=2 {
            control.check()?;
            let dir = root.join(name);
            let file = format!("player-{viewer}.jsonl");
            let raw = sources.bytes(&dir.join(&file), limits.source_bytes)?;
            let header_bytes = raw
                .split_inclusive(|&b| b == b'\n')
                .next()
                .ok_or_else(|| Error::contract("original journal header absent"))?;
            if header_bytes.len() > limits.record_bytes {
                return Err(Error::resource("original header record limit"));
            }
            let h = json::parse(header_bytes, &budget, &control)?;
            drop(raw);
            let (identity, model) = role_identity(&h.root, m, ep, spec, viewer, &budget, &control)?;
            let declaration = OriginalHeader::new(model, &identity, &budget, &control)?;
            let selected = schema == 4 && spec.require("learner_viewer")?.integer(1, 2)? == viewer;
            let selected_stage = if selected {
                stage.as_deref_mut().map(|s| (s, digests.len()))
            } else {
                None
            };
            let input = if let Some((_, window)) = tree {
                if selected {
                    Journal::original_tree(
                        &dir,
                        &file,
                        budget.clone(),
                        control.clone(),
                        limits.source_bytes,
                        limits.record_bytes,
                        &declaration,
                        selected_stage,
                        window,
                    )?
                } else {
                    Journal::original(
                        &dir,
                        &file,
                        budget.clone(),
                        control.clone(),
                        limits.source_bytes,
                        limits.record_bytes,
                        &declaration,
                        None,
                    )?
                }
            } else {
                Journal::original(
                    &dir,
                    &file,
                    budget.clone(),
                    control.clone(),
                    limits.source_bytes,
                    limits.record_bytes,
                    &declaration,
                    selected_stage,
                )?
            };
            let end = input.end_value();
            require(
                end.require("end_kind")? == ep.require("end_kind")?
                    && end.require("outcome")? == nullable(ep, "outcome")
                    && end.require("simulator_abort_reason")?
                        == nullable(ep, "simulator_abort_reason"),
                "original journal/episode closure differs",
            )?;
            let evidence = input.sources.proof_files(&root)?;
            for key in evidence.root.object()?.keys() {
                if key.ends_with(".columns") {
                    _metadata.push(budget.reserve(1024 + 4 * key.len())?)?;
                    require(
                        referenced.insert(key.clone()),
                        "sidecar referenced more than once",
                    )?;
                }
            }
            let replay_name = format!("{name}/replay-player-{viewer}.json");
            if replays.contains(&replay_name) {
                let replay = sources.json(&root.join(&replay_name))?;
                validation::reconcile_replay_controlled(
                    &replay.root,
                    viewer,
                    &input.accepted_actions()?,
                    nullable(ep, "outcome"),
                    &control,
                )?;
                let closure =
                    validation::replay_closure(&replay.root, field(ep, "game_id")?, viewer)?;
                if let Some(c) = closure {
                    if field(c, "type")? == "terminal" {
                        require(
                            !nullable(ep, "outcome").is_null(),
                            "Replay cannot invent unknown natural outcome",
                        )?;
                    }
                }
                closures.push(closure.map(|c| json::copy(c, &control)).transpose()?);
            } else {
                require(
                    kind != "natural"
                        && ep
                            .require("health")?
                            .get("replay_or_finalization_failure")
                            .is_some_and(json::truthy),
                    "missing original Replay without explicit health failure",
                )?;
                missing_replays.push(json::text(&replay_name));
            }
            let reason = if kind == "natural" {
                Value::Null
            } else {
                json::copy(
                    ep.get("simulator_abort_reason")
                        .filter(|v| json::truthy(v))
                        .or_else(|| ep.get("failure").filter(|v| json::truthy(v)))
                        .unwrap_or(ep.require("end_kind")?),
                    &control,
                )?
            };
            if !reason.is_null() {
                validation::text_value(&reason)?;
            }
            accepted_total += input.accepted_count();
            if selected {
                require(
                    field(input.header_value(), "model_version")? == "2"
                        && ["mlp", "gru", "transformer"]
                            .contains(&field(input.header_value(), "model_id")?),
                    "learner role must be a neural model2",
                )?;
                let label = field(input.header_value(), "policy")?;
                if let Some(old) = &learner_label {
                    require(old == label, "cohort mixes frozen learner labels")?;
                } else {
                    learner_label = Some(label.to_owned());
                    learner_model = Some(doc(model.unwrap(), &budget, &control)?);
                }
                let frames = input.correspondence_frames()?;
                let h = input.header_value();
                digests.push(episode_digest(
                    h,
                    h.require("viewer")?,
                    h.require("bank_episode")?,
                    &input.boundary_value(),
                    frames.root.array()?,
                    end.require("end_kind")?,
                    end.require("outcome")?,
                    &reason,
                    &budget,
                    &control,
                )?);
                accepted_learner += input.accepted_count();
                if let Some(stage) = stage.as_deref_mut() {
                    staged.push(if tree.is_some() {
                        input.staged_tree_episode(stage, digests.len() - 1, &reason)?
                    } else {
                        input.staged_episode(stage, digests.len() - 1, &reason)?
                    });
                }
            }
            original_sources.push(input.into_sources())?;
        }
        if closures.len() == 2 {
            require(
                closures[0] == closures[1],
                "participant Replay closure conflict",
            )?;
        }
        let surviving = closures.first().and_then(Option::as_ref);
        if let Some(closure) = surviving {
            if field(closure, "type")? == "aborted" {
                require(
                    !nullable(ep, "simulator_abort_reason").is_null(),
                    "abort Replay lacks original cause",
                )?;
            }
        }
        if !nullable(ep, "simulator_abort_reason").is_null() && !closures.is_empty() {
            require(
                surviving.and_then(|c| c.get("reason"))
                    == Some(nullable(ep, "simulator_abort_reason"))
                    && kind
                        == if nullable(ep, "simulator_abort_reason").as_str()
                            == Some("turn_limit_exceeded")
                        {
                            "truncated"
                        } else {
                            "aborted"
                        },
                "original abort cause/Replay projection differs",
            )?;
        }
        let count = closure_counts
            .get(kind)
            .and_then(Value::as_i64)
            .unwrap_or(0)
            + 1;
        closure_counts
            .as_object_mut()
            .unwrap()
            .insert(kind.into(), json::integer(count as i128));
    }
    require(
        referenced == all_columns,
        "missing/extra original candidate sidecars",
    )?;
    if stage.is_some() {
        require(
            if let Some((kind, _)) = tree {
                digests.len()
                    == match kind {
                        TreeSourceKind::Train => 16,
                        TreeSourceKind::FinalValidation => 64,
                    }
            } else {
                digests.len() == 16
                    && field(m, "collection_kind")? == "ppo_batch"
                    && split == "train"
            },
            "selected cohort must preserve its full original16/64 neural-role population",
        )?;
    }
    let mut file_owner = sources.proof_files(&root)?;
    let files = &mut file_owner.root;
    for owner in original_sources.iter() {
        let evidence = owner.proof_files(&root)?;
        for (key, value) in evidence.root.object()? {
            if let Some(old) = files.get(key) {
                require(old == value, "original evidence changed across owners")?;
            } else {
                _metadata.push(budget.reserve(1024 + 4 * key.len())?)?;
                files
                    .as_object_mut()
                    .unwrap()
                    .insert(key.clone(), json::copy(value, &control)?);
            }
        }
    }
    let manifest_sha = field(files.require("manifest.json")?, "sha256")?;
    let mut proof = json::object([
        ("schema_version", json::integer(1)),
        ("kind", json::text("structured-component-source-proof")),
        ("qualified", Value::Bool(false)),
        ("parent_qualification", json::text("not_performed")),
        ("manifest_sha256", json::text(manifest_sha)),
        (
            "source_set_sha256",
            json::text(&digest(&files, &budget, &control)?),
        ),
        ("games", json::integer(episodes.len() as i128)),
        ("accepted_decisions", json::integer(accepted_total as i128)),
        ("closure_counts", closure_counts),
        ("missing_replays", json::array(missing_replays)),
        ("files", json::copy(files, &control)?),
    ]);
    for key in ["action_contract", "purpose", "synthetic_capacity"] {
        proof
            .as_object_mut()
            .unwrap()
            .insert(key.into(), json::copy(m.require(key)?, &control)?);
    }
    let proof = doc(&proof, &budget, &control)?;
    let metadata = doc(&json::array(staged), &budget, &control)?;
    let schedule_sha256 = digest(m.require("schedule")?, &budget, &control)?;
    let bindings_sha256 = digest(m.require("policy_bindings")?, &budget, &control)?;
    original_sources.push(sources)?;
    for owner in original_sources.iter() {
        owner.recheck()?;
    }
    for owner in original_sources.iter() {
        owner.final_snapshots()?;
    }
    Ok(CollectionScan {
        directory: root,
        manifest,
        sources: original_sources,
        proof,
        schedule_sha256,
        bindings_sha256,
        episodes: metadata,
        digests,
        learner_label,
        model: learner_model,
        accepted_learner,
        _metadata,
    })
}

#[cfg(test)]
mod collection_tests {
    use super::*;
    use sha2::Digest;
    fn fixture(label: &str) -> (PathBuf, Arc<Budget>, Control) {
        let budget = Budget::new(512 * 1024 * 1024);
        let control = Control::new(crate::budget::monotonic() + 60.0).unwrap();
        let input = json::parse(
            include_bytes!("../tests/fixtures/structured_collection_inputs.json"),
            &budget,
            &control,
        )
        .unwrap();
        let selected = input
            .root
            .array()
            .unwrap()
            .iter()
            .find(|v| v.get("label").and_then(Value::as_str) == Some(label))
            .unwrap();
        let directory = std::env::temp_dir().join(format!(
            "gf-collection-scan-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        for (name, hex) in selected.require("files").unwrap().object().unwrap() {
            let path = directory.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let hex = hex.string().unwrap();
            let bytes = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect::<Vec<_>>();
            std::fs::write(path, bytes).unwrap();
        }
        (directory, budget, control)
    }
    fn write(path: &Path, value: &Value, budget: &Arc<Budget>, control: &Control) {
        std::fs::write(path, &*json::canonical(value, budget, control).unwrap()).unwrap();
    }
    #[test]
    fn historical_no_staging_batch_cardinality_rejects_before_participant_lookup() {
        let (root, budget, control) = fixture("natural");
        let path = root.join("manifest.json");
        let mut manifest = json::parse(&std::fs::read(&path).unwrap(), &budget, &control).unwrap();
        manifest
            .root
            .as_object_mut()
            .unwrap()
            .insert("config_inputs".into(), json::array([]));
        manifest
            .root
            .as_object_mut()
            .unwrap()
            .insert("code".into(), json::object([]));
        manifest
            .root
            .as_object_mut()
            .unwrap()
            .insert("binary_sha256".into(), json::text(&"a".repeat(64)));
        manifest
            .root
            .as_object_mut()
            .unwrap()
            .insert("server_config_sha256".into(), json::text(&"b".repeat(64)));
        manifest.root.as_object_mut().unwrap().insert(
            "simulator".into(),
            json::object([("revision", Value::Null), ("spec_sha256", Value::Null)]),
        );
        let mut seats = json::object([]);
        for group in manifest.root.require("schedule").unwrap().array().unwrap() {
            for spec in group.require("episodes").unwrap().array().unwrap() {
                seats.as_object_mut().unwrap().insert(
                    field(spec, "episode_id").unwrap().into(),
                    spec.require("learner_viewer").unwrap().clone(),
                );
            }
        }
        let parent_path = root.parent().unwrap().join("recorded-parent");
        manifest
            .root
            .as_object_mut()
            .unwrap()
            .insert("schema_version".into(), json::integer(4));
        manifest
            .root
            .as_object_mut()
            .unwrap()
            .insert("collection_kind".into(), json::text("ppo_batch"));
        manifest.root.as_object_mut().unwrap().insert(
            "ppo".into(),
            json::object([
                ("parent_run_id", json::text("recorded-parent")),
                ("parent_run_path", json::text(parent_path.to_str().unwrap())),
                ("batch_id", json::text("batch-000000")),
                ("collection_generation", json::integer(0)),
                ("learner_seats", seats),
            ]),
        );
        write(&path, &manifest.root, &budget, &control);
        let m = &manifest.root;
        let state = json::object([
            ("run_id", json::text("recorded-parent")),
            ("config_sha256", m.require("config_sha256").unwrap().clone()),
            (
                "effective_config",
                m.require("effective_config").unwrap().clone(),
            ),
            ("config_inputs", m.require("config_inputs").unwrap().clone()),
            (
                "identity",
                json::object([
                    ("code", m.require("code").unwrap().clone()),
                    (
                        "integration",
                        json::object([
                            ("binary_sha256", m.require("binary_sha256").unwrap().clone()),
                            (
                                "server_config_sha256",
                                m.require("server_config_sha256").unwrap().clone(),
                            ),
                            (
                                "declared_simulator_revision",
                                m.require("simulator_contract_revision").unwrap().clone(),
                            ),
                            (
                                "source_revision",
                                m.require("simulator")
                                    .unwrap()
                                    .require("revision")
                                    .unwrap()
                                    .clone(),
                            ),
                            (
                                "source_spec_sha256",
                                m.require("simulator")
                                    .unwrap()
                                    .require("spec_sha256")
                                    .unwrap()
                                    .clone(),
                            ),
                        ]),
                    ),
                ]),
            ),
        ]);
        let error = inspect_collection(
            &root,
            Some((&parent_path, &state)),
            &CollectionLimits {
                source_bytes: 64 * 1024 * 1024,
                record_bytes: 8 * 1024 * 1024,
            },
            None,
            budget.clone(),
            control,
        )
        .err()
        .unwrap();
        assert!(
            matches!(error,Error::Contract(message) if message.contains("eight original swapped pairs"))
        );
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn baseline_original_population_and_mixed_interruptions_preserve_raw_evidence() {
        for (label, accepted) in [("natural", 4), ("mixed", 2)] {
            let (root, budget, control) = fixture(label);
            let result = inspect_collection(
                &root,
                None,
                &CollectionLimits {
                    source_bytes: 64 * 1024 * 1024,
                    record_bytes: 8 * 1024 * 1024,
                },
                None,
                budget,
                control,
            )
            .unwrap();
            assert_eq!(
                result
                    .proof
                    .root
                    .require("games")
                    .unwrap()
                    .integer(0, 100)
                    .unwrap(),
                2
            );
            assert_eq!(
                result
                    .proof
                    .root
                    .require("accepted_decisions")
                    .unwrap()
                    .integer(0, 100)
                    .unwrap(),
                accepted
            );
            assert_eq!(
                result.proof.root.require("qualified").unwrap(),
                &Value::Bool(false)
            );
            assert!(
                result.episodes.root.array().unwrap().is_empty()
                    && result.digests.is_empty()
                    && result.model.is_none()
            );
            for owner in result.sources.iter() {
                owner.recheck().unwrap();
            }
            drop(result);
            std::fs::remove_dir_all(root).unwrap();
        }
    }
    #[test]
    fn fully_resealed_policy_population_and_incomplete_report_ledger_corruption_rejects() {
        for mutation in [
            "empty_policies",
            "one_policy",
            "empty_label",
            "missing_ledger",
            "array_ledger",
            "bad_name",
        ] {
            let (root, budget, control) = fixture("mixed");
            let path = root.join("manifest.json");
            let mut manifest =
                json::parse(&std::fs::read(&path).unwrap(), &budget, &control).unwrap();
            match mutation {
                "empty_policies" | "one_policy" | "empty_label" => {
                    let policies = if mutation == "empty_policies" {
                        json::array([])
                    } else if mutation == "one_policy" {
                        json::array([json::text("random-structured-v1")])
                    } else {
                        json::array([json::text(""), json::text("heuristic-structured-v1")])
                    };
                    manifest
                        .root
                        .get_mut("schedule")
                        .unwrap()
                        .as_array_mut()
                        .unwrap()[0]
                        .get_mut("episodes")
                        .unwrap()
                        .as_array_mut()
                        .unwrap()[0]
                        .as_object_mut()
                        .unwrap()
                        .insert("policies".into(), policies.clone());
                    manifest
                        .root
                        .get_mut("episodes")
                        .unwrap()
                        .as_array_mut()
                        .unwrap()[0]
                        .as_object_mut()
                        .unwrap()
                        .insert("policies".into(), policies);
                    let ep = &manifest.root.require("episodes").unwrap().array().unwrap()[0];
                    write(
                        &root
                            .join(field(ep, "episode_id").unwrap())
                            .join("result.json"),
                        ep,
                        &budget,
                        &control,
                    );
                }
                "missing_ledger" | "array_ledger" => {
                    let report_path = root.join("report.json");
                    let mut report =
                        json::parse(&std::fs::read(&report_path).unwrap(), &budget, &control)
                            .unwrap();
                    if mutation == "missing_ledger" {
                        report
                            .root
                            .as_object_mut()
                            .unwrap()
                            .remove("health_failures");
                    } else {
                        report
                            .root
                            .as_object_mut()
                            .unwrap()
                            .insert("health_failures".into(), json::array([]));
                    }
                    write(&report_path, &report.root, &budget, &control);
                    manifest.root.as_object_mut().unwrap().insert(
                        "report_sha256".into(),
                        json::text(&format!(
                            "{:x}",
                            sha2::Sha256::digest(std::fs::read(report_path).unwrap())
                        )),
                    );
                }
                "bad_name" => {
                    manifest
                        .root
                        .get_mut("schedule")
                        .unwrap()
                        .as_array_mut()
                        .unwrap()[0]
                        .get_mut("episodes")
                        .unwrap()
                        .as_array_mut()
                        .unwrap()[0]
                        .as_object_mut()
                        .unwrap()
                        .insert("episode_id".into(), json::text(".hidden"));
                }
                _ => unreachable!(),
            }
            write(&path, &manifest.root, &budget, &control);
            assert!(
                inspect_collection(
                    &root,
                    None,
                    &CollectionLimits {
                        source_bytes: 64 * 1024 * 1024,
                        record_bytes: 8 * 1024 * 1024
                    },
                    None,
                    budget.clone(),
                    control.clone()
                )
                .is_err(),
                "mutation {mutation}"
            );
            std::fs::remove_dir_all(root).unwrap();
        }
    }
}

pub(crate) struct ParentScan {
    pub directory: PathBuf,
    pub sources: Sources,
    pub summary: Document,
    pub state: Document,
}
pub(crate) struct CommittedSource {
    pub parent_index: usize,
    pub collection_index: usize,
    pub update: Document,
}
pub(crate) enum ScanMode {
    Live {
        collection: usize,
        parent: usize,
        generation: usize,
    },
    Historical,
}
pub(crate) struct OriginalInputs {
    pub parents: BudgetVec<ParentScan>,
    pub collections: BudgetVec<CollectionScan>,
    pub committed: BudgetVec<CommittedSource>,
    pub mode: ScanMode,
    pub stage: Option<Stage>,
    pub budget: Arc<Budget>,
    pub control: Control,
    _plan: Document,
    _metadata: BudgetVec<Lease>,
}
pub(crate) fn owned_relative(root: &Path, value: &Value) -> Result<PathBuf> {
    let text = value.string()?;
    let path = Path::new(text);
    require(
        !text.is_empty()
            && !path.is_absolute()
            && path
                .components()
                .all(|p| matches!(p, std::path::Component::Normal(_))),
        "nonempty owned relative reference required",
    )?;
    Ok(root.join(path))
}
impl OriginalInputs {
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
            "structured source plan exceeds bounded wire",
        )?;
        let plan = json::parse(encoded, &budget, &control)?;
        let p = &plan.root;
        require(
            !p.require("parents")?.array()?.is_empty()
                || !p.require("collections")?.array()?.is_empty(),
            "structured Source plan cannot be entirely empty",
        )?;
        p.require("schema_version")?.integer(2, 2)?;
        let live = match field(p, "mode")? {
            "live_cohort" => true,
            "historical_parent" => false,
            _ => return Err(Error::contract("unknown structured source mode")),
        };
        p.closed(
            &["schema_version", "mode", "parents", "collections"],
            if live {
                &["cohort_collection_index"]
            } else {
                &[]
            },
        )?;
        if live {
            p.require("cohort_collection_index")?;
        }
        let mut metadata = BudgetVec::new(&budget);
        metadata.push(budget.reserve(65536)?)?;
        let mut parents = BudgetVec::new(&budget);
        let mut parent_paths = BTreeSet::new();
        for value in p.require("parents")?.array()? {
            control.check()?;
            value.closed(&["directory", "require_complete"], &[])?;
            validation::text_value(value.require("directory")?)?;
            let mut sources = Sources::new(
                Path::new(field(value, "directory")?),
                budget.clone(),
                control.clone(),
                source_bytes,
            )?;
            require(
                parent_paths.insert(sources.root.clone()),
                "duplicate original parent directory",
            )?;
            let summary = crate::structured_controls::parent(&mut sources)?;
            let directory = sources.root.clone();
            let state = sources.json(&directory.join("run.json"))?;
            if value.require("require_complete")?.boolean()? {
                require(
                    field(&state.root, "status")? == "complete",
                    "complete healthy original parent required",
                )?;
            }
            metadata.push(budget.reserve(1024 + directory.as_os_str().len() * 4)?)?;
            parents.push(ParentScan {
                directory,
                sources,
                summary,
                state,
            })?;
        }
        let mut requests = Vec::new();
        let mut paths = BTreeSet::new();
        for value in p.require("collections")?.array()? {
            control.check()?;
            value.closed(&["directory", "parent_index"], &[])?;
            validation::text_value(value.require("directory")?)?;
            let path = std::path::absolute(field(value, "directory")?)?;
            require(
                paths.insert(path.clone()),
                "duplicate original collection directory",
            )?;
            let parent = if value.require("parent_index")?.is_null() {
                None
            } else {
                Some(integer(
                    value,
                    "parent_index",
                    0,
                    parents.len() as i128 - 1,
                )?)
            };
            metadata.push(budget.reserve(2048 + path.as_os_str().len() * 4)?)?;
            requests.push((path, parent));
        }
        let selected = if live {
            Some(integer(
                p,
                "cohort_collection_index",
                0,
                requests.len() as i128 - 1,
            )?)
        } else {
            None
        };
        // Every parent brings ALL immutable committed cohorts, even if omitted
        // from the caller's optional explicit selection list.
        let mut committed: BudgetVec<CommittedSource> = BudgetVec::new(&budget);
        for (pi, parent) in parents.iter_mut().enumerate() {
            let count = integer(&parent.state.root, "updates", 0, 1000)?;
            for generation in 1..=count {
                control.check()?;
                let record = parent.sources.json(
                    &parent
                        .directory
                        .join("generations")
                        .join(format!("{generation:06}.json")),
                )?;
                let update = parent.sources.checked_json(
                    &parent
                        .directory
                        .join("updates")
                        .join(format!("{generation:06}.json")),
                    record.root.require("update_sha256")?,
                )?;
                let path = owned_relative(&parent.directory, update.root.require("collection")?)?;
                let index =
                    if let Some(index) = requests.iter().position(|(actual, _)| *actual == path) {
                        require(
                            requests[index].1 == Some(pi),
                            "historical collection parent index differs",
                        )?;
                        index
                    } else {
                        metadata.push(budget.reserve(2048 + path.as_os_str().len() * 4)?)?;
                        paths.insert(path.clone());
                        requests.push((path, Some(pi)));
                        requests.len() - 1
                    };
                require(
                    !committed
                        .iter()
                        .any(|entry| entry.collection_index == index),
                    "duplicate committed collection ownership across generations",
                )?;
                committed.push(CommittedSource {
                    parent_index: pi,
                    collection_index: index,
                    update,
                })?;
            }
        }
        let mode = if let Some(index) = selected {
            let pi = requests[index].1.ok_or_else(|| {
                Error::contract("live neural cohort requires its original parent")
            })?;
            let parent = &parents[pi];
            require(
                field(&parent.state.root, "status")? == "running",
                "live cohort requires running original parent",
            )?;
            let generation = integer(&parent.state.root, "updates", 0, 999)?;
            require(
                requests[index].0
                    == parent
                        .directory
                        .join("collections")
                        .join(format!("batch-{generation:06}")),
                "live output differs from owning original batch",
            )?;
            require(
                !committed
                    .iter()
                    .any(|entry| entry.collection_index == index),
                "live cohort is already committed",
            )?;
            ScanMode::Live {
                collection: index,
                parent: pi,
                generation,
            }
        } else {
            ScanMode::Historical
        };
        let mut stage = if live {
            Some(Stage::new(
                budget.clone(),
                scratch_bytes,
                chunk_bytes,
                control.clone(),
            )?)
        } else {
            None
        };
        let mut collections = BudgetVec::new(&budget);
        for (index, (directory, pi)) in requests.iter().enumerate() {
            control.check()?;
            let mut peek = Sources::new(directory, budget.clone(), control.clone(), source_bytes)?;
            let first = peek.json(&peek.root.join("manifest.json"))?;
            if first.root.get("schema_version").and_then(Value::as_i64) == Some(4) {
                require(
                    selected == Some(index)
                        || committed
                            .iter()
                            .any(|entry| entry.collection_index == index),
                    "uncommitted neural train collection cannot enter historical/baseline authorization",
                )?;
            }
            if first.root.get("collection_kind").and_then(Value::as_str) == Some("ppo_validation") {
                return Err(Error::contract(
                    "external PPO validation parent composition belongs to the separately reviewed C3 source scope",
                ));
            }
            let parent = pi.map(|pi| (&*parents[pi].directory, &parents[pi].state.root));
            let mut scan = inspect_collection(
                directory,
                parent,
                &CollectionLimits {
                    source_bytes,
                    record_bytes,
                },
                if selected == Some(index) {
                    stage.as_mut()
                } else {
                    None
                },
                budget.clone(),
                control.clone(),
            )?;
            require(
                first.root == scan.manifest.root,
                "original collection changed between selection and scan",
            )?;
            scan.sources.push(peek)?;
            if scan
                .manifest
                .root
                .get("collection_kind")
                .and_then(Value::as_str)
                == Some("ppo_validation")
            {
                return Err(Error::contract(
                    "external PPO validation parent composition belongs to the separately reviewed C3 source scope",
                ));
            }
            if let Some(entry) = committed
                .iter()
                .find(|entry| entry.collection_index == index)
            {
                require(
                    scan.manifest
                        .root
                        .require("ppo")?
                        .require("collection_generation")?
                        .integer(0, 999)?
                        == entry.update.root.require("generation")?.integer(1, 1000)? - 1,
                    "committed original source generation differs",
                )?;
                require(
                    scan.proof.root == *entry.update.root.require("source_proof")?,
                    "original Source proof differs from immutable committed evidence",
                )?;
            }
            collections.push(scan)?;
        }
        let result = Self {
            parents,
            collections,
            committed,
            mode,
            stage,
            budget,
            control,
            _plan: plan,
            _metadata: metadata,
        };
        result.recheck()?;
        Ok(result)
    }
    pub(crate) fn recheck(&self) -> Result<()> {
        self.control.check()?;
        for parent in self.parents.iter() {
            parent.sources.recheck()?;
        }
        for collection in self.collections.iter() {
            for owner in collection.sources.iter() {
                owner.recheck()?;
            }
        }
        for parent in self.parents.iter() {
            parent.sources.final_snapshots()?;
        }
        for collection in self.collections.iter() {
            for owner in collection.sources.iter() {
                owner.final_snapshots()?;
            }
        }
        self.control.check()
    }
}

pub(crate) fn semantic_declaration(value: &Value, control: &Control) -> Result<Value> {
    let mut result = json::object([]);
    for key in [
        "input_recipe",
        "generator_version",
        "policy_input_version",
        "end_projection_version",
        "reward_policy_version",
        "reviewed_spec_hashes",
        "simulator_revision",
        "catalog",
        "token_signature",
        "numeric_transform",
        "runtime",
        "numerics",
    ] {
        result
            .as_object_mut()
            .unwrap()
            .insert(key.into(), json::copy(value.require(key)?, control)?);
    }
    Ok(result)
}
pub(crate) fn cohort_binding(
    fingerprint: &str,
    declaration: &Value,
    digests: &[String],
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<Value> {
    let mut value = json::object([
        (
            "recipe",
            json::text("structured-source-cohort-correspondence-v1"),
        ),
        ("control_fingerprint", json::text(fingerprint)),
        (
            "semantics_sha256",
            json::text(&digest(declaration, budget, control)?),
        ),
        ("episode_count", json::integer(digests.len() as i128)),
        (
            "episode_digests",
            json::array(digests.iter().map(|v| json::text(v))),
        ),
    ]);
    let hash = digest(&value, budget, control)?;
    value
        .as_object_mut()
        .unwrap()
        .insert("binding_sha256".into(), json::text(&hash));
    Ok(value)
}
pub(crate) struct Controls {
    pub summary: Document,
    pub authorization: Document,
    pub declarations: Vec<Option<Document>>,
    pub correspondences: Vec<Option<Document>>,
    pub fingerprint: String,
}
impl OriginalInputs {
    pub(crate) fn controls(&mut self) -> Result<Controls> {
        self.recheck()?;
        let b = &self.budget;
        let c = &self.control;
        let mut copy_bytes = 65536usize;
        for parent in self.parents.iter() {
            copy_bytes = copy_bytes
                .checked_add(json::canonical(&parent.summary.root, b, c)?.len())
                .ok_or_else(|| Error::resource("control metadata size overflow"))?;
        }
        for source in self.collections.iter() {
            copy_bytes = copy_bytes
                .checked_add(json::canonical(&source.manifest.root, b, c)?.len())
                .ok_or_else(|| Error::resource("control metadata size overflow"))?;
        }
        for source in self.committed.iter() {
            copy_bytes = copy_bytes
                .checked_add(
                    json::canonical(source.update.root.require("numeric_files")?, b, c)?.len(),
                )
                .ok_or_else(|| Error::resource("control metadata size overflow"))?;
        }
        let _copy = json::admit(copy_bytes, b)?;
        let mode = match self.mode {
            ScanMode::Live { .. } => "live_cohort",
            ScanMode::Historical => "historical_parent",
        };
        let parent_fingerprints = self
            .parents
            .iter()
            .map(|p| json::copy(p.summary.root.require("control_fingerprint")?, c))
            .collect::<Result<Vec<_>>>()?;
        let collection_fingerprints = self
            .collections
            .iter()
            .map(|s| json::copy(s.proof.root.require("source_set_sha256")?, c))
            .collect::<Result<Vec<_>>>()?;
        let fingerprint = digest(
            &json::object([
                ("recipe", json::text("structured-source-controls-v2")),
                ("mode", json::text(mode)),
                ("plan", json::copy(&self._plan.root, c)?),
                ("parents", json::array(parent_fingerprints)),
                ("collections", json::array(collection_fingerprints)),
            ]),
            b,
            c,
        )?;
        let mut parents = Vec::new();
        let mut parent_ack = Vec::new();
        for (index, parent) in self.parents.iter().enumerate() {
            c.check()?;
            let (item, ack) = compact_parent(index, parent, b, c)?;
            parents.push(item);
            parent_ack.push(ack);
        }
        let mut collections = Vec::new();
        let mut collection_ack = Vec::new();
        let mut declarations = Vec::new();
        let mut correspondences = Vec::new();
        for index in 0..self.collections.len() {
            c.check()?;
            let source = &self.collections[index];
            let m = &source.manifest.root;
            let committed = self.committed.iter().find(|u| u.collection_index == index);
            let parent_index = if let Some(entry) = committed {
                Some(entry.parent_index)
            } else {
                match self.mode {
                    ScanMode::Live {
                        collection, parent, ..
                    } if collection == index => Some(parent),
                    _ => None,
                }
            };
            let generation = if let Some(entry) = committed {
                Some(integer(&entry.update.root, "generation", 1, 1000)? - 1)
            } else {
                match self.mode {
                    ScanMode::Live {
                        collection,
                        generation,
                        ..
                    } if collection == index => Some(generation),
                    _ => None,
                }
            };
            let declaration = if let (Some(pi), Some(generation)) = (parent_index, generation) {
                let parent = &mut self.parents[pi];
                let record = &parent.summary.root.require("generations")?.array()?[generation];
                let inference = parent.sources.checked_json(
                    &parent
                        .directory
                        .join("artifacts")
                        .join(format!("{generation:06}"))
                        .join("inference.json"),
                    record.require("inference_sha256")?,
                )?;
                Some(doc(&semantic_declaration(&inference.root, c)?, b, c)?)
            } else {
                None
            };
            let correspondence = declaration
                .as_ref()
                .map(|d| {
                    cohort_binding(&fingerprint, &d.root, &source.digests, b, c)
                        .and_then(|v| doc(&v, b, c))
                })
                .transpose()?;
            let numeric = if let Some(entry) = committed {
                json::object([
                    (
                        "directory",
                        json::text(
                            owned_relative(
                                &self.parents[entry.parent_index].directory,
                                entry.update.root.require("numeric")?,
                            )?
                            .to_str()
                            .ok_or_else(|| Error::contract("nonUTF8 numeric path"))?,
                        ),
                    ),
                    (
                        "sha256",
                        json::copy(entry.update.root.require("numeric_sha256")?, c)?,
                    ),
                    (
                        "files",
                        json::copy(entry.update.root.require("numeric_files")?, c)?,
                    ),
                ])
            } else {
                Value::Null
            };
            let mut item = json::object([
                ("collection_index", json::integer(index as i128)),
                (
                    "directory",
                    json::text(
                        source
                            .directory
                            .to_str()
                            .ok_or_else(|| Error::contract("nonUTF8 collection path"))?,
                    ),
                ),
                (
                    "parent_index",
                    parent_index
                        .map(|v| json::integer(v as i128))
                        .unwrap_or(Value::Null),
                ),
                (
                    "generation",
                    generation
                        .map(|v| json::integer(v as i128))
                        .unwrap_or(Value::Null),
                ),
                (
                    "schema_version",
                    json::copy(m.require("schema_version")?, c)?,
                ),
                ("split", json::copy(m.require("split")?, c)?),
                (
                    "collection_kind",
                    m.get("collection_kind")
                        .map(|v| json::copy(v, c))
                        .transpose()?
                        .unwrap_or(Value::Null),
                ),
                ("run_id", json::copy(m.require("run_id")?, c)?),
                (
                    "manifest_sha256",
                    json::copy(source.proof.root.require("manifest_sha256")?, c)?,
                ),
                ("schedule_sha256", json::text(&source.schedule_sha256)),
                ("bindings_sha256", json::text(&source.bindings_sha256)),
                (
                    "source_set_sha256",
                    json::copy(source.proof.root.require("source_set_sha256")?, c)?,
                ),
                ("games", json::copy(source.proof.root.require("games")?, c)?),
                (
                    "participant_count",
                    json::integer(
                        2 * integer(&source.proof.root, "games", 1, u64::MAX as i128)? as i128,
                    ),
                ),
                (
                    "accepted_decisions",
                    json::copy(source.proof.root.require("accepted_decisions")?, c)?,
                ),
                (
                    "learner_decisions",
                    json::integer(source.accepted_learner as i128),
                ),
                ("journal_schema_version", json::integer(5)),
                ("feature_version", json::text("participant-structured-v1")),
                ("input_recipe", json::text("participant-structured-v1")),
                (
                    "model_version",
                    if parent_index.is_some() {
                        json::text("2")
                    } else {
                        Value::Null
                    },
                ),
                (
                    "generation_schema",
                    if parent_index.is_some() {
                        json::integer(2)
                    } else {
                        Value::Null
                    },
                ),
                (
                    "rollout_schema_version",
                    if parent_index.is_some() {
                        json::integer(3)
                    } else {
                        Value::Null
                    },
                ),
                (
                    "learner_label",
                    source
                        .learner_label
                        .as_ref()
                        .map(|v| json::text(v))
                        .unwrap_or(Value::Null),
                ),
                (
                    "semantics",
                    declaration
                        .as_ref()
                        .map(|d| json::copy(&d.root, c))
                        .transpose()?
                        .unwrap_or(Value::Null),
                ),
                (
                    "correspondence",
                    correspondence
                        .as_ref()
                        .map(|d| json::copy(&d.root, c))
                        .transpose()?
                        .unwrap_or(Value::Null),
                ),
                ("numeric", numeric),
                (
                    "source_proof_sha256",
                    json::text(&digest(&source.proof.root, b, c)?),
                ),
            ]);
            for key in ["action_contract", "purpose", "synthetic_capacity"] {
                item.as_object_mut()
                    .unwrap()
                    .insert(key.into(), json::copy(m.require(key)?, c)?);
            }
            collections.push(item);
            collection_ack.push(json::object([
                ("collection_index", json::integer(index as i128)),
                (
                    "manifest_sha256",
                    json::copy(source.proof.root.require("manifest_sha256")?, c)?,
                ),
                (
                    "expected_schedule_sha256",
                    json::text(&source.schedule_sha256),
                ),
                (
                    "expected_bindings_sha256",
                    json::text(&source.bindings_sha256),
                ),
                (
                    "source_set_sha256",
                    json::copy(source.proof.root.require("source_set_sha256")?, c)?,
                ),
                (
                    "correspondence_sha256",
                    correspondence
                        .as_ref()
                        .map(|d| json::copy(d.root.require("binding_sha256")?, c))
                        .transpose()?
                        .unwrap_or(Value::Null),
                ),
                (
                    "numeric_sha256",
                    committed
                        .map(|u| json::copy(u.update.root.require("numeric_sha256")?, c))
                        .transpose()?
                        .unwrap_or(Value::Null),
                ),
            ]));
            declarations.push(declaration);
            correspondences.push(correspondence);
        }
        let authorization = doc(
            &json::object([
                ("schema_version", json::integer(2)),
                ("mode", json::text(mode)),
                ("control_fingerprint", json::text(&fingerprint)),
                ("parents", json::array(parent_ack)),
                ("collections", json::array(collection_ack)),
            ]),
            b,
            c,
        )?;
        let authorization_bytes = json::canonical(&authorization.root, b, c)?.len();
        let class = if let Some(parent) = self.parents.first() {
            &parent.summary.root
        } else {
            &self.collections[0].manifest.root
        };
        let summary = doc(
            &json::object([
                ("schema_version", json::integer(2)),
                ("kind", json::text("structured-source-controls")),
                ("reader_contract", json::text("structured-v1")),
                ("scope", json::text("engineering-component-originals")),
                ("mode", json::text(mode)),
                ("qualified", Value::Bool(false)),
                ("parent_qualification", json::text("not_performed")),
                (
                    "action_contract",
                    json::copy(class.require("action_contract")?, c)?,
                ),
                ("purpose", json::text("component")),
                ("synthetic_capacity", Value::Bool(false)),
                ("control_fingerprint", json::text(&fingerprint)),
                (
                    "authorization_max_bytes",
                    json::integer(authorization_bytes as i128),
                ),
                ("parents", json::array(parents)),
                ("collections", json::array(collections)),
            ]),
            b,
            c,
        )?;
        self.recheck()?;
        Ok(Controls {
            summary,
            authorization,
            declarations,
            correspondences,
            fingerprint,
        })
    }
}

#[cfg(test)]
mod tree_role_tests {
    use super::*;
    #[test]
    fn original_tree_raw_role_checks_features_cpu_caps_and_null_ancestry_without_checkpoint_io() {
        let budget = Budget::new(64 << 20);
        let control = Control::new(crate::budget::monotonic() + 30.0).unwrap();
        let input = json::parse(
            include_bytes!("../tests/fixtures/structured_tree_role.json"),
            &budget,
            &control,
        )
        .unwrap();
        fn check(value: &Value, budget: &Arc<Budget>, control: &Control) -> Result<()> {
            let (identity, model) = role_identity(
                value.require("header")?,
                value.require("manifest")?,
                value.require("episode")?,
                value.require("spec")?,
                1,
                budget,
                control,
            )?;
            OriginalHeader::new(model, &identity, budget, control)?;
            Ok(())
        }
        check(&input.root, &budget, &control).unwrap();
        let label = input
            .root
            .require("header")
            .unwrap()
            .require("policy")
            .unwrap()
            .string()
            .unwrap()
            .to_owned();
        for case in 0..5 {
            let mut value = json::copy(&input.root, &control).unwrap();
            match case {
                0 => {
                    value["manifest"]["policy_bindings"][&label]["policy_options"]["feature_signature"] =
                        json::text(&"0".repeat(64));
                }
                1 => {
                    value["manifest"]["policy_bindings"][&label]["policy_options"]["inference"]["device"] =
                        json::text("cuda:0");
                }
                2 => {
                    value["manifest"]["policy_bindings"][&label]["policy_options"]["tree_descriptor"]
                        ["model_bytes"] = Value::Bool(true);
                }
                3 => {
                    value["header"]["caps"]["max_decisions"] = json::integer(1);
                }
                _ => {
                    value["header"]["generation_schema"] = json::integer(2);
                }
            }
            assert!(check(&value, &budget, &control).is_err(), "case{case}");
        }
    }
}

pub(crate) fn compact_parent(
    index: usize,
    parent: &ParentScan,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<(Value, Value)> {
    let p = &parent.summary.root;
    let mut item = json::copy(p, c)?;
    item.as_object_mut()
        .unwrap()
        .insert("parent_index".into(), json::integer(index as i128));
    item.as_object_mut().unwrap().insert(
        "run_sha256".into(),
        json::text(
            parent
                .sources
                .verified_sha(&parent.directory.join("run.json"))?,
        ),
    );
    let run = parent
        .state
        .root
        .require("effective_config")?
        .require("run")?;
    item.as_object_mut().unwrap().insert(
        "run_limits".into(),
        json::object([
            (
                "global_cuda_budget_bytes",
                json::copy(run.require("global_cuda_budget_bytes")?, c)?,
            ),
            (
                "max_participant_workers",
                json::copy(run.require("max_participant_workers")?, c)?,
            ),
        ]),
    );
    item.as_object_mut().unwrap().insert(
        "hardware".into(),
        json::copy(parent.state.root.require("hardware")?, c)?,
    );

    let mut sealed = Vec::new();
    let mut controls = Vec::new();
    for generation in p.require("generations")?.array()? {
        c.check()?;
        controls.push(json::copy(generation.require("parent_controls")?, c)?);
        sealed.push(json::object([
            (
                "generation",
                json::copy(generation.require("generation")?, c)?,
            ),
            (
                "inference_sha256",
                json::copy(generation.require("inference_sha256")?, c)?,
            ),
            (
                "resume_sha256",
                json::copy(generation.require("resume_sha256")?, c)?,
            ),
            (
                "artifact_files_sha256",
                json::copy(generation.require("artifact_files_sha256")?, c)?,
            ),
            (
                "retained",
                Value::Bool(field(generation, "artifact_status")? == "retained"),
            ),
        ]));
    }
    let ack = json::object([
        ("parent_index", json::integer(index as i128)),
        (
            "config_sha256",
            json::copy(p.require("state")?.require("config_sha256")?, c)?,
        ),
        (
            "resume_identity",
            json::copy(p.require("state")?.require("resume_identity")?, c)?,
        ),
        ("sealed", json::array(sealed)),
        (
            "controls_sha256",
            json::text(&digest(&json::array(controls), b, c)?),
        ),
    ]);
    Ok((item, ack))
}
