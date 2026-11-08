//! Corrected metadata composition on the shared Source owner. No new Source
//! issuer, policy, game execution, learner or qualification is provided here.
use super::*;
use crate::structured_controls as controls;
use crate::structured_journal::card_instance as dto;

pub(super) fn manifest(
    s: &mut Sources,
    m: &Value,
    parent: Option<(&Path, &Value)>,
    limits: &CollectionLimits,
    b: &Arc<Budget>,
    c: &Control,
    role: CorrectedCollectionRole,
) -> Result<()> {
    require(
        m.require("schema_version")?.integer(5, 5)? == 5
            && field(m, "status")? == "complete"
            && field(m, "collection_role")?
                == if role.final64() {
                    "ppo-final-validation"
                } else {
                    "ppo-batch"
                },
        "complete corrected PPO Collection5 required",
    )?;
    controls::game_binding3(m, b, c)?;
    let cfg = m.require("config")?;
    let actions = cfg.require("actions")?;
    controls::actions3(actions, b, c)?;
    for (key, expected) in [
        ("input_recipe", "participant-structured-v2"),
        ("model_version", "3"),
        ("candidate_codec", "candidate-columns-v2"),
    ] {
        require(
            field(m, key)? == expected,
            "corrected collection context mismatch",
        )?;
    }
    for (key, number) in [
        ("journal_schema_version", 6),
        ("derived_schema_version", 6),
        ("participant_protocol_version", 3),
    ] {
        m.require(key)?.integer(number, number)?;
    }
    require(
        field(m, "token_signature")? == controls::CARD_INSTANCE_SIGNATURE
            && field(m, "config_hash")? == digest(cfg, b, c)?
            && field(m, "action_config_hash")? == digest(actions, b, c)?
            && field(cfg.require("run")?, "split")?
                == if role.final64() {
                    "validation"
                } else {
                    "train"
                },
        "corrected collection Config/input mismatch",
    )?;
    let (parent_root, state) = parent
        .ok_or_else(|| Error::contract("corrected PPO Source requires its original parent"))?;
    require(
        (role.final64() || cfg == state.require("effective_config")?)
            && m.require("inputs")? == state.require("config_inputs")?
            && m.require("code")? == state.require("identity")?.require("code")?,
        "corrected collection differs from frozen parent",
    )?;
    let link = m.require("linkage")?;
    if role.final64() {
        crate::structured_session::final_validation::manifest_controls(
            m,
            parent_root,
            state,
            b,
            c,
        )?;
    } else {
        link.closed(
            &[
                "parent_run_id",
                "parent_run_path",
                "batch_id",
                "collection_generation",
                "learner_seats",
                "schedule_digest",
                "policy_registry_digest",
            ],
            &[],
        )?;
        let generation = integer(link, "collection_generation", 0, 999)?;
        require(
            link.require("parent_run_id")? == state.require("run_id")?
                && field(link, "parent_run_path")?
                    == parent_root
                        .to_str()
                        .ok_or_else(|| Error::contract("nonUTF8 parent root"))?
                && field(link, "batch_id")? == format!("batch-{generation:06}")
                && field(link, "schedule_digest")? == digest(m.require("schedule")?, b, c)?
                && field(link, "policy_registry_digest")?
                    == digest(m.require("policy_bindings")?, b, c)?,
            "corrected collection linkage mismatch",
        )?;
    }
    let schedule = m.require("schedule")?.array()?;
    let groups = m.require("groups")?.array()?;
    let receipts = m.require("runtime_evidence")?.array()?;
    require(
        schedule.len() == role.games() / 2
            && groups.len() == role.games() / 2
            && receipts.len() == role.games() / 2,
        "complete eight swapped original pairs required",
    )?;
    for (group, receipt) in groups.iter().zip(receipts) {
        dto::runtime_receipt(receipt, b, c)?;
        require(
            group
                .get("runtime_evidence_failure")
                .is_none_or(|v| !json::truthy(v))
                && group.require("runtime_evidence")? == receipt
                && receipt.require("binary")?.require("sha256")?
                    == state
                        .require("identity")?
                        .require("integration")?
                        .require("binary_sha256")?,
            "original runtime/group/parent mismatch",
        )?;
    }
    let registry = m.require("policy_bindings")?.object()?;
    let neural = m.require("neural_inputs")?;
    neural.closed(&["input_contract", "policies"], &[])?;
    require(
        field(neural, "input_contract")? == "retained-external-inference-v1",
        "original retained neural contract required",
    )?;
    let named = neural.require("policies")?.object()?;
    let mut expected = BTreeSet::new();
    for (label, binding) in registry {
        c.check()?;
        binding.closed(&["model_id", "policy_options"], &[])?;
        let options = binding.require("policy_options")?;
        if options.is_null() {
            require(
                ["random", "heuristic"].contains(&field(binding, "model_id")?)
                    && *label == format!("{}-card-instance-v2", field(binding, "model_id")?),
                "original baseline registry mismatch",
            )?;
        } else {
            let raw = named
                .get(label)
                .ok_or_else(|| Error::contract("original neural declaration missing"))?;
            require(
                controls::neural_metadata3(raw, actions, b, c)? == *label
                    && raw.require("policy_options")? == options
                    && options.require("model")?.require("model_id")?
                        == binding.require("model_id")?,
                "original neural registry mismatch",
            )?;
            expected.insert(label.as_str());
        }
    }
    require(
        expected.len() <= 6 && expected == named.keys().map(String::as_str).collect(),
        "extra/missing original retained neural declarations",
    )?;
    let policies = m
        .require("policies")?
        .array()?
        .iter()
        .map(ValueExt::string)
        .collect::<Result<BTreeSet<_>>>()?;
    require(
        policies == registry.keys().map(String::as_str).collect(),
        "original policy labels differ from registry",
    )?;
    let mut expected_files = BTreeSet::new();
    for entry in m.require("files")?.array()? {
        c.check()?;
        entry.closed(&["path", "sha256", "bytes"], &[])?;
        let input = owned_relative(&s.root, entry.require("path")?)?;
        let normalized = input.components().collect::<PathBuf>();
        let name = normalized
            .strip_prefix(&s.root)
            .map_err(|_| Error::contract("foreign Source inventory"))?
            .to_str()
            .ok_or_else(|| Error::contract("nonUTF8 Source inventory"))?
            .to_owned();
        require(
            name != "manifest.json"
                && name != "manifest.json.tmp"
                && name != "report.json"
                && expected_files.insert(name),
            "duplicate/foreign original inventory",
        )?;
        let size = std::fs::symlink_metadata(&normalized)?.len();
        if size > limits.source_bytes as u64 {
            return Err(Error::resource("original collection file cap"));
        }
        require(
            entry.require("bytes")?.integer(0, u64::MAX as i128)? == size as i128
                && s.hash(&normalized)? == controls::checksum(entry.require("sha256")?)?,
            "original inventory hash/size differs",
        )?;
    }
    s.pin_tree(&s.root.clone())?;
    let files = s.proof_files(&s.root)?;
    let actual = files
        .root
        .object()?
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    expected_files.extend([role.manifest().into(), "report.json".into()]);
    require(
        actual == expected_files,
        "complete original Collection5 inventory differs",
    )
}
pub(super) fn role<'a>(
    header: &Value,
    m: &'a Value,
    ep: &Value,
    spec: &Value,
    viewer: i128,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<(Value, Option<&'a Value>)> {
    dto::header6(header, b, c)?;
    require(
        header.require("run_id")? == m.require("run_id")?
            && header.require("episode_id")? == spec.require("episode_id")?
            && header.require("game_id")? == ep.require("game_id")?
            && header.require("viewer")?.integer(1, 2)? == viewer
            && header.require("policy")?
                == &spec.require("policies")?.array()?[(viewer - 1) as usize]
            && header.require("sampling_seed")?
                == &spec.require("sampling_seeds")?.array()?[(viewer - 1) as usize]
            && header.require("actions")? == m.require("config")?.require("actions")?,
        "corrected original participant/stream mismatch",
    )?;
    let label = field(header, "policy")?;
    let binding = m.require("policy_bindings")?.require(label)?;
    require(
        header.require("model_id")? == binding.require("model_id")?,
        "corrected original model role mismatch",
    )?;
    let group = m
        .require("schedule")?
        .array()?
        .iter()
        .position(|g| {
            g.get("episodes")
                .and_then(Value::as_array)
                .is_some_and(|rows| {
                    rows.iter()
                        .any(|row| row.get("episode_id") == spec.get("episode_id"))
                })
        })
        .ok_or_else(|| Error::contract("original role schedule missing"))?;
    require(
        header.require("runtime_evidence")? == &m.require("runtime_evidence")?.array()?[group],
        "original participant runtime declaration differs",
    )?;
    let model = if binding.require("policy_options")?.is_null() {
        None
    } else {
        let raw = m
            .require("neural_inputs")?
            .require("policies")?
            .require(label)?;
        require(
            header.require("neural")? == raw
                && raw.require("policy_options")? == binding.require("policy_options")?,
            "original neural role differs from registry",
        )?;
        Some(raw.require("policy_options")?.require("model")?)
    };
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
        "rule_signature",
        "layout_signature",
    ] {
        identity
            .as_object_mut()
            .unwrap()
            .insert(key.into(), json::copy(header.require(key)?, c)?);
    }
    Ok((identity, model))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "bounded genuine read-only acceptance parent; run explicitly"]
    fn actual_card_instance_parent3_and_collection5_originals() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("runs/card-instance-v2-91d5926/ppo-acceptance/mlp-cpu-resume-1");
        let b = Budget::new(2 * 1024 * 1024 * 1024);
        let c = Control::new(crate::budget::monotonic() + 180.).unwrap();
        let mut owner = Sources::new(&root, b.clone(), c.clone(), 64 * 1024 * 1024).unwrap();
        let parent = controls::parent3(&mut owner).unwrap();
        assert_eq!(parent.root["state"]["updates"], json::integer(2));
        let state = owner.json(&root.join("run.json")).unwrap();
        let collection = inspect_card_instance_collection(
            &root.join("collections/batch-000000"),
            (&root, &state.root),
            &CollectionLimits {
                source_bytes: 64 * 1024 * 1024,
                record_bytes: 8 * 1024 * 1024,
            },
            None,
            b.clone(),
            c.clone(),
        )
        .unwrap();
        assert_eq!(collection.digests.len(), 16);
        assert_eq!(collection.proof.root["qualified"], Value::Bool(false));
        for source in collection.sources.iter() {
            source.recheck().unwrap();
        }
        owner.recheck().unwrap();
    }
}
