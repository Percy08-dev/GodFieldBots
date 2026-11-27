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
    let resident = parent.is_some_and(|(_, state)| controls::resident::is_resident(state));
    let version = if resident {7} else if role.hpo() {6} else {5};
    require(
        m.require("schema_version")?.integer(
            version,
            version,
        )? == version
            && field(m, "status")? == "complete"
            && field(m, "collection_role")?
                == if role.hpo() {
                    "hpo-ppo-batch"
                } else if role.final64() {
                    "ppo-final-validation"
                } else {
                    "ppo-batch"
                },
        "complete corrected PPO Collection5 required",
    )?;
    if resident {
        let (_, state) = parent.ok_or_else(|| Error::contract("resident collection requires original parent"))?;
        let max_workers = controls::hpo::spec(state, b, c)?
            .require("operating")?.require("max_participant_workers")?.integer(1, i64::MAX as i128)?;
        controls::resident::retirement(m.require("logical_retirement")?, max_workers, b, c)?;
        let retirement = m.require("logical_retirement")?;
        let link = m.require("linkage")?;
        for (left,right) in [("run_id","parent_run_id"),("collection_id","batch_id"),("generation","collection_generation")] {
            require(retirement.require(left)? == link.require(right)?,"Collection7 original retirement binding differs")?;
        }
    } else { require(m.get("logical_retirement").is_none(),"Collection6 forbids resident retirement")?; }
    controls::game_binding3(m, b, c)?;
    let cfg = m.require("config")?;
    let actions = cfg.require("actions")?;
    controls::actions3(actions, b, c)?;
    require(
        m.require("action_contract")? == actions.require("action_contract")?,
        "corrected collection action contract mismatch",
    )?;
    for (key, expected) in [
        ("input_recipe", "participant-structured-v2"),
        ("model_version", if role.hpo() { "4" } else { "3" }),
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
            && m.require("action_contract")? == state.require("action_contract")?
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
            if role.hpo() {
                &["selection_context"]
            } else {
                &[]
            },
        )?;
        let generation = integer(
            link,
            "collection_generation",
            0,
            if role.hpo() { i64::MAX as i128 } else { 999 },
        )?;
        if role.hpo() {
            let selection_context = link.require("selection_context")?;
            controls::hpo::context(selection_context)?;
            require(
                selection_context
                    .require("batch")?
                    .integer(generation as i128, generation as i128)?
                    == generation as i128
                    && selection_context.require("learning_seed")? == state.require("seed")?
                    && selection_context.require("decision_budget")?
                        == state
                            .require("effective_config")?
                            .require("hpo_spec")?
                            .require("decision_budget")?,
                "HPO original selection context differs from parent",
            )?;
        }
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
    if s.is_logical(&s.root) {
        let registry = s.registry.as_ref().ok_or(Error::Closed)?;
        let (_, original, _) = registry.locate(&s.root).ok_or(Error::Closed)?;
        packed_index(m, &original.description.root, b, c)?;
    }
    for (group, receipt) in groups.iter().zip(receipts) {
        dto::runtime_receipt(receipt, b, c)?;
        require(
            group
                .get("runtime_evidence_failure")
                .is_none_or(|v| !json::truthy(v))
                && group.require("runtime_evidence")? == receipt
                && receipt.require("action_contract")? == m.require("action_contract")?
                && receipt.require("binary")?.require("sha256")?
                    == state
                        .require("identity")?
                        .require("integration")?
                        .require("binary_sha256")?,
            "original runtime/group/parent mismatch",
        )?;
    }
    let registry = m.require("policy_bindings")?.object()?;
    retained_registry(m, actions, b, c)?;
    if !role.final64() {
        if let Some(refresh) = controls::refresh_training_registry(cfg, m.require("policy_bindings")?, b, c)? {
            require(schedule.len() == 8, "mixed opponents require exactly eight pairs")?;
            for (pair, group) in schedule.iter().enumerate() {
                let episodes = group.require("episodes")?.array()?;
                require(episodes.len() == 2, "mixed opponents require swapped pairs")?;
                let left = episodes[0].require("policies")?.array()?;
                let right = episodes[1].require("policies")?.array()?;
                require(left.len() == 2 && right.len() == 2 && left[0] == right[1] && left[1] == right[0],
                    "mixed opponents require identical swapped roles")?;
                let opponent = left[1].string()?;
                let expected = match pair { 0 | 1 => Some("random-card-instance-v2"), 2 => Some("heuristic-card-instance-v2"), 3 => Some(refresh.as_str()), _ => None };
                if let Some(expected) = expected {
                    require(opponent == expected, "mixed pair opponent differs from frozen descriptor")?;
                } else {
                    let binding = registry.get(opponent).ok_or_else(|| Error::contract("mixed snapshot registry missing"))?;
                    require(!["random", "heuristic", "heuristic_refresh"].contains(&field(binding, "model_id")?),
                        "mixed snapshot pair must retain a learned policy")?;
                }
            }
        }
    }
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
        let size = s.file_len(&normalized)? as u64;
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
fn packed_index(m: &Value, description: &Value, b: &Arc<Budget>, c: &Control) -> Result<()> {
    let index = description.require("index")?;
    let groups = index.require("groups")?.array()?;
    let episodes = index.require("episodes")?.array()?;
    let schedule = m.require("schedule")?.array()?;
    let originals = m.require("episodes")?.array()?;
    require(
        groups.len() == schedule.len() && episodes.len() == originals.len(),
        "complete original packed group/episode row sets required",
    )?;
    // Metadata was already admitted, parsed and checked for sorted uniqueness.
    // Semantic equality is canonical JSON equality, preserving bool/number/null.
    for (spec, health) in schedule.iter().zip(m.require("groups")?.array()?) {
        c.check()?;
        let id = spec
            .require("episodes")?
            .array()?
            .first()
            .ok_or_else(|| Error::contract("empty original scheduled group"))?
            .require("episode_id")?;
        let group = groups
            .iter()
            .find(|g| g.get("group_id") == Some(id))
            .ok_or_else(|| Error::contract("missing/foreign packed original group"))?;
        require(
            digest(group.require("record")?, b, c)? == digest(health, b, c)?,
            "packed original group health differs",
        )?;
        for episode in spec.require("episodes")?.array()? {
            let name = episode.require("episode_id")?;
            let row = episodes
                .iter()
                .find(|e| e.get("episode_id") == Some(name))
                .ok_or_else(|| Error::contract("missing/foreign packed original episode"))?;
            let original = originals
                .iter()
                .find(|e| e.get("episode_id") == Some(name))
                .ok_or_else(|| {
                    Error::contract("packed original schedule/result episode differs")
                })?;
            require(
                row.require("group_id")? == id
                    && row.require("game_id")? == original.require("game_id")?,
                "packed original episode group/game differs",
            )?;
            let record = json::object([
                ("end_kind", json::copy(original.require("end_kind")?, c)?),
                ("failure", json::copy(original.require("failure")?, c)?),
            ]);
            require(
                digest(row.require("record")?, b, c)? == digest(&record, b, c)?,
                "packed original episode outcome record differs",
            )?;
            let prefix = format!("{}/", name.string()?);
            for member in description.require("files")?.array()? {
                c.check()?;
                if member.require("name")?.string()?.starts_with(&prefix) {
                    require(
                        member.require("identity")?.require("group_id")? == id,
                        "packed original episode member belongs to different group",
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod packed_index_tests {
    use super::*;
    fn fixture() -> (Value, Value) {
        let health = json::object([
            ("number", json::integer(1)),
            ("flag", Value::Bool(true)),
            ("missing", Value::Null),
        ]);
        let manifest = json::object([
            (
                "schedule",
                json::array(["a", "b"].into_iter().map(|id| {
                    json::object([(
                        "episodes",
                        json::array([json::object([("episode_id", json::text(id))])]),
                    )])
                })),
            ),
            ("groups", json::array([health.clone(), health.clone()])),
            (
                "episodes",
                json::array(["a", "b"].into_iter().map(|id| {
                    json::object([
                        ("episode_id", json::text(id)),
                        ("game_id", json::text(id)),
                        ("end_kind", json::text("natural")),
                        ("failure", Value::Null),
                    ])
                })),
            ),
        ]);
        let description = json::object([
            (
                "index",
                json::object([
                    (
                        "groups",
                        json::array(["a", "b"].into_iter().map(|id| {
                            json::object([("group_id", json::text(id)), ("record", health.clone())])
                        })),
                    ),
                    (
                        "episodes",
                        json::array(["a", "b"].into_iter().map(|id| {
                            json::object([
                                ("episode_id", json::text(id)),
                                ("group_id", json::text(id)),
                                ("game_id", json::text(id)),
                                (
                                    "record",
                                    json::object([
                                        ("end_kind", json::text("natural")),
                                        ("failure", Value::Null),
                                    ]),
                                ),
                            ])
                        })),
                    ),
                ]),
            ),
            (
                "files",
                json::array(["a", "b"].into_iter().map(|id| {
                    json::object([
                        ("name", json::text(&format!("{id}/player-1.jsonl"))),
                        ("identity", json::object([("group_id", json::text(id))])),
                    ])
                })),
            ),
        ]);
        (manifest, description)
    }
    #[test]
    fn packed_index_semantics_reject_every_changed_original_record_and_mapping() {
        let b = Budget::new(64 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let (manifest, original) = fixture();
        packed_index(&manifest, &original, &b, &c).unwrap();
        for case in 0..10 {
            let mut value = original.clone();
            match case {
                0 => value["index"]["groups"][0]["record"]["number"] = Value::Bool(true),
                1 => value["index"]["groups"][0]["record"]["number"] = json::float(1.0).unwrap(),
                2 => value["index"]["groups"][0]["record"]["missing"] = json::integer(0),
                3 => value["index"]["episodes"][0]["game_id"] = Value::Null,
                4 => value["index"]["episodes"][0]["group_id"] = json::text("b"),
                5 => value["index"]["episodes"][0]["record"]["end_kind"] = json::text("truncated"),
                6 => value["index"]["episodes"][0]["record"]["failure"] = Value::Bool(false),
                7 => {
                    value["index"]["episodes"].as_array_mut().unwrap().pop();
                }
                8 => value["index"]["groups"][0]["group_id"] = json::text("foreign"),
                _ => value["files"][0]["identity"]["group_id"] = json::text("b"),
            }
            assert!(
                packed_index(&manifest, &value, &b, &c).is_err(),
                "case{case}"
            );
        }
        assert_eq!(b.used(), 0);
    }
}
// Typed registry validation never selects a PPO learner.
fn retained_registry(m: &Value, actions: &Value, b: &Arc<Budget>, c: &Control) -> Result<()> {
    // The registry kernel is also used by metadata-only old Tree3 diagnostics.
    // Only an explicit Collection6 selects the new HPO registry contract; full
    // Source admission has already required the exact collection version.
    let hpo = match m.get("schema_version") {
        None => false,
        Some(version) => version.integer(5, 7)? >= 6,
    };
    if hpo {
        require(
            m.get("tree_inputs").is_none(),
            "HPO collection cannot adopt Tree inputs",
        )?;
    }
    let registry = m.require("policy_bindings")?.object()?;
    let mut expected_neural = BTreeSet::new();
    let mut expected_tree = BTreeSet::new();
    for (key, contract) in [
        ("neural_inputs", "retained-external-inference-v1"),
        ("tree_inputs", "player-card-refs-v2"),
    ] {
        if let Some(raw) = m.get(key) {
            raw.closed(&["input_contract", "policies"], &[])?;
            require(
                field(raw, "input_contract")? == contract,
                "original typed retained input contract differs",
            )?;
            let policies = raw.require("policies")?.object()?;
            if key == "tree_inputs" {
                require(
                    !policies.is_empty(),
                    "supplied Tree registry must be nonempty",
                )?;
            }
        }
    }
    for (label, binding) in registry {
        c.check()?;
        binding.closed(&["model_id", "policy_options"], &[])?;
        let options = binding.require("policy_options")?;
        if field(binding, "model_id")? == "heuristic_refresh" {
            require(
                *label == dto::refresh_label(actions, options, b, c)?,
                "original configured baseline registry mismatch",
            )?;
        } else if options.is_null() {
            require(
                ["random", "heuristic"].contains(&field(binding, "model_id")?)
                    && *label == format!("{}-card-instance-v2", field(binding, "model_id")?),
                "original baseline registry mismatch",
            )?;
        } else {
            let tree = field(binding, "model_id")? == "lightgbm";
            let key = if tree { "tree_inputs" } else { "neural_inputs" };
            let raw = m.require(key)?.require("policies")?.require(label)?;
            let actual = if tree {
                controls::tree3::metadata(raw, actions, b, c)?
            } else if hpo {
                controls::hpo::neural_metadata(raw, actions, b, c)?
            } else {
                controls::neural_metadata3(raw, actions, b, c)?
            };
            require(
                actual == *label
                    && raw.require("policy_options")? == options
                    && options.require("model")?.require("model_id")?
                        == binding.require("model_id")?,
                "original typed learned registry mismatch",
            )?;
            if tree {
                expected_tree.insert(label.as_str());
            } else {
                expected_neural.insert(label.as_str());
            }
        }
    }
    require(
        expected_neural.len()
            <= if hpo {
                integer(
                    m.require("config")?
                        .require("hpo_spec")?
                        .require("operating")?,
                    "opponent_pool_size",
                    1,
                    i64::MAX as i128,
                )?
                .saturating_add(1)
            } else {
                6
            },
        "original neural registry capacity exceeded",
    )?;
    for (key, expected) in [
        ("neural_inputs", expected_neural),
        ("tree_inputs", expected_tree),
    ] {
        let actual = match m.get(key) {
            Some(raw) => raw
                .require("policies")?
                .object()?
                .keys()
                .map(String::as_str)
                .collect(),
            None => BTreeSet::new(),
        };
        require(
            actual == expected,
            "extra/missing original typed retained declarations",
        )?;
    }
    Ok(())
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
    let refresh = field(binding, "model_id")? == "heuristic_refresh";
    let model = if refresh {
        require(
            header.require("baseline_options")? == binding.require("policy_options")?
                && label
                    == dto::refresh_label(
                        header.require("actions")?,
                        binding.require("policy_options")?,
                        b,
                        c,
                    )?,
            "original configured baseline header/registry mismatch",
        )?;
        None
    } else if binding.require("policy_options")?.is_null() {
        None
    } else {
        let tree = field(binding, "model_id")? == "lightgbm";
        let raw = m
            .require(if tree { "tree_inputs" } else { "neural_inputs" })?
            .require("policies")?
            .require(label)?;
        require(
            header.require(if tree { "tree" } else { "neural" })? == raw
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
    if refresh {
        identity.as_object_mut().unwrap().insert(
            "baseline_options".into(),
            json::copy(header.require("baseline_options")?, c)?,
        );
    }
    Ok((identity, model))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resident_manifest_uses_original_parent_capacity_before_source_checks() {
        let b = Budget::new(128 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let all = json::parse(include_bytes!("../tests/fixtures/hpo_metadata5_rules_v2.json"), &b, &c).unwrap();
        let mut state = all.root["cases"][0]["state"].clone();
        state["run_variant"] = json::text("hpo-ppo-resident-v1");
        state["effective_config"]["hpo_spec"]["schema_version"] = json::integer(3);
        state["effective_config"]["hpo_spec"]["runtime"] = serde_json::json!({"mode":"resident-worker-v1","cache_entries":6,"cache_weight_bytes":134217728});
        state["effective_config"]["hpo_spec"]["operating"]["game_concurrency"] = json::integer(4);
        state["effective_config"]["hpo_spec"]["operating"]["max_participant_workers"] = json::integer(8);
        state["hpo_spec_sha256"] = json::text(&digest(&state["effective_config"]["hpo_spec"], &b, &c).unwrap());
        let mut m = serde_json::json!({"schema_version":7,"status":"complete","collection_role":"hpo-ppo-batch",
            "linkage":{"parent_run_id":"run-a","batch_id":"batch-000000","collection_generation":0},
            "config":{"run":{"max_participant_workers":8},"hpo_spec":{"operating":{"max_participant_workers":8}}}});
        m["logical_retirement"] = controls::resident::tests::retirement_fixture(&b, &c);
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let limits = CollectionLimits {source_bytes:64 << 20, record_bytes:8 << 20};
        let mut s = Sources::new(root, b.clone(), c.clone(), 64 << 20).unwrap();
        // This bounded receipt test deliberately has no policy/game records.
        // Capacity8 passes retirement and reaches the next independent Source check.
        let error = manifest(&mut s, &m, Some((root, &state)), &limits, &b, &c,
            CorrectedCollectionRole::HpoBatch {games:16}).unwrap_err();
        assert!(format!("{error:?}").contains("action_contract"));
        state["effective_config"]["hpo_spec"]["operating"]["game_concurrency"] = json::integer(2);
        state["effective_config"]["hpo_spec"]["operating"]["max_participant_workers"] = json::integer(4);
        state["hpo_spec_sha256"] = json::text(&digest(&state["effective_config"]["hpo_spec"], &b, &c).unwrap());
        // A manifest claiming8 cannot replace the SAME original parent's4.
        let error = manifest(&mut s, &m, Some((root, &state)), &limits, &b, &c,
            CorrectedCollectionRole::HpoBatch {games:16}).unwrap_err();
        assert!(format!("{error:?}").contains("expected physical capacity"));
    }

    #[test]
    fn refresh_registry_dispatches_before_retained_neural_inputs() {
        let b = Budget::new(512 << 20);
        let c = Control::new(crate::budget::monotonic() + 60.).unwrap();
        let fixture = json::parse(
            include_bytes!("../tests/fixtures/card_instance_journal6_rules_v2.json"),
            &b,
            &c,
        )
        .unwrap();
        let actions = &fixture.root["journals"][0]["rows"][0]["actions"];
        let label = "heuristic_refresh-card-instance-v2-636ccfc7684f690b578076713cdf128f28e9dc52c86014bc9b41cbc6b5091af3";
        let binding = json::object([
            ("model_id", json::text("heuristic_refresh")),
            (
                "policy_options",
                json::object([
                    ("version", json::text("1")),
                    ("attack_threshold", json::integer(12)),
                ]),
            ),
        ]);
        let mut registry = json::object([]);
        registry
            .as_object_mut()
            .unwrap()
            .insert(label.into(), binding);
        let m = json::object([
            ("schema_version", json::integer(5)),
            ("policy_bindings", registry),
        ]);
        retained_registry(&m, actions, &b, &c).unwrap();
        let neural = &fixture.root["journals"][1]["rows"][0];
        let nn_label = neural["policy"].string().unwrap();
        let mut mixed = m.clone();
        mixed["policy_bindings"].as_object_mut().unwrap().insert(
            nn_label.into(),
            json::object([
                ("model_id", neural["model_id"].clone()),
                ("policy_options", neural["neural"]["policy_options"].clone()),
            ]),
        );
        let mut policies = json::object([]);
        policies
            .as_object_mut()
            .unwrap()
            .insert(nn_label.into(), neural["neural"].clone());
        mixed.as_object_mut().unwrap().insert(
            "neural_inputs".into(),
            json::object([
                (
                    "input_contract",
                    json::text("retained-external-inference-v1"),
                ),
                ("policies", policies),
            ]),
        );
        retained_registry(&mixed, actions, &b, &c).unwrap();
        mixed["neural_inputs"]["policies"]
            .as_object_mut()
            .unwrap()
            .clear();
        assert!(retained_registry(&mixed, actions, &b, &c).is_err());
        let mut changed = m.clone();
        changed["policy_bindings"][label]["policy_options"]["attack_threshold"] = json::integer(13);
        assert!(retained_registry(&changed, actions, &b, &c).is_err());
        for invalid in [
            Value::Bool(true),
            json::integer(0),
            json::integer(-1),
            json::text("12"),
        ] {
            let mut bad = m.clone();
            bad["policy_bindings"][label]["policy_options"]["attack_threshold"] = invalid;
            assert!(retained_registry(&bad, actions, &b, &c).is_err());
        }
    }

    #[test]
    fn card_instance_tree3_typed_mixed_registry_never_reinterprets_neural_or_baseline() {
        let b = Budget::new(512 << 20);
        let c = Control::new(crate::budget::monotonic() + 60.).unwrap();
        let neural = json::parse(
            include_bytes!("../tests/fixtures/card_instance_journal6_rules_v2.json"),
            &b,
            &c,
        )
        .unwrap();
        let h = &neural.root["journals"][1]["rows"][0];
        let actions = &h["actions"];
        let tree = json::parse(
            include_bytes!("../tests/fixtures/card_instance_tree3_raw_metadata_rules_v2.json"),
            &b,
            &c,
        )
        .unwrap();
        let mut raw = tree.root["tree"].clone();
        let hash = json::text(&digest(actions, &b, &c).unwrap());
        raw["policy_options"]["action_config_hash"] = hash.clone();
        raw["inference_manifest"]["action_config_hash"] = hash;
        let label = controls::tree3::metadata(&raw, actions, &b, &c).unwrap();
        let mut registry = json::object([]);
        registry.as_object_mut().unwrap().insert(
            label.clone(),
            json::object([
                ("model_id", json::text("lightgbm")),
                ("policy_options", raw["policy_options"].clone()),
            ]),
        );
        registry.as_object_mut().unwrap().insert(
            h["policy"].string().unwrap().into(),
            json::object([
                ("model_id", h["model_id"].clone()),
                ("policy_options", h["neural"]["policy_options"].clone()),
            ]),
        );
        registry.as_object_mut().unwrap().insert(
            "random-card-instance-v2".into(),
            json::object([
                ("model_id", json::text("random")),
                ("policy_options", Value::Null),
            ]),
        );
        let mut policies = json::object([]);
        policies
            .as_object_mut()
            .unwrap()
            .insert(label.clone(), raw.clone());
        let mut nn = json::object([]);
        nn.as_object_mut()
            .unwrap()
            .insert(h["policy"].string().unwrap().into(), h["neural"].clone());
        let m = json::object([
            ("policy_bindings", registry),
            (
                "tree_inputs",
                json::object([
                    ("input_contract", json::text("player-card-refs-v2")),
                    ("policies", policies),
                ]),
            ),
            (
                "neural_inputs",
                json::object([
                    (
                        "input_contract",
                        json::text("retained-external-inference-v1"),
                    ),
                    ("policies", nn),
                ]),
            ),
        ]);
        retained_registry(&m, actions, &b, &c).unwrap();
        for mutation in 0..7 {
            let mut bad = m.clone();
            match mutation {
                0 => {
                    bad["tree_inputs"]["policies"]
                        .as_object_mut()
                        .unwrap()
                        .clear();
                }
                1 => {
                    bad["tree_inputs"]["input_contract"] =
                        json::text("retained-external-inference-v1")
                }
                2 => {
                    bad["neural_inputs"]["policies"]
                        .as_object_mut()
                        .unwrap()
                        .insert(label.clone(), raw.clone());
                }
                3 => bad["policy_bindings"][&label]["model_id"] = json::text("mlp"),
                4 => {
                    bad["policy_bindings"]["random-card-instance-v2"]["model_id"] =
                        json::text("lightgbm")
                }
                5 => {
                    bad["tree_inputs"]
                        .as_object_mut()
                        .unwrap()
                        .insert("authority".into(), Value::Bool(true));
                }
                _ => {
                    bad["tree_inputs"]["policies"]
                        .as_object_mut()
                        .unwrap()
                        .clear();
                    bad["policy_bindings"]
                        .as_object_mut()
                        .unwrap()
                        .remove(&label);
                }
            }
            assert!(
                retained_registry(&bad, actions, &b, &c).is_err(),
                "mutation{mutation}"
            );
        }
    }
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
