//! Pure corrected Tree3 RAW metadata. No model/file/pair/Source authority.
//!
//! RAW constructs no Booster. The portable local execution/array formula is
//! exact; actual parser/control/copy ownership is admitted to the SAME native
//! Budget. CPython's graph/alias-specific `_owned` inference guard is deliberately
//! not emulated here. Semantic parity applies when BOTH readers admit an input.
use super::*;

const RECIPE: &[u8] = include_bytes!("../assets/card-instance-tree3-layout-recipe-v1.json");
fn require(ok: bool, text: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::contract(text))
    }
}
fn recipe(b: &Arc<Budget>, c: &Control) -> Result<Document> {
    json::parse(RECIPE, b, c)
}

pub(crate) fn model(value: &Value) -> Result<()> {
    value.closed(
        &[
            "model_id",
            "version",
            "encoder",
            "network",
            "initialization",
        ],
        &[],
    )?;
    is(value, "model_id", "lightgbm")?;
    is(value, "version", "3")?;
    let e = value.require("encoder")?;
    e.closed(
        &[
            "recipe",
            "max_positions",
            "aggregate_window",
            "aggregate_recipe",
        ],
        &[],
    )?;
    is(e, "recipe", "participant-structured-tree-v2")?;
    is(e, "aggregate_recipe", "public-exact-card-refs-v3")?;
    n(e, "max_positions", 1, 128)?;
    n(e, "aggregate_window", 1, 65536)?;
    value.require("network")?.closed(&["num_leaves"], &[])?;
    n(value.require("network")?, "num_leaves", 2, 65536)?;
    value.require("initialization")?.closed(&["recipe"], &[])?;
    is(
        value.require("initialization")?,
        "recipe",
        "seed-derived-v1",
    )
}
fn escape(value: &str) -> String {
    let mut result = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || b"_-".contains(&byte) {
            result.push(byte as char)
        } else {
            result.push_str(&format!("%{byte:02X}"))
        }
    }
    result
}
pub(crate) fn layout(encoder: &Value, b: &Arc<Budget>, c: &Control) -> Result<Document> {
    encoder.closed(
        &[
            "recipe",
            "max_positions",
            "aggregate_window",
            "aggregate_recipe",
        ],
        &[],
    )?;
    is(encoder, "recipe", "participant-structured-tree-v2")?;
    is(encoder, "aggregate_recipe", "public-exact-card-refs-v3")?;
    let positions = n(encoder, "max_positions", 1, 128)? as usize;
    let window = n(encoder, "aggregate_window", 1, 65536)?;
    let r = recipe(b, c)?;
    let base = r.root.require("base_fields")?.array()?;
    let card = r.root.require("position_fields")?.array()?;
    let width = base.len() + positions * card.len();
    let _copies = b.reserve(
        width
            .checked_mul(2048)
            .and_then(|v| v.checked_add(65536))
            .ok_or_else(|| Error::resource("Tree3 layout size overflow"))?,
    )?;
    let mut names = Vec::with_capacity(width);
    let mut mapping = Vec::with_capacity(width);
    let mut transforms = Vec::with_capacity(width);
    let mut indices = Vec::new();
    let mut limits = Vec::new();
    let mut add = |segments: Vec<Value>, field: &Value| -> Result<()> {
        c.check()?;
        let spelling = segments
            .iter()
            .map(|v| v.string().map(escape))
            .collect::<Result<Vec<_>>>()?
            .join(".");
        let kind = field.require("transform")?.string()?;
        if kind == "category" {
            indices.push(json::integer(names.len() as i128));
            limits.push(json::copy(field.require("categorical_limit")?, c)?);
        }
        names.push(json::text(&spelling));
        transforms.push(json::text(kind));
        mapping.push(json::object([
            ("segments", json::array(segments)),
            ("name", json::text(&spelling)),
        ]));
        Ok(())
    };
    for f in base {
        add(f.require("segments")?.array()?.to_vec(), f)?;
    }
    for position in 0..positions {
        for f in card {
            let mut segments = vec![
                json::text("candidate"),
                json::text("position"),
                json::text(&position.to_string()),
            ];
            segments.extend(f.require("segments")?.array()?.iter().cloned());
            add(segments, f)?;
        }
    }
    let mut value = json::copy(r.root.require("fixed")?, c)?;
    for (k, v) in [
        ("encoder", json::copy(encoder, c)?),
        (
            "token_signature",
            json::text(&card_instance_token_signature(window as usize, b, c)?),
        ),
        ("feature_names", json::array(names)),
        ("transforms", json::array(transforms)),
        ("categorical_indices", json::array(indices)),
        ("categorical_limits", json::array(limits)),
        ("raw_field_mapping", json::array(mapping)),
    ] {
        value.as_object_mut().unwrap().insert(k.into(), v);
    }
    let signature = digest(&value, b, c)?;
    value
        .as_object_mut()
        .unwrap()
        .insert("feature_signature".into(), json::text(&signature));
    let bytes = json::canonical(&value, b, c)?;
    json::parse(&bytes, b, c)
}
fn source(value: &Value) -> Result<()> {
    is(value, "source_kind", "card-instance-formal-ppo-originals")?;
    let origin = value.require("source_origin")?;
    origin.closed(
        &[
            "purpose",
            "synthetic_capacity",
            "outcome_origin",
            "qualification_recipe",
        ],
        &[],
    )?;
    is(origin, "purpose", "qualified_game")?;
    is(origin, "outcome_origin", "observed-natural-simulator")?;
    is(
        origin,
        "qualification_recipe",
        "three-formal-parents-final64-v1",
    )?;
    require(
        !origin.require("synthetic_capacity")?.boolean()?,
        "game Tree3 cannot claim synthetic origin",
    )
}
fn execution(
    inference: &Value,
    resources: &Value,
    descriptor: &Value,
    encoder: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<(Value, Value)> {
    inference.closed(
        &["device", "candidate_chunk", "max_candidates", "num_threads"],
        &[],
    )?;
    is(inference, "device", "cpu")?;
    let chunk = n(inference, "candidate_chunk", 1, 65536)?;
    n(inference, "max_candidates", 1, 65536)?;
    let threads = n(inference, "num_threads", 1, 64)?;
    resources.closed(&["max_owned_bytes", "max_array_bytes"], &[])?;
    let owned = n(resources, "max_owned_bytes", 1, i64::MAX as i128)?;
    let array = n(resources, "max_array_bytes", 1, 64i128 << 20)?;
    descriptor.closed(
        &["tree_count", "max_depth", "model_sha256", "model_bytes"],
        &[],
    )?;
    let trees = n(descriptor, "tree_count", 1, i32::MAX as i128)?;
    let depth = n(descriptor, "max_depth", 0, 65535)?;
    checksum(descriptor.require("model_sha256")?)?;
    let bytes = n(descriptor, "model_bytes", 1, 64i128 << 20)?;
    let positions = n(encoder, "max_positions", 1, 128)?;
    let r = recipe(b, c)?;
    let card = n(&r.root, "card_width", 1, 65536)?;
    let context = n(&r.root, "context_width", 1, 65536)?;
    let width = r.root.require("base_fields")?.array()?.len() as i128 + positions * card;
    let fixed = (256i128 << 20) + threads * (64i128 << 20) + 16 * bytes;
    let context_bytes = (context + 18 * card) * 4 + 65536;
    let total = fixed + chunk * (12 * width + 32) + context_bytes;
    require(
        chunk * width * 4 <= array && total <= owned,
        "Tree3 exact array/joint resource caps exceeded",
    )?;
    let costs = json::object([
        ("version", json::text("proposal-cost-v1")),
        ("actor_hidden_width", json::integer(0)),
        ("actor_input_width", json::integer(0)),
        ("candidate_chunk", json::integer(chunk)),
        ("card_width", json::integer(0)),
        ("context_width", json::integer(0)),
        ("fixed_bytes", json::integer(fixed + context_bytes)),
        ("max_preparation_bytes", json::integer(owned)),
        ("max_scoring_operations", json::integer(i64::MAX as i128)),
        ("max_storage_bytes", json::integer(64i128 << 20)),
        ("max_tensor_bytes", json::integer(array)),
        ("position_capacity", json::integer(positions)),
        (
            "tail_operations_per_candidate",
            json::integer(10 * width + 2 * trees * (depth + 1)),
        ),
        ("tail_tensor_bytes_per_candidate", json::integer(4 * width)),
        (
            "tail_working_bytes_per_candidate",
            json::integer(12 * width + 32),
        ),
    ]);
    Ok((costs, json::integer(total)))
}
pub(crate) fn options(
    value: &Value,
    actions: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<String> {
    value.closed(
        &[
            "version",
            "checkpoint",
            "expected_manifest_digest",
            "model",
            "inference",
            "resources",
            "action_config_hash",
            "tree_descriptor",
            "feature_signature",
            "source_kind",
            "source_origin",
            "training_pair",
        ],
        &[],
    )?;
    is(value, "version", "3")?;
    require(
        !text(value, "checkpoint")?.is_empty(),
        "Tree3 own checkpoint required",
    )?;
    checksum(value.require("expected_manifest_digest")?)?;
    require(
        text(value, "action_config_hash")? == digest(actions, b, c)?,
        "Tree3 exact action configuration mismatch",
    )?;
    model(value.require("model")?)?;
    let encoder = value.require("model")?.require("encoder")?;
    let positions = n(encoder, "max_positions", 1, 18)?;
    require(
        n(actions, "max_sequence_length", 1, 18)? <= positions
            && n(actions, "max_candidates", 1, 65536)?
                <= n(value.require("inference")?, "max_candidates", 1, 65536)?
            && n(value.require("inference")?, "candidate_chunk", 256, 256)? == 256,
        "Tree3 real18/chunk256/action limits mismatch",
    )?;
    let derived = layout(encoder, b, c)?;
    eq(
        value.require("feature_signature")?,
        derived.root.require("feature_signature")?,
        "Tree3 feature signature mismatch",
    )?;
    execution(
        value.require("inference")?,
        value.require("resources")?,
        value.require("tree_descriptor")?,
        encoder,
        b,
        c,
    )?;
    source(value)?;
    let pair = value.require("training_pair")?;
    pair.closed(&["request_sha256", "completion_sha256"], &[])?;
    checksum(pair.require("request_sha256")?)?;
    checksum(pair.require("completion_sha256")?)?;
    let mut binding = json::copy(value, c)?;
    binding.as_object_mut().unwrap().remove("checkpoint");
    Ok(format!(
        "lightgbm-tree-v3-{}-{}",
        text(value, "expected_manifest_digest")?,
        digest(&binding, b, c)?
    ))
}
pub(crate) fn metadata(
    raw: &Value,
    actions: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<String> {
    let encoded = json::canonical(raw, b, c)?;
    let _copies = json::admit(encoded.len(), b)?;
    raw.closed(
        &[
            "input_contract",
            "policy_options",
            "inference_manifest",
            "files",
        ],
        &[],
    )?;
    is(raw, "input_contract", "player-card-refs-v2")?;
    let safe = raw.require("policy_options")?;
    let label = options(safe, actions, b, c)?;
    let m = raw.require("inference_manifest")?;
    m.closed(
        &[
            "schema_version",
            "kind",
            "model",
            "architecture_identity",
            "input_recipe",
            "token_signature",
            "feature_recipe",
            "feature_signature",
            "feature_names",
            "categorical_indices",
            "semantics",
            "action_config_hash",
            "native_parameters",
            "runtime",
            "creation_execution",
            "tree_descriptor",
            "model_file",
            "fit_origin",
            "provenance",
            "action_contract",
            "purpose",
            "synthetic_capacity",
            "rule_signature",
            "layout_signature",
            "feature_layout",
            "source_kind",
            "source_origin",
        ],
        &[],
    )?;
    m.require("schema_version")?.integer(4, 4)?;
    is(m, "kind", "structured-tree-inference")?;
    is(m, "fit_origin", "trained")?;
    game_binding3(m, b, c)?;
    source(m)?;
    eq(
        m.require("source_kind")?,
        safe.require("source_kind")?,
        "Tree3 source differs",
    )?;
    eq(
        m.require("source_origin")?,
        safe.require("source_origin")?,
        "Tree3 origin differs",
    )?;
    eq(
        m.require("model")?,
        safe.require("model")?,
        "Tree3 full architecture differs",
    )?;
    eq(
        m.require("action_config_hash")?,
        safe.require("action_config_hash")?,
        "Tree3 action hash differs",
    )?;
    eq(
        m.require("tree_descriptor")?,
        safe.require("tree_descriptor")?,
        "Tree3 own descriptor differs",
    )?;
    let derived = layout(safe.require("model")?.require("encoder")?, b, c)?;
    for key in [
        "token_signature",
        "feature_signature",
        "feature_names",
        "categorical_indices",
        "rule_signature",
        "layout_signature",
    ] {
        eq(
            m.require(key)?,
            derived.root.require(key)?,
            "Tree3 derived feature metadata differs",
        )?;
    }
    eq(
        m.require("feature_layout")?,
        &derived.root,
        "Tree3 full feature layout differs",
    )?;
    is(m, "feature_recipe", "participant-structured-tree-v2")?;
    is(m, "input_recipe", "participant-structured-v2")?;
    let architecture = json::object([
        ("model", json::copy(safe.require("model")?, c)?),
        (
            "feature_signature",
            json::copy(derived.root.require("feature_signature")?, c)?,
        ),
    ]);
    require(
        text(m, "architecture_identity")? == digest(&architecture, b, c)?,
        "Tree3 architecture identity differs",
    )?;
    let r = recipe(b, c)?;
    let mut semantics = json::copy(r.root.require("semantics_base")?, c)?;
    semantics.as_object_mut().unwrap().insert(
        "token_signature".into(),
        json::copy(m.require("token_signature")?, c)?,
    );
    eq(
        m.require("semantics")?,
        &semantics,
        "Tree3 complete semantics differ",
    )?;
    let parameters = m.require("native_parameters")?;
    parameters.closed(
        &[
            "objective",
            "num_class",
            "boosting",
            "linear_tree",
            "zero_as_missing",
        ],
        &[],
    )?;
    is(parameters, "objective", "regression")?;
    is(parameters, "boosting", "gbdt")?;
    parameters.require("num_class")?.integer(1, 1)?;
    require(
        !parameters.require("linear_tree")?.boolean()?
            && !parameters.require("zero_as_missing")?.boolean()?,
        "Tree3 supported numeric parameters required",
    )?;
    let runtime = m.require("runtime")?;
    runtime.closed(
        &["numpy_version", "lightgbm_version", "native_library_sha256"],
        &[],
    )?;
    is(runtime, "numpy_version", "2.5.4")?;
    is(runtime, "lightgbm_version", "4.7.0")?;
    checksum(runtime.require("native_library_sha256")?)?;
    let creation = m.require("creation_execution")?;
    creation.closed(&["inference", "resources", "tree_descriptor"], &[])?;
    eq(
        creation.require("tree_descriptor")?,
        m.require("tree_descriptor")?,
        "Tree3 creation descriptor differs",
    )?;
    execution(
        creation.require("inference")?,
        creation.require("resources")?,
        m.require("tree_descriptor")?,
        safe.require("model")?.require("encoder")?,
        b,
        c,
    )?;
    let file = m.require("model_file")?;
    file.closed(&["file", "sha256", "bytes"], &[])?;
    is(file, "file", "model.txt")?;
    eq(
        file.require("sha256")?,
        safe.require("tree_descriptor")?.require("model_sha256")?,
        "Tree3 model hash differs",
    )?;
    eq(
        file.require("bytes")?,
        safe.require("tree_descriptor")?.require("model_bytes")?,
        "Tree3 model bytes differ",
    )?;
    let provenance = m.require("provenance")?;
    provenance.closed(
        &[
            "schema_version",
            "kind",
            "training_run_id",
            "request_sha256",
            "dataset_manifest_sha256",
            "input_binding",
            "source_kind",
            "source_origin",
        ],
        &[],
    )?;
    provenance.require("schema_version")?.integer(4, 4)?;
    is(provenance, "kind", "structured-fixed-dataset3-fit")?;
    source(provenance)?;
    checksum(provenance.require("request_sha256")?)?;
    checksum(provenance.require("dataset_manifest_sha256")?)?;
    require(
        text(provenance, "training_run_id")?.len() == 32
            && text(provenance, "training_run_id")?
                .bytes()
                .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v)),
        "Tree3 training run identifier differs",
    )?;
    let input_binding = provenance.require("input_binding")?;
    input_binding.closed(
        &[
            "action_contract",
            "rule_signature",
            "layout_signature",
            "purpose",
            "synthetic_capacity",
        ],
        &[],
    )?;
    game_binding3(input_binding, b, c)?;
    eq(
        provenance.require("request_sha256")?,
        safe.require("training_pair")?.require("request_sha256")?,
        "Tree3 own request/provenance differs",
    )?;
    // completion SHA remains a checked opaque own-pair reference. It is NOT an
    // inference provenance field, avoiding the completion/inference hash cycle.
    let files = raw.require("files")?.array()?;
    require(files.len() == 2, "Tree3 exact two own files required")?;
    for (row, (name, hash)) in files.iter().zip([
        ("inference.json", safe.require("expected_manifest_digest")?),
        ("model.txt", file.require("sha256")?),
    ]) {
        row.closed(&["path", "sha256", "identity"], &[])?;
        is(row, "path", name)?;
        checksum(row.require("sha256")?)?;
        eq(row.require("sha256")?, hash, "Tree3 own file hash mismatch")?;
        let ids = row.require("identity")?.array()?;
        require(ids.len() == 5, "Tree3 five-integer owned identity required")?;
        for id in ids {
            id.integer(0, i128::MAX)?;
        }
    }
    Ok(label)
}
pub(crate) fn settings(
    safe: &Value,
    actions: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<(Value, Value)> {
    options(safe, actions, b, c)?;
    let (costs, _) = execution(
        safe.require("inference")?,
        safe.require("resources")?,
        safe.require("tree_descriptor")?,
        safe.require("model")?.require("encoder")?,
        b,
        c,
    )?;
    let caps = json::object([
        ("max_interval_events", json::integer(4096)),
        ("max_episode_events", json::integer(262144)),
        ("max_decisions", Value::Null),
        (
            "max_positions",
            json::copy(actions.require("max_sequence_length")?, c)?,
        ),
        (
            "max_candidates",
            json::copy(actions.require("max_candidates")?, c)?,
        ),
    ]);
    Ok((caps, costs))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn owners() -> (Arc<Budget>, Control) {
        (
            Budget::new(256 << 20),
            Control::new(crate::budget::monotonic() + 60.0).unwrap(),
        )
    }
    #[test]
    fn card_instance_tree3_exact_execution_and_array_thresholds_without_booster() {
        let (b, c) = owners();
        let oracle = json::parse(
            include_bytes!("../tests/fixtures/card_instance_tree3_raw_metadata.json"),
            &b,
            &c,
        )
        .unwrap();
        let mut raw = oracle.root["tree"].clone();
        let actions = &oracle.root["actions"];
        let safe = &raw["policy_options"];
        let (costs, total) = execution(
            &safe["inference"],
            &safe["resources"],
            &safe["tree_descriptor"],
            &safe["model"]["encoder"],
            &b,
            &c,
        )
        .unwrap();
        let total = total.integer(1, i64::MAX as i128).unwrap();
        let array = costs["candidate_chunk"].integer(1, 65536).unwrap()
            * costs["tail_tensor_bytes_per_candidate"]
                .integer(1, i64::MAX as i128)
                .unwrap();
        raw["policy_options"]["resources"]["max_owned_bytes"] = json::integer(total);
        raw["policy_options"]["resources"]["max_array_bytes"] = json::integer(array);
        // RAW owns no Booster. Portable local execution terms are checked
        // exactly; actual typed-control/copy ownership uses the SAME global b.
        metadata(&raw, actions, &b, &c).unwrap();
        raw["policy_options"]["resources"]["max_owned_bytes"] = json::integer(total - 1);
        assert!(metadata(&raw, actions, &b, &c).is_err());
        raw["policy_options"]["resources"]["max_owned_bytes"] = json::integer(total);
        raw["policy_options"]["resources"]["max_array_bytes"] = json::integer(array - 1);
        assert!(metadata(&raw, actions, &b, &c).is_err());
    }
    #[test]
    fn card_instance_tree3_layout_python_full_recipe_and_window_hashes() {
        let (b, c) = owners();
        let references = json::parse(
            include_bytes!("../tests/fixtures/card_instance_tree3_layout_python.json"),
            &b,
            &c,
        )
        .unwrap();
        for row in references.root.array().unwrap() {
            let value = layout(&row["encoder"], &b, &c).unwrap();
            assert_eq!(
                digest(&value.root, &b, &c).unwrap(),
                row["sha256"].string().unwrap()
            );
            assert_eq!(value.root["feature_signature"], row["feature_signature"]);
            assert_eq!(value.root["token_signature"], row["token_signature"]);
            assert_eq!(
                value.root["feature_names"].array().unwrap().len() as i128,
                row["width"].integer(1, 20000).unwrap()
            );
        }
    }
    #[test]
    fn card_instance_tree3_metadata_python_settings_label_and_mutations() {
        let (b, c) = owners();
        let oracle = json::parse(
            include_bytes!("../tests/fixtures/card_instance_tree3_raw_metadata.json"),
            &b,
            &c,
        )
        .unwrap();
        let raw = &oracle.root["tree"];
        let actions = &oracle.root["actions"];
        assert_eq!(
            metadata(raw, actions, &b, &c).unwrap(),
            oracle.root["policy_label"].string().unwrap()
        );
        let (caps, costs) = settings(&raw["policy_options"], actions, &b, &c).unwrap();
        assert_eq!(caps, oracle.root["caps"]);
        assert_eq!(costs, oracle.root["numeric_costs"]);
        for mutation in 0..17 {
            let mut bad = raw.clone();
            match mutation {
                0 => bad["policy_options"]["source_origin"]["purpose"] = json::text("component"),
                1 => {
                    bad["policy_options"]["source_origin"]["synthetic_capacity"] = Value::Bool(true)
                }
                2 => bad["policy_options"]["source_kind"] = json::text("component-fixtures"),
                3 => {
                    bad["policy_options"]["training_pair"]["completion_sha256"] = json::text("bad")
                }
                4 => {
                    bad["inference_manifest"]["provenance"]["request_sha256"] =
                        json::text(&"f".repeat(64))
                }
                5 => {
                    bad["inference_manifest"]["provenance"]
                        .as_object_mut()
                        .unwrap()
                        .insert("completion_sha256".into(), json::text(&"f".repeat(64)));
                }
                6 => bad["inference_manifest"]["feature_names"][0] = json::text("foreign-feature"),
                7 => {
                    bad["inference_manifest"]["model"]["network"]["num_leaves"] = json::integer(32)
                }
                8 => {
                    bad["policy_options"]["model"]["encoder"]
                        .as_object_mut()
                        .unwrap()
                        .insert("private".into(), Value::Null);
                }
                9 => bad["files"][0]["identity"][0] = json::integer(-1),
                10 => bad["files"][0]["path"] = json::text("../inference.json"),
                11 => bad["policy_options"]["resources"]["max_owned_bytes"] = json::integer(1),
                12 => bad["policy_options"]["inference"]["candidate_chunk"] = json::integer(255),
                13 => bad["inference_manifest"]["fit_origin"] = json::text("toy"),
                14 => {
                    bad["inference_manifest"]["semantics"]["input_recipe"] = json::text("foreign")
                }
                15 => bad["policy_options"]["model"]["version"] = json::text("2"),
                _ => {
                    bad["inference_manifest"]["provenance"]["input_binding"]
                        .as_object_mut()
                        .unwrap()
                        .insert("foreign".into(), Value::Null);
                }
            }
            assert!(
                metadata(&bad, actions, &b, &c).is_err(),
                "mutation{mutation}"
            );
        }
        let mut relocated = raw.clone();
        relocated["policy_options"]["checkpoint"] = json::text("relocated-own-model");
        assert_eq!(
            metadata(&relocated, actions, &b, &c).unwrap(),
            metadata(raw, actions, &b, &c).unwrap()
        );
        let tiny = Budget::new(1024);
        assert!(metadata(raw, actions, &tiny, &c).is_err());
        let cancelled = Control::new(crate::budget::monotonic() + 60.0).unwrap();
        cancelled.cancel();
        assert!(metadata(raw, actions, &b, &cancelled).is_err());
    }
}
