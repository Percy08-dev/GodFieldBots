//! Closed resident management evidence. No live ownership or policy authority.
use super::*;
use std::collections::BTreeSet;
fn field<'a>(v: &'a Value, k: &str) -> Result<&'a str> {
    v.require(k)?.string()
}

pub(crate) const CAPABILITY: &str = "collection7;proof4;registry5;pool1";
const OWNER: &[&str] = &[
    "job_ticket",
    "pool_id",
    "pool_epoch",
    "worker_key",
    "pid",
    "viewer",
    "lease_epoch",
];
const ACK: &[&str] = &[
    "schema_version",
    "protocol",
    "run_id",
    "generation",
    "command",
    "request_epoch",
    "checkpoint_sha256",
    "cache_identity",
    "job_ticket",
    "pool_id",
    "pool_epoch",
    "worker_key",
    "pid",
    "viewer",
    "lease_epoch",
    "clean_reset",
    "private_state_empty",
    "pending_io_empty",
    "cuda_synchronized",
    "reset_version",
];
const BASE_BINDING: &[&str] = &[
    "run_id",
    "token",
    "generation",
    "inference_sha256",
    "generation_sha256",
    "policy_sha256",
    "weights_sha256",
    "cache_identity",
];

pub(crate) fn is_resident(state: &Value) -> bool {
    state.get("run_variant").and_then(Value::as_str) == Some("hpo-ppo-resident-v1")
}
pub(crate) fn runtime(value: &Value) -> Result<()> {
    value.closed(&["mode", "cache_entries", "cache_weight_bytes"], &[])?;
    is(value, "mode", "resident-worker-v1")?;
    n(value, "cache_entries", 6, 6)?;
    n(value, "cache_weight_bytes", 134217728, 134217728)?;
    Ok(())
}
fn token(value: &Value) -> Result<()> {
    let raw = value.string()?;
    if raw.is_empty()
        || raw.len() > 128
        || !raw.as_bytes()[0].is_ascii_alphanumeric()
        || !raw
            .bytes()
            .all(|v| v.is_ascii_alphanumeric() || b"_.-".contains(&v))
    {
        return Err(Error::contract("bounded resident token required"));
    }
    Ok(())
}
fn owner(value: &Value) -> Result<()> {
    for name in ["job_ticket", "pool_id"] {
        let text = value.require(name)?.string()?;
        if text.len() != 32
            || !text
                .bytes()
                .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
        {
            return Err(Error::contract("resident job/pool32hex identity required"));
        }
    }
    for name in ["pool_epoch", "worker_key", "pid", "lease_epoch"] {
        n(value, name, 1, i64::MAX as i128)?;
    }
    n(value, "viewer", 1, 2)?;
    Ok(())
}
fn ack(value: &Value) -> Result<()> {
    value.closed(ACK, &[])?;
    owner(value)?;
    n(value, "schema_version", 1, 1)?;
    n(value, "generation", 0, i64::MAX as i128)?;
    n(value, "request_epoch", 1, i64::MAX as i128)?;
    checksum(value.require("checkpoint_sha256")?)?;
    is(value, "protocol", "resident-worker-v1")?;
    is(value, "reset_version", "resident-reset-v1")?;
    let run = value.require("run_id")?.string()?;
    if run.is_empty() || run.chars().count() > 128 {
        return Err(Error::contract("resident run identity required"));
    }
    for key in [
        "clean_reset",
        "private_state_empty",
        "pending_io_empty",
        "cuda_synchronized",
    ] {
        eq(
            value.require(key)?,
            &Value::Bool(true),
            "resident clean reset/private/I/O/CUDA confirmation required",
        )?;
    }
    let operation = field(value, "command")?;
    if !["END", "PARK", "SYNC", "EVICT"].contains(&operation) {
        return Err(Error::contract("resident ACK operation invalid"));
    }
    if operation == "EVICT" {
        checksum(value.require("cache_identity")?)?;
    } else if !value.require("cache_identity")?.is_null() {
        return Err(Error::contract("resident non-EVICT has cache identity"));
    }
    Ok(())
}
pub(crate) fn retirement(value: &Value, max_workers: i128, b: &Arc<Budget>, c: &Control) -> Result<()> {
    if !(1..=i64::MAX as i128).contains(&max_workers) {
        return Err(Error::contract("positive signed63 expected resident capacity required"));
    }
    value.closed(
        &[
            "schema_version",
            "kind",
            "run_id",
            "collection_id",
            "job_ticket",
            "pool_id",
            "pool_epoch",
            "generation",
            "owners",
        ],
        &[],
    )?;
    n(value, "schema_version", 1, 1)?;
    is(value, "kind", "hpo-resident-logical-retirement")?;
    token(value.require("collection_id")?)?;
    n(value, "pool_epoch", 1, i64::MAX as i128)?;
    n(value, "generation", 0, i64::MAX as i128)?;
    let rows = value.require("owners")?.array()?;
    if rows.is_empty() || rows.len() as i128 > max_workers {
        return Err(Error::contract(
            "resident owner set exceeds expected physical capacity",
        ));
    }
    let mut keys = BTreeSet::new();
    let mut viewers = [0i128; 2];
    for row in rows {
        c.check()?;
        let mut fields = OWNER.to_vec();
        fields.extend(["end_ack", "end_ack_sha256", "park_ack", "park_ack_sha256"]);
        row.closed(&fields, &[])?;
        owner(row)?;
        if !keys.insert(n(row, "worker_key", 1, i64::MAX as i128)?) {
            return Err(Error::contract("duplicate resident owner"));
        }
        let viewer = n(row, "viewer", 1, 2)? as usize - 1;
        viewers[viewer] += 1;
        if viewers[viewer] > max_workers / 2 {
            return Err(Error::contract("resident viewer partition exceeds expected capacity"));
        }
        for (name, command) in [("end_ack", "END"), ("park_ack", "PARK")] {
            let a = row.require(name)?;
            ack(a)?;
            is(a, "command", command)?;
            is(row, &format!("{name}_sha256"), &digest(a, b, c)?)?;
            for key in OWNER {
                eq(
                    a.require(key)?,
                    row.require(key)?,
                    "resident ACK owner differs",
                )?;
            }
            for key in ["run_id", "generation"] {
                eq(
                    a.require(key)?,
                    value.require(key)?,
                    "resident ACK run/generation differs",
                )?;
            }
            for key in ["job_ticket", "pool_id", "pool_epoch"] {
                eq(
                    row.require(key)?,
                    value.require(key)?,
                    "resident foreign job/pool",
                )?;
            }
        }
        if n(
            row.require("park_ack")?,
            "request_epoch",
            1,
            i64::MAX as i128,
        )? <= n(
            row.require("end_ack")?,
            "request_epoch",
            1,
            i64::MAX as i128,
        )? {
            return Err(Error::contract("resident PARK must follow END"));
        }
        eq(
            row.require("park_ack")?.require("checkpoint_sha256")?,
            row.require("end_ack")?.require("checkpoint_sha256")?,
            "resident PARK changed END policy",
        )?;
    }
    Ok(())
}
fn binding(
    s: &mut Sources,
    state: &Value,
    records: &[Document],
    model: &Value,
    value: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<usize> {
    let mut keys = BASE_BINDING.to_vec();
    keys.extend(OWNER);
    value.closed(&keys, &[])?;
    owner(value)?;
    token(value.require("token")?)?;
    eq(
        value.require("run_id")?,
        state.require("run_id")?,
        "resident cache pin run differs",
    )?;
    let number = n(value, "generation", 0, records.len() as i128 - 1)? as usize;
    let record = &records[number].root;
    eq(
        value.require("inference_sha256")?,
        record.require("inference_sha256")?,
        "resident cache inference differs",
    )?;
    is(
        value,
        "generation_sha256",
        s.verified_sha(&s.root.join(format!("generations/{number:06}.json")))?,
    )?;
    let mut weights = json::object([]);
    for (name, sha) in record.require("artifact_files")?.object()? {
        if name.starts_with("weights-") {
            weights
                .as_object_mut()
                .unwrap()
                .insert(name.clone(), sha.clone());
        }
    }
    let execution = state.require("execution")?.require("inference")?;
    let actions = state.require("effective_config")?.require("actions")?;
    let mut safe = json::object([
        ("version", json::text("4")),
        (
            "checkpoint",
            json::text(
                s.root
                    .join(format!("artifacts/{number:06}"))
                    .to_str()
                    .ok_or_else(|| Error::contract("resident UTF8checkpoint required"))?,
            ),
        ),
        (
            "expected_manifest_digest",
            record.require("inference_sha256")?.clone(),
        ),
        ("model", model.clone()),
        ("inference", execution.require("binding")?.clone()),
        (
            "cuda_visible_devices",
            execution.require("cuda_visible_devices")?.clone(),
        ),
        ("action_config_hash", json::text(&digest(actions, b, c)?)),
    ]);
    for key in GAME_CLASS {
        safe.as_object_mut()
            .unwrap()
            .insert((*key).into(), state.require(key)?.clone());
    }
    hpo::options(&safe, actions, b, c)?;
    let policy = digest(
        &json::object([
            ("model_id", model.require("model_id")?.clone()),
            ("policy_options", safe),
        ]),
        b,
        c,
    )?;
    let weights = digest(&weights, b, c)?;
    is(value, "policy_sha256", &policy)?;
    is(value, "weights_sha256", &weights)?;
    is(
        value,
        "cache_identity",
        &digest(
            &json::object([
                ("policy_sha256", json::text(&policy)),
                ("weights_sha256", json::text(&weights)),
                (
                    "inference_sha256",
                    record.require("inference_sha256")?.clone(),
                ),
            ]),
            b,
            c,
        )?,
    )?;
    Ok(number)
}
fn release(value: &Value, pin: &Value, b: &Arc<Budget>, c: &Control) -> Result<()> {
    value.closed(
        &["schema_version", "kind", "binding_sha256"],
        &["ack", "ack_sha256", "resource"],
    )?;
    n(value, "schema_version", 1, 1)?;
    is(value, "binding_sha256", &digest(pin, b, c)?)?;
    match field(value, "kind")? {
        "hpo-resident-cache-eviction" => {
            value.closed(
                &[
                    "schema_version",
                    "kind",
                    "binding_sha256",
                    "ack",
                    "ack_sha256",
                ],
                &[],
            )?;
            let a = value.require("ack")?;
            ack(a)?;
            is(a, "command", "EVICT")?;
            is(value, "ack_sha256", &digest(a, b, c)?)?;
            eq(
                a.require("cache_identity")?,
                pin.require("cache_identity")?,
                "resident release cache identity differs",
            )?;
            eq(
                a.require("checkpoint_sha256")?,
                pin.require("inference_sha256")?,
                "resident release checkpoint differs",
            )?;
            for key in OWNER {
                if *key != "lease_epoch" {
                    eq(
                        a.require(key)?,
                        pin.require(key)?,
                        "resident release physical owner differs",
                    )?;
                }
            }
            n(
                a,
                "lease_epoch",
                n(pin, "lease_epoch", 1, i64::MAX as i128)?,
                i64::MAX as i128,
            )?;
            eq(
                a.require("run_id")?,
                pin.require("run_id")?,
                "resident cache release run differs",
            )?;
            n(
                a,
                "generation",
                n(pin, "generation", 0, i64::MAX as i128)?,
                i64::MAX as i128,
            )?;
        }
        "hpo-resident-cache-reaped" => {
            value.closed(
                &["schema_version", "kind", "binding_sha256", "resource"],
                &[],
            )?;
            let resource = value.require("resource")?;
            resource.closed(&["kind", "key", "pid", "outcome"], &[])?;
            is(resource, "kind", "worker")?;
            is(resource, "outcome", "reaped")?;
            eq(
                resource.require("key")?,
                pin.require("worker_key")?,
                "resident release worker differs",
            )?;
            eq(
                resource.require("pid")?,
                pin.require("pid")?,
                "resident release PID differs",
            )?;
        }
        _ => return Err(Error::contract("resident release evidence kind invalid")),
    }
    Ok(())
}
pub(crate) fn cache_references(
    s: &mut Sources,
    state: &Value,
    records: &[Document],
    model: &Value,
    registry: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<BTreeSet<usize>> {
    let refs = registry.require("cache_references")?.object()?;
    let mut protected = BTreeSet::new();
    for (name, pin) in refs {
        if field(pin, "token")? != name {
            return Err(Error::contract("resident cache token differs"));
        }
        protected.insert(binding(s, state, records, model, pin, b, c)?);
    }
    let root = s.root.join("cache-reference-receipts");
    if !root.exists() && !root.is_symlink() {
        if !refs.is_empty() {
            return Err(Error::contract("resident cache pin lacks original journal"));
        }
        s.absent_control(&root)?;
        return Ok(protected);
    }
    let (names, _names_charge) = s.child_names(&root)?;
    if names.iter().any(|(_, dir)| !dir) {
        return Err(Error::contract(
            "resident cache journal requires token directories",
        ));
    }
    s.inventory(
        &root,
        names.iter().map(|(name, _)| name.clone()).collect(),
        true,
    )?;
    let mut tokens = BTreeSet::new();
    for (name, _) in names {
        c.check()?;
        token(&json::text(&name))?;
        tokens.insert(name.clone());
        let directory = root.join(&name);
        let (files, _files_charge) = s.child_names(&directory)?;
        if files.iter().any(|(name, dir)| {
            *dir || ![
                "acquire.intent.json",
                "acquire.receipt.json",
                "release.intent.json",
                "release.receipt.json",
            ]
            .contains(&name.as_str())
        }) {
            return Err(Error::contract("foreign resident cache journal entry"));
        }
        let files: BTreeSet<_> = files.into_iter().map(|(name, _)| name).collect();
        s.inventory(&directory, files.iter().cloned().collect(), false)?;
        if !files.contains("acquire.intent.json")
            || files.contains("release.intent.json") && !files.contains("acquire.receipt.json")
        {
            return Err(Error::contract(
                "resident cache transaction lacks original acquisition",
            ));
        }
        let mut original: Option<Value> = None;
        let mut number = 0;
        for phase in ["acquire", "release"] {
            let path = directory.join(format!("{phase}.intent.json"));
            if !files.contains(&format!("{phase}.intent.json")) {
                if files.contains(&format!("{phase}.receipt.json")) {
                    return Err(Error::contract(
                        "resident completion has no original intent",
                    ));
                }
                continue;
            }
            let intent = s.json(&path)?;
            let mut fields = vec!["schema_version", "kind", "operation_id", "binding"];
            if phase == "release" {
                fields.extend(["acquire_receipt_sha256", "release_evidence"]);
            }
            intent.root.closed(&fields, &[])?;
            n(&intent.root, "schema_version", 1, 1)?;
            is(
                &intent.root,
                "kind",
                &format!("hpo-cache-reference-{phase}-intent"),
            )?;
            crate::journal::uuid(intent.root.require("operation_id")?.string()?)?;
            let pin = intent.root.require("binding")?;
            number = binding(s, state, records, model, pin, b, c)?;
            is(pin, "token", &name)?;
            if let Some(original) = &original {
                eq(pin, original, "resident journal pin binding changed")?;
            } else {
                original = Some(pin.clone());
            }
            if phase == "release" {
                is(
                    &intent.root,
                    "acquire_receipt_sha256",
                    s.verified_sha(&directory.join("acquire.receipt.json"))?,
                )?;
                release(intent.root.require("release_evidence")?, pin, b, c)?;
            }
            if files.contains(&format!("{phase}.receipt.json")) {
                let receipt = s.json(&directory.join(format!("{phase}.receipt.json")))?;
                receipt.root.closed(
                    &[
                        "schema_version",
                        "kind",
                        "binding",
                        "intent_sha256",
                        "registry_sha256",
                        "confirmed_unix",
                    ],
                    &[],
                )?;
                n(&receipt.root, "schema_version", 1, 1)?;
                is(
                    &receipt.root,
                    "kind",
                    &format!("hpo-cache-reference-{phase}-receipt"),
                )?;
                is(&receipt.root, "intent_sha256", s.verified_sha(&path)?)?;
                eq(
                    receipt.root.require("binding")?,
                    pin,
                    "resident receipt pin differs",
                )?;
                checksum(receipt.root.require("registry_sha256")?)?;
                let time = receipt.root.require("confirmed_unix")?.number()?;
                if !time.is_finite() || time <= 0. {
                    return Err(Error::contract("resident receipt timestamp invalid"));
                }
            }
        }
        let pin =
            original.ok_or_else(|| Error::contract("resident journal original binding absent"))?;
        let active = refs.get(&name);
        if let Some(active) = active {
            eq(
                active,
                &pin,
                "resident active pin differs from original intent",
            )?;
        }
        let released = files.contains("release.receipt.json");
        if released && active.is_some()
            || files.contains("acquire.receipt.json")
                && !files.contains("release.intent.json")
                && active.is_none()
        {
            return Err(Error::contract(
                "resident cache reference vanished or resurrected without original transaction",
            ));
        }
        if !released {
            protected.insert(number);
        }
    }
    if refs.keys().any(|key| !tokens.contains(key)) {
        return Err(Error::contract(
            "resident active pin lacks original journal",
        ));
    }
    Ok(protected)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn retirement_fixture(b: &Arc<Budget>, c: &Control) -> Value {
        let mut value = serde_json::json!({
            "schema_version":1, "kind":"hpo-resident-logical-retirement", "run_id":"run-a",
            "collection_id":"batch-000000", "job_ticket":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "pool_id":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "pool_epoch":1, "generation":0, "owners":[]
        });
        for index in 0..8 {
            let mut row = serde_json::json!({"job_ticket":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "pool_id":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "pool_epoch":1,
                "worker_key":index+1,"pid":1000+index,"viewer":1+index%2,"lease_epoch":1});
            let mut a = row.clone();
            for (key, item) in serde_json::json!({"schema_version":1,"protocol":"resident-worker-v1",
                "run_id":"run-a","generation":0,"checkpoint_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                "cache_identity":null,"clean_reset":true,"private_state_empty":true,"pending_io_empty":true,
                "cuda_synchronized":true,"reset_version":"resident-reset-v1"}).as_object().unwrap() {
                a[key] = item.clone();
            }
            a["command"] = json::text("END"); a["request_epoch"] = json::integer(1);
            row["end_ack_sha256"] = json::text(&digest(&a, &b, &c).unwrap()); row["end_ack"] = a.clone();
            a["command"] = json::text("PARK"); a["request_epoch"] = json::integer(2);
            row["park_ack_sha256"] = json::text(&digest(&a, &b, &c).unwrap()); row["park_ack"] = a;
            value["owners"].as_array_mut().unwrap().push(row);
        }
        value
    }

    #[test]
    fn configurable_retirement_checks_original_capacity_and_all_owner_acks() {
        let b = Budget::new(8 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let value = retirement_fixture(&b, &c);
        retirement(&value, 8, &b, &c).unwrap();
        retirement(&value, 9, &b, &c).unwrap();
        for capacity in [0, 4, 7, i64::MAX as i128 + 1] {
            assert!(retirement(&value, capacity, &b, &c).is_err());
        }
        let mut broken = value.clone();
        broken["owners"][7] = broken["owners"][0].clone();
        assert!(retirement(&broken, 8, &b, &c).is_err());
        broken = value.clone();
        broken["owners"][7]["viewer"] = json::integer(1);
        for name in ["end_ack", "park_ack"] {
            broken["owners"][7][name]["viewer"] = json::integer(1);
            broken["owners"][7][format!("{name}_sha256")] = json::text(&digest(&broken["owners"][7][name], &b, &c).unwrap());
        }
        assert!(retirement(&broken, 9, &b, &c).is_err());
        broken = value.clone();
        broken["owners"][7]["park_ack"]["request_epoch"] = json::integer(1);
        broken["owners"][7]["park_ack_sha256"] = json::text(&digest(&broken["owners"][7]["park_ack"], &b, &c).unwrap());
        assert!(retirement(&broken, 8, &b, &c).is_err());
        broken = value.clone();
        broken["owners"][7]["end_ack"]["private_state_empty"] = Value::Bool(false);
        broken["owners"][7]["end_ack_sha256"] = json::text(&digest(&broken["owners"][7]["end_ack"], &b, &c).unwrap());
        assert!(retirement(&broken, 8, &b, &c).is_err());
    }

    #[test]
    #[ignore = "explicit bounded unqualified CPU originals; GFB_RESIDENT_NATIVE_FIXTURES required"]
    fn source6_reads_original_registry5_pins_and_collection7_proof4() {
        let root = std::path::PathBuf::from(
            std::env::var("GFB_RESIDENT_NATIVE_FIXTURES").expect("explicit fixture root required"),
        );
        for name in ["absent", "acquired", "pending", "released", "live"] {
            let directory = root.join(name);
            let b = Budget::new(1usize << 30);
            let c = Control::new(crate::budget::monotonic() + 60.).unwrap();
            let live = name == "live";
            let mut plan = json::object([
                ("schema_version", json::integer(6)),
                (
                    "mode",
                    json::text(if live {
                        "live_cohort"
                    } else {
                        "historical_parent"
                    }),
                ),
                (
                    "parents",
                    json::array([json::object([
                        ("directory", json::text(directory.to_str().unwrap())),
                        ("require_complete", Value::Bool(false)),
                    ])]),
                ),
                (
                    "collections",
                    json::array(if live {
                        vec![json::object([
                            (
                                "directory",
                                json::text(
                                    directory.join("collections/batch-000000").to_str().unwrap(),
                                ),
                            ),
                            ("parent_index", json::integer(0)),
                        ])]
                    } else {
                        vec![]
                    }),
                ),
            ]);
            if live {
                plan["cohort_collection_index"] = json::integer(0);
            }
            let encoded = json::canonical(&plan, &b, &c).unwrap();
            let opened = crate::structured_source::unadmitted_hpo_component(|| {
                crate::structured_source::OriginalInputs::open5(
                    &encoded,
                    None,
                    b.clone(),
                    c.clone(),
                    64 << 20,
                    8 << 20,
                    1 << 30,
                    64 << 20,
                )
            });
            let mut inputs =
                opened.unwrap_or_else(|error| panic!("resident fixture {name}: {error:?}"));
            assert!(inputs.format.resident());
            let pending = inputs.controls().unwrap();
            assert_eq!(
                pending.summary.root["parent"]["registry"]["schema_version"],
                json::integer(5)
            );
            if live {
                let raw = &pending.summary.root["collections"][0];
                retirement(&raw["logical_retirement"], 4, &b, &c).unwrap();
                let selection =
                    json::parse(&std::fs::read(root.join("selection.json")).unwrap(), &b, &c)
                        .unwrap();
                let mut control = raw["control"].clone();
                control["selection"] = selection.root.clone();
                let mut correspondence = json::object([
                    (
                        "recipe",
                        json::text("card-instance-hpo-source-correspondence-v1"),
                    ),
                    (
                        "control_fingerprint",
                        json::text(&digest(&control, &b, &c).unwrap()),
                    ),
                    (
                        "semantics_sha256",
                        json::text(&digest(&raw["semantics"], &b, &c).unwrap()),
                    ),
                    (
                        "episode_count",
                        json::integer(raw["episode_digests"].array().unwrap().len() as i128),
                    ),
                    ("episode_digests", raw["episode_digests"].clone()),
                ]);
                correspondence["binding_sha256"] =
                    json::text(&digest(&correspondence, &b, &c).unwrap());
                let recipe = &inputs.parents[0].state.root["effective_config"]["hpo_spec"]["training_recipe"];
                let request = json::object([
                    ("schema_version", json::integer(5)),
                    ("pending_sha256", json::text(&pending.fingerprint)),
                    ("parent", pending.authorization.root["parent"].clone()),
                    (
                        "collections",
                        json::array([json::object([
                            ("collection_index", json::integer(0)),
                            ("selection", selection.root.clone()),
                            ("training_recipe", recipe.clone()),
                            (
                                "expected_optimizer_step_delta",
                                json::integer(hpo::optimizer_delta(recipe, 6).unwrap()),
                            ),
                            ("expected_correspondence", correspondence),
                        ])]),
                    ),
                ]);
                let (final_controls, _) = inputs.authorize_hpo(&pending, &request).unwrap();
                let proof = &final_controls.summary.root["collections"][0]["source_proof"];
                assert_eq!(proof["schema_version"], json::integer(4));
                assert_eq!(proof["logical_retirement"], raw["logical_retirement"]);
            }
            plan["schema_version"] = json::integer(5);
            assert!(
                crate::structured_source::OriginalInputs::open5(
                    &json::canonical(&plan, &b, &c).unwrap(),
                    None,
                    b.clone(),
                    c.clone(),
                    64 << 20,
                    8 << 20,
                    1 << 30,
                    64 << 20
                )
                .is_err()
            );
        }
    }
}
