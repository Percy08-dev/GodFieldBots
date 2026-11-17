//! HPO metadata only. Original owners remain in the existing Source/session.
//! No sampler, reward, GAE, normalization or numerical optimizer lives here.
use super::*;

pub(crate) const SELECTION_VERSION: &str = "hpo-endpoint-selection-v1";
pub(crate) const CAPABILITY: &str = "model4;collection6;proof3;numeric5;selection1;provider1";
const OPERATING: &[&str] = &[
    "managed_ram_bytes",
    "managed_cuda_bytes",
    "game_concurrency",
    "max_participant_workers",
    "trial_concurrency",
    "source_budget_bytes",
    "numeric_budget_bytes",
    "save_budget_bytes",
    "collector_budget_bytes",
    "parent_budget_bytes",
    "max_updates",
    "inference_latest",
    "inference_periodic_decisions",
    "pool_every_decisions",
    "opponent_pool_size",
    "resume_latest",
];
const STORAGE: &[&str] = &[
    "parent_root",
    "numeric_root",
    "optuna_path",
    "originals_root",
    "windows_c_root",
    "destination_capacity_bytes",
    "admission_reservation_bytes",
    "windows_c_min_free_bytes",
    "ext4_control_required",
];

fn absolute(value: &Value) -> Result<()> {
    let raw = value.string()?;
    if raw.is_empty() || raw.contains('\0') || !Path::new(raw).is_absolute() {
        return Err(Error::contract(
            "HPO original destination must be an absolute path",
        ));
    }
    Ok(())
}

pub(crate) fn spec<'a>(state: &'a Value, b: &Arc<Budget>, c: &Control) -> Result<&'a Value> {
    let value = state.require("effective_config")?.require("hpo_spec")?;
    let mixed = refresh_training(state.require("effective_config")?, b, c)?.is_some();
    let resident = resident::is_resident(state);
    value.closed(
        &[
            "schema_version",
            "kind",
            "decision_budget",
            "training_recipe",
            "effective_settings_sha256",
            "base_config_sha256",
            "campaign_config_sha256",
            "study_identity",
            "operating",
            "storage",
            "selection_version",
            "deadline_version",
            "deadline_inputs",
        ],
        if resident { &["training_opponents", "runtime"] } else if mixed { &["training_opponents"] } else { &[] },
    )?;
    let version = if resident { 3 } else if mixed { 2 } else { 1 };
    n(value, "schema_version", version, version)?;
    if resident { resident::runtime(value.require("runtime")?)?; }
    is(value, "kind", "ppo-hpo-run")?;
    is(value, "selection_version", SELECTION_VERSION)?;
    is(value, "deadline_version", "hpo-deadline-v1")?;
    n(value, "decision_budget", 1, i64::MAX as i128)?;
    training(value.require("training_recipe")?)?;
    for key in [
        "effective_settings_sha256",
        "base_config_sha256",
        "campaign_config_sha256",
        "study_identity",
    ] {
        checksum(value.require(key)?)?;
    }
    let operating = value.require("operating")?;
    operating.closed(OPERATING, &[])?;
    for key in OPERATING {
        n(operating, key, 1, i64::MAX as i128)?;
    }
    let games = n(operating, "game_concurrency", 1, i64::MAX as i128)?;
    let workers = n(operating, "max_participant_workers", 1, i64::MAX as i128)?;
    if workers < 2 * games {
        return Err(Error::contract(
            "HPO simultaneous player worker projection invalid",
        ));
    }
    n(operating, "trial_concurrency", 1, 1)?;
    n(operating, "resume_latest", 2, 1024)?;
    let storage = value.require("storage")?;
    storage.closed(STORAGE, &[])?;
    for key in [
        "parent_root",
        "numeric_root",
        "optuna_path",
        "windows_c_root",
    ] {
        absolute(storage.require(key)?)?;
    }
    if !storage.require("originals_root")?.is_null() {
        absolute(storage.require("originals_root")?)?;
    }
    let capacity = n(storage, "destination_capacity_bytes", 1, i64::MAX as i128)?;
    n(storage, "admission_reservation_bytes", 1, capacity)?;
    n(storage, "windows_c_min_free_bytes", 1, i64::MAX as i128)?;
    eq(
        storage.require("ext4_control_required")?,
        &Value::Bool(true),
        "HPO ext4 control requirement changed",
    )?;
    let inputs = value.require("deadline_inputs")?;
    let timers = [
        "run_wall_seconds",
        "base_wall_seconds",
        "base_admission_seconds",
        "base_execution_seconds",
        "base_finalization_seconds",
    ];
    inputs.closed(&timers, &[])?;
    for key in timers {
        let t = finite(inputs, key)?;
        if t <= 0. || t > 1e20 {
            return Err(Error::contract("HPO deadline input invalid"));
        }
    }
    let wall = finite(inputs, "base_wall_seconds")?;
    let admission = finite(inputs, "base_admission_seconds")?;
    let execution = finite(inputs, "base_execution_seconds")?;
    let finalization = finite(inputs, "base_finalization_seconds")?;
    if !(admission < execution && execution < wall)
        || execution + finalization > wall + 0.001
        || finite(inputs, "run_wall_seconds")? <= finalization
    {
        return Err(Error::contract(
            "HPO original deadline projection inputs inconsistent",
        ));
    }
    is(state, "hpo_spec_sha256", &digest(value, b, c)?)?;
    Ok(value)
}

pub(crate) fn run(state: &Value, b: &Arc<Budget>, c: &Control) -> Result<Value> {
    let _copies = json::admit(json::canonical(state, b, c)?.len(), b)?;
    let packed = state.get("storage_binding").is_some();
    let mut keys: Vec<&str> = "schema_version algorithm run_id status config_sha256 effective_config config_inputs identity seed device created_unix last_unix wall_deadline_unix admission_deadline_unix execution_deadline_unix updates accepted_decisions optimizer_steps generation_sha256 collection_seconds update_seconds invocations completed_updates model_id model_version input_recipe action_contract purpose synthetic_capacity resume_identity execution rule_signature layout_signature run_variant hpo_spec_sha256 collected_decisions used_decisions stop_reason".split_whitespace().collect();
    if packed {
        keys.push("storage_binding");
    }
    state.closed(
        &keys,
        &[
            "hardware",
            "elapsed_seconds",
            "artifact_bytes",
            "peak_coordinator_memory_bytes",
            "coordinator_memory_scope",
            "timing_scope",
            "failure",
            "quarantine",
            "cleanup_failure",
            "budget_overrun_seconds",
        ],
    )?;
    n(state, "schema_version", 5, 5)?;
    is(state, "algorithm", "ppo")?;
    if !resident::is_resident(state) { is(state, "run_variant", "hpo-ppo-v1")?; }
    is(state, "model_version", "4")?;
    is(state, "input_recipe", "participant-structured-v2")?;
    validation_text(state.require("run_id")?)?;
    for key in [
        "failure",
        "quarantine",
        "cleanup_failure",
        "budget_overrun_seconds",
    ] {
        if state.get(key).is_some_and(crate::controls::truthy) {
            return Err(Error::contract("HPO run contains failure evidence"));
        }
    }
    let cfg = state.require("effective_config")?;
    cfg.closed(
        &[
            "schema_version",
            "kind",
            "run",
            "server",
            "actions",
            "model",
            "opponent",
            "evaluation",
            "training",
            "preprocessing",
            "hpo_spec",
        ],
        if packed { &["storage"] } else { &[] },
    )?;
    n(cfg, "schema_version", 1, 1)?;
    is(cfg, "kind", "experiment")?;
    is(state, "config_sha256", &digest(cfg, b, c)?)?;
    if packed {
        eq(
            state.require("storage_binding")?,
            &crate::logical_artifacts::binding(),
            "HPO packed storage binding differs",
        )?;
        let storage = cfg.require("storage")?;
        storage.closed(
            &[
                "format",
                "db_root",
                "ram_root",
                "budget_mib",
                "queue_mib",
                "max_pack_mib",
            ],
            &[],
        )?;
        is(storage, "format", "sqlite-group-pack-v1")?;
        for key in ["db_root", "ram_root"] {
            absolute(storage.require(key)?)?;
        }
        let budget = n(storage, "budget_mib", 193, i64::MAX as i128)?;
        n(storage, "queue_mib", 1, budget)?;
        n(storage, "max_pack_mib", 1, 256)?;
    }
    let specification = spec(state, b, c)?;
    let recipe = specification.require("training_recipe")?;
    let operating = specification.require("operating")?;
    let storage = specification.require("storage")?;
    if !storage.require("originals_root")?.is_null()
        && (!packed
            || storage.require("originals_root")? != cfg.require("storage")?.require("db_root")?)
    {
        return Err(Error::contract(
            "HPO original storage projection differs from sole spec",
        ));
    }
    let model = pure_model(cfg, c)?;
    model4(&model, b, c)?;
    n(&model, "max_positions", 1, 18)?;
    eq(
        model.require("model_id")?,
        state.require("model_id")?,
        "HPO model kind mismatch",
    )?;
    let mut effective_recipe = json::copy(recipe, c)?;
    if text(&model, "model_id")? == "mlp" {
        effective_recipe["tbptt_length"] = json::integer(32);
    }
    is(
        specification,
        "effective_settings_sha256",
        &digest(
            &json::object([
                ("model", json::text(&architecture(&model, b, c)?)),
                ("training", effective_recipe),
            ]),
            b,
            c,
        )?,
    )?;
    actions3(cfg.require("actions")?, b, c)?;
    game_binding3(state, b, c)?;
    let train = cfg.require("training")?;
    train.closed(
        &[
            "algorithm",
            "device",
            "seeds",
            "max_updates",
            "max_owned_bytes",
            "admission_reservation_bytes",
            "cleanup_seconds",
            "max_log_bytes",
        ],
        &[],
    )?;
    is(train, "algorithm", "ppo")?;
    for (key, reference) in [
        ("max_updates", operating.require("max_updates")?),
        (
            "max_owned_bytes",
            storage.require("destination_capacity_bytes")?,
        ),
        (
            "admission_reservation_bytes",
            storage.require("admission_reservation_bytes")?,
        ),
    ] {
        n(train, key, 1, i64::MAX as i128)?;
        eq(
            train.require(key)?,
            reference,
            "HPO runtime training projection differs",
        )?;
    }
    n(train, "max_log_bytes", 1, i32::MAX as i128)?;
    let seed = n(state, "seed", 0, i32::MAX as i128)?;
    if train.require("seeds")?.array()?.len() != 1
        || train.require("seeds")?.array()?[0].integer(seed, seed)? != seed
    {
        return Err(Error::contract("HPO one learning seed projection mismatch"));
    }
    eq(
        train.require("device")?,
        state.require("device")?,
        "HPO learner device mismatch",
    )?;
    if !["cpu", "cuda:0"].contains(&text(state, "device")?) {
        return Err(Error::contract("HPO learner device unsupported"));
    }
    let settings = cfg.require("preprocessing")?;
    settings.closed(
        &[
            "memory_budget_mib",
            "cache_budget_mib",
            "chunk_target_mib",
            "max_source_file_mib",
            "max_record_mib",
            "max_in_flight",
            "workers",
            "backend",
            "reader",
            "scratch_budget_mib",
            "wall_seconds",
            "cleanup_seconds",
        ],
        &[],
    )?;
    is(settings, "reader", "participant-structured-v2")?;
    let memory = n(settings, "memory_budget_mib", 1, i64::MAX as i128)?;
    if memory * (1 << 20) > n(operating, "source_budget_bytes", 1, i64::MAX as i128)? {
        return Err(Error::contract(
            "HPO source reader budget projection mismatch",
        ));
    }
    n(settings, "cache_budget_mib", 0, memory)?;
    n(settings, "chunk_target_mib", 1, memory)?;
    n(
        settings,
        "max_record_mib",
        1,
        n(settings, "max_source_file_mib", 1, i64::MAX as i128)?,
    )?;
    n(settings, "scratch_budget_mib", 1, i64::MAX as i128)?;
    let workers = n(settings, "workers", 1, 8)?;
    let in_flight = n(settings, "max_in_flight", 1, i64::MAX as i128)?;
    let source_wall = finite(settings, "wall_seconds")?;
    let source_cleanup = finite(settings, "cleanup_seconds")?;
    if source_cleanup <= 0.
        || source_cleanup >= source_wall
        || text(settings, "backend")? == "python" && (workers != 1 || in_flight != 1)
        || text(settings, "backend")? == "rust" && ![1, 2, 4, 8].contains(&workers)
    {
        return Err(Error::contract(
            "HPO original preprocessing limits inconsistent",
        ));
    }
    validate_preprocessing3(state, b, c)?;
    let integration = state.require("identity")?.require("integration")?;
    integration.closed(
        &[
            "binary_sha256",
            "server_config_sha256",
            "declared_simulator_revision",
            "source_revision",
            "source_spec_sha256",
        ],
        &["binary_build_revision"],
    )?;
    let server = cfg.require("server")?;
    server.object()?;
    if server
        .get("external_url")
        .is_some_and(|value| !value.is_null())
    {
        return Err(Error::contract(
            "HPO original requires locally owned Simulator runtime",
        ));
    }
    eq(
        server.require("simulator_revision")?,
        state
            .require("action_contract")?
            .require("simulator_revision")?,
        "HPO original server source revision differs",
    )?;
    if text(settings, "backend")? == "rust" {
        is(
            state
                .require("identity")?
                .require("preprocessing")?
                .require("binary")?
                .require("build")?,
            "card_instance_hpo",
            CAPABILITY,
        )?;
    }
    let run = cfg.require("run")?;
    run.closed(
        &[
            "name",
            "split",
            "output_dir",
            "seeds",
            "sampling_seeds",
            "pairs_per_seed",
            "concurrency",
            "wall_seconds",
            "admission_seconds",
            "execution_seconds",
            "finalization_seconds",
            "report_reserve_seconds",
            "max_decisions",
            "global_cuda_budget_bytes",
            "max_participant_workers",
        ],
        &[],
    )?;
    is(run, "split", "train")?;
    for (key, reference) in [
        ("concurrency", "game_concurrency"),
        ("global_cuda_budget_bytes", "managed_cuda_bytes"),
        ("max_participant_workers", "max_participant_workers"),
    ] {
        n(run, key, 1, i64::MAX as i128)?;
        eq(
            run.require(key)?,
            operating.require(reference)?,
            "HPO runtime resource projection mismatch",
        )?;
    }
    let inputs = specification.require("deadline_inputs")?;
    let wall = finite(inputs, "run_wall_seconds")?;
    let finalization = finite(inputs, "base_finalization_seconds")?;
    let execute = wall - finalization;
    let admission = execute
        * (finite(inputs, "base_admission_seconds")? / finite(inputs, "base_execution_seconds")?);
    for (key, expected) in [
        ("wall_seconds", wall),
        ("execution_seconds", execute),
        ("admission_seconds", admission),
        ("finalization_seconds", finalization),
    ] {
        if (finite(run, key)? - expected).abs() > 0.001 {
            return Err(Error::contract("HPO derived deadline duration mismatch"));
        }
    }
    let cleanup = finite(train, "cleanup_seconds")?;
    let report = finite(run, "report_reserve_seconds")?;
    if cleanup <= 0. || cleanup > 60. || report <= 0. || cleanup + report > finalization {
        return Err(Error::contract(
            "HPO original cleanup/report reserve mismatch",
        ));
    }
    let created = finite(state, "created_unix")?;
    let last = finite(state, "last_unix")?;
    if created <= 0. || !(created <= last && last <= created + wall) {
        return Err(Error::contract("HPO original clock provenance mismatch"));
    }
    for (key, duration) in [
        ("wall_deadline_unix", wall),
        ("execution_deadline_unix", execute),
        ("admission_deadline_unix", admission),
    ] {
        if (finite(state, key)? - created - duration).abs() > 0.001 {
            return Err(Error::contract(
                "HPO original absolute deadline was rewritten",
            ));
        }
    }
    let execution = state.require("execution")?;
    execution.closed(&["learner", "inference"], &[])?;
    let configured = cfg.require("model")?;
    for role in ["learner", "inference"] {
        let actual = execution.require(role)?;
        actual.closed(&["binding", "cuda_visible_devices"], &[])?;
        binding(actual.require("binding")?)?;
        let mut expected = json::copy(configured.require("inference")?, c)?;
        if role == "learner" {
            expected["device"] = train.require("device")?.clone();
        }
        eq(
            actual.require("binding")?,
            &expected,
            "HPO exact execution binding differs",
        )?;
        visibility(configured.require("cuda_visible_devices")?)?;
        let expected_visibility = if text(&expected, "device")? == "cpu" {
            json::text("")
        } else {
            configured.require("cuda_visible_devices")?.clone()
        };
        eq(
            actual.require("cuda_visible_devices")?,
            &expected_visibility,
            "HPO execution visibility mismatch",
        )?;
    }
    let expected_resume = json::object([
        ("version", json::text("hpo-ppo-resume-v1")),
        ("config_sha256", state.require("config_sha256")?.clone()),
        ("identity", state.require("identity")?.clone()),
        ("execution", execution.clone()),
        ("classification", clone_fields(state, GAME_CLASS, c)?),
        (
            "storage_binding",
            state.get("storage_binding").cloned().unwrap_or(Value::Null),
        ),
    ]);
    is(state, "resume_identity", &digest(&expected_resume, b, c)?)?;
    let updates = n(
        state,
        "updates",
        0,
        n(train, "max_updates", 1, i64::MAX as i128)?,
    )?;
    let used = n(
        state,
        "used_decisions",
        0,
        n(specification, "decision_budget", 1, i64::MAX as i128)?,
    )?;
    n(state, "accepted_decisions", used, used)?;
    n(state, "collected_decisions", used, i64::MAX as i128)?;
    n(state, "optimizer_steps", 0, i32::MAX as i128)?;
    finite(state, "collection_seconds")?;
    finite(state, "update_seconds")?;
    let status = text(state, "status")?;
    let reason = state.require("stop_reason")?;
    let completed = state.require("completed_updates")?;
    if !["running", "paused", "incomplete", "complete"].contains(&status)
        || status == "running" && !reason.is_null()
        || status == "paused" && (updates == 0 || reason.as_str() != Some("pause"))
        || status == "incomplete"
            && (!matches!(
                reason.as_str(),
                Some("update_cap" | "time_cap" | "storage_cap" | "empty_collection")
            ) || used == n(specification, "decision_budget", 1, i64::MAX as i128)?)
        || status == "complete"
            && (updates == 0
                || !reason.is_null()
                || used != n(specification, "decision_budget", 1, i64::MAX as i128)?
                || completed.integer(updates, updates)? != updates)
        || status != "complete" && !completed.is_null()
    {
        return Err(Error::contract(
            "HPO status/completion/decision budget mismatch",
        ));
    }
    let invocations = state.require("invocations")?.array()?;
    if status != "running" {
        if invocations.is_empty()
            || (finite(state, "elapsed_seconds")? - (last - created)).abs() > 0.001
        {
            return Err(Error::contract(
                "HPO closed run elapsed/cleanup history mismatch",
            ));
        }
        n(
            state,
            "artifact_bytes",
            0,
            n(train, "max_owned_bytes", 1, i64::MAX as i128)?,
        )?;
    }
    for invocation in invocations {
        invocation.closed(&["mode", "elapsed_seconds", "child_health"], &[])?;
        if !["fresh", "resume"].contains(&text(invocation, "mode")?) {
            return Err(Error::contract("HPO invocation mode invalid"));
        }
        finite(invocation, "elapsed_seconds")?;
        let health = invocation.require("child_health")?;
        health.closed(&["exit_code", "stderr_bytes"], &[])?;
        n(health, "exit_code", 0, 0)?;
        n(health, "stderr_bytes", 0, i64::MAX as i128)?;
    }
    let hardware = state.require("hardware")?;
    hardware.closed(
        &[
            "device",
            "name",
            "cuda_runtime",
            "device_count",
            "cuda_driver_api_version",
            "exact_driver_build",
        ],
        &[],
    )?;
    eq(
        hardware.require("device")?,
        state.require("device")?,
        "HPO recorded hardware device mismatch",
    )?;
    validation_text(hardware.require("name")?)?;
    validation_text(hardware.require("exact_driver_build")?)?;
    let devices = n(hardware, "device_count", 0, 1024)?;
    if text(state, "device")? == "cpu" {
        if devices != 0
            || text(hardware, "name")? != "cpu"
            || !hardware.require("cuda_driver_api_version")?.is_null()
        {
            return Err(Error::contract("HPO CPU hardware observation inconsistent"));
        }
    } else {
        if devices == 0 {
            return Err(Error::contract("HPO CUDA hardware evidence missing"));
        }
        validation_text(hardware.require("cuda_runtime")?)?;
        n(hardware, "cuda_driver_api_version", 1, i64::MAX as i128)?;
    }
    Ok(model)
}

pub(crate) fn architecture(model: &Value, b: &Arc<Budget>, c: &Control) -> Result<String> {
    model4(model, b, c)?;
    let mut pure = json::copy(model, c)?;
    pure.as_object_mut().unwrap().remove("resources");
    digest(
        &json::object([
            ("model", pure),
            (
                "token_signature",
                json::text(&card_instance_token_signature(
                    n(model.require("event")?, "aggregate_window", 1, 65536)? as usize,
                    b,
                    c,
                )?),
            ),
        ]),
        b,
        c,
    )
}

pub(crate) fn semantic_keys() -> Vec<&'static str> {
    SEMANTICS
        .iter()
        .copied()
        .filter(|key| *key != "reward_policy_version")
        .chain(["rule_signature", "spec_inventory_sha256"])
        .collect()
}

pub(crate) fn inference(
    value: &Value,
    actions: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<()> {
    let encoded = json::canonical(value, b, c)?;
    let _metadata = json::admit(encoded.len(), b)?;
    actions3(actions, b, c)?;
    let mut keys = vec![
        "schema_version",
        "kind",
        "model",
        "architecture_identity",
        "action_config_hash",
        "producer",
        "weights",
    ];
    keys.extend(semantic_keys());
    keys.extend(GAME_CLASS);
    value.closed(&keys, &[])?;
    n(value, "schema_version", 5, 5)?;
    is(value, "kind", "structured-inference")?;
    game_binding3(value, b, c)?;
    producer(value.require("producer")?)?;
    let mut model = json::copy(value.require("model")?, c)?;
    if model.get("resources").is_some() {
        return Err(Error::contract(
            "inference5 contains pure architecture only",
        ));
    }
    // Identical technical shape-validation aids to the bounded Python recorded
    // inference parser. These values grant no resource/allocation permission.
    model.as_object_mut().unwrap().insert(
        "resources".into(),
        json::object([
            ("max_parameters", json::integer(100_000_000)),
            ("max_owned_bytes", json::integer(12i128 << 30)),
            ("max_tensor_bytes", json::integer(64i128 << 20)),
            ("cuda_context_bytes", json::integer(512i128 << 20)),
            ("workspace_bytes", json::integer(256i128 << 20)),
        ]),
    );
    let shapes = model4(&model, b, c)?;
    n(&model, "max_positions", 1, 18)?;
    semantic_corrected(value, &model, b, c, true)?;
    is(value, "architecture_identity", &architecture(&model, b, c)?)?;
    is(value, "action_config_hash", &digest(actions, b, c)?)?;
    let descriptors = bundle(value.require("weights")?)?;
    is(value.require("weights")?, "prefix", "weights")?;
    if descriptors.len() != shapes.len() {
        return Err(Error::contract(
            "inference5 exact named weight inventory differs",
        ));
    }
    for (name, shape) in shapes {
        let descriptor = descriptors
            .get(&name)
            .ok_or_else(|| Error::contract("inference5 named weight missing"))?;
        is(descriptor, "dtype", "float32")?;
        eq(
            descriptor.require("shape")?,
            &json::array(shape.into_iter().map(json::integer)),
            "inference5 named weight shape mismatch",
        )?;
    }
    Ok(())
}

pub(crate) fn options(
    value: &Value,
    actions: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<String> {
    options_context(value, actions, b, c, true)
}

pub(crate) fn neural_metadata(
    value: &Value,
    actions: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<String> {
    neural_metadata_context(value, actions, b, c, true)
}

pub(crate) fn training(value: &Value) -> Result<()> {
    value.closed(
        &[
            "version",
            "algorithm",
            "optimizer",
            "reward",
            "gamma",
            "collection_games",
            "adam_learning_rate",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "epochs",
            "minibatch_size",
            "clip_range",
            "value_coefficient",
            "entropy_coefficient",
            "gae_lambda",
            "max_grad_norm",
            "tbptt_length",
        ],
        &[],
    )?;
    for (key, expected) in [
        ("version", "hpo-ppo-v1"),
        ("algorithm", "ppo"),
        ("optimizer", "adam"),
        ("reward", "natural-terminal-v1"),
    ] {
        is(value, key, expected)?;
    }
    if finite(value, "gamma")? != 1. {
        return Err(Error::contract("HPO gamma is fixed at one"));
    }
    let games = n(value, "collection_games", 8, 65536)?;
    if games % 8 != 0 {
        return Err(Error::contract(
            "HPO whole seat-paired collection must be a multiple of eight",
        ));
    }
    n(value, "epochs", 1, 1000)?;
    n(value, "minibatch_size", 1, 1 << 20)?;
    n(value, "tbptt_length", 1, 1024)?;
    for (name, lo, hi, strict_lo, strict_hi) in [
        ("adam_learning_rate", 0., 1., true, false),
        ("adam_beta1", 0., 1., false, true),
        ("adam_beta2", 0., 1., false, true),
        ("adam_epsilon", 0., 1., true, false),
        ("clip_range", 0., 1., true, true),
        ("value_coefficient", 0., 1e6, false, false),
        ("entropy_coefficient", 0., 1e6, false, false),
        ("gae_lambda", 0., 1., false, false),
        ("max_grad_norm", 0., 1e6, true, false),
    ] {
        let actual = finite(value, name)?;
        if actual < lo || actual > hi || strict_lo && actual == lo || strict_hi && actual == hi {
            return Err(Error::contract("HPO finite training recipe range mismatch"));
        }
    }
    Ok(())
}

pub(crate) fn optimizer_delta(recipe: &Value, used: usize) -> Result<i128> {
    training(recipe)?;
    let epochs = n(recipe, "epochs", 1, 1000)?;
    let batch = n(recipe, "minibatch_size", 1, 1 << 20)?;
    Ok(epochs * ((used as i128 + batch - 1) / batch))
}

pub(crate) fn file_receipt<'a>(value: &'a Value, expected: &str) -> Result<&'a str> {
    value.closed(
        &["sha256", "status", "deletion_intent", "deletion_receipt"],
        &[],
    )?;
    if checksum(value.require("sha256")?)? != expected {
        return Err(Error::contract("HPO per-file original SHA differs"));
    }
    let status = text(value, "status")?;
    if status == "retained" {
        if !value.require("deletion_intent")?.is_null()
            || !value.require("deletion_receipt")?.is_null()
        {
            return Err(Error::contract(
                "HPO retained file cannot claim deletion evidence",
            ));
        }
        return Ok(status);
    }
    if !["deleting", "deleted", "failed"].contains(&status) {
        return Err(Error::contract("unknown HPO per-file retention status"));
    }
    let intent = value.require("deletion_intent")?;
    intent.closed(
        &[
            "operation_id",
            "staging_basename",
            "sha256",
            "original_identity",
        ],
        &[],
    )?;
    let operation = text(intent, "operation_id")?;
    if operation.is_empty()
        || operation.len() > 128
        || !operation.is_ascii()
        || !operation.as_bytes()[0].is_ascii_alphanumeric()
        || !operation
            .bytes()
            .all(|v| v.is_ascii_alphanumeric() || b"_.-".contains(&v))
    {
        return Err(Error::contract(
            "HPO owned deletion operation identity invalid",
        ));
    }
    let staging = text(intent, "staging_basename")?;
    if staging.len() != 38
        || !staging.starts_with("prune-")
        || !staging[6..]
            .bytes()
            .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
    {
        return Err(Error::contract(
            "HPO owned deletion staging basename invalid",
        ));
    }
    if checksum(intent.require("sha256")?)? != expected {
        return Err(Error::contract("HPO owned deletion intent SHA mismatch"));
    }
    let identity = intent.require("original_identity")?.array()?;
    if identity.len() != 5 {
        return Err(Error::contract(
            "HPO original deletion identity requires exact stat5",
        ));
    }
    for part in identity {
        part.integer(0, i128::MAX)?;
    }
    let receipt = value.require("deletion_receipt")?;
    if status == "deleted" {
        receipt.closed(
            &[
                "operation_id",
                "sha256",
                "original_identity",
                "completed_unix",
            ],
            &[],
        )?;
        for key in ["operation_id", "sha256", "original_identity"] {
            eq(
                receipt.require(key)?,
                intent.require(key)?,
                "HPO completed receipt differs from original intent",
            )?;
        }
        let completed = finite(receipt, "completed_unix")?;
        if completed <= 0. || completed > 1e20 {
            return Err(Error::contract("HPO completed deletion time invalid"));
        }
    } else if !receipt.is_null() {
        return Err(Error::contract(
            "HPO uncertain deletion cannot claim completion",
        ));
    }
    Ok(status)
}

pub(crate) fn generation(
    value: &Value,
    state: &Value,
    number: usize,
    previous: Option<&Value>,
    expected_parent: Option<&str>,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<Vec<usize>> {
    value.closed(
        &[
            "schema_version",
            "run_id",
            "generation",
            "parent_sha256",
            "model_id",
            "model_version",
            "input_recipe",
            "action_contract",
            "purpose",
            "synthetic_capacity",
            "inference_sha256",
            "resume_sha256",
            "artifact_files",
            "accepted_decisions",
            "optimizer_steps",
            "update_sha256",
            "parent_controls",
            "rule_signature",
            "layout_signature",
            "hpo_spec_sha256",
            "collected_decisions",
            "used_decisions",
        ],
        &[],
    )?;
    n(value, "schema_version", 4, 4)?;
    n(value, "generation", number as i128, number as i128)?;
    for key in [
        "run_id",
        "model_id",
        "model_version",
        "input_recipe",
        "hpo_spec_sha256",
    ] {
        eq(
            value.require(key)?,
            state.require(key)?,
            "HPO generation immutable identity differs",
        )?;
    }
    for key in GAME_CLASS {
        eq(
            value.require(key)?,
            state.require(key)?,
            "HPO generation public input binding differs",
        )?;
    }
    match expected_parent {
        None if value.require("parent_sha256")?.is_null() => {}
        Some(sha) if checksum(value.require("parent_sha256")?)? == sha => {}
        _ => {
            return Err(Error::contract(
                "HPO generation original parent hash mismatch",
            ));
        }
    }
    let used = n(
        value,
        "used_decisions",
        0,
        n(state, "used_decisions", 0, i64::MAX as i128)?,
    )?;
    n(value, "accepted_decisions", used, used)?;
    let collected = n(
        value,
        "collected_decisions",
        used,
        n(state, "collected_decisions", 0, i64::MAX as i128)?,
    )?;
    let steps = n(
        value,
        "optimizer_steps",
        0,
        n(state, "optimizer_steps", 0, i32::MAX as i128)?,
    )?;
    let configuration = spec(state, b, c)?;
    let operating = configuration.require("operating")?;
    let mut pool = if number == 0 {
        if used != 0
            || collected != 0
            || steps != 0
            || !value.require("update_sha256")?.is_null()
            || previous.is_some()
        {
            return Err(Error::contract(
                "HPO generation0 must be an untrained fresh state",
            ));
        }
        vec![0]
    } else {
        checksum(value.require("update_sha256")?)?;
        let previous =
            previous.ok_or_else(|| Error::contract("HPO generation previous state missing"))?;
        if used <= n(previous, "used_decisions", 0, i64::MAX as i128)? {
            return Err(Error::contract(
                "HPO generation must advance unique used decisions",
            ));
        }
        let prior = previous
            .require("parent_controls")?
            .require("snapshot_pool")?
            .array()?;
        let _scratch = b.reserve(65536 + prior.len() * 32)?;
        let mut pool = prior
            .iter()
            .map(|v| v.integer(0, number as i128).map(|n| n as usize))
            .collect::<Result<Vec<_>>>()?;
        let interval = n(operating, "pool_every_decisions", 1, i64::MAX as i128)?;
        if used / interval > n(previous, "used_decisions", 0, i64::MAX as i128)? / interval {
            pool.push(number);
            let size = n(operating, "opponent_pool_size", 1, i64::MAX as i128)? as usize;
            if pool.len() > size {
                pool.drain(..pool.len() - size);
            }
        }
        pool
    };
    let controls = value.require("parent_controls")?;
    controls.closed(
        &[
            "schedule_opponent_rng",
            "snapshot_pool",
            "collection_position",
        ],
        &[],
    )?;
    n(
        controls,
        "collection_position",
        number as i128,
        number as i128,
    )?;
    pcg(controls.require("schedule_opponent_rng")?)?;
    eq(
        controls.require("snapshot_pool")?,
        &json::array(pool.iter().map(|n| json::integer(*n as i128))),
        "HPO decision-threshold snapshot pool mismatch",
    )?;
    let files = value.require("artifact_files")?;
    filemap(files)?;
    eq(
        files.require("inference.json")?,
        value.require("inference_sha256")?,
        "HPO immutable inference hash/map mismatch",
    )?;
    eq(
        files.require("resume.json")?,
        value.require("resume_sha256")?,
        "HPO immutable resume hash/map mismatch",
    )?;
    let mut names =
        std::collections::BTreeSet::from(["inference.json".to_owned(), "resume.json".to_owned()]);
    for prefix in ["weights", "resume"] {
        let count = files
            .object()?
            .keys()
            .filter(|name| name.starts_with(&format!("{prefix}-")) && name.ends_with(".npz"))
            .count();
        if count == 0 || count > 65536 {
            return Err(Error::contract(
                "HPO physical bundle1 shard inventory invalid",
            ));
        }
        for index in 0..count {
            for ext in ["npz", "json"] {
                names.insert(format!("{prefix}-{index:06}.{ext}"));
            }
        }
    }
    if names.len() != files.object()?.len() || names.iter().any(|name| files.get(name).is_none()) {
        return Err(Error::contract(
            "HPO immutable artifact map contains missing/foreign shard members",
        ));
    }
    c.check()?;
    Ok(std::mem::take(&mut pool))
}

fn optimizer_recipe(recipe: &Value) -> Result<Value> {
    training(recipe)?;
    Ok(json::object([
        ("name", json::text("Adam")),
        ("lr", recipe.require("adam_learning_rate")?.clone()),
        (
            "betas",
            json::array([
                recipe.require("adam_beta1")?.clone(),
                recipe.require("adam_beta2")?.clone(),
            ]),
        ),
        ("eps", recipe.require("adam_epsilon")?.clone()),
        ("weight_decay", json::integer(0)),
        ("amsgrad", Value::Bool(false)),
        ("maximize", Value::Bool(false)),
        ("foreach", Value::Null),
        ("capturable", Value::Bool(false)),
        ("differentiable", Value::Bool(false)),
        ("fused", Value::Null),
        ("decoupled_weight_decay", Value::Bool(false)),
    ]))
}

fn resume_metadata(
    value: &Value,
    inference: &Value,
    state: &Value,
    record: &Value,
    model: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<()> {
    let mut keys = vec![
        "schema_version",
        "kind",
        "inference_digest",
        "model",
        "binding",
        "resume_identity",
        "producer",
        "optimizer_steps",
        "optimizer_recipe",
        "parameter_states",
        "cuda_rng_count",
        "shuffle_rng",
        "state",
        "action_config_hash",
        "training_recipe",
    ];
    keys.extend(semantic_keys());
    keys.extend(GAME_CLASS);
    value.closed(&keys, &[])?;
    n(value, "schema_version", 5, 5)?;
    is(value, "kind", "structured-resume")?;
    game_binding3(value, b, c)?;
    semantic_corrected(value, model, b, c, true)?;
    producer(value.require("producer")?)?;
    for key in semantic_keys() {
        eq(
            value.require(key)?,
            inference.require(key)?,
            "HPO paired artifact semantics differ",
        )?;
    }
    eq(
        value.require("model")?,
        model,
        "HPO resume full model/resources mismatch",
    )?;
    eq(
        value.require("inference_digest")?,
        record.require("inference_sha256")?,
        "HPO resume/inference original link mismatch",
    )?;
    eq(
        value.require("action_config_hash")?,
        inference.require("action_config_hash")?,
        "HPO paired action configuration differs",
    )?;
    eq(
        value.require("binding")?,
        state
            .require("execution")?
            .require("learner")?
            .require("binding")?,
        "HPO saved execution differs",
    )?;
    eq(
        value.require("resume_identity")?,
        state.require("resume_identity")?,
        "HPO saved resume identity mismatch",
    )?;
    eq(
        value.require("optimizer_steps")?,
        record.require("optimizer_steps")?,
        "HPO saved optimizer counter mismatch",
    )?;
    let recipe = spec(state, b, c)?.require("training_recipe")?;
    eq(
        value.require("training_recipe")?,
        recipe,
        "HPO saved full training recipe differs",
    )?;
    eq(
        value.require("optimizer_recipe")?,
        &optimizer_recipe(recipe)?,
        "HPO saved Adam descriptor differs",
    )?;
    eq(
        value.require("producer")?.require("source_digest")?,
        inference.require("producer")?.require("source_digest")?,
        "HPO paired producer source differs",
    )?;
    let rc = value.require("producer")?.require("code_revision")?;
    let ic = inference.require("producer")?.require("code_revision")?;
    if !rc.is_null() && !ic.is_null() {
        eq(rc, ic, "HPO paired producer revision differs")?;
    }
    pcg(value.require("shuffle_rng")?)?;
    let cuda = if text(state, "device")? == "cpu" {
        0
    } else {
        n(state.require("hardware")?, "device_count", 1, 1024)?
    };
    n(value, "cuda_rng_count", cuda, cuda)?;
    is(value.require("state")?, "prefix", "resume")?;
    let descriptors = bundle(value.require("state")?)?;
    let shapes = model4(model, b, c)?;
    let states = value.require("parameter_states")?.object()?;
    if states.len() != shapes.len() {
        return Err(Error::contract("HPO Adam named ownership differs"));
    }
    let mut expected: BTreeMap<String, (&str, Option<Vec<i128>>)> = BTreeMap::new();
    expected.insert("rng.torch_cpu".into(), ("uint8", None));
    for index in 0..cuda {
        expected.insert(format!("rng.torch_cuda.{index}"), ("uint8", None));
    }
    for (name, shape) in shapes {
        let entry = states
            .get(&name)
            .ok_or_else(|| Error::contract("HPO Adam parameter metadata missing"))?;
        entry.closed(&["present"], &[])?;
        if entry.require("present")?.boolean()? {
            expected.insert(format!("adam.{name}.step"), ("int64", Some(vec![])));
            for part in ["exp_avg", "exp_avg_sq"] {
                expected.insert(
                    format!("adam.{name}.{part}"),
                    ("float32", Some(shape.clone())),
                );
            }
        }
    }
    if expected.len() != descriptors.len() {
        return Err(Error::contract("HPO Adam/RNG tensor set mismatch"));
    }
    for (name, (dtype, shape)) in expected {
        let actual = descriptors
            .get(&name)
            .ok_or_else(|| Error::contract("HPO saved Adam/RNG tensor missing"))?;
        is(actual, "dtype", dtype)?;
        if let Some(shape) = shape {
            eq(
                actual.require("shape")?,
                &json::array(shape.into_iter().map(json::integer)),
                "HPO saved optimizer shape differs",
            )?;
        } else {
            let shape = actual.require("shape")?.array()?;
            if shape.len() != 1 {
                return Err(Error::contract("HPO saved RNG vector shape invalid"));
            }
            shape[0].integer(1, 64i128 << 20)?;
        }
    }
    Ok(())
}

fn bundle_files(s: &mut Sources, path: &Path, value: &Value) -> Result<()> {
    bundle(value)?;
    for shard in value.require("shards")?.array()? {
        let name = text(shard, "file")?.replace(".npz", ".json");
        let side = s.checked_json(&path.join(&name), value.require("files")?.require(&name)?)?;
        eq(
            &side.root,
            shard.require("metadata")?,
            "HPO saved shard sidecar differs from original descriptor",
        )?;
    }
    s.file_map(path, value.require("files")?, true)
}

fn artifact(
    s: &mut Sources,
    state: &Value,
    record: &Value,
    entry: &Value,
    model: &Value,
    inference_required: bool,
    resume_required: bool,
    protected_semantics: &Value,
) -> Result<Value> {
    let b = s.budget.clone();
    let c = s.control.clone();
    entry.closed(&["files"], &[])?;
    let original = record.require("artifact_files")?.object()?;
    let files = entry.require("files")?.object()?;
    if original.len() != files.len() || original.keys().any(|key| !files.contains_key(key)) {
        return Err(Error::contract(
            "HPO per-file registry differs from immutable artifact map",
        ));
    }
    let mut retained = json::object([]);
    for (name, sha) in original {
        c.check()?;
        let status = file_receipt(&files[name], checksum(sha)?)?;
        if ["deleting", "failed"].contains(&status) {
            return Err(Error::contract(
                "HPO uncertain deletion cannot form a healthy parent",
            ));
        }
        if status == "retained" {
            retained
                .as_object_mut()
                .unwrap()
                .insert(name.clone(), sha.clone());
        }
    }
    let full = retained.object()?.len() == original.len();
    let infer = retained.get("inference.json").is_some();
    let resume = retained.get("resume.json").is_some();
    let infer_closed = original
        .keys()
        .filter(|name| name.as_str() == "inference.json" || name.starts_with("weights-"))
        .all(|name| retained.get(name).is_some());
    if (inference_required || infer) && !infer_closed || (resume_required || resume) && !full {
        return Err(Error::contract(
            "HPO protected saved-file dependency closure missing",
        ));
    }
    let number = n(record, "generation", 0, i64::MAX as i128)?;
    let path = s.root.join("artifacts").join(format!("{number:06}"));
    s.inventory(&path, retained.object()?.keys().cloned().collect(), false)?;
    s.file_map(&path, &retained, true)?;
    if !infer {
        return json::copy(protected_semantics, &c);
    }
    let inf = s.checked_json(
        &path.join("inference.json"),
        record.require("inference_sha256")?,
    )?;
    inference(
        &inf.root,
        state.require("effective_config")?.require("actions")?,
        &b,
        &c,
    )?;
    let mut pure = model.clone();
    pure.as_object_mut().unwrap().remove("resources");
    eq(
        inf.root.require("model")?,
        &pure,
        "HPO saved pure inference architecture differs from full run model",
    )?;
    bundle_files(s, &path, inf.root.require("weights")?)?;
    if resume {
        let res = s.checked_json(&path.join("resume.json"), record.require("resume_sha256")?)?;
        resume_metadata(&res.root, &inf.root, state, record, model, &b, &c)?;
        bundle_files(s, &path, res.root.require("state")?)?;
        let mut actual = inf
            .root
            .require("weights")?
            .require("files")?
            .object()?
            .clone();
        for (name, sha) in res.root.require("state")?.require("files")?.object()? {
            if actual.insert(name.clone(), sha.clone()).is_some() {
                return Err(Error::contract("HPO artifact bundles overlap"));
            }
        }
        actual.insert(
            "inference.json".into(),
            record.require("inference_sha256")?.clone(),
        );
        actual.insert(
            "resume.json".into(),
            record.require("resume_sha256")?.clone(),
        );
        eq(
            &Value::Object(actual),
            record.require("artifact_files")?,
            "HPO saved full original artifact inventory differs",
        )?;
    } else {
        for (name, sha) in inf.root.require("weights")?.require("files")?.object()? {
            eq(
                sha,
                &original[name],
                "HPO retained inference weights differ from immutable map",
            )?;
        }
    }
    clone_fields(&inf.root, &semantic_keys(), &c)
}

fn frozen_registry(
    root: &Path,
    state: &Value,
    records: &[Document],
    generation: usize,
    model: &Value,
    update: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<String> {
    let mut registry = json::object([
        (
            "random-card-instance-v2",
            json::object([
                ("model_id", json::text("random")),
                ("policy_options", Value::Null),
            ]),
        ),
        (
            "heuristic-card-instance-v2",
            json::object([
                ("model_id", json::text("heuristic")),
                ("policy_options", Value::Null),
            ]),
        ),
    ]);
    let cfg = state.require("effective_config")?;
    if let Some(options) = refresh_training(cfg, b, c)? {
        let label = crate::structured_journal::card_instance::refresh_label(cfg.require("actions")?, options, b, c)?;
        registry.as_object_mut().unwrap().insert(label, json::object([
            ("model_id", json::text("heuristic_refresh")), ("policy_options", options.clone())]));
    }
    let mut numbers = vec![generation];
    for index in records[generation]
        .root
        .require("parent_controls")?
        .require("snapshot_pool")?
        .array()?
    {
        numbers.push(index.integer(0, generation as i128)? as usize);
    }
    let execution = state.require("execution")?.require("inference")?;
    let actions = state.require("effective_config")?.require("actions")?;
    let mut current = None;
    for number in numbers {
        let mut safe = json::object([
            ("version", json::text("4")),
            (
                "checkpoint",
                json::text(
                    root.join("artifacts")
                        .join(format!("{number:06}"))
                        .to_str()
                        .ok_or_else(|| Error::contract("HPO UTF-8 checkpoint path required"))?,
                ),
            ),
            (
                "expected_manifest_digest",
                records[number].root.require("inference_sha256")?.clone(),
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
        let label = options(&safe, actions, b, c)?;
        if number == generation {
            current = Some(label.clone());
        }
        registry.as_object_mut().unwrap().insert(
            label,
            json::object([
                ("model_id", model.require("model_id")?.clone()),
                ("policy_options", safe),
            ]),
        );
    }
    eq(
        update.require("policy_bindings")?,
        &registry,
        "HPO original current/pool policy registry differs from committed controls",
    )?;
    current.ok_or_else(|| Error::contract("HPO frozen current policy missing"))
}

fn update_metadata(
    s: &mut Sources,
    value: &Value,
    state: &Value,
    previous: &Value,
    record: &Value,
    model: &Value,
    previous_hash: &str,
    semantics: &Value,
    policy: &str,
) -> Result<(usize, usize, i128)> {
    let b = s.budget.clone();
    let c = s.control.clone();
    let mut keys = vec![
        "schema_version",
        "status",
        "run_id",
        "generation",
        "parent_sha256",
        "batch_id",
        "collection",
        "collection_manifest_sha256",
        "numeric",
        "numeric_sha256",
        "accepted_decisions",
        "source_proof",
        "metrics",
        "planned_games",
        "actual_games",
        "health",
        "policy_bindings",
        "learner_seats",
        "collection_seconds",
        "update_seconds",
        "model_id",
        "model_version",
        "input_recipe",
        "action_contract",
        "purpose",
        "synthetic_capacity",
        "numeric_files",
        "rule_signature",
        "layout_signature",
        "hpo_spec_sha256",
        "collected_decisions",
        "used_decisions",
        "overshoot_decisions",
        "expected_optimizer_step_delta",
    ];
    if let Some(binding) = state.get("storage_binding") {
        keys.push("storage_binding");
        eq(
            value.require("storage_binding")?,
            binding,
            "HPO packed update storage binding differs",
        )?;
    }
    value.closed(&keys, &[])?;
    n(value, "schema_version", 5, 5)?;
    is(value, "status", "complete")?;
    let generation = n(record, "generation", 1, i64::MAX as i128)?;
    n(value, "generation", generation, generation)?;
    is(value, "parent_sha256", previous_hash)?;
    let batch = generation - 1;
    is(value, "batch_id", &format!("batch-{batch:06}"))?;
    is(
        value,
        "collection",
        &format!("collections/batch-{batch:06}"),
    )?;
    is(value, "numeric", &format!("numeric/batch-{batch:06}"))?;
    for key in [
        "run_id",
        "model_id",
        "model_version",
        "input_recipe",
        "hpo_spec_sha256",
    ] {
        eq(
            value.require(key)?,
            state.require(key)?,
            "HPO update immutable identity differs",
        )?;
    }
    for key in GAME_CLASS {
        eq(
            value.require(key)?,
            state.require(key)?,
            "HPO update public input binding differs",
        )?;
    }
    let recipe = spec(state, &b, &c)?.require("training_recipe")?;
    let games = n(recipe, "collection_games", 8, 65536)?;
    n(value, "planned_games", games, games)?;
    n(value, "actual_games", games, games)?;
    if !value.require("health")?.object()?.is_empty() {
        return Err(Error::contract(
            "HPO committed update contains unhealthy games",
        ));
    }
    let used = n(value, "used_decisions", 1, i64::MAX as i128)? as usize;
    let collected = n(value, "collected_decisions", used as i128, i64::MAX as i128)? as usize;
    n(value, "accepted_decisions", used as i128, used as i128)?;
    n(
        value,
        "overshoot_decisions",
        (collected - used) as i128,
        (collected - used) as i128,
    )?;
    let delta = optimizer_delta(recipe, used)?;
    n(value, "expected_optimizer_step_delta", delta, delta)?;
    let proof = value.require("source_proof")?;
    let packed = state.get("storage_binding").is_some();
    let mut proof_keys = vec![
        "schema_version",
        "kind",
        "qualified",
        "parent_qualification",
        "manifest_sha256",
        "source_set_sha256",
        "files",
        "games",
        "accepted_decisions",
        "closure_counts",
        "missing_replays",
        "selection",
        "expected_optimizer_step_delta",
    ];
    proof_keys.extend(GAME_CLASS);
    if packed {
        proof_keys.extend(["storage", "custody"]);
    }
    let resident = resident::is_resident(state);
    if resident { proof_keys.push("logical_retirement"); }
    proof.closed(&proof_keys, &[])?;
    n(proof, "schema_version", if resident {4} else {3}, if resident {4} else {3})?;
    if resident {
        let max_workers = n(spec(state, &b, &c)?.require("operating")?, "max_participant_workers", 1, i64::MAX as i128)?;
        resident::retirement(proof.require("logical_retirement")?, max_workers, &b, &c)?;
        let retirement = proof.require("logical_retirement")?;
        eq(retirement.require("run_id")?, state.require("run_id")?, "Proof4 original run differs")?;
        eq(retirement.require("generation")?, previous.require("generation")?, "Proof4 original generation differs")?;
        is(retirement, "collection_id", value.require("batch_id")?.string()?)?;
    }
    is(
        proof,
        "kind",
        if packed {
            "card-instance-hpo-packed-original-source-proof"
        } else {
            "card-instance-hpo-original-source-proof"
        },
    )?;
    eq(
        proof.require("qualified")?,
        &Value::Bool(false),
        "HPO Source proof cannot grant qualification",
    )?;
    is(proof, "parent_qualification", "not_performed")?;
    for key in GAME_CLASS {
        eq(
            proof.require(key)?,
            state.require(key)?,
            "HPO Source proof public input binding differs",
        )?;
    }
    n(proof, "games", games, games)?;
    n(
        proof,
        "accepted_decisions",
        collected as i128,
        i64::MAX as i128,
    )?;
    n(proof, "expected_optimizer_step_delta", delta, delta)?;
    eq(
        proof.require("manifest_sha256")?,
        value.require("collection_manifest_sha256")?,
        "HPO original collection hash differs",
    )?;
    checksum(proof.require("manifest_sha256")?)?;
    is(
        proof,
        "source_set_sha256",
        &digest(proof.require("files")?, &b, &c)?,
    )?;
    let selected = proof.require("selection")?;
    let context_value = json::object([
        ("selection_version", json::text(SELECTION_VERSION)),
        (
            "decision_budget",
            state
                .require("effective_config")?
                .require("hpo_spec")?
                .require("decision_budget")?
                .clone(),
        ),
        ("used_before", previous.require("used_decisions")?.clone()),
        (
            "remaining",
            json::integer(
                n(spec(state, &b, &c)?, "decision_budget", 1, i64::MAX as i128)?
                    - n(previous, "used_decisions", 0, i64::MAX as i128)?,
            ),
        ),
        ("learning_seed", state.require("seed")?.clone()),
        ("batch", json::integer(batch)),
    ]);
    let counts = selected
        .require("episode_counts")?
        .array()?
        .iter()
        .map(|v| v.integer(0, 1024).map(|n| n as usize))
        .collect::<Result<Vec<_>>>()?;
    if counts.len() != games as usize
        || selection(selected, &counts, &context_value, &b, &c)? != used
        || counts.iter().sum::<usize>() != collected
    {
        return Err(Error::contract(
            "HPO update selected/collected original count mismatch",
        ));
    }
    eq(
        proof.require("closure_counts")?,
        &json::object([("natural", json::integer(games))]),
        "HPO whole original natural closure counts differ",
    )?;
    if !proof.require("missing_replays")?.array()?.is_empty() {
        return Err(Error::contract(
            "HPO committed batch has missing original replays",
        ));
    }
    let seats = value.require("learner_seats")?.object()?;
    if seats.len() != games as usize {
        return Err(Error::contract(
            "HPO complete original learner viewer population differs",
        ));
    }
    for viewer in seats.values() {
        viewer.integer(1, 2)?;
    }
    let numeric_path = relative(&s.root, value.require("numeric")?)?;
    let numeric = s.checked_json(
        &numeric_path.join("rollout.json"),
        value.require("numeric_sha256")?,
    )?;
    let raw = &numeric.root;
    let mut numeric_keys = vec![
        "schema_version",
        "kind",
        "architecture_identity",
        "action_config_hash",
        "policy_version",
        "decision_count",
        "episodes",
        "data",
        "training_recipe",
        "selection",
        "expected_optimizer_step_delta",
    ];
    numeric_keys.extend(semantic_keys());
    numeric_keys.extend(GAME_CLASS);
    raw.closed(&numeric_keys, &[])?;
    n(raw, "schema_version", 5, 5)?;
    is(raw, "kind", "structured-rollout")?;
    is(raw, "policy_version", policy)?;
    n(raw, "decision_count", used as i128, used as i128)?;
    n(raw, "expected_optimizer_step_delta", delta, delta)?;
    eq(
        raw.require("training_recipe")?,
        recipe,
        "HPO numeric training recipe differs",
    )?;
    eq(
        raw.require("selection")?,
        selected,
        "HPO numeric selection differs from Source proof",
    )?;
    for key in GAME_CLASS {
        eq(
            raw.require(key)?,
            state.require(key)?,
            "HPO numeric public input binding differs",
        )?;
    }
    for key in semantic_keys() {
        eq(
            raw.require(key)?,
            semantics.require(key)?,
            "HPO numeric public semantics differ from protected G0",
        )?;
    }
    is(raw, "architecture_identity", &architecture(model, &b, &c)?)?;
    is(
        raw,
        "action_config_hash",
        &digest(
            state.require("effective_config")?.require("actions")?,
            &b,
            &c,
        )?,
    )?;
    let episodes = raw.require("episodes")?.array()?;
    if episodes.len() != games as usize {
        return Err(Error::contract(
            "HPO numeric must retain every complete learner episode",
        ));
    }
    let mut owners = std::collections::BTreeSet::new();
    for (episode, count) in episodes.iter().zip(&counts) {
        game_binding3(episode, &b, &c)?;
        is(episode, "end_kind", "natural")?;
        crate::validation::result(episode.require("outcome")?)?;
        if episode.require("outcome")?.is_null()
            || !episode.require("reason")?.is_null()
            || episode.require("frames")?.array()?.len() != *count
            || episode.require("selections")?.array()?.len() != *count
            || episode.require("has_targets")? != &Value::Bool(*count != 0)
            || !owners.insert((n(episode, "actor", 1, 2)?, text(episode, "episode")?))
        {
            return Err(Error::contract(
                "HPO numeric original natural participant episode differs",
            ));
        }
        for frame in episode.require("frames")?.array()? {
            game_binding3(frame, &b, &c)?;
        }
    }
    let mut actual = raw.require("data")?.require("files")?.object()?.clone();
    actual.insert(
        "rollout.json".into(),
        value.require("numeric_sha256")?.clone(),
    );
    eq(
        &Value::Object(actual),
        value.require("numeric_files")?,
        "HPO numeric exact file inventory differs",
    )?;
    s.inventory(
        &numeric_path,
        value
            .require("numeric_files")?
            .object()?
            .keys()
            .cloned()
            .collect(),
        false,
    )?;
    bundle_files(s, &numeric_path, raw.require("data")?)?;
    let metrics = value.require("metrics")?;
    metrics.closed(
        &[
            "policy_loss",
            "value_mse",
            "entropy",
            "approx_kl",
            "total_loss",
            "optimizer_steps",
            "optimizer_step_delta",
            "collected_decisions",
            "used_decisions",
            "parity",
            "tbptt_length",
            "epochs",
            "endpoint_evaluations",
            "context_decisions",
            "suffix_decisions",
            "update_seconds",
        ],
        &["elapsed_seconds", "save_seconds", "peak_cpu_memory_bytes"],
    )?;
    for key in [
        "policy_loss",
        "value_mse",
        "entropy",
        "approx_kl",
        "total_loss",
    ] {
        if !metrics.require(key)?.number()?.is_finite() {
            return Err(Error::contract("HPO nonfinite update metric"));
        }
    }
    let steps = n(previous, "optimizer_steps", 0, i32::MAX as i128)? + delta;
    n(metrics, "optimizer_steps", steps, steps)?;
    n(metrics, "optimizer_step_delta", delta, delta)?;
    n(
        metrics,
        "collected_decisions",
        collected as i128,
        collected as i128,
    )?;
    n(metrics, "used_decisions", used as i128, used as i128)?;
    let epochs = n(recipe, "epochs", 1, 1000)?;
    let tbptt = n(recipe, "tbptt_length", 1, 1024)?;
    n(metrics, "epochs", epochs, epochs)?;
    n(metrics, "tbptt_length", tbptt, tbptt)?;
    let endpoints = epochs * used as i128;
    n(metrics, "endpoint_evaluations", endpoints, endpoints)?;
    let context_count = n(metrics, "context_decisions", endpoints, endpoints * 1024)?;
    n(
        metrics,
        "suffix_decisions",
        endpoints,
        context_count.min(endpoints * tbptt),
    )?;
    let parity = metrics.require("parity")?;
    parity.closed(
        &["atol", "rtol", "max_abs_error", "decisions", "checks"],
        &[],
    )?;
    if finite(parity, "atol")? != 1e-5 || finite(parity, "rtol")? != 1e-5 {
        return Err(Error::contract("HPO parity tolerance recipe differs"));
    }
    finite(parity, "max_abs_error")?;
    n(parity, "decisions", collected as i128, collected as i128)?;
    eq(
        parity.require("checks")?,
        &json::array(
            [
                "saved",
                "cpu-cache",
                "cpu-full",
                "learner-full",
                "learner-split",
            ]
            .into_iter()
            .map(json::text),
        ),
        "HPO whole-original parity recipe differs",
    )?;
    for key in [
        "update_seconds",
        "elapsed_seconds",
        "save_seconds",
        "peak_cpu_memory_bytes",
    ] {
        if metrics.get(key).is_some() {
            finite(metrics, key)?;
        }
    }
    finite(value, "collection_seconds")?;
    finite(value, "update_seconds")?;
    Ok((collected, used, delta))
}

pub(crate) fn parent(s: &mut Sources, packed: bool) -> Result<Document> {
    let root = s.root.clone();
    let b = s.budget.clone();
    let c = s.control.clone();
    let state = s.json(&root.join("run.json"))?;
    let r = &state.root;
    let model = run(r, &b, &c)?;
    if packed != r.get("storage_binding").is_some() {
        return Err(Error::contract(
            "explicit HPO parent physical variant differs",
        ));
    }
    let count = n(r, "updates", 0, i64::MAX as i128)? as usize;
    let control_bytes = count
        .checked_add(1)
        .and_then(|n| n.checked_mul(4096))
        .ok_or_else(|| Error::resource("HPO generation control admission overflow"))?;
    let _copies = b.reserve(control_bytes)?;
    s.inventory(
        &root.join("generations"),
        (0..=count).map(|n| format!("{n:06}.json")).collect(),
        false,
    )?;
    s.inventory(
        &root.join("updates"),
        (1..=count).map(|n| format!("{n:06}.json")).collect(),
        false,
    )?;
    s.inventory(
        &root.join("artifacts"),
        (0..=count).map(|n| format!("{n:06}")).collect(),
        true,
    )?;
    let registry = s.json(&root.join("artifact-registry.json"))?;
    registry.root.closed(
        &["schema_version", "generations", "evaluation_references"],
        if resident::is_resident(r) { &["cache_references"] } else { &[] },
    )?;
    let registry_version = if resident::is_resident(r) {5} else {4};
    n(&registry.root, "schema_version", registry_version, registry_version)?;
    let entries = registry.root.require("generations")?.object()?;
    if entries.len() != count + 1 || (0..=count).any(|n| !entries.contains_key(&n.to_string())) {
        return Err(Error::contract(
            "HPO per-file registry generation population differs",
        ));
    }
    let mut records = Vec::new();
    let mut previous_hash = None;
    for number in 0..=count {
        c.check()?;
        let record = s.json(&root.join("generations").join(format!("{number:06}.json")))?;
        generation(
            &record.root,
            r,
            number,
            records.last().map(|v: &Document| &v.root),
            previous_hash.as_deref(),
            &b,
            &c,
        )?;
        previous_hash = Some(
            s.verified_sha(&root.join("generations").join(format!("{number:06}.json")))?
                .to_owned(),
        );
        records.push(record);
    }
    is(
        r,
        "generation_sha256",
        previous_hash
            .as_deref()
            .ok_or_else(|| Error::contract("HPO G0 missing"))?,
    )?;
    let latest = s.json(&root.join("latest.json"))?;
    latest.root.closed(&["generation", "sha256"], &[])?;
    n(&latest.root, "generation", count as i128, count as i128)?;
    eq(
        latest.root.require("sha256")?,
        r.require("generation_sha256")?,
        "HPO latest/run original link differs",
    )?;
    let configuration = spec(r, &b, &c)?;
    let operating = configuration.require("operating")?;
    let resume_latest = n(operating, "resume_latest", 2, 1024)? as usize;
    let inference_latest = n(operating, "inference_latest", 1, i64::MAX as i128)? as usize;
    let mut required = std::collections::BTreeSet::from([0usize]);
    for number in count.saturating_add(1).saturating_sub(inference_latest)..=count {
        required.insert(number);
    }
    for number in count.saturating_add(1).saturating_sub(resume_latest)..=count {
        required.insert(number);
    }
    for number in records[count]
        .root
        .require("parent_controls")?
        .require("snapshot_pool")?
        .array()?
    {
        required.insert(number.integer(0, count as i128)? as usize);
    }
    let interval = n(
        operating,
        "inference_periodic_decisions",
        1,
        i64::MAX as i128,
    )?;
    for number in 1..=count {
        if n(&records[number].root, "used_decisions", 0, i64::MAX as i128)? / interval
            > n(
                &records[number - 1].root,
                "used_decisions",
                0,
                i64::MAX as i128,
            )? / interval
        {
            required.insert(number);
        }
    }
    for (token, reference) in registry.root.require("evaluation_references")?.object()? {
        if token.is_empty()
            || token.len() > 128
            || !token.is_ascii()
            || !token.as_bytes()[0].is_ascii_alphanumeric()
            || !token
                .bytes()
                .all(|v| v.is_ascii_alphanumeric() || b"_.-".contains(&v))
        {
            return Err(Error::contract("HPO evaluation reference token invalid"));
        }
        reference.closed(&["generation", "inference_sha256"], &[])?;
        let number = n(reference, "generation", 0, count as i128)? as usize;
        eq(
            reference.require("inference_sha256")?,
            records[number].root.require("inference_sha256")?,
            "HPO pinned evaluation inference differs",
        )?;
        required.insert(number);
    }
    if resident::is_resident(r) {
        required.extend(resident::cache_references(s, r, &records, &model, &registry.root, &b, &c)?);
    } else if root.join("cache-reference-receipts").exists() || root.join("cache-reference-receipts").is_symlink() {
        return Err(Error::contract("fresh HPO forbids resident cache journal"));
    }
    let mut declaration = Value::Null;
    let mut collected = 0usize;
    let mut used = 0usize;
    let mut steps = 0i128;
    for number in 0..=count {
        let current = &records[number].root;
        let next = artifact(
            s,
            r,
            current,
            &entries[&number.to_string()],
            &model,
            required.contains(&number),
            number == 0 || number >= count.saturating_add(1).saturating_sub(resume_latest),
            &declaration,
        )?;
        if number == 0 {
            declaration = next;
        } else {
            eq(
                &next,
                &declaration,
                "HPO run public/runtime semantics changed across generations",
            )?;
        }
        if number != 0 {
            let update = s.checked_json(
                &root.join("updates").join(format!("{number:06}.json")),
                current.require("update_sha256")?,
            )?;
            let policy =
                frozen_registry(&root, r, &records, number - 1, &model, &update.root, &b, &c)?;
            let (a, u, delta) = update_metadata(
                s,
                &update.root,
                r,
                &records[number - 1].root,
                current,
                &model,
                s.verified_sha(
                    &root
                        .join("generations")
                        .join(format!("{:06}.json", number - 1)),
                )?
                .to_owned()
                .as_str(),
                &declaration,
                &policy,
            )?;
            collected = collected
                .checked_add(a)
                .ok_or_else(|| Error::resource("HPO cumulative original count overflow"))?;
            used = used
                .checked_add(u)
                .ok_or_else(|| Error::resource("HPO cumulative used count overflow"))?;
            steps += delta;
        }
        n(
            current,
            "collected_decisions",
            collected as i128,
            collected as i128,
        )?;
        n(current, "used_decisions", used as i128, used as i128)?;
        n(current, "optimizer_steps", steps, steps)?;
    }
    n(
        r,
        "collected_decisions",
        collected as i128,
        collected as i128,
    )?;
    n(r, "used_decisions", used as i128, used as i128)?;
    n(r, "optimizer_steps", steps, steps)?;
    let mut result = json::object([
        ("schema_version", json::integer(5)),
        ("kind", json::text("card-instance-hpo-parent-controls")),
        (
            "validation_scope",
            json::text("metadata-maps-filehashes-only"),
        ),
        ("npz_validated", Value::Bool(false)),
        ("journal6_validated", Value::Bool(false)),
        ("qualified", Value::Bool(false)),
        ("parent_qualification", json::text("not_performed")),
        (
            "directory",
            json::text(
                root.to_str()
                    .ok_or_else(|| Error::contract("HPO original UTF-8 path required"))?,
            ),
        ),
        ("state", json::copy(r, &c)?),
        ("model", model),
        ("execution", r.require("execution")?.clone()),
        ("semantics", declaration),
        (
            "actions_sha256",
            json::text(&digest(
                r.require("effective_config")?.require("actions")?,
                &b,
                &c,
            )?),
        ),
        (
            "identity_sha256",
            json::text(&digest(r.require("identity")?, &b, &c)?),
        ),
        (
            "generations",
            json::array(records.iter().map(|v| v.root.clone())),
        ),
        ("registry", registry.root.clone()),
        ("control_fingerprint", json::text(&s.fingerprint()?)),
    ]);
    for key in GAME_CLASS {
        result
            .as_object_mut()
            .unwrap()
            .insert((*key).into(), r.require(key)?.clone());
    }
    if packed {
        result.as_object_mut().unwrap().insert(
            "storage_binding".into(),
            r.require("storage_binding")?.clone(),
        );
    }
    let raw = json::canonical(&result, &b, &c)?;
    json::parse(&raw, &b, &c)
}

pub(crate) fn verify_derived_artifact(
    s: &mut Sources,
    state: &Value,
    generation: usize,
    steps: usize,
) -> Result<Value> {
    let b = s.budget.clone();
    let c = s.control.clone();
    let path = s
        .root
        .join("artifacts")
        .join(format!("{:06}", generation + 1));
    let inference = s.json(&path.join("inference.json"))?;
    let resume = s.json(&path.join("resume.json"))?;
    let inf_sha = s.verified_sha(&path.join("inference.json"))?.to_owned();
    let res_sha = s.verified_sha(&path.join("resume.json"))?.to_owned();
    let mut files = json::object([]);
    for bundle in [
        inference.root.require("weights")?,
        resume.root.require("state")?,
    ] {
        for (name, sha) in bundle.require("files")?.object()? {
            if files
                .as_object_mut()
                .unwrap()
                .insert(name.clone(), sha.clone())
                .is_some()
            {
                return Err(Error::contract("HPO derived artifact file collision"));
            }
        }
    }
    files["inference.json"] = json::text(&inf_sha);
    files["resume.json"] = json::text(&res_sha);
    let links = json::object([
        ("generation", json::integer((generation + 1) as i128)),
        ("inference_sha256", json::text(&inf_sha)),
        ("resume_sha256", json::text(&res_sha)),
        ("artifact_files", files.clone()),
        ("optimizer_steps", json::integer(steps as i128)),
    ]);
    let entry = json::object([(
        "files",
        Value::Object(
            files
                .object()?
                .iter()
                .map(|(name, sha)| {
                    (
                        name.clone(),
                        json::object([
                            ("sha256", sha.clone()),
                            ("status", json::text("retained")),
                            ("deletion_intent", Value::Null),
                            ("deletion_receipt", Value::Null),
                        ]),
                    )
                })
                .collect(),
        ),
    )]);
    let model = run(state, &b, &c)?;
    artifact(s, state, &links, &entry, &model, true, true, &Value::Null)?;
    Ok(files)
}

pub(crate) fn context(value: &Value) -> Result<()> {
    value.closed(
        &[
            "selection_version",
            "decision_budget",
            "used_before",
            "remaining",
            "learning_seed",
            "batch",
        ],
        &[],
    )?;
    is(value, "selection_version", SELECTION_VERSION)?;
    let budget = n(value, "decision_budget", 1, i64::MAX as i128)?;
    let before = n(value, "used_before", 0, budget)?;
    n(value, "remaining", budget - before, budget - before)?;
    n(value, "learning_seed", 0, i64::MAX as i128)?;
    n(value, "batch", 0, i64::MAX as i128)?;
    Ok(())
}

pub(crate) fn selection(
    value: &Value,
    counts: &[usize],
    expected_context: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<usize> {
    value.closed(
        &[
            "context",
            "derived_seed",
            "episode_counts",
            "collected_decisions",
            "used_decisions",
            "indices",
            "indices_sha256",
        ],
        &[],
    )?;
    context(value.require("context")?)?;
    context(expected_context)?;
    eq(
        value.require("context")?,
        expected_context,
        "HPO selection original context mismatch",
    )?;
    let declared = value.require("episode_counts")?.array()?;
    if declared.len() != counts.len() {
        return Err(Error::contract(
            "HPO complete original episode count mismatch",
        ));
    }
    let mut collected = 0usize;
    for (actual, declared) in counts.iter().zip(declared) {
        c.check()?;
        declared.integer(*actual as i128, *actual as i128)?;
        if *actual > 1024 {
            return Err(Error::contract(
                "HPO original own-decision capacity exceeded",
            ));
        }
        collected = collected
            .checked_add(*actual)
            .ok_or_else(|| Error::resource("HPO collected count overflow"))?;
    }
    n(
        value,
        "collected_decisions",
        collected as i128,
        collected as i128,
    )?;
    let used = (n(expected_context, "remaining", 0, i64::MAX as i128)? as usize).min(collected);
    n(value, "used_decisions", used as i128, used as i128)?;
    let seed = digest(
        &json::object([
            ("recipe", json::text("hpo-ppo-selection-seed-v1")),
            ("context", json::copy(expected_context, c)?),
        ]),
        b,
        c,
    )?;
    let seed = u128::from_str_radix(&seed[..32], 16)
        .map_err(|_| Error::contract("invalid HPO derived seed"))?;
    if uint128(value.require("derived_seed")?)? != seed {
        return Err(Error::contract(
            "HPO original independently derived seed mismatch",
        ));
    }
    let indices = value.require("indices")?.array()?;
    if indices.len() != used {
        return Err(Error::contract(
            "HPO selected unique endpoint count mismatch",
        ));
    }
    let mut previous = -1i128;
    for index in indices {
        c.check()?;
        previous = index.integer(previous + 1, collected as i128 - 1)?;
    }
    is(
        value,
        "indices_sha256",
        &digest(value.require("indices")?, b, c)?,
    )?;
    Ok(used)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "actual held CPU bytes, explicitly unqualified; requires GFB_HPO_NATIVE_LIVE_FIXTURE"]
    fn source5_unadmitted_actual_live_numeric_bind_dynamic_prepare_and_faults() {
        let fixture = std::path::PathBuf::from(
            std::env::var("GFB_HPO_NATIVE_LIVE_FIXTURE")
                .expect("explicit component fixture required"),
        );
        let root = fixture.join("parent");
        let b = Budget::new(1usize << 30);
        let c = Control::new(crate::budget::monotonic() + 120.).unwrap();
        let component = json::parse(
            &std::fs::read(fixture.join("component.json")).unwrap(),
            &b,
            &c,
        )
        .unwrap();
        let empty = component.root.get("selection").is_some();
        let numeric = json::parse(
            &std::fs::read(if empty {
                fixture.join("component.json")
            } else {
                fixture.join("staged-numeric/rollout.json")
            })
            .unwrap(),
            &b,
            &c,
        )
        .unwrap();
        let plan = json::object([
            ("schema_version", json::integer(5)),
            ("mode", json::text("live_cohort")),
            (
                "parents",
                json::array([json::object([
                    ("directory", json::text(root.to_str().unwrap())),
                    ("require_complete", Value::Bool(false)),
                ])]),
            ),
            (
                "collections",
                json::array([json::object([
                    (
                        "directory",
                        json::text(root.join("collections/batch-000000").to_str().unwrap()),
                    ),
                    ("parent_index", json::integer(0)),
                ])]),
            ),
            ("cohort_collection_index", json::integer(0)),
        ]);
        let encoded = json::canonical(&plan, &b, &c).unwrap();
        let open = || {
            crate::structured_session::CardInstanceNativeSession::open(
                &encoded,
                b.clone(),
                c.clone(),
                64 << 20,
                8 << 20,
                512 << 20,
                8 << 20,
            )
        };
        // The normal constructor still refuses these explicitly CPU originals.
        assert!(open().is_err());
        let mut session = crate::structured_source::unadmitted_hpo_component(open).unwrap();
        assert!(session.begin_update().is_err());
        let pending = json::parse(&session.control_summary().unwrap(), &b, &c).unwrap();
        fn file_hash(path: &std::path::Path) -> Value {
            use sha2::Digest;
            json::text(&format!(
                "{:x}",
                sha2::Sha256::digest(std::fs::read(path).unwrap())
            ))
        }
        let immutable = json::object([(
            "generations/000000.json",
            file_hash(&root.join("generations/000000.json")),
        )]);
        let request = json::object([
            ("schema_version", json::integer(5)),
            (
                "pending_sha256",
                json::text(&digest(&pending.root, &b, &c).unwrap()),
            ),
            (
                "parent",
                json::object([
                    ("run_sha256", file_hash(&root.join("run.json"))),
                    (
                        "config_sha256",
                        pending.root["parent"]["state"]["config_sha256"].clone(),
                    ),
                    (
                        "resume_identity",
                        pending.root["parent"]["state"]["resume_identity"].clone(),
                    ),
                    (
                        "generation_files_sha256",
                        json::text(&digest(&immutable, &b, &c).unwrap()),
                    ),
                    (
                        "registry_sha256",
                        file_hash(&root.join("artifact-registry.json")),
                    ),
                ]),
            ),
            (
                "collections",
                json::array([json::object([
                    ("collection_index", json::integer(0)),
                    ("selection", numeric.root["selection"].clone()),
                    ("training_recipe", numeric.root["training_recipe"].clone()),
                    (
                        "expected_optimizer_step_delta",
                        numeric.root["expected_optimizer_step_delta"].clone(),
                    ),
                    (
                        "expected_correspondence",
                        component.root["correspondence"].clone(),
                    ),
                ])]),
            ),
        ]);
        session
            .authorize_hpo(&json::canonical(&request, &b, &c).unwrap())
            .unwrap();
        let payload = session.export_cohort().unwrap();
        if empty {
            let packet = json::parse(payload.metadata(), &b, &c).unwrap();
            assert_eq!(packet.root["metadata"]["decision_count"], json::integer(0));
            let episodes = packet.root["metadata"]["episodes"].array().unwrap();
            assert_eq!(episodes.len(), 8);
            assert!(
                episodes
                    .iter()
                    .all(|episode| episode["frames"].array().unwrap().is_empty()
                        && episode["end_kind"].as_str() == Some("natural"))
            );
            drop(packet);
            drop(payload);
            session.retire_payload().unwrap();
            assert!(session.begin_update().is_err());
            assert!(session.prepare_commit().is_err());
            assert!(!root.join("numeric/batch-000000").exists());
            assert!(!root.join("artifacts/000001").exists());
            session.close().unwrap();
            drop(session);
            drop(numeric);
            drop(pending);
            drop(encoded);
            drop(component);
            assert_eq!(b.used(), 0);
            return;
        }
        assert!(session.retire_payload().is_err());
        drop(payload);
        session.retire_payload().unwrap();
        assert_eq!(b.numeric_used(), None);
        assert!(session.begin_update().is_err());
        fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
            std::fs::create_dir(to).unwrap();
            for entry in std::fs::read_dir(from).unwrap() {
                let entry = entry.unwrap();
                assert!(entry.file_type().unwrap().is_file());
                std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
            }
        }
        copy_tree(
            &fixture.join("staged-numeric"),
            &root.join("numeric/batch-000000"),
        );
        session
            .bind_numeric(&json::canonical(&component.root["binding"], &b, &c).unwrap())
            .unwrap();
        assert!(
            session
                .bind_numeric(&json::canonical(&component.root["binding"], &b, &c).unwrap())
                .is_err()
        );
        session.begin_update().unwrap();
        copy_tree(
            &fixture.join("staged-checkpoint"),
            &root.join("artifacts/000001"),
        );
        let receipt = json::parse(session.prepare_commit().unwrap(), &b, &c).unwrap();
        assert_eq!(
            receipt.root["kind"],
            json::text("card-instance-hpo-source-prepared")
        );
        assert_eq!(receipt.root["used_decisions"], json::integer(3));
        assert_eq!(receipt.root["collected_decisions"], json::integer(8));
        assert_eq!(
            receipt.root["expected_optimizer_step_delta"],
            component.root["metrics"]["optimizer_step_delta"]
        );
        assert_eq!(
            receipt.root["optimizer_steps"],
            component.root["metrics"]["optimizer_steps"]
        );
        // Successful preparation retires the SAME core before exposing receipt.
        assert!(session.recheck().is_err());
        session.close().unwrap();
        drop(session);
        drop(receipt);
        std::fs::remove_dir_all(root.join("numeric/batch-000000")).unwrap();
        std::fs::remove_dir_all(root.join("artifacts/000001")).unwrap();
        for fault in 0..2 {
            let mut session = crate::structured_source::unadmitted_hpo_component(open).unwrap();
            let pending = json::parse(&session.control_summary().unwrap(), &b, &c).unwrap();
            let mut authorization = request.clone();
            authorization["pending_sha256"] = json::text(&digest(&pending.root, &b, &c).unwrap());
            session
                .authorize_hpo(&json::canonical(&authorization, &b, &c).unwrap())
                .unwrap();
            let payload = session.export_cohort().unwrap();
            drop(payload);
            session.retire_payload().unwrap();
            copy_tree(
                &fixture.join("staged-numeric"),
                &root.join("numeric/batch-000000"),
            );
            let mut binding = component.root["binding"].clone();
            if fault == 0 {
                // A rehashed wrong USED count must not gain numeric authority.
                let mut wrong = numeric.root.clone();
                wrong["decision_count"] = json::integer(4);
                std::fs::write(
                    root.join("numeric/batch-000000/rollout.json"),
                    &*json::canonical(&wrong, &b, &c).unwrap(),
                )
                .unwrap();
                let checksum = file_hash(&root.join("numeric/batch-000000/rollout.json"));
                binding["manifest_sha256"] = checksum.clone();
                binding["files"]["rollout.json"] = checksum;
                assert!(
                    session
                        .bind_numeric(&json::canonical(&binding, &b, &c).unwrap())
                        .is_err()
                );
                assert!(session.begin_update().is_err());
            } else {
                session
                    .bind_numeric(&json::canonical(&binding, &b, &c).unwrap())
                    .unwrap();
                session.begin_update().unwrap();
                copy_tree(
                    &fixture.join("staged-checkpoint"),
                    &root.join("artifacts/000001"),
                );
                let path = root.join("artifacts/000001/resume.json");
                let mut wrong = json::parse(&std::fs::read(&path).unwrap(), &b, &c)
                    .unwrap()
                    .root;
                wrong["optimizer_steps"] = json::integer(0);
                std::fs::write(path, &*json::canonical(&wrong, &b, &c).unwrap()).unwrap();
            }
            assert!(session.prepare_commit().is_err());
            // No failed precommit receipt can escape through a second call.
            assert!(session.prepare_commit().is_err());
            session.close().unwrap();
            drop(session);
            std::fs::remove_dir_all(root.join("numeric/batch-000000")).unwrap();
            if fault == 1 {
                std::fs::remove_dir_all(root.join("artifacts/000001")).unwrap();
            }
        }
        drop(numeric);
        drop(pending);
        drop(encoded);
        drop(component);
        assert_eq!(b.used(), 0);
    }

    #[test]
    #[ignore = "bounded private complete CPU originals; requires GFB_HPO_NATIVE_TEST_PARENT"]
    fn source5_held_controlled_parent_validates_original_run_registry_and_artifacts() {
        let root = std::path::PathBuf::from(
            std::env::var("GFB_HPO_NATIVE_TEST_PARENT")
                .expect("explicit private fixture path required"),
        );
        let b = Budget::new(1usize << 30);
        let c = Control::new(crate::budget::monotonic() + 120.).unwrap();
        let mut original = Sources::new(&root, b.clone(), c.clone(), 64 << 20).unwrap();
        let summary = parent(&mut original, false).unwrap();
        assert_eq!(summary.root["schema_version"], json::integer(5));
        assert_eq!(summary.root["qualified"], Value::Bool(false));
        assert_eq!(summary.root["npz_validated"], Value::Bool(false));
        assert!(summary.root["generations"].array().unwrap().len() >= 2);
        let steps = summary.root["generations"][1]["optimizer_steps"]
            .integer(1, i32::MAX as i128)
            .unwrap() as usize;
        let files =
            verify_derived_artifact(&mut original, &summary.root["state"], 0, steps).unwrap();
        assert_eq!(files, summary.root["generations"][1]["artifact_files"]);
        assert!(
            verify_derived_artifact(&mut original, &summary.root["state"], 0, steps + 1).is_err()
        );
        original.recheck().unwrap();
        drop(original);
        let plan = json::object([
            ("schema_version", json::integer(5)),
            ("mode", json::text("historical_parent")),
            (
                "parents",
                json::array([json::object([
                    ("directory", json::text(root.to_str().unwrap())),
                    ("require_complete", Value::Bool(false)),
                ])]),
            ),
            ("collections", json::array([])),
        ]);
        let wire = json::canonical(&plan, &b, &c).unwrap();
        let mut inputs = crate::structured_source::OriginalInputs::open5(
            &wire,
            None,
            b.clone(),
            c.clone(),
            64 << 20,
            8 << 20,
            1 << 30,
            64 << 20,
        )
        .unwrap();
        let pending = inputs.controls().unwrap();
        assert_eq!(pending.summary.root["schema_version"], json::integer(5));
        assert_eq!(
            pending.summary.root["kind"],
            json::text("card-instance-hpo-pending-controls")
        );
        for collection in pending.summary.root["collections"].array().unwrap() {
            assert!(collection.get("source_proof").is_none());
            assert!(collection.get("correspondence").is_none());
            assert_eq!(collection["episode_counts"].array().unwrap().len(), 8);
            assert_eq!(collection["episode_digests"].array().unwrap().len(), 8);
        }
        inputs.recheck().unwrap();
    }

    #[test]
    #[ignore = "bounded private complete CPU originals; requires GFB_HPO_NATIVE_TEST_PARENT"]
    fn source5_two_stage_authorization_freezes_selection_and_refuses_stale_phase() {
        let root = std::path::PathBuf::from(std::env::var("GFB_HPO_NATIVE_TEST_PARENT").unwrap());
        let b = Budget::new(1usize << 30);
        let c = Control::new(crate::budget::monotonic() + 120.).unwrap();
        let plan = json::object([
            ("schema_version", json::integer(5)),
            ("mode", json::text("historical_parent")),
            (
                "parents",
                json::array([json::object([
                    ("directory", json::text(root.to_str().unwrap())),
                    ("require_complete", Value::Bool(false)),
                ])]),
            ),
            ("collections", json::array([])),
        ]);
        let encoded = json::canonical(&plan, &b, &c).unwrap();
        let mut session = crate::structured_session::CohortSession::open5(
            &encoded,
            None,
            b.clone(),
            c.clone(),
            64 << 20,
            8 << 20,
            1 << 30,
            64 << 20,
        )
        .unwrap();
        let pending = json::parse(&session.summary().unwrap(), &b, &c).unwrap();
        assert!(session.export().is_err());
        let mut collections = Vec::new();
        for raw in pending.root["collections"].array().unwrap() {
            let index = raw["collection_index"]
                .integer(0, i64::MAX as i128)
                .unwrap() as usize;
            let original = session
                .inputs
                .committed
                .iter()
                .find(|row| row.collection_index == index)
                .unwrap();
            let selection = original.update.root["source_proof"]["selection"].clone();
            let mut control = raw["control"].clone();
            control["selection"] = selection.clone();
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
            collections.push(json::object([("collection_index",json::integer(index as i128)),("selection",selection),
                ("training_recipe",session.inputs.parents[0].state.root["effective_config"]["hpo_spec"]["training_recipe"].clone()),
                ("expected_optimizer_step_delta",original.update.root["expected_optimizer_step_delta"].clone()),
                ("expected_correspondence",correspondence)]));
        }
        let request = json::object([
            ("schema_version", json::integer(5)),
            ("pending_sha256", json::text(&session.controls.fingerprint)),
            (
                "parent",
                session.controls.authorization.root["parent"].clone(),
            ),
            ("collections", json::array(collections)),
        ]);
        let request = json::canonical(&request, &b, &c).unwrap();
        let final_wire = session.authorize_hpo(&request).unwrap();
        let final_summary = json::parse(&final_wire, &b, &c).unwrap();
        assert_eq!(
            final_summary.root["kind"],
            json::text("card-instance-hpo-source-controls")
        );
        for raw in final_summary.root["collections"].array().unwrap() {
            assert_eq!(raw["source_proof"]["schema_version"], json::integer(3));
            assert_eq!(raw["source_proof"]["qualified"], Value::Bool(false));
            assert_eq!(raw["selection"], raw["source_proof"]["selection"]);
        }
        assert!(session.authorize_hpo(&request).is_err());
        session.close().unwrap();
    }

    #[test]
    fn generation4_binds_original_maps_counts_and_decision_pool_controls() {
        let b = Budget::new(128 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let all = json::parse(
            include_bytes!("../tests/fixtures/hpo_metadata5_rules_v2.json"),
            &b,
            &c,
        )
        .unwrap();
        for case in all.root["cases"].array().unwrap() {
            let mut state = case["state"].clone();
            let zero = &case["generation"];
            assert_eq!(
                generation(zero, &state, 0, None, None, &b, &c).unwrap(),
                vec![0]
            );
            let mut next = zero.clone();
            next["generation"] = json::integer(1);
            next["parent_sha256"] = state["generation_sha256"].clone();
            next["update_sha256"] = json::text(&"b".repeat(64));
            next["used_decisions"] = json::integer(3);
            next["accepted_decisions"] = json::integer(3);
            next["collected_decisions"] = json::integer(8);
            next["optimizer_steps"] = json::integer(
                optimizer_delta(&state["effective_config"]["hpo_spec"]["training_recipe"], 3)
                    .unwrap(),
            );
            next["parent_controls"]["collection_position"] = json::integer(1);
            next["parent_controls"]["snapshot_pool"] =
                json::array([json::integer(0), json::integer(1)]);
            state["used_decisions"] = json::integer(3);
            state["accepted_decisions"] = json::integer(3);
            state["collected_decisions"] = json::integer(8);
            state["optimizer_steps"] = next["optimizer_steps"].clone();
            assert_eq!(
                generation(
                    &next,
                    &state,
                    1,
                    Some(zero),
                    Some(state["generation_sha256"].string().unwrap()),
                    &b,
                    &c
                )
                .unwrap(),
                vec![0, 1]
            );
            next["parent_controls"]["snapshot_pool"] = json::array([json::integer(0)]);
            assert!(
                generation(
                    &next,
                    &state,
                    1,
                    Some(zero),
                    Some(state["generation_sha256"].string().unwrap()),
                    &b,
                    &c
                )
                .is_err()
            );
        }
    }

    #[test]
    fn run5_spec_and_original_clocks_schema_are_closed_without_legacy_caps() {
        let b = Budget::new(128 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let all = json::parse(
            include_bytes!("../tests/fixtures/hpo_metadata5_rules_v2.json"),
            &b,
            &c,
        )
        .unwrap();
        for case in all.root["cases"].array().unwrap() {
            let state = &case["state"];
            spec(state, &b, &c).unwrap();
            let model = run(state, &b, &c).unwrap();
            assert_eq!(model["model_id"], state["model_id"]);
            assert!(
                state["effective_config"]["run"]["wall_seconds"]
                    .number()
                    .unwrap()
                    > 7200.
            );
            let mut higher = state.clone();
            higher["effective_config"]["hpo_spec"]["operating"]["managed_cuda_bytes"] =
                json::integer(20i128 << 30);
            higher["effective_config"]["run"]["global_cuda_budget_bytes"] =
                json::integer(20i128 << 30);
            reseal_run(&mut higher, &b, &c);
            assert!(run(&higher, &b, &c).is_ok());
            let mut broken = state.clone();
            broken["effective_config"]["hpo_spec"]["deadline_inputs"]["base_admission_seconds"] =
                json::integer(7200);
            reseal_run(&mut broken, &b, &c);
            assert!(spec(&broken, &b, &c).is_err());
            broken = state.clone();
            broken["wall_deadline_unix"] =
                json::float(broken["wall_deadline_unix"].number().unwrap() + 1.).unwrap();
            assert!(run(&broken, &b, &c).is_err());
            broken = state.clone();
            broken["schema_version"] = json::integer(4);
            assert!(run(&broken, &b, &c).is_err());
            broken = state.clone();
            broken["stop_reason"] = json::text("pause");
            assert!(run(&broken, &b, &c).is_err());
            broken = state.clone();
            broken["effective_config"]["server"]["external_url"] =
                json::text("http://external.invalid");
            reseal_run(&mut broken, &b, &c);
            assert!(run(&broken, &b, &c).is_err());
            broken = state.clone();
            broken["effective_config"]["server"]["simulator_revision"] =
                json::text("unrecorded-source");
            reseal_run(&mut broken, &b, &c);
            assert!(run(&broken, &b, &c).is_err());
            broken = state.clone();
            broken["identity"]["integration"]["foreign"] = Value::Bool(true);
            reseal_run(&mut broken, &b, &c);
            assert!(run(&broken, &b, &c).is_err());
        }
    }

    #[test]
    fn hpo_configurable_game_and_worker_capacity_preserves_projection() {
        let b = Budget::new(128 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let all = json::parse(include_bytes!("../tests/fixtures/hpo_metadata5_rules_v2.json"), &b, &c).unwrap();
        for case in all.root["cases"].array().unwrap() {
            let mut state = case["state"].clone();
            state["effective_config"]["hpo_spec"]["operating"]["game_concurrency"] = json::integer(4);
            state["effective_config"]["hpo_spec"]["operating"]["max_participant_workers"] = json::integer(8);
            state["effective_config"]["run"]["concurrency"] = json::integer(4);
            state["effective_config"]["run"]["max_participant_workers"] = json::integer(8);
            reseal_run(&mut state, &b, &c);
            assert!(run(&state, &b, &c).is_ok());
            state["effective_config"]["run"]["max_participant_workers"] = json::integer(4);
            reseal_run(&mut state, &b, &c);
            assert!(run(&state, &b, &c).is_err());
            state["effective_config"]["hpo_spec"]["operating"]["game_concurrency"] = json::integer(i64::MAX as i128 / 2);
            state["effective_config"]["hpo_spec"]["operating"]["max_participant_workers"] = json::integer(i64::MAX as i128);
            reseal_run(&mut state, &b, &c);
            assert!(spec(&state, &b, &c).is_ok());
            state["effective_config"]["hpo_spec"]["operating"]["game_concurrency"] = json::integer(i64::MAX as i128);
            reseal_run(&mut state, &b, &c);
            assert!(spec(&state, &b, &c).is_err());
        }
    }

    #[test]
    fn resident_run5_spec3_is_explicit_and_preserves_old_native_contract() {
        let b = Budget::new(128 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let all = json::parse(include_bytes!("../tests/fixtures/hpo_metadata5_rules_v2.json"), &b, &c).unwrap();
        for case in all.root["cases"].array().unwrap() {
            let old = case["state"].clone();
            let mut state = old.clone();
            state["run_variant"] = json::text("hpo-ppo-resident-v1");
            state["effective_config"]["hpo_spec"]["schema_version"] = json::integer(3);
            state["effective_config"]["hpo_spec"]["runtime"] = json::object([
                ("mode", json::text("resident-worker-v1")),
                ("cache_entries", json::integer(6)),
                ("cache_weight_bytes", json::integer(134217728)),
            ]);
            reseal_run(&mut state, &b, &c);
            assert!(run(&state, &b, &c).is_ok());
            state["effective_config"]["hpo_spec"]["runtime"]["cache_entries"] = json::integer(7);
            reseal_run(&mut state, &b, &c);
            assert!(run(&state, &b, &c).is_err());
            assert!(run(&old, &b, &c).is_ok());
        }
        assert_eq!(CAPABILITY, "model4;collection6;proof3;numeric5;selection1;provider1");
    }

    fn reseal_run(state: &mut Value, b: &Arc<Budget>, c: &Control) {
        state["hpo_spec_sha256"] =
            json::text(&digest(&state["effective_config"]["hpo_spec"], b, c).unwrap());
        state["config_sha256"] = json::text(&digest(&state["effective_config"], b, c).unwrap());
        state["resume_identity"] = json::text(
            &digest(
                &json::object([
                    ("version", json::text("hpo-ppo-resume-v1")),
                    ("config_sha256", state["config_sha256"].clone()),
                    ("identity", state["identity"].clone()),
                    ("execution", state["execution"].clone()),
                    (
                        "classification",
                        clone_fields(state, GAME_CLASS, c).unwrap(),
                    ),
                    (
                        "storage_binding",
                        state.get("storage_binding").cloned().unwrap_or(Value::Null),
                    ),
                ]),
                b,
                c,
            )
            .unwrap(),
        );
    }

    #[test]
    fn source5_formats_are_explicit_and_old_semantic_sets_stay_exact() {
        assert!(SourceFormat::ResidentHpo.hpo() && SourceFormat::ResidentHpo.resident());
        assert!(SourceFormat::PackedResidentHpo.hpo() && SourceFormat::PackedResidentHpo.packed());
        assert!(!SourceFormat::Hpo.resident());
        assert_eq!(resident::CAPABILITY,"collection7;proof4;registry5;pool1");
        assert!(SourceFormat::Hpo.hpo());
        assert!(SourceFormat::Hpo.corrected());
        assert!(!SourceFormat::Hpo.packed());
        assert!(SourceFormat::PackedHpo.hpo() && SourceFormat::PackedHpo.packed());
        assert!(!SourceFormat::Corrected.hpo());
        assert!(!SourceFormat::PackedCorrected.hpo());
        assert!(
            SourceFormat::Corrected
                .semantics()
                .contains(&"reward_policy_version")
        );
        assert!(
            !SourceFormat::Hpo
                .semantics()
                .contains(&"reward_policy_version")
        );
        assert_eq!(
            SourceFormat::Hpo.semantics(),
            SourceFormat::PackedHpo.semantics()
        );
    }

    #[test]
    fn inference5_and_options4_keep_pure_and_execution_models_distinct() {
        let b = Budget::new(64 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let cases = json::parse(
            include_bytes!("../tests/fixtures/card_instance_models3_rules_v2.json"),
            &b,
            &c,
        )
        .unwrap();
        for case in cases.root.require("cases").unwrap().array().unwrap() {
            let mut model = case.require("model").unwrap().clone();
            model["version"] = json::text("4");
            model["initialization"] = json::text("orthogonal-activation-aware-v1");
            model["temporal"]
                .as_object_mut()
                .unwrap()
                .remove("sequence_length");
            model["temporal"]["max_decisions"] = json::integer(1024);
            let mut manifest = case.require("manifest").unwrap().clone();
            manifest["schema_version"] = json::integer(5);
            manifest["model"] = model.clone();
            manifest["model"]
                .as_object_mut()
                .unwrap()
                .remove("resources");
            manifest
                .as_object_mut()
                .unwrap()
                .remove("reward_policy_version");
            manifest["architecture_identity"] = json::text(&architecture(&model, &b, &c).unwrap());
            inference(&manifest, case.require("actions").unwrap(), &b, &c).unwrap();
            let mut raw = case.require("neural").unwrap().clone();
            let mut bytes = json::canonical(&manifest, &b, &c).unwrap();
            bytes.push(b'\n').unwrap();
            let sha = format!("{:x}", Sha256::digest(&*bytes));
            raw["inference_manifest"] = manifest.clone();
            raw["policy_options"]["version"] = json::text("4");
            raw["policy_options"]["model"] = model;
            raw["policy_options"]["expected_manifest_digest"] = json::text(&sha);
            for file in raw["files"].as_array_mut().unwrap() {
                if file["path"].as_str() == Some("inference.json") {
                    file["sha256"] = json::text(&sha);
                }
            }
            let label = neural_metadata(&raw, case.require("actions").unwrap(), &b, &c).unwrap();
            assert!(label.contains("-ppo-v4-"));
            let journals = json::parse(
                include_bytes!("../tests/fixtures/card_instance_journal6_rules_v2.json"),
                &b,
                &c,
            )
            .unwrap();
            let mut header = journals.root["journals"][0]["rows"][0].clone();
            header["model_version"] = json::text("4");
            header["model_id"] = raw["policy_options"]["model"]["model_id"].clone();
            header["policy"] = json::text(&label);
            header["neural"] = raw.clone();
            header["actions"] = case["actions"].clone();
            header["action_config_hash"] = json::text(&digest(&case["actions"], &b, &c).unwrap());
            header["aggregate_window"] =
                raw["policy_options"]["model"]["event"]["aggregate_window"].clone();
            header["token_signature"] = raw["inference_manifest"]["token_signature"].clone();
            let (caps, costs) = worker_settings3(
                &raw["policy_options"]["model"],
                &raw["policy_options"]["inference"],
                &case["actions"],
                &b,
                &c,
            )
            .unwrap();
            header["caps"] = caps;
            header["numeric_costs"] = costs;
            crate::structured_journal::card_instance::header6(&header, &b, &c).unwrap();
            let mut bad_header = header;
            bad_header["model_version"] = json::text("3");
            assert!(
                crate::structured_journal::card_instance::header6(&bad_header, &b, &c).is_err()
            );
            assert!(inference4(&manifest, case.require("actions").unwrap(), &b, &c).is_err());
            let mut broken = manifest.clone();
            broken["model"]["resources"] = raw["policy_options"]["model"]["resources"].clone();
            assert!(inference(&broken, case.require("actions").unwrap(), &b, &c).is_err());
            broken = manifest;
            broken["reward_policy_version"] = json::text("natural-terminal-v1");
            assert!(inference(&broken, case.require("actions").unwrap(), &b, &c).is_err());
        }
    }

    #[test]
    fn registry4_file_receipt_preserves_uncertain_and_exact_owned_identity() {
        let sha = "a".repeat(64);
        let identity = json::array((1..=5).map(json::integer));
        let intent = json::object([
            ("operation_id", json::text("owned-operation")),
            (
                "staging_basename",
                json::text(&format!("prune-{}", "b".repeat(32))),
            ),
            ("sha256", json::text(&sha)),
            ("original_identity", identity.clone()),
        ]);
        let completed = json::object([
            ("operation_id", json::text("owned-operation")),
            ("sha256", json::text(&sha)),
            ("original_identity", identity),
            ("completed_unix", json::float(1.).unwrap()),
        ]);
        let mut file = json::object([
            ("sha256", json::text(&sha)),
            ("status", json::text("retained")),
            ("deletion_intent", Value::Null),
            ("deletion_receipt", Value::Null),
        ]);
        assert_eq!(file_receipt(&file, &sha).unwrap(), "retained");
        file["status"] = json::text("deleting");
        file["deletion_intent"] = intent;
        assert_eq!(file_receipt(&file, &sha).unwrap(), "deleting");
        file["status"] = json::text("deleted");
        assert!(file_receipt(&file, &sha).is_err());
        file["deletion_receipt"] = completed;
        assert_eq!(file_receipt(&file, &sha).unwrap(), "deleted");
        file["deletion_receipt"]["original_identity"][4] = json::integer(999);
        assert!(file_receipt(&file, &sha).is_err());
        file["status"] = json::text("failed");
        assert!(file_receipt(&file, &sha).is_err());
        file["deletion_receipt"] = Value::Null;
        assert_eq!(file_receipt(&file, &sha).unwrap(), "failed");
    }

    #[test]
    fn full_training_recipe_and_configurable_step_arithmetic_are_closed() {
        let b = Budget::new(4 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let value = json::parse(br#"{"version":"hpo-ppo-v1","algorithm":"ppo","optimizer":"adam","reward":"natural-terminal-v1","gamma":1.0,"collection_games":24,"adam_learning_rate":0.002,"adam_beta1":0.8,"adam_beta2":0.98,"adam_epsilon":0.0000001,"epochs":3,"minibatch_size":2,"clip_range":0.1,"value_coefficient":0.3,"entropy_coefficient":0.02,"gae_lambda":0.6,"max_grad_norm":1.0,"tbptt_length":8}"#,&b,&c).unwrap();
        training(&value.root).unwrap();
        assert_eq!(optimizer_delta(&value.root, 5).unwrap(), 9);
        assert_eq!(optimizer_delta(&value.root, 0).unwrap(), 0);
        for (key, bad) in [
            ("gamma", json::float(0.99).unwrap()),
            ("collection_games", json::integer(10)),
            ("epochs", Value::Bool(true)),
            ("clip_range", json::integer(1)),
            ("adam_beta1", json::integer(1)),
            ("gae_lambda", json::integer(-1)),
            ("tbptt_length", json::integer(0)),
        ] {
            let mut broken = value.root.clone();
            broken[key] = bad;
            assert!(training(&broken).is_err(), "{key}");
        }
        let mut broken = value.root.clone();
        broken["device"] = json::text("cuda:0");
        assert!(training(&broken).is_err());
    }

    #[test]
    fn closed_selection_validates_original_counts_without_choosing_subset() {
        let b = Budget::new(16 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let context = json::parse(br#"{"selection_version":"hpo-endpoint-selection-v1","decision_budget":2,"used_before":0,"remaining":2,"learning_seed":17,"batch":1000000}"#, &b, &c).unwrap();
        let counts = [3usize, 0, 4];
        let seed = u128::from_str_radix(
            &digest(
                &json::object([
                    ("recipe", json::text("hpo-ppo-selection-seed-v1")),
                    ("context", context.root.clone()),
                ]),
                &b,
                &c,
            )
            .unwrap()[..32],
            16,
        )
        .unwrap();
        let indices = json::array([json::integer(0), json::integer(6)]);
        let mut value = json::object([
            ("context", context.root.clone()),
            (
                "derived_seed",
                Value::Number(json::Number::from_u128(seed).unwrap()),
            ),
            (
                "episode_counts",
                json::array(counts.iter().map(|v| json::integer(*v as i128))),
            ),
            ("collected_decisions", json::integer(7)),
            ("used_decisions", json::integer(2)),
            ("indices", indices.clone()),
            (
                "indices_sha256",
                json::text(&digest(&indices, &b, &c).unwrap()),
            ),
        ]);
        assert_eq!(
            selection(&value, &counts, &context.root, &b, &c).unwrap(),
            2
        );
        // Structural support accepts an alternative valid subset; independently
        // replayed Python sampling is the authority for which subset is chosen.
        value["indices"] = json::array([json::integer(2), json::integer(3)]);
        value["indices_sha256"] = json::text(&digest(&value["indices"], &b, &c).unwrap());
        assert!(selection(&value, &counts, &context.root, &b, &c).is_ok());
        value["episode_counts"] =
            json::array([json::integer(2), json::integer(0), json::integer(5)]);
        assert!(selection(&value, &counts, &context.root, &b, &c).is_err());
        value["episode_counts"] = json::array(counts.iter().map(|v| json::integer(*v as i128)));
        value["indices"] = json::array([json::integer(3), json::integer(2)]);
        value["indices_sha256"] = json::text(&digest(&value["indices"], &b, &c).unwrap());
        assert!(selection(&value, &counts, &context.root, &b, &c).is_err());
        value["derived_seed"] = Value::Bool(true);
        assert!(selection(&value, &counts, &context.root, &b, &c).is_err());
        value["derived_seed"] = Value::Number(json::Number::from_u128(seed).unwrap());
        value["episode_counts"] =
            json::array([json::integer(0), json::integer(0), json::integer(0)]);
        value["collected_decisions"] = json::integer(0);
        value["used_decisions"] = json::integer(0);
        value["indices"] = json::array([]);
        value["indices_sha256"] = json::text(&digest(&value["indices"], &b, &c).unwrap());
        assert_eq!(
            selection(&value, &[0, 0, 0], &context.root, &b, &c).unwrap(),
            0
        );
    }
}
