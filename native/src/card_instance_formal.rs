//! Closed formal Run3 control validation, never eligibility or learner authority.
//! Original sidecar ownership stays in the SAME parent Sources traversal.
use crate::{
    budget::{Budget, Control},
    error::{Error, Result},
    json::{self, Document, Value, ValueExt},
    structured_controls::{checksum, digest},
    structured_session::Sources,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;
pub(crate) const VARIANT: &str = "formal-parent-v1";
pub(crate) const EXTRA: &[&str] = &["run_variant", "formal_declaration", "formal_stop"];
pub(crate) const RECIPE_SHA: &str =
    "871fc8d4591e62d302bb2f79d98d474da15cbdb19c86506e5a70557f4ecd67b2";
const ALGORITHM_SHA: &str = "318eab51e51ab242c629111634ff25e986c1a0a0ece0d246694fd72a27c90cc9";
const RECIPE_ID: &str = "card-instance-formal-ppo-v1";
const ASSET: &[u8] = include_bytes!("../assets/card-instance-formal-ppo-v1.json");
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::contract(message))
    }
}
fn exact(a: &Value, b: &Value, budget: &Arc<Budget>, control: &Control) -> Result<bool> {
    Ok(&*json::canonical(a, budget, control)? == &*json::canonical(b, budget, control)?)
}
pub(crate) fn discriminator(state: &Value) -> Result<bool> {
    match state.get("run_variant") {
        Some(value) => {
            require(value.string()? == VARIANT, "unknown formal Run3 variant")?;
            Ok(true)
        }
        None => Ok(false),
    }
}
fn recipe(b: &Arc<Budget>, c: &Control) -> Result<Document> {
    require(
        format!("{:x}", Sha256::digest(ASSET)) == RECIPE_SHA,
        "formal literal recipe byte SHA differs",
    )?;
    let doc = json::parse(ASSET, b, c)?;
    let r = &doc.root;
    require(
        r.require("schema_version")?.integer(1, 1)? == 1
            && r.require("kind")?.string()? == "card-instance-formal-ppo-recipe"
            && r.require("recipe_id")?.string()? == RECIPE_ID,
        "formal recipe discriminator differs",
    )?;
    let mut algorithm = json::copy(r.require("algorithm_recipe")?, c)?;
    require(
        algorithm.require("content_sha256")?.string()? == ALGORITHM_SHA,
        "formal algorithm reference differs",
    )?;
    algorithm.as_object_mut().unwrap().remove("content_sha256");
    require(
        digest(&algorithm, b, c)? == ALGORITHM_SHA,
        "formal algorithm content differs",
    )?;
    Ok(doc)
}
fn checked_config(config: &Value, recipe: &Value, b: &Arc<Budget>, c: &Control) -> Result<Value> {
    let id = config.require("model")?.require("model_id")?.string()?;
    let profile = recipe
        .require("model_profiles")?
        .get(id)
        .ok_or_else(|| Error::contract("formal model profile differs"))?;
    let device = config
        .require("model")?
        .require("inference")?
        .require("device")?;
    let backend = config.require("preprocessing")?.require("backend")?;
    require(
        recipe
            .require("allowed_inference_devices")?
            .array()?
            .contains(device)
            && recipe
                .require("allowed_reader_backends")?
                .array()?
                .contains(backend),
        "formal backend/device differs",
    )?;
    let seeds = config.require("training")?.require("seeds")?.array()?;
    require(seeds.len() == 1, "one formal seed slot required")?;
    let seed = seeds[0].integer(0, i64::MAX as i128)?;
    require([17, 29, 43].contains(&seed), "formal seed slot differs")?;
    let mut expected = json::copy(recipe.require("base_config")?, c)?;
    expected["model"] = json::copy(profile, c)?;
    expected["model"]["inference"]["device"] = json::copy(device, c)?;
    expected["preprocessing"]["backend"] = json::copy(backend, c)?;
    expected["training"]["seeds"] = json::array([json::integer(seed)]);
    for (section, key) in [
        ("run", "name"),
        ("run", "output_dir"),
        ("server", "source_dir"),
        ("server", "binary"),
        ("server", "config"),
    ] {
        let value = config.require(section)?.require(key)?;
        require(
            !value.string()?.is_empty(),
            "formal nonempty path/name required",
        )?;
        expected[section][key] = json::copy(value, c)?;
    }
    require(
        exact(config, &expected, b, c)?,
        "formal full literal Config differs",
    )?;
    Ok(expected)
}
fn execution(config: &Value, c: &Control) -> Result<Value> {
    let inference = json::copy(config.require("model")?.require("inference")?, c)?;
    let mut learner = json::copy(&inference, c)?;
    learner["device"] = json::copy(config.require("training")?.require("device")?, c)?;
    let mut result = json::object([]);
    for (role, binding) in [("learner", learner), ("inference", inference)] {
        let visibility = if binding.require("device")?.string()? == "cpu" {
            json::text("")
        } else {
            json::copy(config.require("model")?.require("cuda_visible_devices")?, c)?
        };
        result.as_object_mut().unwrap().insert(
            role.into(),
            json::object([("binding", binding), ("cuda_visible_devices", visibility)]),
        );
    }
    Ok(result)
}
pub(crate) fn reference(v: &Value) -> Result<()> {
    v.closed(
        &[
            "relative_path",
            "sha256",
            "recipe_id",
            "recipe_sha256",
            "algorithm_recipe_sha256",
            "seed_slot",
        ],
        &[],
    )?;
    checksum(v.require("sha256")?)?;
    require(
        v.require("relative_path")?.string()? == "formal-declaration.json"
            && v.require("recipe_id")?.string()? == RECIPE_ID
            && checksum(v.require("recipe_sha256")?)? == RECIPE_SHA
            && checksum(v.require("algorithm_recipe_sha256")?)? == ALGORITHM_SHA
            && [17, 29, 43].contains(&v.require("seed_slot")?.integer(0, i64::MAX as i128)?),
        "formal declaration reference differs",
    )
}
fn declaration(
    v: &Value,
    state: &Value,
    recipe: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<()> {
    v.closed(
        &[
            "schema_version",
            "kind",
            "recipe_id",
            "recipe_sha256",
            "algorithm_recipe_sha256",
            "seed_slot",
            "config_sha256",
            "config_inputs_sha256",
            "input_binding",
            "execution_identity_sha256",
            "identity_sha256",
        ],
        &[],
    )?;
    let cfg = checked_config(state.require("effective_config")?, recipe, b, c)?;
    let reference = state.require("formal_declaration")?;
    let identity = state.require("identity")?;
    require(
        exact(identity.require("formal_parent")?, reference, b, c)?,
        "formal SAME run/identity declaration required",
    )?;
    let mut base = json::copy(identity, c)?;
    base.as_object_mut().unwrap().remove("formal_parent");
    let exec = execution(&cfg, c)?;
    let config_sha = digest(&cfg, b, c)?;
    let inputs_sha = digest(state.require("config_inputs")?, b, c)?;
    let wanted = json::object([
        ("schema_version", json::integer(1)),
        (
            "kind",
            json::text("card-instance-formal-parent-declaration"),
        ),
        ("recipe_id", json::text(RECIPE_ID)),
        ("recipe_sha256", json::text(RECIPE_SHA)),
        ("algorithm_recipe_sha256", json::text(ALGORITHM_SHA)),
        ("seed_slot", json::copy(&cfg["training"]["seeds"][0], c)?),
        ("config_sha256", json::text(&config_sha)),
        ("config_inputs_sha256", json::text(&inputs_sha)),
        (
            "input_binding",
            json::copy(recipe.require("input_binding")?, c)?,
        ),
        (
            "execution_identity_sha256",
            json::text(&digest(&exec, b, c)?),
        ),
        ("identity_sha256", json::text(&digest(&base, b, c)?)),
    ]);
    require(
        exact(v, &wanted, b, c)?
            && exact(
                reference.require("seed_slot")?,
                wanted.require("seed_slot")?,
                b,
                c,
            )?
            && exact(state.require("execution")?, &exec, b, c)?
            && state.require("config_sha256")?.string()? == config_sha
            && base.require("configuration_inputs_sha256")?.string()? == inputs_sha
            && base
                .require("integration")?
                .require("server_config_sha256")?
                == recipe.require("server_config_sha256")?,
        "formal declaration/base identity/config/execution differs",
    )
}
fn finite(v: &Value) -> Result<f64> {
    let value = v.number()?;
    require(value.is_finite(), "formal finite clock required")?;
    Ok(value)
}
// Preserve Python's numeric type and operation order. Normal runtime anchors
// are binary64; integer-controlled fixtures retain integer arithmetic/echoes.
pub(crate) fn arithmetic(a: &Value, b: &Value, subtract: bool) -> Result<Value> {
    if a.as_number().is_some_and(json::integer_token)
        && b.as_number().is_some_and(json::integer_token)
    {
        let x = a.integer(i128::MIN, i128::MAX)?;
        let y = b.integer(i128::MIN, i128::MAX)?;
        let result = if subtract {
            x.checked_sub(y)
        } else {
            x.checked_add(y)
        }
        .ok_or_else(|| Error::contract("formal integer clock overflow"))?;
        Ok(json::integer(result))
    } else {
        let (x, y) = (finite(a)?, finite(b)?);
        json::float(if subtract { x - y } else { x + y })
    }
}
fn add(a: &Value, b: &Value) -> Result<Value> {
    arithmetic(a, b, false)
}
fn sub(a: &Value, b: &Value) -> Result<Value> {
    arithmetic(a, b, true)
}
fn stop_expected(state: &Value, raw: &Value, b: &Arc<Budget>, c: &Control) -> Result<Value> {
    let clock = raw.require("clock")?;
    clock.closed(
        &[
            "invocation_started_unix",
            "invocation_started_monotonic",
            "observed_monotonic",
            "admission_deadline_monotonic",
            "execution_deadline_monotonic",
            "wall_deadline_monotonic",
        ],
        &[],
    )?;
    let u = clock.require("invocation_started_unix")?;
    let m = clock.require("invocation_started_monotonic")?;
    let o = clock.require("observed_monotonic")?;
    let mut wanted_clock = json::object([
        ("invocation_started_unix", json::copy(u, c)?),
        ("invocation_started_monotonic", json::copy(m, c)?),
        ("observed_monotonic", json::copy(o, c)?),
    ]);
    for (target, key) in [
        ("admission_deadline_monotonic", "admission_deadline_unix"),
        ("execution_deadline_monotonic", "execution_deadline_unix"),
        ("wall_deadline_monotonic", "wall_deadline_unix"),
    ] {
        wanted_clock
            .as_object_mut()
            .unwrap()
            .insert(target.into(), sub(&add(m, state.require(key)?)?, u)?);
    }
    for value in wanted_clock.object()?.values() {
        finite(value)?;
    }
    let train = state.require("effective_config")?.require("training")?;
    let owned = raw.require("owned_bytes")?.integer(0, u64::MAX as i128)?;
    let maximum = train
        .require("max_owned_bytes")?
        .integer(10i128 << 30, 10i128 << 30)?;
    let reserve = train
        .require("admission_reservation_bytes")?
        .integer(1i128 << 30, 1i128 << 30)?;
    let updates = train.require("max_updates")?.integer(1000, 1000)?;
    require(
        finite(o)? < finite(wanted_clock.require("execution_deadline_monotonic")?)?
            && finite(o)? < finite(wanted_clock.require("wall_deadline_monotonic")?)?
            && owned <= maximum,
        "formal execution/wall/actual storage overrun cannot complete",
    )?;
    require(
        finite(u)? >= finite(state.require("created_unix")?)?
            && finite(m)? >= 0.
            && finite(o)? >= finite(m)?,
        "formal original invocation clock invalid",
    )?;
    let generation = state.require("updates")?.integer(0, 1000)?;
    let reason = if generation >= updates {
        "max_updates"
    } else if finite(o)? >= finite(wanted_clock.require("admission_deadline_monotonic")?)? {
        "admission_deadline"
    } else if owned + reserve > maximum {
        "storage_admission"
    } else {
        return Err(Error::contract("formal complete has no normal stop"));
    };
    require(generation > 0, "formal zero-update completion ineligible")?;
    let elapsed = sub(&add(u, &sub(o, m)?)?, state.require("created_unix")?)?;
    let now = add(u, &sub(o, m)?)?;
    require(
        finite(&elapsed)? >= 0.
            && finite(state.require("last_unix")?)? >= finite(&sub(&now, &json::float(0.1)?)?)?
            && state.require("completed_updates")?.integer(0, 1000)? == generation
            && state
                .require("optimizer_steps")?
                .integer(1, i64::MAX as i128)?
                > 0,
        "formal positive committed prefix/original elapsed differs",
    )?;
    let _copy = b.reserve(65536)?;
    Ok(json::object([
        ("schema_version", json::integer(1)),
        ("kind", json::text("card-instance-formal-stop")),
        ("recipe_id", json::text(RECIPE_ID)),
        ("recipe_sha256", json::text(RECIPE_SHA)),
        (
            "declaration_sha256",
            json::copy(state.require("formal_declaration")?.require("sha256")?, c)?,
        ),
        ("reason", json::text(reason)),
        ("generation", json::integer(generation)),
        ("clock", wanted_clock),
        ("elapsed_seconds", elapsed),
        ("owned_bytes", json::integer(owned)),
        ("admission_reservation_bytes", json::integer(reserve)),
        ("max_owned_bytes", json::integer(maximum)),
        ("max_updates", json::integer(updates)),
    ]))
}
pub(crate) fn validate_values(
    state: &Value,
    declared: &Value,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<()> {
    require(discriminator(state)?, "explicit formal variant required")?;
    reference(state.require("formal_declaration")?)?;
    let recipe = recipe(b, c)?;
    declaration(declared, state, &recipe.root, b, c)?;
    match state.require("status")?.string()? {
        "running" | "paused" => require(
            state.require("formal_stop")?.is_null(),
            "running/paused formal stop must be null",
        ),
        "complete" => {
            let raw = state.require("formal_stop")?;
            raw.closed(
                &[
                    "schema_version",
                    "kind",
                    "recipe_id",
                    "recipe_sha256",
                    "declaration_sha256",
                    "reason",
                    "generation",
                    "clock",
                    "elapsed_seconds",
                    "owned_bytes",
                    "admission_reservation_bytes",
                    "max_owned_bytes",
                    "max_updates",
                ],
                &[],
            )?;
            require(
                exact(raw, &stop_expected(state, raw, b, c)?, b, c)?,
                "formal stop reason/clock/usage arithmetic differs",
            )
        }
        _ => Err(Error::contract("healthy formal state required")),
    }
}
pub(crate) fn validate(s: &mut Sources, state: &Value) -> Result<()> {
    let reference = state.require("formal_declaration")?;
    self::reference(reference)?;
    let sidecar = s.checked_json(
        &s.root.join("formal-declaration.json"),
        reference.require("sha256")?,
    )?;
    let budget = s.budget.clone();
    let control = s.control.clone();
    validate_values(state, &sidecar.root, &budget, &control)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixtures() -> (Arc<Budget>, Control, Document) {
        let b = Budget::new(128 * 1024 * 1024);
        let c = Control::new(crate::budget::monotonic() + 120.).unwrap();
        let doc = json::parse(
            include_bytes!("../tests/fixtures/card_instance_formal_controls_python.json"),
            &b,
            &c,
        )
        .unwrap();
        (b, c, doc)
    }
    #[test]
    fn formal_python_full_recipe_declaration_stop_clock_goldens_are_exact() {
        let (b, c, doc) = fixtures();
        for case in doc.root["cases"].array().unwrap() {
            validate_values(&case["state"], &case["declaration"], &b, &c)
                .unwrap_or_else(|error| panic!("{}: {error:?}", case["label"]));
        }
    }
    #[test]
    fn formal_discriminators_reject_ordinary_mixed_bool_and_unknown() {
        assert!(!discriminator(&json::object([])).unwrap());
        for value in [
            Value::Bool(true),
            Value::Null,
            json::text("ordinary"),
            json::integer(1),
        ] {
            assert!(discriminator(&json::object([("run_variant", value)])).is_err());
        }
    }
    #[test]
    fn formal_mutated_full_config_identity_reference_stop_and_counters_refuse() {
        let (b, c, doc) = fixtures();
        let case = &doc.root["cases"][0];
        for mutation in 0..26 {
            let mut state = case["state"].clone();
            let mut declared = case["declaration"].clone();
            match mutation {
                0 => state["run_variant"] = Value::Bool(true),
                1 => declared["schema_version"] = Value::Bool(true),
                2 => {
                    declared
                        .as_object_mut()
                        .unwrap()
                        .insert("qualified".into(), Value::Bool(true));
                }
                3 => state["identity"]["unrelated_nested"]["keep"] = Value::Bool(false),
                4 => state["formal_declaration"]["sha256"] = json::text(&"0".repeat(64)),
                5 => {
                    let mut base = state["identity"].clone();
                    base.as_object_mut().unwrap().remove("formal_parent");
                    base.as_object_mut().unwrap().remove("unrelated_nested");
                    declared["identity_sha256"] = json::text(&digest(&base, &b, &c).unwrap());
                }
                6 => state["effective_config"]["training"]["max_updates"] = json::integer(2),
                7 => {
                    state["effective_config"]["training"]["max_owned_bytes"] =
                        json::integer(2i128 << 30)
                }
                8 => {
                    state["effective_config"]["preprocessing"]["reader"] =
                        json::text("structured-v1")
                }
                9 => {
                    state["effective_config"]["model"]["inference"]["device"] = json::text("cuda:1")
                }
                10 => {
                    state["effective_config"]["training"]["seeds"] =
                        json::array([json::integer(17), json::integer(29)])
                }
                11 => state["effective_config"]["run"]["wall_seconds"] = json::integer(7199),
                12 => state["formal_stop"]["reason"] = json::text("admission_deadline"),
                13 => {
                    state["formal_stop"]["clock"]["observed_monotonic"] =
                        state["formal_stop"]["clock"]["execution_deadline_monotonic"].clone()
                }
                14 => state["formal_stop"]["elapsed_seconds"] = json::float(1.).unwrap(),
                15 => {
                    state["formal_stop"]["clock"]["admission_deadline_monotonic"] =
                        json::float(1.).unwrap()
                }
                16 => {
                    state["last_unix"] =
                        json::float(finite(&state["last_unix"]).unwrap() - 0.001).unwrap()
                }
                17 => state["updates"] = json::integer(0),
                18 => state["completed_updates"] = json::integer(999),
                19 => state["optimizer_steps"] = json::integer(0),
                20 => state["status"] = json::text("paused"),
                21 => state["formal_declaration"]["seed_slot"] = Value::Bool(true),
                22 => {
                    state["identity"]["configuration_inputs_sha256"] = json::text(&"0".repeat(64))
                }
                23 => declared["input_binding"]["purpose"] = json::text("component"),
                24 => state["formal_stop"]["owned_bytes"] = json::integer((10i128 << 30) + 1),
                _ => state["formal_stop"]["schema_version"] = Value::Bool(true),
            }
            assert!(
                validate_values(&state, &declared, &b, &c).is_err(),
                "mutation {mutation}"
            );
        }
    }
    #[test]
    fn formal_original_sidecar_sha_stat_is_retained_without_source_authority() {
        let (b, c, doc) = fixtures();
        let case = &doc.root["cases"][0];
        let root = std::env::temp_dir().join(format!(
            "gf-formal-control-only-{}-{}",
            std::process::id(),
            crate::budget::monotonic().to_bits()
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("formal-declaration.json");
        let bytes = case["declaration_utf8"].string().unwrap().as_bytes();
        std::fs::write(&path, bytes).unwrap();
        let mut sources = Sources::new(&root, b, c, 64 * 1024 * 1024).unwrap();
        validate(&mut sources, &case["state"]).unwrap();
        sources.recheck().unwrap();
        std::fs::write(&path, b"changed").unwrap();
        assert!(sources.recheck().is_err());
        let replacement = root.join("replacement");
        std::fs::write(&replacement, bytes).unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        assert!(sources.recheck().is_err());
        drop(sources);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn formal_cancel_and_low_budget_do_not_return_validation_success() {
        let (_, _, doc) = fixtures();
        let case = &doc.root["cases"][0];
        let b = Budget::new(1024);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        assert!(validate_values(&case["state"], &case["declaration"], &b, &c).is_err());
        assert_eq!(b.used(), 0);
        let b = Budget::new(16 * 1024 * 1024);
        c.cancel();
        assert_eq!(
            validate_values(&case["state"], &case["declaration"], &b, &c).unwrap_err(),
            Error::Cancelled
        );
    }
}
