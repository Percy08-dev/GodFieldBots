//! Exact metadata-only model2 parent validation. ZIP/NPY values remain Python S2.
use crate::{
    budget::{Budget, BudgetVec, Control},
    error::{Error, Result},
    json::{self, Document, Value, ValueExt},
    structured_session::Sources,
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path, sync::Arc};

#[path = "hpo_controls.rs"]
pub(crate) mod hpo;
#[path = "hpo_resident.rs"]
pub(crate) mod resident;
#[path = "card_instance_tree_raw.rs"]
pub(crate) mod tree3;

/// Closed curriculum options. This changes training identity only; no policy math.
pub(crate) fn refresh_training<'a>(cfg: &'a Value, b: &Arc<Budget>, c: &Control) -> Result<Option<&'a Value>> {
    let descriptor = if let Some(spec) = cfg.get("hpo_spec") {
        if cfg.get("training_opponents").is_some() {
            return Err(Error::contract("HPO opponents belong exclusively to spec2"));
        }
        match n(spec, "schema_version", 1, 3)? {
            1 => {
                if spec.get("training_opponents").is_some() {
                    return Err(Error::contract("HPO spec1 rejects training_opponents"));
                }
                return Ok(None);
            }
            3 if spec.get("training_opponents").is_none() => return Ok(None),
            _ => {
                n(spec.require("training_recipe")?, "collection_games", 16, 16)?;
                spec.require("training_opponents")?
            }
        }
    } else if let Some(value) = cfg.get("training_opponents") {
        value
    } else {
        return Ok(None);
    };
    descriptor.closed(&["version", "preset", "refresh"], &[])?;
    is(descriptor, "version", "1")?;
    is(descriptor, "preset", "mixed-refresh-16-v1")?;
    let options = descriptor.require("refresh")?;
    crate::structured_journal::card_instance::refresh_label(cfg.require("actions")?, options, b, c)?;
    Ok(Some(options))
}

pub(crate) fn refresh_training_registry(cfg: &Value, registry: &Value, b: &Arc<Budget>, c: &Control) -> Result<Option<String>> {
    let options = refresh_training(cfg, b, c)?;
    let mut found = 0;
    let expected = options.map(|value| crate::structured_journal::card_instance::refresh_label(cfg.require("actions")?, value, b, c)).transpose()?;
    for (label, binding) in registry.object()? {
        if text(binding, "model_id")? == "heuristic_refresh" {
            found += 1;
            binding.closed(&["model_id", "policy_options"], &[])?;
            if expected.as_deref() != Some(label.as_str()) || Some(binding.require("policy_options")?) != options {
                return Err(Error::contract("training refresh registry differs from frozen descriptor"));
            }
        }
    }
    if found != usize::from(expected.is_some()) {
        return Err(Error::contract("training refresh registry population differs from frozen descriptor"));
    }
    Ok(expected)
}

pub const INPUT: &str = "participant-structured-v1";
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceFormat {
    Engineering,
    Corrected,
    PackedCorrected,
    Hpo,
    PackedHpo,
    ResidentHpo,
    PackedResidentHpo,
}
impl SourceFormat {
    pub(crate) fn corrected(self) -> bool {
        self != Self::Engineering
    }
    pub(crate) fn packed(self) -> bool {
        matches!(self, Self::PackedCorrected | Self::PackedHpo | Self::PackedResidentHpo)
    }
    pub(crate) fn hpo(self) -> bool {
        matches!(self, Self::Hpo | Self::PackedHpo | Self::ResidentHpo | Self::PackedResidentHpo)
    }
    pub(crate) fn resident(self) -> bool {
        matches!(self, Self::ResidentHpo | Self::PackedResidentHpo)
    }
    fn class(self) -> &'static [&'static str] {
        if self.corrected() { GAME_CLASS } else { CLASS }
    }
    fn semantics(self) -> Vec<&'static str> {
        if self.hpo() {
            return hpo::semantic_keys();
        }
        let mut keys = SEMANTICS.to_vec();
        if self.corrected() {
            keys.extend(["rule_signature", "spec_inventory_sha256"]);
        }
        keys
    }
}
const GAME_CLASS: &[&str] = &[
    "action_contract",
    "purpose",
    "synthetic_capacity",
    "rule_signature",
    "layout_signature",
];
fn classification_context(
    v: &Value,
    contract: &Value,
    format: SourceFormat,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<()> {
    if format.corrected() {
        game_binding3(v, b, c)?;
        eq(v.require("action_contract")?, contract, "corrected enclosing action contract mismatch")
    } else {
        classification(v, contract)
    }
}
fn semantic_context(
    v: &Value,
    m: &Value,
    format: SourceFormat,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<()> {
    if format.hpo() {
        semantic_corrected(v, m, b, c, true)
    } else if format.corrected() {
        semantic3(v, m, b, c)
    } else {
        semantic(v, m, b, c)
    }
}
fn architecture_context(
    m: &Value,
    format: SourceFormat,
    b: &Arc<Budget>,
    c: &Control,
) -> Result<String> {
    if format.hpo() {
        hpo::architecture(m, b, c)
    } else if format.corrected() {
        architecture3(m, b, c)
    } else {
        architecture(m, b, c)
    }
}
const CLASS: &[&str] = &["action_contract", "purpose", "synthetic_capacity"];
const SEMANTICS: &[&str] = &[
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
];
const RECIPE: &[u8] = include_bytes!("../assets/structured-token-recipe-v1.json");
const EXACT_RECIPE: &[u8] =
    include_bytes!("../assets/structured-token-recipe-public-exact-v2.json");
pub(crate) const CARD_INSTANCE_RECIPE: &[u8] =
    include_bytes!("../assets/card-instance-token-recipe-corrected-v2.json");
#[path = "card_instance_formal.rs"]
pub(crate) mod formal;

pub(crate) const CARD_INSTANCE_SIGNATURE: &str =
    "ed0c61152b78fa984aad342be22c6b9d6ef49b12b8f0d1c0f0de89724468002f";

/// Pure diagnostic identity; this does not advertise a reader capability.
pub(crate) fn card_instance_token_signature(
    window: usize,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<String> {
    if !(1..=65536).contains(&window) {
        return Err(Error::contract("card-instance aggregate window bound"));
    }
    let mut recipe = json::parse(CARD_INSTANCE_RECIPE, budget, control)?;
    if digest(&recipe.root, budget, control)? != CARD_INSTANCE_SIGNATURE {
        return Err(Error::contract(
            "embedded corrected card-instance recipe changed",
        ));
    }
    recipe
        .root
        .as_object_mut()
        .expect("pinned recipe")
        .insert("history_window".into(), json::integer(window as i128));
    digest(&recipe.root, budget, control)
}

fn historical(model: &Value) -> Result<bool> {
    match model.require("event")?.get("aggregate_recipe") {
        None => Ok(true),
        Some(value) if value.string()? == "public-exact-v2" => Ok(false),
        _ => Err(Error::contract("unsupported closed aggregate recipe")),
    }
}

pub fn checksum(value: &Value) -> Result<&str> {
    let text = value.string()?;
    if text.len() != 64
        || !text
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(Error::contract("structured required SHA256 invalid"));
    }
    Ok(text)
}
pub fn basename(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 200
        || !name.is_ascii()
        || !name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
        || matches!(name, "." | "..")
    {
        return Err(Error::contract("structured owned basename required"));
    }
    Ok(())
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value.require(key)?.string()
}
fn n(value: &Value, key: &str, low: i128, high: i128) -> Result<i128> {
    value.require(key)?.integer(low, high)
}
fn eq(a: &Value, b: &Value, message: &str) -> Result<()> {
    if a != b {
        return Err(Error::contract(message));
    }
    Ok(())
}
fn is(value: &Value, key: &str, expected: &str) -> Result<()> {
    if text(value, key)? != expected {
        return Err(Error::contract(format!("structured {key} mismatch")));
    }
    Ok(())
}
pub fn digest(value: &Value, budget: &Arc<Budget>, control: &Control) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(&*json::canonical(value, budget, control)?)
    ))
}
pub fn token_signature(
    window: i128,
    old: bool,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<String> {
    if !(1..=65536).contains(&window) {
        return Err(Error::contract("structured aggregate window bound"));
    }
    let mut recipe = json::parse(if old { RECIPE } else { EXACT_RECIPE }, budget, control)?;
    recipe
        .root
        .as_object_mut()
        .expect("pinned recipe")
        .insert("history_window".into(), json::integer(window));
    digest(&recipe.root, budget, control)
}
fn classification(value: &Value, contract: &Value) -> Result<()> {
    eq(
        value.require("action_contract")?,
        contract,
        "structured action contract differs",
    )?;
    is(value, "purpose", "component")?;
    if value.require("synthetic_capacity")? != &Value::Bool(false) {
        return Err(Error::contract(
            "structured game control rejects synthetic capacity",
        ));
    }
    Ok(())
}
pub(crate) fn contract(value: &Value) -> Result<()> {
    value.closed(
        &[
            "profile_id",
            "owner_confirmation_sha256",
            "adopted_rules_sha256",
            "eligibility",
        ],
        &[],
    )?;
    for (key, expected) in [
        ("profile_id", "owner-proposal-engineering-20261101-v1"),
        (
            "owner_confirmation_sha256",
            "a0e6acb7fdc708ce4bd1b11f23d5dd3f0e41373cc1c1c1bef0a43a3e3efc2cb2",
        ),
        (
            "adopted_rules_sha256",
            "f4132843bb700947d51e19b57bed1b3bc9bcaf5f63c0cfa60998f1ddc3d6d978",
        ),
        ("eligibility", "engineering_only"),
    ] {
        is(value, key, expected)?;
    }
    Ok(())
}
fn uint128(value: &Value) -> Result<u128> {
    let Value::Number(value) = value else {
        return Err(Error::contract("PCG uint128 required"));
    };
    if !json::integer_token(value) {
        return Err(Error::contract("PCG integer token required"));
    }
    (if value.as_str() == "-0" {
        "0"
    } else {
        value.as_str()
    })
    .parse()
    .map_err(|_| Error::contract("PCG uint128 range"))
}
fn pcg(value: &Value) -> Result<()> {
    value.closed(&["bit_generator", "state", "has_uint32", "uinteger"], &[])?;
    is(value, "bit_generator", "PCG64")?;
    value.require("state")?.closed(&["state", "inc"], &[])?;
    uint128(value.require("state")?.require("state")?)?;
    if uint128(value.require("state")?.require("inc")?)? % 2 != 1 {
        return Err(Error::contract("PCG increment must be odd"));
    }
    n(value, "has_uint32", 0, 1)?;
    n(value, "uinteger", 0, u32::MAX as i128)?;
    Ok(())
}
fn controls(value: &Value, generation: usize) -> Result<()> {
    value.closed(
        &[
            "schedule_opponent_rng",
            "snapshot_pool",
            "collection_position",
        ],
        &[],
    )?;
    n(
        value,
        "collection_position",
        generation as i128,
        generation as i128,
    )?;
    pcg(value.require("schedule_opponent_rng")?)?;
    let expected: Vec<Value> = (0..=generation)
        .filter(|i| i % 10 == 0)
        .rev()
        .take(5)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|n| json::integer(n as i128))
        .collect();
    eq(
        value.require("snapshot_pool")?,
        &Value::Array(expected),
        "structured snapshot pool mismatch",
    )
}
pub(crate) fn binding(value: &Value) -> Result<()> {
    value.closed(
        &[
            "device",
            "candidate_chunk",
            "max_candidates",
            "managed_budget_bytes",
        ],
        &[],
    )?;
    if !["cpu", "cuda:0"].contains(&text(value, "device")?) {
        return Err(Error::contract("structured execution device unsupported"));
    }
    n(value, "candidate_chunk", 1, 65536)?;
    n(value, "max_candidates", 1, 65536)?;
    n(value, "managed_budget_bytes", 1, 12i128 << 30)?;
    Ok(())
}
pub(crate) fn visibility(value: &Value) -> Result<()> {
    if value.is_null() {
        return Ok(());
    }
    let text = value.string()?;
    if text.is_empty() {
        return Ok(());
    }
    if text.len() > 1024 {
        return Err(Error::contract("CUDA selector too long"));
    }
    let mut seen = Vec::new();
    for item in text.split(',') {
        let index = item == "0"
            || !item.starts_with('0')
                && item.bytes().all(|c| c.is_ascii_digit())
                && !item.is_empty();
        let uuid = item.strip_prefix("GPU-").is_some_and(|s| {
            s.len() == 36
                && s.bytes().enumerate().all(|(i, c)| {
                    if [8, 13, 18, 23].contains(&i) {
                        c == b'-'
                    } else {
                        c.is_ascii_hexdigit()
                    }
                })
        });
        if seen.len() >= 32 || seen.contains(&item) || !(index || uuid) {
            return Err(Error::contract("CUDA selector invalid"));
        }
        seen.push(item);
    }
    Ok(())
}
fn widths(value: &Value) -> Result<Vec<i128>> {
    let values = value.array()?;
    if !(1..=4).contains(&values.len()) {
        return Err(Error::contract("structured hidden widths bound"));
    }
    values.iter().map(|value| value.integer(1, 1024)).collect()
}
fn shape(shapes: &mut BTreeMap<String, Vec<i128>>, name: &str, dims: &[i128]) {
    shapes.insert(name.into(), dims.to_vec());
}
fn linear(shapes: &mut BTreeMap<String, Vec<i128>>, name: &str, source: i128, target: i128) {
    shape(shapes, &format!("{name}.weight"), &[target, source]);
    shape(shapes, &format!("{name}.bias"), &[target]);
}
fn mlp(shapes: &mut BTreeMap<String, Vec<i128>>, name: &str, widths: &[i128]) {
    for (i, w) in widths.windows(2).enumerate() {
        linear(shapes, &format!("{name}.{}", i * 2), w[0], w[1]);
    }
}
fn gru(
    shapes: &mut BTreeMap<String, Vec<i128>>,
    name: &str,
    source: i128,
    width: i128,
    layers: i128,
) {
    for i in 0..layers {
        shape(
            shapes,
            &format!("{name}.weight_ih_l{i}"),
            &[3 * width, if i == 0 { source } else { width }],
        );
        shape(
            shapes,
            &format!("{name}.weight_hh_l{i}"),
            &[3 * width, width],
        );
        for part in ["ih", "hh"] {
            shape(shapes, &format!("{name}.bias_{part}_l{i}"), &[3 * width]);
        }
    }
}
fn attention(shapes: &mut BTreeMap<String, Vec<i128>>, name: &str, width: i128, feed: i128) {
    for part in ["query", "key", "value", "output"] {
        linear(shapes, &format!("{name}.{part}"), width, width);
    }
    linear(shapes, &format!("{name}.feedforward1"), width, feed);
    linear(shapes, &format!("{name}.feedforward2"), feed, width);
    for norm in ["norm1", "norm2"] {
        shape(shapes, &format!("{name}.{norm}.weight"), &[width]);
        shape(shapes, &format!("{name}.{norm}.bias"), &[width]);
    }
}

pub(crate) fn model(
    value: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<BTreeMap<String, Vec<i128>>> {
    model_context(value, budget, control, ModelFormat::Engineering)
}
pub(crate) fn model3(
    value: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<BTreeMap<String, Vec<i128>>> {
    model_context(value, budget, control, ModelFormat::Corrected)
}

pub(crate) fn model4(
    value: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<BTreeMap<String, Vec<i128>>> {
    model_context(value, budget, control, ModelFormat::Hpo)
}

#[cfg(test)]
mod hpo_model_tests {
    use super::*;

    #[test]
    fn model4_has_own_temporal_activation_and_shape_contract() {
        let budget = Budget::new(64 << 20);
        let control = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let cases = json::parse(
            include_bytes!("../tests/fixtures/card_instance_models3_rules_v2.json"),
            &budget,
            &control,
        )
        .unwrap();
        for case in cases.root.require("cases").unwrap().array().unwrap() {
            let original = case.require("model").unwrap();
            let mut value = original.clone();
            value["version"] = json::text("4");
            value["initialization"] = json::text("orthogonal-activation-aware-v1");
            value["temporal"]
                .as_object_mut()
                .unwrap()
                .remove("sequence_length");
            value["temporal"]["max_decisions"] = json::integer(1024);
            assert_eq!(
                model4(&value, &budget, &control).unwrap(),
                model3(original, &budget, &control).unwrap()
            );
            value["card"]["activation"] = json::text("tanh");
            value["heads"]["activation"] = json::text("tanh");
            value["temporal"]["activation"] = json::text("tanh");
            if value["model_id"].as_str() == Some("mlp") {
                value["temporal"]["num_layers"] = json::integer(3);
                let shapes = model4(&value, &budget, &control).unwrap();
                let hidden = value["temporal"]["hidden_size"].integer(1, 1024).unwrap();
                assert_eq!(shapes["temporal_dense.0.weight"], vec![hidden, hidden]);
                assert_eq!(shapes["temporal_dense.2.bias"], vec![hidden]);
            } else {
                assert!(model4(&value, &budget, &control).is_ok());
            }
            assert!(model3(&value, &budget, &control).is_err());
            let mut broken = value.clone();
            broken["temporal"]["sequence_length"] = json::integer(32);
            assert!(model4(&broken, &budget, &control).is_err());
            broken = value.clone();
            broken["card"]["activation"] = json::text("elu");
            assert!(model4(&broken, &budget, &control).is_err());
            broken = value;
            broken["temporal"]["max_decisions"] = json::integer(8);
            assert!(model4(&broken, &budget, &control).is_err());
        }
    }
}
#[derive(Clone, Copy)]
enum ModelFormat {
    Engineering,
    Corrected,
    Hpo,
}
fn model_context(
    value: &Value,
    budget: &Arc<Budget>,
    control: &Control,
    format: ModelFormat,
) -> Result<BTreeMap<String, Vec<i128>>> {
    let corrected = !matches!(format, ModelFormat::Engineering);
    let hpo = matches!(format, ModelFormat::Hpo);
    value.closed(
        &[
            "model_id",
            "version",
            "input_recipe",
            "max_positions",
            "state",
            "card",
            "hand",
            "event",
            "temporal",
            "heads",
            "resources",
            "initialization",
            "dropout",
            "precision",
        ],
        &[],
    )?;
    is(
        value,
        "version",
        if hpo {
            "4"
        } else if corrected {
            "3"
        } else {
            "2"
        },
    )?;
    is(
        value,
        "input_recipe",
        if corrected {
            "participant-structured-v2"
        } else {
            INPUT
        },
    )?;
    is(
        value,
        "initialization",
        if hpo {
            "orthogonal-activation-aware-v1"
        } else {
            "orthogonal-structured-v1"
        },
    )?;
    n(value, "dropout", 0, 0)?;
    is(value, "precision", "ieee-fp32")?;
    let kind = text(value, "model_id")?;
    if !["mlp", "gru", "transformer"].contains(&kind) {
        return Err(Error::contract("structured model unsupported"));
    }
    let positions = n(value, "max_positions", 1, 128)?;
    let state = value.require("state")?;
    state.closed(&["kind", "hidden_sizes", "output_size", "activation"], &[])?;
    let card = value.require("card")?;
    card.closed(
        &[
            "definition_embedding",
            "ability_embedding",
            "output_size",
            "activation",
        ],
        &[],
    )?;
    let hand = value.require("hand")?;
    hand.closed(
        &[
            "kind",
            "hidden_sizes",
            "latent_size",
            "pooling",
            "attention_layers",
            "attention_heads",
            "attention_feedforward",
            "activation",
        ],
        &[],
    )?;
    let event = value.require("event")?;
    let old = if corrected {
        is(event, "aggregate_recipe", "public-exact-card-refs-v3")?;
        false
    } else {
        historical(value)?
    };
    let mut event_keys = vec![
        "hidden_size",
        "num_layers",
        "aggregate_window",
        "max_interval",
        "max_episode",
    ];
    if !old {
        event_keys.push("aggregate_recipe");
    }
    event.closed(&event_keys, &[])?;
    let temporal = value.require("temporal")?;
    let mut temporal_keys = vec![
        "kind",
        "hidden_size",
        "num_layers",
        "num_heads",
        "feedforward_size",
        "max_decisions",
        "activation",
    ];
    if !hpo {
        temporal_keys.push("sequence_length");
    }
    temporal.closed(&temporal_keys, &[])?;
    let heads = value.require("heads")?;
    heads.closed(
        &["actor_hidden_sizes", "critic_hidden_sizes", "activation"],
        &[],
    )?;
    let resources = value.require("resources")?;
    resources.closed(
        &[
            "max_parameters",
            "max_owned_bytes",
            "max_tensor_bytes",
            "cuda_context_bytes",
            "workspace_bytes",
        ],
        &[],
    )?;
    for item in [state, card, hand, temporal, heads] {
        if !hpo {
            is(item, "activation", "relu")?;
        } else if !["relu", "tanh"].contains(&text(item, "activation")?) {
            return Err(Error::contract("model4 dense activation unsupported"));
        }
    }
    let sw = widths(state.require("hidden_sizes")?)?;
    let hw = widths(hand.require("hidden_sizes")?)?;
    let aw = widths(heads.require("actor_hidden_sizes")?)?;
    let vw = widths(heads.require("critic_hidden_sizes")?)?;
    let so = n(state, "output_size", 1, 1024)?;
    let ce = n(card, "definition_embedding", 1, 1024)?;
    let ae = n(card, "ability_embedding", 1, 1024)?;
    let co = n(card, "output_size", 1, 1024)?;
    let hl = n(hand, "latent_size", 1, 1024)?;
    let h_layers = n(hand, "attention_layers", 1, 4)?;
    let h_heads = n(hand, "attention_heads", 1, 16)?;
    let h_feed = n(hand, "attention_feedforward", 1, 1024)?;
    let eh = n(event, "hidden_size", 1, 1024)?;
    let el = n(event, "num_layers", 1, 4)?;
    let ei = n(event, "max_interval", 1, 65536)?;
    let ee = n(event, "max_episode", ei, i32::MAX as i128)?;
    n(event, "aggregate_window", 1, 65536.min(ee))?;
    let th = n(temporal, "hidden_size", 1, 1024)?;
    let tl = n(temporal, "num_layers", 1, 4)?;
    let t_heads = n(temporal, "num_heads", 1, 16)?;
    let tf = n(temporal, "feedforward_size", 1, 1024)?;
    let td = n(temporal, "max_decisions", 1, 1024)?;
    if hpo {
        n(temporal, "max_decisions", 1024, 1024)?;
    } else {
        n(temporal, "sequence_length", 1, td)?;
    }
    is(temporal, "kind", kind)?;
    let sk = text(state, "kind")?;
    let hk = text(hand, "kind")?;
    if !["mlp", "identity"].contains(&sk)
        || !["direct", "deepsets", "attention"].contains(&hk)
        || !["sum", "mean", "max"].contains(&text(hand, "pooling")?)
    {
        return Err(Error::contract("structured encoder unsupported"));
    }
    if sk == "identity" && (sw != [128] || so != 64 || hpo && text(state, "activation")? != "relu")
    {
        return Err(Error::contract("inactive state settings"));
    }
    if hk != "attention" && (h_layers, h_heads, h_feed) != (2, 4, 128) {
        return Err(Error::contract("inactive hand attention settings"));
    }
    if hk == "direct"
        && (hw != [128]
            || hl != 128
            || text(hand, "pooling")? != "mean"
            || hpo && text(hand, "activation")? != "relu")
    {
        return Err(Error::contract("inactive direct settings"));
    }
    if hk == "attention" && co % h_heads != 0 || kind == "transformer" && th % t_heads != 0 {
        return Err(Error::contract("attention head divisibility"));
    }
    if kind != "transformer" && (t_heads, tf) != (4, 256) || !hpo && kind == "mlp" && tl != 1 {
        return Err(Error::contract("inactive temporal settings"));
    }
    let maxp = n(resources, "max_parameters", 1, 100_000_000)?;
    let maxt = n(resources, "max_tensor_bytes", 1, 64i128 << 20)?;
    n(resources, "max_owned_bytes", 1, 12i128 << 30)?;
    n(resources, "cuda_context_bytes", 1, 4i128 << 30)?;
    n(resources, "workspace_bytes", 1, 4i128 << 30)?;
    let recipe = json::parse(
        if corrected {
            CARD_INSTANCE_RECIPE
        } else if old {
            RECIPE
        } else {
            EXACT_RECIPE
        },
        budget,
        control,
    )?;
    let cats = |key: &str| -> Result<Vec<i128>> {
        recipe
            .root
            .require(key)?
            .array()?
            .iter()
            .map(|x| Ok(x.array()?.len() as i128 + 2))
            .collect()
    };
    let sc = cats("state_categories")?;
    let cc = cats("card_categories")?;
    let ec = cats("event_categories")?;
    let sn = recipe.root.require("state_numeric")?.array()?.len() as i128;
    let cn = recipe.root.require("card_numeric")?.array()?.len() as i128;
    let en = recipe.root.require("event_numeric")?.array()?.len() as i128;
    let state_width = 2 * sn + sc.len() as i128 + sc.iter().sum::<i128>() + 18 * cc[0];
    let card_direct = 2 * cn + cc.iter().sum::<i128>();
    let event_width = 2 * en + ec.iter().sum::<i128>();
    let history = if corrected {
        recipe.root.require("history_fields")?.array()?.len() as i128
    } else {
        recipe.root.require("event_categories")?.array()?[0]
            .array()?
            .len() as i128
            + (ec.len() as i128 - 4) * 2
            + en * 2
            + if old {
                0
            } else {
                recipe.root.require("history_append")?.array()?.len() as i128
            }
    };
    let card_in = ce + ae + cc[2..].iter().sum::<i128>() + 2 * cn;
    let card_actor = co + cc[if corrected { 10 } else { 6 }] + 2;
    let frame = if sk == "identity" { state_width } else { so }
        + if hk == "direct" { 18 * card_direct } else { hl }
        + eh
        + history;
    let candidate = recipe
        .root
        .require("candidate_categories")?
        .array()?
        .iter()
        .map(|x| Ok(x.array()?.len() as i128 + 2))
        .collect::<Result<Vec<_>>>()?
        .iter()
        .sum::<i128>()
        + 8;
    let mut shapes = BTreeMap::new();
    if sk == "mlp" {
        let mut w = vec![state_width];
        w.extend(sw);
        w.push(so);
        mlp(&mut shapes, "state_encoder", &w);
    }
    shape(&mut shapes, "definition_embedding.weight", &[cc[0], ce]);
    shape(&mut shapes, "ability_embedding.weight", &[cc[1], ae]);
    linear(&mut shapes, "card_encoder", card_in, co);
    if hk != "direct" {
        let mut w = vec![co];
        w.extend(hw);
        w.push(hl);
        mlp(&mut shapes, "hand_encoder", &w);
    }
    if hk == "attention" {
        for i in 0..h_layers {
            attention(&mut shapes, &format!("hand_attention.{i}"), co, h_feed);
        }
    }
    gru(&mut shapes, "event_encoder", event_width, eh, el);
    linear(&mut shapes, "frame_projection", frame, th);
    if hpo && kind == "mlp" {
        for index in 0..tl - 1 {
            linear(
                &mut shapes,
                &format!("temporal_dense.{}", 2 * index),
                th,
                th,
            );
        }
    }
    if kind == "gru" {
        gru(&mut shapes, "temporal", th, th, tl);
    } else if kind == "transformer" {
        for i in 0..tl {
            attention(&mut shapes, &format!("temporal.{i}"), th, tf);
        }
    }
    let mut w = vec![th + candidate + positions * card_actor];
    w.extend(aw);
    w.push(1);
    mlp(&mut shapes, "actor", &w);
    let mut w = vec![th];
    w.extend(vw);
    w.push(1);
    mlp(&mut shapes, "critic", &w);
    if shapes
        .values()
        .map(|s| s.iter().product::<i128>())
        .sum::<i128>()
        > maxp
        || shapes
            .values()
            .any(|s| s.iter().product::<i128>() * 4 > maxt)
    {
        return Err(Error::resource("structured parameter/tensor cap"));
    }
    Ok(shapes)
}

fn bundle<'a>(value: &'a Value) -> Result<BTreeMap<String, &'a Value>> {
    value.closed(
        &[
            "schema_version",
            "prefix",
            "shards",
            "files",
            "tensor_count",
            "decoded_bytes",
        ],
        &[],
    )?;
    n(value, "schema_version", 1, 1)?;
    let prefix = text(value, "prefix")?;
    if prefix.is_empty()
        || !prefix.as_bytes()[0].is_ascii_lowercase()
        || !prefix
            .bytes()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
    {
        return Err(Error::contract("structured shard prefix invalid"));
    }
    let shards = value.require("shards")?.array()?;
    if shards.is_empty() || shards.len() > 65536 {
        return Err(Error::contract("structured shard count bound"));
    }
    let files = value.require("files")?.object()?;
    let mut names = Vec::new();
    let mut tensors = BTreeMap::new();
    let mut total = 0;
    for (index, shard) in shards.iter().enumerate() {
        shard.closed(&["file", "metadata"], &[])?;
        let name = format!("{prefix}-{index:06}.npz");
        if text(shard, "file")? != name {
            return Err(Error::contract("structured shard order/path mismatch"));
        }
        let meta = shard.require("metadata")?;
        meta.closed(&["sha256", "arrays"], &[])?;
        checksum(meta.require("sha256")?)?;
        if files.get(&name) != Some(meta.require("sha256")?) {
            return Err(Error::contract("structured shard hash binding"));
        }
        names.push(name.clone());
        names.push(name.replace(".npz", ".json"));
        let arrays = meta.require("arrays")?.object()?;
        if arrays.is_empty() {
            return Err(Error::contract("empty shard"));
        }
        let mut bytes = 0;
        for (key, desc) in arrays {
            if key.is_empty()
                || !key
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
                || tensors.contains_key(key)
            {
                return Err(Error::contract("invalid/duplicate named tensor"));
            }
            desc.closed(&["dtype", "shape", "bytes"], &[])?;
            let unit = match text(desc, "dtype")? {
                "float32" => 4,
                "int64" => 8,
                "uint8" => 1,
                _ => return Err(Error::contract("unsupported typed dtype")),
            };
            let shape = desc.require("shape")?.array()?;
            if shape.len() > 8 {
                return Err(Error::contract("tensor dimensions bound"));
            }
            let dimensions = shape
                .iter()
                .map(|dim| dim.integer(0, 64i128 << 20))
                .collect::<Result<Vec<_>>>()?;
            let mut size = if dimensions.contains(&0) {
                0i128
            } else {
                1i128
            };
            if size != 0 {
                for dim in dimensions {
                    size = size
                        .checked_mul(dim)
                        .ok_or_else(|| Error::resource("tensor product overflow"))?;
                }
            }
            size = size
                .checked_mul(unit)
                .ok_or_else(|| Error::resource("tensor product overflow"))?;
            if n(desc, "bytes", 0, 64i128 << 20)? != size {
                return Err(Error::contract("typed tensor byte mismatch"));
            }
            bytes += size;
            if bytes > 64i128 << 20 {
                return Err(Error::resource("decoded shard exceeds64MiB"));
            }
            tensors.insert(key.clone(), desc);
        }
        total += bytes;
    }
    if total != n(value, "decoded_bytes", 0, 12i128 << 30)?
        || tensors.len() as i128 != n(value, "tensor_count", 1, 1_000_000)?
        || files.len() != names.len()
        || names.iter().any(|name| !files.contains_key(name))
    {
        return Err(Error::contract("typed bundle totals/filemap mismatch"));
    }
    for (name, sha) in files {
        basename(name)?;
        checksum(sha)?;
    }
    Ok(tensors)
}
fn semantic(value: &Value, model: &Value, budget: &Arc<Budget>, control: &Control) -> Result<()> {
    for (key, expected) in [
        ("input_recipe", INPUT),
        ("generator_version", "public-atomic-diverse-v4"),
        ("policy_input_version", "player-atomic-options-v1"),
        ("end_projection_version", "simulator-closure-v1"),
        ("reward_policy_version", "natural-terminal-v1"),
        (
            "simulator_revision",
            "ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6",
        ),
        ("numeric_transform", "signed-log1p-fp32-v1"),
    ] {
        is(value, key, expected)?;
    }
    crate::journal::catalog_meta(value.require("catalog")?)?;
    if digest(value.require("reviewed_spec_hashes")?, budget, control)?
        != "17dc86ba14c8fac378e352acd5c7c55429e910b37482954f74ca3673f8b1767d"
    {
        return Err(Error::contract("structured reviewed spec hashes mismatch"));
    }
    if text(value, "token_signature")?
        != token_signature(
            n(model.require("event")?, "aggregate_window", 1, 65536)?,
            historical(model)?,
            budget,
            control,
        )?
    {
        return Err(Error::contract("structured token signature mismatch"));
    }
    runtime_semantics(value)
}
fn runtime_semantics(value: &Value) -> Result<()> {
    let runtime = value.require("runtime")?;
    runtime.closed(&["torch", "numpy", "python", "gil_enabled"], &[])?;
    for key in ["torch", "numpy", "python"] {
        if text(runtime, key)?.is_empty() {
            return Err(Error::contract("empty recorded runtime"));
        }
    }
    runtime.require("gil_enabled")?.boolean()?;
    let numerics = value.require("numerics")?;
    numerics.closed(
        &[
            "precision",
            "tf32",
            "amp",
            "deterministic",
            "cublas_workspace_config",
            "threads",
        ],
        &[],
    )?;
    is(numerics, "precision", "ieee-fp32")?;
    is(numerics, "cublas_workspace_config", ":4096:8")?;
    n(numerics, "threads", 1, 1)?;
    for (key, expected) in [("tf32", false), ("amp", false), ("deterministic", true)] {
        if numerics.require(key)? != &Value::Bool(expected) {
            return Err(Error::contract("structured numerical recipe mismatch"));
        }
    }
    Ok(())
}
fn producer(value: &Value) -> Result<()> {
    value.closed(&["code_revision", "source_digest"], &[])?;
    checksum(value.require("source_digest")?)?;
    if !value.require("code_revision")?.is_null() {
        let text = text(value, "code_revision")?;
        if text.len() != 40
            || !text
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(Error::contract("recorded code revision invalid"));
        }
    }
    Ok(())
}
fn filemap(value: &Value) -> Result<()> {
    for (name, sha) in value.object()? {
        basename(name)?;
        checksum(sha)?;
    }
    Ok(())
}
fn clone_fields(value: &Value, keys: &[&str], control: &Control) -> Result<Value> {
    let mut output = json::Map::new();
    for key in keys {
        output.insert((*key).into(), json::copy(value.require(key)?, control)?);
    }
    Ok(Value::Object(output))
}
fn pure_model(config: &Value, control: &Control) -> Result<Value> {
    let raw = config.require("model")?;
    let mut output = raw.object()?.clone();
    for key in ["checkpoint", "inference", "cuda_visible_devices"] {
        if output.remove(key).is_none() {
            return Err(Error::contract("missing configured model field"));
        }
    }
    if !raw.require("checkpoint")?.is_null() {
        return Err(Error::contract("parent Config must be fresh model2"));
    }
    control.check()?;
    Ok(Value::Object(output))
}
pub(crate) fn architecture(
    model: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<String> {
    let mut fields = model.object()?.clone();
    fields.remove("resources");
    digest(
        &json::object([
            ("model", Value::Object(fields)),
            (
                "token_signature",
                json::text(&token_signature(
                    n(model.require("event")?, "aggregate_window", 1, 65536)?,
                    historical(model)?,
                    budget,
                    control,
                )?),
            ),
        ]),
        budget,
        control,
    )
}
pub(crate) fn actions(config: &Value) -> Result<()> {
    let action = config.require("actions")?;
    action.closed(
        &[
            "action_contract",
            "version",
            "generator",
            "use_action_options",
            "targets",
            "max_sequence_length",
            "max_candidates",
            "max_expansions",
            "max_temporary_candidates",
            "count_states",
            "count_milliseconds",
            "search_milliseconds",
            "channel_weights",
        ],
        &[],
    )?;
    is(action, "version", "2")?;
    is(action, "generator", "public-atomic-diverse-v4")?;
    contract(action.require("action_contract")?)?;
    if action.require("use_action_options")? != &Value::Bool(true) {
        return Err(Error::contract("structured actions require hints"));
    }
    n(action, "max_sequence_length", 1, 128)?;
    n(action, "max_candidates", 1, 65536)?;
    n(action, "max_expansions", 1, i32::MAX as i128)?;
    n(action, "max_temporary_candidates", 1, i32::MAX as i128)?;
    n(action, "count_states", 1, i32::MAX as i128)?;
    n(action, "count_milliseconds", 1, 1000)?;
    n(action, "search_milliseconds", 1, 10000)?;
    let target = action.require("targets")?.array()?;
    if target.is_empty()
        || target.len() > 2
        || target.iter().any(|t| {
            !t.as_str()
                .is_some_and(|t| ["self", "opponent"].contains(&t))
        })
        || target.len() == 2 && target[0] == target[1]
    {
        return Err(Error::contract("closed action targets required"));
    }
    let weights = action.require("channel_weights")?.array()?;
    if weights.len() != 7 {
        return Err(Error::contract("seven channels required"));
    }
    for w in weights {
        w.integer(1, i32::MAX as i128)?;
    }
    Ok(())
}
fn finite(value: &Value, key: &str) -> Result<f64> {
    let v = value.require(key)?.number()?;
    if !v.is_finite() || v < 0. {
        return Err(Error::contract(
            "finite nonnegative control number required",
        ));
    }
    Ok(v)
}

fn artifact(
    s: &mut Sources,
    path: &Path,
    record: &Value,
    receipt: &Value,
    state: &Value,
    model_value: &Value,
    shapes: &BTreeMap<String, Vec<i128>>,
    format: SourceFormat,
) -> Result<Value> {
    let budget = s.budget.clone();
    let control = s.control.clone();
    let contract = state.require("action_contract")?;
    let inference = s.checked_json(
        &path.join("inference.json"),
        record.require("inference_sha256")?,
    )?;
    let inf = &inference.root;
    let mut keys = vec![
        "schema_version",
        "kind",
        "model",
        "architecture_identity",
        "action_config_hash",
        "producer",
        "weights",
    ];
    keys.extend(format.semantics());
    keys.extend(format.class());
    inf.closed(&keys, &[])?;
    n(
        inf,
        "schema_version",
        if format.corrected() { 4 } else { 3 },
        if format.corrected() { 4 } else { 3 },
    )?;
    is(inf, "kind", "structured-inference")?;
    classification_context(inf, contract, format, &budget, &control)?;
    producer(inf.require("producer")?)?;
    semantic_context(inf, model_value, format, &budget, &control)?;
    eq(
        inf.require("model")?,
        model_value,
        "artifact/full frozen model mismatch",
    )?;
    if text(inf, "architecture_identity")?
        != architecture_context(model_value, format, &budget, &control)?
        || text(inf, "action_config_hash")?
            != digest(
                state.require("effective_config")?.require("actions")?,
                &budget,
                &control,
            )?
    {
        return Err(Error::contract("artifact architecture/action mismatch"));
    }
    let weights = bundle(inf.require("weights")?)?;
    is(inf.require("weights")?, "prefix", "weights")?;
    if weights.len() != shapes.len() {
        return Err(Error::contract("named parameter descriptor count"));
    }
    for (name, dims) in shapes {
        let desc = weights
            .get(name)
            .ok_or_else(|| Error::contract("named parameter missing"))?;
        is(desc, "dtype", "float32")?;
        eq(
            desc.require("shape")?,
            &Value::Array(dims.iter().map(|d| json::integer(*d)).collect()),
            "named parameter shape mismatch",
        )?;
    }
    receipt.closed(
        &[
            "status",
            "inference_sha256",
            "original_files",
            "deletion_intent",
            "deletion_receipt",
        ],
        &[],
    )?;
    eq(
        receipt.require("original_files")?,
        record.require("artifact_files")?,
        "original artifact maps mismatch",
    )?;
    eq(
        receipt.require("inference_sha256")?,
        record.require("inference_sha256")?,
        "registry inference link mismatch",
    )?;
    let files = record.require("artifact_files")?;
    filemap(files)?;
    if files.require("inference.json")? != record.require("inference_sha256")?
        || files.require("resume.json")? != record.require("resume_sha256")?
    {
        return Err(Error::contract("required manifest links mismatch"));
    }
    let retained = text(receipt, "status")? == "retained";
    if retained {
        if !receipt.require("deletion_intent")?.is_null()
            || !receipt.require("deletion_receipt")?.is_null()
        {
            return Err(Error::contract("retained artifact has deletion evidence"));
        }
        let resume = s.checked_json(&path.join("resume.json"), record.require("resume_sha256")?)?;
        let r = &resume.root;
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
        ];
        keys.extend(format.semantics());
        keys.extend(format.class());
        if format.corrected() {
            keys.push("action_config_hash");
            is(r, "action_config_hash", text(inf, "action_config_hash")?)?;
        }
        r.closed(&keys, &[])?;
        n(
            r,
            "schema_version",
            if format.corrected() { 4 } else { 3 },
            if format.corrected() { 4 } else { 3 },
        )?;
        is(r, "kind", "structured-resume")?;
        classification_context(r, contract, format, &budget, &control)?;
        semantic_context(r, model_value, format, &budget, &control)?;
        producer(r.require("producer")?)?;
        for key in format.semantics() {
            eq(
                r.require(key)?,
                inf.require(key)?,
                "paired semantic declarations mismatch",
            )?;
        }
        eq(r.require("model")?, model_value, "paired model mismatch")?;
        eq(
            r.require("inference_digest")?,
            record.require("inference_sha256")?,
            "resume inference link mismatch",
        )?;
        eq(
            r.require("binding")?,
            state
                .require("execution")?
                .require("learner")?
                .require("binding")?,
            "resume learner binding mismatch",
        )?;
        binding(r.require("binding")?)?;
        eq(
            r.require("resume_identity")?,
            state.require("resume_identity")?,
            "resume identity mismatch",
        )?;
        eq(
            r.require("optimizer_steps")?,
            record.require("optimizer_steps")?,
            "resume optimizer counter mismatch",
        )?;
        eq(
            r.require("producer")?.require("source_digest")?,
            inf.require("producer")?.require("source_digest")?,
            "paired producer digest mismatch",
        )?;
        let rc = r.require("producer")?.require("code_revision")?;
        let ic = inf.require("producer")?.require("code_revision")?;
        if !rc.is_null() && !ic.is_null() {
            eq(rc, ic, "paired code revision mismatch")?;
        }
        let recipe = r.require("optimizer_recipe")?;
        recipe.closed(
            &[
                "name",
                "lr",
                "betas",
                "eps",
                "weight_decay",
                "amsgrad",
                "maximize",
                "foreach",
                "capturable",
                "differentiable",
                "fused",
                "decoupled_weight_decay",
            ],
            &[],
        )?;
        is(recipe, "name", "Adam")?;
        if finite(recipe, "lr")? != 0.0003
            || finite(recipe, "eps")? != 1e-8
            || finite(recipe, "weight_decay")? != 0.
            || recipe.require("betas")?.array()?.len() != 2
            || recipe.require("betas")?.array()?[0].number()? != 0.9
            || recipe.require("betas")?.array()?[1].number()? != 0.999
        {
            return Err(Error::contract("Adam recipe mismatch"));
        }
        for key in [
            "amsgrad",
            "maximize",
            "capturable",
            "differentiable",
            "decoupled_weight_decay",
        ] {
            if recipe.require(key)? != &Value::Bool(false) {
                return Err(Error::contract("Adam execution option mismatch"));
            }
        }
        for key in ["foreach", "fused"] {
            if !recipe.require(key)?.is_null() {
                return Err(Error::contract("Adam nullable option mismatch"));
            }
        }
        let count = n(r, "cuda_rng_count", 0, 1024)?;
        let expected = if text(state, "device")? == "cpu" {
            0
        } else {
            n(state.require("hardware")?, "device_count", 1, 1024)?
        };
        if count != expected {
            return Err(Error::contract("resume CUDA RNG count mismatch"));
        }
        pcg(r.require("shuffle_rng")?)?;
        is(r.require("state")?, "prefix", "resume")?;
        let descriptors = bundle(r.require("state")?)?;
        let states = r.require("parameter_states")?.object()?;
        if states.len() != shapes.len() {
            return Err(Error::contract("Adam named ownership mismatch"));
        }
        let mut expected: BTreeMap<String, (&str, Option<Vec<i128>>)> = BTreeMap::new();
        expected.insert("rng.torch_cpu".into(), ("uint8", None));
        for i in 0..count {
            expected.insert(format!("rng.torch_cuda.{i}"), ("uint8", None));
        }
        for (name, shape) in shapes {
            let entry = states
                .get(name)
                .ok_or_else(|| Error::contract("Adam parameter missing"))?;
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
        if descriptors.len() != expected.len() {
            return Err(Error::contract("Adam/RNG tensor set mismatch"));
        }
        for (name, (dtype, shape)) in expected {
            let desc = descriptors
                .get(&name)
                .ok_or_else(|| Error::contract("Adam/RNG descriptor missing"))?;
            is(desc, "dtype", dtype)?;
            if let Some(shape) = shape {
                eq(
                    desc.require("shape")?,
                    &Value::Array(shape.into_iter().map(json::integer).collect()),
                    "Adam tensor shape mismatch",
                )?;
            } else {
                let shape = desc.require("shape")?.array()?;
                if shape.len() != 1 || shape[0].integer(1, 64i128 << 20)? == 0 {
                    return Err(Error::contract("RNG vector shape"));
                }
            }
        }
        let mut actual = json::Map::new();
        for bundle_key in [inf.require("weights")?, r.require("state")?] {
            for (name, sha) in bundle_key.require("files")?.object()? {
                if actual.insert(name.clone(), sha.clone()).is_some() {
                    return Err(Error::contract("mixed artifact bundle ownership"));
                }
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
            files,
            "exact full artifact map mismatch",
        )?;
        for b in [inf.require("weights")?, r.require("state")?] {
            for shard in b.require("shards")?.array()? {
                let name = text(shard, "file")?.replace(".npz", ".json");
                let side =
                    s.checked_json(&path.join(&name), b.require("files")?.require(&name)?)?;
                eq(
                    &side.root,
                    shard.require("metadata")?,
                    "inline shard sidecar mismatch",
                )?;
            }
        }
        s.file_map(path, files, true)?;
        s.inventory(path, files.object()?.keys().cloned().collect(), false)?;
    } else {
        is(receipt, "status", "deleted")?;
        let intent = receipt.require("deletion_intent")?;
        intent.closed(&["files", "staging_basename"], &[])?;
        let staging = text(intent, "staging_basename")?;
        if staging.len() != 38
            || !staging.starts_with("prune-")
            || !staging[6..]
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(Error::contract("pruning staging basename"));
        }
        match std::fs::symlink_metadata(path.parent().expect("artifact parent").join(staging)) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            _ => return Err(Error::contract("completed pruning stage must be absent")),
        }
        let mut deleted = files.object()?.clone();
        deleted.remove("inference.json");
        eq(
            intent.require("files")?,
            &Value::Object(deleted.clone()),
            "prune intent exact map",
        )?;
        let receipt_files = receipt.require("deletion_receipt")?;
        receipt_files.closed(&["files"], &[])?;
        eq(
            receipt_files.require("files")?,
            &Value::Object(deleted),
            "prune receipt exact map",
        )?;
        for (name, sha) in inf.require("weights")?.require("files")?.object()? {
            eq(
                files.require(name)?,
                sha,
                "pruned original weights map mismatch",
            )?;
        }
        let required = 2 + inf.require("weights")?.require("files")?.object()?.len();
        let rest = files
            .object()?
            .len()
            .checked_sub(required)
            .ok_or_else(|| Error::contract("pruned map missing"))?;
        if rest == 0 || rest % 2 != 0 {
            return Err(Error::contract("pruned resume shard pairs missing"));
        }
        for i in 0..rest / 2 {
            for ext in ["npz", "json"] {
                checksum(files.require(&format!("resume-{i:06}.{ext}"))?)?;
            }
        }
        s.inventory(path, vec!["inference.json".into()], false)?;
    }
    clone_fields(inf, &format.semantics(), &control)
}

fn relative(root: &Path, value: &Value) -> Result<std::path::PathBuf> {
    let path = Path::new(value.string()?);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|p| !matches!(p, std::path::Component::Normal(_)))
    {
        return Err(Error::contract("relative owned path required"));
    }
    Ok(root.join(path))
}
fn update_metadata(
    s: &mut Sources,
    u: &Value,
    state: &Value,
    m: &Value,
    sem: &Value,
    previous: &str,
    previous_inf: &str,
    number: usize,
    format: SourceFormat,
) -> Result<i128> {
    let budget = s.budget.clone();
    let control = s.control.clone();
    let root = s.root.clone();
    u.closed(
        &[
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
        ],
        if format.packed() {
            &["rule_signature", "layout_signature", "storage_binding"]
        } else if format.corrected() {
            &["rule_signature", "layout_signature"]
        } else {
            &[]
        },
    )?;
    n(
        u,
        "schema_version",
        if format.packed() {
            4
        } else if format.corrected() {
            3
        } else {
            2
        },
        if format.packed() {
            4
        } else if format.corrected() {
            3
        } else {
            2
        },
    )?;
    is(u, "status", "complete")?;
    if format.packed() {
        eq(
            u.require("storage_binding")?,
            state.require("storage_binding")?,
            "packed update storage binding differs",
        )?;
    }
    is(
        u,
        "model_version",
        if format.corrected() { "3" } else { "2" },
    )?;
    is(
        u,
        "input_recipe",
        if format.corrected() {
            "participant-structured-v2"
        } else {
            INPUT
        },
    )?;
    eq(
        u.require("model_id")?,
        state.require("model_id")?,
        "update model mismatch",
    )?;
    eq(
        u.require("run_id")?,
        state.require("run_id")?,
        "update run mismatch",
    )?;
    n(u, "generation", number as i128, number as i128)?;
    if checksum(u.require("parent_sha256")?)? != previous {
        return Err(Error::contract("update parent hash mismatch"));
    }
    classification_context(
        u,
        state.require("action_contract")?,
        format,
        &budget,
        &control,
    )?;
    if format.corrected() {
        refresh_training_registry(state.require("effective_config")?, u.require("policy_bindings")?, &budget, &control)?;
    }
    n(u, "planned_games", 16, 16)?;
    n(u, "actual_games", 16, 16)?;
    if !u.require("health")?.object()?.is_empty() {
        return Err(Error::contract("update unhealthy"));
    }
    let seats = u.require("learner_seats")?.object()?;
    if seats.len() != 16 {
        return Err(Error::contract("all16 learner roles required"));
    }
    for viewer in seats.values() {
        viewer.integer(1, 2)?;
    }
    checksum(u.require("collection_manifest_sha256")?)?;
    checksum(u.require("numeric_sha256")?)?;
    let collection = s.checked_json(
        &relative(&root, u.require("collection")?)?.join("manifest.json"),
        u.require("collection_manifest_sha256")?,
    )?;
    n(
        &collection.root,
        "schema_version",
        if format.corrected() { 5 } else { 4 },
        if format.corrected() { 5 } else { 4 },
    )?;
    n(
        &collection.root,
        "journal_schema_version",
        if format.corrected() { 6 } else { 5 },
        if format.corrected() { 6 } else { 5 },
    )?;
    classification_context(
        &collection.root,
        state.require("action_contract")?,
        format,
        &budget,
        &control,
    )?;
    let numeric = relative(&root, u.require("numeric")?)?;
    let rollout = s.checked_json(&numeric.join("rollout.json"), u.require("numeric_sha256")?)?;
    let r = &rollout.root;
    n(
        r,
        "schema_version",
        if format.corrected() { 4 } else { 3 },
        if format.corrected() { 4 } else { 3 },
    )?;
    is(r, "kind", "structured-rollout")?;
    classification_context(
        r,
        state.require("action_contract")?,
        format,
        &budget,
        &control,
    )?;
    if text(r, "architecture_identity")? != architecture_context(m, format, &budget, &control)?
        || text(r, "action_config_hash")?
            != digest(
                state.require("effective_config")?.require("actions")?,
                &budget,
                &control,
            )?
    {
        return Err(Error::contract("numeric model/action mismatch"));
    }
    for key in format.semantics() {
        eq(
            r.require(key)?,
            sem.require(key)?,
            "numeric recorded declaration mismatch",
        )?;
    }
    let execution = state.require("execution")?.require("inference")?;
    let exec = digest(
        &json::object([
            ("inference", execution.require("binding")?.clone()),
            ("resources", m.require("resources")?.clone()),
            (
                "cuda_visible_devices",
                execution.require("cuda_visible_devices")?.clone(),
            ),
        ]),
        &budget,
        &control,
    )?;
    let label = if format.corrected() {
        let mut options = json::object([
            ("version", json::text("3")),
            ("checkpoint", json::text("recorded")),
            ("expected_manifest_digest", json::text(previous_inf)),
            ("model", json::copy(m, &control)?),
            (
                "inference",
                json::copy(execution.require("binding")?, &control)?,
            ),
            (
                "cuda_visible_devices",
                json::copy(execution.require("cuda_visible_devices")?, &control)?,
            ),
            (
                "action_config_hash",
                json::text(&digest(
                    state.require("effective_config")?.require("actions")?,
                    &budget,
                    &control,
                )?),
            ),
        ]);
        for key in GAME_CLASS {
            options
                .as_object_mut()
                .unwrap()
                .insert((*key).into(), json::copy(state.require(key)?, &control)?);
        }
        options3(
            &options,
            state.require("effective_config")?.require("actions")?,
            &budget,
            &control,
        )?
    } else {
        format!("{}-ppo-v2-{previous_inf}-{exec}", text(m, "model_id")?)
    };
    is(r, "policy_version", &label)?;
    let episodes = r.require("episodes")?.array()?;
    if episodes.len() != 16 {
        return Err(Error::contract("numeric all16 episodes required"));
    }
    let mut count = 0;
    let mut owners = Vec::new();
    for episode in episodes {
        control.check()?;
        classification_context(
            episode,
            state.require("action_contract")?,
            format,
            &budget,
            &control,
        )?;
        is(episode, "end_kind", "natural")?;
        crate::validation::result(episode.require("outcome")?)?;
        if episode.require("outcome")?.is_null() || !episode.require("reason")?.is_null() {
            return Err(Error::contract("natural closure/outcome mismatch"));
        }
        let actor = n(episode, "actor", 1, 2)?;
        let id = text(episode, "episode")?;
        if id.is_empty() || owners.iter().any(|(a, b)| *a == actor && *b == id) {
            return Err(Error::contract("duplicate numeric participant owner"));
        }
        owners.push((actor, id));
        let frames = episode.require("frames")?.array()?;
        if frames.len() != episode.require("selections")?.array()?.len()
            || episode.require("has_targets")? != &Value::Bool(!frames.is_empty())
        {
            return Err(Error::contract("numeric frames/selections/target mismatch"));
        }
        for frame in frames {
            classification_context(
                frame,
                state.require("action_contract")?,
                format,
                &budget,
                &control,
            )?;
        }
        count += frames.len() as i128;
    }
    if n(r, "decision_count", 1, u64::MAX as i128)? != count
        || n(u, "accepted_decisions", 1, u64::MAX as i128)? != count
    {
        return Err(Error::contract("numeric learner-only count mismatch"));
    }
    let proof = u.require("source_proof")?;
    n(
        proof,
        "schema_version",
        if format.packed() { 2 } else { 1 },
        if format.packed() { 2 } else { 1 },
    )?;
    is(
        proof,
        "kind",
        if format.packed() {
            "card-instance-packed-original-source-proof"
        } else if format.corrected() {
            "card-instance-original-source-proof"
        } else {
            "structured-component-source-proof"
        },
    )?;
    classification_context(
        proof,
        state.require("action_contract")?,
        format,
        &budget,
        &control,
    )?;
    is(proof, "parent_qualification", "not_performed")?;
    if proof.require("qualified")? != &Value::Bool(false) {
        return Err(Error::contract("engineering source cannot qualify"));
    }
    n(proof, "games", 16, 16)?;
    eq(
        proof.require("manifest_sha256")?,
        u.require("collection_manifest_sha256")?,
        "Source manifest link mismatch",
    )?;
    checksum(proof.require("source_set_sha256")?)?;
    n(proof, "accepted_decisions", count, u64::MAX as i128)?;
    bundle(r.require("data")?)?;
    let mut files = r.require("data")?.require("files")?.object()?.clone();
    files.insert("rollout.json".into(), u.require("numeric_sha256")?.clone());
    eq(
        u.require("numeric_files")?,
        &Value::Object(files.clone()),
        "numeric exact file map mismatch",
    )?;
    s.file_map(&numeric, u.require("numeric_files")?, true)?;
    s.inventory(&numeric, files.keys().cloned().collect(), false)?;
    for shard in r.require("data")?.require("shards")?.array()? {
        let name = text(shard, "file")?.replace(".npz", ".json");
        let side = s.checked_json(
            &numeric.join(&name),
            r.require("data")?.require("files")?.require(&name)?,
        )?;
        eq(
            &side.root,
            shard.require("metadata")?,
            "numeric shard sidecar mismatch",
        )?;
    }
    finite(u, "collection_seconds")?;
    finite(u, "update_seconds")?;
    Ok(count)
}

pub fn parent(s: &mut Sources) -> Result<Document> {
    parent_context(s, SourceFormat::Engineering)
}
pub(crate) fn parent3(s: &mut Sources) -> Result<Document> {
    parent_context(s, SourceFormat::Corrected)
}
pub(crate) fn parent4(s: &mut Sources) -> Result<Document> {
    parent_context(s, SourceFormat::PackedCorrected)
}
fn parent_context(s: &mut Sources, format: SourceFormat) -> Result<Document> {
    let root = s.root.clone();
    let budget = s.budget.clone();
    let control = s.control.clone();
    let state = s.json(&root.join("run.json"))?;
    let r = &state.root;
    let formal = format.corrected() && self::formal::discriminator(r)?;
    let run_fields = "schema_version algorithm run_id status config_sha256 effective_config config_inputs identity seed device created_unix last_unix wall_deadline_unix admission_deadline_unix execution_deadline_unix updates accepted_decisions optimizer_steps generation_sha256 collection_seconds update_seconds invocations completed_updates model_id model_version input_recipe action_contract purpose synthetic_capacity resume_identity execution";
    let optional = "hardware elapsed_seconds artifact_bytes peak_coordinator_memory_bytes coordinator_memory_scope timing_scope failure quarantine cleanup_failure budget_overrun_seconds";
    r.closed(
        &[
            run_fields.split_whitespace().collect::<Vec<_>>(),
            if format.corrected() {
                vec!["rule_signature", "layout_signature"]
            } else {
                vec![]
            },
            if format.packed() {
                vec!["storage_binding"]
            } else {
                vec![]
            },
            if formal {
                self::formal::EXTRA.to_vec()
            } else {
                vec![]
            },
        ]
        .concat(),
        &optional.split_whitespace().collect::<Vec<_>>(),
    )?;
    n(
        r,
        "schema_version",
        if format.packed() {
            4
        } else if format.corrected() {
            3
        } else {
            2
        },
        if format.packed() {
            4
        } else if format.corrected() {
            3
        } else {
            2
        },
    )?;
    is(r, "algorithm", "ppo")?;
    is(
        r,
        "model_version",
        if format.corrected() { "3" } else { "2" },
    )?;
    is(
        r,
        "input_recipe",
        if format.corrected() {
            "participant-structured-v2"
        } else {
            INPUT
        },
    )?;
    if text(r, "run_id")?.is_empty()
        || !["running", "paused", "complete"].contains(&text(r, "status")?)
    {
        return Err(Error::contract("healthy component Run2 required"));
    }
    for key in [
        "failure",
        "quarantine",
        "cleanup_failure",
        "budget_overrun_seconds",
    ] {
        if r.get(key).is_some_and(crate::controls::truthy) {
            return Err(Error::contract(
                "structured parent contains failure evidence",
            ));
        }
    }
    if formal {
        self::formal::validate(s, r)?;
    }
    let cfg = r.require("effective_config")?;
    if refresh_training(cfg, &budget, &control)?.is_some() && (!format.corrected() || formal) {
        return Err(Error::contract("mixed opponents require nonformal corrected PPO"));
    }
    if format.packed() {
        eq(
            r.require("storage_binding")?,
            &crate::logical_artifacts::binding(),
            "packed run storage binding differs",
        )?;
        is(cfg.require("storage")?, "format", "sqlite-group-pack-v1")?;
    } else if cfg
        .get("storage")
        .is_some_and(|v| v.get("format").and_then(Value::as_str) == Some("sqlite-group-pack-v1"))
    {
        return Err(Error::contract("directory run rejects packed storage"));
    }
    let action_contract = cfg.require("actions")?.require("action_contract")?;
    if format.corrected() {
        actions3(cfg.require("actions")?, &budget, &control)?;
        validate_preprocessing3(r, &budget, &control)?;
    } else {
        actions(cfg)?;
    }
    classification_context(r, action_contract, format, &budget, &control)?;
    if digest(cfg, &budget, &control)? != checksum(r.require("config_sha256")?)? {
        return Err(Error::contract("frozen Config checksum mismatch"));
    }
    if r.require("identity")?.object()?.is_empty() {
        return Err(Error::contract("recorded identity required"));
    }
    r.require("config_inputs")?.array()?;
    let pure = pure_model(cfg, &control)?;
    // Bind the descriptor to the ORIGINAL protected generation0/map/inference,
    // never infer a historical model solely from a missing recipe field.
    let boot = s.json(&root.join("generations/000000.json"))?;
    checksum(boot.root.require("inference_sha256")?)?;
    eq(
        boot.root
            .require("artifact_files")?
            .require("inference.json")?,
        boot.root.require("inference_sha256")?,
        "g0 descriptor bootstrap map mismatch",
    )?;
    let first = s.checked_json(
        &root.join("artifacts/000000/inference.json"),
        boot.root.require("inference_sha256")?,
    )?;
    eq(
        first.root.require("model")?,
        &pure,
        "g0 descriptor bootstrap frozen model mismatch",
    )?;
    let expected_token = if format.corrected() {
        card_instance_token_signature(
            n(pure.require("event")?, "aggregate_window", 1, 65536)? as usize,
            &budget,
            &control,
        )?
    } else {
        token_signature(
            n(pure.require("event")?, "aggregate_window", 1, 65536)?,
            historical(&pure)?,
            &budget,
            &control,
        )?
    };
    is(&first.root, "token_signature", &expected_token)?;
    let shapes = if format.corrected() {
        model3(&pure, &budget, &control)?
    } else {
        model(&pure, &budget, &control)?
    };
    eq(
        pure.require("model_id")?,
        r.require("model_id")?,
        "parent model kind mismatch",
    )?;
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
    let seed = n(
        r,
        "seed",
        0,
        if format.corrected() {
            i64::MAX as i128
        } else {
            i32::MAX as i128
        },
    )?;
    if train.require("seeds")?.array()?.len() != 1
        || train.require("seeds")?.array()?[0].integer(
            0,
            if format.corrected() {
                i64::MAX as i128
            } else {
                i32::MAX as i128
            },
        )? != seed
    {
        return Err(Error::contract("seed differs from frozen training seed"));
    }
    eq(
        train.require("device")?,
        r.require("device")?,
        "learning device mismatch",
    )?;
    is(cfg.require("run")?, "split", "train")?;
    let max_updates = n(train, "max_updates", 1, 1000)?;
    let max_owned = n(train, "max_owned_bytes", 1, 10i128 << 30)?;
    n(
        train,
        "admission_reservation_bytes",
        1,
        (1i128 << 30).min(max_owned),
    )?;
    n(train, "max_log_bytes", 1, i32::MAX as i128)?;
    let cleanup = finite(train, "cleanup_seconds")?;
    if cleanup <= 0. || cleanup > 60. {
        return Err(Error::contract("cleanup budget invalid"));
    }
    let execution = r.require("execution")?;
    execution.closed(&["learner", "inference"], &[])?;
    for role in ["learner", "inference"] {
        let e = execution.require(role)?;
        e.closed(&["binding", "cuda_visible_devices"], &[])?;
        binding(e.require("binding")?)?;
        visibility(e.require("cuda_visible_devices")?)?;
        let expected_device = if role == "learner" {
            r.require("device")?
        } else {
            cfg.require("model")?
                .require("inference")?
                .require("device")?
        };
        eq(
            e.require("binding")?.require("device")?,
            expected_device,
            "execution Config device mismatch",
        )?;
        let configured = cfg.require("model")?.require("cuda_visible_devices")?;
        visibility(configured)?;
        let expected = if expected_device.string()? == "cpu" {
            json::text("")
        } else {
            configured.clone()
        };
        eq(
            e.require("cuda_visible_devices")?,
            &expected,
            "execution visibility mismatch",
        )?;
    }
    eq(
        execution.require("inference")?.require("binding")?,
        cfg.require("model")?.require("inference")?,
        "frozen inference binding mismatch",
    )?;
    let mut expected_resume = json::object([
        ("recipe", json::text("structured-ppo-resume-identity-v1")),
        ("config_sha256", r.require("config_sha256")?.clone()),
        ("identity", r.require("identity")?.clone()),
        ("execution", execution.clone()),
        ("input_binding", clone_fields(r, format.class(), &control)?),
    ]);
    if format.packed() {
        expected_resume["recipe"] = json::text("card-instance-packed-ppo-resume-identity-v1");
        expected_resume.as_object_mut().unwrap().insert(
            "storage_binding".into(),
            json::copy(r.require("storage_binding")?, &control)?,
        );
    }
    if digest(&expected_resume, &budget, &control)? != checksum(r.require("resume_identity")?)? {
        return Err(Error::contract("domain-separated resume identity mismatch"));
    }
    let created = finite(r, "created_unix")?;
    let last = finite(r, "last_unix")?;
    let wall = finite(r, "wall_deadline_unix")?;
    let admission = finite(r, "admission_deadline_unix")?;
    let execute = finite(r, "execution_deadline_unix")?;
    if !(created <= last
        && last <= wall
        && created < admission
        && admission <= execute
        && execute < wall)
        || wall - created > 7200.
        || admission - created > 5400.
        || execute - created > 7140.
    {
        return Err(Error::contract("original structured parent clocks invalid"));
    }
    for (limit, key) in [
        (wall, "wall_seconds"),
        (admission, "admission_seconds"),
        (execute, "execution_seconds"),
    ] {
        if (limit - created - finite(cfg.require("run")?, key)?).abs() > 0.001 {
            return Err(Error::contract("original Config deadline mismatch"));
        }
    }
    let count = n(r, "updates", 0, max_updates)? as usize;
    let total_count = n(r, "accepted_decisions", 0, u64::MAX as i128)?;
    let total_steps = n(r, "optimizer_steps", 0, i32::MAX as i128)?;
    finite(r, "collection_seconds")?;
    finite(r, "update_seconds")?;
    if !r.require("completed_updates")?.is_null() {
        n(r, "completed_updates", 0, 1000)?;
    }
    if text(r, "status")? == "complete"
        && (count == 0 || r.require("completed_updates")?.integer(0, 1000)? != count as i128)
    {
        return Err(Error::contract(
            "complete component run must have finalized updates",
        ));
    }
    if ["paused", "complete"].contains(&text(r, "status")?) {
        if (finite(r, "elapsed_seconds")? - (last - created)).abs() > 0.001 {
            return Err(Error::contract("closing elapsed mismatch"));
        }
        n(r, "artifact_bytes", 0, max_owned)?;
    }
    for inv in r.require("invocations")?.array()? {
        control.check()?;
        let health = inv.require("child_health")?;
        health.closed(&["exit_code", "stderr_bytes"], &[])?;
        n(health, "exit_code", 0, 0)?;
        n(health, "stderr_bytes", 0, u64::MAX as i128)?;
    }
    let hardware = r.require("hardware")?;
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
        r.require("device")?,
        "hardware declared device mismatch",
    )?;
    if text(hardware, "name")?.is_empty() || text(hardware, "exact_driver_build")?.is_empty() {
        return Err(Error::contract("hardware description empty"));
    }
    if !hardware.require("cuda_runtime")?.is_null() {
        text(hardware, "cuda_runtime")?;
    }
    n(hardware, "device_count", 0, 1024)?;
    if text(r, "device")? == "cpu" {
        n(hardware, "device_count", 0, 0)?;
        is(hardware, "name", "cpu")?;
        if !hardware.require("cuda_driver_api_version")?.is_null() {
            return Err(Error::contract("CPU hardware driver must be null"));
        }
    } else {
        n(hardware, "device_count", 1, 1024)?;
        n(hardware, "cuda_driver_api_version", 1, i32::MAX as i128)?;
    }
    let latest = s.json(&root.join("latest.json"))?;
    latest.root.closed(&["generation", "sha256"], &[])?;
    n(&latest.root, "generation", count as i128, count as i128)?;
    checksum(latest.root.require("sha256")?)?;
    eq(
        latest.root.require("sha256")?,
        r.require("generation_sha256")?,
        "latest/run generation mismatch",
    )?;
    let registry = s.json(&root.join("artifact-registry.json"))?;
    registry
        .root
        .closed(&["schema_version", "generations"], &[])?;
    n(
        &registry.root,
        "schema_version",
        if format.corrected() { 3 } else { 2 },
        if format.corrected() { 3 } else { 2 },
    )?;
    let reg = registry.root.require("generations")?.object()?;
    if reg.len() != count + 1 {
        return Err(Error::contract("registry generation set mismatch"));
    }
    s.inventory(
        &root.join("generations"),
        (0..=count).map(|i| format!("{i:06}.json")).collect(),
        false,
    )?;
    s.inventory(
        &root.join("updates"),
        (1..=count).map(|i| format!("{i:06}.json")).collect(),
        false,
    )?;
    s.inventory(
        &root.join("artifacts"),
        (0..=count).map(|i| format!("{i:06}")).collect(),
        true,
    )?;
    let mut summaries = BudgetVec::new(&budget);
    let mut previous: Option<String> = None;
    let mut previous_inference: Option<String> = None;
    let mut accepted = 0;
    let mut steps = 0;
    let mut last_semantics = None;
    for number in 0..=count {
        control.check()?;
        let path = root.join("generations").join(format!("{number:06}.json"));
        let generation = s.json(&path)?;
        let g = &generation.root;
        g.closed(
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
            ],
            if format.corrected() {
                &["rule_signature", "layout_signature"]
            } else {
                &[]
            },
        )?;
        n(
            g,
            "schema_version",
            if format.corrected() { 3 } else { 2 },
            if format.corrected() { 3 } else { 2 },
        )?;
        n(g, "generation", number as i128, number as i128)?;
        is(
            g,
            "model_version",
            if format.corrected() { "3" } else { "2" },
        )?;
        is(
            g,
            "input_recipe",
            if format.corrected() {
                "participant-structured-v2"
            } else {
                INPUT
            },
        )?;
        eq(
            g.require("model_id")?,
            r.require("model_id")?,
            "generation model mismatch",
        )?;
        eq(
            g.require("run_id")?,
            r.require("run_id")?,
            "generation run mismatch",
        )?;
        classification_context(g, action_contract, format, &budget, &control)?;
        if number == 0 {
            if !g.require("parent_sha256")?.is_null() || !g.require("update_sha256")?.is_null() {
                return Err(Error::contract("generation0 links must be explicitly null"));
            }
        } else {
            if checksum(g.require("parent_sha256")?)? != previous.as_deref().expect("previous") {
                return Err(Error::contract("generation parent link mismatch"));
            }
            checksum(g.require("update_sha256")?)?;
        }
        checksum(g.require("inference_sha256")?)?;
        checksum(g.require("resume_sha256")?)?;
        filemap(g.require("artifact_files")?)?;
        controls(g.require("parent_controls")?, number)?;
        let receipt = reg
            .get(&number.to_string())
            .ok_or_else(|| Error::contract("registry generation missing"))?;
        let protected = number == 0 || number % 10 == 0 || number == count;
        if protected && text(receipt, "status")? != "retained" {
            return Err(Error::contract("required retained generation pruned"));
        }
        let semantics = artifact(
            s,
            &root.join("artifacts").join(format!("{number:06}")),
            g,
            receipt,
            r,
            &pure,
            &shapes,
            format,
        )?;
        let mut summary = clone_fields(
            g,
            &[
                "generation",
                "parent_sha256",
                "update_sha256",
                "inference_sha256",
                "resume_sha256",
                "accepted_decisions",
                "optimizer_steps",
                "parent_controls",
            ],
            &control,
        )?;
        let fields = summary.as_object_mut().expect("summary object");
        fields.insert("generation_sha256".into(), json::text(&s.hash(&path)?));
        fields.insert(
            "artifact_files_sha256".into(),
            json::text(&digest(g.require("artifact_files")?, &budget, &control)?),
        );
        fields.insert("artifact_status".into(), receipt.require("status")?.clone());
        if text(receipt, "status")? == "deleted" {
            fields.insert(
                "pruned_descriptor".into(),
                clone_fields(
                    receipt,
                    &["original_files", "deletion_intent", "deletion_receipt"],
                    &control,
                )?,
            );
        } else {
            fields.insert("pruned_descriptor".into(), Value::Null);
        }
        if number == 0 {
            n(g, "accepted_decisions", 0, 0)?;
            n(g, "optimizer_steps", 0, 0)?;
        } else {
            let update = s.checked_json(
                &root.join("updates").join(format!("{number:06}.json")),
                g.require("update_sha256")?,
            )?;
            let amount = update_metadata(
                s,
                &update.root,
                r,
                &pure,
                &semantics,
                previous.as_deref().expect("previous"),
                previous_inference.as_deref().expect("previous inference"),
                number,
                format,
            )?;
            accepted += amount;
            steps += 4 * ((amount + 127) / 128);
            n(g, "accepted_decisions", accepted, accepted)?;
            n(g, "optimizer_steps", steps, steps)?;
            n(
                update.root.require("metrics")?,
                "optimizer_steps",
                steps,
                steps,
            )?;
        }
        let bytes = json::canonical(&summary, &budget, &control)?;
        summaries.push(json::parse(&bytes, &budget, &control)?)?;
        previous = Some(s.hash(&path)?);
        previous_inference = Some(text(g, "inference_sha256")?.to_owned());
        last_semantics = Some(semantics);
    }
    if previous.as_deref() != Some(text(r, "generation_sha256")?)
        || accepted != total_count
        || steps != total_steps
    {
        return Err(Error::contract("parent final counters/generation mismatch"));
    }
    let clocks = clone_fields(
        r,
        &[
            "created_unix",
            "last_unix",
            "wall_deadline_unix",
            "admission_deadline_unix",
            "execution_deadline_unix",
            "completed_updates",
        ],
        &control,
    )?;
    let mut state_summary = clone_fields(
        r,
        &[
            "run_id",
            "status",
            "seed",
            "updates",
            "accepted_decisions",
            "optimizer_steps",
            "model_id",
            "model_version",
            "input_recipe",
            "config_sha256",
            "resume_identity",
            "generation_sha256",
        ],
        &control,
    )?;
    if format.packed() {
        state_summary
            .as_object_mut()
            .unwrap()
            .insert("schema_version".into(), json::integer(4));
        state_summary.as_object_mut().unwrap().insert(
            "storage_binding".into(),
            json::copy(r.require("storage_binding")?, &control)?,
        );
    }
    if formal {
        for key in self::formal::EXTRA {
            state_summary
                .as_object_mut()
                .unwrap()
                .insert((*key).into(), json::copy(r.require(key)?, &control)?);
        }
    }
    let mut generation_values = Vec::new();
    for item in summaries.iter() {
        control.check()?;
        generation_values.push(json::copy(&item.root, &control)?);
    }
    let mut summary = json::object([
        (
            "schema_version",
            json::integer(if format.packed() {
                4
            } else if format.corrected() {
                3
            } else {
                2
            }),
        ),
        (
            "kind",
            json::text(if format.packed() {
                "card-instance-packed-parent-controls"
            } else if format.corrected() {
                "card-instance-parent-controls"
            } else {
                "structured-parent-controls"
            }),
        ),
        (
            "validation_scope",
            json::text("metadata-maps-filehashes-only"),
        ),
        ("npz_validated", Value::Bool(false)),
        (
            if format.corrected() {
                "journal6_validated"
            } else {
                "journal5_validated"
            },
            Value::Bool(false),
        ),
        ("qualified", Value::Bool(false)),
        ("parent_qualification", json::text("not_performed")),
        (
            "directory",
            json::text(
                root.to_str()
                    .ok_or_else(|| Error::contract("UTF8 parent directory required"))?,
            ),
        ),
        ("state", state_summary),
        ("clocks", clocks),
        ("model", pure),
        ("execution", execution.clone()),
        ("semantics", last_semantics.expect("g0 semantics")),
        (
            "actions_sha256",
            json::text(&digest(cfg.require("actions")?, &budget, &control)?),
        ),
        ("action_contract", action_contract.clone()),
        (
            "purpose",
            json::text(if format.corrected() {
                "qualified_game"
            } else {
                "component"
            }),
        ),
        ("synthetic_capacity", Value::Bool(false)),
        (
            "identity_sha256",
            json::text(&digest(r.require("identity")?, &budget, &control)?),
        ),
        ("generations", Value::Array(generation_values)),
        ("control_fingerprint", json::text(&s.fingerprint()?)),
    ]);
    if format.corrected() {
        for key in ["rule_signature", "layout_signature"] {
            summary
                .as_object_mut()
                .unwrap()
                .insert(key.into(), json::copy(r.require(key)?, &control)?);
        }
    }
    if format.packed() {
        summary.as_object_mut().unwrap().insert(
            "storage_binding".into(),
            json::copy(r.require("storage_binding")?, &control)?,
        );
    }
    let bytes = json::canonical(&summary, &budget, &control)?;
    json::parse(&bytes, &budget, &control)
}

// Tree2 is a concrete non-PPO RAW role. This derives metadata only: neither
// deterministic floating feature transforms nor a Booster runs in this module.
pub(crate) fn tree_model(value: &Value) -> Result<()> {
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
    is(value, "version", "2")?;
    let encoder = value.require("encoder")?;
    encoder.closed(
        &[
            "recipe",
            "max_positions",
            "aggregate_window",
            "aggregate_recipe",
        ],
        &[],
    )?;
    is(encoder, "recipe", "participant-structured-tree-v1")?;
    is(encoder, "aggregate_recipe", "public-exact-v2")?;
    n(encoder, "max_positions", 1, 128)?;
    n(encoder, "aggregate_window", 1, 65536)?;
    value.require("network")?.closed(&["num_leaves"], &[])?;
    n(value.require("network")?, "num_leaves", 2, 65536)?;
    value.require("initialization")?.closed(&["recipe"], &[])?;
    is(
        value.require("initialization")?,
        "recipe",
        "seed-derived-v1",
    )
}
pub(crate) fn tree_execution(options: &Value) -> Result<()> {
    options.closed(
        &[
            "version",
            "checkpoint",
            "expected_manifest_digest",
            "model",
            "inference",
            "resources",
            "tree_descriptor",
            "feature_signature",
            "action_contract",
            "purpose",
            "synthetic_capacity",
        ],
        &["training_binding"],
    )?;
    if let Some(binding) = options.get("training_binding") {
        binding.closed(&["directory", "request_sha256", "completion_sha256"], &[])?;
        if text(binding, "directory")?.is_empty() {
            return Err(Error::contract(
                "Tree2 explicit owned training directory required",
            ));
        }
        checksum(binding.require("request_sha256")?)?;
        checksum(binding.require("completion_sha256")?)?;
    }
    is(options, "version", "2")?;
    if text(options, "checkpoint")?.is_empty() {
        return Err(Error::contract("Tree2 own checkpoint required"));
    }
    checksum(options.require("expected_manifest_digest")?)?;
    checksum(options.require("feature_signature")?)?;
    tree_model(options.require("model")?)?;
    let inference = options.require("inference")?;
    inference.closed(
        &["device", "candidate_chunk", "max_candidates", "num_threads"],
        &[],
    )?;
    is(inference, "device", "cpu")?;
    let chunk = n(inference, "candidate_chunk", 1, 65536)?;
    n(inference, "max_candidates", 1, 65536)?;
    let threads = n(inference, "num_threads", 1, 64)?;
    let resources = options.require("resources")?;
    resources.closed(&["max_owned_bytes", "max_array_bytes"], &[])?;
    let owned = n(resources, "max_owned_bytes", 1, i64::MAX as i128)?;
    let array = n(resources, "max_array_bytes", 1, 64i128 << 20)?;
    let descriptor = options.require("tree_descriptor")?;
    descriptor.closed(
        &["tree_count", "max_depth", "model_sha256", "model_bytes"],
        &[],
    )?;
    n(descriptor, "tree_count", 1, i32::MAX as i128)?;
    n(descriptor, "max_depth", 0, 65535)?;
    checksum(descriptor.require("model_sha256")?)?;
    let bytes = n(descriptor, "model_bytes", 1, 64i128 << 20)?;
    let positions = n(
        options.require("model")?.require("encoder")?,
        "max_positions",
        1,
        128,
    )?;
    let width = 1071 + 32 * positions;
    let total =
        (256i128 << 20) + threads * (64i128 << 20) + 16 * bytes + chunk * (12 * width + 32) + 72008;
    if chunk * width * 4 > array || total > owned {
        return Err(Error::resource(
            "Tree2 RAW declared array/joint resources exceeded",
        ));
    }
    Ok(())
}
pub(crate) fn tree_feature_signature(
    model: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<String> {
    tree_model(model)?;
    let layout = tree_layout(model.require("encoder")?, budget, control)?;
    Ok(layout
        .root
        .require("feature_signature")?
        .string()?
        .to_owned())
}
pub(crate) fn tree_layout(
    encoder: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<Document> {
    encoder.closed(
        &[
            "recipe",
            "max_positions",
            "aggregate_window",
            "aggregate_recipe",
        ],
        &[],
    )?;
    is(encoder, "recipe", "participant-structured-tree-v1")?;
    is(encoder, "aggregate_recipe", "public-exact-v2")?;
    n(encoder, "aggregate_window", 1, 65536)?;
    let positions = n(encoder, "max_positions", 1, 128)? as usize;
    let width = 1071 + 32 * positions;
    let _layout = budget.reserve(1024 * width + 65536)?;
    let recipe = json::parse(EXACT_RECIPE, budget, control)?;
    let state = recipe.root.require("state_numeric")?.array()?;
    let card = recipe.root.require("card_numeric")?.array()?;
    let mut fields: Vec<(String, &'static str)> = Vec::with_capacity(width);
    for (prefix, transform) in [("state.numeric", "numeric"), ("state.present", "presence")] {
        for name in state {
            fields.push((format!("{prefix}.{}", name.string()?), transform));
        }
    }
    let state_categories = [
        "phase",
        "required_choice",
        "first_player",
        "active_player",
        "current_player",
        "result",
        "own.disease",
        "own.guardian",
        "opponent.disease",
        "opponent.guardian",
    ];
    for (prefix, transform) in [
        ("state.category", "category"),
        ("state.category_present", "presence"),
    ] {
        for name in state_categories {
            fields.push((format!("{prefix}.{name}"), transform));
        }
    }
    fn add_card(
        fields: &mut Vec<(String, &'static str)>,
        prefix: &str,
        card: &[Value],
    ) -> Result<()> {
        for (part, transform) in [("numeric", "numeric"), ("present", "presence")] {
            for name in card {
                fields.push((format!("{prefix}.{part}.{}", name.string()?), transform));
            }
        }
        for name in [
            "definition",
            "ability",
            "kind",
            "attribute",
            "disease",
            "guardian",
            "source",
        ] {
            fields.push((format!("{prefix}.category.{name}"), "category"));
        }
        fields.push((format!("{prefix}.mask"), "presence"));
        Ok(())
    }
    for slot in 0..18 {
        add_card(&mut fields, &format!("own.{slot}"), card)?;
    }
    for slot in 0..18 {
        fields.push((format!("opponent.learned.{slot}"), "category"));
    }
    for name in recipe.root.require("event_categories")?.array()?[0].array()? {
        fields.push((format!("history.event.{}", name.string()?), "numeric"));
    }
    for field in [
        "player",
        "attacker",
        "target",
        "active_player",
        "buyer",
        "seller",
        "from",
        "to",
    ] {
        for role in ["self", "opponent"] {
            fields.push((format!("history.actor.{field}.{role}"), "numeric"));
        }
    }
    for name in recipe.root.require("event_numeric")?.array()? {
        for stat in ["sum", "present"] {
            fields.push((
                format!("history.numeric.{}.{stat}", name.string()?),
                "numeric",
            ));
        }
    }
    for name in recipe.root.require("history_append")?.array()? {
        fields.push((format!("history.{}", name.string()?), "numeric"));
    }
    fields.push(("candidate.kind".into(), "category"));
    fields.push(("candidate.target".into(), "category"));
    for (prefix, transform) in [
        ("candidate.numeric", "numeric"),
        ("candidate.present", "presence"),
    ] {
        for name in ["hp", "mp", "money", "accept"] {
            fields.push((format!("{prefix}.{name}"), transform));
        }
    }
    fields.push(("candidate.length".into(), "numeric"));
    for slot in 0..18 {
        fields.push((format!("candidate.used_slot.{slot}"), "numeric"));
    }
    for position in 0..positions {
        add_card(&mut fields, &format!("candidate.position.{position}"), card)?;
    }
    if fields.len() != width {
        return Err(Error::contract("Tree2 pinned layout width mismatch"));
    }
    control.check()?;
    let mut value = json::object([
        ("recipe", json::text("participant-structured-tree-v1")),
        (
            "numeric_transform",
            json::text("numpy-fp32-signed-log1p-v1"),
        ),
        ("encoder", json::copy(encoder, control)?),
        (
            "token_signature",
            json::text(&token_signature(
                n(encoder, "aggregate_window", 1, 65536)?,
                false,
                budget,
                control,
            )?),
        ),
        (
            "feature_names",
            json::array(fields.iter().map(|(name, _)| json::text(name))),
        ),
        (
            "categorical_indices",
            json::array(
                fields
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, kind))| *kind == "category")
                    .map(|(index, _)| json::integer(index as i128)),
            ),
        ),
        (
            "transforms",
            json::array(fields.iter().map(|(_, kind)| json::text(kind))),
        ),
    ]);
    let signature = digest(&value, budget, control)?;
    value
        .as_object_mut()
        .unwrap()
        .insert("feature_signature".into(), json::text(&signature));
    let bytes = json::canonical(&value, budget, control)?;
    json::parse(&bytes, budget, control)
}

#[cfg(test)]
mod tree_metadata_tests {
    use super::*;
    #[test]
    fn exact_tree_feature_signatures_match_independent_python_metadata() {
        let budget = Budget::new(256 << 20);
        let control = Control::new(crate::budget::monotonic() + 30.0).unwrap();
        let cases = json::parse(
            include_bytes!("../tests/fixtures/structured_tree_metadata.json"),
            &budget,
            &control,
        )
        .unwrap();
        for case in cases.root.array().unwrap() {
            assert_eq!(
                tree_feature_signature(case.require("model").unwrap(), &budget, &control).unwrap(),
                case.require("feature_signature").unwrap().string().unwrap()
            );
        }
    }
}

// Corrected pure metadata validation. No files, Torch, learned policy or Source
// permission are supplied by these helpers; NPZ values remain the sealed owner.
fn corrected_pins(budget: &Arc<Budget>, control: &Control) -> Result<Document> {
    json::parse(
        include_bytes!("../assets/card-instance-validation-pins-v2.json"),
        budget,
        control,
    )
}
pub(crate) fn contract3(value: &Value, budget: &Arc<Budget>, control: &Control) -> Result<()> {
    if value == crate::structured_journal::card_instance::defense_v3_card_instance_contract() {
        return control.check();
    }
    let pins = corrected_pins(budget, control)?;
    let expected = pins.root.require("action_contract")?;
    value.closed(
        &expected
            .object()?
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        &[],
    )?;
    eq(value, expected, "corrected contract mismatch")
}
pub(crate) fn game_binding3(value: &Value, budget: &Arc<Budget>, control: &Control) -> Result<()> {
    use crate::structured_journal::card_instance::WireProfile;
    let action = value.require("action_contract")?;
    contract3(action, budget, control)?;
    let profile = WireProfile::expected(action)?;
    eq(
        value.require("rule_signature")?,
        profile.rule_signature(),
        "game rule signature mismatch",
    )?;
    is(value, "layout_signature", CARD_INSTANCE_SIGNATURE)?;
    is(value, "purpose", "qualified_game")?;
    eq(
        value.require("synthetic_capacity")?,
        &Value::Bool(false),
        "game rejects synthetic capacity",
    )
}
pub(crate) fn actions3(value: &Value, budget: &Arc<Budget>, control: &Control) -> Result<()> {
    value.closed(
        &[
            "action_contract",
            "version",
            "generator",
            "use_action_options",
            "targets",
            "max_sequence_length",
            "max_candidates",
            "max_expansions",
            "max_temporary_candidates",
            "count_states",
            "count_milliseconds",
            "search_milliseconds",
            "channel_weights",
        ],
        &[],
    )?;
    contract3(value.require("action_contract")?, budget, control)?;
    is(value, "version", "3")?;
    is(value, "generator", "public-card-instance-diverse-v5")?;
    eq(
        value.require("use_action_options")?,
        &Value::Bool(true),
        "actions3 requires atomic hints",
    )?;
    for (key, high) in [
        ("max_sequence_length", 18),
        ("max_candidates", 65536),
        ("max_expansions", i32::MAX as i128),
        ("max_temporary_candidates", i32::MAX as i128),
        ("count_states", i32::MAX as i128),
        ("count_milliseconds", 1000),
        ("search_milliseconds", 10000),
    ] {
        n(value, key, 1, high)?;
    }
    let targets = value.require("targets")?.array()?;
    if targets.is_empty()
        || targets.len() > 2
        || targets
            .iter()
            .any(|v| !["self", "opponent"].contains(&v.as_str().unwrap_or("")))
        || targets.len() == 2 && targets[0] == targets[1]
    {
        return Err(Error::contract("actions3 target order/set mismatch"));
    }
    let weights = value.require("channel_weights")?.array()?;
    if weights.len() != 7 {
        return Err(Error::contract("actions3 requires seven channels"));
    }
    for weight in weights {
        weight.integer(1, i32::MAX as i128)?;
    }
    Ok(())
}
pub(crate) fn architecture3(
    model: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<String> {
    model3(model, budget, control)?;
    let _copy = budget.reserve(65536)?;
    let mut fields = json::copy(model, control)?;
    fields.as_object_mut().unwrap().remove("resources");
    digest(
        &json::object([
            ("model", fields),
            (
                "token_signature",
                json::text(&card_instance_token_signature(
                    n(model.require("event")?, "aggregate_window", 1, 65536)? as usize,
                    budget,
                    control,
                )?),
            ),
        ]),
        budget,
        control,
    )
}
fn semantic3(value: &Value, model: &Value, budget: &Arc<Budget>, control: &Control) -> Result<()> {
    semantic_corrected(value, model, budget, control, false)
}
fn semantic_corrected(
    value: &Value,
    model: &Value,
    budget: &Arc<Budget>,
    control: &Control,
    hpo: bool,
) -> Result<()> {
    let pins = corrected_pins(budget, control)?;
    for (key, expected) in [
        ("input_recipe", "participant-structured-v2"),
        ("generator_version", "public-card-instance-diverse-v5"),
        ("policy_input_version", "player-card-refs-v2"),
        ("end_projection_version", "simulator-closure-v1"),
        ("reward_policy_version", "natural-terminal-v1"),
        ("numeric_transform", "signed-log1p-fp32-v1"),
    ] {
        if hpo && key == "reward_policy_version" {
            continue;
        }
        is(value, key, expected)?;
    }
    use crate::structured_journal::card_instance::{WireProfile, defense_v3_release};
    let profile = WireProfile::expected(value.require("action_contract")?)?;
    let contract = profile.frame_contract();
    for key in ["simulator_revision", "spec_inventory_sha256"] {
        eq(
            value.require(key)?,
            contract.require(key)?,
            "corrected semantic revision/spec mismatch",
        )?;
    }
    eq(
        value.require("reviewed_spec_hashes")?,
        if profile == WireProfile::DefenseV3 {
            defense_v3_release().require("spec_files")?
        } else {
            pins.root.require("reviewed_spec_hashes")?
        },
        "corrected reviewed specs mismatch",
    )?;
    eq(
        value.require("catalog")?,
        &profile.catalog_metadata(),
        "corrected catalog mismatch",
    )?;
    is(
        value,
        "token_signature",
        &card_instance_token_signature(
            n(model.require("event")?, "aggregate_window", 1, 65536)? as usize,
            budget,
            control,
        )?,
    )?;
    runtime_semantics(value)
}
pub(crate) fn inference4(
    value: &Value,
    actions: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<()> {
    let encoded = json::canonical(value, budget, control)?;
    let _metadata = json::admit(encoded.len(), budget)?;
    actions3(actions, budget, control)?;
    let mut keys = vec![
        "schema_version",
        "kind",
        "model",
        "architecture_identity",
        "action_config_hash",
        "producer",
        "weights",
        "rule_signature",
        "layout_signature",
        "spec_inventory_sha256",
    ];
    keys.extend(SEMANTICS);
    keys.extend(CLASS);
    value.closed(&keys, &[])?;
    n(value, "schema_version", 4, 4)?;
    is(value, "kind", "structured-inference")?;
    game_binding3(value, budget, control)?;
    producer(value.require("producer")?)?;
    let model = value.require("model")?;
    let shapes = model3(model, budget, control)?;
    n(model, "max_positions", 1, 18)?;
    semantic3(value, model, budget, control)?;
    is(
        value,
        "architecture_identity",
        &architecture3(model, budget, control)?,
    )?;
    is(
        value,
        "action_config_hash",
        &digest(actions, budget, control)?,
    )?;
    let descriptors = bundle(value.require("weights")?)?;
    is(value.require("weights")?, "prefix", "weights")?;
    if descriptors.len() != shapes.len() {
        return Err(Error::contract("inference4 named weight count mismatch"));
    }
    for (name, shape) in shapes {
        let descriptor = descriptors
            .get(&name)
            .ok_or_else(|| Error::contract("inference4 named weight missing"))?;
        is(descriptor, "dtype", "float32")?;
        eq(
            descriptor.require("shape")?,
            &json::array(shape.into_iter().map(json::integer)),
            "inference4 weight shape mismatch",
        )?;
    }
    Ok(())
}
/// Pure safe-options validation and exact label. Resources/runtime are bound in
/// the label but no current machine, artifact path or CUDA process is queried.
pub(crate) fn options3(
    value: &Value,
    actions: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<String> {
    options_context(value, actions, budget, control, false)
}
fn options_context(
    value: &Value,
    actions: &Value,
    budget: &Arc<Budget>,
    control: &Control,
    hpo: bool,
) -> Result<String> {
    actions3(actions, budget, control)?;
    value.closed(
        &[
            "version",
            "checkpoint",
            "expected_manifest_digest",
            "model",
            "inference",
            "cuda_visible_devices",
            "action_config_hash",
            "action_contract",
            "rule_signature",
            "layout_signature",
            "purpose",
            "synthetic_capacity",
        ],
        &[],
    )?;
    is(value, "version", if hpo { "4" } else { "3" })?;
    validation_text(value.require("checkpoint")?)?;
    game_binding3(value, budget, control)?;
    is(
        value,
        "action_config_hash",
        &digest(actions, budget, control)?,
    )?;
    let model = value.require("model")?;
    if hpo {
        model4(model, budget, control)?;
    } else {
        model3(model, budget, control)?;
    }
    n(model, "max_positions", 1, 18)?;
    let inference = value.require("inference")?;
    binding(inference)?;
    visibility(value.require("cuda_visible_devices")?)?;
    let manifest = checksum(value.require("expected_manifest_digest")?)?;
    let _copy = budget.reserve(65536)?;
    let mut identity = json::object([
        (
            "execution",
            json::object([
                ("inference", json::copy(inference, control)?),
                (
                    "resources",
                    json::copy(model.require("resources")?, control)?,
                ),
                (
                    "cuda_visible_devices",
                    json::copy(value.require("cuda_visible_devices")?, control)?,
                ),
            ]),
        ),
        (
            "action_config_hash",
            json::copy(value.require("action_config_hash")?, control)?,
        ),
    ]);
    for key in [
        "action_contract",
        "rule_signature",
        "layout_signature",
        "purpose",
        "synthetic_capacity",
    ] {
        identity
            .as_object_mut()
            .unwrap()
            .insert(key.into(), json::copy(value.require(key)?, control)?);
    }
    Ok(format!(
        "{}-ppo-v{}-{manifest}-{}",
        text(model, "model_id")?,
        if hpo { "4" } else { "3" },
        digest(&identity, budget, control)?
    ))
}
fn validation_text(value: &Value) -> Result<()> {
    crate::validation::text_value(value)
}

/// Exact Python resource_estimate formulas; all products are i128 after closed
/// bounds, and each declared tensor remains below its configured64MiB limit.
pub(crate) fn resource3(
    model: &Value,
    inference: &Value,
    candidates: Option<i128>,
    suffix: Option<i128>,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<Value> {
    let hpo = text(model, "version")? == "4";
    let shapes = if hpo {
        model4(model, budget, control)?
    } else {
        model3(model, budget, control)?
    };
    binding(inference)?;
    let recipe = json::parse(CARD_INSTANCE_RECIPE, budget, control)?;
    let cats = |name: &str| -> Result<Vec<i128>> {
        recipe
            .root
            .require(name)?
            .array()?
            .iter()
            .map(|v| Ok(v.array()?.len() as i128 + 2))
            .collect()
    };
    let sc = cats("state_categories")?;
    let cc = cats("card_categories")?;
    let ec = cats("event_categories")?;
    let sn = recipe.root.require("state_numeric")?.array()?.len() as i128;
    let cn = recipe.root.require("card_numeric")?.array()?.len() as i128;
    let en = recipe.root.require("event_numeric")?.array()?.len() as i128;
    let state_width = 2 * sn + sc.len() as i128 + sc.iter().sum::<i128>() + 18 * cc[0];
    let card_direct = 2 * cn + cc.iter().sum::<i128>();
    let event_width = 2 * en + ec.iter().sum::<i128>();
    let temporal = model.require("temporal")?;
    let hand = model.require("hand")?;
    let event = model.require("event")?;
    let heads = model.require("heads")?;
    let decisions = n(temporal, "max_decisions", 1, 1024)?;
    let interval = n(event, "max_interval", 1, 65536)?;
    let count = candidates.unwrap_or(n(inference, "max_candidates", 1, 65536)?);
    if count < 1 || count > n(inference, "max_candidates", 1, 65536)? {
        return Err(Error::contract("resource candidate count mismatch"));
    }
    let suffix = if let Some(suffix) = suffix {
        suffix
    } else if hpo {
        1
    } else {
        decisions.min(n(temporal, "sequence_length", 1, decisions)?)
    };
    if suffix < 1 || suffix > decisions {
        return Err(Error::contract("resource suffix mismatch"));
    }
    let positions = n(model, "max_positions", 1, 128)?;
    let co = n(model.require("card")?, "output_size", 1, 1024)?;
    let state = if text(model.require("state")?, "kind")? == "identity" {
        state_width
    } else {
        n(model.require("state")?, "output_size", 1, 1024)?
    };
    let frame = state
        + if text(hand, "kind")? == "direct" {
            18 * card_direct
        } else {
            n(hand, "latent_size", 1, 1024)?
        }
        + n(event, "hidden_size", 1, 1024)?
        + 387;
    let candidate = 26;
    let actor = n(temporal, "hidden_size", 1, 1024)? + candidate + positions * (co + cc[10] + 2);
    let widths = widths(heads.require("actor_hidden_sizes")?)?;
    let first = widths[0];
    let chunk = n(inference, "candidate_chunk", 1, 65536)?.min(count);
    let eh = n(event, "hidden_size", 1, 1024)?;
    let el = n(event, "num_layers", 1, 4)?;
    let tl = n(temporal, "num_layers", 1, 4)?;
    let th = n(temporal, "hidden_size", 1, 1024)?;
    let tensors = [
        ("raw_state", decisions * state_width * 4),
        ("raw_cards", decisions * 18 * card_direct * 4),
        ("interval_tokens", interval * event_width * 4),
        ("interval_gru", interval * eh * el * 4 * 12),
        (
            "hand_attention",
            suffix
                * n(hand, "attention_layers", 1, 4)?
                * n(hand, "attention_heads", 1, 16)?
                * 18
                * 18
                * 4
                * 8,
        ),
        ("frame_history", decisions * frame * 4),
        (
            "temporal_attention",
            tl * n(temporal, "num_heads", 1, 16)? * decisions * decisions * 4,
        ),
        ("temporal_cache", 2 * tl * decisions * th * 4),
        ("candidate_references", count * positions * 8),
        ("candidate_columns", count * (candidate + 2) * 4),
        ("chunk_dense_reference", chunk * actor * 4),
        ("chunk_position_gather", chunk * positions * first * 4),
        (
            "chunk_checkpoint_replay",
            chunk * widths.iter().sum::<i128>() * 4 * 8,
        ),
        ("position_card_projection", positions * 19 * first * 4),
        (
            "position_card_projection_unpadded",
            positions * 18 * first * 4,
        ),
        ("position_card_projection_padding", positions * first * 4),
        ("global_logits_probability_gradient", count * 4 * 5),
    ];
    let resources = model.require("resources")?;
    if tensors
        .iter()
        .any(|(_, v)| *v > n(resources, "max_tensor_bytes", 1, 64i128 << 20).unwrap_or(0))
    {
        return Err(Error::resource("resource tensor cap"));
    }
    let parameters = shapes
        .values()
        .map(|s| s.iter().product::<i128>())
        .sum::<i128>();
    let parameter_bytes = parameters * 4 * 6 + shapes.len() as i128 * 8;
    let context = n(resources, "workspace_bytes", 1, 4i128 << 30)?
        + if text(inference, "device")? == "cuda:0" {
            n(resources, "cuda_context_bytes", 1, 4i128 << 30)?
        } else {
            0
        };
    let activation = 4
        * (tensors.iter().map(|(_, v)| v).sum::<i128>()
            + (suffix - 1) * (interval * event_width * 4 + interval * eh * el * 4 * 12));
    let total = parameter_bytes + context + activation;
    if total
        > n(resources, "max_owned_bytes", 1, 12i128 << 30)?.min(n(
            inference,
            "managed_budget_bytes",
            1,
            12i128 << 30,
        )?)
    {
        return Err(Error::resource("resource managed cap"));
    }
    Ok(json::object([
        (
            "parameter_optimizer_cpu_reference_bytes",
            json::integer(parameter_bytes),
        ),
        ("context_workspace_bytes", json::integer(context)),
        ("activation_and_copy_bytes", json::integer(activation)),
        ("total_bytes", json::integer(total)),
        ("parameters", json::integer(parameters)),
        (
            "tensor_bytes",
            json::object(tensors.into_iter().map(|(k, v)| (k, json::integer(v)))),
        ),
    ]))
}

pub(crate) fn worker_settings3(
    model: &Value,
    inference: &Value,
    actions: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<(Value, Value)> {
    actions3(actions, budget, control)?;
    let _metadata = budget.reserve(65536)?;
    let peak = resource3(model, inference, None, Some(1), budget, control)?;
    let fixed = resource3(model, inference, Some(1), Some(1), budget, control)?;
    if n(actions, "max_sequence_length", 1, 18)? > n(model, "max_positions", 1, 18)?
        || n(actions, "max_candidates", 1, 65536)? > n(inference, "max_candidates", 1, 65536)?
    {
        return Err(Error::contract("game model/actions capacity mismatch"));
    }
    let event = model.require("event")?;
    let temporal = model.require("temporal")?;
    let heads = model.require("heads")?;
    let co = n(model.require("card")?, "output_size", 1, 1024)?;
    let positions = n(model, "max_positions", 1, 18)?;
    let actor = n(temporal, "hidden_size", 1, 1024)? + 26 + positions * (co + 7);
    let mut widths = widths(heads.require("actor_hidden_sizes")?)?;
    let hidden = widths[0];
    let mut ops = hidden;
    widths.push(1);
    for pair in widths.windows(2) {
        ops += 2 * pair[0] * pair[1] + pair[1];
    }
    ops += widths[1..widths.len() - 1].iter().sum::<i128>();
    let caps = json::object([
        (
            "max_interval_events",
            json::copy(event.require("max_interval")?, control)?,
        ),
        (
            "max_episode_events",
            json::copy(event.require("max_episode")?, control)?,
        ),
        (
            "max_decisions",
            json::copy(temporal.require("max_decisions")?, control)?,
        ),
        (
            "max_positions",
            json::copy(actions.require("max_sequence_length")?, control)?,
        ),
        (
            "max_candidates",
            json::copy(actions.require("max_candidates")?, control)?,
        ),
    ]);
    let costs = json::object([
        ("version", json::text("proposal-cost-v1")),
        (
            "context_width",
            json::copy(temporal.require("hidden_size")?, control)?,
        ),
        ("card_width", json::integer(co + 7)),
        ("actor_input_width", json::integer(actor)),
        ("actor_hidden_width", json::integer(hidden)),
        ("position_capacity", json::integer(positions)),
        (
            "candidate_chunk",
            json::copy(inference.require("candidate_chunk")?, control)?,
        ),
        ("tail_operations_per_candidate", json::integer(ops)),
        (
            "tail_working_bytes_per_candidate",
            json::integer(widths.iter().sum::<i128>() * 32),
        ),
        (
            "tail_tensor_bytes_per_candidate",
            json::integer(widths.iter().max().unwrap() * 4),
        ),
        (
            "fixed_bytes",
            json::copy(fixed.require("total_bytes")?, control)?,
        ),
        (
            "max_preparation_bytes",
            json::integer(
                n(
                    model.require("resources")?,
                    "max_owned_bytes",
                    1,
                    12i128 << 30,
                )?
                .min(n(inference, "managed_budget_bytes", 1, 12i128 << 30)?),
            ),
        ),
        ("max_scoring_operations", json::integer(i64::MAX as i128)),
        ("max_storage_bytes", json::integer(64i128 << 20)),
        (
            "max_tensor_bytes",
            json::copy(
                model.require("resources")?.require("max_tensor_bytes")?,
                control,
            )?,
        ),
    ]);
    drop(peak);
    Ok((caps, costs))
}
pub(crate) fn neural_metadata3(
    value: &Value,
    actions: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<String> {
    neural_metadata_context(value, actions, budget, control, false)
}
fn neural_metadata_context(
    value: &Value,
    actions: &Value,
    budget: &Arc<Budget>,
    control: &Control,
    hpo: bool,
) -> Result<String> {
    let encoded = json::canonical(value, budget, control)?;
    let _metadata = json::admit(encoded.len(), budget)?;
    value.closed(
        &[
            "input_contract",
            "policy_options",
            "inference_manifest",
            "files",
        ],
        &[],
    )?;
    is(value, "input_contract", "retained-external-inference-v1")?;
    let options = value.require("policy_options")?;
    let label = options_context(options, actions, budget, control, hpo)?;
    let manifest = value.require("inference_manifest")?;
    if hpo {
        self::hpo::inference(manifest, actions, budget, control)?;
    } else {
        inference4(manifest, actions, budget, control)?;
    }
    let mut pure = json::copy(options.require("model")?, control)?;
    if hpo {
        pure.as_object_mut().unwrap().remove("resources");
        let mut encoded = json::canonical(manifest, budget, control)?;
        encoded.push(b'\n')?;
        is(
            options,
            "expected_manifest_digest",
            &format!("{:x}", Sha256::digest(&*encoded)),
        )?;
    }
    eq(
        manifest.require("model")?,
        &pure,
        "neural manifest/options model mismatch",
    )?;
    let inference = options.require("inference")?;
    if text(inference, "device")? == "cpu"
        && options.require("cuda_visible_devices")?.as_str() != Some("")
    {
        return Err(Error::contract("CPU child visibility must be empty"));
    }
    worker_settings3(
        options.require("model")?,
        inference,
        actions,
        budget,
        control,
    )?;
    let mut expected = manifest
        .require("weights")?
        .require("files")?
        .object()?
        .clone();
    expected.insert(
        "inference.json".into(),
        json::copy(options.require("expected_manifest_digest")?, control)?,
    );
    let files = value.require("files")?.array()?;
    let mut seen = std::collections::BTreeSet::new();
    if files.len() != expected.len() {
        return Err(Error::contract("neural external file count mismatch"));
    }
    for file in files {
        control.check()?;
        file.closed(&["path", "sha256", "identity"], &[])?;
        let name = text(file, "path")?;
        basename(name)?;
        if !seen.insert(name) || expected.get(name) != Some(file.require("sha256")?) {
            return Err(Error::contract("neural external file/hash mismatch"));
        }
        checksum(file.require("sha256")?)?;
        let identity = file.require("identity")?.array()?;
        if identity.len() != 5 {
            return Err(Error::contract("neural stat identity shape"));
        }
        for number in identity {
            number.integer(0, i128::MAX)?;
        }
    }
    Ok(label)
}

fn validate_preprocessing3(state: &Value, b: &Arc<Budget>, c: &Control) -> Result<()> {
    use crate::structured_journal::card_instance::{WireProfile, defense_v3_release};
    let action = state.require("action_contract")?;
    let profile = WireProfile::expected(action)?;
    let identity = state.require("identity")?;
    let v = identity.require("preprocessing")?;
    let settings = state
        .require("effective_config")?
        .require("preprocessing")?;
    let backend = text(v, "backend")?;
    let mut keys = vec!["backend", "reader_contract", "implementation"];
    keys.extend(GAME_CLASS);
    if backend == "rust" {
        keys.push("binary");
    }
    v.closed(&keys, &[])?;
    game_binding3(v, b, c)?;
    eq(v.require("action_contract")?, action, "preprocessing action contract mismatch")?;
    is(v, "reader_contract", "participant-structured-v2")?;
    is(settings, "reader", "participant-structured-v2")?;
    eq(
        settings.require("backend")?,
        v.require("backend")?,
        "parent preprocessing backend differs",
    )?;
    match backend {
        "python" => is(v, "implementation", "python-reference-player-card-refs-v2")?,
        "rust" => {
            is(v, "implementation", "rust-player-card-refs-v2")?;
            let binary = v.require("binary")?;
            binary.closed(
                &[
                    "backend",
                    "extension_path",
                    "extension_sha256",
                    "build",
                    "python_abi",
                    "gil_disabled",
                ],
                &[],
            )?;
            is(binary, "backend", "native")?;
            checksum(binary.require("extension_sha256")?)?;
            if !text(binary, "extension_path")?.starts_with('/')
                || binary.require("gil_disabled")? != &Value::Bool(false)
            {
                return Err(Error::contract("recorded native binary path/GIL mismatch"));
            }
            is(binary, "python_abi", "cpython-314")?;
            let build = binary.require("build")?;
            build.object()?;
            checksum(build.require("native_source_sha256")?)?;
            is(
                build,
                "card_instance_contract",
                "participant-structured-v2;journal6;rollout4;generation3;model3",
            )?;
            is(build, "card_instance_signature", CARD_INSTANCE_SIGNATURE)?;
            is(build, "card_instance_models", "mlp,gru,transformer")?;
            is(build, "abi", "cp314-gil")?;
            if profile == WireProfile::DefenseV3 {
                is(build, "card_instance_defense_v3", crate::structured_journal::card_instance::DEFENSE_V3_PROFILE)?;
            }
        }
        _ => {
            return Err(Error::contract(
                "recorded game preprocessing backend unsupported",
            ));
        }
    }
    let integration = identity.require("integration")?;
    let p = corrected_pins(b, c)?;
    let (binary, specs) = if profile == WireProfile::DefenseV3 {
        let release = defense_v3_release();
        (release.require("binary_sha256")?, release.require("spec_files")?)
    } else {
        (p.root.require("approved_binary_sha256")?, p.root.require("approved_spec_files")?)
    };
    eq(
        integration.require("binary_sha256")?,
        binary,
        "parent runtime binary mismatch",
    )?;
    for key in ["declared_simulator_revision", "source_revision"] {
        eq(
            integration.require(key)?,
            action.require("simulator_revision")?,
            "parent runtime source mismatch",
        )?;
    }
    checksum(integration.require("server_config_sha256")?)?;
    eq(
        integration.require("source_spec_sha256")?,
        specs,
        "parent runtime normative inventory mismatch",
    )
}
pub(crate) fn binding3_value(value: &Value, c: &Control) -> Result<Value> {
    clone_fields(value, GAME_CLASS, c)
}
pub(crate) fn semantic3_declaration(value: &Value, c: &Control) -> Result<Value> {
    clone_fields(value, &SourceFormat::Corrected.semantics(), c)
}
pub(crate) fn bundle_metadata(value: &Value) -> Result<BTreeMap<String, &Value>> {
    bundle(value)
}
pub(crate) fn corrected_game_binding(
    profile: crate::structured_journal::card_instance::WireProfile,
    c: &Control,
) -> Result<Value> {
    Ok(json::object([
        (
            "action_contract",
            json::copy(profile.frame_contract(), c)?,
        ),
        (
            "rule_signature",
            json::copy(profile.rule_signature(), c)?,
        ),
        ("layout_signature", json::text(CARD_INSTANCE_SIGNATURE)),
        ("purpose", json::text("qualified_game")),
        ("synthetic_capacity", Value::Bool(false)),
    ]))
}

/// Validate the newly derived paired artifact before lineage publication, using
/// the same sealed metadata/file kernels. This never constructs a generation.
pub(crate) fn verify_derived_artifact3(
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
                .insert(name.clone(), json::copy(sha, &c)?)
                .is_some()
            {
                return Err(Error::contract("derived artifact file collision"));
            }
        }
    }
    files["inference.json"] = json::text(&inf_sha);
    files["resume.json"] = json::text(&res_sha);
    let record = json::object([
        ("inference_sha256", json::text(&inf_sha)),
        ("resume_sha256", json::text(&res_sha)),
        ("artifact_files", json::copy(&files, &c)?),
        ("optimizer_steps", json::integer(steps as i128)),
    ]);
    let receipt = json::object([
        ("status", json::text("retained")),
        ("inference_sha256", json::text(&inf_sha)),
        ("original_files", files),
        ("deletion_intent", Value::Null),
        ("deletion_receipt", Value::Null),
    ]);
    let model = pure_model(state.require("effective_config")?, &c)?;
    let shapes = model3(&model, &b, &c)?;
    artifact(
        s,
        &path,
        &record,
        &receipt,
        state,
        &model,
        &shapes,
        SourceFormat::Corrected,
    )?;
    Ok(record)
}


#[cfg(test)]
mod defense_priority_tests {
    use super::*;

    #[test]
    fn captured_old_mlp_and_final64_bindings_are_explicitly_unsupported() {
        let b = Budget::new(512 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let mlp = json::parse(
            include_bytes!("../tests/fixtures/card_instance_actual_mlp_reference.json"),
            &b,
            &c,
        )
        .unwrap();
        let final64 = json::parse(
            include_bytes!("../tests/fixtures/card_instance_final64_originals.json"),
            &b,
            &c,
        )
        .unwrap();
        let hex = final64.root["files"]["manifest.json"].string().unwrap();
        let bytes: Vec<_> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let manifest = json::parse(&bytes, &b, &c).unwrap();
        for binding in [&mlp.root["collections"][0]["proof"], &manifest.root] {
            assert_eq!(
                binding["rule_signature"].string().unwrap(),
                "3f55f044225fb20c63d683145cdceb397c36f0b7161c7b16793f72210db3e86d"
            );
            assert_eq!(
                game_binding3(binding, &b, &c).err(),
                Some(Error::contract("game rule signature mismatch"))
            );
        }
    }

    #[test]
    fn current_binding_accepts_defense_rules_and_rejects_superseded_rules() {
        let b = Budget::new(128 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let mut binding = corrected_game_binding(Default::default(), &c).unwrap();
        binding["rule_signature"] =
            json::text("081fa1e0b6c186c563c949e65fbce19b77f07db505516330d3ac377b1b02e2d6");
        game_binding3(&binding, &b, &c).unwrap();
        for old in [
            "3f55f044225fb20c63d683145cdceb397c36f0b7161c7b16793f72210db3e86d",
            "0000000000000000000000000000000000000000000000000000000000000000",
        ] {
            binding["rule_signature"] = json::text(old);
            assert!(game_binding3(&binding, &b, &c).is_err());
        }
    }
}

#[cfg(test)]
mod refresh_opponents_tests {
    use super::*;
    #[test]
    fn mixed_training_descriptor_and_registry_are_closed_and_preserve_large_thresholds() {
        let b = Budget::new(128 << 20);
        let c = Control::new(crate::budget::monotonic() + 30.).unwrap();
        let fixture = json::parse(include_bytes!("../tests/fixtures/card_instance_journal6_rules_v2.json"), &b, &c).unwrap();
        let actions = fixture.root["journals"][0]["rows"][0]["actions"].clone();
        let descriptor = json::object([
            ("version", json::text("1")), ("preset", json::text("mixed-refresh-16-v1")),
            ("refresh", json::object([("version", json::text("1")), ("attack_threshold", json::integer(12))]))]);
        let mut cfg = json::object([("actions", actions), ("training_opponents", descriptor)]);
        for threshold in [json::integer(12), json::parse(b"1361129467683753853853498429727072845824", &b, &c).unwrap().root.clone()] {
            cfg["training_opponents"]["refresh"]["attack_threshold"] = threshold;
            let options = refresh_training(&cfg, &b, &c).unwrap().unwrap();
            let label = crate::structured_journal::card_instance::refresh_label(&cfg["actions"], options, &b, &c).unwrap();
            let mut registry = json::object([]);
            registry.as_object_mut().unwrap().insert(label.clone(), json::object([
                ("model_id", json::text("heuristic_refresh")), ("policy_options", options.clone())]));
            assert_eq!(refresh_training_registry(&cfg, &registry, &b, &c).unwrap(), Some(label.clone()));
            let mut wrong = registry.clone();
            wrong[&label]["policy_options"]["attack_threshold"] = json::integer(13);
            assert!(refresh_training_registry(&cfg, &wrong, &b, &c).is_err());
            assert!(refresh_training_registry(&cfg, &json::object([]), &b, &c).is_err());
            let mut old = cfg.clone();
            old.as_object_mut().unwrap().remove("training_opponents");
            assert!(refresh_training_registry(&old, &registry, &b, &c).is_err());
        }
        for (key, bad) in [("version", json::text("2")), ("preset", json::text("other")), ("extra", Value::Null)] {
            let mut wrong = cfg.clone();
            wrong["training_opponents"].as_object_mut().unwrap().insert(key.into(), bad);
            assert!(refresh_training(&wrong, &b, &c).is_err());
        }
        for threshold in [json::integer(0), json::integer(-1), Value::Bool(true), json::float(12.).unwrap()] {
            let mut wrong = cfg.clone();
            wrong["training_opponents"]["refresh"]["attack_threshold"] = threshold;
            assert!(refresh_training(&wrong, &b, &c).is_err());
        }
        let descriptor = cfg.as_object_mut().unwrap().remove("training_opponents").unwrap();
        cfg.as_object_mut().unwrap().insert("hpo_spec".into(), json::object([
            ("schema_version", json::integer(2)), ("training_opponents", descriptor),
            ("training_recipe", json::object([("collection_games", json::integer(16))]))]));
        assert!(refresh_training(&cfg, &b, &c).unwrap().is_some());
        cfg["hpo_spec"]["training_recipe"]["collection_games"] = json::integer(8);
        assert!(refresh_training(&cfg, &b, &c).is_err());
        cfg["hpo_spec"]["schema_version"] = json::integer(1);
        assert!(refresh_training(&cfg, &b, &c).is_err());
    }
}
