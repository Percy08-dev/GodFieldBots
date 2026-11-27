//! Corrected public DTO diagnostics. These helpers confer no Source, artifact,
//! label, runtime-attestation or learning authority. No engine or policy runs.
use super::{ColumnContext, Table};
use crate::{
    budget::{Budget, Control},
    error::{Error, Result},
    json::{self, Document, Value, ValueExt},
    structured_controls::{checksum, digest},
    validation,
};
use std::{
    collections::BTreeSet,
    sync::{Arc, OnceLock},
};

const PINS: &[u8] = include_bytes!("../assets/card-instance-validation-pins-v2.json");
pub(crate) fn embedded_pins() -> &'static Value {
    static PIN: OnceLock<Value> = OnceLock::new();
    PIN.get_or_init(|| serde_json::from_slice(PINS).expect("built-in validation contract"))
}

pub(crate) fn defense_v3_release() -> &'static Value {
    static RELEASE: OnceLock<Value> = OnceLock::new();
    RELEASE.get_or_init(|| {
        serde_json::from_slice(include_bytes!("../assets/card-instance-defense-v3-release.json"))
            .expect("built-in Simulator v3 release pins")
    })
}

pub(crate) const DEFENSE_V3_PROFILE: &str = "card-instance-simulator-v3-shared-20261127-defense";

pub(crate) fn defense_v3_card_instance_contract() -> &'static Value {
    static CONTRACT: OnceLock<Value> = OnceLock::new();
    CONTRACT.get_or_init(|| {
        let mut contract = embedded_pins()["action_contract"].clone();
        for (key, value) in [
            ("profile_id", DEFENSE_V3_PROFILE),
            ("gameplay_api", "v3"),
            ("rule_set_version", "true-match-1v1-v3"),
            ("catalog_version", "public-ja-v4"),
            ("catalog_sha256", "c27c3341dcb0dfdf998322071641c32e9b5f22c8651427ab81a1a49fcb15cb27"),
            ("simulator_revision", "997a0de59a456dd84699605c178f12ec74addf84"),
        ] {
            contract[key] = serde_json::json!(value);
        }
        contract["spec_inventory_sha256"] = defense_v3_release()["spec_inventory_sha256"].clone();
        contract
    })
}

/// Raw Simulator wire identity, independent of the shared numerical layout.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum WireProfile {
    #[default]
    V2,
    DefenseV3,
}

impl WireProfile {
    pub(crate) fn expected(contract: &Value) -> Result<Self> {
        if contract == defense_v3_card_instance_contract() {
            Ok(Self::DefenseV3)
        } else if contract == &embedded_pins()["action_contract"] {
            Ok(Self::V2)
        } else {
            Err(Error::contract("unsupported expected card-instance contract"))
        }
    }

    pub(crate) fn frame_contract(self) -> &'static Value {
        match self {
            Self::V2 => &embedded_pins()["action_contract"],
            Self::DefenseV3 => defense_v3_card_instance_contract(),
        }
    }

    pub(crate) fn gameplay_api(self) -> &'static str {
        self.frame_contract()["gameplay_api"].as_str().unwrap()
    }

    pub(crate) fn rule_set(self) -> &'static str {
        self.frame_contract()["rule_set_version"].as_str().unwrap()
    }

    pub(crate) fn event_schema(self) -> &'static str {
        match self {
            Self::V2 => "2",
            Self::DefenseV3 => "3",
        }
    }

    pub(crate) fn rule_signature(self) -> &'static Value {
        static DEFENSE_V3_RULE_SIGNATURE: OnceLock<Value> = OnceLock::new();
        match self {
            Self::V2 => &embedded_pins()["rule_signature"],
            Self::DefenseV3 => DEFENSE_V3_RULE_SIGNATURE.get_or_init(|| {
                json::text("be963cce45e25982ecd0512ebd91fc7dba26c38b44bd66f726813ab88b8b28a2")
            }),
        }
    }

    pub(crate) fn catalog_metadata(self) -> Value {
        let contract = self.frame_contract();
        serde_json::json!({
            "catalog_schema_version": 2, "catalog_version": contract["catalog_version"],
            "content_sha256": contract["catalog_sha256"], "definition_count": 296,
            "locale": "ja", "rule_set_version": self.rule_set(),
        })
    }
}

pub(crate) const KINDS: &[&str] = &[
    "pray",
    "play_sequence",
    "respond",
    "discard",
    "sacrifice",
    "exchange",
    "sell",
    "buy",
    "buy_decision",
    "invoke_phenomenon",
];
const CARD_LISTS: &[&str] = &[
    "usable_cards",
    "exchange_cards",
    "sell_cards",
    "buy_cards",
    "phenomenon_cards",
];
fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::contract(message))
    }
}
fn is(value: &Value, key: &str, expected: &str) -> Result<()> {
    require(
        value.require(key)?.string()? == expected,
        "corrected DTO semantic mismatch",
    )
}
fn pins(budget: &Arc<Budget>, control: &Control) -> Result<Document> {
    json::parse(PINS, budget, control)
}
pub(crate) fn refresh_label(
    actions: &Value,
    options: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<String> {
    options.closed(&["version", "attack_threshold"], &[])?;
    is(options, "version", "1")?;
    let number = options
        .require("attack_threshold")?
        .as_number()
        .ok_or_else(|| Error::contract("refresh positive integer threshold required"))?;
    let raw = number.as_str();
    require(
        json::integer_token(number)
            && !raw.starts_with('-')
            && raw.bytes().any(|byte| byte != b'0'),
        "refresh positive integer threshold required",
    )?;
    crate::structured_controls::actions3(actions, budget, control)?;
    let profile = WireProfile::expected(actions.require("action_contract")?)?;
    let identity = json::object([
        ("options", json::copy(options, control)?),
        (
            "action_contract",
            json::copy(actions.require("action_contract")?, control)?,
        ),
        (
            "rule_signature",
            json::copy(profile.rule_signature(), control)?,
        ),
        (
            "action_config_hash",
            json::text(&digest(actions, budget, control)?),
        ),
    ]);
    Ok(format!(
        "heuristic_refresh-card-instance-v2-{}",
        digest(&identity, budget, control)?
    ))
}

pub(crate) fn card_ref(value: &Value) -> Result<&str> {
    let text = value.string()?;
    require(
        text.len() == 35
            && text.starts_with("c1_")
            && text.as_bytes()[3..]
                .iter()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c)),
        "invalid opaque card reference",
    )?;
    Ok(text)
}
fn contract(value: &Value, expected: &Value) -> Result<()> {
    // Pin exact keys AND scalar types; Value's integer/float equality is not
    // used as a substitute for this closed public contract.
    value.closed(
        &[
            "profile_id",
            "simulator_revision",
            "rule_set_version",
            "gameplay_api",
            "card_ref_scheme",
            "catalog_schema_version",
            "catalog_version",
            "catalog_sha256",
            "spec_inventory_sha256",
            "policy_input_version",
            "input_recipe",
        ],
        &[],
    )?;
    for (key, pin) in expected.object()? {
        if key == "catalog_schema_version" {
            value.require(key)?.integer(2, 2)?;
        } else {
            is(value, key, pin.string()?)?;
        }
    }
    Ok(())
}
fn typed_metadata(choice: Option<&str>) -> Value {
    let reference = json::object([
        ("type", json::text("string")),
        ("pattern", json::text("^c1_[0-9a-f]{32}$")),
        ("sources", json::array([json::text("own.cards[].card_ref")])),
    ]);
    let refs = json::object([
        ("type", json::text("array")),
        ("items", reference.clone()),
        ("ordered", Value::Bool(true)),
        ("unique", Value::Bool(true)),
    ]);
    let target = json::object([
        ("type", json::text("integer")),
        ("enum", json::array([json::integer(1), json::integer(2)])),
        (
            "sources",
            json::array([
                json::text("own.player_id"),
                json::text("opponent.player_id"),
            ]),
        ),
    ]);
    let numeric = |low, high| {
        json::object([
            ("type", json::text("integer")),
            ("minimum", json::integer(low)),
            ("maximum", json::integer(high)),
        ])
    };
    let descriptor = |kind: &str, fields: Value| {
        json::object([
            ("type", json::text(kind)),
            ("required_fields", fields),
            ("optional_fields", json::object([])),
            ("additional_fields", Value::Bool(false)),
        ])
    };
    json::array(match choice {
        Some("turn_action") => vec![
            descriptor("pray", json::object([])),
            descriptor(
                "play_sequence",
                json::object([
                    ("ordered_card_refs", refs.clone()),
                    ("target", target.clone()),
                ]),
            ),
            descriptor("discard", json::object([("card_refs", refs.clone())])),
            descriptor("sacrifice", json::object([("card_refs", refs)])),
            descriptor(
                "exchange",
                json::object([
                    ("card_ref", reference.clone()),
                    ("hp", numeric(i32::MIN as i128, i32::MAX as i128)),
                    ("mp", numeric(0, u32::MAX as i128)),
                    ("money", numeric(0, u32::MAX as i128)),
                ]),
            ),
            descriptor(
                "sell",
                json::object([
                    ("card_ref", reference.clone()),
                    ("sold_card_ref", reference.clone()),
                    ("target", target.clone()),
                ]),
            ),
            descriptor(
                "buy",
                json::object([("card_ref", reference.clone()), ("target", target)]),
            ),
            descriptor("invoke_phenomenon", json::object([("card_ref", reference)])),
        ],
        Some("defense") => vec![descriptor(
            "respond",
            json::object([("ordered_card_refs", refs)]),
        )],
        Some("buy_accept_reject") => vec![descriptor(
            "buy_decision",
            json::object([("accept", json::object([("type", json::text("boolean"))]))]),
        )],
        _ => vec![],
    })
}
fn status(value: &Value) -> Result<()> {
    value.closed(&["disease", "fog", "flash", "dream", "dark_cloud"], &[])?;
    let disease = value.require("disease")?;
    require(
        disease.is_null() || ["cold", "fever", "hell", "heaven"].contains(&disease.string()?),
        "unknown disease",
    )?;
    for field in ["fog", "flash", "dream", "dark_cloud"] {
        value.require(field)?.boolean()?;
    }
    Ok(())
}

/// Original schema2 HTTP observation. Dataset/Replay representations are not
/// accepted under this entry point, and no legality is inferred from syntax.
pub(crate) fn observation_for_profile(
    value: &Value,
    viewer: Option<i128>,
    budget: &Arc<Budget>,
    control: &Control,
    disabled: bool,
    profile: WireProfile,
) -> Result<()> {
    control.check()?;
    let _containers = budget.reserve(65536)?;
    value.closed(
        &[
            "observation_schema_version",
            "state_revision",
            "viewer_player_id",
            "turn",
            "first_player",
            "active_player",
            "current_player",
            "phase",
            "required_choice",
            "end_times",
            "own",
            "opponent",
            "result",
            "rule_set_version",
            "card_ref_scheme",
            "action_schema_version",
            "actionable",
            "typed_action_schema",
            "turns_until_end_of_world",
        ],
        &[],
    )?;
    is(value, "observation_schema_version", "2")?;
    is(value, "rule_set_version", profile.rule_set())?;
    is(value, "card_ref_scheme", "aes256-card-ref-v1")?;
    value.require("action_schema_version")?.integer(2, 2)?;
    let actor = validation::player(value.require("viewer_player_id")?)?;
    require(
        viewer.is_none_or(|v| v == actor),
        "observation viewer mismatch",
    )?;
    for key in ["first_player", "active_player"] {
        validation::player(value.require(key)?)?;
    }
    let current = value.require("current_player")?;
    if !current.is_null() {
        validation::player(current)?;
    }
    value
        .require("state_revision")?
        .integer(0, u64::MAX as i128)?;
    for key in ["turn", "turns_until_end_of_world"] {
        value.require(key)?.integer(0, u32::MAX as i128)?;
    }
    value.require("end_times")?.boolean()?;
    let phase = value.require("phase")?.string()?;
    require(
        [
            "awaiting_action",
            "awaiting_response",
            "resolving",
            "terminal",
        ]
        .contains(&phase),
        "unknown phase",
    )?;
    let choice = value.require("required_choice")?;
    let choice = if choice.is_null() {
        None
    } else {
        Some(choice.string()?)
    };
    require(
        choice.is_none_or(|v| {
            [
                "turn_action",
                "defense",
                "buy_accept_reject",
                "additional_choice",
            ]
            .contains(&v)
        }),
        "unknown choice",
    )?;
    let own = value.require("own")?;
    let opponent = value.require("opponent")?;
    own.closed(
        &[
            "player_id",
            "hp",
            "mp",
            "money",
            "cards",
            "status",
            "guardian",
            "ascended",
        ],
        &[],
    )?;
    opponent.closed(
        &["player_id", "learned_miracles", "ascended"],
        &["hp", "mp", "money", "status", "guardian"],
    )?;
    require(
        validation::player(own.require("player_id")?)? == actor
            && validation::player(opponent.require("player_id")?)? == 3 - actor,
        "observation participant mismatch",
    )?;
    for participant in [own, opponent] {
        participant.require("ascended")?.boolean()?;
        for field in ["hp", "mp", "money"] {
            if let Some(v) = participant.get(field) {
                v.integer(
                    if field == "hp" { i32::MIN as i128 } else { 0 },
                    if field == "hp" {
                        i32::MAX as i128
                    } else {
                        u32::MAX as i128
                    },
                )?;
            }
        }
        if let Some(v) = participant.get("status") {
            status(v)?;
        }
        if let Some(v) = participant.get("guardian").filter(|v| !v.is_null()) {
            validation::text_value(v)?;
        }
    }
    let cards = own.require("cards")?.array()?;
    require(cards.len() <= 18, "own card capacity exceeded")?;
    let mut seen = BTreeSet::new();
    for card in cards {
        control.check()?;
        card.closed(&["kind", "card_ref", "definition_id"], &["state"])?;
        require(
            seen.insert(card_ref(card.require("card_ref")?)?),
            "duplicate owned reference",
        )?;
        validation::text_value(card.require("definition_id")?)?;
        match card.require("kind")?.string()? {
            "artifact" => require(
                card.get("state").is_none(),
                "artifact cannot have miracle state",
            )?,
            "miracle" => require(
                ["unused", "learned"].contains(&card.require("state")?.string()?),
                "unknown miracle state",
            )?,
            _ => return Err(Error::contract("unknown instance kind")),
        }
    }
    let learned = opponent.require("learned_miracles")?.array()?;
    require(learned.len() <= 18, "opponent learned capacity exceeded")?;
    for name in learned {
        validation::text_value(name)?;
    }
    if own.require("status")?.require("fog")?.boolean()? {
        require(
            opponent
                .object()?
                .keys()
                .all(|k| ["player_id", "learned_miracles", "ascended"].contains(&k.as_str())),
            "Fog opponent visibility mismatch",
        )?;
    }
    validation::result(value.require("result")?)?;
    let can_act = current.as_i64() == Some(actor as i64)
        && value.require("result")?.is_null()
        && (["awaiting_action", "awaiting_response"].contains(&phase)
            || phase == "resolving" && choice == Some("buy_accept_reject"));
    require(
        !can_act || choice != Some("additional_choice"),
        "additional choice unsupported",
    )?;
    let expected = typed_metadata(if can_act && !disabled { choice } else { None });
    require(
        value.require("actionable")?.boolean()? == !expected.array()?.is_empty()
            && value.require("typed_action_schema")? == &expected,
        "typed v2 metadata/actionability mismatch",
    )
}
fn hint_entries<'a>(
    rows: &'a Value,
    own: &[Value],
    control: &Control,
) -> Result<BTreeSet<&'a str>> {
    let mut seen = BTreeSet::new();
    for row in rows.array()? {
        control.check()?;
        row.closed(&["card_ref", "definition_id"], &[])?;
        let reference = card_ref(row.require("card_ref")?)?;
        validation::text_value(row.require("definition_id")?)?;
        require(
            seen.insert(reference)
                && own.iter().any(|card| {
                    card.get("card_ref") == row.get("card_ref")
                        && card.get("definition_id") == row.get("definition_id")
                }),
            "duplicate/foreign/perception-mismatched hint",
        )?;
    }
    Ok(seen)
}
#[cfg(test)]
pub(crate) fn hints(value: &Value, obs: &Value, control: &Control) -> Result<()> {
    hints_for_profile(value, obs, control, WireProfile::V2)
}

pub(crate) fn hints_for_profile(
    value: &Value,
    obs: &Value,
    control: &Control,
    profile: WireProfile,
) -> Result<()> {
    value.closed(
        &["state_revision", "viewer_player_id", "action_options"],
        &[],
    )?;
    require(
        value
            .require("state_revision")?
            .integer(0, u64::MAX as i128)?
            == obs
                .require("state_revision")?
                .integer(0, u64::MAX as i128)?
            && validation::player(value.require("viewer_player_id")?)?
                == obs.require("viewer_player_id")?.integer(1, 2)?,
        "hint wrapper revision/actor mismatch",
    )?;
    let v = value.require("action_options")?;
    v.closed(
        &[
            "action_options_schema_version",
            "rule_set_version",
            "card_ref_scheme",
            "actor",
            "state_revision",
            "required_choice",
            "candidate_groups",
            "valid_targets",
            "can_pray",
            "can_discard",
            "can_sacrifice",
            "can_allow",
            "usable_cards",
            "exchange_cards",
            "sell_cards",
            "buy_cards",
            "phenomenon_cards",
        ],
        &[],
    )?;
    is(v, "action_options_schema_version", "2")?;
    is(v, "rule_set_version", profile.rule_set())?;
    is(v, "card_ref_scheme", "aes256-card-ref-v1")?;
    require(
        validation::player(v.require("actor")?)?
            == obs.require("viewer_player_id")?.integer(1, 2)?
            && v.require("state_revision")?.integer(0, u64::MAX as i128)?
                == obs
                    .require("state_revision")?
                    .integer(0, u64::MAX as i128)?
            && v.require("required_choice")? == obs.require("required_choice")?,
        "atomic hint actor/revision/choice mismatch",
    )?;
    let own = obs.require("own")?.require("cards")?.array()?;
    let usable = hint_entries(v.require("usable_cards")?, own, control)?;
    for name in &CARD_LISTS[1..] {
        hint_entries(v.require(name)?, own, control)?;
    }
    let groups = v.require("candidate_groups")?.object()?;
    if profile == WireProfile::DefenseV3 {
        require(
            groups.contains_key("defense:direct") == (v["required_choice"] == "defense"),
            "direct defense group differs from required choice",
        )?;
    }
    for (name, rows) in groups {
        require(!name.is_empty(), "empty candidate group")?;
        let group_cards = hint_entries(rows, own, control)?;
        require(
            group_cards.is_subset(&usable),
            "group is not usable-card subset",
        )?;
        if profile == WireProfile::DefenseV3 && name == "defense:direct" {
            require(
                v["usable_cards"]
                    .array()?
                    .iter()
                    .filter(|row| group_cards.contains(row["card_ref"].as_str().unwrap()))
                    .eq(rows.array()?),
                "direct defense order differs from usable cards",
            )?;
        }
    }
    let mut targets = BTreeSet::new();
    for target in v.require("valid_targets")?.array()? {
        require(
            targets.insert(validation::player(target)?),
            "duplicate target hint",
        )?;
    }
    for name in ["can_pray", "can_discard", "can_sacrifice", "can_allow"] {
        v.require(name)?.boolean()?;
    }
    Ok(())
}
/// Validates one original actionable own frame. Perceived definition hashes
/// authenticate supplied definitions only; no missing values are looked up.
pub(crate) fn frame(value: &Value, budget: &Arc<Budget>, control: &Control) -> Result<()> {
    frame_for_profile(value, budget, control, WireProfile::V2, false)
}

pub(crate) fn frame_for_rollout(value: &Value, budget: &Arc<Budget>, control: &Control, profile: WireProfile) -> Result<()> {
    frame_for_profile(value, budget, control, profile, true)
}

pub(crate) fn frame_for_profile(
    value: &Value,
    budget: &Arc<Budget>,
    control: &Control,
    profile: WireProfile,
    reuse: bool,
) -> Result<()> {
    control.check()?;
    value.closed(
        &[
            "actor",
            "episode",
            "revision",
            "interval_start",
            "boundary",
            "interval_start_hash",
            "payload",
            "frame_hash",
        ],
        &[],
    )?;
    let actor = value.require("actor")?.integer(1, 2)?;
    validation::text_value(value.require("episode")?)?;
    let revision = value.require("revision")?.integer(0, u64::MAX as i128)?;
    let boundary = value.require("boundary")?;
    boundary.closed(&["end", "prefix_hash"], &[])?;
    let end = boundary.require("end")?.integer(0, i32::MAX as i128)?;
    let start = value.require("interval_start")?.integer(0, end)?;
    require(end - start <= 65536, "frame interval capacity exceeded")?;
    for hash in [
        boundary.require("prefix_hash")?,
        value.require("interval_start_hash")?,
        value.require("frame_hash")?,
    ] {
        checksum(hash)?;
    }
    let payload = value.require("payload")?;
    payload.closed(
        &[
            "observation",
            "typed_metadata",
            "catalog",
            "action_options",
            "history_end",
        ],
        &[],
    )?;
    // Charge bounded validation containers before constructing typed metadata,
    // sets and the call-local frame-hash binding.
    let raw = json::canonical(value, budget, control)?;
    require(raw.len() <= 64 * 1024 * 1024, "frame exceeds64MiB")?;
    let _copy = json::admit(raw.len() + 65536, budget)?;
    let obs = payload.require("observation")?;
    observation_for_profile(obs, Some(actor), budget, control, false, profile)?;
    require(
        obs.require("actionable")?.boolean()?
            && obs.require("current_player")?.integer(1, 2)? == actor
            && obs
                .require("state_revision")?
                .integer(0, u64::MAX as i128)?
                == revision
            && payload.require("typed_metadata")? == obs.require("typed_action_schema")?,
        "original actor/frame/metadata mismatch",
    )?;
    hints_for_profile(payload.require("action_options")?, obs, control, profile)?;
    if !payload.require("history_end")?.is_null() {
        validation::text_value(payload.require("history_end")?)?;
    }
    // Cache only this exact embedded document, never the supplied frame or
    // the observed definition digests it still has to prove.
    let parsed;
    let pins = if reuse {
        control.check()?;
        embedded_pins()
    } else {
        parsed = pins(budget, control)?;
        &parsed.root
    };
    let catalog = payload.require("catalog")?.object()?;
    let cards = obs.require("own")?.require("cards")?.array()?;
    let names = cards
        .iter()
        .map(|v| v.require("definition_id")?.string())
        .collect::<Result<BTreeSet<_>>>()?;
    require(
        catalog.len() == names.len() && catalog.keys().all(|k| names.contains(k.as_str())),
        "catalog must equal own perceived subset",
    )?;
    for (name, definition) in catalog {
        control.check()?;
        let expected = pins
            .require("definition_sha256")?
            .get(name)
            .ok_or_else(|| Error::contract("unknown perceived definition"))?
            .string()?;
        require(
            digest(definition, budget, control)? == expected,
            "perceived definition differs from pinned catalog",
        )?;
        require(
            definition.require("definition_id")?.string()? == name,
            "catalog key/definition mismatch",
        )?;
    }
    for card in cards {
        let definition = payload
            .require("catalog")?
            .require(card.require("definition_id")?.string()?)?;
        require(
            definition
                .require("classification")?
                .require("definition_class")?
                .string()?
                == "owned_card"
                && (card.require("kind")?.string()? == "miracle")
                    == (definition.require("kind")?.string()? == "miracles"),
            "perceived kind/class mismatch",
        )?;
    }
    let mut binding = json::copy(value, control)?;
    let fields = binding.as_object_mut().unwrap();
    fields.remove("frame_hash");
    // DecisionFrame._binding() adds these layout identities even though they
    // are not fields in the serialized frame DTO. Historical card frames bind
    // recipe/policy only; corrected frames additionally bind the full contract.
    let contract = profile.frame_contract();
    for key in ["input_recipe", "policy_input_version"] {
        fields.insert(key.into(), json::copy(contract.require(key)?, control)?);
    }
    fields.insert("action_contract".into(), json::copy(contract, control)?);
    require(
        digest(&binding, budget, control)? == value.require("frame_hash")?.string()?,
        "frame hash mismatch",
    )
}

pub(super) fn references(frame: &Value, control: &Control) -> Result<Vec<Value>> {
    frame
        .require("payload")?
        .require("observation")?
        .require("own")?
        .require("cards")?
        .array()?
        .iter()
        .map(|card| {
            control.check()?;
            json::copy(card.require("card_ref")?, control)
        })
        .collect()
}
pub(super) fn action(
    kind: &str,
    target: &str,
    numerics: &[i64; 4],
    present: &[bool; 4],
    slots: &[u8],
    frame: &Value,
    refs: &[Value],
    control: &Control,
) -> Result<Value> {
    let mut result = json::object([("type", json::text(kind))]);
    let elements = slots
        .iter()
        .map(|&slot| {
            refs.get(slot as usize)
                .ok_or_else(|| Error::contract("slot outside original cards"))
                .and_then(|v| json::copy(v, control))
        })
        .collect::<Result<Vec<_>>>()?;
    let object = result.as_object_mut().unwrap();
    match kind {
        "play_sequence" | "respond" => {
            object.insert("ordered_card_refs".into(), json::array(elements));
        }
        "discard" | "sacrifice" => {
            object.insert("card_refs".into(), json::array(elements));
        }
        "exchange" | "buy" | "invoke_phenomenon" | "sell" => {
            require(
                elements.len() == if kind == "sell" { 2 } else { 1 },
                "carrier count mismatch",
            )?;
            object.insert("card_ref".into(), elements[0].clone());
            if kind == "sell" {
                object.insert("sold_card_ref".into(), elements[1].clone());
            }
        }
        "pray" | "buy_decision" => require(elements.is_empty(), "unexpected candidate refs")?,
        _ => return Err(Error::contract("unknown corrected candidate kind")),
    }
    if target != "omitted" {
        require(
            ["self", "opponent"].contains(&target),
            "v2 target cannot be null",
        )?;
        let actor = frame.require("actor")?.integer(1, 2)?;
        object.insert(
            "target".into(),
            json::integer(if target == "self" { actor } else { 3 - actor }),
        );
    }
    for (index, name) in ["hp", "mp", "money", "accept"].iter().enumerate() {
        if present[index] {
            let value = if *name == "accept" {
                require(
                    (0..=1).contains(&numerics[index]),
                    "noncanonical acceptance",
                )?;
                Value::Bool(numerics[index] == 1)
            } else {
                json::integer(numerics[index] as i128)
            };
            object.insert((*name).into(), value);
        }
    }
    let descriptors = frame
        .require("payload")?
        .require("typed_metadata")?
        .array()?;
    let descriptor = descriptors
        .iter()
        .find(|v| v.get("type").and_then(Value::as_str) == Some(kind))
        .ok_or_else(|| Error::contract("candidate outside typed metadata"))?;
    let required = descriptor.require("required_fields")?.object()?;
    require(
        required.keys().all(|k| result.get(k).is_some())
            && result
                .object()?
                .keys()
                .all(|k| k == "type" || required.contains_key(k)),
        "candidate fields differ from v2 syntax",
    )?;
    for (key, low, high) in [
        ("hp", i32::MIN as i128, i32::MAX as i128),
        ("mp", 0, u32::MAX as i128),
        ("money", 0, u32::MAX as i128),
    ] {
        if let Some(v) = result.get(key) {
            v.integer(low, high)?;
        }
    }
    Ok(result)
}
pub(super) fn column_binding(
    descriptor: &Value,
    budget: &Arc<Budget>,
    control: &Control,
    profile: WireProfile,
) -> Result<()> {
    let pins = pins(budget, control)?;
    contract(
        descriptor.require("action_contract")?,
        profile.frame_contract(),
    )?;
    let codebook = json::object([
        ("codec", json::text("candidate-columns-v2")),
        (
            "contract",
            json::copy(pins.root.require("action_contract")?, control)?,
        ),
        (
            "action_kinds",
            json::array(KINDS.iter().map(|v| json::text(v))),
        ),
        (
            "targets",
            json::array(super::TARGETS.iter().map(|v| json::text(v))),
        ),
    ]);
    require(
        descriptor.require("codebook_sha256")?.string()? == digest(&codebook, budget, control)?,
        "codec2 codebook mismatch",
    )
}
pub(crate) fn columns(
    payload: &[u8],
    descriptor: &Value,
    original: &Value,
    max_positions: usize,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<Table> {
    frame(original, budget, control)?;
    require(
        (1..=18).contains(&max_positions),
        "diagnostic codec2 requires real18 capacity",
    )?;
    super::columns_context(
        payload,
        descriptor,
        original,
        max_positions,
        false,
        budget,
        control,
        ColumnContext::Corrected,
    )
}

pub(crate) fn component_columns(
    payload: &[u8],
    descriptor: &Value,
    original: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<Table> {
    frame(original, budget, control)?;
    super::columns_context(
        payload,
        descriptor,
        original,
        18,
        false,
        budget,
        control,
        ColumnContext::CorrectedComponent,
    )
}

pub(crate) fn component_binding(budget: &Arc<Budget>, control: &Control) -> Result<Value> {
    let pins = pins(budget, control)?;
    Ok(json::object([
        (
            "action_contract",
            json::copy(pins.root.require("action_contract")?, control)?,
        ),
        ("purpose", json::text("component")),
        ("synthetic_capacity", Value::Bool(false)),
        (
            "rule_signature",
            json::copy(pins.root.require("rule_signature")?, control)?,
        ),
    ]))
}

pub(crate) fn is_concealed_draw(value: &Value) -> bool {
    value["type"] == "artifact_drawn" && value["definition_id"] == "concealed"
}

/// Public schema2 event diagnostic with exact Guardian/definition namespaces.
pub(crate) fn event(
    value: &Value,
    recipe: &crate::structured_tokens::Recipe,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<()> {
    event_for_profile(value, recipe, budget, control, WireProfile::V2, None)
}

pub(crate) fn event_for_profile(
    value: &Value,
    recipe: &crate::structured_tokens::Recipe,
    budget: &Arc<Budget>,
    control: &Control,
    profile: WireProfile,
    viewer: Option<i128>,
) -> Result<()> {
    control.check()?;
    require(
        recipe.is_corrected(),
        "corrected event requires explicit recipe",
    )?;
    let tag = value.require("type")?.string()?;
    if profile == WireProfile::DefenseV3
        && ["attack_declared", "attack_missed"].contains(&tag)
        && value.get("attacker").is_none()
    {
        let keys: &[&str] = if tag == "attack_declared" {
            &["type", "target", "attack", "attribute", "definition_ids"]
        } else {
            &["type", "target"]
        };
        value.closed(keys, &[])?;
        let target = validation::player(value.require("target")?)?;
        require(viewer.is_none_or(|v| v == target), "anonymous attack requires target viewer")?;
        if tag == "attack_declared" {
            value.require("attack")?.integer(i32::MIN as i128, i32::MAX as i128)?;
            require(
                ["none", "fire", "water", "wood", "earth", "light", "dark"]
                    .contains(&value.require("attribute")?.string()?),
                "unknown attack attribute",
            )?;
            let definitions = recipe.definition_ids()?;
            let guardians = recipe.guardian_ids()?;
            for id in value.require("definition_ids")?.array()? {
                control.check()?;
                require(definitions.contains(id) || guardians.contains(id), "attack definition namespace mismatch")?;
            }
        }
        return Ok(());
    }
    if is_concealed_draw(value) {
        value.closed(&["type", "player", "definition_id"], &[])?;
        return validation::player(value.require("player")?).map(|_| ());
    }
    if tag == "safe" {
        value.closed(&["type", "player"], &[])?;
        validation::player(value.require("player")?)?;
    } else {
        let bytes = json::canonical(value, budget, control)?;
        let _copy = json::admit(bytes.len() + 65536, budget)?;
        validation::validate_events_controlled(
            &json::array([json::copy(value, control)?]),
            control,
        )?;
    }
    if tag == "status_changed" {
        require(
            [
                "cold",
                "fever",
                "hell",
                "heaven",
                "fog",
                "flash",
                "dream",
                "dark_cloud",
            ]
            .contains(&value.require("status")?.string()?),
            "unknown public status",
        )?;
    }
    if let Some(name) = value.get("definition_id") {
        let namespace = if ["guardian_set", "guardian_left"].contains(&tag) {
            recipe.guardian_ids()?
        } else {
            recipe.definition_ids()?
        };
        require(
            namespace.contains(name)
                || (tag == "guardian_acted" && recipe.guardian_ids()?.contains(name)),
            "event identifier namespace mismatch",
        )?;
    }
    Ok(())
}

/// Deliberately partial: expected own-policy/runtime declarations are supplied
/// by the caller. Success never validates their artifacts, files, labels,
/// runtime attestation, Source eligibility or full execution/capacity semantics.
pub(crate) struct DiagnosticHeaderExpected<'a> {
    pub actions: &'a Value,
    pub neural: Option<&'a Value>,
    pub tree: Option<&'a Value>,
    pub runtime_evidence: &'a Value,
}
pub(crate) fn diagnostic_header_envelope(
    value: &Value,
    expected: DiagnosticHeaderExpected<'_>,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<()> {
    control.check()?;
    let refresh = value.require("model_id")?.string()? == "heuristic_refresh";
    let tree = value.require("model_version")?.string()? == "3"
        && value.require("model_id")?.string()? == "lightgbm";
    let neural = ["3", "4"].contains(&value.require("model_version")?.string()?) && !tree;
    value.closed(
        &[
            "kind",
            "schema_version",
            "feature_version",
            "input_recipe",
            "reward_version",
            "policy_input_version",
            "end_projection_version",
            "catalog",
            "episode_id",
            "viewer",
            "policy",
            "run_id",
            "game_id",
            "action_config_hash",
            "generator_version",
            "bank_episode",
            "caps",
            "numeric_costs",
            "model_id",
            "model_version",
            "generation_schema",
            "aggregate_window",
            "token_signature",
            "sampling_seed",
            "use_action_options",
            "action_contract",
            "rule_signature",
            "layout_signature",
            "purpose",
            "synthetic_capacity",
            "protocol_version",
            "actions",
            "runtime_evidence",
        ],
        if refresh {
            &["baseline_options"]
        } else if neural {
            &["neural"]
        } else if tree {
            &["tree"]
        } else {
            &[]
        },
    )?;
    value.require("schema_version")?.integer(6, 6)?;
    value.require("protocol_version")?.integer(3, 3)?;
    for (key, text) in [
        ("kind", "header"),
        ("feature_version", "participant-structured-v2"),
        ("input_recipe", "participant-structured-v2"),
        ("reward_version", "natural-terminal-v1"),
        ("policy_input_version", "player-card-refs-v2"),
        ("end_projection_version", "simulator-closure-v1"),
        ("generator_version", "public-card-instance-diverse-v5"),
        ("purpose", "qualified_game"),
    ] {
        is(value, key, text)?;
    }
    let action_contract = expected.actions.require("action_contract")?;
    let profile = WireProfile::expected(action_contract)?;
    contract(value.require("action_contract")?, profile.frame_contract())?;
    require(
        value.require("rule_signature")? == profile.rule_signature()
            && value.require("layout_signature")?.string()?
                == crate::structured_controls::CARD_INSTANCE_SIGNATURE
            && !value.require("synthetic_capacity")?.boolean()?
            && value.require("use_action_options")?.boolean()?
            && value.require("generation_schema")?.is_null(),
        "J6 corrected binding mismatch",
    )?;
    require(
        value.require("actions")? == expected.actions
            && value.require("action_config_hash")?.string()?
                == digest(expected.actions, budget, control)?
            && value.require("runtime_evidence")? == expected.runtime_evidence,
        "J6 expected declaration mismatch",
    )?;
    require(value.require("catalog")? == &profile.catalog_metadata(), "J6 catalog mismatch")?;
    for key in ["episode_id", "policy", "run_id", "game_id", "bank_episode"] {
        validation::text_value(value.require(key)?)?;
    }
    value.require("viewer")?.integer(1, 2)?;
    value
        .require("sampling_seed")?
        .integer(0, u64::MAX as i128)?;
    let caps = value.require("caps")?;
    caps.closed(
        &[
            "max_interval_events",
            "max_episode_events",
            "max_decisions",
            "max_positions",
            "max_candidates",
        ],
        &[],
    )?;
    let interval = caps.require("max_interval_events")?.integer(1, 65536)?;
    let episode = caps
        .require("max_episode_events")?
        .integer(interval, i32::MAX as i128)?;
    if !caps.require("max_decisions")?.is_null() {
        caps.require("max_decisions")?.integer(1, 1024)?;
    }
    caps.require("max_positions")?.integer(1, 18)?;
    caps.require("max_candidates")?.integer(1, 65536)?;
    require(
        caps.require("max_positions")? == expected.actions.require("max_sequence_length")?
            && caps.require("max_candidates")? == expected.actions.require("max_candidates")?,
        "J6 action caps mismatch",
    )?;
    super::numeric_costs(value.require("numeric_costs")?)?;
    let window = value
        .require("aggregate_window")?
        .integer(1, episode.min(65536))?;
    require(
        value.require("token_signature")?.string()?
            == crate::structured_controls::card_instance_token_signature(
                window as usize,
                budget,
                control,
            )?,
        "J6 actual-window signature mismatch",
    )?;
    let model = value.require("model_id")?.string()?;
    if neural {
        require(
            ["mlp", "gru", "transformer"].contains(&model),
            "unknown model3 family",
        )?;
        let recorded = value.require("neural")?;
        recorded.closed(
            &[
                "input_contract",
                "policy_options",
                "inference_manifest",
                "files",
            ],
            &[],
        )?;
        is(recorded, "input_contract", "retained-external-inference-v1")?;
        require(
            expected.neural == Some(recorded) && expected.tree.is_none(),
            "J6 expected neural declaration mismatch",
        )?;
    } else if tree {
        is(value, "model_version", "3")?;
        let recorded = value.require("tree")?;
        recorded.closed(
            &[
                "input_contract",
                "policy_options",
                "inference_manifest",
                "files",
            ],
            &[],
        )?;
        is(recorded, "input_contract", "player-card-refs-v2")?;
        require(
            expected.tree == Some(recorded) && expected.neural.is_none(),
            "J6 expected own Tree declaration mismatch",
        )?;
    } else if refresh {
        is(value, "model_version", "1")?;
        require(
            expected.neural.is_none()
                && expected.tree.is_none()
                && value.require("aggregate_window")?.integer(128, 128)? == 128
                && value.require("policy")?.string()?
                    == refresh_label(
                        expected.actions,
                        value.require("baseline_options")?,
                        budget,
                        control,
                    )?,
            "J6 configured baseline identity mismatch",
        )?;
    } else {
        is(value, "model_version", "1")?;
        require(
            ["random", "heuristic"].contains(&model)
                && expected.neural.is_none()
                && expected.tree.is_none()
                && value.require("policy")?.string()? == format!("{model}-card-instance-v2"),
            "J6 baseline identity mismatch",
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structured_tokens::{Recipe, candidate_tokens, frame_tokens};
    use sha2::{Digest, Sha256};

    fn owners() -> (Arc<Budget>, Control) {
        (
            Budget::new(512 * 1024 * 1024),
            Control::new(crate::budget::monotonic() + 60.0).unwrap(),
        )
    }
    fn parse(raw: &[u8], b: &Arc<Budget>, c: &Control) -> Value {
        json::parse(raw, b, c).unwrap().root.clone()
    }
    fn seal(frame: &mut Value, b: &Arc<Budget>, c: &Control) {
        let mut binding = frame.clone();
        binding.as_object_mut().unwrap().remove("frame_hash");
        let contract = pins(b, c).unwrap().root["action_contract"].clone();
        binding["input_recipe"] = contract["input_recipe"].clone();
        binding["policy_input_version"] = contract["policy_input_version"].clone();
        binding["action_contract"] = contract;
        let hash = digest(&binding, b, c).unwrap();
        frame
            .as_object_mut()
            .unwrap()
            .insert("frame_hash".into(), json::text(&hash));
    }
    fn fixture(names: &[(&str, Option<&str>)], end: usize, b: &Arc<Budget>, c: &Control) -> Value {
        let mut obs = parse(
            include_bytes!("../../tests/fixtures/card_instance_v2_runtime_observation.json"),
            b,
            c,
        );
        let all = parse(
            include_bytes!("../../tests/fixtures/card_instance_v2_catalog.json"),
            b,
            c,
        );
        let mut catalog = json::object([]);
        let mut cards = Vec::new();
        let mut rows = Vec::new();
        for (index, (name, state)) in names.iter().enumerate() {
            let reference = json::text(&format!("c1_{index:032x}"));
            let mut card = json::object([
                ("definition_id", json::text(name)),
                ("card_ref", reference.clone()),
                (
                    "kind",
                    json::text(if state.is_some() {
                        "miracle"
                    } else {
                        "artifact"
                    }),
                ),
            ]);
            if let Some(state) = state {
                card.as_object_mut()
                    .unwrap()
                    .insert("state".into(), json::text(state));
            }
            cards.push(card);
            rows.push(json::object([
                ("card_ref", reference),
                ("definition_id", json::text(name)),
            ]));
            let definition = all
                .require("definitions")
                .unwrap()
                .array()
                .unwrap()
                .iter()
                .find(|d| d.get("definition_id").and_then(Value::as_str) == Some(*name))
                .unwrap();
            catalog
                .as_object_mut()
                .unwrap()
                .insert((*name).into(), definition.clone());
        }
        obs["own"]["cards"] = json::array(cards);
        obs["typed_action_schema"] = typed_metadata(Some("turn_action"));
        let hints = json::object([
            ("action_options_schema_version", json::text("2")),
            ("actor", json::integer(1)),
            ("state_revision", json::integer(0)),
            ("required_choice", json::text("turn_action")),
            ("rule_set_version", json::text("true-match-1v1-v2")),
            ("card_ref_scheme", json::text("aes256-card-ref-v1")),
            ("candidate_groups", json::object([])),
            (
                "valid_targets",
                json::array([json::integer(1), json::integer(2)]),
            ),
            ("can_pray", Value::Bool(true)),
            ("can_discard", Value::Bool(true)),
            ("can_sacrifice", Value::Bool(false)),
            ("can_allow", Value::Bool(false)),
            ("usable_cards", json::array(rows)),
            ("exchange_cards", json::array([])),
            ("sell_cards", json::array([])),
            ("buy_cards", json::array([])),
            ("phenomenon_cards", json::array([])),
        ]);
        let mut frame = json::object([
            ("actor", json::integer(1)),
            ("episode", json::text("pure-corrected-diagnostic")),
            ("revision", json::integer(0)),
            ("interval_start", json::integer(0)),
            ("interval_start_hash", json::text(&"0".repeat(64))),
            (
                "boundary",
                json::object([
                    ("end", json::integer(end as i128)),
                    ("prefix_hash", json::text(&"1".repeat(64))),
                ]),
            ),
            (
                "payload",
                json::object([
                    ("typed_metadata", obs["typed_action_schema"].clone()),
                    ("observation", obs),
                    ("catalog", catalog),
                    ("history_end", Value::Null),
                    (
                        "action_options",
                        json::object([
                            ("state_revision", json::integer(0)),
                            ("viewer_player_id", json::integer(1)),
                            ("action_options", hints),
                        ]),
                    ),
                ]),
            ),
        ]);
        seal(&mut frame, b, c);
        frame
    }
    fn one_column(
        frame: &Value,
        kind: usize,
        target: usize,
        slots: &[u8],
        nums: [i64; 4],
        present: [bool; 4],
        b: &Arc<Budget>,
        c: &Control,
    ) -> (Vec<u8>, Value) {
        let mut bytes = vec![0; 64 + slots.len()];
        bytes[0] = kind as u8;
        bytes[1] = target as u8;
        for i in 0..4 {
            bytes[8 + i * 8..16 + i * 8].copy_from_slice(&nums[i].to_le_bytes());
            bytes[40 + i] = u8::from(present[i]);
        }
        bytes[56..64].copy_from_slice(&(slots.len() as i64).to_le_bytes());
        bytes[64..].copy_from_slice(slots);
        let p = pins(b, c).unwrap();
        let classification = p.root.require("action_contract").unwrap().clone();
        let mut table = json::object([
            ("frame_hash", frame["frame_hash"].clone()),
            ("kinds", json::array([json::text(KINDS[kind])])),
            (
                "targets",
                json::array([json::text(super::super::TARGETS[target])]),
            ),
            (
                "numerics",
                json::array([json::array(nums.map(|n| json::integer(n as i128)))]),
            ),
            (
                "presence",
                json::array([json::array(present.map(Value::Bool))]),
            ),
            (
                "offsets",
                json::array([json::integer(0), json::integer(slots.len() as i128)]),
            ),
            (
                "slots",
                json::array(slots.iter().map(|n| json::integer(*n as i128))),
            ),
            ("action_contract", classification.clone()),
            ("purpose", json::text("qualified_game")),
            ("synthetic_capacity", Value::Bool(false)),
        ]);
        let table_hash = digest(&table, b, c).unwrap();
        table
            .as_object_mut()
            .unwrap()
            .insert("table_hash".into(), json::text(&table_hash));
        let mut descriptor = json::object([
            ("schema_version", json::integer(2)),
            ("codec", json::text("candidate-columns-v2")),
            ("file", json::text("player-1-decision-0.columns")),
            ("count", json::integer(1)),
            ("elements", json::integer(slots.len() as i128)),
            ("bytes", json::integer(bytes.len() as i128)),
            (
                "sha256",
                json::text(&format!("{:x}", Sha256::digest(&bytes))),
            ),
            ("frame_hash", frame["frame_hash"].clone()),
            ("table_hash", json::text(&table_hash)),
            ("action_contract", classification.clone()),
            ("purpose", json::text("qualified_game")),
            ("synthetic_capacity", Value::Bool(false)),
        ]);
        let codebook = json::object([
            ("codec", json::text("candidate-columns-v2")),
            ("contract", classification),
            (
                "action_kinds",
                json::array(KINDS.iter().map(|k| json::text(k))),
            ),
            (
                "targets",
                json::array(super::super::TARGETS.iter().map(|k| json::text(k))),
            ),
        ]);
        descriptor["codebook_sha256"] = json::text(&digest(&codebook, b, c).unwrap());
        descriptor["columns"] = json::array(
            [
                ("kinds", "uint8", vec![1], 0, 1),
                ("targets", "uint8", vec![1], 1, 1),
                ("numerics", "int64", vec![1, 4], 8, 32),
                ("presence", "uint8", vec![1, 4], 40, 4),
                ("offsets", "int64", vec![2], 48, 16),
                ("slots", "uint8", vec![slots.len()], 64, slots.len()),
            ]
            .into_iter()
            .map(|(name, dtype, shape, offset, bytes)| {
                json::object([
                    ("name", json::text(name)),
                    ("dtype", json::text(dtype)),
                    (
                        "shape",
                        json::array(shape.into_iter().map(|n| json::integer(n as i128))),
                    ),
                    ("offset", json::integer(offset)),
                    ("bytes", json::integer(bytes as i128)),
                ])
            }),
        );
        (bytes, descriptor)
    }
    #[test]
    fn corrected_recipe_pins_widths_and_actual_window_without_legacy_reinterpretation() {
        let (b, c) = owners();
        let recipe = Recipe::corrected(128, &b, &c).unwrap();
        assert_eq!(
            recipe.signature,
            crate::structured_controls::CARD_INSTANCE_SIGNATURE
        );
        assert_ne!(
            Recipe::corrected(2, &b, &c).unwrap().signature,
            recipe.signature
        );
        assert_ne!(
            Recipe::new(128, &b, &c).unwrap().signature,
            recipe.signature
        );
        assert_eq!(recipe.definition_ids().unwrap().len(), 296);
        assert_eq!(recipe.guardian_ids().unwrap().len(), 10);
        assert!(Recipe::corrected(0, &b, &c).is_err());
    }
    #[test]
    fn refs_are_opaque_features_bind_kind_state_original_order_and_duplicates() {
        let (b, c) = owners();
        let recipe = Recipe::corrected(128, &b, &c).unwrap();
        let mut original = fixture(
            &[
                ("bronze-club", None),
                ("fireball", Some("unused")),
                ("fireball", Some("learned")),
            ],
            0,
            &b,
            &c,
        );
        frame(&original, &b, &c).unwrap();
        let before = frame_tokens(&original, &[], &recipe, &b, &c).unwrap();
        assert_eq!(before.value["history"].array().unwrap().len(), 387);
        assert_eq!(before.value["card_numerics"][0].array().unwrap().len(), 26);
        assert_eq!(
            before.value["card_categories"][0].array().unwrap().len(),
            11
        );
        for i in 0..3 {
            assert_eq!(
                before.value["card_numerics"][i][8],
                json::integer(i as i128 + 1)
            );
            let reference = json::text(&format!("c1_{:032x}", u128::MAX - i as u128));
            original["payload"]["observation"]["own"]["cards"][i]["card_ref"] = reference.clone();
            original["payload"]["action_options"]["action_options"]["usable_cards"][i]["card_ref"] =
                reference;
        }
        assert_eq!(before.value["card_categories"][0][6], json::integer(2));
        assert_eq!(before.value["card_categories"][1][7], json::integer(2));
        assert_eq!(before.value["card_categories"][2][7], json::integer(3));
        seal(&mut original, &b, &c);
        frame(&original, &b, &c).unwrap();
        let after = frame_tokens(&original, &[], &recipe, &b, &c).unwrap();
        assert_ne!(before.value["frame_hash"], after.value["frame_hash"]);
        for name in [
            "state_numerics",
            "state_present",
            "state_categories",
            "state_category_present",
            "card_numerics",
            "card_present",
            "card_categories",
            "card_mask",
            "opponent_learned",
            "history",
        ] {
            assert_eq!(before.value[name], after.value[name]);
        }
        original["payload"]["observation"]["own"]["cards"][1]["card_ref"] =
            original["payload"]["observation"]["own"]["cards"][0]["card_ref"].clone();
        seal(&mut original, &b, &c);
        assert!(frame(&original, &b, &c).is_err());
    }
    #[test]
    fn learned_7_12_18_and_empty_have_exact_padding_and_no_invented_tokens() {
        let (b, c) = owners();
        let recipe = Recipe::corrected(128, &b, &c).unwrap();
        for count in [0, 7, 12, 18] {
            let raw = fixture(&vec![("fireball", Some("learned")); count], 0, &b, &c);
            frame(&raw, &b, &c).unwrap();
            let tokens = frame_tokens(&raw, &[], &recipe, &b, &c).unwrap();
            assert_eq!(
                tokens.value["card_mask"]
                    .array()
                    .unwrap()
                    .iter()
                    .filter(|v| **v == Value::Bool(true))
                    .count(),
                count
            );
            for row in &tokens.value["card_numerics"].array().unwrap()[count..] {
                assert!(row.array().unwrap().iter().all(|v| v == &json::integer(0)));
            }
        }
    }
    #[test]
    fn guardian_acted_closed_union_accepts_every_pinned_id_and_rejects_unknowns() {
        let (b, c) = owners();
        let recipe = Recipe::corrected(128, &b, &c).unwrap();
        for name in recipe
            .definition_ids()
            .unwrap()
            .iter()
            .chain(recipe.guardian_ids().unwrap())
        {
            let raw = json::object([
                ("type", json::text("guardian_acted")),
                ("player", json::integer(1)),
                ("definition_id", name.clone()),
            ]);
            event(&raw, &recipe, &b, &c).unwrap();
        }
        for tag in ["guardian_acted", "guardian_set", "guardian_left", "artifact_used"] {
            let raw = json::object([
                ("type", json::text(tag)),
                ("player", json::integer(1)),
                ("definition_id", json::text("unknown-public-id")),
            ]);
            assert!(event(&raw, &recipe, &b, &c).is_err());
        }
    }
    #[test]
    fn concealed_draws_are_validated_but_occupy_no_token_or_history_rows() {
        let (b, c) = owners();
        let recipe = Recipe::corrected(2, &b, &c).unwrap();
        let draw = parse(
            br#"{"type":"artifact_drawn","player":1,"definition_id":"bronze-club"}"#,
            &b,
            &c,
        );
        let marker = parse(
            br#"{"type":"artifact_drawn","player":1,"definition_id":"concealed"}"#,
            &b,
            &c,
        );
        let safe = parse(br#"{"type":"safe","player":2}"#, &b, &c);
        let mut events = vec![draw.clone(), marker.clone(), safe.clone(), marker.clone()];
        for (source, filtered, count, truncated) in [
            (vec![marker.clone(); 8], vec![], 0, 0),
            (events.clone(), vec![draw.clone(), safe.clone()], 2, 0),
            (
                {
                    events.insert(0, draw.clone());
                    events
                },
                vec![draw.clone(), draw, safe],
                2,
                1,
            ),
        ] {
            let frame = fixture(&[], source.len(), &b, &c);
            let tokens = frame_tokens(&frame, &source, &recipe, &b, &c).unwrap();
            let visible_frame = fixture(&[], filtered.len(), &b, &c);
            let visible = frame_tokens(&visible_frame, &filtered, &recipe, &b, &c).unwrap();
            for key in [
                "interval_numerics",
                "interval_present",
                "interval_categories",
            ] {
                assert_eq!(tokens.value[key], visible.value[key]);
            }
            assert_eq!(
                tokens.value["event_end"],
                json::integer(source.len() as i128)
            );
            assert_eq!(tokens.value["history"][369], json::integer(count));
            assert_eq!(tokens.value["history"][370], json::integer(truncated));
            assert_eq!(
                tokens.value["history"][371],
                json::integer(if count == 0 { 0 } else { 1 })
            );
        }
        let mut wrong = marker.clone();
        wrong["type"] = json::text("artifact_used");
        assert!(event(&wrong, &recipe, &b, &c).is_err());
        let mut unknown = marker;
        unknown["definition_id"] = json::text("unknown-public-id");
        assert!(event(&unknown, &recipe, &b, &c).is_err());
    }

    #[test]
    fn guardian_namespaces_safe_history_offsets_and_future_tail_are_exact() {
        let (b, c) = owners();
        let recipe = Recipe::corrected(2, &b, &c).unwrap();
        let guardian = recipe.guardian_ids().unwrap()[0].clone();
        let definition = recipe.definition_ids().unwrap()[0].clone();
        let events = vec![
            json::object([
                ("type", json::text("guardian_set")),
                ("player", json::integer(1)),
                ("definition_id", guardian.clone()),
            ]),
            json::object([
                ("type", json::text("guardian_left")),
                ("player", json::integer(2)),
                ("definition_id", guardian.clone()),
            ]),
            json::object([
                ("type", json::text("guardian_acted")),
                ("player", json::integer(1)),
                ("definition_id", definition.clone()),
            ]),
            json::object([("type", json::text("safe")), ("player", json::integer(1))]),
        ];
        let raw = fixture(&[], 4, &b, &c);
        let tokens = frame_tokens(&raw, &events, &recipe, &b, &c).unwrap();
        assert_eq!(tokens.value["history"][24], json::integer(1));
        assert_eq!(tokens.value["history"][63], json::integer(1));
        assert_eq!(tokens.value["history"][369], json::integer(2));
        assert_eq!(tokens.value["history"][370], json::integer(1));
        assert_eq!(tokens.value["history"][371], json::integer(2));
        let mut future = events.clone();
        future.push(json::object([("type", json::text("unknown-future-event"))]));
        assert_eq!(
            tokens.value,
            frame_tokens(&raw, &future, &recipe, &b, &c).unwrap().value
        );
        for (tag, wrong) in [
            ("guardian_set", definition.clone()),
            ("guardian_left", definition),
            ("artifact_used", guardian),
            ("guardian_acted", json::text("unknown-public-id")),
        ] {
            assert!(
                event(
                    &json::object([
                        ("type", json::text(tag)),
                        ("player", json::integer(1)),
                        ("definition_id", wrong)
                    ]),
                    &recipe,
                    &b,
                    &c
                )
                .is_err()
            );
        }
    }
    #[test]
    fn codec2_reuses_exact_binary_decoder_and_rejects_mixed_codebooks_and_slots() {
        let (b, c) = owners();
        let original = fixture(
            &[("bronze-club", None), ("fireball", Some("learned"))],
            0,
            &b,
            &c,
        );
        let recipe = Recipe::corrected(128, &b, &c).unwrap();
        // Both artifact and learned miracle refs are valid v2 discard syntax.
        let (bytes, mut descriptor) =
            one_column(&original, 3, 0, &[0, 1], [0; 4], [false; 4], &b, &c);
        let table = columns(&bytes, &descriptor, &original, 18, &b, &c).unwrap();
        assert_eq!(
            table.value["slots"],
            json::array([json::integer(0), json::integer(1)])
        );
        assert_eq!(
            candidate_tokens(&table.value, &recipe, &b, &c)
                .unwrap()
                .value["kinds"],
            json::array([json::integer(5)])
        );
        assert!(super::super::columns(&bytes, &descriptor, &original, 18, false, &b, &c).is_err());
        descriptor["codebook_sha256"] = json::text(&"0".repeat(64));
        assert!(columns(&bytes, &descriptor, &original, 18, &b, &c).is_err());
        let (bytes, descriptor) = one_column(&original, 3, 0, &[0, 0], [0; 4], [false; 4], &b, &c);
        assert!(columns(&bytes, &descriptor, &original, 18, &b, &c).is_err());
        let (mut bytes, descriptor) = one_column(&original, 3, 0, &[0], [0; 4], [false; 4], &b, &c);
        bytes[2] = 1; // canonical alignment padding and payload hash must fail.
        assert!(columns(&bytes, &descriptor, &original, 18, &b, &c).is_err());
    }
    #[test]
    fn all_action_families_use_only_v2_syntax_and_original_refs() {
        let (b, c) = owners();
        let mut original = fixture(
            &[("bronze-club", None), ("fireball", Some("learned"))],
            0,
            &b,
            &c,
        );
        for kind in 0..KINDS.len() {
            let choice = if KINDS[kind] == "respond" {
                "defense"
            } else if KINDS[kind] == "buy_decision" {
                "buy_accept_reject"
            } else {
                "turn_action"
            };
            original["payload"]["observation"]["required_choice"] = json::text(choice);
            original["payload"]["observation"]["typed_action_schema"] =
                typed_metadata(Some(choice));
            original["payload"]["typed_metadata"] = typed_metadata(Some(choice));
            original["payload"]["action_options"]["action_options"]["required_choice"] =
                json::text(choice);
            seal(&mut original, &b, &c);
            let target = if ["play_sequence", "sell", "buy"].contains(&KINDS[kind]) {
                3
            } else {
                0
            };
            let slots: &[u8] = match KINDS[kind] {
                "pray" | "buy_decision" | "respond" => &[],
                "sell" => &[0, 1],
                _ => &[1],
            };
            let (nums, present) = match KINDS[kind] {
                "exchange" => ([1, 0, 0, 0], [true, true, true, false]),
                "buy_decision" => ([0, 0, 0, 1], [false, false, false, true]),
                _ => ([0; 4], [false; 4]),
            };
            let (bytes, descriptor) =
                one_column(&original, kind, target, slots, nums, present, &b, &c);
            columns(&bytes, &descriptor, &original, 18, &b, &c).unwrap();
        }
    }
    #[test]
    fn rollout_frame_reuses_only_embedded_pins() {
        let (budget, control) = owners();
        let original = fixture(&[("bronze-club", None)], 0, &budget, &control);
        assert_eq!(embedded_pins(), &pins(&budget, &control).unwrap().root);
        for reuse in [false, true] {
            frame_for_profile(&original, &budget, &control, WireProfile::V2, reuse).unwrap();
        }
        let mut altered = original;
        altered["payload"]["catalog"]["bronze-club"]["attack"] = json::integer(999);
        seal(&mut altered, &budget, &control);
        let baseline = frame_for_profile(&altered, &budget, &control, WireProfile::V2, false).unwrap_err();
        let reused = frame_for_profile(&altered, &budget, &control, WireProfile::V2, true).unwrap_err();
        assert_eq!(baseline, reused);
    }

    #[test]
    fn original_frame_rejects_foreign_stale_hints_fog_and_perceived_values() {
        let (b, c) = owners();
        let original = fixture(&[("bronze-club", None)], 0, &b, &c);
        for mutation in 0..5 {
            let mut changed = original.clone();
            match mutation {
                0 => changed["payload"]["action_options"]["state_revision"] = json::integer(1),
                1 => {
                    changed["payload"]["action_options"]["action_options"]["usable_cards"][0]["card_ref"] =
                        json::text(&format!("c1_{:032x}", 99))
                }
                2 => changed["payload"]["observation"]["own"]["status"]["fog"] = Value::Bool(true),
                3 => changed["payload"]["catalog"]["bronze-club"]["attack"] = json::integer(999),
                _ => changed["payload"]["observation"]["action_schema_version"] = json::integer(1),
            }
            seal(&mut changed, &b, &c);
            assert!(frame(&changed, &b, &c).is_err());
        }
        let mut fog = original.clone();
        fog["payload"]["observation"]["own"]["status"]["fog"] = Value::Bool(true);
        fog["payload"]["observation"]["opponent"] = json::object([
            ("player_id", json::integer(2)),
            ("learned_miracles", json::array([])),
            ("ascended", Value::Bool(false)),
        ]);
        seal(&mut fog, &b, &c);
        frame(&fog, &b, &c).unwrap();
        let recipe = Recipe::corrected(128, &b, &c).unwrap();
        let tokens = frame_tokens(&fog, &[], &recipe, &b, &c).unwrap();
        assert_eq!(tokens.value["state_present"][12], Value::Bool(false));
        assert_eq!(
            tokens.value["state_category_present"][8],
            Value::Bool(false)
        );
    }
    #[test]
    fn cancelled_and_underbudget_diagnostics_do_not_return_tokens() {
        let (b, c) = owners();
        let raw = fixture(&[("bronze-club", None)], 0, &b, &c);
        let recipe = Recipe::corrected(128, &b, &c).unwrap();
        let tiny = Budget::new(1);
        assert!(frame(&raw, &tiny, &c).is_err());
        assert!(frame_tokens(&raw, &[], &recipe, &tiny, &c).is_err());
        c.cancel();
        assert_eq!(frame(&raw, &b, &c).err(), Some(Error::Cancelled));
        assert_eq!(
            frame_tokens(&raw, &[], &recipe, &b, &c).err(),
            Some(Error::Cancelled)
        );
    }
    #[test]
    fn j6_envelope_is_partial_binds_expected_metadata_and_keeps_generation_null() {
        let (b, c) = owners();
        let old = parse(
            include_bytes!("../tests/fixtures/structured_cohort16_inputs.json"),
            &b,
            &c,
        );
        let raw = old["episodes"][0]["journal"]
            .string()
            .unwrap()
            .lines()
            .next()
            .unwrap();
        let mut header = parse(raw.as_bytes(), &b, &c);
        let p = pins(&b, &c).unwrap();
        let actions = json::object([
            ("version", json::text("3")),
            ("generator", json::text("public-card-instance-diverse-v5")),
            ("action_contract", p.root["action_contract"].clone()),
            ("use_action_options", Value::Bool(true)),
            (
                "targets",
                json::array([json::text("opponent"), json::text("self")]),
            ),
            ("max_sequence_length", json::integer(18)),
            ("max_candidates", json::integer(32768)),
            ("max_expansions", json::integer(262144)),
            ("max_temporary_candidates", json::integer(262144)),
            ("count_states", json::integer(16384)),
            ("count_milliseconds", json::integer(50)),
            ("search_milliseconds", json::integer(1000)),
            (
                "channel_weights",
                json::array((0..7).map(|_| json::integer(1))),
            ),
        ]);
        // Synthetic supplied declaration: deliberately no runtime/artifact
        // attestation is asserted by this header-envelope-only diagnostic.
        let runtime = json::object([("diagnostic_fixture", Value::Bool(true))]);
        for (key, text) in [
            ("feature_version", "participant-structured-v2"),
            ("input_recipe", "participant-structured-v2"),
            ("policy_input_version", "player-card-refs-v2"),
            ("generator_version", "public-card-instance-diverse-v5"),
            ("purpose", "qualified_game"),
            ("model_id", "random"),
            ("model_version", "1"),
            ("policy", "random-card-instance-v2"),
        ] {
            header[key] = json::text(text);
        }
        header["schema_version"] = json::integer(6);
        header["protocol_version"] = json::integer(3);
        header["generation_schema"] = Value::Null;
        header["action_contract"] = p.root["action_contract"].clone();
        header["rule_signature"] = p.root["rule_signature"].clone();
        header["layout_signature"] =
            json::text(crate::structured_controls::CARD_INSTANCE_SIGNATURE);
        header["actions"] = actions.clone();
        header["action_config_hash"] = json::text(&digest(&actions, &b, &c).unwrap());
        header["runtime_evidence"] = runtime.clone();
        header["aggregate_window"] = json::integer(2);
        header["token_signature"] = json::text(
            &crate::structured_controls::card_instance_token_signature(2, &b, &c).unwrap(),
        );
        header["caps"]["max_positions"] = json::integer(18);
        header["caps"]["max_candidates"] = json::integer(32768);
        header["catalog"] = json::object([
            ("catalog_schema_version", json::integer(2)),
            ("catalog_version", json::text("public-ja-v3")),
            ("rule_set_version", json::text("true-match-1v1-v2")),
            ("locale", json::text("ja")),
            (
                "content_sha256",
                p.root["action_contract"]["catalog_sha256"].clone(),
            ),
            ("definition_count", json::integer(296)),
        ]);
        diagnostic_header_envelope(
            &header,
            DiagnosticHeaderExpected {
                tree: None,
                actions: &actions,
                neural: None,
                runtime_evidence: &runtime,
            },
            &b,
            &c,
        )
        .unwrap();
        header["generation_schema"] = json::integer(3);
        assert!(
            diagnostic_header_envelope(
                &header,
                DiagnosticHeaderExpected {
                    tree: None,
                    actions: &actions,
                    neural: None,
                    runtime_evidence: &runtime
                },
                &b,
                &c
            )
            .is_err()
        );
        header["generation_schema"] = Value::Null;
        header["token_signature"] = json::text(crate::structured_controls::CARD_INSTANCE_SIGNATURE);
        assert!(
            diagnostic_header_envelope(
                &header,
                DiagnosticHeaderExpected {
                    tree: None,
                    actions: &actions,
                    neural: None,
                    runtime_evidence: &runtime
                },
                &b,
                &c
            )
            .is_err()
        );
        header["token_signature"] = json::text(
            &crate::structured_controls::card_instance_token_signature(2, &b, &c).unwrap(),
        );
        let neural = json::object([
            (
                "input_contract",
                json::text("retained-external-inference-v1"),
            ),
            ("policy_options", json::object([])),
            ("inference_manifest", json::object([])),
            ("files", json::array([])),
        ]);
        for model in ["mlp", "gru", "transformer"] {
            header["model_id"] = json::text(model);
            header["model_version"] = json::text("3");
            header["policy"] = json::text("supplied-diagnostic-label-not-validated");
            header["neural"] = neural.clone();
            diagnostic_header_envelope(
                &header,
                DiagnosticHeaderExpected {
                    tree: None,
                    actions: &actions,
                    neural: Some(&neural),
                    runtime_evidence: &runtime,
                },
                &b,
                &c,
            )
            .unwrap();
            let changed = json::object([]);
            assert!(
                diagnostic_header_envelope(
                    &header,
                    DiagnosticHeaderExpected {
                        tree: None,
                        actions: &actions,
                        neural: Some(&changed),
                        runtime_evidence: &runtime
                    },
                    &b,
                    &c
                )
                .is_err()
            );
        }
    }

    #[test]
    fn every_public_event_has_exact_category_and_historical_ids_need_no_current_catalog() {
        let (b, c) = owners();
        let recipe = Recipe::corrected(128, &b, &c).unwrap();
        let mut events = parse(
            br#"[
            {"type":"turn_started","turn":1,"active_player":1},
            {"type":"action_accepted","player":1},
            {"type":"artifact_used","player":1,"definition_id":"bronze-club"},
            {"type":"artifact_drawn","player":1,"definition_id":"bronze-club"},
            {"type":"devil_appeared","player":1,"definition_id":"bronze-club"},
            {"type":"artifact_discarded","player":1,"definition_id":"bronze-club"},
            {"type":"guardian_set","player":1,"definition_id":"placeholder"},
            {"type":"guardian_left","player":1,"definition_id":"placeholder"},
            {"type":"guardian_acted","player":1,"definition_id":"bronze-club"},
            {"type":"miracle_used","player":1,"definition_id":"fireball","mp_spent":0},
            {"type":"attack_declared","attacker":1,"target":2,"attack":5},
            {"type":"attack_missed","attacker":1,"target":2},
            {"type":"damage_applied","player":2,"amount":5,"remaining_hp":35},
            {"type":"healed","player":1,"amount":5,"hp":45},
            {"type":"resource_changed","player":1,"hp_delta":0,"mp_delta":-1,"money_delta":2},
            {"type":"buy_offered","buyer":1,"seller":2,"definition_id":"bronze-club","price":5},
            {"type":"sale_offered","seller":1,"target":2,"definition_id":"bronze-club","price":5},
            {"type":"artifact_transferred","from":1,"to":2,"definition_id":"bronze-club","price":5},
            {"type":"sun_amulet_activated","player":1},
            {"type":"status_changed","player":1,"status":"dark_cloud","active":true},
            {"type":"phenomenon_invoked","definition_id":"fireball"},
            {"type":"player_ascended","player":1},
            {"type":"player_defeated","player":2},
            {"type":"game_ended","result":"draw"},
            {"type":"safe","player":1}
        ]"#,
            &b,
            &c,
        );
        events[6]["definition_id"] = recipe.guardian_ids().unwrap()[0].clone();
        events[7]["definition_id"] = recipe.guardian_ids().unwrap()[0].clone();
        let raw = fixture(&[], 25, &b, &c); // No currently possessed definitions.
        frame(&raw, &b, &c).unwrap();
        let tokens = frame_tokens(&raw, events.array().unwrap(), &recipe, &b, &c).unwrap();
        for index in 0..25 {
            assert_eq!(
                tokens.value["interval_categories"][index][0],
                json::integer(index as i128 + 2)
            );
            assert_eq!(tokens.value["history"][index], json::integer(1));
        }
        assert_eq!(tokens.value["interval_numerics"].array().unwrap().len(), 25);
        assert_eq!(tokens.value["history"][369], json::integer(25));
        events[19]["status"] = json::text("unknown-status");
        assert!(frame_tokens(&raw, events.array().unwrap(), &recipe, &b, &c).is_err());
    }
    #[test]
    fn literal_python_corrected_frame_binds_recipe_policy_and_complete_contract() {
        let (b, c) = owners();
        let literal = parse(
            include_bytes!("../tests/fixtures/card_instance_corrected_frame.json"),
            &b,
            &c,
        );
        let original = literal.require("frame").unwrap();
        assert_eq!(
            original["frame_hash"].string().unwrap(),
            "4a6ccf18f798ce2f1bc9761300d4be11354cdebf128cf1c43e88d1a55f040c7e"
        );
        frame(original, &b, &c).unwrap();
        // A correct digest of the unbound wire fields must still be rejected.
        let mut changed = original.clone();
        changed.as_object_mut().unwrap().remove("frame_hash");
        let unbound_hash = digest(&changed, &b, &c).unwrap();
        changed["frame_hash"] = json::text(&unbound_hash);
        assert!(frame(&changed, &b, &c).is_err());
        // Historical 07e card frames bind recipe/policy, but no full contract.
        let mut historical_binding = original.clone();
        historical_binding
            .as_object_mut()
            .unwrap()
            .remove("frame_hash");
        historical_binding["input_recipe"] = json::text("participant-structured-v2");
        historical_binding["policy_input_version"] = json::text("player-card-refs-v2");
        changed["frame_hash"] = json::text(&digest(&historical_binding, &b, &c).unwrap());
        assert!(frame(&changed, &b, &c).is_err());
        // Also reject a resealed digest carrying the old full contract.
        let mut old_contract = pins(&b, &c).unwrap().root["action_contract"].clone();
        old_contract["profile_id"] = json::text("card-instance-simulator-v2-20261103");
        old_contract["simulator_revision"] = json::text("07e8185f98c503aa41200a80ea554b4fd21088ad");
        old_contract["spec_inventory_sha256"] =
            json::text("8381079eae8a68378d8419c7f7245c62358833f0dd9045667021e112a1e9bad5");
        historical_binding["action_contract"] = old_contract;
        changed["frame_hash"] = json::text(&digest(&historical_binding, &b, &c).unwrap());
        assert!(frame(&changed, &b, &c).is_err());
    }
}

/// Complete public receipt metadata validation, never current executable/PID
/// attestation. The external original owner remains responsible for those reads.
pub(crate) fn runtime_receipt(
    value: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<()> {
    value.closed(
        &[
            "schema_version",
            "action_contract",
            "build",
            "binary",
            "source",
            "executed",
        ],
        &[],
    )?;
    let expected = value.require("action_contract")?;
    let profile = WireProfile::expected(expected)?;
    let v3 = profile == WireProfile::DefenseV3;
    let schema = if v3 { 2 } else { 1 };
    value.require("schema_version")?.integer(schema, schema)?;
    contract(expected, profile.frame_contract())?;
    let build = value.require("build")?;
    let parsed;
    if v3 {
        require(build == defense_v3_release(), "runtime approved v3 release mismatch")?;
    } else {
        parsed = pins(budget, control)?;
        build.closed(
            &["binary_sha256", "spec_inventory_sha256", "change_inventory_sha256", "spec_files", "change_files"],
            &[],
        )?;
        for (key, pin) in [
            ("binary_sha256", "approved_binary_sha256"),
            ("change_inventory_sha256", "approved_change_sha256"),
            ("spec_files", "approved_spec_files"),
            ("change_files", "approved_change_files"),
        ] {
            require(build.require(key)? == parsed.root.require(pin)?, "runtime approved release mismatch")?;
        }
        require(
            build.require("spec_inventory_sha256")? == expected.require("spec_inventory_sha256")?,
            "runtime approved spec inventory mismatch",
        )?;
    }
    let binary = value.require("binary")?;
    binary.closed(&["path", "sha256", "identity"], &[])?;
    let source = value.require("source")?;
    source.closed(&["path", "revision", "spec_files", "change_files"], if v3 { &["dirty"] } else { &[] })?;
    require(
        (!v3 || source.require("dirty")? == &Value::Bool(false))
            && binary.require("sha256")? == build.require("binary_sha256")?
            && source.require("revision")? == expected.require("simulator_revision")?
            && source.require("spec_files")? == build.require("spec_files")?
            && source.require("change_files")? == build.require("change_files")?,
        "runtime source/binary release mismatch",
    )?;
    for v in [binary, source] {
        require(
            v.require("path")?.string()?.starts_with('/'),
            "runtime absolute path required",
        )?;
    }
    let identity = |v: &Value| -> Result<()> {
        let rows = v.array()?;
        require(rows.len() == 5, "runtime stat identity shape")?;
        for n in rows {
            n.integer(0, i128::MAX)?;
        }
        Ok(())
    };
    identity(binary.require("identity")?)?;
    let executed = value.require("executed")?;
    executed.closed(&["pid", "start_time", "sha256", "identity"], &[])?;
    executed.require("pid")?.integer(1, i128::MAX)?;
    executed.require("start_time")?.integer(1, i128::MAX)?;
    identity(executed.require("identity")?)?;
    require(
        executed.require("sha256")? == build.require("binary_sha256")?,
        "executed release mismatch",
    )
}
/// Full J6 header metadata validator, distinct from the deliberately partial
/// diagnostic envelope above. Does not inspect artifacts or mint Source power.
pub(crate) fn header6(value: &Value, budget: &Arc<Budget>, control: &Control) -> Result<()> {
    let _scratch = budget.reserve(256 * 1024)?;
    let actions = value.require("actions")?;
    crate::structured_controls::actions3(actions, budget, control)?;
    let neural = value.get("neural");
    diagnostic_header_envelope(
        value,
        DiagnosticHeaderExpected {
            actions,
            neural,
            tree: value.get("tree"),
            runtime_evidence: value.require("runtime_evidence")?,
        },
        budget,
        control,
    )?;
    let receipt = value.require("runtime_evidence")?;
    runtime_receipt(receipt, budget, control)?;
    require(
        receipt.require("action_contract")? == value.require("action_contract")?,
        "J6 runtime receipt profile mismatch",
    )?;
    let (caps, costs) = if value.require("model_version")?.string()? == "3"
        && value.require("model_id")?.string()? == "lightgbm"
    {
        let raw = value.require("tree")?;
        let label = crate::structured_controls::tree3::metadata(raw, actions, budget, control)?;
        require(
            value.require("policy")?.string()? == label,
            "J6 Tree3 own policy label mismatch",
        )?;
        let safe = raw.require("policy_options")?;
        require(
            value.require("aggregate_window")?
                == safe
                    .require("model")?
                    .require("encoder")?
                    .require("aggregate_window")?,
            "J6 Tree3 actual window mismatch",
        )?;
        crate::structured_controls::tree3::settings(safe, actions, budget, control)?
    } else if ["3", "4"].contains(&value.require("model_version")?.string()?) {
        let raw = value.require("neural")?;
        let hpo = value.require("model_version")?.string()? == "4";
        let label = if hpo {
            crate::structured_controls::hpo::neural_metadata(raw, actions, budget, control)?
        } else {
            crate::structured_controls::neural_metadata3(raw, actions, budget, control)?
        };
        require(
            value.require("policy")?.string()? == label,
            "J6 neural execution label mismatch",
        )?;
        let options = raw.require("policy_options")?;
        is(
            options,
            "version",
            value.require("model_version")?.string()?,
        )?;
        let model = options.require("model")?;
        require(
            value.require("model_id")? == model.require("model_id")?
                && value.require("aggregate_window")?
                    == model.require("event")?.require("aggregate_window")?,
            "J6 model/window identity mismatch",
        )?;
        crate::structured_controls::worker_settings3(
            model,
            options.require("inference")?,
            actions,
            budget,
            control,
        )?
    } else {
        let p = pins(budget, control)?;
        (
            json::object([
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
            ]),
            json::copy(p.root.require("baseline_numeric_costs")?, control)?,
        )
    };
    require(
        value.require("caps")? == &caps && value.require("numeric_costs")? == &costs,
        "J6 exact caps/numeric costs mismatch",
    )
}

#[cfg(test)]
mod full_validation_tests {
    use super::*;
    use crate::{
        structured_controls as controls,
        structured_tokens::{self, Recipe},
    };
    fn owners() -> (Arc<Budget>, Control) {
        (
            Budget::new(1024 * 1024 * 1024),
            Control::new(crate::budget::monotonic() + 120.).unwrap(),
        )
    }
    fn fixture(raw: &[u8], b: &Arc<Budget>, c: &Control) -> Document {
        json::parse(raw, b, c).unwrap()
    }
    fn bytes(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }
    fn tensor_shape(v: &Value) -> Vec<usize> {
        if let Some(rows) = v.as_array() {
            let mut result = vec![rows.len()];
            if let Some(first) = rows.first() {
                result.extend(tensor_shape(first));
            }
            result
        } else {
            vec![]
        }
    }
    fn tensor_bytes(v: &Value, dtype: &str, out: &mut Vec<u8>) {
        if let Some(rows) = v.as_array() {
            for row in rows {
                tensor_bytes(row, dtype, out)
            }
        } else if dtype == "uint8" {
            out.push(u8::from(v.boolean().unwrap()));
        } else {
            out.extend_from_slice(
                &(v.integer(i64::MIN as i128, i64::MAX as i128).unwrap() as i64).to_le_bytes(),
            );
        }
    }
    #[test]
    fn defense_v3_journal6_header_binds_rule_signature_catalog_and_release() {
        let (b, c) = owners();
        let all = fixture(include_bytes!("../tests/fixtures/card_instance_journal6_rules_v2.json"), &b, &c);
        let mut header = all.root["journals"][0]["rows"][0].clone();
        let legacy = header.clone();
        let profile = WireProfile::DefenseV3;
        let served = defense_v3_release();
        let action = profile.frame_contract();
        let receipt = &mut header["runtime_evidence"];
        receipt["schema_version"] = json::integer(2);
        receipt["action_contract"] = action.clone();
        receipt["build"] = served.clone();
        receipt["source"]["dirty"] = Value::Bool(false);
        receipt["source"]["revision"] = action["simulator_revision"].clone();
        receipt["source"]["spec_files"] = served["spec_files"].clone();
        receipt["source"]["change_files"] = served["change_files"].clone();
        receipt["binary"]["sha256"] = served["binary_sha256"].clone();
        receipt["executed"]["sha256"] = served["binary_sha256"].clone();
        header["catalog"] = profile.catalog_metadata();
        header["action_contract"] = action.clone();
        header["actions"]["action_contract"] = action.clone();
        header["action_config_hash"] = json::text(&digest(&header["actions"], &b, &c).unwrap());
        header["rule_signature"] = profile.rule_signature().clone();
        header6(&header, &b, &c).unwrap();
        header6(&legacy, &b, &c).unwrap();
        for key in ["rule_signature", "catalog", "runtime_evidence", "action_contract"] {
            let mut mixed = header.clone();
            mixed[key] = legacy[key].clone();
            assert!(header6(&mixed, &b, &c).is_err(), "{key}");
        }
    }

    #[test]
    fn card_instance_tree3_full_j6_raw_tokens_and_closed_header_transactions() {
        let (b, c) = owners();
        let all = fixture(
            include_bytes!("../tests/fixtures/card_instance_tree3_raw_journal_rules_v2.json"),
            &b,
            &c,
        );
        let case = &all.root["journals"][0];
        let dir = directory();
        write_journal(&dir, &case["rows"], case, &b, &c);
        let journal = super::super::Journal::open_corrected(
            &dir,
            "player-1.jsonl",
            b.clone(),
            c.clone(),
            16 << 20,
            4 << 20,
        )
        .unwrap();
        let encoded = journal.diagnostic().unwrap();
        let raw = json::parse(&encoded, &b, &c).unwrap();
        assert_eq!(
            raw.root["decisions"][0]["frame_tokens"],
            case["frame_tokens"]
        );
        assert_eq!(
            raw.root["decisions"][0]["candidate_tokens"],
            case["candidate_tokens"]
        );
        assert_eq!(raw.root["header"], case["rows"][0]);
        assert_eq!(
            raw.root["projection"]["declaration"],
            json::text("tree-raw-corrected-own-header")
        );
        assert!(!raw.root["qualified"].boolean().unwrap());
        assert!(
            controls::model3(&case["rows"][0]["tree"]["policy_options"]["model"], &b, &c).is_err()
        );
        drop(journal);
        for mutation in 0..11 {
            let mut rows = case["rows"].clone();
            match mutation {
                0 => rows[2]["selection"]["logprob"] = json::integer(0),
                1 => rows[2]["selection"]["value"] = json::integer(0),
                2 => {
                    let tree = rows[0]["tree"].clone();
                    rows[0]
                        .as_object_mut()
                        .unwrap()
                        .insert("neural".into(), tree);
                }
                3 => {
                    rows[0]["tree"]["policy_options"]["source_origin"]["purpose"] =
                        json::text("component")
                }
                4 => rows[0]["model_version"] = json::text("2"),
                5 => {
                    rows[0]["tree"]["policy_options"]["training_pair"]["request_sha256"] =
                        json::text(&"f".repeat(64))
                }
                6 => rows[0]["aggregate_window"] = json::integer(1),
                7 => {
                    rows[0]["tree"]["inference_manifest"]["feature_layout"]["raw_field_mapping"]
                        [0]["segments"][0] = json::text("other-private")
                }
                8 => rows[0]["viewer"] = json::integer(2),
                9 => rows[0]["numeric_costs"]["tail_operations_per_candidate"] = json::integer(0),
                _ => rows[2]["frame"]["boundary"]["prefix_hash"] = json::text(&"0".repeat(64)),
            }
            write_journal(&dir, &rows, case, &b, &c);
            assert!(
                super::super::Journal::open_corrected(
                    &dir,
                    "player-1.jsonl",
                    b.clone(),
                    c.clone(),
                    16 << 20,
                    4 << 20
                )
                .is_err(),
                "mutation{mutation}"
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn corrected_python_token_arrays_codec_dtypes_shapes_offsets_are_exact() {
        let (b, c) = owners();
        let all = fixture(
            include_bytes!("../tests/fixtures/card_instance_tokens_python.json"),
            &b,
            &c,
        );
        for case in all.root["cases"].array().unwrap() {
            let raw = &case["frame"];
            frame(raw, &b, &c).unwrap();
            let recipe = Recipe::corrected(2, &b, &c).unwrap();
            let tokens = structured_tokens::frame_tokens(
                raw,
                case["events"].array().unwrap(),
                &recipe,
                &b,
                &c,
            )
            .unwrap();
            assert_eq!(
                tokens.value, case["frame_tokens"],
                "{} frame",
                case["label"]
            );
            let table = columns(
                &bytes(case["bytes_hex"].string().unwrap()),
                &case["columns"],
                raw,
                18,
                &b,
                &c,
            )
            .unwrap();
            assert_eq!(table.value, case["table"], "{} table", case["label"]);
            let candidates =
                structured_tokens::candidate_tokens(&table.value, &recipe, &b, &c).unwrap();
            assert_eq!(
                candidates.value, case["candidate_tokens"],
                "{} candidates",
                case["label"]
            );
            for (name, expected) in case["arrays"].object().unwrap() {
                let (group, key) = name.split_once('.').unwrap();
                let values = if group == "frame" {
                    &tokens.value[key]
                } else {
                    &candidates.value[key]
                };
                assert_eq!(
                    json::array(
                        tensor_shape(values)
                            .into_iter()
                            .map(|n| json::integer(n as i128))
                    ),
                    expected["shape"],
                    "{name} shape"
                );
                let mut actual = Vec::new();
                tensor_bytes(values, expected["dtype"].string().unwrap(), &mut actual);
                assert_eq!(
                    actual,
                    bytes(expected["bytes_hex"].string().unwrap()),
                    "{name} dtype/bytes"
                );
            }
        }
    }
    #[test]
    fn corrected_models3_inference4_labels_worker_resources_match_python_nine_configs() {
        let (b, c) = owners();
        let all = fixture(
            include_bytes!("../tests/fixtures/card_instance_models3_rules_v2.json"),
            &b,
            &c,
        );
        for case in all.root["cases"].array().unwrap() {
            let shapes = controls::model3(&case["model"], &b, &c).unwrap();
            let actual = Value::Object(
                shapes
                    .into_iter()
                    .map(|(k, v)| (k, json::array(v.into_iter().map(json::integer))))
                    .collect(),
            );
            assert_eq!(actual, case["shapes"], "{} shapes", case["label"]);
            assert_eq!(
                controls::options3(&case["cuda_options"], &case["actions"], &b, &c).unwrap(),
                case["cuda_label"].string().unwrap()
            );
            assert_eq!(
                controls::resource3(
                    &case["model"],
                    &case["cuda_options"]["inference"],
                    None,
                    None,
                    &b,
                    &c
                )
                .unwrap(),
                case["resources_cuda"]
            );

            controls::inference4(&case["manifest"], &case["actions"], &b, &c).unwrap();
            assert_eq!(
                controls::options3(&case["options"], &case["actions"], &b, &c).unwrap(),
                case["policy_label"].string().unwrap()
            );
            assert_eq!(
                controls::neural_metadata3(&case["neural"], &case["actions"], &b, &c).unwrap(),
                case["policy_label"].string().unwrap()
            );
            let (caps, costs) = controls::worker_settings3(
                &case["model"],
                &case["inference"],
                &case["actions"],
                &b,
                &c,
            )
            .unwrap();
            assert_eq!(caps, case["caps"], "{} caps", case["label"]);
            assert_eq!(costs, case["costs"], "{} costs", case["label"]);
            assert_eq!(
                controls::resource3(&case["model"], &case["inference"], None, None, &b, &c)
                    .unwrap(),
                case["resources"],
                "{} resources",
                case["label"]
            );
            assert_eq!(
                controls::resource3(&case["model"], &case["inference"], Some(1), Some(1), &b, &c)
                    .unwrap(),
                case["resources_one"],
                "{} one resources",
                case["label"]
            );
        }
    }
    #[test]
    fn corrected_inference_and_execution_reject_malformed_mixed_and_resource_metadata() {
        let (b, c) = owners();
        let all = fixture(
            include_bytes!("../tests/fixtures/card_instance_models3_rules_v2.json"),
            &b,
            &c,
        );
        let base = &all.root["cases"][0];
        for mutation in 0..9 {
            let mut manifest = base["manifest"].clone();
            match mutation {
                0 => manifest["schema_version"] = json::integer(3),
                1 => manifest["model"]["version"] = json::text("2"),
                2 => manifest["token_signature"] = json::text(controls::CARD_INSTANCE_SIGNATURE),
                3 => manifest["rule_signature"] = json::text(&"0".repeat(64)),
                4 => manifest["action_config_hash"] = json::text(&"0".repeat(64)),
                5 => manifest["numerics"]["amp"] = Value::Bool(true),
                6 => manifest["runtime"]["foreign"] = Value::Null,
                7 => {
                    manifest["action_contract"]["profile_id"] =
                        json::text("card-instance-simulator-v2-20261103")
                }
                _ => manifest["weights"]["decoded_bytes"] = json::integer(0),
            };
            assert!(
                controls::inference4(&manifest, &base["actions"], &b, &c).is_err(),
                "mutation{mutation}"
            );
        }
        let mut neural = base["neural"].clone();
        neural["policy_options"]["cuda_visible_devices"] = Value::Null;
        assert!(controls::neural_metadata3(&neural, &base["actions"], &b, &c).is_err());
        let mut inference = base["inference"].clone();
        inference["managed_budget_bytes"] = json::integer(1);
        assert!(controls::resource3(&base["model"], &inference, None, None, &b, &c).is_err());
        let mut model = base["model"].clone();
        model["resources"]["max_tensor_bytes"] = json::integer(1);
        assert!(controls::model3(&model, &b, &c).is_err());
        let mut neural = base["neural"].clone();
        neural["files"][0]["sha256"] = json::text(&"f".repeat(64));
        assert!(controls::neural_metadata3(&neural, &base["actions"], &b, &c).is_err());
    }
    fn journal_case<'a>(root: &'a Value, index: usize) -> &'a Value {
        &root["journals"][index]
    }
    fn directory() -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "gf-card-instance-j6-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&p).unwrap();
        p
    }
    fn write_journal(
        dir: &std::path::Path,
        rows: &Value,
        case: &Value,
        b: &Arc<Budget>,
        c: &Control,
    ) {
        let mut text = Vec::new();
        for row in rows.array().unwrap() {
            text.extend_from_slice(&json::canonical(row, b, c).unwrap());
            text.push(b'\n');
        }
        let viewer = rows[0]["viewer"].integer(1, 2).unwrap();
        std::fs::write(dir.join(format!("player-{viewer}.jsonl")), text).unwrap();
        std::fs::write(
            dir.join(case["rows"][2]["table"]["file"].string().unwrap()),
            bytes(case["bytes_hex"].string().unwrap()),
        )
        .unwrap();
    }
    #[test]
    fn full_journal6_transaction_bank_report_and_natural_closure_accept_four_python_records() {
        let (b, c) = owners();
        let all = fixture(
            include_bytes!("../tests/fixtures/card_instance_journal6_rules_v2.json"),
            &b,
            &c,
        );
        for case in all.root["journals"].array().unwrap() {
            let dir = directory();
            write_journal(&dir, &case["rows"], case, &b, &c);
            let journal = super::super::Journal::open_corrected(
                &dir,
                "player-1.jsonl",
                b.clone(),
                c.clone(),
                16 * 1024 * 1024,
                4 * 1024 * 1024,
            )
            .unwrap();
            assert_eq!(journal.accepted_count(), 1);
            assert_eq!(journal.end_value()["end_kind"], json::text("natural"));
            assert!(
                super::super::Journal::open(
                    &dir,
                    "player-1.jsonl",
                    b.clone(),
                    c.clone(),
                    16 * 1024 * 1024,
                    4 * 1024 * 1024
                )
                .is_err()
            );
            drop(journal);
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
    #[test]
    fn full_journal6_rejects_missing_foreign_stale_future_prefix_and_transaction_mutations() {
        let (b, c) = owners();
        let all = fixture(
            include_bytes!("../tests/fixtures/card_instance_journal6_rules_v2.json"),
            &b,
            &c,
        );
        let case = journal_case(&all.root, 1);
        for mutation in 0..15 {
            let mut rows = case["rows"].clone();
            match mutation {
                0 => rows[1]["delta"]["actor"] = json::integer(2),
                1 => rows[1]["delta"]["previous_hash"] = json::text(&"0".repeat(64)),
                2 => {
                    rows[1]["delta"]["events"].as_array_mut().unwrap().pop();
                }
                3 => rows[2]["frame"]["boundary"]["end"] = json::integer(5),
                4 => rows[2]["search_report"]["rule_signature"] = json::text(&"0".repeat(64)),
                5 => {
                    rows[3]["response"]["command_id"] =
                        json::text("22222222-2222-2222-2222-222222222222")
                }
                6 => rows[3]["response"]["state_revision"] = json::integer(2),
                7 => rows[4]["reward"] = json::integer(1),
                8 => rows[5]["terminal_reward"] = json::integer(-1),
                9 => rows[2]["selection"]["logprob"] = Value::Null,
                10 => rows[0]["generation_schema"] = json::integer(3),
                11 => {
                    rows[0]["runtime_evidence"]["executed"]["sha256"] = json::text(&"0".repeat(64))
                }
                12 => {
                    rows.as_array_mut().unwrap().swap(2, 3);
                }
                13 => {
                    rows[2]["frame"]["interval_start"] = json::integer(1);
                }
                _ => {
                    rows[2]["frame"]["interval_start_hash"] = json::text(&"f".repeat(64));
                }
            }
            if [3, 13, 14].contains(&mutation) {
                let mut binding = rows[2]["frame"].clone();
                binding.as_object_mut().unwrap().remove("frame_hash");
                let p = pins(&b, &c).unwrap();
                binding["input_recipe"] = json::text("participant-structured-v2");
                binding["policy_input_version"] = json::text("player-card-refs-v2");
                binding["action_contract"] = p.root["action_contract"].clone();
                rows[2]["frame"]["frame_hash"] = json::text(&digest(&binding, &b, &c).unwrap());
                frame(&rows[2]["frame"], &b, &c).unwrap(); // Reach the bank guard with a valid resealed frame.
            }
            let dir = directory();
            write_journal(&dir, &rows, case, &b, &c);
            assert!(
                super::super::Journal::open_corrected(
                    &dir,
                    "player-1.jsonl",
                    b.clone(),
                    c.clone(),
                    16 * 1024 * 1024,
                    4 * 1024 * 1024
                )
                .is_err(),
                "mutation{mutation}"
            );
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
    #[test]
    fn full_journal6_preserves_empty_natural_and_interrupted_pending_outcomes() {
        let (b, c) = owners();
        let all = fixture(
            include_bytes!("../tests/fixtures/card_instance_journal6_rules_v2.json"),
            &b,
            &c,
        );
        let case = journal_case(&all.root, 1);
        for kind in ["natural", "cancelled", "incomplete", "aborted", "truncated"] {
            let mut end = case["rows"][5].clone();
            end["end_kind"] = json::text(kind);
            let rows = if kind == "natural" {
                json::array([case["rows"][0].clone(), end])
            } else {
                end["outcome"] = Value::Null;
                end["terminal_reward"] = Value::Null;
                end["final_observation"] = Value::Null;
                end["pending_decisions"] = json::array([json::integer(0)]);
                if kind == "truncated" {
                    end["simulator_abort_reason"] = json::text("turn_limit_exceeded");
                }
                if kind == "aborted" {
                    end["simulator_abort_reason"] = json::text("participant_disconnected");
                }
                json::array([
                    case["rows"][0].clone(),
                    case["rows"][1].clone(),
                    case["rows"][2].clone(),
                    end,
                ])
            };
            let dir = directory();
            write_journal(&dir, &rows, case, &b, &c);
            let journal = super::super::Journal::open_corrected(
                &dir,
                "player-1.jsonl",
                b.clone(),
                c.clone(),
                16 * 1024 * 1024,
                4 * 1024 * 1024,
            )
            .unwrap();
            assert_eq!(journal.accepted_count(), 0);
            assert_eq!(journal.end_value()["end_kind"], json::text(kind));
            assert_eq!(
                journal.end_value()["terminal_reward"],
                if kind == "natural" {
                    json::integer(0)
                } else {
                    Value::Null
                }
            );
            drop(journal);
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
    #[test]
    fn full_journal6_python_viewer_win_loss_draw_rewards_are_exact() {
        let (b, c) = owners();
        let all = fixture(
            include_bytes!("../tests/fixtures/card_instance_journal6_rewards_rules_v2.json"),
            &b,
            &c,
        );
        assert_eq!(all.root["cases"].array().unwrap().len(), 6);
        let mut covered = std::collections::BTreeSet::new();
        for case in all.root["cases"].array().unwrap() {
            let viewer = case["viewer"].integer(1, 2).unwrap();
            let expected = case["expected_reward"].integer(-1, 1).unwrap();
            covered.insert((viewer, expected));
            let dir = directory();
            write_journal(&dir, &case["rows"], case, &b, &c);
            let journal = super::super::Journal::open_corrected(
                &dir,
                &format!("player-{viewer}.jsonl"),
                b.clone(),
                c.clone(),
                16 * 1024 * 1024,
                4 * 1024 * 1024,
            )
            .unwrap();
            assert_eq!(journal.header_value()["viewer"], json::integer(viewer));
            assert_eq!(journal.accepted_count(), 1);
            assert_eq!(journal.end_value()["end_kind"], json::text("natural"));
            assert_eq!(
                journal.end_value()["terminal_reward"],
                json::integer(expected)
            );
            assert_eq!(journal.end_value()["outcome"], case["rows"][5]["outcome"]);
            drop(journal);
            std::fs::remove_dir_all(dir).unwrap();
        }
        assert_eq!(
            covered,
            [(1, -1), (1, 0), (1, 1), (2, -1), (2, 0), (2, 1)]
                .into_iter()
                .collect()
        );
    }
    #[test]
    fn full_journal6_rejects_wrong_signs_and_fully_rebound_viewer_rewards() {
        let (b, c) = owners();
        let all = fixture(
            include_bytes!("../tests/fixtures/card_instance_journal6_rewards_rules_v2.json"),
            &b,
            &c,
        );
        let cases = all.root["cases"].array().unwrap();
        let mut wrong_signs = 0;
        let mut rebound_viewers = 0;
        for case in cases {
            let viewer = case["viewer"].integer(1, 2).unwrap();
            let expected = case["expected_reward"].integer(-1, 1).unwrap();
            let wrong = if expected == 0 { 1 } else { -expected };
            for mutation in ["end", "transition", "both"] {
                let mut rows = case["rows"].clone();
                if mutation != "transition" {
                    rows[5]["terminal_reward"] = json::integer(wrong);
                }
                if mutation != "end" {
                    rows[4]["reward"] = json::integer(wrong);
                }
                let dir = directory();
                write_journal(&dir, &rows, case, &b, &c);
                let error = super::super::Journal::open_corrected(
                    &dir,
                    &format!("player-{viewer}.jsonl"),
                    b.clone(),
                    c.clone(),
                    16 * 1024 * 1024,
                    4 * 1024 * 1024,
                )
                .err()
                .expect("wrong reward must reject");
                assert_eq!(
                    error,
                    Error::contract(if mutation == "transition" {
                        "natural final accepted reward mismatch"
                    } else {
                        "natural reward perspective mismatch"
                    })
                );
                std::fs::remove_dir_all(dir).unwrap();
                wrong_signs += 1;
            }
            if expected == 0 {
                continue;
            } // A draw stays zero after changing viewer.
            let other = cases
                .iter()
                .find(|other| {
                    other["viewer"].as_i64() == Some((3 - viewer) as i64)
                        && other["rows"][5]["outcome"] == case["rows"][5]["outcome"]
                })
                .unwrap();
            let mut rows = other["rows"].clone();
            // The literal other-viewer record already has independently produced
            // and Python-reader-validated frame, bank, table and selection hashes.
            // Keep that fully rebound identity, but copy the original seat's reward.
            header6(&rows[0], &b, &c).unwrap();
            frame(&rows[2]["frame"], &b, &c).unwrap();
            rows[4]["reward"] = json::integer(expected);
            rows[5]["terminal_reward"] = json::integer(expected);
            let dir = directory();
            write_journal(&dir, &rows, other, &b, &c);
            let error = super::super::Journal::open_corrected(
                &dir,
                &format!("player-{}.jsonl", 3 - viewer),
                b.clone(),
                c.clone(),
                16 * 1024 * 1024,
                4 * 1024 * 1024,
            )
            .err()
            .expect("rebound viewer reward must reject");
            assert_eq!(
                error,
                Error::contract("natural reward perspective mismatch")
            );
            std::fs::remove_dir_all(dir).unwrap();
            rebound_viewers += 1;
        }
        assert_eq!((wrong_signs, rebound_viewers), (18, 4));
    }
}

pub(crate) fn action_value2(value: &Value, obs: Option<&Value>) -> Result<()> {
    let kind = value.require("type")?.string()?;
    let choice = if kind == "respond" {
        "defense"
    } else if kind == "buy_decision" {
        "buy_accept_reject"
    } else {
        "turn_action"
    };
    let schemas = typed_metadata(Some(choice));
    let descriptor = schemas
        .array()?
        .iter()
        .find(|v| v.get("type").and_then(Value::as_str) == Some(kind))
        .ok_or_else(|| Error::contract("unknown public v2 action"))?;
    let fields = descriptor.require("required_fields")?.object()?;
    let mut keys = vec!["type"];
    keys.extend(fields.keys().map(String::as_str));
    value.closed(&keys, &[])?;
    let mut references = Vec::new();
    for key in fields.keys() {
        let v = value.require(key)?;
        match key.as_str() {
            "ordered_card_refs" | "card_refs" => {
                for r in v.array()? {
                    references.push(card_ref(r)?);
                }
            }
            "card_ref" | "sold_card_ref" => references.push(card_ref(v)?),
            "target" => {
                validation::player(v)?;
            }
            "accept" => {
                v.boolean()?;
            }
            "hp" => {
                v.integer(i32::MIN as i128, i32::MAX as i128)?;
            }
            "mp" | "money" => {
                v.integer(0, u32::MAX as i128)?;
            }
            _ => return Err(Error::contract("unknown v2 action field")),
        }
    }
    require(
        references.len() <= 18
            && references.iter().collect::<BTreeSet<_>>().len() == references.len(),
        "repeated/over-capacity public references",
    )?;
    if let Some(obs) = obs {
        require(
            obs.require("required_choice")?.as_str() == Some(choice),
            "own Replay action differs from required choice",
        )?;
        let own = obs.require("own")?.require("cards")?.array()?;
        require(
            references.iter().all(|r| {
                own.iter()
                    .any(|v| v.get("card_ref").and_then(Value::as_str) == Some(*r))
            }),
            "own Replay action contains foreign reference",
        )?;
    }
    Ok(())
}
/// Complete participant-visible Replay2 shape and own accepted-action closure.
/// Replay is structural/control evidence only and never enters staged features.
pub(crate) fn reconcile_replay_for_profile<'a>(
    value: &'a Value,
    game: &str,
    viewer: i128,
    accepted: &[&Value],
    outcome: &Value,
    b: &Arc<Budget>,
    c: &Control,
    profile: WireProfile,
) -> Result<Option<&'a Value>> {
    c.check()?;
    value.closed(
        &[
            "game_id",
            "viewer_player_id",
            "requested_turn",
            "rule_set_version",
            "observation_schema_version",
            "public_event_schema_version",
            "card_ref_scheme",
            "episode_end",
            "turns",
        ],
        &[],
    )?;
    require(
        value.require("game_id")?.string()? == game
            && validation::player(value.require("viewer_player_id")?)? == viewer,
        "Replay2 game/viewer differs",
    )?;
    for (key, pin) in [
        ("rule_set_version", profile.rule_set()),
        ("observation_schema_version", "2"),
        ("public_event_schema_version", profile.event_schema()),
        ("card_ref_scheme", "aes256-card-ref-v1"),
    ] {
        is(value, key, pin)?;
    }
    if !value.require("requested_turn")?.is_null() {
        value
            .require("requested_turn")?
            .integer(0, u32::MAX as i128)?;
    }
    let closure = value.require("episode_end")?;
    if !closure.is_null() {
        match closure.require("type")?.string()? {
            "terminal" => {
                closure.closed(&["type", "result"], &[])?;
                validation::result(closure.require("result")?)?;
                require(
                    !closure.require("result")?.is_null(),
                    "Replay2 terminal outcome required",
                )?;
            }
            "aborted" => {
                closure.closed(&["type", "reason", "player", "decision"], &[])?;
                validation::text_value(closure.require("reason")?)?;
                if !closure.require("player")?.is_null() {
                    validation::player(closure.require("player")?)?;
                }
                let d = closure.require("decision")?;
                d.closed(&["turn", "decision_index", "invalid_attempt_count"], &[])?;
                for key in ["turn", "decision_index"] {
                    if !d.require(key)?.is_null() {
                        d.require(key)?.integer(0, u32::MAX as i128)?;
                    }
                }
                d.require("invalid_attempt_count")?
                    .integer(0, u32::MAX as i128)?;
            }
            _ => return Err(Error::contract("Replay2 closure type unsupported")),
        }
    }
    let recipe = crate::structured_tokens::Recipe::corrected(128, b, c)?;
    let mut index = 0usize;
    for turn in value.require("turns")?.array()? {
        c.check()?;
        turn.closed(&["turn", "active_player", "decisions"], &[])?;
        turn.require("turn")?.integer(0, u32::MAX as i128)?;
        validation::player(turn.require("active_player")?)?;
        for d in turn.require("decisions")?.array()? {
            c.check()?;
            d.closed(
                &[
                    "actor",
                    "observation",
                    "action_options",
                    "visible_action",
                    "visible_events",
                    "aborted",
                ],
                &[],
            )?;
            let actor = validation::player(d.require("actor")?)?;
            let aborted = d.require("aborted")?.boolean()?;
            let obs = d.require("observation")?;
            observation_for_profile(obs, Some(viewer), b, c, aborted, profile)?;
            require(
                obs.require("current_player")?.integer(1, 2)? == actor,
                "Replay2 actor/observation differs",
            )?;
            if !d.require("action_options")?.is_null() {
                let raw = json::canonical(d.require("action_options")?, b, c)?;
                let _scratch = json::admit(raw.len() + 65536, b)?;
                hints_for_profile(
                    &json::object([
                        (
                            "state_revision",
                            json::copy(obs.require("state_revision")?, c)?,
                        ),
                        ("viewer_player_id", json::integer(viewer)),
                        (
                            "action_options",
                            json::copy(d.require("action_options")?, c)?,
                        ),
                    ]),
                    obs,
                    c,
                    profile,
                )?;
            }
            let visible = d.require("visible_action")?;
            if !visible.is_null() {
                match visible.require("type")?.string()? {
                    "concealed" => visible.closed(&["type"], &[])?,
                    "action" => {
                        visible.closed(&["type", "action"], &[])?;
                        action_value2(
                            visible.require("action")?,
                            if actor == viewer { Some(obs) } else { None },
                        )?;
                    }
                    _ => return Err(Error::contract("Replay2 visible action type unsupported")),
                }
            }
            for e in d.require("visible_events")?.array()? {
                event_for_profile(e, &recipe, b, c, profile, Some(viewer))?;
            }
            if actor == viewer {
                if aborted {
                    require(
                        visible.is_null(),
                        "aborted Replay2 action cannot be accepted",
                    )?;
                } else if !visible.is_null() {
                    require(
                        visible.require("type")?.string()? == "action"
                            && index < accepted.len()
                            && visible.require("action")? == accepted[index],
                        "own accepted Replay2 action differs",
                    )?;
                    index += 1;
                }
            }
        }
    }
    require(
        index == accepted.len(),
        "accepted action count/Replay2 differs",
    )?;
    if !outcome.is_null() {
        require(
            !closure.is_null()
                && closure.require("type")?.string()? == "terminal"
                && closure.require("result")? == outcome,
            "Replay2 terminal outcome differs",
        )?;
    }
    Ok(if closure.is_null() {
        None
    } else {
        Some(closure)
    })
}
