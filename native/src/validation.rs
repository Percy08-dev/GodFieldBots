//! Closed participant DTO checks, pinned to Simulator ffcffa09 / catalog v2.
//!
//! This module borrows the owned JSON document. It never rebuilds a participant
//! snapshot or consults another participant, engine state, or rejection details.
use crate::budget::{BudgetVec, Control};
use crate::error::{Error, Result};
use crate::json::Value;

fn poll(control: Option<&Control>) -> Result<()> {
    if let Some(control) = control {
        control.check()?;
    }
    Ok(())
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Key<'a> {
    Number(i128),
    Text(&'a [u8]),
}
fn key(value: &Value) -> Result<Key<'_>> {
    match value {
        Value::String(text) => Ok(Key::Text(text.bytes())),
        _ => Ok(Key::Number(value.integer(0, u64::MAX as i128)?)),
    }
}
fn indexed<'a>(
    array: &'a Value,
    control: Option<&Control>,
    extract: impl Fn(&'a Value) -> Result<Key<'a>>,
) -> Result<BudgetVec<Key<'a>>> {
    let Value::Array(entries) = array else {
        return fail("array required");
    };
    let mut keys = BudgetVec::with_capacity(entries.budget(), entries.len())?;
    for entry in entries.iter() {
        poll(control)?;
        keys.push(extract(entry)?)?;
    }
    sort_keys(&mut keys, control)?;
    Ok(keys)
}
fn sort_keys(keys: &mut [Key<'_>], control: Option<&Control>) -> Result<()> {
    sort_controlled(keys, control, Ord::cmp)
}
pub(crate) fn sort_controlled<T>(
    keys: &mut [T],
    control: Option<&Control>,
    compare: impl Fn(&T, &T) -> std::cmp::Ordering,
) -> Result<()> {
    // In-place heapsort permits a fallible deadline/cancellation poll inside the
    // sorting work, while std comparators cannot return a cancellation error.
    fn down<T>(
        keys: &mut [T],
        mut root: usize,
        end: usize,
        control: Option<&Control>,
        compare: &impl Fn(&T, &T) -> std::cmp::Ordering,
    ) -> Result<()> {
        loop {
            poll(control)?;
            let Some(mut child) = root
                .checked_mul(2)
                .and_then(|n| n.checked_add(1))
                .filter(|n| *n < end)
            else {
                break;
            };
            if child + 1 < end && compare(&keys[child], &keys[child + 1]).is_lt() {
                child += 1;
            }
            if !compare(&keys[root], &keys[child]).is_lt() {
                break;
            }
            keys.swap(root, child);
            root = child;
        }
        Ok(())
    }
    for root in (0..keys.len() / 2).rev() {
        down(keys, root, keys.len(), control, &compare)?;
    }
    for end in (1..keys.len()).rev() {
        poll(control)?;
        keys.swap(0, end);
        down(keys, 0, end, control, &compare)?;
    }
    Ok(())
}
fn duplicated(keys: &[Key<'_>], control: Option<&Control>) -> Result<bool> {
    for pair in keys.windows(2) {
        poll(control)?;
        if pair[0] == pair[1] {
            return Ok(true);
        }
    }
    Ok(false)
}
/// Catalog mapping keys are the perceived references. Their inner definition IDs
/// need not agree: the historical validator intentionally accepts that shape.
pub fn catalog_entry<'a>(catalog: &'a Value, reference: &Value) -> Result<&'a Value> {
    let Value::String(reference) = reference else {
        return fail("string reference required");
    };
    catalog
        .as_object()?
        .iter()
        .find(|(key, _)| key.bytes() == reference.bytes())
        .map(|(_, value)| value)
        .ok_or_else(|| Error::contract("unknown own-visible definition"))
}

const U64: i128 = u64::MAX as i128;
const U32: i128 = u32::MAX as i128;
const I32_MIN: i128 = i32::MIN as i128;
const I32_MAX: i128 = i32::MAX as i128;
pub const KINDS: &[&str] = &[
    "weapons",
    "armor",
    "miracles",
    "sundries",
    "trade",
    "guardians",
    "phenomena",
    "devils",
];
pub const ATTRIBUTES: &[&str] = &["fire", "water", "wood", "stone", "light", "darkness"];
pub const NUMERIC: &[&str] = &[
    "attack",
    "defense",
    "ability_value",
    "price",
    "mp_cost",
    "hit_rate",
];

fn fail<T>(message: &str) -> Result<T> {
    Err(Error::contract(message))
}
fn is(value: &Value, expected: &str) -> bool {
    value.as_str().is_ok_and(|v| v == expected)
}
fn member(value: &Value, expected: &[&str]) -> bool {
    value.as_str().is_ok_and(|v| expected.contains(&v))
}
pub fn text_value(value: &Value) -> Result<()> {
    match value {
        Value::String(text) if !text.bytes().is_empty() => Ok(()),
        _ => fail("expected nonempty string"),
    }
}
pub fn player(value: &Value) -> Result<i128> {
    value.integer(1, 2)
}
pub fn result(value: &Value) -> Result<()> {
    if value.is_null() || is(value, "draw") {
        return Ok(());
    }
    value.closed(&["win"], &[])?;
    let win = value.require("win")?;
    win.closed(&["winner"], &[])?;
    player(win.require("winner")?)?;
    Ok(())
}
fn status(value: &Value) -> Result<()> {
    value.closed(&["disease", "fog", "flash", "dream", "dark_cloud"], &[])?;
    let disease = value.require("disease")?;
    if !disease.is_null() && !member(disease, &["cold", "fever", "hell", "heaven"]) {
        return fail("unknown disease");
    }
    for key in ["fog", "flash", "dream", "dark_cloud"] {
        value.require(key)?.as_bool()?;
    }
    Ok(())
}

// Exact generated typed metadata, checked structurally instead of allocating a
// second JSON tree. In particular integer constants are not booleans or floats.
fn string_list(value: &Value, expected: &[&str]) -> Result<()> {
    let values = value.as_array()?;
    if values.len() != expected.len() {
        return fail("typed metadata mismatch");
    }
    for (item, expected) in values.iter().zip(expected) {
        if !is(item, expected) {
            return fail("typed metadata mismatch");
        }
    }
    Ok(())
}
fn string_schema(value: &Value, expected: &str) -> Result<()> {
    value.closed(&["type", "const"], &[])?;
    if !is(value.require("type")?, "string") || !is(value.require("const")?, expected) {
        return fail("typed metadata mismatch");
    }
    Ok(())
}
fn integer_schema(value: &Value, low: i128, high: i128) -> Result<()> {
    value.closed(&["type", "minimum", "maximum"], &[])?;
    if !is(value.require("type")?, "integer")
        || value.require("minimum")?.integer(low, low)? != low
        || value.require("maximum")?.integer(high, high)? != high
    {
        return fail("typed metadata mismatch");
    }
    Ok(())
}
fn instance_schema(value: &Value) -> Result<()> {
    value.closed(&["type", "minimum", "maximum", "sources"], &[])?;
    if !is(value.require("type")?, "integer") {
        return fail("typed metadata mismatch");
    }
    value.require("minimum")?.integer(0, 0)?;
    value.require("maximum")?.integer(U64, U64)?;
    string_list(value.require("sources")?, &["own.hand[].instance_id"])
}
fn player_schema(value: &Value) -> Result<()> {
    value.closed(&["type", "enum", "sources"], &[])?;
    if !is(value.require("type")?, "integer") {
        return fail("typed metadata mismatch");
    }
    let entries = value.require("enum")?.as_array()?;
    if entries.len() != 2 {
        return fail("typed metadata mismatch");
    }
    entries[0].integer(1, 1)?;
    entries[1].integer(2, 2)?;
    string_list(
        value.require("sources")?,
        &["own.player_id", "opponent.player_id"],
    )
}
fn element_schema(value: &Value) -> Result<()> {
    value.closed(&["one_of"], &[])?;
    let entries = value.require("one_of")?.as_array()?;
    if entries.len() != 2 {
        return fail("typed metadata mismatch");
    }
    for (i, entry) in entries.iter().enumerate() {
        entry.closed(
            &[
                "type",
                "required_fields",
                "optional_fields",
                "additional_fields",
            ],
            &[],
        )?;
        if !is(entry.require("type")?, "object") || entry.require("additional_fields")?.as_bool()? {
            return fail("typed metadata mismatch");
        }
        entry.require("optional_fields")?.closed(&[], &[])?;
        let required = entry.require("required_fields")?;
        if i == 0 {
            required.closed(&["source", "instance_id"], &[])?;
            string_schema(required.require("source")?, "hand_artifact")?;
            instance_schema(required.require("instance_id")?)?;
        } else {
            required.closed(&["source", "definition_id"], &[])?;
            string_schema(required.require("source")?, "learned_miracle")?;
            let definition = required.require("definition_id")?;
            definition.closed(&["type", "sources"], &[])?;
            if !is(definition.require("type")?, "string") {
                return fail("typed metadata mismatch");
            }
            string_list(definition.require("sources")?, &["own.learned_miracles[]"])?;
        }
    }
    Ok(())
}
fn array_schema(value: &Value, typed: bool) -> Result<()> {
    value.closed(&["type", "items", "ordered"], &[])?;
    if !is(value.require("type")?, "array") || !value.require("ordered")?.as_bool()? {
        return fail("typed metadata mismatch");
    }
    if typed {
        element_schema(value.require("items")?)
    } else {
        instance_schema(value.require("items")?)
    }
}
fn descriptors(value: &Value, choice: Option<&str>, can_act: bool) -> Result<bool> {
    let tags: &[&str] = if !can_act {
        &[]
    } else {
        match choice {
            Some("turn_action") => &[
                "pray",
                "play_sequence",
                "discard",
                "sacrifice",
                "exchange",
                "sell",
                "buy",
            ],
            Some("defense") => &["respond", "respond_sequence"],
            Some("buy_accept_reject") => &["buy_decision"],
            _ => &[],
        }
    };
    let entries = value.as_array()?;
    if entries.len() != tags.len() {
        return fail("typed metadata mismatch");
    }
    for (entry, tag) in entries.iter().zip(tags) {
        entry.closed(
            &[
                "type",
                "required_fields",
                "optional_fields",
                "additional_fields",
            ],
            &[],
        )?;
        if !is(entry.require("type")?, tag) || entry.require("additional_fields")?.as_bool()? {
            return fail("typed metadata mismatch");
        }
        let required = entry.require("required_fields")?;
        let optional = entry.require("optional_fields")?;
        if *tag == "play_sequence" {
            optional.closed(&["target"], &[])?;
            let target = optional.require("target")?;
            target.closed(&["one_of"], &[])?;
            let alternatives = target.require("one_of")?.as_array()?;
            if alternatives.len() != 2 {
                return fail("typed metadata mismatch");
            }
            player_schema(&alternatives[0])?;
            alternatives[1].closed(&["type"], &[])?;
            if !is(alternatives[1].require("type")?, "null") {
                return fail("typed metadata mismatch");
            }
        } else {
            optional.closed(&[], &[])?;
        }
        match *tag {
            "pray" => required.closed(&[], &[])?,
            "play_sequence" => {
                required.closed(&["ordered_elements"], &[])?;
                array_schema(required.require("ordered_elements")?, true)?;
            }
            "respond" | "respond_sequence" => {
                required.closed(&["ordered_defenses"], &[])?;
                array_schema(
                    required.require("ordered_defenses")?,
                    *tag == "respond_sequence",
                )?;
            }
            "discard" | "sacrifice" => {
                required.closed(&["instance_ids"], &[])?;
                array_schema(required.require("instance_ids")?, false)?;
            }
            "exchange" => {
                required.closed(&["artifact", "hp", "mp", "money"], &[])?;
                instance_schema(required.require("artifact")?)?;
                integer_schema(required.require("hp")?, I32_MIN, I32_MAX)?;
                for key in ["mp", "money"] {
                    integer_schema(required.require(key)?, 0, U32)?;
                }
            }
            "sell" | "buy" => {
                required.closed(
                    if *tag == "sell" {
                        &["artifact", "sold_artifact", "target"]
                    } else {
                        &["artifact", "target"]
                    },
                    &[],
                )?;
                instance_schema(required.require("artifact")?)?;
                if *tag == "sell" {
                    instance_schema(required.require("sold_artifact")?)?;
                }
                player_schema(required.require("target")?)?;
            }
            "buy_decision" => {
                required.closed(&["accept"], &[])?;
                let accept = required.require("accept")?;
                accept.closed(&["type"], &[])?;
                if !is(accept.require("type")?, "boolean") {
                    return fail("typed metadata mismatch");
                }
            }
            _ => unreachable!(),
        }
    }
    Ok(!tags.is_empty())
}

pub fn observation(value: &Value, viewer: Option<i128>) -> Result<()> {
    observation_impl(value, viewer, None)
}
pub fn observation_controlled(
    value: &Value,
    viewer: Option<i128>,
    control: &Control,
) -> Result<()> {
    observation_impl(value, viewer, Some(control))
}
fn observation_impl(value: &Value, viewer: Option<i128>, control: Option<&Control>) -> Result<()> {
    poll(control)?;
    value.closed(
        &[
            "observation_schema_version",
            "action_schema_version",
            "actionable",
            "typed_action_schema",
            "state_revision",
            "viewer_player_id",
            "turn",
            "turns_until_end_of_world",
            "first_player",
            "active_player",
            "current_player",
            "phase",
            "required_choice",
            "end_times",
            "own",
            "opponent",
            "result",
        ],
        &[],
    )?;
    if !is(value.require("observation_schema_version")?, "1") {
        return fail("unsupported observation/action version");
    }
    value.require("action_schema_version")?.integer(1, 1)?;
    let actor = player(value.require("viewer_player_id")?)?;
    if viewer.is_some_and(|v| v != actor) {
        return fail("viewer mismatch");
    }
    for key in ["first_player", "active_player"] {
        player(value.require(key)?)?;
    }
    let current = value.require("current_player")?;
    if !current.is_null() {
        player(current)?;
    }
    value.require("state_revision")?.integer(0, U64)?;
    for key in ["turn", "turns_until_end_of_world"] {
        value.require(key)?.integer(0, U32)?;
    }
    value.require("end_times")?.as_bool()?;
    let phase = value.require("phase")?.as_str()?;
    if ![
        "awaiting_action",
        "awaiting_response",
        "resolving",
        "terminal",
    ]
    .contains(&phase)
    {
        return fail("unknown phase");
    }
    let choice = value.require("required_choice")?;
    if !choice.is_null()
        && !member(
            choice,
            &[
                "turn_action",
                "defense",
                "buy_accept_reject",
                "additional_choice",
            ],
        )
    {
        return fail("unknown choice");
    }
    let own = value.require("own")?;
    let opponent = value.require("opponent")?;
    own.closed(
        &[
            "player_id",
            "hp",
            "mp",
            "money",
            "hand",
            "learned_miracles",
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
    if player(own.require("player_id")?)? != actor
        || player(opponent.require("player_id")?)? != 3 - actor
    {
        return fail("participant mismatch");
    }
    for participant in [own, opponent] {
        participant.require("ascended")?.as_bool()?;
        for key in ["hp", "mp", "money"] {
            if let Some(item) = participant.get(key) {
                item.integer(
                    if key == "hp" { I32_MIN } else { 0 },
                    if key == "hp" { I32_MAX } else { U32 },
                )?;
            }
        }
        if let Some(item) = participant.get("status") {
            status(item)?;
        }
        if let Some(item) = participant.get("guardian") {
            if !item.is_null() {
                text_value(item)?;
            }
        }
        for item in participant.require("learned_miracles")?.as_array()? {
            poll(control)?;
            text_value(item)?;
        }
    }
    let hand = own.require("hand")?.as_array()?;
    for artifact in hand {
        poll(control)?;
        artifact.closed(&["instance_id", "definition_id"], &[])?;
        artifact.require("instance_id")?.integer(0, U64)?;
        text_value(artifact.require("definition_id")?)?;
    }
    let instances = indexed(own.require("hand")?, control, |artifact| {
        key(artifact.require("instance_id")?)
    })?;
    if duplicated(&instances, control)? {
        return fail("duplicate hand instance");
    }
    result(value.require("result")?)?;
    let input_phase = ["awaiting_action", "awaiting_response"].contains(&phase)
        || phase == "resolving" && is(choice, "buy_accept_reject");
    let can_act = current.integer(1, 2).is_ok_and(|v| v == actor)
        && value.require("result")?.is_null()
        && input_phase;
    if can_act && is(choice, "additional_choice") {
        return fail("additional_choice has no v1 action contract");
    }
    let actionable = descriptors(
        value.require("typed_action_schema")?,
        choice.as_str().ok(),
        can_act,
    )?;
    if value.require("actionable")?.as_bool()? != actionable {
        return fail("typed metadata mismatch");
    }
    Ok(())
}

pub fn validate_definition(value: &Value) -> Result<()> {
    value.closed(
        &["definition_id", "display_name", "kind"],
        &[
            "attack",
            "defense",
            "ability_value",
            "attribute",
            "price",
            "mp_cost",
            "additional_attack",
            "hit_rate",
            "ability",
            "effect_text",
            "curse",
            "guardian",
        ],
    )?;
    for key in ["definition_id", "display_name"] {
        text_value(value.require(key)?)?;
    }
    if !member(value.require("kind")?, KINDS) {
        return fail("unknown catalog kind");
    }
    for key in NUMERIC {
        if let Some(item) = value.get(key) {
            let unsigned = ["price", "mp_cost", "hit_rate"].contains(key);
            item.integer(
                if unsigned { 0 } else { I32_MIN },
                if *key == "hit_rate" {
                    100
                } else if unsigned {
                    U32
                } else {
                    I32_MAX
                },
            )?;
        }
    }
    if let Some(item) = value.get("attribute") {
        if !member(item, ATTRIBUTES) {
            return fail("unknown attribute");
        }
    }
    if let Some(item) = value.get("additional_attack") {
        item.as_bool()?;
    }
    for key in ["ability", "effect_text"] {
        if let Some(item) = value.get(key) {
            text_value(item)?;
        }
    }
    for key in ["curse", "guardian"] {
        if let Some(item) = value.get(key) {
            item.closed(&["id", "name", "description"], &[])?;
            for (_, field) in item.as_object()? {
                text_value(field)?;
            }
        }
    }
    Ok(())
}

pub fn actor_action_options(value: &Value, obs: &Value, raw: bool) -> Result<()> {
    observation(obs, None)?;
    actor_action_options_impl(value, obs, raw, None)
}
fn actor_action_options_impl(
    value: &Value,
    obs: &Value,
    raw: bool,
    control: Option<&Control>,
) -> Result<()> {
    poll(control)?;
    // Caller already validated Observation; this does not duplicate its walk.
    if !obs.require("actionable")?.as_bool()?
        || obs.require("current_player")? != obs.require("viewer_player_id")?
    {
        return fail("ActionOptions require current actionable actor");
    }
    value.closed(
        if raw {
            &["state_revision", "action_options"]
        } else {
            &["state_revision", "action_options", "viewer_player_id"]
        },
        &[],
    )?;
    if value.require("state_revision")?.integer(0, U64)?
        != obs.require("state_revision")?.integer(0, U64)?
    {
        return fail("ActionOptions revision mismatch");
    }
    if !raw
        && player(value.require("viewer_player_id")?)? != player(obs.require("viewer_player_id")?)?
    {
        return fail("ActionOptions actor mismatch");
    }
    let body = value.require("action_options")?;
    body.closed(
        &[
            "usable_hand_elements",
            "usable_learned_miracles",
            "exchange_artifacts",
            "sell_artifacts",
            "buy_artifacts",
            "valid_targets",
            "can_pray",
            "can_discard",
            "can_sacrifice",
            "can_allow",
            "required_choice",
        ],
        &[],
    )?;
    if body.require("required_choice")? != obs.require("required_choice")? {
        return fail("ActionOptions choice mismatch");
    }
    let own = obs.require("own")?;
    let hand = indexed(own.require("hand")?, control, |card| {
        key(card.require("instance_id")?)
    })?;
    let learned = indexed(own.require("learned_miracles")?, control, key)?;
    for field in [
        "usable_hand_elements",
        "usable_learned_miracles",
        "exchange_artifacts",
        "sell_artifacts",
        "buy_artifacts",
        "valid_targets",
    ] {
        let entries = body.require(field)?;
        for entry in entries.as_array()? {
            poll(control)?;
            if field == "usable_learned_miracles" {
                text_value(entry)?;
            } else if field == "valid_targets" {
                player(entry)?;
            } else {
                entry.integer(0, U64)?;
            }
        }
        let references = indexed(entries, control, key)?;
        if duplicated(&references, control)? {
            return fail("duplicate or foreign ActionOptions reference");
        }
        for reference in references.iter() {
            poll(control)?;
            let allowed = if field == "valid_targets" {
                true
            } else if field == "usable_learned_miracles" {
                learned.binary_search(reference).is_ok()
            } else {
                hand.binary_search(reference).is_ok()
            };
            if !allowed {
                return fail("duplicate or foreign ActionOptions reference");
            }
        }
    }
    for key in ["can_pray", "can_discard", "can_sacrifice", "can_allow"] {
        body.require(key)?.as_bool()?;
    }
    Ok(())
}

pub fn validate_worker_input(value: &Value, use_action_options: Option<bool>) -> Result<()> {
    validate_worker_input_impl(value, use_action_options, None, false)
}
pub fn validate_worker_input_controlled(
    value: &Value,
    use_action_options: Option<bool>,
    control: &Control,
) -> Result<()> {
    validate_worker_input_impl(value, use_action_options, Some(control), false)
}
pub fn validate_journal_input_controlled(
    value: &Value,
    use_action_options: Option<bool>,
    control: &Control,
) -> Result<()> {
    validate_worker_input_impl(value, use_action_options, Some(control), true)
}
fn validate_worker_input_impl(
    value: &Value,
    use_action_options: Option<bool>,
    control: Option<&Control>,
    journal_coercion: bool,
) -> Result<()> {
    poll(control)?;
    value.closed(
        &[
            "observation",
            "events",
            "history_end",
            "typed_metadata",
            "catalog",
            "action_options",
        ],
        &[],
    )?;
    let obs = value.require("observation")?;
    observation_impl(obs, None, control)?;
    let events = value.require("events")?;
    if !worker_events(events)?.is_empty() {
        validate_events_impl(events, control)?;
    }
    let end = value.require("history_end")?;
    if !end.is_null() {
        text_value(end)?;
    }
    // Both trees must be canonical-equivalent; the Observation metadata itself
    // has just been checked against the exact closed generated schema.
    let metadata = value.require("typed_metadata")?;
    let expected = obs.require("typed_action_schema")?;
    let empty_iterable = match metadata {
        Value::String(text) => text.bytes().is_empty(),
        Value::Object(entries) => entries.is_empty(),
        _ => false,
    };
    if metadata != expected
        && !(journal_coercion && expected.as_array()?.is_empty() && empty_iterable)
    {
        return fail("worker payload boundary mismatch");
    }
    let catalog = value.require("catalog")?.as_object()?;
    let own = obs.require("own")?;
    let hand = own.require("hand")?.as_array()?;
    let learned = own.require("learned_miracles")?.as_array()?;
    let Value::Object(catalog_values) = value.require("catalog")? else {
        unreachable!()
    };
    let mut visible = BudgetVec::with_capacity(
        catalog_values.budget(),
        hand.len()
            .checked_add(learned.len())
            .ok_or_else(|| Error::resource("visibility capacity overflow"))?,
    )?;
    for card in hand {
        poll(control)?;
        visible.push(key(card.require("definition_id")?)?)?;
    }
    for entry in learned {
        poll(control)?;
        visible.push(key(entry)?)?;
    }
    sort_keys(&mut visible, control)?;
    let mut catalog_keys = BudgetVec::with_capacity(catalog_values.budget(), catalog.len())?;
    for (reference, definition) in catalog {
        poll(control)?;
        let reference = Key::Text(reference.bytes());
        if visible.binary_search(&reference).is_err() {
            return fail("worker payload boundary mismatch");
        }
        catalog_keys.push(reference)?;
        validate_definition(definition)?;
    }
    sort_keys(&mut catalog_keys, control)?;
    for reference in visible.iter() {
        poll(control)?;
        if catalog_keys.binary_search(reference).is_err() {
            return fail("unknown own-visible definition");
        }
    }
    let options = value.require("action_options")?;
    if use_action_options.is_some_and(|expected| expected == options.is_null()) {
        return fail("ActionOptions presence/mode mismatch");
    }
    if !options.is_null() {
        actor_action_options_impl(options, obs, false, control)?;
    }
    Ok(())
}

/// The worker's historical list(events) conversion accepts empty iterables.
/// Nonempty string/object iterables yield strings and fail the closed event DTO.
pub fn worker_events(value: &Value) -> Result<&[Value]> {
    match value {
        Value::Array(entries) => Ok(entries),
        Value::String(text) if text.bytes().is_empty() => Ok(&[]),
        Value::Object(entries) if entries.is_empty() => Ok(&[]),
        _ => fail("event array required"),
    }
}

pub fn validate_events(value: &Value) -> Result<()> {
    validate_events_impl(value, None)
}
pub fn validate_events_controlled(value: &Value, control: &Control) -> Result<()> {
    validate_events_impl(value, Some(control))
}
fn validate_events_impl(value: &Value, control: Option<&Control>) -> Result<()> {
    for event in value.as_array()? {
        poll(control)?;
        let tag = event.require("type")?.as_str()?;
        let fields: &[&str] = match tag {
            "turn_started" => &["type", "turn", "active_player"],
            "action_accepted" | "sun_amulet_activated" | "player_ascended" | "player_defeated" => {
                &["type", "player"]
            }
            "artifact_used" | "artifact_drawn" | "devil_appeared" | "artifact_discarded"
            | "guardian_set" | "guardian_left" | "guardian_acted" => {
                &["type", "player", "definition_id"]
            }
            "miracle_used" => &["type", "player", "definition_id", "mp_spent"],
            "attack_declared" => &["type", "attacker", "target", "attack"],
            "attack_missed" => &["type", "attacker", "target"],
            "damage_applied" | "healed" => &["type", "player", "amount"],
            "resource_changed" => &["type", "player", "hp_delta", "mp_delta", "money_delta"],
            "buy_offered" => &["type", "buyer", "seller", "definition_id", "price"],
            "sale_offered" => &["type", "seller", "target", "definition_id", "price"],
            "artifact_transferred" => &["type", "from", "to", "definition_id", "price"],
            "status_changed" => &["type", "player", "status", "active"],
            "phenomenon_invoked" => &["type", "definition_id"],
            "game_ended" => &["type", "result"],
            _ => return fail("unknown player event"),
        };
        event.closed(
            fields,
            match tag {
                "damage_applied" => &["remaining_hp"],
                "healed" => &["hp"],
                _ => &[],
            },
        )?;
        for (key, item) in event.as_object()? {
            match key.as_str()? {
                "player" | "attacker" | "target" | "active_player" | "buyer" | "seller"
                | "from" | "to" => {
                    player(item)?;
                }
                "turn" | "price" | "mp_spent" => {
                    item.integer(0, U32)?;
                }
                "attack" | "amount" | "remaining_hp" | "hp" | "hp_delta" | "mp_delta"
                | "money_delta" => {
                    item.integer(I32_MIN, I32_MAX)?;
                }
                "active" => {
                    item.as_bool()?;
                }
                "result" => result(item)?,
                _ => {
                    text_value(item)?;
                }
            }
        }
    }
    Ok(())
}

pub fn replay_closure<'a>(
    replay: &'a Value,
    game: &str,
    viewer: i128,
) -> Result<Option<&'a Value>> {
    if !is(replay.require("game_id")?, game)
        || replay.require("viewer_player_id")?.integer(1, 2)? != viewer
        || !is(replay.require("observation_schema_version")?, "1")
    {
        return fail("replay game/perspective/version/closure mismatch");
    }
    let value = replay.require("episode_end")?;
    if value.is_null() {
        return Ok(None);
    }
    match value.require("type")?.as_str()? {
        "terminal" => {
            value.closed(&["type", "result"], &[])?;
            result(value.require("result")?)?;
            if value.require("result")?.is_null() {
                return fail("terminal closure requires result");
            }
        }
        "aborted" => {
            value.closed(&["type", "reason", "player", "decision"], &[])?;
            text_value(value.require("reason")?)?;
            if !value.require("player")?.is_null() {
                player(value.require("player")?)?;
            }
            let decision = value.require("decision")?;
            decision.closed(&["turn", "decision_index", "invalid_attempt_count"], &[])?;
            for key in ["turn", "decision_index"] {
                let item = decision.require(key)?;
                if !item.is_null() {
                    item.integer(0, U32)?;
                }
            }
            decision.require("invalid_attempt_count")?.integer(0, U32)?;
        }
        _ => return fail("unknown episode closure type"),
    }
    Ok(Some(value))
}

fn equals_player(value: &Value, expected: i128) -> bool {
    match value {
        Value::Bool(value) => i128::from(*value) == expected,
        Value::Int(_) => value.integer(1, 2) == Ok(expected),
        Value::Float(value) => *value == expected as f64,
        _ => false,
    }
}
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Int(text) => text.bytes() != b"0",
        Value::Float(value) => *value != 0.,
        Value::String(text) => !text.bytes().is_empty(),
        Value::Array(values) => !values.is_empty(),
        Value::Object(values) => !values.is_empty(),
    }
}
fn replay_action_equal(left: &Value, right: &Value) -> Result<bool> {
    left.as_object()?;
    right.as_object()?;
    if left.get("type").is_some_and(|v| is(v, "play_sequence"))
        && right.get("type").is_some_and(|v| is(v, "play_sequence"))
    {
        // Replay emits null for the optional omitted target. No owned copies.
        let a = left.as_object()?;
        let b = right.as_object()?;
        for (key, value) in a {
            if key == "target" {
                continue;
            }
            if !right
                .get(key.as_str()?)
                .is_some_and(|other| value.eq_python(other))
            {
                return Ok(false);
            }
        }
        for (key, _) in b {
            if key != "target" && left.get(key.as_str()?).is_none() {
                return Ok(false);
            }
        }
        let target_a = left.get("target").filter(|v| !v.is_null());
        let target_b = right.get("target").filter(|v| !v.is_null());
        Ok(match (target_a, target_b) {
            (Some(a), Some(b)) => a.eq_python(b),
            (None, None) => true,
            _ => false,
        })
    } else {
        Ok(left.eq_python(right))
    }
}
pub fn reconcile_replay(
    replay: &Value,
    viewer: i128,
    accepted: &[&Value],
    outcome: &Value,
) -> Result<()> {
    reconcile_replay_impl(replay, viewer, accepted, outcome, None)
}
pub fn reconcile_replay_controlled(
    replay: &Value,
    viewer: i128,
    accepted: &[&Value],
    outcome: &Value,
    control: &Control,
) -> Result<()> {
    reconcile_replay_impl(replay, viewer, accepted, outcome, Some(control))
}
fn reconcile_replay_impl(
    replay: &Value,
    viewer: i128,
    accepted: &[&Value],
    outcome: &Value,
    control: Option<&Control>,
) -> Result<()> {
    poll(control)?;
    if !equals_player(replay.require("viewer_player_id")?, viewer)
        || !is(replay.require("observation_schema_version")?, "1")
    {
        return fail("replay perspective/version mismatch");
    }
    let mut index = 0;
    if let Some(turns) = replay.get("turns") {
        for turn in worker_events(turns)? {
            turn.as_object()?;
            poll(control)?;
            if let Some(decisions) = turn.get("decisions") {
                for decision in worker_events(decisions)? {
                    decision.as_object()?;
                    poll(control)?;
                    if decision
                        .get("actor")
                        .is_none_or(|v| !equals_player(v, viewer))
                    {
                        continue;
                    }
                    if let Some(visible) =
                        decision.get("visible_action").filter(|value| truthy(value))
                    {
                        visible.as_object()?;
                        if visible.get("type").is_some_and(|v| is(v, "action")) {
                            let action = visible.require("action")?;
                            if index >= accepted.len()
                                || !replay_action_equal(action, accepted[index])?
                            {
                                return fail("accepted actions/replay mismatch");
                            }
                            index += 1;
                        }
                    }
                }
            }
        }
    }
    if index != accepted.len() {
        return fail("accepted actions/replay mismatch");
    }
    if !outcome.is_null() {
        let end = replay.require("episode_end")?;
        end.closed(&["type", "result"], &[])?;
        if !is(end.require("type")?, "terminal") || !end.require("result")?.eq_python(outcome) {
            return fail("terminal outcome/replay mismatch");
        }
    }
    Ok(())
}
