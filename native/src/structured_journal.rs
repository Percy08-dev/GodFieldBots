//! Original Journal5 and raw candidate-columns validation, separate from Source
//! qualification and numerical export. No legality probes or model execution.
use crate::{
    budget::{Budget, Control, Lease},
    error::{Error, Result},
    json::{self, Value, ValueExt},
    structured_controls::{checksum, digest},
    validation,
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, sync::Arc};

// Pure corrected DTOs only. No public Python ABI or Source authority.
#[path = "card_instance_dto.rs"]
pub(crate) mod card_instance;

#[derive(Clone, Copy)]
enum ColumnContext {
    Engineering,
    Corrected,
    CorrectedComponent,
}

pub const KINDS: &[&str] = &[
    "pray",
    "play_sequence",
    "respond",
    "respond_sequence",
    "discard",
    "sacrifice",
    "exchange",
    "sell",
    "buy",
    "buy_decision",
];
pub const TARGETS: &[&str] = &["omitted", "null", "self", "opponent"];
pub struct Table {
    pub value: Value,
    _lease: Lease,
}
fn fail<T>(message: &str) -> Result<T> {
    Err(Error::contract(message))
}
fn n(value: &Value, key: &str, lo: i128, hi: i128) -> Result<usize> {
    Ok(value.require(key)?.integer(lo, hi)? as usize)
}
fn copy_fields(value: &Value, fields: &[&'static str], control: &Control) -> Result<Value> {
    Ok(json::object(
        fields
            .iter()
            .map(|&key| Ok((key, json::copy(value.require(key)?, control)?)))
            .collect::<Result<Vec<_>>>()?,
    ))
}
pub fn frame(frame: &Value, budget: &Arc<Budget>, control: &Control) -> Result<()> {
    control.check()?;
    frame.closed(
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
    let actor = frame.require("actor")?.integer(1, 2)?;
    validation::text_value(frame.require("episode")?)?;
    let revision = frame.require("revision")?.integer(0, u64::MAX as i128)?;
    let boundary = frame.require("boundary")?;
    boundary.closed(&["end", "prefix_hash"], &[])?;
    let end = n(boundary, "end", 0, i32::MAX as i128)?;
    let start = n(frame, "interval_start", 0, end as i128)?;
    if end - start > 65536 {
        return fail("structured frame event interval exceeds capacity");
    }
    checksum(boundary.require("prefix_hash")?)?;
    checksum(frame.require("interval_start_hash")?)?;
    checksum(frame.require("frame_hash")?)?;
    let payload = frame.require("payload")?;
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
    // The original parsed frame owns the raw document admission. Charge this
    // one call-local validation copy before creating it, then discard it.
    let encoded = json::canonical(payload, budget, control)?;
    let _scratch = json::admit(encoded.len(), budget)?;
    let mut input = json::copy(payload, control)?;
    input
        .as_object_mut()
        .unwrap()
        .insert("events".into(), json::array([]));
    validation::validate_worker_input_controlled(&input, Some(true), control)?;
    let obs = payload.require("observation")?;
    if obs.require("viewer_player_id")?.integer(1, 2)? != actor
        || obs.require("current_player")?.integer(1, 2)? != actor
        || !obs.require("actionable")?.boolean()?
        || obs
            .require("state_revision")?
            .integer(0, u64::MAX as i128)?
            != revision
    {
        return fail("structured frame actor/revision mismatch");
    }
    let own = obs.require("own")?;
    let hand = own.require("hand")?.array()?;
    let learned = own.require("learned_miracles")?.array()?;
    if hand.len() + learned.len() > 18
        || obs
            .require("opponent")?
            .require("learned_miracles")?
            .array()?
            .len()
            > 18
    {
        return fail("structured frame own/public learned capacity");
    }
    let mut physical = BTreeSet::new();
    for card in hand {
        control.check()?;
        if !physical.insert(digest(card.require("instance_id")?, budget, control)?) {
            return fail("duplicate hand physical reference");
        }
    }
    physical.clear();
    for name in learned {
        control.check()?;
        if !physical.insert(name.string()?.to_owned()) {
            return fail("duplicate learned physical reference");
        }
    }
    for (key, definition) in payload.require("catalog")?.object()? {
        control.check()?;
        if definition.require("definition_id")?.string() != Ok(key.as_str()) {
            return fail("perceived catalog key/definition mismatch");
        }
    }
    if own.require("status")?.require("fog")?.boolean()?
        && obs
            .require("opponent")?
            .object()?
            .keys()
            .any(|key| !["player_id", "learned_miracles", "ascended"].contains(&key.as_str()))
    {
        return fail("Fog opponent visibility mismatch");
    }
    let binding = copy_fields(
        frame,
        &[
            "actor",
            "episode",
            "revision",
            "interval_start",
            "interval_start_hash",
            "boundary",
            "payload",
        ],
        control,
    )?;
    if digest(&binding, budget, control)? != frame.require("frame_hash")?.string()? {
        return fail("structured frame hash mismatch");
    }
    Ok(())
}
fn references(frame: &Value, control: &Control) -> Result<Vec<Value>> {
    let own = frame
        .require("payload")?
        .require("observation")?
        .require("own")?;
    let mut refs = Vec::with_capacity(18);
    for card in own.require("hand")?.array()? {
        refs.push(json::object([
            ("source", json::text("hand_artifact")),
            (
                "instance_id",
                json::copy(card.require("instance_id")?, control)?,
            ),
        ]));
    }
    for name in own.require("learned_miracles")?.array()? {
        refs.push(json::object([
            ("source", json::text("learned_miracle")),
            ("definition_id", json::copy(name, control)?),
        ]));
    }
    Ok(refs)
}
fn action(
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
        .map(|&i| {
            refs.get(i as usize)
                .ok_or_else(|| Error::contract("candidate slot outside original hand"))
                .and_then(|v| json::copy(v, control))
        })
        .collect::<Result<Vec<_>>>()?;
    match kind {
        "play_sequence" | "respond_sequence" => {
            result.as_object_mut().unwrap().insert(
                if kind == "play_sequence" {
                    "ordered_elements"
                } else {
                    "ordered_defenses"
                }
                .into(),
                json::array(elements),
            );
        }
        "respond" | "discard" | "sacrifice" => {
            let mut ids = Vec::new();
            for element in elements {
                if element.require("source")?.string()? != "hand_artifact" {
                    return fail("candidate requires own hand reference");
                }
                ids.push(json::copy(element.require("instance_id")?, control)?);
            }
            result.as_object_mut().unwrap().insert(
                if kind == "respond" {
                    "ordered_defenses"
                } else {
                    "instance_ids"
                }
                .into(),
                json::array(ids),
            );
        }
        "exchange" | "buy" | "sell" => {
            if elements.len() != if kind == "sell" { 2 } else { 1 } {
                return fail("invalid trade carrier references");
            }
            for (i, element) in elements.iter().enumerate() {
                if element.require("source")?.string()? != "hand_artifact" {
                    return fail("trade requires hand references");
                }
                result.as_object_mut().unwrap().insert(
                    if i == 0 { "artifact" } else { "sold_artifact" }.into(),
                    json::copy(element.require("instance_id")?, control)?,
                );
            }
        }
        _ => {
            if !elements.is_empty() {
                return fail("unexpected candidate references");
            }
        }
    }
    if target != "omitted" {
        result.as_object_mut().unwrap().insert(
            "target".into(),
            if target == "null" {
                Value::Null
            } else {
                json::integer(if target == "self" {
                    frame.require("actor")?.integer(1, 2)?
                } else {
                    3 - frame.require("actor")?.integer(1, 2)?
                })
            },
        );
    }
    for (i, key) in ["hp", "mp", "money", "accept"].iter().enumerate() {
        if present[i] {
            if *key == "accept" && !(0..=1).contains(&numerics[i]) {
                return fail("noncanonical candidate acceptance");
            }
            result.as_object_mut().unwrap().insert(
                (*key).into(),
                if *key == "accept" {
                    Value::Bool(numerics[i] != 0)
                } else {
                    json::integer(numerics[i] as i128)
                },
            );
        }
    }
    let descriptors = frame
        .require("payload")?
        .require("typed_metadata")?
        .array()?;
    let descriptor = descriptors
        .iter()
        .find(|v| v.get("type").and_then(Value::as_str) == Some(kind))
        .ok_or_else(|| Error::contract("candidate kind outside typed metadata"))?;
    let required = descriptor.require("required_fields")?.object()?;
    let optional = descriptor.require("optional_fields")?.object()?;
    for key in required.keys() {
        if result.get(key).is_none() {
            return fail("candidate missing required field");
        }
    }
    for key in result.object()?.keys() {
        if key != "type" && !required.contains_key(key) && !optional.contains_key(key) {
            return fail("candidate field outside typed metadata");
        }
    }
    if required.contains_key("target") && result.get("target").is_some_and(Value::is_null) {
        return fail("required candidate target cannot be null");
    }
    for (key, lo, hi) in [
        ("hp", i32::MIN as i128, i32::MAX as i128),
        ("mp", 0, u32::MAX as i128),
        ("money", 0, u32::MAX as i128),
    ] {
        if let Some(value) = result.get(key) {
            value.integer(lo, hi)?;
        }
    }
    Ok(result)
}

pub fn columns(
    payload: &[u8],
    descriptor: &Value,
    frame: &Value,
    max_positions: usize,
    synthetic: bool,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<Table> {
    columns_context(
        payload,
        descriptor,
        frame,
        max_positions,
        synthetic,
        budget,
        control,
        ColumnContext::Engineering,
    )
}

fn columns_context(
    payload: &[u8],
    descriptor: &Value,
    frame: &Value,
    max_positions: usize,
    synthetic: bool,
    budget: &Arc<Budget>,
    control: &Control,
    context: ColumnContext,
) -> Result<Table> {
    control.check()?;
    let corrected = !matches!(context, ColumnContext::Engineering);
    descriptor.closed(
        &[
            "schema_version",
            "codec",
            "file",
            "count",
            "elements",
            "bytes",
            "sha256",
            "frame_hash",
            "table_hash",
            "columns",
            "action_contract",
            "purpose",
            "synthetic_capacity",
        ],
        if corrected { &["codebook_sha256"] } else { &[] },
    )?;
    let schema = if corrected { 2 } else { 1 };
    if descriptor
        .require("schema_version")?
        .integer(schema, schema)?
        != schema
        || descriptor.require("codec")?.string()?
            != if corrected {
                "candidate-columns-v2"
            } else {
                "candidate-columns-v1"
            }
        || descriptor.require("purpose")?.string()?
            != if matches!(context, ColumnContext::Corrected) {
                "qualified_game"
            } else {
                "component"
            }
        || descriptor.require("synthetic_capacity")?.boolean()? != synthetic
    {
        return fail("candidate column recipe/classification mismatch");
    }
    crate::structured_controls::basename(descriptor.require("file")?.string()?)?;
    if !descriptor.require("file")?.string()?.as_bytes()[0].is_ascii_alphanumeric() {
        return fail("candidate column basename must begin with ASCII alphanumeric");
    }
    if corrected {
        if synthetic {
            return fail("codec2 diagnostic requires real18 references");
        }
        card_instance::column_binding(descriptor, budget, control)?;
    } else {
        crate::structured_controls::contract(descriptor.require("action_contract")?)?;
    }
    for key in ["sha256", "frame_hash", "table_hash"] {
        checksum(descriptor.require(key)?)?;
    }
    if descriptor.require("frame_hash")? != frame.require("frame_hash")? {
        return fail("candidate column original frame mismatch");
    }
    let count = n(descriptor, "count", 1, 65536)?;
    let elements = n(descriptor, "elements", 0, (count * 128) as i128)?;
    if !(1..=128).contains(&max_positions) {
        return fail("candidate position capacity invalid");
    }
    let size = n(descriptor, "bytes", 1, 64 * 1024 * 1024)?;
    if payload.len() != size {
        return fail("candidate column byte size mismatch");
    }
    let mut payload_hash = Sha256::new();
    for block in payload.chunks(32768) {
        control.check()?;
        payload_hash.update(block);
    }
    control.check()?;
    if format!("{:x}", payload_hash.finalize()) != descriptor.require("sha256")?.string()? {
        return fail("candidate column byte size/hash mismatch");
    }
    let scalar_count = count
        .checked_mul(13)
        .and_then(|n| n.checked_add(elements))
        .ok_or_else(|| Error::resource("candidate ownership overflow"))?;
    let _lease = budget.reserve(
        scalar_count
            .checked_mul(1024)
            .and_then(|n| n.checked_add(65536))
            .ok_or_else(|| Error::resource("candidate ownership overflow"))?,
    )?;
    let layout = [
        ("kinds", "uint8", vec![count], count),
        ("targets", "uint8", vec![count], count),
        ("numerics", "int64", vec![count, 4], count * 32),
        ("presence", "uint8", vec![count, 4], count * 4),
        ("offsets", "int64", vec![count + 1], (count + 1) * 8),
        ("slots", "uint8", vec![elements], elements),
    ];
    let mut offset = 0;
    let mut parts = Vec::new();
    let declared = descriptor.require("columns")?.array()?;
    if declared.len() != 6 {
        return fail("candidate fixed six columns required");
    }
    for ((name, dtype, shape, length), value) in layout.iter().zip(declared) {
        control.check()?;
        let aligned = if *dtype == "int64" {
            (offset + 7) / 8 * 8
        } else {
            offset
        };
        let expected = json::object([
            ("name", json::text(name)),
            ("dtype", json::text(dtype)),
            (
                "shape",
                json::array(shape.iter().map(|&v| json::integer(v as i128))),
            ),
            ("offset", json::integer(aligned as i128)),
            ("bytes", json::integer(*length as i128)),
        ]);
        // canonical distinguishes bool/float metadata from integer shapes.
        if json::canonical(value, budget, control)?[..]
            != json::canonical(&expected, budget, control)?[..]
        {
            return fail("candidate column exact layout mismatch");
        }
        if aligned + length > size || payload[offset..aligned].iter().any(|&v| v != 0) {
            return fail("candidate padding/layout overflow");
        }
        parts.push(&payload[aligned..aligned + length]);
        offset = aligned + length;
    }
    if offset != size {
        return fail("candidate trailing bytes");
    }
    let read =
        |bytes: &[u8], i: usize| i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
    let mut offsets = Vec::with_capacity(count + 1);
    for i in 0..=count {
        offsets.push(read(parts[4], i));
    }
    if offsets[0] != 0 || offsets[count] != elements as i64 {
        return fail("candidate offsets endpoints mismatch");
    }
    let refs = if corrected {
        card_instance::references(frame, control)?
    } else {
        references(frame, control)?
    };
    let mut kinds = Vec::with_capacity(count);
    let mut targets = Vec::with_capacity(count);
    let mut numerics = Vec::with_capacity(count);
    let mut presence = Vec::with_capacity(count);
    for i in 0..count {
        control.check()?;
        let kind = *(if corrected {
            card_instance::KINDS
        } else {
            KINDS
        })
        .get(parts[0][i] as usize)
        .ok_or_else(|| Error::contract("unknown raw candidate kind"))?;
        let target = *TARGETS
            .get(parts[1][i] as usize)
            .ok_or_else(|| Error::contract("unknown raw relative target"))?;
        if offsets[i] < 0
            || offsets[i + 1] < offsets[i]
            || offsets[i + 1] > elements as i64
            || offsets[i + 1] - offsets[i] > max_positions as i64
        {
            return fail("candidate row offset/capacity mismatch");
        }
        let row = &parts[5][offsets[i] as usize..offsets[i + 1] as usize];
        if row.iter().any(|&v| v > 17)
            || (!synthetic
                && (row.len() > 18 || row.iter().collect::<BTreeSet<_>>().len() != row.len()))
        {
            return fail("candidate game physical references duplicate/overflow");
        }
        let mut values = [0; 4];
        let mut exists = [false; 4];
        for j in 0..4 {
            values[j] = read(parts[2], i * 4 + j);
            let bit = parts[3][i * 4 + j];
            if bit > 1
                || values[j] < i32::MIN as i64
                || values[j] > u32::MAX as i64
                || bit == 0 && values[j] != 0
            {
                return fail("candidate numeric/presence mismatch");
            }
            exists[j] = bit == 1;
        }
        if corrected {
            card_instance::action(kind, target, &values, &exists, row, frame, &refs, control)?;
        } else {
            action(kind, target, &values, &exists, row, frame, &refs, control)?;
        }
        kinds.push(json::text(kind));
        targets.push(json::text(target));
        numerics.push(json::array(
            values.iter().map(|&v| json::integer(v as i128)),
        ));
        presence.push(json::array(exists.iter().map(|&v| Value::Bool(v))));
    }
    let mut value = json::object([
        (
            "frame_hash",
            json::copy(frame.require("frame_hash")?, control)?,
        ),
        ("kinds", json::array(kinds)),
        ("targets", json::array(targets)),
        ("numerics", json::array(numerics)),
        ("presence", json::array(presence)),
        (
            "offsets",
            json::array(offsets.iter().map(|&v| json::integer(v as i128))),
        ),
        (
            "slots",
            json::array(parts[5].iter().map(|&v| json::integer(v as i128))),
        ),
        (
            "action_contract",
            json::copy(descriptor.require("action_contract")?, control)?,
        ),
        (
            "purpose",
            json::text(if matches!(context, ColumnContext::Corrected) {
                "qualified_game"
            } else {
                "component"
            }),
        ),
        ("synthetic_capacity", Value::Bool(synthetic)),
    ]);
    if digest(&value, budget, control)? != descriptor.require("table_hash")?.string()? {
        return fail("candidate canonical table hash mismatch");
    }
    value.as_object_mut().unwrap().insert(
        "table_hash".into(),
        json::copy(descriptor.require("table_hash")?, control)?,
    );
    Ok(Table { value, _lease })
}

fn selected_action(table: &Value, index: usize, frame: &Value, control: &Control) -> Result<Value> {
    selected_action_context(table, index, frame, control, false)
}
fn selected_action_context(
    table: &Value,
    index: usize,
    frame: &Value,
    control: &Control,
    corrected: bool,
) -> Result<Value> {
    let refs = if corrected {
        card_instance::references(frame, control)?
    } else {
        references(frame, control)?
    };
    let offsets = table.require("offsets")?.array()?;
    let start = offsets[index].integer(0, i64::MAX as i128)? as usize;
    let end = offsets[index + 1].integer(start as i128, i64::MAX as i128)? as usize;
    let slots = table.require("slots")?.array()?[start..end]
        .iter()
        .map(|v| v.integer(0, 17).map(|v| v as u8))
        .collect::<Result<Vec<_>>>()?;
    let mut values = [0; 4];
    let mut exists = [false; 4];
    for j in 0..4 {
        values[j] = table.require("numerics")?.array()?[index].array()?[j]
            .integer(i32::MIN as i128, u32::MAX as i128)? as i64;
        exists[j] = table.require("presence")?.array()?[index].array()?[j].boolean()?;
    }
    (if corrected {
        card_instance::action
    } else {
        action
    })(
        table.require("kinds")?.array()?[index].string()?,
        table.require("targets")?.array()?[index].string()?,
        &values,
        &exists,
        &slots,
        frame,
        &refs,
        control,
    )
}
fn finite(value: &Value, nonnegative: bool) -> Result<f64> {
    let number = value.number()?;
    if !number.is_finite() || nonnegative && number < 0.0 {
        return fail("finite bounded diagnostic number required");
    }
    Ok(number)
}
fn timings(value: &Value) -> Result<()> {
    for item in value.object()?.values() {
        finite(item, true)?;
    }
    Ok(())
}
fn numeric_costs(value: &Value) -> Result<()> {
    value.closed(
        &[
            "version",
            "context_width",
            "card_width",
            "actor_input_width",
            "actor_hidden_width",
            "position_capacity",
            "candidate_chunk",
            "tail_operations_per_candidate",
            "tail_working_bytes_per_candidate",
            "tail_tensor_bytes_per_candidate",
            "fixed_bytes",
            "max_preparation_bytes",
            "max_scoring_operations",
            "max_storage_bytes",
            "max_tensor_bytes",
        ],
        &[],
    )?;
    if value.require("version")?.string()? != "proposal-cost-v1" {
        return fail("unsupported proposal cost recipe");
    }
    let mut positive = 0;
    for key in [
        "context_width",
        "card_width",
        "actor_input_width",
        "actor_hidden_width",
    ] {
        positive += usize::from(n(value, key, 0, 1 << 20)? > 0);
    }
    if positive != 0 && positive != 4 {
        return fail("mixed numeric dimensions");
    }
    let mut tail = Vec::new();
    for key in [
        "tail_operations_per_candidate",
        "tail_working_bytes_per_candidate",
        "tail_tensor_bytes_per_candidate",
    ] {
        tail.push(n(value, key, 0, i64::MAX as i128)?);
    }
    if tail[1] < tail[2] || positive == 4 && tail.contains(&0) {
        return fail("incomplete numeric tail coefficients");
    }
    n(value, "position_capacity", 1, 128)?;
    n(value, "candidate_chunk", 1, 65536)?;
    n(value, "fixed_bytes", 0, 12i128 << 30)?;
    n(value, "max_preparation_bytes", 1, 12i128 << 30)?;
    n(value, "max_scoring_operations", 1, i64::MAX as i128)?;
    n(value, "max_storage_bytes", 1, 12i128 << 30)?;
    n(value, "max_tensor_bytes", 1, 64i128 << 20)?;
    Ok(())
}
fn rng_state(value: &Value) -> Result<()> {
    let state = value.array()?;
    if state.len() != 3 || state[0].integer(3, 3)? != 3 {
        return fail("closed MT19937 state required");
    }
    let words = state[1].array()?;
    if words.len() != 625 {
        return fail("closed MT19937 word count");
    }
    for value in &words[..624] {
        value.integer(0, u32::MAX as i128)?;
    }
    words[624].integer(0, 624)?;
    if !state[2].is_null() {
        if !state[2]
            .as_number()
            .is_some_and(|v| !json::integer_token(v))
        {
            return fail("Gaussian cache must be float or null");
        }
        finite(&state[2], false)?;
    }
    Ok(())
}
fn initial_rng(seed: u64, budget: &Arc<Budget>, control: &Control) -> Result<Value> {
    let hashed = digest(
        &json::object([
            ("recipe", json::text("structured-search-stream-v1")),
            ("sampling_seed", json::integer(seed as i128)),
        ]),
        budget,
        control,
    )?;
    let seed = u64::from_str_radix(&hashed[..16], 16)
        .map_err(|_| Error::contract("search seed digest"))?;
    Ok(initial_mt(seed))
}
// CPython3.14 GIL integer-seed INITIAL state only. No draws, search execution,
// game RNG, schedule RNG or source authorization is provided by this helper.
fn initial_mt(seed: u64) -> Value {
    let keys = if seed >> 32 == 0 {
        vec![seed as u32]
    } else {
        vec![seed as u32, (seed >> 32) as u32]
    };
    let mut mt = [0u32; 624];
    mt[0] = 19650218;
    for i in 1..624 {
        mt[i] = 1812433253u32
            .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
            .wrapping_add(i as u32);
    }
    let (mut i, mut j) = (1, 0);
    for _ in 0..624 {
        mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1664525))
            .wrapping_add(keys[j])
            .wrapping_add(j as u32);
        i += 1;
        j += 1;
        if i >= 624 {
            mt[0] = mt[623];
            i = 1;
        }
        if j >= keys.len() {
            j = 0;
        }
    }
    for _ in 0..623 {
        mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1566083941))
            .wrapping_sub(i as u32);
        i += 1;
        if i >= 624 {
            mt[0] = mt[623];
            i = 1;
        }
    }
    mt[0] = 0x80000000;
    json::array([
        json::integer(3),
        json::array(
            mt.into_iter()
                .map(|v| json::integer(v as i128))
                .chain([json::integer(624)]),
        ),
        Value::Null,
    ])
}
const OMISSIONS: &[&str] = &[
    "same-definition distinct normal-attack miracle combination exception",
    "mixed utility with weapon or utility and order",
    "pure Doll and CutCost armor attack-support placement/companions",
    "Tear/Herb explicit-opponent versus user-fixed effect mapping",
    "guardian-present or unknown-guardian use",
    "underfull soap/broom removal",
    "opponent broom without a permitted visible count",
    "per-definition omitted/null defaults",
    "normative target-document discrepancy",
    "runtime conformance and reconciled reflection opportunity contract",
];
fn report(
    value: &Value,
    header: &Value,
    table: &Value,
    expected_rng: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<()> {
    report_context(value, header, table, expected_rng, budget, control, false)
}
fn report_context(
    value: &Value,
    header: &Value,
    table: &Value,
    expected_rng: &Value,
    budget: &Arc<Budget>,
    control: &Control,
    corrected: bool,
) -> Result<()> {
    value.closed(
        &[
            "action_contract",
            "generator_version",
            "config_hash",
            "numeric_costs_hash",
            "table_hash",
            "numeric_costs",
            "precount",
            "attempts",
            "prunes",
            "discovered",
            "mandatory",
            "selected",
            "selected_elements",
            "channel_counts",
            "traversal_complete",
            "presentation_complete",
            "cutoff_reasons",
            "omissions",
            "rng_before",
            "rng_after",
            "count_ms",
            "search_ms",
            "selection_ms",
        ],
        if corrected { &["rule_signature"] } else { &[] },
    )?;
    if value.require("action_contract")? != header.require("action_contract")?
        || value.require("generator_version")? != header.require("generator_version")?
        || value.require("config_hash")? != header.require("action_config_hash")?
        || value.require("table_hash")? != table.require("table_hash")?
    {
        return fail("search report identity mismatch");
    }
    if corrected {
        if value.require("rule_signature")? != header.require("rule_signature")? {
            return fail("report rule signature mismatch");
        }
        let actions = header.require("actions")?;
        if n(value, "attempts", 0, i64::MAX as i128)?
            > n(actions, "max_expansions", 1, i32::MAX as i128)?
            || n(value, "discovered", 0, i64::MAX as i128)?
                > n(actions, "max_temporary_candidates", 1, i32::MAX as i128)?
        {
            return fail("report exceeds frozen action budgets");
        }
    }
    for key in ["config_hash", "numeric_costs_hash", "table_hash"] {
        checksum(value.require(key)?)?;
    }
    numeric_costs(value.require("numeric_costs")?)?;
    if value.require("numeric_costs")? != header.require("numeric_costs")?
        || digest(value.require("numeric_costs")?, budget, control)?
            != value.require("numeric_costs_hash")?.string()?
    {
        return fail("search report cost binding mismatch");
    }
    let precount = value.require("precount")?;
    precount.closed(
        &[
            "mode",
            "saturated",
            "count_states",
            "complete",
            "prefixes",
            "attempts",
            "excluded",
            "targets",
            "numeric_variants",
            "elements",
            "preparation_bytes",
            "scoring_operations",
            "storage_bytes",
            "full_enumeration_admitted",
        ],
        &[],
    )?;
    let mode = precount.require("mode")?.string()?;
    if !["exact", "upper_bound", "unknown"].contains(&mode) {
        return fail("unknown precount mode");
    }
    let saturated = precount.require("saturated")?.boolean()?;
    let full = precount.require("full_enumeration_admitted")?.boolean()?;
    if full && (mode != "exact" || saturated) {
        return fail("inconsistent precount admission");
    }
    for (key, v) in precount.object()? {
        if !["mode", "saturated", "full_enumeration_admitted"].contains(&key.as_str()) {
            v.integer(0, i64::MAX as i128)?;
        }
    }
    let attempts = n(value, "attempts", 0, i64::MAX as i128)?;
    let prunes = n(value, "prunes", 0, i64::MAX as i128)?;
    let discovered = n(value, "discovered", 0, i64::MAX as i128)?;
    let mandatory = n(value, "mandatory", 0, i64::MAX as i128)?;
    let selected = n(value, "selected", 1, i64::MAX as i128)?;
    let elements = n(value, "selected_elements", 0, i64::MAX as i128)?;
    if prunes > attempts
        || mandatory > selected
        || selected > discovered
        || selected != table.require("kinds")?.array()?.len()
        || elements != table.require("slots")?.array()?.len()
    {
        return fail("search report counts inconsistent");
    }
    let channels = value.require("channel_counts")?.array()?;
    if channels.len() != 7 {
        return fail("seven search channel assignments required");
    }
    let sum = channels.iter().try_fold(0i128, |sum, v| {
        Ok::<_, Error>(sum + v.integer(0, selected as i128)?)
    })?;
    if sum != (selected - mandatory) as i128 {
        return fail("search channel assignment count mismatch");
    }
    let traversal = value.require("traversal_complete")?.boolean()?;
    let presentation = value.require("presentation_complete")?.boolean()?;
    if presentation != (traversal && selected == discovered) {
        return fail("search completeness mismatch");
    }
    let mut reasons = BTreeSet::new();
    for reason in value.require("cutoff_reasons")?.array()? {
        let text = reason.string()?;
        if ![
            "count_time",
            "count_states",
            "search_time",
            "max_expansions",
            "max_temporary_candidates",
            "max_candidates",
            "numeric_costs",
            "max_sequence_length",
        ]
        .contains(&text)
            || !reasons.insert(text)
        {
            return fail("unknown/repeated cutoff reason");
        }
    }
    let omissions = value.require("omissions")?.array()?;
    let expected_omissions = if corrected {
        &["unlisted direct-use utility definitions"][..]
    } else {
        OMISSIONS
    };
    if omissions.len() != expected_omissions.len()
        || !omissions
            .iter()
            .zip(expected_omissions)
            .all(|(v, &s)| v.as_str() == Some(s))
    {
        return fail("search omission recipe mismatch");
    }
    for key in ["count_ms", "search_ms", "selection_ms"] {
        finite(value.require(key)?, true)?;
    }
    rng_state(value.require("rng_before")?)?;
    rng_state(value.require("rng_after")?)?;
    if json::canonical(value.require("rng_before")?, budget, control)?[..]
        != json::canonical(expected_rng, budget, control)?[..]
    {
        return fail("independent search RNG stream discontinuity");
    }
    Ok(())
}

fn uuid(value: &Value) -> Result<()> {
    let text = value.string()?;
    if text.len() != 36
        || !text.bytes().enumerate().all(|(i, c)| {
            if [8, 13, 18, 23].contains(&i) {
                c == b'-'
            } else {
                c.is_ascii_digit() || (b'a'..=b'f').contains(&c)
            }
        })
    {
        return fail("canonical command UUID required");
    }
    Ok(())
}
fn command(
    value: &Value,
    frame: &Value,
    selected: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<()> {
    value.closed(&["command_id", "expected_revision", "action"], &[])?;
    uuid(value.require("command_id")?)?;
    if value
        .require("expected_revision")?
        .integer(0, u64::MAX as i128)?
        != frame.require("revision")?.integer(0, u64::MAX as i128)?
        || json::canonical(value.require("action")?, budget, control)?[..]
            != json::canonical(selected, budget, control)?[..]
    {
        return fail("command not reconstructed from ORIGINAL frame");
    }
    Ok(())
}
fn acknowledgement(response: &Value, command: &Value) -> Result<()> {
    response.closed(
        &[
            "command_id",
            "previous_revision",
            "state_revision",
            "status",
            "result",
        ],
        &[],
    )?;
    if response.require("command_id")? != command.require("command_id")? {
        return fail("acknowledgement command mismatch");
    }
    let previous = response
        .require("previous_revision")?
        .integer(0, u64::MAX as i128)?;
    let current = response
        .require("state_revision")?
        .integer(0, u64::MAX as i128)?;
    if previous
        != command
            .require("expected_revision")?
            .integer(0, u64::MAX as i128)?
        || current != previous + 1
    {
        return fail("acknowledgement revision mismatch");
    }
    let status = response.require("status")?.string()?;
    if !["active", "terminal", "aborted"].contains(&status) {
        return fail("unknown positive ACK status");
    }
    validation::result(response.require("result")?)?;
    if (status == "terminal") == response.require("result")?.is_null() {
        return fail("ACK outcome/status mismatch");
    }
    Ok(())
}
/// Original role declaration checked by the Source composer, not a public
/// "allow historical" flag. It never authorizes Source or a checkpoint.
pub(crate) struct OriginalHeader<'a> {
    pub(crate) model: Option<&'a Value>,
    identity: &'a Value,
    corrected: bool,
}
impl<'a> OriginalHeader<'a> {
    pub(crate) fn new(
        model: Option<&'a Value>,
        identity: &'a Value,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<Self> {
        identity.closed(
            &[
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
            ],
            &[],
        )?;
        if let Some(model) = model {
            if model.require("model_id")?.string()? == "lightgbm" {
                crate::structured_controls::tree_model(model)?;
                let encoder = model.require("encoder")?;
                let window = encoder.require("aggregate_window")?.integer(1, 65536)?;
                if identity.require("model_version")?.string()? != "2"
                    || identity.require("model_id")?.string()? != "lightgbm"
                    || !identity.require("generation_schema")?.is_null()
                    || identity.require("aggregate_window")?.integer(1, 65536)? != window
                    || identity.require("token_signature")?.string()?
                        != crate::structured_controls::token_signature(
                            window, false, budget, control,
                        )?
                    || identity
                        .require("caps")?
                        .require("max_positions")?
                        .integer(1, 128)?
                        > encoder.require("max_positions")?.integer(1, 128)?
                {
                    return fail("recorded original nonPPO Tree2 role descriptor mismatch");
                }
                return Ok(Self {
                    model: Some(model),
                    identity,
                    corrected: false,
                });
            }
            crate::structured_controls::model(model, budget, control)?;
            let window = model
                .require("event")?
                .require("aggregate_window")?
                .integer(1, 65536)?;
            let old = model.require("event")?.get("aggregate_recipe").is_none();
            if identity.require("model_version")?.string()? != "2"
                || identity.require("model_id")? != model.require("model_id")?
                || identity.require("aggregate_window")?.integer(1, 65536)? != window
                || identity.require("token_signature")?.string()?
                    != crate::structured_controls::token_signature(window, old, budget, control)?
            {
                return fail("recorded original neural role descriptor mismatch");
            }
            let caps = identity.require("caps")?;
            for (cap, field) in [
                ("max_interval_events", "max_interval"),
                ("max_episode_events", "max_episode"),
            ] {
                if caps.require(cap)? != model.require("event")?.require(field)? {
                    return fail("original neural event caps differ");
                }
            }
            if caps.require("max_decisions")?
                != model.require("temporal")?.require("max_decisions")?
                || caps.require("max_positions")?.integer(1, 128)?
                    > model.require("max_positions")?.integer(1, 128)?
            {
                return fail("original neural history/position caps differ");
            }
        } else {
            if identity.require("model_version")?.string()? != "1"
                || !["random", "heuristic"].contains(&identity.require("model_id")?.string()?)
            {
                return fail("original baseline role descriptor required");
            }
            let window = identity.require("aggregate_window")?.integer(1, 65536)?;
            let sig = identity.require("token_signature")?.string()?;
            if sig != crate::structured_controls::token_signature(window, false, budget, control)?
                && sig
                    != crate::structured_controls::token_signature(window, true, budget, control)?
            {
                return fail("unknown original baseline token descriptor");
            }
        }
        Ok(Self {
            model,
            identity,
            corrected: false,
        })
    }
    pub(crate) fn new_corrected(
        model: Option<&'a Value>,
        identity: &'a Value,
        b: &Arc<Budget>,
        c: &Control,
    ) -> Result<Self> {
        let base = [
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
        ];
        identity.closed(&base, &[])?;
        crate::structured_controls::game_binding3(identity, b, c)?;
        if !identity.require("generation_schema")?.is_null() {
            return fail("J6 generation schema must remain null");
        }
        let window = n(identity, "aggregate_window", 1, 65536)?;
        if identity.require("token_signature")?.string()?
            != crate::structured_controls::card_instance_token_signature(window, b, c)?
        {
            return fail("corrected original token identity mismatch");
        }
        if let Some(model) = model {
            crate::structured_controls::model3(model, b, c)?;
            if identity.require("model_version")?.string()? != "3"
                || identity.require("model_id")? != model.require("model_id")?
                || window != n(model.require("event")?, "aggregate_window", 1, 65536)?
            {
                return fail("corrected original neural role differs");
            }
        } else if identity.require("model_version")?.string()? != "1"
            || !["random", "heuristic"].contains(&identity.require("model_id")?.string()?)
        {
            return fail("corrected original baseline role differs");
        }
        Ok(Self {
            model,
            identity,
            corrected: true,
        })
    }
    fn check(&self, header: &Value, _budget: &Arc<Budget>, _control: &Control) -> Result<()> {
        for (key, value) in self.identity.object()? {
            if header.require(key)? != value {
                return fail("original journal role declaration differs from frozen Source");
            }
        }
        Ok(())
    }
}
enum Destination<'a, 'm> {
    Diagnostic,
    CorrectedDiagnostic,
    Original {
        declaration: &'a OriginalHeader<'m>,
        stage: Option<(&'a mut crate::structured_staging::Stage, usize)>,
        tree_window: Option<usize>,
    },
}

fn header(
    value: &Value,
    budget: &Arc<Budget>,
    control: &Control,
    original: Option<&OriginalHeader<'_>>,
) -> Result<()> {
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
            "purpose",
            "synthetic_capacity",
        ],
        &[],
    )?;
    crate::structured_controls::contract(value.require("action_contract")?)?;
    for (key, expected) in [
        ("kind", "header"),
        ("feature_version", "participant-structured-v1"),
        ("input_recipe", "participant-structured-v1"),
        ("reward_version", "natural-terminal-v1"),
        ("policy_input_version", "player-atomic-options-v1"),
        ("end_projection_version", "simulator-closure-v1"),
        ("generator_version", "public-atomic-diverse-v4"),
        ("purpose", "component"),
    ] {
        if value.require(key)?.string() != Ok(expected) {
            return fail("journal5 semantic identity mismatch");
        }
    }
    value.require("schema_version")?.integer(5, 5)?;
    value.require("viewer")?.integer(1, 2)?;
    value
        .require("sampling_seed")?
        .integer(0, u64::MAX as i128)?;
    if value.require("synthetic_capacity")?.boolean()?
        || !value.require("use_action_options")?.boolean()?
    {
        return fail("journal5 requires nonsynthetic actor hints");
    }
    for key in ["episode_id", "policy", "run_id", "game_id", "bank_episode"] {
        validation::text_value(value.require(key)?)?;
    }
    checksum(value.require("action_config_hash")?)?;
    let catalog = value.require("catalog")?;
    catalog.closed(
        &[
            "catalog_schema_version",
            "catalog_version",
            "rule_set_version",
            "locale",
            "content_sha256",
            "definition_count",
        ],
        &[],
    )?;
    catalog.require("catalog_schema_version")?.integer(1, 1)?;
    catalog.require("definition_count")?.integer(296, 296)?;
    for (key, expected) in [
        ("catalog_version", "public-ja-v2"),
        ("rule_set_version", "true-match-1v1-v1"),
        ("locale", "ja"),
        (
            "content_sha256",
            "481ddfb0c068a1d2aaaf2dbcb6a5b4bd04396dfcc5c995e14e47b71222b6a51d",
        ),
    ] {
        if catalog.require(key)?.string() != Ok(expected) {
            return fail("journal5 catalog identity mismatch");
        }
    }
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
    let interval = n(caps, "max_interval_events", 1, 65536)?;
    n(
        caps,
        "max_episode_events",
        interval as i128,
        i32::MAX as i128,
    )?;
    if !caps.require("max_decisions")?.is_null() {
        n(caps, "max_decisions", 1, 1024)?;
    }
    n(caps, "max_positions", 1, 128)?;
    n(caps, "max_candidates", 1, 65536)?;
    numeric_costs(value.require("numeric_costs")?)?;
    let window = n(
        value,
        "aggregate_window",
        1,
        65536.min(n(caps, "max_episode_events", 1, i32::MAX as i128)? as i128),
    )?;
    let model = value.require("model_id")?.string()?;
    match value.require("model_version")?.string()? {
        "2" if model == "lightgbm" => {
            if !value.require("generation_schema")?.is_null() {
                return fail("Tree2 RAW role has no PPO generation schema");
            }
        }
        "2" => {
            if !["mlp", "gru", "transformer"].contains(&model) {
                return fail("journal5 neural model mismatch");
            }
            value.require("generation_schema")?.integer(2, 2)?;
        }
        "1" => {
            if !["random", "heuristic"].contains(&model)
                || !value.require("generation_schema")?.is_null()
                || original.is_none() && window != 128
            {
                return fail("baseline diagnostic requires corrected128 header");
            }
        }
        _ => return fail("unknown journal5 policy version"),
    }
    if let Some(declaration) = original {
        declaration.check(value, budget, control)?;
    } else if crate::structured_controls::token_signature(window as i128, false, budget, control)?
        != value.require("token_signature")?.string()?
    {
        return fail("journal5 active corrected token signature required");
    }
    Ok(())
}
struct Decision {
    document: usize,
    table: Option<Table>,
    frame_tokens: Option<crate::structured_tokens::OwnedTokens>,
    candidate_tokens: Option<crate::structured_tokens::OwnedTokens>,
    binding: Option<json::Document>,
    staged_frame: Option<Value>,
}
pub struct Journal {
    pub sources: crate::structured_session::Sources,
    data: ParsedJournal,
}
pub(crate) struct ParsedJournal {
    documents: crate::budget::BudgetVec<json::Document>,
    bank: crate::budget::BudgetVec<Value>,
    prefixes: crate::budget::BudgetVec<[u8; 32]>,
    decisions: crate::budget::BudgetVec<Decision>,
    accepted: crate::budget::BudgetVec<usize>,
    end: usize,
    max_json: usize,
    corrected: bool,
}
fn hash_bytes(text: &str) -> Result<[u8; 32]> {
    if text.len() != 64 {
        return fail("event prefix digest length");
    }
    let mut bytes = [0; 32];
    for i in 0..32 {
        bytes[i] = u8::from_str_radix(&text[2 * i..2 * i + 2], 16)
            .map_err(|_| Error::contract("invalid event prefix SHA"))?;
    }
    Ok(bytes)
}
fn hash_text(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
pub(crate) fn event_prefix_initial(
    actor: i128,
    episode: &Value,
    contract: Option<&Value>,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<[u8; 32]> {
    let mut binding = vec![
        json::text(if contract.is_some() {
            "participant-structured-v2"
        } else {
            "participant-structured-v1"
        }),
        json::text("event-bank"),
        json::integer(actor),
        json::copy(episode, control)?,
    ];
    if let Some(contract) = contract {
        binding.push(json::copy(contract, control)?);
    }
    hash_bytes(&digest(&json::array(binding), budget, control)?)
}
pub(crate) fn event_prefix_next(
    previous: &[u8; 32],
    event: &Value,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<[u8; 32]> {
    let encoded = json::canonical(event, budget, control)?;
    let mut hash = Sha256::new();
    hash.update(previous);
    hash.update(&*encoded);
    Ok(hash.finalize().into())
}
pub(crate) fn event_prefix_text(prefix: &[u8; 32]) -> String {
    hash_text(prefix)
}
impl Journal {
    pub(crate) fn open_corrected(
        directory: &std::path::Path,
        journal_file: &str,
        budget: Arc<Budget>,
        control: Control,
        max_json: usize,
        max_record: usize,
    ) -> Result<Self> {
        Self::open_destination(
            directory,
            journal_file,
            budget,
            control,
            max_json,
            max_record,
            Destination::CorrectedDiagnostic,
        )
    }
    pub fn open(
        directory: &std::path::Path,
        journal_file: &str,
        budget: Arc<Budget>,
        control: Control,
        max_json: usize,
        max_record: usize,
    ) -> Result<Self> {
        Self::open_destination(
            directory,
            journal_file,
            budget,
            control,
            max_json,
            max_record,
            Destination::Diagnostic,
        )
    }
    pub(crate) fn original_corrected(
        directory: &std::path::Path,
        journal_file: &str,
        budget: Arc<Budget>,
        control: Control,
        max_json: usize,
        max_record: usize,
        declaration: &OriginalHeader<'_>,
        stage: Option<(&mut crate::structured_staging::Stage, usize)>,
    ) -> Result<Self> {
        if !declaration.corrected || stage.is_some() && declaration.model.is_none() {
            return fail("corrected original entry requires its closed declaration");
        }
        Self::open_destination(
            directory,
            journal_file,
            budget,
            control,
            max_json,
            max_record,
            Destination::Original {
                declaration,
                stage,
                tree_window: None,
            },
        )
    }
    pub(crate) fn original(
        directory: &std::path::Path,
        journal_file: &str,
        budget: Arc<Budget>,
        control: Control,
        max_json: usize,
        max_record: usize,
        declaration: &OriginalHeader<'_>,
        stage: Option<(&mut crate::structured_staging::Stage, usize)>,
    ) -> Result<Self> {
        if declaration.corrected {
            return fail("old original Journal entry rejects corrected declarations");
        }
        if stage.is_some()
            && (declaration.model.is_none()
                || declaration.model.unwrap().require("model_id")?.string()? == "lightgbm"
                || declaration
                    .model
                    .unwrap()
                    .require("event")?
                    .get("aggregate_recipe")
                    .and_then(Value::as_str)
                    != Some("public-exact-v2"))
        {
            return fail("historical/baseline raw Source cannot produce active neural arrays");
        }
        Self::open_destination(
            directory,
            journal_file,
            budget,
            control,
            max_json,
            max_record,
            Destination::Original {
                declaration,
                stage,
                tree_window: None,
            },
        )
    }
    pub(crate) fn original_tree(
        directory: &std::path::Path,
        journal_file: &str,
        budget: Arc<Budget>,
        control: Control,
        max_json: usize,
        max_record: usize,
        declaration: &OriginalHeader<'_>,
        stage: Option<(&mut crate::structured_staging::Stage, usize)>,
        window: usize,
    ) -> Result<Self> {
        if let Some(model) = declaration.model {
            if model.require("model_id")?.string()? == "lightgbm"
                || model
                    .require("event")?
                    .require("aggregate_recipe")?
                    .string()?
                    != "public-exact-v2"
            {
                return fail("TreeDataset selected source must be current376 neural2");
            }
        } else {
            return fail("TreeDataset selected source requires its original neural header");
        }
        Self::open_destination(
            directory,
            journal_file,
            budget,
            control,
            max_json,
            max_record,
            Destination::Original {
                declaration,
                stage,
                tree_window: Some(window),
            },
        )
    }
    fn open_destination(
        directory: &std::path::Path,
        journal_file: &str,
        budget: Arc<Budget>,
        control: Control,
        max_json: usize,
        max_record: usize,
        destination: Destination<'_, '_>,
    ) -> Result<Self> {
        let mut sources = crate::structured_session::Sources::new(
            directory,
            budget.clone(),
            control.clone(),
            max_json,
        )?;
        let data = Self::parse_destination(
            &mut sources,
            journal_file,
            budget,
            control,
            max_json,
            max_record,
            destination,
        )?;
        Ok(Self { sources, data })
    }
    fn parse_destination(
        sources: &mut crate::structured_session::Sources,
        journal_file: &str,
        budget: Arc<Budget>,
        control: Control,
        max_json: usize,
        max_record: usize,
        mut destination: Destination<'_, '_>,
    ) -> Result<ParsedJournal> {
        let diagnostic = matches!(
            &destination,
            Destination::Diagnostic | Destination::CorrectedDiagnostic
        );
        let corrected = matches!(&destination, Destination::CorrectedDiagnostic)
            || matches!(&destination,Destination::Original{declaration,..} if declaration.corrected);
        crate::structured_controls::basename(journal_file)?;
        if max_record == 0 || max_record > max_json {
            return fail("journal5 record bound invalid");
        }
        let bytes = sources.bytes(&sources.root.join(journal_file), max_json)?;
        let mut documents = crate::budget::BudgetVec::new(&budget);
        for line in bytes.split_inclusive(|&v| v == b'\n') {
            control.check()?;
            if line.len() > max_record {
                return fail("journal5 record byte cap exceeded");
            }
            documents.push(json::parse(line, &budget, &control)?)?;
        }
        drop(bytes);
        if documents.is_empty() {
            return fail("journal5 header required");
        }
        if corrected {
            card_instance::header6(&documents[0].root, &budget, &control)?;
            if let Destination::Original { declaration, .. } = &destination {
                declaration.check(&documents[0].root, &budget, &control)?;
            }
        } else {
            header(
                &documents[0].root,
                &budget,
                &control,
                match &destination {
                    Destination::Diagnostic | Destination::CorrectedDiagnostic => None,
                    Destination::Original { declaration, .. } => Some(declaration),
                },
            )?;
        }
        let viewer = documents[0].root.require("viewer")?.integer(1, 2)?;
        if journal_file != format!("player-{viewer}.jsonl") {
            return fail("journal filename/viewer mismatch");
        }
        let initial = event_prefix_initial(
            viewer,
            documents[0].root.require("bank_episode")?,
            if corrected {
                Some(documents[0].root.require("action_contract")?)
            } else {
                None
            },
            &budget,
            &control,
        )?;
        let mut prefixes = crate::budget::BudgetVec::new(&budget);
        prefixes.push(initial)?;
        let seed = documents[0]
            .root
            .require("sampling_seed")?
            .integer(0, u64::MAX as i128)? as u64;
        let mut expected_rng = initial_rng(seed, &budget, &control)?;
        let recipe = if corrected
            || diagnostic
            || matches!(&destination, Destination::Original { stage: Some(_), .. })
        {
            Some((if corrected {
                crate::structured_tokens::Recipe::corrected
            } else {
                crate::structured_tokens::Recipe::new
            })(
                match &destination {
                    Destination::Original {
                        tree_window: Some(window),
                        ..
                    } => *window,
                    _ => n(&documents[0].root, "aggregate_window", 1, 65536)?,
                },
                &budget,
                &control,
            )?)
        } else {
            None
        };
        let _rng_metadata = budget.reserve(65536)?;
        let mut bank = crate::budget::BudgetVec::new(&budget);
        let mut decisions: crate::budget::BudgetVec<Decision> =
            crate::budget::BudgetVec::new(&budget);
        let mut accepted = crate::budget::BudgetVec::new(&budget);
        let (
            mut pending,
            mut last_accepted,
            mut awaiting_next,
            mut terminal,
            mut lifecycle,
            mut end,
        ) = (None, None, None, None, None, None);
        let mut delta_pending = false;
        for index in 1..documents.len() {
            control.check()?;
            if end.is_some() {
                return fail("records after journal5 end");
            }
            // Event values move into the single owned bank, while their
            // original document admission remains alive through close.
            if documents[index].root.require("kind")?.string()? == "events" {
                let row = &documents[index].root;
                row.closed(&["kind", "delta", "boundary"], &[])?;
                if pending.is_some() || delta_pending || lifecycle.is_some() {
                    return fail("journal5 event delta ordering mismatch");
                }
                let caps = documents[0].root.require("caps")?;
                if !caps.require("max_decisions")?.is_null()
                    && decisions.len() + 1 > n(caps, "max_decisions", 1, 1024)?
                {
                    return fail("journal5 decision cap exceeded");
                }
                let delta = row.require("delta")?;
                delta.closed(
                    &["actor", "episode", "start", "previous_hash", "events"],
                    &[],
                )?;
                if delta.require("actor")?.integer(1, 2)? != viewer
                    || delta.require("episode")? != documents[0].root.require("bank_episode")?
                    || n(delta, "start", 0, i32::MAX as i128)? != bank.len()
                    || checksum(delta.require("previous_hash")?)?
                        != hash_text(prefixes.last().unwrap())
                {
                    return fail("journal5 delta owner/offset/prefix mismatch");
                }
                let events = delta.require("events")?.array()?;
                if events.len() > n(caps, "max_interval_events", 1, 65536)?
                    || bank.len() + events.len()
                        > n(caps, "max_episode_events", 1, i32::MAX as i128)?
                {
                    return fail("journal5 event capacity exceeded");
                }
                if corrected {
                    for event in events {
                        card_instance::event(event, recipe.as_ref().unwrap(), &budget, &control)?;
                    }
                } else {
                    validation::validate_events_controlled(delta.require("events")?, &control)?;
                }
                let count = events.len();
                for event_index in 0..count {
                    control.check()?;
                    let event = documents[index]
                        .root
                        .get_mut("delta")
                        .unwrap()
                        .get_mut("events")
                        .unwrap()
                        .as_array_mut()
                        .unwrap()[event_index]
                        .take();
                    prefixes.push(event_prefix_next(
                        prefixes.last().unwrap(),
                        &event,
                        &budget,
                        &control,
                    )?)?;
                    bank.push(event)?;
                }
                let boundary = documents[index].root.require("boundary")?;
                boundary.closed(&["end", "prefix_hash"], &[])?;
                if n(boundary, "end", 0, i32::MAX as i128)? != bank.len()
                    || checksum(boundary.require("prefix_hash")?)?
                        != hash_text(prefixes.last().unwrap())
                {
                    return fail("journal5 delta boundary mismatch");
                }
                delta_pending = true;
                continue;
            }
            let row = &documents[index].root;
            let h = &documents[0].root;
            match row.require("kind")?.string()? {
                "decision" => {
                    row.closed(
                        &[
                            "kind",
                            "decision",
                            "frame",
                            "table",
                            "search_report",
                            "selection",
                            "command",
                            "timings",
                        ],
                        &[],
                    )?;
                    let decision = n(row, "decision", 0, 65535)?;
                    if decision != decisions.len()
                        || pending.is_some()
                        || last_accepted.is_some()
                        || terminal.is_some()
                        || lifecycle.is_some()
                        || !delta_pending
                        || awaiting_next.is_some_and(|v| v != decision)
                    {
                        return fail("journal5 decision ordering mismatch");
                    }
                    let caps = h.require("caps")?;
                    if !caps.require("max_decisions")?.is_null()
                        && decision + 1 > n(caps, "max_decisions", 1, 1024)?
                    {
                        return fail("journal5 decision cap exceeded");
                    }
                    awaiting_next = None;
                    let f = row.require("frame")?;
                    if corrected {
                        card_instance::frame(f, &budget, &control)?;
                    } else {
                        frame(f, &budget, &control)?;
                    }
                    let start = n(f, "interval_start", 0, bank.len() as i128)?;
                    let previous = decisions
                        .last()
                        .map(|d| {
                            if let Some(binding) = &d.binding {
                                binding.root.require("frame")
                            } else {
                                documents[d.document].root.require("frame")
                            }
                        })
                        .transpose()?;
                    let previous_end = previous
                        .map(|v| n(v.require("boundary")?, "end", 0, bank.len() as i128))
                        .transpose()?
                        .unwrap_or(0);
                    if f.require("actor")?.integer(1, 2)? != viewer
                        || f.require("episode")? != h.require("bank_episode")?
                        || n(f.require("boundary")?, "end", 0, i32::MAX as i128)? != bank.len()
                        || checksum(f.require("boundary")?.require("prefix_hash")?)?
                            != hash_text(prefixes.last().unwrap())
                        || checksum(f.require("interval_start_hash")?)?
                            != hash_text(&prefixes[start])
                        || start != previous_end
                        || bank.len() - start > n(caps, "max_interval_events", 1, 65536)?
                        || previous.is_some_and(|v| {
                            v.require("revision").unwrap().as_u64()
                                >= f.require("revision").unwrap().as_u64()
                        })
                    {
                        return fail("journal5 frame owner/interval/prefix/revision mismatch");
                    }
                    let descriptor = row.require("table")?;
                    if n(descriptor, "count", 1, 65536)? > n(caps, "max_candidates", 1, 65536)? {
                        return fail("journal5 candidate cap exceeded");
                    }
                    let expected_file = format!("player-{viewer}-decision-{decision:010}.columns");
                    if descriptor.require("file")?.string()? != expected_file {
                        return fail("foreign actor/decision sidecar");
                    }
                    let raw =
                        sources.bytes(&sources.root.join(&expected_file), 64 * 1024 * 1024)?;
                    let table = columns_context(
                        &raw,
                        descriptor,
                        f,
                        n(caps, "max_positions", 1, 128)?,
                        false,
                        &budget,
                        &control,
                        if corrected {
                            ColumnContext::Corrected
                        } else {
                            ColumnContext::Engineering
                        },
                    )?;
                    drop(raw);
                    report_context(
                        row.require("search_report")?,
                        h,
                        &table.value,
                        &expected_rng,
                        &budget,
                        &control,
                        corrected,
                    )?;
                    expected_rng = json::copy(
                        row.require("search_report")?.require("rng_after")?,
                        &control,
                    )?;
                    let selection = row.require("selection")?;
                    selection.closed(
                        &["table_hash", "index", "policy_version", "logprob", "value"],
                        &[],
                    )?;
                    if selection.require("table_hash")? != table.value.require("table_hash")?
                        || selection.require("policy_version")? != h.require("policy")?
                    {
                        return fail("journal5 Selection identity mismatch");
                    }
                    let selected = n(
                        selection,
                        "index",
                        0,
                        table.value.require("kinds")?.array()?.len() as i128 - 1,
                    )?;
                    for key in ["logprob", "value"] {
                        let v = selection.require(key)?;
                        if v.is_null() {
                            if (h.require("model_version")?.string()? == "2"
                                || corrected && h.require("model_version")?.string()? == "3")
                                && h.require("model_id")?.string()? != "lightgbm"
                            {
                                return fail(
                                    "neural Selection requires finite BOTH policy statistics",
                                );
                            }
                        } else {
                            if h.require("model_id")?.string()? == "lightgbm"
                                || corrected && h.require("model_version")?.string()? == "1"
                            {
                                return fail(
                                    "Tree2 RAW Selection must have null policy statistics",
                                );
                            }
                            let number = finite(v, false)?;
                            if key == "logprob" && number > 0.0 {
                                return fail("positive log probability");
                            }
                        }
                    }
                    let selected =
                        selected_action_context(&table.value, selected, f, &control, corrected)?;
                    command(row.require("command")?, f, &selected, &budget, &control)?;
                    timings(row.require("timings")?)?;
                    let (frame_tokens, candidate_tokens) = if diagnostic {
                        let recipe = recipe.as_ref().unwrap();
                        (
                            Some(crate::structured_tokens::frame_tokens(
                                f, &bank, recipe, &budget, &control,
                            )?),
                            Some(crate::structured_tokens::candidate_tokens(
                                &table.value,
                                recipe,
                                &budget,
                                &control,
                            )?),
                        )
                    } else {
                        (None, None)
                    };
                    let binding = if diagnostic {
                        None
                    } else {
                        let frame_meta = copy_fields(
                            f,
                            &[
                                "actor",
                                "episode",
                                "revision",
                                "interval_start",
                                "boundary",
                                "interval_start_hash",
                                "frame_hash",
                            ],
                            &control,
                        )?;
                        let payload = json::canonical(f.require("payload")?, &budget, &control)?;
                        let value = json::object([
                            ("frame", frame_meta),
                            (
                                "payload_sha256",
                                json::text(&format!("{:x}", Sha256::digest(&*payload))),
                            ),
                            (
                                "table_hash",
                                json::copy(table.value.require("table_hash")?, &control)?,
                            ),
                            ("selection", json::copy(selection, &control)?),
                            ("command", json::copy(row.require("command")?, &control)?),
                        ]);
                        let encoded = json::canonical(&value, &budget, &control)?;
                        Some(json::parse(&encoded, &budget, &control)?)
                    };
                    decisions.push(Decision {
                        document: index,
                        table: Some(table),
                        frame_tokens,
                        candidate_tokens,
                        binding,
                        staged_frame: None,
                    })?;
                    pending = Some(decision);
                    delta_pending = false;
                }
                "acknowledgement" => {
                    row.closed(&["kind", "decision", "response"], &[])?;
                    let decision = n(row, "decision", 0, 65535)?;
                    if pending != Some(decision) {
                        return fail("ACK without unique pending decision");
                    }
                    acknowledgement(
                        row.require("response")?,
                        if let Some(binding) = &decisions[decision].binding {
                            binding.root.require("command")?
                        } else {
                            documents[decisions[decision].document]
                                .root
                                .require("command")?
                        },
                    )?;
                    let lifecycle_ack =
                        row.require("response")?.require("status")?.string()? != "active";
                    if !diagnostic {
                        if let Destination::Original {
                            stage: Some((stage, episode_index)),
                            tree_window,
                            ..
                        } = &mut destination
                        {
                            let d = &decisions[decision];
                            let meta = if tree_window.is_some() {
                                stage.selected_frame(
                                    *episode_index,
                                    accepted.len(),
                                    documents[d.document].root.require("frame")?,
                                    &d.table.as_ref().unwrap().value,
                                    d.binding
                                        .as_ref()
                                        .unwrap()
                                        .root
                                        .require("selection")?
                                        .require("index")?
                                        .integer(0, i64::MAX as i128)?
                                        as usize,
                                    &bank,
                                    recipe.as_ref().unwrap(),
                                )?
                            } else {
                                stage.frame(
                                    *episode_index,
                                    accepted.len(),
                                    documents[d.document].root.require("frame")?,
                                    &d.table.as_ref().unwrap().value,
                                    &bank,
                                    recipe.as_ref().unwrap(),
                                )?
                            };
                            decisions[decision].staged_frame = Some(meta);
                        }
                        decisions[decision].table.take();
                        let document = decisions[decision].document;
                        documents[document] = json::parse(b"null", &budget, &control)?;
                    }
                    accepted.push(decision)?;
                    last_accepted = Some(decision);
                    pending = None;
                    if lifecycle_ack {
                        lifecycle = Some(index);
                    }
                }
                "transition" => {
                    row.closed(
                        &["kind", "decision", "next_decision", "reward", "terminal"],
                        &[],
                    )?;
                    let decision = n(row, "decision", 0, 65535)?;
                    let reward = row.require("reward")?.integer(-1, 1)?;
                    let next = if row.require("next_decision")?.is_null() {
                        None
                    } else {
                        Some(n(row, "next_decision", 0, 65535)?)
                    };
                    if pending.is_some() || last_accepted != Some(decision) {
                        return fail("transition lacks positive accepted action");
                    }
                    if row.require("terminal")?.boolean()? {
                        if next.is_some() {
                            return fail("terminal next pointer");
                        }
                        terminal = Some(index);
                    } else {
                        if reward != 0 || next != Some(decisions.len()) {
                            return fail("intermediate transition mismatch");
                        }
                        awaiting_next = next;
                    }
                    last_accepted = None;
                }
                "end" => {
                    row.closed(
                        &[
                            "kind",
                            "end_kind",
                            "outcome",
                            "terminal_reward",
                            "final_observation",
                            "pending_decisions",
                            "simulator_abort_reason",
                        ],
                        &[],
                    )?;
                    let kind = row.require("end_kind")?.string()?;
                    if !["natural", "truncated", "aborted", "cancelled", "incomplete"]
                        .contains(&kind)
                    {
                        return fail("unknown journal5 closure kind");
                    }
                    let expected = json::array(pending.map(|v| json::integer(v as i128)));
                    if row.require("pending_decisions")? != &expected
                        || awaiting_next.is_some()
                        || delta_pending
                    {
                        return fail("end pending/transition mismatch");
                    }
                    let reason = row.require("simulator_abort_reason")?;
                    if !reason.is_null() {
                        validation::text_value(reason)?;
                        if kind
                            != if reason.string()? == "turn_limit_exceeded" {
                                "truncated"
                            } else {
                                "aborted"
                            }
                        {
                            return fail("abort classification mismatch");
                        }
                    }
                    if kind == "natural" {
                        let outcome = row.require("outcome")?;
                        validation::result(outcome)?;
                        if outcome.is_null() || pending.is_some() {
                            return fail("natural closure lacks observed outcome");
                        }
                        let reward = if outcome.as_str() == Some("draw") {
                            0
                        } else if outcome.require("win")?.require("winner")?.integer(1, 2)?
                            == viewer
                        {
                            1
                        } else {
                            -1
                        };
                        if row.require("terminal_reward")?.integer(-1, 1)? != reward {
                            return fail("natural reward perspective mismatch");
                        }
                        let final_obs = row.require("final_observation")?;
                        if corrected {
                            card_instance::observation(final_obs, Some(viewer), &budget, &control)?;
                        } else {
                            validation::observation_controlled(final_obs, Some(viewer), &control)?;
                        }
                        if final_obs.require("phase")?.string()? != "terminal"
                            || final_obs.require("result")? != outcome
                        {
                            return fail("natural closure final observation mismatch");
                        }
                        if let Some(&last) = accepted.last() {
                            let term = terminal.ok_or_else(|| {
                                Error::contract(
                                    "accepted action missing natural terminal transition",
                                )
                            })?;
                            if n(&documents[term].root, "decision", 0, 65535)? != last
                                || documents[term].root.require("reward")?.integer(-1, 1)? != reward
                            {
                                return fail("natural final accepted reward mismatch");
                            }
                        }
                    } else if !row.require("outcome")?.is_null()
                        || !row.require("terminal_reward")?.is_null()
                        || !row.require("final_observation")?.is_null()
                        || terminal.is_some()
                    {
                        return fail("interrupted closure invents natural targets/state");
                    }
                    if let Some(lifecycle) = lifecycle {
                        let response = documents[lifecycle].root.require("response")?;
                        let expected = if response.require("status")?.string()? == "terminal" {
                            "natural"
                        } else if reason.as_str() == Some("turn_limit_exceeded") {
                            "truncated"
                        } else {
                            "aborted"
                        };
                        if kind != expected
                            || row.require("outcome")? != response.require("result")?
                        {
                            return fail("lifecycle/closure mismatch");
                        }
                    }
                    end = Some(index);
                }
                _ => return fail("unknown journal5 record kind"),
            }
        }
        let end = end.ok_or_else(|| Error::contract("journal5 final closure required"))?;
        sources.recheck()?;
        Ok(ParsedJournal {
            documents,
            bank,
            prefixes,
            decisions,
            accepted,
            end,
            max_json,
            corrected,
        })
    }
    pub(crate) fn header_value(&self) -> &Value {
        self.data.header_value()
    }
    pub(crate) fn end_value(&self) -> &Value {
        self.data.end_value()
    }
    pub(crate) fn accepted_count(&self) -> usize {
        self.data.accepted_count()
    }
    pub(crate) fn boundary_value(&self) -> Value {
        self.data.boundary_value()
    }
    pub(crate) fn accepted_actions(&self) -> Result<Vec<&Value>> {
        self.data.accepted_actions()
    }
    pub(crate) fn correspondence_frames(&self) -> Result<json::Document> {
        self.data.correspondence_frames(&self.sources)
    }
    pub(crate) fn staged_episode(
        &self,
        stage: &mut crate::structured_staging::Stage,
        index: usize,
        reason: &Value,
    ) -> Result<Value> {
        self.data
            .staged_episode(&self.sources, stage, index, reason)
    }
    pub(crate) fn staged_tree_episode(
        &self,
        stage: &mut crate::structured_staging::Stage,
        index: usize,
        reason: &Value,
    ) -> Result<Value> {
        self.data
            .staged_tree_episode(&self.sources, stage, index, reason)
    }
    pub(crate) fn restage_corrected(
        sources: &mut crate::structured_session::Sources,
        journal_file: &str,
        max_json: usize,
        max_record: usize,
        declaration: &OriginalHeader<'_>,
        stage: &mut crate::structured_staging::Stage,
        index: usize,
    ) -> Result<ParsedJournal> {
        if !declaration.corrected {
            return fail("recorded restage requires corrected original declaration");
        }
        Self::parse_destination(
            sources,
            journal_file,
            sources.budget.clone(),
            sources.control.clone(),
            max_json,
            max_record,
            Destination::Original {
                declaration,
                stage: Some((stage, index)),
                tree_window: None,
            },
        )
    }
    pub(crate) fn into_sources(self) -> crate::structured_session::Sources {
        self.sources
    }
    pub fn diagnostic(&self) -> Result<crate::budget::BudgetVec<u8>> {
        let s = &self.sources;
        s.control.check()?;
        s.recheck()?;
        let scratch = s.budget.reserve(
            s.budget
                .used()
                .checked_mul(2)
                .and_then(|n| n.checked_add(65536))
                .ok_or_else(|| Error::resource("diagnostic copy overflow"))?,
        )?;
        let mut decisions = Vec::with_capacity(self.data.decisions.len());
        for (index, decision) in self.data.decisions.iter().enumerate() {
            s.control.check()?;
            let row = &self.data.documents[decision.document].root;
            decisions.push(json::object([
                ("decision", json::integer(index as i128)),
                ("frame", json::copy(row.require("frame")?, &s.control)?),
                (
                    "table",
                    json::copy(
                        &decision
                            .table
                            .as_ref()
                            .ok_or(crate::error::Error::Closed)?
                            .value,
                        &s.control,
                    )?,
                ),
                (
                    "selection",
                    json::copy(row.require("selection")?, &s.control)?,
                ),
                (
                    "search_report",
                    json::copy(row.require("search_report")?, &s.control)?,
                ),
                (
                    "frame_tokens",
                    json::copy(
                        &decision
                            .frame_tokens
                            .as_ref()
                            .ok_or(crate::error::Error::Closed)?
                            .value,
                        &s.control,
                    )?,
                ),
                (
                    "candidate_tokens",
                    json::copy(
                        &decision
                            .candidate_tokens
                            .as_ref()
                            .ok_or(crate::error::Error::Closed)?
                            .value,
                        &s.control,
                    )?,
                ),
            ]));
        }
        let result = json::object([
            ("kind", json::text("structured-journal-tokens")),
            (
                "validation_scope",
                json::text(if self.data.corrected {
                    "journal6-codec2-rawtokens-only"
                } else {
                    "journal5-columns-rawtokens-only"
                }),
            ),
            ("qualified", Value::Bool(false)),
            ("parent_qualification", json::text("not_performed")),
            (
                "projection",
                json::object([
                    (
                        "aggregate_recipe",
                        json::text(if self.data.corrected {
                            "public-exact-card-refs-v3"
                        } else {
                            "public-exact-v2"
                        }),
                    ),
                    (
                        "aggregate_window",
                        json::copy(
                            self.data.documents[0].root.require("aggregate_window")?,
                            &s.control,
                        )?,
                    ),
                    (
                        "declaration",
                        json::text(
                            if self.data.documents[0]
                                .root
                                .require("model_version")?
                                .string()?
                                == "1"
                            {
                                "baseline-corrected128"
                            } else if self.data.documents[0].root.require("model_id")?.string()?
                                == "lightgbm"
                            {
                                "tree-raw-corrected-own-header"
                            } else {
                                "neural-own-header"
                            },
                        ),
                    ),
                ]),
            ),
            (
                "header",
                json::copy(&self.data.documents[0].root, &s.control)?,
            ),
            (
                "bank",
                json::object([
                    (
                        "actor",
                        json::copy(self.data.documents[0].root.require("viewer")?, &s.control)?,
                    ),
                    (
                        "episode",
                        json::copy(
                            self.data.documents[0].root.require("bank_episode")?,
                            &s.control,
                        )?,
                    ),
                    (
                        "events",
                        json::array(
                            self.data
                                .bank
                                .iter()
                                .map(|v| json::copy(v, &s.control))
                                .collect::<Result<Vec<_>>>()?,
                        ),
                    ),
                    (
                        "prefixes",
                        json::array(self.data.prefixes.iter().map(|v| json::text(&hash_text(v)))),
                    ),
                ]),
            ),
            ("decisions", json::array(decisions)),
            (
                "accepted",
                json::array(self.data.accepted.iter().map(|&v| json::integer(v as i128))),
            ),
            (
                "end",
                json::copy(&self.data.documents[self.data.end].root, &s.control)?,
            ),
        ]);
        let encoded = json::canonical(&result, &s.budget, &s.control)?;
        if encoded.len() > self.data.max_json {
            return fail("diagnostic JSON wire cap exceeded; no truncation");
        }
        s.recheck()?;
        drop(scratch);
        Ok(encoded)
    }
}

impl ParsedJournal {
    pub(crate) fn header_value(&self) -> &Value {
        &self.documents[0].root
    }
    pub(crate) fn end_value(&self) -> &Value {
        &self.documents[self.end].root
    }
    pub(crate) fn accepted_count(&self) -> usize {
        self.accepted.len()
    }
    pub(crate) fn boundary_value(&self) -> Value {
        json::object([
            ("end", json::integer(self.bank.len() as i128)),
            (
                "prefix_hash",
                json::text(&hash_text(self.prefixes.last().unwrap())),
            ),
        ])
    }
    pub(crate) fn accepted_actions(&self) -> Result<Vec<&Value>> {
        self.accepted
            .iter()
            .map(|&i| {
                self.decisions[i]
                    .binding
                    .as_ref()
                    .ok_or(Error::Closed)?
                    .root
                    .require("command")?
                    .require("action")
            })
            .collect()
    }
    pub(crate) fn correspondence_frames(
        &self,
        s: &crate::structured_session::Sources,
    ) -> Result<json::Document> {
        s.control.check()?;
        let mut raw = 65536usize;
        for &i in self.accepted.iter() {
            raw = raw
                .checked_add(
                    json::canonical(
                        &self.decisions[i]
                            .binding
                            .as_ref()
                            .ok_or(Error::Closed)?
                            .root,
                        &s.budget,
                        &s.control,
                    )?
                    .len(),
                )
                .ok_or_else(|| Error::resource("raw correspondence size overflow"))?;
        }
        let _copy = json::admit(raw, &s.budget)?;
        let mut result = Vec::with_capacity(self.accepted.len());
        for &i in self.accepted.iter() {
            s.control.check()?;
            let binding = &self.decisions[i]
                .binding
                .as_ref()
                .ok_or(Error::Closed)?
                .root;
            let mut entry = copy_fields(
                binding.require("frame")?,
                &[
                    "frame_hash",
                    "revision",
                    "interval_start",
                    "interval_start_hash",
                    "boundary",
                ],
                &s.control,
            )?;
            for key in ["payload_sha256", "table_hash", "selection"] {
                entry
                    .as_object_mut()
                    .unwrap()
                    .insert(key.into(), json::copy(binding.require(key)?, &s.control)?);
            }
            result.push(entry);
        }
        let bytes = json::canonical(&json::array(result), &s.budget, &s.control)?;
        json::parse(&bytes, &s.budget, &s.control)
    }
    pub(crate) fn staged_episode(
        &self,
        s: &crate::structured_session::Sources,
        stage: &mut crate::structured_staging::Stage,
        index: usize,
        reason: &Value,
    ) -> Result<Value> {
        s.control.check()?;
        if self.header_value().require("model_version")?.string()?
            != if self.corrected { "3" } else { "2" }
            || self.header_value().require("model_id")?.string()? == "lightgbm"
        {
            return fail("baseline raw role cannot be a PPO payload");
        }
        let mut selections = Vec::with_capacity(self.accepted.len());
        let mut frames = Vec::with_capacity(self.accepted.len());
        let raw = self
            .accepted
            .len()
            .checked_mul(16384)
            .and_then(|n| n.checked_add(65536))
            .ok_or_else(|| Error::resource("episode metadata size overflow"))?;
        let _copy = s.budget.reserve(raw)?;
        for &i in self.accepted.iter() {
            let d = &self.decisions[i];
            frames.push(json::copy(
                d.staged_frame.as_ref().ok_or(Error::Closed)?,
                &s.control,
            )?);
            selections.push(json::copy(
                d.binding
                    .as_ref()
                    .ok_or(Error::Closed)?
                    .root
                    .require("selection")?,
                &s.control,
            )?);
        }
        let blocks = stage.finish_episode_arrays(index, &self.bank, &selections)?;
        let header = self.header_value();
        let end = self.end_value();
        let mut result = json::object([
            ("actor", json::copy(header.require("viewer")?, &s.control)?),
            (
                "episode",
                json::copy(header.require("bank_episode")?, &s.control)?,
            ),
            ("boundary", self.boundary_value()),
            ("raw_blocks", blocks),
            ("frames", json::array(frames)),
            ("selections", json::array(selections)),
            (
                "end_kind",
                json::copy(end.require("end_kind")?, &s.control)?,
            ),
            ("outcome", json::copy(end.require("outcome")?, &s.control)?),
            ("reason", json::copy(reason, &s.control)?),
        ]);
        for key in if self.corrected {
            &[
                "action_contract",
                "purpose",
                "synthetic_capacity",
                "rule_signature",
                "layout_signature",
            ][..]
        } else {
            &["action_contract", "purpose", "synthetic_capacity"][..]
        } {
            result
                .as_object_mut()
                .unwrap()
                .insert((*key).into(), json::copy(header.require(key)?, &s.control)?);
        }
        Ok(result)
    }
    pub(crate) fn staged_tree_episode(
        &self,
        s: &crate::structured_session::Sources,
        stage: &mut crate::structured_staging::Stage,
        index: usize,
        reason: &Value,
    ) -> Result<Value> {
        s.control.check()?;
        let _copy = s.budget.reserve(65536 + self.accepted.len() * 16384)?;
        let mut frames = Vec::new();
        let mut selections = Vec::new();
        for &i in self.accepted.iter() {
            s.control.check()?;
            frames.push(json::copy(
                self.decisions[i]
                    .staged_frame
                    .as_ref()
                    .ok_or(Error::Closed)?,
                &s.control,
            )?);
            selections.push(json::copy(
                self.decisions[i]
                    .binding
                    .as_ref()
                    .ok_or(Error::Closed)?
                    .root
                    .require("selection")?,
                &s.control,
            )?);
        }
        let header = self.header_value();
        let end = self.end_value();
        let mut value = json::object([
            ("actor", json::copy(header.require("viewer")?, &s.control)?),
            (
                "episode",
                json::copy(header.require("bank_episode")?, &s.control)?,
            ),
            ("boundary", self.boundary_value()),
            ("raw_blocks", stage.finish_episode_raw(index, &self.bank)?),
            ("frames", json::array(frames)),
            ("original_selections", json::array(selections)),
            (
                "end_kind",
                json::copy(end.require("end_kind")?, &s.control)?,
            ),
            ("outcome", json::copy(end.require("outcome")?, &s.control)?),
            ("reason", json::copy(reason, &s.control)?),
        ]);
        for key in ["action_contract", "purpose", "synthetic_capacity"] {
            value
                .as_object_mut()
                .unwrap()
                .insert(key.into(), json::copy(header.require(key)?, &s.control)?);
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mt_integer_seed_initial_states_match_independent_cpython314_goldens() {
        let budget = Budget::new(4 * 1024 * 1024);
        let control = Control::new(crate::budget::monotonic() + 30.0).unwrap();
        let fixtures = json::parse(
            include_bytes!("../tests/fixtures/cpython314_mt_initial.json"),
            &budget,
            &control,
        )
        .unwrap();
        for case in fixtures.root.array().unwrap() {
            let seed = case
                .require("seed")
                .unwrap()
                .integer(0, u64::MAX as i128)
                .unwrap() as u64;
            let actual = initial_mt(seed);
            assert_eq!(
                actual,
                *case.require("state").unwrap(),
                "direct MT integer seed {seed}"
            );
            rng_state(&actual).unwrap();
        }
    }
}

#[cfg(test)]
mod original_source_tests {
    use super::*;
    #[test]
    fn historical62_original_raw_walk_never_activates_token_projection() {
        let budget = Budget::new(128 * 1024 * 1024);
        let control = Control::new(crate::budget::monotonic() + 30.0).unwrap();
        let fixture = json::parse(
            include_bytes!("../tests/fixtures/structured_cohort16_inputs.json"),
            &budget,
            &control,
        )
        .unwrap();
        let episode = &fixture.root.require("episodes").unwrap().array().unwrap()[0];
        let mut model = fixture.root.require("model").unwrap().clone();
        model
            .get_mut("event")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove("aggregate_recipe");
        let raw = episode.require("journal").unwrap().string().unwrap();
        let mut h = json::parse(raw.lines().next().unwrap().as_bytes(), &budget, &control).unwrap();
        let window = h
            .root
            .require("aggregate_window")
            .unwrap()
            .integer(1, 65536)
            .unwrap();
        let old =
            crate::structured_controls::token_signature(window, true, &budget, &control).unwrap();
        h.root
            .as_object_mut()
            .unwrap()
            .insert("token_signature".into(), json::text(&old));
        let mut encoded = json::canonical(&h.root, &budget, &control)
            .unwrap()
            .to_vec();
        encoded.push(b'\n');
        encoded.extend_from_slice(raw.split_once('\n').unwrap().1.as_bytes());
        let root = std::env::temp_dir().join(format!(
            "gf-historical-raw-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let file = episode.require("journal_file").unwrap().string().unwrap();
        std::fs::write(root.join(file), encoded).unwrap();
        for (name, hex) in episode.require("columns").unwrap().object().unwrap() {
            let hex = hex.string().unwrap();
            let bytes = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect::<Vec<_>>();
            std::fs::write(root.join(name), bytes).unwrap();
        }
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
                .insert(key.into(), h.root.require(key).unwrap().clone());
        }
        let declaration = OriginalHeader::new(Some(&model), &identity, &budget, &control).unwrap();
        let journal = Journal::original(
            &root,
            file,
            budget.clone(),
            control.clone(),
            64 * 1024 * 1024,
            8 * 1024 * 1024,
            &declaration,
            None,
        )
        .unwrap();
        assert_eq!(journal.accepted_count(), 1);
        assert!(
            journal
                .data
                .decisions
                .iter()
                .all(|d| d.frame_tokens.is_none()
                    && d.candidate_tokens.is_none()
                    && d.table.is_none())
        );
        assert!(
            Journal::open(
                &root,
                file,
                budget.clone(),
                control.clone(),
                64 * 1024 * 1024,
                8 * 1024 * 1024
            )
            .is_err()
        );
        let mut stage = crate::structured_staging::Stage::new(
            budget.clone(),
            1024 * 1024,
            65536,
            control.clone(),
        )
        .unwrap();
        assert!(
            Journal::original(
                &root,
                file,
                budget.clone(),
                control.clone(),
                64 * 1024 * 1024,
                8 * 1024 * 1024,
                &declaration,
                Some((&mut stage, 0))
            )
            .is_err()
        );
        assert!(stage.descriptors().unwrap().array().unwrap().is_empty());
        stage.close().unwrap();
        journal.sources.recheck().unwrap();
        drop(journal);
        std::fs::remove_dir_all(root).unwrap();
    }
}
