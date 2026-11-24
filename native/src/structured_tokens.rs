//! Exact model-neutral participant tokens. No model, engine or catalog lookup.
//! Callers validate original frame/bank identities before invoking these helpers.
use crate::{
    budget::{Budget, Control, Lease},
    error::{Error, Result},
    json::{self, Document, Value, ValueExt},
    structured_controls::{digest, token_signature},
    structured_journal::card_instance::is_concealed_draw,
};
use std::sync::Arc;

const EXACT_RECIPE: &[u8] =
    include_bytes!("../assets/structured-token-recipe-public-exact-v2.json");
#[derive(Clone, Copy, PartialEq, Eq)]
enum Context {
    Engineering,
    CorrectedCardInstance,
}
pub struct Recipe {
    document: Document,
    pub window: usize,
    pub signature: String,
    context: Context,
}
pub struct OwnedTokens {
    pub value: Value,
    _lease: Lease,
}
impl Recipe {
    pub fn new(window: usize, budget: &Arc<Budget>, control: &Control) -> Result<Self> {
        let signature = token_signature(window as i128, false, budget, control)?;
        let mut document = json::parse(EXACT_RECIPE, budget, control)?;
        document
            .root
            .as_object_mut()
            .expect("pinned recipe")
            .insert("history_window".into(), json::integer(window as i128));
        if digest(&document.root, budget, control)? != signature {
            return Err(Error::contract(
                "embedded exact token recipe identity mismatch",
            ));
        }
        Ok(Self {
            document,
            window,
            signature,
            context: Context::Engineering,
        })
    }
    /// Explicit corrected diagnostic context, never inferred from input keys.
    pub(crate) fn corrected(
        window: usize,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<Self> {
        let signature =
            crate::structured_controls::card_instance_token_signature(window, budget, control)?;
        let mut document = json::parse(
            crate::structured_controls::CARD_INSTANCE_RECIPE,
            budget,
            control,
        )?;
        document
            .root
            .as_object_mut()
            .expect("pinned recipe")
            .insert("history_window".into(), json::integer(window as i128));
        Ok(Self {
            document,
            window,
            signature,
            context: Context::CorrectedCardInstance,
        })
    }
    pub(crate) fn is_corrected(&self) -> bool {
        self.context == Context::CorrectedCardInstance
    }
    pub(crate) fn guardian_ids(&self) -> Result<&[Value]> {
        self.list("state_categories")?[7].array()
    }
    pub(crate) fn definition_ids(&self) -> Result<&[Value]> {
        self.list("card_categories")?[0].array()
    }
    fn list(&self, field: &str) -> Result<&[Value]> {
        self.document.root.require(field)?.array()
    }
    fn category(&self, value: Option<&Value>, vocabulary: &Value) -> Result<i64> {
        let Some(value) = value.filter(|v| !v.is_null()) else {
            return Ok(1);
        };
        vocabulary
            .array()?
            .iter()
            .position(|entry| entry == value)
            .map(|i| i as i64 + 2)
            .ok_or_else(|| Error::contract("unknown exact structured category"))
    }
    fn categories(&self, field: &str, values: &[Option<Value>]) -> Result<Vec<i64>> {
        let vocabularies = self.list(field)?;
        if vocabularies.len() != values.len() {
            return Err(Error::contract("exact category shape mismatch"));
        }
        values
            .iter()
            .zip(vocabularies)
            .map(|(value, vocabulary)| self.category(value.as_ref(), vocabulary))
            .collect()
    }
}
fn number(value: &Value) -> Result<i64> {
    if let Value::Bool(v) = value {
        return Ok(i64::from(*v));
    }
    Ok(value.integer(i64::MIN as i128, i64::MAX as i128)? as i64)
}
fn numbers(value: &Value, fields: &[Value]) -> Result<(Vec<i64>, Vec<bool>)> {
    let mut values = Vec::with_capacity(fields.len());
    let mut present = Vec::with_capacity(fields.len());
    for field in fields {
        let value = value.get(field.string()?);
        values.push(value.map(number).transpose()?.unwrap_or(0));
        present.push(value.is_some());
    }
    Ok((values, present))
}
fn ints(values: impl IntoIterator<Item = i64>) -> Value {
    json::array(values.into_iter().map(|v| json::integer(v as i128)))
}
fn flags(values: impl IntoIterator<Item = bool>) -> Value {
    json::array(values.into_iter().map(Value::Bool))
}
fn relative(value: Option<&Value>, actor: i128) -> Result<Option<Value>> {
    value
        .filter(|v| !v.is_null())
        .map(|v| {
            Ok(json::text(if v.integer(1, 2)? == actor {
                "self"
            } else {
                "opponent"
            }))
        })
        .transpose()
}
fn result(value: Option<&Value>, actor: i128) -> Result<Option<Value>> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    Ok(Some(if value.as_str() == Some("draw") {
        json::text("draw")
    } else {
        json::text(
            if value.require("win")?.require("winner")?.integer(1, 2)? == actor {
                "self_win"
            } else {
                "opponent_win"
            },
        )
    }))
}
fn add(destination: &mut i64, value: i64) -> Result<()> {
    *destination = destination
        .checked_add(value)
        .ok_or_else(|| Error::contract("structured int64 history overflow"))?;
    Ok(())
}
fn event(value: &Value, actor: i128, recipe: &Recipe) -> Result<(Vec<i64>, Vec<bool>, Vec<i64>)> {
    let (values, present) = numbers(value, recipe.list("event_numeric")?)?;
    let mut cats = vec![
        Some(value.require("type")?.clone()),
        value.get("definition_id").cloned(),
        value.get("status").cloned(),
        result(value.get("result"), actor)?,
    ];
    for key in [
        "player",
        "attacker",
        "target",
        "active_player",
        "buyer",
        "seller",
        "from",
        "to",
    ] {
        cats.push(relative(value.get(key), actor)?);
    }
    Ok((
        values,
        present,
        recipe.categories("event_categories", &cats)?,
    ))
}
fn reservation(budget: &Arc<Budget>, scalar_count: usize) -> Result<Lease> {
    // All token scalars are signed integers/booleans. This also charges their
    // temporary Value containers and nested row vectors before construction.
    let bytes = scalar_count
        .checked_mul(1024)
        .and_then(|n| n.checked_add(65536))
        .ok_or_else(|| Error::resource("structured token ownership overflow"))?;
    budget.reserve(bytes)
}

pub fn frame_tokens(
    frame: &Value,
    events: &[Value],
    recipe: &Recipe,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<OwnedTokens> {
    frame_tokens_context(frame, events, recipe, budget, control, false)
}
/// Rust-only live History owns validation of every appended event exactly once.
/// The token/aggregation kernel below is identical to the validated DTO path.
pub(crate) fn frame_tokens_owned(
    frame: &Value,
    events: &[Value],
    recipe: &Recipe,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<OwnedTokens> {
    frame_tokens_context(frame, events, recipe, budget, control, true)
}
fn frame_tokens_context(
    frame: &Value,
    events: &[Value],
    recipe: &Recipe,
    budget: &Arc<Budget>,
    control: &Control,
    owned: bool,
) -> Result<OwnedTokens> {
    control.check()?;
    let actor = frame.require("actor")?.integer(1, 2)?;
    let start = frame
        .require("interval_start")?
        .integer(0, events.len() as i128)? as usize;
    let end = frame
        .require("boundary")?
        .require("end")?
        .integer(start as i128, events.len() as i128)? as usize;
    if end - start > 65536 {
        return Err(Error::contract("structured interval token capacity"));
    }
    let corrected = recipe.is_corrected();
    if corrected && !owned {
        // Validate only this participant's original prefix. A later tail is
        // never read to construct an earlier frame's tokens.
        for value in &events[..end] {
            crate::structured_journal::card_instance::event(value, recipe, budget, control)?;
        }
    }
    let scalar_count = (end - start)
        .checked_mul(34)
        .and_then(|n| n.checked_add(2048))
        .ok_or_else(|| Error::resource("structured interval token size overflow"))?;
    let lease = reservation(budget, scalar_count)?;
    let payload = frame.require("payload")?;
    let obs = payload.require("observation")?;
    let options = payload
        .require("action_options")?
        .require("action_options")?;
    let mut values = json::object([]);
    for field in &recipe.list("state_numeric")?[..4] {
        values.as_object_mut().unwrap().insert(
            field.string()?.into(),
            json::copy(obs.require(field.string()?)?, control)?,
        );
    }
    for side in ["own", "opponent"] {
        let participant = obs.require(side)?;
        for key in ["hp", "mp", "money", "ascended"] {
            if let Some(value) = participant.get(key) {
                values
                    .as_object_mut()
                    .unwrap()
                    .insert(format!("{side}.{key}"), json::copy(value, control)?);
            }
        }
        if let Some(status) = participant.get("status") {
            for key in ["fog", "flash", "dream", "dark_cloud"] {
                if let Some(value) = status.get(key) {
                    values
                        .as_object_mut()
                        .unwrap()
                        .insert(format!("{side}.{key}"), json::copy(value, control)?);
                }
            }
        }
    }
    for key in ["can_pray", "can_discard", "can_sacrifice", "can_allow"] {
        values
            .as_object_mut()
            .unwrap()
            .insert(key.into(), json::copy(options.require(key)?, control)?);
    }
    for (label, player) in [("self", actor), ("opponent", 3 - actor)] {
        values.as_object_mut().unwrap().insert(
            format!("target_{label}"),
            Value::Bool(
                options
                    .require("valid_targets")?
                    .array()?
                    .iter()
                    .any(|v| v.as_i64() == Some(player as i64)),
            ),
        );
    }
    let (state_numbers, state_present) = numbers(&values, recipe.list("state_numeric")?)?;
    let mut cats = vec![
        obs.get("phase").cloned(),
        obs.get("required_choice").cloned(),
    ];
    for key in ["first_player", "active_player", "current_player"] {
        cats.push(relative(obs.get(key), actor)?);
    }
    cats.push(result(obs.get("result"), actor)?);
    let mut category_present = vec![true; 6];
    for side in ["own", "opponent"] {
        let participant = obs.require(side)?;
        cats.push(
            participant
                .get("status")
                .and_then(|v| v.get("disease"))
                .cloned(),
        );
        cats.push(participant.get("guardian").cloned());
        category_present.extend([
            participant.get("status").is_some(),
            participant.get("guardian").is_some(),
        ]);
    }
    let state_cats = recipe.categories("state_categories", &cats)?;
    let own = obs.require("own")?;
    let mut cards = Vec::with_capacity(18);
    if recipe.is_corrected() {
        let originals = own.require("cards")?.array()?;
        if originals.len() > 18 {
            return Err(Error::contract("own card token capacity"));
        }
        for (index, card) in originals.iter().enumerate() {
            cards.push((
                card.require("definition_id")?,
                card.require("kind")?.string()?,
                card.require("card_ref")?,
                index + 1,
                card.get("state"),
            ));
        }
    } else {
        let hand = own.require("hand")?.array()?;
        let learned = own.require("learned_miracles")?.array()?;
        if hand.len() + learned.len() > 18 {
            return Err(Error::contract("own card token capacity"));
        }
        for card in hand {
            cards.push((
                card.require("definition_id")?,
                "hand_artifact",
                card.require("instance_id")?,
                0,
                None,
            ));
        }
        for (index, name) in learned.iter().enumerate() {
            cards.push((name, "learned_miracle", name, index + 1, None));
        }
    }
    if cards.len() > 18 {
        return Err(Error::contract("own card token capacity"));
    }
    let mut card_numbers = Vec::with_capacity(18);
    let mut card_present = Vec::with_capacity(18);
    let mut card_cats = Vec::with_capacity(18);
    for (name, source, reference, order, state) in cards {
        control.check()?;
        let definition = payload.require("catalog")?.require(name.string()?)?;
        let mut numbers_value = json::object([]);
        for field in recipe.list("card_numeric")? {
            if let Some(value) = definition.get(field.string()?) {
                numbers_value
                    .as_object_mut()
                    .unwrap()
                    .insert(field.string()?.into(), json::copy(value, control)?);
            }
        }
        let available = |key: &str| -> Result<bool> {
            let rows = options.require(key)?.array()?;
            if recipe.is_corrected() {
                Ok(rows
                    .iter()
                    .any(|row| row.get("card_ref") == Some(reference)))
            } else {
                Ok(rows.contains(reference))
            }
        };
        let usable = if recipe.is_corrected() {
            available("usable_cards")?
        } else {
            available(if source == "hand_artifact" {
                "usable_hand_elements"
            } else {
                "usable_learned_miracles"
            })?
        };
        numbers_value
            .as_object_mut()
            .unwrap()
            .insert("usable".into(), Value::Bool(usable));
        numbers_value.as_object_mut().unwrap().insert(
            if recipe.is_corrected() {
                "possession_position"
            } else {
                "learned_order"
            }
            .into(),
            json::integer(order as i128),
        );
        for key in ["exchange", "sell", "buy"] {
            let enabled = if recipe.is_corrected() {
                available(&format!("{key}_cards"))?
            } else {
                source == "hand_artifact" && available(&format!("{key}_artifacts"))?
            };
            numbers_value
                .as_object_mut()
                .unwrap()
                .insert(key.into(), Value::Bool(enabled));
        }
        if recipe.is_corrected() {
            numbers_value.as_object_mut().unwrap().insert(
                "phenomenon".into(),
                Value::Bool(available("phenomenon_cards")?),
            );
            let roles = definition
                .require("classification")?
                .require("use_roles")?
                .array()?;
            for field in &recipe.list("card_numeric")?[13..] {
                let key = field.string()?;
                let role = key
                    .strip_prefix("role.")
                    .ok_or_else(|| Error::contract("corrected role recipe mismatch"))?;
                numbers_value.as_object_mut().unwrap().insert(
                    key.into(),
                    Value::Bool(roles.iter().any(|v| v.as_str() == Some(role))),
                );
            }
        }
        let (n, p) = numbers(&numbers_value, recipe.list("card_numeric")?)?;
        card_numbers.push(ints(n));
        card_present.push(flags(p));
        let mut categories = vec![
            Some(json::copy(name, control)?),
            definition.get("ability").cloned(),
            definition.get("kind").cloned(),
            definition.get("attribute").cloned(),
            definition.get("curse").and_then(|v| v.get("id")).cloned(),
            definition
                .get("guardian")
                .and_then(|v| v.get("id"))
                .cloned(),
        ];
        categories.push(Some(json::text(source)));
        if recipe.is_corrected() {
            let classification = definition.require("classification")?;
            categories.extend([
                state.cloned(),
                classification.get("definition_class").cloned(),
                classification.get("miracle_subclass").cloned(),
                Some(json::text(&if source == "artifact" {
                    "artifact".into()
                } else {
                    format!(
                        "miracle_{}",
                        state
                            .ok_or_else(|| Error::contract("miracle state missing"))?
                            .string()?
                    )
                })),
            ]);
        }
        card_cats.push(ints(recipe.categories("card_categories", &categories)?));
    }
    let own_count = card_numbers.len();
    while card_numbers.len() < 18 {
        card_numbers.push(ints(vec![0; recipe.list("card_numeric")?.len()]));
        card_present.push(flags(vec![false; recipe.list("card_numeric")?.len()]));
        card_cats.push(ints(vec![0; recipe.list("card_categories")?.len()]));
    }
    let mut opponent = Vec::with_capacity(18);
    for name in obs
        .require("opponent")?
        .require("learned_miracles")?
        .array()?
    {
        opponent.push(recipe.category(Some(name), &recipe.list("card_categories")?[0])?);
    }
    if opponent.len() > 18 {
        return Err(Error::contract("opponent public learned capacity"));
    }
    opponent.resize(18, 0);
    let mut interval_numbers = Vec::new();
    let mut interval_present = Vec::new();
    let mut interval_cats = Vec::new();
    for value in &events[start..end] {
        control.check()?;
        if corrected && is_concealed_draw(value) {
            continue;
        }
        let (n, p, c) = event(value, actor, recipe)?;
        interval_numbers.push(ints(n));
        interval_present.push(flags(p));
        interval_cats.push(ints(c));
    }
    let event_count = recipe.list("event_categories")?[0].array()?.len();
    let numeric_start = event_count + 16;
    let definition_start = numeric_start + 2 * recipe.list("event_numeric")?.len();
    let definition_vocab = &recipe.list("event_categories")?[1];
    let window_index = definition_start + definition_vocab.array()?.len();
    let actor_start = window_index + 2;
    let mut history = vec![0i64; actor_start + 16];
    let window_start = if corrected {
        0
    } else {
        end.saturating_sub(recipe.window)
    };
    let mut count = 0;
    let mut truncated = window_start != 0;
    // The statistics use bounded integer sums. Accumulate from the end so
    // concealed draws use no slots and the first extra visible row proves truncation.
    for value in events[window_start..end].iter().rev() {
        control.check()?;
        if corrected && is_concealed_draw(value) {
            continue;
        }
        if count == recipe.window {
            truncated = true;
            break;
        }
        count += 1;
        let (n, p, c) = event(value, actor, recipe)?;
        add(&mut history[(c[0] - 2) as usize], 1)?;
        for (role, &category) in c[4..].iter().enumerate() {
            if category >= 2 {
                add(
                    &mut history[event_count + 2 * role + (category - 2) as usize],
                    1,
                )?;
            }
        }
        for (i, (&number, &exists)) in n.iter().zip(&p).enumerate() {
            add(&mut history[numeric_start + 2 * i], number)?;
            add(&mut history[numeric_start + 2 * i + 1], i64::from(exists))?;
        }
        if let Some(name) = value.get("definition_id") {
            let category = recipe.category(Some(name), definition_vocab)?;
            if category < 2 {
                return Err(Error::contract("missing exact event definition ID"));
            }
            add(&mut history[definition_start + (category - 2) as usize], 1)?;
        }
        let player = value
            .get("player")
            .or_else(|| value.get("attacker"))
            .or_else(|| value.get("active_player"));
        if let Some(player) = player.filter(|v| !v.is_null()) {
            let player = player.integer(1, 2)?;
            let base = actor_start + if player == actor { 0 } else { 8 };
            add(&mut history[base], 1)?;
            for (i, key) in [
                "amount",
                "hp_delta",
                "mp_delta",
                "money_delta",
                "mp_spent",
                "price",
                "attack",
            ]
            .iter()
            .enumerate()
            {
                if let Some(v) = value.get(*key) {
                    add(&mut history[base + 1 + i], number(v)?)?;
                }
            }
        }
    }
    history[window_index] = count as i64;
    history[window_index + 1] = i64::from(truncated);
    let value = json::object([
        (
            "frame_hash",
            json::copy(frame.require("frame_hash")?, control)?,
        ),
        ("recipe_signature", json::text(&recipe.signature)),
        ("aggregate_window", json::integer(recipe.window as i128)),
        ("actor", json::integer(actor)),
        ("episode", json::copy(frame.require("episode")?, control)?),
        ("revision", json::copy(frame.require("revision")?, control)?),
        ("event_start", json::integer(start as i128)),
        ("event_end", json::integer(end as i128)),
        ("state_numerics", ints(state_numbers)),
        ("state_present", flags(state_present)),
        ("state_categories", ints(state_cats)),
        ("state_category_present", flags(category_present)),
        ("card_numerics", json::array(card_numbers)),
        ("card_present", json::array(card_present)),
        ("card_categories", json::array(card_cats)),
        ("card_mask", flags((0..18).map(|i| i < own_count))),
        ("opponent_learned", ints(opponent)),
        ("interval_numerics", json::array(interval_numbers)),
        ("interval_present", json::array(interval_present)),
        ("interval_categories", json::array(interval_cats)),
        ("history", ints(history)),
    ]);
    control.check()?;
    Ok(OwnedTokens {
        value,
        _lease: lease,
    })
}

pub fn candidate_tokens(
    table: &Value,
    recipe: &Recipe,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<OwnedTokens> {
    control.check()?;
    let count = table.require("kinds")?.array()?.len();
    let elements = table.require("slots")?.array()?.len();
    let lease = reservation(
        budget,
        count
            .checked_mul(11)
            .and_then(|v| v.checked_add(elements))
            .ok_or_else(|| Error::resource("candidate token size overflow"))?,
    )?;
    let categories = recipe.list("candidate_categories")?;
    let kinds = table
        .require("kinds")?
        .array()?
        .iter()
        .map(|v| recipe.category(Some(v), &categories[0]))
        .collect::<Result<Vec<_>>>()?;
    let targets = table
        .require("targets")?
        .array()?
        .iter()
        .map(|v| recipe.category(Some(v), &categories[1]))
        .collect::<Result<Vec<_>>>()?;
    let value = json::object([
        (
            "table_hash",
            json::copy(table.require("table_hash")?, control)?,
        ),
        (
            "frame_hash",
            json::copy(table.require("frame_hash")?, control)?,
        ),
        ("recipe_signature", json::text(&recipe.signature)),
        ("kinds", ints(kinds)),
        ("targets", ints(targets)),
        ("numerics", json::copy(table.require("numerics")?, control)?),
        ("present", json::copy(table.require("presence")?, control)?),
        ("offsets", json::copy(table.require("offsets")?, control)?),
        ("slots", json::copy(table.require("slots")?, control)?),
    ]);
    Ok(OwnedTokens {
        value,
        _lease: lease,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structured_journal;
    #[test]
    fn pure_history_key_presence_null_fallback_is_not_a_journal_acceptance_fixture() {
        let budget = Budget::new(64 * 1024 * 1024);
        let control = Control::new(crate::budget::monotonic() + 30.0).unwrap();
        let fixtures = json::parse(
            include_bytes!("../tests/fixtures/structured_tokens_v2.json"),
            &budget,
            &control,
        )
        .unwrap();
        let frame = fixtures.root.array().unwrap()[0].require("frame").unwrap();
        // These exercise the lower pure aggregate rule, including explicit
        // null keys; the original J5 validator independently rejects events
        // outside its closed public schema before calling this token helper.
        let events = json::parse(
            br#"[
            {"type":"healed","player":null,"attacker":1,"amount":100},
            {"type":"healed","attacker":null,"active_player":1,"amount":200},
            {"type":"healed","attacker":1,"active_player":2,"amount":-3},
            {"type":"healed","player":2,"attacker":1,"amount":4},
            {"type":"healed","active_player":1,"amount":2},
            {"type":"phenomenon_invoked","definition_id":"fireball"}
        ]"#,
            &budget,
            &control,
        )
        .unwrap();
        let recipe = Recipe::new(128, &budget, &control).unwrap();
        let result = frame_tokens(
            frame,
            events.root.array().unwrap(),
            &recipe,
            &budget,
            &control,
        )
        .unwrap();
        let history = result.value.require("history").unwrap().array().unwrap();
        assert_eq!(history[360], json::integer(2));
        assert_eq!(history[361], json::integer(-1));
        assert_eq!(history[368], json::integer(1));
        assert_eq!(history[369], json::integer(4));
        assert!(crate::validation::validate_events(&events.root).is_err());
    }
    #[test]
    fn python_actual_frame_columns_and_raw_tokens_match_exactly() {
        let budget = Budget::new(512 * 1024 * 1024);
        let control = Control::new(crate::budget::monotonic() + 30.0).unwrap();
        let fixtures = json::parse(
            include_bytes!("../tests/fixtures/structured_tokens_v2.json"),
            &budget,
            &control,
        )
        .unwrap();
        for case in fixtures.root.array().unwrap() {
            let frame = case.require("frame").unwrap();
            structured_journal::frame(frame, &budget, &control).unwrap();
            let events = case.require("events").unwrap().array().unwrap();
            let recipe = Recipe::new(
                case.require("window").unwrap().integer(1, 65536).unwrap() as usize,
                &budget,
                &control,
            )
            .unwrap();
            let result = frame_tokens(frame, events, &recipe, &budget, &control).unwrap();
            assert_eq!(
                result.value,
                *case.require("frame_tokens").unwrap(),
                "frame {}",
                case.require("label").unwrap()
            );
            let hex = case.require("bytes").unwrap().string().unwrap();
            let bytes = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect::<Vec<_>>();
            let synthetic = case
                .require("columns")
                .unwrap()
                .require("synthetic_capacity")
                .unwrap()
                .boolean()
                .unwrap();
            let table = structured_journal::columns(
                &bytes,
                case.require("columns").unwrap(),
                frame,
                128,
                synthetic,
                &budget,
                &control,
            )
            .unwrap();
            if synthetic {
                assert!(
                    structured_journal::columns(
                        &bytes,
                        case.require("columns").unwrap(),
                        frame,
                        128,
                        false,
                        &budget,
                        &control
                    )
                    .is_err()
                );
            }
            assert_eq!(table.value, *case.require("table").unwrap());
            let tokens = candidate_tokens(&table.value, &recipe, &budget, &control).unwrap();
            assert_eq!(tokens.value, *case.require("candidate_tokens").unwrap());
            // The same observed frame never receives future bank events.
            let mut future = events.to_vec();
            future.push(events[0].clone());
            assert_eq!(
                frame_tokens(frame, &future, &recipe, &budget, &control)
                    .unwrap()
                    .value,
                result.value
            );
        }
    }
}
