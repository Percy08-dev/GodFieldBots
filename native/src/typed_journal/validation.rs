//! Concrete participant checks, matching validation.rs without rebuilding a Value tree.
use super::{
    dto::contract::{Presence, UniqueMap, Validate},
    dto::text::Text,
    dto::*,
    ensure, required,
};
use crate::{
    budget::{Budget, BudgetVec, Control},
    error::{Error, Result},
};
use std::sync::Arc;

fn shape(value: &impl Validate) -> Result<()> {
    value.validate().map_err(Error::contract)
}
fn text_is(value: &Presence<Text<'_>>, expected: &str) -> bool {
    value
        .value()
        .is_some_and(|value| value.as_ref() == expected)
}
fn sources(value: &Schema<'_>, expected: &[&str]) -> Result<()> {
    let actual = required(&value.sources)?;
    ensure(
        actual.len() == expected.len()
            && actual.iter().zip(expected).all(|(a, b)| a.as_ref() == *b),
        "typed metadata sources mismatch",
    )
}
fn fields<'a, 'raw>(
    value: &'a UniqueMap<Text<'raw>, Schema<'raw>>,
    expected: &[&str],
) -> Result<()> {
    ensure(
        value.0.len() == expected.len()
            && expected
                .iter()
                .all(|key| value.0.keys().any(|actual| actual.as_ref() == *key)),
        "typed metadata fields mismatch",
    )
}
fn field<'a, 'raw>(
    value: &'a UniqueMap<Text<'raw>, Schema<'raw>>,
    name: &str,
) -> Result<&'a Schema<'raw>> {
    value
        .0
        .iter()
        .find_map(|(key, value)| (key.as_ref() == name).then_some(value))
        .ok_or_else(|| Error::contract("missing typed schema field"))
}
fn integer_bounds(value: &Schema<'_>, low: i128, high: i128) -> Result<()> {
    ensure(
        text_is(&value.r#type, "integer")
            && *required(&value.minimum)? == low
            && *required(&value.maximum)? == high,
        "typed metadata integer mismatch",
    )
}
fn integer_schema(value: &Schema<'_>, low: i128, high: i128) -> Result<()> {
    integer_bounds(value, low, high)?;
    ensure(
        value.sources.missing(),
        "scalar integer schema has source directives",
    )
}
fn instance_schema(value: &Schema<'_>) -> Result<()> {
    integer_bounds(value, 0, u64::MAX as i128)?;
    sources(value, &["own.hand[].instance_id"])
}
fn player_schema(value: &Schema<'_>) -> Result<()> {
    ensure(
        text_is(&value.r#type, "integer")
            && required(&value.r#enum)?.as_slice()
                == [
                    super::dto::contract::Player(1),
                    super::dto::contract::Player(2),
                ],
        "typed metadata player mismatch",
    )?;
    sources(value, &["own.player_id", "opponent.player_id"])
}
fn object_schema<'a, 'raw>(
    value: &'a Schema<'raw>,
    tag: &str,
    names: &[&str],
    optional: &[&str],
) -> Result<(
    &'a UniqueMap<Text<'raw>, Schema<'raw>>,
    &'a UniqueMap<Text<'raw>, Schema<'raw>>,
)> {
    ensure(
        text_is(&value.r#type, tag) && !*required(&value.additional_fields)?,
        "typed metadata object mismatch",
    )?;
    let required_fields = required(&value.required_fields)?;
    let optional_fields = required(&value.optional_fields)?;
    fields(required_fields, names)?;
    fields(optional_fields, optional)?;
    Ok((required_fields, optional_fields))
}
fn element_schema(value: &Schema<'_>) -> Result<()> {
    let alternatives = required(&value.one_of)?;
    ensure(
        alternatives.len() == 2,
        "typed metadata element alternatives mismatch",
    )?;
    let (hand, _) = object_schema(&alternatives[0], "object", &["source", "instance_id"], &[])?;
    let (learned, _) = object_schema(
        &alternatives[1],
        "object",
        &["source", "definition_id"],
        &[],
    )?;
    for (fields, tag) in [(hand, "hand_artifact"), (learned, "learned_miracle")] {
        let source = field(fields, "source")?;
        ensure(
            text_is(&source.r#type, "string") && text_is(&source.r#const, tag),
            "typed metadata source mismatch",
        )?;
    }
    instance_schema(field(hand, "instance_id")?)?;
    let definition = field(learned, "definition_id")?;
    ensure(
        text_is(&definition.r#type, "string"),
        "typed metadata definition mismatch",
    )?;
    sources(definition, &["own.learned_miracles[]"])
}
fn array_schema(value: &Schema<'_>, typed: bool) -> Result<()> {
    ensure(
        text_is(&value.r#type, "array") && *required(&value.ordered)?,
        "typed metadata array mismatch",
    )?;
    if typed {
        element_schema(required(&value.items)?)
    } else {
        instance_schema(required(&value.items)?)
    }
}
fn descriptors(
    entries: &[Schema<'_>],
    choice: Option<&str>,
    can_act: bool,
    control: &Control,
) -> Result<bool> {
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
    ensure(entries.len() == tags.len(), "typed metadata mismatch")?;
    for (entry, tag) in entries.iter().zip(tags) {
        control.check()?;
        let names: &[&str] = match *tag {
            "pray" => &[],
            "play_sequence" => &["ordered_elements"],
            "respond" | "respond_sequence" => &["ordered_defenses"],
            "discard" | "sacrifice" => &["instance_ids"],
            "exchange" => &["artifact", "hp", "mp", "money"],
            "sell" => &["artifact", "sold_artifact", "target"],
            "buy" => &["artifact", "target"],
            "buy_decision" => &["accept"],
            _ => unreachable!(),
        };
        let (required_fields, optional_fields) = object_schema(
            entry,
            tag,
            names,
            if *tag == "play_sequence" {
                &["target"]
            } else {
                &[]
            },
        )?;
        if *tag == "play_sequence" {
            let alternatives = required(&field(optional_fields, "target")?.one_of)?;
            ensure(
                alternatives.len() == 2,
                "typed metadata target alternatives mismatch",
            )?;
            player_schema(&alternatives[0])?;
            ensure(
                text_is(&alternatives[1].r#type, "null"),
                "typed metadata target null mismatch",
            )?;
        }
        match *tag {
            "pray" => {}
            "play_sequence" => array_schema(field(required_fields, "ordered_elements")?, true)?,
            "respond" | "respond_sequence" => array_schema(
                field(required_fields, "ordered_defenses")?,
                *tag == "respond_sequence",
            )?,
            "discard" | "sacrifice" => {
                array_schema(field(required_fields, "instance_ids")?, false)?
            }
            "exchange" => {
                instance_schema(field(required_fields, "artifact")?)?;
                integer_schema(
                    field(required_fields, "hp")?,
                    i32::MIN as i128,
                    i32::MAX as i128,
                )?;
                for key in ["mp", "money"] {
                    integer_schema(field(required_fields, key)?, 0, u32::MAX as i128)?;
                }
            }
            "buy" | "sell" => {
                instance_schema(field(required_fields, "artifact")?)?;
                if *tag == "sell" {
                    instance_schema(field(required_fields, "sold_artifact")?)?;
                }
                player_schema(field(required_fields, "target")?)?;
            }
            "buy_decision" => ensure(
                text_is(&field(required_fields, "accept")?.r#type, "boolean"),
                "typed metadata accept mismatch",
            )?,
            _ => unreachable!(),
        }
    }
    Ok(!tags.is_empty())
}
pub fn observation(
    value: &Observation<'_>,
    viewer: Option<u8>,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<()> {
    shape(value)?;
    observation_semantics(value, viewer, budget, control)
}
pub(crate) fn observation_semantics(
    value: &Observation<'_>,
    viewer: Option<u8>,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<()> {
    control.check()?;
    ensure(
        viewer.is_none_or(|viewer| viewer == value.viewer_player_id.0),
        "viewer mismatch",
    )?;
    let mut hand = BudgetVec::with_capacity(budget, value.own.hand.len())?;
    for artifact in &value.own.hand {
        control.check()?;
        hand.push(artifact.instance_id)?;
    }
    crate::budget::sort_unstable_by(&mut hand, Ord::cmp, control)?;
    ensure(
        !hand.windows(2).any(|pair| pair[0] == pair[1]),
        "duplicate hand instance",
    )?;
    let choice = value.required_choice.as_ref().map(AsRef::as_ref);
    let phase = value.phase.as_ref();
    let input_phase = ["awaiting_action", "awaiting_response"].contains(&phase)
        || (phase == "resolving" && choice == Some("buy_accept_reject"));
    let can_act = value.current_player == Some(value.viewer_player_id)
        && value.result.is_none()
        && input_phase;
    ensure(
        !(can_act && choice == Some("additional_choice")),
        "additional_choice has no v1 action contract",
    )?;
    let actionable = descriptors(&value.typed_action_schema, choice, can_act, control)?;
    ensure(value.actionable == actionable, "typed metadata mismatch")
}
fn unique_members<T: Ord>(
    entries: &[T],
    allowed: impl Fn(&T) -> bool,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<()> {
    let mut sorted = BudgetVec::with_capacity(budget, entries.len())?;
    for entry in entries {
        control.check()?;
        ensure(
            allowed(entry),
            "duplicate or foreign ActionOptions reference",
        )?;
        sorted.push(entry)?;
    }
    crate::budget::sort_unstable_by(&mut sorted, Ord::cmp, control)?;
    ensure(
        !sorted.windows(2).any(|pair| pair[0] == pair[1]),
        "duplicate or foreign ActionOptions reference",
    )
}
pub fn input(
    value: &Input<'_>,
    mode: Option<bool>,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<()> {
    shape(value)?;
    input_semantics(value, mode, budget, control)
}
pub(crate) fn input_semantics(
    value: &Input<'_>,
    mode: Option<bool>,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<()> {
    let obs = &value.observation;
    observation_semantics(obs, None, budget, control)?;
    ensure(
        value.typed_metadata == obs.typed_action_schema,
        "worker payload boundary mismatch",
    )?;
    // Map keys are perceived references. Inner definition IDs may intentionally differ.
    let mut visible = BudgetVec::with_capacity(
        budget,
        obs.own
            .hand
            .len()
            .checked_add(obs.own.learned_miracles.len())
            .ok_or_else(|| Error::resource("visibility capacity overflow"))?,
    )?;
    for card in &obs.own.hand {
        control.check()?;
        visible.push(&card.definition_id)?;
    }
    for reference in &obs.own.learned_miracles {
        control.check()?;
        visible.push(reference)?;
    }
    for reference in visible.iter() {
        control.check()?;
        ensure(
            value.catalog.0.contains_key(*reference),
            "unknown own-visible definition",
        )?;
    }
    for reference in value.catalog.0.keys() {
        control.check()?;
        ensure(
            visible.iter().any(|visible| *visible == reference),
            "worker payload boundary mismatch",
        )?;
    }
    ensure(
        mode.is_none_or(|expected| expected == value.action_options.is_some()),
        "ActionOptions presence/mode mismatch",
    )?;
    if let Some(options) = &value.action_options {
        ensure(
            obs.actionable && obs.current_player == Some(obs.viewer_player_id),
            "ActionOptions require current actionable actor",
        )?;
        ensure(
            options.viewer_player_id == obs.viewer_player_id,
            "ActionOptions actor mismatch",
        )?;
        ensure(
            options.state_revision == obs.state_revision,
            "ActionOptions revision mismatch",
        )?;
        let body = &options.action_options;
        ensure(
            body.required_choice == obs.required_choice,
            "ActionOptions choice mismatch",
        )?;
        for entries in [
            &body.usable_hand_elements,
            &body.exchange_artifacts,
            &body.sell_artifacts,
            &body.buy_artifacts,
        ] {
            unique_members(
                entries,
                |reference| {
                    obs.own
                        .hand
                        .iter()
                        .any(|card| card.instance_id == *reference)
                },
                budget,
                control,
            )?;
        }
        unique_members(
            &body.usable_learned_miracles,
            |reference| obs.own.learned_miracles.contains(reference),
            budget,
            control,
        )?;
        unique_members(&body.valid_targets, |_| true, budget, control)?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{budget::monotonic, json::Value};
    pub(crate) fn fixture() -> Value {
        let observation: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/observation_p1_public.json"
        ))
        .unwrap();
        let catalog: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/catalog_public_v2.json"
        ))
        .unwrap();
        let mut visible = serde_json::Map::new();
        for artifact in observation["own"]["hand"].as_array().unwrap() {
            let id = artifact["definition_id"].as_str().unwrap();
            let definition = catalog["definitions"]
                .as_array()
                .unwrap()
                .iter()
                .find(|d| d["definition_id"] == id)
                .unwrap();
            visible.insert(id.to_owned(), definition.clone());
        }
        for reference in observation["own"]["learned_miracles"].as_array().unwrap() {
            let id = reference.as_str().unwrap();
            let definition = catalog["definitions"]
                .as_array()
                .unwrap()
                .iter()
                .find(|d| d["definition_id"] == id)
                .unwrap();
            visible.insert(id.to_owned(), definition.clone());
        }
        serde_json::json!({"typed_metadata":observation["typed_action_schema"],"observation":observation,
            "events":[],"history_end":null,"catalog":visible,"action_options":null})
    }
    fn check(value: &Value) -> bool {
        let budget = Budget::new(128 * 1024 * 1024);
        let control = Control::new(monotonic() + 10.).unwrap();
        let wire = serde_json::to_vec(value).unwrap();
        let parsed: Input = match serde_json::from_slice(&wire) {
            Ok(value) => value,
            Err(_) => return false,
        };
        input(&parsed, Some(false), &budget, &control).is_ok()
    }
    #[test]
    fn typed_input_matches_legacy_semantics_and_preserves_perceived_catalog_keys() {
        let original = fixture();
        assert!(check(&original));
        crate::validation::validate_worker_input(&original, Some(false)).unwrap();
        let mut perceived = original.clone();
        let (_, definition) = perceived["catalog"]
            .as_object_mut()
            .unwrap()
            .iter_mut()
            .next()
            .unwrap();
        definition["definition_id"] = Value::String("perceived-inner-id".into());
        assert!(check(&perceived));
        crate::validation::validate_worker_input(&perceived, Some(false)).unwrap();
        let mut cases = Vec::new();
        let mut value = original.clone();
        value["observation"]["viewer_player_id"] = serde_json::json!(2);
        cases.push(value);
        let mut value = original.clone();
        value["observation"]["actionable"] = serde_json::json!(false);
        cases.push(value);
        let mut value = original.clone();
        value["typed_metadata"] = serde_json::json!([]);
        cases.push(value);
        let mut value = original.clone();
        value["catalog"] = serde_json::json!({});
        cases.push(value);
        let mut value = original.clone();
        let duplicate = value["observation"]["own"]["hand"][0].clone();
        value["observation"]["own"]["hand"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        cases.push(value);
        let mut value = original.clone();
        value["observation"]["typed_action_schema"][0]["additional_fields"] =
            serde_json::json!(true);
        cases.push(value);
        for key in ["hp", "mp", "money"] {
            let mut value = original.clone();
            for metadata in [&mut value["observation"]["typed_action_schema"]] {
                let exchange = metadata
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|entry| entry["type"] == "exchange")
                    .unwrap();
                exchange["required_fields"][key]["sources"] = serde_json::json!(["snapshot.rng"]);
            }
            value["typed_metadata"] = value["observation"]["typed_action_schema"].clone();
            cases.push(value);
        }
        for value in cases {
            assert!(!check(&value));
            assert!(crate::validation::validate_worker_input(&value, Some(false)).is_err());
        }
    }
    #[test]
    fn atomic_hints_bind_actor_revision_visible_refs_uniqueness_and_mode() {
        let mut value = fixture();
        let id = value["observation"]["own"]["hand"][0]["instance_id"].clone();
        value["action_options"] = serde_json::json!({"viewer_player_id":value["observation"]["viewer_player_id"],"state_revision":value["observation"]["state_revision"],"action_options":{
            "required_choice":value["observation"]["required_choice"],"usable_hand_elements":[id],"usable_learned_miracles":[],
            "exchange_artifacts":[],"sell_artifacts":[],"buy_artifacts":[],"valid_targets":[1,2],"can_pray":false,"can_discard":true,"can_sacrifice":true,"can_allow":false}});
        let valid = |value: &Value, mode: bool| {
            let budget = Budget::new(128 * 1024 * 1024);
            let control = Control::new(monotonic() + 10.).unwrap();
            let wire = serde_json::to_vec(value).unwrap();
            let typed: Input = serde_json::from_slice(&wire).unwrap();
            (
                input(&typed, Some(mode), &budget, &control).is_ok(),
                crate::validation::validate_worker_input(value, Some(mode)).is_ok(),
            )
        };
        assert_eq!(valid(&value, true), (true, true));
        assert_eq!(valid(&value, false), (false, false));
        let mut mutations = Vec::new();
        let mut v = value.clone();
        v["action_options"]["viewer_player_id"] = serde_json::json!(2);
        mutations.push(v);
        let mut v = value.clone();
        v["action_options"]["state_revision"] = serde_json::json!(u64::MAX);
        mutations.push(v);
        let mut v = value.clone();
        v["action_options"]["action_options"]["usable_hand_elements"] =
            serde_json::json!([u64::MAX]);
        mutations.push(v);
        let mut v = value.clone();
        let id = value["observation"]["own"]["hand"][0]["instance_id"].clone();
        v["action_options"]["action_options"]["usable_hand_elements"] = serde_json::json!([id, id]);
        mutations.push(v);
        let mut v = value.clone();
        v["action_options"]["action_options"]["usable_learned_miracles"] =
            serde_json::json!(["foreign"]);
        mutations.push(v);
        let mut v = value.clone();
        v["action_options"]["action_options"]["required_choice"] = serde_json::json!("defense");
        mutations.push(v);
        for mutation in mutations {
            assert_eq!(valid(&mutation, true), (false, false));
        }
    }
}
