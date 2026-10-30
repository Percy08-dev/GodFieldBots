//! Concrete borrowed DTO traversal. Shared parent sinks, bucket/hash math and
//! recipe configuration preserve feature order and bits without a Value adapter.
use super::*;
use crate::typed_journal::{dto::contract::Validate, dto::text::Text, dto::*, required};

struct Lookup<'a, 'raw> {
    input: &'a Input<'raw>,
    hand: BudgetVec<(u64, &'a Definition<'raw>)>,
    learned: BudgetVec<&'a Text<'raw>>,
    usable_hand: BudgetVec<u64>,
    usable_learned: BudgetVec<&'a Text<'raw>>,
}
impl<'a, 'raw> Lookup<'a, 'raw> {
    fn new(input: &'a Input<'raw>, budget: &Arc<Budget>, control: &Control) -> Result<Self> {
        let own = &input.observation.own;
        let mut hand = BudgetVec::with_capacity(budget, own.hand.len())?;
        for card in &own.hand {
            control.check()?;
            hand.push((
                card.instance_id,
                input
                    .catalog
                    .0
                    .get(&card.definition_id)
                    .ok_or_else(|| Error::contract("unknown own-visible definition"))?,
            ))?;
        }
        crate::budget::sort_unstable_by(&mut hand, |a, b| a.0.cmp(&b.0), control)?;
        let mut learned = BudgetVec::with_capacity(budget, own.learned_miracles.len())?;
        for reference in &own.learned_miracles {
            control.check()?;
            learned.push(reference)?;
        }
        crate::budget::sort_unstable_by(&mut learned, Ord::cmp, control)?;
        let body = input.action_options.as_ref().map(|v| &v.action_options);
        let mut usable_hand =
            BudgetVec::with_capacity(budget, body.map_or(0, |b| b.usable_hand_elements.len()))?;
        let mut usable_learned =
            BudgetVec::with_capacity(budget, body.map_or(0, |b| b.usable_learned_miracles.len()))?;
        if let Some(body) = body {
            for key in &body.usable_hand_elements {
                control.check()?;
                usable_hand.push(*key)?;
            }
            for key in &body.usable_learned_miracles {
                control.check()?;
                usable_learned.push(key)?;
            }
        }
        crate::budget::sort_unstable_by(&mut usable_hand, Ord::cmp, control)?;
        crate::budget::sort_unstable_by(&mut usable_learned, Ord::cmp, control)?;
        Ok(Self {
            input,
            hand,
            learned,
            usable_hand,
            usable_learned,
        })
    }
    fn card(&self, reference: &Text<'raw>) -> Result<&'a Definition<'raw>> {
        self.input
            .catalog
            .0
            .get(reference)
            .ok_or_else(|| Error::contract("unknown own-visible definition"))
    }
    fn hand(&self, key: u64) -> Result<Reference<'a, 'raw>> {
        let index = self
            .hand
            .binary_search_by_key(&key, |entry| entry.0)
            .map_err(|_| Error::contract("foreign encoded hand reference"))?;
        Ok(Reference {
            card: self.hand[index].1,
            key: Key::Hand(key),
        })
    }
    fn element(&self, value: &'a Element<'raw>) -> Result<Reference<'a, 'raw>> {
        match value.source.as_ref() {
            "hand_artifact" => self.hand(*required(&value.instance_id)?),
            "learned_miracle" => {
                let key = required(&value.definition_id)?;
                if self.learned.binary_search(&key).is_err() {
                    return Err(Error::contract("foreign encoded learned reference"));
                }
                Ok(Reference {
                    card: self.card(key)?,
                    key: Key::Learned(key),
                })
            }
            _ => Err(Error::contract("unknown encoded source")),
        }
    }
}
#[derive(Clone, Copy)]
enum Key<'a, 'raw> {
    Hand(u64),
    Learned(&'a Text<'raw>),
}
#[derive(Clone, Copy)]
struct Reference<'a, 'raw> {
    card: &'a Definition<'raw>,
    key: Key<'a, 'raw>,
}
fn references<'a, 'raw>(
    lookup: &Lookup<'a, 'raw>,
    action: &'a Action<'raw>,
    config: &FeatureConfig,
) -> Result<([Option<Reference<'a, 'raw>>; 3], usize)> {
    let mut output = [None; 3];
    let tag = action.r#type.as_ref();
    let count = match tag {
        "play_sequence" => {
            let entries = required(&action.ordered_elements)?;
            if entries.len() > config.max_elements {
                return Err(Error::contract("candidate exceeds encoder capacity"));
            }
            for (slot, value) in output.iter_mut().zip(entries) {
                *slot = Some(lookup.element(value)?);
            }
            entries.len()
        }
        "respond" | "respond_sequence" => {
            let entries = required(&action.ordered_defenses)?;
            if entries.len() > config.max_elements {
                return Err(Error::contract("candidate exceeds encoder capacity"));
            }
            for (slot, value) in output.iter_mut().zip(entries) {
                *slot = Some(match (tag, value) {
                    ("respond", Defense::Instance(key)) => lookup.hand(*key)?,
                    ("respond_sequence", Defense::Element(value)) => lookup.element(value)?,
                    _ => return Err(Error::contract("ordered defense tag/shape mismatch")),
                });
            }
            entries.len()
        }
        "discard" | "sacrifice" => {
            let entries = required(&action.instance_ids)?;
            if entries.len() > config.max_elements {
                return Err(Error::contract("candidate exceeds encoder capacity"));
            }
            for (slot, key) in output.iter_mut().zip(entries) {
                *slot = Some(lookup.hand(*key)?);
            }
            entries.len()
        }
        "exchange" | "sell" | "buy" => {
            output[0] = Some(lookup.hand(*required(&action.artifact)?)?);
            if tag == "sell" {
                output[1] = Some(lookup.hand(*required(&action.sold_artifact)?)?);
                2
            } else {
                1
            }
        }
        "pray" | "buy_decision" => 0,
        _ => return Err(Error::contract("unsupported encoded action")),
    };
    Ok((output, count))
}
fn numbers(card: &Definition<'_>) -> [Option<i128>; 6] {
    [
        card.attack.value().map(|v| *v as i128),
        card.defense.value().map(|v| *v as i128),
        card.ability_value.value().map(|v| *v as i128),
        card.price.value().map(|v| *v as i128),
        card.mp_cost.value().map(|v| *v as i128),
        card.hit_rate.value().map(|v| *v as i128),
    ]
}
fn definition(
    sink: &mut impl Sink,
    prefix: Prefix<'_>,
    card: Option<&Definition<'_>>,
    config: &FeatureConfig,
) -> Result<()> {
    let kind = card.map(|v| v.kind.as_ref());
    category_scalar(sink, prefix, &[".kind"], kind, kind.is_none(), KINDS)?;
    let attribute = card.and_then(|v| v.attribute.value()).map(AsRef::as_ref);
    category_scalar(
        sink,
        prefix,
        &[".attribute"],
        attribute,
        attribute.is_none(),
        ATTRIBUTES,
    )?;
    let values = card.map_or([None; 6], numbers);
    for (key, value) in NUMERIC.iter().zip(values) {
        optional_scalar(sink, prefix, key, value.map(|v| v as f64))?;
    }
    let additional = card.and_then(|v| v.additional_attack.value());
    sink.put(
        prefix,
        &[".additional_attack"],
        None,
        additional.map_or(0., |v| u8::from(*v) as f64),
    )?;
    sink.put(
        prefix,
        &[".additional_attack.missing"],
        None,
        u8::from(additional.is_none()) as f64,
    )?;
    sink.hash(
        prefix,
        &[".definition."],
        card.into_iter().map(|v| Ok(v.definition_id.as_ref())),
        config.hash_buckets,
    )?;
    sink.hash(
        prefix,
        &[".ability."],
        card.and_then(|v| v.ability.value())
            .into_iter()
            .map(|v| Ok(v.as_ref())),
        config.hash_buckets,
    )
}
struct Participant<'a, 'raw> {
    resources: [Option<f64>; 3],
    ascended: bool,
    status: Option<&'a Status<'raw>>,
    guardian_present: bool,
    guardian: Option<&'a str>,
    learned: usize,
}
fn calculate<'a, 'raw>(
    input: &'a Input<'raw>,
    action: Option<&'a Action<'raw>>,
    config: &FeatureConfig,
    sink: &mut impl Sink,
    control: &Control,
    common: bool,
    lookup: &Lookup<'a, 'raw>,
) -> Result<()> {
    control.check()?;
    let obs = &input.observation;
    let own = &obs.own;
    let opponent = &obs.opponent;
    let viewer = own.player_id.0;
    let options = input.action_options.as_ref().map(|v| &v.action_options);
    if common {
        let participants = [
            (
                "own",
                Participant {
                    resources: [
                        Some(own.hp as f64),
                        Some(own.mp as f64),
                        Some(own.money as f64),
                    ],
                    ascended: own.ascended,
                    status: Some(&own.status),
                    guardian_present: true,
                    guardian: own.guardian.as_ref().map(AsRef::as_ref),
                    learned: own.learned_miracles.len(),
                },
            ),
            (
                "opponent",
                Participant {
                    resources: [
                        opponent.hp.value().map(|v| *v as f64),
                        opponent.mp.value().map(|v| *v as f64),
                        opponent.money.value().map(|v| *v as f64),
                    ],
                    ascended: opponent.ascended,
                    status: opponent.status.value(),
                    guardian_present: opponent.guardian.present(),
                    guardian: opponent.guardian.value().map(AsRef::as_ref),
                    learned: opponent.learned_miracles.len(),
                },
            ),
        ];
        for (name, participant) in participants {
            let prefix = Prefix::new(name);
            for (key, value) in ["hp", "mp", "money"].iter().zip(participant.resources) {
                optional_scalar(sink, prefix, key, value)?;
            }
            sink.put(
                prefix,
                &[".ascended"],
                None,
                u8::from(participant.ascended) as f64,
            )?;
            sink.put(
                prefix,
                &[".status.missing"],
                None,
                u8::from(participant.status.is_none()) as f64,
            )?;
            let flags = participant
                .status
                .map_or([false; 4], |s| [s.fog, s.flash, s.dream, s.dark_cloud]);
            for (key, value) in ["fog", "flash", "dream", "dark_cloud"].iter().zip(flags) {
                sink.put(prefix, &[".", key], None, u8::from(value) as f64)?;
            }
            let disease = participant
                .status
                .and_then(|s| s.disease.as_ref())
                .map(AsRef::as_ref);
            category_scalar(
                sink,
                prefix,
                &[".disease"],
                disease,
                disease.is_none(),
                &["cold", "fever", "hell", "heaven"],
            )?;
            sink.put(
                prefix,
                &[".guardian.missing"],
                None,
                u8::from(!participant.guardian_present) as f64,
            )?;
            sink.hash(
                prefix,
                &[".guardian."],
                participant.guardian.into_iter().map(Ok),
                config.hash_buckets,
            )?;
            sink.put(
                prefix,
                &[".learned_count"],
                None,
                participant.learned as f64,
            )?;
        }
        for (key, value) in [
            ("turn", obs.turn as f64),
            (
                "turns_until_end_of_world",
                obs.turns_until_end_of_world as f64,
            ),
            ("end_times", u8::from(obs.end_times) as f64),
        ] {
            sink.put(Prefix::new(key), &[], None, value)?;
        }
        for (key, value) in [
            ("current_player", obs.current_player.map(|v| v.0)),
            ("active_player", Some(obs.active_player.0)),
            ("first_player", Some(obs.first_player.0)),
        ] {
            sink.put(
                Prefix::new(key),
                &[".own"],
                None,
                u8::from(value == Some(viewer)) as f64,
            )?;
            sink.put(
                Prefix::new(key),
                &[".opponent"],
                None,
                u8::from(value == Some(3 - viewer)) as f64,
            )?;
            sink.put(
                Prefix::new(key),
                &[".missing"],
                None,
                u8::from(value.is_none()) as f64,
            )?;
        }
        let choice = obs.required_choice.as_ref().map(AsRef::as_ref);
        category_scalar(
            sink,
            Prefix::new("choice"),
            &[],
            choice,
            choice.is_none(),
            &["turn_action", "defense", "buy_accept_reject"],
        )?;
        for (name, count) in [
            ("hand", own.hand.len()),
            ("learned", own.learned_miracles.len()),
        ] {
            let prefix = Prefix::new(name);
            let card = |index: usize| -> Result<&Definition<'raw>> {
                control.check()?;
                lookup.card(if name == "hand" {
                    &own.hand[index].definition_id
                } else {
                    &own.learned_miracles[index]
                })
            };
            sink.put(prefix, &[".count"], None, count as f64)?;
            for kind in KINDS {
                let mut total = 0;
                for index in 0..count {
                    total += usize::from(card(index)?.kind.as_ref() == *kind);
                }
                sink.put(prefix, &[".kind.", kind], None, total as f64)?;
            }
            for (number_index, key) in NUMERIC.iter().enumerate() {
                let mut sum = 0i128;
                let mut missing = 0usize;
                for index in 0..count {
                    let value = numbers(card(index)?)[number_index];
                    sum += value.unwrap_or(0);
                    missing += usize::from(value.is_none());
                }
                sink.put(prefix, &[".", key], None, sum as f64)?;
                sink.put(prefix, &[".", key, ".missing"], None, missing as f64)?;
            }
            sink.hash(
                prefix,
                &[".definitions."],
                (0..count).map(|index| Ok(card(index)?.definition_id.as_ref())),
                config.hash_buckets,
            )?;
            sink.hash(
                prefix,
                &[".abilities."],
                (0..count).filter_map(|index| match card(index) {
                    Ok(card) => card.ability.value().map(|value| Ok(value.as_ref())),
                    Err(error) => Some(Err(error)),
                }),
                config.hash_buckets,
            )?;
        }
        sink.put(
            Prefix::new("hints"),
            &[".missing"],
            None,
            u8::from(options.is_none()) as f64,
        )?;
        let flags = options.map_or([false; 4], |o| {
            [o.can_pray, o.can_discard, o.can_sacrifice, o.can_allow]
        });
        for (key, value) in ["can_pray", "can_discard", "can_sacrifice", "can_allow"]
            .iter()
            .zip(flags)
        {
            sink.put(
                Prefix::new("hints"),
                &[".", key],
                None,
                u8::from(value) as f64,
            )?;
        }
        let counts = options.map_or([0; 6], |o| {
            [
                o.usable_hand_elements.len(),
                o.usable_learned_miracles.len(),
                o.exchange_artifacts.len(),
                o.sell_artifacts.len(),
                o.buy_artifacts.len(),
                o.valid_targets.len(),
            ]
        });
        for (key, count) in [
            "usable_hand_elements",
            "usable_learned_miracles",
            "exchange_artifacts",
            "sell_artifacts",
            "buy_artifacts",
            "valid_targets",
        ]
        .iter()
        .zip(counts)
        {
            sink.put(
                Prefix::new("hints"),
                &[".", key, ".count"],
                None,
                count as f64,
            )?;
        }
    }
    let tag = action.map_or("pray", |a| a.r#type.as_ref());
    for action_tag in ACTIONS {
        sink.put(
            Prefix::new("action"),
            &[".", action_tag],
            None,
            u8::from(tag == *action_tag) as f64,
        )?;
    }
    sink.put(Prefix::new("action"), &[".missing"], None, 0.)?;
    let target = action.and_then(|a| a.target.value()).map(|v| v.0);
    for (key, value) in [
        (".own", target == Some(viewer)),
        (".opponent", target == Some(3 - viewer)),
        (".unspecified", target.is_none()),
    ] {
        sink.put(Prefix::new("target"), &[key], None, u8::from(value) as f64)?;
    }
    let action_values = action.map_or([None; 4], |a| {
        [
            a.hp.value().map(|v| *v as f64),
            a.mp.value().map(|v| *v as f64),
            a.money.value().map(|v| *v as f64),
            a.accept.value().map(|v| u8::from(*v) as f64),
        ]
    });
    for (key, value) in ["hp", "mp", "money", "accept"].iter().zip(action_values) {
        optional_scalar(sink, Prefix::new("action"), key, value)?;
    }
    let (refs, count) = if let Some(action) = action {
        references(lookup, action, config)?
    } else {
        ([None; 3], 0)
    };
    sink.put(Prefix::new("action"), &[".elements"], None, count as f64)?;
    for (position, reference) in refs.iter().enumerate().take(config.max_elements) {
        let prefix = Prefix {
            root: "element",
            position: Some(position),
        };
        sink.put(
            prefix,
            &[".present"],
            None,
            u8::from(reference.is_some()) as f64,
        )?;
        sink.put(
            prefix,
            &[".hand"],
            None,
            u8::from(reference.is_some_and(|r| matches!(r.key, Key::Hand(_)))) as f64,
        )?;
        sink.put(
            prefix,
            &[".learned"],
            None,
            u8::from(reference.is_some_and(|r| matches!(r.key, Key::Learned(_)))) as f64,
        )?;
        definition(sink, prefix, reference.map(|r| r.card), config)?;
        let usable = options.is_some()
            && reference.is_some_and(|reference| match reference.key {
                Key::Hand(key) => lookup.usable_hand.binary_search(&key).is_ok(),
                Key::Learned(key) => lookup.usable_learned.binary_search(&key).is_ok(),
            });
        sink.put(prefix, &[".usable"], None, u8::from(usable) as f64)?;
    }
    if common && config.history == "public_aggregate" {
        let all = &input.events;
        let events = &all[all.len().saturating_sub(config.history_window)..];
        let prefix = Prefix::new("history");
        sink.put(
            prefix,
            &[".window_truncated"],
            None,
            u8::from(all.len() > config.history_window) as f64,
        )?;
        sink.put(prefix, &[".count"], None, events.len() as f64)?;
        for tag in EVENTS {
            let mut count = 0;
            for event in events {
                control.check()?;
                count += usize::from(event.r#type.as_ref() == *tag);
            }
            sink.put(prefix, &[".type.", tag], None, count as f64)?;
        }
        for (role, participant) in [("own", viewer), ("opponent", 3 - viewer)] {
            let selected = || {
                events.iter().filter(|event| {
                    event
                        .player
                        .value()
                        .or_else(|| event.attacker.value())
                        .or_else(|| event.active_player.value())
                        .is_some_and(|p| p.0 == participant)
                })
            };
            let mut sums = [0i128; 7];
            let mut count = 0;
            for event in selected() {
                control.check()?;
                count += 1;
                let values = [
                    event.amount.value().map(|v| *v as i128),
                    event.hp_delta.value().map(|v| *v as i128),
                    event.mp_delta.value().map(|v| *v as i128),
                    event.money_delta.value().map(|v| *v as i128),
                    event.mp_spent.value().map(|v| *v as i128),
                    event.price.value().map(|v| *v as i128),
                    event.attack.value().map(|v| *v as i128),
                ];
                for (sum, value) in sums.iter_mut().zip(values) {
                    *sum += value.unwrap_or(0);
                }
            }
            sink.put(prefix, &[".", role, ".count"], None, count as f64)?;
            for (key, sum) in [
                "amount",
                "hp_delta",
                "mp_delta",
                "money_delta",
                "mp_spent",
                "price",
                "attack",
            ]
            .iter()
            .zip(sums)
            {
                sink.put(prefix, &[".", role, ".", key], None, sum as f64)?;
            }
        }
        sink.hash(
            prefix,
            &[".definitions."],
            events
                .iter()
                .filter_map(|event| event.definition_id.value().map(|v| Ok(v.as_ref()))),
            config.hash_buckets,
        )?;
    }
    Ok(())
}

pub struct PreparedDecision<'a, 'raw> {
    input: &'a Input<'raw>,
    config: &'a FeatureConfig,
    base: BudgetVec<f64>,
    lookup: Lookup<'a, 'raw>,
    action_start: usize,
    action_end: usize,
}
impl<'a, 'raw> PreparedDecision<'a, 'raw> {
    pub fn new(
        input: &'a Input<'raw>,
        config: &'a FeatureConfig,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<Self> {
        config.validate()?;
        control.check()?;
        let lookup = Lookup::new(input, budget, control)?;
        let mut base = BudgetVec::with_capacity(budget, config.width())?;
        for _ in 0..config.width() {
            base.push(0.)?;
        }
        let mut sink = NumericSink {
            output: &mut base,
            position: 0,
            control: Some(control),
        };
        calculate(input, None, config, &mut sink, control, true, &lookup)?;
        if sink.position != config.width() {
            return Err(Error::contract("invalid encoder vector width"));
        }
        let action_start = 107 + 6 * config.hash_buckets;
        let action_end = action_start + 23 + config.max_elements * (34 + 2 * config.hash_buckets);
        Ok(Self {
            input,
            config,
            base,
            lookup,
            action_start,
            action_end,
        })
    }
    pub fn state(&self, output: &mut [f64], control: &Control) -> Result<()> {
        control.check()?;
        if output.len() != self.base.len() {
            return Err(Error::contract("invalid encoder state vector width"));
        }
        output.copy_from_slice(&self.base);
        Ok(())
    }
    pub fn encode(
        &self,
        action: &'a Action<'raw>,
        output: &mut [f64],
        control: &Control,
    ) -> Result<()> {
        control.check()?;
        action.validate().map_err(Error::contract)?;
        if output.len() != self.base.len() {
            return Err(Error::contract("invalid encoder vector width"));
        }
        output.copy_from_slice(&self.base);
        let mut sink = NumericSink {
            output: &mut output[self.action_start..self.action_end],
            position: 0,
            control: Some(control),
        };
        calculate(
            self.input,
            Some(action),
            self.config,
            &mut sink,
            control,
            false,
            &self.lookup,
        )?;
        if sink.position != sink.output.len() {
            return Err(Error::contract("invalid encoder vector width"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{budget::monotonic, json::Value};
    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }
    #[test]
    fn both_recipes_all_action_forms_and_prepared_states_match_legacy_bits() {
        let base = crate::typed_journal::validation::tests::fixture();
        let mut variants = vec![base.clone()];
        let mut fog = base.clone();
        for key in ["hp", "mp", "money", "status", "guardian"] {
            fog["observation"]["opponent"]
                .as_object_mut()
                .unwrap()
                .remove(key);
        }
        variants.push(fog);
        let mut rich = base.clone();
        let catalog: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/catalog_public_v2.json"
        ))
        .unwrap();
        let miracle = catalog["definitions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|card| card["kind"] == "miracles")
            .unwrap()
            .clone();
        let reference = miracle["definition_id"].as_str().unwrap().to_owned();
        rich["observation"]["own"]["learned_miracles"]
            .as_array_mut()
            .unwrap()
            .push(Value::String(reference.clone()));
        rich["catalog"]
            .as_object_mut()
            .unwrap()
            .insert(reference.clone(), miracle);
        let first_key = rich["observation"]["own"]["hand"][0]["definition_id"]
            .as_str()
            .unwrap()
            .to_owned();
        rich["catalog"][&first_key]["definition_id"] = serde_json::json!("perceived-inner-id");
        rich["catalog"][&first_key]["ability"] = serde_json::json!("escaped \"ability\"");
        let ids: Vec<_> = rich["observation"]["own"]["hand"]
            .as_array()
            .unwrap()
            .iter()
            .map(|card| card["instance_id"].clone())
            .collect();
        rich["action_options"] = serde_json::json!({"viewer_player_id":rich["observation"]["viewer_player_id"],"state_revision":rich["observation"]["state_revision"],"action_options":{
            "required_choice":rich["observation"]["required_choice"],"usable_hand_elements":ids,"usable_learned_miracles":[reference],
            "exchange_artifacts":[],"sell_artifacts":[],"buy_artifacts":[],"valid_targets":[1,2],"can_pray":true,"can_discard":true,"can_sacrifice":false,"can_allow":false}});
        rich["events"] = serde_json::json!([
            {"type":"turn_started","turn":1,"active_player":1},
            {"type":"damage_applied","player":2,"amount":12},
            {"type":"miracle_used","player":1,"definition_id":reference,"mp_spent":3},
            {"type":"resource_changed","player":2,"hp_delta":-1,"mp_delta":2,"money_delta":3}]);
        variants.push(rich);
        for value in variants {
            let budget = Budget::new(128 * 1024 * 1024);
            let control = Control::new(monotonic() + 10.).unwrap();
            crate::validation::validate_worker_input(&value, None).unwrap();
            let wire = serde_json::to_vec(&value).unwrap();
            let input: Input = serde_json::from_slice(&wire).unwrap();
            crate::typed_journal::validation::input(&input, None, &budget, &control).unwrap();
            let id = input.observation.own.hand[0].instance_id;
            let mut actions = vec![
                serde_json::json!({"type":"pray"}),
                serde_json::json!({"type":"play_sequence","ordered_elements":[{"source":"hand_artifact","instance_id":id}],"target":null}),
                serde_json::json!({"type":"respond","ordered_defenses":[]}),
                serde_json::json!({"type":"respond","ordered_defenses":[id]}),
                serde_json::json!({"type":"respond_sequence","ordered_defenses":[{"source":"hand_artifact","instance_id":id}]}),
                serde_json::json!({"type":"discard","instance_ids":[id]}),
                serde_json::json!({"type":"sacrifice","instance_ids":[id]}),
                serde_json::json!({"type":"exchange","artifact":id,"hp":-2,"mp":3,"money":4}),
                serde_json::json!({"type":"sell","artifact":id,"sold_artifact":id,"target":2}),
                serde_json::json!({"type":"buy","artifact":id,"target":1}),
                serde_json::json!({"type":"buy_decision","accept":false}),
            ];
            if let Some(learned) = input.observation.own.learned_miracles.first() {
                actions.push(serde_json::json!({"type":"play_sequence","ordered_elements":[{"source":"learned_miracle","definition_id":learned.as_ref()}],"target":2}));
            }
            let action_wires: Vec<_> = actions
                .iter()
                .map(|action| serde_json::to_vec(action).unwrap())
                .collect();
            let typed_actions: Vec<Action> = action_wires
                .iter()
                .map(|wire| serde_json::from_slice(wire).unwrap())
                .collect();
            for history in ["none", "public_aggregate"] {
                let config = FeatureConfig {
                    history: history.to_owned(),
                    max_elements: 3,
                    hash_buckets: 32,
                    history_window: 2,
                };
                let old = crate::feature::PreparedDecision::new(&value, &config, &budget, &control)
                    .unwrap();
                let new = PreparedDecision::new(&input, &config, &budget, &control).unwrap();
                let (mut a, mut b) = (vec![0.; config.width()], vec![0.; config.width()]);
                old.state(&mut a, &control).unwrap();
                new.state(&mut b, &control).unwrap();
                assert_eq!(bits(&a), bits(&b), "state {history}");
                for (action, typed) in actions.iter().zip(&typed_actions) {
                    old.encode(action, &mut a, &control).unwrap();
                    new.encode(typed, &mut b, &control).unwrap();
                    assert_eq!(bits(&a), bits(&b), "{history}: {action}");
                }
                let foreign = serde_json::json!({"type":"discard","instance_ids":[u64::MAX]});
                let wire = serde_json::to_vec(&foreign).unwrap();
                let typed: Action = serde_json::from_slice(&wire).unwrap();
                assert!(old.encode(&foreign, &mut a, &control).is_err());
                assert!(new.encode(&typed, &mut b, &control).is_err());
            }
            assert_eq!(budget.used(), 0);
        }
    }
}
