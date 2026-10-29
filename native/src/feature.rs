//! Pure participant/action float64 recipe. No learning transforms or simulator rules.
use crate::budget::{Budget, BudgetVec, Control, Lease};
use crate::error::{Error, Result};
use crate::json::Value;
use crate::validation::{ATTRIBUTES, KINDS, NUMERIC};
use sha2::{Digest, Sha256};
use std::fmt::Write;
use std::ops::Deref;
use std::sync::Arc;
fn poll(control: Option<&Control>) -> Result<()> {
    if let Some(control) = control {
        control.check()?;
    }
    Ok(())
}

const ACTIONS: &[&str] = &[
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
const EVENTS: &[&str] = &[
    "turn_started",
    "action_accepted",
    "artifact_used",
    "artifact_drawn",
    "devil_appeared",
    "artifact_discarded",
    "guardian_set",
    "guardian_left",
    "guardian_acted",
    "miracle_used",
    "attack_declared",
    "attack_missed",
    "damage_applied",
    "healed",
    "resource_changed",
    "buy_offered",
    "sale_offered",
    "artifact_transferred",
    "sun_amulet_activated",
    "status_changed",
    "phenomenon_invoked",
    "player_ascended",
    "player_defeated",
    "game_ended",
];

#[derive(Clone, Debug)]
pub struct FeatureConfig {
    pub history: String,
    pub max_elements: usize,
    pub hash_buckets: usize,
    pub history_window: usize,
}
impl FeatureConfig {
    pub fn from_value(value: &Value) -> Result<Self> {
        value.closed(
            &["history", "max_elements", "hash_buckets", "history_window"],
            &[],
        )?;
        let history = value.require("history")?.as_str()?;
        if !["none", "public_aggregate"].contains(&history) {
            return Err(Error::contract("unsupported encoder history"));
        }
        Ok(Self {
            history: history.to_owned(),
            max_elements: value.require("max_elements")?.integer(2, 3)? as usize,
            hash_buckets: value.require("hash_buckets")?.integer(2, 4096)? as usize,
            history_window: value.require("history_window")?.integer(1, 100000)? as usize,
        })
    }
    pub fn validate(&self) -> Result<()> {
        if !["none", "public_aggregate"].contains(&self.history.as_str())
            || !(2..=3).contains(&self.max_elements)
            || !(2..=4096).contains(&self.hash_buckets)
            || !(1..=100000).contains(&self.history_window)
        {
            return Err(Error::contract("invalid encoder config"));
        }
        Ok(())
    }
    pub fn width(&self) -> usize {
        // Common participant 2*(19+B); world/role/choice 16; inventory
        // 2*(21+2B); hints 11; candidate 23+elements*(34+2B).
        130 + 6 * self.hash_buckets
            + self.max_elements * (34 + 2 * self.hash_buckets)
            + if self.history == "public_aggregate" {
                42 + self.hash_buckets
            } else {
                0
            }
    }
}

#[derive(Clone, Copy)]
struct Prefix<'a> {
    root: &'a str,
    position: Option<usize>,
}
impl<'a> Prefix<'a> {
    fn new(root: &'a str) -> Self {
        Self {
            root,
            position: None,
        }
    }
}
trait Sink {
    fn put(
        &mut self,
        prefix: Prefix<'_>,
        suffix: &[&str],
        index: Option<usize>,
        value: f64,
    ) -> Result<()>;
    fn hash<'a>(
        &mut self,
        prefix: Prefix<'_>,
        suffix: &[&str],
        values: impl Iterator<Item = Result<&'a str>>,
        size: usize,
    ) -> Result<()>;
}
fn bucket(value: &str, size: usize) -> usize {
    let hash = Sha256::digest(value.as_bytes());
    (u64::from_be_bytes(hash[..8].try_into().expect("eight digest bytes")) % size as u64) as usize
}
struct NumericSink<'a> {
    output: &'a mut [f64],
    position: usize,
    control: Option<&'a Control>,
}
impl Sink for NumericSink<'_> {
    fn put(&mut self, _: Prefix<'_>, _: &[&str], _: Option<usize>, value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(Error::contract("nonfinite/invalid encoder vector"));
        }
        let slot = self
            .output
            .get_mut(self.position)
            .ok_or_else(|| Error::contract("invalid encoder vector width"))?;
        *slot = value;
        self.position += 1;
        Ok(())
    }
    fn hash<'a>(
        &mut self,
        _: Prefix<'_>,
        _: &[&str],
        values: impl Iterator<Item = Result<&'a str>>,
        size: usize,
    ) -> Result<()> {
        let stop = self
            .position
            .checked_add(size)
            .ok_or_else(|| Error::contract("invalid encoder vector width"))?;
        let slots = self
            .output
            .get_mut(self.position..stop)
            .ok_or_else(|| Error::contract("invalid encoder vector width"))?;
        slots.fill(0.);
        for value in values {
            poll(self.control)?;
            slots[bucket(value?, size)] += 1.;
        }
        self.position = stop;
        Ok(())
    }
}
// All names in the pinned participant-action-v1 recipe are ASCII and at most
// 64 bytes. Fixed capacities avoid uncharged String/Vec growth peaks.
const NAME_CAPACITY: usize = 64;
// CPython 3.14 compact-ASCII strings fit well within 256 bytes at that length.
// Four pointer slots per item cover list allocation growth and replacement
// overlap during explicit append conversion; 1024 covers container headers.
const PYTHON_NAME_BYTES: usize = 256 + 4 * size_of::<usize>();
const PYTHON_LIST_HEADER_BYTES: usize = 1024;

pub struct FeatureNames {
    names: Vec<String>,
    // Includes native buffers AND the simultaneous Python conversion buffers.
    // The binding must keep this owner alive until conversion has succeeded.
    charge: Lease,
}
impl FeatureNames {
    pub fn reserved_bytes(&self) -> usize {
        self.charge.bytes()
    }
}
impl Deref for FeatureNames {
    type Target = [String];
    fn deref(&self) -> &[String] {
        &self.names
    }
}
struct NameSink<'a> {
    output: Vec<String>,
    control: &'a Control,
}
fn decimal_len(mut value: usize) -> usize {
    let mut length = 1;
    while value >= 10 {
        length += 1;
        value /= 10;
    }
    length
}
impl Sink for NameSink<'_> {
    fn put(
        &mut self,
        prefix: Prefix<'_>,
        suffix: &[&str],
        index: Option<usize>,
        _: f64,
    ) -> Result<()> {
        self.control.check()?;
        let length = prefix.root.len()
            + prefix
                .position
                .map_or(0, |position| 1 + decimal_len(position))
            + suffix.iter().map(|part| part.len()).sum::<usize>()
            + index.map_or(0, decimal_len);
        if length > NAME_CAPACITY
            || !prefix.root.is_ascii()
            || suffix.iter().any(|part| !part.is_ascii())
        {
            return Err(Error::resource(
                "feature name exceeds pinned metadata allocation bound",
            ));
        }
        if self.output.len() == self.output.capacity() {
            return Err(Error::contract("invalid encoder recipe width"));
        }
        let mut name = String::new();
        name.try_reserve_exact(NAME_CAPACITY)
            .map_err(|_| Error::resource("feature name allocation failed"))?;
        if name.capacity() != NAME_CAPACITY {
            return Err(Error::resource(
                "unexpected feature name allocator capacity",
            ));
        }
        name.push_str(prefix.root);
        if let Some(position) = prefix.position {
            write!(&mut name, ".{position}").expect("String formatting");
        }
        for part in suffix {
            name.push_str(part);
        }
        if let Some(index) = index {
            write!(&mut name, "{index}").expect("String formatting");
        }
        self.output.push(name);
        Ok(())
    }
    fn hash<'a>(
        &mut self,
        prefix: Prefix<'_>,
        suffix: &[&str],
        _: impl Iterator<Item = Result<&'a str>>,
        size: usize,
    ) -> Result<()> {
        for index in 0..size {
            self.put(prefix, suffix, Some(index), 0.)?;
        }
        Ok(())
    }
}

fn field<'a>(value: Option<&'a Value>, key: &str) -> Option<&'a Value> {
    value.and_then(|v| v.get(key))
}
fn array(value: Option<&Value>) -> Result<&[Value]> {
    match value {
        Some(v) => v.as_array(),
        None => Ok(&[]),
    }
}
fn string(value: Option<&Value>) -> Option<&str> {
    value.and_then(|v| v.as_str().ok())
}
fn number(value: Option<&Value>) -> Result<f64> {
    match value {
        None => Ok(0.),
        Some(Value::Bool(v)) => Ok(u8::from(*v) as f64),
        Some(v) => v.number(),
    }
}
fn integer(value: Option<&Value>) -> Result<i128> {
    match value {
        None => Ok(0),
        Some(v) => v.integer(i32::MIN as i128, u64::MAX as i128),
    }
}
fn optional(
    sink: &mut impl Sink,
    prefix: Prefix<'_>,
    source: Option<&Value>,
    key: &str,
) -> Result<()> {
    let value = field(source, key);
    sink.put(prefix, &[".", key], None, number(value)?)?;
    sink.put(
        prefix,
        &[".", key, ".missing"],
        None,
        u8::from(value.is_none()) as f64,
    )
}
fn category(
    sink: &mut impl Sink,
    prefix: Prefix<'_>,
    suffix: &[&str],
    value: Option<&Value>,
    categories: &[&str],
) -> Result<()> {
    // Stack-only composition: suffix lengths in this recipe never exceed two.
    for category in categories {
        let parts = [
            suffix.first().copied().unwrap_or(""),
            suffix.get(1).copied().unwrap_or(""),
            ".",
            category,
        ];
        sink.put(
            prefix,
            &parts,
            None,
            u8::from(string(value) == Some(category)) as f64,
        )?;
    }
    let parts = [
        suffix.first().copied().unwrap_or(""),
        suffix.get(1).copied().unwrap_or(""),
        ".missing",
    ];
    sink.put(
        prefix,
        &parts,
        None,
        u8::from(value.is_none_or(Value::is_null)) as f64,
    )
}
fn definition(
    sink: &mut impl Sink,
    prefix: Prefix<'_>,
    card: Option<&Value>,
    config: &FeatureConfig,
) -> Result<()> {
    category(sink, prefix, &[".kind"], field(card, "kind"), KINDS)?;
    category(
        sink,
        prefix,
        &[".attribute"],
        field(card, "attribute"),
        ATTRIBUTES,
    )?;
    for key in NUMERIC {
        optional(sink, prefix, card, key)?;
    }
    let additional = field(card, "additional_attack");
    sink.put(prefix, &[".additional_attack"], None, number(additional)?)?;
    sink.put(
        prefix,
        &[".additional_attack.missing"],
        None,
        u8::from(additional.is_none()) as f64,
    )?;
    sink.hash(
        prefix,
        &[".definition."],
        field(card, "definition_id").into_iter().map(Value::as_str),
        config.hash_buckets,
    )?;
    sink.hash(
        prefix,
        &[".ability."],
        field(card, "ability").into_iter().map(Value::as_str),
        config.hash_buckets,
    )
}
fn bytes(value: &Value) -> Result<&[u8]> {
    if let Value::String(text) = value {
        Ok(text.bytes())
    } else {
        Err(Error::contract("string reference required"))
    }
}
struct Lookup<'a> {
    catalog: BudgetVec<(&'a [u8], &'a Value)>,
    hand: BudgetVec<(i128, &'a Value)>,
    learned: BudgetVec<&'a [u8]>,
    usable_hand: BudgetVec<i128>,
    usable_learned: BudgetVec<&'a [u8]>,
}
impl<'a> Lookup<'a> {
    fn new(input: &'a Value, control: Option<&Control>) -> Result<Self> {
        let Value::Object(entries) = input.require("catalog")? else {
            return Err(Error::contract("catalog object required"));
        };
        let budget = entries.budget();
        let mut catalog = BudgetVec::with_capacity(budget, entries.len())?;
        for (key, card) in entries.iter() {
            poll(control)?;
            catalog.push((key.bytes(), card))?;
        }
        crate::validation::sort_controlled(&mut catalog, control, |a, b| a.0.cmp(b.0))?;
        let own = input.require("observation")?.require("own")?;
        let entries = own.require("hand")?.as_array()?;
        let mut hand = BudgetVec::with_capacity(budget, entries.len())?;
        for card in entries {
            poll(control)?;
            let reference = bytes(card.require("definition_id")?)?;
            let index = catalog
                .binary_search_by(|entry| entry.0.cmp(reference))
                .map_err(|_| Error::contract("unknown own-visible definition"))?;
            hand.push((
                card.require("instance_id")?.integer(0, u64::MAX as i128)?,
                catalog[index].1,
            ))?;
        }
        crate::validation::sort_controlled(&mut hand, control, |a, b| a.0.cmp(&b.0))?;
        let entries = own.require("learned_miracles")?.as_array()?;
        let mut learned = BudgetVec::with_capacity(budget, entries.len())?;
        for reference in entries {
            poll(control)?;
            learned.push(bytes(reference)?)?;
        }
        crate::validation::sort_controlled(&mut learned, control, Ord::cmp)?;
        let options = input
            .get("action_options")
            .filter(|v| !v.is_null())
            .and_then(|v| v.get("action_options"));
        let entries = array(field(options, "usable_hand_elements"))?;
        let mut usable_hand = BudgetVec::with_capacity(budget, entries.len())?;
        for reference in entries {
            poll(control)?;
            usable_hand.push(reference.integer(0, u64::MAX as i128)?)?;
        }
        crate::validation::sort_controlled(&mut usable_hand, control, Ord::cmp)?;
        let entries = array(field(options, "usable_learned_miracles"))?;
        let mut usable_learned = BudgetVec::with_capacity(budget, entries.len())?;
        for reference in entries {
            poll(control)?;
            usable_learned.push(bytes(reference)?)?;
        }
        crate::validation::sort_controlled(&mut usable_learned, control, Ord::cmp)?;
        Ok(Self {
            catalog,
            hand,
            learned,
            usable_hand,
            usable_learned,
        })
    }
    fn card(&self, reference: &Value) -> Result<&'a Value> {
        let reference = bytes(reference)?;
        let index = self
            .catalog
            .binary_search_by(|entry| entry.0.cmp(reference))
            .map_err(|_| Error::contract("unknown own-visible definition"))?;
        Ok(self.catalog[index].1)
    }
}

#[derive(Clone, Copy)]
struct Reference<'a> {
    source: &'static str,
    card: &'a Value,
    key: &'a Value,
}
fn hand_reference<'a>(lookup: &Lookup<'a>, key: &'a Value) -> Result<Reference<'a>> {
    let instance = key.integer(0, u64::MAX as i128)?;
    let index = lookup
        .hand
        .binary_search_by_key(&instance, |entry| entry.0)
        .map_err(|_| Error::contract("foreign encoded hand reference"))?;
    Ok(Reference {
        source: "hand_artifact",
        card: lookup.hand[index].1,
        key,
    })
}
fn typed_reference<'a>(lookup: &Lookup<'a>, value: &'a Value) -> Result<Reference<'a>> {
    match value.require("source")?.as_str()? {
        "hand_artifact" => {
            value.closed(&["source", "instance_id"], &[])?;
            hand_reference(lookup, value.require("instance_id")?)
        }
        "learned_miracle" => {
            value.closed(&["source", "definition_id"], &[])?;
            let key = value.require("definition_id")?;
            if !matches!(key, Value::String(_)) {
                return Err(Error::contract("foreign encoded learned reference"));
            }
            if lookup.learned.binary_search(&bytes(key)?).is_err() {
                return Err(Error::contract("foreign encoded learned reference"));
            }
            Ok(Reference {
                source: "learned_miracle",
                card: lookup.card(key)?,
                key,
            })
        }
        _ => Err(Error::contract("unknown encoded source")),
    }
}
fn references<'a>(
    lookup: &Lookup<'a>,
    action: &'a Value,
    config: &FeatureConfig,
) -> Result<([Option<Reference<'a>>; 3], usize)> {
    let tag = action.require("type")?.as_str()?;
    action.closed(
        match tag {
            "pray" => &["type"],
            "play_sequence" => &["type", "ordered_elements"],
            "respond" | "respond_sequence" => &["type", "ordered_defenses"],
            "discard" | "sacrifice" => &["type", "instance_ids"],
            "exchange" => &["type", "artifact", "hp", "mp", "money"],
            "sell" => &["type", "artifact", "sold_artifact", "target"],
            "buy" => &["type", "artifact", "target"],
            "buy_decision" => &["type", "accept"],
            _ => return Err(Error::contract("unsupported encoded action")),
        },
        if tag == "play_sequence" {
            &["target"]
        } else {
            &[]
        },
    )?;
    if let Some(target) = action.get("target") {
        if !target.is_null() {
            target.integer(1, 2)?;
        }
    }
    if let Some(accept) = action.get("accept") {
        accept.as_bool()?;
    }
    for key in ["hp", "mp", "money"] {
        if let Some(value) = action.get(key) {
            value.integer(
                if key == "hp" { i32::MIN as i128 } else { 0 },
                u32::MAX as i128,
            )?;
        }
    }
    let mut output = [None; 3];
    let elements = match tag {
        "play_sequence" => Some((action.require("ordered_elements")?.as_array()?, true)),
        "respond_sequence" => Some((action.require("ordered_defenses")?.as_array()?, true)),
        "respond" => Some((action.require("ordered_defenses")?.as_array()?, false)),
        "discard" | "sacrifice" => Some((action.require("instance_ids")?.as_array()?, false)),
        _ => None,
    };
    let count;
    if let Some((entries, typed)) = elements {
        count = entries.len();
        if count > config.max_elements {
            return Err(Error::contract("candidate exceeds encoder capacity"));
        }
        for (slot, entry) in output.iter_mut().zip(entries) {
            *slot = Some(if typed {
                typed_reference(lookup, entry)?
            } else {
                hand_reference(lookup, entry)?
            });
        }
    } else if ["exchange", "sell", "buy"].contains(&tag) {
        output[0] = Some(hand_reference(lookup, action.require("artifact")?)?);
        count = if tag == "sell" {
            output[1] = Some(hand_reference(lookup, action.require("sold_artifact")?)?);
            2
        } else {
            1
        };
    } else {
        count = 0;
    }
    Ok((output, count))
}

fn calculate(
    input: Option<&Value>,
    action: Option<&Value>,
    config: &FeatureConfig,
    sink: &mut impl Sink,
    control: Option<&Control>,
    common: bool,
    lookup: Option<&Lookup<'_>>,
) -> Result<()> {
    poll(control)?;
    let obs = field(input, "observation");
    let own = field(obs, "own");
    let opponent = field(obs, "opponent");
    let viewer = field(own, "player_id").map_or(Ok(1), |v| v.integer(1, 2))?;
    let options = field(input, "action_options")
        .filter(|v| !v.is_null())
        .and_then(|v| v.get("action_options"));
    if common {
        for (name, participant) in [("own", own), ("opponent", opponent)] {
            let prefix = Prefix::new(name);
            for key in ["hp", "mp", "money"] {
                optional(sink, prefix, participant, key)?;
            }
            sink.put(
                prefix,
                &[".ascended"],
                None,
                number(field(participant, "ascended"))?,
            )?;
            let status = field(participant, "status");
            sink.put(
                prefix,
                &[".status.missing"],
                None,
                u8::from(status.is_none()) as f64,
            )?;
            for key in ["fog", "flash", "dream", "dark_cloud"] {
                sink.put(prefix, &[".", key], None, number(field(status, key))?)?;
            }
            category(
                sink,
                prefix,
                &[".disease"],
                field(status, "disease"),
                &["cold", "fever", "hell", "heaven"],
            )?;
            let guardian = field(participant, "guardian");
            sink.put(
                prefix,
                &[".guardian.missing"],
                None,
                u8::from(guardian.is_none()) as f64,
            )?;
            sink.hash(
                prefix,
                &[".guardian."],
                guardian
                    .filter(|v| !v.is_null())
                    .into_iter()
                    .map(Value::as_str),
                config.hash_buckets,
            )?;
            sink.put(
                prefix,
                &[".learned_count"],
                None,
                array(field(participant, "learned_miracles"))?.len() as f64,
            )?;
        }
        for key in ["turn", "turns_until_end_of_world", "end_times"] {
            sink.put(Prefix::new(key), &[], None, number(field(obs, key))?)?;
        }
        for key in ["current_player", "active_player", "first_player"] {
            let item = field(obs, key);
            sink.put(
                Prefix::new(key),
                &[".own"],
                None,
                u8::from(item.is_some_and(|v| v.integer(1, 2) == Ok(viewer))) as f64,
            )?;
            sink.put(
                Prefix::new(key),
                &[".opponent"],
                None,
                u8::from(item.is_some_and(|v| v.integer(1, 2) == Ok(3 - viewer))) as f64,
            )?;
            sink.put(
                Prefix::new(key),
                &[".missing"],
                None,
                u8::from(item.is_none_or(Value::is_null)) as f64,
            )?;
        }
        category(
            sink,
            Prefix::new("choice"),
            &[],
            field(obs, "required_choice"),
            &["turn_action", "defense", "buy_accept_reject"],
        )?;
        for (name, entries) in [
            ("hand", array(field(own, "hand"))?),
            ("learned", array(field(own, "learned_miracles"))?),
        ] {
            let prefix = Prefix::new(name);
            let cards = || {
                entries.iter().map(|entry| {
                    poll(control)?;
                    let reference = if name == "hand" {
                        entry.require("definition_id")?
                    } else {
                        entry
                    };
                    lookup
                        .ok_or_else(|| Error::contract("missing feature lookup"))?
                        .card(reference)
                })
            };
            sink.put(prefix, &[".count"], None, entries.len() as f64)?;
            for kind in KINDS {
                let mut count = 0usize;
                for card in cards() {
                    if string(card?.get("kind")) == Some(kind) {
                        count += 1;
                    }
                }
                sink.put(prefix, &[".kind.", kind], None, count as f64)?;
            }
            for key in NUMERIC {
                let mut sum = 0i128;
                let mut missing = 0usize;
                for card in cards() {
                    let value = card?.get(key);
                    sum += integer(value)?;
                    missing += usize::from(value.is_none());
                }
                sink.put(prefix, &[".", key], None, sum as f64)?;
                sink.put(prefix, &[".", key, ".missing"], None, missing as f64)?;
            }
            sink.hash(
                prefix,
                &[".definitions."],
                cards().map(|card| card?.require("definition_id")?.as_str()),
                config.hash_buckets,
            )?;
            sink.hash(
                prefix,
                &[".abilities."],
                cards().filter_map(|card| match card {
                    Ok(card) => card.get("ability").map(Value::as_str),
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
        for key in ["can_pray", "can_discard", "can_sacrifice", "can_allow"] {
            sink.put(
                Prefix::new("hints"),
                &[".", key],
                None,
                number(field(options, key))?,
            )?;
        }
        for key in [
            "usable_hand_elements",
            "usable_learned_miracles",
            "exchange_artifacts",
            "sell_artifacts",
            "buy_artifacts",
            "valid_targets",
        ] {
            sink.put(
                Prefix::new("hints"),
                &[".", key, ".count"],
                None,
                array(field(options, key))?.len() as f64,
            )?;
        }
    }
    let tag = field(action, "type");
    // The template names use the same default pray action as the Python recipe.
    for action_tag in ACTIONS {
        sink.put(
            Prefix::new("action"),
            &[".", action_tag],
            None,
            u8::from(string(tag).unwrap_or("pray") == *action_tag) as f64,
        )?;
    }
    sink.put(Prefix::new("action"), &[".missing"], None, 0.)?;
    let target = field(action, "target");
    sink.put(
        Prefix::new("target"),
        &[".own"],
        None,
        u8::from(target.is_some_and(|v| v.integer(1, 2) == Ok(viewer))) as f64,
    )?;
    sink.put(
        Prefix::new("target"),
        &[".opponent"],
        None,
        u8::from(target.is_some_and(|v| v.integer(1, 2) == Ok(3 - viewer))) as f64,
    )?;
    sink.put(
        Prefix::new("target"),
        &[".unspecified"],
        None,
        u8::from(target.is_none_or(Value::is_null)) as f64,
    )?;
    for key in ["hp", "mp", "money", "accept"] {
        optional(sink, Prefix::new("action"), action, key)?;
    }
    let (refs, count) = if let (Some(_input), Some(action)) = (input, action) {
        references(
            lookup.ok_or_else(|| Error::contract("missing feature lookup"))?,
            action,
            config,
        )?
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
            u8::from(reference.is_some_and(|r| r.source == "hand_artifact")) as f64,
        )?;
        sink.put(
            prefix,
            &[".learned"],
            None,
            u8::from(reference.is_some_and(|r| r.source == "learned_miracle")) as f64,
        )?;
        definition(sink, prefix, reference.map(|r| r.card), config)?;
        let usable =
            if let (Some(_options), Some(reference), Some(lookup)) = (options, reference, lookup) {
                if reference.source == "hand_artifact" {
                    lookup
                        .usable_hand
                        .binary_search(&reference.key.integer(0, u64::MAX as i128)?)
                        .is_ok()
                } else {
                    lookup
                        .usable_learned
                        .binary_search(&bytes(reference.key)?)
                        .is_ok()
                }
            } else {
                false
            };
        sink.put(prefix, &[".usable"], None, u8::from(usable) as f64)?;
    }
    if common && config.history == "public_aggregate" {
        let all = if let Some(events) = field(input, "events") {
            crate::validation::worker_events(events)?
        } else {
            &[]
        };
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
            let mut count = 0usize;
            for event in events {
                poll(control)?;
                count += usize::from(string(event.get("type")) == Some(tag));
            }
            sink.put(prefix, &[".type.", tag], None, count as f64)?;
        }
        for (role, participant) in [("own", viewer), ("opponent", 3 - viewer)] {
            let selected = || {
                events.iter().filter(|event| {
                    event
                        .get("player")
                        .or_else(|| event.get("attacker"))
                        .or_else(|| event.get("active_player"))
                        .is_some_and(|v| v.integer(1, 2) == Ok(participant))
                })
            };
            let mut count = 0usize;
            for _ in selected() {
                poll(control)?;
                count += 1;
            }
            sink.put(prefix, &[".", role, ".count"], None, count as f64)?;
            for key in [
                "amount",
                "hp_delta",
                "mp_delta",
                "money_delta",
                "mp_spent",
                "price",
                "attack",
            ] {
                let mut sum = 0i128;
                for event in selected() {
                    poll(control)?;
                    sum += integer(event.get(key))?;
                }
                sink.put(prefix, &[".", role, ".", key], None, sum as f64)?;
            }
        }
        sink.hash(
            prefix,
            &[".definitions."],
            events
                .iter()
                .filter_map(|event| event.get("definition_id").map(Value::as_str)),
            config.hash_buckets,
        )?;
    }
    Ok(())
}

/// Generate pinned recipe metadata with admission before any name allocation.
/// The returned owner retains both native capacity and Python publication charge;
/// bindings must hold it through explicit list/string conversion and its polls.
pub fn feature_names_controlled(
    config: &FeatureConfig,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<FeatureNames> {
    control.check()?;
    config.validate()?;
    let width = config.width();
    let bytes_per_name = size_of::<String>() + NAME_CAPACITY + PYTHON_NAME_BYTES;
    let bytes = width
        .checked_mul(bytes_per_name)
        .and_then(|n| n.checked_add(PYTHON_LIST_HEADER_BYTES))
        .ok_or_else(|| Error::resource("feature name allocation capacity overflow"))?;
    let charge = budget.reserve(bytes)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(width)
        .map_err(|_| Error::resource("feature metadata allocation failed"))?;
    if output.capacity() != width {
        return Err(Error::resource(
            "unexpected feature metadata allocator capacity",
        ));
    }
    let mut sink = NameSink { output, control };
    calculate(None, None, config, &mut sink, Some(control), true, None)?;
    control.check()?;
    if sink.output.len() != width {
        return Err(Error::contract("invalid encoder recipe width"));
    }
    Ok(FeatureNames {
        names: sink.output,
        charge,
    })
}
/// Encode a borrowed, already-validated actor snapshot into caller-owned storage.
/// Candidate syntax/references and configuration remain checked on every call.
pub fn encode(
    input: &Value,
    action: &Value,
    config: &FeatureConfig,
    output: &mut [f64],
) -> Result<()> {
    encode_impl(input, action, config, output, None)
}
pub fn encode_controlled(
    input: &Value,
    action: &Value,
    config: &FeatureConfig,
    output: &mut [f64],
    control: &Control,
) -> Result<()> {
    encode_impl(input, action, config, output, Some(control))
}
fn encode_impl(
    input: &Value,
    action: &Value,
    config: &FeatureConfig,
    output: &mut [f64],
    control: Option<&Control>,
) -> Result<()> {
    config.validate()?;
    if output.len() != config.width() {
        return Err(Error::contract("invalid encoder vector width"));
    }
    let lookup = Lookup::new(input, control)?;
    let mut sink = NumericSink {
        output,
        position: 0,
        control,
    };
    calculate(
        Some(input),
        Some(action),
        config,
        &mut sink,
        control,
        true,
        Some(&lookup),
    )?;
    if sink.position != sink.output.len() {
        return Err(Error::contract("invalid encoder vector width"));
    }
    Ok(())
}

/// One actor snapshot's common/history features. The owned base uses the same
/// invocation budget and no numeric feature caches escape this decision.
pub struct PreparedDecision<'a> {
    input: &'a Value,
    config: &'a FeatureConfig,
    base: BudgetVec<f64>,
    lookup: Lookup<'a>,
    action_start: usize,
    action_end: usize,
}
impl<'a> PreparedDecision<'a> {
    pub fn state(&self, output: &mut [f64], control: &Control) -> Result<()> {
        control.check()?;
        if output.len() != self.base.len() {
            return Err(Error::contract("invalid encoder state vector width"));
        }
        output.copy_from_slice(&self.base);
        Ok(())
    }
    /// The caller must first validate the owned actor snapshot. No policy input
    /// is rebuilt here; only the recipe's caller-independent numeric base is kept.
    pub fn new(
        input: &'a Value,
        config: &'a FeatureConfig,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<Self> {
        config.validate()?;
        control.check()?;
        let lookup = Lookup::new(input, Some(control))?;
        let mut base = BudgetVec::with_capacity(budget, config.width())?;
        for _ in 0..config.width() {
            base.push(0.)?;
        }
        let mut sink = NumericSink {
            output: &mut base,
            position: 0,
            control: Some(control),
        };
        calculate(
            Some(input),
            None,
            config,
            &mut sink,
            Some(control),
            true,
            Some(&lookup),
        )?;
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
    pub fn encode(&self, action: &Value, output: &mut [f64], control: &Control) -> Result<()> {
        control.check()?;
        if output.len() != self.config.width() {
            return Err(Error::contract("invalid encoder vector width"));
        }
        output.copy_from_slice(&self.base);
        let mut sink = NumericSink {
            output: &mut output[self.action_start..self.action_end],
            position: 0,
            control: Some(control),
        };
        calculate(
            Some(self.input),
            Some(action),
            self.config,
            &mut sink,
            Some(control),
            false,
            Some(&self.lookup),
        )?;
        if sink.position != sink.output.len() {
            return Err(Error::contract("invalid encoder vector width"));
        }
        Ok(())
    }
}
