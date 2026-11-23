//! Bounded lazy current-v5 search; clocks and CPython streams are explicit owners.
use super::{
    Config, Input, Result, ensure, error, hash,
    rng::PythonRng,
    rules::{self, Analysis, Family},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    time::Instant,
};
const SAT: u64 = i64::MAX as u64;
fn add(xs: &[u64]) -> u64 {
    xs.iter().fold(0, |a, b| a.saturating_add(*b).min(SAT))
}
fn mul(xs: &[u64]) -> u64 {
    xs.iter().fold(1, |a, b| a.saturating_mul(*b).min(SAT))
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Costs {
    pub version: String,
    pub context_width: u64,
    pub card_width: u64,
    pub actor_input_width: u64,
    pub actor_hidden_width: u64,
    pub position_capacity: u64,
    pub candidate_chunk: u64,
    pub tail_operations_per_candidate: u64,
    pub tail_working_bytes_per_candidate: u64,
    pub tail_tensor_bytes_per_candidate: u64,
    pub fixed_bytes: u64,
    pub max_preparation_bytes: u64,
    pub max_scoring_operations: u64,
    pub max_storage_bytes: u64,
    pub max_tensor_bytes: u64,
}
impl Costs {
    pub fn parse(v: &Value) -> Result<Self> {
        let c: Self = serde_json::from_value(v.clone()).map_err(|e| error(e.to_string()))?;
        ensure(c.version == "proposal-cost-v1", "numeric cost version")?;
        let d = [
            c.context_width,
            c.card_width,
            c.actor_input_width,
            c.actor_hidden_width,
        ];
        ensure(
            d.iter().all(|x| *x <= 1 << 20)
                && (d.iter().all(|x| *x == 0) || d.iter().all(|x| *x > 0)),
            "numeric dimensions",
        )?;
        ensure(
            c.tail_working_bytes_per_candidate >= c.tail_tensor_bytes_per_candidate
                && [
                    c.tail_operations_per_candidate,
                    c.tail_working_bytes_per_candidate,
                    c.tail_tensor_bytes_per_candidate,
                ]
                .iter()
                .all(|x| *x <= SAT && (c.actor_hidden_width == 0 || *x > 0)),
            "numeric tail coefficients",
        )?;
        ensure(
            (1..=128).contains(&c.position_capacity)
                && (1..=65536).contains(&c.candidate_chunk)
                && c.fixed_bytes <= 12 << 30
                && (1..=12 << 30).contains(&c.max_preparation_bytes)
                && (1..=SAT).contains(&c.max_scoring_operations)
                && (1..=12 << 30).contains(&c.max_storage_bytes)
                && (1..=64 << 20).contains(&c.max_tensor_bytes),
            "numeric cost limits",
        )?;
        Ok(c)
    }
    fn estimate(&self, n: u64, e: u64, p: usize, t: u64, te: u64, prefix: u64) -> Result<Estimate> {
        ensure(
            p as u64 <= self.position_capacity,
            "generator exceeds numeric position capacity",
        )?;
        let categorical = if self.actor_input_width > 0 {
            self.actor_input_width
                .checked_sub(self.context_width + self.position_capacity * self.card_width)
                .filter(|x| *x > 0)
                .ok_or_else(|| error("numeric actor dimensions do not cover positions"))?
        } else {
            0
        };
        let storage = add(&[mul(&[n, 64]), mul(&[e, 8]), 8]);
        let chunk = n.min(self.candidate_chunk);
        let projection = mul(&[self.position_capacity, 19, self.actor_hidden_width, 4]);
        let unpadded = mul(&[self.position_capacity, 18, self.actor_hidden_width, 4]);
        let padding = mul(&[self.position_capacity, self.actor_hidden_width, 4]);
        let gather = mul(&[chunk, self.position_capacity, self.actor_hidden_width, 4]);
        let columns = mul(&[n, categorical, 4]);
        let logits = mul(&[n, 20]);
        let head = mul(&[chunk, self.actor_hidden_width, 32]);
        let working = mul(&[chunk, self.tail_working_bytes_per_candidate]);
        let tensor = mul(&[chunk, self.tail_tensor_bytes_per_candidate]);
        let encoder = mul(&[add(&[self.context_width, mul(&[18, self.card_width])]), 4]);
        let temporary = add(&[mul(&[t, 2048]), mul(&[te, 8]), mul(&[prefix, 32])]);
        let preparation = add(&[
            self.fixed_bytes,
            temporary,
            storage,
            encoder,
            projection,
            unpadded,
            padding,
            gather,
            columns,
            logits,
            head,
            working,
        ]);
        let operations = add(&[
            mul(&[
                2,
                self.position_capacity,
                18,
                self.card_width,
                self.actor_hidden_width,
            ]),
            mul(&[2, n, categorical, self.actor_hidden_width]),
            mul(&[n, self.position_capacity, self.actor_hidden_width]),
            mul(&[2, self.context_width, self.actor_hidden_width]),
            mul(&[n, self.tail_operations_per_candidate]),
        ]);
        let largest = *[
            mul(&[n, 64]),
            mul(&[e, 8]),
            projection,
            gather,
            columns,
            logits,
            head,
            encoder,
            tensor,
        ]
        .iter()
        .max()
        .unwrap();
        Ok(Estimate {
            preparation,
            operations,
            storage,
            largest,
        })
    }
}
struct Estimate {
    preparation: u64,
    operations: u64,
    storage: u64,
    largest: u64,
}
impl Estimate {
    fn fits(&self, c: &Costs) -> bool {
        self.preparation <= c.max_preparation_bytes
            && self.operations <= c.max_scoring_operations
            && self.storage <= c.max_storage_bytes
            && self.largest <= c.max_tensor_bytes
    }
}
/// A supplied transcript is exact and exhaustible. Real games use monotonic time.
pub enum Clock {
    Fixed { values: Vec<f64>, index: usize },
    Real { origin: Instant, calls: usize },
}
impl Clock {
    pub fn parse(v: &Value) -> Result<Self> {
        if v.is_null() {
            return Ok(Self::Real {
                origin: Instant::now(),
                calls: 0,
            });
        }
        let values: Vec<f64> =
            serde_json::from_value(v.clone()).map_err(|e| error(e.to_string()))?;
        ensure(
            values.iter().all(|v| v.is_finite() && *v >= 0.)
                && values.windows(2).all(|p| p[0] <= p[1]),
            "monotonic finite clock transcript required",
        )?;
        Ok(Self::Fixed { values, index: 0 })
    }
    pub fn calls(&self) -> usize {
        match self {
            Self::Fixed { index, .. } => *index,
            Self::Real { calls, .. } => *calls,
        }
    }
    fn now(&mut self) -> Result<f64> {
        match self {
            Self::Fixed { values, index } => {
                let v = *values
                    .get(*index)
                    .ok_or_else(|| error("fixed clock transcript exhausted"))?;
                *index += 1;
                Ok(v)
            }
            Self::Real { origin, calls } => {
                *calls += 1;
                Ok(origin.elapsed().as_secs_f64())
            }
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Proposal {
    pub kind: String,
    pub target: String,
    pub refs: Vec<usize>,
    pub numerics: [i64; 4],
    pub presence: [bool; 4],
}
impl Proposal {
    fn new(kind: &str, target: &str, refs: Vec<usize>) -> Self {
        Self {
            kind: kind.into(),
            target: target.into(),
            refs,
            numerics: [0; 4],
            presence: [false; 4],
        }
    }
    fn exchange(slot: usize, a: [i64; 3]) -> Self {
        Self {
            numerics: [a[0], a[1], a[2], 0],
            presence: [true, true, true, false],
            ..Self::new("exchange", "omitted", vec![slot])
        }
    }
}
#[derive(Clone, Debug)]
struct Summary {
    attribute: String,
    effects: Vec<String>,
    attack: i64,
    resources: [i64; 3],
    refresh: usize,
    remaining: Vec<usize>,
    length: usize,
}
pub struct Space<'a> {
    pub input: &'a Input,
    pub config: &'a Config,
    choices: [Vec<usize>; 5],
    pub maximum: usize,
    allocations: Vec<[i64; 3]>,
}
impl<'a> Space<'a> {
    pub fn new(input: &'a Input, config: &'a Config) -> Result<Self> {
        let mut choices: [Vec<usize>; 5] = Default::default();
        for c in &input.cards {
            if input.usable[c.slot] {
                if c.role("weapon")
                    || c.additional()
                    || c.subclass == "weapon"
                    || c.ability == "cutCost"
                {
                    choices[0].push(c.slot);
                }
                if rules::defense_kind(c).is_some() || c.ability == "cutCost" {
                    choices[1].push(c.slot);
                }
                if rules::UTILITY.contains(&c.id.as_str()) || c.ability == "cutCost" {
                    choices[2].push(c.slot);
                }
                if c.subclass == "status" || c.ability == "cutCost" {
                    choices[3].push(c.slot);
                }
            }
        }
        choices[4] = input
            .cards
            .iter()
            .filter(|c| {
                c.miracle
                    || c.kind != "weapons"
                        && (input.end_times
                            || !matches!(c.id.as_str(), "sun-amulet" | "dangerous-mortar"))
            })
            .map(|c| c.slot)
            .collect();
        let total = input.hp + input.mp + input.money;
        ensure(
            input.trade[0].is_empty() || (0..=297).contains(&total),
            "hinted exchange carrier has no conserved range-valid allocation",
        )?;
        let allocations = if input.trade[0].is_empty() {
            vec![]
        } else {
            allocations(input)
        };
        Ok(Self {
            input,
            config,
            choices,
            maximum: config.max_sequence_length.min(input.cards.len()).min(18),
            allocations,
        })
    }
    fn choices(&self, f: Family) -> &[usize] {
        &self.choices[match f {
            Family::Offense => 0,
            Family::Defense => 1,
            Family::Utility => 2,
            Family::Status => 3,
            Family::Discard => 4,
        }]
    }
    fn families(&self) -> Vec<Family> {
        if self.input.choice == "defense" {
            return if self.choices(Family::Defense).is_empty() {
                vec![]
            } else {
                vec![Family::Defense]
            };
        }
        if self.input.choice != "turn_action" {
            return vec![];
        }
        let mut f: Vec<_> = [Family::Offense, Family::Utility, Family::Status]
            .into_iter()
            .filter(|f| !self.choices(*f).is_empty() && !self.input.targets.is_empty())
            .collect();
        if !self.choices(Family::Discard).is_empty()
            && if self.input.end_times {
                self.input.can_sacrifice
            } else {
                self.input.can_discard
            }
        {
            f.push(Family::Discard);
        }
        f
    }
    pub fn analysis(&self, f: Family, r: &[usize]) -> Analysis {
        rules::analyze(
            &self.input.cards,
            r,
            f,
            self.input.mp,
            self.input.flash,
            self.config.max_sequence_length,
            &self.input.usable,
            self.input.context,
            self.config.requires_powder_weapon,
        )
    }
    fn valid(&self, f: Family, r: &[usize]) -> bool {
        if f == Family::Discard {
            r.len() <= self.maximum && r.windows(2).all(|p| p[0] < p[1])
        } else {
            self.analysis(f, r).viable
        }
    }
    fn targets(&self, f: Family, r: &[usize]) -> Vec<String> {
        if f == Family::Discard {
            return vec!["omitted".into()];
        }
        let facts = self.analysis(f, r);
        if !facts.complete {
            return vec![];
        }
        if f == Family::Defense {
            return vec!["omitted".into()];
        }
        self.input
            .targets
            .iter()
            .filter(|t| !facts.opponent_only || t.as_str() == "opponent")
            .cloned()
            .collect()
    }
    fn completions(&self, f: Family, r: &[usize]) -> Vec<Proposal> {
        let kind = match f {
            Family::Discard => {
                if self.input.end_times {
                    "sacrifice"
                } else {
                    "discard"
                }
            }
            Family::Defense => "respond",
            _ => "play_sequence",
        };
        self.targets(f, r)
            .into_iter()
            .map(|t| Proposal::new(kind, &t, r.to_vec()))
            .collect()
    }
    fn basic_count(&self) -> u64 {
        if self.input.choice == "buy_accept_reject" {
            2
        } else {
            u64::from(
                self.input.choice == "defense" && self.input.can_allow
                    || self.input.choice == "turn_action" && self.input.can_pray,
            )
        }
    }
    fn mandatory(&self) -> Vec<Proposal> {
        let mut out = vec![];
        if self.input.choice == "buy_accept_reject" {
            for accept in [0, 1] {
                out.push(Proposal {
                    numerics: [0, 0, 0, accept],
                    presence: [false, false, false, true],
                    ..Proposal::new("buy_decision", "omitted", vec![])
                });
            }
        } else if self.input.choice == "defense" && self.input.can_allow {
            out.push(Proposal::new("respond", "omitted", vec![]));
        } else if self.input.choice == "turn_action" {
            if self.input.can_pray {
                out.push(Proposal::new("pray", "omitted", vec![]));
            }
            for &s in &self.input.trade[2] {
                for t in &self.input.targets {
                    out.push(Proposal::new("buy", t, vec![s]));
                }
            }
            for &s in &self.input.trade[0] {
                out.push(Proposal::exchange(s, self.allocations[0]));
            }
            for &s in &self.input.phenomena {
                out.push(Proposal::new("invoke_phenomenon", "omitted", vec![s]));
            }
        }
        out
    }
    fn prefix_capacity(&self) -> u64 {
        self.families()
            .iter()
            .map(|f| self.choices(*f).len() as u64)
            .sum::<u64>()
            * 3
            * self.maximum as u64
            * (self.maximum as u64 + 1)
            / 2
    }
    fn summary(&self, p: &Proposal) -> Summary {
        let cards: Vec<_> = p.refs.iter().map(|r| &self.input.cards[*r]).collect();
        let facts = if p.kind == "play_sequence" {
            self.analysis(family(&self.input.cards[p.refs[0]]), &p.refs)
        } else if p.kind == "respond" && !cards.is_empty() {
            self.analysis(Family::Defense, &p.refs)
        } else {
            Analysis {
                attribute: "neutral".into(),
                ..Default::default()
            }
        };
        let (freed, opportunities, removed): (usize, usize, Vec<usize>) = match p.kind.as_str() {
            "play_sequence" | "respond" => {
                let r: Vec<_> = cards
                    .iter()
                    .filter(|c| !c.miracle)
                    .map(|c| c.slot)
                    .collect();
                (r.len(), cards.len(), r)
            }
            "discard" | "sacrifice" => (
                cards.len(),
                if p.kind == "sacrifice" {
                    cards.len()
                } else {
                    0
                },
                p.refs.clone(),
            ),
            "pray" => (usize::from(self.input.cards.len() == 18), 1, vec![]),
            _ => {
                let r = if p.kind == "sell" && p.target == "self" {
                    p.refs[..1].to_vec()
                } else {
                    p.refs.clone()
                };
                (r.len(), 1, r)
            }
        };
        Summary {
            attribute: facts.attribute,
            effects: facts.effects,
            attack: facts.attack,
            resources: if p.kind == "exchange" {
                [p.numerics[0], p.numerics[1], p.numerics[2]]
            } else {
                [
                    facts.spent,
                    cards.len() as i64,
                    cards.iter().map(|c| c.price).sum(),
                ]
            },
            refresh: 18
                .min(opportunities)
                .min((18 + freed).saturating_sub(self.input.cards.len())),
            remaining: self
                .input
                .cards
                .iter()
                .filter(|c| !removed.contains(&c.slot))
                .map(|c| c.slot)
                .collect(),
            length: cards.len(),
        }
    }
}
pub fn family(c: &rules::Card) -> Family {
    if rules::UTILITY.contains(&c.id.as_str()) {
        Family::Utility
    } else if c.subclass == "status" {
        Family::Status
    } else {
        Family::Offense
    }
}
fn grid(lower: i64, upper: i64) -> Vec<i64> {
    let mut values: Vec<_> = (0..26).map(|i| lower + (upper - lower) * i / 25).collect();
    values.dedup();
    let mut out = vec![];
    for v in [values[0], *values.last().unwrap(), values[values.len() / 2]]
        .into_iter()
        .chain(values)
    {
        if !out.contains(&v) {
            out.push(v);
        }
    }
    out
}
fn allocations(input: &Input) -> Vec<[i64; 3]> {
    let total = input.hp + input.mp + input.money;
    if !(0..=297).contains(&total) {
        return vec![];
    }
    let mut out = vec![];
    let current = [input.hp, input.mp, input.money];
    if current.iter().all(|v| (0..=99).contains(v)) {
        out.push(current);
    } else {
        let hp = 99.min(total);
        let mp = 99.min(total - hp);
        out.push([hp, mp, total - hp - mp]);
    }
    for order in [[0, 1, 2], [1, 2, 0], [2, 0, 1]] {
        for first in grid(0.max(total - 198), 99.min(total)) {
            let remaining = total - first;
            for second in grid(0.max(remaining - 99), 99.min(remaining)) {
                let mut v = [0; 3];
                v[order[0]] = first;
                v[order[1]] = second;
                v[order[2]] = remaining - second;
                out.push(v);
            }
        }
    }
    let balanced = total / 3;
    out.push([
        balanced,
        (total - balanced) / 2,
        total - balanced - (total - balanced) / 2,
    ]);
    if (1..=199).contains(&total) {
        out.push([1, 99.min(total - 1), 0.max(total - 100)]);
    }
    let costs: BTreeSet<_> = input
        .cards
        .iter()
        .filter(|c| c.miracle)
        .map(|c| c.cost)
        .collect();
    for cost in costs {
        for mp in [cost - 1, cost, cost + 1] {
            if (0.max(total - 198)..=99.min(total)).contains(&mp) {
                for hp in [0.max(total - mp - 99), 99.min(total - mp)] {
                    out.push([hp, mp, total - hp - mp]);
                }
            }
        }
    }
    out
}
#[derive(Default, Clone, Serialize)]
struct Totals {
    complete: u64,
    prefixes: u64,
    attempts: u64,
    excluded: u64,
    targets: u64,
    numeric_variants: u64,
    elements: u64,
}
impl Totals {
    fn add(&mut self, c: u64, e: u64, t: u64, n: u64, a: u64, x: u64) {
        self.complete = add(&[self.complete, c]);
        self.elements = add(&[self.elements, e]);
        self.targets = add(&[self.targets, t]);
        self.numeric_variants = add(&[self.numeric_variants, n]);
        self.attempts = add(&[self.attempts, a]);
        self.excluded = add(&[self.excluded, x]);
    }
}
fn upper(s: &Space<'_>) -> Totals {
    let mut t = Totals::default();
    let basics = s.basic_count();
    if s.input.choice == "turn_action" {
        let buy = (s.input.trade[2].len() * s.input.targets.len()) as u64;
        let sell = if s.config.max_sequence_length >= 2 {
            (s.input.trade[1].len()
                * s.input
                    .cards
                    .iter()
                    .filter(|c| c.hand)
                    .count()
                    .saturating_sub(1)
                * s.input.targets.len()) as u64
        } else {
            0
        };
        let ex = s.input.trade[0].len() as u64 * (3 * 26 * 26 + 3 + s.input.cards.len() as u64 * 6);
        let phenomena = s.input.phenomena.len() as u64;
        let fixed = basics + buy + sell + ex + phenomena;
        t.add(
            fixed,
            buy + 2 * sell + ex + phenomena,
            buy + sell,
            ex,
            fixed,
            0,
        );
    } else {
        t.add(
            basics,
            0,
            0,
            if s.input.choice == "buy_accept_reject" {
                basics
            } else {
                0
            },
            basics,
            0,
        );
    }
    for f in s.families() {
        let size = s.choices(f).len() as u64;
        let mut previous = 1;
        let targets = if matches!(f, Family::Offense | Family::Utility | Family::Status) {
            s.input.targets.len() as u64
        } else {
            1
        };
        for depth in 1..=s.maximum as u64 {
            let count = if depth > size {
                0
            } else if f == Family::Discard {
                let mut c = 1u128;
                for i in 0..depth {
                    c = c * (size - i) as u128 / (i + 1) as u128;
                }
                c.min(SAT as u128) as u64
            } else {
                mul(&(0..depth).map(|i| size - i).collect::<Vec<_>>())
            };
            let complete = mul(&[count, targets]);
            let attempts = mul(&[previous, (size + 1).saturating_sub(depth)]);
            t.add(
                complete,
                mul(&[complete, depth]),
                if matches!(f, Family::Offense | Family::Utility | Family::Status) {
                    complete
                } else {
                    0
                },
                0,
                add(&[attempts, complete]),
                attempts,
            );
            t.prefixes = add(&[t.prefixes, count]);
            previous = count;
        }
    }
    t
}
#[derive(Serialize)]
struct Precount {
    mode: String,
    saturated: bool,
    count_states: u64,
    #[serde(flatten)]
    totals: Totals,
    preparation_bytes: u64,
    scoring_operations: u64,
    storage_bytes: u64,
    full_enumeration_admitted: bool,
}
struct Counter<'a> {
    states: u64,
    reasons: BTreeSet<String>,
    deadline: f64,
    clock: &'a mut Clock,
    config: &'a Config,
}
impl Counter<'_> {
    fn tick(&mut self) -> Result<bool> {
        if self.states >= self.config.count_states {
            self.reasons.insert("count_states".into());
            return Ok(false);
        }
        if self.clock.now()? >= self.deadline {
            self.reasons.insert("count_time".into());
            return Ok(false);
        }
        self.states += 1;
        Ok(true)
    }
}
fn count_tree(
    s: &Space<'_>,
    f: Family,
    prefix: &mut Vec<usize>,
    counter: &mut Counter<'_>,
    t: &mut Totals,
) -> Result<bool> {
    if prefix.len() >= s.maximum {
        return Ok(true);
    }
    for &slot in s.choices(f) {
        if prefix.contains(&slot) {
            continue;
        }
        if !counter.tick()? {
            return Ok(false);
        }
        prefix.push(slot);
        if !s.valid(f, prefix) {
            t.add(0, 0, 0, 0, 1, 1);
            prefix.pop();
            continue;
        }
        t.prefixes += 1;
        let count = s.targets(f, prefix).len() as u64;
        t.add(
            count,
            count * prefix.len() as u64,
            if matches!(f, Family::Offense | Family::Utility | Family::Status) {
                count
            } else {
                0
            },
            0,
            1 + count,
            0,
        );
        if !count_tree(s, f, prefix, counter, t)? {
            return Ok(false);
        }
        prefix.pop();
    }
    Ok(true)
}
fn precount(
    s: &Space<'_>,
    costs: &Costs,
    clock: &mut Clock,
) -> Result<(Precount, BTreeSet<String>)> {
    let started = clock.now()?;
    let mut counter = Counter {
        states: 0,
        reasons: BTreeSet::new(),
        deadline: started + s.config.count_milliseconds as f64 / 1000.,
        clock,
        config: s.config,
    };
    let mut t = Totals::default();
    let mut exact = counter.tick()?;
    if exact {
        let basics = s.basic_count();
        t.add(
            basics,
            0,
            0,
            if s.input.choice == "buy_accept_reject" {
                basics
            } else {
                0
            },
            basics,
            0,
        );
        if s.input.choice == "turn_action" {
            let buy = (s.input.trade[2].len() * s.input.targets.len()) as u64;
            let sell = if s.config.max_sequence_length >= 2 {
                s.input.trade[1]
                    .iter()
                    .map(|slot| {
                        s.input
                            .cards
                            .iter()
                            .filter(|c| c.hand && c.slot != *slot)
                            .count() as u64
                    })
                    .sum::<u64>()
                    * s.input.targets.len() as u64
            } else {
                0
            };
            let phen = s.input.phenomena.len() as u64;
            t.add(
                buy + sell + phen,
                buy + sell * 2 + phen,
                buy + sell,
                0,
                buy + sell + phen,
                0,
            );
            if !s.input.trade[0].is_empty() {
                let mut unique = HashSet::new();
                let mut visits = 0;
                for allocation in &s.allocations {
                    if !counter.tick()? {
                        exact = false;
                        break;
                    }
                    visits += 1;
                    unique.insert(*allocation);
                }
                if exact {
                    let count = (unique.len() * s.input.trade[0].len()) as u64;
                    let attempts = visits * s.input.trade[0].len() as u64;
                    t.add(count, count, 0, count, attempts, attempts - count);
                }
            }
        }
        if exact {
            for family in s.families() {
                if !count_tree(s, family, &mut vec![], &mut counter, &mut t)? {
                    exact = false;
                    break;
                }
            }
        }
    }
    if !exact {
        t = upper(s);
    }
    let estimate = costs.estimate(
        t.complete,
        t.elements,
        s.config.max_sequence_length,
        t.complete,
        t.elements,
        s.prefix_capacity(),
    )?;
    let saturated = [
        t.complete,
        t.prefixes,
        t.attempts,
        t.excluded,
        t.targets,
        t.numeric_variants,
        t.elements,
        estimate.preparation,
        estimate.operations,
        estimate.storage,
        estimate.largest,
    ]
    .contains(&SAT);
    let admitted = exact
        && !saturated
        && t.complete > 0
        && t.complete
            <= s.config
                .max_candidates
                .min(s.config.max_temporary_candidates) as u64
        && t.attempts <= s.config.max_expansions
        && estimate.fits(costs);
    Ok((
        Precount {
            mode: if exact { "exact" } else { "upper_bound" }.into(),
            saturated,
            count_states: counter.states,
            totals: t,
            preparation_bytes: estimate.preparation,
            scoring_operations: estimate.operations,
            storage_bytes: estimate.storage,
            full_enumeration_admitted: admitted,
        },
        counter.reasons,
    ))
}
// SearchLimit is distinct from malformed input/mandatory-coverage failure.
#[derive(Debug)]
enum StepError {
    Limit,
    Failed(crate::error::Error),
}
impl From<crate::error::Error> for StepError {
    fn from(e: crate::error::Error) -> Self {
        Self::Failed(e)
    }
}
type Step<T> = std::result::Result<T, StepError>;
struct Collector<'a> {
    space: &'a Space<'a>,
    costs: &'a Costs,
    clock: &'a mut Clock,
    deadline: f64,
    discovery_deadline: f64,
    proposals: Vec<Proposal>,
    index: HashMap<Proposal, usize>,
    mandatory: Vec<usize>,
    summaries: Vec<Summary>,
    assigned: HashMap<usize, usize>,
    attempts: u64,
    prunes: u64,
    elements: u64,
    reasons: BTreeSet<String>,
}
impl<'a> Collector<'a> {
    fn tick(&mut self, mandatory: bool) -> Step<()> {
        let reason = if self.attempts >= self.space.config.max_expansions {
            Some("max_expansions")
        } else if self.clock.now()? >= self.discovery_deadline {
            Some("search_time")
        } else {
            None
        };
        if let Some(reason) = reason {
            self.reasons.insert(reason.into());
            return Err(if mandatory {
                StepError::Failed(error(format!("mandatory coverage cannot fit {reason}")))
            } else {
                StepError::Limit
            });
        }
        self.attempts += 1;
        Ok(())
    }
    fn offer(&mut self, p: Proposal, mandatory: bool) -> Step<()> {
        if let Some(&i) = self.index.get(&p) {
            self.prunes += 1;
            if mandatory && !self.mandatory.contains(&i) {
                self.mandatory.push(i);
            }
            return Ok(());
        }
        let count = self.proposals.len() + 1;
        let elements = self.elements + p.refs.len() as u64;
        let reason = if count > self.space.config.max_temporary_candidates {
            Some("max_temporary_candidates")
        } else if mandatory && self.mandatory.len() + 1 > self.space.config.max_candidates {
            Some("max_candidates")
        } else {
            let estimate = self.costs.estimate(
                count.min(self.space.config.max_candidates) as u64,
                elements.min(
                    (self.space.config.max_candidates * self.space.config.max_sequence_length)
                        as u64,
                ),
                self.space.config.max_sequence_length,
                count as u64,
                elements,
                self.space.prefix_capacity(),
            )?;
            if !estimate.fits(self.costs) {
                Some("numeric_costs")
            } else {
                None
            }
        };
        if let Some(r) = reason {
            self.reasons.insert(r.into());
            return Err(if mandatory {
                StepError::Failed(error(format!("mandatory coverage cannot fit {r}")))
            } else {
                StepError::Limit
            });
        }
        let i = self.proposals.len();
        self.summaries.push(self.space.summary(&p));
        self.index.insert(p.clone(), i);
        self.proposals.push(p);
        self.elements = elements;
        if mandatory {
            self.mandatory.push(i);
        }
        Ok(())
    }
}
struct Walker {
    family: Family,
    root: Vec<usize>,
    order: Vec<usize>,
    lengths: Vec<Option<usize>>,
    length_index: usize,
    stack: Vec<(Vec<usize>, usize)>,
}
impl Walker {
    fn new(
        family: Family,
        root: Vec<usize>,
        order: Vec<usize>,
        lengths: Vec<Option<usize>>,
    ) -> Self {
        let stack = if lengths.is_empty() {
            vec![]
        } else {
            vec![(root.clone(), 0)]
        };
        Self {
            family,
            root,
            order,
            lengths,
            length_index: 0,
            stack,
        }
    }
    fn next(&mut self, c: &mut Collector<'_>) -> Step<Option<Option<Vec<usize>>>> {
        loop {
            if self.stack.is_empty() {
                self.length_index += 1;
                if self.length_index >= self.lengths.len() {
                    return Ok(None);
                }
                self.stack.push((self.root.clone(), 0));
            }
            let length = self.lengths[self.length_index];
            let (prefix, next) = self.stack.last_mut().unwrap();
            if prefix.len() >= c.space.maximum
                || length.is_some_and(|l| prefix.len() >= l)
                || *next >= self.order.len()
            {
                self.stack.pop();
                continue;
            }
            let slot = self.order[*next];
            *next += 1;
            if prefix.contains(&slot) {
                continue;
            }
            c.tick(false)?;
            let mut refs = prefix.clone();
            refs.push(slot);
            if !c.space.valid(self.family, &refs) {
                return Ok(Some(None));
            }
            self.stack.push((refs.clone(), 0));
            let presented = if length.is_none_or(|l| l == refs.len()) {
                refs
            } else {
                vec![]
            };
            return Ok(Some(Some(presented)));
        }
    }
}
struct Fixed {
    sell: usize,
    hand: usize,
    target: usize,
    exchange: usize,
    allocation: usize,
}
impl Fixed {
    fn new() -> Self {
        Self {
            sell: 0,
            hand: 0,
            target: 0,
            exchange: 0,
            allocation: 1,
        }
    }
    fn next(&mut self, c: &mut Collector<'_>) -> Step<Option<Proposal>> {
        let s = c.space;
        if s.input.choice != "turn_action" {
            return Ok(None);
        }
        while s.config.max_sequence_length >= 2 && self.sell < s.input.trade[1].len() {
            while self.hand < s.input.cards.len() {
                let card = &s.input.cards[self.hand];
                if !card.hand || card.slot == s.input.trade[1][self.sell] {
                    self.hand += 1;
                    continue;
                }
                if self.target < s.input.targets.len() {
                    let p = Proposal::new(
                        "sell",
                        &s.input.targets[self.target],
                        vec![s.input.trade[1][self.sell], card.slot],
                    );
                    self.target += 1;
                    c.tick(false)?;
                    return Ok(Some(p));
                }
                self.target = 0;
                self.hand += 1;
            }
            self.hand = 0;
            self.sell += 1;
        }
        while self.exchange < s.input.trade[0].len() {
            if self.allocation < s.allocations.len() {
                let p = Proposal::exchange(
                    s.input.trade[0][self.exchange],
                    s.allocations[self.allocation],
                );
                self.allocation += 1;
                c.tick(false)?;
                return Ok(Some(p));
            }
            self.exchange += 1;
            self.allocation = 1;
        }
        Ok(None)
    }
}
enum Job {
    Fixed(Fixed),
    Walk(Walker),
}
fn length_order(maximum: usize, band: usize) -> Vec<Option<usize>> {
    if maximum < 2 {
        return vec![];
    }
    let mut lengths = vec![];
    for l in [maximum, 2, (maximum + 2) / 2]
        .into_iter()
        .chain(2..=maximum)
    {
        if !lengths.contains(&l) {
            lengths.push(l);
        }
    }
    lengths
        .into_iter()
        .skip(band)
        .step_by(3)
        .map(Some)
        .collect()
}
fn breadth<K: Ord>(others: &[usize], key: impl Fn(usize) -> K) -> VecDeque<usize> {
    let mut ordered = others.to_vec();
    ordered.sort_by_key(|i| key(*i));
    let mut out = VecDeque::new();
    let mut intervals = VecDeque::from([(0, ordered.len())]);
    while let Some((lower, upper)) = intervals.pop_front() {
        if lower >= upper {
            continue;
        }
        let middle = (lower + upper) / 2;
        out.push_back(ordered[lower]);
        if upper - lower > 1 {
            out.push_back(ordered[upper - 1]);
        }
        if middle != lower && middle != upper - 1 {
            out.push_back(ordered[middle]);
        }
        intervals.extend([(lower + 1, middle), (middle + 1, upper - 1)]);
    }
    out
}
fn grouped<K: Eq + std::hash::Hash>(
    others: &[usize],
    key: impl Fn(usize) -> K,
    c: &mut Collector<'_>,
    deadline: f64,
) -> Result<VecDeque<usize>> {
    let mut groups: Vec<VecDeque<usize>> = vec![];
    let mut index = HashMap::new();
    for &i in others {
        let k = key(i);
        let g = *index.entry(k).or_insert_with(|| {
            groups.push(VecDeque::new());
            groups.len() - 1
        });
        groups[g].push_back(i);
    }
    let mut out = VecDeque::new();
    let mut active: Vec<usize> = (0..groups.len()).collect();
    while !active.is_empty() {
        if c.clock.now()? >= deadline {
            c.reasons.insert("search_time".into());
            break;
        }
        for &g in &active {
            out.push_back(groups[g].pop_front().unwrap());
        }
        active.retain(|g| !groups[*g].is_empty());
    }
    Ok(out)
}
fn select(c: &mut Collector<'_>, rng: &mut PythonRng) -> Result<Vec<usize>> {
    let mut selected = c.mandatory.clone();
    let mut used: HashSet<_> = selected.iter().copied().collect();
    let others: Vec<_> = (0..c.proposals.len())
        .filter(|i| !used.contains(i))
        .collect();
    if others.is_empty() {
        return Ok(selected);
    }
    let preparation_deadline = c.clock.now()? + (c.deadline - c.clock.now()?).max(0.) * 0.5;
    let mut prepared = vec![];
    for (index, &p) in others.iter().enumerate() {
        if index % 64 == 0 && c.clock.now()? >= preparation_deadline {
            c.reasons.insert("search_time".into());
            break;
        }
        prepared.push(p);
    }
    let others = prepared;
    let keys: Vec<_> = c
        .summaries
        .iter()
        .zip(&c.proposals)
        .map(|(s, p)| {
            (
                s.attribute.clone(),
                s.effects.clone(),
                p.kind.clone(),
                p.target.clone(),
            )
        })
        .collect();
    let categorical = grouped(&others, |i| keys[i].clone(), c, preparation_deadline)?;
    let mut exploration = others.clone();
    rng.shuffle(&mut exploration);
    let keys: Vec<_> = c.summaries.iter().map(|s| s.remaining.clone()).collect();
    let remaining = grouped(&others, |i| keys[i].clone(), c, preparation_deadline)?;
    let mut channels = [
        categorical,
        breadth(&others, |i| c.summaries[i].attack),
        breadth(&others, |i| c.summaries[i].resources),
        breadth(&others, |i| c.summaries[i].refresh),
        remaining,
        breadth(&others, |i| c.summaries[i].length),
        exploration.into(),
    ];
    let limit = c.space.config.max_candidates;
    let mut selected_len = selected.len();
    for i in 0..7 {
        if selected_len >= limit {
            break;
        }
        match take(i, c, &mut channels, &mut selected, &mut used)? {
            None => return Ok(selected),
            Some(true) => selected_len += 1,
            Some(false) => (),
        }
    }
    while selected_len < limit && channels.iter().any(|ch| !ch.is_empty()) {
        let mut progress = false;
        for (i, weight) in c.space.config.channel_weights.iter().enumerate() {
            for _ in 0..*weight {
                match take(i, c, &mut channels, &mut selected, &mut used)? {
                    None => return Ok(selected),
                    Some(false) => break,
                    Some(true) => {
                        selected_len += 1;
                        progress = true;
                    }
                }
            }
        }
        if !progress {
            break;
        }
    }

    if selected_len >= limit && selected_len < c.proposals.len() {
        c.reasons.insert("max_candidates".into());
    }
    Ok(selected)
}
fn take(
    channel: usize,
    c: &mut Collector<'_>,
    channels: &mut [VecDeque<usize>; 7],
    selected: &mut Vec<usize>,
    used: &mut HashSet<usize>,
) -> Result<Option<bool>> {
    if c.clock.now()? >= c.deadline {
        c.reasons.insert("search_time".into());
        return Ok(None);
    }
    let mut skipped = 0;
    while channels[channel].front().is_some_and(|i| used.contains(i)) {
        channels[channel].pop_front();
        skipped += 1;
        if skipped % 64 == 0 && c.clock.now()? >= c.deadline {
            c.reasons.insert("search_time".into());
            return Ok(None);
        }
    }
    if channels[channel].is_empty() || selected.len() >= c.space.config.max_candidates {
        return Ok(Some(false));
    }
    let p = channels[channel].pop_front().unwrap();
    selected.push(p);
    c.assigned.insert(p, channel);
    used.insert(p);
    Ok(Some(true))
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Table {
    pub frame_hash: String,
    pub kinds: Vec<String>,
    pub targets: Vec<String>,
    pub numerics: Vec<[i64; 4]>,
    pub presence: Vec<[bool; 4]>,
    pub offsets: Vec<usize>,
    pub slots: Vec<usize>,
    pub action_contract: Value,
    pub purpose: String,
    pub synthetic_capacity: bool,
    pub table_hash: String,
}
impl Table {
    pub fn proposals(&self) -> Vec<Proposal> {
        (0..self.kinds.len())
            .map(|i| Proposal {
                kind: self.kinds[i].clone(),
                target: self.targets[i].clone(),
                refs: self.slots[self.offsets[i]..self.offsets[i + 1]].to_vec(),
                numerics: self.numerics[i],
                presence: self.presence[i],
            })
            .collect()
    }
    fn content(&self) -> Result<Value> {
        let mut v = serde_json::to_value(self).map_err(|e| error(e.to_string()))?;
        v.as_object_mut().unwrap().remove("table_hash");
        Ok(v)
    }
    pub fn validate(&self, input: &Input, config: &Config) -> Result<()> {
        let n = self.kinds.len();
        ensure(
            n > 0
                && n <= config.max_candidates
                && self.targets.len() == n
                && self.numerics.len() == n
                && self.presence.len() == n
                && self.offsets.len() == n + 1
                && self.offsets[0] == 0
                && self.offsets[n] == self.slots.len()
                && self.offsets.windows(2).all(|p| p[0] <= p[1])
                && self.slots.iter().all(|s| *s < input.cards.len()),
            "invalid native table geometry",
        )?;
        ensure(
            self.frame_hash == input.frame_hash
                && self.action_contract == config.action_contract
                && self.purpose == "qualified_game"
                && !self.synthetic_capacity
                && hash(&self.content()?)? == self.table_hash,
            "native supplied table binding mismatch",
        )?;
        for p in self.proposals() {
            ensure(
                input.action_kinds.contains(&p.kind),
                "candidate outside typed metadata",
            )?;
            let targeted = matches!(p.kind.as_str(), "play_sequence" | "sell" | "buy");
            ensure(
                if targeted {
                    matches!(p.target.as_str(), "self" | "opponent")
                } else {
                    p.target == "omitted"
                },
                "candidate target presence mismatch",
            )?;
            let refs_ok = match p.kind.as_str() {
                "play_sequence" | "discard" | "sacrifice" => !p.refs.is_empty(),
                "respond" => true,
                "sell" => p.refs.len() == 2,
                "exchange" | "buy" | "invoke_phenomenon" => p.refs.len() == 1,
                "pray" | "buy_decision" => p.refs.is_empty(),
                _ => false,
            };
            ensure(
                refs_ok
                    && p.refs.len() <= 18
                    && p.refs.iter().collect::<HashSet<_>>().len() == p.refs.len(),
                "invalid native table refs",
            )?;
            let presence = match p.kind.as_str() {
                "exchange" => [true, true, true, false],
                "buy_decision" => [false, false, false, true],
                _ => [false; 4],
            };
            ensure(
                p.presence == presence
                    && p.numerics
                        .iter()
                        .zip(p.presence)
                        .all(|(n, present)| present || *n == 0),
                "candidate numeric presence mismatch",
            )?;
            if p.kind == "exchange" {
                ensure(
                    (i32::MIN as i64..=i32::MAX as i64).contains(&p.numerics[0])
                        && (0..=u32::MAX as i64).contains(&p.numerics[1])
                        && (0..=u32::MAX as i64).contains(&p.numerics[2]),
                    "candidate resource range",
                )?;
            }
            if p.kind == "buy_decision" {
                ensure(
                    (0..=1).contains(&p.numerics[3]),
                    "candidate acceptance range",
                )?;
            }
        }

        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct Generated {
    pub table: Table,
    pub report: Value,
}
impl Generated {
    pub fn value(&self) -> Value {
        json!({"table":self.table,"report":self.report})
    }
}
pub fn generate(
    input: &Input,
    config: &Config,
    rules: &Value,
    costs: &Costs,
    rng: &mut PythonRng,
    clock: &mut Clock,
) -> Result<Value> {
    Ok(generate_typed(input, config, rules, costs, rng, clock)?.value())
}
pub fn generate_typed(
    input: &Input,
    config: &Config,
    rules: &Value,
    costs: &Costs,
    rng: &mut PythonRng,
    clock: &mut Clock,
) -> Result<Generated> {
    let count_started = clock.now()?;
    let space = Space::new(input, config)?;
    costs.estimate(0, 0, config.max_sequence_length, 0, 0, 0)?;
    let before = rng.state();
    let (precount, reasons) = precount(&space, costs, clock)?;
    let search_started = clock.now()?;
    let deadline = search_started + config.search_milliseconds as f64 / 1000.;
    let mut c = Collector {
        space: &space,
        costs,
        clock,
        deadline,
        discovery_deadline: deadline - config.search_milliseconds as f64 / 1000. * 0.25,
        proposals: vec![],
        index: HashMap::new(),
        mandatory: vec![],
        summaries: vec![],
        assigned: HashMap::new(),
        attempts: 0,
        prunes: 0,
        elements: 0,
        reasons,
    };
    if config.max_sequence_length < 2
        && !input.trade[1].is_empty()
        && input.cards.iter().filter(|c| c.hand).count() >= 2
        && !input.targets.is_empty()
    {
        c.reasons.insert("max_sequence_length".into());
    }
    let mandatory = (|| -> Step<()> {
        for p in space.mandatory() {
            c.tick(true)?;
            c.offer(p, true)?;
        }
        Ok(())
    })();
    if let Err(e) = mandatory {
        return Err(match e {
            StepError::Failed(e) => e,
            StepError::Limit => error("mandatory limit"),
        });
    }
    let mut roots = vec![];
    for family in space.families() {
        let mut family_roots = vec![];
        for &slot in space.choices(family) {
            if let Err(e) = c.tick(true) {
                return Err(match e {
                    StepError::Failed(e) => e,
                    StepError::Limit => error("mandatory limit"),
                });
            }
            let r = vec![slot];
            if space.valid(family, &r) {
                family_roots.push(r.clone());
                for p in space.completions(family, &r) {
                    if let Err(e) = c.tick(true) {
                        return Err(match e {
                            StepError::Failed(e) => e,
                            StepError::Limit => error("mandatory limit"),
                        });
                    }
                    if let Err(e) = c.offer(p, true) {
                        return Err(match e {
                            StepError::Failed(e) => e,
                            StepError::Limit => error("mandatory limit"),
                        });
                    }
                }
            } else {
                c.prunes += 1;
            }
        }
        roots.push((family, family_roots));
    }
    let mut jobs = VecDeque::from([Job::Fixed(Fixed::new())]);
    for (family, roots) in roots {
        if precount.full_enumeration_admitted {
            let order = space.choices(family).to_vec();
            for root in roots {
                jobs.push_back(Job::Walk(Walker::new(
                    family,
                    root,
                    order.clone(),
                    vec![None],
                )));
            }
        } else {
            let mut order = space.choices(family).to_vec();
            rng.shuffle(&mut order);
            for root in roots {
                for band in 0..3 {
                    jobs.push_back(Job::Walk(Walker::new(
                        family,
                        root.clone(),
                        order.clone(),
                        length_order(space.maximum, band),
                    )));
                }
            }
        }
    }
    let traversal = (|| -> Step<()> {
        while let Some(mut job) = jobs.pop_front() {
            match &mut job {
                Job::Fixed(f) => {
                    let Some(p) = f.next(&mut c)? else {
                        continue;
                    };
                    c.offer(p, false)?;
                }
                Job::Walk(w) => {
                    let Some(item) = w.next(&mut c)? else {
                        continue;
                    };
                    match item {
                        None => c.prunes += 1,
                        Some(r) if !r.is_empty() => {
                            for p in space.completions(w.family, &r) {
                                c.tick(false)?;
                                c.offer(p, false)?;
                            }
                        }
                        _ => (),
                    }
                }
            }
            jobs.push_back(job);
        }
        Ok(())
    })();
    let complete = match traversal {
        Ok(()) => true,
        Err(StepError::Limit) => false,
        Err(StepError::Failed(e)) => return Err(e),
    };
    ensure(!c.proposals.is_empty(), "no completed v5 proposal found")?;
    let selection_started = c.clock.now()?;
    let mut selected = select(&mut c, rng)?;
    let selected_elements = selected
        .iter()
        .map(|i| c.proposals[*i].refs.len() as u64)
        .sum();
    ensure(
        costs
            .estimate(
                selected.len() as u64,
                selected_elements,
                config.max_sequence_length,
                c.proposals.len() as u64,
                c.elements,
                space.prefix_capacity(),
            )?
            .fits(costs),
        "selected table exceeds numeric cost envelope",
    )?;
    let mut offsets = vec![0];
    let mut slots = vec![];
    let mut keep = selected.len();
    for (index, &p) in selected.iter().enumerate() {
        if c.clock.now()? >= c.deadline {
            c.reasons.insert("search_time".into());
            ensure(
                index >= c.mandatory.len(),
                "mandatory table construction exceeds search deadline",
            )?;
            keep = index;
            break;
        }
        slots.extend(&c.proposals[p].refs);
        offsets.push(slots.len());
    }
    selected.truncate(keep);
    ensure(!selected.is_empty(), "invalid compact candidate columns")?;
    let mut table = Table {
        frame_hash: input.frame_hash.clone(),
        kinds: selected
            .iter()
            .map(|p| c.proposals[*p].kind.clone())
            .collect(),
        targets: selected
            .iter()
            .map(|p| c.proposals[*p].target.clone())
            .collect(),
        numerics: selected.iter().map(|p| c.proposals[*p].numerics).collect(),
        presence: selected.iter().map(|p| c.proposals[*p].presence).collect(),
        offsets,
        slots,
        action_contract: config.action_contract.clone(),
        purpose: "qualified_game".into(),
        synthetic_capacity: false,
        table_hash: String::new(),
    };
    table.table_hash = hash(&table.content()?)?;
    let mut channel_counts = [0u64; 7];
    for p in &selected {
        if let Some(i) = c.assigned.get(p) {
            channel_counts[*i] += 1;
        }
    }
    let finished = c.clock.now()?;
    if finished >= c.deadline {
        c.reasons.insert("search_time".into());
    }
    let report = json!({"action_contract":config.action_contract,"generator_version":config.generator,"config_hash":hash(&serde_json::to_value(config).unwrap())?,"numeric_costs_hash":hash(&serde_json::to_value(costs).unwrap())?,"table_hash":table.table_hash,"numeric_costs":costs,"precount":precount,"attempts":c.attempts,"prunes":c.prunes,"discovered":c.proposals.len(),"mandatory":c.mandatory.len(),"selected":selected.len(),"selected_elements":table.slots.len(),"channel_counts":channel_counts,"traversal_complete":complete,"presentation_complete":complete&&selected.len()==c.proposals.len(),"cutoff_reasons":c.reasons,"omissions":["unlisted direct-use utility definitions"],"rng_before":before,"rng_after":rng.state(),"count_ms":(search_started-count_started)*1000.,"search_ms":(selection_started-search_started)*1000.,"selection_ms":(finished-selection_started)*1000.,"rule_signature":config.signature(rules)?});
    Ok(Generated { table, report })
}
