//! Participant-local Random, Heuristic and Refresh state. No thread-local policy state.
use super::{
    Config, Input, Result, VisibleEvent, ensure, error, hash,
    rng::PythonRng,
    rules::{self, Family, Incoming},
    search::{Proposal, Table, family},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

fn require_game_capacity(config: &Config) -> Result<()> {
    ensure(
        config.max_sequence_length <= 18,
        "corrected baseline kind/game capacity required",
    )
}

pub fn heuristic(input: &Input, config: &Config, rows: &[Proposal]) -> Result<usize> {
    require_game_capacity(config)?;
    let mut best = None;
    for (i, p) in rows.iter().enumerate() {
        let cards: Vec<_> = p.refs.iter().map(|r| &input.cards[*r]).collect();
        let score = match p.kind.as_str() {
            "play_sequence" | "respond" if !cards.is_empty() => {
                let f = if p.kind == "respond" {
                    Family::Defense
                } else {
                    family(cards[0])
                };
                let a = rules::analyze(
                    &input.cards,
                    &p.refs,
                    f,
                    input.mp,
                    input.flash,
                    config.max_sequence_length,
                    &input.usable,
                    input.context,
                    config.requires_powder_weapon,
                    config.mirage_danger_requires_opponent,
                );
                ensure(
                    a.complete,
                    "generated baseline sequence lacks complete public analysis",
                )?;
                let mut score = if p.kind == "play_sequence" {
                    let mut x = a.attack as f64;
                    if a.attribute == "darkness" {
                        x += 6.;
                    }
                    if p.target == "self" {
                        x = -x;
                    }
                    x
                } else {
                    a.effective
                        .iter()
                        .map(|r| input.cards[*r].defense.max(0))
                        .sum::<i64>() as f64
                };
                score -= 0.05 * p.refs.len() as f64;
                score
            }
            "pray" => 2.,
            "sacrifice" => 1. + 0.1 * p.refs.len() as f64,
            "discard" => -1. - 0.05 * cards.iter().map(|c| c.price).sum::<i64>() as f64,
            "buy_decision" => p.numerics[3] as f64,
            "exchange" => 0.2 * (p.numerics[0] - input.hp) as f64,
            _ => 0.,
        };
        if best.is_none_or(|(_, value)| score > value) {
            best = Some((i, score));
        }
    }
    best.map(|(i, _)| i)
        .ok_or_else(|| error("empty baseline table"))
}
#[derive(Clone, Debug, Default)]
struct Fact {
    mask: u32,
    selected: usize,
    consumed: usize,
    spent: i64,
    attack: i64,
    offense: bool,
    removed: u32,
    opportunities: usize,
    uncertain: bool,
}
fn facts(input: &Input, config: &Config, rows: &[Proposal]) -> Result<Vec<Fact>> {
    let mut out = Vec::with_capacity(rows.len());
    for p in rows {
        let mask = p.refs.iter().fold(0, |m, r| m | (1 << r));
        let artifacts = p
            .refs
            .iter()
            .filter(|r| !input.cards[**r].miracle)
            .fold(0u32, |m, r| m | (1 << r));
        let mut fact = Fact {
            mask,
            selected: p.refs.len(),
            consumed: artifacts.count_ones() as usize,
            ..Default::default()
        };
        match p.kind.as_str() {
            "play_sequence" => {
                let f = family(&input.cards[p.refs[0]]); // Refresh uses the public default context for turn actions.
                let a = rules::analyze(
                    &input.cards,
                    &p.refs,
                    f,
                    input.mp,
                    input.flash,
                    config.max_sequence_length,
                    &input.usable,
                    Incoming::Unknown,
                    config.requires_powder_weapon,
                    config.mirage_danger_requires_opponent,
                );
                ensure(
                    a.complete,
                    "refresh sequence lacks complete public analysis",
                )?;
                let renewal = f == Family::Utility && p.target == "self";
                fact.spent = a.spent;
                fact.attack = a.attack;
                fact.offense = f == Family::Offense && p.target == "opponent";
                if renewal {
                    fact.removed = artifacts;
                    fact.opportunities = p.refs.len();
                    fact.uncertain = a
                        .effects
                        .iter()
                        .any(|s| s == "removeUsedMiracles" || s == "removeItems");
                }
            }
            "pray" => {
                fact.opportunities = 1;
                fact.uncertain = input.cards.len() == 18;
            }
            "sacrifice" => {
                fact.removed = mask;
                fact.opportunities = p.refs.len();
            }
            "exchange" if p.numerics[..3] == [input.hp, input.mp, input.money] => {
                fact.removed = mask;
                fact.opportunities = 1;
            }
            "sell" => {
                let price = input.cards[p.refs[1]].price;
                if p.target != "self" || price <= input.money + input.mp {
                    fact.removed = if p.target == "self" {
                        1 << p.refs[0]
                    } else {
                        mask
                    };
                    fact.opportunities = 1;
                    fact.spent = if p.target == "self" {
                        (price - input.money).max(0)
                    } else {
                        0
                    };
                }
            }
            "buy" | "invoke_phenomenon" => {
                fact.removed = mask;
                fact.opportunities = 1;
                fact.uncertain = p.kind == "invoke_phenomenon";
            }
            _ => (),
        }
        out.push(fact);
    }
    Ok(out)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pending {
    turn: u64,
    attack: i64,
    decision: u64,
    damage: i64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Previous {
    actor: u8,
    episode: String,
    revision: u64,
    frame_hash: String,
    boundary: usize,
    prefix_hash: String,
    turn: u64,
    table_hash: String,
    index: usize,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefreshState {
    last: Option<Previous>,
    own_turn: Option<u64>,
    decision: u64,
    samples: Vec<(i64, u64)>,
    pending: Option<Pending>,
}
impl RefreshState {
    pub fn select(
        &mut self,
        input: &Input,
        config: &Config,
        table: &Table,
        threshold: u64,
    ) -> Result<usize> {
        require_game_capacity(config)?;
        let rows = table.proposals();
        if let Some(last) = &self.last {
            ensure(
                input.actor == last.actor && input.episode == last.episode,
                "refresh history changed participant/episode without reset",
            )?;
            ensure(
                last.boundary <= input.boundary
                    && input.prefix_hash(last.boundary) == last.prefix_hash,
                "refresh history prefix changed",
            )?;
            if input.frame_hash == last.frame_hash {
                ensure(
                    table.table_hash == last.table_hash,
                    "repeated refresh frame changed supplied candidates",
                )?;
                ensure(
                    last.index < table.kinds.len(),
                    "invalid restored refresh selection",
                )?;
                return Ok(last.index);
            }
            ensure(
                input.revision > last.revision
                    && input.interval_start == last.boundary
                    && input.turn >= last.turn,
                "refresh history frontier/revision/turn did not advance",
            )?;
        }
        let mut next = self.clone();
        if input.choice == "turn_action" && Some(input.turn) != next.own_turn {
            next.own_turn = Some(input.turn);
            next.decision = next
                .decision
                .checked_add(1)
                .ok_or_else(|| error("refresh decision overflow"))?;
        }
        if let Some(previous) = &self.last {
            for event in &input.events[previous.boundary..input.boundary] {
                let Some(pending) = next.pending.as_mut() else {
                    break;
                };
                match event {
                    VisibleEvent::Turn(turn) if *turn > pending.turn => {
                        next.samples
                            .push(((pending.attack - pending.damage).max(0), pending.decision));
                        next.pending = None;
                    }
                    VisibleEvent::Damage { player, amount } if *player != input.actor => {
                        pending.damage += (*amount).max(0)
                    }
                    _ => (),
                }
            }
        }
        next.samples
            .retain(|(_, origin)| next.decision - *origin <= 3);
        let index = if input.choice == "turn_action" {
            let facts = facts(input, config, &rows)?;
            let usable: Vec<_> = next
                .samples
                .iter()
                .filter(|(_, origin)| (1..=3).contains(&(next.decision - *origin)))
                .collect();
            let count = usable.len() as i128;
            let blocked: i128 = usable.iter().map(|(value, _)| *value as i128).sum();
            let eligible: Vec<_> = facts
                .iter()
                .enumerate()
                .filter(|(_, f)| {
                    f.offense
                        && f.attack > 0
                        && if count == 0 {
                            f.attack as u64 >= threshold
                        } else {
                            count * f.attack as i128 >= 2 * blocked
                        }
                })
                .map(|(i, _)| i)
                .collect();
            let index = if !eligible.is_empty() {
                *eligible
                    .iter()
                    .max_by_key(|i| {
                        let f = &facts[**i];
                        (
                            if count > 0 {
                                count * f.attack as i128 - blocked
                            } else {
                                f.attack as i128
                            },
                            -f.spent,
                            -(f.consumed as i64),
                            -(**i as i64),
                        )
                    })
                    .unwrap()
            } else {
                renew_or_fallback(input, config, &rows, &facts, threshold)?
            };
            if facts[index].offense && facts[index].attack > 0 {
                next.pending = Some(Pending {
                    turn: input.turn,
                    attack: facts[index].attack,
                    decision: next.decision,
                    damage: 0,
                });
            }
            index
        } else {
            heuristic(input, config, &rows)?
        };
        next.last = Some(Previous {
            actor: input.actor,
            episode: input.episode.clone(),
            revision: input.revision,
            frame_hash: input.frame_hash.clone(),
            boundary: input.boundary,
            prefix_hash: input.prefix_hash(input.boundary),
            turn: input.turn,
            table_hash: table.table_hash.clone(),
            index,
        });
        *self = next;
        Ok(index)
    }
}
fn renew_or_fallback(
    input: &Input,
    config: &Config,
    rows: &[Proposal],
    facts: &[Fact],
    threshold: u64,
) -> Result<usize> {
    let possession = input.cards.len();
    let remaining = (1usize << possession) - 1;
    let mut retention = vec![0i64; 1 << possession];
    for f in facts {
        if f.offense && f.attack > 0 {
            retention[f.mask as usize] = retention[f.mask as usize].max(f.attack);
        }
    }
    for bit in 0..possession {
        for mask in 0..retention.len() {
            if mask & (1 << bit) != 0 {
                retention[mask] = retention[mask].max(retention[mask ^ (1 << bit)]);
            }
        }
    }
    let mut renewals = vec![];
    for (i, f) in facts.iter().enumerate() {
        let freed = if rows[i].kind == "pray" && possession == 18 {
            1
        } else {
            f.removed.count_ones() as usize
        };
        let opportunities = f.opportunities.min((18 + freed).saturating_sub(possession));
        if opportunities > 0 {
            let retained = if f.uncertain {
                0
            } else {
                retention[remaining ^ f.removed as usize] as u64
            };
            renewals.push((
                retained.min(threshold),
                opportunities,
                -f.spent,
                -(f.selected as i64),
                -(i as i64),
            ));
        }
    }
    if let Some(best) = renewals.iter().max() {
        return Ok((-best.4) as usize);
    }
    if possession == 18 {
        if let Some(i) = (0..rows.len())
            .filter(|i| rows[*i].kind == "discard")
            .max_by_key(|i| {
                (
                    (retention[remaining ^ facts[*i].mask as usize] as u64).min(threshold),
                    -(facts[*i].selected as i64),
                    -(*i as i64),
                )
            })
        {
            return Ok(i);
        }
    }
    if let Some(i) = (0..facts.len())
        .filter(|i| facts[*i].offense)
        .min_by_key(|i| (facts[*i].attack, facts[*i].spent, facts[*i].selected, *i))
    {
        return Ok(i);
    }
    heuristic(input, config, rows)
}
/// Validate the complete Refresh option; only the private comparison value is
/// bounded. Callers retain the original JSON integer for capture and identity.
pub(crate) fn comparison_threshold(value: &Value) -> Result<u64> {
    let number = value
        .as_number()
        .ok_or_else(|| error("refresh positive integer threshold required"))?;
    let text = number.as_str();
    ensure(
        crate::json::integer_token(number)
            && !text.starts_with('-')
            && text.bytes().any(|b| b != b'0'),
        "refresh positive integer threshold required",
    )?;
    // Public attacks saturate at i64::MAX, so every above-u64 threshold has
    // identical comparisons. Never replace the caller's original JSON value.
    Ok(text.parse::<u64>().unwrap_or(u64::MAX))
}

pub fn select_diagnostic(
    input: &Input,
    config: &Config,
    rules: &Value,
    v: &Value,
) -> Result<Value> {
    require_game_capacity(config)?;
    let generated = &v["generated"];
    let table: Table =
        serde_json::from_value(generated["table"].clone()).map_err(|e| error(e.to_string()))?;
    table.validate(input, config)?;
    ensure(
        generated["report"]["config_hash"] == hash(&serde_json::to_value(config).unwrap())?
            && generated["report"]["rule_signature"] == config.signature(rules)?
            && generated["report"]["table_hash"] == table.table_hash,
        "baseline generated config/rule binding mismatch",
    )?;
    let kind = v["kind"]
        .as_str()
        .ok_or_else(|| error("baseline kind required"))?;
    let (index, label, rng_state, state) = match kind {
        "random" => {
            let mut rng = PythonRng::from_state(&v["rng_state"])?;
            let index = rng.range(table.kinds.len() as u64)? as usize;
            (
                index,
                "random-card-instance-v2".into(),
                rng.state(),
                Value::Null,
            )
        }
        "heuristic" => {
            let rng = PythonRng::from_state(&v["rng_state"])?;
            (
                heuristic(input, config, &table.proposals())?,
                "heuristic-card-instance-v2".into(),
                rng.state(),
                Value::Null,
            )
        }
        "refresh" => {
            let options = &v["refresh_options"];
            use crate::json::ValueExt;
            options.closed(&["version", "attack_threshold"], &[])?;
            ensure(options["version"] == "1", "refresh version")?;
            let threshold = comparison_threshold(&options["attack_threshold"])?;
            let mut state: RefreshState = if v["state"].is_null() {
                RefreshState::default()
            } else {
                serde_json::from_value(v["state"].clone()).map_err(|e| error(e.to_string()))?
            };
            ensure(
                state
                    .samples
                    .iter()
                    .all(|(blocked, origin)| *blocked >= 0 && *origin <= state.decision)
                    && state.pending.as_ref().is_none_or(|p| {
                        p.attack > 0 && p.damage >= 0 && p.decision <= state.decision
                    }),
                "invalid restored refresh counters",
            )?;
            let index = state.select(input, config, &table, threshold)?;
            let label = format!(
                "heuristic_refresh-card-instance-v2-{}",
                hash(
                    &json!({"options":options,"action_contract":config.action_contract,"rule_signature":config.signature(rules)?,"action_config_hash":hash(&serde_json::to_value(config).unwrap())?})
                )?
            );
            (
                index,
                label,
                Value::Null,
                serde_json::to_value(state).unwrap(),
            )
        }
        _ => return Err(error("unknown native baseline kind")),
    };
    Ok(
        json!({"selection":{"table_hash":table.table_hash,"index":index,"policy_version":label,"logprob":null,"value":null},"rng_state":rng_state,"state":state}),
    )
}
