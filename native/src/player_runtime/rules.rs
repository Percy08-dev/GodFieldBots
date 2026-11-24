//! Adopted public candidate rules only; never a game legality oracle.
use serde::{Deserialize, Serialize};
use serde_json::Value;
pub const SAT: i64 = i64::MAX;
pub const UTILITY: &[&str] = &[
    "smile-dew",
    "heart-dew",
    "romance-water",
    "galaxy-geyser",
    "smile-flower",
    "heart-flower",
    "romance-fragrance",
    "spring",
    "treasure",
    "tone",
    "song",
    "smile-shell",
    "heart-shell",
    "thump-thump-tear",
    "heaven-herb",
    "goddess-s-soap",
    "nocturnal-broom",
    "guardian-pot",
    "release",
];
const EXCLUSIVE: &[&str] = &[
    "reflectAnything",
    "reflectWeapon",
    "reflectMiracle",
    "bounceWeapon",
    "bounceMiracle",
];
const RINGS: &[&str] = &[
    "counterAtk",
    "counterCurse",
    "counterBoost2xMP",
    "counter2xAtk",
    "counterTakeCP",
];
#[derive(Clone, Debug)]
pub struct Card {
    pub slot: usize,
    pub miracle: bool,
    pub hand: bool,
    pub id: String,
    pub kind: String,
    pub ability: String,
    pub subclass: String,
    pub roles: Vec<String>,
    pub cost: i64,
    pub attack: i64,
    pub defense: i64,
    pub price: i64,
    pub ability_value: i64,
    pub attribute: String,
}
impl Card {
    pub fn new(slot: usize, row: &Value, d: &Value) -> Self {
        let string =
            |v: &Value, k: &str, default: &str| v[k].as_str().unwrap_or(default).to_owned();
        let num = |k: &str, default: i64| d[k].as_i64().unwrap_or(default);
        let miracle = row["kind"] == "miracle";
        Self {
            slot,
            miracle,
            hand: !miracle || row["state"] == "unused",
            id: string(d, "definition_id", ""),
            kind: string(d, "kind", ""),
            ability: string(d, "ability", ""),
            subclass: string(&d["classification"], "miracle_subclass", ""),
            roles: d["classification"]["use_roles"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r.as_str().unwrap().to_owned())
                .collect(),
            cost: num("mp_cost", 0),
            attack: num("attack", 0),
            defense: num("defense", 0),
            price: num("price", 0),
            ability_value: num("ability_value", 2),
            attribute: string(d, "attribute", "neutral"),
        }
    }
    pub fn role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }
    pub fn additional(&self) -> bool {
        self.role("additional_attack")
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Offense,
    Defense,
    Utility,
    Status,
    Discard,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Incoming {
    Unknown,
    Miracle,
    Weapon,
}
#[derive(Clone, Debug, Default)]
pub struct Analysis {
    pub viable: bool,
    pub complete: bool,
    pub payment: i64,
    pub spent: i64,
    pub effective: Vec<usize>,
    pub attack: i64,
    pub attribute: String,
    pub effects: Vec<String>,
    pub opponent_only: bool,
}
pub fn defense_kind(c: &Card) -> Option<bool> {
    if EXCLUSIVE.contains(&c.ability.as_str())
        || (["blockWeapon", "blockMiracle"].contains(&c.ability.as_str()) && c.defense <= 0)
    {
        Some(true)
    } else if c.role("numeric_defense")
        || c.defense > 0
        || RINGS.contains(&c.ability.as_str())
        || c.ability == "filterAtkElement"
    {
        Some(false)
    } else {
        None
    }
}
fn attribute(left: Option<&str>, right: &str) -> String {
    match left {
        None => right.to_owned(),
        Some(l) if l == right => right.to_owned(),
        Some("light") if ["fire", "water", "wood", "stone"].contains(&right) => right.to_owned(),
        Some(l) if right == "light" && ["fire", "water", "wood", "stone"].contains(&l) => {
            l.to_owned()
        }
        _ => "neutral".to_owned(),
    }
}
pub fn analyze(
    cards: &[Card],
    refs: &[usize],
    family: Family,
    mp: i64,
    flash: bool,
    maximum: usize,
    usable: &[bool],
    context: Incoming,
    requires_powder_weapon: bool,
    mirage_danger_requires_opponent: bool,
) -> Analysis {
    let empty = || Analysis {
        attribute: "neutral".into(),
        ..Default::default()
    };
    if refs.is_empty()
        || refs.len() > maximum.min(18)
        || family == Family::Defense && flash && refs.len() > 1
    {
        return empty();
    }
    let mut mask = 0u32;
    for &r in refs {
        if r >= cards.len() || !usable[r] || mask & (1 << r) != 0 {
            return empty();
        }
        mask |= 1 << r;
    }
    let mut suppressed = 0u32;
    let mut waived = 0u32;
    for i in 1..refs.len() {
        if cards[refs[i]].ability == "cutCost" && cards[refs[i - 1]].miracle {
            suppressed |= 1 << refs[i];
            waived |= 1 << refs[i - 1];
        }
    }
    let effective: Vec<usize> = refs
        .iter()
        .copied()
        .filter(|r| suppressed & (1 << r) == 0)
        .collect();
    let mut uncertain = false;
    let (mut has_powder, mut has_weapon) = (false, false);
    match family {
        Family::Utility | Family::Status => {
            if effective.len() != 1 {
                return empty();
            }
            let root = &cards[effective[0]];
            if family == Family::Utility && !UTILITY.contains(&root.id.as_str())
                || family == Family::Status && root.subclass != "status"
            {
                return empty();
            }
        }
        Family::Defense => {
            if refs
                .iter()
                .enumerate()
                .any(|(i, r)| i != 0 && cards[*r].ability == "filterAtkElement")
            {
                return empty();
            }
            for &r in &effective {
                let c = &cards[r];
                let Some(mut exclusive) = defense_kind(c) else {
                    return empty();
                };
                if c.defense > 0
                    && ["blockMiracle", "bounceMiracle", "reflectMiracle"]
                        .contains(&c.ability.as_str())
                {
                    match context {
                        Incoming::Miracle => exclusive = true,
                        Incoming::Weapon => exclusive = false,
                        Incoming::Unknown if effective.len() > 1 => {
                            exclusive = false;
                            uncertain = true;
                        }
                        _ => (),
                    }
                }
                if exclusive && effective.len() != 1 {
                    return empty();
                }
            }
        }
        Family::Offense => {
            let normal: Vec<_> = effective
                .iter()
                .copied()
                .filter(|r| cards[*r].miracle && cards[*r].subclass == "weapon")
                .collect();
            let artifacts: Vec<_> = effective
                .iter()
                .copied()
                .filter(|r| {
                    !cards[*r].miracle && (cards[*r].role("weapon") || cards[*r].additional())
                })
                .collect();
            if !normal.is_empty()
                && (!artifacts.is_empty()
                    || effective[0] != normal[0]
                    || normal.iter().any(|r| cards[*r].id != cards[normal[0]].id))
            {
                return empty();
            }
            for (i, &r) in effective.iter().enumerate() {
                let c = &cards[r];
                if requires_powder_weapon {
                    has_powder |= c.id == "strength-powder";
                    has_weapon |= c.kind == "weapons";
                }
                if c.miracle {
                    if !["weapon", "additional_weapon", "attack_support"]
                        .contains(&c.subclass.as_str())
                        || c.subclass == "attack_support" && i == 0
                    {
                        return empty();
                    }
                } else if !artifacts.contains(&r) || !c.additional() && i != 0 {
                    return empty();
                }
            }
        }
        Family::Discard => return empty(),
    }
    let payment: i64 = refs
        .iter()
        .map(|r| {
            if cards[*r].miracle && waived & (1 << r) == 0 {
                cards[*r].cost
            } else {
                0
            }
        })
        .sum();
    let affordable = payment <= mp;
    let last = &cards[*refs.last().unwrap()];
    let salvage = !affordable
        && last.miracle
        && refs.len() < maximum.min(18)
        && payment
            - if waived & (1 << last.slot) != 0 {
                0
            } else {
                last.cost
            }
            <= mp
        && !(family == Family::Defense && flash)
        && cards
            .iter()
            .any(|c| mask & (1 << c.slot) == 0 && usable[c.slot] && c.ability == "cutCost");
    let (mut attack, mut attr, mut wand, mut reset, mut mirages) =
        (0i64, None::<String>, None::<String>, false, 0i64);
    if family == Family::Offense {
        for &r in &effective {
            let c = &cards[r];
            if c.ability == "doubleAtk" {
                attack = attack.saturating_mul(c.ability_value.max(1));
            } else if c.ability == "attackEveryEnemy" {
                attr = Some("neutral".into());
                wand = None;
                reset = true;
                mirages += 1;
            } else {
                let amount = if c.ability == "atkBy2xMP" {
                    2 * (mp - payment).max(0)
                } else {
                    c.attack.max(0)
                };
                attack = attack.saturating_add(amount);
                if amount != 0 || c.ability == "atkBy2xMP" {
                    if c.ability == "setElement" {
                        attr = Some(if reset {
                            c.attribute.clone()
                        } else {
                            attribute(attr.as_deref(), &c.attribute)
                        });
                        wand = Some(c.attribute.clone());
                    } else {
                        attr = Some(attribute(
                            attr.as_deref(),
                            if c.additional() {
                                wand.as_deref().unwrap_or(&c.attribute)
                            } else {
                                &c.attribute
                            },
                        ));
                    }
                }
            }
        }
    }
    let rounds = if effective.iter().any(|r| cards[*r].ability == "attackTwice") {
        2
    } else {
        1
    } * mirages.max(1);
    let spent = if family == Family::Offense
        && effective.iter().any(|r| cards[*r].ability == "atkBy2xMP")
    {
        mp
    } else {
        payment
    };
    let mut effects: Vec<_> = effective
        .iter()
        .map(|r| cards[*r].ability.clone())
        .filter(|s| !s.is_empty())
        .collect();
    effects.sort();
    effects.dedup();
    Analysis {
        viable: affordable || salvage,
        complete: affordable && !uncertain && (!has_powder || has_weapon),
        payment,
        spent,
        attack: attack.saturating_mul(rounds),
        attribute: attr.unwrap_or("neutral".into()),
        effects,
        opponent_only: mirages > 0
            && (mirage_danger_requires_opponent
                || !effective.iter().any(|r| cards[*r].ability == "danger")),
        effective,
    }
}
