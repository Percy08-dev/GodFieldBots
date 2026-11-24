//! Own-participant policy core and owned serving boundary; game actors integrate directly.
pub mod baselines;
pub mod batch;
pub mod game;
pub mod history;
pub mod http;
mod journal_candidates;
pub mod numeric;
pub mod metrics;
pub mod requests;
pub mod rng;
pub mod rules;
pub mod scheduler;
pub mod search;
pub mod training;
use crate::{
    budget::{Budget, Control},
    error::{Error, Result},
    json::{self, Value, ValueExt},
    structured_controls::digest,
};
use rules::{Card, Incoming};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::{Arc, LazyLock};
pub(crate) fn error(s: impl Into<String>) -> Error {
    Error::contract(s)
}
pub(crate) fn ensure(ok: bool, s: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(error(s)) }
}
pub fn hash(v: &Value) -> Result<String> {
    digest(v, &Budget::new(12usize << 30), &Control::new(f64::MAX)?)
}
fn canonical(v: &Value) -> Result<Vec<u8>> {
    Ok(json::canonical(v, &Budget::new(12usize << 30), &Control::new(f64::MAX)?)?.to_vec())
}
fn shared_card_instance_contract() -> &'static Value {
    static CONTRACT: LazyLock<Value> = LazyLock::new(|| {
        let mut pins: Value = serde_json::from_slice(include_bytes!(
            "../../assets/card-instance-validation-pins-v2.json"
        ))
        .expect("checked-in card-instance pins");
        let contract = &mut pins["action_contract"];
        contract["profile_id"] = json!("card-instance-simulator-v2-shared-20261118");
        contract["simulator_revision"] = json!("072dabbca1458c9526897781d2fd64edeeb4f9ca");
        contract["spec_inventory_sha256"] =
            json!("da4bde0ce82d91dc10c4a8a7f012c1cb95b0079113344453b956f638fb1d39d4");
        contract.take()
    });
    &CONTRACT
}

fn strength_powder_card_instance_contract() -> &'static Value {
    static CONTRACT: LazyLock<Value> = LazyLock::new(|| {
        let mut contract = shared_card_instance_contract().clone();
        contract["profile_id"] = json!("card-instance-simulator-v2-shared-20261123");
        contract["simulator_revision"] = json!("3172da407e37a4ad57ac82e70a96910c3ac6fbb8");
        contract["spec_inventory_sha256"] =
            json!("6403eef771b64c34e78eb653c2baefe65ceca5308051b506a7092151e9316c59");
        contract
    });
    &CONTRACT
}

fn dream_compat_card_instance_contract() -> &'static Value {
    static CONTRACT: LazyLock<Value> = LazyLock::new(|| {
        let mut contract = strength_powder_card_instance_contract().clone();
        contract["profile_id"] = json!("card-instance-simulator-v2-shared-20261124");
        contract["simulator_revision"] = json!("9e47d4c5f8cd3f60437f9aeabc95076c1dc04321");
        contract["spec_inventory_sha256"] =
            json!("afefd6f3b9113b813c90aaae7bb7b18f1e6299e52bf814ef34ec22112362ab5e");
        contract
    });
    &CONTRACT
}

pub(crate) const MOON_DREAM_PROFILE: &str = "card-instance-simulator-v2-shared-20261125";

fn moon_dream_card_instance_contract() -> &'static Value {
    static CONTRACT: LazyLock<Value> = LazyLock::new(|| {
        let mut contract = dream_compat_card_instance_contract().clone();
        contract["profile_id"] = json!(MOON_DREAM_PROFILE);
        contract["simulator_revision"] = json!("f394a7b57b0ffb12c6b5782ff52fdb8842ec7534");
        contract["spec_inventory_sha256"] =
            json!("c2a0e37607bff98e342b618c5057887cdf6adca981b81731245adf5853cdfc78");
        contract
    });
    &CONTRACT
}

pub(crate) const PESTLE_MIRAGE_PROFILE: &str = "card-instance-simulator-v2-shared-20261125-pestle";

fn pestle_mirage_card_instance_contract() -> &'static Value {
    static CONTRACT: LazyLock<Value> = LazyLock::new(|| {
        let mut contract = moon_dream_card_instance_contract().clone();
        contract["profile_id"] = json!(PESTLE_MIRAGE_PROFILE);
        contract["simulator_revision"] = json!("6d50603dab3f6d2df34ea4e433bab7dca96f52c1");
        contract["spec_inventory_sha256"] =
            json!("05b68761d1c23a3784d04093b417875b2b5893739a1e6c73367a969dc83db753");
        contract
    });
    &CONTRACT
}

fn is_shared_card_instance_contract(contract: &Value) -> bool {
    contract == shared_card_instance_contract()
        || contract == strength_powder_card_instance_contract()
        || contract == dream_compat_card_instance_contract()
        || contract == moon_dream_card_instance_contract()
        || contract == pestle_mirage_card_instance_contract()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub action_contract: Value,
    pub version: String,
    pub generator: String,
    pub use_action_options: bool,
    pub targets: Vec<String>,
    pub max_sequence_length: usize,
    pub max_candidates: usize,
    pub max_expansions: u64,
    pub max_temporary_candidates: usize,
    pub count_states: u64,
    pub count_milliseconds: u64,
    pub search_milliseconds: u64,
    pub channel_weights: [u64; 7],
    #[serde(skip)]
    requires_powder_weapon: bool,
    #[serde(skip)]
    mirage_danger_requires_opponent: bool,
}
impl Config {
    pub fn parse(v: &Value, rules: &Value) -> Result<Self> {
        let mut c: Self = serde_json::from_value(v.clone()).map_err(|e| error(e.to_string()))?;
        ensure(
            is_shared_card_instance_contract(&c.action_contract),
            "approved shared action contract required",
        )?;
        ensure(
            *rules
                == json!({"version":"2","profile_id":"owner-card-instance-candidates-20261117-v2","owner_confirmation_sha256":"bdd1792cde50eae2338b5f068436e40ca87cfbdf0c276ad096651bc0ff27375c"}),
            "current adopted public rules required",
        )?;
        ensure(
            c.version == "3"
                && c.generator == "public-card-instance-diverse-v5"
                && c.use_action_options,
            "current v5 action recipe required",
        )?;
        ensure(
            !c.targets.is_empty()
                && c.targets.len() <= 2
                && c.targets
                    .iter()
                    .all(|t| ["self", "opponent"].contains(&t.as_str()))
                && (c.targets.len() == 1 || c.targets[0] != c.targets[1]),
            "invalid target config",
        )?;
        for (n, max) in [
            (c.max_sequence_length as u64, 128),
            (c.max_candidates as u64, 65536),
            (c.max_expansions, i32::MAX as u64),
            (c.max_temporary_candidates as u64, i32::MAX as u64),
            (c.count_states, i32::MAX as u64),
            (c.count_milliseconds, 1000),
            (c.search_milliseconds, 10000),
        ] {
            ensure(n > 0 && n <= max, "invalid search bound")?;
        }
        ensure(
            c.channel_weights
                .iter()
                .all(|w| *w > 0 && *w <= i32::MAX as u64),
            "invalid channel weight",
        )?;
        let is_pestle_mirage_profile = c.action_contract["profile_id"] == PESTLE_MIRAGE_PROFILE;
        c.mirage_danger_requires_opponent = is_pestle_mirage_profile;
        c.requires_powder_weapon = is_pestle_mirage_profile
            || c.action_contract == *strength_powder_card_instance_contract()
            || c.action_contract == *dream_compat_card_instance_contract()
            || c.action_contract == *moon_dream_card_instance_contract();
        Ok(c)
    }
    fn simulator_binary_sha256(&self) -> &'static str {
        if self.action_contract["profile_id"] == PESTLE_MIRAGE_PROFILE {
            "500d55cd8eb03126f65d3c24fe58b6091bf8acdb2fc20aed76622a8679f4000d"
        } else if self.action_contract == *moon_dream_card_instance_contract() {
            "15137c72e670a90a64fd474030ee9024c91b97b1282c8d103db895f54a2bfa37"
        } else if self.action_contract == *dream_compat_card_instance_contract() {
            "f95fadbc8eaad23fbe63e17ac9bdb0ebeca4b237956dedeab09d86f30cfd94bc"
        } else if self.action_contract == *strength_powder_card_instance_contract() {
            "45ac711b2f172dfd89c23edf30ba0045ab0704c0215a6995d4e8626af1cfb923"
        } else {
            "76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d"
        }
    }
    pub fn signature(&self, rules: &Value) -> Result<String> {
        hash(
            &json!({"generator_version":self.generator,"input_recipe":self.action_contract["input_recipe"],"action_contract":self.action_contract,"rules":rules}),
        )
    }
}
#[derive(Clone, Debug)]
pub enum VisibleEvent {
    Turn(u64),
    Damage { player: u8, amount: i64 },
    Other,
}
#[derive(Clone, Debug)]
pub struct Input {
    pub actor: u8,
    pub episode: String,
    pub revision: u64,
    pub frame_hash: String,
    pub interval_start: usize,
    pub boundary: usize,
    pub hashes: Arc<Vec<[u8; 32]>>,
    pub turn: u64,
    pub choice: String,
    pub action_kinds: Vec<String>,
    pub cards: Vec<Card>,
    pub usable: Vec<bool>,
    pub targets: Vec<String>,
    pub trade: [Vec<usize>; 3],
    pub phenomena: Vec<usize>,
    pub hp: i64,
    pub mp: i64,
    pub money: i64,
    pub flash: bool,
    pub end_times: bool,
    pub can_pray: bool,
    pub can_discard: bool,
    pub can_sacrifice: bool,
    pub can_allow: bool,
    pub context: Incoming,
    pub events: Arc<Vec<VisibleEvent>>,
}
impl Input {
    /// Original public DTO boundary. Corrected-layout frame hashes retain their
    /// historical contract; Config independently binds the shared runtime contract.
    pub fn parse(
        frame: &Value,
        events: &Value,
        config: &Config,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<Self> {
        crate::structured_journal::card_instance::frame(frame, budget, control)?;
        let actor = frame["actor"].as_u64().unwrap() as u8;
        let events = events.array()?;
        let boundary = frame["boundary"]["end"].as_u64().unwrap() as usize;
        let start = frame["interval_start"].as_u64().unwrap() as usize;
        ensure(
            events.len() == boundary,
            "own events must equal exact frame prefix",
        )?;
        let pins: Value = serde_json::from_slice(include_bytes!(
            "../../assets/card-instance-validation-pins-v2.json"
        ))
        .unwrap();
        let mut chain = crate::structured_journal::event_prefix_initial(
            actor as i128,
            &frame["episode"],
            Some(&pins["action_contract"]),
            budget,
            control,
        )?;
        let mut hashes = vec![chain];
        let recipe = crate::structured_tokens::Recipe::corrected(16, budget, control)?;
        let mut typed_events = Vec::with_capacity(events.len());
        for e in events {
            crate::structured_journal::card_instance::event(e, &recipe, budget, control)?;
            chain = crate::structured_journal::event_prefix_next(&chain, e, budget, control)?;
            hashes.push(chain);
            typed_events.push(match e["type"].as_str().unwrap() {
                "turn_started" => VisibleEvent::Turn(e["turn"].as_u64().unwrap()),
                "damage_applied" => VisibleEvent::Damage {
                    player: e["player"].as_u64().unwrap() as u8,
                    amount: e["amount"].as_i64().unwrap(),
                },
                _ => VisibleEvent::Other,
            });
        }
        ensure(
            crate::structured_journal::event_prefix_text(&hashes[boundary])
                == frame["boundary"]["prefix_hash"]
                && crate::structured_journal::event_prefix_text(&hashes[start])
                    == frame["interval_start_hash"],
            "own event prefix binding mismatch",
        )?;
        Self::from_owned_projection(
            frame,
            events,
            config,
            Arc::new(hashes),
            Arc::new(typed_events),
        )
    }
    pub(crate) fn prefix_hash(&self, index: usize) -> String {
        crate::structured_journal::event_prefix_text(&self.hashes[index])
    }
    /// Rust-only live actor boundary. History has already validated each new
    /// event and incrementally bound this exact own frame/prefix. No public DTO
    /// or historical reader may call this in place of its untrusted validation.
    pub(crate) fn from_owned_projection(
        frame: &Value,
        events: &[Value],
        config: &Config,
        hashes: Arc<Vec<[u8; 32]>>,
        typed_events: Arc<Vec<VisibleEvent>>,
    ) -> Result<Self> {
        let raw = &frame["payload"];
        let obs = &raw["observation"];
        let own = &obs["own"];
        let hints = &raw["action_options"]["action_options"];
        let actor = frame["actor"].as_u64().unwrap() as u8;
        let episode = frame["episode"].as_str().unwrap().to_owned();
        let boundary = frame["boundary"]["end"].as_u64().unwrap() as usize;
        let start = frame["interval_start"].as_u64().unwrap() as usize;
        let rows = own["cards"].as_array().unwrap();
        let cards: Vec<_> = rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                Card::new(
                    i,
                    row,
                    &raw["catalog"][row["definition_id"].as_str().unwrap()],
                )
            })
            .collect();
        let refs: std::collections::HashMap<_, _> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r["card_ref"].as_str().unwrap(), i))
            .collect();
        let slots = |key: &str| -> Vec<usize> {
            hints[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| refs[r["card_ref"].as_str().unwrap()])
                .collect()
        };
        let mut usable = vec![false; cards.len()];
        for slot in slots("usable_cards") {
            usable[slot] = true;
        }
        let targets = config
            .targets
            .iter()
            .filter(|t| {
                hints["valid_targets"].as_array().unwrap().iter().any(|v| {
                    v.as_u64()
                        == Some(if t.as_str() == "self" {
                            actor as u64
                        } else {
                            3 - actor as u64
                        })
                })
            })
            .cloned()
            .collect();
        let mut context = Incoming::Unknown;
        if obs["required_choice"] == "defense" && own["status"]["fog"] == false {
            if let Some(i) = events.iter().rposition(|e| e["type"] == "action_accepted") {
                let b = &events[i..];
                if (b.len() == 3 || b.len() == 4)
                    && b[0]["player"].as_u64() == Some((3 - actor) as u64)
                    && b[1]["type"] == "artifact_used"
                    && b[1]["player"] == b[0]["player"]
                    && b[b.len() - 1]["type"] == "attack_declared"
                    && b[b.len() - 1]["attacker"] == b[0]["player"]
                    && b[b.len() - 1]["target"].as_u64() == Some(actor as u64)
                {
                    if b.len() == 4 {
                        if b[2]["type"] == "miracle_used"
                            && b[2]["player"] == b[1]["player"]
                            && b[2]["definition_id"] == b[1]["definition_id"]
                        {
                            context = Incoming::Miracle;
                        }
                    } else if cards.iter().any(|c| {
                        !c.miracle
                            && Some(c.id.as_str()) == b[1]["definition_id"].as_str()
                            && c.role("weapon")
                    }) {
                        context = Incoming::Weapon;
                    }
                }
            }
        }
        Ok(Self {
            actor,
            episode,
            revision: frame["revision"].as_u64().unwrap(),
            frame_hash: frame["frame_hash"].as_str().unwrap().into(),
            interval_start: start,
            boundary,
            hashes,
            turn: obs["turn"].as_u64().unwrap(),
            choice: obs["required_choice"].as_str().unwrap().into(),
            action_kinds: raw["typed_metadata"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d["type"].as_str().unwrap().to_owned())
                .collect(),
            cards,
            usable,
            targets,
            trade: [
                slots("exchange_cards"),
                slots("sell_cards"),
                slots("buy_cards"),
            ],
            phenomena: slots("phenomenon_cards"),
            hp: own["hp"].as_i64().unwrap(),
            mp: own["mp"].as_i64().unwrap(),
            money: own["money"].as_i64().unwrap(),
            flash: own["status"]["flash"].as_bool().unwrap(),
            end_times: obs["end_times"].as_bool().unwrap(),
            can_pray: hints["can_pray"].as_bool().unwrap(),
            can_discard: hints["can_discard"].as_bool().unwrap(),
            can_sacrifice: hints["can_sacrifice"].as_bool().unwrap(),
            can_allow: hints["can_allow"].as_bool().unwrap(),
            context,
            events: typed_events,
        })
    }
}
fn decode(encoded: &[u8]) -> Result<json::Document> {
    let b = Budget::new(12usize << 30);
    json::parse(encoded, &b, &Control::new(f64::MAX)?)
}
pub fn search_diagnostic(encoded: &[u8]) -> Result<Vec<u8>> {
    let doc = decode(encoded)?;
    let v = &doc.root;
    v.closed(
        &[
            "frame",
            "events",
            "actions",
            "rules",
            "numeric_costs",
            "rng_state",
            "clock",
        ],
        &[],
    )?;
    let config = Config::parse(&v["actions"], &v["rules"])?;
    let input = Input::parse(
        &v["frame"],
        &v["events"],
        &config,
        &Budget::new(12usize << 30),
        &Control::new(f64::MAX)?,
    )?;
    let costs = search::Costs::parse(&v["numeric_costs"])?;
    let mut rng = rng::PythonRng::from_state(&v["rng_state"])?;
    let mut clock = search::Clock::parse(&v["clock"])?;
    let generated = search::generate(&input, &config, &v["rules"], &costs, &mut rng, &mut clock)?;
    canonical(&json!({"generated":generated,"rng_state":rng.state(),"clock_calls":clock.calls()}))
}
pub fn select_diagnostic(encoded: &[u8]) -> Result<Vec<u8>> {
    let doc = decode(encoded)?;
    let v = &doc.root;
    v.closed(
        &[
            "frame",
            "events",
            "actions",
            "rules",
            "generated",
            "kind",
            "rng_state",
            "state",
            "refresh_options",
        ],
        &[],
    )?;
    let config = Config::parse(&v["actions"], &v["rules"])?;
    let input = Input::parse(
        &v["frame"],
        &v["events"],
        &config,
        &Budget::new(12usize << 30),
        &Control::new(f64::MAX)?,
    )?;
    let result = baselines::select_diagnostic(&input, &config, &v["rules"], v)?;
    canonical(&result)
}
pub fn rng_diagnostic(encoded: &[u8]) -> Result<Vec<u8>> {
    let doc = decode(encoded)?;
    let v = &doc.root;
    v.closed(&["seed", "bits", "ranges", "shuffle"], &[])?;
    let mut rng = rng::PythonRng::seed(v["seed"].integer(0, u64::MAX as i128)? as u64);
    let mut bits = vec![];
    for k in v["bits"].array()? {
        bits.push(rng.bits(k.integer(0, 64)? as u32)?);
    }
    let mut ranges = vec![];
    for n in v["ranges"].array()? {
        ranges.push(rng.range(n.integer(1, u64::MAX as i128)? as u64)?);
    }
    let mut shuffled: Vec<_> = (0..v["shuffle"].integer(0, 65536)? as usize).collect();
    rng.shuffle(&mut shuffled);
    canonical(
        &json!({"bits":bits,"ranges":ranges,"shuffle":shuffled,"state":rng.state(),"search_seed":rng::search_seed(v["seed"].as_u64().unwrap())?}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_baseline_capacity_rejects_before_refresh_state_mutation() {
        let originals: Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/rust_player_v5/originals.json"
        ))
        .unwrap();
        let episode = &originals["episodes"][3];
        let frame = &episode["decisions"][0]["frame"];
        for maximum in [18, 19, 128] {
            let mut actions = episode["actions"].clone();
            actions["max_sequence_length"] = json!(maximum);
            let config = Config::parse(&actions, &episode["rules"]).unwrap();
            let input = Input::parse(
                frame,
                &json!([]),
                &config,
                &Budget::new(12usize << 30),
                &Control::new(f64::MAX).unwrap(),
            )
            .unwrap();
            let costs = search::Costs::parse(&episode["numeric_costs"]).unwrap();
            let mut rng = rng::PythonRng::seed(17);
            let mut clock = search::Clock::parse(&json!(vec![0.; 64])).unwrap();
            let generated = search::generate_typed(
                &input,
                &config,
                &episode["rules"],
                &costs,
                &mut rng,
                &mut clock,
            )
            .unwrap();
            assert!(!generated.table.kinds.is_empty());
            let heuristic = baselines::heuristic(&input, &config, &generated.table.proposals());
            let mut state = baselines::RefreshState::default();
            let before = serde_json::to_value(&state).unwrap();
            let refresh = state.select(&input, &config, &generated.table, 12);
            if maximum == 18 {
                assert_eq!(heuristic.unwrap(), 0);
                assert_eq!(refresh.unwrap(), 0);
            } else {
                assert_eq!(
                    heuristic.unwrap_err(),
                    error("corrected baseline kind/game capacity required")
                );
                assert_eq!(
                    refresh.unwrap_err(),
                    error("corrected baseline kind/game capacity required")
                );
                assert_eq!(serde_json::to_value(&state).unwrap(), before);
            }
        }
    }
    #[test]
    fn saved_python_v5_tables_reports_streams_and_policy_states() {
        let originals: Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/rust_player_v5/originals.json"
        ))
        .unwrap();
        for episode in originals["episodes"].as_array().unwrap() {
            let mut events = vec![];
            let mut state = Value::Null;
            for decision in episode["decisions"].as_array().unwrap() {
                events.extend(
                    decision["delta"]["events"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .cloned(),
                );
                let payload = json!({"frame":decision["frame"],"events":events,"actions":episode["actions"],"rules":episode["rules"],"numeric_costs":episode["numeric_costs"],"rng_state":decision["search_rng_before"],"clock":decision["clock_trace"].as_array().unwrap().iter().map(|r|r["seconds"].clone()).collect::<Vec<_>>()});
                let actual: Value = serde_json::from_slice(
                    &search_diagnostic(&serde_json::to_vec(&payload).unwrap()).unwrap(),
                )
                .unwrap();
                assert_eq!(
                    actual["generated"], decision["generated"],
                    "episode {}",
                    episode["name"]
                );
                assert_eq!(actual["rng_state"], decision["search_rng_after"]);
                assert_eq!(
                    actual["clock_calls"],
                    decision["clock_trace"].as_array().unwrap().len()
                );
                for kind in ["random", "heuristic", "refresh"] {
                    let expected = &decision["policies"][kind];
                    let payload = json!({"frame":decision["frame"],"events":events,"actions":episode["actions"],"rules":episode["rules"],"generated":actual["generated"],"kind":kind,"rng_state":expected["rng_before"],"state":if kind=="refresh"{state.clone()}else{Value::Null},"refresh_options":episode["refresh_options"]});
                    let result: Value = serde_json::from_slice(
                        &select_diagnostic(&serde_json::to_vec(&payload).unwrap()).unwrap(),
                    )
                    .unwrap();
                    assert_eq!(result["selection"], expected["selection"]);
                    assert_eq!(result["rng_state"], expected["rng_after"]);
                    if kind == "refresh" {
                        state = result["state"].clone();
                    }
                }
            }
        }
    }
}
