//! Own-participant event bank and corrected-layout decision continuity.
use super::{Config, Input, VisibleEvent, ensure, hash};
use crate::{
    budget::{Budget, Control},
    error::Result,
    json::{Value, ValueExt},
    structured_journal::{self, card_instance::{self, WireProfile}},
};
use serde_json::json;
use std::sync::Arc;

pub struct History {
    pub actor: u8,
    pub episode: String,
    pub events: Vec<Value>,
    pub cursor: Option<String>,
    prefixes: Arc<Vec<[u8; 32]>>,
    chain: [u8; 32],
    typed: Arc<Vec<VisibleEvent>>,
    last: usize,
    revision: Option<u64>,
    budget: Arc<Budget>,
    profile: WireProfile,
    recipe: crate::structured_tokens::Recipe,
}
impl History {
    pub fn new(actor: u8, episode: String) -> Result<Self> {
        Self::new_for_profile(actor, episode, WireProfile::V2)
    }

    pub(super) fn new_for_profile(actor: u8, episode: String, profile: WireProfile) -> Result<Self> {
        let budget = Budget::new(12usize << 30);
        let c = Control::new(f64::MAX)?;
        let initial = structured_journal::event_prefix_initial(
            actor as i128,
            &json!(episode),
            Some(profile.frame_contract()),
            &budget,
            &c,
        )?;
        let recipe = crate::structured_tokens::Recipe::corrected(16, &budget, &c)?;
        Ok(Self {
            actor,
            episode,
            events: vec![],
            cursor: None,
            prefixes: Arc::new(vec![initial]),
            chain: initial,
            typed: Arc::new(vec![]),
            last: 0,
            revision: None,
            budget,
            profile,
            recipe,
        })
    }
    pub(super) fn event_recipe(&self) -> &crate::structured_tokens::Recipe {
        &self.recipe
    }

    pub fn append(
        &mut self,
        events: Vec<Value>,
        cursor: Option<String>,
        max_episode: usize,
        max_interval: usize,
    ) -> Result<()> {
        ensure(
            self.events.len() + events.len() <= max_episode,
            "model episode event capacity exceeded",
        )?;
        ensure(
            self.events.len() + events.len() - self.last <= max_interval,
            "model interval event capacity exceeded",
        )?;
        let c = Control::new(f64::MAX)?;
        for e in &events {
            card_instance::event_for_profile(e, &self.recipe, &self.budget, &c, self.profile, Some(self.actor as i128))?;
        }
        let prefixes = Arc::get_mut(&mut self.prefixes)
            .ok_or_else(|| super::error("own prefix still borrowed by a live decision"))?;
        let typed = Arc::get_mut(&mut self.typed)
            .ok_or_else(|| super::error("own typed events still borrowed by a live decision"))?;
        for e in events {
            self.chain = structured_journal::event_prefix_next(&self.chain, &e, &self.budget, &c)?;
            prefixes.push(self.chain);
            typed.push(match e["type"].as_str().unwrap() {
                "turn_started" => VisibleEvent::Turn(e["turn"].as_u64().unwrap()),
                "damage_applied" => VisibleEvent::Damage {
                    player: e["player"].as_u64().unwrap() as u8,
                    amount: e["amount"].as_i64().unwrap(),
                },
                _ => VisibleEvent::Other,
            });
            self.events.push(e);
        }
        self.cursor = cursor;
        Ok(())
    }
    pub fn input(&self, frame: &Value, config: &Config) -> Result<Input> {
        ensure(self.profile == config.wire_profile(), "history policy profile mismatch")?;
        Input::from_owned_projection(
            frame,
            &self.events,
            config,
            self.prefixes.clone(),
            self.typed.clone(),
        )
    }
    pub fn frame(
        &mut self,
        observation: Value,
        mut options: Value,
        definitions: &Value,
    ) -> Result<(Value, Value)> {
        let c = Control::new(f64::MAX)?;
        // Http::observe already validated this exact participant observation.
        ensure(
            observation["viewer_player_id"] == self.actor
                && observation["current_player"] == self.actor
                && observation["actionable"] == true,
            "own actionable HTTP observation required",
        )?;
        let revision = observation["state_revision"].as_u64().unwrap();
        ensure(
            self.revision.is_none_or(|r| revision > r),
            "own decision revision must advance",
        )?;
        ensure(
            options["state_revision"] == revision,
            "ActionOptions observation revision mismatch",
        )?;
        options.closed(&["state_revision", "action_options"], &[])?;
        options["viewer_player_id"] = json!(self.actor);
        ensure(observation["rule_set_version"] == self.profile.rule_set(), "history observation profile mismatch")?;
        card_instance::hints_for_profile(&options, &observation, &c, self.profile)?;
        let cards = observation["own"]["cards"].as_array().unwrap();
        let mut catalog = serde_json::Map::new();
        for card in cards {
            let id = card["definition_id"].as_str().unwrap();
            let definition = definitions
                .get(id)
                .ok_or_else(|| super::error("missing perceived own definition"))?;
            ensure(
                definition["classification"]["definition_class"] == "owned_card"
                    && (card["kind"] == "miracle") == (definition["kind"] == "miracles"),
                "perceived kind/class mismatch",
            )?;
            catalog.insert(id.into(), definition.clone());
        }
        let boundary = json!({"end":self.events.len(),"prefix_hash":structured_journal::event_prefix_text(self.prefixes.last().unwrap())});
        let delta = json!({"actor":self.actor,"episode":self.episode,"start":self.last,"previous_hash":structured_journal::event_prefix_text(&self.prefixes[self.last]),"events":self.events[self.last..]});
        let mut frame = json!({"actor":self.actor,"episode":self.episode,"revision":revision,"interval_start":self.last,"interval_start_hash":structured_journal::event_prefix_text(&self.prefixes[self.last]),"boundary":boundary,"payload":{"typed_metadata":observation["typed_action_schema"],"observation":observation,"catalog":catalog,"action_options":options,"history_end":self.cursor}});
        let mut binding = frame.clone();
        let contract = self.profile.frame_contract();
        for key in ["input_recipe", "policy_input_version"] {
            binding[key] = contract[key].clone();
        }
        binding["action_contract"] = contract.clone();
        frame["frame_hash"] = json!(hash(&binding)?);
        // Frame is generated here from validated HTTP and pinned own definitions.
        self.last = self.events.len();
        self.revision = Some(revision);
        Ok((frame, delta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defense_v3_fixture() -> Value {
        serde_json::from_slice(include_bytes!("../../tests/fixtures/defense_v3_live_own_view.json"))
            .unwrap()
    }

    #[test]
    fn defense_v3_raw_history_and_tokens_preserve_fog_without_an_attacker() {
        let fixture = defense_v3_fixture();
        let config = Config::parse(&fixture["actions"], &fixture["rules"]).unwrap();
        assert_eq!(
            config.signature(&fixture["rules"]).unwrap(),
            config.wire_profile().rule_signature().as_str().unwrap(),
        );
        let real = &fixture["fog"];
        let original = &real["frame"];
        let payload = &original["payload"];
        let episode = original["episode"].as_str().unwrap();
        let mut history = History::new_for_profile(2, episode.into(), config.wire_profile()).unwrap();
        let control = Control::new(f64::MAX).unwrap();
        let rows = real["events"].as_array().unwrap();
        history.append(rows.clone(), payload["history_end"].as_str().map(str::to_owned), 128, 128).unwrap();
        let mut options = payload["action_options"].clone();
        options.as_object_mut().unwrap().remove("viewer_player_id");
        let (frame, delta) = history
            .frame(payload["observation"].clone(), options, &payload["catalog"])
            .unwrap();
        assert_eq!(frame, *original);
        assert_eq!(delta["events"], real["events"]);
        assert!(card_instance::frame(&frame, &history.budget, &control).is_err());
        Input::parse(&frame, &real["events"], &config, &history.budget, &control).unwrap();

        let tokens = crate::structured_tokens::frame_tokens_for_profile(
            &frame, rows, &history.recipe, &history.budget, &control, config.wire_profile(),
        ).unwrap().value;
        let fog_row = rows.iter().filter(|event| !card_instance::is_concealed_draw(event))
            .position(|event| event["type"] == "attack_declared" && event.get("attacker").is_none())
            .unwrap();
        assert_eq!(tokens["interval_categories"][fog_row][5], 1); // missing attacker
        assert_eq!(tokens["interval_categories"][fog_row][6], 2); // self target
        assert_eq!(tokens["interval_numerics"][fog_row][3], 3); // punch damage
        assert_eq!(tokens["interval_present"][fog_row][3], true);
    }

    #[test]
    fn defense_v3_anonymous_attack_forms_are_closed_and_viewer_bound() {
        let mut legacy = History::new(2, "fog-view".into()).unwrap();
        let mut history = History::new_for_profile(2, "fog-view".into(), WireProfile::DefenseV3).unwrap();
        let declaration = json!({"type":"attack_declared","target":2,"attack":30,
                                 "attribute":"fire","definition_ids":["fireball","fireball"]});
        let miss = json!({"type":"attack_missed","target":2});
        for valid in [declaration.clone(), miss.clone()] {
            assert!(legacy.append(vec![valid.clone()], None, 8, 8).is_err());
            history.append(vec![valid], None, 8, 8).unwrap();
        }
        for (key, bad) in [
            ("attacker", json!(1)),
            ("target", json!(1)),
            ("attribute", json!("neutral")),
            ("definition_ids", json!(["unknown-definition"])),
        ] {
            let mut altered = declaration.clone();
            altered[key] = bad;
            assert!(history.append(vec![altered], None, 8, 8).is_err());
        }
        assert!(history.append(vec![json!({"type":"attack_missed","target":1})], None, 8, 8).is_err());
        assert_eq!(history.events, [declaration, miss]);
        let named = json!({"type":"attack_declared","attacker":2,"target":1,"attack":30});
        legacy.append(vec![named.clone()], None, 8, 8).unwrap();
        history.append(vec![named], None, 8, 8).unwrap();
    }

    #[test]
    fn defense_v3_p46_keeps_cut_cost_support_and_requires_direct_or_first_rainbow() {
        use super::super::{rng::PythonRng, rules::{Card, Family}, search};
        let fixture = defense_v3_fixture();
        let config = Config::parse(&fixture["actions"], &fixture["rules"]).unwrap();
        let budget = Budget::new(256 << 20);
        let control = Control::new(f64::MAX).unwrap();
        let mut input = Input::parse(&fixture["p46"]["frame"], &fixture["p46"]["events"], &config, &budget, &control).unwrap();
        let hood = input.cards.iter().position(|card| card.id == "spiritual-hood").unwrap();
        let turbulence = input.cards.iter().position(|card| card.id == "turbulence").unwrap();
        assert_eq!(input.mp, 0);
        assert_eq!(input.defense_direct, Some(1 << turbulence));
        assert!(input.usable[hood]);
        let costs = search::Costs::parse(&fixture["numeric_costs"]).unwrap();
        let mut rng = PythonRng::seed(17);
        let mut clock = search::Clock::parse(&json!(vec![0.; 256])).unwrap();
        let candidates = search::generate_typed(&input, &config, &fixture["rules"], &costs, &mut rng, &mut clock).unwrap();
        assert_eq!(candidates.table.kinds, ["respond", "respond"]);
        assert_eq!(candidates.table.offsets, [0, 0, 2]);
        assert_eq!(candidates.table.slots, [turbulence, hood]);

        let rainbow = input.cards.len();
        let old: Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/rust_player_v5/originals.json"
        )).unwrap();
        let payload = &old["episodes"].as_array().unwrap().iter()
            .find(|row| row["name"] == "defense-rainbow-first")
            .unwrap()["decisions"][0]["frame"]["payload"];
        input.cards.push(Card::new(rainbow, &payload["observation"]["own"]["cards"][0],
                                   &payload["catalog"]["rainbow-curtain"]));
        input.usable.push(true);
        input.defense_direct = Some((1 << turbulence) | (1 << rainbow));
        let complete = |input: &Input, refs: &[usize]| {
            search::Space::new(input, &config).unwrap().analysis(Family::Defense, refs).complete
        };
        assert!(complete(&input, &[rainbow, hood]));
        assert!(!complete(&input, &[hood, rainbow]));
        assert!(!complete(&input, &[rainbow, turbulence, hood])); // Turbulence is exclusive
        assert!(!complete(&input, &[turbulence, rainbow, hood]));
        input.flash = true;
        assert!(!complete(&input, &[turbulence, hood]));
        assert!(!complete(&input, &[rainbow, hood]));
        input.defense_direct = Some((1 << turbulence) | (1 << hood));
        assert!(complete(&input, &[hood])); // neutral attack's direct hint
        input.flash = false;
        input.defense_direct = Some(1 << turbulence);
        assert!(!complete(&input, &[rainbow, hood])); // Rainbow needs its own direct hint to clear
    }

    #[test]
    fn defense_v3_direct_group_requires_the_exact_actor_subset_and_usable_order() {
        let fixture = defense_v3_fixture();
        let payload = &fixture["p46"]["frame"]["payload"];
        let observation = &payload["observation"];
        let options = &payload["action_options"];
        let control = Control::new(f64::MAX).unwrap();
        let check = |options: &Value| card_instance::hints_for_profile(options, observation, &control, WireProfile::DefenseV3);
        check(options).unwrap();
        let mut altered = options.clone();
        let groups = altered["action_options"]["candidate_groups"].as_object_mut().unwrap();
        groups.remove("defense:direct");
        assert!(check(&altered).is_err());
        altered["action_options"]["candidate_groups"]["defense:direct"] = json!([]);
        check(&altered).unwrap();
        let mut unordered = options["action_options"]["usable_cards"].as_array().unwrap().clone();
        unordered.reverse();
        altered["action_options"]["candidate_groups"]["defense:direct"] = json!(unordered);
        assert!(check(&altered).is_err());
        altered["action_options"]["candidate_groups"]["defense:direct"] = json!([
            {"card_ref":"c1_ffffffffffffffffffffffffffffffff","definition_id":"turbulence"}
        ]);
        assert!(check(&altered).is_err());
    }

    #[test]
    fn concealed_draws_count_as_raw_events_but_never_serving_or_learner_rows() {
        use super::super::{game::storage::Mailbox, numeric::ServingRow, rng::PythonRng, search};

        let originals: Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/rust_player_v5/originals.json"
        ))
        .unwrap();
        let episode = &originals["episodes"][3];
        let first = &episode["decisions"][0]["frame"];
        let payload = &first["payload"];
        let actor = first["actor"].as_u64().unwrap() as u8;
        let config = Config::parse(&episode["actions"], &episode["rules"]).unwrap();
        let costs = search::Costs::parse(&episode["numeric_costs"]).unwrap();
        let control = Control::new(f64::MAX).unwrap();
        let owner = Mailbox::new(64 * 1024 * 1024, 1).unwrap();
        let marker =
            json!({"type": "artifact_drawn", "player": actor, "definition_id": "concealed"});
        for count in [0, 1] {
            let mut history = History::new(actor, "concealed-draw".into()).unwrap();
            let mut source = vec![marker.clone(); 8];
            if count == 1 {
                source.insert(
                    4,
                    json!({"type":"artifact_drawn", "player":actor, "definition_id":"bronze-club"}),
                );
            }
            assert!(history.append(source.clone(), None, 16, 7).is_err());
            history.append(source.clone(), None, 16, 16).unwrap();
            let mut options = payload["action_options"].clone();
            options.as_object_mut().unwrap().remove("viewer_player_id");
            let (frame, delta) = history
                .frame(payload["observation"].clone(), options, &payload["catalog"])
                .unwrap();
            assert_eq!(delta["events"], json!(source));
            let input = history.input(&frame, &config).unwrap();
            assert_eq!(input.boundary, count + 8);
            let mut rng = PythonRng::seed(17);
            let mut clock = search::Clock::parse(&json!(vec![0.; 1024])).unwrap();
            let generated = search::generate_typed(
                &input,
                &config,
                &episode["rules"],
                &costs,
                &mut rng,
                &mut clock,
            )
            .unwrap();
            let (serving, _row) = ServingRow::build_training(
                &frame,
                &history.events,
                &input,
                &config,
                &generated.table,
                128,
                18,
                &history.budget,
                &control,
                &owner.account().account,
            )
            .unwrap();
            assert_eq!(serving.events, count);
            for (name, width) in [
                ("interval_numerics", 11),
                ("interval_present", 11),
                ("interval_categories", 12),
            ] {
                assert_eq!(serving.fields[name].shape(), [count, width]);
            }
        }
    }

    #[test]
    fn saved_original_event_banks_frames_and_deltas_remain_exact() {
        let saved: Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/rust_player_v5/originals.json"
        ))
        .unwrap();
        for episode in saved["episodes"].as_array().unwrap() {
            let first = &episode["decisions"][0]["frame"];
            let mut history = History::new(
                first["actor"].as_u64().unwrap() as u8,
                first["episode"].as_str().unwrap().into(),
            )
            .unwrap();
            for decision in episode["decisions"].as_array().unwrap() {
                let frame = &decision["frame"];
                let payload = &frame["payload"];
                history
                    .append(
                        decision["delta"]["events"].as_array().unwrap().clone(),
                        payload["history_end"].as_str().map(str::to_owned),
                        262144,
                        65536,
                    )
                    .unwrap();
                let mut raw_options = payload["action_options"].clone();
                raw_options
                    .as_object_mut()
                    .unwrap()
                    .remove("viewer_player_id");
                let (actual, delta) = history
                    .frame(
                        payload["observation"].clone(),
                        raw_options,
                        &payload["catalog"],
                    )
                    .unwrap();
                assert_eq!(actual, *frame);
                assert_eq!(delta, decision["delta"]);
                let config = Config::parse(&episode["actions"], &episode["rules"]).unwrap();
                let control = Control::new(f64::MAX).unwrap();
                let trusted = history.input(&actual, &config).unwrap();
                assert!(Arc::ptr_eq(&trusted.hashes, &history.prefixes));
                assert!(Arc::ptr_eq(&trusted.events, &history.typed));
                let checked = Input::parse(
                    &actual,
                    &json!(history.events),
                    &config,
                    &history.budget,
                    &control,
                )
                .unwrap();
                assert_eq!(trusted.hashes, checked.hashes);
                let costs = super::super::search::Costs::parse(&episode["numeric_costs"]).unwrap();
                let make = |input: &Input| {
                    let mut rng =
                        super::super::rng::PythonRng::from_state(&decision["search_rng_before"])
                            .unwrap();
                    let values: Vec<_> = decision["clock_trace"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v["seconds"].clone())
                        .collect();
                    let mut clock = super::super::search::Clock::parse(&json!(values)).unwrap();
                    super::super::search::generate_typed(
                        input,
                        &config,
                        &episode["rules"],
                        &costs,
                        &mut rng,
                        &mut clock,
                    )
                    .unwrap()
                };
                let generated = make(&trusted);
                assert_eq!(generated.value(), make(&checked).value());
                assert_eq!(generated.value(), decision["generated"]);
                let a = super::super::numeric::ServingRow::build_owned(
                    &actual,
                    &history.events,
                    &trusted,
                    &config,
                    &generated.table,
                    128,
                    18,
                    &history.budget,
                    &control,
                )
                .unwrap();
                let b = super::super::numeric::ServingRow::build(
                    &actual,
                    &history.events,
                    &checked,
                    &config,
                    &generated.table,
                    128,
                    18,
                    &history.budget,
                    &control,
                )
                .unwrap();
                assert_buffers(&a, &b);
                drop(trusted);
                drop(checked);
                assert_eq!(Arc::strong_count(&history.prefixes), 1);
                assert_eq!(Arc::strong_count(&history.typed), 1);
            }
        }
    }
    fn assert_buffers(
        a: &super::super::numeric::ServingRow,
        b: &super::super::numeric::ServingRow,
    ) {
        use super::super::numeric::Buffer;
        assert_eq!(a.fields.len(), b.fields.len());
        for (name, left) in &a.fields {
            match (left, &b.fields[name]) {
                (Buffer::Float(x, xs), Buffer::Float(y, ys)) => {
                    assert_eq!(x, y, "{name}");
                    assert_eq!(xs, ys);
                }
                (Buffer::Long(x, xs), Buffer::Long(y, ys)) => {
                    assert_eq!(x, y, "{name}");
                    assert_eq!(xs, ys);
                }
                (Buffer::Bool(x, xs), Buffer::Bool(y, ys)) => {
                    assert_eq!(x, y, "{name}");
                    assert_eq!(xs, ys);
                }
                _ => panic!("dtype changed: {name}"),
            }
        }
    }
    #[tokio::test(flavor = "current_thread")]
    async fn incremental_owner_rejects_new_bad_delta_and_reclaims_borrows_on_abort_or_error() {
        let saved: Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/rust_player_v5/originals.json"
        ))
        .unwrap();
        let e = &saved["episodes"][3];
        let frame = &e["decisions"][0]["frame"];
        let config = Config::parse(&e["actions"], &e["rules"]).unwrap();
        let mut history = History::new(1, frame["episode"].as_str().unwrap().into()).unwrap();
        let before = history.prefixes.clone();
        assert!(
            history
                .append(vec![json!({"type":"private_unknown"})], None, 128, 8)
                .is_err()
        );
        assert!(history.events.is_empty() && Arc::ptr_eq(&before, &history.prefixes));
        drop(before);
        let input = history.input(frame, &config).unwrap();
        assert!(history.append(vec![], None, 128, 8).is_err());
        let owner = tokio::spawn(async move {
            let _input = input;
            std::future::pending::<()>().await;
        });
        tokio::task::yield_now().await;
        owner.abort();
        assert!(owner.await.unwrap_err().is_cancelled());
        history.append(vec![], None, 128, 8).unwrap();
        let failed = (|| -> Result<()> {
            let _input = history.input(frame, &config)?;
            Err(super::super::error("inference failed"))
        })();
        assert!(failed.is_err());
        history.append(vec![], None, 128, 8).unwrap();
        assert_eq!(Arc::strong_count(&history.prefixes), 1);
        assert_eq!(Arc::strong_count(&history.typed), 1);
    }
    #[test]
    fn bounded_live_projection_cost_excludes_whole_prefix_validation() {
        use std::time::Instant;
        let saved: Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/rust_player_v5/originals.json"
        ))
        .unwrap();
        let e = &saved["episodes"][3];
        let original = &e["decisions"][0]["frame"];
        let config = Config::parse(&e["actions"], &e["rules"]).unwrap();
        for size in [0, 1024, 8192] {
            let mut history = History::new(1, "bounded-projection".into()).unwrap();
            let at = Instant::now();
            history.append(vec![json!({"type":"resource_changed","player":1,"hp_delta":0,"mp_delta":0,"money_delta":0});size],None,16384,16384).unwrap();
            let admission = at.elapsed();
            let mut raw = original["payload"]["action_options"].clone();
            raw.as_object_mut().unwrap().remove("viewer_player_id");
            let (frame, _) = history
                .frame(
                    original["payload"]["observation"].clone(),
                    raw,
                    &original["payload"]["catalog"],
                )
                .unwrap();
            let at = Instant::now();
            for _ in 0..100 {
                let input = history.input(&frame, &config).unwrap();
                assert!(Arc::ptr_eq(&input.hashes, &history.prefixes));
                std::hint::black_box(input);
            }
            let typed = at.elapsed();
            let events = json!(history.events);
            let at = Instant::now();
            let checked = Input::parse(
                &frame,
                &events,
                &config,
                &history.budget,
                &Control::new(f64::MAX).unwrap(),
            )
            .unwrap();
            std::hint::black_box(checked);
            eprintln!(
                "history={size} new_event_admission_us={} trusted_100_us={} diagnostic_once_us={}",
                admission.as_micros(),
                typed.as_micros(),
                at.elapsed().as_micros()
            );
        }
    }
}
