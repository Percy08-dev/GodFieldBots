//! Native Journal6 writer for the current owned HPO collection.
//! The caller retains the exact Mailbox record until validation and output
//! custody succeed. This module cannot mint a durable collection or Source.
use super::super::{
    ensure, error,
    http::Http,
    journal_candidates,
    rng::{PythonRng, search_seed},
    search::Table,
    training::Episode,
};
use super::capture::OwnedAction;
use crate::{
    budget::{Budget, BudgetVec, Control, JsonAdmission},
    error::Result,
    json::{self as wire, Value, ValueExt},
    structured_controls::{SourceFormat, digest},
    structured_journal::{
        self as journal, card_instance::{self, WireProfile}, event_prefix_initial, event_prefix_next, event_prefix_text,
    },
    structured_source::episode_digest_context,
};
use serde::{
    Deserializer,
    de::{DeserializeSeed, IgnoredAny, MapAccess, Visitor},
};
use serde_json::{Map, json};
use sha2::{Digest, Sha256};
use std::{
    fmt,
    sync::Arc,
};

const CODEC: &str = "candidate-columns-v2";
type Member = (String, BudgetVec<u8>);

#[derive(Clone, Copy)]
enum PendingProjection {
    Pending,
    Generated,
    Table,
}

impl<'de> DeserializeSeed<'de> for PendingProjection {
    type Value = Value;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Value, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for PendingProjection {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a pending capture object, generated object or candidate table object")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> std::result::Result<Value, M::Error> {
        let mut result = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            let value = match (self, key.as_str()) {
                (Self::Pending, "generated") => map.next_value_seed(Self::Generated)?,
                (Self::Generated, "table") => map.next_value_seed(Self::Table)?,
                (Self::Table, field)
                    if !matches!(field, "action_contract" | "table_hash" | "kinds" | "slots") =>
                {
                    // The independent typed check still reads every column from
                    // the same original before the pending record is accepted.
                    map.next_value::<IgnoredAny>()?;
                    continue;
                }
                _ => map.next_value()?,
            };
            result.insert(key, value);
        }
        Ok(Value::Object(result))
    }
}

fn parse_pending(original: &[u8], control: &Control) -> Result<Value> {
    control.check()?;
    wire::reject_reserved_number_members(original, control)?;
    let mut decoder = serde_json::Deserializer::from_slice(original);
    let root = PendingProjection::Pending
        .deserialize(&mut decoder)
        .map_err(|failure| error(failure.to_string()))?;
    decoder
        .end()
        .map_err(|failure| error(failure.to_string()))?;
    control.check()?;
    Ok(root)
}

enum CandidateOrigin<'a> {
    Wire {
        table: &'a Value,
        original: &'a [u8],
    },
    Actor(&'a Table),
}

impl CandidateOrigin<'_> {
    fn action_contract(&self) -> &Value {
        match self {
            Self::Wire { table, .. } => &table["action_contract"],
            Self::Actor(table) => &table.action_contract,
        }
    }

    fn report_table(&self) -> journal::ReportTable<'_> {
        match self {
            Self::Wire { table, .. } => journal::ReportTable::Original(table),
            Self::Actor(table) => journal::ReportTable::Typed {
                hash: &table.table_hash,
                kinds: table.kinds.len(),
                slots: table.slots.len(),
            },
        }
    }

    fn check<'a>(
        &'a self,
        frame: &'a Value,
        selected: usize,
        max_positions: usize,
        size: usize,
    ) -> Result<journal_candidates::Candidates<'a>> {
        match self {
            Self::Wire { original, .. } => journal_candidates::check(original, max_positions, size),
            Self::Actor(table) => {
                journal_candidates::check_actor_table(frame, selected, table, max_positions, size)
            }
        }
    }
}

struct PendingFields<'a> {
    run: &'a str,
    trial: &'a str,
    actor: usize,
    frame: &'a Value,
    delta: &'a Value,
    candidates: CandidateOrigin<'a>,
    search: &'a Value,
    selected: usize,
    command: &'a Value,
    scores: &'a Value,
    search_before: &'a Value,
    search_after: &'a Value,
    policy_before: &'a Value,
    policy_after: &'a Value,
    sampling_seed: u64,
    policy: &'a Value,
}

impl<'a> PendingFields<'a> {
    fn from_wire(data: &'a Value, original: &'a [u8]) -> Result<Self> {
        data.closed(
            &[
                "schema",
                "run",
                "trial",
                "game",
                "episode",
                "actor",
                "frame",
                "delta",
                "generated",
                "selected",
                "scores",
                "command",
                "search_rng_before",
                "search_rng_after",
                "policy_state_before",
                "policy_state_after",
                "sampling_seed",
                "policy",
            ],
            &[],
        )?;
        let generated = data.require("generated")?;
        generated.closed(&["table", "report"], &[])?;
        Ok(Self {
            run: data.require("run")?.string()?,
            trial: data.require("trial")?.string()?,
            actor: data.require("actor")?.integer(1, 2)? as usize,
            frame: data.require("frame")?,
            delta: data.require("delta")?,
            candidates: CandidateOrigin::Wire {
                table: generated.require("table")?,
                original,
            },
            search: generated.require("report")?,
            selected: number(data, "selected", usize::MAX)?,
            command: data.require("command")?,
            scores: data.require("scores")?,
            search_before: data.require("search_rng_before")?,
            search_after: data.require("search_rng_after")?,
            policy_before: data.require("policy_state_before")?,
            policy_after: data.require("policy_state_after")?,
            sampling_seed: data
                .require("sampling_seed")?
                .integer(0, u64::MAX as i128)? as u64,
            policy: data.require("policy")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec2_aligns_numeric_columns_and_preserves_empty_slots() {
        let (columns, bytes) = candidate_layout(3, 0);
        assert_eq!(bytes, 152);
        assert_eq!(
            columns[2],
            json!({"name":"numerics","dtype":"int64","shape":[3,4],"offset":8,"bytes":96})
        );
        assert_eq!(
            columns[4],
            json!({"name":"offsets","dtype":"int64","shape":[4],"offset":120,"bytes":32})
        );
        assert_eq!(
            columns[5],
            json!({"name":"slots","dtype":"uint8","shape":[0],"offset":152,"bytes":0})
        );
    }

    fn writer_fixture() -> (EpisodeWriter, Value) {
        let fixture: Value = serde_json::from_slice(include_bytes!(
            "../../../../tests/fixtures/rust_player_v5/originals.json"
        ))
        .unwrap();
        let episode = fixture["episodes"][0].clone();
        let original = &episode["decisions"][0];
        let frame = &original["frame"];
        let report = &original["generated"]["report"];
        let seed = episode["sampling_seed"].as_u64().unwrap();
        let deadline = f64::MAX;
        let policy = json!({"kind":"mlp","seed":seed,"epoch":0,
            "checkpoint":"a".repeat(64),"numeric_costs":episode["numeric_costs"]});
        let header = |actor| {
            json!({"kind":"header","schema_version":6,"viewer":actor,
            "purpose":"qualified_game","synthetic_capacity":false,"model_id":"mlp","model_version":"4",
            "bank_episode":frame["episode"],"episode_id":frame["episode"],"run_id":"run",
            "game_id":"registered-context","policy":"mlp-fixture","actions":episode["actions"],
            "runtime_evidence":null,"action_contract":report["action_contract"],"layout_signature":"c".repeat(64),
            "catalog":WireProfile::V2.catalog_metadata(),
            "sampling_seed":seed,"numeric_costs":episode["numeric_costs"],"generator_version":report["generator_version"],
            "action_config_hash":report["config_hash"],"rule_signature":report["rule_signature"],
            "caps":{"max_positions":18,"max_interval_events":128,"max_episode_events":10000,
                    "max_candidates":10000,"max_decisions":1000}})
        };
        let validation =
            JournalValidation::new(&Budget::new(64 << 20), &Control::new(deadline).unwrap())
                .unwrap();
        let writer = EpisodeWriter::new(
            Arc::new(validation),
            [header(1), header(2)],
            "trial".into(),
            [policy.clone(), policy.clone()],
            episode["numeric_costs"].clone(),
            Value::Null,
            1,
            deadline,
            episode["rules"].clone(),
            pick(
                &header(1),
                &[
                    "action_contract",
                    "rule_signature",
                    "layout_signature",
                    "purpose",
                    "synthetic_capacity",
                ],
            )
            .unwrap(),
        )
        .unwrap();
        (writer, episode)
    }

    fn pending_fixture(episode: &Value, policy: &Value) -> Value {
        let original = &episode["decisions"][0];
        let frame = &original["frame"];
        let seed = &episode["sampling_seed"];
        let deadline = f64::MAX;
        let command = json!({"command_id":"9d5cd67b-5999-4b0d-9b30-c7274346dd63",
            "expected_revision":frame["revision"],"action":original["ordered_actions"][0]});
        let capture = json!({"schema":"rust-player-pending-v1","run":"run","trial":"trial","game":"actual-game",
            "episode":frame["episode"],"actor":1,"frame":frame,"delta":original["delta"],
            "generated":original["generated"],"selected":0,"scores":{
                "identity":{"run":"run","trial":"trial","game":"actual-game","player":1,"epoch":0,
                    "request":0,"revision":frame["revision"],"checkpoint":policy["checkpoint"],"deadline":deadline},
                "sample":0,"logprob":-0.0,"value":f64::from(0.1f32)},
            "command":command,"search_rng_before":original["search_rng_before"],"search_rng_after":original["search_rng_after"],
            "policy_state_before":null,"policy_state_after":null,"sampling_seed":seed,"policy":policy});
        capture
    }

    fn terminal_fixture(
        writer: &EpisodeWriter,
        game: Value,
        end_kind: &str,
        reason: Value,
    ) -> Value {
        let header = &writer.seats[0].header;
        let episode = &header["episode_id"];
        json!({
            "schema":"rust-player-terminal-v1", "run":header["run_id"],"trial":writer.trial,
            "game":game,"episode":episode,"runtime_evidence":header["runtime_evidence"],
            "actions":header["actions"],"rules":writer.rules,"numeric_costs":writer.numeric_costs,
            "clock":writer.clock,"policies":writer.policies,"terminal_observations":[],"replays":[],
            "report":{"episode":episode,"game":game,"end_kind":end_kind,"outcome":null,
                "simulator_abort_reason":reason,"decisions":0,"error":"injected failure",
                "cleanup_errors":[],"storage_accepted":false,"inference_released":true},
            "banks":[{"actor":1,"episode":episode,"events":writer.seats[0].events,"cursor":null},
                     {"actor":2,"episode":episode,"events":writer.seats[1].events,"cursor":null}],
        })
    }

    fn consume_fixture(writer: &mut EpisodeWriter, kind: &str, value: &Value) -> Result<()> {
        let budget = Budget::new(64 << 20);
        let control = Control::new(f64::MAX).unwrap();
        writer.consume(kind, &serde_json::to_vec(value).unwrap(), &budget, &control)
    }

    fn journal_rows(members: &[Member], actor: u8) -> Vec<Value> {
        let filename = format!("player-{actor}.jsonl");
        let bytes = &members
            .iter()
            .find(|(name, _)| *name == filename)
            .unwrap()
            .1;
        std::str::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn ack_fixture(capture: &Value, status: &str, result: Value) -> Value {
        let revision = capture["frame"]["revision"].as_u64().unwrap();
        json!({"schema":"rust-player-ack-v1","game":"actual-game","episode":capture["episode"],"actor":1,
            "command":capture["command"],"acknowledgement":{"public_event_schema_version":"2",
                "rule_set_version":"true-match-1v1-v2","card_ref_scheme":"aes256-card-ref-v1",
                "command_id":capture["command"]["command_id"],"previous_revision":revision,
                "state_revision":revision + 1,"status":status,"events":[],"result":result,
                "next_observation":null,"action_options":null}})
    }

    fn consume_unprojected_pending(
        writer: &mut EpisodeWriter,
        original: &[u8],
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<()> {
        let document = wire::parse(original, budget, control)?;
        writer.consume_data("pending", original, &document.root, budget, control)
    }

    #[test]
    fn pending_projection_keeps_only_report_fields_in_the_candidate_table() {
        let (writer, episode) = writer_fixture();
        let capture = pending_fixture(&episode, &writer.policies[0]);
        let original = serde_json::to_vec(&capture).unwrap();
        let control = Control::new(f64::MAX).unwrap();
        let mut expected = capture;
        expected["generated"]["table"]
            .as_object_mut()
            .unwrap()
            .retain(|key, _| {
                matches!(
                    key.as_str(),
                    "action_contract" | "table_hash" | "kinds" | "slots"
                )
            });

        assert_eq!(parse_pending(&original, &control).unwrap(), expected);
    }

    #[test]
    fn pending_projection_preserves_journal_candidate_and_numeric_bytes() {
        let (mut control_writer, episode) = writer_fixture();
        let (mut projected_writer, _) = writer_fixture();
        let capture = pending_fixture(&episode, &control_writer.policies[0]);
        let original = serde_json::to_vec(&capture).unwrap();
        let budget = Budget::new(64 << 20);
        let control = Control::new(f64::MAX).unwrap();
        let ack = ack_fixture(&capture, "active", Value::Null);
        consume_unprojected_pending(&mut control_writer, &original, &budget, &control).unwrap();
        projected_writer
            .consume("pending", &original, &budget, &control)
            .unwrap();
        for writer in [&mut control_writer, &mut projected_writer] {
            consume_fixture(writer, "ack", &ack).unwrap();
        }

        assert_eq!(
            &**control_writer.seats[0].journal.as_ref().unwrap(),
            &**projected_writer.seats[0].journal.as_ref().unwrap()
        );
        assert_eq!(control_writer.members.len(), 1);
        assert_eq!(projected_writer.members.len(), 1);
        assert_eq!(
            &*control_writer.members[0].1,
            &*projected_writer.members[0].1
        );
        let before = &control_writer.learner_rows()[0];
        let after = &projected_writer.learner_rows()[0];
        assert_eq!(&*before.payload, &*after.payload);
        assert_eq!(before.logprob.to_bits(), after.logprob.to_bits());
        assert_eq!(before.value.to_bits(), after.value.to_bits());
    }

    fn owned_pending_fixture(raw: &Value) -> super::super::capture::OwnedPending {
        use super::super::capture::{OwnedPending, Scores};
        use crate::player_runtime::search::Generated;
        let scores = &raw["scores"];
        OwnedPending {
            schema: "rust-player-pending-v1",
            run: raw["run"].as_str().unwrap().to_owned(),
            trial: raw["trial"].as_str().unwrap().to_owned(),
            game: raw["game"].as_str().unwrap().to_owned(),
            episode: raw["episode"].as_str().unwrap().to_owned(),
            actor: raw["actor"].as_u64().unwrap() as u8,
            frame: raw["frame"].clone(),
            delta: raw["delta"].clone(),
            generated: Generated {
                table: serde_json::from_value(raw["generated"]["table"].clone()).unwrap(),
                report: raw["generated"]["report"].clone(),
            },
            selected: raw["selected"].as_u64().unwrap() as usize,
            scores: Some(Scores {
                identity: serde_json::from_value(scores["identity"].clone()).unwrap(),
                logprob: scores["logprob"].as_f64().unwrap(),
                sample: scores["sample"].as_u64().unwrap() as usize,
                value: scores["value"].as_f64().unwrap(),
            }),
            command: raw["command"].clone(),
            search_rng_before: raw["search_rng_before"].clone(),
            search_rng_after: raw["search_rng_after"].clone(),
            policy_state_before: raw["policy_state_before"].clone(),
            policy_state_after: raw["policy_state_after"].clone(),
            sampling_seed: raw["sampling_seed"].as_u64().unwrap(),
            policy: serde_json::from_value(raw["policy"].clone()).unwrap(),
        }
    }

    #[test]
    fn owned_pending_and_ack_match_wire_journal_codec_and_numeric_bits() {
        use super::super::capture::OwnedAck;

        for (logprob, value) in [
            (-0.0, f64::from(f32::from_bits(1))),
            (-f64::from(f32::from_bits(1)), f64::from(0.1f32)),
        ] {
            let (mut wire_writer, episode) = writer_fixture();
            let (mut typed_writer, _) = writer_fixture();
            let capture = pending_fixture(&episode, &wire_writer.policies[0]);
            let mut typed = owned_pending_fixture(&capture);
            let scores = typed.scores.as_mut().unwrap();
            scores.logprob = logprob;
            scores.value = value;
            let policy = serde_json::to_value(&typed.policy).unwrap();
            wire_writer.policies[0] = policy.clone();
            typed_writer.policies[0] = policy;
            let pending = OwnedAction::Pending(typed);
            let original = serde_json::to_vec(&pending).unwrap();
            let budget = Budget::new(64 << 20);
            let control = Control::new(f64::MAX).unwrap();

            wire_writer
                .consume("pending", &original, &budget, &control)
                .unwrap();
            typed_writer
                .consume_owned("pending", &original, &pending, &budget, &control)
                .unwrap();
            let raw_ack = ack_fixture(&capture, "active", Value::Null);
            let ack = OwnedAction::Ack(OwnedAck {
                schema: "rust-player-ack-v1",
                game: raw_ack["game"].as_str().unwrap().to_owned(),
                episode: raw_ack["episode"].as_str().unwrap().to_owned(),
                actor: raw_ack["actor"].as_u64().unwrap() as u8,
                command: raw_ack["command"].clone(),
                acknowledgement: raw_ack["acknowledgement"].clone(),
            });
            let bytes = serde_json::to_vec(&ack).unwrap();
            wire_writer
                .consume("ack", &bytes, &budget, &control)
                .unwrap();
            typed_writer
                .consume_owned("ack", &bytes, &ack, &budget, &control)
                .unwrap();

            assert_eq!(
                &**wire_writer.seats[0].journal.as_ref().unwrap(),
                &**typed_writer.seats[0].journal.as_ref().unwrap()
            );
            assert_eq!(wire_writer.members[0].0, typed_writer.members[0].0);
            assert_eq!(&*wire_writer.members[0].1, &*typed_writer.members[0].1);
            let expected = &wire_writer.learner_rows()[0];
            let recorded = &typed_writer.learner_rows()[0];
            assert_eq!(&*expected.payload, &*recorded.payload);
            assert_eq!(expected.correspondence, recorded.correspondence);
            assert_eq!(expected.logprob.to_bits(), recorded.logprob.to_bits());
            assert_eq!(expected.value.to_bits(), recorded.value.to_bits());
            assert_eq!(logprob.to_bits(), recorded.logprob.to_bits());
            assert_eq!(value.to_bits(), recorded.value.to_bits());
        }
    }

    #[test]
    fn owned_pending_validates_all_candidate_rows_before_custody() {
        let (mut writer, episode) = writer_fixture();
        let capture = pending_fixture(&episode, &writer.policies[0]);
        let mut pending = owned_pending_fixture(&capture);
        writer.policies[0] = serde_json::to_value(&pending.policy).unwrap();
        // Candidate 0 is selected. A malformed unselected row must also fail.
        pending.generated.table.numerics[1] = [1, 0, 0, 0];
        let owned = OwnedAction::Pending(pending);
        let original = serde_json::to_vec(&owned).unwrap();
        let budget = Budget::new(64 << 20);
        let control = Control::new(f64::MAX).unwrap();
        assert!(
            writer
                .consume_owned("pending", &original, &owned, &budget, &control)
                .is_err()
        );
        assert!(writer.pending.is_none() && writer.members.is_empty());
        assert!(writer.game.is_none());
    }

    #[test]
    fn pending_projection_rejects_invalid_originals_before_custody() {
        let (writer, episode) = writer_fixture();
        let capture = pending_fixture(&episode, &writer.policies[0]);
        let original = serde_json::to_string(&capture).unwrap();
        let duplicate = original.replacen(r#""numerics":"#, r#""nume\u0072ics":[],"numerics":"#, 1);
        assert_ne!(original, duplicate);
        let mut invalid_row = capture.clone();
        invalid_row["generated"]["table"]["numerics"][1] = json!([0, 0, 0]);
        let mut extra_key = capture.clone();
        extra_key["generated"]["table"]["extra"] = Value::Null;
        let mut reserved_member = capture;
        reserved_member["generated"]["table"]["numerics"][1] =
            json!({"$serde_json::private::Number":"0"});
        for original in [
            duplicate,
            format!("{original} invalid suffix"),
            serde_json::to_string(&invalid_row).unwrap(),
            serde_json::to_string(&extra_key).unwrap(),
            serde_json::to_string(&reserved_member).unwrap(),
        ] {
            let (mut writer, _) = writer_fixture();
            let (mut unprojected, _) = writer_fixture();
            let budget = Budget::new(64 << 20);
            let control = Control::new(f64::MAX).unwrap();
            let bytes = original.as_bytes();
            let failure = writer
                .consume("pending", bytes, &budget, &control)
                .unwrap_err();
            let baseline = consume_unprojected_pending(&mut unprojected, bytes, &budget, &control);
            assert_eq!(failure, baseline.unwrap_err());
            assert!(writer.pending.is_none() && writer.members.is_empty());
        }
    }

    #[test]
    fn original_pending_emits_matching_sidecar_and_waits_for_exact_ack() {
        let (mut writer, episode) = writer_fixture();
        let original = &episode["decisions"][0];
        let frame = &original["frame"];
        let deadline = f64::MAX;
        let mut capture = pending_fixture(&episode, &writer.policies[0]);
        let command = capture["command"].clone();
        let budget = Budget::new(64 << 20);
        let control = Control::new(deadline).unwrap();
        let frame_hash = capture["frame"]["frame_hash"].clone();
        capture["frame"]["frame_hash"] = json!("0".repeat(64));
        assert!(
            writer
                .consume(
                    "pending",
                    &serde_json::to_vec(&capture).unwrap(),
                    &budget,
                    &control
                )
                .is_err()
        );
        assert!(writer.pending.is_none() && writer.members.is_empty());
        capture["frame"]["frame_hash"] = frame_hash;
        writer
            .consume(
                "pending",
                &serde_json::to_vec(&capture).unwrap(),
                &budget,
                &control,
            )
            .unwrap();
        let revision = frame["revision"].as_u64().unwrap();
        let mut ack = ack_fixture(&capture, "active", Value::Null);
        ack["acknowledgement"]["state_revision"] = json!(revision + 2);
        assert!(
            writer
                .consume("ack", &serde_json::to_vec(&ack).unwrap(), &budget, &control)
                .is_err()
        );
        assert_eq!(writer.learner_rows().len(), 0);
        ack["acknowledgement"]["state_revision"] = json!(revision + 1);
        writer
            .consume("ack", &serde_json::to_vec(&ack).unwrap(), &budget, &control)
            .unwrap();
        let bytes = writer.seats[0].journal.as_ref().unwrap();
        let lines: Vec<Value> = std::str::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0]["game_id"], "actual-game");
        assert_eq!(
            lines[1],
            json!({"kind":"events","delta":original["delta"],"boundary":frame["boundary"]})
        );
        assert_eq!(lines[2]["frame"], *frame);
        assert_eq!(
            lines[2]["table"]["codebook_sha256"],
            "ef5f5f827dd142d45b15dc54f713d337f9dbc5e7e53607a028fa9f46b16e447e"
        );
        assert_eq!(
            lines[2]["table"]["file"],
            "player-1-decision-0000000000.columns"
        );
        assert_eq!(
            lines[2]["table"]["sha256"],
            format!("{:x}", Sha256::digest(&*writer.members[0].1))
        );
        assert_eq!(
            lines[3],
            json!({"decision":0,"kind":"acknowledgement","response":{
            "command_id":command["command_id"],"previous_revision":revision,"state_revision":revision + 1,
            "status":"active","result":null}})
        );
        assert_eq!(
            writer.learner_rows()[0].logprob.to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(
            &*writer.learner_rows()[0].payload,
            &*wire::canonical(&frame["payload"], &budget, &control).unwrap()
        );
        assert!(writer.into_members().is_err());
    }

    #[test]
    fn interrupted_terminal_keeps_pending_decision_and_available_replay() {
        let (mut writer, episode) = writer_fixture();
        let capture = pending_fixture(&episode, &writer.policies[0]);
        consume_fixture(&mut writer, "pending", &capture).unwrap();
        let reason = json!("turn_limit_exceeded");
        let mut terminal =
            terminal_fixture(&writer, json!("actual-game"), "truncated", reason.clone());
        terminal["replays"] = json!([{"viewer_player_id":2, "original_evidence":"retained"}]);
        let first = terminal["banks"][0]["events"][0].clone();
        terminal["banks"][0]["events"][0] =
            json!({"type":"turn_started","turn":1,"active_player":1});
        assert!(consume_fixture(&mut writer, "terminal", &terminal).is_err());
        terminal["banks"][0]["events"][0] = first;

        // Events observed after the last pending record remain in its original
        // terminal bank. They cannot create another accepted decision.
        let events = terminal["banks"][0]["events"].as_array_mut().unwrap();
        events.push(json!({"type":"turn_started","turn":"bad-turn","active_player":1}));
        let final_event = events.len() - 1;
        assert!(consume_fixture(&mut writer, "terminal", &terminal).is_err());
        terminal["banks"][0]["events"][final_event] =
            json!({"type":"turn_started","turn":2,"active_player":1});
        consume_fixture(&mut writer, "terminal", &terminal).unwrap();
        assert!(!writer.requires_training());
        let members = writer.into_members().unwrap();
        let rows = journal_rows(&members, 1);
        assert_eq!(rows.len(), 4);
        assert_eq!(
            rows.last().unwrap(),
            &json!({"kind":"end","end_kind":"truncated",
            "outcome":null,"terminal_reward":null,"final_observation":null,"pending_decisions":[0],
            "simulator_abort_reason":reason})
        );
        let other = journal_rows(&members, 2);
        assert_eq!(other.len(), 2);
        assert_eq!(other.last().unwrap()["pending_decisions"], json!([]));
        let replay = &members
            .iter()
            .find(|(n, _)| n == "replay-player-2.json")
            .unwrap()
            .1;
        assert_eq!(
            serde_json::from_slice::<Value>(replay).unwrap(),
            terminal["replays"][0]
        );
        let original_terminal = &members
            .iter()
            .find(|(name, _)| name == "native-terminal.json")
            .unwrap()
            .1;
        assert_eq!(
            serde_json::from_slice::<Value>(original_terminal).unwrap()["banks"],
            terminal["banks"],
        );
    }

    #[test]
    fn aborted_ack_with_null_result_still_binds_terminal_disposition() {
        let (mut writer, episode) = writer_fixture();
        let capture = pending_fixture(&episode, &writer.policies[0]);
        consume_fixture(&mut writer, "pending", &capture).unwrap();
        consume_fixture(
            &mut writer,
            "ack",
            &ack_fixture(&capture, "aborted", Value::Null),
        )
        .unwrap();

        let mut terminal =
            terminal_fixture(&writer, json!("actual-game"), "truncated", Value::Null);
        assert!(consume_fixture(&mut writer, "terminal", &terminal).is_err());
        terminal["report"]["simulator_abort_reason"] = json!("turn_limit_exceeded");
        consume_fixture(&mut writer, "terminal", &terminal).unwrap();
        assert!(!writer.requires_training());
        let members = writer.into_members().unwrap();
        let rows = journal_rows(&members, 1);
        assert_eq!(rows[3]["kind"], "acknowledgement");
        assert_eq!(rows[4]["pending_decisions"], json!([]));
        assert_eq!(rows[4]["terminal_reward"], Value::Null);
    }

    #[test]
    fn precreation_failure_keeps_original_terminal_without_seat_journals() {
        let (mut writer, _) = writer_fixture();
        let mut terminal = terminal_fixture(&writer, Value::Null, "incomplete", Value::Null);
        terminal["banks"][0]["events"] =
            json!([{"type":"turn_started","turn":1,"active_player":1}]);
        assert!(consume_fixture(&mut writer, "terminal", &terminal).is_err());
        terminal["banks"][0]["events"] = json!([]);
        consume_fixture(&mut writer, "terminal", &terminal).unwrap();
        assert_eq!(writer.game_id(), None);
        assert!(!writer.requires_training());
        let members = writer.into_members().unwrap();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].0, "native-terminal.json");
        assert_eq!(
            serde_json::from_slice::<Value>(&members[0].1).unwrap(),
            pick(&terminal, &["report", "banks", "terminal_observations"]).unwrap()
        );
    }

    #[test]
    fn a_created_game_without_decisions_keeps_both_seat_journals() {
        let (mut writer, _) = writer_fixture();
        let terminal = terminal_fixture(&writer, json!("actual-game"), "incomplete", Value::Null);
        consume_fixture(&mut writer, "terminal", &terminal).unwrap();
        assert_eq!(writer.game_id(), Some("actual-game"));
        let members = writer.into_members().unwrap();
        assert_eq!(members.len(), 3);
        for actor in [1, 2] {
            let rows = journal_rows(&members, actor);
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0]["game_id"], "actual-game");
            assert_eq!(rows[1]["end_kind"], "incomplete");
        }
    }

    #[test]
    fn natural_terminal_with_cleanup_error_keeps_real_rewards_without_learner_evidence() {
        let (mut writer, episode) = writer_fixture();
        let capture = pending_fixture(&episode, &writer.policies[0]);
        consume_fixture(&mut writer, "pending", &capture).unwrap();
        let outcome = json!({"win":{"winner":1}});
        consume_fixture(
            &mut writer,
            "ack",
            &ack_fixture(&capture, "terminal", outcome.clone()),
        )
        .unwrap();

        let mut terminal = terminal_fixture(&writer, json!("actual-game"), "natural", Value::Null);
        terminal["report"]["outcome"] = outcome.clone();
        terminal["report"]["error"] = Value::Null;
        terminal["report"]["cleanup_errors"] = json!(["unable to fetch seat2 Replay"]);
        let observations: Vec<_> = [1, 2]
            .into_iter()
            .map(|actor| {
                let mut obs = capture["frame"]["payload"]["observation"].clone();
                obs["viewer_player_id"] = json!(actor);
                obs["own"]["player_id"] = json!(actor);
                obs["opponent"]["player_id"] = json!(3 - actor);
                obs["phase"] = json!("terminal");
                obs["required_choice"] = Value::Null;
                obs["current_player"] = Value::Null;
                obs["actionable"] = json!(false);
                obs["typed_action_schema"] = json!([]);
                obs["result"] = outcome.clone();
                obs
            })
            .collect();
        terminal["terminal_observations"] = json!(observations);
        consume_fixture(&mut writer, "terminal", &terminal).unwrap();
        assert!(!writer.requires_training());
        let members = writer.into_members().unwrap();
        let first = journal_rows(&members, 1);
        let second = journal_rows(&members, 2);
        assert_eq!(first[4]["kind"], "transition");
        assert_eq!(first[4]["terminal"], true);
        assert_eq!(first[5]["terminal_reward"], 1);
        assert_eq!(first[5]["final_observation"], observations[0]);
        assert_eq!(second.len(), 2);
        assert_eq!(second[1]["terminal_reward"], -1);
        assert_eq!(second[1]["final_observation"], observations[1]);
    }
}

pub(super) struct RowProof {
    pub revision: u64,
    pub frame_hash: String,
    pub table_hash: String,
    pub candidates: usize,
    pub selected: i64,
    pub logprob: f64,
    pub value: f64,
    pub correspondence: Value,
    pub payload: BudgetVec<u8>,
}

pub(super) struct LearnerEvidence {
    pub rows: Vec<RowProof>,
    pub events: Vec<BudgetVec<u8>>,
    pub boundary: Value,
    pub outcome: Value,
    pub facts: Value,
    _admission: Arc<JsonAdmission>,
}
impl LearnerEvidence {
    pub(super) fn retained_bytes(&self) -> usize {
        self.events
            .iter()
            .map(BudgetVec::capacity_bytes)
            .sum::<usize>()
            + self
                .rows
                .iter()
                .map(|row| row.payload.capacity_bytes())
                .sum::<usize>()
    }
}

enum LifecycleAck {
    Natural(Value),
    Aborted,
}

struct Seat {
    header: Value,
    journal: Option<BudgetVec<u8>>,
    events: Vec<Value>,
    prefix: [u8; 32],
    search_state: Value,
    policy_state: Option<Value>,
    accepted: Vec<Value>,
    rows: Vec<RowProof>,
    last_revision: Option<u64>,
    last_accepted: Option<usize>,
    decisions: usize,
    lifecycle_ack: Option<LifecycleAck>,
    owned: Vec<Arc<JsonAdmission>>,
}
impl Seat {
    fn new(header: Value) -> Result<Self> {
        let seed = seed(&header)?;
        Ok(Self {
            header,
            journal: None,
            events: Vec::new(),
            prefix: [0; 32],
            search_state: PythonRng::seed(search_seed(seed)?).state(),
            policy_state: None,
            accepted: Vec::new(),
            rows: Vec::new(),
            last_revision: None,
            last_accepted: None,
            decisions: 0,
            lifecycle_ack: None,
            owned: Vec::new(),
        })
    }

    fn reserve_line_bytes(&mut self, size: usize) -> Result<()> {
        self.journal.as_mut().expect("opened journal").reserve(size)
    }
    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        self.journal
            .as_mut()
            .expect("opened journal")
            .extend_from_slice(bytes)
    }
}

struct Pending {
    viewer: usize,
    command: Value,
    proof: Option<RowProof>,
}

pub(super) struct EpisodeWriter {
    validation: Arc<JournalValidation>,
    profile: WireProfile,
    seats: [Seat; 2],
    trial: String,
    policies: [Value; 2],
    numeric_costs: Value,
    clock: Value,
    learner: usize,
    deadline: f64,
    rules: Value,
    classification: Value,
    game: Option<String>,
    pending: Option<Pending>,
    revision: Option<u64>,
    terminal: Option<Value>,
    members: Vec<Member>,
}

fn seed(header: &Value) -> Result<u64> {
    Ok(header
        .require("sampling_seed")?
        .integer(0, u64::MAX as i128)? as u64)
}

fn pinned_frame_contract() -> &'static Value {
    &card_instance::embedded_pins()["action_contract"]
}

fn number(object: &Value, key: &str, max: usize) -> Result<usize> {
    Ok(object.require(key)?.integer(0, max as i128)? as usize)
}

fn candidate_layout(count: usize, elements: usize) -> (Vec<Value>, usize) {
    let mut offset = 0usize;
    let mut columns = Vec::with_capacity(6);
    for (name, dtype, shape, bytes) in [
        ("kinds", "uint8", vec![count], count),
        ("targets", "uint8", vec![count], count),
        ("numerics", "int64", vec![count, 4], 32 * count),
        ("presence", "uint8", vec![count, 4], 4 * count),
        ("offsets", "int64", vec![count + 1], 8 * (count + 1)),
        ("slots", "uint8", vec![elements], elements),
    ] {
        if dtype == "int64" {
            offset = offset.next_multiple_of(8);
        }
        columns
            .push(json!({"name":name,"dtype":dtype,"shape":shape,"offset":offset,"bytes":bytes}));
        offset += bytes;
    }
    (columns, offset)
}

fn emit(
    rows: impl IntoIterator<Item = Value>,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<BudgetVec<u8>> {
    let mut output = BudgetVec::new(budget);
    for row in rows {
        let bytes = wire::canonical(&row, budget, control)?;
        output.extend_from_slice(&bytes)?;
        output.push(b'\n')?;
    }
    Ok(output)
}

fn encode_object(value: &Value, budget: &Arc<Budget>, control: &Control) -> Result<BudgetVec<u8>> {
    wire::canonical(value, budget, control)
}

fn pick(source: &Value, names: &[&str]) -> Result<Value> {
    Ok(Value::Object(
        names
            .iter()
            .map(|key| Ok(((*key).into(), source.require(key)?.clone())))
            .collect::<Result<_>>()?,
    ))
}

pub(super) struct JournalValidation {
    recipe: crate::structured_tokens::Recipe,
    codebook_sha256: String,
}

impl JournalValidation {
    pub(super) fn new(budget: &Arc<Budget>, control: &Control) -> Result<Self> {
        let recipe = crate::structured_tokens::Recipe::corrected(16, budget, control)?;
        let codebook_sha256 = digest(
            &json!({"codec":CODEC,"contract":pinned_frame_contract(),
                "action_kinds":card_instance::KINDS,"targets":journal::TARGETS}),
            budget,
            control,
        )?;
        Ok(Self {
            recipe,
            codebook_sha256,
        })
    }
}

fn terminal_requires_training(report: &Value) -> bool {
    report["end_kind"] == "natural"
        && report["error"].is_null()
        && report["cleanup_errors"]
            .as_array()
            .is_some_and(Vec::is_empty)
        && report["inference_released"] == true
}

impl EpisodeWriter {
    pub(super) fn new(
        validation: Arc<JournalValidation>,
        headers: [Value; 2],
        trial: String,
        policies: [Value; 2],
        numeric_costs: Value,
        clock: Value,
        learner_viewer: u8,
        deadline: f64,
        rules: Value,
        classification: Value,
    ) -> Result<Self> {
        const CLASSIFICATION: &[&str] = &[
            "action_contract",
            "rule_signature",
            "layout_signature",
            "purpose",
            "synthetic_capacity",
        ];
        classification.closed(CLASSIFICATION, &[])?;
        let profile = super::super::wire_profile_for_runtime(classification.require("action_contract")?)?;
        ensure(
            matches!(learner_viewer, 1 | 2) && !trial.is_empty(),
            "invalid rollout writer context",
        )?;
        ensure(
            deadline.is_finite() && deadline > 0.0,
            "invalid rollout writer deadline",
        )?;
        let learner = usize::from(learner_viewer - 1);
        ensure(
            policies[learner]["kind"] == "mlp",
            "rollout writer learner must be MLP",
        )?;
        for (i, (header, policy)) in headers.iter().zip(&policies).enumerate() {
            let kind = policy.require("kind")?.string()?;
            let model = if kind == "refresh" {
                "heuristic_refresh"
            } else {
                kind
            };
            ensure(
                matches!(kind, "mlp" | "heuristic" | "random" | "refresh")
                    && header["kind"] == "header"
                    && header["schema_version"] == 6
                    && header["viewer"] == i + 1
                    && header["purpose"] == "qualified_game"
                    && header["synthetic_capacity"] == false
                    && header["model_id"] == model
                    && header["model_version"] == if kind == "mlp" { "4" } else { "1" }
                    && header["bank_episode"] == header["episode_id"]
                    && header["episode_id"] == headers[0]["episode_id"]
                    && header["run_id"] == headers[0]["run_id"]
                    && header["actions"] == headers[0]["actions"]
                    && header["actions"]["action_contract"] == header["action_contract"]
                    && header["catalog"] == profile.catalog_metadata()
                    && header["runtime_evidence"] == headers[0]["runtime_evidence"]
                    && super::super::is_shared_card_instance_contract(&header["action_contract"])
                    && CLASSIFICATION
                        .iter()
                        .all(|k| header[*k] == classification[*k]),
                "rollout writer header or policy binding differs",
            )?;
            ensure(
                header["numeric_costs"] == *policy.get("numeric_costs").unwrap_or(&numeric_costs)
                    && policy["seed"] == seed(header)?,
                "rollout writer costs or policy seed differ",
            )?;
            ensure(
                !header.require("episode_id")?.string()?.is_empty()
                    && !header.require("run_id")?.string()?.is_empty(),
                "empty rollout identity",
            )?;
        }
        let [first, second] = headers;
        Ok(Self {
            validation,
            profile,
            seats: [Seat::new(first)?, Seat::new(second)?],
            trial,
            policies,
            numeric_costs,
            clock,
            learner,
            deadline,
            rules,
            classification,
            game: None,
            pending: None,
            revision: None,
            terminal: None,
            members: Vec::new(),
        })
    }

    pub(super) fn consume(
        &mut self,
        kind: &str,
        original: &[u8],
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<()> {
        control.check()?;
        ensure(self.terminal.is_none(), "rollout record after terminal")?;
        if kind == "pending" {
            let _admission = wire::admit(original.len(), budget)?;
            let data = parse_pending(original, control)?;
            self.consume_data(kind, original, &data, budget, control)
        } else {
            let document = wire::parse(original, budget, control)?;
            self.consume_data(kind, original, &document.root, budget, control)
        }
    }

    pub(super) fn consume_owned(
        &mut self,
        kind: &str,
        original: &[u8],
        capture: &OwnedAction,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<()> {
        control.check()?;
        ensure(self.terminal.is_none(), "rollout record after terminal")?;
        // The mailbox serializes this exact immutable capture before enqueue.
        // Preserve original-byte admission and the reserved-member scan while
        // borrowing the actor's values instead of decoding another copy.
        let _admission = wire::admit(original.len(), budget)?;
        wire::reject_reserved_number_members(original, control)?;
        match capture {
            OwnedAction::Pending(pending) => {
                ensure(kind == "pending", "rollout typed pending kind differs")?;
                self.check_original_binding(
                    kind,
                    pending.schema,
                    &pending.episode,
                    Some(&pending.game),
                )?;
                let policy = serde_json::to_value(&pending.policy)
                    .map_err(|failure| error(failure.to_string()))?;
                let scores = serde_json::to_value(&pending.scores)
                    .map_err(|failure| error(failure.to_string()))?;
                self.consume_pending(
                    PendingFields {
                        run: &pending.run,
                        trial: &pending.trial,
                        actor: usize::from(pending.actor),
                        frame: &pending.frame,
                        delta: &pending.delta,
                        candidates: CandidateOrigin::Actor(&pending.generated.table),
                        search: &pending.generated.report,
                        selected: pending.selected,
                        command: &pending.command,
                        scores: &scores,
                        search_before: &pending.search_rng_before,
                        search_after: &pending.search_rng_after,
                        policy_before: &pending.policy_state_before,
                        policy_after: &pending.policy_state_after,
                        sampling_seed: pending.sampling_seed,
                        policy: &policy,
                    },
                    &pending.game,
                    budget,
                    control,
                )
            }
            OwnedAction::Ack(ack) => {
                ensure(kind == "ack", "rollout typed ACK kind differs")?;
                self.check_original_binding(kind, ack.schema, &ack.episode, Some(&ack.game))?;
                self.apply_ack(
                    usize::from(ack.actor),
                    &ack.command,
                    &ack.acknowledgement,
                    budget,
                    control,
                )
            }
        }
    }

    fn check_original_binding(
        &self,
        kind: &str,
        schema: &str,
        episode: &str,
        game: Option<&str>,
    ) -> Result<()> {
        let expected_schema = match kind {
            "pending" => "rust-player-pending-v1",
            "ack" => "rust-player-ack-v1",
            "terminal" => "rust-player-terminal-v1",
            _ => return Err(error("unknown rollout storage record")),
        };
        ensure(
            schema == expected_schema && self.seats[0].header["episode_id"] == episode,
            "rollout record kind/episode differs",
        )?;
        ensure(
            game.is_none_or(|game| !game.is_empty()),
            "empty rollout game identity",
        )?;
        ensure(
            self.game.as_deref().is_none_or(|id| Some(id) == game),
            "rollout original game differs",
        )
    }

    fn consume_data(
        &mut self,
        kind: &str,
        original: &[u8],
        data: &Value,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<()> {
        let raw_game = data.require("game")?;
        let game = if kind == "terminal" && raw_game.is_null() {
            None
        } else {
            Some(raw_game.string()?)
        };
        self.check_original_binding(
            kind,
            data.require("schema")?.string()?,
            data.require("episode")?.string()?,
            game,
        )?;
        match kind {
            "pending" => self.consume_pending(
                PendingFields::from_wire(data, original)?,
                game.expect("pending game checked"),
                budget,
                control,
            ),
            "ack" => self.consume_ack(data, budget, control),
            "terminal" => self.consume_terminal(data, game, budget, control),
            _ => unreachable!("record kind checked"),
        }
    }

    fn open(&mut self, game: &str, budget: &Arc<Budget>, control: &Control) -> Result<()> {
        if self.game.is_some() {
            return Ok(());
        }
        for (index, seat) in self.seats.iter_mut().enumerate() {
            seat.header["game_id"] = json!(game);
            seat.prefix = event_prefix_initial(
                (index + 1) as i128,
                seat.header.require("bank_episode")?,
                Some(self.profile.frame_contract()),
                budget,
                control,
            )?;
            seat.journal = Some(emit([seat.header.clone()], budget, control)?);
        }
        self.game = Some(game.into());
        Ok(())
    }

    fn consume_pending(
        &mut self,
        data: PendingFields<'_>,
        game: &str,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<()> {
        let actor = data.actor;
        ensure(matches!(actor, 1 | 2), "invalid rollout pending actor")?;
        let viewer = actor - 1;
        let seat = &self.seats[viewer];
        let policy = &self.policies[viewer];
        ensure(
            self.pending.is_none()
                && !self.seats.iter().any(|s| s.lifecycle_ack.is_some())
                && seat.header["run_id"] == data.run
                && data.trial == self.trial
                && data.policy == policy
                && seat.header["sampling_seed"] == data.sampling_seed,
            "rollout pending chronology or context differs",
        )?;
        let frame = data.frame;
        card_instance::frame_for_rollout(frame, budget, control, self.profile)?;
        let revision = frame.require("revision")?.integer(0, u64::MAX as i128)? as u64;
        ensure(
            self.revision.is_none_or(|previous| previous == revision)
                && seat
                    .last_revision
                    .is_none_or(|previous| previous < revision)
                && frame["actor"] == actor
                && frame["episode"] == seat.header["bank_episode"],
            "rollout frame/accepted revision mismatch",
        )?;
        let delta = data.delta;
        delta.closed(
            &["actor", "episode", "start", "previous_hash", "events"],
            &[],
        )?;
        let events = delta.require("events")?.array()?;
        let caps = seat.header.require("caps")?;
        let interval_cap = number(caps, "max_interval_events", 65536)?;
        let episode_cap = number(caps, "max_episode_events", i32::MAX as usize)?;
        let original_prefix = if self.game.is_none() {
            event_prefix_initial(
                actor as i128,
                seat.header.require("bank_episode")?,
                Some(self.profile.frame_contract()),
                budget,
                control,
            )?
        } else {
            seat.prefix
        };
        ensure(
            delta["actor"] == actor
                && delta["episode"] == frame["episode"]
                && delta["start"] == seat.events.len()
                && frame["interval_start"] == delta["start"]
                && delta["previous_hash"] == event_prefix_text(&original_prefix)
                && delta["previous_hash"] == frame["interval_start_hash"]
                && events.len() <= interval_cap
                && seat.events.len() + events.len() <= episode_cap
                && frame["boundary"]["end"] == seat.events.len() + events.len(),
            "rollout event delta original prefix differs",
        )?;
        let mut prefix = original_prefix;
        for event in events {
            card_instance::event_for_profile(event, &self.validation.recipe, budget, control, self.profile, Some(actor as i128))?;
            prefix = event_prefix_next(&prefix, event, budget, control)?;
        }
        ensure(
            frame["boundary"]["prefix_hash"] == event_prefix_text(&prefix),
            "rollout frame event boundary differs",
        )?;
        let search = data.search;
        ensure(
            data.candidates.action_contract() == &seat.header["action_contract"]
                && &search["rng_before"] == data.search_before
                && &search["rng_after"] == data.search_after,
            "rollout candidate contract or original RNG differs",
        )?;
        journal::report_context_for_table(
            search,
            &seat.header,
            data.candidates.report_table(),
            &seat.search_state,
            budget,
            control,
            true,
        )?;
        let candidates = search
            .require("selected")?
            .integer(1, number(caps, "max_candidates", 65536)? as i128)?
            as usize;
        let elements = number(search, "selected_elements", 18 * candidates)?;
        let selected = data.selected;
        ensure(selected < candidates, "selected candidate outside table")?;
        let (columns, size) = candidate_layout(candidates, elements);
        let max_positions = caps.require("max_positions")?.integer(1, 18)? as usize;
        let count = seat.decisions;
        ensure(
            count <= u32::MAX as usize
                && (caps["max_decisions"].is_null()
                    || count < number(caps, "max_decisions", usize::MAX)?),
            "rollout decision capacity exceeded",
        )?;
        let temporary = budget.reserve(size)?;
        let sidecar = data
            .candidates
            .check(frame, selected, max_positions, size)?;
        ensure(
            sidecar.frame_hash == frame["frame_hash"]
                && sidecar.table_hash == search["table_hash"]
                && sidecar.selected == selected
                && sidecar.count == candidates
                && sidecar.elements == elements,
            "rollout frame/table/report binding differs",
        )?;
        let command = data.command;
        let action = wire::parse(&sidecar.action, budget, control)?;
        journal::command(command, frame, &action.root, budget, control)?;
        let learned = policy["kind"] == "mlp";
        let raw_scores = data.scores;
        let (logprob, value, policy_state) = if learned {
            raw_scores.closed(&["identity", "logprob", "sample", "value"], &[])?;
            let expected = json!({"run":seat.header["run_id"],"trial":self.trial,"game":game,
                "player":actor,"epoch":policy["epoch"],"request":count,"revision":revision,
                "checkpoint":policy["checkpoint"],"deadline":self.deadline});
            ensure(
                raw_scores["identity"] == expected
                    && raw_scores["sample"] == selected
                    && data.policy_before.is_null()
                    && data.policy_after.is_null(),
                "rollout learned policy score identity differs",
            )?;
            let score = |key| {
                raw_scores
                    .require(key)?
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| error("rollout score must be finite"))
            };
            (Some(score("logprob")?), Some(score("value")?), None)
        } else {
            ensure(
                raw_scores.is_null() && viewer != self.learner,
                "rollout baseline provided learned scores",
            )?;
            let before = data.policy_before;
            let after = data.policy_after;
            before.closed(&["rng", "refresh"], &[])?;
            after.closed(&["rng", "refresh"], &[])?;
            let mut rng = PythonRng::from_state(before.require("rng")?)?;
            ensure(
                match &seat.policy_state {
                    Some(previous) => previous == before,
                    None => before["rng"] == PythonRng::seed(seed(&seat.header)?).state(),
                },
                "rollout baseline RNG continuity differs",
            )?;
            if policy["kind"] == "random" {
                ensure(
                    rng.range(candidates as u64)? == selected as u64,
                    "rollout Random selection differs",
                )?;
            }
            ensure(
                rng.state() == after["rng"],
                "rollout baseline RNG transition differs",
            )?;
            (None, None, Some(after.clone()))
        };
        let name = format!("player-{actor}-decision-{count:010}.columns");
        let descriptor = json!({"schema_version":2,"codec":CODEC,"file":name,"count":candidates,
            "elements":elements,"bytes":size,"sha256":sidecar.sha256,"frame_hash":sidecar.frame_hash,
            "table_hash":sidecar.table_hash,"columns":columns,"action_contract":seat.header["action_contract"],
            "purpose":"qualified_game","synthetic_capacity":false,"codebook_sha256":self.validation.codebook_sha256});
        let selection = json!({"table_hash":sidecar.table_hash,"index":selected,"policy_version":seat.header["policy"],
            "logprob":raw_scores["logprob"],"value":raw_scores["value"]});
        let proof = if viewer == self.learner {
            let payload = wire::canonical(frame.require("payload")?, budget, control)?;
            Some(RowProof {
                revision,
                frame_hash: sidecar.frame_hash.into(),
                table_hash: sidecar.table_hash.into(),
                candidates,
                selected: selected as i64,
                logprob: logprob.ok_or_else(|| error("learner logprob absent"))?,
                value: value.ok_or_else(|| error("learner value absent"))?,
                correspondence: json!({"frame_hash":frame["frame_hash"],"payload_sha256":format!("{:x}",Sha256::digest(&*payload)),
                    "revision":revision,"interval_start":frame["interval_start"],"interval_start_hash":frame["interval_start_hash"],
                    "boundary":frame["boundary"],"table_hash":sidecar.table_hash,"selection":selection}),
                payload,
            })
        } else {
            None
        };
        let mut rows = Vec::with_capacity(3);
        if let Some(last) = seat.last_accepted {
            rows.push(json!({"kind":"transition","decision":last,"next_decision":count,"reward":0,"terminal":false}));
        }
        rows.push(json!({"kind":"events","delta":delta,"boundary":frame["boundary"]}));
        rows.push(
            json!({"kind":"decision","decision":count,"frame":frame,"table":descriptor,
            "search_report":search,"selection":selection,"command":command,"timings":{}}),
        );
        let lines = emit(rows, budget, control)?;
        let mut member = BudgetVec::with_capacity(budget, size)?;
        member.extend_from_slice(&sidecar.payload)?;
        drop(sidecar);
        drop(temporary);
        // Owned parsed event/policy state and row proof outlive the transport parse.
        let retained = wire::canonical(
            &json!({"events":events,"search":search["rng_after"],"policy":policy_state,
            "row":proof.as_ref().map(|p| &p.correspondence),"command":command}),
            budget,
            control,
        )?;
        let owned = wire::admit(retained.len(), budget)?;
        self.open(game, budget, control)?;
        let seat = &mut self.seats[viewer];
        seat.reserve_line_bytes(lines.len())?;
        control.check()?;
        seat.append(&lines)?;
        self.members.push((name, member));
        seat.owned.push(owned);
        seat.events.extend(events.iter().cloned());
        seat.prefix = prefix;
        seat.search_state = search["rng_after"].clone();
        seat.policy_state = policy_state;
        seat.last_accepted = None;
        seat.last_revision = Some(revision);
        seat.decisions += 1;
        self.pending = Some(Pending {
            viewer,
            command: command.clone(),
            proof,
        });
        Ok(())
    }

    fn consume_ack(&mut self, data: &Value, budget: &Arc<Budget>, control: &Control) -> Result<()> {
        data.closed(
            &[
                "schema",
                "game",
                "episode",
                "actor",
                "command",
                "acknowledgement",
            ],
            &[],
        )?;
        self.apply_ack(
            data.require("actor")?.integer(1, 2)? as usize,
            data.require("command")?,
            data.require("acknowledgement")?,
            budget,
            control,
        )
    }

    fn apply_ack(
        &mut self,
        actor: usize,
        command: &Value,
        actual: &Value,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<()> {
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| error("rollout ACK lacks pending original"))?;
        let viewer = pending.viewer;
        ensure(
            actor == viewer + 1 && command == &pending.command,
            "rollout ACK command/actor differs",
        )?;
        Http::validate_act_ack_for_profile(
            actual,
            (viewer + 1) as u8,
            &pending.command,
            Some(&self.validation.recipe),
            self.profile,
        )?;
        let ack = pick(
            actual,
            &[
                "command_id",
                "previous_revision",
                "state_revision",
                "status",
                "result",
            ],
        )?;
        journal::acknowledgement(&ack, &pending.command)?;
        let seat = &mut self.seats[viewer];
        let index = seat.decisions - 1;
        let append = emit(
            [json!({"kind":"acknowledgement","decision":index,"response":ack})],
            budget,
            control,
        )?;
        seat.reserve_line_bytes(append.len())?;
        control.check()?;
        seat.append(&append)?;
        let pending = self.pending.take().expect("checked pending");
        self.revision = actual["state_revision"].as_u64();
        seat.last_accepted = Some(index);
        match ack["status"].as_str().expect("validated status") {
            "terminal" => seat.lifecycle_ack = Some(LifecycleAck::Natural(ack["result"].clone())),
            "aborted" => seat.lifecycle_ack = Some(LifecycleAck::Aborted),
            _ => {}
        }
        if ack["status"] == "active" || ack["status"] == "terminal" {
            seat.accepted.push(pending.command["action"].clone());
            if let Some(proof) = pending.proof {
                seat.rows.push(proof);
            }
        }
        Ok(())
    }

    fn consume_terminal(
        &mut self,
        data: &Value,
        game: Option<&str>,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<()> {
        data.closed(
            &[
                "schema",
                "report",
                "run",
                "trial",
                "episode",
                "game",
                "runtime_evidence",
                "actions",
                "rules",
                "numeric_costs",
                "clock",
                "policies",
                "terminal_observations",
                "replays",
                "banks",
            ],
            &[],
        )?;
        let header = &self.seats[0].header;
        ensure(
            data["run"] == header["run_id"]
                && data["trial"] == self.trial
                && data["runtime_evidence"] == header["runtime_evidence"]
                && data["actions"] == header["actions"]
                && data["rules"] == self.rules
                && data["numeric_costs"] == self.numeric_costs
                && data["clock"] == self.clock
                && data["policies"].array()? == self.policies,
            "rollout terminal controls differ",
        )?;
        let report = data.require("report")?;
        report.closed(
            &[
                "episode",
                "game",
                "end_kind",
                "outcome",
                "simulator_abort_reason",
                "decisions",
                "error",
                "cleanup_errors",
                "storage_accepted",
                "inference_released",
            ],
            &[],
        )?;
        ensure(
            report["episode"] == data["episode"]
                && report["game"] == data["game"]
                && report["storage_accepted"] == false,
            "rollout terminal report identity differs",
        )?;
        if !report["error"].is_null() {
            report["error"].string()?;
        }
        for error in report["cleanup_errors"].array()? {
            error.string()?;
        }
        report["inference_released"].boolean()?;
        let end_kind = report.require("end_kind")?.string()?;
        ensure(
            matches!(
                end_kind,
                "natural" | "truncated" | "aborted" | "cancelled" | "incomplete"
            ),
            "unknown rollout terminal disposition",
        )?;
        let reason = report.require("simulator_abort_reason")?;
        let abort_kind = if reason.is_null() {
            "aborted"
        } else {
            crate::validation::text_value(reason)?;
            let kind = if reason == "turn_limit_exceeded" {
                "truncated"
            } else {
                "aborted"
            };
            ensure(end_kind == kind, "rollout abort reason/disposition differs")?;
            kind
        };
        let outcome = report.require("outcome")?;
        let natural = end_kind == "natural";
        let healthy = terminal_requires_training(report);
        let observations = data.require("terminal_observations")?.array()?;
        let banks = data.require("banks")?.array()?;
        let replays = data.require("replays")?.array()?;
        ensure(banks.len() == 2, "both original rollout banks required")?;
        if natural {
            crate::validation::result(outcome)?;
            ensure(
                game.is_some() && !outcome.is_null() && self.pending.is_none(),
                "natural rollout requires a game, an outcome and no pending decisions",
            )?;
        }
        if healthy {
            ensure(
                observations.len() == 2
                    && replays.len() == 2
                    && report["decisions"] == self.seats.iter().map(|s| s.decisions).sum::<usize>()
                    && self.seats[self.learner].decisions == self.seats[self.learner].rows.len(),
                "healthy rollout terminal evidence differs",
            )?;
        }

        let winner = outcome.get("win").and_then(|v| v["winner"].as_u64());
        let mut terminal_lines = Vec::with_capacity(2);
        for (viewer, seat) in self.seats.iter().enumerate() {
            let actor = viewer + 1;
            let bank = &banks[viewer];
            bank.closed(&["actor", "episode", "events", "cursor"], &[])?;
            let events = bank["events"].array()?;
            ensure(
                bank["actor"] == actor
                    && bank["episode"] == data["episode"]
                    && events.starts_with(&seat.events),
                "rollout terminal bank differs from journal event prefix",
            )?;
            if game.is_none() {
                ensure(
                    events.is_empty(),
                    "rollout bank events without a created game",
                )?;
                continue;
            }
            for event in events {
                card_instance::event_for_profile(event, &self.validation.recipe, budget, control, self.profile, Some(actor as i128))?;
            }
            if let Some(ack) = &seat.lifecycle_ack {
                let (expected, ack_outcome) = match ack {
                    LifecycleAck::Natural(result) => ("natural", result),
                    LifecycleAck::Aborted => (abort_kind, &Value::Null),
                };
                ensure(
                    end_kind == expected && ack_outcome == outcome,
                    "rollout lifecycle ACK differs from terminal disposition",
                )?;
            }
            let mut rows = Vec::with_capacity(2);
            let (reward, observation) = if natural {
                let observation = observations
                    .iter()
                    .rfind(|v| v["viewer_player_id"] == actor)
                    .ok_or_else(|| error("missing own terminal observation"))?;
                card_instance::observation_for_profile(observation, Some(actor as i128), budget, control, false, self.profile)?;
                ensure(
                    observation["phase"] == "terminal" && observation["result"] == *outcome,
                    "rollout final observation/outcome differs",
                )?;
                let reward = winner.map_or(0, |winner| if winner == actor as u64 { 1 } else { -1 });
                if let Some(last) = seat.last_accepted {
                    rows.push(
                        json!({"kind":"transition","decision":last,"next_decision":null,
                        "reward":reward,"terminal":true}),
                    );
                }
                (json!(reward), observation)
            } else {
                (Value::Null, &Value::Null)
            };
            let pending: Vec<_> = self
                .pending
                .as_ref()
                .filter(|p| p.viewer == viewer)
                .map(|_| seat.decisions - 1)
                .into_iter()
                .collect();
            rows.push(json!({"kind":"end","end_kind":end_kind,
                "outcome": if natural { outcome } else { &Value::Null },
                "terminal_reward":reward,"final_observation":observation,
                "pending_decisions":pending,"simulator_abort_reason":reason}));
            terminal_lines.push(emit(rows, budget, control)?);
        }

        // In an unhealthy game a participant Replay can be unavailable or its
        // reconciliation can be the error under investigation. Preserve the
        // bytes the actor actually got. Only a healthy terminal qualifies them
        // as evidence for the original learner.
        let mut replay_members = [false; 2];
        let mut members = Vec::with_capacity(replays.len() + 1);
        for replay in replays {
            let viewer = replay
                .get("viewer_player_id")
                .or_else(|| replay.get("viewer"))
                .ok_or_else(|| error("rollout Replay viewer required"))?
                .integer(1, 2)? as usize;
            ensure(
                !replay_members[viewer - 1],
                "duplicate rollout Replay viewer",
            )?;
            replay_members[viewer - 1] = true;
            if healthy {
                let accepted: Vec<_> = self.seats[viewer - 1].accepted.iter().collect();
                let closure = card_instance::reconcile_replay_for_profile(
                    replay,
                    game.expect("natural rollout game checked"),
                    viewer as i128,
                    &accepted,
                    outcome,
                    budget,
                    control,
                    self.profile,
                )?;
                ensure(
                    closure.is_some_and(|v| v["type"] == "terminal" && v["result"] == *outcome),
                    "rollout Replay terminal closure differs",
                )?;
            }
            members.push((
                format!("replay-player-{viewer}.json"),
                encode_object(replay, budget, control)?,
            ));
        }
        members.push((
            "native-terminal.json".into(),
            encode_object(
                &pick(data, &["report", "banks", "terminal_observations"])?,
                budget,
                control,
            )?,
        ));
        if let Some(game) = game {
            self.open(game, budget, control)?;
            for (seat, end) in self.seats.iter_mut().zip(&terminal_lines) {
                seat.reserve_line_bytes(end.len())?;
            }
            control.check()?;
            for (seat, end) in self.seats.iter_mut().zip(&terminal_lines) {
                seat.append(end)?;
            }
        }
        self.members.extend(members);
        self.terminal = Some(report.clone());
        Ok(())
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.members
            .iter()
            .map(|(_, bytes)| bytes.capacity_bytes())
            .sum::<usize>()
            + self
                .seats
                .iter()
                .map(|seat| {
                    seat.journal.as_ref().map_or(0, BudgetVec::capacity_bytes)
                        + seat
                            .rows
                            .iter()
                            .map(|r| r.payload.capacity_bytes())
                            .sum::<usize>()
                })
                .sum::<usize>()
            + self
                .pending
                .as_ref()
                .and_then(|p| p.proof.as_ref())
                .map_or(0, |r| r.payload.capacity_bytes())
    }

    #[cfg(test)]
    fn learner_rows(&self) -> &[RowProof] {
        &self.seats[self.learner].rows
    }
    fn learner_boundary(&self) -> Value {
        let seat = &self.seats[self.learner];
        json!({"end":seat.events.len(),"prefix_hash":event_prefix_text(&seat.prefix)})
    }
    pub(super) fn game_id(&self) -> Option<&str> {
        self.game.as_deref()
    }

    pub(super) fn terminal(&self) -> Option<&Value> {
        self.terminal.as_ref()
    }
    pub(super) fn requires_training(&self) -> bool {
        self.terminal
            .as_ref()
            .is_some_and(terminal_requires_training)
    }

    pub(super) fn into_completed(
        mut self,
        training: &Arc<Episode>,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<(Vec<Member>, LearnerEvidence)> {
        control.check()?;
        ensure(
            self.requires_training(),
            "rollout learner evidence requires healthy natural terminal",
        )?;
        let terminal = self
            .terminal
            .as_ref()
            .ok_or_else(|| error("rollout evidence requires the original terminal"))?;
        let outcome = terminal.require("outcome")?;
        let winner = if outcome == "draw" {
            None
        } else {
            Some(outcome.require("win")?.require("winner")?.integer(1, 2)? as u8)
        };
        let learner = &self.seats[self.learner];
        let game = self
            .game
            .as_deref()
            .ok_or_else(|| error("rollout terminal lacks original game identity"))?;
        let header = &learner.header;
        let policy = &self.policies[self.learner];
        let viewer = (self.learner + 1) as u8;
        let checkpoint = policy.require("checkpoint")?.string()?;
        let epoch = policy.require("epoch")?.integer(0, u64::MAX as i128)? as u64;
        let episode = header.require("episode_id")?;
        training.require_identity(
            header.require("run_id")?.string()?,
            &self.trial,
            episode.string()?,
            game,
            viewer,
            checkpoint,
            epoch,
            learner.rows.len(),
            winner,
        )?;
        for (decision, row) in learner.rows.iter().enumerate() {
            control.check()?;
            training.require_row(
                decision,
                row.revision,
                &row.frame_hash,
                &row.table_hash,
                row.candidates,
                row.selected,
                row.logprob,
                row.value,
            )?;
        }

        let boundary = self.learner_boundary();
        let digest = episode_digest_context(
            &self.classification,
            &json!(viewer),
            header.require("bank_episode")?,
            &boundary,
            learner.rows.iter().map(|row| &row.correspondence),
            terminal.require("end_kind")?,
            outcome,
            terminal.require("simulator_abort_reason")?,
            budget,
            control,
            SourceFormat::Corrected,
        )?;
        let facts = json!({"episode_id":episode,"game_id":game,"viewer":viewer,"checkpoint":checkpoint,
            "epoch":epoch,"decisions":learner.rows.len(),"episode_digest":digest});
        let events = learner
            .events
            .iter()
            .map(|event| wire::canonical(event, budget, control))
            .collect::<Result<Vec<_>>>()?;
        let mut retained = 0usize;
        for data in learner
            .rows
            .iter()
            .map(|row| &row.correspondence)
            .chain([&facts, &boundary, outcome])
        {
            let len = wire::canonical(data, budget, control)?.len();
            retained = retained
                .checked_add(len)
                .ok_or_else(|| crate::error::Error::resource("learner evidence size overflow"))?;
        }
        let _admission = wire::admit(retained, budget)?;
        let outcome = outcome.clone();
        let rows = std::mem::take(&mut self.seats[self.learner].rows);
        let evidence = LearnerEvidence {
            rows,
            events,
            boundary,
            outcome,
            facts,
            _admission,
        };
        Ok((self.into_members()?, evidence))
    }

    pub(super) fn into_members(mut self) -> Result<Vec<Member>> {
        ensure(
            self.terminal.is_some(),
            "rollout journal original terminal required",
        )?;
        if self.game.is_some() {
            for (viewer, seat) in self.seats.into_iter().enumerate() {
                self.members.push((
                    format!("player-{}.jsonl", viewer + 1),
                    seat.journal
                        .ok_or_else(|| error("rollout journal not opened"))?,
                ));
            }
        }
        Ok(self.members)
    }
}
