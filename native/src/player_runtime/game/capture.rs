//! Borrowed, closed native capture facts; no legacy journal/Source authority.
use super::{GameReport, Participant, PolicySpec};
use crate::{
    json::Value,
    player_runtime::{
        requests::Identity,
        search::{Generated, Table},
    },
};
use serde::Serialize;

#[derive(Serialize)]
pub struct Scores {
    pub identity: Identity,
    pub logprob: f64,
    pub sample: usize,
    pub value: f64,
}
#[derive(Serialize)]
pub struct GeneratedRef<'a> {
    pub table: &'a Table,
    pub report: &'a Value,
}
impl<'a> From<&'a Generated> for GeneratedRef<'a> {
    fn from(g: &'a Generated) -> Self {
        Self {
            table: &g.table,
            report: &g.report,
        }
    }
}
#[derive(Serialize)]
pub struct Pending<'a> {
    pub schema: &'static str,
    pub run: &'a str,
    pub trial: &'a str,
    pub game: &'a str,
    pub episode: &'a str,
    pub actor: u8,
    pub frame: &'a Value,
    pub delta: &'a Value,
    pub generated: GeneratedRef<'a>,
    pub selected: usize,
    pub scores: Option<&'a Scores>,
    pub command: &'a Value,
    pub search_rng_before: &'a Value,
    pub search_rng_after: &'a Value,
    pub policy_state_before: &'a Value,
    pub policy_state_after: &'a Value,
    pub sampling_seed: u64,
    pub policy: &'a PolicySpec,
}
#[derive(Serialize)]
pub struct Ack<'a> {
    pub schema: &'static str,
    pub game: &'a str,
    pub episode: &'a str,
    pub actor: u8,
    pub command: &'a Value,
    pub acknowledgement: &'a Value,
}
#[derive(Serialize)]
pub struct Bank<'a> {
    actor: u8,
    episode: &'a str,
    events: &'a [Value],
    cursor: &'a Option<String>,
}
impl<'a> From<&'a Participant> for Bank<'a> {
    fn from(p: &'a Participant) -> Self {
        Self {
            actor: p.history.actor,
            episode: &p.history.episode,
            events: &p.history.events,
            cursor: &p.history.cursor,
        }
    }
}
#[derive(Serialize)]
pub struct Terminal<'a> {
    pub schema: &'static str,
    pub report: &'a GameReport,
    pub run: &'a str,
    pub trial: &'a str,
    pub episode: &'a str,
    pub game: &'a Option<String>,
    pub runtime_evidence: &'a Value,
    pub actions: &'a Value,
    pub rules: &'a Value,
    pub numeric_costs: &'a Value,
    pub clock: &'a Value,
    pub policies: &'a [PolicySpec; 2],
    pub terminal_observations: &'a [Value],
    pub replays: &'a [Value],
    pub banks: [Bank<'a>; 2],
}

// The actor owns these observations and search results until mailbox acceptance.
// The mailbox can keep a weak reference to this exact input to serialization.
pub(super) enum OwnedAction {
    Pending(OwnedPending),
    Ack(OwnedAck),
}

pub(super) struct OwnedPending {
    pub schema: &'static str,
    pub run: String,
    pub trial: String,
    pub game: String,
    pub episode: String,
    pub actor: u8,
    pub frame: Value,
    pub delta: Value,
    pub generated: Generated,
    pub selected: usize,
    pub scores: Option<Scores>,
    pub command: Value,
    pub search_rng_before: Value,
    pub search_rng_after: Value,
    pub policy_state_before: Value,
    pub policy_state_after: Value,
    pub sampling_seed: u64,
    pub policy: PolicySpec,
}

pub(super) struct OwnedAck {
    pub schema: &'static str,
    pub game: String,
    pub episode: String,
    pub actor: u8,
    pub command: Value,
    pub acknowledgement: Value,
}

impl OwnedAction {
    pub(super) fn routing(&self) -> (&'static str, &str, &str) {
        match self {
            Self::Pending(pending) => ("pending", &pending.episode, &pending.game),
            Self::Ack(ack) => ("ack", &ack.episode, &ack.game),
        }
    }
}

impl Serialize for OwnedAction {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Pending(pending) => Pending {
                schema: pending.schema,
                run: &pending.run,
                trial: &pending.trial,
                game: &pending.game,
                episode: &pending.episode,
                actor: pending.actor,
                frame: &pending.frame,
                delta: &pending.delta,
                generated: (&pending.generated).into(),
                selected: pending.selected,
                scores: pending.scores.as_ref(),
                command: &pending.command,
                search_rng_before: &pending.search_rng_before,
                search_rng_after: &pending.search_rng_after,
                policy_state_before: &pending.policy_state_before,
                policy_state_after: &pending.policy_state_after,
                sampling_seed: pending.sampling_seed,
                policy: &pending.policy,
            }
            .serialize(serializer),
            Self::Ack(ack) => Ack {
                schema: ack.schema,
                game: &ack.game,
                episode: &ack.episode,
                actor: ack.actor,
                command: &ack.command,
                acknowledgement: &ack.acknowledgement,
            }
            .serialize(serializer),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn borrowed_pending_preserves_facts_and_exact_torch_item_scalars() {
        let saved: Value = serde_json::from_slice(include_bytes!(
            "../../../../tests/fixtures/rust_player_v5/originals.json"
        ))
        .unwrap();
        let e = &saved["episodes"][0];
        let d = &e["decisions"][0];
        let frame = &d["frame"];
        let table: Table = serde_json::from_value(d["generated"]["table"].clone()).unwrap();
        let policy = PolicySpec {
            kind: "mlp".into(),
            seed: 17,
            checkpoint: Some("a".repeat(64)),
            epoch: 0,
            limits: None,
            attack_threshold: json!(12),
            numeric_costs: None,
        };
        for (logprob, value) in [
            (-0.1f32, 0.1f32),
            (-0.0f32, -0.0f32),
            (-1., f32::MIN_POSITIVE),
        ] {
            let identity = Identity {
                run: "r".into(),
                trial: "t".into(),
                game: "actual".into(),
                player: 1,
                epoch: 0,
                request: 0,
                revision: frame["revision"].as_u64().unwrap(),
                checkpoint: "a".repeat(64),
                deadline: 1.,
            };
            // torch.item promotes the actual FP32 value exactly. The initial
            // serde f32-shortest transport prototype is not that double fact.
            let old_scores = json!({"identity":identity,"sample":0,"logs":[f64::from(logprob),-1.0f64],"value":f64::from(value)});
            let scores = Scores {
                identity,
                logprob: f64::from(logprob),
                sample: 0,
                value: f64::from(value),
            };
            let command = json!({"command_id":"00000000-0000-4000-8000-000000000001","expected_revision":frame["revision"],"action":{"type":"respond","ordered_card_refs":[]}});
            let state = json!({"original":true});
            let pending = Pending {
                schema: "rust-player-pending-v1",
                run: "r",
                trial: "t",
                game: "actual",
                episode: frame["episode"].as_str().unwrap(),
                actor: 1,
                frame,
                delta: &d["delta"],
                generated: GeneratedRef {
                    table: &table,
                    report: &d["generated"]["report"],
                },
                selected: 0,
                scores: Some(&scores),
                command: &command,
                search_rng_before: &d["search_rng_before"],
                search_rng_after: &d["search_rng_after"],
                policy_state_before: &state,
                policy_state_after: &state,
                sampling_seed: 17,
                policy: &policy,
            };
            let actual: Value =
                serde_json::from_slice(&serde_json::to_vec(&pending).unwrap()).unwrap();
            let expected = json!({"schema":"rust-player-pending-v1","run":"r","trial":"t","game":"actual","episode":frame["episode"],"actor":1,"frame":frame,"delta":d["delta"],"generated":d["generated"],"selected":0,"scores":{"identity":old_scores["identity"],"sample":0,"logprob":old_scores["logs"][0],"value":old_scores["value"]},"command":command,"search_rng_before":d["search_rng_before"],"search_rng_after":d["search_rng_after"],"policy_state_before":state,"policy_state_after":state,"sampling_seed":17,"policy":policy});
            assert_eq!(actual, expected);
            assert_eq!(
                actual["scores"]["logprob"].as_f64().unwrap().to_bits(),
                f64::from(logprob).to_bits()
            );
            assert_eq!(
                actual["scores"]["value"].as_f64().unwrap().to_bits(),
                f64::from(value).to_bits()
            );
            assert_eq!(
                actual["generated"]["table"]["table_hash"],
                d["generated"]["table"]["table_hash"]
            );
            assert_eq!(actual["frame"]["frame_hash"], frame["frame_hash"]);
            assert!(!actual["scores"].as_object().unwrap().contains_key("logs"));
        }
    }
    #[test]
    fn borrowed_terminal_and_ack_preserve_original_parsed_facts() {
        let policy = PolicySpec {
            kind: "heuristic".into(),
            seed: 17,
            checkpoint: None,
            epoch: 0,
            limits: None,
            attack_threshold: json!(12),
            numeric_costs: None,
        };
        let policies = [policy.clone(), policy];
        let seats = [
            Participant::new(1, "planned".into(), policies[0].clone(), false).unwrap(),
            Participant::new(2, "planned".into(), policies[1].clone(), false).unwrap(),
        ];
        let report = GameReport {
            episode: "planned".into(),
            game: Some("actual".into()),
            end_kind: "truncated".into(),
            outcome: Value::Null,
            simulator_abort_reason: None,
            decisions: 0,
            error: Some("decision_limit".into()),
            cleanup_errors: vec![],
            storage_accepted: false,
            inference_released: true,
        };
        let game = Some("actual".to_owned());
        let evidence = json!({"schema_version":2});
        let controls = json!({"value":1});
        let views = vec![json!({"viewer":1}), json!({"viewer":2})];
        let terminal = Terminal {
            schema: "rust-player-terminal-v1",
            report: &report,
            run: "r",
            trial: "t",
            episode: "planned",
            game: &game,
            runtime_evidence: &evidence,
            actions: &controls,
            rules: &controls,
            numeric_costs: &controls,
            clock: &Value::Null,
            policies: &policies,
            terminal_observations: &views,
            replays: &views,
            banks: [(&seats[0]).into(), (&seats[1]).into()],
        };
        let old = json!({"schema":"rust-player-terminal-v1","report":report,"run":"r","trial":"t","episode":"planned","game":game,"runtime_evidence":evidence,"actions":controls,"rules":controls,"numeric_costs":controls,"clock":null,"policies":policies,"terminal_observations":views,"replays":views,"banks":seats.iter().map(|p|json!({"actor":p.history.actor,"episode":p.history.episode,"events":p.history.events,"cursor":p.history.cursor})).collect::<Vec<_>>()});
        let actual: Value =
            serde_json::from_slice(&serde_json::to_vec(&terminal).unwrap()).unwrap();
        assert_eq!(actual, old);
        let ack = Ack {
            schema: "rust-player-ack-v1",
            game: "actual",
            episode: "planned",
            actor: 1,
            command: &controls,
            acknowledgement: &controls,
        };
        let actual: Value = serde_json::from_slice(&serde_json::to_vec(&ack).unwrap()).unwrap();
        assert_eq!(
            actual,
            json!({"schema":"rust-player-ack-v1","game":"actual","episode":"planned","actor":1,"command":controls,"acknowledgement":controls})
        );
    }
}
