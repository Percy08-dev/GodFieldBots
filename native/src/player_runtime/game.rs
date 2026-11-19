//! One external game transition path, used unchanged by both schedulers.
mod capture;
pub mod storage;
use super::{
    Config, Input,
    baselines::{self, RefreshState},
    batch::Dispatcher,
    ensure, error,
    history::History,
    http::{Http, Server},
    numeric::ServingRow,
    requests::Identity,
    rng::{PythonRng, search_seed},
    search::{self, Costs, Table},
};
use crate::{
    budget::{Budget, Control},
    error::{Error, Result},
    json::{Value, ValueExt},
    structured_journal::card_instance,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use storage::Mailbox;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub max_decisions: usize,
    pub max_episode_events: usize,
    pub max_interval_events: usize,
    pub max_candidates: usize,
    pub max_positions: usize,
    pub aggregate_window: usize,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicySpec {
    pub kind: String,
    pub seed: u64,
    #[serde(default)]
    pub checkpoint: Option<String>,
    #[serde(default)]
    pub epoch: u64,
    #[serde(default)]
    pub limits: Option<Limits>,
    #[serde(default = "threshold")]
    pub attack_threshold: Value,
}
fn threshold() -> Value {
    json!(12)
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameSpec {
    pub episode: String,
    pub run: String,
    pub trial: String,
    pub server: Value,
    pub actions: Value,
    pub rules: Value,
    pub numeric_costs: Value,
    pub policies: [PolicySpec; 2],
    pub max_decisions: usize,
    pub deadline: f64,
    pub cleanup_deadline: f64,
    #[serde(default)]
    pub runtime_evidence: Value,
    #[serde(default)]
    pub clock: Value,
}
impl GameSpec {
    pub fn validate(&self) -> Result<()> {
        let config = Config::parse(&self.actions, &self.rules)?;
        ensure(
            !self.episode.is_empty()
                && !self.run.is_empty()
                && !self.trial.is_empty()
                && self.max_decisions > 0,
            "invalid game routing/budget",
        )?;
        super::requests::wait_duration(self.deadline)?;
        super::requests::wait_duration(self.cleanup_deadline)?;
        ensure(
            self.cleanup_deadline >= self.deadline,
            "game cleanup deadline must cover execution deadline",
        )?;
        Costs::parse(&self.numeric_costs)?;
        for p in &self.policies {
            ensure(
                ["random", "heuristic", "refresh", "mlp"].contains(&p.kind.as_str()),
                "unknown native policy",
            )?;
            if p.kind != "mlp" {
                ensure(
                    config.max_sequence_length <= 18,
                    "corrected baseline kind/game capacity required",
                )?;
            } else {
                let checkpoint = p
                    .checkpoint
                    .as_deref()
                    .ok_or_else(|| error("MLP checkpoint required"))?;
                let l = p
                    .limits
                    .as_ref()
                    .ok_or_else(|| error("configured MLP model limits required"))?;
                ensure(
                    checkpoint.len() == 64
                        && checkpoint
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                    "invalid checkpoint",
                )?;
                ensure(
                    l.max_decisions > 0
                        && l.max_episode_events > 0
                        && l.max_interval_events > 0
                        && l.max_candidates > 0
                        && (1..=128).contains(&l.max_positions)
                        && (1..=65536).contains(&l.aggregate_window)
                        && config.max_sequence_length <= l.max_positions
                        && config.max_candidates <= l.max_candidates,
                    "configured model capacity mismatch",
                )?;
            }
            baselines::comparison_threshold(&p.attack_threshold)?;
        }
        let _: Server =
            serde_json::from_value(self.server.clone()).map_err(|e| error(e.to_string()))?;
        // Full schema2 receipt validation belongs to the existing Python live
        // process owner/facade. Never reinterpret it as historical schema1.
        ensure(
            self.runtime_evidence["schema_version"].as_u64() == Some(2)
                && self.runtime_evidence["action_contract"] == config.action_contract
                && self.runtime_evidence["source"]["dirty"] == false
                && self.runtime_evidence["source"]["revision"]
                    == config.action_contract["simulator_revision"]
                && self.runtime_evidence["binary"]["sha256"]
                    == "76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d"
                && self.runtime_evidence["executed"]["sha256"]
                    == self.runtime_evidence["binary"]["sha256"]
                && self.runtime_evidence["executed"]["pid"]
                    .as_u64()
                    .is_some_and(|p| p > 0),
            "actual executed shared Simulator receipt required",
        )?;
        Ok(())
    }
}

/// Policy interface is scoped to one participant's input, table and own state.
/// It has no HTTP, routing, storage, rejection diagnostics or other seat access.
trait OwnPolicy: Send {
    fn select(&mut self, input: &Input, config: &Config, table: &Table) -> Result<Option<usize>>;
    fn state(&self) -> Value;
}
struct Baseline {
    kind: String,
    rng: PythonRng,
    refresh: RefreshState,
    threshold: u64,
}
impl OwnPolicy for Baseline {
    fn select(&mut self, input: &Input, config: &Config, table: &Table) -> Result<Option<usize>> {
        Ok(Some(match self.kind.as_str() {
            "random" => self.rng.range(table.kinds.len() as u64)? as usize,
            "heuristic" => baselines::heuristic(input, config, &table.proposals())?,
            "refresh" => self.refresh.select(input, config, table, self.threshold)?,
            _ => unreachable!(),
        }))
    }
    fn state(&self) -> Value {
        json!({"rng":self.rng.state(),"refresh":self.refresh})
    }
}
struct Mlp;
impl OwnPolicy for Mlp {
    fn select(&mut self, _: &Input, _: &Config, _: &Table) -> Result<Option<usize>> {
        Ok(None)
    }
    fn state(&self) -> Value {
        Value::Null
    }
}
struct Participant {
    history: History,
    search: PythonRng,
    policy: Box<dyn OwnPolicy>,
    spec: PolicySpec,
    decisions: usize,
    accepted: Vec<Value>,
}
impl Participant {
    fn new(actor: u8, episode: String, spec: PolicySpec) -> Result<Self> {
        let policy: Box<dyn OwnPolicy> = if spec.kind == "mlp" {
            Box::new(Mlp)
        } else {
            Box::new(Baseline {
                kind: spec.kind.clone(),
                rng: PythonRng::seed(spec.seed),
                refresh: RefreshState::default(),
                threshold: baselines::comparison_threshold(&spec.attack_threshold)?,
            })
        };
        Ok(Self {
            history: History::new(actor, episode)?,
            search: PythonRng::seed(search_seed(spec.seed)?),
            policy,
            spec,
            decisions: 0,
            accepted: vec![],
        })
    }
    fn limits(&self) -> Limits {
        self.spec.limits.clone().unwrap_or(Limits {
            max_decisions: usize::MAX,
            max_episode_events: i32::MAX as usize,
            max_interval_events: 65536,
            max_candidates: 65536,
            max_positions: 18,
            aggregate_window: 128,
        })
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct GameReport {
    pub episode: String,
    pub game: Option<String>,
    pub end_kind: String,
    pub outcome: Value,
    pub simulator_abort_reason: Option<String>,
    pub decisions: usize,
    pub error: Option<String>,
    pub cleanup_errors: Vec<String>,
    pub storage_accepted: bool,
    pub inference_released: bool,
}
struct Actor {
    spec: GameSpec,
    http: Http,
    game: Option<String>,
    seats: [Participant; 2],
    report: GameReport,
    final_observations: Vec<Value>,
    replays: Vec<Value>,
}
fn command_id() -> Result<String> {
    use std::io::Read;
    let mut b = [0u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut b)?;
    b[6] = (b[6] & 15) | 64;
    b[8] = (b[8] & 63) | 128;
    let h: String = b.iter().map(|v| format!("{v:02x}")).collect();
    Ok(format!(
        "{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    ))
}
fn action(frame: &Value, table: &Table, index: usize) -> Result<Value> {
    ensure(
        index < table.kinds.len(),
        "selected candidate outside own table",
    )?;
    let kind = &table.kinds[index];
    let target = &table.targets[index];
    let slots = &table.slots[table.offsets[index]..table.offsets[index + 1]];
    let numerics = &table.numerics[index];
    let presence = &table.presence[index];
    let refs = &frame["payload"]["observation"]["own"]["cards"];
    let elements: Vec<Value> = slots.iter().map(|i| refs[*i]["card_ref"].clone()).collect();
    let mut a = json!({"type":kind});
    match kind.as_str() {
        "play_sequence" | "respond" => a["ordered_card_refs"] = json!(elements),
        "discard" | "sacrifice" => a["card_refs"] = json!(elements),
        "exchange" | "sell" | "buy" | "invoke_phenomenon" => {
            a["card_ref"] = elements[0].clone();
            if kind == "sell" {
                a["sold_card_ref"] = elements[1].clone();
            }
        }
        "pray" | "buy_decision" => {}
        _ => return Err(error("unknown typed candidate")),
    }
    if target != "omitted" {
        a["target"] = json!(if target == "self" {
            frame["actor"].as_u64().unwrap()
        } else {
            3 - frame["actor"].as_u64().unwrap()
        });
    }
    for (i, k) in ["hp", "mp", "money", "accept"].iter().enumerate() {
        if presence[i] {
            a[*k] = if i == 3 {
                json!(numerics[i] == 1)
            } else {
                json!(numerics[i])
            };
        }
    }
    card_instance::action_value2(&a, Some(&frame["payload"]["observation"]))?;
    Ok(a)
}
impl Actor {
    fn new(spec: GameSpec, client: reqwest::Client) -> Result<Self> {
        let http = Http::new(
            client,
            serde_json::from_value(spec.server.clone()).map_err(|e| error(e.to_string()))?,
            spec.deadline,
        )?;
        let seats = [
            Participant::new(1, spec.episode.clone(), spec.policies[0].clone())?,
            Participant::new(2, spec.episode.clone(), spec.policies[1].clone())?,
        ];
        let report = GameReport {
            episode: spec.episode.clone(),
            game: None,
            end_kind: "incomplete".into(),
            outcome: Value::Null,
            simulator_abort_reason: None,
            decisions: 0,
            error: None,
            cleanup_errors: vec![],
            storage_accepted: false,
            inference_released: false,
        };
        Ok(Self {
            spec,
            http,
            game: None,
            seats,
            report,
            final_observations: vec![],
            replays: vec![],
        })
    }
    async fn play(&mut self, dispatcher: &Arc<Dispatcher>, storage: &Arc<Mailbox>) -> Result<()> {
        self.http.attest().await?;
        let definitions = self.http.catalog().await?;
        let game = self.http.create().await?;
        self.game = Some(game.clone());
        self.report.game = Some(game.clone());
        self.http.attest().await?; // Required post-Create-before-Join fence.
        for viewer in [1, 2] {
            self.http.join(&game, &command_id()?, viewer).await?;
        }
        let config = Config::parse(&self.spec.actions, &self.spec.rules)?;
        let costs = Costs::parse(&self.spec.numeric_costs)?;
        let budget = Budget::new(12usize << 30);
        let control = Control::new(f64::MAX)?;
        let mut obs = self.http.observe(&game, 1).await?;
        loop {
            if !obs["result"].is_null() {
                self.report.end_kind = "natural".into();
                self.report.outcome = obs["result"].clone();
                break;
            }
            if self.report.decisions >= self.spec.max_decisions {
                self.report.end_kind = "truncated".into();
                return Err(error("decision_limit"));
            }
            let actor = obs["current_player"].integer(1, 2)? as u8;
            if obs["viewer_player_id"] != actor {
                obs = self.http.observe(&game, actor).await?;
            }
            ensure(
                obs["actionable"] == true,
                "unsupported non-actionable actor",
            )?;
            let seat = &mut self.seats[(actor - 1) as usize];
            let limits = seat.limits();
            ensure(
                seat.decisions < limits.max_decisions,
                "configured own model decision capacity exceeded",
            )?;
            let (events, cursor) = self
                .http
                .events(
                    &game,
                    actor,
                    seat.history.cursor.clone(),
                    limits
                        .max_interval_events
                        .min(limits.max_episode_events - seat.history.events.len()),
                )
                .await?;
            seat.history.append(
                events,
                cursor,
                limits.max_episode_events,
                limits.max_interval_events,
            )?;
            let options = self.http.options(&game, actor).await?;
            let (frame, delta) = seat.history.frame(obs.clone(), options, &definitions)?;
            let input = seat.history.input(&frame, &config)?;
            let search_before = seat.search.state();
            let policy_before = seat.policy.state();
            let mut clock = search::Clock::parse(&self.spec.clock)?;
            let generated = search::generate_typed(
                &input,
                &config,
                &self.spec.rules,
                &costs,
                &mut seat.search,
                &mut clock,
            )?;
            ensure(
                generated.table.kinds.len() <= limits.max_candidates,
                "configured candidate capacity exceeded",
            )?;
            let mut scores = None;
            let selected =
                if let Some(index) = seat.policy.select(&input, &config, &generated.table)? {
                    index
                } else {
                    let row = ServingRow::build_owned(
                        &frame,
                        &seat.history.events,
                        &input,
                        &config,
                        &generated.table,
                        limits.aggregate_window,
                        limits.max_positions,
                        &budget,
                        &control,
                    )?;
                    let identity = Identity {
                        run: self.spec.run.clone(),
                        trial: self.spec.trial.clone(),
                        game: game.clone(),
                        player: actor,
                        epoch: seat.spec.epoch,
                        request: seat.decisions as u64,
                        revision: input.revision,
                        checkpoint: seat.spec.checkpoint.clone().unwrap(),
                        deadline: self.spec.deadline,
                    };
                    let (_, handle) = dispatcher
                        .enqueue_async(identity.clone(), row, seat.spec.seed)
                        .await?;
                    let answer = handle.wait_async(self.spec.deadline).await?;
                    ensure(
                        answer.identity == identity,
                        "own inference answer identity mismatch",
                    )?;
                    let selected = answer.sample;
                    scores = Some(capture::Scores {
                        identity: answer.identity,
                        logprob: f64::from(answer.logs[selected]),
                        sample: selected,
                        value: f64::from(answer.value),
                    });
                    selected
                };
            let command = json!({"command_id":command_id()?,"expected_revision":input.revision,"action":action(&frame,&generated.table,selected)?});
            // Immutable validated pending and writer custody precede the first Act.
            let search_after = seat.search.state();
            let policy_after = seat.policy.state();
            let pending = capture::Pending {
                schema: "rust-player-pending-v1",
                run: &self.spec.run,
                trial: &self.spec.trial,
                game: &game,
                episode: &self.spec.episode,
                actor,
                frame: &frame,
                delta: &delta,
                generated: (&generated).into(),
                selected,
                scores: scores.as_ref(),
                command: &command,
                search_rng_before: &search_before,
                search_rng_after: &search_after,
                policy_state_before: &policy_before,
                policy_state_after: &policy_after,
                sampling_seed: seat.spec.seed,
                policy: &seat.spec,
            };
            storage
                .publish(
                    "pending",
                    &self.spec.episode,
                    &game,
                    &pending,
                    None,
                    self.spec.deadline,
                )
                .await?;
            let ack = self.http.act(&game, actor, &command).await?;
            if ack["status"] == "terminal" {
                self.report.end_kind = "natural".into();
                self.report.outcome = ack["result"].clone();
            } else if ack["status"] == "aborted" {
                self.report.end_kind = "aborted".into();
            }
            seat.accepted.push(command["action"].clone());
            seat.decisions += 1;
            self.report.decisions += 1;
            let record = capture::Ack {
                schema: "rust-player-ack-v1",
                game: &game,
                episode: &self.spec.episode,
                actor,
                command: &command,
                acknowledgement: &ack,
            };
            storage
                .publish(
                    "ack",
                    &self.spec.episode,
                    &game,
                    &record,
                    None,
                    self.spec.deadline,
                )
                .await?;
            if ack["status"] != "active" {
                break;
            }
            obs = self.http.observe(&game, actor).await?;
            tokio::task::yield_now().await;
        }
        Ok(())
    }
    async fn cleanup(&mut self, dispatcher: &Arc<Dispatcher>) {
        self.http.deadline = self.spec.cleanup_deadline;
        let Some(game) = self.game.clone() else {
            self.report.inference_released = true;
            return;
        };
        // Ended actor never readmits. Retirement erases private sampling state;
        // release waits actual exported service leases, not merely actor promises.
        for actor in [1, 2] {
            let p = &self.spec.policies[(actor - 1) as usize];
            let identity = Identity {
                run: self.spec.run.clone(),
                trial: self.spec.trial.clone(),
                game: game.clone(),
                player: actor,
                epoch: p.epoch,
                request: 0,
                revision: 0,
                checkpoint: p.checkpoint.clone().unwrap_or_default(),
                deadline: self.http.deadline,
            };
            if let Err(e) = dispatcher.retire(&identity) {
                self.report.cleanup_errors.push(e.to_string());
            }
        }
        let mut closures = vec![];
        for viewer in [1, 2] {
            match self.http.observe(&game, viewer).await {
                Ok(obs) => {
                    if !self.report.outcome.is_null() && obs["result"] != self.report.outcome {
                        self.report
                            .cleanup_errors
                            .push("terminal participant result mismatch".into());
                    }
                    self.final_observations.push(obs);
                }
                Err(e) => self
                    .report
                    .cleanup_errors
                    .push(format!("observe seat{viewer}: {e}")),
            }
            match self.http.replay(&game, viewer).await {
                Ok(replay) => {
                    let accepted: Vec<_> =
                        self.seats[(viewer - 1) as usize].accepted.iter().collect();
                    match card_instance::reconcile_replay2(
                        &replay,
                        &game,
                        viewer as i128,
                        &accepted,
                        &self.report.outcome,
                        &Budget::new(12usize << 30),
                        &Control::new(f64::MAX).unwrap(),
                    ) {
                        Ok(closure) => {
                            closures.push(closure.cloned());
                        }
                        Err(e) => self
                            .report
                            .cleanup_errors
                            .push(format!("Replay seat{viewer}: {e}")),
                    };
                    self.replays.push(replay);
                }
                Err(e) => self
                    .report
                    .cleanup_errors
                    .push(format!("Replay seat{viewer}: {e}")),
            }
        }
        if closures.len() == 2 {
            if closures[0] != closures[1] {
                self.report
                    .cleanup_errors
                    .push("participant Replay closure identities disagree".into());
            } else if let Some(closure) = &closures[0] {
                if self.report.outcome.is_null() {
                    if closure["type"] == "aborted" {
                        let reason = closure["reason"].as_str().unwrap();
                        self.report.simulator_abort_reason = Some(reason.into());
                        self.report.end_kind = if reason == "turn_limit_exceeded" {
                            "truncated"
                        } else {
                            "aborted"
                        }
                        .into();
                    } else {
                        self.report.cleanup_errors.push(
                            "terminal Replay cannot fabricate an unknown natural outcome".into(),
                        );
                    }
                }
            } else if self.report.end_kind == "aborted" {
                self.report
                    .cleanup_errors
                    .push("simulator abort lacks reconciled closure evidence".into());
            }
        }
        if let Err(e) = self.http.delete(&game).await {
            self.report.cleanup_errors.push(format!("Delete: {e}"));
        }
        match dispatcher
            .release_game_async(&self.spec.run, &self.spec.trial, &game, self.http.deadline)
            .await
        {
            Ok(()) => self.report.inference_released = true,
            Err(e) => self
                .report
                .cleanup_errors
                .push(format!("inference release: {e}")),
        }
    }
}
pub async fn run(
    spec: GameSpec,
    client: reqwest::Client,
    dispatcher: Arc<Dispatcher>,
    storage: Arc<Mailbox>,
    cancel: Arc<super::scheduler::Cancellation>,
) -> GameReport {
    let episode = spec.episode.clone();
    let mut actor = match Actor::new(spec, client) {
        Ok(v) => v,
        Err(e) => {
            return GameReport {
                episode,
                game: None,
                end_kind: "incomplete".into(),
                outcome: Value::Null,
                simulator_abort_reason: None,
                decisions: 0,
                error: Some(e.to_string()),
                cleanup_errors: vec![],
                storage_accepted: false,
                inference_released: false,
            };
        }
    };
    let result = tokio::select! {biased;_=cancel.wait()=>Err(Error::Cancelled),result=actor.play(&dispatcher,&storage)=>result};
    if let Err(e) = result {
        if matches!(e, Error::ResourceLimit(_)) {
            cancel.cancel();
            storage.close(e.clone());
        }
        if actor.report.outcome.is_null()
            && actor.report.end_kind != "aborted"
            && actor.report.end_kind != "truncated"
        {
            actor.report.end_kind = match e {
                Error::Cancelled => "cancelled",
                Error::Deadline => "truncated",
                _ => "incomplete",
            }
            .into();
        }
        actor.report.error = Some(e.to_string());
    }
    actor.cleanup(&dispatcher).await;
    let game = actor.game.as_deref().unwrap_or("");
    let release_key = if actor.report.inference_released && actor.report.cleanup_errors.is_empty() {
        Some((
            actor.spec.run.clone(),
            actor.spec.trial.clone(),
            episode.clone(),
        ))
    } else {
        None
    };
    let terminal = capture::Terminal {
        schema: "rust-player-terminal-v1",
        report: &actor.report,
        run: &actor.spec.run,
        trial: &actor.spec.trial,
        episode: &actor.spec.episode,
        game: &actor.game,
        runtime_evidence: &actor.spec.runtime_evidence,
        actions: &actor.spec.actions,
        rules: &actor.spec.rules,
        numeric_costs: &actor.spec.numeric_costs,
        clock: &actor.spec.clock,
        policies: &actor.spec.policies,
        terminal_observations: &actor.final_observations,
        replays: &actor.replays,
        banks: [(&actor.seats[0]).into(), (&actor.seats[1]).into()],
    };
    let deadline = actor.spec.cleanup_deadline;
    // Cancellation/failure still reports the explicit final diagnostics through
    // the seam if it remains open; no manufactured archive success.
    match storage
        .publish("terminal", &episode, game, &terminal, release_key, deadline)
        .await
    {
        Ok(()) => actor.report.storage_accepted = true,
        Err(e) => {
            cancel.cancel();
            storage.close(e.clone());
            actor
                .report
                .cleanup_errors
                .push(format!("terminal handoff: {e}"));
        }
    };
    actor.report
}
