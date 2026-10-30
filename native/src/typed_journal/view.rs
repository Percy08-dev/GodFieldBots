//! Closed reader dispatch: explicit decision facts/actions/numeric preparation.
//! This is not an object/array/property adapter over the ordinary input tree.
use super::{dto, dto::contract::Validate, ensure, journal, required};
use crate::{
    budget::{Budget, BudgetVec, Control},
    error::{Error, Result},
    feature,
    json::{Value, ValueExt},
    parallel::JournalData,
    session::Reader,
};
use serde::Deserialize;
use std::sync::Arc;

pub(crate) enum JournalView<'records, 'raw> {
    Legacy(crate::journal::Journal<'records>),
    Typed(journal::Journal<'records, 'raw>),
}
pub(crate) struct JournalProvenance<'a> {
    pub header: &'a Value,
    pub end: Option<&'a Value>,
    pub decision_count: usize,
    pub accepted_count: usize,
}
impl<'records, 'raw> JournalView<'records, 'raw> {
    pub fn validate(
        data: &'records JournalData,
        parsed: Option<&'records journal::Records<'raw>>,
        reader: Reader,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<Self> {
        match (data, parsed, reader) {
            (JournalData::Legacy(records), None, Reader::Legacy) => {
                crate::journal::validate(records, budget, control).map(Self::Legacy)
            }
            (JournalData::Typed(_), Some(records), Reader::TypedBorrowedV1) => {
                journal::validate(records, budget, control).map(Self::Typed)
            }
            _ => Err(Error::contract("mixed Journal representation/storage")),
        }
    }
    pub fn provenance(&self) -> JournalProvenance<'_> {
        match self {
            Self::Legacy(j) => JournalProvenance {
                header: j.header,
                end: j.end,
                decision_count: j.decisions.len(),
                accepted_count: j.accepted.len(),
            },
            Self::Typed(j) => JournalProvenance {
                header: &j.header,
                end: j.end.as_ref(),
                decision_count: j.decisions.len(),
                accepted_count: j.accepted.len(),
            },
        }
    }
    pub fn header(&self) -> &Value {
        self.provenance().header
    }
    pub fn end(&self) -> Option<&Value> {
        self.provenance().end
    }
    pub fn decision_count(&self) -> usize {
        self.provenance().decision_count
    }
    pub fn decision(&self, index: usize) -> Result<DecisionView<'_, 'raw>> {
        match self {
            Self::Legacy(j) => j
                .decisions
                .get(index)
                .map(|record| DecisionView::Legacy(record)),
            Self::Typed(j) => j
                .decisions
                .get(index)
                .map(|record| DecisionView::Typed(record)),
        }
        .ok_or_else(|| Error::contract("decision index outside validated Journal"))
    }
    pub fn reconcile_replay(
        &self,
        replay: &Value,
        viewer: i128,
        outcome: &Value,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<()> {
        match self {
            Self::Legacy(journal) => {
                let mut accepted = BudgetVec::with_capacity(budget, journal.accepted.len())?;
                for decision in journal.accepted.iter() {
                    control.check()?;
                    accepted.push(decision.require("command")?.require("action")?)?;
                }
                crate::validation::reconcile_replay_controlled(
                    replay, viewer, &accepted, outcome, control,
                )
            }
            Self::Typed(journal) => reconcile_typed(replay, viewer, journal, outcome, control),
        }
    }
}
#[derive(Clone, Copy)]
pub(crate) enum DecisionView<'a, 'raw> {
    Legacy(&'a Value),
    Typed(&'a dto::Record<'raw>),
}
#[derive(Clone, Copy)]
pub(crate) enum ActionRef<'a, 'raw> {
    Legacy(&'a Value),
    Typed(&'a dto::Action<'raw>),
}
pub(crate) enum PreparedFeatures<'a, 'raw> {
    Legacy(feature::PreparedDecision<'a>),
    Typed(feature::typed::PreparedDecision<'a, 'raw>),
}
impl<'a, 'raw> DecisionView<'a, 'raw> {
    pub fn number(self) -> Result<u64> {
        match self {
            Self::Legacy(v) => Ok(v.require("decision")?.integer(0, u64::MAX as i128)? as u64),
            Self::Typed(v) => Ok(*required(&v.decision)?),
        }
    }
    pub fn identity_number(self, control: &Control) -> Result<Value> {
        match self {
            Self::Legacy(v) => crate::json::copy(v.require("decision")?, control),
            Self::Typed(v) => Ok(crate::json::integer(*required(&v.decision)? as i128)),
        }
    }
    pub fn selected(self) -> Result<usize> {
        let index = match self {
            Self::Legacy(v) => v
                .require("selection")?
                .require("index")?
                .integer(0, u64::MAX as i128)? as u64,
            Self::Typed(v) => required(&v.selection)?.index,
        };
        usize::try_from(index).map_err(|_| Error::contract("selected index overflow"))
    }
    pub fn policy_version(self) -> Result<&'a str> {
        match self {
            Self::Legacy(v) => v.require("selection")?.require("policy_version")?.string(),
            Self::Typed(v) => Ok(required(&v.selection)?.policy_version.as_ref()),
        }
    }
    pub fn old_statistics(self) -> Result<(Option<f64>, Option<f64>)> {
        match self {
            Self::Legacy(v) => {
                let s = v.require("selection")?;
                let get = |name| -> Result<Option<f64>> {
                    let v = s.require(name)?;
                    if v.is_null() {
                        Ok(None)
                    } else {
                        Ok(Some(v.number()?))
                    }
                };
                Ok((get("logprob")?, get("value")?))
            }
            Self::Typed(v) => {
                let s = required(&v.selection)?;
                Ok((s.logprob, s.value))
            }
        }
    }
    pub fn candidate_count(self) -> Result<usize> {
        match self {
            Self::Legacy(v) => Ok(v.require("candidates")?.require("actions")?.array()?.len()),
            Self::Typed(v) => Ok(required(&v.candidates)?.actions.len()),
        }
    }
    pub fn action(self, index: usize) -> Result<ActionRef<'a, 'raw>> {
        match self {
            Self::Legacy(v) => v
                .require("candidates")?
                .require("actions")?
                .array()?
                .get(index)
                .map(ActionRef::Legacy),
            Self::Typed(v) => required(&v.candidates)?
                .actions
                .get(index)
                .map(ActionRef::Typed),
        }
        .ok_or_else(|| Error::contract("candidate index outside validated decision"))
    }
    pub fn command_action(self) -> Result<ActionRef<'a, 'raw>> {
        match self {
            Self::Legacy(v) => Ok(ActionRef::Legacy(v.require("command")?.require("action")?)),
            Self::Typed(v) => Ok(ActionRef::Typed(&required(&v.command)?.action)),
        }
    }
    pub fn prepare(
        self,
        config: &'a feature::FeatureConfig,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<PreparedFeatures<'a, 'raw>> {
        match self {
            Self::Legacy(v) => {
                feature::PreparedDecision::new(v.require("input")?, config, budget, control)
                    .map(PreparedFeatures::Legacy)
            }
            Self::Typed(v) => {
                feature::typed::PreparedDecision::new(required(&v.input)?, config, budget, control)
                    .map(PreparedFeatures::Typed)
            }
        }
    }
}
impl<'a, 'raw> PreparedFeatures<'a, 'raw> {
    pub fn state(&self, output: &mut [f64], control: &Control) -> Result<()> {
        match self {
            Self::Legacy(p) => p.state(output, control),
            Self::Typed(p) => p.state(output, control),
        }
    }
    pub fn encode(
        &self,
        action: ActionRef<'a, 'raw>,
        output: &mut [f64],
        control: &Control,
    ) -> Result<()> {
        match (self, action) {
            (Self::Legacy(p), ActionRef::Legacy(action)) => p.encode(action, output, control),
            (Self::Typed(p), ActionRef::Typed(action)) => p.encode(action, output, control),
            _ => Err(Error::contract("mixed action/feature representations")),
        }
    }
}
fn reconcile_typed(
    replay: &Value,
    viewer: i128,
    journal: &journal::Journal<'_, '_>,
    outcome: &Value,
    control: &Control,
) -> Result<()> {
    control.check()?;
    ensure(
        crate::validation::equals_player(replay.require("viewer_player_id")?, viewer)
            && replay.require("observation_schema_version")?.string()? == "1",
        "replay perspective/version mismatch",
    )?;
    let mut index = 0;
    if let Some(turns) = replay.get("turns") {
        for turn in crate::validation::worker_events(turns)? {
            control.check()?;
            turn.object()?;
            if let Some(decisions) = turn.get("decisions") {
                for decision in crate::validation::worker_events(decisions)? {
                    control.check()?;
                    decision.object()?;
                    if decision
                        .get("actor")
                        .is_none_or(|actor| !crate::validation::equals_player(actor, viewer))
                    {
                        continue;
                    }
                    if let Some(visible) = decision
                        .get("visible_action")
                        .filter(|v| crate::json::truthy(v))
                    {
                        visible.object()?;
                        if visible
                            .get("type")
                            .is_some_and(|v| v.as_str() == Some("action"))
                        {
                            ensure(
                                index < journal.accepted.len(),
                                "accepted actions/replay mismatch",
                            )?;
                            // Standard Serde borrows this small action view directly from
                            // legacy Replay Value; no serialize/parse or policy-input DOM.
                            let action = dto::Action::deserialize(visible.require("action")?)
                                .map_err(|e| Error::contract(e.to_string()))?;
                            action.validate().map_err(Error::contract)?;
                            let saved = &required(&journal.accepted[index].command)?.action;
                            let equal = if action.r#type.as_ref() == "play_sequence"
                                && saved.r#type.as_ref() == "play_sequence"
                            {
                                action.ordered_elements == saved.ordered_elements
                                    && action.target.value() == saved.target.value()
                            } else {
                                action == *saved
                            };
                            ensure(equal, "accepted actions/replay mismatch")?;
                            index += 1;
                        }
                    }
                }
            }
        }
    }
    ensure(
        index == journal.accepted.len(),
        "accepted actions/replay mismatch",
    )?;
    if !outcome.is_null() {
        let end = replay.require("episode_end")?;
        end.closed(&["type", "result"], &[])?;
        ensure(
            end.require("type")?.string()? == "terminal"
                && end.require("result")?.eq_python(outcome),
            "terminal outcome/replay mismatch",
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        budget::monotonic, json, parallel::EpisodeReader, session::Reader, source::Sources,
    };
    use std::{fs, path::PathBuf};
    struct Fixture(PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn worker_owned_raw_scopes_reconcile_replay_and_reject_mixed_features() {
        let fixture = Fixture(std::env::temp_dir().join(format!(
            "gf-typed-views-{}-{}",
            std::process::id(),
            monotonic().to_bits()
        )));
        fs::create_dir(&fixture.0).unwrap();
        let values = super::super::journal::tests::fixture();
        let bytes = super::super::journal::tests::wire(&values);
        for name in ["player-1.jsonl", "player-2.jsonl"] {
            fs::write(fixture.0.join(name), &bytes).unwrap();
        }
        let replay = serde_json::json!({"game_id":"g","viewer_player_id":1,"observation_schema_version":"1",
            "turns":[{"decisions":[{"actor":1,"visible_action":{"type":"action","action":{"type":"pray"}}}]}],
            "episode_end":{"type":"terminal","result":"draw"}});
        for name in ["replay-player-1.json", "replay-player-2.json"] {
            fs::write(fixture.0.join(name), serde_json::to_vec(&replay).unwrap()).unwrap();
        }
        for workers in [1, 4] {
            let budget = Budget::new(128 * 1024 * 1024);
            let control = Control::new(monotonic() + 10.).unwrap();
            let telemetry = Arc::new(crate::telemetry::Telemetry::default());
            let mut sources = Sources::with_telemetry(
                budget.clone(),
                64 * 1024 * 1024,
                8 * 1024 * 1024,
                &telemetry,
            );
            let paths = [fixture.0.clone()];
            let mut reader = EpisodeReader::new(
                &paths,
                workers,
                workers,
                &budget,
                &control,
                64 * 1024 * 1024,
                8 * 1024 * 1024,
                Reader::TypedBorrowedV1,
            );
            let (_, data) = reader.next(&mut sources).unwrap().unwrap();
            let records = data
                .records1
                .parse_typed(&control, &telemetry)
                .unwrap()
                .unwrap();
            let typed = JournalView::validate(
                &data.records1,
                Some(&records),
                Reader::TypedBorrowedV1,
                &budget,
                &control,
            )
            .unwrap();
            assert!(
                JournalView::validate(&data.records1, None, Reader::Legacy, &budget, &control)
                    .is_err()
            );
            assert!(
                JournalView::validate(
                    &data.records1,
                    Some(&records),
                    Reader::Legacy,
                    &budget,
                    &control
                )
                .is_err()
            );
            typed
                .reconcile_replay(&replay, 1, &serde_json::json!("draw"), &budget, &control)
                .unwrap();
            let mut bad = replay.clone();
            bad["turns"][0]["decisions"][0]["visible_action"]["action"]["artifact"] =
                serde_json::json!(1);
            assert!(
                typed
                    .reconcile_replay(&bad, 1, &serde_json::json!("draw"), &budget, &control)
                    .is_err()
            );
            let config = feature::FeatureConfig {
                history: "none".into(),
                max_elements: 3,
                hash_buckets: 32,
                history_window: 128,
            };
            let decision = typed.decision(0).unwrap();
            let prepared = decision.prepare(&config, &budget, &control).unwrap();
            let mut output = vec![0.; config.width()];
            prepared
                .encode(decision.action(0).unwrap(), &mut output, &control)
                .unwrap();
            assert!(
                prepared
                    .encode(
                        ActionRef::Legacy(&values[1]["command"]["action"]),
                        &mut output,
                        &control
                    )
                    .is_err()
            );
            let metrics: Value = serde_json::from_str(&telemetry.json(0, 0, 0)).unwrap();
            assert_eq!(metrics["typed_journal_files"], 1);
            assert_eq!(metrics["typed_journal_records"], 5);
            assert_eq!(metrics["legacy_journal_files"], 0);
            assert_eq!(metrics["reader_contract"], "typed-borrowed-v1");
            // The retained numeric copy remains usable after every DTO/raw/source owner is gone.
            let expected = output.clone();
            drop(prepared);
            drop(typed);
            drop(records);
            drop(data);
            drop(reader);
            drop(sources);
            assert_eq!(output, expected);
            assert_eq!(budget.used(), 0);
        }
        // Legacy views preserve the old scalar identity formatting path.
        let budget = Budget::new(128 * 1024 * 1024);
        let control = Control::new(monotonic() + 10.).unwrap();
        let docs: Vec<_> = values
            .iter()
            .map(|value| {
                json::parse(&serde_json::to_vec(value).unwrap(), &budget, &control).unwrap()
            })
            .collect();
        let data = JournalData::Legacy(docs);
        assert!(
            JournalView::validate(&data, None, Reader::TypedBorrowedV1, &budget, &control).is_err()
        );
        let view = JournalView::validate(&data, None, Reader::Legacy, &budget, &control).unwrap();
        assert_eq!(
            view.decision(0).unwrap().identity_number(&control).unwrap(),
            values[1]["decision"]
        );
    }
}
