//! Structural journal ledger; reward perspective stays in Python's learning layer.
use crate::budget::{Budget, BudgetVec, Control};
use crate::error::{Error, Result};
use crate::json::ValueExt;
use crate::json::{Document, Value};
use crate::validation;
use std::sync::Arc;

pub const FEATURE_VERSION: &str = "participant-json-v2";
pub const REWARD_VERSION: &str = "natural-terminal-v1";
pub const POLICY_INPUT_VERSION: &str = "player-atomic-options-v1";
pub const END_PROJECTION_VERSION: &str = "simulator-closure-v1";
pub const GENERATOR_VERSION: &str = "public-atomic-subset-v3";
pub const SCORER_VERSION: &str = "public-content-v1";
pub const CATALOG_HASH: &str = "481ddfb0c068a1d2aaaf2dbcb6a5b4bd04396dfcc5c995e14e47b71222b6a51d";

pub fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value.require(key)?.string()
}
pub fn int(value: &Value, key: &str) -> Result<i128> {
    value.require(key)?.integer(0, u64::MAX as i128)
}
pub fn equal(actual: &Value, expected: &Value, message: &str) -> Result<()> {
    if !actual.eq_python(expected) {
        return Err(Error::contract(message));
    }
    Ok(())
}
fn expect_text(value: &Value, key: &str, expected: &str) -> Result<()> {
    if string(value, key)? != expected {
        return Err(Error::contract(format!("incompatible {key}")));
    }
    Ok(())
}
fn nonempty(value: &Value) -> Result<()> {
    if value.string()?.is_empty() {
        return Err(Error::contract("nonempty string required"));
    }
    Ok(())
}
pub fn catalog_meta(value: &Value) -> Result<()> {
    value.closed(
        &[
            "catalog_schema_version",
            "catalog_version",
            "rule_set_version",
            "locale",
            "content_sha256",
            "definition_count",
        ],
        &[],
    )?;
    if value.require("catalog_schema_version")?.integer(1, 1)? != 1
        || value.require("definition_count")?.integer(296, 296)? != 296
    {
        return Err(Error::contract("incompatible catalog metadata"));
    }
    for (key, expected) in [
        ("catalog_version", "public-ja-v2"),
        ("rule_set_version", "true-match-1v1-v1"),
        ("locale", "ja"),
        ("content_sha256", CATALOG_HASH),
    ] {
        expect_text(value, key, expected)?;
    }
    Ok(())
}

pub struct Journal<'a> {
    pub header: &'a Value,
    pub decisions: BudgetVec<&'a Value>,
    pub accepted: BudgetVec<&'a Value>,
    pub transitions: BudgetVec<&'a Value>,
    pub end: Option<&'a Value>,
}

pub fn validate<'a>(
    records: &'a [Document],
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<Journal<'a>> {
    let header = &records
        .first()
        .ok_or_else(|| Error::contract("missing journal header"))?
        .root;
    header.closed(
        &[
            "kind",
            "schema_version",
            "feature_version",
            "reward_version",
            "catalog",
            "episode_id",
            "viewer",
            "policy",
            "run_id",
            "game_id",
            "action_config_hash",
            "generator_version",
            "scorer_version",
            "end_projection_version",
            "policy_input_version",
            "use_action_options",
        ],
        &[],
    )?;
    expect_text(header, "kind", "header")?;
    header.require("schema_version")?.integer(4, 4)?;
    for (key, expected) in [
        ("feature_version", FEATURE_VERSION),
        ("reward_version", REWARD_VERSION),
        ("policy_input_version", POLICY_INPUT_VERSION),
        ("end_projection_version", END_PROJECTION_VERSION),
        ("generator_version", GENERATOR_VERSION),
        ("scorer_version", SCORER_VERSION),
    ] {
        expect_text(header, key, expected)?;
    }
    catalog_meta(header.require("catalog")?)?;
    let mode = header.require("use_action_options")?.boolean()?;
    let viewer = header.require("viewer")?.integer(1, 2)?;
    for key in [
        "episode_id",
        "policy",
        "run_id",
        "game_id",
        "action_config_hash",
    ] {
        nonempty(header.require(key)?)?;
    }
    let mut ledger = Journal {
        header,
        decisions: BudgetVec::new(budget),
        accepted: BudgetVec::new(budget),
        transitions: BudgetVec::new(budget),
        end: None,
    };
    let mut pending: Option<usize> = None;
    let mut last_accepted: Option<usize> = None;
    let mut awaiting_next: Option<usize> = None;
    let mut terminal_transition: Option<&Value> = None;
    let mut last_revision = 0;
    let mut lifecycle_ack: Option<&Value> = None;
    for record in &records[1..] {
        control.check()?;
        let record = &record.root;
        if ledger.end.is_some() {
            return Err(Error::contract("records after journal end"));
        }
        match string(record, "kind")? {
            "decision" => {
                record.closed(
                    &[
                        "kind",
                        "decision",
                        "input",
                        "candidates",
                        "selection",
                        "command",
                        "selection_elapsed_ms",
                    ],
                    &[],
                )?;
                let index = usize::try_from(int(record, "decision")?)
                    .map_err(|_| Error::contract("decision index overflow"))?;
                if index != ledger.decisions.len()
                    || pending.is_some()
                    || last_accepted.is_some()
                    || terminal_transition.is_some()
                    || lifecycle_ack.is_some()
                {
                    return Err(Error::contract("invalid decision ordering"));
                }
                if awaiting_next.is_some_and(|next| next != index) {
                    return Err(Error::contract("transition/next decision mismatch"));
                }
                awaiting_next = None;
                let raw = record.require("input")?;
                validation::validate_journal_input_controlled(raw, Some(mode), control)?;
                equal(
                    raw.require("observation")?.require("viewer_player_id")?,
                    header.require("viewer")?,
                    "mixed participant journal",
                )?;
                let candidates = record.require("candidates")?;
                candidates.closed(
                    &[
                        "actions",
                        "generator_version",
                        "scorer_version",
                        "config_hash",
                        "attempts",
                        "public_prunes",
                        "approximate_drops",
                        "latency_ms",
                        "cutoff",
                        "cutoff_reasons",
                    ],
                    &[],
                )?;
                for (left, right) in [
                    ("generator_version", "generator_version"),
                    ("scorer_version", "scorer_version"),
                    ("config_hash", "action_config_hash"),
                ] {
                    equal(
                        candidates.require(left)?,
                        header.require(right)?,
                        "incompatible candidate versions/config",
                    )?;
                }
                let selection = record.require("selection")?;
                selection.closed(&["index", "policy_version", "logprob", "value"], &[])?;
                let selected = usize::try_from(int(selection, "index")?)
                    .map_err(|_| Error::contract("selected index overflow"))?;
                let actions = candidates.require("actions")?.array()?;
                if selected >= actions.len() {
                    return Err(Error::contract("invalid stored candidate selection"));
                }
                for field in ["logprob", "value"] {
                    let number = selection.require(field)?;
                    if !number.is_null() && !number.number()?.is_finite() {
                        return Err(Error::contract("invalid policy numeric output"));
                    }
                }
                let command = record.require("command")?;
                command.closed(&["command_id", "expected_revision", "action"], &[])?;
                uuid(string(command, "command_id")?)?;
                let revision = int(command, "expected_revision")?;
                if revision < last_revision {
                    return Err(Error::contract("participant revision moved backwards"));
                }
                equal(
                    command.require("expected_revision")?,
                    raw.require("observation")?.require("state_revision")?,
                    "command/input revision mismatch",
                )?;
                equal(
                    command.require("action")?,
                    &actions[selected],
                    "command/candidate mismatch",
                )?;
                ledger.decisions.push(record)?;
                pending = Some(index);
            }
            "acknowledgement" => {
                record.closed(&["kind", "decision", "response"], &[])?;
                let index = int(record, "decision")? as usize;
                if pending != Some(index) {
                    return Err(Error::contract(
                        "acknowledgement without unique pending decision",
                    ));
                }
                let response = record.require("response")?;
                acknowledgement(response, ledger.decisions[index].require("command")?)?;
                if string(response, "status")? != "active" {
                    lifecycle_ack = Some(response);
                }
                last_revision = int(response, "state_revision")?;
                ledger.accepted.push(ledger.decisions[index])?;
                pending = None;
                last_accepted = Some(index);
            }
            "transition" => {
                record.closed(
                    &["kind", "decision", "next_decision", "reward", "terminal"],
                    &[],
                )?;
                let index = int(record, "decision")? as usize;
                if Some(index) != last_accepted || terminal_transition.is_some() {
                    return Err(Error::contract("transition lacks last accepted action"));
                }
                if record.require("terminal")?.boolean()? {
                    record.require("reward")?.integer(-1, 1)?;
                    if !record.require("next_decision")?.is_null() {
                        return Err(Error::contract("terminal transition has next decision"));
                    }
                    terminal_transition = Some(record);
                } else {
                    if lifecycle_ack.is_some() {
                        return Err(Error::contract(
                            "continuing transition after lifecycle acknowledgement",
                        ));
                    }
                    record.require("reward")?.integer(0, 0)?;
                    if int(record, "next_decision")? != ledger.decisions.len() as i128 {
                        return Err(Error::contract("invalid continuing transition"));
                    }
                    awaiting_next = Some(ledger.decisions.len());
                }
                ledger.transitions.push(record)?;
                last_accepted = None;
            }
            "end" => {
                record.closed(
                    &[
                        "kind",
                        "end_kind",
                        "outcome",
                        "terminal_reward",
                        "final_observation",
                        "pending_decisions",
                        "simulator_abort_reason",
                    ],
                    &[],
                )?;
                let kind = string(record, "end_kind")?;
                if !["natural", "truncated", "aborted", "cancelled", "incomplete"].contains(&kind)
                    || awaiting_next.is_some()
                {
                    return Err(Error::contract("invalid journal end/order"));
                }
                let pending_values = record.require("pending_decisions")?.array()?;
                if let Some(index) = pending {
                    if pending_values.len() != 1
                        || pending_values[0].integer(0, u64::MAX as i128)? != index as i128
                    {
                        return Err(Error::contract("pending request ledger mismatch"));
                    }
                } else if !pending_values.is_empty() {
                    return Err(Error::contract("pending request ledger mismatch"));
                }
                let reason = record.require("simulator_abort_reason")?;
                if !reason.is_null() && kind != abort_kind(reason.string()?)? {
                    return Err(Error::contract(
                        "simulator abort cause/projected kind mismatch",
                    ));
                }
                if let Some(ack) = lifecycle_ack {
                    let expected = if string(ack, "status")? == "terminal" {
                        "natural"
                    } else if !reason.is_null() {
                        abort_kind(reason.string()?)?
                    } else {
                        "aborted"
                    };
                    if kind != expected {
                        return Err(Error::contract("lifecycle acknowledgement/end mismatch"));
                    }
                    equal(
                        record.require("outcome")?,
                        ack.require("result")?,
                        "lifecycle acknowledgement/end mismatch",
                    )?;
                }
                if kind == "natural" {
                    let outcome = record.require("outcome")?;
                    validation::result(outcome)?;
                    if outcome.is_null() {
                        return Err(Error::contract("missing natural result"));
                    }
                    let final_obs = record.require("final_observation")?;
                    validation::observation_controlled(final_obs, Some(viewer), control)?;
                    if pending.is_some() || string(final_obs, "phase")? != "terminal" {
                        return Err(Error::contract("natural result/observation mismatch"));
                    }
                    equal(
                        final_obs.require("result")?,
                        outcome,
                        "natural result/observation mismatch",
                    )?;
                    record.require("terminal_reward")?.integer(-1, 1)?;
                    // Perspective is intentionally checked by Python from both raw
                    // outcomes/rewards. Rust only checks the persisted ledger agrees.
                    if let Some(last) = ledger.accepted.last() {
                        let terminal = terminal_transition
                            .ok_or_else(|| Error::contract("missing terminal transition"))?;
                        equal(
                            terminal.require("decision")?,
                            last.require("decision")?,
                            "terminal transition/decision mismatch",
                        )?;
                        equal(
                            terminal.require("reward")?,
                            record.require("terminal_reward")?,
                            "terminal transition/reward mismatch",
                        )?;
                    } else if terminal_transition.is_some() {
                        return Err(Error::contract(
                            "terminal transition without accepted action",
                        ));
                    }
                } else if !record.require("terminal_reward")?.is_null()
                    || !record.require("outcome")?.is_null()
                    || terminal_transition.is_some()
                    || !record.require("final_observation")?.is_null()
                {
                    return Err(Error::contract("interrupted journal has terminal label"));
                }
                ledger.end = Some(record);
            }
            _ => return Err(Error::contract("unknown journal record")),
        }
    }
    if terminal_transition.is_some() && ledger.end.is_none() {
        return Err(Error::contract(
            "terminal transition lacks matching natural end",
        ));
    }
    if awaiting_next.is_some() {
        return Err(Error::contract("transition lacks recorded next decision"));
    }
    Ok(ledger)
}
pub fn acknowledgement(response: &Value, command: &Value) -> Result<()> {
    response.closed(
        &[
            "command_id",
            "previous_revision",
            "state_revision",
            "status",
            "result",
        ],
        &[],
    )?;
    equal(
        response.require("command_id")?,
        command.require("command_id")?,
        "acknowledgement command mismatch",
    )?;
    if int(response, "previous_revision")? != int(command, "expected_revision")?
        || int(response, "state_revision")? != int(command, "expected_revision")? + 1
    {
        return Err(Error::contract("acknowledgement revision mismatch"));
    }
    let status = string(response, "status")?;
    if !["active", "terminal", "aborted"].contains(&status) {
        return Err(Error::contract("invalid accepted acknowledgement status"));
    }
    validation::result(response.require("result")?)?;
    if (status == "terminal") != !response.require("result")?.is_null() {
        return Err(Error::contract("acknowledgement outcome/status mismatch"));
    }
    Ok(())
}
fn uuid(value: &str) -> Result<()> {
    if value.len() != 36
        || value.bytes().enumerate().any(|(i, b)| {
            if [8, 13, 18, 23].contains(&i) {
                b != b'-'
            } else {
                !b.is_ascii_digit() && !(b'a'..=b'f').contains(&b)
            }
        })
    {
        return Err(Error::contract("invalid command UUID"));
    }
    Ok(())
}
pub fn abort_kind(reason: &str) -> Result<&'static str> {
    if reason.is_empty() {
        return Err(Error::contract("nonempty abort reason required"));
    }
    Ok(if reason == "turn_limit_exceeded" {
        "truncated"
    } else {
        "aborted"
    })
}
