//! Borrowed record storage is external to the ledger. Only Header/End become
//! small canonical provenance controls; ordinary decisions remain concrete DTOs.
use super::{dto::contract::Validate, dto::*, ensure, nullable, required, validation};
use crate::{
    budget::{Budget, BudgetVec, Control, JsonAdmission},
    error::{Error, Result},
    json::{self, Value, ValueExt},
};
use serde::Deserialize;
use std::sync::Arc;

pub struct Records<'raw> {
    pub records: Vec<Record<'raw>>,
    control_bound: usize,
    _admission: Arc<JsonAdmission>,
}
pub fn parse_admitted<'raw>(
    raw: &'raw [u8],
    max_record: usize,
    admission: &Arc<JsonAdmission>,
    control: &Control,
) -> Result<Records<'raw>> {
    control.check()?;
    std::str::from_utf8(raw).map_err(|_| Error::contract("invalid UTF-8 journal"))?;
    let mut records = Vec::new();
    let mut start = 0;
    let mut control_bound = 0usize;
    while start < raw.len() {
        control.check()?;
        let mut end = start;
        while end < raw.len() && raw[end] != b'\r' && raw[end] != b'\n' {
            end += 1;
        }
        if end - start > max_record {
            return Err(Error::resource("journal exceeds max_record_mib"));
        }
        let bytes = &raw[start..end];
        let mut decoder = serde_json::Deserializer::from_slice(bytes);
        control.check()?;
        let record = Record::deserialize(&mut decoder)
            .map_err(|error| Error::contract(error.to_string()))?;
        decoder
            .end()
            .map_err(|error| Error::contract(error.to_string()))?;
        control.check()?;
        if matches!(record.kind.as_ref(), "header" | "end") {
            control_bound = control_bound
                .checked_add(bytes.len())
                .ok_or_else(|| Error::resource("typed control size overflow"))?;
        }
        records.push(record);
        start = end;
        if raw.get(start) == Some(&b'\r') {
            start += 1;
            if raw.get(start) == Some(&b'\n') {
                start += 1;
            }
        } else if raw.get(start) == Some(&b'\n') {
            start += 1;
        }
    }
    control.check()?;
    Ok(Records {
        records,
        control_bound,
        _admission: admission.clone(),
    })
}
pub struct Journal<'records, 'raw> {
    pub header: Value,
    pub decisions: BudgetVec<&'records Record<'raw>>,
    pub accepted: BudgetVec<&'records Record<'raw>>,
    pub transitions: BudgetVec<&'records Record<'raw>>,
    pub end: Option<Value>,
    pub end_record: Option<&'records Record<'raw>>,
    _controls: Arc<JsonAdmission>,
}
pub fn validate<'records, 'raw>(
    records: &'records Records<'raw>,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<Journal<'records, 'raw>> {
    let first = records
        .records
        .first()
        .ok_or_else(|| Error::contract("missing journal header"))?;
    ensure(first.kind.as_ref() == "header", "missing journal header")?;
    first.validate().map_err(Error::contract)?;
    // Admission precedes either control tree's allocation. Full End includes its
    // terminal Observation because the frozen historical source hash requires it.
    let controls = json::admit(records.control_bound, budget)?;
    let header = serde_json::to_value(first).map_err(|e| Error::contract(e.to_string()))?;
    crate::journal::catalog_meta(header.require("catalog")?)?;
    for (key, expected) in [
        ("feature_version", crate::journal::FEATURE_VERSION),
        ("reward_version", crate::journal::REWARD_VERSION),
        ("policy_input_version", crate::journal::POLICY_INPUT_VERSION),
        (
            "end_projection_version",
            crate::journal::END_PROJECTION_VERSION,
        ),
        ("generator_version", crate::journal::GENERATOR_VERSION),
        ("scorer_version", crate::journal::SCORER_VERSION),
    ] {
        ensure(
            crate::journal::string(&header, key)? == expected,
            "incompatible typed journal versions",
        )?;
    }
    let viewer = required(&first.viewer)?.0;
    let mode = *required(&first.use_action_options)?;
    let mut ledger = Journal {
        header,
        decisions: BudgetVec::new(budget),
        accepted: BudgetVec::new(budget),
        transitions: BudgetVec::new(budget),
        end: None,
        end_record: None,
        _controls: controls,
    };
    let mut pending: Option<usize> = None;
    let mut last_accepted: Option<usize> = None;
    let mut awaiting_next: Option<usize> = None;
    let mut terminal_transition: Option<&Record<'raw>> = None;
    let mut lifecycle_ack: Option<&Acknowledgement<'raw>> = None;
    let mut last_revision = 0u64;
    for record in &records.records[1..] {
        control.check()?;
        ensure(ledger.end.is_none(), "records after journal end")?;
        record.validate().map_err(Error::contract)?;
        match record.kind.as_ref() {
            "decision" => {
                let index = usize::try_from(*required(&record.decision)?)
                    .map_err(|_| Error::contract("decision index overflow"))?;
                ensure(
                    index == ledger.decisions.len()
                        && pending.is_none()
                        && last_accepted.is_none()
                        && terminal_transition.is_none()
                        && lifecycle_ack.is_none(),
                    "invalid decision ordering",
                )?;
                ensure(
                    awaiting_next.is_none_or(|next| next == index),
                    "transition/next decision mismatch",
                )?;
                awaiting_next = None;
                let input = required(&record.input)?;
                validation::input_semantics(input, Some(mode), budget, control)?;
                ensure(
                    input.observation.viewer_player_id.0 == viewer,
                    "mixed participant journal",
                )?;
                let candidates = required(&record.candidates)?;
                ensure(
                    candidates.generator_version == *required(&first.generator_version)?
                        && candidates.scorer_version == *required(&first.scorer_version)?
                        && candidates.config_hash == *required(&first.action_config_hash)?,
                    "incompatible candidate versions/config",
                )?;
                let selection = required(&record.selection)?;
                let selected = usize::try_from(selection.index)
                    .map_err(|_| Error::contract("selected index overflow"))?;
                ensure(
                    selected < candidates.actions.len(),
                    "invalid stored candidate selection",
                )?;
                let command = required(&record.command)?;
                crate::journal::uuid(command.command_id.as_ref())?;
                ensure(
                    command.expected_revision >= last_revision,
                    "participant revision moved backwards",
                )?;
                ensure(
                    command.expected_revision == input.observation.state_revision,
                    "command/input revision mismatch",
                )?;
                ensure(
                    command.action == candidates.actions[selected],
                    "command/candidate mismatch",
                )?;
                ledger.decisions.push(record)?;
                pending = Some(index);
            }
            "acknowledgement" => {
                let index = usize::try_from(*required(&record.decision)?)
                    .map_err(|_| Error::contract("decision index overflow"))?;
                ensure(
                    pending == Some(index),
                    "acknowledgement without unique pending decision",
                )?;
                let response = required(&record.response)?;
                let command = required(&ledger.decisions[index].command)?;
                ensure(
                    response.command_id == command.command_id,
                    "acknowledgement command mismatch",
                )?;
                ensure(
                    response.previous_revision == command.expected_revision
                        && command.expected_revision.checked_add(1)
                            == Some(response.state_revision),
                    "acknowledgement revision mismatch",
                )?;
                ensure(
                    (response.status.as_ref() == "terminal") == response.result.is_some(),
                    "acknowledgement outcome/status mismatch",
                )?;
                if response.status.as_ref() != "active" {
                    lifecycle_ack = Some(response);
                }
                last_revision = response.state_revision;
                ledger.accepted.push(ledger.decisions[index])?;
                pending = None;
                last_accepted = Some(index);
            }
            "transition" => {
                let index = usize::try_from(*required(&record.decision)?)
                    .map_err(|_| Error::contract("decision index overflow"))?;
                ensure(
                    last_accepted == Some(index) && terminal_transition.is_none(),
                    "transition lacks last accepted action",
                )?;
                if *required(&record.terminal)? {
                    ensure(
                        (-1..=1).contains(required(&record.reward)?),
                        "terminal reward range",
                    )?;
                    ensure(
                        nullable(&record.next_decision)?.is_none(),
                        "terminal transition has next decision",
                    )?;
                    terminal_transition = Some(record);
                } else {
                    ensure(
                        lifecycle_ack.is_none(),
                        "continuing transition after lifecycle acknowledgement",
                    )?;
                    ensure(
                        *required(&record.reward)? == 0
                            && *required(&record.next_decision)? == ledger.decisions.len() as u64,
                        "invalid continuing transition",
                    )?;
                    awaiting_next = Some(ledger.decisions.len());
                }
                ledger.transitions.push(record)?;
                last_accepted = None;
            }
            "end" => {
                ensure(awaiting_next.is_none(), "invalid journal end/order")?;
                let kind = required(&record.end_kind)?.as_ref();
                let pending_values = required(&record.pending_decisions)?;
                ensure(
                    match pending {
                        Some(index) => pending_values.as_slice() == [index as u64],
                        None => pending_values.is_empty(),
                    },
                    "pending request ledger mismatch",
                )?;
                let reason = nullable(&record.simulator_abort_reason)?.map(AsRef::as_ref);
                if let Some(reason) = reason {
                    ensure(
                        kind == crate::journal::abort_kind(reason)?,
                        "simulator abort cause/projected kind mismatch",
                    )?;
                }
                let outcome = nullable(&record.outcome)?;
                if let Some(ack) = lifecycle_ack {
                    let expected = if ack.status.as_ref() == "terminal" {
                        "natural"
                    } else if let Some(reason) = reason {
                        crate::journal::abort_kind(reason)?
                    } else {
                        "aborted"
                    };
                    ensure(
                        kind == expected && outcome == ack.result.as_ref(),
                        "lifecycle acknowledgement/end mismatch",
                    )?;
                }
                if kind == "natural" {
                    let outcome =
                        outcome.ok_or_else(|| Error::contract("missing natural result"))?;
                    let final_obs = required(&record.final_observation)?;
                    validation::observation_semantics(final_obs, Some(viewer), budget, control)?;
                    ensure(
                        pending.is_none()
                            && final_obs.phase.as_ref() == "terminal"
                            && final_obs.result.as_ref() == Some(outcome),
                        "natural result/observation mismatch",
                    )?;
                    let reward = *required(&record.terminal_reward)?;
                    ensure((-1..=1).contains(&reward), "terminal reward range")?;
                    if let Some(last) = ledger.accepted.last() {
                        let terminal = terminal_transition
                            .ok_or_else(|| Error::contract("missing terminal transition"))?;
                        ensure(
                            required(&terminal.decision)? == required(&last.decision)?
                                && *required(&terminal.reward)? == reward,
                            "terminal transition/decision/reward mismatch",
                        )?;
                    } else {
                        ensure(
                            terminal_transition.is_none(),
                            "terminal transition without accepted action",
                        )?;
                    }
                } else {
                    ensure(
                        nullable(&record.terminal_reward)?.is_none()
                            && outcome.is_none()
                            && terminal_transition.is_none()
                            && nullable(&record.final_observation)?.is_none(),
                        "interrupted journal has terminal label",
                    )?;
                }
                // Shape validation above excludes every ordinary decision field.
                ledger.end =
                    Some(serde_json::to_value(record).map_err(|e| Error::contract(e.to_string()))?);
                ledger.end_record = Some(record);
            }
            _ => return Err(Error::contract("unknown journal record")),
        }
    }
    ensure(
        terminal_transition.is_none() || ledger.end.is_some(),
        "terminal transition lacks matching natural end",
    )?;
    ensure(
        awaiting_next.is_none(),
        "transition lacks recorded next decision",
    )?;
    Ok(ledger)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::budget::monotonic;
    pub(crate) fn fixture() -> Vec<Value> {
        let input = super::super::validation::tests::fixture();
        let revision = input["observation"]["state_revision"].as_u64().unwrap();
        let mut final_obs = input["observation"].clone();
        for (key, value) in [
            ("phase", serde_json::json!("terminal")),
            ("current_player", Value::Null),
            ("required_choice", Value::Null),
            ("actionable", serde_json::json!(false)),
            ("typed_action_schema", serde_json::json!([])),
            ("result", serde_json::json!("draw")),
            ("state_revision", serde_json::json!(revision + 1)),
        ] {
            final_obs[key] = value;
        }
        let header = serde_json::json!({"kind":"header","schema_version":4,"feature_version":crate::journal::FEATURE_VERSION,
            "reward_version":crate::journal::REWARD_VERSION,"policy_input_version":crate::journal::POLICY_INPUT_VERSION,
            "use_action_options":false,"end_projection_version":crate::journal::END_PROJECTION_VERSION,
            "catalog":{"catalog_schema_version":1,"catalog_version":"public-ja-v2","rule_set_version":"true-match-1v1-v1","locale":"ja","content_sha256":crate::journal::CATALOG_HASH,"definition_count":296},
            "episode_id":"e","viewer":1,"policy":"p","run_id":"r","game_id":"g","action_config_hash":"h",
            "generator_version":crate::journal::GENERATOR_VERSION,"scorer_version":crate::journal::SCORER_VERSION});
        let decision = serde_json::json!({"kind":"decision","decision":0,"input":input,
            "candidates":{"actions":[{"type":"pray"}],"generator_version":crate::journal::GENERATOR_VERSION,
                "scorer_version":crate::journal::SCORER_VERSION,"config_hash":"h","attempts":1,"public_prunes":0,
                "approximate_drops":0,"latency_ms":1.,"cutoff":false,"cutoff_reasons":[]},
            "selection":{"index":0,"policy_version":"p","logprob":-0.1,"value":0.5},
            "command":{"action":{"type":"pray"},"command_id":"00000000-0000-0000-0000-000000000001","expected_revision":revision},
            "selection_elapsed_ms":1.});
        let ack = serde_json::json!({"kind":"acknowledgement","decision":0,"response":{
            "command_id":"00000000-0000-0000-0000-000000000001","previous_revision":revision,
            "state_revision":revision+1,"status":"terminal","result":"draw"}});
        let transition = serde_json::json!({"kind":"transition","decision":0,"next_decision":null,"reward":0,"terminal":true});
        let end = serde_json::json!({"kind":"end","end_kind":"natural","outcome":"draw","terminal_reward":0,
            "final_observation":final_obs,"pending_decisions":[],"simulator_abort_reason":null});
        vec![header, decision, ack, transition, end]
    }
    pub(crate) fn wire(values: &[Value]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for value in values {
            serde_json::to_writer(&mut bytes, value).unwrap();
            bytes.push(b'\n');
        }
        bytes
    }
    fn accepted(values: &[Value]) -> (bool, bool) {
        let budget = Budget::new(128 * 1024 * 1024);
        let control = Control::new(monotonic() + 10.).unwrap();
        let bytes = wire(values);
        let admission = json::admit(bytes.len(), &budget).unwrap();
        let typed = parse_admitted(&bytes, 8 * 1024 * 1024, &admission, &control)
            .and_then(|records| validate(&records, &budget, &control).map(|_| ()));
        let old: Vec<_> = values
            .iter()
            .map(|value| {
                json::parse_admitted(&serde_json::to_vec(value).unwrap(), &admission, &control)
                    .unwrap()
            })
            .collect();
        let legacy = crate::journal::validate(&old, &budget, &control);
        (typed.is_ok(), legacy.is_ok())
    }
    #[test]
    fn ledger_matches_legacy_ack_transition_end_and_control_provenance() {
        let values = fixture();
        assert_eq!(accepted(&values), (true, true));
        let budget = Budget::new(128 * 1024 * 1024);
        let control = Control::new(monotonic() + 10.).unwrap();
        let bytes = wire(&values);
        let admission = json::admit(bytes.len(), &budget).unwrap();
        let records = parse_admitted(&bytes, 8 * 1024 * 1024, &admission, &control).unwrap();
        let journal = validate(&records, &budget, &control).unwrap();
        assert_eq!(journal.header, values[0]);
        assert_eq!(journal.end.as_ref(), values.last());
        assert_eq!(
            (
                journal.decisions.len(),
                journal.accepted.len(),
                journal.transitions.len()
            ),
            (1, 1, 1)
        );
        drop(journal);
        drop(records);
        drop(admission);
        assert_eq!(budget.used(), 0);
        let mut cases = Vec::new();
        let mut value = values.clone();
        value[2]["response"]["command_id"] = serde_json::json!("wrong");
        cases.push(value);
        let mut value = values.clone();
        value[2]["response"]["state_revision"] = serde_json::json!(0);
        cases.push(value);
        let mut value = values.clone();
        value[2]["response"]["status"] = serde_json::json!("active");
        cases.push(value);
        let mut value = values.clone();
        value[3]["reward"] = serde_json::json!(1);
        cases.push(value);
        let mut value = values.clone();
        value.remove(3);
        cases.push(value);
        let mut value = values.clone();
        value.remove(2);
        cases.push(value);
        let mut value = values.clone();
        value[4]["pending_decisions"] = serde_json::json!([0]);
        cases.push(value);
        let mut value = values.clone();
        value[4]["final_observation"]["viewer_player_id"] = serde_json::json!(2);
        cases.push(value);
        let mut value = values.clone();
        value[4]["outcome"] = serde_json::json!({"win":{"winner":1}});
        cases.push(value);
        let mut value = values.clone();
        value.push(values[1].clone());
        cases.push(value);
        for value in cases {
            assert_eq!(accepted(&value), (false, false));
        }
    }
    #[test]
    fn common_record_checks_preserve_reserved_keys_and_framing() {
        let values = fixture();
        let bytes = wire(&values);
        let budget = Budget::new(128 * 1024 * 1024);
        let control = Control::new(monotonic() + 10.).unwrap();
        let admission = json::admit(bytes.len(), &budget).unwrap();
        assert!(parse_admitted(&bytes, 1, &admission, &control).is_err());
        let reserved = br#"{"$serde_json::private::Number":"1"}"#;
        assert!(parse_admitted(reserved, 1024, &admission, &control).is_err());
        let crlf = String::from_utf8(bytes)
            .unwrap()
            .replace('\n', "\r\n")
            .into_bytes();
        let records = parse_admitted(&crlf, 8 * 1024 * 1024, &admission, &control).unwrap();
        validate(&records, &budget, &control).unwrap();
    }

    // These explicit outcomes were checked against the old production decoder
    // (reserved-member scan plus the unmodified DTOs) before removing its scan.
    // They remain an independent acceptance oracle, rather than deriving the
    // expected result from the new DTO decoder with the old scan prepended.
    fn assert_wire_acceptance(raw: &[u8], expected: bool) {
        let budget = Budget::new(128 * 1024 * 1024);
        let control = Control::new(monotonic() + 10.).unwrap();
        let admission = json::admit(raw.len(), &budget).unwrap();
        let result = parse_admitted(raw, 8 * 1024 * 1024, &admission, &control);
        assert_eq!(result.is_ok(), expected, "{}", String::from_utf8_lossy(raw));
        drop(result);
        drop(admission);
        assert_eq!(budget.used(), 0);
    }

    fn schema_record(schema: &str) -> String {
        let mut input = fixture()[1]["input"].clone();
        input["typed_metadata"] = serde_json::json!([]);
        let input = serde_json::to_string(&input).unwrap().replace(
            "\"typed_metadata\":[]",
            &format!("\"typed_metadata\":[{schema}]"),
        );
        format!(r#"{{"kind":"decision","input":{input}}}"#)
    }

    #[test]
    fn frozen_record_wire_acceptance_covers_fixed_custom_and_sequence_forms() {
        for (wire, expected) in [
            (r#"{"kind":"x"}"#, true),
            (r#"{"kind":"$serde_json::private::Number"}"#, true),
            (r#"{"kind":"\u0024serde_json::private::Number"}"#, true),
            (r#"{"kind":"\ud83d\ude00"}"#, true),
            (r#"{"kind":"\ud800"}"#, false),
            (r#"{"kind":"\udc00"}"#, false),
            (r#"{"kind":"x","$serde_json::private::Number":"1"}"#, false),
            (
                r#"{"kind":"x","\u0024serde_json::private::Number":"1"}"#,
                false,
            ),
            (
                r#"{"kind":"x","catalog":{"$serde_json::private::Number":"1"}}"#,
                false,
            ),
            (r#"{"kind":"x","kind":"x"}"#, false),
            (r#"{"kind":"x","\u006bind":"x"}"#, false),
            (r#"{"kind":"x","decision":18446744073709551615}"#, true),
            (r#"{"kind":"x","decision":18446744073709551616}"#, false),
            (r#"{"kind":"x","decision":0.0}"#, false),
            (r#"{"kind":"x","decision":-1}"#, false),
            (r#"{"kind":"x","selection_elapsed_ms":1e999}"#, false),
            (r#"{"kind":"x","outcome":"draw"}"#, true),
            (r#"{"kind":"x","outcome":{"win":{"winner":1}}}"#, true),
            (r#"{"kind":"x","outcome":{"win":[2]}}"#, true),
            (
                r#"{"kind":"x","outcome":{"$serde_json::private::Number":"1"}}"#,
                false,
            ),
            (
                r#"{"kind":"x","outcome":{"win":{"winner":1},"\u0024serde_json::private::Number":"1"}}"#,
                false,
            ),
            (
                r#"{"kind":"x","outcome":{"win":{"winner":1,"$serde_json::private::Number":"1"}}}"#,
                false,
            ),
            (r#"{"kind":"x","outcome":18446744073709551616}"#, false),
            (r#"{"kind":"x","outcome":9007199254740993.0}"#, false),
            (r#"{"kind":"x","outcome":1e999}"#, false),
            (
                r#"{"kind":"x","command":{"command_id":"c","expected_revision":0,"action":{"type":"respond","ordered_defenses":[18446744073709551615]}}}"#,
                true,
            ),
            (
                r#"{"kind":"x","command":{"command_id":"c","expected_revision":0,"action":{"type":"respond_sequence","ordered_defenses":[{"source":"hand_artifact","instance_id":1}]}}}"#,
                true,
            ),
            (
                r#"{"kind":"x","command":{"command_id":"c","expected_revision":0,"action":{"type":"respond","ordered_defenses":[{"$serde_json::private::Number":"1"}]}}}"#,
                false,
            ),
            (
                r#"{"kind":"x","command":{"command_id":"c","expected_revision":0,"action":{"type":"respond","ordered_defenses":[{"source":"hand_artifact","instance_id":1,"\u0024serde_json::private::Number":"1"}]}}}"#,
                false,
            ),
            (
                r#"{"kind":"x","command":{"command_id":"c","expected_revision":0,"action":{"type":"respond","ordered_defenses":[18446744073709551616]}}}"#,
                false,
            ),
            (
                r#"{"kind":"x","command":{"command_id":"c","expected_revision":0,"action":{"type":"respond","ordered_defenses":[9007199254740993.0]}}}"#,
                false,
            ),
            (
                r#"{"kind":"x","command":{"command_id":"c","expected_revision":0,"action":{"type":"respond","ordered_defenses":[1e999]}}}"#,
                false,
            ),
            (
                r#"{"kind":"x","selection":[0,"$serde_json::private::Number",null,null]}"#,
                true,
            ),
            (r#"{"kind":"x","selection":[0,"p",null,null,0]}"#, false),
            (r#"["x"]"#, true),
            (r#"{"kind":"x"} {}"#, false),
            (r#"{"kind":"x"}\n"#, false),
        ] {
            assert_wire_acceptance(wire.as_bytes(), expected);
        }
        assert_wire_acceptance(b"{\"kind\":\"\xff\"}", false);
        for ending in ["", "\n", "\r", "\r\n"] {
            assert_wire_acceptance(format!(r#"{{"kind":"x"}}{ending}"#).as_bytes(), true);
        }
        assert_wire_acceptance(b"{\"kind\":\"x\"}\n\n", false);
    }

    #[test]
    fn frozen_dynamic_map_acceptance_covers_recursive_escaped_and_nonfirst_keys() {
        for (schema, expected) in [
            (r#"{"const":"$serde_json::private::Number"}"#, true),
            (r#"{"required_fields":{"x":{}},"optional_fields":{}}"#, true),
            (
                r#"{"required_fields":{"$serde_json::private::Number":{}}}"#,
                false,
            ),
            (
                r#"{"required_fields":{"\u0024serde_json::private::Number":{}}}"#,
                false,
            ),
            (
                r#"{"required_fields":{"x":{},"$serde_json::private::Numbe\u0072":{}}}"#,
                false,
            ),
            (
                r#"{"optional_fields":{"$serde_json::private::Number":{}}}"#,
                false,
            ),
            (
                r#"{"items":{"one_of":[{"optional_fields":{"x":{"required_fields":{"$serde_json::private::Number":{}}}}}]}}"#,
                false,
            ),
            (r#"{"required_fields":{"x":{},"\u0078":{}}}"#, false),
            (r#"{"required_fields":{"\ud800":{}}}"#, false),
            (
                r#"{"minimum":-170141183460469231731687303715884105728,"maximum":170141183460469231731687303715884105727}"#,
                true,
            ),
            (
                r#"{"minimum":-170141183460469231731687303715884105729}"#,
                false,
            ),
            (
                r#"{"maximum":170141183460469231731687303715884105728}"#,
                false,
            ),
        ] {
            assert_wire_acceptance(schema_record(schema).as_bytes(), expected);
        }
        let input = fixture()[1]["input"].clone();
        let definition = r#"{"definition_id":"d","display_name":"$serde_json::private::Number","kind":"weapon"}"#;
        for (keys, expected) in [
            (format!(r#""ordinary":{definition}"#), true),
            (
                format!(r#""$serde_json::private::Number":{definition}"#),
                false,
            ),
            (
                format!(r#""\u0024serde_json::private::Number":{definition}"#),
                false,
            ),
            (
                format!(r#""x":{definition},"$serde_json::private::Number":{definition}"#),
                false,
            ),
            (format!(r#""x":{definition},"\u0078":{definition}"#), false),
        ] {
            let mut input = input.clone();
            input["catalog"] = serde_json::json!({});
            let object = serde_json::to_string(&input)
                .unwrap()
                .replace("\"catalog\":{}", &format!("\"catalog\":{{{keys}}}"));
            assert_wire_acceptance(
                format!(r#"{{"kind":"x","input":{object}}}"#).as_bytes(),
                expected,
            );
            let sequence = format!(
                "[{},{},{},{},{{{keys}}},{}]",
                input["observation"],
                input["events"],
                input["history_end"],
                input["typed_metadata"],
                input["action_options"]
            );
            assert_wire_acceptance(
                format!(r#"{{"kind":"x","input":{sequence}}}"#).as_bytes(),
                expected,
            );
        }
    }

    #[test]
    fn frozen_typed_recursion_boundary_preserves_standard_serde_limit() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                // Record, Input, typed_metadata array and root Schema use four levels.
                for depth in [122, 123, 124, 125] {
                    let schema =
                        format!("{}{{}}{}", "{\"items\":".repeat(depth), "}".repeat(depth));
                    assert_wire_acceptance(schema_record(&schema).as_bytes(), depth < 124);
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
