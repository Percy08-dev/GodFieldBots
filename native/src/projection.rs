//! Source linkage and exact legacy source-proof serialization, without policy I/O.
use crate::json::ValueExt;
use crate::{
    budget::{Budget, BudgetVec, Control},
    controls,
    error::{Error, Result},
    journal::{self, Journal, equal, string},
    json::{self, Value},
    session::{Collection, path_text},
    source::Sources,
    validation,
};
use std::{path::Path, sync::Arc};

pub const SIMULATOR_REVISION: &str = "ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6";
const REVIEWED_SPECS_DIGEST: &str =
    "17dc86ba14c8fac378e352acd5c7c55429e910b37482954f74ca3673f8b1767d";

fn optional(value: &Value, key: &str, control: &Control) -> Result<Value> {
    match value.get(key) {
        Some(v) => json::copy(v, control),
        None => Ok(Value::Null),
    }
}

pub fn validate_manifest(
    collection: &Collection,
    budget: &Arc<Budget>,
    control: &Control,
    sources: &Sources,
) -> Result<()> {
    let m = &collection.manifest.root;
    let report = &collection.report.root;
    if !controls::numeric_field(m, "journal_schema_version", 4)
        || string(m, "feature_version")? != journal::FEATURE_VERSION
        || string(m, "reward_policy_version")? != journal::REWARD_VERSION
        || string(m, "end_projection_version")? != journal::END_PROJECTION_VERSION
    {
        return Err(Error::contract(
            "journal/manifest semantic versions mismatch",
        ));
    }
    journal::catalog_meta(m.require("catalog")?)?;
    if string(m, "simulator_contract_revision")? != SIMULATOR_REVISION
        || string(m, "observation_schema_version")? != "1"
        || m.require("action_schema_version")?.integer(1, 1)? != 1
        || string(m, "policy_input_version")? != journal::POLICY_INPUT_VERSION
        || controls::digest(
            m.require("reviewed_simulator_spec_sha256")?,
            budget,
            control,
        )? != REVIEWED_SPECS_DIGEST
    {
        return Err(Error::contract(
            "incompatible source simulator semantics/spec snapshots",
        ));
    }
    let config = m.require("effective_config")?;
    if string(config.require("actions")?, "scorer")? != "public_heuristic_v1" {
        return Err(Error::contract("source search scorer mismatch"));
    }
    if sources.digest(&collection.directory.join("report.json"), control)?
        != string(m, "report_sha256")?
    {
        return Err(Error::contract("source report hash mismatch"));
    }
    report.require("schema_version")?.integer(1, 1)?;
    if !matches!(report.get("comparison_valid"), Some(Value::Bool(true))) {
        return Err(Error::contract("source report not successful"));
    }
    for key in ["failures", "health_failures", "group_health_failures"] {
        if !report.require(key)?.object()?.is_empty() {
            return Err(Error::contract("source report not successful"));
        }
    }
    if controls::field_truthy(report, "budget_overrun_seconds")
        || report
            .get("report_status")
            .is_some_and(|v| v.string().is_ok_and(|s| s == "minimal"))
    {
        return Err(Error::contract("source report not successful"));
    }
    for key in [
        "finalization_failure",
        "budget_overrun_seconds",
        "correction_save_failures",
    ] {
        if controls::field_truthy(m, key) {
            return Err(Error::contract("run finalization failure"));
        }
    }
    Ok(())
}

pub struct EpisodeLinks<'a> {
    pub episode: &'a Value,
    pub group: &'a Value,
    pub spec: &'a Value,
    pub health: &'a Value,
}
pub fn links<'a>(manifest: &'a Value, id: &str) -> Result<EpisodeLinks<'a>> {
    let mut episode = None;
    for item in manifest.require("episodes")?.array()? {
        if string(item, "episode_id")? == id {
            if episode.replace(item).is_some() {
                return Err(Error::contract("duplicate manifest episode"));
            }
        }
    }
    let episode =
        episode.ok_or_else(|| Error::contract("journal episode missing from manifest"))?;
    let mut found = None;
    for group in manifest.require("schedule")?.array()? {
        for spec in group.require("episodes")?.array()? {
            if string(spec, "episode_id")? == id {
                if found.replace((group, spec)).is_some() {
                    return Err(Error::contract("duplicate scheduled identity"));
                }
            }
        }
    }
    let (group, spec) = found.ok_or_else(|| Error::contract("missing scheduled identity"))?;
    for (left, right) in [("pair_id", "pair_id"), ("policies", "policies")] {
        equal(
            episode.require(left)?,
            spec.require(right)?,
            "source schedule/result mismatch",
        )?;
    }
    equal(
        episode.require("seed")?,
        group.require("seed")?,
        "source schedule/result mismatch",
    )?;
    let mut health = None;
    for candidate in manifest.require("groups")?.array()? {
        if candidate
            .require("episode_ids")?
            .array()?
            .iter()
            .any(|v| v.string().is_ok_and(|s| s == id))
        {
            if health.replace(candidate).is_some() {
                return Err(Error::contract("duplicate group health"));
            }
        }
    }
    let health = health.ok_or_else(|| Error::contract("missing group health"))?;
    for key in [
        "failure",
        "client_close_failure",
        "budget_overrun_seconds",
        "finalization_cancelled",
        "cleanup_failure",
    ] {
        if controls::field_truthy(health, key) {
            return Err(Error::contract("episode or group health failure"));
        }
    }
    if let Some(process) = health.get("process").filter(|v| !v.is_null()) {
        if !controls::numeric_field(process, "exit_code", 0)
            || string(process, "shutdown")? != "stdin_eof"
        {
            return Err(Error::contract("episode or group process health failure"));
        }
        for key in ["log_error", "reap_failure", "budget_overrun_seconds"] {
            if controls::field_truthy(process, key) {
                return Err(Error::contract("episode or group process health failure"));
            }
        }
    }
    Ok(EpisodeLinks {
        episode,
        group,
        spec,
        health,
    })
}

pub fn source_metadata(
    collection: &Collection,
    source: &Path,
    replay: &Value,
    journal: &Journal<'_>,
    links: &EpisodeLinks<'_>,
    sources: &Sources,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<Value> {
    let header = journal.header;
    let m = &collection.manifest.root;
    let viewer = header.require("viewer")?.integer(1, 2)?;
    equal(
        header.require("episode_id")?,
        links.episode.require("episode_id")?,
        "journal/manifest episode identity mismatch",
    )?;
    equal(
        header.require("episode_id")?,
        links.spec.require("episode_id")?,
        "journal/schedule episode identity mismatch",
    )?;
    if source
        .parent()
        .and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
        != Some(string(header, "episode_id")?)
    {
        return Err(Error::contract(
            "journal/directory episode identity mismatch",
        ));
    }
    equal(
        header.require("run_id")?,
        m.require("run_id")?,
        "journal/manifest run identity mismatch",
    )?;
    equal(
        header.require("game_id")?,
        links.episode.require("game_id")?,
        "journal/manifest game identity mismatch",
    )?;
    equal(
        header.require("policy")?,
        &links.episode.require("policies")?.array()?[(viewer - 1) as usize],
        "source game/model linkage mismatch",
    )?;
    let configured_mode = m
        .require("effective_config")?
        .require("actions")?
        .require("use_action_options")?
        .boolean()?;
    if header.require("use_action_options")?.boolean()? != configured_mode {
        return Err(Error::contract("source ActionOptions mode mismatch"));
    }
    if controls::digest(
        m.require("effective_config")?.require("actions")?,
        budget,
        control,
    )? != string(header, "action_config_hash")?
    {
        return Err(Error::contract("source configuration hash mismatch"));
    }
    let learner = links.spec.require("learner_viewer")?.integer(1, 2)?;
    if viewer == learner {
        let expected = format!(
            "mlp-ppo-v1-{}",
            string(header, "policy")?
                .strip_prefix("mlp-")
                .unwrap_or(string(header, "policy")?)
        );
        for decision in journal.decisions.iter() {
            control.check()?;
            if string(decision.require("selection")?, "policy_version")? != expected {
                return Err(Error::contract(
                    "PPO learner selection differs from frozen inference identity",
                ));
            }
        }
    }
    let end = journal
        .end
        .ok_or_else(|| Error::contract("missing natural journal end"))?;
    if string(end, "end_kind")? != "natural"
        || journal.accepted.len() != journal.decisions.len()
        || !end.require("pending_decisions")?.array()?.is_empty()
    {
        return Err(Error::contract(
            "ineligible interrupted or pending source journal",
        ));
    }
    equal(
        end.require("end_kind")?,
        links.episode.require("end_kind")?,
        "journal/manifest end projection mismatch",
    )?;
    equal(
        end.require("simulator_abort_reason")?,
        links.episode.require("simulator_abort_reason")?,
        "journal/manifest abort projection mismatch",
    )?;
    equal(
        end.require("outcome")?,
        links.episode.require("outcome")?,
        "journal/manifest terminal outcome mismatch",
    )?;
    equal(
        replay.require("game_id")?,
        header.require("game_id")?,
        "replay game identity mismatch",
    )?;
    let mut accepted = BudgetVec::with_capacity(budget, journal.accepted.len())?;
    for decision in journal.accepted.iter() {
        accepted.push(decision.require("command")?.require("action")?)?;
    }
    let _replay_timer = sources
        .telemetry()
        .phase_timer(crate::telemetry::Phase::ReplayValidation);
    validation::reconcile_replay_controlled(
        replay,
        viewer,
        &accepted,
        links.episode.require("outcome")?,
        control,
    )?;
    let closure = validation::replay_closure(replay, string(header, "game_id")?, viewer)?;
    if closure.is_some_and(|c| {
        c.get("type")
            .is_some_and(|v| v.string().is_ok_and(|s| s == "aborted"))
    }) {
        return Err(Error::contract("natural source has aborted replay closure"));
    }
    drop(_replay_timer);
    let _proof_timer = sources
        .telemetry()
        .phase_timer(crate::telemetry::Phase::ProofStaging);
    let report = &collection.report.root;
    let report_path = collection.directory.join("report.json");
    let report_diagnostic = json::object([
        ("path", json::text(path_text(&report_path)?)),
        (
            "expected_sha256",
            json::copy(m.require("report_sha256")?, control)?,
        ),
        ("sha256", json::text(sources.digest(&report_path, control)?)),
        (
            "comparison_valid",
            json::copy(report.require("comparison_valid")?, control)?,
        ),
        (
            "failures",
            json::copy(report.require("failures")?, control)?,
        ),
        (
            "health_failures",
            json::copy(report.require("health_failures")?, control)?,
        ),
        (
            "group_health_failures",
            json::copy(report.require("group_health_failures")?, control)?,
        ),
    ]);
    let lineage = json::object([
        ("run_id", json::copy(header.require("run_id")?, control)?),
        ("game_id", json::copy(header.require("game_id")?, control)?),
        (
            "episode_id",
            json::copy(header.require("episode_id")?, control)?,
        ),
        ("seed", json::copy(links.group.require("seed")?, control)?),
        (
            "pair_id",
            json::copy(links.spec.require("pair_id")?, control)?,
        ),
        ("ancestry", optional(links.group, "ancestry", control)?),
        ("split", json::copy(m.require("split")?, control)?),
        (
            "learner_viewer",
            json::copy(links.spec.require("learner_viewer")?, control)?,
        ),
    ]);
    let replay_path = source
        .parent()
        .ok_or_else(|| Error::contract("source parent missing"))?
        .join(format!("replay-player-{viewer}.json"));
    Ok(json::object([
        (
            "path",
            json::text(path_text(&std::fs::canonicalize(source)?)?),
        ),
        ("sha256", json::text(sources.digest(source, control)?)),
        (
            "manifest_sha256",
            json::text(sources.digest(&collection.directory.join("manifest.json"), control)?),
        ),
        ("header", json::copy(header, control)?),
        ("lineage", lineage),
        (
            "decision_count",
            json::integer(journal.decisions.len() as i128),
        ),
        ("end", json::copy(end, control)?),
        ("episode_result", json::copy(links.episode, control)?),
        ("group_health", json::copy(links.health, control)?),
        (
            "replay_sha256",
            json::text(sources.digest(&replay_path, control)?),
        ),
        ("report", report_diagnostic),
        ("eligible", Value::Bool(true)),
        ("ineligible_reasons", Value::Array(Vec::new())),
    ]))
}
