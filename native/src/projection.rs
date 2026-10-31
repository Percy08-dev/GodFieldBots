//! Source linkage and exact legacy source-proof serialization, without policy I/O.
use crate::json::ValueExt;
use crate::{
    budget::{Budget, Control, Lease},
    controls,
    error::{Error, Result},
    journal::{self, equal, string},
    json::{self, Value},
    session::{Collection, path_text},
    source::Sources,
    validation,
};
use std::{collections::HashMap, path::Path, sync::Arc};

pub const SIMULATOR_REVISION: &str = "ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6";
const REVIEWED_SPECS_DIGEST: &str =
    "17dc86ba14c8fac378e352acd5c7c55429e910b37482954f74ca3673f8b1767d";

fn ppo_selection_version(label: &str) -> Result<String> {
    let (identity, checksum) = ["mlp", "gru", "transformer"]
        .iter()
        .find_map(|identity| {
            label
                .strip_prefix(&format!("{identity}-"))
                .map(|checksum| (*identity, checksum))
        })
        .filter(|(_, checksum)| !checksum.is_empty())
        .ok_or_else(|| Error::contract("unsupported frozen PPO policy label"))?;
    Ok(format!("{identity}-ppo-v1-{checksum}"))
}

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
    pub ordinal: usize,
    pub episode: &'a Value,
    pub group: &'a Value,
    pub spec: &'a Value,
    pub health: &'a Value,
}
/// Local borrowed indexes: never stored self-referentially in Collection.
/// Hash iteration order is never used for output or episode consumption.
pub struct EpisodeIndex<'a> {
    episodes: HashMap<&'a str, (usize, &'a Value)>,
    schedule: HashMap<&'a str, (&'a Value, &'a Value)>,
    health: HashMap<&'a str, &'a Value>,
    _lease: Lease,
}
impl<'a> EpisodeIndex<'a> {
    pub fn new(manifest: &'a Value, budget: &Arc<Budget>, control: &Control) -> Result<Self> {
        control.check()?;
        let episodes = manifest.require("episodes")?.array()?;
        let schedule = manifest.require("schedule")?.array()?;
        let groups = manifest.require("groups")?.array()?;
        let mut scheduled = 0usize;
        let mut memberships = 0usize;
        for group in schedule {
            control.check()?;
            scheduled = scheduled
                .checked_add(group.require("episodes")?.array()?.len())
                .ok_or_else(|| Error::resource("episode index capacity overflow"))?;
        }
        for group in groups {
            control.check()?;
            memberships = memberships
                .checked_add(group.require("episode_ids")?.array()?.len())
                .ok_or_else(|| Error::resource("episode index capacity overflow"))?;
        }
        let bytes = episodes
            .len()
            .checked_add(scheduled)
            .and_then(|n| n.checked_add(memberships))
            .and_then(|n| n.checked_mul(160))
            .and_then(|n| n.checked_add(1024))
            .ok_or_else(|| Error::resource("episode index capacity overflow"))?;
        let lease = budget.reserve(bytes)?;
        let mut result = Self {
            episodes: HashMap::new(),
            schedule: HashMap::new(),
            health: HashMap::new(),
            _lease: lease,
        };
        result
            .episodes
            .try_reserve(episodes.len())
            .map_err(|_| Error::resource("episode index allocation failed"))?;
        result
            .schedule
            .try_reserve(scheduled)
            .map_err(|_| Error::resource("episode index allocation failed"))?;
        result
            .health
            .try_reserve(memberships)
            .map_err(|_| Error::resource("episode index allocation failed"))?;
        for (ordinal, episode) in episodes.iter().enumerate() {
            control.check()?;
            let id = string(episode, "episode_id")?;
            if result.episodes.insert(id, (ordinal, episode)).is_some() {
                return Err(Error::contract("duplicate manifest episode"));
            }
        }
        for group in schedule {
            control.check()?;
            for spec in group.require("episodes")?.array()? {
                control.check()?;
                if result
                    .schedule
                    .insert(string(spec, "episode_id")?, (group, spec))
                    .is_some()
                {
                    return Err(Error::contract("duplicate scheduled identity"));
                }
            }
        }
        for group in groups {
            control.check()?;
            for id in group.require("episode_ids")?.array()? {
                control.check()?;
                // Preserve the old any(string().is_ok_and(...)) membership:
                // nonstrings are ignored and repeats within one group count once.
                if let Some(id) = id.as_str() {
                    // The prior lookup only queried actual manifest episodes;
                    // unrelated health memberships cannot introduce conflicts.
                    if !result.episodes.contains_key(id) {
                        continue;
                    }
                    if let Some(previous) = result.health.insert(id, group) {
                        if !std::ptr::eq(previous, group) {
                            return Err(Error::contract("duplicate group health"));
                        }
                    }
                }
            }
        }
        control.check()?;
        Ok(result)
    }
    pub fn scheduled_count(&self) -> usize {
        self.schedule.len()
    }
    pub fn links(&self, id: &str, control: &Control) -> Result<EpisodeLinks<'a>> {
        control.check()?;
        let &(ordinal, episode) = self
            .episodes
            .get(id)
            .ok_or_else(|| Error::contract("journal episode missing from manifest"))?;
        let &(group, spec) = self
            .schedule
            .get(id)
            .ok_or_else(|| Error::contract("missing scheduled identity"))?;
        let &health = self
            .health
            .get(id)
            .ok_or_else(|| Error::contract("missing group health"))?;
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
        control.check()?;
        Ok(EpisodeLinks {
            ordinal,
            episode,
            group,
            spec,
            health,
        })
    }
}

/// Actual preflight source hashes, shared immutably by all jobs in a collection.
/// File hashes still come exclusively from each job's own Sources evidence.
pub(crate) struct CollectionDigests {
    pub manifest: String,
    report: String,
    _metadata: crate::budget::Lease,
}
impl CollectionDigests {
    pub fn new(
        collection: &Collection,
        sources: &Sources,
        budget: &Arc<Budget>,
        control: &Control,
    ) -> Result<Self> {
        let manifest = sources.digest(&collection.directory.join("manifest.json"), control)?;
        let report = sources.digest(&collection.directory.join("report.json"), control)?;
        let metadata = budget.reserve(manifest.len() + report.len())?;
        Ok(Self {
            manifest: manifest.to_owned(),
            report: report.to_owned(),
            _metadata: metadata,
        })
    }
}

pub(crate) fn source_metadata(
    collection: &Collection,
    source: &Path,
    replay: &Value,
    journal: &crate::typed_journal::view::JournalView<'_, '_>,
    links: &EpisodeLinks<'_>,
    sources: &Sources,
    digests: &CollectionDigests,
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<Value> {
    let facts = journal.provenance();
    let header = facts.header;
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
        let expected = ppo_selection_version(string(header, "policy")?)?;
        for index in 0..facts.decision_count {
            control.check()?;
            if journal.decision(index)?.policy_version()? != expected {
                return Err(Error::contract(
                    "PPO learner selection differs from frozen inference identity",
                ));
            }
        }
    }
    let end = facts
        .end
        .ok_or_else(|| Error::contract("missing natural journal end"))?;
    if string(end, "end_kind")? != "natural"
        || facts.accepted_count != facts.decision_count
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
    let _replay_timer = sources
        .telemetry()
        .phase_timer(crate::telemetry::Phase::ReplayValidation);
    journal.reconcile_replay(
        replay,
        viewer,
        links.episode.require("outcome")?,
        budget,
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
        ("sha256", json::text(&digests.report)),
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
        ("manifest_sha256", json::text(&digests.manifest)),
        ("header", json::copy(header, control)?),
        ("lineage", lineage),
        (
            "decision_count",
            json::integer(facts.decision_count as i128),
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

#[cfg(test)]
mod episode_index_tests {
    use super::*;
    use crate::budget::monotonic;
    use serde_json::json;

    fn manifest(count: usize) -> Value {
        let episodes: Vec<_> = (0..count)
            .map(|i| {
                json!({
                    "episode_id": format!("e{i}"), "pair_id": format!("p{i}"),
                    "policies": ["a", "b"], "seed": i,
                })
            })
            .collect();
        let schedule: Vec<_> = (0..count)
            .rev()
            .map(|i| {
                json!({
                    "seed": i, "episodes": [{"episode_id": format!("e{i}"),
                        "pair_id": format!("p{i}"), "policies": ["a", "b"]}],
                })
            })
            .collect();
        let groups: Vec<_> = (0..count)
            .rev()
            .map(|i| {
                json!({
                    "episode_ids": [format!("e{i}")],
                    "process": {"exit_code": 0, "shutdown": "stdin_eof"},
                })
            })
            .collect();
        json!({"episodes": episodes, "schedule": schedule, "groups": groups})
    }
    fn control() -> Control {
        Control::new(monotonic() + 30.).unwrap()
    }

    #[test]
    fn indexed_links_preserve_original_ordinal_with_permuted_schedule_and_health() {
        let mut m = manifest(8);
        m["episodes"].as_array_mut().unwrap().reverse();
        let budget = Budget::new(1024 * 1024);
        let control = control();
        let index = EpisodeIndex::new(&m, &budget, &control).unwrap();
        for i in 0..8 {
            let links = index.links(&format!("e{i}"), &control).unwrap();
            assert_eq!(links.ordinal, 7 - i);
            assert!(std::ptr::eq(links.episode, &m["episodes"][7 - i]));
            assert_eq!(links.spec["episode_id"], format!("e{i}"));
            assert_eq!(links.group["seed"], i);
        }
    }

    #[test]
    fn missing_and_duplicate_sources_and_schedule_conflicts_fail_closed() {
        let budget = Budget::new(1024 * 1024);
        let control = control();
        for (field, expected) in [
            ("episodes", "duplicate manifest episode"),
            ("schedule", "duplicate scheduled identity"),
            ("groups", "duplicate group health"),
        ] {
            let mut m = manifest(2);
            let duplicate = m[field][0].clone();
            m[field].as_array_mut().unwrap().push(duplicate);
            assert!(matches!(EpisodeIndex::new(&m, &budget, &control),
                            Err(Error::Contract(message)) if message == expected));
        }
        for (field, expected) in [
            ("episodes", "journal episode missing from manifest"),
            ("schedule", "missing scheduled identity"),
            ("groups", "missing group health"),
        ] {
            let mut m = manifest(1);
            m[field].as_array_mut().unwrap().clear();
            let index = EpisodeIndex::new(&m, &budget, &control).unwrap();
            assert!(matches!(index.links("e0", &control),
                            Err(Error::Contract(message)) if message == expected));
        }
        for field in ["pair_id", "policies"] {
            let mut m = manifest(1);
            m["schedule"][0]["episodes"][0][field] = json!("different");
            let index = EpisodeIndex::new(&m, &budget, &control).unwrap();
            assert!(matches!(index.links("e0", &control),
                            Err(Error::Contract(message)) if message == "source schedule/result mismatch"));
        }
        let mut m = manifest(1);
        m["schedule"][0]["seed"] = json!(999);
        let index = EpisodeIndex::new(&m, &budget, &control).unwrap();
        assert!(matches!(index.links("e0", &control),
                        Err(Error::Contract(message)) if message == "source schedule/result mismatch"));
    }

    #[test]
    fn health_membership_preserves_same_record_repeats_and_ignored_nonstrings() {
        let mut m = manifest(1);
        m["groups"][0]["episode_ids"] = json!(["e0", null, 1, {}, "e0"]);
        let budget = Budget::new(1024 * 1024);
        let control = control();
        let index = EpisodeIndex::new(&m, &budget, &control).unwrap();
        assert!(index.links("e0", &control).is_ok());
        drop(index);
        for _ in 0..2 {
            m["groups"]
                .as_array_mut()
                .unwrap()
                .push(json!({"episode_ids": ["foreign"]}));
        }
        let index = EpisodeIndex::new(&m, &budget, &control).unwrap();
        assert_eq!(index.health.len(), 1);
        assert!(index.links("e0", &control).is_ok());
        drop(index);
        m["groups"]
            .as_array_mut()
            .unwrap()
            .push(json!({"episode_ids": ["e0"]}));
        assert!(matches!(EpisodeIndex::new(&m, &budget, &control),
                        Err(Error::Contract(message)) if message == "duplicate group health"));
    }

    #[test]
    fn index_capacity_is_linear_reserved_once_and_queries_reuse_it() {
        for count in [1, 32, 256, 1024] {
            let m = manifest(count);
            let budget = Budget::new(2 * 1024 * 1024);
            let control = control();
            let index = EpisodeIndex::new(&m, &budget, &control).unwrap();
            // Deterministic structural scaling: three E-entry hash tables,
            // one linear reservation, then queries through those retained maps.
            assert_eq!(
                (
                    index.episodes.len(),
                    index.schedule.len(),
                    index.health.len()
                ),
                (count, count, count)
            );
            let admitted = 1024 + 160 * 3 * count;
            assert_eq!(budget.used(), admitted);
            for i in (0..count).rev() {
                assert_eq!(index.links(&format!("e{i}"), &control).unwrap().ordinal, i);
            }
            assert_eq!(budget.used(), admitted);
            drop(index);
            assert_eq!(budget.used(), 0);
        }
        let m = manifest(1);
        let budget = Budget::new(1023);
        let control = control();
        assert!(matches!(
            EpisodeIndex::new(&m, &budget, &control),
            Err(Error::ResourceLimit(_))
        ));
        assert_eq!(budget.used(), 0);
        control.cancel();
        assert!(matches!(
            EpisodeIndex::new(&m, &budget, &control),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    #[ignore = "bounded release microbenchmark; run explicitly without concurrent work"]
    fn benchmark_repeated_scans_vs_episode_index() {
        use std::{hint::black_box, time::Instant};
        let m = manifest(1024);
        let ids: Vec<_> = (0..1024).map(|i| format!("e{i}")).collect();
        let budget = Budget::new(2 * 1024 * 1024);
        let control = control();
        let start = Instant::now();
        for _ in 0..3 {
            for id in &ids {
                let episodes = m["episodes"].as_array().unwrap();
                let episode = episodes
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e["episode_id"].as_str() == Some(id))
                    .last()
                    .unwrap();
                let scheduled = m["schedule"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|group| {
                        group["episodes"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(move |spec| (group, spec))
                    })
                    .filter(|(_, s)| s["episode_id"].as_str() == Some(id))
                    .last()
                    .unwrap();
                let health = m["groups"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|g| {
                        g["episode_ids"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|v| v.as_str() == Some(id))
                    })
                    .last()
                    .unwrap();
                black_box((episode, scheduled, health));
            }
        }
        let scan_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let index = EpisodeIndex::new(&m, &budget, &control).unwrap();
        for _ in 0..3 {
            for id in &ids {
                black_box(index.links(id, &control).unwrap());
            }
        }
        println!(
            "episode_lookup episodes=1024 repetitions=3 scan_seconds={} indexed_build_and_queries_seconds={}",
            scan_seconds,
            start.elapsed().as_secs_f64()
        );
    }
}

#[cfg(test)]
mod ppo_identity_tests {
    use super::*;
    #[test]
    fn closed_ppo_policy_labels_preserve_mlp_and_gru_and_reject_bare_or_unknown() {
        assert_eq!(ppo_selection_version("mlp-p").unwrap(), "mlp-ppo-v1-p");
        assert_eq!(ppo_selection_version("gru-p").unwrap(), "gru-ppo-v1-p");
        assert_eq!(
            ppo_selection_version("transformer-p").unwrap(),
            "transformer-ppo-v1-p"
        );
        for label in ["p", "", "mlp-", "gru-", "transformer-", "future-p"] {
            assert!(ppo_selection_version(label).is_err(), "accepted {label}");
        }
    }
}
