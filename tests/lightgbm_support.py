"""Explicit synthetic original-run bundles for gate/state-machine unit tests."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict
from pathlib import Path

from gf_cli.config import ExperimentConfig, ServerConfig
from gf_common.domain import (
    CATALOG_META,
    END_PROJECTION_VERSION,
    POLICY_INPUT_VERSION,
    REVIEWED_SPEC_HASHES,
    SIMULATOR_REVISION,
    EpisodeResult,
    canonical,
    digest,
)
from gf_common.schedule import schedule, serialize_schedule
from gf_evaluation.evaluation import report
from gf_training.journal import FEATURE_VERSION, JOURNAL_SCHEMA, REWARD_VERSION, ParticipantJournal
from tests.conftest import fixture
from tests.test_journal import final_state, journal_decision


def synthetic_run(directory: Path, make_input, *, split="train", seeds=("00000000-0000-0000-0000-000000000001",),
                  sampling=(17,), outcomes=None, decisions=True, ancestry=None, planned=None, both_viewers=False):
    directory.mkdir()
    effective = asdict(ExperimentConfig(server=ServerConfig(fixed_seed_experiment=True)))
    effective["run"].update(split=split, seeds=list(seeds), sampling_seeds=list(sampling), pairs_per_seed=1)
    planned = planned or schedule(tuple(seeds), tuple(sampling), 1, "random", "random")
    run_id = str(uuid.uuid4())
    episodes = []
    for group in planned:
        for spec in group.episodes:
            path = directory / spec.episode_id
            path.mkdir()
            game = uuid.uuid4().hex[:26].upper()
            outcome = outcomes(spec) if callable(outcomes) else outcomes if outcomes is not None else "draw"
            for viewer in (1, 2):
                journal = ParticipantJournal(path, spec.episode_id, viewer, spec.policies[viewer - 1], run_id=run_id, game_id=game,
                                             action_config_hash=digest(effective["actions"]))
                value = make_input()
                if viewer == 2 and both_viewers:
                    from gf_common.domain import DecisionInput
                    from gf_common.validation import validate_worker_input
                    raw = value.to_dict()
                    obs = raw['observation']
                    obs.update(viewer_player_id=2, current_player=2, active_player=2, state_revision=1)
                    obs['own']['player_id'] = 2
                    obs['opponent']['player_id'] = 1
                    raw['action_options'].update(viewer_player_id=2, state_revision=1)
                    value = validate_worker_input(DecisionInput(obs, tuple(raw['events']), raw['history_end'], tuple(raw['typed_metadata']), raw['catalog'], raw['action_options']))
                actions = []
                if (viewer == 1 or both_viewers) and decisions:
                    journal_decision(journal, value)
                    actions = [json.loads(journal.path.read_text().splitlines()[1])["command"]["action"]]
                if viewer == 1 or both_viewers:
                    final = final_state(value, outcome)
                else:
                    final = fixture("observation_p2_public.json")
                    final.update(result=outcome, phase="terminal", actionable=False, typed_action_schema=[], current_player=None, required_choice=None)
                journal.finish("natural", outcome, final)
                replay = {"game_id": game, "viewer_player_id": viewer, "observation_schema_version": "1",
                          "episode_end": {"type": "terminal", "result": outcome},
                          "turns": [{"decisions": [{"actor": viewer, "visible_action": {"type": "action", "action": action}} for action in actions]}]}
                (path / f"replay-player-{viewer}.json").write_text(canonical(replay))
            episodes.append(EpisodeResult(spec.episode_id, spec.pair_id, group.seed, spec.policies, game_id=game,
                                          outcome=outcome, end_kind="natural", replay_reconciled=True, first_player=1))
    schedule_data = serialize_schedule(planned, 3)
    if ancestry is not None:
        for group in schedule_data:
            group["ancestry"] = ancestry
    summary = report(episodes, planned, bootstrap_samples=2)
    encoded_report = canonical(summary).encode()
    (directory / "report.json").write_bytes(encoded_report)
    manifest = {"schema_version": 3, "run_id": run_id, "status": "complete", "effective_config": effective,
                "config_sha256": digest(effective), "split": split, "elapsed_seconds": 1.0,
                "schedule": schedule_data, "episodes": [asdict(episode) for episode in episodes],
                "groups": [{"episode_ids": [spec.episode_id for spec in group.episodes]} for group in planned],
                "report_sha256": hashlib.sha256(encoded_report).hexdigest(), "journal_schema_version": JOURNAL_SCHEMA,
                "feature_version": FEATURE_VERSION, "reward_policy_version": REWARD_VERSION,
                "end_projection_version": END_PROJECTION_VERSION, "policy_input_version": POLICY_INPUT_VERSION,
                "catalog": CATALOG_META, "simulator_contract_revision": SIMULATOR_REVISION,
                "reviewed_simulator_spec_sha256": REVIEWED_SPEC_HASHES, "observation_schema_version": "1", "action_schema_version": 1}
    (directory / "manifest.json").write_text(canonical(manifest))
    return manifest
