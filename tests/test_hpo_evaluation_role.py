"""Eval6 is an explicit closed role, never a learning/final64 capability."""
import copy
import time
from dataclasses import asdict

import pytest

from gf_common.card_instance_collection import (
    FINAL_VALIDATION,
    PPO_BATCH,
    HpoCollectionRole,
    HpoEvaluationRole,
    checked_hpo_evaluation_manifest,
    checked_linkage,
    checked_role,
    hpo_batch_role,
    hpo_evaluation_groups,
    hpo_evaluation_role,
)
from gf_common.domain import ContractError, digest
from gf_common.schedule import serialize_schedule


def binding(comparison=False):
    left = {"run_id": "left", "run_path": "/parents/left", "training_seed": 17,
        "generation": 2, "generation_sha256": "a" * 64, "inference_sha256": "b" * 64,
        "source_receipt_sha256": "c" * 64, "policy_label": "left-policy"}
    right = {**left, "run_id": "right", "run_path": "/parents/right", "training_seed": 23,
             "policy_label": "right-policy", "inference_sha256": "d" * 64} if comparison else None
    registry = {left["policy_label"]: {"model_id": "mlp", "policy_options": {
        "version": "4", "expected_manifest_digest": left["inference_sha256"]}}}
    if right is None:
        registry["heuristic-card-instance-v2"] = {"model_id": "heuristic", "policy_options": None}
    else:
        registry[right["policy_label"]] = {"model_id": "gru", "policy_options": {
            "version": "4", "expected_manifest_digest": right["inference_sha256"]}}
    link = {"schema_version": 1, "kind": "hpo-evaluation-linkage", "campaign_id": "campaign",
        "evaluation_id": "evaluation", "gpu_ticket": "e" * 32,
        "purpose": "comparison" if comparison else "objective", "conditions_sha256": "f" * 64,
        "left_parent": left, "right_parent": right,
        "game_seeds": ["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"],
        "learner_seats": {}, "schedule_digest": "0" * 64, "policy_registry_digest": digest(registry)}
    groups = hpo_evaluation_groups(link)
    schedule = serialize_schedule(groups, 4)
    link["learner_seats"] = {spec["episode_id"]: spec["learner_viewer"]
                              for group in schedule for spec in group["episodes"]}
    link["schedule_digest"] = digest(schedule)
    return link, schedule, registry


@pytest.mark.parametrize("comparison", [False, True])
def test_distinct_closed_role_swaps_same_own_streams_and_binds_panel(comparison):
    link, schedule, registry = binding(comparison)
    role = hpo_evaluation_role(4)
    assert type(role) is HpoEvaluationRole
    assert not isinstance(role, HpoCollectionRole)
    assert checked_role(role) is role
    assert checked_linkage(link, role=role, schedule=schedule, registry=registry) == link
    for group in schedule:
        first, second = group["episodes"]
        assert first["policies"] == second["policies"][::-1]
        assert first["sampling_seeds"] == second["sampling_seeds"][::-1]
        assert [first["learner_viewer"], second["learner_viewer"]] == [1, 2]
    if not comparison:
        assert schedule[0]["episodes"][0]["sampling_seeds"] == [6980607022302246907,16157659044000002144]


@pytest.mark.parametrize("role", [PPO_BATCH, FINAL_VALIDATION, hpo_batch_role(8)])
def test_eval_linkage_cannot_be_read_as_training_or_final64(role):
    link, schedule, registry = binding()
    with pytest.raises(ContractError):
        checked_linkage(link, role=role, schedule=schedule, registry=registry)


@pytest.mark.parametrize("mutation", ["extra", "missing", "wrong-role", "wrong-stream", "extra-game",
    "missing-game", "duplicate-game", "game-seed", "right-parent", "selection", "final64", "policy",
    "hash", "seat", "seed-bool", "manifest-sha", "relative-parent", "unknown-parent"])
def test_eval_linkage_refuses_substitution_or_unclosed_data(mutation):
    link, schedule, registry = binding()
    if mutation == "extra": link["unknown"] = None
    elif mutation == "missing": del link["conditions_sha256"]
    elif mutation == "wrong-role": link["purpose"] = "training"
    elif mutation == "wrong-stream": schedule[0]["episodes"][0]["sampling_seeds"][0] += 1
    elif mutation == "extra-game": schedule.append(copy.deepcopy(schedule[0]))
    elif mutation == "missing-game": schedule.pop()
    elif mutation == "duplicate-game": schedule[1] = copy.deepcopy(schedule[0])
    elif mutation == "game-seed": schedule[0]["seed"] = "00000000-0000-0000-0000-000000000003"
    elif mutation == "right-parent": link["right_parent"] = copy.deepcopy(link["left_parent"])
    elif mutation == "selection": link["selection_context"] = {}
    elif mutation == "final64": link["validation_invocation"] = {}
    elif mutation == "policy": registry["extra-policy"] = registry["left-policy"]
    elif mutation == "hash": link["schedule_digest"] = "0" * 64
    elif mutation == "seat": link["learner_seats"][schedule[0]["episodes"][0]["episode_id"]] = 2
    elif mutation == "seed-bool": link["left_parent"]["training_seed"] = True
    elif mutation == "manifest-sha": registry["left-policy"]["policy_options"]["expected_manifest_digest"] = "0" * 64
    elif mutation == "relative-parent": link["left_parent"]["run_path"] = "relative"
    else: link["left_parent"]["weights"] = "/hidden"
    with pytest.raises(ContractError):
        checked_linkage(link, role=hpo_evaluation_role(4), schedule=schedule, registry=registry)


@pytest.mark.parametrize("games", [0, 1, 3, True])
def test_eval_role_requires_positive_complete_seat_pairs(games):
    with pytest.raises(ContractError): hpo_evaluation_role(games)


def test_raw_role_values_do_not_bypass_exact_eval_factory():
    role = hpo_evaluation_role(2)
    with pytest.raises(ContractError): checked_role(HpoEvaluationRole(**{**asdict(role), "role": "hpo-ppo-batch"}))


def test_complete_manifest_rejects_training_fields_and_never_authorizes_source():
    from gf_common.card_instance_collection import (
        HPO_EVALUATION_CORE_FIELDS,
        HPO_EVALUATION_PUBLISHED_FIELDS,
    )

    link, schedule, registry = binding()
    manifest = dict.fromkeys(HPO_EVALUATION_CORE_FIELDS | HPO_EVALUATION_PUBLISHED_FIELDS)
    manifest.update(schema_version=6, model_version="4", collection_role="hpo-evaluation", linkage=link,
        schedule=schedule, policy_bindings=registry, config={"run": {"split": "validation"}},
        kind="card-instance-collection", command="evaluate", run_id=link["evaluation_id"], status="complete",
        source_eligibility="not-authorized-by-collection-metadata", schedule_schema_version=4,
        journal_schema_version=6, derived_schema_version=6, participant_protocol_version=3,
        candidate_codec="candidate-columns-v2", config_hash=digest({"run": {"split": "validation"}}),
        report_sha256="0" * 64,
        episodes=[{"episode_id": spec["episode_id"], "game_id": f"game-{index}-{seat}",
                   "seed": group["seed"], "policies": spec["policies"], "pair_id": spec["pair_id"],
                   "end_kind": "natural", "outcome": "draw", "failure": None, "health": {},
                   "replay_reconciled": True, "decisions": 0}
            for index, group in enumerate(schedule) for seat, spec in enumerate(group["episodes"])])
    assert checked_hpo_evaluation_manifest(manifest) is manifest
    for field in ("selection_context", "qualified", "parent_qualification", "validation_invocation", "tree_inputs"):
        with pytest.raises(ContractError): checked_hpo_evaluation_manifest({**manifest, field: None})
    with pytest.raises(ContractError): checked_hpo_evaluation_manifest({**manifest, "source_eligibility": "qualified"})
    with pytest.raises(ContractError): checked_hpo_evaluation_manifest({**manifest, "report_sha256": None})


def actual_metadata(tmp_path):
    """Actual CPU inference files, hand-authored game facts: no executed Source."""
    pytest.importorskip("torch")
    from gf_common.domain import RuntimePolicyBinding, canonical
    from gf_models.arrays import sha256
    from gf_training.card_instance_collection import hpo_evaluation_collection_metadata
    from gf_training.journal import atomic_json
    from tests.test_card_instance_baseline_runtime import receipt
    from tests.test_card_instance_learning import ACTIONS
    from tests.test_hpo_artifacts import configured_for, retained, saved

    learner, checkpoint, saved_receipt = saved(tmp_path, trained=False)
    owner = retained(configured_for(learner, checkpoint))
    try:
        label, execution = owner.binding(configured_for(learner, checkpoint))
        link, _, _ = binding()
        link["left_parent"].update(policy_label=label, inference_sha256=saved_receipt["inference_digest"])
        registry = {label: execution, "heuristic-card-instance-v2": RuntimePolicyBinding("heuristic")}
        schedule = serialize_schedule(hpo_evaluation_groups(link), 4)
        link.update(schedule_digest=digest(schedule), policy_registry_digest=digest({name: asdict(value) for name, value in registry.items()}),
            learner_seats={spec["episode_id"]: spec["learner_viewer"] for group in schedule for spec in group["episodes"]})
        directory = tmp_path / "collection"
        directory.mkdir()
        (directory / "opaque.txt").write_text("original immutable diagnostic")
        report = {"schema_version": 1, "healthy": True, "comparison_valid": True, "planned_games": 4,
                  "recorded_games": 4, "missing_games": 0, "natural_denominator": 4,
                  "planned_pairs": 2, "complete_pairs": 2, "end_counts": {"natural": 4},
                  "wins": 0, "draws": 4, "losses": 0, "paired_score": .5,
                  "failures": {}, "health_failures": {}, "group_health_failures": {}}
        atomic_json(directory / "report.json", report)
        runtimes = [receipt() for _ in schedule]
        for index, runtime in enumerate(runtimes):
            runtime["executed"].update(pid=123 + index, start_time=1 + index)
        worker_cap = 4  # Historical two-game, two-seat HPO evaluation capacity.
        manifest = hpo_evaluation_collection_metadata(actions=ACTIONS,
            config={"run": {"split": "validation", "max_participant_workers": worker_cap},
                "hpo_spec": {"operating": {"max_participant_workers": worker_cap}}, "actions": asdict(ACTIONS)},
            runtime_evidence=runtimes, schedule=schedule,
            inventory=[{"path": "opaque.txt", "sha256": sha256(directory / "opaque.txt"), "bytes": (directory / "opaque.txt").stat().st_size}],
            policies=list(registry), bots_revision="fixture", policy_bindings=registry,
            neural_inputs=owner.finalize(deadline=time.monotonic() + 30), linkage=link)
        manifest.update(kind="card-instance-collection", run_id=link["evaluation_id"], command="evaluate",status="complete",
            schedule_schema_version=4, code={"revision": "fixture"}, inputs=[],
            groups=[{"runtime_evidence": copy.deepcopy(runtime),
                     "process": {"exit_code": 0, "shutdown": "stdin_eof", "log_error": None}}
                    for runtime in runtimes],resource_admission={},
            episodes=[{"episode_id": spec["episode_id"], "game_id": f"game-{index}-{seat}","seed": group["seed"],
                       "policies": spec["policies"],"pair_id": spec["pair_id"],"end_kind":"natural","outcome":"draw",
                       "failure":None,"health":{},"replay_reconciled":True,"decisions":0}
                for index, group in enumerate(schedule) for seat,spec in enumerate(group["episodes"])],
            report_sha256=sha256(directory / "report.json"))
        report["episodes"] = copy.deepcopy(manifest["episodes"])
        atomic_json(directory / "report.json",report)
        manifest["report_sha256"] = sha256(directory / "report.json")
        atomic_json(directory / "manifest.json", manifest)
        assert canonical(manifest)
        return directory, manifest, ACTIONS
    finally:
        owner.release()


def test_metadata_publication_and_actual_original_inspector_share_exact_eval_role(tmp_path):
    from gf_common.preprocessing import PreparationConfig
    from gf_training.card_instance_collection import _publication_schema
    from gf_training.card_instance_collection_source import inspect_game_manifest
    from gf_training.hpo_collection import check_hpo_collection
    from gf_training.structured_source import _SourceOwner

    directory, manifest, actions = actual_metadata(tmp_path)
    assert _publication_schema(manifest)
    owner = _SourceOwner(directory, PreparationConfig(), time.monotonic()+30)
    rows, specs = inspect_game_manifest(owner, manifest, actions,"validation",role=hpo_evaluation_role(4))
    assert len(rows) == len(specs) == 4
    assert all(row["decisions"] == 0 for row in rows.values())
    for role in (PPO_BATCH, FINAL_VALIDATION, hpo_batch_role(8)):
        with pytest.raises(ContractError):
            inspect_game_manifest(owner,manifest,actions,"validation",role=role)
    with pytest.raises(ContractError): check_hpo_collection(manifest)


@pytest.mark.parametrize("fault", ["report-bytes", "report-hash", "failed-game", "extra-top-field", "missing-report-field"])
def test_saved_eval_metadata_never_hides_failed_or_changed_originals(tmp_path, fault):
    from gf_common.preprocessing import PreparationConfig
    from gf_training.card_instance_collection_source import inspect_game_manifest
    from gf_training.structured_source import _SourceOwner

    directory, manifest, actions = actual_metadata(tmp_path)
    if fault == "report-bytes": (directory / "report.json").write_text("{}")
    elif fault == "report-hash": manifest["report_sha256"] = "0" * 64
    elif fault == "failed-game": manifest["episodes"][0]["end_kind"] = "aborted"
    elif fault == "extra-top-field": manifest["qualification"] = True
    else: del manifest["report_sha256"]
    owner = _SourceOwner(directory,PreparationConfig(),time.monotonic()+30)
    with pytest.raises(ContractError):
        inspect_game_manifest(owner,manifest,actions,"validation",role=hpo_evaluation_role(4))


def test_exact_eval_plan_freezes_inputs_and_requires_original_held_job(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gf_cli.hpo_evaluation import HpoEvaluationPlan
    from gf_training import hpo_gpu_lease as gpu

    monkeypatch.setattr(gpu,"_GLOBAL_ROOT",tmp_path / "gpu")
    link, _, registry = binding()
    from gf_common.domain import RuntimePolicyBinding
    runtime = {label: RuntimePolicyBinding(**value) for label,value in registry.items()}
    groups = hpo_evaluation_groups(link)
    with gpu.admit_gpu_job(gpu.JobContext("campaign","evaluation","evaluation","left","evaluation"),global_cuda_bytes=1024,max_workers=2) as job:
        link["gpu_ticket"] = job.ticket
        started = time.monotonic()
        plan = HpoEvaluationPlan.create(groups,runtime,tmp_path / "evaluation",link,
            started_monotonic=started,admission_deadline=started+10,execution_deadline=started+20,
            work_deadline=started+25,wall_deadline=started+30)
        link["purpose"] = "training"
        registry["left-policy"]["policy_options"]["version"] = "3"
        assert plan.linkage["purpose"] == "objective"
        assert plan.bindings["left-policy"].policy_options["version"] == "4"
        plan.require_job(job)
        with pytest.raises(ContractError): plan.require(SimpleNamespace(config=SimpleNamespace()),gpu_job=job)
        job.confirm_termination()
        with pytest.raises(ContractError): plan.require(SimpleNamespace(),gpu_job=job)


def test_eval_plan_rejects_deadline_extension_and_unclosed_bindings(tmp_path):
    from gf_cli.hpo_evaluation import HpoEvaluationPlan
    from gf_common.domain import RuntimePolicyBinding

    link,_,registry = binding()
    runtime = {label: RuntimePolicyBinding(**value) for label,value in registry.items()}
    with pytest.raises(ContractError):
        HpoEvaluationPlan.create(hpo_evaluation_groups(link),runtime,tmp_path / "evaluation",link,
            started_monotonic=10,admission_deadline=30,execution_deadline=20,work_deadline=40,wall_deadline=50)
    from gf_common.domain import EpisodeGroup
    group = hpo_evaluation_groups(link)[0]
    with pytest.raises(ContractError):
        HpoEvaluationPlan.create((EpisodeGroup(group.seed,list(group.episodes)),*hpo_evaluation_groups(link)[1:]),runtime,
            tmp_path / "evaluation",link,started_monotonic=10,admission_deadline=20,
            execution_deadline=30,work_deadline=40,wall_deadline=50)


@pytest.mark.parametrize("fault", ["exit", "shutdown", "log", "reap", "process-budget", "failure", "client-close",
    "storage", "group-unknown", "group-no-process", "group-no-runtime", "group-extra", "group-missing", "duplicate",
    "receipt-order", "process-no-exit", "process-extra", "process-bool", "process-type", "episode-health",
    "episode-health-unknown", "report-no-group-summary", "report-group-summary-type", "report-count",
    "report-episode", "report-failures", "report-schema", "report-end-count", "report-missing-game", "report-pairs",
    "report-score", "report-extra-nan", "report-duplicate-health"])
def test_raw_group_and_episode_health_cannot_be_hidden_by_healthy_saved_report(tmp_path,fault):
    import json

    from gf_common.preprocessing import PreparationConfig
    from gf_models.arrays import sha256
    from gf_training.card_instance_collection_source import inspect_game_manifest
    from gf_training.journal import atomic_json
    from gf_training.structured_source import _SourceOwner

    directory, manifest, actions = actual_metadata(tmp_path)
    report = json.loads((directory / "report.json").read_text())
    group = manifest["groups"][0]
    process = group["process"]
    if fault == "exit": process["exit_code"] = 9;process["shutdown"] = "killed"
    elif fault == "shutdown": process["shutdown"] = "kill"
    elif fault == "log": process["log_error"] = "log drain failed"
    elif fault == "reap": process["reap_failure"] = "not joined"
    elif fault == "process-budget": process["budget_overrun_seconds"] = .1
    elif fault == "failure": group["failure"] = "deadline"
    elif fault == "client-close": group["client_close_failure"] = "failed"
    elif fault == "storage": group["storage_validation_failure"] = "original mismatch"
    elif fault == "group-unknown": group["unknown_cleanup_proof"] = None
    elif fault == "group-no-process": del group["process"]
    elif fault == "group-no-runtime": del group["runtime_evidence"]
    elif fault == "group-extra": manifest["groups"].append(copy.deepcopy(group))
    elif fault == "group-missing": manifest["groups"].pop()
    elif fault == "duplicate":
        manifest["groups"][1] = copy.deepcopy(group)
        manifest["runtime_evidence"][1] = copy.deepcopy(group["runtime_evidence"])
    elif fault == "receipt-order": manifest["groups"].reverse()
    elif fault == "process-no-exit": del process["exit_code"]
    elif fault == "process-extra": process["cleanup_ok"] = True
    elif fault == "process-bool": process["exit_code"] = False
    elif fault == "process-type": group["process"] = None
    elif fault == "episode-health": manifest["episodes"][0]["health"]["save_failure"] = "failed"
    elif fault == "episode-health-unknown": manifest["episodes"][0]["health"]["unknown_proof"] = None
    elif fault == "report-no-group-summary": del report["group_health_failures"]
    elif fault == "report-group-summary-type": report["group_health_failures"] = []
    elif fault == "report-count": report["recorded_games"] = 3
    elif fault == "report-episode": report["episodes"][0]["outcome"] = {"win":{"winner":1}}
    elif fault == "report-failures": report["failures"] = {"save_failure":1}
    elif fault == "report-schema": report["schema_version"] = True
    elif fault == "report-end-count": report["end_counts"] = {"natural":3,"aborted":1}
    elif fault == "report-missing-game": report["missing_games"] = 1
    elif fault == "report-pairs": report["complete_pairs"] = 1
    elif fault == "report-score": report["paired_score"] = .75
    elif fault == "report-extra-nan": report["wall_seconds"] = float("nan")
    if fault == "report-extra-nan":
        (directory / "report.json").write_text(json.dumps(report))
    elif fault == "report-duplicate-health":
        from gf_common.domain import canonical
        (directory / "report.json").write_text(canonical(report).replace('"group_health_failures":{}',
            '"group_health_failures":{},"group_health_failures":{}'))
    else:
        atomic_json(directory / "report.json",report)
    manifest["report_sha256"] = sha256(directory / "report.json")
    natural_outcomes = copy.deepcopy([(row["end_kind"],row["outcome"]) for row in manifest["episodes"]])
    owner = _SourceOwner(directory,PreparationConfig(),time.monotonic()+30)
    with pytest.raises(ContractError):
        inspect_game_manifest(owner,manifest,actions,"validation",role=hpo_evaluation_role(4))
    assert [(row["end_kind"],row["outcome"]) for row in manifest["episodes"]] == natural_outcomes


def test_saved_healthy_report_uses_actual_left_seat_natural_outcomes(tmp_path):
    import json

    from gf_common.preprocessing import PreparationConfig
    from gf_models.arrays import sha256
    from gf_training.card_instance_collection_source import inspect_game_manifest
    from gf_training.journal import atomic_json
    from gf_training.structured_source import _SourceOwner

    directory,manifest,actions = actual_metadata(tmp_path)
    outcomes = [{"win":{"winner":1}},"draw",{"win":{"winner":2}},{"win":{"winner":1}}]
    for row,outcome in zip(manifest["episodes"],outcomes,strict=True): row["outcome"] = outcome
    report = json.loads((directory / "report.json").read_text())
    report.update(wins=1,draws=1,losses=2,paired_score=.375,episodes=copy.deepcopy(manifest["episodes"]),
        diagnostic={"nested":[1e308,-1e308,0,False,None]})
    atomic_json(directory / "report.json",report)
    manifest["report_sha256"] = sha256(directory / "report.json")
    owner = _SourceOwner(directory,PreparationConfig(),time.monotonic()+30)
    rows,_ = inspect_game_manifest(owner,manifest,actions,"validation",role=hpo_evaluation_role(4))
    assert [row["outcome"] for row in rows.values()] == outcomes


@pytest.mark.parametrize("literal", ["1e999", "-1e999"])
@pytest.mark.parametrize("location", ["wall_seconds","diagnostic","nested"])
def test_eval_report_exponent_overflow_is_never_finite_evidence(tmp_path,literal,location):
    import json

    from gf_common.domain import canonical
    from gf_common.preprocessing import PreparationConfig
    from gf_models.arrays import sha256
    from gf_training.card_instance_collection_source import inspect_game_manifest
    from gf_training.structured_source import _SourceOwner

    directory,manifest,actions = actual_metadata(tmp_path)
    report = json.loads((directory / "report.json").read_text())
    if location == "nested": report["diagnostic"] = {"nested":[{"value":"OVERFLOW"}]}
    else: report[location] = "OVERFLOW"
    raw = canonical(report).replace('"OVERFLOW"',literal)
    assert "Infinity" not in raw and "NaN" not in raw
    (directory / "report.json").write_text(raw)
    manifest["report_sha256"] = sha256(directory / "report.json")
    outcomes = copy.deepcopy([row["outcome"] for row in manifest["episodes"]])
    owner = _SourceOwner(directory,PreparationConfig(),time.monotonic()+30)
    with pytest.raises(ContractError):
        inspect_game_manifest(owner,manifest,actions,"validation",role=hpo_evaluation_role(4))
    assert [row["outcome"] for row in manifest["episodes"]] == outcomes
