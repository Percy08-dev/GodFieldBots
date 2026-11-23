"""CPU checkpoint boundaries through the common owned PPO invocation."""
import asyncio
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from gf_models.sharded_arrays import read_json
from tests.hpo_owned_support import owned_runtime, run_owned


def operating_schedule(*updates, cap=None):
    operating = {}
    if cap is not None:
        operating["max_updates"] = cap
    return dict(operating, checkpoint_evaluation={
        "enabled": True, "wall_seconds": 120., "updates": list(updates),
    })


def test_owned_checkpoint_schedule_commits_g0_g1_and_finishes_at_final_d(tmp_path, monkeypatch):
    from tests.test_structured_numeric_worker import assert_artifacts_equal

    runtime, run_id, _, _ = owned_runtime(tmp_path, monkeypatch,
        operating=operating_schedule(0, 1, 2))
    directory = tmp_path / "parents" / run_id

    async def run():
        g0, ops0 = await run_owned(runtime, directory, pause_after_update=0)
        assert (g0["status"], g0["updates"], g0["used_decisions"]) == ("paused", 0, 0)
        assert not any((directory / "collections").iterdir())
        assert not any((directory / "updates").iterdir())
        assert read_json(directory / "artifacts/000000/resume.json")["optimizer_steps"] == 0
        assert ops0.receipt is not None
        initial_json = (directory / "generations/000000.json").read_bytes()
        initial_wall = read_json(directory / "run.json")["wall_deadline_unix"]

        g1, ops1 = await run_owned(runtime, directory, mode="resume", pause_after_update=1)
        assert (g1["status"], g1["updates"], g1["used_decisions"]) == ("paused", 1, 16)
        assert ops1.receipt is not None
        assert (directory / "generations/000000.json").read_bytes() == initial_json

        final, ops2 = await run_owned(runtime, directory, mode="resume", pause_after_update=2)
        assert (final["status"], final["updates"], final["used_decisions"]) == ("complete", 2, 32)
        assert read_json(directory / "run.json")["wall_deadline_unix"] == initial_wall
        assert ops2.completion is not None
        ops2.completion.close()

        uninterrupted_root = tmp_path / "continuous"
        uninterrupted_root.mkdir()
        unbroken, unbroken_id, _, _ = owned_runtime(uninterrupted_root, monkeypatch,
            operating=operating_schedule(0, 1, 2))
        comparison = uninterrupted_root / "parents" / unbroken_id
        continuous, continuous_ops = await run_owned(unbroken, comparison)
        assert (continuous["status"], continuous["used_decisions"]) == ("complete", 32)
        assert continuous_ops.completion is not None
        continuous_ops.completion.close()
        assert_artifacts_equal(directory / "artifacts/000002", comparison / "artifacts/000002")

    asyncio.run(run())


def test_owned_checkpoint_schedule_stops_at_update_cap_before_scheduled_pause(tmp_path, monkeypatch):
    runtime, run_id, _, _ = owned_runtime(tmp_path, monkeypatch,
        operating=operating_schedule(1, cap=1))
    directory = tmp_path / "parents" / run_id
    result, ops = asyncio.run(run_owned(runtime, directory, pause_after_update=1))
    assert (result["status"], result["stop_reason"], result["updates"]) == ("incomplete", "update_cap", 1)
    assert ops.receipt is not None and ops.completion is None


def test_owned_zero_pause_requires_zero_in_the_active_schedule(tmp_path, monkeypatch):
    runtime, run_id, _, _ = owned_runtime(tmp_path, monkeypatch,
        operating=operating_schedule(1))
    directory = tmp_path / "parents" / run_id
    with pytest.raises(ContractError, match="pause target"):
        asyncio.run(run_owned(runtime, directory, pause_after_update=0))
    assert not directory.exists()


@pytest.mark.parametrize("enabled, reaches_d, fault", [
    (True, False, None), (False, False, None), (True, True, None),
    (True, False, "skipped"), (True, False, "failed"), (True, False, "unconfirmed"),
    (True, False, "tampered"),
])
def test_legacy_scheduler_closes_each_job_before_panel_and_binds_only_final_d(
        tmp_path, monkeypatch, enabled, reaches_d, fault):
    from gf_cli import hpo, hpo_evaluation
    from gf_cli.hpo_storage import CampaignStore
    from gf_models.arrays import atomic_json, sha256
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_ppo import HpoRunResult

    monkeypatch.setattr(gpu, "_GLOBAL_ROOT", tmp_path / "global")
    target = 3 if reaches_d else 2
    config = SimpleNamespace(schema_version=4, decision_budget=2 if reaches_d else 100,
        storage=SimpleNamespace(parent_root=tmp_path / "parents"),
        operating=SimpleNamespace(managed_cuda_bytes=1024, max_participant_workers=1),
        checkpoint_evaluation=SimpleNamespace(updates=(0, target)) if enabled else None,
        run_wall_seconds=120.)
    loaded = SimpleNamespace(config=config)
    materialized = SimpleNamespace(config=SimpleNamespace(hpo_spec={"schema_version": 4}),
                                  sha256="a" * 64, effective={})
    proposal = SimpleNamespace(number=1, sha256="b" * 64, settings=None)
    events = []
    folder = "proposals/000001/seeds/17"
    original_operation_ids = []
    original_deadline = time.time() + 119.

    async def train(_loaded, _collector, root, *, gpu_job, mode, pause_after_update=None, **_kwargs):
        generation = 0 if pause_after_update == 0 else 2
        events.append((mode, pause_after_update, generation))
        original_operation_ids.append(gpu_job._admission["context"]["operation_id"])
        if generation == 0:
            status, reason = "paused", "pause"
        elif reaches_d:
            status, reason = "complete", None
        else:
            status, reason = "incomplete", "update_cap"
        state = {"run_id": root.name, "seed": 17, "config_sha256": materialized.sha256,
            "status": status, "stop_reason": reason, "updates": generation, "used_decisions": generation,
            "admission_deadline_unix": original_deadline, "execution_deadline_unix": original_deadline + .2,
            "wall_deadline_unix": original_deadline + .5}
        path = root / "generations" / f"{generation:06}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, {"generation": generation})
        state["generation_sha256"] = sha256(path)
        atomic_json(root / "run.json", state)
        return HpoRunResult({"run_dir": str(root), **state}, gpu_job.confirm_termination())

    async def panel(store, _loaded, _campaign, _proposal, *, generation, deadline, directory, **_kwargs):
        gpu.require_resolved_namespace()
        assert deadline <= time.monotonic() + 120.
        events.append(("panel", None, generation))
        if generation == 0 and fault is not None:
            panel_path = f"{folder}/checkpoint-evaluations/0"
            error = RuntimeError("panel failed")
            if fault == "tampered":
                closure_path = f"{folder}/training-slices/000000/closure.json"
                closure, _ = store.get(closure_path)
                atomic_json(store.directory / closure_path, dict(closure, used_decisions=3))
                return
            if fault == "skipped":
                raise TimeoutError("pre-intent timeout")
            evaluation_id = "failed-evaluation"
            store.put(panel_path + "/evaluation.intent.json", {"evaluation_id": evaluation_id})
            if fault == "failed":
                context = gpu.JobContext("campaign", evaluation_id, "evaluation", directory.name, evaluation_id)
                with gpu.admit_gpu_job(context, global_cuda_bytes=1024, max_workers=1) as job:
                    retired = job.confirm_termination()
                error.hpo_gpu_termination_receipt = retired
                store.put(panel_path + "/evaluation-failed-gpu-termination.json", {
                    "schema_version": 1, "kind": "hpo-failed-evaluation-termination",
                    "evaluation_id": evaluation_id, "gpu_termination": asdict(retired),
                    "error_type": type(error).__name__, "evaluation_eligible": False,
                })
            raise error

    def index(*_args, terminal_generation, **_kwargs):
        assert terminal_generation == 2

    def training_receipt(store, _loaded, _campaign, _proposal, *, training_seed, run_id, directory,
                         termination, **_kwargs):
        assert reaches_d, "incomplete checkpoint became eligible for the objective"
        path = directory / "run.json"
        state = read_json(path)
        source = {"run_id": run_id, "run_path": str(directory), "training_seed": training_seed,
            "config_sha256": state["config_sha256"], "state_sha256": sha256(path), "generation": 2,
            "generation_sha256": state["generation_sha256"], "used_decisions": 2}
        operation = hpo._terminal_training_operation(store, loaded, proposal, source, asdict(termination))
        assert operation == original_operation_ids[-1] and operation != run_id
        with pytest.raises(ContractError, match="settled original training slice"):
            hpo._terminal_training_operation(store, loaded, proposal, dict(source, generation=3),
                                             asdict(termination))
        events.append(("objective-receipt", None, 2))

    modules = {
        "gf_cli.hpo_config": SimpleNamespace(materialize_hpo_run=lambda *_args, **_kwargs: materialized),
        "gf_cli.card_instance": SimpleNamespace(),
        "gf_training.hpo_ppo": SimpleNamespace(HpoRunResult=HpoRunResult, train_hpo_ppo=train),
    }
    monkeypatch.setattr(hpo, "owned_runtime_module", lambda name, _: modules[name])
    monkeypatch.setattr(hpo, "persist_training_receipt", training_receipt)
    monkeypatch.setattr(hpo_evaluation, "evaluate_checkpoint", panel)
    monkeypatch.setattr(hpo_evaluation, "persist_best_checkpoint", index, raising=False)
    with CampaignStore(tmp_path / "campaign", mode="fresh", max_owned_bytes=1024**2) as store:
        status_path = f"{folder}/checkpoint-evaluations/0/status.json"
        if fault == "tampered":
            with pytest.raises(ContractError, match="settled prior training slice changed"):
                asyncio.run(hpo._train_seed(store, loaded, {"campaign_id": "campaign"}, proposal, 17))
            assert events == [("fresh", 0, 0), ("panel", None, 0)]
            return
        if fault == "unconfirmed":
            with pytest.raises(RuntimeError, match="panel failed"):
                asyncio.run(hpo._train_seed(store, loaded, {"campaign_id": "campaign"}, proposal, 17))
            assert events == [("fresh", 0, 0), ("panel", None, 0)]
            assert not store.exists(status_path)
            return
        if reaches_d:
            asyncio.run(hpo._train_seed(store, loaded, {"campaign_id": "campaign"}, proposal, 17))
        else:
            with pytest.raises(ContractError, match="training stopped incomplete"):
                asyncio.run(hpo._train_seed(store, loaded, {"campaign_id": "campaign"}, proposal, 17))
        assert store.exists(f"{folder}/training.terminal.json") == reaches_d
        if fault is not None:
            assert store.get(status_path)[0]["status"] == fault
        if enabled:
            assert events == [("fresh", 0, 0), ("panel", None, 0),
                              ("resume", target, 2),
                              ("objective-receipt" if reaches_d else "panel", None, 2)]
            assert len(set(original_operation_ids)) == 2
            prev, prev_sha = store.get(f"{folder}/training-slices/000000/closure.json")
            following, _ = store.get(f"{folder}/training-slices/000001/intent.json")
            assert following["previous_closure_sha256"] == prev_sha
            assert following["previous_generation_sha256"] == prev["generation_sha256"]
        else:
            assert events == [("fresh", None, 2)]
            assert not (Path(store.directory) / folder / "training-slices").exists()
