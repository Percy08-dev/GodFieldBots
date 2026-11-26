"""Real game faults retain the native Journal terminal and its original evidence."""

import json
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict, replace

import pytest

from gf_common.domain import ContractError, digest
from gf_common.group_pack import MIB, StorageSettings, decode_pack
from gf_simulator_client.rust_player import RustPlayerRuntime
from gf_training.group_pack_resources import StorageBudget
from gf_training.rust_player_storage import (
    PolicyJournalContext,
    RustPlayerStorage,
    StorageGroup,
    episode_encoding_plan,
)
from tests.test_rust_player_runtime import fault_proxy, game_spec, managed, runtime_config
from tests.test_rust_player_storage import journal_context
from tests.test_rust_player_storage import sacrifice_release as sacrifice_release  # noqa: PLC0414 - pytest fixture export


def _with_cpu_learner(spec, directory):
    pytest.importorskip("torch")
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.rust_player import RustPlayerInference, serialized_torch
    from gf_models.structured_artifacts import save_game_inference_v4
    from gf_models.structured_binding import ConfiguredNeuralModelV4
    from gf_models.structured_config import InferenceBindingV2
    from gf_models.structured_network import make_network
    from gf_models.structured_runtime import own_game_worker_settings_v3
    from tests.model3_support import LAYOUT
    from tests.test_hpo_network import small_model4

    context = journal_context(spec)
    model = small_model4()
    binding = InferenceBindingV2(max_candidates=64, candidate_chunk=2)
    with serialized_torch():
        network = make_network(model, 17, binding, layout=LAYOUT)
    save_game_inference_v4(
        directory,
        network,
        model,
        digest(asdict(context.actions)),
        action_contract=context.actions.action_contract,
        rules=context.rules,
    )
    configured = ConfiguredNeuralModelV4(
        **{name: getattr(model, name) for name in model.__dataclass_fields__},
        checkpoint=str(directory),
        inference=binding,
    )
    owner = RetainedGameArtifacts(
        (configured,),
        context.actions,
        rules=context.rules,
        max_owned_bytes=128 * MIB,
        preflight_deadline=spec["deadline"],
        finalization_deadline=spec["cleanup_deadline"],
    )
    try:
        label, execution = owner.binding(configured)
        caps, costs, _, _ = own_game_worker_settings_v3(
            "mlp", context.actions, execution.policy_options, rules=context.rules
        )
        native = {
            "kind": "mlp",
            "seed": spec["policies"][1]["seed"],
            "checkpoint": execution.policy_options["expected_manifest_digest"],
            "epoch": 0,
            "limits": {
                "max_decisions": model.temporal.max_decisions,
                "max_episode_events": model.event.max_episode,
                "max_interval_events": model.event.max_interval,
                "max_candidates": binding.max_candidates,
                "max_positions": model.max_positions,
                "aggregate_window": model.event.aggregate_window,
            },
            "attack_threshold": 12,
            "numeric_costs": asdict(costs),
        }
        policy = PolicyJournalContext(
            native,
            {
                "policy": label,
                "model_id": "mlp",
                "caps": asdict(caps),
                "numeric_costs": asdict(costs),
                "sampling_seed": native["seed"],
                "neural": owner.journal_metadata(label),
            },
        )
        baseline = context.policies[0]
        baseline = replace(
            baseline, native=baseline.native | {"numeric_costs": baseline.journal["numeric_costs"]}
        )
        spec["policies"] = [baseline.native, native]
        spec["learner_viewer"] = 2
        service = RustPlayerInference(device="cpu", cache_entries=2, cache_weight_bytes=64 * MIB)
        service.publish(network, native["checkpoint"])
        return replace(context, policies=(baseline, policy), learner_viewer=2), service
    finally:
        owner.release()


@pytest.mark.parametrize(
    "fault,disposition,reason",
    [
        ("wrong_attestation", "incomplete", None),
        ("stale_hints", "incomplete", None),
        ("mismatch_ack", "incomplete", None),
        ("abort_turn_limit", "truncated", "turn_limit_exceeded"),
        ("abort_timeout", "aborted", "player_action_timeout"),
    ],
)
def test_actual_rust_rollout_preserves_fault_terminal(
    tmp_path, sacrifice_release, fault, disposition, reason
):
    with (
        managed(tmp_path / "simulator", **sacrifice_release) as (startup, receipt),
        fault_proxy(startup, fault) as (routed, proxy),
    ):
        deadline = time.monotonic() + 30
        spec = game_spec(routed, receipt, deadline=deadline)
        spec["actions"]["action_contract"] = receipt["action_contract"]
        context, service = _with_cpu_learner(spec, tmp_path / "inference")
        settings = StorageSettings(
            db_root=tmp_path / "catalog",
            ram_root=tmp_path / "ram",
            budget_bytes=512 * MIB,
            queue_bytes=4 * MIB,
        )
        runtime = RustPlayerRuntime(
            runtime_config()
            | {"runtime_identity": service.runtime, "storage_bytes": settings.budget_bytes},
            service=service,
        )
        work = tmp_path / "work" / "collection"
        durable = tmp_path / "durable" / "collection"
        writer = None
        try:
            writer = RustPlayerStorage(
                runtime,
                durable,
                run_id=context.run_id,
                contexts=(),
                groups=(StorageGroup("group", (context.episode_id,), {}),),
                settings=settings,
                workspace=StorageBudget(settings, deadline),
                workspace_bytes=256 * MIB,
                deadline=deadline,
                cleanup_deadline=spec["cleanup_deadline"],
                deferred_contexts=True,
                context_plans=(episode_encoding_plan(context),),
                training=True,
                native_rollout=True,
                work_directory=work,
                durable_mount=(tmp_path, tmp_path.stat().st_dev),
            )
            writer.bind_contexts((context,))
            writer._finish_rollout_registration()
            runtime.start([spec])
            reader = writer._begin_rollout_read()
            try:
                packet = reader.next_terminal(deadline)
                bundle = packet.bundle()
                original = json.loads(memoryview(bundle.view()).tobytes())
                prior = json.loads(packet.prior_receipts())
                assert not packet.has_evidence()
                assert packet.same_original(bundle)
                writer.accept_rollout(bundle, packet)
                runtime.storage_ack(bundle)
            except BaseException:
                writer._finish_rollout_read(failed=True)
                raise
            else:
                writer._finish_rollout_read(failed=False)
            del packet, bundle, reader

            runtime.wait(deadline)
            report = runtime.completed_reports(
                [(context.run_id, context.trial, context.episode_id)]
            )[0]["report"]
            assert report["storage_accepted"] is True
            assert report["end_kind"] == disposition and report["simulator_abort_reason"] == reason

            inventory = writer.inventory()
            try:
                database = f"{(work / 'collection.sqlite3').as_uri()}?mode=ro"
                with closing(sqlite3.connect(database, uri=True)) as db:
                    (pack,) = db.execute("SELECT pack FROM groups").fetchone()
                archive = {m.name: m.data for m in decode_pack(pack)}
                assert set(archive) == {row[0] for row in inventory.rows}
                with pytest.raises(ContractError, match="complete committed learner trajectories"):
                    writer.training_evidence(inventory, (context.episode_id,))
            finally:
                inventory.close()
            assert not durable.exists()

            names = set(archive)
            prefix = context.episode_id + "/"
            terminal = json.loads(archive[prefix + "native-terminal.json"])
            assert terminal == {
                k: original[k] for k in ("report", "banks", "terminal_observations")
            }
            for replay in original["replays"]:
                viewer = replay["viewer_player_id"]
                assert json.loads(archive[f"{prefix}replay-player-{viewer}.json"]) == replay
            records = [r["kind"] for r in prior]
            if fault == "wrong_attestation":
                assert not records and report["game"] is None
                assert names == {prefix + "native-terminal.json"}
            else:
                pending_count = 0
                for viewer in (1, 2):
                    rows = [
                        json.loads(line)
                        for line in archive[f"{prefix}player-{viewer}.jsonl"].splitlines()
                    ]
                    assert rows[0]["game_id"] == report["game"]
                    end = rows[-1]
                    assert end["end_kind"] == disposition
                    assert end["simulator_abort_reason"] == reason
                    assert (
                        end["outcome"] is end["terminal_reward"] is end["final_observation"] is None
                    )
                    pending_count += len(end["pending_decisions"])
                    assert not any(row["kind"] == "transition" for row in rows)
                if fault == "stale_hints":
                    assert not records and pending_count == 0
                elif fault == "mismatch_ack":
                    assert records == ["pending"] and pending_count == 1
                else:
                    assert records == ["pending", "ack"] and pending_count == 0
            assert proxy["act_started"].is_set() == bool(records)
        finally:
            runtime.cancel("rollout terminal integration test complete")
            runtime.join()
            if writer is not None:
                writer.close()
        assert runtime.recording_account().used == 0
