"""Opt-in actual CUDA participant bootstrap/IPC coverage; no Simulator/source claim.

Fresh process: CUBLAS_WORKSPACE_CONFIG=:4096:8 GFB_RUN_STRUCTURED_PROTOCOL_CUDA=1.
Configured visibility0 is explicit; unavailable CUDA fails rather than falling back.
"""
from __future__ import annotations

import asyncio
import math
import os
from dataclasses import asdict

import pytest

from gf_common.domain import digest
from gf_common.structured import EventBank
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_models.structured_baselines import make_structured_policy
from gf_models.structured_binding import policy_label, safe_options_v2
from gf_models.structured_config import InferenceBindingV2, ModelConfigV2, TemporalConfigV2
from gf_simulator_client.structured_resources import ReservationLedger
from gf_simulator_client.structured_worker import StructuredParticipantWorker
from gf_training.structured_numeric import Learner
from tests.test_structured_protocol import action_settings, own_frame

pytestmark = pytest.mark.skipif(os.environ.get("GFB_RUN_STRUCTURED_PROTOCOL_CUDA") != "1",
                               reason="explicit actual CUDA participant protocol opt-in required")


@pytest.mark.parametrize("kind", ["mlp", "gru", "transformer"])
def test_actual_cuda_child_two_frames_cpu_reference_sampling_replay_and_reap(make_input, tmp_path, kind):
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    # Full documented attention defaults; only the small synthetic action table
    # and two decision endpoints are bounded for this runtime component gate.
    model = ModelConfigV2(model_id=kind, temporal=TemporalConfigV2(kind=kind, num_layers=2 if kind == "transformer" else 1))
    learner = Learner(model, 17, InferenceBindingV2(), digest(asdict(action_settings())),
                      digest({"component": "cuda-protocol-only"}))
    artifact = learner.save(tmp_path / "component-checkpoint")
    options = safe_options_v2({"version": "2", "checkpoint": str(tmp_path / "component-checkpoint"),
        "expected_manifest_digest": artifact["inference_digest"], "model": asdict(model),
        "inference": asdict(learner.binding), "cuda_visible_devices": "", "allow_toy": True})
    binding = InferenceBindingV2(device="cuda:0")
    options["inference"], options["cuda_visible_devices"] = asdict(binding), "0"
    expected_label = policy_label(learner.model, binding, options["expected_manifest_digest"], cuda_visible_devices="0")
    async def run():
        ledger = ReservationLedger()
        replayed = []
        for _ in range(2):  # fresh episode initialization resets the child CPU Generator.
            worker = StructuredParticipantWorker(make_structured_policy, kind, 2**63, action_settings(), ledger=ledger, policy_options=options)
            selected_rows, history = [], []
            try:
                assert ledger.cuda_bytes > 0 and len(ledger.workers) == 1
                await worker.initialize()
                assert worker.expected_policy == expected_label and worker.startup_elapsed_ms > 0
                bank = EventBank(1, "opaque-cuda-component")
                value = make_input(hand=("bronze-club", "steel-armor"), learned=("fireball", "aura"))
                for revision in (0, 2):
                    frame, delta = own_frame(value, bank, revision, [{"type": "action_accepted", "player": 1}])
                    generated, selection = await worker.select_frame(frame, delta)
                    history.append(tokenize_frame(frame, bank.view(), aggregate_window=learner.model.event.aggregate_window))
                    candidates = tokenize_table(generated.table, frame, learner.model.max_positions,
                                                aggregate_window=learner.model.event.aggregate_window)
                    with learner.torch.no_grad():
                        logs, expected_value, _ = learner.network(tuple(history), candidates)
                    assert selection.logprob == pytest.approx(float(logs[selection.index]), abs=1e-5, rel=1e-5)
                    assert selection.value == pytest.approx(float(expected_value), abs=1e-5, rel=1e-5)
                    assert selection.policy_version == expected_label
                    assert all(math.isfinite(value) and value >= 0 for value in worker.last_timings.values())
                    assert {"input_pack_transfer_host_ms", "forward_enqueue_host_ms", "sync_host_ms", "return_ms", "sampling_ms"} <= set(worker.last_timings)
                    assert worker.lease.decisions == {} and ledger.cuda_bytes > 0
                    selected_rows.append((generated.table.table_hash, selection.index, selection.logprob, selection.value))
            finally:
                await worker.close()
            assert worker.process.poll() is not None and not ledger.workers and ledger.cuda_bytes == 0
            replayed.append(selected_rows)
        assert replayed[0] == replayed[1]
    asyncio.run(run())
