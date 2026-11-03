"""Corrected game inference components, never real games or Source evidence."""
import asyncio
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gf_common.card_instance import ActionConfigV3, card_instance_contract
from gf_common.card_instance_rules import card_instance_rules
from gf_common.diverse_actions import PublicCandidateGeneratorV3
from gf_common.domain import ContractError, RuntimePolicyBinding, canonical, digest
from gf_common.structured import Selection, event_delta
from gf_common.structured_tokens import tokenize_table
from gf_models import structured_artifacts as artifacts
from gf_models.arrays import sha256
from gf_models.card_instance_artifacts import RetainedGameArtifacts, validate_neural_metadata
from gf_models.structured_binding import (
    ConfiguredNeuralModelV3,
)
from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_network import make_network
from gf_models.structured_policy import GameStructuredPolicyV3, StructuredPolicyV3
from gf_models.structured_runtime import admit_game_workers_v3, own_game_worker_settings_v3
from gf_simulator_client.card_instance_worker import CardInstanceParticipantWorker
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.card_instance_collection import collection_metadata
from gf_training.card_instance_journal import CardInstanceJournal, read_card_instance_journal
from tests.model3_support import LAYOUT, model3, raw_inputs
from tests.test_card_instance_baseline_runtime import receipt
from tests.test_model3_artifacts import update

RULES = card_instance_rules()
ACTIONS = ActionConfigV3(action_contract=LAYOUT.action_contract, max_candidates=64)
ACTION_HASH = digest(asdict(ACTIONS))


def binding_for(device):
    if device != "cpu" and os.environ.get("GF_RUN_CUDA") != "1":
        pytest.skip("actual HOST CUDA cases use GF_RUN_CUDA=1")
    if device != "cpu" and not torch.cuda.is_available():
        pytest.fail("requested HOST CUDA check has no device")
    return InferenceBindingV2(device=device, max_candidates=64, candidate_chunk=2)


def saved(tmp_path, temporal="mlp", hand="attention", device="cpu", window=128):
    model, binding = model3(hand, temporal), binding_for(device)
    model = replace(model, event=replace(model.event, aggregate_window=window))
    network = make_network(model, 17, binding, layout=LAYOUT)
    directory = tmp_path / "game-artifact"
    manifest = artifacts.save_game_inference_v3(directory, network, model, ACTION_HASH,
        action_contract=LAYOUT.action_contract, rules=RULES)
    configured = ConfiguredNeuralModelV3(**{name: getattr(model, name) for name in model.__dataclass_fields__},
        checkpoint=str(directory), inference=binding,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES") if device != "cpu" else None)
    owner = RetainedGameArtifacts((configured,), ACTIONS, rules=RULES, max_owned_bytes=256 * 1024**2,
        preflight_deadline=time.monotonic() + 30, finalization_deadline=time.monotonic() + 120)
    label, execution = owner.binding(configured)
    return owner, label, execution, network, model, binding, manifest


def generated(execution, *, seed=17, count=1):
    frames, _, originals, bank, _ = raw_inputs(count=count)
    caps, costs, _, _ = own_game_worker_settings_v3(execution.model_id, ACTIONS, execution.policy_options, rules=RULES)
    search_seed = int(digest({"recipe": "structured-search-stream-v1", "sampling_seed": seed})[:16], 16)
    generator = PublicCandidateGeneratorV3(ACTIONS, rules=RULES, clock=lambda: 0.0)
    game = generator.generate_game(originals[-1], bank.view(), random.Random(search_seed), costs)
    component = generator.generate(originals[-1], bank.view(), random.Random(search_seed), costs)
    return frames, originals[-1], bank, game, component, caps, costs


@pytest.mark.parametrize("device", ["cpu", "cuda:0"], ids=["cpu", "cuda"])
@pytest.mark.parametrize("hand", ["direct", "deepsets", "attention"])
@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_game_all_modes_inference_resume_and_same_weight_component_numerics(tmp_path, device, hand, temporal):
    owner, label, execution, network, model, binding, manifest = saved(tmp_path, temporal, hand, device)
    directory = Path(execution.policy_options["checkpoint"])
    assert not (directory / "resume.json").exists()  # Inference-only ownership.
    frames, frame, bank, game, component, _, costs = generated(execution, count=3)
    game_tokens = tokenize_table(game.table, frame, layout=LAYOUT)
    comp_tokens = tokenize_table(component.table, frame, layout=LAYOUT)
    other = make_network(model, 17, binding, layout=LAYOUT)
    component_policy = StructuredPolicyV3(other, model, binding, "1" * 64, layout=LAYOUT,
        action_contract=LAYOUT.action_contract, rules=RULES, purpose="component", synthetic_capacity=False,
        action_config_hash=ACTION_HASH)
    with torch.no_grad():
        for left, right in zip(network(frames, game_tokens), other(frames, comp_tokens), strict=True):
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
    policy = GameStructuredPolicyV3.from_artifact(directory, sha256(directory / "inference.json"), model,
        binding, ACTION_HASH, action_contract=LAYOUT.action_contract, rules=RULES)
    assert policy.version == label
    with pytest.raises(ContractError):
        policy.select_frame(frame, bank.view(), component)
    with pytest.raises(ContractError):
        component_policy.select_frame(frame, bank.view(), game)
    originals = raw_inputs(count=3)[2]
    for index, original in enumerate(originals):
        current = PublicCandidateGeneratorV3(ACTIONS, rules=RULES, clock=lambda: 0.0).generate_game(
            original, bank.view(), random.Random(17), costs)
        with torch.no_grad():
            logs, value, _ = network(frames[:index + 1], tokenize_table(current.table, original, layout=LAYOUT))
        selected = policy.select_frame(original, bank.view(), current)
        assert math.isclose(selected.logprob, float(logs[selected.index].cpu()), abs_tol=1e-5, rel_tol=1e-5)
        assert math.isclose(selected.value, float(value.cpu()), abs_tol=1e-5, rel_tol=1e-5)
    # Keep the full temporal prefix for numerical resume while actual policy
    # selection starts at the original first decision below.
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0003)
    shuffle = np.random.Generator(np.random.PCG64(17))
    update(network, optimizer, shuffle, frames, game_tokens)
    # A paired inference must represent the same normal update boundary.
    paired = tmp_path / "paired"
    artifacts.save_game_inference_v3(paired, network, model, ACTION_HASH,
        action_contract=LAYOUT.action_contract, rules=RULES)
    artifacts.save_game_resume_v3(paired, network, optimizer, shuffle, 1, model, binding, "0" * 64,
        action_config_hash=ACTION_HASH, action_contract=LAYOUT.action_contract, rules=RULES)
    restored, _ = artifacts.load_game_inference_v3(paired, expected_digest=sha256(paired / "inference.json"),
        model=model, binding=binding, action_config_hash=ACTION_HASH, action_contract=LAYOUT.action_contract, rules=RULES)
    target_optimizer = torch.optim.Adam(restored.parameters(), lr=0.0003)
    target_shuffle = np.random.Generator(np.random.PCG64(999))
    assert artifacts.restore_game_resume_v3(paired, restored, target_optimizer, target_shuffle, model, binding, "0" * 64,
        action_config_hash=ACTION_HASH, action_contract=LAYOUT.action_contract, rules=RULES) == 1
    assert canonical(shuffle.bit_generator.state) == canonical(target_shuffle.bit_generator.state)
    for (name, original), (other_name, actual) in zip(network.named_parameters(), restored.named_parameters(), strict=True):
        assert name == other_name
        torch.testing.assert_close(original, actual, atol=0, rtol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(optimizer.state[original][key], target_optimizer.state[actual][key], atol=0, rtol=0)
    # Both existing component wrappers and bound networks refuse relabeling.
    with pytest.raises(ContractError):
        artifacts.save_inference_v3(tmp_path / "component-relabel", network, model, ACTION_HASH,
            layout=LAYOUT, action_contract=LAYOUT.action_contract, rules=RULES,
            purpose="component", synthetic_capacity=False)
    with pytest.raises(ContractError):
        artifacts.save_game_inference_v3(tmp_path / "game-relabel", other, model, ACTION_HASH,
            action_contract=LAYOUT.action_contract, rules=RULES)
    assert not (tmp_path / "component-relabel").exists() and not (tmp_path / "game-relabel").exists()
    assert manifest["layout_signature"] == owner.journal_metadata(label)["policy_options"]["layout_signature"]
    owner.finalize(deadline=time.monotonic() + 30)
    owner.release()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"], ids=["cpu", "cuda"])
@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_actual_game_worker_journal6_original_refs_and_sealed_collection5(tmp_path, temporal, device, monkeypatch):
    monkeypatch.setenv("GAME_SEED", "management-secret")
    monkeypatch.setenv("GF_SERVER_URL", "http://management-secret")
    owner, label, execution, _, _, _, _ = saved(tmp_path, temporal, device=device)
    _, frame, bank, _, _, caps, costs = generated(execution)

    async def run():
        ledger = ReservationLedger()
        worker = CardInstanceParticipantWorker(temporal, 17, ACTIONS, ledger=ledger,
            policy_options=execution.policy_options, artifact_inputs=owner)
        try:
            initial_bytes = canonical(worker.initial)
            assert "management-secret" not in initial_bytes and "factory" not in worker.initial
            assert worker.initial["policy_options"] == execution.policy_options
            environment = Path(f"/proc/{worker.process.pid}/environ").read_bytes()
            assert b"GAME_SEED" not in environment and b"management-secret" not in environment
            generated_rows, selected = await worker.select_frame(frame,
                event_delta(
                    frame.actor, frame.episode, 0, bank.view()._prefixes[0], bank.view().events(), layout=LAYOUT))
            assert selected.policy_version == label
            assert selected.table_hash == generated_rows.table.table_hash
            assert all(math.isfinite(number) for number in (selected.logprob, selected.value))
            assert generated_rows.table.frame_hash == frame.frame_hash
            return generated_rows, selected
        finally:
            await worker.close()
            assert worker.process.poll() is not None and not ledger.workers and not ledger.cuda_bytes

    rows, selected = asyncio.run(run())
    delta = event_delta(frame.actor, frame.episode, 0, bank.view()._prefixes[0], bank.view().events(), layout=LAYOUT)
    folder = tmp_path / "journal"
    folder.mkdir()
    journal = CardInstanceJournal(folder, "episode", 1, label, run_id="run", game_id="game", actions=ACTIONS,
        bank_episode=bank.episode, caps=asdict(caps), numeric_costs=asdict(costs), model_id=temporal,
        runtime_evidence=receipt(), sampling_seed=17, neural=owner.journal_metadata(label))
    command = {"command_id": str(uuid.uuid4()), "expected_revision": frame.revision,
        "action": selected.validate(rows.table, frame)}
    journal.begin_frame(frame, delta, rows, selected, command, {})
    journal.finish("incomplete", None)
    data = read_card_instance_journal(journal.final_path, actions=ACTIONS)
    assert data.header["neural"] == owner.journal_metadata(label)
    assert data.decisions[0].selection == selected
    assert data.bank.events() == bank.view().events()
    metadata = collection_metadata(actions=ACTIONS, config={}, runtime_evidence=[receipt()],
        schedule=[{"fixture": True}], inventory=[{"fixture": True}], policies=[label], bots_revision="fixture",
        policy_bindings={label: execution}, neural_inputs=owner.finalize(deadline=time.monotonic() + 30))
    assert metadata["model_version"] == "3"
    assert metadata["source_eligibility"] == "not-authorized-by-collection-metadata"
    assert metadata["neural_inputs"]["policies"][label] == data.header["neural"]
    owner.release()


@pytest.mark.parametrize("mutation", ["old_profile", "old_signature", "component", "wide_capacity"])
def test_bad_game_context_refuses_before_network_allocation(tmp_path, monkeypatch, mutation):
    owner, _, execution, _, model, binding, _ = saved(tmp_path)
    path = Path(execution.policy_options["checkpoint"]) / "inference.json"
    value = json.loads(path.read_text())
    contract = LAYOUT.action_contract
    if mutation == "old_profile":
        contract = card_instance_contract()
    elif mutation == "old_signature":
        value["token_signature"] = "72db3a64f6d087a5f889c4e3cf6b4eb46a8595fd334e1249ca2d51f76496e1ad"
    elif mutation == "component":
        value["purpose"] = "component"
    else:
        model = replace(model, max_positions=19)
    path.write_text(json.dumps(value))
    import gf_models.structured_network as networks
    monkeypatch.setattr(networks, "make_network", lambda *args, **kwargs: pytest.fail("bad input allocated a network"))
    with pytest.raises(ContractError):
        artifacts.load_game_inference_v3(path.parent, expected_digest=sha256(path), model=model, binding=binding,
            action_config_hash=ACTION_HASH, action_contract=contract, rules=RULES)
    owner.release()


@pytest.mark.parametrize("mutation", ["append", "replace", "remove"])
def test_external_original_mutation_after_load_invalidates_finalization(tmp_path, mutation):
    owner, _, execution, _, model, binding, _ = saved(tmp_path)
    directory = Path(execution.policy_options["checkpoint"])
    artifacts.load_game_inference_v3(directory, expected_digest=sha256(directory / "inference.json"),
        model=model, binding=binding, action_config_hash=ACTION_HASH, action_contract=LAYOUT.action_contract, rules=RULES)
    path = next(directory.glob("weights-*.npz"))
    if mutation == "append":
        path.write_bytes(path.read_bytes() + b"late mutation")
    elif mutation == "replace":
        original = path.read_bytes()
        path.unlink()
        path.write_bytes(original)
    else:
        path.unlink()
    with pytest.raises((ContractError, OSError)):
        owner.finalize(deadline=time.monotonic() + 30)
    owner.release()


def test_joint_four_worker_admission_and_no_double_charge_of_covered_temporaries(tmp_path):
    # Metadata-only CUDA envelopes need no device allocation.
    owner, _, execution, _, _, _, _ = saved(tmp_path)
    options = copy.deepcopy(execution.policy_options)
    options["inference"]["device"] = "cuda:0"
    options["cuda_visible_devices"] = None
    options["model"]["resources"]["cuda_context_bytes"] = 3 * 1024**3
    cuda = RuntimePolicyBinding("mlp", options)
    one = admit_game_workers_v3(ACTIONS, (cuda, cuda), rules=RULES, game_concurrency=1)
    assert one["workers"] == 2 and one["decision_temporary_bytes"] == 0
    assert one["peak_cuda_bytes"] == one["worker_resident_bytes"]
    with pytest.raises(ContractError, match="joint"):
        admit_game_workers_v3(ACTIONS, (cuda, cuda), rules=RULES, game_concurrency=2)
    with pytest.raises(ContractError):
        admit_game_workers_v3(ACTIONS, (execution, execution), rules=RULES, game_concurrency=2, max_workers=3)
    owner.release()


def test_torch_free_sealed_owner_and_journal_metadata_validation(tmp_path):
    owner, label, _, _, _, _, _ = saved(tmp_path)
    raw = owner.journal_metadata(label)
    (tmp_path / "metadata.json").write_text(json.dumps(raw))
    script = '''import importlib.abc, json, sys, time
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self, name, path=None, target=None):
  if name == 'torch' or name.startswith('torch.'):
   raise AssertionError('Torch imported during retained metadata inspection')
sys.meta_path.insert(0, NoTorch())
from pathlib import Path
from gf_common.card_instance import ActionConfigV3, corrected_card_instance_contract
from gf_common.card_instance_rules import card_instance_rules
from gf_models.card_instance_artifacts import validate_neural_metadata
from gf_models.structured_sealed import SealedScan, inspect_retained_game_inference_v3
actions=ActionConfigV3(action_contract=corrected_card_instance_contract(),max_candidates=64)
rules=card_instance_rules()
raw=json.loads(Path(sys.argv[1]).read_text())
validate_neural_metadata(raw,actions=actions,rules=rules)
inspect_retained_game_inference_v3(Path(raw['policy_options']['checkpoint']),expected_inference_digest=raw['policy_options']['expected_manifest_digest'],actions=actions,rules=rules,scan=SealedScan(max_owned_bytes=268435456,deadline=time.monotonic()+10))
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "metadata.json")], check=False, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    owner.release()


def test_release_performs_no_filesystem_io_and_closes_live_binding(tmp_path, monkeypatch):
    owner, label, _, _, _, _, _ = saved(tmp_path)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("release performed filesystem I/O"))
    owner.release()
    owner.release()
    with pytest.raises(ContractError):
        owner.journal_metadata(label)


@pytest.mark.parametrize("failure", ["spawn", "timeout", "cancel", "released_owner"])
def test_neural_worker_failures_preserve_reap_and_reservation_ownership(tmp_path, monkeypatch, failure):
    owner, _, execution, _, _, _, _ = saved(tmp_path)
    import gf_simulator_client.card_instance_worker as workers

    async def run():
        ledger = ReservationLedger()
        if failure == "spawn":
            monkeypatch.setattr(workers.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected spawn")))
            with pytest.raises(OSError, match="injected"):
                CardInstanceParticipantWorker("mlp", 17, ACTIONS, ledger=ledger,
                    policy_options=execution.policy_options, artifact_inputs=owner)
        elif failure == "released_owner":
            owner.release()
            with pytest.raises(ContractError):
                CardInstanceParticipantWorker("mlp", 17, ACTIONS, ledger=ledger,
                    policy_options=execution.policy_options, artifact_inputs=owner)
        else:
            worker = CardInstanceParticipantWorker("mlp", 17, ACTIONS, timeout=0.000001 if failure == "timeout" else 10,
                ledger=ledger, policy_options=execution.policy_options, artifact_inputs=owner)
            task = asyncio.create_task(worker.initialize())
            if failure == "cancel":
                await asyncio.sleep(0)
                task.cancel()
            with pytest.raises((TimeoutError, asyncio.CancelledError)):
                await task
            await worker.close()
            assert worker.process.poll() is not None
        assert not ledger.workers and not ledger.cuda_bytes

    asyncio.run(run())
    owner.release()


def test_game_context_refusal_preserves_cache_and_sampling_rng(tmp_path):
    owner, label, execution, _, model, binding, _ = saved(tmp_path)
    options = execution.policy_options
    policy = GameStructuredPolicyV3.from_artifact(Path(options["checkpoint"]), options["expected_manifest_digest"],
        model, binding, ACTION_HASH, action_contract=LAYOUT.action_contract, rules=RULES)
    _, frame, bank, game, component, _, _ = generated(execution)
    before = policy.rng.get_state().clone()
    for invalid in (component, replace(game, report=replace(game.report, config_hash="f" * 64))):
        with pytest.raises(ContractError):
            policy.select_frame(frame, bank.view(), invalid)
        assert policy.cache.position == 0 and torch.equal(before, policy.rng.get_state())
    global_rng = torch.get_rng_state().clone()
    selected = policy.select_frame(frame, bank.view(), game)
    assert selected.policy_version == label and math.isfinite(selected.logprob) and math.isfinite(selected.value)
    assert torch.equal(global_rng, torch.get_rng_state())
    assert policy.cache.position == 1
    owner.release()


@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_game_empty_hand_empty_events_have_finite_singleton_selection(tmp_path, temporal):
    owner, _, execution, _, model, binding, _ = saved(tmp_path, temporal)
    _, _, frames, bank, _ = raw_inputs(count=1, empty=True, events=False)
    frame = frames[0]
    _, costs, _, _ = own_game_worker_settings_v3(temporal, ACTIONS, execution.policy_options, rules=RULES)
    game = PublicCandidateGeneratorV3(ACTIONS, rules=RULES, clock=lambda: 0.0).generate_game(
        frame, bank.view(), random.Random(17), costs)
    policy = GameStructuredPolicyV3.from_artifact(Path(execution.policy_options["checkpoint"]),
        execution.policy_options["expected_manifest_digest"], model, binding, ACTION_HASH,
        action_contract=LAYOUT.action_contract, rules=RULES)
    selected = policy.select_frame(frame, bank.view(), game)
    assert selected.validate(game.table, frame) == {"type": "pray"}
    assert selected.logprob == 0.0 and math.isfinite(selected.value)
    owner.release()


@pytest.mark.parametrize("mutation", ["manifest_purpose", "action_hash", "file_hash", "extra_field", "model_version"])
def test_crossed_sealed_metadata_refuses_without_external_reads(tmp_path, monkeypatch, mutation):
    owner, label, _, _, _, _, _ = saved(tmp_path)
    value = owner.journal_metadata(label)
    if mutation == "manifest_purpose":
        value["inference_manifest"]["purpose"] = "component"
    elif mutation == "action_hash":
        value["policy_options"]["action_config_hash"] = "0" * 64
    elif mutation == "file_hash":
        value["files"][0]["sha256"] = "0" * 64
    elif mutation == "extra_field":
        value["unexpected"] = True
    else:
        value["policy_options"]["model"]["version"] = "2"
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("metadata validation read external files"))
    with pytest.raises(ContractError):
        validate_neural_metadata(value, actions=ACTIONS, rules=RULES)
    owner.release()


@pytest.mark.parametrize("stats", [(None, None), (0.0, None), (None, 0.0)])
def test_neural_journal_requires_both_old_policy_statistics_before_decision_append(tmp_path, stats):
    owner, label, execution, _, _, _, _ = saved(tmp_path)
    _, frame, bank, game, _, caps, costs = generated(execution)
    directory = tmp_path / "journal"
    directory.mkdir()
    journal = CardInstanceJournal(directory, "episode", 1, label, run_id="run", game_id="game", actions=ACTIONS,
        bank_episode=bank.episode, caps=asdict(caps), numeric_costs=asdict(costs), model_id="mlp",
        runtime_evidence=receipt(), neural=owner.journal_metadata(label))
    delta = event_delta(frame.actor, frame.episode, 0, bank.view()._prefixes[0], bank.view().events(), layout=LAYOUT)
    selected = Selection(game.table.table_hash, 0, label, *stats)
    command = {"command_id": str(uuid.uuid4()), "expected_revision": frame.revision,
        "action": selected.validate(game.table, frame)}
    before = journal.bank.boundary
    with pytest.raises(ContractError):
        journal.begin_frame(frame, delta, game, selected, command, {})
    assert journal.bank.boundary == before and not journal.pending
    journal.finish("incomplete", None)
    owner.release()


def test_neural_journal_uses_actual_window_and_collection_requires_complete_sealed_bindings(tmp_path):
    owner, label, execution, _, _, _, _ = saved(tmp_path, window=2)
    _, _, bank, _, _, caps, costs = generated(execution)
    directory = tmp_path / "journal"
    directory.mkdir()
    journal = CardInstanceJournal(directory, "episode", 1, label, run_id="run", game_id="game", actions=ACTIONS,
        bank_episode=bank.episode, caps=asdict(caps), numeric_costs=asdict(costs), model_id="mlp",
        runtime_evidence=receipt(), neural=owner.journal_metadata(label))
    assert journal.header["aggregate_window"] == 2
    assert journal.header["token_signature"] != journal.header["layout_signature"]
    journal.finish("incomplete", None)
    assert read_card_instance_journal(journal.final_path, actions=ACTIONS).header["neural"] == owner.journal_metadata(label)
    options = {"actions": ACTIONS, "config": {}, "runtime_evidence": [receipt()], "schedule": [{}], "inventory": [{}],
        "policies": [label], "bots_revision": "fixture"}
    sealed = owner.finalize(deadline=time.monotonic() + 30)
    for missing in ({"neural_inputs": sealed}, {"policy_bindings": {label: execution}},
                    {"policy_bindings": {label: execution}, "neural_inputs": {**sealed, "policies": {}}}):
        with pytest.raises(ContractError):
            collection_metadata(**options, **missing)
    owner.release()


@pytest.mark.parametrize("mutation", ["missing_options", "null_options", "baseline_options", "factory", "old_layout",
    "bad_visibility", "cpu_none", "cpu_nonempty"])
def test_tagged_neural_initialization_refuses_before_torch_factory_import(tmp_path, mutation):
    owner, _, execution, _, _, _, _ = saved(tmp_path)
    _, _, _, _, _, caps, costs = generated(execution)
    from gf_common.card_instance_rules import rule_signature
    from gf_common.card_instance_runtime import game_binding
    initial = {"protocol_version": 3, "model_id": "mlp", "sampling_seed": 17, "actions": asdict(ACTIONS),
        "rules": asdict(RULES), "caps": asdict(caps), "numeric_costs": asdict(costs),
        **game_binding(ACTIONS.action_contract, rule_signature(ACTIONS.action_contract, RULES)),
        "policy_options": copy.deepcopy(execution.policy_options)}
    if mutation == "missing_options":
        del initial["policy_options"]
    elif mutation == "null_options":
        initial["policy_options"] = None
    elif mutation == "baseline_options":
        initial["model_id"] = "random"
    elif mutation == "factory":
        initial["factory"] = {"module": "arbitrary", "name": "factory"}
    elif mutation == "old_layout":
        initial["layout_signature"] = "72db3a64f6d087a5f889c4e3cf6b4eb46a8595fd334e1249ca2d51f76496e1ad"
    elif mutation == "cpu_none":
        initial["policy_options"]["cuda_visible_devices"] = None
    elif mutation == "cpu_nonempty":
        initial["policy_options"]["cuda_visible_devices"] = "0"
    else:
        initial["policy_options"]["cuda_visible_devices"] = "not-a-device"
    path = tmp_path / "init.json"
    path.write_text(json.dumps(initial))
    script = '''import importlib.abc, io, json, runpy, sys
from types import SimpleNamespace
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self, name, path=None, target=None):
  if name == 'torch' or name.startswith('torch.') or name == 'gf_models.structured_policy':
   raise AssertionError('invalid init reached Torch/factory import')
sys.meta_path.insert(0, NoTorch())
from gf_simulator_client.framing import read_frame, write_frame
with open(sys.argv[1]) as source: initial=json.load(source)
source, output=io.BytesIO(), io.BytesIO()
write_frame(source, initial);source.seek(0)
original_stdin, original_stdout=sys.stdin,sys.stdout
sys.stdin, sys.stdout=SimpleNamespace(buffer=source),SimpleNamespace(buffer=output)
runpy.run_module('gf_simulator_client.card_instance_bootstrap',run_name='__main__')
sys.stdin,sys.stdout=original_stdin,original_stdout
output.seek(0);reply=read_frame(output)
assert reply['worker_error'] == 'ContractError',reply
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script, str(path)], check=False, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    owner.release()


def test_retained_original_expectations_are_independent_of_all_returned_nested_metadata(tmp_path):
    owner, label, execution, _, _, _, _ = saved(tmp_path)
    original = owner.journal_metadata(label)
    execution.policy_options["model"]["heads"]["actor_hidden_sizes"][0] += 1
    returned = owner.journal_metadata(label)
    returned["inference_manifest"]["model"]["heads"]["actor_hidden_sizes"][0] += 2
    returned["policy_options"]["inference"]["device"] = "cuda:0"
    returned["files"][0]["identity"][0] += 1
    final = owner.finalize(deadline=time.monotonic() + 30)
    final["policies"][label]["inference_manifest"]["weights"]["files"].clear()
    assert owner.journal_metadata(label) == original
    assert owner.finalize(deadline=time.monotonic() + 30)["policies"][label] == original
    owner.release()


@pytest.mark.parametrize("phase", ["before_child_load", "after_child_load"])
def test_original_mutation_around_real_worker_load_never_yields_healthy_inputs(tmp_path, phase):
    owner, _, execution, _, _, _, _ = saved(tmp_path)
    path = next(Path(execution.policy_options["checkpoint"]).glob("weights-*.npz"))

    async def run():
        ledger = ReservationLedger()
        if phase == "before_child_load":
            path.write_bytes(path.read_bytes() + b"mutated")
        worker = CardInstanceParticipantWorker("mlp", 17, ACTIONS, ledger=ledger,
            policy_options=execution.policy_options, artifact_inputs=owner)
        try:
            if phase == "before_child_load":
                with pytest.raises((ContractError, RuntimeError)):
                    await worker.initialize()
            else:
                await worker.initialize()
                path.write_bytes(path.read_bytes() + b"mutated")
        finally:
            await worker.close()
        assert not ledger.workers and not ledger.cuda_bytes and worker.process.poll() is not None

    asyncio.run(run())
    with pytest.raises((ContractError, OSError)):
        owner.finalize(deadline=time.monotonic() + 30)
    with pytest.raises(ContractError):
        owner.journal_metadata("any")
    owner.release()


def test_late_deadline_failure_is_sticky_and_release_still_has_no_io(tmp_path):
    owner, label, _, _, _, _, _ = saved(tmp_path)
    with pytest.raises(TimeoutError):
        owner.finalize(deadline=time.monotonic() - 1)
    with pytest.raises(ContractError):
        owner.finalize(deadline=time.monotonic() + 30)
    with pytest.raises(ContractError):
        owner.journal_metadata(label)
    owner.release()
