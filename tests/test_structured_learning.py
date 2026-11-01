"""Synthetic structured numeric/artifact components; no gameplay/quality gate."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace

import numpy as np
import pytest

from gf_common.domain import ContractError, canonical, digest
from gf_common.structured import CandidateTable, DecisionFrame, EventBank, EventDelta, Selection
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_common.validation import decision_input
from gf_models.arrays import atomic_json, sha256
from gf_models.sharded_arrays import inspect_bundle, read_bundle, read_json, write_bundle
from gf_models.structured_artifacts import inspect_inference, load_inference
from gf_models.structured_config import InferenceBindingV2, architecture_identity
from gf_models.structured_policy import StructuredPolicy
from gf_training.structured_numeric import Learner
from gf_training.structured_rollout import (
    EpisodeRecords,
    StructuredRollout,
    read_rollout,
    write_rollout,
)
from tests.conftest import public_options
from tests.test_structured_network import small_model

ACTION_HASH = digest({"synthetic": "component-only-actions2"})
RESUME_ID = digest({"synthetic": "component-only-config"})


def learner_for(temporal="mlp", hand="attention", device="cpu"):
    model = small_model(hand=hand, temporal=temporal)
    model = replace(model, temporal=replace(model.temporal, max_decisions=8, sequence_length=2),
                    event=replace(model.event, max_interval=8))
    return Learner(model, 17, InferenceBindingV2(device=device), ACTION_HASH, RESUME_ID)


def records_for(make_input, learner, *, count=3, empty=False, events=True, candidates=4, episode="episode"):
    torch = learner.torch
    value = make_input(hand=() if empty else ("bronze-club", "steel-armor"),
                       learned=() if empty else ("fireball", "aura"))
    bank = EventBank(1, episode)
    frames, tables, selections, history = [], [], [], []
    for index in range(count):
        start = bank.boundary.end
        if events:
            event = {"type": "resource_changed", "player": 1, "hp_delta": -index, "mp_delta": 1, "money_delta": 0}
            bank.append(EventDelta(1, episode, start, bank.boundary.prefix_hash, [event]))
            history.append(event)
        obs = value.to_dict()["observation"]
        obs["state_revision"] = index
        obs["own"]["hp"] -= index
        current = decision_input(obs, history, None, value.catalog, public_options(obs, value.catalog))
        frame = DecisionFrame.from_input(current, bank, start)
        if empty:
            actions = [{"type": "pray"}]
        elif candidates == 4:
            actions = [{"type": "play_sequence", "target": 2, "ordered_elements": [frame.references[0]]},
                       {"type": "play_sequence", "target": 1, "ordered_elements": [frame.references[2]] * 2},
                       {"type": "play_sequence", "target": None, "ordered_elements": [frame.references[0], frame.references[3]]},
                       {"type": "discard", "instance_ids": [2]}]
        else:
            # Repeated rows are an explicitly synthetic normalization stress,
            # not output from the deduplicating production candidate generator.
            actions = [{"type": "play_sequence", "target": 2, "ordered_elements": [frame.references[0]]}] * candidates
        table = CandidateTable.from_actions(frame, actions)
        frames.append(frame)
        tables.append(table)
        tokens = tuple(tokenize_frame(item, bank.view(), aggregate_window=learner.model.event.aggregate_window) for item in frames)
        encoded_table = tokenize_table(table, frame, learner.model.max_positions, aggregate_window=learner.model.event.aggregate_window)
        with torch.no_grad():
            logs, value_tensor, _ = learner.network(tokens, encoded_table)
        learner.synchronize()
        selected = index % len(actions)
        selections.append(Selection(table.table_hash, selected, "synthetic-component-policy", float(logs[selected].cpu()), float(value_tensor.cpu())))
    return EpisodeRecords(bank.view(), tuple(frames), tuple(tables), tuple(selections), "natural", {"win": {"winner": 1}})


def rollout_for(make_input, learner, **settings):
    return StructuredRollout((records_for(make_input, learner, **settings),), learner.model, ACTION_HASH)


def assert_same_learners(left, right):
    torch = left.torch
    assert left.steps == right.steps
    assert canonical(left.shuffle.bit_generator.state) == canonical(right.shuffle.bit_generator.state)
    for (name, original), (other, restored) in zip(left.network.named_parameters(), right.network.named_parameters(), strict=True):
        assert name == other
        torch.testing.assert_close(original, restored, atol=0, rtol=0)
        state1, state2 = left.optimizer.state.get(original), right.optimizer.state.get(restored)
        assert bool(state1) == bool(state2), name
        if state1:
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(state1[key], state2[key], atol=0, rtol=0, msg=name)


def test_shards_boundaries_closed_inventory_and_corruption(tmp_path):
    arrays = {"b": np.arange(30, dtype=np.int64), "a": np.ones(30, dtype=np.float32), "c": np.zeros(32, dtype=np.uint8)}
    manifest = write_bundle(tmp_path, "test", arrays, shard_bytes=272)
    assert len(manifest["shards"]) == 2
    assert set(manifest["files"]) == {"test-000000.npz", "test-000000.json", "test-000001.npz", "test-000001.json"}
    for name, array in read_bundle(tmp_path, manifest).items():
        np.testing.assert_array_equal(array, arrays[name])
        assert not array.flags.writeable
    for changes in ({"schema_version": 2}, {"decoded_bytes": True}, {"prefix": "../outside"}):
        with pytest.raises(ContractError):
            inspect_bundle(tmp_path, {**manifest, **changes})
    altered = copy.deepcopy(manifest)
    altered["shards"][1]["file"] = "../test-000001.npz"
    with pytest.raises(ContractError):
        read_bundle(tmp_path, altered)
    (tmp_path / "test-000001.npz").write_bytes(b"corrupt")
    with pytest.raises(ContractError, match="hash"):
        read_bundle(tmp_path, manifest)


def test_tensor_size_validation_precedes_array_scan_and_manifest_allocation(tmp_path, monkeypatch):
    import gf_models.sharded_arrays as module
    huge = np.lib.stride_tricks.as_strided(np.zeros(1, np.uint8), shape=(64 * 1024**2 + 1,), strides=(0,))
    def forbidden(*args, **kwargs):
        pytest.fail("oversized tensor triggered an array scan/allocation")
    monkeypatch.setattr(module.np, "isfinite", forbidden)
    with pytest.raises(ContractError, match="oversized"):
        write_bundle(tmp_path, "huge", {"tensor": huge})
    assert not list(tmp_path.iterdir())


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version":1,"schema_version":3}')
    with pytest.raises(ContractError, match="manifest"):
        read_json(path)


@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_typed_rollout_preserves_tokens_banks_targets_and_future_tail(make_input, tmp_path, temporal):
    learner = learner_for(temporal)
    record = records_for(make_input, learner)
    # Later participant-visible terminal tail belongs to the episode bank, never
    # to an earlier own-decision interval or aggregate.
    bank = EventBank(1, record.bank.episode)
    bank.append(EventDelta(1, bank.episode, 0, bank.boundary.prefix_hash, record.bank.events()))
    bank.append(EventDelta(1, bank.episode, bank.boundary.end, bank.boundary.prefix_hash,
                           [{"type": "player_defeated", "player": 2}]))
    record = EpisodeRecords(bank.view(), record.frames, record.tables, record.selections, record.end_kind, record.outcome)
    original = StructuredRollout((record,), learner.model, ACTION_HASH)
    manifest = write_rollout(tmp_path / "rollout", original, shard_bytes=32768)
    loaded = read_rollout(tmp_path / "rollout", sha256(tmp_path / "rollout" / "rollout.json"), learner.model)
    assert len(manifest["data"]["shards"]) > 1
    assert loaded.episodes[0].records.bank.boundary.end == 4
    assert loaded.episodes[0].frames[-1].event_end == 3
    assert loaded.episodes[0].frames == original.episodes[0].frames
    assert loaded.episodes[0].tables == original.episodes[0].tables
    np.testing.assert_array_equal(loaded.episodes[0].advantage, original.episodes[0].advantage)
    np.testing.assert_array_equal(loaded.episodes[0].target, original.episodes[0].target)
    assert learner.parity(loaded)["decisions"] == len(loaded)


@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_two_updates_equal_one_save_restore_resume_all_named_state(make_input, tmp_path, temporal):
    uninterrupted = learner_for(temporal)
    first = rollout_for(make_input, uninterrupted)
    uninterrupted.update(first)
    identity = uninterrupted.save(tmp_path / "generation", shard_bytes=400000)
    assert all(not PathName.startswith("/") for PathName in identity["files"])
    restored = learner_for(temporal)
    restored.restore(tmp_path / "generation")
    assert_same_learners(uninterrupted, restored)
    next_rollout = rollout_for(make_input, uninterrupted, episode="next")
    uninterrupted.update(next_rollout)
    restored.update(next_rollout)
    assert_same_learners(uninterrupted, restored)


@pytest.mark.parametrize("hand", ["direct", "deepsets", "attention"])
def test_optimizer_absence_survives_resume_then_first_interval_gradient(make_input, tmp_path, hand):
    learner = learner_for(hand=hand)
    learner.update(rollout_for(make_input, learner, empty=True, events=False))
    event_parameters = {name: parameter for name, parameter in learner.network.named_parameters() if name.startswith("event_encoder")}
    assert all(parameter not in learner.optimizer.state for parameter in event_parameters.values())
    learner.save(tmp_path / "generation")
    restored = learner_for(hand=hand)
    restored.restore(tmp_path / "generation")
    assert_same_learners(learner, restored)
    assert all(parameter not in restored.optimizer.state for name, parameter in restored.network.named_parameters() if name.startswith("event_encoder"))
    richer = rollout_for(make_input, learner, episode="later")
    learner.update(richer)
    restored.update(richer)
    assert_same_learners(learner, restored)
    for name, parameter in restored.network.named_parameters():
        if name.startswith("event_encoder"):
            assert restored.optimizer.state[parameter]["step"].item() == 4


def test_whole_batch_parity_failure_precedes_every_adam_change(make_input):
    learner = learner_for("gru")
    record = records_for(make_input, learner)
    selections = list(record.selections)
    selections[-1] = replace(selections[-1], value=selections[-1].value + .1)
    invalid = EpisodeRecords(record.bank, record.frames, record.tables, tuple(selections), record.end_kind, record.outcome)
    rollout = StructuredRollout((invalid,), learner.model, ACTION_HASH)
    before = {name: value.detach().clone() for name, value in learner.network.state_dict().items()}
    with pytest.raises(ContractError, match="parity"):
        learner.update(rollout)
    assert learner.steps == 0 and not learner.optimizer.state
    for name, tensor in learner.network.state_dict().items():
        assert learner.torch.equal(tensor, before[name])


def test_resource_override_for_new_inference_but_exact_resume_binding(make_input, tmp_path):
    learner = learner_for()
    learner.update(rollout_for(make_input, learner))
    identity = learner.save(tmp_path / "generation")
    different = replace(learner.model, resources=replace(learner.model.resources, max_owned_bytes=2 * 1024**3))
    assert architecture_identity(different) == architecture_identity(learner.model)
    network, _ = load_inference(tmp_path / "generation", model=different, binding=learner.binding,
                                 expected_digest=identity["inference_digest"])
    assert network.config == different
    resumed = Learner(different, 17, learner.binding, ACTION_HASH, RESUME_ID)
    with pytest.raises(ContractError, match="resume"):
        resumed.restore(tmp_path / "generation")
    wrong_binding = Learner(learner.model, 17, replace(learner.binding, candidate_chunk=2), ACTION_HASH, RESUME_ID)
    with pytest.raises(ContractError, match="resume"):
        wrong_binding.restore(tmp_path / "generation")
    with pytest.raises(ContractError, match="architecture"):
        inspect_inference(tmp_path / "generation", model=replace(learner.model, event=replace(learner.model.event, aggregate_window=7)))


def test_inference_cache_full_sampling_reset_and_ten_thousand_candidates(make_input, tmp_path):
    learner = learner_for("transformer")
    record = records_for(make_input, learner, count=1, candidates=10001)
    identity = learner.save(tmp_path / "generation")
    policy = StructuredPolicy.from_artifact(tmp_path / "generation", identity["inference_digest"], learner.model, learner.binding, ACTION_HASH)
    frame, table = record.frames[0], record.tables[0]
    policy.reset(71)
    first = policy.select_frame(frame, record.bank, table)
    assert first.logprob == pytest.approx(-np.log(10001), abs=1e-5)
    assert first.table_hash == table.table_hash
    policy.reset(71)
    assert policy.select_frame(frame, record.bank, table) == first
    assert identity["inference_digest"] in first.policy_version
    assert digest(asdict(learner.binding)) in first.policy_version


@pytest.mark.parametrize("kind", ["truncated", "aborted", "unsupported", "infrastructure_failure"])
def test_zero_decision_and_interrupted_closures_never_invent_targets(make_input, tmp_path, kind):
    learner = learner_for()
    bank = EventBank(1, "zero").view()
    zero = EpisodeRecords(bank, (), (), (), "natural", "draw")
    value = rollout_for(make_input, learner)
    aborted = EpisodeRecords(value.episodes[0].records.bank, value.episodes[0].records.frames,
                             value.episodes[0].records.tables, value.episodes[0].records.selections,
                             kind, reason="explicit-component-interruption")
    rollout = StructuredRollout((zero, aborted), learner.model, ACTION_HASH)
    assert not rollout.eligible and len(rollout) == 3
    assert all(episode.advantage is None and episode.target is None for episode in rollout.episodes)
    write_rollout(tmp_path / "rollout", rollout)
    restored = read_rollout(tmp_path / "rollout", sha256(tmp_path / "rollout" / "rollout.json"), learner.model)
    assert restored.episodes[0].records.outcome == "draw"
    assert restored.episodes[1].records.end_kind == kind
    assert len(restored) == 3
    with pytest.raises(ContractError, match="no learning targets"):
        restored.vector(0)
    with pytest.raises(ContractError, match="natural"):
        learner.update(restored)


def test_npy_headers_are_validated_before_numpy_decode(tmp_path, monkeypatch):
    import gf_models.sharded_arrays as module
    manifest = write_bundle(tmp_path, "headers", {"tensor": np.ones(1, np.uint8)})
    shard = manifest["shards"][0]
    path = tmp_path / shard["file"]
    np.savez(path, tensor=np.asarray([{"not": "a numeric tensor"}], dtype=object))
    shard["metadata"]["sha256"] = sha256(path)
    manifest["files"][path.name] = sha256(path)
    atomic_json(path.with_suffix(".json"), shard["metadata"])
    manifest["files"][path.with_suffix(".json").name] = sha256(path.with_suffix(".json"))
    def forbidden(*args, **kwargs):
        pytest.fail("NumPy decoded an object/header-mismatched tensor")
    monkeypatch.setattr(module.np, "load", forbidden)
    with pytest.raises(ContractError, match="header"):
        read_bundle(tmp_path, manifest)


def rewritten_rollout(directory, manifest, arrays):
    for name in manifest["data"]["files"]:
        (directory / name).unlink()
    manifest["data"] = write_bundle(directory, "rollout", arrays, shard_bytes=32768)
    atomic_json(directory / "rollout.json", manifest)
    return sha256(directory / "rollout.json")


def test_rehashed_raw_token_corruption_is_rejected_against_original_frame(make_input, tmp_path):
    learner = learner_for()
    rollout = rollout_for(make_input, learner)
    directory = tmp_path / "rollout"
    manifest = write_rollout(directory, rollout, shard_bytes=32768)
    arrays = {key: value.copy() for key, value in read_bundle(directory, manifest["data"]).items()}
    arrays["e.0.f.0.state_numerics"][4] += 1
    digest_after_tamper = rewritten_rollout(directory, manifest, arrays)
    with pytest.raises(ContractError, match="stored tokens"):
        read_rollout(directory, digest_after_tamper, learner.model)


def test_rehashed_foreign_slot_cannot_become_an_owned_candidate(make_input, tmp_path):
    learner = learner_for()
    rollout = rollout_for(make_input, learner, count=1)
    directory = tmp_path / "rollout"
    manifest = write_rollout(directory, rollout, shard_bytes=32768)
    arrays = {key: value.copy() for key, value in read_bundle(directory, manifest["data"]).items()}
    arrays["e.0.f.0.candidate.slots"][0] = 17
    frame = rollout.episodes[0].records.frames[0]
    original = rollout.episodes[0].records.tables[0]
    content = original._content()
    content["slots"] = tuple(map(int, arrays["e.0.f.0.candidate.slots"]))
    new_table_hash = digest(content)
    manifest["episodes"][0]["frames"][0]["table_hash"] = new_table_hash
    manifest["episodes"][0]["selections"][0]["table_hash"] = new_table_hash
    after_tamper = rewritten_rollout(directory, manifest, arrays)
    with pytest.raises(ContractError, match="slot"):
        read_rollout(directory, after_tamper, learner.model)
    assert len(frame.references) == 4


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("kind", "legacy-inference"),
                                          ("numeric_transform", "different"), ("generator_version", "v3"),
                                          ("reviewed_spec_hashes", {}), ("unexpected", 1)])
def test_inference_unknown_or_mixed_semantics_reject_before_network(make_input, tmp_path, monkeypatch, field, value):
    import gf_models.structured_network as module
    learner = learner_for()
    learner.save(tmp_path / "generation")
    manifest = read_json(tmp_path / "generation" / "inference.json")
    manifest[field] = value
    atomic_json(tmp_path / "generation" / "inference.json", manifest)
    def forbidden(*args, **kwargs):
        pytest.fail("network constructed before manifest compatibility check")
    monkeypatch.setattr(module, "make_network", forbidden)
    with pytest.raises(ContractError):
        load_inference(tmp_path / "generation", model=learner.model, binding=learner.binding)


def test_resume_rng_and_named_presence_exact_roundtrip(make_input, tmp_path):
    learner = learner_for("gru")
    learner.update(rollout_for(make_input, learner))
    learner.torch.rand(7)
    learner.shuffle.random(5)
    cpu_before = learner.torch.get_rng_state().clone()
    shuffle_before = canonical(learner.shuffle.bit_generator.state)
    learner.save(tmp_path / "generation")
    expected = learner.torch.rand(8)
    resumed = learner_for("gru")
    resumed.restore(tmp_path / "generation")
    assert resumed.torch.equal(resumed.torch.get_rng_state(), cpu_before)
    assert canonical(resumed.shuffle.bit_generator.state) == shuffle_before
    assert resumed.torch.equal(resumed.torch.rand(8), expected)
    resume = read_json(tmp_path / "generation" / "resume.json")
    assert set(resume["parameter_states"]) == set(dict(resumed.network.named_parameters()))
    assert all(type(state["present"]) is bool for state in resume["parameter_states"].values())


def test_resume_identity_and_runtime_mismatch_leave_live_learner_unchanged(make_input, tmp_path):
    learner = learner_for()
    learner.update(rollout_for(make_input, learner))
    learner.save(tmp_path / "generation")
    other = learner_for()
    before = {name: value.clone() for name, value in other.network.state_dict().items()}
    manifest = read_json(tmp_path / "generation" / "resume.json")
    manifest["runtime"]["python"] = "different"
    atomic_json(tmp_path / "generation" / "resume.json", manifest)
    with pytest.raises(ContractError, match="resume"):
        other.restore(tmp_path / "generation")
    assert other.steps == 0 and not other.optimizer.state
    for name, value in other.network.state_dict().items():
        assert other.torch.equal(value, before[name])


def test_mlp_validates_history_but_encodes_only_current_frame(make_input, monkeypatch):
    learner = learner_for()
    rollout = rollout_for(make_input, learner)
    frames, candidates = rollout.endpoint(2)
    work = []
    original = learner.network.encode_frame
    def counted(frame):
        work.append(frame.frame_hash)
        return original(frame)
    monkeypatch.setattr(learner.network, "encode_frame", counted)
    actual = learner.network(frames, candidates)
    assert work == [frames[-1].frame_hash]
    altered = replace(frames[0], state_numerics=(999,) + frames[0].state_numerics[1:])
    work.clear()
    other = learner.network((altered, *frames[1:]), candidates)
    assert work == [frames[-1].frame_hash]
    for expected, value in zip(actual, other, strict=True):
        learner.torch.testing.assert_close(expected, value, atol=0, rtol=0)
    wrong_window = replace(frames[0], aggregate_window=7)
    with pytest.raises(ContractError):
        learner.network((wrong_window, *frames[1:]), candidates)


def test_logical_shuffle_128_endpoint_clip_steps_preserved(make_input, monkeypatch):
    learner = learner_for(hand="direct")
    records = tuple(records_for(make_input, learner, count=1, events=False, episode=f"ep-{index}") for index in range(129))
    rollout = StructuredRollout(records, learner.model, ACTION_HASH)
    expected_rng = np.random.Generator(np.random.PCG64(17))
    expected = []
    for _ in range(4):
        expected.extend(rollout.endpoint(int(index))[0][-1].frame_hash for index in expected_rng.permutation(129))
    visited, clips = [], []
    actual_forward = learner.network.forward
    actual_parity = learner.parity
    recording = False
    def parity(batch):
        nonlocal recording
        result = actual_parity(batch)
        recording = True
        return result
    def forward(frames, candidates, **options):
        if recording:
            visited.append(frames[-1].frame_hash)
        return actual_forward(frames, candidates, **options)
    actual_clip = learner.torch.nn.utils.clip_grad_norm_
    def clip(*args, **kwargs):
        clips.append(len(visited))
        return actual_clip(*args, **kwargs)
    monkeypatch.setattr(learner, "parity", parity)
    monkeypatch.setattr(learner.network, "forward", forward)
    monkeypatch.setattr(learner.torch.nn.utils, "clip_grad_norm_", clip)
    learner.update(rollout)
    assert visited == expected
    assert clips == [128, 129, 257, 258, 386, 387, 515, 516]
    assert learner.steps == 8


def test_cpu_learner_and_resume_do_not_seed_or_snapshot_cuda_rng(make_input, tmp_path, monkeypatch):
    import torch
    def forbidden(*args, **kwargs):
        pytest.fail("CPU numeric component inspected/changed CUDA ownership")
    for name in ("manual_seed_all", "get_rng_state_all", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    learner = learner_for()
    learner.update(rollout_for(make_input, learner, count=1))
    learner.save(tmp_path / "generation")
    learner.restore(tmp_path / "generation")


def test_corrupt_resume_state_is_prevalidated_before_staged_network_or_rng_change(make_input, tmp_path, monkeypatch):
    import gf_models.structured_network as module
    learner = learner_for()
    learner.update(rollout_for(make_input, learner))
    directory = tmp_path / "generation"
    learner.save(directory)
    resume = read_json(directory / "resume.json")
    arrays = {name: value.copy() for name, value in read_bundle(directory, resume["state"]).items()}
    arrays["rng.torch_cpu"] = np.zeros(3, dtype=np.uint8)
    for name in resume["state"]["files"]:
        (directory / name).unlink()
    resume["state"] = write_bundle(directory, "resume", arrays)
    atomic_json(directory / "resume.json", resume)
    rng_before = learner.torch.get_rng_state().clone()
    shuffle_before = canonical(learner.shuffle.bit_generator.state)
    parameters_before = {name: tensor.clone() for name, tensor in learner.network.state_dict().items()}
    def forbidden(*args, **kwargs):
        pytest.fail("staged model constructed before resume shard/RNG validation")
    monkeypatch.setattr(module, "make_network", forbidden)
    with pytest.raises(ContractError, match="RNG"):
        learner.restore(directory)
    assert learner.torch.equal(learner.torch.get_rng_state(), rng_before)
    assert canonical(learner.shuffle.bit_generator.state) == shuffle_before
    for name, tensor in learner.network.state_dict().items():
        assert learner.torch.equal(tensor, parameters_before[name])


def test_staged_resume_accounts_live_learner_before_new_allocation(make_input, tmp_path, monkeypatch):
    import gf_training.structured_numeric as module
    learner = learner_for()
    learner.update(rollout_for(make_input, learner))
    directory = tmp_path / "generation"
    learner.save(directory)
    original = module.resource_estimate
    estimate = original(learner.model, learner.binding)["total_bytes"]
    learner.binding = replace(learner.binding, managed_budget_bytes=estimate)
    def forbidden(*args, **kwargs):
        pytest.fail("staged inference allocated before live+staged resource admission")
    monkeypatch.setattr(module, "load_inference", forbidden)
    with pytest.raises(ContractError, match="staged resume"):
        learner.restore(directory)


def test_bounded_manifest_read_binds_the_same_decoded_bytes(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"value":1}\n')
    original = sha256(path)
    assert read_json(path, expected_digest=original) == {"value": 1}
    path.write_text('{"value":2}\n')
    with pytest.raises(ContractError, match="hash"):
        read_json(path, expected_digest=original)


def test_rollout_owned_token_and_packing_admission_precedes_array_allocation(make_input, tmp_path, monkeypatch):
    import gf_training.structured_rollout as module
    learner = learner_for()
    record = records_for(make_input, learner)
    tiny = replace(learner.model, resources=replace(learner.model.resources, max_owned_bytes=1))
    def forbidden(*args, **kwargs):
        pytest.fail("rollout allocation/encoding preceded ownership admission")
    with monkeypatch.context() as patch:
        patch.setattr(module, "tokenize_frame", forbidden)
        with pytest.raises(ContractError, match="before encoding"):
            StructuredRollout((record,), tiny, ACTION_HASH)
    rollout = StructuredRollout((record,), learner.model, ACTION_HASH)
    rollout.model = replace(learner.model, resources=replace(learner.model.resources, max_owned_bytes=rollout.resident_bytes + 1))
    monkeypatch.setattr(module, "_array", forbidden)
    monkeypatch.setattr(module, "_blob", forbidden)
    with pytest.raises(ContractError, match="planning|packing"):
        write_rollout(tmp_path / "packing", rollout)


def test_whole_rollout_and_endpoint_are_jointly_admitted_before_reference_model(make_input, monkeypatch):
    import gf_training.structured_numeric as module
    learner = learner_for()
    rollout = rollout_for(make_input, learner)
    costs = []
    for index in range(len(rollout)):
        frames, candidates = rollout.endpoint(index)
        costs.append(module.resource_estimate(learner.model, learner.binding, decisions=len(frames),
            interval_events=max(len(frame.interval_numerics) for frame in frames),
            candidates=len(candidates.kinds), suffix_decisions=min(len(frames), learner.model.temporal.sequence_length))["total_bytes"])
    learner.binding = replace(learner.binding, managed_budget_bytes=max(costs) + rollout.resident_bytes - 1)
    def forbidden(*args, **kwargs):
        pytest.fail("reference network allocated before whole-input admission")
    monkeypatch.setattr(module, "make_network", forbidden)
    with pytest.raises(ContractError, match="whole rollout"):
        learner.parity(rollout)
    assert learner.steps == 0 and not learner.optimizer.state


def test_resume_shuffle_rng_is_closed_and_strict_before_state_publication(make_input, tmp_path):
    learner = learner_for()
    learner.save(tmp_path / "generation")
    path = tmp_path / "generation" / "resume.json"
    original = read_json(path)
    for changes in ({"unknown": 1}, {"has_uint32": True}, {"bit_generator": "other"},
                    {"state": {"state": 0, "inc": 2}}):
        manifest = copy.deepcopy(original)
        manifest["shuffle_rng"].update(changes)
        atomic_json(path, manifest)
        with pytest.raises(ContractError):
            learner.restore(tmp_path / "generation")
        assert learner.steps == 0 and not learner.optimizer.state


def test_new_production_modules_import_without_torch():
    import subprocess
    import sys
    code = ("import sys; import gf_models.sharded_arrays; import gf_models.structured_artifacts; "
            "import gf_models.structured_policy; import gf_training.structured_rollout; "
            "import gf_training.structured_numeric; assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("field,value", [("candidate_chunk", 1), ("max_candidates", 1000),
                                          ("managed_budget_bytes", 2 * 1024**3)])
def test_policy_rejects_relabelled_same_device_network_binding_before_cache(monkeypatch, field, value):
    learner = learner_for()
    incompatible = replace(learner.binding, **{field: value})
    def forbidden(*args, **kwargs):
        pytest.fail("cache allocated before captured network binding validation")
    monkeypatch.setattr(learner.network, "new_cache", forbidden)
    with pytest.raises(ContractError, match="config/runtime binding"):
        StructuredPolicy(learner.network, learner.model, incompatible, "a" * 64)


def test_policy_rejects_relabelled_active_resources_before_cache(monkeypatch):
    learner = learner_for()
    incompatible = replace(learner.model, resources=replace(learner.model.resources, max_owned_bytes=2 * 1024**3))
    assert architecture_identity(incompatible) == architecture_identity(learner.model)
    def forbidden(*args, **kwargs):
        pytest.fail("cache allocated before captured active resource validation")
    monkeypatch.setattr(learner.network, "new_cache", forbidden)
    with pytest.raises(ContractError, match="config/runtime binding"):
        StructuredPolicy(learner.network, incompatible, learner.binding, "a" * 64)


def test_matching_policy_binding_labels_actual_envelope_and_fresh_artifact_overrides(tmp_path):
    from gf_models.structured_config import resource_estimate
    learner = learner_for()
    policy = StructuredPolicy(learner.network, learner.model, learner.binding, "a" * 64)
    assert policy.reserved_bytes == resource_estimate(learner.model, learner.network.binding)["total_bytes"]
    assert policy.binding == learner.network.binding
    assert digest(asdict(learner.network.binding)) in policy.version
    identity = learner.save(tmp_path / "generation")
    active = replace(learner.model, resources=replace(learner.model.resources, max_owned_bytes=2 * 1024**3))
    binding = replace(learner.binding, candidate_chunk=1, max_candidates=1000, managed_budget_bytes=2 * 1024**3)
    fresh = StructuredPolicy.from_artifact(tmp_path / "generation", identity["inference_digest"], active, binding, ACTION_HASH)
    assert fresh.network.config == fresh.model == active
    assert fresh.network.binding == fresh.binding == binding
    assert fresh.reserved_bytes == resource_estimate(active, binding)["total_bytes"]
    assert digest(asdict(binding)) in fresh.version


@pytest.mark.parametrize("admit_metadata", [False, True])
def test_reader_joint_raw_owner_envelope_rejects_before_decode_or_frame_parse(make_input, tmp_path, monkeypatch, admit_metadata):
    import gf_training.structured_rollout as module
    learner = learner_for()
    rollout = rollout_for(make_input, learner, count=1, empty=True, events=False)
    directory = tmp_path / "rollout"
    manifest = write_rollout(directory, rollout)
    decoded = manifest["data"]["decoded_bytes"]
    descriptors = inspect_bundle(directory, manifest["data"])
    combined = module._read_reservation(manifest, descriptors)
    assert decoded < combined
    metadata_budget = 32 * module.json_bytes(manifest) + 65536
    assert metadata_budget < combined
    limit = (max(decoded, metadata_budget) + combined) // 2 if admit_metadata else max(decoded + 1, min(64000, combined - 1))
    active = replace(learner.model, resources=replace(learner.model.resources, max_owned_bytes=limit))
    inspect_bundle(directory, manifest["data"], max_decoded_bytes=limit)
    def forbidden(*args, **kwargs):
        pytest.fail("raw owner/decoded arrays allocated before joint reader admission")
    monkeypatch.setattr(module, "read_bundle", forbidden)
    monkeypatch.setattr(module, "DecisionFrame", forbidden)
    with pytest.raises(ContractError, match="reservation"):
        read_rollout(directory, sha256(directory / "rollout.json"), active)


@pytest.mark.parametrize("mismatch", ["pooling", "resources"])
def test_save_inference_rejects_same_shape_relabel_before_copy_or_directory(tmp_path, monkeypatch, mismatch):
    import gf_models.structured_artifacts as module
    from gf_models.structured_config import parameter_shapes
    learner = learner_for(hand="deepsets")
    changed = (replace(learner.model, hand=replace(learner.model.hand, pooling="max")) if mismatch == "pooling" else
               replace(learner.model, resources=replace(learner.model.resources, max_owned_bytes=2 * 1024**3)))
    assert parameter_shapes(learner.model) == parameter_shapes(changed)
    def forbidden(*args, **kwargs):
        pytest.fail("tensor copied before actual network semantics validation")
    monkeypatch.setattr(module, "_cpu_array", forbidden)
    destination = tmp_path / "invalid"
    with pytest.raises(ContractError, match="full model config"):
        module.save_inference(destination, learner.network, changed, ACTION_HASH)
    assert not destination.exists()


@pytest.mark.parametrize("mismatch", ["pooling", "resources", "binding"])
def test_save_resume_rejects_relabelled_actual_context_before_copy(make_input, tmp_path, monkeypatch, mismatch):
    import gf_models.structured_artifacts as module
    learner = learner_for(hand="deepsets")
    directory = tmp_path / "generation"
    module.save_inference(directory, learner.network, learner.model, ACTION_HASH)
    model, binding = learner.model, learner.binding
    if mismatch == "pooling":
        model = replace(model, hand=replace(model.hand, pooling="max"))
    elif mismatch == "resources":
        model = replace(model, resources=replace(model.resources, max_owned_bytes=2 * 1024**3))
    else:
        binding = replace(binding, candidate_chunk=1)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    def forbidden(*args, **kwargs):
        pytest.fail("resume tensor copied before actual context validation")
    monkeypatch.setattr(module, "_cpu_array", forbidden)
    with pytest.raises(ContractError, match="mismatch"):
        module.save_resume(directory, learner.network, learner.optimizer, learner.shuffle, learner.steps,
                           model, binding, RESUME_ID)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


@pytest.mark.parametrize("option,value", [("lr", .001), ("betas", (.8, .999)), ("eps", 1e-6),
    ("weight_decay", .01), ("amsgrad", True), ("maximize", True), ("foreach", False),
    ("capturable", True), ("differentiable", True), ("fused", False),
    ("decoupled_weight_decay", True), ("unknown", 1)])
def test_save_rejects_noncanonical_adam_actual_options_before_publication(tmp_path, monkeypatch, option, value):
    import gf_models.structured_artifacts as module
    learner = learner_for()
    learner.optimizer.param_groups[0][option] = value
    def forbidden(*args, **kwargs):
        pytest.fail("Adam option mismatch copied state before validation")
    monkeypatch.setattr(module, "_cpu_array", forbidden)
    destination = tmp_path / "invalid"
    with pytest.raises(ContractError, match="Adam"):
        learner.save(destination)
    assert not destination.exists()


@pytest.mark.parametrize("mismatch", ["missing", "reordered", "foreign", "foreign_state", "defaults", "wrong_type", "extra_group"])
def test_save_rejects_optimizer_parameter_and_type_ownership(tmp_path, monkeypatch, mismatch):
    import gf_models.structured_artifacts as module
    learner = learner_for()
    torch = learner.torch
    parameters = list(learner.network.parameters())
    if mismatch == "missing":
        learner.optimizer.param_groups[0]["params"] = parameters[:-1]
    elif mismatch == "reordered":
        learner.optimizer.param_groups[0]["params"] = list(reversed(parameters))
    elif mismatch == "foreign":
        learner.optimizer.param_groups[0]["params"] = [torch.nn.Parameter(torch.zeros_like(parameters[0])), *parameters[1:]]
    elif mismatch == "foreign_state":
        learner.optimizer.state[torch.nn.Parameter(torch.ones(1))] = {}
    elif mismatch == "defaults":
        learner.optimizer.defaults["lr"] = .001
    elif mismatch == "wrong_type":
        learner.optimizer = torch.optim.SGD(parameters, lr=.0003)
    else:
        learner.optimizer.param_groups.append(dict(learner.optimizer.param_groups[0]))
    def forbidden(*args, **kwargs):
        pytest.fail("invalid optimizer copied state before ownership validation")
    monkeypatch.setattr(module, "_cpu_array", forbidden)
    destination = tmp_path / "invalid"
    with pytest.raises(ContractError, match="Adam"):
        learner.save(destination)
    assert not destination.exists()


def test_negative_rehashed_adam_variance_rejects_without_live_state_change(make_input, tmp_path, monkeypatch):
    import gf_training.structured_numeric as module
    learner = learner_for()
    learner.update(rollout_for(make_input, learner))
    directory = tmp_path / "generation"
    learner.save(directory)
    manifest = read_json(directory / "resume.json")
    arrays = {name: value.copy() for name, value in read_bundle(directory, manifest["state"]).items()}
    variance_name = next(name for name in arrays if name.endswith(".exp_avg_sq"))
    arrays[variance_name].flat[0] = -1
    for name in manifest["state"]["files"]:
        (directory / name).unlink()
    manifest["state"] = write_bundle(directory, "resume", arrays)
    atomic_json(directory / "resume.json", manifest)
    torch = learner.torch
    before_weights = {name: value.clone() for name, value in learner.network.state_dict().items()}
    before_states = {name: {key: value.clone() for key, value in learner.optimizer.state[parameter].items()}
                     for name, parameter in learner.network.named_parameters() if parameter in learner.optimizer.state}
    before_rng, before_shuffle, before_steps = torch.get_rng_state().clone(), canonical(learner.shuffle.bit_generator.state), learner.steps
    def forbidden(*args, **kwargs):
        pytest.fail("staged model allocated before negative variance rejection")
    monkeypatch.setattr(module, "load_inference", forbidden)
    with pytest.raises(ContractError, match="negative Adam variance"):
        learner.restore(directory)
    assert learner.steps == before_steps and torch.equal(torch.get_rng_state(), before_rng)
    assert canonical(learner.shuffle.bit_generator.state) == before_shuffle
    for name, value in learner.network.state_dict().items():
        assert torch.equal(value, before_weights[name])
    for name, parameter in learner.network.named_parameters():
        if name in before_states:
            for key, value in learner.optimizer.state[parameter].items():
                assert torch.equal(value, before_states[name][key])


@pytest.mark.parametrize("collision", ["owned-000001.npz", "owned-000001.json", "owned-000001.json.tmp"])
def test_all_shard_destinations_preflight_dangling_symlinks_before_any_write(tmp_path, collision):
    directory = tmp_path / "owned"
    directory.mkdir()
    outside = tmp_path / "outside-must-stay-absent"
    link = directory / collision
    link.symlink_to(outside)
    with pytest.raises(ContractError, match="preexisting"):
        write_bundle(directory, "owned", {"a": np.zeros(8, np.uint8), "b": np.zeros(8, np.uint8)}, shard_bytes=8)
    assert not outside.exists() and link.is_symlink()
    assert {path.name for path in directory.iterdir()} == {collision}


@pytest.mark.parametrize("collision", ["resume.json", "resume.json.tmp"])
def test_resume_manifest_destinations_preflight_dangling_symlinks(make_input, tmp_path, collision):
    import gf_models.structured_artifacts as module
    learner = learner_for()
    directory = tmp_path / "owned"
    module.save_inference(directory, learner.network, learner.model, ACTION_HASH)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    outside = tmp_path / "outside-must-stay-absent"
    (directory / collision).symlink_to(outside)
    with pytest.raises(ContractError, match="preexisting"):
        module.save_resume(directory, learner.network, learner.optimizer, learner.shuffle, learner.steps,
                           learner.model, learner.binding, RESUME_ID)
    assert not outside.exists()
    assert all((directory / name).read_bytes() == value for name, value in before.items())
    assert not list(directory.glob("resume-*.npz"))


@pytest.mark.parametrize("manifest_name", ["inference.json", "inference.json.tmp", "rollout.json", "rollout.json.tmp"])
def test_new_artifact_directory_does_not_reinterpret_preexisting_manifest_links(make_input, tmp_path, manifest_name):
    import gf_models.structured_artifacts as module
    learner = learner_for()
    directory = tmp_path / "owned"
    directory.mkdir()
    outside = tmp_path / "outside-must-stay-absent"
    (directory / manifest_name).symlink_to(outside)
    with pytest.raises(ContractError):
        if manifest_name.startswith("inference"):
            module.save_inference(directory, learner.network, learner.model, ACTION_HASH)
        else:
            write_rollout(directory, rollout_for(make_input, learner))
    assert not outside.exists()
    assert {path.name for path in directory.iterdir()} == {manifest_name}


def test_actual_1024_frame_sharded_persistence_and_1025_early_rejection(make_input, tmp_path, monkeypatch):
    import gf_training.structured_rollout as module
    from gf_models.structured_config import ModelConfigV2
    value = make_input(hand=())
    bank = EventBank(1, "synthetic-persistence-capacity")
    frames, tables, selections = [], [], []
    for revision in range(1025):
        raw = value.to_dict()
        raw["observation"]["state_revision"] = revision
        raw["action_options"]["state_revision"] = revision
        current = replace(value, observation=raw["observation"], action_options=raw["action_options"])
        frame = DecisionFrame.from_input(current, bank, 0)
        table = CandidateTable.from_actions(frame, [{"type": "pray"}])
        frames.append(frame)
        tables.append(table)
        selections.append(Selection(table.table_hash, 0, "synthetic-persistence-only", 0., 0.))
    model = ModelConfigV2()
    records = EpisodeRecords(bank.view(), tuple(frames[:1024]), tuple(tables[:1024]), tuple(selections[:1024]), "natural", "draw")
    rollout = StructuredRollout((records,), model, ACTION_HASH)
    assert len(rollout) == 1024
    directory = tmp_path / "maximum"
    manifest = write_rollout(directory, rollout, shard_bytes=1024**2)
    assert len(manifest["data"]["shards"]) > 1
    assert manifest["data"]["tensor_count"] == 1024 * (len(module.FRAME_FIELDS) + len(module.EVENT_FIELDS) + len(module.CANDIDATE_FIELDS) + 1) + 5
    assert all(sum(item["bytes"] for item in shard["metadata"]["arrays"].values()) <= 1024**2 for shard in manifest["data"]["shards"])
    loaded = read_rollout(directory, sha256(directory / "rollout.json"), model)
    assert len(loaded) == 1024 and len(loaded.episodes[0].records.frames) == 1024
    assert loaded.endpoint(1023)[0][-1].frame_hash == frames[1023].frame_hash
    assert loaded.episodes[0].frames == rollout.episodes[0].frames
    assert loaded.episodes[0].tables == rollout.episodes[0].tables
    too_many = EpisodeRecords(bank.view(), tuple(frames), tuple(tables), tuple(selections), "natural", "draw")
    def forbidden(*args, **kwargs):
        pytest.fail("1025-frame input allocated/decoded before capacity rejection")
    with monkeypatch.context() as patch:
        patch.setattr(module, "tokenize_frame", forbidden)
        with pytest.raises(ContractError, match="1024"):
            StructuredRollout((too_many,), model, ACTION_HASH)
    # An untrusted manifest must hit the same bound before decoding even one
    # typed shard or constructing a raw DecisionFrame.
    manifest["episodes"][0]["frames"].append(copy.deepcopy(manifest["episodes"][0]["frames"][-1]))
    manifest["episodes"][0]["selections"].append(copy.deepcopy(manifest["episodes"][0]["selections"][-1]))
    atomic_json(directory / "rollout.json", manifest)
    monkeypatch.setattr(module, "read_bundle", forbidden)
    monkeypatch.setattr(module, "DecisionFrame", forbidden)
    with pytest.raises(ContractError, match="1024"):
        read_rollout(directory, sha256(directory / "rollout.json"), model)


def test_complete_rollout_manifest_plan_checks_descriptors_before_packing(make_input, tmp_path, monkeypatch):
    import gf_models.sharded_arrays as shards
    import gf_training.structured_rollout as module
    from gf_models.sharded_arrays import json_bytes
    model = learner_for().model
    model = replace(model, temporal=replace(model.temporal, max_decisions=64))
    learner = Learner(model, 17, InferenceBindingV2(), ACTION_HASH, RESUME_ID)
    rollout = rollout_for(make_input, learner, count=50, events=False)
    original_directory = tmp_path / "original"
    manifest = write_rollout(original_directory, rollout)
    full_size = (original_directory / "rollout.json").stat().st_size
    assert full_size == json_bytes(manifest)
    old_incomplete_size = len(canonical(manifest["episodes"]).encode()) + 65536
    assert old_incomplete_size < full_size
    scaled_limit = (old_incomplete_size + full_size) // 2
    monkeypatch.setattr(shards, "MAX_MANIFEST_BYTES", scaled_limit)
    monkeypatch.setattr(module, "MAX_MANIFEST_BYTES", scaled_limit)
    def forbidden(*args, **kwargs):
        pytest.fail("tensor packed/written before complete manifest size admission")
    with monkeypatch.context() as patch:
        patch.setattr(module, "_array", forbidden)
        patch.setattr(module, "_blob", forbidden)
        patch.setattr(module, "write_bundle", forbidden)
        with pytest.raises(ContractError, match="complete JSON manifest"):
            write_rollout(tmp_path / "too-large", rollout)
    assert not list((tmp_path / "too-large").iterdir())
    with pytest.raises(ContractError, match="reservation"):
        shards.read_json(original_directory / "rollout.json")
    # Writer and reader agree on the exact boundary, including final newline.
    monkeypatch.setattr(shards, "MAX_MANIFEST_BYTES", full_size)
    monkeypatch.setattr(module, "MAX_MANIFEST_BYTES", full_size)
    matching = write_rollout(tmp_path / "exact-bound", rollout)
    assert (tmp_path / "exact-bound" / "rollout.json").stat().st_size == full_size
    assert shards.read_json(tmp_path / "exact-bound" / "rollout.json") == matching


def raw_record_with_interval(make_input, count):
    value = make_input(hand=())
    bank = EventBank(1, "raw-interval-caps")
    events = tuple({"type": "action_accepted", "player": 1} for _ in range(count))
    bank.append(EventDelta(1, bank.episode, 0, bank.boundary.prefix_hash, events))
    frame = DecisionFrame.from_input(replace(value, events=events), bank, 0)
    table = CandidateTable.from_actions(frame, [{"type": "pray"}])
    selection = Selection(table.table_hash, 0, "synthetic-unencoded", 0., 0.)
    return EpisodeRecords(bank.view(), (frame,), (table,), (selection,), "natural", "draw")


def test_policy_known_interval_and_candidate_caps_precede_tokenization(make_input, monkeypatch):
    import gf_models.structured_policy as module
    base = learner_for()
    interval_record = raw_record_with_interval(make_input, 3)
    model = replace(base.model, event=replace(base.model.event, max_interval=2))
    narrow = Learner(model, 17, base.binding, ACTION_HASH, RESUME_ID)
    interval_policy = StructuredPolicy(narrow.network, model, narrow.binding, "a" * 64)
    record = records_for(make_input, base, count=1)
    one_binding = replace(base.binding, max_candidates=1)
    one = Learner(base.model, 17, one_binding, ACTION_HASH, RESUME_ID)
    one_policy = StructuredPolicy(one.network, one.model, one.binding, "b" * 64)
    def forbidden(*args, **kwargs):
        pytest.fail("known policy scalar cap was checked after tokenization")
    monkeypatch.setattr(module, "tokenize_frame", forbidden)
    monkeypatch.setattr(module, "tokenize_table", forbidden)
    with pytest.raises(ContractError):
        interval_policy.select_frame(interval_record.frames[0], interval_record.bank, interval_record.tables[0])
    with pytest.raises(ContractError):
        one_policy.select_frame(record.frames[0], record.bank, record.tables[0])
    assert interval_policy.cache.position == one_policy.cache.position == 0


def test_rollout_known_caps_and_candidate_tuple_owners_precede_tokenization(make_input, monkeypatch):
    import gf_training.structured_rollout as module
    learner = learner_for()
    interval = raw_record_with_interval(make_input, 3)
    record = records_for(make_input, learner, count=1, events=False, candidates=10001)
    raw_bytes = module._owned_bytes((record,))
    limit = raw_bytes + 64001
    narrow = replace(learner.model, resources=replace(learner.model.resources, max_owned_bytes=limit))
    def forbidden(*args, **kwargs):
        pytest.fail("known rollout cap/tuple ownership checked after tokenization")
    monkeypatch.setattr(module, "tokenize_frame", forbidden)
    monkeypatch.setattr(module, "tokenize_table", forbidden)
    with pytest.raises(ContractError, match="before encoding"):
        StructuredRollout((record,), narrow, ACTION_HASH)
    interval_model = replace(learner.model, event=replace(learner.model.event, max_interval=2))
    with pytest.raises(ContractError):
        StructuredRollout((interval,), interval_model, ACTION_HASH)


def test_actual_adam_option_types_and_weight_dtype_are_checked_before_copy(tmp_path, monkeypatch):
    import gf_models.structured_artifacts as module
    learner = learner_for()
    learner.optimizer.param_groups[0]["betas"] = [.9, .999]
    def forbidden(*args, **kwargs):
        pytest.fail("noncanonical save context reached tensor copy")
    monkeypatch.setattr(module, "_cpu_array", forbidden)
    with pytest.raises(ContractError, match="Adam option types"):
        learner.save(tmp_path / "bad-betas")
    learner.optimizer.param_groups[0]["betas"] = (.9, .999)
    learner.network.double()
    with pytest.raises(ContractError, match="FP32"):
        module.save_inference(tmp_path / "bad-dtype", learner.network, learner.model, ACTION_HASH)
    assert not (tmp_path / "bad-betas").exists() and not (tmp_path / "bad-dtype").exists()


def test_known_episode_and_position_caps_leave_policy_cache_unchanged_before_tokens(make_input, monkeypatch):
    import gf_models.structured_policy as module
    learner = learner_for()
    record = records_for(make_input, learner)
    capped_model = replace(learner.model, event=replace(learner.model.event, aggregate_window=2, max_interval=2, max_episode=2))
    capped = Learner(capped_model, 17, learner.binding, ACTION_HASH, RESUME_ID)
    policy = StructuredPolicy(capped.network, capped.model, capped.binding, "a" * 64)
    for index in range(2):
        policy.select_frame(record.frames[index], record.bank, record.tables[index])
    position_model = replace(learner.model, max_positions=1)
    narrow = Learner(position_model, 17, learner.binding, ACTION_HASH, RESUME_ID)
    narrow_policy = StructuredPolicy(narrow.network, narrow.model, narrow.binding, "b" * 64)
    def forbidden(*args, **kwargs):
        pytest.fail("known episode/position cap checked after tokenization")
    monkeypatch.setattr(module, "tokenize_frame", forbidden)
    monkeypatch.setattr(module, "tokenize_table", forbidden)
    with pytest.raises(ContractError):
        policy.select_frame(record.frames[-1], record.bank, record.tables[-1])
    with pytest.raises(ContractError, match="position capacity"):
        narrow_policy.select_frame(record.frames[0], record.bank, record.tables[0])
    assert policy.cache.position == 2 and narrow_policy.cache.position == 0
