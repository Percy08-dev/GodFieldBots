"""Mandatory profile discrimination; component flags never authorize real sources."""
from __future__ import annotations

import copy
from dataclasses import asdict

import pytest

from gf_common.action_contract import engineering_contract, require_qualified_contract
from gf_common.domain import ContractError, canonical
from gf_models.arrays import atomic_json, sha256
from gf_models.structured_artifacts import load_inference, read_resume
from gf_models.structured_binding import input_binding, safe_options_v2
from gf_models.structured_policy import StructuredPolicy
from gf_training.structured_rollout import read_rollout, write_rollout
from tests.test_structured_learning import learner_for, records_for, rollout_for

FIELDS = ("action_contract", "purpose", "synthetic_capacity")


def damage(record, mutation):
    if mutation.startswith("missing_"):
        del record[mutation.removeprefix("missing_")]
    elif mutation == "unknown_contract":
        record["action_contract"]["profile_id"] = "unknown"
    elif mutation == "zero_contract":
        record["action_contract"]["adopted_rules_sha256"] = "0" * 64
    elif mutation == "qualified_claim":
        record["purpose"] = "qualified_game"
    elif mutation == "capacity_flip":
        record["synthetic_capacity"] = not record["synthetic_capacity"]
    else:
        record["synthetic_capacity"] = 1


@pytest.mark.parametrize("format", ["inference", "resume", "rollout"])
@pytest.mark.parametrize("mutation", ["missing_action_contract", "missing_purpose", "missing_synthetic_capacity",
    "unknown_contract", "zero_contract", "qualified_claim", "capacity_flip", "capacity_integer"])
def test_rehashed_metadata_rejects_before_archive_decode_network_or_raw_owner(make_input, tmp_path, monkeypatch, format, mutation):
    import gf_models.structured_artifacts as artifacts
    import gf_models.structured_network as network
    import gf_training.structured_rollout as rollout_module
    learner = learner_for()
    directory = tmp_path / format
    if format == "rollout":
        manifest = write_rollout(directory, rollout_for(make_input, learner, count=1))
        path = directory / "rollout.json"
    else:
        learner.save(directory)
        import json
        path = directory / f"{format}.json"
        manifest = json.loads(path.read_text())
    damage(manifest, mutation)
    atomic_json(path, manifest)
    def forbidden(*args, **kwargs):
        pytest.fail("incompatible metadata reached archive decoding/network/raw token ownership")
    for module in (artifacts, rollout_module):
        monkeypatch.setattr(module, "inspect_bundle", forbidden)
        monkeypatch.setattr(module, "read_bundle", forbidden)
    monkeypatch.setattr(network, "make_network", forbidden)
    monkeypatch.setattr(rollout_module, "DecisionFrame", forbidden)
    expected = {"action_contract": engineering_contract(), "purpose": "component", "synthetic_capacity": True}
    with pytest.raises(ContractError):
        if format == "inference":
            load_inference(directory, expected_digest=sha256(path), model=learner.model, binding=learner.binding, **expected)
        elif format == "resume":
            read_resume(directory, learner.model, learner.binding, learner.resume_identity, **expected)
        else:
            read_rollout(directory, sha256(path), learner.model, **expected)


@pytest.mark.parametrize("level", ["episode", "table"])
@pytest.mark.parametrize("field", FIELDS)
def test_rollout_requires_each_nested_classification_before_any_decode(make_input, tmp_path, monkeypatch, level, field):
    import gf_training.structured_rollout as module
    learner = learner_for()
    manifest = write_rollout(tmp_path / "rollout", rollout_for(make_input, learner, count=1))
    target = manifest["episodes"][0] if level == "episode" else manifest["episodes"][0]["frames"][0]
    del target[field]
    atomic_json(tmp_path / "rollout/rollout.json", manifest)
    def forbidden(*args, **kwargs):
        pytest.fail("nested missing classification reached archive ownership")
    monkeypatch.setattr(module, "inspect_bundle", forbidden)
    with pytest.raises(ContractError):
        read_rollout(tmp_path / "rollout", sha256(tmp_path / "rollout/rollout.json"), learner.model,
                     action_contract=engineering_contract(), purpose="component", synthetic_capacity=True)


def test_capacity_relabel_resume_rejects_before_live_adam_parameters_or_rng_change(make_input, tmp_path):
    synthetic = learner_for()
    synthetic.save(tmp_path / "synthetic")
    live = learner_for(synthetic_capacity=False)
    live.update(rollout_for(make_input, live, count=1))
    parameters = {name: value.clone() for name, value in live.network.state_dict().items()}
    optimizer = copy.deepcopy(live.optimizer.state_dict())
    rng, shuffle, steps = live.torch.get_rng_state().clone(), canonical(live.shuffle.bit_generator.state), live.steps
    with pytest.raises(ContractError, match="classification"):
        live.restore(tmp_path / "synthetic")
    assert live.steps == steps and canonical(live.shuffle.bit_generator.state) == shuffle
    assert live.torch.equal(live.torch.get_rng_state(), rng)
    for name, value in live.network.state_dict().items():
        assert live.torch.equal(value, parameters[name])
    current = live.optimizer.state_dict()
    assert current["param_groups"] == optimizer["param_groups"] and set(current["state"]) == set(optimizer["state"])
    for index, state in current["state"].items():
        for key, value in state.items():
            assert live.torch.equal(value, optimizer["state"][index][key])


def test_policy_and_learner_reject_table_or_rollout_classification_before_tokens_or_parity(make_input, monkeypatch):
    import gf_models.structured_policy as module
    synthetic, ordinary = learner_for(), learner_for(synthetic_capacity=False)
    rollout = rollout_for(make_input, ordinary, count=1)
    def forbidden(*args, **kwargs):
        pytest.fail("mismatched classification reached tokenization/parity")
    monkeypatch.setattr(rollout, "endpoint", forbidden)
    with pytest.raises(ContractError, match="classification"):
        synthetic.admit(rollout)
    record = records_for(make_input, ordinary, count=1)
    policy = StructuredPolicy(synthetic.network, synthetic.model, synthetic.binding, "a" * 64,
                              action_contract=engineering_contract(), purpose="component", synthetic_capacity=True)
    policy.reset(17)
    rng = policy.rng.get_state().clone()
    monkeypatch.setattr(module, "tokenize_frame", forbidden)
    with pytest.raises(ContractError, match="classification"):
        policy.select_frame(record.frames[0], record.bank, record.tables[0])
    assert policy.cache.position == 0 and synthetic.torch.equal(policy.rng.get_state(), rng)


def test_artifact_capacity_override_is_not_resource_override(tmp_path, monkeypatch):
    import gf_models.structured_network as module
    learner = learner_for()
    identity = learner.save(tmp_path / "synthetic")
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic artifact reached False policy construction")
    monkeypatch.setattr(module, "make_network", forbidden)
    with pytest.raises(ContractError, match="classification"):
        StructuredPolicy.from_artifact(tmp_path / "synthetic", identity["inference_digest"], learner.model, learner.binding,
            learner.action_config_hash, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)


def test_neural_runtime_factory_rejects_capacity_artifact_before_spawn(tmp_path, monkeypatch):
    import gf_simulator_client.structured_worker as module
    from gf_models.structured_baselines import make_structured_policy
    from gf_simulator_client.structured_resources import ReservationLedger
    from tests.test_structured_protocol import action_settings
    learner = learner_for()
    options = {"version": "2", "checkpoint": str(tmp_path / "unread"), "expected_manifest_digest": "a" * 64,
               "model": asdict(learner.model), "inference": asdict(learner.binding), "cuda_visible_devices": "",
               **input_binding(engineering_contract(), "component", True)}
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic artifact spawned a game-proposal runtime")
    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    ledger = ReservationLedger()
    with pytest.raises(ContractError, match="classification"):
        module.StructuredParticipantWorker(make_structured_policy, "mlp", 0, action_settings(), ledger=ledger, policy_options=options)
    assert not ledger.workers


@pytest.mark.parametrize("mutation", ["missing_action_contract", "missing_purpose", "missing_synthetic_capacity",
    "unknown_contract", "zero_contract", "qualified_claim", "capacity_flip", "capacity_integer"])
def test_safe_options_and_journal_require_full_descriptor_and_game_classification(make_input, tmp_path, monkeypatch, mutation):
    import gf_training.structured_journal as module
    from tests.test_structured_protocol import journal_setup, neural_component
    _, options = neural_component(tmp_path)
    damage(options, mutation)
    with pytest.raises(ContractError):
        safe_options_v2(options, action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
    directory = tmp_path / "journal"
    directory.mkdir()
    writer, frame, delta, generated, selection, command = journal_setup(make_input, directory)
    writer.begin_frame(frame, delta, generated, selection, command, {})
    writer.finish("incomplete", None)
    import json
    rows = [json.loads(line) for line in writer.final_path.read_text().splitlines()]
    damage(rows[0], mutation)
    def forbidden(*args, **kwargs):
        pytest.fail("missing journal profile reached frame/sidecar decode")
    monkeypatch.setattr(module, "parse_frame", forbidden)
    monkeypatch.setattr(module, "read_candidates", forbidden)
    with pytest.raises(ContractError):
        module.read_structured_journal(writer.final_path, records=rows, action_contract=engineering_contract(),
                                       purpose="component", synthetic_capacity=False)


def test_component_classification_never_upgrades_original_source_authority():
    # Runtime/component bindings can validate without granting qualified-source
    # authority. A caller's intended purpose cannot alter the owner descriptor.
    input_binding(engineering_contract(), "component", False)
    with pytest.raises(ContractError, match="engineering_only"):
        require_qualified_contract(engineering_contract())
    with pytest.raises(ContractError, match="engineering_only"):
        input_binding(engineering_contract(), "qualified_game", False)


@pytest.mark.parametrize("surface", ["ready", "reply", "report", "table"])
@pytest.mark.parametrize("mutation", ["missing_action_contract", "zero_contract"])
def test_participant_metadata_failure_reaps_child_and_holds_no_reservation(make_input, monkeypatch, surface, mutation):
    import asyncio
    import random

    from gf_common.diverse_actions import PublicCandidateGeneratorV2
    from gf_common.structured import EventBank
    from gf_common.structured_wire import generated_dict
    from gf_models.structured_baselines import make_structured_policy
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_simulator_client.structured_worker import StructuredParticipantWorker
    from tests.test_structured_protocol import action_settings, own_frame
    async def run():
        ledger = ReservationLedger()
        worker = StructuredParticipantWorker(make_structured_policy, "random", 17, action_settings(), ledger=ledger)
        if surface == "ready":
            reply = {"protocol_version": 2, "kind": "ready", "policy_version": worker.expected_policy, "model_load_ms": 0.,
                     **input_binding(engineering_contract(), "component", False)}
            damage(reply, mutation)
        else:
            await worker.initialize()
            bank = EventBank(1, "metadata-component")
            frame, delta = own_frame(make_input(hand=("bronze-club",)), bank, 0)
            rng = random.Random(0)
            rng.setstate(worker._search_state)
            generated = PublicCandidateGeneratorV2(action_settings(), clock=lambda: 0.).generate(frame, bank.view(), rng, worker.costs)
            policy = make_structured_policy("random", action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
            selected = policy.select_frame(frame, bank.view(), generated.table)
            reply = {"protocol_version": 2, "kind": "selection", "generated": generated_dict(generated), "selection": asdict(selected),
                     "timings": {"generation_ms": 0., "policy_ms": 0., "worker_decision_ms": 0.},
                     **input_binding(engineering_contract(), "component", False)}
            damage(reply if surface == "reply" else reply["generated"][surface], mutation)
        async def injected(*args, **kwargs):
            return reply
        monkeypatch.setattr(worker, "_roundtrip", injected)
        with pytest.raises(ContractError):
            if surface == "ready":
                await worker.initialize()
            else:
                await worker.select_frame(frame, delta)
        assert worker.closed and worker.process.poll() is not None and worker.lease.released
        assert not ledger.workers
    asyncio.run(run())


def test_game_journal_and_factory_reject_synthetic_capacity_before_owned_io(tmp_path):
    from gf_common.domain import digest
    from gf_models.structured_baselines import make_structured_policy
    from gf_models.structured_runtime import own_worker_settings
    from gf_training.structured_journal import StructuredJournal
    from tests.test_structured_protocol import action_settings
    settings = action_settings()
    caps, costs, _, _ = own_worker_settings("random", settings, None)
    with pytest.raises(ContractError, match="synthetic capacity"):
        StructuredJournal(tmp_path, "component", 1, "random-structured-v1", run_id="run", game_id="game",
            action_config_hash=digest(asdict(settings)), bank_episode="own", caps=asdict(caps), numeric_costs=asdict(costs),
            model_id="random", model_version="1", action_contract=settings.action_contract, purpose="component", synthetic_capacity=True)
    with pytest.raises(ContractError, match="synthetic capacity"):
        make_structured_policy("mlp", {"checkpoint": str(tmp_path / "never-read")}, "a" * 64,
            action_contract=settings.action_contract, purpose="component", synthetic_capacity=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("saved_capacity", [False, True])
def test_loaded_network_cannot_be_wrapped_or_resaved_under_opposite_classification(tmp_path, monkeypatch, saved_capacity):
    import gf_models.structured_artifacts as artifacts
    learner = learner_for(synthetic_capacity=saved_capacity)
    identity = learner.save(tmp_path / "original")
    expected = input_binding(learner.action_contract, learner.purpose, saved_capacity)
    network, _ = load_inference(tmp_path / "original", expected_digest=identity["inference_digest"],
        model=learner.model, binding=learner.binding, action_config_hash=learner.action_config_hash,
        action_contract=learner.action_contract, purpose=learner.purpose, synthetic_capacity=saved_capacity)
    tag = network._structured_input_binding
    before = {name: tensor.clone() for name, tensor in network.state_dict().items()}
    def forbidden(*args, **kwargs):
        pytest.fail("network relabel reached cache allocation or tensor copying")
    monkeypatch.setattr(network, "new_cache", forbidden)
    monkeypatch.setattr(artifacts, "_cpu_array", forbidden)
    with pytest.raises(ContractError, match="network input.*classification"):
        StructuredPolicy(network, learner.model, learner.binding, identity["inference_digest"],
            action_contract=learner.action_contract, purpose="component", synthetic_capacity=not saved_capacity)
    destination = tmp_path / "relabelled"
    with pytest.raises(ContractError, match="network input.*classification"):
        artifacts.save_inference(destination, network, learner.model, learner.action_config_hash,
            action_contract=learner.action_contract, purpose="component", synthetic_capacity=not saved_capacity)
    assert not destination.exists() and network._structured_input_binding is tag
    for name, value in network.state_dict().items():
        assert learner.torch.equal(value, before[name])
    assert asdict(tag) == expected


@pytest.mark.parametrize("bad_binding", ["contract", "purpose"])
def test_loaded_network_invalid_contract_or_purpose_rejects_before_cache_or_copy(tmp_path, monkeypatch, bad_binding):
    import gf_models.structured_artifacts as artifacts
    learner = learner_for()
    identity = learner.save(tmp_path / "original")
    network, _ = load_inference(tmp_path / "original", action_contract=learner.action_contract,
                                purpose="component", synthetic_capacity=True)
    contract = learner.action_contract if bad_binding == "purpose" else {**asdict(learner.action_contract), "profile_id": "other"}
    purpose = "qualified_game" if bad_binding == "purpose" else "component"
    def forbidden(*args, **kwargs):
        pytest.fail("invalid network binding reached cache allocation or tensor copying")
    monkeypatch.setattr(network, "new_cache", forbidden)
    monkeypatch.setattr(artifacts, "_cpu_array", forbidden)
    with pytest.raises(ContractError):
        StructuredPolicy(network, learner.model, learner.binding, identity["inference_digest"],
            action_contract=contract, purpose=purpose, synthetic_capacity=True)
    with pytest.raises(ContractError):
        artifacts.save_inference(tmp_path / "never", network, learner.model, learner.action_config_hash,
            action_contract=contract, purpose=purpose, synthetic_capacity=True)
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize("capacity", [False, True])
def test_fresh_component_network_binds_once_and_matching_roundtrip_keeps_immutable_tag(tmp_path, capacity):
    from dataclasses import FrozenInstanceError

    from gf_models.structured_artifacts import save_inference
    from gf_models.structured_network import make_network
    learner = learner_for(synthetic_capacity=capacity)
    network = make_network(learner.model, 17, learner.binding)
    assert not hasattr(network, "_structured_input_binding")
    names = set(network.state_dict())
    policy = StructuredPolicy(network, learner.model, learner.binding, "a" * 64,
        action_contract=learner.action_contract, purpose="component", synthetic_capacity=capacity)
    tag = network._structured_input_binding
    with pytest.raises(FrozenInstanceError):
        tag.synthetic_capacity = not capacity
    assert set(network.state_dict()) == names  # tag is not a model parameter/buffer.
    save_inference(tmp_path / "fresh", network, learner.model, learner.action_config_hash,
        action_contract=learner.action_contract, purpose="component", synthetic_capacity=capacity)
    assert network._structured_input_binding is tag and policy.cache.position == 0
    loaded, _ = load_inference(tmp_path / "fresh", action_contract=learner.action_contract,
                               purpose="component", synthetic_capacity=capacity)
    assert loaded._structured_input_binding == tag
    matching = StructuredPolicy(loaded, learner.model, learner.binding, sha256(tmp_path / "fresh/inference.json"),
        action_contract=learner.action_contract, purpose="component", synthetic_capacity=capacity)
    assert matching.cache.position == 0


def test_resume_and_learner_reject_tag_relabel_before_live_state_mutation(tmp_path, monkeypatch):
    import gf_models.structured_artifacts as artifacts
    learner = learner_for()
    learner.save(tmp_path / "original")
    before_files = {path.name: path.read_bytes() for path in (tmp_path / "original").iterdir()}
    tag, rng = learner.network._structured_input_binding, learner.torch.get_rng_state().clone()
    before = {name: value.clone() for name, value in learner.network.state_dict().items()}
    def forbidden(*args, **kwargs):
        pytest.fail("network relabel reached tensor/RNG archive decoding")
    monkeypatch.setattr(artifacts, "_cpu_array", forbidden)
    monkeypatch.setattr(artifacts, "read_resume", forbidden)
    with pytest.raises(ContractError, match="network input.*classification"):
        artifacts.save_resume(tmp_path / "original", learner.network, learner.optimizer, learner.shuffle, learner.steps,
            learner.model, learner.binding, learner.resume_identity, action_contract=learner.action_contract,
            purpose="component", synthetic_capacity=False)
    with pytest.raises(ContractError, match="network input.*classification"):
        artifacts.restore_resume(tmp_path / "original", learner.network, learner.optimizer, learner.shuffle,
            learner.model, learner.binding, learner.resume_identity, action_contract=learner.action_contract,
            purpose="component", synthetic_capacity=False)
    learner.synthetic_capacity = False
    learner.input_binding = input_binding(learner.action_contract, "component", False)
    # The authoritative Config can reject the public relabel before a tag check.
    with pytest.raises(ContractError, match="classification"):
        learner.save(tmp_path / "relabelled")
    with pytest.raises(ContractError, match="classification"):
        learner.restore(tmp_path / "original")
    assert not (tmp_path / "relabelled").exists()
    assert learner.network._structured_input_binding is tag and not learner.optimizer.state and learner.steps == 0
    assert learner.torch.equal(learner.torch.get_rng_state(), rng)
    assert {path.name: path.read_bytes() for path in (tmp_path / "original").iterdir()} == before_files
    for name, value in learner.network.state_dict().items():
        assert learner.torch.equal(value, before[name])


def test_network_tag_relabel_rejects_before_learner_parity_or_policy_cache_step(make_input, monkeypatch):
    import gf_models.structured_policy as module
    learner = learner_for()
    ordinary = learner_for(synthetic_capacity=False)
    rollout = rollout_for(make_input, ordinary, count=1)
    policy = StructuredPolicy(learner.network, learner.model, learner.binding, "a" * 64,
        action_contract=learner.action_contract, purpose="component", synthetic_capacity=True)
    learner.synthetic_capacity = policy.synthetic_capacity = False
    learner.input_binding = policy.input_binding = input_binding(learner.action_contract, "component", False)
    def forbidden(*args, **kwargs):
        pytest.fail("tag mismatch reached parity/tokenization")
    monkeypatch.setattr(rollout, "endpoint", forbidden)
    monkeypatch.setattr(module, "tokenize_frame", forbidden)
    with pytest.raises(ContractError, match="classification"):
        learner.admit(rollout)
    record = records_for(make_input, ordinary, count=1)
    rng = policy.rng.get_state().clone()
    with pytest.raises(ContractError, match="network input.*classification"):
        policy.select_frame(record.frames[0], record.bank, record.tables[0])
    assert policy.cache.position == 0 and learner.steps == 0 and learner.torch.equal(policy.rng.get_state(), rng)
