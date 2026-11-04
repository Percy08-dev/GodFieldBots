"""Real sealed inference artifacts; registry ownership remains separate from Source."""
import copy
import shutil
import time
from dataclasses import asdict, replace

import pytest

from gf_common.domain import ContractError, RuntimePolicyBinding, canonical
from gf_models.card_instance_artifacts import RetainedGameArtifacts
from gf_models.structured_artifacts import save_game_inference_v3
from gf_models.structured_binding import ConfiguredNeuralModelV3
from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_network import make_network
from tests.model3_support import LAYOUT, model3
from tests.test_card_instance_learning import ACTIONS, RULES


def actual_registry(tmp_path, count):
    models, registry = [], {'random-card-instance-v2': RuntimePolicyBinding('random'),
        'heuristic-card-instance-v2': RuntimePolicyBinding('heuristic')}
    generations = (11, 0, 10) if count == 3 else tuple(range(count))
    for index, generation in enumerate(generations):
        model, binding = model3(), InferenceBindingV2(max_candidates=64)
        network = make_network(model, 17, binding, layout=LAYOUT)
        import torch
        with torch.no_grad():
            next(network.parameters()).reshape(-1)[0].add_(index * .01)
        directory = tmp_path / f'gen-{generation:06}'
        from gf_common.domain import digest
        save_game_inference_v3(directory, network, model, digest(asdict(ACTIONS)),
            action_contract=ACTIONS.action_contract, rules=RULES)
        configured = ConfiguredNeuralModelV3(**{key: getattr(model, key) for key in model.__dataclass_fields__},
            checkpoint=str(directory), inference=binding)
        owner = RetainedGameArtifacts((configured,), ACTIONS, rules=RULES, max_owned_bytes=256 * 1024**2,
            preflight_deadline=time.monotonic() + 30, finalization_deadline=time.monotonic() + 60)
        try:
            label, execution = owner.binding(configured)
            registry[label] = execution
            models.append(configured)
        finally:
            owner.release()
    assert len(registry) == count + 2
    return tuple(models), registry


def own(registry):
    return RetainedGameArtifacts.for_registry(registry, ACTIONS, rules=RULES, max_owned_bytes=256 * 1024**2,
        preflight_deadline=time.monotonic() + 30, finalization_deadline=time.monotonic() + 60)


@pytest.mark.parametrize('count', [3, 6])
def test_registry_three_generation11_and_six_actual_artifacts_share_owned_lifecycle(tmp_path, count):
    models, registry = actual_registry(tmp_path, count)
    frozen = canonical({key: asdict(value) for key, value in registry.items()})
    owner = own(registry)
    try:
        expected = {key for key, value in registry.items() if value.policy_options is not None}
        assert set(owner.finalize(deadline=time.monotonic() + 30)['policies']) == expected
        for model in models:
            label, binding = owner.binding(replace(model, cuda_visible_devices=''))
            assert canonical(asdict(binding)) == canonical(asdict(registry[label]))
            owner.require_options(binding.model_id, binding.policy_options)
            metadata = owner.journal_metadata(label)
            assert metadata['policy_options']['expected_manifest_digest'] == registry[label].policy_options['expected_manifest_digest']
            metadata['policy_options']['expected_manifest_digest'] = 'f' * 64
            assert owner.journal_metadata(label)['policy_options'] == registry[label].policy_options
        assert canonical({key: asdict(value) for key, value in registry.items()}) == frozen
        assert len(owner._scan.files) >= count * 3
    finally:
        owner.release()
    with pytest.raises(ContractError, match='released'):
        owner.journal_metadata(next(iter(expected)))
    with pytest.raises(ContractError, match='one or two'):
        RetainedGameArtifacts(models, ACTIONS, rules=RULES, max_owned_bytes=256 * 1024**2,
            preflight_deadline=time.monotonic() + 30, finalization_deadline=time.monotonic() + 60)


def test_registry_seventh_actual_artifact_rejects_before_owner_or_file_reads(tmp_path, monkeypatch):
    _, registry = actual_registry(tmp_path, 7)
    def forbidden(*args, **kwargs):
        pytest.fail('seventh identity reached sealed owner allocation')
    monkeypatch.setattr(RetainedGameArtifacts, '_initialize', forbidden)
    with pytest.raises(ContractError):
        own(registry)


@pytest.mark.parametrize('mutation', ['baseline-label', 'baseline-model', 'unknown-model', 'neural-label', 'neural-options'])
def test_registry_closed_label_options_and_baseline_admission_before_files(tmp_path, monkeypatch, mutation):
    _, registry = actual_registry(tmp_path, 1)
    if mutation == 'baseline-label':
        registry['random'] = registry.pop('random-card-instance-v2')
    elif mutation == 'baseline-model':
        registry['random-card-instance-v2'] = RuntimePolicyBinding('unknown')
    else:
        label = next(key for key, value in registry.items() if value.policy_options is not None)
        value = registry.pop(label)
        if mutation == 'unknown-model':
            registry[label] = RuntimePolicyBinding('unknown', value.policy_options)
        elif mutation == 'neural-label':
            registry['copied-label'] = value
        else:
            options = copy.deepcopy(value.policy_options)
            options['extra'] = True
            registry[label] = RuntimePolicyBinding(value.model_id, options)
    def forbidden(*args, **kwargs):
        pytest.fail('malformed registry reached file ownership')
    monkeypatch.setattr(RetainedGameArtifacts, '_initialize', forbidden)
    with pytest.raises(ContractError):
        own(registry)


def test_registry_valid_different_current_artifact_cannot_rebind_expected_digest(tmp_path):
    models, registry = actual_registry(tmp_path, 3)
    source, destination = models[1].checkpoint, models[0].checkpoint
    from pathlib import Path
    for path in Path(source).iterdir():
        shutil.copyfile(path, Path(destination) / path.name)
    with pytest.raises(ContractError, match='hash|digest'):
        own(registry)


def test_registry_retains_all_original_files_until_final_recheck(tmp_path):
    models, registry = actual_registry(tmp_path, 3)
    owner = own(registry)
    try:
        from pathlib import Path
        path = next(Path(models[-1].checkpoint).glob('weights-*.npz'))
        path.write_bytes(path.read_bytes() + b'late mutation')
        with pytest.raises(ContractError):
            owner.finalize(deadline=time.monotonic() + 30)
        with pytest.raises(ContractError, match='failed'):
            owner.binding(models[0])
    finally:
        owner.release()


@pytest.mark.parametrize('count', [1, 3, 6])
def test_actual_source_binding_registry_enters_strict_environment_without_rebinding(tmp_path, monkeypatch, count):
    from gf_common.domain import clone, digest
    from gf_models.arrays import sha256
    from gf_simulator_client.card_instance_environment import CardInstanceSimulatorEnvironment
    from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
    from gf_simulator_client.environment import EnvironmentSettings
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.card_instance_journal import CardInstanceJournal
    from gf_training.card_instance_source import _binding
    from gf_training.lineage import checkpoint_path, run_lock
    from tests.card_instance_learning_support import initial_parent
    from tests.test_card_instance_baseline_runtime import receipt

    directory = tmp_path / 'parent'
    with run_lock(directory, create=True):
        learner, state, initial, _ = initial_parent(directory)
        generations = (0,) if count == 1 else (11, 0, 10) if count == 3 else (51, 10, 20, 30, 40, 50)
        registry = {'random-card-instance-v2': RuntimePolicyBinding('random'),
                    'heuristic-card-instance-v2': RuntimePolicyBinding('heuristic')}
        for index, generation in enumerate(generations):
            if generation == 0:
                record = initial[0]
            else:
                with learner.torch.no_grad():
                    next(learner.network.parameters()).reshape(-1)[0].add_(.01)
                learner.save(checkpoint_path(directory, generation))
                record = {'inference_sha256': sha256(checkpoint_path(directory, generation) / 'inference.json')}
            label, binding = _binding(directory, state, record, generation)
            registry[label] = binding
            # Dataclass tuples and JSON lists have identical wire bytes. The
            # actual producer must emit the closed JSON shape used by the owner.
            raw_model = asdict(learner.model)
            assert raw_model != clone(raw_model)
            assert canonical(raw_model) == canonical(clone(raw_model))
            assert digest(raw_model) == digest(binding.policy_options['model'])
        assert len(registry) == count + 2
        owner = own(registry)
        try:
            runtime = CardInstanceRuntimeEvidence(tmp_path / 'unused-binary', tmp_path / 'unused-source')
            monkeypatch.setattr(runtime, 'preflight', lambda: receipt())
            output = tmp_path / 'never-created'
            settings = EnvironmentSettings(str(runtime.binary), 'unused-server-config', output, 'fixture', ACTIONS,
                policy_bindings=registry)
            environment = CardInstanceSimulatorEnvironment(settings, CardInstanceJournal,
                ledger=ReservationLedger(), runtime_evidence=runtime, artifact_inputs=owner)
            assert environment.settings.policy_bindings is registry and not output.exists()
            for label, binding in registry.items():
                if binding.policy_options is not None:
                    assert owner.journal_metadata(label)['policy_options'] == binding.policy_options
                    assert isinstance(binding.policy_options['model']['state']['hidden_sizes'], list)
            label = next(key for key, value in registry.items() if value.policy_options is not None)
            original = registry[label]
            bad_options = clone(original.policy_options)
            bad_options['inference']['candidate_chunk'] += 1
            bad_registry = {**registry, label: RuntimePolicyBinding(original.model_id, bad_options)}
            assert canonical(bad_options) != canonical(original.policy_options)
            from dataclasses import replace
            with pytest.raises(ContractError, match='options differ'):
                CardInstanceSimulatorEnvironment(replace(settings, policy_bindings=bad_registry), CardInstanceJournal,
                    ledger=ReservationLedger(), runtime_evidence=runtime, artifact_inputs=owner)
            assert not output.exists()
        finally:
            owner.release()
