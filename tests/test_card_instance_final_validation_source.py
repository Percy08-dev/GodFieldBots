"""Baseline-only raw originals test sharing; NEVER formal NN Source admission."""
import copy
import hashlib
import time
from pathlib import Path

import pytest

from gf_common.card_instance_collection import FINAL_VALIDATION, PPO_BATCH
from gf_common.domain import ContractError, canonical, digest
from gf_training import card_instance_collection_source as manifest_kernel
from gf_training.card_instance_collection_source import inspect_game_manifest
from gf_training.structured_source import _inspect_component_collection, _SourceOwner
from tests.card_instance_final_validation_support import baseline_originals, rewrite_manifest


def source_owner(directory, loaded):
    return _SourceOwner(directory, loaded.config.preprocessing, time.monotonic() + 60, max_owned_bytes=1024**3)


@pytest.mark.parametrize('decisions', [False, True])
def test_shared_raw64_retains_zero_or_nonzero_own_endpoints_without_nn_or_formal_authority(tmp_path, decisions):
    directory = tmp_path / 'UNADMITTED-baseline-originals'
    loaded, _, _ = baseline_originals(directory, decisions=decisions)
    owner = source_owner(directory, loaded)
    result = _inspect_component_collection(owner, loaded.config.actions, 'validation', _game=True, _game_role=FINAL_VALIDATION)
    assert result.proof['games'] == 64 and result.proof['qualified'] is False
    assert result.proof['parent_qualification'] == 'not_performed'
    assert len(result.episodes) == 128
    selected = [row for row in result.episodes if result.manifest['linkage']['learner_seats'][row.episode_id] == row.viewer]
    assert len(selected) == 64 and sum(len(row.records.frames) for row in selected) == (64 if decisions else 0)
    assert {row.viewer for row in selected} == {1, 2}
    assert all(row.model_id in ('random', 'heuristic') for row in selected)
    with pytest.raises(ContractError):
        _inspect_component_collection(source_owner(directory, loaded), loaded.config.actions, 'validation', _game=True)
    with pytest.raises(ContractError):
        _inspect_component_collection(source_owner(directory, loaded), loaded.config.actions, 'validation', _game_role=FINAL_VALIDATION)


def test_extracted_default_manifest_matches_old16_outputs_and_owned_snapshots_exactly(tmp_path):
    directory = tmp_path / 'UNADMITTED-batch-originals'
    loaded, manifest, _ = baseline_originals(directory, role=PPO_BATCH)
    old, new = source_owner(directory, loaded), source_owner(directory, loaded)
    # Immutable pre-extraction oracle; remains independent after the old private
    # name becomes an alias. Expected SHA was pinned BEFORE applying that alias.
    fixture = Path(__file__).parent / 'fixtures/card_instance_manifest_ppo_batch_v1.txt'
    raw = fixture.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '7aee25eb917d6ed5abd73f6c44b8692ca9c43e72d42328c3bc2604c927b7815e'
    original = dict(manifest_kernel.__dict__)
    exec(compile(raw, str(fixture), 'exec'), original)  # noqa: S102 - fixed hash-pinned repository oracle
    old_manifest = original['_game_manifest']
    assert old_manifest(old, copy.deepcopy(manifest), loaded.config.actions, 'train') == inspect_game_manifest(new, copy.deepcopy(manifest), loaded.config.actions, 'train')
    assert canonical(old.files) == canonical(new.files)
    assert old.inventories == new.inventories and old.used == new.used
    result = _inspect_component_collection(source_owner(directory, loaded), loaded.config.actions, 'train', _game=True)
    assert result.proof['games'] == 16 and len(result.episodes) == 32


@pytest.mark.parametrize('mutation', ['role', 'linkage7', 'extra-link', 'generation', 'split', 'cohort', 'seat', 'runtime', 'report', 'replay', 'header', 'inventory', 'projection', 'invocation'])
def test_final64_raw_corruption_refuses_without_promoting_baseline_fixture(tmp_path, mutation):
    directory = tmp_path / 'UNADMITTED-baseline-originals'
    loaded, manifest, report = baseline_originals(directory)
    if mutation == 'role': manifest['collection_role'] = 'ppo-batch'
    elif mutation == 'linkage7':
        manifest['linkage'].pop('final_generation'); manifest['linkage']['collection_generation'] = 7
    elif mutation == 'extra-link': manifest['linkage']['collection_generation'] = 7
    elif mutation == 'generation': manifest['linkage']['final_generation'] = 1001
    elif mutation == 'split': manifest['config']['run']['split'] = 'train'; manifest['config_hash'] = digest(manifest['config'])
    elif mutation == 'cohort': manifest['episodes'].pop()
    elif mutation == 'seat': manifest['linkage']['learner_seats'][next(iter(manifest['linkage']['learner_seats']))] = 0
    elif mutation == 'runtime': manifest['runtime_evidence'][0]['executed'] = None
    elif mutation == 'report':
        report['natural_denominator'] = 63
        (directory / 'report.json').write_text(canonical(report))
    elif mutation == 'replay': next(directory.glob('*/replay-player-1.json')).unlink()
    elif mutation == 'header':
        path = next(directory.glob('*/player-1.jsonl'))
        path.write_text(path.read_text().replace('"sampling_seed":', '"sampling_seed":0,"UNADMITTED-original-seed":', 1))
    elif mutation == 'inventory': (directory / 'late-file.txt').write_text('UNADMITTED late inventory')
    elif mutation == 'projection': manifest['config_projection']['extra'] = True
    else: manifest['validation_invocation']['execution_deadline_monotonic'] += 1
    rewrite_manifest(directory, manifest)
    with pytest.raises((ContractError, FileNotFoundError)):
        _inspect_component_collection(source_owner(directory, loaded), loaded.config.actions, 'validation', _game=True, _game_role=FINAL_VALIDATION)


def test_raw_original_guard_detects_byte_restoration_and_extra_inventory(tmp_path):
    directory = tmp_path / 'UNADMITTED-baseline-originals'
    loaded, _, _ = baseline_originals(directory)
    owner = source_owner(directory, loaded)
    _inspect_component_collection(owner, loaded.config.actions, 'validation', _game=True, _game_role=FINAL_VALIDATION)
    path = next(directory.glob('*/player-1.jsonl'))
    raw = path.read_bytes()
    path.write_bytes(raw + b'\n'); path.write_bytes(raw)
    with pytest.raises(ContractError): owner.finalize()


def test_final_nn_metadata_producer_rejects_baseline_only_structural_fixture(tmp_path):
    from gf_common.domain import RuntimePolicyBinding
    from gf_training.card_instance_collection import final_validation_collection_metadata
    directory = tmp_path / 'UNADMITTED-baseline-originals'
    loaded, manifest, _ = baseline_originals(directory)
    with pytest.raises(ContractError):
        final_validation_collection_metadata(actions=loaded.config.actions, config=manifest['config'],
            runtime_evidence=manifest['runtime_evidence'], schedule=manifest['schedule'], inventory=manifest['files'],
            policies=manifest['policies'], bots_revision='UNADMITTED',
            policy_bindings={key: RuntimePolicyBinding(**value) for key, value in manifest['policy_bindings'].items()},
            neural_inputs=manifest['neural_inputs'], linkage=manifest['linkage'],
            config_projection=manifest['config_projection'], validation_invocation=manifest['validation_invocation'],
            elapsed_seconds=manifest['elapsed_seconds'])
