"""Real-source isolation, full validation and lazy fixture import regressions."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from gf_common.domain import ContractError
from gf_models.arrays import sha256
from gf_training.fixed_data import build_dataset, inspect_dataset
from gf_training.journal import atomic_json, project_journals
from gf_training.lineage import read_json, validate_chain
from tests.shared_ppo_sources import SharedPPOSources, _reseal


def test_support_imports_are_lazy_and_do_not_collect_test_modules():
    script = '''
import sys
import tests.conftest
import tests.shared_ppo_sources
assert 'torch' not in sys.modules
assert not any(name.startswith('tests.test_') for name in sys.modules)
'''
    subprocess.run([sys.executable, '-c', script], check=True, timeout=30)


def test_shared_full_source_copy_matches_original_rows_and_numeric(
    tmp_path, frozen_sources, _ppo_source_template,
):
    parents, validations = frozen_sources
    root = parents[0].parent
    assert root != _ppo_source_template.root
    _, original_train, original_validation = inspect_dataset(_ppo_source_template.root / 'original-dataset.json')
    _, train, validation = inspect_dataset(root / 'clone-dataset.json')
    assert train == original_train and validation == original_validation
    assert len(train) == 48 and len(validation) == 192
    assert [read_json(parent / 'run.json')['seed'] for parent in parents] == [17, 29, 43]
    assert all(read_json(value / 'manifest.json')['elapsed_seconds'] == 1. for value in validations)
    for relative, _ in _ppo_source_template.inventory:
        original, copied = _ppo_source_template.root / relative, root / relative
        assert copied.is_file() and not copied.is_symlink()
        assert original.stat().st_ino != copied.stat().st_ino
        # Policy inputs, accepted selections, IDs, Replay and all numeric and
        # checkpoint bytes remain exact; only specified management records change.
        if ('/artifacts/' in relative or '/numeric/' in relative
                or copied.suffix in ('.jsonl', '.log') or copied.name.startswith('replay-')):
            assert copied.read_bytes() == original.read_bytes()
    for parent, value in zip(parents, validations, strict=True):
        validate_chain(parent, require_artifacts=True)
        for directory in (parent / 'collections/batch-000000', value):
            manifest = read_json(directory / 'manifest.json')
            assert manifest['ppo']['parent_run_path'] == str(parent)
            for binding in manifest['policy_bindings'].values():
                if binding['policy_options']:
                    assert Path(binding['policy_options']['checkpoint']).is_relative_to(parent)
        update = read_json(parent / 'updates/000001.json')
        _, proof = project_journals(sorted((parent / 'collections').glob('*/*/player-*.jsonl')))
        assert update['source_proof']['source_set_sha256'] == proof['source_set_sha256']
    assert str(_ppo_source_template.root) not in (root / 'clone-dataset.json').read_text()


def test_mutated_copy_rejected_and_next_private_copy_unchanged(
    tmp_path, frozen_sources, _ppo_source_template,
):
    parents, validations = frozen_sources
    journal = next((parents[0] / 'collections').glob('*/*/player-1.jsonl'))
    original = journal.read_bytes()
    journal.write_bytes(original + b'\n')
    with pytest.raises((ContractError, ValueError)):
        build_dataset(parents, validations, tmp_path / 'mutated.json')
    later_parents, later_validations = _ppo_source_template.clone(tmp_path / 'later')
    later = later_parents[0] / journal.relative_to(parents[0])
    assert later.read_bytes() == original and later.stat().st_ino != journal.stat().st_ino
    build_dataset(later_parents, later_validations, tmp_path / 'later-dataset.json')


def test_template_admission_rejects_chain_valid_stale_original_proof(
    frozen_sources, monkeypatch,
):
    parents, _ = frozen_sources
    root = parents[0].parent
    update = read_json(parents[0] / 'updates/000001.json')
    update['source_proof']['source_set_sha256'] = '0' * 64
    _reseal(parents[0], update)
    validate_chain(parents[0])  # Chain validity alone cannot admit this source.
    import tests.shared_ppo_sources as support
    monkeypatch.setattr(support, 'project_journals', lambda *a, **kw: pytest.fail('must not repair original proof'))
    with pytest.raises(ContractError, match='proof changed'):
        SharedPPOSources.admit(root)
    assert not (root / 'original-dataset.json').exists()


@pytest.mark.parametrize('corruption', ['bytes', 'symlink', 'version'])
def test_changed_template_fails_before_copy(tmp_path, frozen_sources, corruption):
    parents, _ = frozen_sources
    root = parents[0].parent
    template = SharedPPOSources.admit(root)
    if corruption == 'bytes':
        numeric = next((parents[0] / 'numeric').rglob('*.npz'))
        numeric.write_bytes(b'corrupt')
    elif corruption == 'symlink':
        (parents[0] / 'unexpected-link').symlink_to(parents[0] / 'run.json')
    else:
        state = read_json(parents[0] / 'run.json')
        state['schema_version'] = 999
        atomic_json(parents[0] / 'run.json', state)
    destination = tmp_path / 'must-not-publish'
    with pytest.raises(ContractError, match='inventory|regular'):
        template.clone(destination)
    assert not destination.exists()


def test_unsupported_template_version_is_not_resealed(frozen_sources):
    parents, _ = frozen_sources
    manifest_path = parents[0] / 'collections/batch-000000/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['schema_version'] = 999
    atomic_json(manifest_path, manifest)
    update = read_json(parents[0] / 'updates/000001.json')
    update['collection_manifest_sha256'] = sha256(manifest_path)
    update['source_proof']['manifest_sha256'] = sha256(manifest_path)
    _reseal(parents[0], update)
    validate_chain(parents[0])
    with pytest.raises(ContractError, match='unsupported shared PPO collection layout/version'):
        SharedPPOSources.admit(parents[0].parent)
