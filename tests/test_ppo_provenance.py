"""Installed package provenance excludes unrelated site-packages and ancestor Git."""
import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import pytest

from gf_common.domain import ContractError
from gf_training.ppo_run import source_identity


def test_installed_like_identity_hashes_only_owned_packages_without_imports(tmp_path, monkeypatch):
    import gf_training.ppo_run as module
    names = ('gf_common', 'gf_models', 'gf_training', 'gf_cli',
             'gf_evaluation', 'gf_simulator_client', 'godfieldbots')
    site = tmp_path / 'ancestor-repository' / '.venv/lib/python3.14/site-packages'
    expected = {}
    specs = {}
    for name in names:
        package = site / name
        package.mkdir(parents=True)
        for relative in ('__init__.py', 'owned.py'):
            path = package / relative
            path.write_text(f'# owned {name}/{relative}\n')
            expected[f'{name}/{relative}'] = hashlib.sha256(path.read_bytes()).hexdigest()
        specs[name] = importlib.util.spec_from_file_location(name, package / '__init__.py')
    own_module = site / 'gf_training/ppo_run.py'
    own_module.write_text('# installed own coordinator module\n')
    expected['gf_training/ppo_run.py'] = hashlib.sha256(own_module.read_bytes()).hexdigest()
    unrelated = site / 'torch'
    unrelated.mkdir()
    (unrelated / '__init__.py').write_text('raise AssertionError("must never import unrelated package")\n')
    # A nearby project file and an ancestor Git directory cannot attest this wheel.
    (site.parent / 'pyproject.toml').write_text('[project]\nname="unrelated"\n')
    (tmp_path / 'ancestor-repository/.git').mkdir()
    monkeypatch.setattr(module, '__file__', str(own_module))
    monkeypatch.setattr(module.importlib.util, 'find_spec', lambda name: specs[name])
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: pytest.fail('installed wheel must not inspect ancestor Git'))
    original_modules = set(sys.modules)
    result = source_identity(time.monotonic() + 10)
    assert result['module_sha256'] == expected
    assert result['code'] == {'revision': None, 'dirty': None, 'installed_build_revision': None}
    assert not ({'torch', 'lightgbm', 'pyarrow', 'gf_cli'} - original_modules) & set(sys.modules)


def test_source_identity_checkout_scope_and_deadline():
    result = source_identity(time.monotonic() + 10)
    root = Path(__file__).resolve().parents[1] / 'src'
    names = ('gf_common', 'gf_models', 'gf_training', 'gf_cli',
             'gf_evaluation', 'gf_simulator_client', 'godfieldbots')
    expected = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for name in names for path in (root / name).rglob('*.py')}
    assert result['module_sha256'] == expected
    assert result['code']['revision'] is not None
    with pytest.raises(TimeoutError):
        source_identity(time.monotonic() - 1)


def test_missing_owned_package_rejects_incomplete_source_identity(monkeypatch):
    import gf_training.ppo_run as module
    monkeypatch.setattr(module.importlib.util, 'find_spec', lambda name: None)
    with pytest.raises(ContractError, match='owned Bots package'):
        source_identity(time.monotonic() + 10)


def test_source_layout_archive_does_not_inherit_ancestor_git(tmp_path, monkeypatch):
    import gf_training.ppo_run as module
    ancestor = tmp_path / 'unrelated-repository'
    (ancestor / '.git').mkdir(parents=True)
    archive = ancestor / 'source-archive'
    source = archive / 'src'
    specs = {}
    expected = {}
    for name in ('gf_common', 'gf_models', 'gf_training', 'gf_cli',
                 'gf_evaluation', 'gf_simulator_client', 'godfieldbots'):
        package = source / name
        package.mkdir(parents=True)
        initial = package / '__init__.py'
        initial.write_text(f'# archived {name}\n')
        expected[f'{name}/__init__.py'] = hashlib.sha256(initial.read_bytes()).hexdigest()
        specs[name] = importlib.util.spec_from_file_location(name, initial)
    own_module = source / 'gf_training/ppo_run.py'
    own_module.write_text('# archived coordinator\n')
    expected['gf_training/ppo_run.py'] = hashlib.sha256(own_module.read_bytes()).hexdigest()
    (archive / 'pyproject.toml').write_text('[project]\nname="godfieldbots"\n')
    monkeypatch.setattr(module, '__file__', str(own_module))
    monkeypatch.setattr(module.importlib.util, 'find_spec', lambda name: specs[name])
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: pytest.fail('archive cannot inherit ancestor Git'))
    result = source_identity(time.monotonic() + 10)
    assert result['module_sha256'] == expected
    assert result['code'] == {'revision': None, 'dirty': None, 'installed_build_revision': None}
