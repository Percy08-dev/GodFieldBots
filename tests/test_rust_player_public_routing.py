"""Public routing/diagnostic DATA controls; final training exec is always mocked."""
import json
import os
from pathlib import Path

import pytest

from tests.test_rust_player_config import NORMAL_BASE, runtime


def native_args(tmp_path, *, command='train', timed=True):
    table = '{' + ','.join(key + '=' + json.dumps(value) for key, value in runtime().items()) + '}'
    args = [command, '--config', str(NORMAL_BASE), '--set', 'run.player_runtime=' + table,
        '--set', 'training.device="cpu"', '--set', 'model.inference.device="cpu"']
    if timed:
        args += ['--set', 'run.post_host_seconds=30.0']
    if command in ('train', 'train-formal'):
        args += ['--run-dir', str(tmp_path / 'parent')]
    return args


def test_timed_public_train_routes_to_actual_sealed_factory_before_legacy_guard(tmp_path, monkeypatch, capsys):
    import gf_cli
    from gf_cli import config
    from gf_simulator_client import rust_player_companion
    observed = []
    def legacy_forbidden(*args, **kwargs):
        raise AssertionError('new train selection entered legacy guard/handler')
    def final_exec(executable, argv, env):
        launch = json.loads(os.pread(int(argv[2]), 65536, 0))
        operation = json.loads(Path(launch['operation_path']).read_bytes())
        observed.append((launch, operation))
        raise OSError('test-only final exec refusal')
    monkeypatch.setattr(config, 'require_storage_command', legacy_forbidden)
    monkeypatch.setattr(rust_player_companion.os, 'execve', final_exec)
    assert gf_cli.main(native_args(tmp_path) + ['--pause-after-update', '1']) == 2
    assert len(observed) == 1
    launch, operation = observed[0]
    assert launch['mode'] == 'ordinary' and launch['locks'][0]['lock_kind'] == 'parent'
    assert operation['execution_request'] == {'mode': 'fresh', 'pause_after_update': 1}
    output = capsys.readouterr()
    assert not output.out and 'test-only final exec refusal' in output.err


@pytest.mark.parametrize('command,timed', [('train', False), ('train-formal', True), ('collect', True), ('evaluate', True)])
def test_untimed_or_other_public_native_commands_remain_guarded(tmp_path, monkeypatch, capsys, command, timed):
    import gf_cli
    from gf_simulator_client import rust_player_companion
    def forbidden(*args, **kwargs):
        raise AssertionError('unsupported command reached companion discovery/exec')
    monkeypatch.setattr(rust_player_companion, 'discover_companion', forbidden)
    assert gf_cli.main(native_args(tmp_path, command=command, timed=timed)) == 2
    assert not (tmp_path / 'parent').exists()
    assert not capsys.readouterr().out


def test_public_native_resume_requires_run_dir_before_factory(tmp_path, monkeypatch):
    import gf_cli
    from gf_cli import rust_player_entry
    def forbidden(*args, **kwargs):
        raise AssertionError('missing run directory reached factory')
    monkeypatch.setattr(rust_player_entry, 'exec_ordinary', forbidden)
    args = native_args(tmp_path)
    del args[-2:]
    assert gf_cli.main(args + ['--mode', 'resume']) == 2


def test_legacy_public_train_keeps_existing_handler(tmp_path, monkeypatch, capsys):
    import gf_cli
    from gf_cli import rust_player_entry
    from gf_training import card_instance_ppo
    calls = []
    async def legacy(loaded, collect, directory, **kwargs):
        calls.append((directory, kwargs))
        return {'status': 'paused', 'updates': 0}
    def forbidden(*args, **kwargs):
        raise AssertionError('legacy train entered native factory')
    monkeypatch.setattr(card_instance_ppo, 'train_card_instance_ppo', legacy)
    monkeypatch.setattr(rust_player_entry, 'exec_ordinary', forbidden)
    assert gf_cli.main(['train', '--config', str(NORMAL_BASE), '--run-dir', str(tmp_path / 'legacy')]) == 0
    assert calls == [(tmp_path / 'legacy', {'mode': 'fresh', 'pause_after_update': None})]
    assert json.loads(capsys.readouterr().out) == {'status': 'paused', 'updates': 0}
