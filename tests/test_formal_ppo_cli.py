"""Explicit formal CLI routing, no model/process/Source work."""
from pathlib import Path

from gf_cli import main
from gf_training.card_instance_ppo import _GameOperations

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/experiments/card-instance-mlp-ppo-acceptance.toml'


def test_shortened_formal_cli_rejects_before_model_or_output(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('shortened formal entry allocated runtime/model')
    monkeypatch.setattr(_GameOperations, 'validate_config', forbidden)
    target = tmp_path / 'not-created'
    assert main(['train-formal', '--config', str(CONFIG), '--run-dir', str(target)]) == 2
    assert not target.exists()
    assert 'frozen literal variant' in capsys.readouterr().err


def test_ordinary_train_keeps_short_recipe_and_never_selects_formal_entry(tmp_path, monkeypatch):
    import gf_training.card_instance_ppo as module
    selected = []
    async def ordinary(value, collector, directory, **kwargs):
        selected.append(value.config.training.max_updates)
        return {'scope': 'routing-only'}
    async def forbidden(*args, **kwargs):
        raise AssertionError('ordinary train selected formal')
    monkeypatch.setattr(module, 'train_card_instance_ppo', ordinary)
    monkeypatch.setattr(module, 'train_formal_card_instance_ppo', forbidden)
    assert main(['train', '--config', str(CONFIG), '--run-dir', str(tmp_path / 'ordinary')]) == 0
    assert selected == [2]


def test_formal_train_reuses_same_collector_route_with_explicit_fresh_resume_arguments(tmp_path, monkeypatch):
    import gf_training.card_instance_ppo as module
    selected = []
    async def formal(value, collector, directory, **kwargs):
        selected.append((value.config.training.max_updates, kwargs['mode'], kwargs['pause_after_update']))
        return {'scope': 'formal-routing-only-NOT-executed'}
    monkeypatch.setattr(module, 'train_formal_card_instance_ppo', formal)
    assert main(['train-formal', '--config', str(CONFIG), '--run-dir', str(tmp_path / 'formal'),
        '--set', 'training.max_updates=1000', '--mode', 'resume', '--pause-after-update', '10']) == 0
    assert selected == [(1000, 'resume', 10)]
