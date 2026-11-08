"""CLI composition/early refusal only; no Source, model, native gate or fit runs."""
from __future__ import annotations

import asyncio
import importlib.abc
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

import gf_cli
import gf_cli.card_instance as composition
from gf_cli.config import load_config
from gf_common.card_instance import CORRECTED_SIMULATOR_REVISION
from gf_common.domain import ContractError
from gf_models.config import TrainingConfig
from gf_models.structured_tree_config import ConfiguredTreeModelV3

ROOT = Path(__file__).parents[1]
TOTAL = 8*1024**3


def config_file(tmp_path, *, backend='python', reader='participant-structured-v2', model='lightgbm',
                checkpoint=None, prep_wall=300, training_wall=123, memory=8192):
    file = tmp_path/'tree-data.toml'
    checkpoint_line = '' if checkpoint is None else 'checkpoint='+json.dumps(str(checkpoint))+'\n'
    file.write_text('schema_version=1\nkind="experiment"\n[references]\nactions='
        +json.dumps(str(ROOT/'configs/actions/card-instance-v2-corrected.toml'))+'\n'
        '[server]\ngameplay_api="v2"\nsimulator_revision="'+CORRECTED_SIMULATOR_REVISION
        +'"\nfixed_seed_experiment=true\n[model]\nversion="3"\nmodel_id='+json.dumps(model)+'\n'
        +checkpoint_line+'[preprocessing]\nbackend='+json.dumps(backend)+'\nreader='+json.dumps(reader)
        +'\nwall_seconds='+str(prep_wall)+'\nmemory_budget_mib='+str(memory)+'\n'
        '[training]\nalgorithm="lightgbm_mc"\ncost_mode="fixed_data"\nseeds=[17]\nwall_seconds='+str(training_wall)+'\n')
    return file


def input_paths(tmp_path):
    parents, validations = [], []
    for seed in (17,29,43):
        parent, validation = tmp_path/f'parent-{seed}', tmp_path/f'validation-{seed}'
        parent.mkdir(); validation.mkdir()
        parents.append(parent); validations.append(validation)
    # A path-presence placeholder, never a qualified Dataset or Source.
    manifest = tmp_path/'unadmitted-input.json'
    manifest.write_text('{}\n')
    return tuple(parents), tuple(validations), manifest


class NoSourceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('gf_training.card_instance_game_tree_data', 'gf_training.structured_tree_fit'):
            raise AssertionError('early refusal/help imported Source/fit implementation')


def test_new_help_is_lazy_and_old_subcommand_help_stays_available(capsys):
    gate = NoSourceImports()
    sys.meta_path.insert(0, gate)
    try:
        for args in (['game-dataset','--help'], ['game-dataset','build','--help'],
                     ['game-dataset','inspect','--help'], ['fit-game-tree','--help'], ['train','--help']):
            with pytest.raises(SystemExit) as result:
                gf_cli.main(args)
            assert result.value.code == 0
        text = capsys.readouterr().out
        assert '--parents' in text and '--validations' in text and '--expected-digest' in text
        assert '--pause-after-update' in text
    finally:
        sys.meta_path.remove(gate)


@pytest.mark.parametrize('backend', ['python','rust'])
def test_tree3_same_corrected_participant_reader_is_type_admitted(backend, tmp_path):
    loaded = load_config(config_file(tmp_path, backend=backend))
    assert type(loaded.config.model) is ConfiguredTreeModelV3
    assert type(loaded.config.training) is TrainingConfig
    assert loaded.config.preprocessing.backend == backend
    assert loaded.config.preprocessing.reader == 'participant-structured-v2'


@pytest.mark.parametrize('override', [
    {'model': {'version': '2'}}, {'model': {'encoder': {'max_positions': 19}}},
    {'preprocessing': {'reader': 'structured-v1'}}, {'preprocessing': {'reader': 'typed-borrowed-v1'}},
    {'training': {'algorithm': 'ppo'}},
])
def test_tree_reader_config_crossovers_reject(override, tmp_path):
    file = config_file(tmp_path)
    with pytest.raises(ContractError):
        load_config(file, override)


def test_existing_nn_ppo_participant_reader_is_unchanged():
    loaded = load_config(ROOT/'configs/experiments/card-instance-mlp-ppo-acceptance.toml')
    assert loaded.config.model.model_id == 'mlp' and loaded.config.training.algorithm == 'ppo'
    assert loaded.config.preprocessing.reader == 'participant-structured-v2'


def test_controlled_build_dispatch_preserves_order_settings_encoder_and_full_budget(tmp_path, monkeypatch, capsys):
    # Mock dispatch is composition evidence only: no originals are opened.
    import gf_training.card_instance_game_tree_data as data
    config = config_file(tmp_path)
    loaded = load_config(config)
    parents, validations, _ = input_paths(tmp_path)
    output = tmp_path/'fresh-output'
    calls = []
    def build(*args, **kwargs):
        calls.append((args,kwargs))
        return {'controlled-build-dispatch-not-executed': True}
    monkeypatch.setattr(data,'build_game_tree_dataset',build)
    before = time.monotonic()
    assert gf_cli.main(['game-dataset','build','--config',str(config),'--parents',*map(str,parents),
        '--validations',*map(str,validations),'--output',str(output)]) == 0
    after = time.monotonic()
    args, kwargs = calls[0]
    assert args == (parents, validations, output, loaded.config.preprocessing)
    assert kwargs['encoder'] == loaded.config.model.encoder
    assert kwargs['max_owned_bytes'] == TOTAL
    assert before+300 <= kwargs['deadline'] <= after+300
    assert 'ownership' not in kwargs and not output.exists()
    assert json.loads(capsys.readouterr().out) == {'controlled-build-dispatch-not-executed': True}


def test_controlled_inspect_and_fit_dispatch_preserve_digest_child_cap_and_original_deadline(tmp_path, monkeypatch, capsys):
    import gf_training.card_instance_game_tree_data as data
    import gf_training.structured_tree_fit as fit
    config = config_file(tmp_path)
    loaded = load_config(config)
    _, _, manifest = input_paths(tmp_path)
    calls = []
    def inspect(*args, **kwargs):
        calls.append(('inspect',args,kwargs)); return {'controlled-inspect-dispatch-not-executed': True}
    async def train(*args, **kwargs):
        calls.append(('fit',args,kwargs)); return {'controlled-fit-dispatch-not-executed': True}
    monkeypatch.setattr(data,'inspect_game_tree_dataset',inspect)
    monkeypatch.setattr(fit,'fit_game_dataset',train)
    digest = 'a'*64
    before = time.monotonic()
    assert gf_cli.main(['game-dataset','inspect',str(manifest),'--config',str(config),'--expected-digest',digest]) == 0
    output = tmp_path/'fresh-fit'
    assert gf_cli.main(['fit-game-tree','--config',str(config),'--manifest',str(manifest),
        '--expected-digest',digest,'--output',str(output),'--seed','19']) == 0
    after = time.monotonic()
    for kind, args, kwargs in calls:
        assert args == (manifest,) and kwargs['settings'] == loaded.config.preprocessing
        assert kwargs['expected_digest'] == digest and kwargs['max_owned_bytes'] == TOTAL
        seconds = 300 if kind == 'inspect' else 123
        assert before+seconds <= kwargs['deadline'] <= after+seconds
        assert 'ownership' not in kwargs
    _, _, kwargs = calls[1]
    assert kwargs['model'] == loaded.config.model and kwargs['model'].resources.max_owned_bytes == 2*1024**3
    assert kwargs['training'] == loaded.config.training and kwargs['directory'] == output and kwargs['seed'] == 19
    assert not output.exists()
    assert len(capsys.readouterr().out.splitlines()) == 2


@pytest.mark.parametrize('operation', ['build','inspect','fit'])
def test_missing_inputs_refuse_before_lazy_source_import_or_output(operation, tmp_path, capsys):
    file = config_file(tmp_path)
    missing, output = tmp_path/'missing', tmp_path/'must-not-exist'
    gate = NoSourceImports(); sys.meta_path.insert(0,gate)
    # Ensure a previously imported dispatch module cannot hide eager resolution.
    names = ('gf_training.card_instance_game_tree_data','gf_training.structured_tree_fit')
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    try:
        if operation == 'build':
            args = ['game-dataset','build','--config',str(file),'--parents',*[str(missing)]*3,
                '--validations',*[str(missing)]*3,'--output',str(output)]
        elif operation == 'inspect':
            args = ['game-dataset','inspect',str(missing),'--config',str(file),'--expected-digest','a'*64]
        else:
            args = ['fit-game-tree','--config',str(file),'--manifest',str(missing),
                '--expected-digest','a'*64,'--output',str(output),'--seed','17']
        assert gf_cli.main(args) == 2
        assert json.loads(capsys.readouterr().err)['error'] == 'ContractError'
        assert not output.exists()
    finally:
        sys.meta_path.remove(gate); sys.modules.update(saved)


@pytest.mark.parametrize('digest', ['', 'a'*63, 'A'*64, '../Source', 'g'*64])
def test_digest_syntax_is_checked_before_source_or_output(digest, tmp_path, capsys):
    file = config_file(tmp_path)
    _, _, manifest = input_paths(tmp_path)
    output = tmp_path/'must-not-exist'
    gate = NoSourceImports(); sys.meta_path.insert(0,gate)
    try:
        assert gf_cli.main(['fit-game-tree','--config',str(file),'--manifest',str(manifest),
            '--expected-digest',digest,'--output',str(output),'--seed','17']) == 2
        assert json.loads(capsys.readouterr().err)['error'] == 'ContractError'
        assert not output.exists()
    finally:
        sys.meta_path.remove(gate)


@pytest.mark.parametrize('mutation', ['legacy_reader','low_total','checkpoint','large_child','long_fit','expired','nan_start'])
def test_cheap_configuration_gates_refuse_without_source_or_child(mutation, tmp_path, monkeypatch):
    file = config_file(tmp_path, reader='legacy' if mutation == 'legacy_reader' else 'participant-structured-v2',
        memory=8191 if mutation == 'low_total' else 8192,
        checkpoint=tmp_path/'old-checkpoint' if mutation == 'checkpoint' else None,
        training_wall=7201 if mutation == 'long_fit' else 123)
    loaded = load_config(file)
    if mutation == 'large_child':
        model = replace(loaded.config.model, resources=replace(loaded.config.model.resources,max_owned_bytes=TOTAL+1))
        loaded = replace(loaded, config=replace(loaded.config,model=model))
    _, _, manifest = input_paths(tmp_path)
    output = tmp_path/'must-not-exist'
    started = float('nan') if mutation == 'nan_start' else time.monotonic()-(301 if mutation == 'expired' else 0)
    monkeypatch.setattr(composition,'_fresh_game_tree_output',lambda path: pytest.fail('allocation/output reached before admission'))
    with pytest.raises(ContractError):
        asyncio.run(composition.fit_game_tree(loaded,manifest,'a'*64,output,17,started=started))
    assert not output.exists()


@pytest.mark.parametrize('seed', [-1, 2**31])
def test_seed_range_is_existing_fit_range(seed, tmp_path, capsys):
    file = config_file(tmp_path)
    _, _, manifest = input_paths(tmp_path)
    output = tmp_path/'must-not-exist'
    assert gf_cli.main(['fit-game-tree','--config',str(file),'--manifest',str(manifest),
        '--expected-digest','a'*64,'--output',str(output),'--seed',str(seed)]) == 2
    assert json.loads(capsys.readouterr().err)['error'] == 'ContractError' and not output.exists()


def test_tree_train_still_cannot_become_ppo_learner(tmp_path, capsys):
    file = config_file(tmp_path)
    assert gf_cli.main(['train','--config',str(file),'--run-dir',str(tmp_path/'must-not-exist')]) == 2
    assert 'NN3 learner' in json.loads(capsys.readouterr().err)['message']
    assert not (tmp_path/'must-not-exist').exists()


def test_optional_free_help_and_missing_inputs_in_clean_interpreter(tmp_path):
    file = config_file(tmp_path)
    script = '''import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','lightgbm','pyarrow','gf_native') or name in ('gf_training.card_instance_game_tree_data','gf_training.structured_tree_fit'):
   raise AssertionError('optional/Source import before cheap gate: '+name)
sys.meta_path.insert(0,Deny())
import gf_cli
try:gf_cli.main(['fit-game-tree','--help'])
except SystemExit as error:assert error.code==0
assert gf_cli.main(['game-dataset','inspect',sys.argv[2],'--config',sys.argv[1],'--expected-digest','a'*64])==2
'''
    result = subprocess.run([sys.executable,'-E','-c',script,str(file),str(tmp_path/'missing')],
        capture_output=True,text=True,check=False,timeout=10)
    assert result.returncode == 0, result.stderr


def test_direct_fit_uses_original_clock_and_never_reduces_declared_child_cap(tmp_path, monkeypatch):
    import gf_training.structured_tree_fit as fit
    file = config_file(tmp_path)
    loaded = load_config(file)
    model = replace(loaded.config.model, resources=replace(loaded.config.model.resources,max_owned_bytes=3*1024**3))
    loaded = replace(loaded,config=replace(loaded.config,model=model))
    _, _, manifest = input_paths(tmp_path)
    output = tmp_path/'fresh'
    original_started = time.monotonic()-10
    captured = {}
    async def dispatch(*args, **kwargs):
        captured.update(kwargs);return {'controlled-dispatch-not-executed': True}
    monkeypatch.setattr(fit,'fit_game_dataset',dispatch)
    result = asyncio.run(composition.fit_game_tree(loaded,manifest,'a'*64,output,17,started=original_started))
    assert result == {'controlled-dispatch-not-executed': True}
    assert captured['deadline'] == original_started+123
    assert captured['model'] is model and model.resources.max_owned_bytes == 3*1024**3
    assert captured['max_owned_bytes'] == TOTAL and not output.exists()


def test_nn_or_baseline_config_cannot_dispatch_game_tree_source(tmp_path, capsys):
    for file in (ROOT/'configs/experiments/card-instance-mlp-ppo-acceptance.toml',
                 ROOT/'configs/experiments/card-instance-v2-corrected-contract.toml'):
        assert gf_cli.main(['game-dataset','inspect',str(tmp_path/'missing'),'--config',str(file),
            '--expected-digest','a'*64]) == 2
        error = json.loads(capsys.readouterr().err)
        assert error['error'] == 'ContractError' and 'Tree3' in error['message']


@pytest.mark.parametrize('operation', ['build','fit'])
def test_existing_output_refuses_before_source_entry(operation, tmp_path, monkeypatch, capsys):
    import gf_training.card_instance_game_tree_data as data
    import gf_training.structured_tree_fit as fit
    file = config_file(tmp_path)
    parents,validations,manifest = input_paths(tmp_path)
    output = tmp_path/'already-exists';output.mkdir()
    monkeypatch.setattr(data,'build_game_tree_dataset',lambda *a, **kw: pytest.fail('Source build entered'))
    async def forbidden(*args, **kwargs):
        pytest.fail('Source fit entered')
    monkeypatch.setattr(fit,'fit_game_dataset',forbidden)
    if operation == 'build':
        args = ['game-dataset','build','--config',str(file),'--parents',*map(str,parents),
            '--validations',*map(str,validations),'--output',str(output)]
    else:
        args = ['fit-game-tree','--config',str(file),'--manifest',str(manifest),
            '--expected-digest','a'*64,'--output',str(output),'--seed','17']
    assert gf_cli.main(args) == 2
    assert 'fresh path' in json.loads(capsys.readouterr().err)['message']
    assert list(output.iterdir()) == []


@pytest.mark.parametrize('args', [
    ['game-dataset','build','--config','unused','--parents','a','b','--validations','d','e','f','--output','g'],
    ['game-dataset','inspect','unused','--config','unused'],
    ['fit-game-tree','--config','unused','--manifest','unused','--expected-digest','a'*64,'--output','unused'],
    ['fit-game-tree','--config','unused','--manifest','unused','--expected-digest','a'*64,'--output','unused','--seed','17','--mode','resume'],
])
def test_closed_cli_arguments_require_complete_explicit_inputs(args):
    with pytest.raises(SystemExit) as raised:
        gf_cli.main(args)
    assert raised.value.code == 2
