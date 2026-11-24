"""Explicit route/collector control doubles; no actual model/game/Source positive."""
import asyncio
import copy
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_cli import card_instance as cli
from gf_common.card_instance_collection import NATIVE_FINAL_VALIDATION_MARKER
from gf_common.domain import ContractError, EpisodeResult, GroupResult, RuntimePolicyBinding
from gf_training import card_instance_collection as records
from gf_training import card_instance_final_validation as module
from tests.test_card_instance_baseline_runtime import receipt
from tests.test_card_instance_final_validation_contract import contract
from tests.test_card_instance_final_validation_owner import actor_context
from tests.test_formal_ppo_control import loaded

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('marker', [None, 'run3-formal-parent-v1;declaration1;stop1', NATIVE_FINAL_VALIDATION_MARKER])
def test_separate_native64_adoption_refuses_before_runtime_output_model_actor(tmp_path, monkeypatch, marker):
    from gf_training import native_backend
    original = loaded(tmp_path)
    cfg = replace(original.config, preprocessing=replace(original.config.preprocessing, backend='rust'))
    value = replace(original, config=cfg)
    native = SimpleNamespace(build_info=lambda: {'card_instance_formal_controls': 'run3-formal-parent-v1;declaration1;stop1',
        'card_instance_final_validation': marker})
    monkeypatch.setattr(native_backend, 'load_native', lambda: native)
    monkeypatch.setattr(native_backend, 'card_instance_reader_settings', lambda *args, **_: {})
    monkeypatch.setattr(cli, 'CardInstanceRuntimeEvidence', lambda *args, **kwargs: pytest.fail('runtime before native64 refusal'))
    with pytest.raises(ContractError, match='no fallback'):
        module.require_actor_backend(value)
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('kind', ['short', 'ordinary', 'fixture'])
def test_explicit_library_short_or_unadmitted_originals_refuse_before_create(tmp_path, kind):
    script = '''
import asyncio, importlib.abc, json, sys
from pathlib import Path
blocked = ('torch', 'lightgbm', 'gf_native')
attempts = []
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            attempts.append(fullname)
            raise AssertionError('early refusal imported ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from gf_cli import card_instance as cli
from gf_cli.config import load_config
from gf_common.domain import ContractError
from tests.test_formal_ppo_control import loaded
root, tmp, kind = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
value = (load_config(root/'configs/experiments/card-instance-mlp-ppo-acceptance.toml')
    if kind == 'short' else loaded(tmp))
parent = tmp/'parent'
parent.mkdir()
(parent/'run.json').write_text(json.dumps({'scope': kind}))
def forbidden(*args, **kwargs):
    attempts.append('runtime/Create')
    raise AssertionError('runtime/Create before formal admission')
cli.CardInstanceRuntimeEvidence = cli.CardInstanceSimulatorEnvironment = forbidden
try:
    asyncio.run(cli.collect_formal_final_validation(value, parent, tmp/'output'))
except ContractError:
    pass
else:
    raise AssertionError('unadmitted originals entered final validation')
assert not (tmp/'output').exists()
assert not attempts
assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules for prefix in blocked)
print(json.dumps({'status': 'REFUSED', 'kind': kind}))
'''
    result = subprocess.run([sys.executable, '-E', '-c', script, str(ROOT), str(tmp_path), kind],
        text=True, capture_output=True, cwd=ROOT, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    assert json.loads(result.stdout) == {'status': 'REFUSED', 'kind': kind}
    assert not (tmp_path/'output').exists()


def test_cli_validate_formal_policy_is_separate_and_reachable(monkeypatch, capsys, tmp_path):
    import gf_cli
    calls = []
    async def collect(value, parent, output):
        calls.append((value.config.run.split, parent, output))
        return {'scope': 'UNADMITTED-route-only'}
    monkeypatch.setattr(cli, 'collect_formal_final_validation', collect)
    status = gf_cli.main(['validate-formal-policy', '--config', str(ROOT / 'configs/experiments/card-instance-mlp-ppo-acceptance.toml'),
        '--parent-run', str(tmp_path / 'parent'), '--output', str(tmp_path / 'output')])
    assert status == 0 and calls == [('train', tmp_path / 'parent', tmp_path / 'output')]
    assert json.loads(capsys.readouterr().out)['scope'] == 'UNADMITTED-route-only'


@pytest.mark.parametrize('mutation', [None, 'generation', 'projection', 'clock', 'cost', 'registry', 'inference'])
def test_final_metadata_producer_closed_fields_and_actual_inference_digest(tmp_path, monkeypatch, mutation):
    original, projected, registry, schedule, linkage = contract(tmp_path)
    label = linkage['policy_label']
    metadata = {'schema_version': 5, 'config': projected, 'config_hash': __import__('gf_common.domain', fromlist=['digest']).digest(projected),
        'policy_bindings': copy.deepcopy(registry), 'neural_inputs': {'policies': {label: {'scope': 'UNADMITTED-metadata-unit'}}}}
    monkeypatch.setattr(records, 'collection_metadata', lambda **kwargs: copy.deepcopy(metadata))
    projection = module.config_projection(original.effective, list(original.inputs), projected)
    invocation = module.validation_invocation(started_unix=1., started_monotonic=1000., caller_deadline_monotonic=8200., run=projected['run'])
    cost = 1.
    if mutation == 'generation': linkage['final_generation'] = 1001
    elif mutation == 'projection': projection['extra'] = 0
    elif mutation == 'clock': invocation['execution_deadline_monotonic'] += 1
    elif mutation == 'cost': cost = 7201.
    elif mutation == 'registry': metadata['policy_bindings']['unknown-card-instance-v2'] = {'model_id': 'random', 'policy_options': None}
    elif mutation == 'inference': metadata['policy_bindings'][label]['policy_options']['expected_manifest_digest'] = 'b' * 64
    def produce():
        return records.final_validation_collection_metadata(actions=original.config.actions, config=projected,
            runtime_evidence=[], schedule=schedule, inventory=[], policies=list(registry), bots_revision='UNADMITTED',
            policy_bindings={key: RuntimePolicyBinding(**raw) for key, raw in registry.items()}, neural_inputs={}, linkage=linkage,
            config_projection=projection, validation_invocation=invocation, elapsed_seconds=cost)
    if mutation is not None:
        with pytest.raises(ContractError): produce()
    else:
        result = produce()
        assert set(result) - set(metadata) == {'collection_role', 'linkage', 'config_projection', 'validation_invocation', 'elapsed_seconds'}
        assert result['linkage']['final_generation'] == 1000
        assert result['linkage']['final_inference_sha256'] == metadata['policy_bindings'][label]['policy_options']['expected_manifest_digest']


@pytest.mark.parametrize('fault', [None, 'group', 'cancel', 'artifact-finalize', 'publication', 'late-parent'])
def test_normal_corrected_collector_uses_one_fresh_plan_actor_ledger_and_cost_order(tmp_path, monkeypatch, fault):
    import gf_cli
    from gf_models import card_instance_artifacts, structured_runtime
    _value, _parent, ledger, context, scope = actor_context(tmp_path, monkeypatch)
    seen = {'groups': [], 'settings': [], 'released': 0}
    class Runtime:
        def __init__(self, *args, **kwargs): self.deadline = kwargs['deadline']
        def preflight(self): pass
    class Artifacts:
        def journal_metadata(self, label): return {'scope': 'UNADMITTED-artifact-double'}
        def finalize(self, **kwargs):
            if fault == 'artifact-finalize': raise OSError('UNADMITTED artifact finalization failure')
            return {'scope': 'UNADMITTED-neural-metadata-double'}
        def release(self): seen['released'] += 1
    def artifacts(*args, **kwargs):
        seen['artifact_budget'] = kwargs['max_owned_bytes']
        return Artifacts()
    class Environment:
        def __init__(self, settings, journal, **kwargs):
            assert kwargs['ledger'] is ledger
            seen['settings'].append(settings)
        async def run_group(self, group):
            seen['groups'].append(group)
            if len(seen['groups']) == 1 and fault in ('group', 'cancel'):
                if fault == 'cancel': raise asyncio.CancelledError('UNADMITTED controlled cancellation')
                raise OSError('UNADMITTED group failure')
            return GroupResult([EpisodeResult(spec.episode_id, spec.pair_id, group.seed, spec.policies,
                game_id=spec.episode_id, outcome='draw', end_kind='natural', replay_reconciled=True)
                for spec in group.episodes], {'runtime_evidence': receipt()})
    def metadata(**kwargs):
        seen['metadata'] = kwargs
        return {'schema_version': 5, 'collection_role': 'ppo-final-validation', 'config_projection': kwargs['config_projection'],
            'validation_invocation': kwargs['validation_invocation'], 'elapsed_seconds': kwargs['elapsed_seconds'], 'linkage': kwargs['linkage']}
    monkeypatch.setattr(cli, 'CardInstanceRuntimeEvidence', Runtime)
    monkeypatch.setattr(card_instance_artifacts.RetainedGameArtifacts, 'for_registry', artifacts)
    monkeypatch.setattr(structured_runtime, 'admit_game_workers_v3', lambda *args, **kwargs: {'peak_cuda_bytes': 0})
    monkeypatch.setattr(cli, 'CardInstanceSimulatorEnvironment', Environment)
    monkeypatch.setattr(cli, '_validate_recorded_episodes', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, '_inventory', lambda *args: [])
    monkeypatch.setattr(cli, 'final_validation_collection_metadata', metadata)
    monkeypatch.setattr(gf_cli, '_code_provenance', lambda *args: {'revision': 'UNADMITTED'})
    if fault == 'publication':
        import os
        original_pwrite = os.pwrite
        failed = []
        def write(descriptor, data, offset):
            if not failed:
                failed.append(True)
                raise OSError('UNADMITTED publication write failure')
            return original_pwrite(descriptor, data, offset)
        monkeypatch.setattr(os, 'pwrite', write)
    if fault == 'late-parent':
        original_guard = cli._validate_final_collection
        guards = []
        def guard(*args):
            original_guard(*args)
            guards.append(True)
            if len(guards) == 3:
                raise ContractError('UNADMITTED changed parent before publication')
        monkeypatch.setattr(cli, '_validate_final_collection', guard)
    try:
        async def run():
            await context.__aenter__()
            if fault is not None:
                expected = asyncio.CancelledError if fault == 'cancel' else ContractError if fault == 'late-parent' else OSError
                with pytest.raises(expected) as error:
                    await cli.collect_owned_final_validation(context)
                assert seen['released'] == 1 and not ledger.workers and ledger.learner is None
                path = context.plan.output / 'manifest.json'
                assert not path.exists() or path.read_bytes().startswith(b'!')
                await context.__aexit__(type(error.value), error.value, error.value.__traceback__)
                assert context.cleanup_complete
                return
            result = await cli.collect_owned_final_validation(context)
            assert type(result) is module._IssuedFinalValidationCollection and len(seen['groups']) == 32
            report = json.loads((result.directory / 'report.json').read_text())
            assert not (result.directory / 'manifest.json').exists()
            manifest = json.loads((result.directory / 'manifest.json.tmp').read_text())
            duration = context.plan.wall_deadline - context.plan.started_monotonic
            module._checked_returned_report(report, result.report, manifest['elapsed_seconds'], duration)
            assert report['planned_games'] == report['natural_denominator'] == 64
            assert report['wall_seconds'] <= manifest['elapsed_seconds'] <= result.report['wall_seconds'] <= duration
            assert seen['settings'][0].admission_deadline == context.plan.admission_deadline
            assert seen['settings'][0].execution_deadline == context.plan.execution_deadline
            assert seen['settings'][0].wall_deadline == context.plan.wall_deadline
            assert seen['metadata']['linkage'] is context.plan.linkage
            assert ledger.learner is None and ledger.cuda_bytes == 0 and not ledger.workers
            assert seen['released'] == 1 and seen['artifact_budget'] > 0
            with pytest.raises(ContractError, match='accepted original Source'):
                await context.aclose()  # No Source admission in this collector control double.
            assert not (result.directory / 'manifest.json').exists()
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)



@pytest.mark.parametrize('final', [False, True])
def test_existing_publisher_includes_report_write_cost_only_for_final_role(tmp_path, monkeypatch, final):
    from gf_common.domain import canonical
    clock = [1000.]
    monkeypatch.setattr(records, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    directory = tmp_path / 'publication'
    directory.mkdir()
    owner = records.CollectionPublication(directory, work_deadline=1010., cleanup_deadline=1020.)
    report = {'healthy': True, 'wall_seconds': 2.}
    manifest = {'schema_version': 5, 'scope': 'UNADMITTED-publication-control', 'elapsed_seconds': .1}
    if final:
        manifest.update(collection_role='ppo-final-validation', validation_invocation={
            'started_unix': 1., 'started_monotonic': 1000., 'caller_deadline_monotonic': 8200.,
            'admission_deadline_monotonic': 6400., 'execution_deadline_monotonic': 8140., 'wall_deadline_monotonic': 8200.})
    before = copy.deepcopy(manifest)
    write = owner._write
    def step(name, value, **kwargs):
        write(name, value, **kwargs)
        if name == 'report.json': clock[0] += 7.
    monkeypatch.setattr(owner, '_write', step)
    if final:
        owner.begin_final_validation(report, manifest)
        assert not (directory / 'manifest.json').exists()
        owner.finish_final_validation()
    else:
        owner.commit(report, manifest)
    persisted = json.loads((directory / 'manifest.json').read_text())
    assert persisted['elapsed_seconds'] == (7. if final else .1)
    assert (directory / 'report.json').read_bytes() == (canonical(report) + '\n').encode()
    assert manifest == before
    if not final:
        assert (directory / 'manifest.json').read_bytes() == (canonical(before) + '\n').encode()



@pytest.mark.parametrize('duration', [0, 59, 60, float('inf'), float('nan')])
def test_explicit_entry_rejects_caller_cap_before_original_parent_io(tmp_path, monkeypatch, duration):
    from gf_training.card_instance_formal import FormalGameParent
    from tests.formal_control_support import unadmitted_current_synthetic_formal_shape
    monkeypatch.setattr(module, 'formal_preflight', unadmitted_current_synthetic_formal_shape)
    value = loaded(tmp_path)
    monkeypatch.setattr(FormalGameParent, '__enter__', lambda *args: pytest.fail('parent I/O before caller reserve rejection'))
    monkeypatch.setattr(cli, 'CardInstanceRuntimeEvidence', lambda *args, **kwargs: pytest.fail('runtime before caller reserve rejection'))
    with pytest.raises(ContractError, match='finalization reserve'):
        asyncio.run(cli.collect_formal_final_validation(value, tmp_path / 'parent', tmp_path / 'output',
            deadline=time.monotonic() + duration))
    assert not (tmp_path / 'parent').exists() and not (tmp_path / 'output').exists()
