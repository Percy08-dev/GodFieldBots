"""Additive controls-only candidate tests; no J5/NPZ/source capability claim."""
import asyncio
import copy
import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from gf_common.domain import ContractError, digest
from gf_common.preprocessing import PreparationConfig
from gf_models.arrays import atomic_json, sha256
from gf_models.structured_binding import model_defaults_v2
from gf_training.native_structured import NativeStructuredControls
from tests.native_structured_support import metadata_parent

native = pytest.importorskip('gf_native')
pytestmark = pytest.mark.skipif(not hasattr(native, 'StructuredParentControls'), reason='candidate controls diagnostic not installed')
SETTINGS = PreparationConfig(backend='rust', reader='structured-v1', memory_budget_mib=512)


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_three_models_metadata_and_no_npz_claim(tmp_path, kind):
    model = model_defaults_v2(kind).architecture()
    state, record, inference = metadata_parent(tmp_path / 'parent', model)
    with NativeStructuredControls(tmp_path / 'parent', SETTINGS, deadline=time.monotonic() + 60) as owner:
        result = owner.result
        assert result['model'] == json.loads(json.dumps(inference['model']))
        assert result['state']['seed'] == state['seed'] and result['state']['config_sha256'] == state['config_sha256']
        assert result['generations'][0]['inference_sha256'] == record['inference_sha256']
        assert result['npz_validated'] is result['journal5_validated'] is result['qualified'] is False
        assert result['kind'] == 'structured-parent-controls' and result['parent_qualification'] == 'not_performed'
        owner.recheck()
        assert owner._session.used_bytes() >= owner._session.control_reserved_bytes() > 0
    assert owner._session.used_bytes() == 0


@pytest.mark.parametrize('window', [1, 128, 1024, 65536])
def test_embedded_actual_window_signature(tmp_path, window):
    model = model_defaults_v2().architecture()
    model = replace(model, event=replace(model.event, aggregate_window=window))
    _, _, inference = metadata_parent(tmp_path / 'parent', model)
    with NativeStructuredControls(tmp_path / 'parent', SETTINGS, deadline=time.monotonic() + 60) as owner:
        assert owner.result['semantics']['token_signature'] == inference['token_signature']


def test_summary_original_inventory_rechecked_and_diagnostic_scope_preserved(tmp_path):
    metadata_parent(tmp_path / 'parent', model_defaults_v2().architecture())
    with pytest.raises(ValueError, match='inventory|snapshot|changed'), NativeStructuredControls(
            tmp_path / 'parent', SETTINGS, deadline=time.monotonic() + 60) as owner:
        atomic_json(tmp_path / 'parent/generations/foreign.json', {})
        owner.recheck()
    info = dict(native.build_info())
    assert not hasattr(native.StructuredParentControls, 'authorize')
    assert not hasattr(native.StructuredParentControls, 'export_cohort')
    assert info['recipe_identity'] == 'native-preprocessing-v1'
    assert info['encoder_version'] == 'participant-action-v1'


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_actual_c1_two_update_retained_pruned_metadata_differential(tmp_path, make_input, kind):
    pytest.importorskip('torch')
    from gf_training.lineage import run_lock
    from gf_training.structured_lineage import (
        retain_component_generations,
        validate_component_chain,
    )
    from tests.structured_lineage_support import make_chain
    parent, _ = make_chain(tmp_path, make_input, kind=kind)
    with run_lock(parent):
        retain_component_generations(parent, 2, max_owned_bytes=512 * 1024**2, deadline=time.monotonic() + 60)
    state, records = validate_component_chain(parent, max_owned_bytes=512 * 1024**2, deadline=time.monotonic() + 60)
    with NativeStructuredControls(parent, SETTINGS, deadline=time.monotonic() + 60) as owner:
        value = owner.result
        assert value['state']['accepted_decisions'] == state['accepted_decisions']
        assert value['state']['optimizer_steps'] == state['optimizer_steps']
        for old, new in zip(records, value['generations'], strict=True):
            for key in ('generation', 'inference_sha256', 'resume_sha256', 'parent_sha256', 'update_sha256',
                        'accepted_decisions', 'optimizer_steps', 'parent_controls'):
                assert old[key] == new[key]
        assert value['generations'][1]['artifact_status'] == 'deleted'
        descriptor = value['generations'][1]['pruned_descriptor']
        assert set(descriptor) == {'original_files', 'deletion_intent', 'deletion_receipt'}
        assert set(descriptor['deletion_intent']) == {'files', 'staging_basename'}
        assert set(descriptor['deletion_receipt']) == {'files'}


def variants():
    from gf_models.structured_config import HandConfigV2, StateConfigV2
    base = model_defaults_v2().architecture()
    identity = replace(base, state=StateConfigV2(kind='identity'), hand=HandConfigV2(kind='direct'))
    gru = model_defaults_v2('gru').architecture()
    gru = replace(gru, state=replace(gru.state, hidden_sizes=(17,), output_size=19),
        card=replace(gru.card, definition_embedding=3, ability_embedding=5, output_size=15),
        hand=replace(gru.hand, kind='deepsets', hidden_sizes=(23,), latent_size=21, pooling='sum'),
        event=replace(gru.event, hidden_size=7, max_interval=4, max_episode=16, aggregate_window=1),
        temporal=replace(gru.temporal, hidden_size=15, num_layers=2, max_decisions=8, sequence_length=3),
        heads=replace(gru.heads, actor_hidden_sizes=(31, 11), critic_hidden_sizes=(17,)))
    transformer = model_defaults_v2('transformer').architecture()
    transformer = replace(transformer, state=replace(transformer.state, hidden_sizes=(24, 12), output_size=20),
        card=replace(transformer.card, output_size=24),
        hand=replace(transformer.hand, hidden_sizes=(48,), latent_size=32, attention_layers=1,
                     attention_heads=3, attention_feedforward=48, pooling='max'),
        temporal=replace(transformer.temporal, hidden_size=60, num_heads=3, feedforward_size=84,
                         max_decisions=16, sequence_length=5),
        event=replace(transformer.event, hidden_size=12, num_layers=2, aggregate_window=1024),
        heads=replace(transformer.heads, actor_hidden_sizes=(64, 16), critic_hidden_sizes=(32, 16)))
    mean = replace(gru, hand=replace(gru.hand, pooling='mean'))
    maximum = replace(gru, hand=replace(gru.hand, pooling='max'))
    return (identity, gru, transformer, mean, maximum)


@pytest.mark.parametrize('model', variants(), ids=['state-identity-direct', 'gru-sum-odd-width', 'tf2-attention-nondefault', 'gru-mean', 'gru-max'])
def test_python_parameter_shapes_independent_pairwise_reference(tmp_path, model):
    metadata_parent(tmp_path / 'parent', model)
    with NativeStructuredControls(tmp_path / 'parent', SETTINGS, deadline=time.monotonic() + 60) as owner:
        assert owner.result['model'] == json.loads(json.dumps(model, default=lambda v: v.__dict__))


def test_partial_uninitialized_adam_metadata_from_actual_c1(tmp_path, make_input, monkeypatch):
    pytest.importorskip('torch')
    import tests.structured_lineage_support as support
    from gf_training.structured_numeric import Learner
    from tests.test_structured_learning import learner_for, records_for
    original_save = Learner.save
    def save(self, directory, **kwargs):
        # Direct-hand frame projection exceeds this attention fixture's tiny
        #400k shard setting; use the unchanged real64MiB format bound.
        return original_save(self, directory, **{**kwargs, 'shard_bytes': 64 * 1024**2})
    monkeypatch.setattr(Learner, 'save', save)
    monkeypatch.setattr(support, 'learner_for', lambda kind, **kwargs: learner_for(kind, hand='direct', **kwargs))
    monkeypatch.setattr(support, 'records_for', lambda *args, **kwargs: records_for(*args, empty=True, events=False, **kwargs))
    parent, _ = support.make_chain(tmp_path, make_input, updates=1)
    resume = json.loads((parent / 'artifacts/000001/resume.json').read_text())
    assert resume['parameter_states']['event_encoder.weight_ih_l0'] == {'present': False}
    assert any(item['present'] for item in resume['parameter_states'].values())
    with NativeStructuredControls(parent, SETTINGS, deadline=time.monotonic() + 60) as owner:
        assert owner.result['state']['optimizer_steps'] == 4


@pytest.mark.parametrize('mutation', ['float_schema', 'bool_schema', 'missing_contract', 'null_hash', 'wrong_config', 'seed_mismatch', 'inactive_hand', 'tensor_bound', 'unknown_field'])
def test_closed_strict_parent_metadata_rejects_before_summary(tmp_path, mutation):
    from gf_common.action_contract import engineering_contract
    from gf_models.structured_binding import input_binding
    from gf_training.structured_lineage import resume_identity
    root = tmp_path / 'parent'
    state, _, _ = metadata_parent(root, model_defaults_v2().architecture())
    if mutation == 'float_schema':
        state['schema_version'] = 2.0
    elif mutation == 'bool_schema':
        state['schema_version'] = True
    elif mutation == 'missing_contract':
        del state['action_contract']
    elif mutation == 'null_hash':
        state['generation_sha256'] = None
    elif mutation == 'wrong_config':
        state['effective_config']['server']['binary'] = '/foreign/unexecuted'
    elif mutation == 'seed_mismatch':
        state['effective_config']['training']['seeds'] = [42]
    elif mutation == 'inactive_hand':
        state['effective_config']['model']['hand'].update(kind='direct', latent_size=17)
    elif mutation == 'tensor_bound':
        state['effective_config']['model']['resources']['max_tensor_bytes'] = 1
    else:
        state['unknown'] = True
    if mutation in ('seed_mismatch', 'inactive_hand', 'tensor_bound'):
        state['config_sha256'] = digest(state['effective_config'])
        state['resume_identity'] = resume_identity(state['config_sha256'], state['identity'], state['execution'], input_binding(engineering_contract(), 'component', False))
    atomic_json(root / 'run.json', state)
    with pytest.raises((ValueError, MemoryError)):
        native.StructuredParentControls(str(root), b'{"memory_bytes":536870912,"max_json_bytes":67108864}', time.monotonic() + 60)


@pytest.mark.parametrize('settings', [b'{"memory_bytes":true,"max_json_bytes":1}', b'{"memory_bytes":0,"max_json_bytes":1}',
    b'{"memory_bytes":1.0,"max_json_bytes":1}', b'{"memory_bytes":1,"max_json_bytes":null}', b'{"memory_bytes":1,"max_json_bytes":1,"unknown":1}'])
def test_diagnostic_settings_strict_before_any_path_io(tmp_path, settings):
    with pytest.raises(ValueError):
        native.StructuredParentControls(str(tmp_path / 'absent'), settings, time.monotonic() + 60)
    assert not (tmp_path / 'absent').exists()


@pytest.mark.parametrize('kind', ['fifo', 'symlink', 'replacement'])
def test_regular_original_fd_and_namespace_checks(tmp_path, kind):
    root = tmp_path / 'parent'
    metadata_parent(root, model_defaults_v2().architecture())
    target = root / 'run.json'
    if kind in ('fifo', 'symlink'):
        data = target.read_bytes()
        target.unlink()
        if kind == 'fifo':
            os.mkfifo(target)
        else:
            outside = tmp_path / 'outside.json'
            outside.write_bytes(data)
            target.symlink_to(outside)
        started = time.monotonic()
        with pytest.raises((ValueError, OSError)):
            native.StructuredParentControls(str(root), b'{"memory_bytes":536870912,"max_json_bytes":67108864}', time.monotonic() + 5)
        assert time.monotonic() - started < 5
    else:
        owner = native.StructuredParentControls(str(root), b'{"memory_bytes":536870912,"max_json_bytes":67108864}', time.monotonic() + 60)
        owner.summary()
        replacement = tmp_path / 'replacement.json'
        replacement.write_bytes(target.read_bytes())
        os.replace(replacement, target)
        with pytest.raises(ValueError):
            owner.recheck()
        with pytest.raises(ValueError):
            owner.close()
        assert owner.used_bytes() == 0


def test_shared_summary_and_caller_leases_stay_admitted_until_close(tmp_path):
    root = tmp_path / 'parent'
    metadata_parent(root, model_defaults_v2().architecture())
    owner = native.StructuredParentControls(str(root), b'{"memory_bytes":67108864,"max_json_bytes":1048576}', time.monotonic() + 60)
    owner.summary()
    before = owner.used_bytes()
    assert owner.control_reserved_bytes() > 32 * 1024**2
    owner.recheck(1024**2)
    assert owner.used_bytes() == before + 1024**2
    with pytest.raises(MemoryError):
        owner.recheck(64 * 1024**2)
    assert owner.used_bytes() == before + 1024**2
    owner.close()
    owner.close()
    assert owner.used_bytes() == 0
    with pytest.raises((ValueError, RuntimeError)):
        owner.recheck()


def test_expired_or_insufficient_budget_has_no_result(tmp_path):
    with pytest.raises(TimeoutError):
        native.StructuredParentControls(str(tmp_path / 'absent'), b'{"memory_bytes":536870912,"max_json_bytes":67108864}', time.monotonic() - 1)
    root = tmp_path / 'parent'
    metadata_parent(root, model_defaults_v2().architecture())
    with pytest.raises(MemoryError):
        native.StructuredParentControls(str(root), b'{"memory_bytes":1,"max_json_bytes":67108864}', time.monotonic() + 60)
    with pytest.raises(MemoryError):
        native.StructuredParentControls(str(root), b'{"memory_bytes":536870912,"max_json_bytes":1}', time.monotonic() + 60)


def test_async_constructor_cancellation_joins_before_releasing_parent_lock(tmp_path, monkeypatch):
    from gf_training.lineage import run_lock
    root = tmp_path / 'parent'
    metadata_parent(root, model_defaults_v2().architecture())
    actual = native.StructuredParentControls
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def delayed(*args, **kwargs):
        entered.set()
        release.wait(10)
        try:
            return actual(*args, **kwargs)
        finally:
            finished.set()
    monkeypatch.setattr(native, 'StructuredParentControls', delayed)
    async def run():
        owner = NativeStructuredControls(root, SETTINGS, deadline=time.monotonic() + 60)
        task = asyncio.create_task(owner.__aenter__())
        try:
            while not entered.is_set():
                await asyncio.sleep(.001)
            task.cancel('original native constructor cancellation')
            await asyncio.sleep(.01)
            assert not task.done() and not finished.is_set()
            with pytest.raises(ContractError), run_lock(root):
                pass
            release.set()
            with pytest.raises(asyncio.CancelledError, match='original native constructor cancellation'):
                await task
            assert finished.is_set() and owner._closed
            with run_lock(root):
                pass
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_embedded_recipe_golden_and_build_source_binding():
    from gf_common.structured_history import HISTORICAL_AGGREGATE
    from gf_common.structured_tokens import signature
    root = Path(__file__).parents[1] / 'native'
    asset = root / 'assets/structured-token-recipe-v1.json'
    assert sha256(asset) == '80d62e776bc244cddb340429507bc040d6e4d85e5674d6a276c7a9a3ac7b6622'
    recipe = json.loads(asset.read_text())
    for window in (1, 128, 1024, 65536):
        assert digest({**recipe, 'history_window': window}) == signature(window, aggregate_recipe=HISTORICAL_AGGREGATE)
    changed = copy.deepcopy(recipe)
    changed['card_categories'][0][0] += '-changed'
    assert digest(changed) != signature(128)
    script = (root / 'build.rs').read_text()
    assert 'assets/structured-token-recipe-v1.json' in script
    assert 'cargo:rerun-if-changed=' in script and 'GFB_NATIVE_SOURCE_SHA256' in script
    current = root / 'assets/structured-token-recipe-public-exact-v2.json'
    assert sha256(current) == '3a574b7ccc6e3c9753fb2479410e5bc22bbae0909e3bca5c66ce3eccf45f4466'
    active = json.loads(current.read_text())
    for window in (1, 128, 1024, 65536):
        assert digest({**active, 'history_window': window}) == signature(window)
    assert 'assets/structured-token-recipe-public-exact-v2.json' in script


def test_native_actual_historical62_parent_uses_exact_old_descriptor():
    from tests.test_structured_history_compat import ROOT
    with NativeStructuredControls(ROOT, SETTINGS, deadline=time.monotonic() + 60) as owner:
        value = owner.result
        original = json.loads((ROOT / 'artifacts/000000/inference.json').read_text())
        assert value['model'] == original['model']
        assert value['semantics']['token_signature'] == original['token_signature']
        assert 'aggregate_recipe' not in value['model']['event']
        assert value['state']['updates'] == 2 and value['generations'][1]['artifact_status'] == 'deleted'


@pytest.fixture
def pruned(tmp_path, make_input):
    pytest.importorskip('torch')
    from gf_training.lineage import run_lock
    from gf_training.structured_lineage import retain_component_generations
    from tests.structured_lineage_support import make_chain
    root, _ = make_chain(tmp_path, make_input)
    with run_lock(root):
        retain_component_generations(root, 2, max_owned_bytes=512 * 1024**2, deadline=time.monotonic() + 60)
    return root


@pytest.mark.parametrize('mutation', ['intent_field', 'intent_files', 'receipt_files', 'stage_present', 'partial', 'extra_file', 'missing_sha', 'chain'])
def test_pruned_and_chain_metadata_cannot_use_old_hash_shortcuts(pruned, mutation):
    root = pruned
    path = root / 'artifact-registry.json'
    registry = json.loads(path.read_text())
    entry = registry['generations']['1']
    if mutation == 'intent_field':
        del entry['deletion_intent']['staging_basename']
    elif mutation == 'intent_files':
        entry['deletion_intent']['files'] = {}
    elif mutation == 'receipt_files':
        entry['deletion_receipt']['files'] = {}
    elif mutation == 'stage_present':
        (root / 'artifacts' / entry['deletion_intent']['staging_basename']).mkdir()
    elif mutation == 'partial':
        entry['status'] = 'deleting'
        entry['deletion_receipt'] = None
    elif mutation == 'extra_file':
        (root / 'artifacts/000001/foreign').write_bytes(b'evidence')
    elif mutation == 'missing_sha':
        record_path = root / 'generations/000001.json'
        record = json.loads(record_path.read_text())
        record['update_sha256'] = None
        atomic_json(record_path, record)
    else:
        record_path = root / 'generations/000001.json'
        record = json.loads(record_path.read_text())
        record['parent_sha256'] = '0' * 64
        atomic_json(record_path, record)
    atomic_json(path, registry)
    with pytest.raises((ValueError, MemoryError)):
        native.StructuredParentControls(str(root), b'{"memory_bytes":536870912,"max_json_bytes":67108864}', time.monotonic() + 60)


@pytest.mark.parametrize('mutation', ['name', 'shape', 'dtype', 'adam_presence', 'nullable_dtype'])
def test_named_parameter_and_adam_descriptor_metadata_is_strict(tmp_path, mutation):
    root = tmp_path / 'parent'
    state, record, _ = metadata_parent(root, model_defaults_v2().architecture())
    artifact = root / 'artifacts/000000'
    inference = json.loads((artifact / 'inference.json').read_text())
    resume = json.loads((artifact / 'resume.json').read_text())
    if mutation in ('name', 'shape', 'dtype'):
        shard = inference['weights']['shards'][0]
        arrays = shard['metadata']['arrays']
        name = next(iter(arrays))
        if mutation == 'name':
            arrays[name + '.unknown'] = arrays.pop(name)
        elif mutation == 'shape':
            arrays[name]['shape'] = [1]
            arrays[name]['bytes'] = 4
        else:
            arrays[name]['dtype'] = 'int64'
            arrays[name]['bytes'] *= 2
        atomic_json(artifact / shard['file'].replace('.npz', '.json'), shard['metadata'])
        inference['weights']['files'][shard['file'].replace('.npz', '.json')] = sha256(artifact / shard['file'].replace('.npz', '.json'))
        atomic_json(artifact / 'inference.json', inference)
        resume['inference_digest'] = sha256(artifact / 'inference.json')
    elif mutation == 'adam_presence':
        name = next(iter(resume['parameter_states']))
        resume['parameter_states'][name]['present'] = 0
    else:
        resume['state']['shards'][0]['metadata']['arrays']['rng.torch_cpu']['dtype'] = None
    atomic_json(artifact / 'resume.json', resume)
    record['inference_sha256'] = record['artifact_files']['inference.json'] = sha256(artifact / 'inference.json')
    record['resume_sha256'] = record['artifact_files']['resume.json'] = sha256(artifact / 'resume.json')
    if mutation in ('name', 'shape', 'dtype'):
        record['artifact_files'].update(inference['weights']['files'])
    atomic_json(root / 'generations/000000.json', record)
    state['generation_sha256'] = sha256(root / 'generations/000000.json')
    atomic_json(root / 'run.json', state)
    atomic_json(root / 'latest.json', {'generation': 0, 'sha256': state['generation_sha256']})
    registry = json.loads((root / 'artifact-registry.json').read_text())
    registry['generations']['0'].update(original_files=record['artifact_files'], inference_sha256=record['inference_sha256'])
    atomic_json(root / 'artifact-registry.json', registry)
    with pytest.raises((ValueError, MemoryError)):
        native.StructuredParentControls(str(root), b'{"memory_bytes":536870912,"max_json_bytes":67108864}', time.monotonic() + 60)


@pytest.mark.parametrize('cancel_close', [False, True])
def test_concurrent_active_recheck_close_joins_retires_then_releases_lock(tmp_path, monkeypatch, cancel_close):
    from gf_training.lineage import run_lock
    root = tmp_path / 'parent'
    metadata_parent(root, model_defaults_v2().architecture())
    actual = native.StructuredParentControls
    entered, release = threading.Event(), threading.Event()
    closes = []
    class Proxy:
        def __init__(self, *args, **kwargs):
            self.inner = actual(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def recheck(self, *args):
            entered.set()
            release.wait(10)
            return self.inner.recheck(*args)
        def close(self):
            closes.append(1)
            return self.inner.close()
    monkeypatch.setattr(native, 'StructuredParentControls', Proxy)
    async def run():
        owner = NativeStructuredControls(root, SETTINGS, deadline=time.monotonic() + 60)
        await owner.__aenter__()
        checking = asyncio.create_task(owner.arecheck())
        closing = []
        try:
            while not entered.is_set():
                await asyncio.sleep(.001)
            closing = [asyncio.create_task(owner.aclose()), asyncio.create_task(owner.aclose())]
            await asyncio.sleep(.01)
            assert not owner._closed and not any(task.done() for task in closing)
            with pytest.raises(ContractError), run_lock(root):
                pass
            with pytest.raises(ContractError, match='awaited aclose'):
                owner.close()
            if cancel_close:
                closing[0].cancel('original close cancellation')
            release.set()
            results = await asyncio.gather(checking, *closing, return_exceptions=True)
            assert isinstance(results[0], RuntimeError)
            if cancel_close:
                assert isinstance(results[1], asyncio.CancelledError)
            assert owner._closed and owner._session.used_bytes() == 0 and owner._executor._closed
            assert closes == [1]
            with run_lock(root):
                pass
        finally:
            release.set()
            await asyncio.gather(checking, *closing, return_exceptions=True)
            await owner.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('caller_seconds,expected_seconds', [(60, 15), (7, 7)])
def test_owner_work_deadline_bounds_cancel_constructor_and_rechecks(tmp_path, monkeypatch, caller_seconds, expected_seconds):
    import gf_training.native_structured as module
    root = tmp_path / 'parent'
    metadata_parent(root, model_defaults_v2().architecture())
    started = time.monotonic()
    clock = [started]
    actual_cancel, actual_constructor = native.CancelHandle, native.StructuredParentControls
    captured = []
    def cancel(deadline):
        captured.append(('cancel', deadline))
        return actual_cancel(deadline)
    def constructor(directory, settings, deadline, handle):
        captured.append(('constructor', deadline))
        return actual_constructor(directory, settings, deadline, handle)
    monkeypatch.setattr(native, 'CancelHandle', cancel)
    monkeypatch.setattr(native, 'StructuredParentControls', constructor)
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    settings = replace(SETTINGS, wall_seconds=20, cleanup_seconds=5)
    with NativeStructuredControls(root, settings, deadline=started + caller_seconds) as owner:
        assert owner.deadline == started + expected_seconds
        assert captured == [('cancel', owner.deadline), ('constructor', owner.deadline)]
        owner.recheck()
        clock[0] = owner.deadline
        with pytest.raises(TimeoutError, match='diagnostic deadline'):
            owner.recheck()
        with pytest.raises(TimeoutError, match='diagnostic deadline'):
            _ = owner.result
    assert owner._closed and owner._session.used_bytes() == 0


def test_empty_collection_relative_path_rejected_by_native_and_python(tmp_path):
    import shutil

    from gf_training.structured_lineage import validate_component_chain
    from tests.test_structured_history_compat import ROOT
    root = tmp_path / 'parent'
    shutil.copytree(ROOT, root)
    update_path = root / 'updates/000001.json'
    update = json.loads(update_path.read_text())
    shutil.copyfile(root / update['collection'] / 'manifest.json', root / 'manifest.json')
    update['collection'] = ''
    atomic_json(update_path, update)
    generation_path = root / 'generations/000001.json'
    generation = json.loads(generation_path.read_text())
    generation['update_sha256'] = sha256(update_path)
    atomic_json(generation_path, generation)
    next_update_path = root / 'updates/000002.json'
    next_update = json.loads(next_update_path.read_text())
    next_update['parent_sha256'] = sha256(generation_path)
    atomic_json(next_update_path, next_update)
    next_path = root / 'generations/000002.json'
    latest = json.loads(next_path.read_text())
    latest['parent_sha256'] = sha256(generation_path)
    latest['update_sha256'] = sha256(next_update_path)
    atomic_json(next_path, latest)
    state = json.loads((root / 'run.json').read_text())
    state['generation_sha256'] = sha256(next_path)
    atomic_json(root / 'run.json', state)
    atomic_json(root / 'latest.json', {'generation': 2, 'sha256': state['generation_sha256']})
    with pytest.raises(ContractError, match='relative'):
        validate_component_chain(root, max_owned_bytes=512 * 1024**2, deadline=time.monotonic() + 60)
    with pytest.raises(ValueError, match='relative owned path'):
        native.StructuredParentControls(str(root), b'{"memory_bytes":536870912,"max_json_bytes":67108864}', time.monotonic() + 60)
