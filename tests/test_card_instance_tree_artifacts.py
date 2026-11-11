"""Tree3 artifact4 and actual fixture fresh fits; no games or qualified Sources."""
import asyncio
import hashlib
import json
import shutil
import time
from dataclasses import replace

import numpy as np
import pytest

from gf_common import card_instance_fixture as fixture
from gf_common.domain import ContractError, canonical
from gf_common.preprocessing import PreparationConfig
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.config import TrainingConfig
from gf_models.structured_tree_artifacts import (
    TreeOriginV3,
    fit_toy_v3,
    inspect_artifact,
    inspect_artifact_v3,
    load_artifact_v3,
    save_artifact_v3,
)
from gf_models.structured_tree_config import (
    ConfiguredTreeModelV2,
    ConfiguredTreeModelV3,
    TreeInferenceConfigV2,
)
from gf_models.structured_tree_encoder import PreparedTreeFrameV3, feature_layout_v3
from gf_models.structured_tree_pairing import (
    TREE3_RECORDS,
    copy_training_bundle_v3,
    fit_admission_v3,
    inspect_training,
    inspect_training_v3,
    resolve_checkpoint_v3,
    validate_request,
    validate_request_v3,
)
from gf_models.structured_tree_policy import ComponentStructuredTreePolicyV3
from gf_training.card_instance_tree_data import VerifiedFixtureTreeDataset3, build_fixture_dataset
from gf_training.structured_tree_fit import (
    _fit_component_fixture_dataset,
    _PreparedTreeFit,
    _TreeFitRun,
)
from tests.card_instance_action_support import inputs
from tests.card_instance_tree_fixture_support import construct
from tests.test_card_instance_tree_encoder import prepared

SETTINGS = PreparationConfig(reader='participant-structured-v2', memory_budget_mib=2048)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(canonical(value)+'\n')
    return sha(path)


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    pytest.importorskip('lightgbm')
    _, _, frame, candidates = prepared(inputs([('bronze-club', None), ('fireball', 'learned')]), [{'type': 'pray'}])
    row = next(frame.iter_chunks(candidates))[1]
    matrix = np.repeat(row, 80, axis=0)
    # Vary a numeric public feature while preserving every closed category.
    matrix[:, 0] = np.log1p(np.arange(80, dtype=np.float32))
    labels = np.asarray([1. if i % 3 else -1. for i in range(80)], dtype=np.float32)
    model = ConfiguredTreeModelV3()
    origin = TreeOriginV3('1'*64, fixture.CONTRACT, 'component', False)
    tree = fit_toy_v3([matrix], labels, [matrix.copy()], labels.copy(), model=model,
                      origin=origin, seed=17, rounds=4, early_stopping=2, min_data_in_leaf=5)
    directory = tmp_path_factory.mktemp('tree3-toy')/'artifact'
    save_artifact_v3(directory, tree)
    return directory, tree, matrix


def options(directory, tree, *, trained=False):
    return {'expected_digest': sha(directory/'inference.json'), 'model': tree.model,
        'action_config_hash': tree.origin.action_config_hash, 'action_contract': fixture.CONTRACT,
        'purpose': 'component', 'synthetic_capacity': False,
        'source_kind': 'card-instance-tree-fixture-originals' if trained else None,
        'source_origin': fixture.origin() if trained else None}


def test_actual_toy4_roundtrip_exact_names_category_tags_and_immutable_null_origin(toy, tmp_path):
    directory, tree, matrix = toy
    opts = options(directory, tree)
    manifest, _ = inspect_artifact_v3(directory, **opts)
    assert manifest['schema_version'] == 4
    assert manifest['source_kind'] is manifest['source_origin'] is manifest['provenance'] is None
    assert manifest['semantics']['label_recipe'] == 'component-toy-labels-v1'
    assert set(manifest['semantics']) == {'input_recipe', 'policy_input_version', 'generator_version', 'simulator_revision',
        'catalog', 'reviewed_spec_hashes', 'spec_inventory_sha256', 'rule_signature', 'layout_signature',
        'token_signature', 'numeric_transform', 'label_recipe'}
    assert manifest['feature_layout'] == feature_layout_v3(tree.model.encoder)
    assert len(manifest['feature_names']) == tree.booster.num_feature() == 2810
    assert 'own.0.numeric.role%2Emiracle%3Aweapon' in manifest['feature_names']
    loaded = load_artifact_v3(directory, **opts)
    np.testing.assert_array_equal(loaded.booster.predict(matrix), tree.booster.predict(matrix))
    assert loaded.origin == tree.origin and loaded.training_binding is None
    saved = save_artifact_v3(tmp_path/'resaved', loaded)
    assert saved['source_kind'] is saved['source_origin'] is saved['provenance'] is None
    assert saved['fit_origin'] == 'toy'
    with pytest.raises(ContractError):
        inspect_artifact(directory, expected_digest=opts['expected_digest'], model=ConfiguredTreeModelV2(),
                         action_config_hash='1'*64, action_contract=fixture.CONTRACT, purpose='component', synthetic_capacity=False)
    with pytest.raises(ContractError):
        TreeOriginV3('1'*64, fixture.CONTRACT, 'qualified_game', False)
    with pytest.raises(ContractError):
        replace(loaded.origin, fit_origin='trained')


@pytest.mark.parametrize('mutation', ['feature_layout', 'feature_names', 'categorical_indices', 'source', 'semantics', 'model_text'])
def test_resealed_toy_metadata_or_weights_refuse_before_booster_allocation(toy, tmp_path, monkeypatch, mutation):
    import lightgbm
    directory, tree, _ = toy
    target = tmp_path/'artifact'; shutil.copytree(directory, target)
    raw = json.loads((target/'inference.json').read_bytes())
    if mutation == 'feature_layout':
        raw['feature_layout']['transforms'][0] = 'presence'
    elif mutation == 'feature_names':
        raw['feature_names'][0] += 'x'
    elif mutation == 'categorical_indices':
        raw['categorical_indices'] = raw['categorical_indices'][1:]
    elif mutation == 'source':
        raw['source_kind'] = 'card-instance-tree-fixture-originals'; raw['source_origin'] = fixture.origin()
    elif mutation == 'semantics':
        raw['semantics']['label_recipe'] = 'natural-terminal-v1'
    else:
        text = (target/'model.txt').read_text().replace('state.numeric.turn ', 'state.numeric.fake ', 1)
        (target/'model.txt').write_text(text)
        raw['model_file']['sha256'] = raw['tree_descriptor']['model_sha256'] = sha(target/'model.txt')
        raw['model_file']['bytes'] = raw['tree_descriptor']['model_bytes'] = (target/'model.txt').stat().st_size
        raw['creation_execution']['tree_descriptor'] = raw['tree_descriptor']
    write(target/'inference.json', raw)
    monkeypatch.setattr(lightgbm, 'Booster', lambda *a, **kw: pytest.fail('metadata/text must reject before Booster'))
    with pytest.raises(ContractError):
        load_artifact_v3(target, **options(target, tree))


@pytest.fixture(scope='module')
def actual_dataset(tmp_path_factory):
    pytest.importorskip('torch'); pytest.importorskip('lightgbm')
    root = tmp_path_factory.mktemp('tree3-actual-fit')
    original, original_sha = construct(root/'originals')
    dataset = root/'dataset'
    build_fixture_dataset(original, original_sha, dataset, SETTINGS,
                          max_owned_bytes=2*1024**3, deadline=time.monotonic()+120)
    return root, dataset/'manifest.json', sha(dataset/'manifest.json')


@pytest.fixture(scope='module')
def trained(actual_dataset):
    root, dataset, dataset_sha = actual_dataset
    directory = root/'fit'
    async def run():
        async with asyncio.timeout(120):
            return await _fit_component_fixture_dataset(dataset, expected_digest=dataset_sha, directory=directory,
                model=ConfiguredTreeModelV3(), settings=SETTINGS, training=TrainingConfig(), seed=17)
    from gf_models.structured_tree_pairing import tree_fit_producer
    producer = tree_fit_producer()
    completion = asyncio.run(run())
    assert completion['producer'] == producer == tree_fit_producer()
    return directory, completion


def test_actual_full_population_fresh_fit4_pair_copy_load_and_component_scores(trained, tmp_path):
    directory, completion = trained
    pair = inspect_training_v3(directory)
    assert completion['schema_version'] == pair.request['schema_version'] == pair.inference['schema_version'] == 4
    assert pair.request['row_counts'] == {'train': 96, 'validation': 192}
    assert pair.request['fit_settings'] == {'num_boost_round': 500, 'early_stopping_rounds': 50,
        'learning_rate': .05, 'min_data_in_leaf': 20, 'num_threads': 1}
    assert completion['health']['reaped'] is True and completion['health']['exit_code'] == 0
    assert completion['metrics']['configured_rounds'] == 500
    assert 'shared_fixture_construction' in completion['costs'] and 'shared_generation' not in completion['costs']
    for record in (pair.request, completion, pair.inference, pair.inference['provenance']):
        assert record['source_kind'] == 'card-instance-tree-fixture-originals' and record['source_origin'] == fixture.origin()
    assert pair.inference['semantics']['label_recipe'] == fixture.LABEL_RECIPE
    artifact, digest_, binding = resolve_checkpoint_v3(directory)
    assert artifact == directory/'artifact' and binding == pair.binding
    with pytest.raises(ContractError):
        resolve_checkpoint_v3(artifact)
    with pytest.raises(ContractError):
        inspect_training(directory)
    with pytest.raises(ContractError):
        validate_request(pair.request)
    model = ConfiguredTreeModelV3(inference=TreeInferenceConfigV2(candidate_chunk=1, num_threads=2))
    opts = {'expected_digest': digest_, 'model': model, 'action_config_hash': pair.inference['action_config_hash'],
        'action_contract': fixture.CONTRACT, 'purpose': 'component', 'synthetic_capacity': False,
        'source_kind': 'card-instance-tree-fixture-originals', 'source_origin': fixture.origin()}
    tree = load_artifact_v3(artifact, **opts, training_directory=directory, training_binding=binding)
    assert tree.origin.fit_origin == 'trained' and tree.training_binding == binding
    with pytest.raises(ContractError, match='paired-bundle copy'):
        save_artifact_v3(tmp_path/'invalid-resave', tree)
    with pytest.raises(ContractError, match='explicit complete paired'):
        load_artifact_v3(artifact, **opts)
    copied = copy_training_bundle_v3(directory, tmp_path/'copy')
    assert copied.request_sha256 == binding.request_sha256 and copied.completion_sha256 == binding.completion_sha256
    for file in (*completion['outputs']['files'], 'training.json'):
        assert (directory/file).read_bytes() == (tmp_path/'copy'/file).read_bytes()
    policy = ComponentStructuredTreePolicyV3(tree, digest_)
    frame, bank, _, _ = prepared(inputs([('bronze-club', None), ('fireball', 'learned')]), [{'type': 'pray'}])
    from tests.test_card_instance_tree_encoder import table_for
    table = table_for(frame, [{'type': 'pray'}, {'type': 'play_sequence', 'ordered_card_refs': list(frame.references), 'target': 2}])
    scores = policy.scores(frame, bank.view(), table)
    from gf_common.structured_tokens import tokenize_frame, tokenize_table
    token = tokenize_frame(frame, bank.view(), layout=LAYOUT)
    candidates = tokenize_table(table, frame, 18, layout=LAYOUT)
    direct = np.concatenate([tree.booster.predict(row, num_threads=2) for _, row in PreparedTreeFrameV3.create(token, model).iter_chunks(candidates)])
    np.testing.assert_array_equal(scores, direct)
    assert policy.select_frame(frame, bank.view(), table).index == int(np.argmax(scores))
    assert policy.version.startswith('lightgbm-tree-v3-')


@pytest.mark.parametrize('mutation', ['failed', 'paused', 'request', 'input', 'inference', 'model', 'origin', 'child'])
def test_trained_pair4_rejects_changed_request_completion_inputs_and_origin(trained, tmp_path, mutation):
    directory, _ = trained
    target = tmp_path/'fit'; shutil.copytree(directory, target)
    if mutation in ('failed', 'paused'):
        raw = json.loads((target/'training.json').read_text()); raw['status'] = mutation; write(target/'training.json', raw)
    elif mutation == 'request':
        raw = json.loads((target/'request.json').read_text()); raw['seed'] += 1; write(target/'request.json', raw)
    elif mutation == 'input':
        file = next((target/'inputs').glob('*.npz')); file.write_bytes(b'changed')
    elif mutation == 'inference':
        raw = json.loads((target/'artifact/inference.json').read_text()); raw['feature_signature'] = '0'*64; write(target/'artifact/inference.json', raw)
    elif mutation == 'model':
        (target/'artifact/model.txt').write_bytes(b'changed')
    elif mutation == 'origin':
        raw = json.loads((target/'training.json').read_text()); raw['source_origin']['qualified'] = True; write(target/'training.json', raw)
    else:
        raw = json.loads((target/'child-result.json').read_text()); raw['metrics']['retained_tree_count'] += 1; write(target/'child-result.json', raw)
    with pytest.raises((ContractError, OSError)):
        inspect_training_v3(target)


def test_exact_active_fixture_owner_and_full_numeric_population_required(actual_dataset, tmp_path):
    _, path, dataset_sha = actual_dataset
    model = ConfiguredTreeModelV3()
    fit = {'num_boost_round': 500, 'early_stopping_rounds': 50, 'learning_rate': .05, 'min_data_in_leaf': 20, 'num_threads': 1}
    with pytest.raises(ContractError, match='concrete verified'):
        _PreparedTreeFit({'schema_version': 3}, model, fit, _context=TREE3_RECORDS)
    with VerifiedFixtureTreeDataset3(path, dataset_sha, SETTINGS, max_owned_bytes=2*1024**3,
                                     deadline=time.monotonic()+120) as dataset:
        original = dataset.source
        private = _PreparedTreeFit(dataset, model, fit, _context=TREE3_RECORDS)
        private.prepare()
        assert dataset.source is original and not original._closed
        assert private.counts == private.rows == {'train': 96, 'validation': 192}
        assert {float(value) for split in private.labels.values() for block in split for value in block} == {-1., 0., 1.}
        assert all(block.shape[1] == 2810 for split in private.blocks.values() for block in split)
        assert fit_admission_v3(model, fit, private.rows, retained_bytes=dataset.retained_without_sink())['total_bytes'] < 2*1024**3
        dataset.verify_complete()


def new_owner(actual_dataset, tmp_path, *, settings=SETTINGS):
    _, original, original_sha = actual_dataset
    index = tmp_path/'index'; shutil.copytree(original.parent, index)
    dataset = VerifiedFixtureTreeDataset3(index/'manifest.json', original_sha, settings,
        max_owned_bytes=2*1024**3, deadline=time.monotonic()+7200)
    return _TreeFitRun(dataset, tmp_path/'fit', ConfiguredTreeModelV3(), TrainingConfig(), 17, _context=TREE3_RECORDS)


@pytest.mark.parametrize('phase', ['numeric_before_child', 'index_after_child'])
def test_actual_fit4_mutation_never_completes_and_reaps_same_source(actual_dataset, tmp_path, monkeypatch, phase):
    from gf_training import structured_tree_fit as fitting
    owner = new_owner(actual_dataset, tmp_path)
    original_start = fitting._OwnedTreeFitChild.start
    original_wait = fitting._OwnedTreeFitChild.wait
    async def start(child, request_sha):
        assert owner.dataset._active and not owner.dataset.source._closed
        if phase == 'numeric_before_child':
            next((tmp_path/'fit/inputs').glob('*.npz')).write_bytes(b'changed own input')
        return await original_start(child, request_sha)
    async def wait(child):
        await original_wait(child)
        assert owner.dataset._active and not owner.dataset.source._closed
        if phase == 'index_after_child':
            file = owner.dataset.path
            file.write_bytes(file.read_bytes()+b' ')
    monkeypatch.setattr(fitting._OwnedTreeFitChild, 'start', start)
    monkeypatch.setattr(fitting._OwnedTreeFitChild, 'wait', wait)
    with pytest.raises(ContractError):
        asyncio.run(owner.execute())
    assert owner.child.closed and owner.child.process.returncode is not None
    assert not (tmp_path/'fit/training.json').exists() or json.loads((tmp_path/'fit/training.json').read_text())['status'] != 'complete'
    with pytest.raises((ContractError, OSError)):
        inspect_training_v3(tmp_path/'fit')


def test_actual_fixture_original_changed_after_child_is_not_a_complete_fit(tmp_path, monkeypatch):
    from gf_training import structured_tree_fit as fitting
    original, original_sha = construct(tmp_path/'originals')
    index = tmp_path/'dataset'
    build_fixture_dataset(original, original_sha, index, SETTINGS, max_owned_bytes=2*1024**3, deadline=time.monotonic()+120)
    dataset = VerifiedFixtureTreeDataset3(index/'manifest.json', sha(index/'manifest.json'), SETTINGS,
        max_owned_bytes=2*1024**3, deadline=time.monotonic()+7200)
    owner = _TreeFitRun(dataset, tmp_path/'fit', ConfiguredTreeModelV3(), TrainingConfig(), 17, _context=TREE3_RECORDS)
    original_wait = fitting._OwnedTreeFitChild.wait
    async def wait(child):
        await original_wait(child)
        record = owner.dataset.source.manifest['cohorts'][0]['episodes'][1]['file']
        file = original.parent/record
        file.write_bytes(file.read_bytes()+b' ')
    monkeypatch.setattr(fitting._OwnedTreeFitChild, 'wait', wait)
    with pytest.raises(ContractError):
        asyncio.run(owner.execute())
    assert owner.child.closed and owner.child.process.returncode is not None
    assert json.loads((tmp_path/'fit/training.json').read_text())['status'] == 'failed'
    with pytest.raises(ContractError):
        inspect_training_v3(tmp_path/'fit')


def test_actual_fit4_cancel_preserves_error_and_source_until_reap(actual_dataset, tmp_path, monkeypatch):
    from gf_training import structured_tree_fit as fitting
    owner = new_owner(actual_dataset, tmp_path)
    initiating = asyncio.CancelledError('original fixture fit cancellation')
    original_wait = fitting._OwnedTreeFitChild.wait
    original_close = fitting._OwnedTreeFitChild.close
    async def wait(child):
        await original_wait(child)
        raise initiating
    async def close(child, deadline):
        assert owner.dataset._active and not owner.dataset.source._closed
        return await original_close(child, deadline)
    monkeypatch.setattr(fitting._OwnedTreeFitChild, 'wait', wait)
    monkeypatch.setattr(fitting._OwnedTreeFitChild, 'close', close)
    with pytest.raises(asyncio.CancelledError) as caught:
        asyncio.run(owner.execute())
    assert caught.value is initiating
    assert owner.child.closed and owner.child.process.returncode is not None
    assert owner.dataset._closed and owner.dataset._source._closed
    assert json.loads((tmp_path/'fit/training.json').read_text())['status'] == 'failed'


def test_source_reportable_close_precedes_pair_complete_and_release_is_memory_only(actual_dataset, tmp_path, monkeypatch):
    from gf_training import structured_tree_fit as fitting
    owner = new_owner(actual_dataset, tmp_path)
    original_commit = fitting._CompletionPublication.commit
    reached = []
    def commit(publication, completion):
        assert owner.child.closed and owner.dataset._closed and owner.dataset._source._closed
        assert owner.dataset._source._released
        assert completion['costs']['shared_fixture_construction'] == owner.request['shared_fixture_construction']
        monkeypatch.setattr(owner.dataset._source._scan, 'finalize', lambda: pytest.fail('no late Source IO'))
        monkeypatch.setattr(owner.dataset._index, 'finalize', lambda: pytest.fail('no late index IO'))
        reached.append('closed')
        return original_commit(publication, completion)
    monkeypatch.setattr(fitting._CompletionPublication, 'commit', commit)
    completion = asyncio.run(owner.execute())
    assert completion['status'] == 'complete' and reached == ['closed']
    assert owner.closed and owner.dataset._source._released


@pytest.mark.parametrize('failure', ['timeout', 'cancel', 'unconfirmed_revoke'])
def test_tree3_exact_copy_revokes_late_complete_and_preserves_original_error(trained, tmp_path, monkeypatch, failure):
    from gf_models import structured_tree_pairing as pairing
    directory, _ = trained
    original_revoke = pairing._CopiedCompletion.revoke
    initiating = asyncio.CancelledError('copy cancelled') if failure == 'cancel' else TimeoutError('copy late postcheck')
    def fail(owner):
        raise initiating
    monkeypatch.setattr(pairing._CopiedCompletion, '_postcheck', fail)
    if failure == 'unconfirmed_revoke':
        def unconfirmed(owner):
            raise OSError('revocation not confirmed')
        monkeypatch.setattr(pairing._CopiedCompletion, 'revoke', unconfirmed)
    with pytest.raises(type(initiating)) as caught:
        copy_training_bundle_v3(directory, tmp_path/'copy', deadline=time.monotonic()+90)
    assert caught.value is initiating
    if failure == 'unconfirmed_revoke':
        retained = caught.value._structured_copy_cleanup_owner
        assert retained.fd is not None
        original_revoke(retained)
    with pytest.raises((ContractError, ValueError)):
        inspect_training_v3(tmp_path/'copy')
    inspect_training_v3(directory).finalize()


def test_fixture_cost_is_bound_by_request_and_toy_cannot_claim_it(trained):
    from gf_models.structured_tree_pairing import validate_pair_records
    directory, _ = trained
    pair = inspect_training_v3(directory)
    changed = json.loads(canonical(pair.completion))
    changed['costs']['shared_fixture_construction']['seconds'] += 1
    with pytest.raises(ContractError, match='differs from immutable request'):
        validate_pair_records(pair.request, changed, pair.inference, request_sha256=pair.binding.request_sha256,
            inference_sha256=pair.completion['outputs']['inference_sha256'], model_sha256=pair.completion['outputs']['model_sha256'],
            _context=TREE3_RECORDS)


def test_scratch_refusal_precedes_private_row_allocation_or_output(actual_dataset, tmp_path, monkeypatch):
    from gf_training import structured_tree_fit as fitting
    owner = new_owner(actual_dataset, tmp_path, settings=replace(SETTINGS, scratch_budget_mib=1))
    monkeypatch.setattr(fitting._PreparedTreeFit, 'prepare', lambda *a: pytest.fail('scratch must refuse before raw row allocation'))
    with pytest.raises(ContractError, match='scratch admission'):
        asyncio.run(owner.execute())
    assert not (tmp_path/'fit').exists()


def test_fixture_fresh_fit_rejects_checkpoint_before_source_activity(actual_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(VerifiedFixtureTreeDataset3, '__enter__', lambda *a: pytest.fail('fresh-fit guard precedes Source activity'))
    _, index, index_sha = actual_dataset
    with pytest.raises(ContractError, match='fresh bounded'):
        asyncio.run(_fit_component_fixture_dataset(index, expected_digest=index_sha, directory=tmp_path/'fit',
            model=replace(ConfiguredTreeModelV3(), checkpoint='warm-model'), settings=SETTINGS, training=TrainingConfig(), seed=17))
    assert not (tmp_path/'fit').exists()


def test_wrong_or_missing_trained_pair_or_origin_refuses_before_booster(trained, monkeypatch):
    import lightgbm
    directory, completion = trained
    pair = inspect_training_v3(directory)
    artifact = directory/'artifact'
    opts = {'expected_digest': completion['outputs']['inference_sha256'], 'model': ConfiguredTreeModelV3(),
        'action_config_hash': pair.inference['action_config_hash'], 'action_contract': fixture.CONTRACT,
        'purpose': 'component', 'synthetic_capacity': False, 'source_kind': 'card-instance-tree-fixture-originals',
        'source_origin': fixture.origin()}
    monkeypatch.setattr(lightgbm, 'Booster', lambda *a, **kw: pytest.fail('origin/healthy pair must precede Booster'))
    for change in ({}, {'training_directory': directory, 'training_binding': replace(pair.binding, request_sha256='0'*64)},
                   {'source_origin': {**fixture.origin(), 'qualified': True}}, {'source_kind': 'unknown'},
                   {'model': ConfiguredTreeModelV2()}, {'purpose': 'qualified_game'}):
        with pytest.raises(ContractError):
            load_artifact_v3(artifact, **{**opts, **change})
    for field in ('init_model', 'warm_start'):
        request = {**pair.request, 'fit_settings': {**pair.request['fit_settings'], field: 'existing-model'}}
        with pytest.raises(ContractError):
            validate_request_v3(request)


def test_fixture_pair4_metadata_inspection_has_no_optional_queries_or_imports(trained):
    import subprocess
    import sys
    directory, _ = trained
    script = '''import builtins,importlib.metadata,sys
from pathlib import Path
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','lightgbm','pyarrow','gf_native'):raise AssertionError('optional import during pair inspection')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
importlib.metadata.distribution=lambda *a,**k: (_ for _ in ()).throw(AssertionError('active runtime query'))
from gf_models.structured_tree_pairing import inspect_training_v3
pair=inspect_training_v3(Path(sys.argv[1]))
assert pair.request['row_counts']=={'train':96,'validation':192}
assert pair.inference['source_origin']['qualified'] is False
assert not ({'torch','lightgbm','pyarrow','gf_native'} & sys.modules.keys())
'''
    subprocess.run([sys.executable, '-c', script, str(directory)], check=True)


def test_actual_fixture_fit_asymmetric_threads_preserves_inference_binding(actual_dataset, tmp_path):
    owner = new_owner(actual_dataset, tmp_path)
    owner.training = replace(TrainingConfig(), num_threads=2)
    completion = asyncio.run(owner.execute())
    pair = inspect_training_v3(tmp_path/'fit')
    assert pair.request['fit_settings']['num_threads'] == 2
    assert pair.request['execution']['inference']['num_threads'] == 1
    assert pair.request['row_counts'] == {'train': 96, 'validation': 192}
    assert completion['metrics']['configured_rounds'] == 500
    assert owner.prepared.admission['native_threads'] == 2
    assert owner.prepared.admission['total_bytes'] < 2*1024**3
    assert owner.child.closed and owner.dataset._closed and owner.dataset._source._released


@pytest.mark.parametrize('mode', ['before_close', 'after_close', 'uncertain'])
def test_copy_release_failure_revokes_actual_pair_and_keeps_uncertain_owner(trained, tmp_path, monkeypatch, mode):
    from gf_models import structured_tree_pairing as pairing
    directory, _ = trained
    initiating = OSError('original final copy release failure')
    actual_release = pairing._CopiedCompletion._release
    owners = []
    def release(owner):
        if not owners:
            owners.append(owner)
            if mode == 'after_close':
                actual_release(owner)
            raise initiating
        if mode == 'uncertain':
            raise OSError('descriptor retirement remains uncertain')
        return actual_release(owner)
    monkeypatch.setattr(pairing._CopiedCompletion, '_release', release)
    with pytest.raises(OSError) as caught:
        copy_training_bundle_v3(directory, tmp_path/'copy', deadline=time.monotonic()+90)
    assert caught.value is initiating
    owner = owners[0]
    assert owner.revoked
    if mode == 'uncertain':
        assert caught.value._structured_copy_cleanup_owner is owner and owner.fd is not None
        monkeypatch.setattr(pairing._CopiedCompletion, '_release', actual_release)
        owner.revoke()
    assert owner.fd is None and owner.directory_fd is None and not owner._retained_directories
    with pytest.raises((ContractError, ValueError)):
        inspect_training_v3(tmp_path/'copy')
    inspect_training_v3(directory).finalize()


@pytest.mark.parametrize('mode', ['before_close', 'after_close', 'uncertain'])
def test_fit_publication_release_failure_revokes_actual_complete(trained, tmp_path, monkeypatch, mode):
    from gf_training import structured_tree_fit as fitting
    from gf_training.structured_lineage import _ChainScan, _write
    directory, _ = trained
    target = tmp_path/'fit'; shutil.copytree(directory, target)
    complete = json.loads((target/'training.json').read_bytes())
    publisher = _ChainScan(directory=target, max_owned_bytes=2*1024**3, deadline=time.monotonic()+90)
    _write(target/'training.json', {**complete, 'status': 'running'}, scan=publisher)
    info = target.stat()
    publisher._tree_fit_directory_identity = (info.st_dev, info.st_ino)
    publisher._tree_fit_ancestors = {}
    owner = fitting._CompletionPublication(target, publisher, time.monotonic()+95, 2*1024**3, _context=TREE3_RECORDS)
    initiating = OSError('original final fit publication release failure')
    actual_release = fitting._CompletionPublication._release
    attempted = []
    def release(publication):
        if not attempted:
            attempted.append(True)
            if mode == 'after_close':
                actual_release(publication)
            raise initiating
        if mode == 'uncertain':
            raise OSError('descriptor retirement remains uncertain')
        return actual_release(publication)
    monkeypatch.setattr(fitting._CompletionPublication, '_release', release)
    with pytest.raises(OSError) as caught:
        owner.commit(complete)
    assert caught.value is initiating and owner.revoked and not owner.committed
    if mode == 'uncertain':
        assert caught.value._structured_completion_cleanup_owner is owner and owner.fd is not None
        monkeypatch.setattr(fitting._CompletionPublication, '_release', actual_release)
        owner.revoke()
    assert owner.fd is None and owner.directory_fd is None
    with pytest.raises((ContractError, ValueError)):
        inspect_training_v3(target)


def test_copy_outer_directory_close_failure_precedes_complete_exposure(trained, tmp_path, monkeypatch):
    import os

    from gf_models import structured_tree_pairing as pairing
    directory, _ = trained
    actual_close = os.close
    actual_retire = pairing._CopiedCompletion._retire_directories
    initiating = OSError('retained artifact directory close failed')
    observed = []
    failed = []
    def retire(owner):
        if owner._retained_directories and not observed:
            observed.append(owner)
            fd = owner._retained_directories['artifact']
            def close(candidate):
                if candidate == fd and not failed:
                    failed.append(True)
                    assert not (tmp_path/'copy/training.json').exists()
                    raise initiating
                return actual_close(candidate)
            monkeypatch.setattr(os, 'close', close)
        return actual_retire(owner)
    monkeypatch.setattr(pairing._CopiedCompletion, '_retire_directories', retire)
    with pytest.raises(OSError) as caught:
        copy_training_bundle_v3(directory, tmp_path/'copy', deadline=time.monotonic()+90)
    owner = observed[0]
    assert caught.value is initiating and failed and not owner.exposed
    assert not owner._retained_directories and owner.directory_fd is None
    assert not (tmp_path/'copy/training.json').exists()


@pytest.mark.parametrize('which', ['directory', 'completion'])
def test_close_that_closes_then_raises_still_revokes_copy(trained, tmp_path, monkeypatch, which):
    import os

    from gf_models import structured_tree_pairing as pairing
    directory, _ = trained
    actual_close = os.close
    actual_postcheck = pairing._CopiedCompletion._postcheck
    initiating = OSError('close reported failure after closing')
    observed = []
    failed = []
    def postcheck(owner):
        result = actual_postcheck(owner)
        observed.append(owner)
        fd = owner.directory_fd if which == 'directory' else owner.fd
        def close(candidate):
            if candidate == fd and not failed:
                failed.append(True)
                actual_close(candidate)
                raise initiating
            return actual_close(candidate)
        monkeypatch.setattr(os, 'close', close)
        return result
    monkeypatch.setattr(pairing._CopiedCompletion, '_postcheck', postcheck)
    with pytest.raises(OSError) as caught:
        copy_training_bundle_v3(directory, tmp_path/'copy', deadline=time.monotonic()+90)
    assert caught.value is initiating and failed
    assert observed[0].revoked and observed[0].fd is None and observed[0].directory_fd is None
    with pytest.raises((ContractError, ValueError)):
        inspect_training_v3(tmp_path/'copy')


@pytest.mark.parametrize('field', ['window', 'network'])
def test_toy_actual_booster_cannot_rebind_model_or_window(toy, tmp_path, field):
    from dataclasses import fields

    from gf_models.structured_tree_artifacts import LoadedTreeV3
    from gf_models.structured_tree_config import TreeNetworkConfigV2
    _, tree, _ = toy
    other = (replace(tree.model, encoder=replace(tree.model.encoder, aggregate_window=1)) if field == 'window'
             else replace(tree.model, network=TreeNetworkConfigV2(num_leaves=17)))
    with pytest.raises(ContractError, match='architecture/feature/input binding'):
        replace(tree, model=other)
    with pytest.raises(ContractError, match='architecture/feature/input binding'):
        LoadedTreeV3(tree.booster, other, tree.origin, tree.descriptor)
    # The save boundary also defends an unsupported construction bypass.
    rebound = object.__new__(LoadedTreeV3)
    for descriptor in fields(tree):
        object.__setattr__(rebound, descriptor.name, other if descriptor.name == 'model' else getattr(tree, descriptor.name))
    with pytest.raises(ContractError, match='architecture/feature/input binding'):
        save_artifact_v3(tmp_path/'rebound', rebound)
    assert not (tmp_path/'rebound').exists()


def test_trained_actual_booster_policy_rejects_window_rebind_but_allows_execution_override(trained):
    from dataclasses import fields

    from gf_models.structured_tree_artifacts import LoadedTreeV3
    from gf_models.structured_tree_config import TreeResourcesV2
    directory, completion = trained
    pair = inspect_training_v3(directory)
    model = ConfiguredTreeModelV3()
    opts = {'expected_digest': completion['outputs']['inference_sha256'], 'model': model,
        'action_config_hash': pair.inference['action_config_hash'], 'action_contract': fixture.CONTRACT,
        'purpose': 'component', 'synthetic_capacity': False, 'source_kind': 'card-instance-tree-fixture-originals',
        'source_origin': fixture.origin(), 'training_directory': directory, 'training_binding': pair.binding}
    tree = load_artifact_v3(directory/'artifact', **opts)
    wrong = replace(model, encoder=replace(model.encoder, aggregate_window=1))
    with pytest.raises(ContractError, match='architecture/feature/input binding'):
        replace(tree, model=wrong)
    rebound = object.__new__(LoadedTreeV3)
    for descriptor in fields(tree):
        object.__setattr__(rebound, descriptor.name, wrong if descriptor.name == 'model' else getattr(tree, descriptor.name))
    with pytest.raises(ContractError, match='architecture/feature/input binding'):
        ComponentStructuredTreePolicyV3(rebound, opts['expected_digest'])
    policy = ComponentStructuredTreePolicyV3(tree, opts['expected_digest'])
    frame, bank, _, _ = prepared(inputs([('bronze-club', None)]), [{'type': 'pray'}])
    from tests.test_card_instance_tree_encoder import table_for
    table = table_for(frame, [{'type': 'pray'}])
    policy.model = wrong
    with pytest.raises(ContractError, match='architecture/feature/input binding'):
        policy.scores(frame, bank.view(), table)
    override = replace(model, checkpoint='relocated-own-checkpoint',
        inference=TreeInferenceConfigV2(candidate_chunk=1, num_threads=2),
        resources=TreeResourcesV2(max_owned_bytes=1024**3))
    rebound_execution = replace(tree, model=override)
    active = ComponentStructuredTreePolicyV3(rebound_execution, opts['expected_digest'])
    scores = active.scores(frame, bank.view(), table)
    assert scores.shape == (1,) and np.isfinite(scores).all()
    assert active.tree.origin == tree.origin and active.tree.descriptor == tree.descriptor


def _same_inode_reuse_fault(monkeypatch, publication, path, descriptor, access):
    """The replacement is foreign even when its integer AND inode are identical."""
    import os
    actual_close, actual_open = os.close, os.open
    actual_dup2 = os.dup2
    actual_fstat, actual_pwrite, actual_pread = os.fstat, os.pwrite, os.pread
    initiating = OSError('close failed after same-inode descriptor reuse')
    state = {'replacement': None, 'close_attempts': 0, 'forbidden': []}
    def guard(operation, fd):
        if state['replacement'] is not None and fd == state['replacement']:
            state['forbidden'].append(operation)
            raise AssertionError('foreign replacement FD used by '+operation)
    def close(fd):
        guard('close', fd)
        if fd == descriptor and state['close_attempts'] == 0:
            state['close_attempts'] += 1
            actual_close(fd)
            replacement = actual_open(path, access | os.O_NOFOLLOW)
            if replacement != descriptor:
                actual_dup2(replacement, descriptor)
                actual_close(replacement)
                replacement = descriptor
            assert replacement == descriptor
            state['replacement'] = replacement
            raise initiating
        return actual_close(fd)
    def fstat(fd):
        guard('fstat', fd)
        return actual_fstat(fd)
    def pwrite(fd, data, offset):
        guard('pwrite', fd)
        return actual_pwrite(fd, data, offset)
    def pread(fd, size, offset):
        guard('pread', fd)
        return actual_pread(fd, size, offset)
    monkeypatch.setattr(os, 'close', close)
    monkeypatch.setattr(os, 'fstat', fstat)
    monkeypatch.setattr(os, 'pwrite', pwrite)
    monkeypatch.setattr(os, 'pread', pread)
    return initiating, state, actual_close, actual_fstat


def _actual_completion_publication(trained, tmp_path):
    from gf_training import structured_tree_fit as fitting
    from gf_training.structured_lineage import _ChainScan, _write
    directory, _ = trained
    target = tmp_path/'fit'; shutil.copytree(directory, target)
    complete = json.loads((target/'training.json').read_bytes())
    publisher = _ChainScan(directory=target, max_owned_bytes=2*1024**3, deadline=time.monotonic()+90)
    _write(target/'training.json', {**complete, 'status': 'running'}, scan=publisher)
    info = target.stat()
    publisher._tree_fit_directory_identity = (info.st_dev, info.st_ino)
    publisher._tree_fit_ancestors = {}
    return fitting._CompletionPublication(target, publisher, time.monotonic()+95, 2*1024**3,
                                          _context=TREE3_RECORDS), complete, target


@pytest.mark.parametrize('operation', ['copy', 'fit'])
@pytest.mark.parametrize('access', ['readonly', 'readwrite'])
def test_same_inode_completion_fd_reuse_is_never_touched_and_cannot_bind(trained, tmp_path, monkeypatch, operation, access):
    import os

    from gf_models import structured_tree_pairing as pairing
    from gf_training import structured_tree_fit as fitting
    installed = []
    cls = pairing._CopiedCompletion if operation == 'copy' else fitting._CompletionPublication
    actual_postcheck = cls._postcheck
    target = tmp_path/('copy' if operation == 'copy' else 'fit')
    def postcheck(owner):
        result = actual_postcheck(owner)
        fault = _same_inode_reuse_fault(monkeypatch, owner, target/'training.json', owner.fd,
            os.O_RDONLY if access == 'readonly' else os.O_RDWR)
        installed.append((owner, fault))
        return result
    monkeypatch.setattr(cls, '_postcheck', postcheck)
    if operation == 'fit':
        publication, complete, target = _actual_completion_publication(trained, tmp_path)
    try:
        with pytest.raises(OSError) as caught:
            if operation == 'copy':
                copy_training_bundle_v3(trained[0], target, deadline=time.monotonic()+90)
            else:
                publication.commit(complete)
        owner, (initiating, state, actual_close, actual_fstat) = installed[0]
        assert caught.value is initiating and not state['forbidden'] and state['close_attempts'] == 1
        key = '_structured_copy_cleanup_owner' if operation == 'copy' else '_structured_completion_cleanup_owner'
        assert getattr(caught.value, key) is owner
        assert owner.revoked and owner.fd is None and owner.directory_fd is None and owner._uncertain_releases
        assert actual_fstat(state['replacement']).st_ino == owner.snapshot[1]
        with pytest.raises(ContractError, match='quarantined numbers are not retryable'):
            owner.revoke()
        assert not state['forbidden']
        with pytest.raises((ContractError, ValueError)):
            inspect_training_v3(target)
    finally:
        if installed:
            _, (_, state, actual_close, _) = installed[0]
            if state['replacement'] is not None:
                actual_close(state['replacement'])


@pytest.mark.parametrize('operation', ['copy', 'fit'])
def test_same_inode_directory_fd_reuse_is_terminal_and_replacement_survives(trained, tmp_path, monkeypatch, operation):
    import os

    from gf_models import structured_tree_pairing as pairing
    from gf_training import structured_tree_fit as fitting
    cls = pairing._CopiedCompletion if operation == 'copy' else fitting._CompletionPublication
    actual_postcheck = cls._postcheck
    installed = []
    target = tmp_path/('copy' if operation == 'copy' else 'fit')
    def postcheck(owner):
        result = actual_postcheck(owner)
        fault = _same_inode_reuse_fault(monkeypatch, owner, target, owner.directory_fd, os.O_RDONLY | os.O_DIRECTORY)
        installed.append((owner, fault))
        return result
    monkeypatch.setattr(cls, '_postcheck', postcheck)
    if operation == 'fit':
        publication, complete, target = _actual_completion_publication(trained, tmp_path)
    try:
        with pytest.raises(OSError) as caught:
            if operation == 'copy':
                copy_training_bundle_v3(trained[0], target, deadline=time.monotonic()+90)
            else:
                publication.commit(complete)
        owner, (initiating, state, _, actual_fstat) = installed[0]
        assert caught.value is initiating and owner.revoked and owner._uncertain_releases
        assert not state['forbidden'] and state['close_attempts'] == 1
        assert actual_fstat(state['replacement']).st_ino == target.stat().st_ino
        with pytest.raises((ContractError, ValueError)):
            inspect_training_v3(target)
        with pytest.raises(ContractError, match='quarantined numbers are not retryable'):
            owner.revoke()
        assert not state['forbidden']
    finally:
        if installed:
            _, (_, state, actual_close, _) = installed[0]
            if state['replacement'] is not None:
                actual_close(state['replacement'])


@pytest.mark.parametrize('access', ['readonly', 'readwrite'])
def test_actual_fit_same_inode_fd_reuse_preserves_error_reaps_and_revokes(actual_dataset, tmp_path, monkeypatch, access):
    import os

    from gf_training import structured_tree_fit as fitting
    owner = new_owner(actual_dataset, tmp_path)
    actual_postcheck = fitting._CompletionPublication._postcheck
    installed = []
    def postcheck(publication):
        result = actual_postcheck(publication)
        assert owner.child.closed and owner.dataset._closed
        fault = _same_inode_reuse_fault(monkeypatch, publication, tmp_path/'fit/training.json', publication.fd,
            os.O_RDONLY if access == 'readonly' else os.O_RDWR)
        installed.append((publication, fault))
        return result
    monkeypatch.setattr(fitting._CompletionPublication, '_postcheck', postcheck)
    try:
        with pytest.raises(OSError) as caught:
            asyncio.run(owner.execute())
        publication, (initiating, state, _, actual_fstat) = installed[0]
        assert caught.value is initiating and publication.revoked and not publication.committed
        assert caught.value._structured_completion_cleanup_owner is publication
        assert publication._uncertain_releases and not state['forbidden']
        assert actual_fstat(state['replacement']).st_ino == publication.snapshot[1]
        assert owner.child.closed and owner.child.process.returncode == 0
        assert owner.dataset._closed and owner.dataset._source._released
        with pytest.raises((ContractError, ValueError)):
            inspect_training_v3(tmp_path/'fit')
    finally:
        if installed:
            _, (_, state, actual_close, _) = installed[0]
            if state['replacement'] is not None:
                actual_close(state['replacement'])


def test_retained_copy_directory_same_inode_reuse_never_retried(trained, tmp_path, monkeypatch):
    import os

    from gf_models import structured_tree_pairing as pairing
    actual_retire = pairing._CopiedCompletion._retire_directories
    installed = []
    def retire(owner):
        if owner._retained_directories and not installed:
            fd = owner._retained_directories['artifact']
            fault = _same_inode_reuse_fault(monkeypatch, owner, tmp_path/'copy/artifact', fd,
                                            os.O_RDONLY | os.O_DIRECTORY)
            installed.append((owner, fault))
        return actual_retire(owner)
    monkeypatch.setattr(pairing._CopiedCompletion, '_retire_directories', retire)
    try:
        with pytest.raises(OSError) as caught:
            copy_training_bundle_v3(trained[0], tmp_path/'copy', deadline=time.monotonic()+90)
        owner, (initiating, state, _, actual_fstat) = installed[0]
        assert caught.value is initiating and caught.value._structured_copy_cleanup_owner is owner
        assert not owner.exposed and owner._uncertain_releases and not state['forbidden']
        assert actual_fstat(state['replacement']).st_ino == (tmp_path/'copy/artifact').stat().st_ino
        assert not (tmp_path/'copy/training.json').exists()
        with pytest.raises(ContractError, match='cleanup-only'):
            owner.commit(b'not another publication')
        with pytest.raises(ContractError, match='quarantined numbers are not retryable'):
            owner.revoke()
        assert not state['forbidden']
    finally:
        if installed:
            _, (_, state, actual_close, _) = installed[0]
            if state['replacement'] is not None:
                actual_close(state['replacement'])


@pytest.mark.parametrize('phase', ['before', 'after'])
def test_actual_fixture_release_failure_is_one_use_before_healthy_completion(actual_dataset, tmp_path, monkeypatch, phase):
    owner = new_owner(actual_dataset, tmp_path)
    actual_release = VerifiedFixtureTreeDataset3.release
    first = OSError('actual fixture memory release '+phase+' failure')
    attempts = []
    def release(dataset):
        attempts.append(dataset)
        if phase == 'after':
            actual_release(dataset)
        raise first
    monkeypatch.setattr(VerifiedFixtureTreeDataset3, 'release', release)
    try:
        with pytest.raises(OSError) as caught:
            asyncio.run(owner.execute())
        assert caught.value is first
        assert len(attempts) == 1 and not owner.closed
        assert owner.child.closed and owner.child.process.returncode is not None
        assert owner.dataset._closed and owner.dataset._source._closed
        path = owner.directory/'training.json'
        raw = path.read_bytes()
        assert raw.startswith(b'!') or json.loads(raw)['status'] != 'complete'
        with pytest.raises(OSError) as retried:
            asyncio.run(owner.close())
        assert retried.value is first and len(attempts) == 1 and not owner.closed
    finally:
        # Controlled test-only graph disposal after the failure assertions.
        # The fit-owned failed release stage must remain failed and unqualified.
        actual_release(owner.dataset)
