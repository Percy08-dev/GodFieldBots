"""Actual native categorical toy roundtrips; no real Source/fitting claims."""
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from gf_common.action_contract import engineering_contract
from gf_common.domain import ContractError, canonical
from gf_models.structured_tree_artifacts import (
    LoadedTree,
    TreeOrigin,
    fit_toy,
    inspect_artifact,
    load_artifact,
    save_artifact,
)
from gf_models.structured_tree_config import ConfiguredTreeModelV2
from gf_models.structured_tree_encoder import PreparedTreeFrame, feature_layout
from tests.test_structured_tree_encoder import tokens

pytest.importorskip('lightgbm')


@pytest.fixture
def toy(make_input):
    model = ConfiguredTreeModelV2()
    frame, candidates = tokens(make_input)
    rows = np.concatenate([block for _, block in PreparedTreeFrame.create(frame, model).iter_chunks(candidates)])
    x = np.tile(rows, (16, 1))
    y = np.tile(np.array([0, 1, 2, 3], dtype=np.float32), 16)
    origin = TreeOrigin('a'*64, engineering_contract(), 'component', False)
    fitted = fit_toy([x[:32], x[32:]], y, [x.copy()], y.copy(), model=model, origin=origin,
                     seed=17, rounds=4, min_data_in_leaf=1)
    return fitted, x


def options(directory, tree):
    return {'expected_digest': hashlib.sha256((directory / 'inference.json').read_bytes()).hexdigest(),
        'model': tree.model, 'action_config_hash': tree.origin.action_config_hash,
        'action_contract': tree.origin.action_contract, 'purpose': tree.origin.purpose,
        'synthetic_capacity': tree.origin.synthetic_capacity}


def test_actual_categorical_native_text_roundtrip_all_unused_declarations(toy, tmp_path):
    tree, x = toy
    directory = tmp_path / 'artifact'
    manifest = save_artifact(directory, tree)
    assert manifest['fit_origin'] == 'toy' and manifest['provenance'] is None
    assert manifest['model_file']['file'] == 'model.txt'
    assert manifest['tree_descriptor']['tree_count'] == tree.booster.num_trees()
    loaded = load_artifact(directory, **options(directory, tree))
    np.testing.assert_array_equal(tree.booster.predict(x, num_threads=1), loaded.booster.predict(x, num_threads=1))
    assert loaded.origin == tree.origin and loaded.descriptor == tree.descriptor
    layout = feature_layout(tree.model.encoder)
    actual = loaded.booster.dump_model()['feature_infos']
    assert [i for i, name in enumerate(layout['feature_names']) if actual.get(name, {}).get('values')] == layout['categorical_indices']
    resaved = tmp_path / 'resaved'
    save_artifact(resaved, loaded)
    assert (directory / 'model.txt').read_bytes() == (resaved / 'model.txt').read_bytes()
    assert canonical(manifest) == canonical(json.loads((resaved / 'inference.json').read_bytes()))
    with pytest.raises(ContractError, match='cannot be relabeled'):
        replace(loaded, origin=replace(loaded.origin, synthetic_capacity=True))
    with pytest.raises(ContractError, match='no validated original binding'):
        LoadedTree(__import__('lightgbm').Booster(model_str=(directory / 'model.txt').read_text()), tree.model, tree.origin, tree.descriptor)


def test_toy_active_thread_and_resource_override_resaves_current_creation(toy,tmp_path):
    from dataclasses import asdict

    from gf_models.structured_tree_config import execution_binding
    tree,_=toy
    original_dir=tmp_path/'original'
    original=save_artifact(original_dir,tree)
    active=replace(tree.model,inference=replace(tree.model.inference,candidate_chunk=2,num_threads=2),
        resources=replace(tree.model.resources,max_owned_bytes=3*1024**3,max_array_bytes=32*1024**2))
    loaded=load_artifact(original_dir,**{**options(original_dir,tree),'model':active})
    assert loaded.origin==tree.origin and loaded.training_binding is None and loaded.creation_execution is None
    destination=tmp_path/'resaved'
    resaved=save_artifact(destination,loaded)
    assert resaved['creation_execution']==execution_binding(active,loaded.descriptor)
    assert resaved['creation_execution']!=original['creation_execution']
    assert resaved['creation_execution']['inference']==asdict(active.inference)
    assert resaved['creation_execution']['resources']==asdict(active.resources)
    assert resaved['model']==original['model'] and resaved['provenance'] is None and resaved['fit_origin']=='toy'
    assert (destination/'model.txt').read_bytes()==(original_dir/'model.txt').read_bytes()
    reloaded=load_artifact(destination,**options(destination,loaded))
    assert reloaded.model==active and reloaded.creation_execution is None


@pytest.mark.parametrize('field,value', [('fit_origin','trained'), ('fit_origin','unknown'), ('provenance',{}), ('schema_version',True)])
def test_resealed_unsupported_origin_and_schema_reject(toy, tmp_path, field, value):
    tree, _ = toy
    directory = tmp_path / 'artifact'
    save_artifact(directory, tree)
    path = directory / 'inference.json'
    manifest = json.loads(path.read_bytes())
    manifest[field] = value
    path.write_text(canonical(manifest)+'\n')
    with pytest.raises(ContractError):
        inspect_artifact(directory, **options(directory, tree))


@pytest.mark.parametrize('before,after', [('[zero_as_missing: 0]', '[zero_as_missing: 1]'),
    ('[linear_tree: 0]', '[linear_tree: 1]'), ('[objective: regression]', '[objective: binary]'),
    ('[num_class: 1]', '[num_class: 2]'), ('[boosting: gbdt]', '[boosting: dart]'), ('[num_leaves: 31]', '[num_leaves: 30]')])
def test_resealed_actual_native_parameter_semantics_reject_before_booster(toy, tmp_path, monkeypatch, before, after):
    tree, _ = toy
    directory = tmp_path / 'artifact'
    save_artifact(directory, tree)
    text_path = directory / 'model.txt'
    text = text_path.read_text()
    assert before in text
    text_path.write_text(text.replace(before, after))
    path = directory / 'inference.json'
    manifest = json.loads(path.read_bytes())
    content = text_path.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    manifest['model_file'].update(sha256=sha, bytes=len(content))
    manifest['tree_descriptor']['model_sha256'] = sha
    manifest['tree_descriptor']['model_bytes'] = len(content)
    manifest['creation_execution']['tree_descriptor']['model_sha256'] = sha
    manifest['creation_execution']['tree_descriptor']['model_bytes'] = len(content)
    path.write_text(canonical(manifest)+'\n')
    monkeypatch.setattr(__import__('lightgbm'), 'Booster', lambda **kw: pytest.fail('native load must follow metadata validation'))
    with pytest.raises(ContractError, match='semantics|architecture'):
        load_artifact(directory, **options(directory, tree))


def test_tree_numeric_costs_are_actual_cpu_tree_costs_not_neural_dimensions(toy):
    from gf_models.structured_tree_config import numeric_costs, resource_estimate
    tree, _ = toy
    costs = numeric_costs(tree.model, tree.descriptor)
    width = resource_estimate(tree.model)['feature_width']
    assert costs.context_width == costs.card_width == costs.actor_input_width == costs.actor_hidden_width == 0
    assert costs.tail_tensor_bytes_per_candidate == 4*width
    assert costs.tail_working_bytes_per_candidate == 12*width+32
    assert costs.tail_operations_per_candidate == 10*width+2*tree.descriptor.tree_count*(tree.descriptor.max_depth+1)
    assert tree.descriptor.model_bytes == len(tree.booster.model_to_string(num_iteration=-1).encode())


def test_tree_policy_null_statistics_real_argmax_and_execution_labels(toy, make_input, tmp_path):
    from gf_common.structured import CandidateTable
    from gf_models.structured_tree_policy import StructuredTreePolicy
    from tests.test_structured_records import frame_for
    tree, x = toy
    frame, bank = frame_for(make_input(hand=('bronze-club','bronze-club'), learned=('fireball','aura')))
    table = CandidateTable.from_actions(frame, [{'type':'pray'},
        *[{'type':'play_sequence','ordered_elements':[frame.references[i]],'target':2} for i in (0,1)],
        {'type':'play_sequence','ordered_elements':[frame.references[3],frame.references[0],frame.references[2]],'target':None}],
        action_contract=tree.origin.action_contract, purpose='component', synthetic_capacity=False)
    directory = tmp_path/'artifact'
    save_artifact(directory, tree)
    binding = options(directory, tree)
    policy = StructuredTreePolicy.from_artifact(directory, **binding)
    policy.reset(2**64-1)
    selected = policy.select_frame(frame, bank.view(), table)
    assert selected.index == int(np.argmax(tree.booster.predict(x[:4], num_threads=1)))
    assert selected.logprob is selected.value is None
    assert policy.last_timing['tokenization_ms'] >= 0
    alternate = replace(tree.model, inference=replace(tree.model.inference,candidate_chunk=1))
    rebound = StructuredTreePolicy.from_artifact(directory, **{**binding, 'model':alternate})
    assert rebound.version != policy.version
    assert rebound.select_frame(frame,bank.view(),table).index == selected.index


def test_tree_publication_refuses_existing_symlink_parent_and_preserves_mutated_output(toy, tmp_path, monkeypatch):
    import gf_models.structured_tree_artifacts as module
    tree, _ = toy
    outside = tmp_path/'outside'
    outside.mkdir()
    link = tmp_path/'link'
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ContractError,match='parent'):
        save_artifact(link/'artifact',tree)
    assert not list(outside.iterdir())
    publish = module._publish
    def mutate_first(directory, *args):
        result = publish(directory, *args)
        if args[1] == 'inference.json':
            (directory/'model.txt').write_bytes(b'late replacement')
        return result
    monkeypatch.setattr(module,'_publish',mutate_first)
    with pytest.raises(ContractError,match='written snapshot'):
        save_artifact(tmp_path/'failed',tree)
    assert (tmp_path/'failed/model.txt').read_bytes() == b'late replacement'


def test_final_publication_sweep_catches_earlier_output_changed_during_last_hash(toy, tmp_path, monkeypatch):
    import gf_models.structured_tree_artifacts as module
    tree, _ = toy
    written = module._written
    manifest_checks = []
    def race(directory, directory_fd, name, *args):
        written(directory, directory_fd, name, *args)
        if name == 'inference.json':
            manifest_checks.append(name)
            if len(manifest_checks) == 2:
                with (directory/'model.txt').open('ab') as output:
                    output.write(b'\n')
    monkeypatch.setattr(module, '_written', race)
    with pytest.raises(ContractError, match='written snapshot'):
        save_artifact(tmp_path/'failed',tree)
    assert manifest_checks == ['inference.json','inference.json']


@pytest.mark.parametrize('phase', ['native', 'validation'])
def test_active_load_preserves_same_deadline_after_native_work(toy, tmp_path, monkeypatch, phase):
    import time

    import gf_models.structured_tree_artifacts as module
    tree, _ = toy
    directory = tmp_path/'artifact'
    save_artifact(directory,tree)
    expected_deadline = time.monotonic()+30
    expired = [False]
    checked = module._check
    def check(deadline):
        assert deadline == expected_deadline
        if expired[0]:
            raise TimeoutError('injected same owned deadline after native work')
        checked(deadline)
    monkeypatch.setattr(module, '_check', check)
    if phase == 'native':
        native = module._native
        def delayed_native(*, deadline):
            result = native(deadline=deadline)
            constructor = result.Booster
            def delayed_booster(**kwargs):
                booster = constructor(**kwargs)
                expired[0] = True
                return booster
            from types import SimpleNamespace
            return SimpleNamespace(Booster=delayed_booster)
        monkeypatch.setattr(module,'_native',delayed_native)
    else:
        validate = module._validate_booster
        def delayed_validate(*args, **kwargs):
            result = validate(*args, **kwargs)
            expired[0] = True
            return result
        monkeypatch.setattr(module,'_validate_booster',delayed_validate)
    with pytest.raises(TimeoutError,match='same owned deadline'):
        load_artifact(directory, **options(directory,tree), deadline=expected_deadline)
    assert expired[0]


def _reseal_model(directory,text):
    content=text.encode()
    (directory/'model.txt').write_bytes(content)
    path=directory/'inference.json'
    manifest=json.loads(path.read_bytes())
    sha=hashlib.sha256(content).hexdigest()
    manifest['model_file'].update(sha256=sha,bytes=len(content))
    for descriptor in (manifest['tree_descriptor'],manifest['creation_execution']['tree_descriptor']):
        descriptor.update(model_sha256=sha,model_bytes=len(content))
    path.write_text(canonical(manifest)+'\n')


@pytest.mark.parametrize('direction',['categorical_to_numeric','numeric_to_categorical'])
def test_resealed_real_native_split_kind_change_rejects_before_native_construction(toy,tmp_path,monkeypatch,direction):
    import re

    import lightgbm

    import gf_models.structured_tree_artifacts as module
    tree,_=toy
    directory=tmp_path/'artifact';save_artifact(directory,tree)
    text=(directory/'model.txt').read_text()
    changed=False
    for part in re.split(r'\nTree=[0-9]+\n',text)[1:]:
        features=re.search(r'^split_feature=(.*)$',part,re.MULTILINE)[1].split()
        decisions=re.search(r'^decision_type=(.*)$',part,re.MULTILINE)[1].split()
        for index,decision in enumerate(decisions):
            if bool(int(decision)&1)==(direction=='categorical_to_numeric'):
                replacement=features.copy()
                # Keep the native serialized tree_sizes valid while changing
                # semantics: history feature1000 is numeric, candidate.kind1042
                # is categorical, and both feature tokens retain four digits.
                replacement[index]='1000' if direction=='categorical_to_numeric' else '1042'
                assert len(replacement[index]) == len(features[index])
                text=text.replace('split_feature='+' '.join(features),'split_feature='+' '.join(replacement),1)
                changed=True
                break
        if changed:
            break
    assert changed
    # Reproduce the native-normalized/resealed failure, rather than relying on
    # an invalid tree_sizes declaration to make the metadata parser reject.
    normalized=lightgbm.Booster(model_str=text).model_to_string(num_iteration=-1)
    _reseal_model(directory,normalized)
    monkeypatch.setattr(module,'_native',lambda **kw:pytest.fail('invalid per-node semantics must reject before native construction'))
    with pytest.raises(ContractError,match='scalar tree structure'):
        load_artifact(directory,**options(directory,tree))


@pytest.mark.parametrize('mutation',['split_count','feature_negative','feature_range','zero_missing','unknown_missing'])
def test_resealed_split_vectors_and_missing_flags_reject_in_safe_metadata(toy,tmp_path,monkeypatch,mutation):
    import re

    import gf_models.structured_tree_artifacts as module
    tree,_=toy
    directory=tmp_path/'artifact';save_artifact(directory,tree)
    text=(directory/'model.txt').read_text()
    key='split_feature' if mutation.startswith(('split','feature')) else 'decision_type'
    match=re.search('^'+key+'=(.*)$',text,re.MULTILINE)
    original=match[1].split(); replacement=original.copy()
    if mutation=='split_count':
        replacement.pop()
    elif mutation=='feature_negative':
        replacement[0]='-1'
    elif mutation=='feature_range':
        replacement[0]=str(tree.booster.num_feature())
    elif mutation=='zero_missing':
        replacement[0]=str((int(replacement[0])&~12)|4)
    else:
        replacement[0]=str((int(replacement[0])&~12)|12)
    _reseal_model(directory,text[:match.start(1)]+' '.join(replacement)+text[match.end(1):])
    monkeypatch.setattr(module,'_native',lambda **kw:pytest.fail('malformed split metadata must reject before native'))
    with pytest.raises(ContractError,match='scalar tree structure'):
        load_artifact(directory,**options(directory,tree))


def test_actual_native_dump_split_semantics_must_agree_with_validated_text(toy,tmp_path,monkeypatch):
    import copy

    tree,_=toy
    actual=tree.booster.dump_model
    def changed_dump(**kwargs):
        dump=copy.deepcopy(actual(**kwargs))
        dump['tree_info'][0]['tree_structure']['decision_type']='<='
        return dump
    monkeypatch.setattr(tree.booster,'dump_model',changed_dump)
    with pytest.raises(ContractError,match='per-node categorical'):
        save_artifact(tmp_path/'must-not-exist',tree)
    assert not (tmp_path/'must-not-exist').exists()


@pytest.mark.parametrize('setting',['max_owned_bytes','max_array_bytes','actual_model_envelope'])
def test_resealed_saved_creation_resources_cannot_be_made_valid_by_active_override(toy,tmp_path,setting):
    from gf_models.structured_tree_config import resource_estimate
    tree,_=toy
    directory=tmp_path/'artifact';save_artifact(directory,tree)
    path=directory/'inference.json';manifest=json.loads(path.read_bytes())
    if setting=='actual_model_envelope':
        # The architecture alone fits this declared cap; the actual persisted
        # native model residency must ALSO be charged, independently of active2GiB.
        manifest['creation_execution']['resources']['max_owned_bytes']=resource_estimate(tree.model)['total_bytes']
    else:
        manifest['creation_execution']['resources'][setting]=1
    path.write_text(canonical(manifest)+'\n')
    with pytest.raises(ContractError,match='resource admission'):
        inspect_artifact(directory,**options(directory,tree))


def test_expected_feature_signature_is_checked_in_metadata_before_native(toy,tmp_path,monkeypatch):
    import gf_models.structured_tree_artifacts as module
    tree,_=toy
    directory=tmp_path/'artifact';save_artifact(directory,tree)
    monkeypatch.setattr(module,'_native',lambda **kw:pytest.fail('expected feature mismatch must precede native'))
    with pytest.raises(ContractError,match='inspected feature signature'):
        load_artifact(directory,**options(directory,tree),expected_feature_signature='0'*64)
