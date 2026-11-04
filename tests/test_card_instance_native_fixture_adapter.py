"""SAME concrete component Source diagnostics; no fitting or game authority."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from gf_common.preprocessing import PreparationConfig
from gf_training.card_instance_tree_fixture import TreeFixtureSource
from gf_training.native_card_instance_fixture import native_fixture_diagnostics
from tests.test_card_instance_tree_fixture import copied

pytest_plugins = ('tests.test_card_instance_tree_fixture',)

SETTINGS = PreparationConfig(reader='participant-structured-v2', memory_budget_mib=2048)
LIMIT = 2*1024**3


@pytest.fixture
def native():
    module = pytest.importorskip('gf_native')
    if not hasattr(module, 'CardInstanceFixtureTokens'):
        pytest.skip('isolated fixture-token candidate required')
    return module


def source(path, sha, **options):
    return TreeFixtureSource(path, sha, SETTINGS, max_owned_bytes=LIMIT,
        deadline=time.monotonic()+180, **options)


def complete(owner):
    for _ in owner.selected_inputs():
        pass


def test_source_borrow_bytes_metadata_and_two_reference_streams_do_not_complete(originals):
    path, sha = originals
    with source(path, sha) as owner:
        assert owner.diagnostic_episode_count == 288
        for ordinal in (0, 1, 16, 32, 287):
            with owner.borrow_diagnostic_episode(ordinal) as borrow:
                assert type(borrow.envelope) is bytes and type(borrow.columns) is tuple
                expected = json.loads(borrow.expected_metadata)
                assert borrow.actions.max_candidates == 64
                with pytest.raises(AttributeError):
                    borrow.actions.max_candidates = 1
                assert 'schema_version' not in expected and 'arrays' not in expected and 'kind' not in expected
                assert all(type(value) is bytes for value in borrow.columns)
                first = [decision.sample for decision in borrow.reference_decisions()]
                second = [decision.sample for decision in borrow.reference_decisions()]
                assert first == second and len(first) == len(expected['decisions'])
                assert all('token_prefix' not in row for row in expected['decisions'])
                borrow.guard()
            with pytest.raises(ContractError):
                _ = borrow.envelope
            with pytest.raises(ContractError):
                _ = borrow.actions
            with pytest.raises(ContractError):
                next(borrow.reference_decisions())
        with pytest.raises(ContractError, match='all declared'):
            owner.verify_complete()
        complete(owner)
        assert owner.verify_complete()['counts']['validation']['games'] == 192


def test_borrow_rejects_overlap_and_source_close_without_poisoning_pure_preconditions(originals):
    path, sha = originals
    with source(path, sha) as owner:
        with owner.borrow_diagnostic_episode(1) as borrow:
            with pytest.raises(ContractError, match='overlap'), owner.borrow_diagnostic_episode(0):
                pass
            with pytest.raises(ContractError):
                next(owner.selected_inputs())
            with pytest.raises(ContractError, match='borrow must close'):
                owner.close()
            first = borrow.reference_decisions()
            decision = next(first)
            with pytest.raises(ContractError, match='overlap'):
                next(borrow.reference_decisions())
            list(first)
            assert decision.sample.selected_index >= 0
        with pytest.raises(ContractError):
            _ = decision.frame
        complete(owner)


def test_pending_reference_iterator_is_invalid_after_failed_borrow(originals):
    path, sha = originals
    with pytest.raises(ContractError, match='iterator must retire'), source(path, sha) as owner, owner.borrow_diagnostic_episode(1) as borrow:
        iterator = borrow.reference_decisions()
        next(iterator)
    with pytest.raises(ContractError):
        next(iterator)


def test_borrow_joint_external_budget_is_admitted_before_reference_allocation(originals):
    path, sha = originals
    with pytest.raises(ContractError), source(path, sha) as owner, owner.borrow_diagnostic_episode(1) as borrow:
        borrow.admit_external(LIMIT-1)
    with pytest.raises(ContractError):
        owner.verify_complete()


def test_original_mutation_during_borrow_poisoning_survives_caller_repair(originals, tmp_path):
    path, sha = copied(originals, tmp_path)
    with pytest.raises(ContractError), source(path, sha) as owner, owner.borrow_diagnostic_episode(1) as borrow:
        raw = json.loads(borrow.envelope)
        file = path.parent/raw['decisions'][0]['columns']['file']
        previous = file.read_bytes()
        file.write_bytes(previous+b'changed')
        try:
            with pytest.raises(ContractError):
                borrow.guard()
        finally:
            file.write_bytes(previous)
        with pytest.raises(ContractError):
            borrow.admit_external(0)
    with pytest.raises(ContractError):
        _ = owner.manifest


def test_default_fixture_capability_refuses_before_source_or_output_allocation(tmp_path, monkeypatch):
    import gf_training.native_card_instance_fixture as adapter
    old = SimpleNamespace(build_info=lambda: {'card_instance_signature':'old'})
    monkeypatch.setattr(adapter, 'load_native', lambda: old)
    owner = source(tmp_path/'missing/manifest.json', 'a'*64)
    with pytest.raises(ContractError, match='fixture-token capability'):
        native_fixture_diagnostics(owner, tmp_path/'must-not-exist', max_owned_bytes=LIMIT,
            deadline=time.monotonic()+30)
    assert not (tmp_path/'must-not-exist').exists()


@pytest.mark.parametrize('failure', ['scratch','caller','cancel'])
def test_native_diagnostic_failures_cannot_complete_source(originals, native, tmp_path, monkeypatch, failure):
    import gf_training.native_card_instance_fixture as adapter
    if failure == 'cancel':
        original = adapter._NativeEpisode.construct
        def cancelled(self, *args, **kwargs):
            self.cancel.cancel()
            return original(self, *args, **kwargs)
        monkeypatch.setattr(adapter._NativeEpisode, 'construct', cancelled)
    with pytest.raises((ContractError, RuntimeError, MemoryError)), source(*originals) as owner:
        native_fixture_diagnostics(owner, tmp_path/'diagnostic', max_owned_bytes=LIMIT,
            deadline=time.monotonic()+120, diagnostic_native=native,
            max_scratch_bytes=1 if failure == 'scratch' else None,
            extra_owned_bytes=LIMIT-1 if failure == 'caller' else 0)
    with pytest.raises(ContractError):
        owner.verify_complete()


@pytest.mark.integration
def test_all_nine_cohorts_raw_persistence_and_2810_features_keep_same_source_authority(native, tmp_path):
    if os.environ.get('GF_RUN_FIXTURE_ADAPTER_ORIGINALS') != '1':
        pytest.skip('opt-in durable complete fixture read-only parity')
    root = Path('/home/percy/develop/GodFieldBots/runs/card-instance-v2-91d5926/tree-component-acceptance/fixture-default-1/originals')
    path = root/'manifest.json'
    if not path.is_file():
        pytest.fail('approved durable originals absent')
    from gf_models.arrays import sha256
    settings = PreparationConfig(reader='participant-structured-v2')
    with TreeFixtureSource(path, sha256(path), settings, max_owned_bytes=8*1024**3,
            deadline=time.monotonic()+1800) as owner:
        result = native_fixture_diagnostics(owner, tmp_path/'all-nine', max_owned_bytes=8*1024**3,
            deadline=time.monotonic()+1700, diagnostic_native=native)
        report = result.report
        assert report['episodes'] == 288 and len(report['cohorts']) == 9
        assert sum(value['zero_endpoint_episodes'] for value in report['cohorts'].values()) == 9
        assert report['feature_columns'] == 2810 and report['full_feature_rows'] > 288
        assert report['native_cpu_score_rows'] == 0
        for receipt in report['receipts']:
            container = json.loads((Path(receipt['directory'])/'diagnostic.json').read_text())
            assert set(container) == {'schema_version','kind','source_manifest_sha256','native_binary','metadata','bundle'}
            if receipt['decisions'] == 0:
                assert container['bundle'] is None and container['metadata']['arrays'] == []
        with pytest.raises(ContractError, match='all declared'):
            owner.verify_complete()
        complete(owner)
        owner.verify_complete()
        result.guard()
        snapshot = result.report
        result.close()
        assert snapshot['episodes'] == 288
    with pytest.raises(ContractError):
        _ = result.report


def test_shared_tuple_helper_keeps_old_selected_wrapper_exact():
    from dataclasses import replace

    import numpy as np

    from gf_common.structured_tokens import CandidateTokens, FrameTokens
    from gf_training.native_tree_dataset import _selected_input
    from gf_training.structured_rollout import CANDIDATE_FIELDS, EVENT_FIELDS, FRAME_FIELDS
    value = json.loads((Path(__file__).parents[1]/'native/tests/fixtures/structured_tokens_v2.json').read_text())[0]
    raw = value['frame_tokens']
    def frozen(item):
        return tuple(frozen(v) for v in item) if type(item) is list else item
    tokens = FrameTokens(**{key:frozen(item) for key,item in raw.items()})
    full = CandidateTokens(**{key:frozen(item) for key,item in value['candidate_tokens'].items()})
    index = min(1,len(full.kinds)-1)
    start,end = full.offsets[index:index+2]
    selected = replace(full,kinds=(full.kinds[index],),targets=(full.targets[index],),numerics=(full.numerics[index],),
        present=(full.present[index],),offsets=(0,end-start),slots=full.slots[start:end])
    frame = {**value['frame'], 'table_hash':full.table_hash,'selected_index':0,'original_selection_index':index,
        'token_prefix':'e.0.f.0','event_token_prefix':'e.0.event-tokens.0'}
    arrays = {}
    for prefix,owner,names in (('e.0.f.0',tokens,FRAME_FIELDS),('e.0.event-tokens.0',tokens,EVENT_FIELDS),
            ('e.0.f.0.candidate',selected,CANDIDATE_FIELDS)):
        for name in names:
            arrays[prefix+'.'+name] = np.asarray(getattr(owner,name),dtype=np.uint8 if 'present' in name or 'mask' in name else np.int64)
    selection = {'table_hash':full.table_hash,'index':index,'policy_version':'private-old-wrapper', 'logprob':None,'value':None}
    result = _selected_input({'actor':tokens.actor,'episode':tokens.episode,'outcome':{'win':{'winner':tokens.actor}}},
        frame,selection,arrays,{'token_signature':tokens.recipe_signature,'encoder':{'aggregate_window':tokens.aggregate_window,'max_positions':18}})
    assert result.frame == tokens and result.candidate == selected and result.original_selection.index == index and result.label == 1.


@pytest.mark.parametrize('mutation', ['gap','overlap','prefix','dtype','extra-bank','mask'])
def test_unadmitted_storage_decoder_rejects_malformed_arrays(mutation):
    import copy

    import numpy as np

    from gf_training.native_card_instance_fixture import _arrays, _semantics, _tokens
    literal = json.loads((Path(__file__).parents[1]/'native/tests/fixtures/card_instance_component_tokens_python.json').read_text())['cases'][0]
    metadata = copy.deepcopy(literal['metadata'])
    groups = tuple(tuple(np.frombuffer(bytes.fromhex(value),dtype=dtype) for value in literal['banks_hex'][dtype])
        for dtype in ('uint8','int64','float32'))
    if mutation == 'gap':
        metadata['arrays'][0]['element_offset'] = 1
    elif mutation == 'overlap':
        metadata['arrays'][1]['bank_index'] = metadata['arrays'][0]['bank_index']
        metadata['arrays'][1]['dtype'] = metadata['arrays'][0]['dtype']
        metadata['arrays'][1]['element_offset'] = metadata['arrays'][0]['element_offset']
    elif mutation == 'prefix':
        metadata['decisions'][0]['token_prefix'] = 'e.1.f.0'
    elif mutation == 'dtype':
        metadata['arrays'][0]['dtype'] = 'float32'
    elif mutation == 'extra-bank':
        groups = (*groups[:2], (np.frombuffer(b'',dtype=np.float32),))
    else:
        arrays = _arrays(metadata,groups)
        key = 'e.0.f.0.state_present'
        arrays[key] = arrays[key].copy(); arrays[key][0] = 2
        with pytest.raises(ContractError):
            _tokens(metadata, metadata['decisions'][0], arrays)
        return
    with pytest.raises(ContractError):
        _semantics(metadata)
        _arrays(metadata,groups)


@pytest.mark.integration
def test_all_nine_cohorts_cpu_predictions_match_healthy_default_pair(native, tmp_path):
    if os.environ.get('GF_RUN_FIXTURE_ADAPTER_SCORES') != '1':
        pytest.skip('opt-in read-only actual healthy paired CPU predictor')
    import hashlib
    from dataclasses import asdict

    from gf_common.card_instance_fixture import CONTRACT, origin
    from gf_models.arrays import sha256
    from gf_models.structured_tree_artifacts import load_artifact_v3
    from gf_models.structured_tree_config import ConfiguredTreeModelV3, configured_tree_v3
    from gf_models.structured_tree_pairing import TrainingBinding, inspect_training_v3
    root = Path('/home/percy/develop/GodFieldBots/runs/card-instance-v2-91d5926/tree-component-acceptance/fixture-default-1')
    fit = root/'fit-default-1'
    expected = {'artifact/inference.json':'cffe4fb62c5f3f58300a33a9866ad50ce0614787ee216d63848e7273bba843f1',
        'artifact/model.txt':'19bbc4bdc1e00ddca169b812e4f01fa68f3d0656059abfea7a6b6bb459cd685c',
        'training.json':'b8962cfae1db47a488919e7fa14500ac5c849c1ac10557a2920ab549569f6b50'}
    assert {name:hashlib.sha256((fit/name).read_bytes()).hexdigest() for name in expected} == expected
    request = json.loads((fit/'request.json').read_text())
    model = configured_tree_v3({**asdict(ConfiguredTreeModelV3()),**request['model'],**request['execution']})
    binding = TrainingBinding(str(fit), sha256(fit/'request.json'), expected['training.json'])
    pair = inspect_training_v3(fit, expected_binding=binding, max_owned_bytes=8*1024**3, deadline=time.monotonic()+1800)
    inference = pair.inference
    tree = load_artifact_v3(fit/'artifact', source_kind='card-instance-tree-fixture-originals', source_origin=origin(),
        expected_digest=expected['artifact/inference.json'], model=model, action_config_hash=inference['action_config_hash'],
        action_contract=CONTRACT, purpose='component', synthetic_capacity=False, training_directory=fit,
        training_binding=binding, deadline=time.monotonic()+1800)
    path = root/'originals/manifest.json'
    from gf_simulator_client.structured_resources import ReservationLedger
    ledger = ReservationLedger(max_workers=1) if os.environ.get('GF_RUN_FIXTURE_ADAPTER_FACADE') == '1' else None
    with TreeFixtureSource(path, sha256(path), PreparationConfig(reader='participant-structured-v2'),
            max_owned_bytes=8*1024**3, deadline=time.monotonic()+1800) as owner:
        result = native_fixture_diagnostics(owner, tmp_path/'paired-nine', max_owned_bytes=8*1024**3,
            deadline=time.monotonic()+1700, diagnostic_native=native, tree=tree, pair=pair,
            extra_owned_bytes=pair.scan.retained, scoring_ledger=ledger)
        report = result.report
        assert report['episodes'] == 288 and len(report['cohorts']) == 9
        assert report['native_cpu_score_rows'] == report['full_feature_rows'] > 288
        if ledger is not None:
            assert report['facade_score_rows'] == report['native_cpu_score_rows']
            assert report['scoring_workers_closed'] == 279 and not ledger.workers
            assert all(row['scorer_managed_bytes'] > 0 if row['decisions'] else row['scorer_managed_bytes'] == 0 for row in report['receipts'])
        complete(owner); owner.verify_complete(); result.guard(); pair.finalize(); result.close()
    pair.finalize()
    assert {name:hashlib.sha256((fit/name).read_bytes()).hexdigest() for name in expected} == expected


@pytest.mark.parametrize('target', ['derived-file','output-membership','binary-file'])
def test_unadmitted_result_guard_stays_failed_after_repair(originals, tmp_path, target):
    # Filesystem primitives only: this result cannot complete/qualify a Source.
    from gf_models.structured_sealed import SealedScan
    from gf_training.native_card_instance_fixture import FixtureTokenDiagnostics
    output = tmp_path/'result'; output.mkdir()
    episode = output/'episode-000000'; episode.mkdir()
    derived = episode/'diagnostic.json'; derived.write_bytes(b'private filesystem guard fixture')
    binary_file = tmp_path/'binary-evidence'; binary_file.write_bytes(b'not a loaded extension')
    binary = SealedScan(max_owned_bytes=LIMIT, deadline=time.monotonic()+60)
    with binary.file(binary_file):
        pass
    persisted = SealedScan(max_owned_bytes=LIMIT, deadline=time.monotonic()+60)
    with persisted.file(derived):
        pass
    persisted.inventory(episode, {'diagnostic.json'})
    ancestors = {path:(path.stat().st_dev,path.stat().st_ino) for path in (output,episode)}
    with source(*originals) as owner:
        result = FixtureTokenDiagnostics(owner,binary,persisted,ancestors,
            {'kind':'unadmitted-filesystem-primitive','output':str(output),'episodes':1},None)
        result.guard()
        if target == 'output-membership':
            file = output/'foreign'; file.write_bytes(b'preserve')
            with pytest.raises(ContractError):
                result.guard()
            assert file.read_bytes() == b'preserve'; file.unlink()
        else:
            file = derived if target == 'derived-file' else binary_file
            original = file.read_bytes(); file.write_bytes(original+b'changed')
            with pytest.raises(ContractError):
                result.guard()
            file.write_bytes(original)
        with pytest.raises(ContractError, match='active fixture token diagnostic'):
            _ = result.report
        with pytest.raises(ContractError, match='all declared'):
            owner.verify_complete()
        complete(owner)


def test_borrow_and_reference_yields_hold_no_original_file_descriptors(originals):
    with source(*originals) as owner:
        roots = tuple(str(path) for path in owner.original_roots)
        def check():
            for path in Path('/proc/self/fd').iterdir():
                try:
                    destination = str(path.readlink())
                except FileNotFoundError:
                    continue
                assert not destination.startswith(roots)
        with owner.borrow_diagnostic_episode(1) as borrow:
            check()
            for decision in borrow.reference_decisions():
                assert decision.sample.viewer in (1,2)
                check()
        complete(owner)


def test_explicit_diagnostic_module_cannot_reinterpret_foreign_fixture_marker():
    from gf_training.native_card_instance_fixture import _preflight
    with pytest.raises(ContractError, match='profile differs'):
        _preflight(SimpleNamespace(build_info=lambda: {'card_instance_fixture_tokens':'foreign-v0'}), diagnostic=True)


def test_active_caller_loop_refuses_before_borrow_or_output(native, tmp_path):
    import asyncio

    from gf_simulator_client.structured_resources import ReservationLedger
    async def run():
        owner = source(tmp_path/'missing/manifest.json', 'a'*64)
        with pytest.raises(ContractError, match='active caller loop'):
            native_fixture_diagnostics(owner,tmp_path/'absent',max_owned_bytes=LIMIT,deadline=time.monotonic()+30,
                diagnostic_native=native,tree=object(),pair=object(),scoring_ledger=ReservationLedger(max_workers=1))
    asyncio.run(run())
    assert not (tmp_path/'absent').exists()


@pytest.mark.parametrize('field', ['frame','table','count','nonfinite','generated'])
def test_unadmitted_facade_reply_cannot_cross_original_table_identity(originals, field):
    from gf_models.card_instance_tree_scoring import ScoringResult
    from gf_training.native_card_instance_fixture import _FacadeLoop
    with source(*originals) as owner:
        with owner.borrow_diagnostic_episode(1) as borrow:
            references = borrow.reference_decisions()
            reference = next(references)
            row = reference.sample
            result = ScoringResult('0'*64 if field == 'frame' else row.tokens.frame_hash,
                '0'*64 if field == 'table' else row.candidates.table_hash,
                (float('nan'),)*len(row.candidates.kinds) if field == 'nonfinite' else
                (0.,)*(len(row.candidates.kinds)-(field == 'count')), b'{}', b'{}' if field == 'generated' else None)
            async def reply(*args):
                return result
            async def close():
                return None
            with _FacadeLoop() as loop:
                loop.worker = SimpleNamespace(score_original=reply,close=close,health={'returncode':0})
                with pytest.raises(ContractError, match='original table'):
                    loop.score(reference)
            list(references)
        complete(owner)


def test_owned_facade_loop_preserves_primary_error_and_reachable_cleanup():
    import asyncio

    from gf_training.native_card_instance_fixture import _FacadeLoop
    loop = _FacadeLoop()
    async def failed_close():
        raise OSError('unconfirmed cleanup primitive')
    loop.worker = SimpleNamespace(close=failed_close,health={'returncode':0})
    primary = asyncio.CancelledError('initiating cancellation')
    loop.__exit__(type(primary),primary,None)
    assert primary._fixture_scoring_cleanup_owner is loop
    assert not loop.loop.is_closed()
    assert any('close/reap' in value for value in primary.__notes__)
    async def repaired_close():
        return None
    loop.worker.close = repaired_close
    loop.close()
    assert loop.loop.is_closed() and loop.worker is None


def _logical_remaining(owner):
    from gf_models.structured_tree_config import (
        ConfiguredTreeModelV3,
        TreeResourcesV2,
        resource_estimate,
    )
    fixed = resource_estimate(ConfiguredTreeModelV3(encoder=owner.encoder,
        resources=TreeResourcesV2(max_owned_bytes=owner.limit)))['total_bytes']
    return owner.limit-owner._scan.retained-fixed-owner._current_bytes


@pytest.mark.parametrize('phase', ['object','envelope'])
def test_diagnostic_logical_caller_reservation_refuses_before_envelope_decoders(originals, monkeypatch, phase):
    import gf_training.card_instance_tree_fixture as module
    path, sha = originals
    manifest = json.loads(path.read_text())
    record = manifest['cohorts'][0]['episodes'][1]
    size = (path.parent/record['file']).stat().st_size
    with pytest.raises(ContractError), source(path,sha) as owner:
        # Logical caller ownership only: no multi-GiB allocation is performed.
        json_charge, envelope_charge = 32*size+65536, 64*size+65536
        headroom = 32768 if phase == 'object' else json_charge+envelope_charge//2
        extra = _logical_remaining(owner)-headroom
        calls = []
        def forbidden(*args, **kwargs):
            calls.append(True)
            pytest.fail('heavy envelope decoder called before joint caller admission')
        if phase == 'object':
            monkeypatch.setattr(owner,'_object',forbidden)
        else:
            monkeypatch.setattr(module.fixture,'envelope',forbidden)
        with owner.borrow_diagnostic_episode(1,extra_owned_bytes=extra):
            pytest.fail('insufficient remaining decode budget was admitted')
    assert calls == []


@pytest.mark.parametrize('phase', ['load','reference'])
def test_diagnostic_column_decode_accounts_envelope_and_external_caller_before_decode(originals, monkeypatch, phase):
    import gf_training.card_instance_tree_fixture as module
    with pytest.raises(ContractError), source(*originals) as owner:
        calls = []
        def forbidden(*args, **kwargs):
            calls.append(True)
            pytest.fail('column decoder called before joint external admission')
        if phase == 'load':
            before_envelope = []
            original_temporary = module._FixtureEpisodeBorrow._admit_temporary
            def temporary(borrow,size):
                if not before_envelope:
                    before_envelope.append(owner._current_bytes)
                return original_temporary(borrow,size)
            monkeypatch.setattr(module._FixtureEpisodeBorrow,'_admit_temporary',temporary)
            original_column = owner._column
            def column(*args):
                checked = original_column(*args)
                borrow = owner._diagnostic_borrow
                assert borrow is not None
                assert owner._current_bytes >= before_envelope[0]+len(borrow.envelope)
                borrow.admit_external(_logical_remaining(owner)-32768)
                return checked
            monkeypatch.setattr(owner,'_column',column)
            monkeypatch.setattr(module,'decode_candidate_columns_v2',forbidden)
        with owner.borrow_diagnostic_episode(1) as borrow:
            if phase == 'reference':
                borrow.admit_external(_logical_remaining(owner)-32768)
                monkeypatch.setattr(module,'decode_candidate_columns_v2',forbidden)
                next(borrow.reference_decisions())
            pytest.fail('insufficient remaining column decode budget was admitted')
    assert calls == []


@pytest.fixture(scope='module')
def healthy_adapter_pairs(tmp_path_factory):
    """Disposable resealed counterexample only; original pair is never modified/refit."""
    import hashlib
    import shutil
    from dataclasses import asdict

    from gf_common.card_instance_fixture import CONTRACT, origin
    from gf_common.domain import canonical
    from gf_models.structured_tree_artifacts import load_artifact_v3
    from gf_models.structured_tree_config import ConfiguredTreeModelV3, configured_tree_v3
    from gf_models.structured_tree_pairing import (
        TREE3_RECORDS,
        inspect_training_v3,
        provenance_for_request,
    )
    pytest.importorskip('lightgbm')
    original = ROOT_PAIR = Path('/home/percy/develop/GodFieldBots/runs/card-instance-v2-91d5926/tree-component-acceptance/fixture-default-1/fit-default-1')
    if not original.is_dir():
        pytest.skip('durable healthy component pair unavailable; no synthetic fit substitute')
    def hash_file(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def write(path,value):
        path.write_text(canonical(value)+'\n')
        return hash_file(path)
    protected = {name:hash_file(original/name) for name in ('request.json','training.json','artifact/inference.json','artifact/model.txt','child-result.json')}
    a = tmp_path_factory.mktemp('actual-adapter-pairs')/'pair-a'
    shutil.copytree(original,a)
    pair_a = inspect_training_v3(a,max_owned_bytes=LIMIT,deadline=time.monotonic()+90)
    request_a = pair_a.request
    model = configured_tree_v3({**asdict(ConfiguredTreeModelV3()),**request_a['model'],**request_a['execution']})
    tree = load_artifact_v3(a/'artifact',source_kind='card-instance-tree-fixture-originals',source_origin=origin(),
        expected_digest=pair_a.completion['outputs']['inference_sha256'],model=model,
        action_config_hash=pair_a.inference['action_config_hash'],action_contract=CONTRACT,purpose='component',
        synthetic_capacity=False,training_directory=a,training_binding=pair_a.binding,deadline=time.monotonic()+90)
    b = a.parent/'pair-b'; shutil.copytree(a,b)
    request = json.loads((b/'request.json').read_text())
    request['run_id'] = 'b'*32 if request['run_id'] != 'b'*32 else 'c'*32
    request_sha = write(b/'request.json',request)
    inference = json.loads((b/'artifact/inference.json').read_text())
    inference['provenance'] = provenance_for_request(request,request_sha,_context=TREE3_RECORDS).to_dict()
    inference_sha = write(b/'artifact/inference.json',inference)
    child = json.loads((b/'child-result.json').read_text())
    child.update(request_sha256=request_sha,inference_sha256=inference_sha)
    child_sha = write(b/'child-result.json',child)
    completion = json.loads((b/'training.json').read_text())
    completion.update(run_id=request['run_id'],request_sha256=request_sha)
    completion['outputs']['inference_sha256'] = inference_sha
    completion['outputs']['files'].update({'request.json':request_sha,'artifact/inference.json':inference_sha,'child-result.json':child_sha})
    write(b/'training.json',completion)
    pair_b = inspect_training_v3(b,max_owned_bytes=LIMIT,deadline=time.monotonic()+90)
    assert pair_a.inference['tree_descriptor'] == pair_b.inference['tree_descriptor']
    assert hash_file(a/'artifact/model.txt') == hash_file(b/'artifact/model.txt')
    assert pair_a.inference['provenance'] != pair_b.inference['provenance']
    assert pair_a.binding.request_sha256 != pair_b.binding.request_sha256
    yield tree,pair_a,pair_b
    assert {name:hash_file(ROOT_PAIR/name) for name in protected} == protected


class _PredictorPreflightBoundary(RuntimeError):
    pass


def _predictor_boundary_probe(native, tmp_path, monkeypatch, tree, pair):
    owner = source(tmp_path/'not-opened/manifest.json','a'*64)
    def before_source():
        raise _PredictorPreflightBoundary('paired predictor accepted before any Source operation')
    monkeypatch.setattr(owner,'guard',before_source)
    monkeypatch.setattr(owner,'borrow_diagnostic_episode',lambda *args,**kwargs:pytest.fail('borrow allocated'))
    native_fixture_diagnostics(owner,tmp_path/'must-not-allocate',max_owned_bytes=LIMIT,
        deadline=time.monotonic()+60,diagnostic_native=native,tree=tree,pair=pair)


def test_crossed_healthy_pair_with_identical_model_refuses_before_source_or_output(native, healthy_adapter_pairs, tmp_path, monkeypatch):
    from dataclasses import replace

    from gf_models.structured_tree_policy import ComponentStructuredTreePolicyV3
    tree,_a,b = healthy_adapter_pairs
    crossed = replace(tree,training_binding=b.binding)
    with pytest.raises(ContractError,match='paired evidence'):
        ComponentStructuredTreePolicyV3(crossed,b.completion['outputs']['inference_sha256'],deadline=time.monotonic()+60)
    with pytest.raises(ContractError,match='paired'):
        _predictor_boundary_probe(native,tmp_path,monkeypatch,crossed,b)
    assert not (tmp_path/'must-not-allocate').exists()


def test_matching_predictor_execution_only_overrides_reach_source_boundary(native, healthy_adapter_pairs, tmp_path, monkeypatch):
    from dataclasses import replace
    tree,pair,_ = healthy_adapter_pairs
    execution = replace(tree.model,checkpoint='/execution-only-checkpoint',
        inference=replace(tree.model.inference,candidate_chunk=17,max_candidates=128),
        resources=replace(tree.model.resources,max_owned_bytes=LIMIT-1024**2))
    actual = replace(tree,model=execution)
    with pytest.raises(_PredictorPreflightBoundary,match='paired predictor accepted'):
        _predictor_boundary_probe(native,tmp_path,monkeypatch,actual,pair)
    assert not (tmp_path/'must-not-allocate').exists()


@pytest.mark.parametrize('mutation', ['origin','architecture','feature-layout','feature-names','categorical-indices','feature-signature'])
def test_actual_predictor_binding_and_full_pair_metadata_refuse_before_source(native, healthy_adapter_pairs, tmp_path, monkeypatch, mutation):
    import copy
    from dataclasses import replace

    from gf_models.structured_tree_artifacts import LoadedTreeV3, TreeOriginV3
    tree,a,b = healthy_adapter_pairs
    candidate = tree
    pair = copy.copy(a)
    pair.inference = copy.deepcopy(a.inference)
    if mutation in ('origin','architecture'):
        # Unsupported wrapper-construction bypass must still fail at preflight;
        # both native Booster and underlying healthy pair remain unchanged.
        candidate = object.__new__(LoadedTreeV3)
        for name in LoadedTreeV3.__dataclass_fields__:
            object.__setattr__(candidate,name,getattr(tree,name))
        if mutation == 'origin':
            origin = TreeOriginV3(tree.origin.action_config_hash,tree.origin.action_contract,'component',False,
                'trained',type(tree.origin.provenance).parse(b.inference['provenance']),tree.origin.source_kind,tree.origin.source_origin)
            object.__setattr__(candidate,'origin',origin)
            object.__setattr__(candidate,'training_binding',b.binding)
            pair = b
        else:
            object.__setattr__(candidate,'model',replace(tree.model,network=replace(tree.model.network,num_leaves=17)))
    elif mutation == 'feature-layout':
        pair.inference['feature_layout']['transforms'][0] = 'foreign-transform'
    elif mutation == 'feature-names':
        pair.inference['feature_names'][0] = 'foreign-feature'
    elif mutation == 'categorical-indices':
        pair.inference['categorical_indices'] = []
    else:
        pair.inference['feature_signature'] = '0'*64
    with pytest.raises(ContractError,match='binding|origin|paired'):
        _predictor_boundary_probe(native,tmp_path,monkeypatch,candidate,pair)
    assert not (tmp_path/'must-not-allocate').exists()


@pytest.mark.parametrize('health', [
    {}, {'returncode':None}, {'returncode':False}, {'returncode':1}, {'returncode':-9},
    {'returncode':0,'exit_code':1}, {'returncode':-15,'exit_code':0}, {'returncode':0,'stderr_bytes':-1},
    *({'returncode':-15,key:False} for key in ('emergency_stop','overrun','cleanup_error','abort','pending_error','unknown')),
])
def test_inert_reaped_scorer_unhealthy_close_is_not_counted_or_discarded(health):
    from gf_training.native_card_instance_fixture import _FacadeLoop
    loop = _FacadeLoop()
    async def close():
        return None
    worker = SimpleNamespace(close=close,health=health)
    loop.worker = worker
    try:
        with pytest.raises(ContractError):
            loop.close_worker()
        assert loop.worker is worker and loop.workers_closed == 0 and loop.closed_health == []
        primary = OSError('initiating comparison failure')
        loop.__exit__(type(primary),primary,None)
        assert primary._fixture_scoring_cleanup_owner is loop
        assert loop.worker is worker and loop.workers_closed == 0
    finally:
        # Inert primitive has no process or pending task to reap. This does not
        # claim a healthy close or permit any operational production retry.
        loop.loop.close()


@pytest.mark.parametrize('returncode', [0, -15])
def test_inert_normal_exit_and_sigterm_close_are_recorded(returncode):
    from gf_training.native_card_instance_fixture import _FacadeLoop
    async def close():
        return None
    with _FacadeLoop() as loop:
        loop.worker = SimpleNamespace(close=close,health={'returncode':returncode,'exit_code':returncode,'stderr_bytes':0})
        loop.close_worker()
        assert loop.worker is None and loop.workers_closed == 1
        assert loop.closed_health == [{'returncode':returncode,'exit_code':returncode,'stderr_bytes':0}]


def test_facade_health_is_readonly_snapshot_with_actual_process_returncode():
    from gf_simulator_client.card_instance_tree_scoring_worker import CardInstanceTreeScoringWorker
    worker = object.__new__(CardInstanceTreeScoringWorker)
    worker._transport = SimpleNamespace(health={'exit_code':-15,'returncode':99},process=SimpleNamespace(poll=lambda:-15))
    snapshot = worker.health
    assert snapshot == {'exit_code':-15,'returncode':-15}
    snapshot['emergency_stop'] = True
    assert worker._transport.health == {'exit_code':-15,'returncode':99}
    assert worker.health == {'exit_code':-15,'returncode':-15}


def test_real_component_scorer_close_records_actual_process_returncode():
    from gf_training.native_card_instance_fixture import _FacadeLoop
    from tests.test_card_instance_tree_scoring_worker import worker
    with _FacadeLoop() as loop:
        loop.worker = worker()
        loop.run(loop.worker.initialize())
        loop.close_worker()
        assert loop.workers_closed == 1 and loop.worker is None
        assert loop.closed_health[0]['returncode'] in (0,-15)
        assert set(loop.closed_health[0]) <= {'returncode','exit_code','stderr_bytes'}
