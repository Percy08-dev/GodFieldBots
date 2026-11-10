"""Private candidate checks; packed storage conveys no game/learning authority."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace

import pytest

from gf_common.domain import ContractError, canonical
from gf_common.group_pack import PackMember, canonical_json, encode_pack, unpack_raw
from gf_training.native_group_pack import OriginalArtifactProvider
from tests.test_card_instance_tree_raw import CASE, raw_settings, write_case
from tests.test_group_pack_game_source import owner
from tests.test_group_pack_storage import writer


@pytest.fixture
def candidate():
    native = pytest.importorskip('gf_native')
    if not callable(getattr(getattr(native, 'CardInstanceNativeSession', None), 'from_artifacts', None)):
        pytest.skip('private Native3B candidate required; installed ELF stays unchanged')
    return native


def raw_original(tmp_path):
    directory = tmp_path / 'directory'
    directory.mkdir()
    journal = write_case(directory)
    (directory / '診断.json').write_bytes(b'opaque diagnostics are not parsed as JSON')
    w = writer(tmp_path / 'archive')
    episode = CASE['rows'][0]['episode_id']
    members = tuple(PackMember('journal' if p.suffix == '.jsonl' else 'candidates',
        f'{episode}/{p.name}', p.read_bytes()) for p in sorted(directory.iterdir()))
    pack = encode_pack(members)
    w.submit(episode, pack, raw_bytes=len(unpack_raw(pack)), record={}, episodes=[{
        'episode_id': episode, 'game_id': CASE['rows'][0]['game_id'], 'record': {}}])
    w.store_control('manifest.json', b'{}')
    w.store_control('report.json', b'{}')
    locator = w.seal()
    root = tmp_path / 'logical-root'
    root.mkdir()
    (root / 'collection.locator.json').write_bytes(canonical_json({
        'format': 'sqlite-group-pack-locator-v1', 'schema_version': 1,
        'descriptor': str(locator.path), 'descriptor_sha256': hashlib.sha256(locator.path.read_bytes()).hexdigest()}))
    return journal, owner(root)


def test_raw_original_bytes_tokens_candidates_selections_match_directory(candidate, tmp_path):
    journal, original = raw_original(tmp_path)
    provider = OriginalArtifactProvider((original,))
    reference = packed = None
    try:
        reference = candidate.StructuredJournalTokens.card_instance(str(journal.parent), journal.name,
            raw_settings(), time.monotonic() + 60)
        packed = candidate.StructuredJournalTokens.card_instance_artifacts(str(original.root), CASE['rows'][0]['episode_id'],
            journal.name, provider, raw_settings(memory_bytes=512 * 1024**2 - provider.external_bytes),
            time.monotonic() + 60)
        actual = packed.diagnostic()
        assert actual == reference.diagnostic()
        from gf_common.card_instance import action_config_v3
        from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
        from gf_common.structured_tokens import (
            CandidateTokens,
            FrameTokens,
            tokenize_frame,
            tokenize_table,
        )
        from gf_models.structured_tree_config import ConfiguredTreeModelV3
        from gf_models.structured_tree_encoder import PreparedTreeFrameV3
        from gf_training.card_instance_journal import read_card_instance_journal
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value
        raw = json.loads(actual)
        python = read_card_instance_journal(journal, actions=action_config_v3(raw['header']['actions']))
        for row, decision in zip(raw['decisions'], python.decisions, strict=True):
            frame = FrameTokens(**{key: tuples(value) for key, value in row['frame_tokens'].items()})
            table = CandidateTokens(**{key: tuples(value) for key, value in row['candidate_tokens'].items()})
            expected_frame = tokenize_frame(decision.frame, python.bank, aggregate_window=128, layout=CORRECTED_CARD_INSTANCE_LAYOUT)
            expected_table = tokenize_table(decision.generated.table, decision.frame, aggregate_window=128, layout=CORRECTED_CARD_INSTANCE_LAYOUT)
            model = ConfiguredTreeModelV3()
            native_features = PreparedTreeFrameV3.create(frame, model).iter_chunks(table)
            python_features = PreparedTreeFrameV3.create(expected_frame, model).iter_chunks(expected_table)
            for (start, a), (expected_start, b) in zip(native_features, python_features, strict=True):
                assert start == expected_start and a.dtype == b.dtype and a.shape == b.shape
                assert a.tobytes() == b.tobytes()
        assert not provider._leases and original.artifacts._loans == 0
        packed.recheck()
    finally:
        if packed is not None:
            packed.close()
            assert packed.used_bytes() == 0
        if reference is not None:
            reference.close()
        provider.close()
        original.close()
    assert original.pool.closed and original.pool.budget.used == 0


@pytest.mark.parametrize('fault', ['data-type', 'data-hash', 'data-raise', 'close'])
def test_bridge_explicitly_closes_every_admitted_loan_on_failure(candidate, tmp_path, monkeypatch, fault):
    _, original = raw_original(tmp_path)
    provider = OriginalArtifactProvider((original,))
    original_borrow = provider.borrow
    loans = []
    def borrow(root, name):
        loan = original_borrow(root, name)
        loans.append(loan)
        if fault == 'data-type':
            monkeypatch.setattr(loan, 'data', lambda: bytearray(b'x'))
        elif fault == 'data-hash':
            monkeypatch.setattr(loan, 'data', lambda: b'x')
        elif fault == 'data-raise':
            monkeypatch.setattr(loan, 'data', lambda: (_ for _ in ()).throw(RuntimeError('data failed')))
        else:
            monkeypatch.setattr(loan, 'close', lambda: (_ for _ in ()).throw(RuntimeError('close failed')))
        return loan
    monkeypatch.setattr(provider, 'borrow', borrow)
    try:
        with pytest.raises((ValueError, RuntimeError)) as failure:
            candidate.StructuredJournalTokens.card_instance_artifacts(str(original.root), CASE['rows'][0]['episode_id'],
                'player-1.jsonl', provider, raw_settings(), time.monotonic() + 60)
        assert failure.value._native_original_provider_cleanup_owner is provider
        assert loans
        if fault == 'close':
            assert provider._leases and original.artifacts._loans == 1
            with pytest.raises(ContractError, match='leases'):
                provider.close()
        else:
            assert all(loan._closed for loan in loans)
            assert not provider._leases and original.artifacts._loans == 0
    finally:
        monkeypatch.undo()
        for loan in loans:
            loan.close()
        provider.close()
        original.close()


def test_old_constructor_refuses_plan4_before_provider_or_filesystem_io(candidate, tmp_path):
    from gf_common.group_pack_game import storage_binding
    plan = {'schema_version': 4, 'mode': 'historical_parent',
        'parents': [{'directory': '/no-such-native-pack-parent', 'require_complete': False}], 'collections': [],
        'storage_binding': storage_binding(), 'artifact_registry_sha256': '0' * 64}
    settings = canonical({'memory_bytes': 512 * 1024**2, 'scratch_bytes': 1024**2,
        'max_source_file_bytes': 64 * 1024**2, 'max_record_bytes': 8 * 1024**2,
        'chunk_target_bytes': 1024**2}).encode()
    with pytest.raises(ValueError):
        candidate.CardInstanceNativeSession(canonical(plan).encode(), settings, time.monotonic() + 30)


def test_cancel_after_original_data_closes_the_actual_loan(candidate, tmp_path, monkeypatch):
    _, original = raw_original(tmp_path)
    provider = OriginalArtifactProvider((original,))
    handle = candidate.CancelHandle(time.monotonic() + 30)
    borrow_original = provider.borrow
    loans = []
    def borrow(root, name):
        loan = borrow_original(root, name)
        loans.append(loan)
        data = loan.data
        def cancel_after_data():
            raw = data()
            handle.cancel()
            return raw
        monkeypatch.setattr(loan, 'data', cancel_after_data)
        return loan
    monkeypatch.setattr(provider, 'borrow', borrow)
    try:
        with pytest.raises(RuntimeError, match='cancelled') as failure:
            candidate.StructuredJournalTokens.card_instance_artifacts(str(original.root), CASE['rows'][0]['episode_id'],
                'player-1.jsonl', provider, raw_settings(), time.monotonic()+30, handle)
        assert failure.value._native_original_provider_cleanup_owner is provider
        assert loans and all(loan._closed for loan in loans)
        assert original.artifacts._loans == 0 and not provider._leases
    finally:
        provider.close()
        original.close()


@pytest.mark.parametrize('fault', ['missing-index', 'extra-row-field', 'duplicate-group', 'duplicate-key', 'nonfinite'])
def test_provider_metadata_refuses_before_any_original_member_borrow(candidate, tmp_path, monkeypatch, fault):
    _, original = raw_original(tmp_path)
    provider = OriginalArtifactProvider((original,))
    describe_original = provider.describe
    def describe(root):
        raw = describe_original(root)
        value = json.loads(raw)
        if fault == 'missing-index':
            del value['index']
        elif fault == 'extra-row-field':
            value['index']['episodes'][0]['unexpected'] = None
        elif fault == 'duplicate-group':
            value['index']['groups'].append(value['index']['groups'][0])
        elif fault == 'duplicate-key':
            return raw.replace(b'"index":{', b'"index":{},"index":{', 1)
        else:
            return raw.replace(b'"record":{}', b'"record":{"bad":1e400}', 1)
        return canonical(value).encode()
    monkeypatch.setattr(provider, 'describe', describe)
    monkeypatch.setattr(provider, 'borrow', lambda *_: pytest.fail('malformed metadata reached original member I/O'))
    try:
        with pytest.raises(ValueError):
            candidate.StructuredJournalTokens.card_instance_artifacts(str(original.root), CASE['rows'][0]['episode_id'],
                'player-1.jsonl', provider, raw_settings(), time.monotonic()+30)
        assert original.pool.active is None and not provider._leases
    finally:
        provider.close()
        original.close()


def test_group_pack_marker_is_separate_from_existing_native_capability(monkeypatch):
    from types import SimpleNamespace

    from gf_common.preprocessing import PreparationConfig
    from gf_training import native_backend
    settings = PreparationConfig(backend='rust', reader='participant-structured-v2')
    monkeypatch.setattr(native_backend, 'card_instance_reader_settings', lambda config, module: {'checked': True})
    module = SimpleNamespace(build_info=list, CardInstanceNativeSession=SimpleNamespace(from_artifacts=lambda: None),
        StructuredJournalTokens=SimpleNamespace(card_instance_artifacts=lambda: None))
    with pytest.raises(ContractError, match='capability'):
        native_backend.group_pack_reader_settings(settings, module)
    module.build_info = lambda: [('card_instance_group_pack', native_backend.CARD_INSTANCE_GROUP_PACK)]
    assert native_backend.group_pack_reader_settings(settings, module) == {'checked': True}
    module.CardInstanceNativeSession.from_artifacts = None
    with pytest.raises(ContractError, match='ABI'):
        native_backend.group_pack_reader_settings(settings, module)


@pytest.fixture(params=['mlp', 'gru', 'transformer'])
def packed_parent(candidate, tmp_path, monkeypatch, request):
    """One complete controlled wire5 update; no actual game/formal claim."""
    import asyncio

    from gf_training.card_instance_ppo import train_card_instance_ppo
    from tests import test_group_pack_game_ppo as fixtures
    from tests.test_card_instance_learning import learner_for
    if os.environ.get('GFB_RUN_NATIVE_GROUP_PACK_SOURCE') != '1':
        pytest.skip('explicit bounded native packed Source fixture check')
    saved_parent = os.environ.get('GFB_NATIVE_PACK_PARENT')
    if saved_parent is not None:
        from pathlib import Path

        from gf_common.card_instance import action_config_v3
        from gf_common.preprocessing import preparation_config
        directory = Path(saved_parent)
        state = json.loads((directory / 'run.json').read_bytes())
        assert state['schema_version'] == 4 and state['updates'] == 1 and 'run_variant' not in state
        assert state['model_id'] == request.param
        return directory, preparation_config(state['effective_config']['preprocessing']), action_config_v3(state['effective_config']['actions'])
    import torch
    torch.set_num_threads(1)
    monkeypatch.setattr(fixtures, 'learner_for', lambda: learner_for(temporal=request.param))
    loaded, collector, _ = fixtures.setup(tmp_path, monkeypatch)
    directory = tmp_path / 'parent'
    result = asyncio.run(train_card_instance_ppo(loaded, collector, directory, pause_after_update=1))
    assert result['updates'] == 1 and result['status'] == 'paused'
    print('NATIVE_PACK_FIXTURE_PARENT=' + str(directory), flush=True)
    return directory, loaded.config.preprocessing, loaded.config.actions


def test_packed_source_proof_correspondence_original16_and_numeric_parity(candidate, packed_parent, tmp_path):
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.domain import digest
    from gf_models.sharded_arrays import read_bundle
    from gf_training.native_card_instance import _NativeFinalValidationInputs, _NativeGameInputs
    from gf_training.native_card_instance_final_validation import _decode_recorded_episode_packet
    from gf_training.structured_rollout import (
        _episode_correspondence_digest,
        _source_correspondence,
    )
    from gf_training.structured_source import _inspect_component_collection
    directory, settings, actions = packed_parent
    original = owner(directory / 'collections' / 'batch-000000')
    reference = _inspect_component_collection(original, actions, 'train', _game=True)
    seats = reference.manifest['linkage']['learner_seats']
    selected = tuple(row for row in reference.episodes if seats[row.episode_id] == row.viewer)
    assert len(selected) == 16
    provider = OriginalArtifactProvider((original,))
    plan = {'schema_version': 4, 'mode': 'historical_parent',
        'parents': [{'directory': str(directory), 'require_complete': False}], 'collections': [],
        'storage_binding': json.loads((directory / 'run.json').read_bytes())['storage_binding'],
        'artifact_registry_sha256': provider.registry_sha256}
    try:
        with _NativeGameInputs(plan, replace(settings, backend='rust'), actions=actions,
                max_owned_bytes=1024**3, deadline=time.monotonic()+180, native=candidate,
                artifact_provider=provider) as inputs:
            saved_update = json.loads((directory / 'updates' / '000001.json').read_bytes())
            assert canonical(inputs.proof) == canonical(saved_update['source_proof'])
            assert canonical(inputs.proof) == canonical(reference.proof)
            state = json.loads((directory / 'run.json').read_bytes())
            immutable = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ('generations/000000.json', 'artifacts/000000/inference.json')}
            fingerprint = digest({'recipe': 'card-instance-packed-source-controls-v1',
                'run_id': state['run_id'], 'config_sha256': state['config_sha256'], 'identity': state['identity'],
                'execution': state['execution'], 'plan': reference.manifest['linkage'], 'files': immutable,
                'source_set_sha256': reference.proof['source_set_sha256'], 'storage': reference.proof['storage'],
                'custody': reference.proof['custody']})
            episode_digests = tuple(_episode_correspondence_digest(row.records) for row in selected)
            assert canonical(inputs.correspondence) == canonical(_source_correspondence(fingerprint, inputs.collection['semantics'],
                episode_digests, game=True, storage_binding=state['storage_binding']))
            assert inputs.summary['kind'] == 'card-instance-packed-source-controls'
            assert inputs.summary['parent']['kind'] == 'card-instance-packed-parent-controls'
            assert inputs.original_config_inputs(0) == []
            destination = tmp_path / 'native-numeric'
            exported = inputs.compare_recorded(0, destination)
            expected = json.loads((directory / 'numeric' / 'batch-000000' / 'rollout.json').read_bytes())
            assert canonical(exported) == canonical(expected)
            actual_arrays = read_bundle(destination, exported['data'])
            expected_arrays = read_bundle(directory / 'numeric' / 'batch-000000', expected['data'])
            assert set(actual_arrays) == set(expected_arrays)
            for key, array in actual_arrays.items():
                assert array.dtype == expected_arrays[key].dtype
                assert array.shape == expected_arrays[key].shape
                assert array.tobytes() == expected_arrays[key].tobytes(), key
            del actual_arrays, expected_arrays
            # The unchanged Original16 DTO binds the SAME proof2/source set and
            # own episode digest. This diagnostic cannot construct a parent loan.
            for index, expected_episode in enumerate(selected):
                packet = inputs._native_call(inputs._session.export_recorded_episode_payload, lambda held, ordinal=index: (0, ordinal, held))
                metadata = json.loads(packet[0])
                assert metadata['schema_version'] == 1 and metadata['kind'] == 'card-instance-recorded-episode-payload'
                assert metadata['source_set_sha256'] == inputs.proof['source_set_sha256']
                assert metadata['episode_digest'] == episode_digests[index]
                assert metadata['summary_sha256'] == digest(inputs.summary)
                bound = _NativeFinalValidationInputs._decode_bound(packet)
                def reserve(amount, limit=bound):
                    assert amount <= limit
                decoded = _decode_recorded_episode_packet(packet, summary=inputs.summary,
                    collection_index=0, episode_index=index, episode_id=expected_episode.episode_id,
                    viewer=expected_episode.viewer, actions=actions, rules=card_instance_rules(),
                    aggregate_window=inputs.models[0].event.aggregate_window,
                    max_owned_bytes=1024**3, retained_owned_bytes=inputs.joint_owned_bytes,
                    max_record_bytes=settings.max_record_mib * 1024**2, reserve=reserve, check=inputs._check)
                assert _episode_correspondence_digest(decoded) == episode_digests[index]
                del decoded, metadata, packet
                inputs.recheck()
    finally:
        provider.close()
        original.close()
    assert original.pool.budget.used == 0
