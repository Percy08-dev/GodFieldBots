"""Controls-only original J5/columns/raw-token differential; no Source authority."""
import json
import os
import time
from dataclasses import asdict, replace

import pytest

from gf_common.action_contract import engineering_contract
from gf_common.domain import ContractError, canonical
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_common.structured_wire import frame_dict
from gf_training.structured_journal import read_structured_journal
from tests.test_structured_protocol import journal_setup

native = pytest.importorskip('gf_native')
pytestmark = pytest.mark.skipif(not hasattr(native, 'StructuredJournalTokens'), reason='C2b1 diagnostic candidate required')
SETTINGS = canonical({'memory_bytes': 512 * 1024**2, 'max_json_bytes': 64 * 1024**2, 'max_record_bytes': 8 * 1024**2}).encode()


def journal(make_input, directory, *, natural=True, accepted=True, kind='random', window=128, input_value=None):
    writer, frame, delta, generated, selection, command = journal_setup(make_input, directory,
        model_id=kind, model_version='1' if kind == 'random' else '2', aggregate_window=window, input_value=input_value)
    if kind != 'random':
        selection = replace(selection, logprob=-.12345678901234567, value=.9876543210123456)
    index = writer.begin_frame(frame, delta, generated, selection, command, {'request_ms': 1.})
    outcome = {'win': {'winner': 1}}
    if accepted:
        writer.acknowledge(index, {'command_id': command['command_id'], 'previous_revision': frame.revision,
            'state_revision': frame.revision + 1, 'status': 'terminal' if natural else 'active',
            'result': outcome if natural else None})
    final = frame.observation
    if natural:
        final.update(state_revision=1, actionable=False, typed_action_schema=[], required_choice=None,
                     current_player=None, phase='terminal', result=outcome)
    writer.finish('natural' if natural else 'truncated', outcome if natural else None, final if natural else None)
    return writer.final_path


def owner(path, **settings):
    encoded = SETTINGS if not settings else canonical({'memory_bytes': 512 * 1024**2,
        'max_json_bytes': 64 * 1024**2, 'max_record_bytes': 8 * 1024**2, **settings}).encode()
    return native.StructuredJournalTokens(str(path.parent), path.name, encoded, time.monotonic() + 60)


@pytest.mark.parametrize('kind,window', [('random',128), ('mlp',1), ('gru',1024), ('transformer',65536)])
def test_original_j5_every_raw_token_and_full_precision_selection_matches_python(make_input, tmp_path, kind, window):
    path = journal(make_input, tmp_path, kind=kind, window=window)
    original = read_structured_journal(path, action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    core = owner(path)
    try:
        encoded = core.diagnostic()
        value = json.loads(encoded)
        assert value['kind'] == 'structured-journal-tokens'
        assert value['validation_scope'] == 'journal5-columns-rawtokens-only'
        assert value['qualified'] is False and value['parent_qualification'] == 'not_performed'
        assert value['projection'] == {'aggregate_recipe': 'public-exact-v2', 'aggregate_window': window,
            'declaration': 'baseline-corrected128' if kind == 'random' else 'neural-own-header'}
        assert value['header'] == original.header and value['accepted'] == list(original.accepted)
        assert value['bank']['events'] == list(original.bank.events())
        assert value['bank']['prefixes'] == list(original.bank._prefixes)
        assert value['end'] == original.end
        for expected, actual in zip(original.decisions, value['decisions'], strict=True):
            assert actual['frame'] == frame_dict(expected.frame)
            assert actual['table'] == json.loads(canonical(asdict(expected.generated.table)))
            assert actual['selection'] == asdict(expected.selection)
            assert actual['search_report'] == json.loads(canonical(asdict(expected.generated.report)))
            assert actual['frame_tokens'] == json.loads(canonical(asdict(tokenize_frame(expected.frame, original.bank, aggregate_window=window))))
            assert actual['candidate_tokens'] == json.loads(canonical(asdict(tokenize_table(expected.generated.table, expected.frame, aggregate_window=window))))
        assert core.used_bytes() >= core.control_reserved_bytes() > 0
        core.recheck(1024)
    finally:
        core.close()
    assert core.used_bytes() == 0
    with pytest.raises(RuntimeError):
        core.diagnostic()
    assert 'structured_preprocessing_contract' not in dict(native.build_info())


@pytest.mark.parametrize('accepted', [False, True])
def test_interrupted_closure_keeps_only_positive_ack_indices_and_none_targets(make_input, tmp_path, accepted):
    path = journal(make_input, tmp_path, natural=False, accepted=accepted)
    core = owner(path)
    try:
        result = json.loads(core.diagnostic())
        assert result['accepted'] == ([0] if accepted else [])
        assert result['end']['end_kind'] == 'truncated'
        assert result['end']['outcome'] is result['end']['terminal_reward'] is result['end']['final_observation'] is None
        assert result['end']['pending_decisions'] == ([] if accepted else [0])
    finally:
        core.close()


@pytest.mark.parametrize('mutation', ['version', 'old_signature', 'frame_hash', 'prefix', 'foreign_actor',
    'selection', 'ack', 'closure', 'bool_revision', 'blank_record', 'unknown_record', 'report_rng'])
def test_rehashed_structural_corruption_rejects_originals(make_input, tmp_path, mutation):
    path = journal(make_input, tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    decision = next(row for row in rows if row['kind'] == 'decision')
    if mutation == 'version':
        rows[0]['schema_version'] = 4
    elif mutation == 'old_signature':
        from gf_common.structured_history import HISTORICAL_AGGREGATE
        from gf_common.structured_tokens import signature
        rows[0]['token_signature'] = signature(128, aggregate_recipe=HISTORICAL_AGGREGATE)
    elif mutation == 'frame_hash':
        decision['frame']['frame_hash'] = 'a' * 64
    elif mutation == 'prefix':
        next(row for row in rows if row['kind'] == 'events')['delta']['previous_hash'] = 'a' * 64
    elif mutation == 'foreign_actor':
        decision['frame']['actor'] = 2
    elif mutation == 'selection':
        decision['selection']['index'] = 65535
    elif mutation == 'ack':
        next(row for row in rows if row['kind'] == 'acknowledgement')['response']['state_revision'] += 1
    elif mutation == 'closure':
        rows[-1]['end_kind'] = 'unknown'
    elif mutation == 'bool_revision':
        decision['command']['expected_revision'] = False
    elif mutation == 'unknown_record':
        rows.insert(-1, {'kind': 'unknown'})
    elif mutation == 'report_rng':
        decision['search_report']['rng_before'][1][0] ^= 1
    path.write_text('\n'.join(canonical(row) for row in rows) + '\n' + ('\n' if mutation == 'blank_record' else ''))
    if mutation == 'old_signature':
        # Python's historical structural reader preserves the recorded62 recipe;
        # this active raw-token diagnostic deliberately projects corrected376 only.
        read_structured_journal(path, action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    else:
        with pytest.raises((ContractError, ValueError, json.JSONDecodeError)):
            read_structured_journal(path, action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    with pytest.raises(ValueError):
        owner(path)


def test_raw_sidecar_hash_and_final_same_owner_snapshots(make_input, tmp_path):
    path = journal(make_input, tmp_path)
    core = owner(path)
    raw = next(tmp_path.glob('*.columns'))
    raw.write_bytes(raw.read_bytes()[:-1] + b'\xff')
    try:
        with pytest.raises(ValueError, match='changed|snapshot'):
            core.recheck()
        with pytest.raises(ValueError, match='changed|snapshot'):
            core.diagnostic()
    finally:
        with pytest.raises(ValueError):
            core.close()
    assert core.used_bytes() == 0


@pytest.mark.parametrize('kind', ['fifo', 'symlink'])
def test_nonregular_journal_open_never_follows_or_blocks(tmp_path, kind):
    path = tmp_path / 'player-1.jsonl'
    if kind == 'fifo':
        os.mkfifo(path)
    else:
        outside = tmp_path / 'unowned'
        outside.write_bytes(b'preserved')
        path.symlink_to(outside)
    started = time.monotonic()
    with pytest.raises((OSError, ValueError)):
        owner(path)
    assert time.monotonic() - started < 2


def test_expired_cancelled_and_memory_admission(make_input, tmp_path):
    path = journal(make_input, tmp_path)
    with pytest.raises(MemoryError):
        owner(path, memory_bytes=1)
    with pytest.raises(TimeoutError):
        native.StructuredJournalTokens(str(tmp_path), path.name, SETTINGS, time.monotonic() - 1)
    handle = native.CancelHandle(time.monotonic() + 60)
    handle.cancel()
    with pytest.raises(RuntimeError, match='cancel'):
        native.StructuredJournalTokens(str(tmp_path), path.name, SETTINGS, time.monotonic() + 60, handle)


@pytest.mark.parametrize('actor', [1, 2])
def test_two_original_frames_fog_dream_interval_and_future_prefix_isolation(make_input, tmp_path, actor):
    import random
    import uuid

    from gf_common.diverse_actions import PublicCandidateGeneratorV2
    from gf_common.domain import digest
    from gf_common.structured import EventBank
    from gf_models.structured_baselines import make_structured_policy
    from gf_models.structured_runtime import own_worker_settings
    from gf_training.structured_journal import StructuredJournal
    from tests.test_structured_protocol import action_settings, own_frame
    actions = action_settings()
    caps, costs, _, _ = own_worker_settings('random', actions, None)
    seed = 2**64 - 1
    policy = make_structured_policy('random', action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    policy.reset(seed)
    rng = random.Random(int(digest({'recipe': 'structured-search-stream-v1', 'sampling_seed': seed})[:16], 16))
    bank = EventBank(actor, 'own-two-frame-bank')
    writer = StructuredJournal(tmp_path, 'own-two', actor, policy.version, run_id='component', game_id='own',
        action_config_hash=digest(asdict(actions)), bank_episode=bank.episode, caps=asdict(caps), numeric_costs=asdict(costs),
        model_id='random', model_version='1', sampling_seed=seed, action_contract=engineering_contract(),
        purpose='component', synthetic_capacity=False)
    frames = []
    for revision, definition in [(0, 'bronze-club'), (2, 'angel-axe')]:
        value = make_input(hand=(definition,), learned=('fireball', 'aura'))
        raw = value.to_dict()
        obs = raw['observation']
        obs.update(viewer_player_id=actor, current_player=actor, active_player=actor)
        obs['own']['player_id'], obs['opponent']['player_id'] = actor, 3 - actor
        obs['own']['status'].update(fog=True, dream=revision == 2)
        obs['opponent'] = {key: obs['opponent'][key] for key in ('player_id', 'learned_miracles', 'ascended')}
        raw['action_options']['viewer_player_id'] = actor
        value = replace(value, observation=obs, action_options=raw['action_options'])
        events = [{'type': 'artifact_used', 'player': actor, 'definition_id': 'bronze-club'}] if revision == 0 else [
            {'type': 'miracle_used', 'player': 3 - actor, 'definition_id': 'aura', 'mp_spent': 5}]
        frame, delta = own_frame(value, bank, revision, events)
        generated = PublicCandidateGeneratorV2(actions, clock=lambda: 0.).generate(frame, bank.view(), rng, costs)
        selection = policy.select_frame(frame, bank.view(), generated.table)
        command = {'command_id': str(uuid.uuid4()), 'expected_revision': revision,
                   'action': selection.validate(generated.table, frame)}
        index = writer.begin_frame(frame, delta, generated, selection, command, {})
        writer.acknowledge(index, {'command_id': command['command_id'], 'previous_revision': revision,
            'state_revision': revision + 1, 'status': 'active', 'result': None})
        frames.append(frame)
    writer.finish('cancelled', None)
    data = read_structured_journal(writer.final_path, action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    core = owner(writer.final_path)
    try:
        raw = json.loads(core.diagnostic())
        assert raw['accepted'] == [0, 1] and raw['end']['end_kind'] == 'cancelled'
        assert raw['bank']['actor'] == actor and len(raw['bank']['events']) == 2
        for i, frame in enumerate(frames):
            expected = tokenize_frame(frame, data.bank)
            assert raw['decisions'][i]['frame_tokens'] == json.loads(canonical(asdict(expected)))
            assert len(expected.interval_categories) == 1
        first, second = [row['frame_tokens'] for row in raw['decisions']]
        assert first['card_categories'][0] != second['card_categories'][0]
        assert first['state_present'] != second['state_present'] or first['state_numerics'] != second['state_numerics']
        from gf_common import structured_vocab as vocab
        assert first['history'][62 + vocab.DEFINITIONS.index('aura')] == 0
        assert second['history'][62 + vocab.DEFINITIONS.index('aura')] == 1
        assert first['history'][62 + vocab.DEFINITIONS.index('bronze-club')] == second['history'][62 + vocab.DEFINITIONS.index('bronze-club')] == 1
    finally:
        core.close()


@pytest.mark.parametrize('end_kind', ['natural', 'truncated', 'aborted', 'cancelled', 'incomplete'])
def test_zero_frame_original_closure_is_preserved(make_input, tmp_path, end_kind):
    writer, frame, _, _, _, _ = journal_setup(make_input, tmp_path)
    final = frame.observation
    final.update(actionable=False, typed_action_schema=[], required_choice=None, current_player=None,
                 phase='terminal', result='draw')
    writer.finish(end_kind, 'draw' if end_kind == 'natural' else None, final if end_kind == 'natural' else None)
    core = owner(writer.final_path)
    try:
        raw = json.loads(core.diagnostic())
        assert raw['decisions'] == raw['accepted'] == raw['bank']['events'] == []
        assert len(raw['bank']['prefixes']) == 1 and raw['end']['end_kind'] == end_kind
        assert raw['end']['terminal_reward'] == (0 if end_kind == 'natural' else None)
    finally:
        core.close()


@pytest.mark.parametrize('mutation', ['padding', 'kind', 'target', 'presence', 'offset', 'absent_numeric', 'trailing'])
def test_rehashed_raw_column_corruption_rejects_before_use(make_input, tmp_path, mutation):
    import hashlib
    import struct
    path = journal(make_input, tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    decision = next(row for row in rows if row['kind'] == 'decision')
    descriptor = decision['table']
    sidecar = tmp_path / descriptor['file']
    payload = bytearray(sidecar.read_bytes())
    columns = {column['name']: column for column in descriptor['columns']}
    if mutation == 'padding':
        from gf_common.candidate_columns import column_layout
        descriptor.update(count=1, elements=0)
        descriptor['columns'], descriptor['bytes'] = column_layout(1, 0)
        payload = bytearray(descriptor['bytes'])
        payload[2] = 1  # the fixed u8/u8-to-i64 alignment gap
    elif mutation in ('kind', 'target', 'presence'):
        field = {'kind': 'kinds', 'target': 'targets', 'presence': 'presence'}[mutation]
        payload[columns[field]['offset']] = 255
    elif mutation == 'offset':
        struct.pack_into('<q', payload, columns['offsets']['offset'], -1)
    elif mutation == 'absent_numeric':
        struct.pack_into('<q', payload, columns['numerics']['offset'], 1)
    else:
        payload += b'foreign'
    sidecar.write_bytes(payload)
    descriptor['sha256'] = hashlib.sha256(payload).hexdigest()
    path.write_text(''.join(canonical(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        owner(path)


@pytest.mark.parametrize('mutation', ['synthetic', 'duplicate_physical', 'too_many', 'nullable_neural', 'float_schema', 'unknown_member'])
def test_strict_header_or_game_proposal_cannot_elevate_classification(make_input, tmp_path, mutation):
    import hashlib
    import struct

    from gf_common.domain import digest
    path = journal(make_input, tmp_path, kind='mlp' if mutation == 'nullable_neural' else 'random')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    decision = next(row for row in rows if row['kind'] == 'decision')
    if mutation == 'synthetic':
        decision['table']['synthetic_capacity'] = True
    elif mutation == 'nullable_neural':
        decision['selection']['value'] = None
    elif mutation == 'float_schema':
        rows[0]['schema_version'] = 5.0
    elif mutation == 'unknown_member':
        rows[0]['extra'] = 'not owned metadata'
    else:
        # Replace every row's compact own refs with a repeated learned slot,
        # and rehash BOTH payload and table. Rejection is a physical-boundary
        # check rather than a stale checksum check.
        from gf_common.candidate_columns import column_layout
        descriptor = decision['table']
        count = descriptor['count']
        length = 2 if mutation == 'duplicate_physical' else 19
        if mutation == 'too_many':
            rows[0]['caps']['max_positions'] = 128
        descriptor['elements'] = count * length
        descriptor['columns'], descriptor['bytes'] = column_layout(count, descriptor['elements'])
        core = owner(path)
        try:
            table = json.loads(core.diagnostic())['decisions'][0]['table']
        finally:
            core.close()
        # Each row becomes play_sequence with an otherwise valid learned ref.
        table.update(kinds=['play_sequence'] * count, targets=['omitted'] * count,
                     numerics=[[0] * 4 for _ in range(count)], presence=[[False] * 4 for _ in range(count)],
                     offsets=[i * length for i in range(count + 1)], slots=[0] * descriptor['elements'])
        table.pop('table_hash')
        descriptor['table_hash'] = digest(table)
        payload = bytearray(descriptor['bytes'])
        fields = {column['name']: column for column in descriptor['columns']}
        payload[fields['kinds']['offset']:fields['kinds']['offset'] + count] = bytes([1] * count)
        for i, value in enumerate(table['offsets']):
            struct.pack_into('<q', payload, fields['offsets']['offset'] + 8 * i, value)
        (tmp_path / descriptor['file']).write_bytes(payload)
        descriptor['sha256'] = hashlib.sha256(payload).hexdigest()
    path.write_text(''.join(canonical(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        owner(path)


def test_fixture_exact_token_and_column_values_remain_python_reference():
    from pathlib import Path

    from gf_common.candidate_columns import decode_candidate_columns
    from gf_common.structured import EventBank, EventDelta
    from gf_common.structured_wire import parse_frame
    fixture = Path(__file__).parents[1] / 'native/tests/fixtures/structured_tokens_v2.json'
    for case in json.loads(fixture.read_text()):
        frame = parse_frame(case['frame'])
        bank = EventBank(frame.actor, frame.episode)
        bank.append(EventDelta(frame.actor, frame.episode, 0, bank.boundary.prefix_hash, case['events']))
        table = decode_candidate_columns(bytes.fromhex(case['bytes']), case['columns'], expected_frame_hash=frame.frame_hash,
            action_contract=engineering_contract(), purpose='component', synthetic_capacity=case['columns']['synthetic_capacity'])
        table.validate(frame)
        assert json.loads(canonical(asdict(table))) == case['table']
        assert json.loads(canonical(asdict(tokenize_frame(frame, bank.view(), aggregate_window=case['window'])))) == case['frame_tokens']
        assert json.loads(canonical(asdict(tokenize_table(table, frame, aggregate_window=case['window'])))) == case['candidate_tokens']


def test_eighteen_unique_physical_instances_are_preserved_in_game_proposal(make_input, tmp_path):
    import uuid

    from gf_common.diverse_actions import GeneratedCandidates
    from gf_common.domain import digest
    from gf_common.structured import CandidateTable, EventBank
    from gf_models.structured_runtime import own_worker_settings
    from gf_training.structured_journal import StructuredJournal
    from tests.test_structured_protocol import action_settings, own_frame
    # A typed component script, not an engine legality/completion assertion.
    template_dir = tmp_path / 'template'
    template_dir.mkdir()
    template, _, _, generated, _, _ = journal_setup(make_input, template_dir)
    template.finish('incomplete', None)
    actions = replace(action_settings(), max_sequence_length=18, max_candidates=64)
    caps, costs, _, _ = own_worker_settings('random', actions, None)
    bank = EventBank(1, 'eighteen-distinct')
    frame, delta = own_frame(make_input(hand=('abyss-dart',) * 18), bank, 0)
    table = CandidateTable.from_actions(frame, [{'type': 'play_sequence', 'ordered_elements': list(frame.references), 'target': 2}],
        action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    report = replace(generated.report, config_hash=digest(asdict(actions)), table_hash=table.table_hash,
        discovered=1, mandatory=1, selected=1, selected_elements=18, channel_counts=(0,) * 7,
        traversal_complete=False, presentation_complete=False)
    generated = GeneratedCandidates(table, report)
    writer = StructuredJournal(tmp_path, 'eighteen', 1, 'random-structured-v1', run_id='component', game_id='typed-script',
        action_config_hash=digest(asdict(actions)), bank_episode=bank.episode, caps=asdict(caps), numeric_costs=asdict(costs),
        model_id='random', model_version='1', sampling_seed=17, action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    from gf_common.structured import Selection
    selection = Selection(table.table_hash, 0, 'random-structured-v1')
    command = {'command_id': str(uuid.uuid4()), 'expected_revision': 0, 'action': selection.validate(table, frame)}
    writer.begin_frame(frame, delta, generated, selection, command, {})
    writer.finish('incomplete', None)
    read_structured_journal(writer.final_path, action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    core = owner(writer.final_path)
    try:
        result = json.loads(core.diagnostic())
        actual = result['decisions'][0]
        assert actual['table']['slots'] == list(range(18))
        assert actual['candidate_tokens']['slots'] == list(range(18))
        assert actual['frame_tokens']['card_mask'] == [True] * 18
        assert len({actual['frame_tokens']['card_categories'][i][0] for i in range(18)}) == 1
    finally:
        core.close()


@pytest.mark.parametrize('change', [{'memory_bytes': True}, {'max_json_bytes': 0}, {'max_record_bytes': True},
    {'unknown': 1}, {'max_record_bytes': 65 * 1024**2}])
def test_closed_consumed_diagnostic_settings_reject(make_input, tmp_path, change):
    path = journal(make_input, tmp_path)
    with pytest.raises(ValueError):
        owner(path, **change)


@pytest.mark.parametrize('limit', ['file', 'record'])
def test_encoded_bounds_are_enforced_before_projection(make_input, tmp_path, limit):
    path = journal(make_input, tmp_path)
    if limit == 'file':
        settings = {'max_json_bytes': path.stat().st_size - 1, 'max_record_bytes': 1}
    else:
        settings = {'max_record_bytes': 1}
    with pytest.raises((ValueError, MemoryError)):
        owner(path, **settings)


def test_one_delivery_control_lease_and_cancelled_close_retire_ownership(make_input, tmp_path):
    path = journal(make_input, tmp_path)
    core = owner(path)
    core.diagnostic()
    assert core.control_reserved_bytes() > 0
    with pytest.raises(ValueError, match='already delivered'):
        core.diagnostic()
    with pytest.raises(MemoryError):
        core.recheck(512 * 1024**2)
    # Failed additional caller admission preserves the existing ownership.
    assert core.used_bytes() > 0 and core.control_reserved_bytes() > 0
    core.cancel_handle().cancel()
    with pytest.raises(RuntimeError, match='cancel'):
        core.close()
    assert core.used_bytes() == 0
    core.close()


@pytest.mark.parametrize('kind', ['fifo', 'symlink'])
def test_nonregular_raw_sidecar_never_follows_or_blocks(make_input, tmp_path, kind):
    path = journal(make_input, tmp_path)
    sidecar = next(tmp_path.glob('*.columns'))
    original = sidecar.read_bytes()
    sidecar.unlink()
    if kind == 'fifo':
        os.mkfifo(sidecar)
    else:
        outside = tmp_path / 'outside-evidence'
        outside.write_bytes(original)
        sidecar.symlink_to(outside)
    started = time.monotonic()
    with pytest.raises((OSError, ValueError)):
        owner(path)
    assert time.monotonic() - started < 2
    if kind == 'symlink':
        assert outside.read_bytes() == original


@pytest.mark.parametrize('status', ['dark_cloud', 'darkcloud', 'unknown_public_status'])
def test_exact_event_categories_keep_distinct_spellings_and_reject_unknown(make_input, tmp_path, status):
    writer, frame, delta, generated, selection, command = journal_setup(make_input, tmp_path,
        events=[{'type': 'status_changed', 'player': 1, 'status': status, 'active': True}])
    writer.begin_frame(frame, delta, generated, selection, command, {})
    writer.finish('incomplete', None)
    data = read_structured_journal(writer.final_path, action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    if status == 'unknown_public_status':
        with pytest.raises(ContractError, match='unknown structured category'):
            tokenize_frame(frame, data.bank)
        with pytest.raises(ValueError, match='unknown exact structured category'):
            owner(writer.final_path)
    else:
        core = owner(writer.final_path)
        try:
            encoded = json.loads(core.diagnostic())['decisions'][0]['frame_tokens']
            expected = tokenize_frame(frame, data.bank)
            assert encoded == json.loads(canonical(asdict(expected)))
            from gf_common.structured_tokens import STATUSES
            assert encoded['interval_categories'][0][2] == STATUSES.index(status) + 2
        finally:
            core.close()


def test_optional_free_diagnostic_never_queries_python_models_runtime_or_catalog(make_input, tmp_path):
    import subprocess
    import sys
    path = journal(make_input, tmp_path)
    code = '''
import importlib.abc,importlib.metadata,sys,time
class Forbidden(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='torch' or name.startswith(('torch.','gf_models','gf_common','gf_training')):raise AssertionError(name)
sys.meta_path.insert(0,Forbidden())
def forbidden(*args,**kwargs):raise AssertionError('current runtime query')
importlib.metadata.version=forbidden
import gf_native,json
core=gf_native.StructuredJournalTokens(sys.argv[1],sys.argv[2],sys.argv[3].encode(),time.monotonic()+30)
try:
 result=json.loads(core.diagnostic())
 assert result['qualified'] is False and result['parent_qualification']=='not_performed'
finally:core.close()
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path), path.name, SETTINGS.decode()],
        capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def test_baseline_diagnostic_rejects_instead_of_ignoring_nondefault_recorded_window(make_input, tmp_path):
    from gf_common.structured_tokens import signature
    path = journal(make_input, tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(aggregate_window=1, token_signature=signature(1))
    path.write_text(''.join(canonical(row) + '\n' for row in rows))
    # The generic structural reader supports the recorded window. The bounded
    # baseline diagnostic explicitly implements corrected128, without relabeling.
    read_structured_journal(path, action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    with pytest.raises(ValueError, match='corrected128'):
        owner(path)


@pytest.mark.parametrize('seed', [0, 1, 17, 2**32, 2**63 - 1, 2**64 - 1])
def test_original_search_report_initial_state_matches_cpython_for_header_seed(make_input, tmp_path, seed):
    import random

    from gf_common.domain import digest
    writer, frame, delta, generated, selection, command = journal_setup(make_input, tmp_path, seed=seed)
    expected_seed = int(digest({'recipe': 'structured-search-stream-v1', 'sampling_seed': seed})[:16], 16)
    assert generated.report.rng_before == random.Random(expected_seed).getstate()
    writer.begin_frame(frame, delta, generated, selection, command, {})
    writer.finish('incomplete', None)
    core = owner(writer.final_path)
    try:
        report = json.loads(core.diagnostic())['decisions'][0]['search_report']
        assert report['rng_before'] == json.loads(canonical(random.Random(expected_seed).getstate()))
    finally:
        core.close()


@pytest.mark.parametrize('mutation', ['version', 'word_count', 'negative_word', 'overflow_word', 'index', 'gaussian_integer'])
def test_search_rng_closed_state_and_word_ranges_reject(make_input, tmp_path, mutation):
    path = journal(make_input, tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    report = next(row for row in rows if row['kind'] == 'decision')['search_report']
    state = report['rng_after']
    if mutation == 'version':
        state[0] = 2
    elif mutation == 'word_count':
        state[1].pop()
    elif mutation == 'negative_word':
        state[1][0] = -1
    elif mutation == 'overflow_word':
        state[1][0] = 2**32
    elif mutation == 'index':
        state[1][-1] = 625
    else:
        state[2] = 0
    path.write_text(''.join(canonical(row) + '\n' for row in rows))
    with pytest.raises(ContractError):
        read_structured_journal(path, action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    with pytest.raises(ValueError):
        owner(path)


@pytest.mark.parametrize('seed', [True, -1, 2**64, 1.0])
def test_header_sampling_seed_requires_builtin_uint64(make_input, tmp_path, seed):
    path = journal(make_input, tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['sampling_seed'] = seed
    path.write_text(''.join(canonical(row) + '\n' for row in rows))
    with pytest.raises(ContractError):
        read_structured_journal(path, action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    with pytest.raises(ValueError, match='integer'):
        owner(path)


def test_direct_mt_goldens_are_exact_pinned_python_initial_states():
    import random
    from pathlib import Path
    path = Path(__file__).parents[1] / 'native/tests/fixtures/cpython314_mt_initial.json'
    cases = json.loads(path.read_text())
    assert [case['seed'] for case in cases] == [0, 1, 17, 2**32, 2**63 - 1, 2**64 - 1]
    for case in cases:
        assert case['state'] == json.loads(canonical(random.Random(case['seed']).getstate()))
