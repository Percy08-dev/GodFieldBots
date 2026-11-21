"""Bounded encoding/Pack fixtures; none issue a live collected Source."""
from __future__ import annotations

import os
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError, digest
from gf_common.structured import CandidateTable, Selection
from gf_common.structured_tokens import tokenize_table
from gf_training.structured_rollout import (
    StructuredRollout,
    _write_rollout,
    game_episode_records,
    game_rollout,
)
from tests.model3_support import LAYOUT, model3, raw_inputs
from tests.test_card_instance_learning import ACTIONS, RULES


def records():
    _, _, frames, bank, _ = raw_inputs(count=2)
    tables = tuple(CandidateTable.from_actions(frame,
        [{'type': 'discard', 'card_refs': [frame.references[index]]} for index in (0, 1)],
        action_contract=ACTIONS.action_contract, purpose='qualified_game', synthetic_capacity=False)
        for frame in frames)
    selections = tuple(Selection(table.table_hash, 0, 'packing-fixture', -.25, .125) for table in tables)
    return (game_episode_records(bank.view(), tuple(frames), tables, selections,
        'natural', 'draw', actions=ACTIONS, rules=RULES),)


def source_data():
    """Exercise the private ordinary role with explicitly stubbed external guards."""
    from gf_training.rust_player_source import RustPlayerSource
    source = RustPlayerSource.__new__(RustPlayerSource)
    events = []
    source._records, source.state = records(), 'SEALED'
    source.plan = SimpleNamespace(execution_deadline=time.monotonic()+30,
        linkage={'collection_generation': 0})
    source.parent = SimpleNamespace(generation=0, require=lambda: events.append('parent'))
    learner = SimpleNamespace(model=model3(), _context=SimpleNamespace(rules=RULES))
    source.registry = SimpleNamespace(hpo=False, failed=False, closed=False,
        state={'updates': 0}, pending=source, current=source.parent,
        local=SimpleNamespace(learner=learner), config=SimpleNamespace(actions=ACTIONS))
    source.collection = SimpleNamespace(pack_verification=None, require=lambda *args: events.append('collection'))
    return source, events


def owned_rollout(source):
    from gf_training.rust_player_source import _OrdinarySourceContext
    source.require()  # Same established full guard as persist, before encoding.
    context = _OrdinarySourceContext(source)
    return StructuredRollout(source._records, source.registry.local.learner.model,
        digest(asdict(ACTIONS)), action_contract=ACTIONS.action_contract,
        purpose='qualified_game', synthetic_capacity=False, _context=context)


def test_owned_tokens_reuse_only_checked_records_and_numeric_bytes_match(tmp_path, monkeypatch):
    source, _ = source_data()
    strict = game_rollout(source._records, source.registry.local.learner.model, actions=ACTIONS, rules=RULES)
    validate = CandidateTable.validate
    calls = []
    def capture(table, frame):
        calls.append((table, frame))
        return validate(table, frame)
    monkeypatch.setattr(CandidateTable, 'validate', capture)
    owned = owned_rollout(source)
    assert calls == []
    assert owned.episodes[0].frames == strict.episodes[0].frames
    assert owned.episodes[0].tables == strict.episodes[0].tables
    assert owned.episodes[0].records is source._records[0]
    _write_rollout(tmp_path/'strict', strict)
    _write_rollout(tmp_path/'owned', owned, check=lambda: None, _before_manifest=source.require)
    files = lambda folder: {path.name: path.read_bytes() for path in folder.iterdir()}
    assert files(tmp_path/'strict') == files(tmp_path/'owned')
    game_rollout(source._records, source.registry.local.learner.model, actions=ACTIONS, rules=RULES)
    assert len(calls) == 2  # Stock game/HPO/readback entry retains strict conversion.


@pytest.mark.parametrize('failure', ['hpo', 'pending', 'current', 'failed', 'closed', 'generation', 'deadline'])
def test_owned_context_refuses_wrong_original_source_role(failure):
    from gf_training.rust_player_source import _OrdinarySourceContext
    source, _ = source_data()
    if failure in ('hpo', 'failed', 'closed'):
        setattr(source.registry, failure, True)
    elif failure in ('pending', 'current'):
        setattr(source.registry, failure, object())
    elif failure == 'generation':
        source.registry.state['updates'] = 1
    else:
        source.plan.execution_deadline = time.monotonic()-1
    with pytest.raises((ContractError, TimeoutError)):
        _OrdinarySourceContext(source)


def test_unselected_own_view_error_stays_strict_and_private_context_rejects_foreign_record():
    from gf_training.rust_player_source import _OrdinarySourceContext
    source, _ = source_data()
    record = source._records[0]
    frame, table = record.frames[0], record.tables[0]
    slots = (table.slots[0], 17)
    malformed = replace(table, slots=slots, table_hash=digest(table._content() | {'slots': slots}))
    with pytest.raises(ContractError, match='own view'):
        tokenize_table(malformed, frame, layout=LAYOUT)
    selections = (replace(record.selections[0], table_hash=malformed.table_hash), *record.selections[1:])
    with pytest.raises(ContractError, match='own view'):
        game_episode_records(record.bank, record.frames, (malformed, *record.tables[1:]), selections,
            'natural', 'draw', actions=ACTIONS, rules=RULES)
    context = _OrdinarySourceContext(source)
    foreign = records()[0]
    with pytest.raises(ContractError, match='record'):
        context.tokenize_tables(0, foreign, source.registry.local.learner.model)
    with pytest.raises(ContractError):
        context.tokenize_tables(0, record, replace(source.registry.local.learner.model, max_positions=0))


@pytest.mark.parametrize('failure', ['guard', 'deadline', 'capacity'])
def test_final_guard_and_capacity_refuse_before_completed_manifest(tmp_path, failure):
    source, events = source_data()
    rollout = owned_rollout(source)
    target = tmp_path/'numeric'
    callbacks = []
    expired = False
    def deadline():
        callbacks.append('deadline')
        if expired:
            raise TimeoutError('original deadline')
    def guard():
        nonlocal expired
        callbacks.append('full')
        assert list(target.glob('rollout*.npz'))  # Shards precede completion guard.
        assert not (target/'rollout.json').exists()
        if failure == 'guard':
            raise OSError('original changed')
        expired = True  # Original checking consumed the remaining wall allowance.
    with pytest.raises((OSError, TimeoutError, ContractError)):
        _write_rollout(target, rollout, check=deadline, _before_manifest=guard,
            max_owned_bytes=1 if failure == 'capacity' else None)
    assert not (target/'rollout.json').exists()
    assert events == ['collection', 'parent']
    if failure == 'capacity':
        assert 'full' not in callbacks
    else:
        assert callbacks.count('full') == 1
        assert callbacks[callbacks.index('full')+1:] == ([] if failure == 'guard' else ['deadline'])


def test_source_persist_uses_deadline_loops_full_completion_guard_and_numeric_originals(tmp_path, monkeypatch):
    from gf_common.group_pack import StorageSettings
    from gf_training.group_pack_resources import StorageBudget
    source, events = source_data()
    source.registry.budget = StorageBudget(StorageSettings(), source.plan.execution_deadline)
    source._charges, source._numeric_originals = [], {}
    source._rollout = None
    target = tmp_path/'numeric'
    boundaries = []
    def full(*args):
        boundaries.append('entry' if source._rollout is None else 'completion')
        assert not (target/'rollout.json').exists()
        if source._rollout is not None:
            assert list(target.glob('rollout*.npz'))
    source.collection.require = full
    ticks = []
    actual_deadline = source._packing_deadline
    def deadline():
        ticks.append('deadline')
        actual_deadline()
    monkeypatch.setattr(source, '_packing_deadline', deadline)
    monkeypatch.setattr(source, '_digests', lambda: ('fixture-digest',))
    monkeypatch.setattr(source, '_build_proof', lambda: 'fixture-only-proof')
    try:
        metadata = source.persist(target)
        assert boundaries == ['entry', 'completion']
        assert events == ['parent', 'parent']
        assert len(ticks) > 50 and source.state == 'PERSISTED'
        assert set(source._numeric_originals) == {'rollout.json', *metadata['data']['files']}
        for name, original in source._numeric_originals.items():
            original.recheck()
            assert original.sha256 == source.numeric_files[name]
        # The full guard normally allows PERSISTED; this fixture guard was
        # intentionally about publication ordering, so restore its default.
        source.collection.require = lambda *args: None
        with pytest.raises(ContractError, match='single use'):
            source.persist(tmp_path/'again')
    finally:
        for original in source._numeric_originals.values():
            original.close()
        for charge in source._charges:
            charge.close()
    assert source.registry.budget.used == 0


def packed_originals(tmp_path, *, first_members=None):
    """Real sealed local Pack+retained FDs; no actor, Source or host admission."""
    from gf_common.group_pack import (
        PackMember,
        StorageSettings,
        canonical_json,
        encode_pack,
        unpack_raw,
    )
    from gf_common.preprocessing import PreparationConfig
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.group_pack_source import OriginalFile
    from gf_training.group_pack_worker import GroupPackDatabase
    from gf_training.group_pack_writer import PendingDescriptorOwner
    from gf_training.rust_player_source import RustEmptyCollection, RustPlayerOriginals
    from gf_training.rust_player_storage import RustPendingDescriptor
    deadline = time.monotonic()+30
    settings = StorageSettings(db_root=tmp_path/'db', ram_root=tmp_path/'ram')
    directory = tmp_path/'collection'
    directory.mkdir()
    worker = GroupPackDatabase(max_owned_bytes=512*1024**2)
    ordinal = 0
    def command(kind, payload=None, **fields):
        nonlocal ordinal
        result = worker.execute({'version': 1, 'request_id': ordinal, 'command': kind, **fields}, payload)
        ordinal += 1
        return result
    configured = asdict(settings) | {'db_root': str(settings.db_root), 'ram_root': str(settings.ram_root)}
    command('init', directory=str(directory), run_id='pack-fixture', settings=configured, deadline=deadline)
    for group in ('first', 'second'):
        members = (first_members if group == 'first' and first_members is not None else
            (PackMember('journal', group+'/player-1.jsonl', b'original'),))
        payload = encode_pack(members)
        command('submit', payload, group_id=group, raw_bytes=len(unpack_raw(payload)), record={},
            episodes=[{'episode_id': group, 'game_id': None, 'record': {}}])
    command('control', b'{}', name='manifest.json')
    descriptor = command('seal', publication='pending')
    raw = canonical_json(descriptor)
    path = directory/'collection.pending.json'
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, raw)
    stat = os.fstat(fd)
    retained = PendingDescriptorOwner(directory, path, (stat.st_dev, stat.st_ino), raw,
        descriptor, fd, OriginalFile(directory/'collection.sqlite3', deadline))
    pool = PackedArtifactPool(max_owned_bytes=512*1024**2, deadline=deadline)
    pending = RustPendingDescriptor(retained, pool.budget.reserve(len(raw)))
    owner = RustPlayerOriginals(pending, pool=pool, settings=PreparationConfig(), deadline=deadline)
    inspection = RustEmptyCollection.__new__(RustEmptyCollection)
    inspection.state, inspection.originals = 'INSPECTING', owner
    inspection.collection = SimpleNamespace(originals=owner, pending=pending)
    inspection.registry = SimpleNamespace(hpo=False, pending=inspection)
    return owner, inspection, pending, pool


def test_live_constructor_reader_decodes_valid_logical_journal(tmp_path):
    from gf_common.group_pack import PackMember
    from gf_training.card_instance_journal import read_card_instance_journal
    from tests.test_card_instance_baseline_runtime import ACTIONS as JOURNAL_ACTIONS
    from tests.test_card_instance_baseline_runtime import writer

    folder = tmp_path/'journal-input'
    folder.mkdir()
    journal, frame, delta, generated, selected, command = writer(folder)
    journal.begin_frame(frame, delta, generated, selected, command, {})
    journal.finish('cancelled', None)
    members = (PackMember('journal', 'first/player-1.jsonl', journal.final_path.read_bytes()),
        *(PackMember('candidates', 'first/'+path.name, path.read_bytes()) for path in folder.glob('*.columns')))
    owner, inspection, pending, pool = packed_originals(tmp_path, first_members=members)
    try:
        with owner._ordinary_inspection(inspection):
            parsed = read_card_instance_journal(owner.root/'first/player-1.jsonl', actions=JOURNAL_ACTIONS,
                artifacts=owner.reader, artifact_prefix='first', deadline=owner.deadline)
            assert parsed.decisions[0].generated.table == generated.table
            assert parsed.decisions[0].selection == selected
            assert parsed.decisions[0].command == command
            assert parsed.sidecars == () and len(parsed.logical_sidecars) == 1
            owner.finalize()
    finally:
        owner.close()
        pending.close()
        pool.close()
    assert pool.budget.used == 0


@pytest.mark.parametrize('failure', [None, 'body', 'finalize'])
def test_pack_interval_reuses_only_cached_bytes_and_finalizes_fully(tmp_path, monkeypatch, failure):
    owner, inspection, pending, pool = packed_originals(tmp_path)
    checks = []
    original_check = pending.recheck
    def check():
        checks.append('physical')
        return original_check()
    monkeypatch.setattr(pending, 'recheck', check)
    physical = []
    originals_check = owner.source._originals.recheck
    def pack_check():
        physical.append('pack')
        return originals_check()
    monkeypatch.setattr(owner.source._originals, 'recheck', pack_check)
    try:
        try:
            with owner._ordinary_inspection(inspection):
                assert len(checks) == 1
                assert owner.read(owner.root/'first/player-1.jsonl') == b'original'
                before, pack_before = len(checks), len(physical)
                assert owner.read(owner.root/'first/player-1.jsonl') == b'original'
                assert owner.inventory('**/*') == tuple(sorted(owner._entries))
                assert len(checks) == before and len(physical) == pack_before
                assert owner.read(owner.root/'second/player-1.jsonl') == b'original'
                assert len(physical) > pack_before  # New groups still authenticate originals.
                pack_before = len(physical)
                assert owner.read(owner.root/'manifest.json') == b'{}'
                assert len(physical) > pack_before  # Controls still authenticate bytes/originals.
                if failure == 'body':
                    raise LookupError('projection failure')
                if failure == 'finalize':
                    temporary = pending.path.with_suffix('.replacement')
                    temporary.write_bytes(pending.path.read_bytes())
                    temporary.replace(pending.path)
                owner.finalize()  # Must remain full while the interval is active.
            assert failure is None
        except (LookupError, ValueError):
            assert failure in ('body', 'finalize')
        if failure == 'finalize':
            with pytest.raises(ValueError):
                owner.check()
        else:
            before = len(checks)
            owner.check()
            assert len(checks) == before+1  # Interval cleared on success or error.
            pack_before = len(physical)
            assert owner.reader.names() == tuple(sorted(owner._entries))
            assert len(physical) > pack_before  # Original reader names stays strict.
        assert owner.source._cache_charge > 0 and pool.budget.used > 0
    finally:
        owner.close()
        pending.close()
        pool.close()
    assert pool.budget.used == 0


@pytest.mark.parametrize('path', ['second/player-1.jsonl', 'manifest.json'])
def test_pack_interval_new_group_and_control_still_reject_changed_original(tmp_path, path):
    owner, inspection, pending, pool = packed_originals(tmp_path)
    try:
        with owner._ordinary_inspection(inspection):
            assert owner.read(owner.root/'first/player-1.jsonl') == b'original'
            temporary = pending.path.with_suffix('.replacement')
            temporary.write_bytes(pending.path.read_bytes())
            temporary.replace(pending.path)
            with pytest.raises(ValueError):
                owner.read(owner.root/path)
        with pytest.raises(ValueError):
            owner.check()
    finally:
        owner.close()
        pending.close()
        pool.close()
    assert pool.budget.used == 0


@pytest.mark.parametrize('failure', ['hpo', 'foreign', 'nested', 'closed'])
def test_pack_interval_retains_ordinary_exact_owner_and_closed_gates(tmp_path, failure):
    owner, inspection, pending, pool = packed_originals(tmp_path)
    try:
        if failure == 'hpo':
            inspection.registry.hpo = True
        elif failure == 'foreign':
            inspection.collection.pending = object()
        if failure in ('hpo', 'foreign'):
            with pytest.raises(ContractError), owner._ordinary_inspection(inspection):
                pytest.fail('foreign/nonordinary owner entered optimized interval')
        else:
            with owner._ordinary_inspection(inspection):
                if failure == 'nested':
                    with pytest.raises(ContractError), owner._ordinary_inspection(inspection):
                        pytest.fail('nested interval entered')
                else:
                    owner._closed = True
                    try:
                        with pytest.raises(ContractError, match='closed'):
                            owner.read(owner.root/'first/player-1.jsonl')
                    finally:
                        owner._closed = False
                owner.finalize()
    finally:
        owner.close()
        pending.close()
        pool.close()
    assert pool.budget.used == 0
