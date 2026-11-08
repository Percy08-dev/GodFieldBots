"""Bounded original-source scanning and exact numeric staging (synthetic data)."""
import hashlib
from dataclasses import replace

import numpy as np
import pytest

from gf_common.domain import ContractError, canonical
from gf_common.preprocessing import PreparationConfig
from gf_models.config import EncoderConfig
from gf_models.encoder import encode
from gf_training.fixed_data import (
    build_dataset,
    inspect_dataset,
    inspect_dataset_summary,
    prepare_fixed_inputs,
)
from gf_training.preparation import PreparationSession, ResourceLimit


def test_streaming_summary_and_numeric_match_legacy_samples_and_npz(tmp_path, frozen_sources):
    parents, validations = frozen_sources
    path = tmp_path / 'dataset.json'
    manifest = build_dataset(parents, validations, path)
    assert inspect_dataset_summary(path) == manifest
    actual, train, validation = inspect_dataset(path)
    config = EncoderConfig(history='public_aggregate')
    expected = {'x': np.stack([encode(sample.value, sample.action, config) for sample in train]),
                'y': np.asarray([sample.target for sample in train], dtype=np.float64),
                'validation_x': np.stack([encode(sample.value, sample.action, config) for sample in validation]),
                'validation_y': np.asarray([sample.target for sample in validation], dtype=np.float64)}
    np.savez(tmp_path / 'legacy.npz', **expected)
    settings = PreparationConfig(memory_budget_mib=128, cache_budget_mib=0, chunk_target_mib=16)
    prepared = prepare_fixed_inputs(path, config, preprocessing=settings)
    scratch = prepared.session.scratch_directory
    with prepared:
        assert prepared.dataset == actual
        prepared.write_numeric(tmp_path / 'new.npz')
        assert (tmp_path / 'new.npz').read_bytes() == (tmp_path / 'legacy.npz').read_bytes()
        for name, array in expected.items():
            borrowed = getattr(prepared, name)
            assert borrowed.flags.writeable is False
            assert borrowed.tobytes() == array.tobytes()
            del borrowed
        identity = hashlib.sha256()
        for split, samples in (('train', train), ('validation', validation)):
            for sample in samples:
                identity.update((canonical([split, *sample.identity]) + '\n').encode())
        assert prepared.diagnostics['sample_identity_sha256'] == identity.hexdigest()
        assert prepared.diagnostics['parent_chain_validations'] == 3
        assert prepared.diagnostics['journal_validations'] == 480
        assert prepared.diagnostics['episode_count'] == 240
        assert prepared.diagnostics['managed_peak_bytes'] <= 128 * 1024 * 1024
    assert not scratch.exists()
    assert prepared.diagnostics['cleanup'] == 'complete'
    prepared.finalize()  # No fallible cleanup after successful explicit finalization.
    with pytest.raises(ContractError):
        _ = prepared.x


def test_session_rejects_oversize_file_record_and_episode_before_decode(tmp_path):
    settings = PreparationConfig(memory_budget_mib=64, cache_budget_mib=0, chunk_target_mib=16,
                                 max_source_file_mib=2, max_record_mib=1)
    source = tmp_path / 'source.jsonl'
    source.write_bytes(b' ' * (1024 * 1024 + 1))
    with PreparationSession(settings) as session, pytest.raises(ResourceLimit, match='memory'), session.episode((source,)):
        pytest.fail('episode decoding must not begin over budget')
    with PreparationSession(replace(settings, memory_budget_mib=256)) as session, pytest.raises(ResourceLimit, match='max_record'), session.episode((source,)):
        pytest.fail('record decoding must not begin above record bound')
    source.write_bytes(b' ' * (2 * 1024 * 1024 + 1))
    with PreparationSession(settings) as session, pytest.raises(ResourceLimit, match='max_source'):
        session.read_json(source)


def test_source_tamper_and_cleanup_failure_release_locks_and_scratch(tmp_path, monkeypatch):
    from gf_training import preparation
    source = tmp_path / 'source.json'
    source.write_text('{}')
    owner = PreparationSession()
    owner.read_json(source)
    scratch = owner.scratch()
    source.write_text('{"changed":true}')
    with pytest.raises(ContractError, match='invalidated'):
        owner.finalize()
    assert not scratch.exists() and owner._closed
    owner = PreparationSession()
    owner.scratch()
    def fail(path):
        raise OSError('controlled scratch failure')
    monkeypatch.setattr(preparation.shutil, 'rmtree', fail)
    closed = []
    owner.locks.callback(lambda: closed.append(True))
    with pytest.raises(OSError, match='scratch failure'):
        owner.finalize()
    assert closed == [True] and owner._closed
    assert owner.diagnostics['cleanup'] == 'failed'


def test_index_spills_and_merges_preserve_sorted_rows_with_small_chunks(tmp_path):
    from gf_training.preparation_index import IndexLedger
    settings = PreparationConfig(memory_budget_mib=64, cache_budget_mib=0, chunk_target_mib=1)
    with PreparationSession(settings) as session:
        ledger = IndexLedger(session)
        for index in reversed(range(4000)):
            row = {'run_id': 'run', 'game_id': f'game-{index:06}', 'learner_viewer': index % 2 + 1,
                   'decision': index, 'return': float(index % 3 - 1)}
            ledger.add(('run', row['game_id'], row['learner_viewer'], index), row)
        actual = [__import__('json').loads(encoded)['decision'] for encoded in ledger.sorted_rows()]
        assert actual == list(range(4000))
        assert session.diagnostics['index_spill_runs'] > 16
        assert session.diagnostics['managed_peak_bytes'] <= 64 * 1024 * 1024


def test_index_rejects_existing_output_or_sidecar_before_source_work(tmp_path, monkeypatch):
    import gf_training.fixed_data as data
    monkeypatch.setattr(data, '_validated_dataset', lambda *args: pytest.fail('must reject before source work'))
    output = tmp_path / 'index.parquet'
    sidecar = output.with_suffix('.parquet.json')
    output.write_bytes(b'original index')
    with pytest.raises(ContractError, match='fresh'):
        data.index_dataset(tmp_path / 'missing.json', output)
    assert output.read_bytes() == b'original index'
    output.unlink()
    sidecar.write_bytes(b'original sidecar')
    with pytest.raises(ContractError, match='fresh'):
        data.index_dataset(tmp_path / 'missing.json', output)
    assert sidecar.read_bytes() == b'original sidecar'


@pytest.mark.parametrize('failure', ['second_link', 'post_deadline', 'concurrent_target'])
def test_publication_failure_removes_only_owned_paths(tmp_path, monkeypatch, failure):
    import gf_training.fixed_data as data
    owner = PreparationSession()
    pairs = tuple((tmp_path / f'staged-{index}', tmp_path / f'published-{index}') for index in (1, 2))
    for staged, _ in pairs:
        staged.write_bytes(b'new')
    if failure == 'second_link':
        real = data.os.link
        def link(source, target):
            if target == pairs[1][1]:
                raise OSError('second link controlled failure')
            return real(source, target)
        monkeypatch.setattr(data.os, 'link', link)
    elif failure == 'post_deadline':
        calls = []
        def check(deadline):
            calls.append(True)
            if len(calls) == 2:
                raise TimeoutError('controlled post-publication deadline')
        monkeypatch.setattr(data, 'check_deadline', check)
    else:
        pairs[1][1].write_bytes(b'concurrent original')
    with pytest.raises((OSError, TimeoutError)):
        data._publish_owned(owner, pairs)
    assert not pairs[0][1].exists()
    if failure == 'concurrent_target':
        assert pairs[1][1].read_bytes() == b'concurrent original'
    else:
        assert not pairs[1][1].exists()
    assert owner._closed


@pytest.mark.parametrize('failure', [None, 'tamper', 'cleanup'])
def test_native_runner_finalizes_preparation_before_any_ready_publication(tmp_path, make_input, monkeypatch, failure):
    import asyncio
    import json
    import time
    from dataclasses import asdict

    pytest.importorskip('lightgbm')
    import gf_training.lightgbm_run as runner
    from gf_common.domain import ActionConfig
    from gf_models.config import ModelConfig, TrainingConfig
    from gf_training.journal import AcceptedDecision
    from gf_training.preparation_numeric import PreparedInputs
    source = tmp_path / 'source.json'
    source.write_text('{"healthy":true}')
    model = ModelConfig(model_id='lightgbm', encoder=EncoderConfig(hash_buckets=2))
    training = TrainingConfig(num_boost_round=2, early_stopping_rounds=1, min_data_in_leaf=1,
                              wall_seconds=15, cleanup_seconds=1, cost_mode='fixed_data')
    settings = PreparationConfig(memory_budget_mib=128, cache_budget_mib=0, chunk_target_mib=1,
                                 wall_seconds=15, cleanup_seconds=1)
    wall = time.monotonic() + 15
    session = PreparationSession(settings, deadline=wall - 1, wall_deadline=wall)
    session.read_json(source)
    dataset = {'train_decisions': 4, 'validation_decisions': 2}
    owner = PreparedInputs(session, dataset, model.encoder)
    for split, count in (('train', 4), ('validation', 2)):
        for index in range(count):
            value = make_input()
            owner.accept(AcceptedDecision(value, {'type': 'pray'}, float(index % 2), ('run', 'game', 1, index), split), None)
    owner.seal(dataset)
    directory = tmp_path / 'fit'
    stopped = []
    real_stop = runner._stop
    async def stop(*args, **kwargs):
        await real_stop(*args, **kwargs)
        stopped.append(True)
        if failure == 'tamper':
            source.write_text('{"healthy":false}')
        elif failure == 'cleanup':
            monkeypatch.setattr('gf_training.preparation.shutil.rmtree', lambda path: (_ for _ in ()).throw(OSError('cleanup controlled failure')))
    monkeypatch.setattr(runner, '_stop', stop)
    real_finalize = owner.finalize
    def finalize():
        assert stopped == [True]
        assert not (directory / 'checkpoint').exists()
        assert json.loads((directory / 'staging' / 'inference.json').read_text())['ready'] is False
        real_finalize()
    monkeypatch.setattr(owner, 'finalize', finalize)
    operation = runner._run_numeric(directory, None, None, None, None, model=model, actions=ActionConfig(),
                                    training=training, seed=17, prepared_inputs=owner)
    if failure is None:
        result = asyncio.run(operation)
        assert result['status'] == 'complete'
        assert json.loads((directory / 'checkpoint' / 'inference.json').read_text())['ready'] is True
    else:
        with pytest.raises((ContractError, OSError)):
            asyncio.run(operation)
        result = json.loads((directory / 'training.json').read_text())
        assert result['status'] == 'failed'
        for path in directory.glob('*/inference.json'):
            assert json.loads(path.read_text())['ready'] is False
    assert owner._finalized
    assert session._closed
    assert asdict(model)['encoder'] == asdict(owner.config)


def test_dataset_cli_uses_standalone_preprocessing_without_experiment(tmp_path, monkeypatch, capsys):
    import gf_cli
    import gf_training.fixed_data as data
    preset = tmp_path / 'preprocessing.toml'
    preset.write_text('schema_version = 1\nkind = "preprocessing"\n[preprocessing]\nmemory_budget_mib = 128\ncache_budget_mib = 0\nchunk_target_mib = 16\n')
    calls = []
    def inspect(path, *, deadline, preprocessing):
        import time
        assert preprocessing.memory_budget_mib == 128 and preprocessing.cache_budget_mib == 0
        assert 0 < deadline - time.monotonic() <= preprocessing.wall_seconds
        calls.append(path)
        return {'content_sha256': 'frozen'}
    monkeypatch.setattr(data, 'inspect_dataset_summary', inspect)
    monkeypatch.setattr(gf_cli, 'load_config', lambda *args: pytest.fail('dataset command must not load an experiment'))
    path = tmp_path / 'dataset.json'
    assert gf_cli.main(['dataset', 'inspect', str(path), '--preprocessing-config', str(preset)]) == 0
    assert calls == [path]
    assert '"content_sha256":"frozen"' in capsys.readouterr().out


@pytest.mark.parametrize('encoding', ['utf-8-sig', 'utf-16', 'utf-32'])
def test_original_text_controls_and_journals_reject_alternate_encoding_but_replays_keep_bytes(tmp_path, encoding):
    import json
    payload = '{"value":"公開・日本語"}'
    control = tmp_path / 'run.json'
    journal = tmp_path / 'player-1.jsonl'
    replay = tmp_path / 'replay-player-1.json'
    for path in (control, journal, replay):
        path.write_bytes(payload.encode(encoding))
    with PreparationSession() as session:
        with pytest.raises((UnicodeError, json.JSONDecodeError)):
            session.read_json(control)
        with pytest.raises((UnicodeError, json.JSONDecodeError)), session.episode((journal,)):
            pytest.fail('legacy UTF8 text journal must reject alternate encoding')
        with session.episode((replay,)) as values:
            assert values[replay.resolve()] == json.loads(payload)


def test_jsonl_unicode_and_universal_newlines_match_legacy_text_reader(tmp_path):
    import json
    path = tmp_path / 'player-1.jsonl'
    path.write_bytes('{"name":"first\u2028second"}\r\n{"name":"third"}\r'.encode())
    with path.open(encoding='utf-8') as stream:
        expected = [json.loads(line) for line in stream]
    with PreparationSession() as session, session.episode((path,)) as payload:
        assert payload[path.resolve()] == expected


def test_explicit_fit_deadline_is_never_shortened_by_dataset_command_wall():
    import time
    settings = PreparationConfig(wall_seconds=10, cleanup_seconds=2)
    wall = time.monotonic() + 7200
    with PreparationSession(settings, deadline=wall - 5, wall_deadline=wall) as session:
        assert session.wall_deadline == wall
        assert session.deadline == wall - 5


@pytest.mark.parametrize('scope', ['control', 'episode'])
def test_memory_admission_snapshot_rejects_growth_before_any_decode(tmp_path, monkeypatch, scope):
    import gf_training.preparation as module
    settings = PreparationConfig(memory_budget_mib=40, cache_budget_mib=0, chunk_target_mib=1)
    paths = tuple(tmp_path / name for name in ('player-1.jsonl', 'player-2.jsonl', 'replay-player-1.json', 'replay-player-2.json'))
    for path in paths:
        path.write_bytes(b'{}')
    grown = []
    with PreparationSession(settings) as session:
        charge = session.charge
        def grow_after_admission(amount):
            charge(amount)
            if not grown:
                grown.append(True)
                paths[0].write_bytes(b' ' * (1024 * 1024 - 2) + b'{}')
        monkeypatch.setattr(session, 'charge', grow_after_admission)
        monkeypatch.setattr(module.json, 'loads', lambda *args, **kwargs: pytest.fail('changed source reached JSON decoding'))
        with pytest.raises(ContractError, match='after memory admission'):
            if scope == 'control':
                session.read_json(paths[0])
            else:
                with session.episode(paths):
                    pytest.fail('changed episode passed admission')
        assert session._managed == 0
        assert session.diagnostics['source_reads'] == 0
        assert session.diagnostics['managed_peak_bytes'] < 40 * 1024 * 1024


@pytest.mark.parametrize('change', ['replace', 'same_size'])
def test_admitted_control_identity_rejects_replacement_or_same_size_change(tmp_path, monkeypatch, change):
    path = tmp_path / 'control.json'
    path.write_bytes(b'{}')
    with PreparationSession() as session:
        charge = session.charge
        changed = []
        def mutate(amount):
            charge(amount)
            if changed:
                return
            changed.append(True)
            if change == 'replace':
                replacement = tmp_path / 'replacement.json'
                replacement.write_bytes(b'{}')
                replacement.replace(path)
            else:
                import os
                before = path.stat()
                path.write_bytes(b'[]')
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
        monkeypatch.setattr(session, 'charge', mutate)
        with pytest.raises(ContractError, match='after memory admission'):
            session.read_json(path)
        assert session._managed == 0


def test_iterative_size_counts_aliases_cycles_paths_objects_and_array_ownership(tmp_path):
    import sys
    from dataclasses import dataclass

    from gf_training.preparation import deep_size
    shared = ['日本語', 10**30, None, True, 2.5]
    cycle = [shared, shared]
    cycle.append(cycle)
    expected = sys.getsizeof(cycle) + sys.getsizeof(shared) + sum(sys.getsizeof(item) for item in shared)
    assert deep_size(cycle) == expected
    @dataclass
    class Owned:
        value: object
    owner = Owned(shared)
    expected += sys.getsizeof(owner) + sys.getsizeof(vars(owner)) + sys.getsizeof('value')
    graph = [cycle, owner]
    assert deep_size(graph) == sys.getsizeof(graph) + expected
    path = tmp_path / '日本語'
    assert deep_size([path, path]) == sys.getsizeof([path, path]) + sys.getsizeof(path) + sys.getsizeof(str(path)) + 256
    array = np.arange(20, dtype=np.float64)
    view = array[:10]
    assert deep_size([array, view, array]) == sys.getsizeof([array, view, array]) + sys.getsizeof(array) + sys.getsizeof(view)
    nested = []
    root = nested
    expected = 0
    for _ in range(2000):
        child = []
        nested.append(child)
        expected += sys.getsizeof(nested)
        nested = child
    assert deep_size(root) == expected + sys.getsizeof(nested)


def test_hash_read_buffer_is_bounded_by_its_admitted_file_size(tmp_path, monkeypatch):
    from pathlib import Path
    path = tmp_path / 'source.bin'
    path.write_bytes(b'x')
    real_open = Path.open
    reads = []
    class GrowingReader:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def fileno(self):
            return self.stream.fileno()
        def read(self, size):
            reads.append(size)
            if len(reads) == 1:
                path.write_bytes(b'x' * (1024 * 1024))
            return self.stream.read(size)
    def open_source(self, *args, **kwargs):
        stream = real_open(self, *args, **kwargs)
        return GrowingReader(stream) if self == path and args == ('rb',) else stream
    monkeypatch.setattr(Path, 'open', open_source)
    with PreparationSession() as session:
        with pytest.raises(ContractError, match='admitted hash length'):
            session.hash_file(path)
        assert reads == [2]
        assert session._managed == 0
        assert session.diagnostics['source_reads'] == 0


def test_late_index_writer_close_does_not_start_unbounded_hashing(tmp_path, make_input, monkeypatch):
    import json

    pq = pytest.importorskip('pyarrow.parquet')
    import gf_training.fixed_data as data
    from gf_training.journal import AcceptedDecision
    from gf_training.q_data import RunBundle
    source = tmp_path / 'dataset.json'
    frozen = {'content_sha256': 'frozen'}
    source.write_text(json.dumps(frozen))
    collection = {'run_id': 'run', 'split': 'train', 'episodes': [{'game_id': 'game', 'episode_id': 'episode'}],
                  'schedule': [{'episodes': [{'episode_id': 'episode', 'policies': ['learner', 'opponent']}]}],
                  'ppo': {'parent_run_id': 'parent', 'batch_id': 'batch', 'collection_generation': 1}}
    (tmp_path / 'manifest.json').write_text(json.dumps(collection))
    bundle = RunBundle(tmp_path, collection, (), 0.)
    expired = []
    class ExpiringSession(PreparationSession):
        def check(self, *, cleanup=False):
            if expired and not cleanup:
                raise TimeoutError('writer close passed the execution deadline')
            super().check(cleanup=cleanup)
    monkeypatch.setattr(data, 'PreparationSession', ExpiringSession)
    def validate(path, session, sink):
        sink(AcceptedDecision(make_input(), {'type': 'pray'}, 1., ('run', 'game', 1, 0), 'train'), bundle)
        return frozen
    monkeypatch.setattr(data, '_validated_dataset', validate)
    real_writer = pq.ParquetWriter
    class LateWriter:
        def __init__(self, *args, **kwargs):
            self.writer = real_writer(*args, **kwargs)
        def write_table(self, table):
            self.writer.write_table(table)
        def close(self):
            self.writer.close()
            expired.append(True)
    monkeypatch.setattr(pq, 'ParquetWriter', LateWriter)
    monkeypatch.setattr(data, 'sha256', lambda *args, **kwargs: pytest.fail('hashing started after the writer exhausted the deadline'))
    output = tmp_path / 'index.parquet'
    with pytest.raises(TimeoutError, match='writer close'):
        data.index_dataset(source, output)
    assert not output.exists() and not output.with_suffix('.parquet.json').exists()
