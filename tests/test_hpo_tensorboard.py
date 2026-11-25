import asyncio
import io
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from gf_common.domain import canonical, digest
from gf_common.hpo_update_bytes import encode_update
from gf_training.hpo_update_storage import update_name


class HistoryFiles:
    def __init__(self, root, *, compressed):
        self.root = root
        self.compressed = compressed
        for directory in ('.rust-player', 'generations', 'updates'):
            (root / directory).mkdir(parents=True)
        effective = {
            'training': {'max_updates': 1000},
            'hpo_spec': {'schema_version': 6, 'decision_budget': 500,
                'storage': {'hpo_diagnostics': {
                    'update_compression': 'zstd' if compressed else 'none',
                    'http_log_ram_root': '/tmp/ram', 'http_log_archive_root': '/tmp/archive',
                    'http_log_retention': 'errors', 'http_log_compression': 'zstd'}}},
        }
        self.state = {
            'schema_version': 5, 'run_variant': 'hpo-ppo-owned-rust-player-v1',
            'run_id': root.name, 'model_id': 'mlp', 'seed': 17, 'status': 'running',
            'updates': 0, 'generation_sha256': None, 'effective_config': effective,
            'config_sha256': digest(effective),
        }
        self.write('.rust-player/operation-fresh.json', {
            'schema_version': 2, 'operation_type': 'hpo-train', 'run_id': root.name,
            'config_sha256': self.state['config_sha256'],
        })
        self.write('run.json', self.state)
        self.hashes = {0: self.generation(0, None)}

    def write(self, path, data):
        data = canonical(data).encode()
        (self.root / path).write_bytes(data)
        return sha256(data).hexdigest()

    def generation(self, n, parent_sha, *, update_parent=None):
        update_sha = None
        if n:
            metrics = {'total_loss': n / 10, 'policy_loss': -.03, 'value_mse': .4,
                'entropy': .8, 'approx_kl': .01, 'unknown': 999}
            if n == 1:
                metrics['diagnostics'] = {'schema_version': 1, 'clip_fraction': .3,
                    'explained_variance': None, 'pre_clip_grad_norm_mean': True,
                    'pre_clip_grad_norm_max': 17, 'learning_rate': .001}
            data = encode_update({
                'schema_version': 8, 'status': 'complete', 'run_id': self.root.name,
                'generation': n, 'parent_sha256': update_parent or parent_sha,
                'metrics': metrics, 'collection_seconds': 3.4, 'update_seconds': 1.2,
            }, compressed=self.compressed)
            path = self.root / 'updates' / update_name(n, compressed=self.compressed)
            path.write_bytes(data)
            update_sha = sha256(data).hexdigest()
        return self.write(f'generations/{n:06d}.json', {
            'schema_version': 6, 'run_id': self.root.name, 'generation': n,
            'parent_sha256': parent_sha, 'update_sha256': update_sha,
            'used_decisions': n * 2, 'collected_decisions': n * 3,
        })

    def commit_pointer(self, n, *, publish=True):
        self.write('latest.json', {'generation': n, 'sha256': self.hashes[n]})
        if publish:
            self.publish(n)

    def publish(self, n):
        self.state.update(updates=n, generation_sha256=self.hashes[n])
        self.write('run.json', self.state)

    def add(self, n, *, publish=True, update_parent=None):
        self.hashes[n] = self.generation(n, self.hashes[n - 1], update_parent=update_parent)
        self.commit_pointer(n, publish=publish)


@pytest.fixture(params=[False, True], ids=['json', 'zstd'])
def history_files(tmp_path, request):
    return HistoryFiles(tmp_path / 'standalone', compressed=request.param)


def test_backfill_then_reads_only_new_commits(history_files, monkeypatch):
    from gf_cli import hpo_progress_records as records

    history_files.add(1)
    history_files.add(2)
    history = records.StandaloneScalarHistory(history_files.root)
    reads = []
    real_read = records.read_hpo_update

    def count_read(parent, number, *args, **kwargs):
        reads.append(number)
        return real_read(parent, number, *args, **kwargs)

    monkeypatch.setattr(records, 'read_hpo_update', count_read)
    rows = history.poll()
    assert [row.step for row in rows] == [1, 2]
    assert rows[0].values == {
        'ppo/total_loss': .1, 'ppo/policy_loss': -.03, 'ppo/value_mse': .4,
        'ppo/entropy': .8, 'ppo/approx_kl': .01, 'diagnostics/clip_fraction': .3,
        'diagnostics/pre_clip_grad_norm_max': 17, 'diagnostics/learning_rate': .001,
        'timing/collection_seconds': 3.4, 'timing/update_seconds': 1.2,
        'progress/used_decisions': 2, 'progress/collected_decisions': 3,
    }
    assert 'diagnostics/clip_fraction' not in rows[1].values
    assert sorted(reads) == [1, 2]

    assert history.poll() == []
    history_files.add(3, publish=False)
    assert history.poll() is None
    assert sorted(reads) == [1, 2]
    history_files.publish(3)
    assert [row.step for row in history.poll()] == [3]
    assert history.poll() == []
    assert sorted(reads) == [1, 2, 3]


def test_requires_verified_generation_zero_and_no_point_at_zero(history_files):
    from gf_cli.hpo_progress_records import StandaloneScalarHistory

    history = StandaloneScalarHistory(history_files.root)
    assert history.poll() is None
    history_files.commit_pointer(0)
    assert history.poll() == []
    history_files.add(1)
    assert [row.step for row in history.poll()] == [1]


@pytest.mark.parametrize('fault', ['update-bytes', 'update-parent', 'previous-anchor', 'rewind'])
def test_rejects_entire_invalid_suffix(history_files, fault):
    from gf_cli.hpo_progress_records import StandaloneScalarHistory

    history_files.add(1)
    history = StandaloneScalarHistory(history_files.root)
    assert [row.step for row in history.poll()] == [1]
    if fault == 'rewind':
        history_files.commit_pointer(0)
    else:
        if fault == 'previous-anchor':
            history_files.hashes[1] = 'f' * 64
        history_files.add(2, update_parent='e' * 64 if fault == 'update-parent' else None)
        history_files.add(3)
        if fault == 'update-bytes':
            name = update_name(2, compressed=history_files.compressed)
            (history_files.root / 'updates' / name).write_bytes(b'changed')
    assert history.poll() is None
    if fault == 'previous-anchor':
        history_files.hashes[1] = sha256(
            (history_files.root / 'generations/000001.json').read_bytes()).hexdigest()
    history_files.add(2)
    history_files.add(3)
    assert [row.step for row in history.poll()] == [2, 3]


@pytest.mark.parametrize(('tty', 'screen', 'once', 'json', 'tensorboard', 'starts'), [
    (True, True, False, False, True, 1),
    (True, False, False, False, True, 1),
    (True, True, True, False, True, 0),
    (True, True, False, True, True, 0),
    (False, False, False, False, True, 0),
    (True, True, False, False, False, 0),
])
def test_watch_tensorboard_terminal_line(
    tmp_path, monkeypatch, tty, screen, once, json, tensorboard, starts,
):
    from gf_cli import hpo_progress, hpo_tensorboard

    files = HistoryFiles(tmp_path / 'standalone', compressed=False)
    files.add(1)

    class Output(io.StringIO):
        def isatty(self):
            return tty

    started, closed, overviews = [], [], []
    url = 'TensorBoard: http://127.0.0.1:4321/'

    class Session:
        def __init__(self, root, *, interval):
            started.append((root, interval))

        def line(self):
            return url

        def publish_overview(self, overview):
            overviews.append(overview)

        def close(self):
            closed.append(True)

    frames = 0

    async def stop_after_two(_interval):
        nonlocal frames
        frames += 1
        if frames == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(hpo_progress, '_terminal_screen',
                        lambda _output: ('ENTER', 'LEAVE', 'CLEAR') if screen else None)
    monkeypatch.setattr(hpo_progress.asyncio, 'sleep', stop_after_two)
    monkeypatch.setattr(hpo_tensorboard, 'TensorBoardWatch', Session)
    output = Output()
    coroutine = hpo_progress.watch(files.root, output=output, once=once, as_json=json,
                                   tensorboard=tensorboard, interval=.2)
    if once:
        assert asyncio.run(coroutine) == 0
    else:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(coroutine)
    assert started == [(files.root, .2)] * starts
    assert closed == [True] * starts
    assert len(overviews) == 2 * starts
    assert all(step == 1 and '## Committed progress' in text for step, text in overviews)
    assert output.getvalue().splitlines().count(url) == starts * (2 if screen else 1)
    if screen and not (once or json):
        assert output.getvalue().endswith('LEAVE')


def test_tensorboard_missing_is_optional(tmp_path, monkeypatch):
    from gf_cli import hpo_tensorboard as tb

    monkeypatch.setattr(tb.importlib.util, 'find_spec', lambda _: None)
    session = tb.TensorBoardWatch(tmp_path, interval=1.)
    try:
        assert 'not installed' in session.line()
    finally:
        session.close()


@pytest.mark.parametrize('fail', [False, True])
def test_companion_status_and_reap(tmp_path, monkeypatch, fail):
    from gf_cli import hpo_tensorboard as tb

    popen = tb.subprocess.Popen
    started = []
    script = """
import os
import sys
from pathlib import Path

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
directory = Path(sys.argv[1])
if sys.argv[2] == 'True':
    sys.exit(9)
(directory / 'status').write_text('4321 ready', encoding='ascii')
assert sys.stdin.read() == ''
"""

    def spawn(args, **kwargs):
        directory = Path(args[args.index('--work-dir') + 1])
        child = popen([sys.executable, '-c', script, str(directory), str(fail)], **kwargs)
        started.append((child, directory))
        return child

    monkeypatch.setattr(tb.importlib.util, 'find_spec', lambda _: object())
    monkeypatch.setattr(tb.subprocess, 'Popen', spawn)
    session = tb.TensorBoardWatch(tmp_path, interval=1.)
    child, directory = started[0]
    try:
        import time

        deadline = time.monotonic() + 3
        while 'starting' in session.line() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ('exit 9' if fail else 'http://127.0.0.1:4321/') in session.line()
    finally:
        session.close()
    assert child.poll() == (9 if fail else 0)
    assert not directory.exists()


def test_worker_writes_tensorboard_scalar_events(tmp_path, monkeypatch):
    from gf_cli import hpo_tensorboard as tb

    program = pytest.importorskip('tensorboard.program')
    loader = pytest.importorskip('tensorboard.backend.event_processing.event_file_loader')
    summary_pb2 = pytest.importorskip('tensorboard.compat.proto.summary_pb2')
    files = HistoryFiles(tmp_path / 'standalone', compressed=True)
    files.add(1)
    files.add(2)
    work_dir = tmp_path / 'dashboard'
    work_dir.mkdir()

    monkeypatch.setattr(program.TensorBoard, 'launch', lambda self: 'http://127.0.0.1:4321/')
    select = tb.select.select

    def stop_after_first_poll(readable, writable, exceptional, timeout):
        if readable == [sys.stdin]:
            return (readable if timeout else [], [], [])
        return select(readable, writable, exceptional, timeout)

    monkeypatch.setattr(tb.select, 'select', stop_after_first_poll)
    tb._worker(files.root, work_dir, 1.)

    assert (work_dir / 'status').read_text(encoding='ascii') == '4321 ready'
    [events] = (work_dir / 'events').glob('events.out.tfevents.*')
    summaries = [
        (event.step, {value.tag: value for value in event.summary.value})
        for event in loader.LegacyEventFileLoader(str(events)).Load() if event.HasField('summary')
    ]
    assert [step for step, _ in summaries] == [1, 2]
    loss = summaries[0][1]['ppo/total_loss']
    assert loss.metadata.plugin_data.plugin_name == 'scalars'
    assert loss.metadata.data_class == summary_pb2.DATA_CLASS_SCALAR
    assert list(loss.tensor.double_val) == [.1]
    assert 'diagnostics/clip_fraction' not in summaries[1][1]


def _watch_observation():
    return {
        'availability': 'available',
        'snapshot': {
            'standalone': True, 'phase': 'collect', 'status': 'running',
            'batch_games_ended': 51, 'batch_games_planned': 256,
            'transient_learner_decisions': 72, 'invocation_elapsed_seconds': 200.5,
            'target_updates': 145, 'eta_seconds': 111.4,
        },
        'durable': {
            'model': 'mlp', 'seed': 17, 'availability': 'available', 'status': 'running',
            'elapsed_seconds': 210.6, 'target_updates': 1000, 'decision_budget': 50000,
            'committed_updates': 12, 'committed_used': 1212, 'committed_collected': 1220,
            'latest_metrics': {'total_loss': .05, 'entropy': 1.24, 'approx_kl': .008},
            'latest_durations': {'collection_seconds': 12.3, 'update_seconds': 7.4},
            'last_collection_games_per_second': 8.2, 'last_collection_actions_per_second': 190.4,
        },
    }


def test_run_overview_uses_watch_values_in_four_groups():
    from gf_cli.hpo_progress import _standalone_overview

    value = _watch_observation()
    step, markdown = _standalone_overview(value)
    assert step == 12
    run, batch, committed, latest = markdown.split('\n\n## ')
    assert run.startswith('## Run\n')
    assert '**Model** mlp' in run and '**Seed** 17' in run and '**Recorded status** running' in run
    assert batch.startswith('Current batch\n')
    for item in ('collect', '51 / 256', '72', '200.50 s', '**Invocation ETA** 111 s'):
        assert item in batch
    assert committed.startswith('Committed progress\n')
    for item in ('12 / 145', '1,212 / 50,000', '1,220'):
        assert item in committed
    assert latest.startswith('Latest update\n')
    for item in ('0.05', '1.24', '0.008', '12.30 s', '7.40 s', '8.20 games/s · 190.40 actions/s'):
        assert item in latest


def test_run_overview_marks_independent_freshness_and_hides_stale_live_values():
    from gf_cli.hpo_progress import _standalone_overview

    value = _watch_observation()
    value['availability'] = 'stale'
    value['durable']['host_result'] = 'awaiting-supervised-finalization'
    step, markdown = _standalone_overview(value)
    assert step == 12
    assert 'Pending supervised finalization' in markdown
    assert '**Live observation** Stale' in markdown
    assert '**Run elapsed** 210.60 s' in markdown
    assert '**Updates** 12 / 1,000' in markdown
    assert '**Phase**' not in markdown and '51 / 256' not in markdown and 'Invocation ETA' not in markdown
    assert '0.05' in markdown and 'Last verified' not in markdown

    value['availability'] = 'available'
    value['durable']['availability'] = 'stale'
    step, markdown = _standalone_overview(value)
    assert step == 12
    assert '**Phase** collect' in markdown and '51 / 256' in markdown
    assert 'Last verified run records' in markdown and 'Invocation ETA' not in markdown
    assert '**Updates** 12 / 1,000' in markdown

    value['availability'] = 'missing'
    value['snapshot'] = None
    value['durable']['availability'] = 'partial'
    value['durable']['committed_updates'] = None
    value['durable']['latest_metrics'] = None
    step, markdown = _standalone_overview(value)
    assert step is None
    assert '**Live observation** Unavailable' in markdown
    assert 'Committed records unavailable' in markdown
    assert '**Updates**' not in markdown and '**Loss**' not in markdown
    assert '**Step 0** Unconfirmed placeholder' in markdown
    del value['durable']
    value.update(availability='available', snapshot={'standalone': True, 'phase': 'learn'})
    assert _standalone_overview(value) is None


def test_overview_preserves_confirmed_generation_zero():
    from gf_cli.hpo_progress import _standalone_overview

    value = _watch_observation()
    value['durable'].update(committed_updates=0, committed_used=0, committed_collected=0)
    step, markdown = _standalone_overview(value)
    assert step == 0 and '**Updates** 0 / 145' in markdown
    assert 'Unconfirmed placeholder' not in markdown


def test_overview_snapshot_replaces_only_changed_values(tmp_path, monkeypatch):
    import json

    from gf_cli import hpo_tensorboard as tb

    monkeypatch.setattr(tb.importlib.util, 'find_spec', lambda _: object())
    child = type('Child', (), {'poll': lambda self: None})()
    monkeypatch.setattr(tb.subprocess, 'Popen', lambda *args, **kwargs: child)
    session = tb.TensorBoardWatch(tmp_path, interval=1.)
    assert session._workspace is not None
    directory = Path(session._workspace.name)
    snapshot = directory / 'overview.json'
    try:
        session.publish_overview((1, '## First'))
        first = json.loads(snapshot.read_bytes())
        assert first['step'] == 1 and first['markdown'] == '## First'
        original = snapshot.stat().st_ino
        session.publish_overview((1, '## First'))
        assert snapshot.stat().st_ino == original
        session.publish_overview((2, '## Second'))
        assert snapshot.stat().st_ino != original
        assert json.loads(snapshot.read_bytes())['markdown'] == '## Second'
        snapshot.with_suffix('.next').mkdir()
        session.publish_overview((3, '## Fails'))
        assert not snapshot.exists()
    finally:
        session._workspace.cleanup()


def test_overview_reuses_native_text_routes_and_sanitizes_markdown(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from gf_cli import hpo_tensorboard as tb

    base = pytest.importorskip('tensorboard.plugins.base_plugin')
    util = pytest.importorskip('tensorboard.plugin_util')
    client = pytest.importorskip('werkzeug.test')
    plugins = tb._overview_plugins(tmp_path)
    overview = next(plugin for plugin in plugins if getattr(plugin, 'plugin_name', None) == 'text')
    assert sum(plugin is overview for plugin in plugins) == 1
    assert any(getattr(plugin, 'plugin_name', None) == 'scalars' for plugin in plugins)
    instance = overview(SimpleNamespace(sampling_hints={}, data_provider=None))
    assert instance.is_active()
    metadata = instance.frontend_metadata()
    assert isinstance(metadata, base.FrontendMetadata)
    assert metadata.tab_name == 'Overview'
    assert metadata.element_name == 'tf-text-dashboard'
    routes = instance.get_plugin_apps()
    tags = client.Client(routes['/tags']).get('/tags').get_json()
    assert tags == {'.': ['HPO watch']}
    http = client.Client(routes['/text'])

    def current_row():
        [row] = http.get('/text', query_string={'run': '.', 'tag': 'HPO watch'}).get_json()
        return row

    missing = current_row()
    assert missing['step'] == 0 and 'overview unavailable' in missing['text']
    assert 'Step 0 is an unconfirmed placeholder' in missing['text']
    snapshot = tmp_path / 'overview.json'
    snapshot.write_text(json.dumps({
        'wall_time': 123.45, 'step': 17, 'markdown': '## Run\n\n<script>bad()</script>\n\n**Seed** 17',
    }))
    first = current_row()
    assert first['step'] == 17
    assert '<h2>Run</h2>' in first['text'] and '<strong>Seed</strong> 17' in first['text']
    assert '<script>' not in first['text']
    second = {'wall_time': 123.46, 'step': 18, 'markdown': '## Current batch'}
    snapshot.write_text(json.dumps(second))
    assert current_row()['step'] == 18 and '<h2>Current batch</h2>' in current_row()['text']
    snapshot.write_text('malformed')
    corrupt = current_row()
    assert corrupt['step'] == 0 and 'Step 0 is an unconfirmed placeholder' in corrupt['text']
    snapshot.write_text(json.dumps({**second, 'step': True}))
    assert 'overview unavailable' in current_row()['text']
    assert http.get('/text', query_string={'run': 'anything', 'tag': 'HPO watch'}).get_json() == []

    def bad_render(_):
        raise RuntimeError('third-party rendering failure')

    monkeypatch.setattr(util, 'markdown_to_safe_html', bad_render)
    snapshot.write_text(json.dumps(second))
    degraded = current_row()
    assert degraded['step'] == 18 and 'overview unavailable' in degraded['text']
    assert 'unconfirmed placeholder' not in degraded['text']
    snapshot.write_text(json.dumps({**second, 'step': None}))
    unconfirmed = current_row()
    assert unconfirmed['step'] == 0 and 'Step 0 is an unconfirmed placeholder' in unconfirmed['text']
