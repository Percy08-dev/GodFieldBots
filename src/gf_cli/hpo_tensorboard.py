"""Optional loopback TensorBoard companion for the interactive HPO watcher."""

from __future__ import annotations

import argparse
import importlib.util
import os
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from gf_cli.hpo_progress_records import StandaloneScalarHistory

_STARTING = 'TensorBoard: starting'


class TensorBoardWatch:
    """Own the optional process, its liveness pipe and private event workspace."""

    def __init__(self, root: Path, *, interval: float) -> None:
        self._workspace: tempfile.TemporaryDirectory[str] | None = None
        self._child: subprocess.Popen[bytes] | None = None
        self._startup = _STARTING
        if importlib.util.find_spec('tensorboard') is None:
            self._startup = 'TensorBoard: optional extra tensorboard is not installed'
            return
        # An inherited TMPDIR may point at the observed run. The directory
        # for derived events must always be outside the read-only parent.
        temp_root = Path('/tmp')
        if temp_root.is_relative_to(root.absolute()):
            self._startup = 'TensorBoard: temporary workspace unavailable outside the run'
            return
        try:
            self._workspace = tempfile.TemporaryDirectory(prefix='gf-hpo-tb-', dir=temp_root)
            self._child = subprocess.Popen(
                [sys.executable, '-m', 'gf_cli.hpo_tensorboard', '--path', str(root.absolute()),
                 '--work-dir', self._workspace.name, '--interval', str(interval)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
            )
        except OSError:
            self._startup = 'TensorBoard: companion could not start'
            self.close()

    def line(self) -> str:
        if self._child is None or self._workspace is None:
            return self._startup
        exit_code = self._child.poll()
        if exit_code is not None:
            return f'TensorBoard: companion stopped (exit {exit_code})'
        try:
            text = (Path(self._workspace.name) / 'status').read_text(encoding='ascii')
        except FileNotFoundError:
            return _STARTING
        except (OSError, UnicodeError):
            return 'TensorBoard: companion status unavailable'
        port, _, status = text.partition(' ')
        if not port.isdecimal() or len(port) > 5 or not 0 < int(port) <= 65535:
            return 'TensorBoard: companion status unavailable'
        line = f'TensorBoard: http://127.0.0.1:{port}/'
        if status == 'waiting':
            line += ' (waiting for consistent committed records)'
        return line

    def close(self) -> None:
        child, self._child = self._child, None
        workspace, self._workspace = self._workspace, None
        if child is not None:
            assert child.stdin is not None
            child.stdin.close()
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        if workspace is not None:
            workspace.cleanup()


def _status(work_dir: Path, port: int, waiting: bool) -> None:
    pending = work_dir / 'status.next'
    pending.write_text(f'{port} {"waiting" if waiting else "ready"}', encoding='ascii')
    pending.replace(work_dir / 'status')


def _worker(root: Path, work_dir: Path, interval: float) -> None:
    program = importlib.import_module('tensorboard.program')
    event_pb2 = importlib.import_module('tensorboard.compat.proto.event_pb2')
    summary_pb2 = importlib.import_module('tensorboard.compat.proto.summary_pb2')
    tensor_pb2 = importlib.import_module('tensorboard.compat.proto.tensor_pb2')
    types_pb2 = importlib.import_module('tensorboard.compat.proto.types_pb2')
    writer_module = importlib.import_module('tensorboard.summary.writer.event_file_writer')

    logdir = work_dir / 'events'
    board = program.TensorBoard()
    board.configure(['tensorboard', '--host=127.0.0.1', '--port=0', '--load_fast=false',
                     '--logdir', str(logdir), '--reload_interval', str(interval)])
    history = StandaloneScalarHistory(root)
    writer = writer_module.EventFileWriter(str(logdir))
    try:
        port = urlsplit(board.launch()).port
        assert port is not None
        metadata = summary_pb2.SummaryMetadata(
            plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name='scalars'),
            data_class=summary_pb2.DATA_CLASS_SCALAR,
        )
        previous_waiting = None
        while not select.select([sys.stdin], [], [], 0)[0]:
            rows = history.poll()
            if select.select([sys.stdin], [], [], 0)[0]:
                break
            if rows:
                observed = time.time()
                for row in rows:
                    summary = summary_pb2.Summary(value=[
                        summary_pb2.Summary.Value(
                            tag=tag, metadata=metadata, tensor=tensor_pb2.TensorProto(
                                dtype=types_pb2.DT_DOUBLE, double_val=[value]),
                        ) for tag, value in row.values.items()
                    ])
                    writer.add_event(event_pb2.Event(
                        wall_time=observed, step=row.step, summary=summary))
                # No retry after a write or flush failure, which might have
                # published a prefix to this event stream.
                writer.flush()
            waiting = rows is None
            if waiting != previous_waiting:
                _status(work_dir, port, waiting)
                previous_waiting = waiting
            if select.select([sys.stdin], [], [], interval)[0]:
                break
    finally:
        writer.close()


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--interval', type=float, required=True)
    args = parser.parse_args()
    _worker(args.path, args.work_dir, args.interval)


if __name__ == '__main__':
    _main()
