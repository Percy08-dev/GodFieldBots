"""Owned campaign records and actual local destination admission; no GPU I/O."""
from __future__ import annotations

import fcntl
import hashlib
import os
import re
import stat
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, Self

from gf_cli.hpo_config import HPODiagnosticStorageConfig, HPOStorageConfig
from gf_common.domain import ContractError, Json, integer
from gf_common.group_pack import canonical_json, parse_json
from gf_common.hpo_storage import (
    HPODiagnosticsConfig,
    check_sqlite_mounts,
    require_zstd,
    windows_drive,
)
from gf_common.hpo_storage import destination_mount as _destination_mount
from gf_common.hpo_storage import existing_ancestor as _existing
from gf_common.hpo_storage import read_mounts as _mounts
from gf_common.hpo_storage import safe_storage_path as _safe


def _deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError('HPO storage admission deadline')


def _sync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _directories(path: Path) -> None:
    _safe(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Existence after an earlier fsync failure is not durability evidence.
    identities = []
    for ancestor in (path, *path.parents):
        info = ancestor.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ContractError('HPO storage ancestry is not a directory')
        identities.append((ancestor, info.st_dev, info.st_ino))
        _sync(ancestor)
    for ancestor, device, inode in identities:
        info = ancestor.lstat()
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (device, inode):
            raise ContractError('HPO storage ancestry changed during synchronization')


class CampaignBusy(ContractError):
    """An actual campaign owner already holds the campaign lock."""


def _read_campaign_record(path: Path, *, limit: int, newline: bool) -> tuple[Json, str]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        raise ContractError('campaign record must be an actual owned regular file')
    if info.st_size > limit + 1:
        raise ContractError('campaign record exceeds declared parent envelope')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as source:
        opened = os.fstat(source.fileno())
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ContractError('campaign record changed before read')
        data = source.read(limit + 2)
        after = os.fstat(source.fileno())
        current = path.lstat()
        before_identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if any((item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns) != before_identity
               for item in (after, current)):
            raise ContractError('campaign record changed during read')
    if not newline:
        encoded = data
    else:
        if not data.endswith(b'\n'):
            raise ContractError('canonical completed campaign record required')
        encoded = data[:-1]
    value = parse_json(encoded, canonical=True, limit=limit)
    if type(value) is not dict:
        raise ContractError('closed campaign record object required')
    return value, hashlib.sha256(data).hexdigest()


def read_campaign_manifest(directory: Path, *, max_owned_bytes: int) -> Json:
    """Read the immutable manifest for preflight before opening a writable owner."""
    path = directory / 'campaign.json'
    _safe(path)
    value, _ = _read_campaign_record(path, limit=max_owned_bytes // 32, newline=True)
    checksum = value.get('loaded_config_sha256')
    if (value.get('kind') != 'hpo-campaign' or type(checksum) is not str
            or re.fullmatch(r'[0-9a-f]{64}', checksum) is None):
        raise ContractError('original campaign configuration hash required')
    return value


class CampaignStore:
    """One actual same-thread campaign flock and immutable bounded JSON files."""

    def __init__(self, directory: Path, *, mode: str, max_owned_bytes: int) -> None:
        integer(max_owned_bytes, 1, 2**63 - 1)
        if mode not in ('fresh', 'resume') or not isinstance(directory, Path):
            raise ContractError('explicit fresh/resume campaign storage required')
        self._limit = max_owned_bytes // 32
        if self._limit < 2:
            raise ContractError('campaign metadata cannot fit declared parent envelope')
        self.directory = directory
        _safe(directory)
        if mode == 'fresh':
            _directories(directory.parent)
            directory.mkdir(mode=0o700, exist_ok=False)
        elif not directory.is_dir():
            raise ContractError('resume requires original campaign directory')
        _directories(directory)
        info = directory.lstat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ContractError('campaign directory must be private and user-owned')
        self._identity = (info.st_dev, info.st_ino)
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._poisoned = False
        self._control: Any = None
        self._borrower: Any = None
        self._lock: BinaryIO | None = None
        fd = os.open(directory / 'lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                raise ContractError('unowned campaign lock')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise CampaignBusy('another process holds this campaign') from error
            self._lock = os.fdopen(fd, 'r+b')
        except BaseException:
            os.close(fd)
            raise
        try:
            _sync(directory)
        except BaseException:
            self.close()
            raise

    def __copy__(self) -> Any:
        raise TypeError('campaign storage owners cannot be copied')

    def __deepcopy__(self, memo: Any) -> Any:
        raise TypeError('campaign storage owners cannot be copied')

    def __reduce__(self) -> Any:
        raise TypeError('campaign storage owners cannot be serialized')

    @classmethod
    def borrow(cls, control: Any, *, max_owned_bytes: int) -> Self:
        """Borrow only the actual authenticated campaign OFD; close-only custody."""
        from gf_training.rust_player_supervision import HostControl

        if type(control) is not HostControl:
            raise ContractError('authenticated original campaign HostControl required')
        control._require()
        integer(max_owned_bytes, 1, 2**63 - 1)
        owner = cls.__new__(cls)
        owner.directory = Path(control.campaign_bootstrap['campaign_directory'])
        owner._limit = max_owned_bytes // 32
        if owner._limit < 2:
            raise ContractError('campaign metadata cannot fit declared parent envelope')
        _safe(owner.directory)
        info = owner.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ContractError('campaign directory must be private and user-owned')
        owner._identity = info.st_dev, info.st_ino
        owner._pid, owner._thread = os.getpid(), threading.get_ident()
        owner._poisoned = False
        owner._control, owner._borrower = control, None
        owner._lock = None
        # Duplication and registration are performed by the authenticated owner;
        # no caller-supplied FD or new lock description is accepted here.
        owner._lock, owner._borrower = control._borrow_campaign_store(owner, owner.directory)
        owner._require()
        return owner

    def _require(self) -> None:
        if (self._lock is None or self._lock.closed or self._poisoned
                or os.getpid() != self._pid or threading.get_ident() != self._thread):
            raise ContractError('original held healthy campaign storage owner required')
        if self._control is not None:
            self._control._require_campaign_borrower(self._borrower, self)
            if self._control._terminal:
                raise ContractError('accepted terminal host may only close its campaign borrower and exit')
        _safe(self.directory)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != self._identity:
            raise ContractError('campaign namespace identity changed')

    def _path(self, name: str) -> Path:
        self._require()
        if type(name) is not str or name == 'lock' or '\x00' in name or any(part in ('', '.', '..') for part in name.split('/')):
            raise ContractError('safe relative campaign record path required')
        path = Path(name)
        if path.is_absolute() or path.parts[0] == 'telemetry':
            raise ContractError('foreign campaign record path')
        result = self.directory / path
        _safe(result)
        return result

    def get(self, name: str) -> tuple[Json, str]:
        path = self._path(name)
        try:
            return _read_campaign_record(path, limit=self._limit, newline=name.split('/')[0] != 'supervision')
        except FileNotFoundError:
            raise
        except ValueError as error:
            self._poisoned = True
            raise ContractError('corrupt or incompatible campaign record') from error

    def exists(self, name: str) -> bool:
        path = self._path(name)
        if path.exists() or path.is_symlink():
            self.get(name)
            return True
        return False

    def put(self, name: str, value: Json) -> str:
        path = self._path(name)
        try:
            try:
                data = canonical_json(value, limit=self._limit) + (b'' if name.split('/')[0] == 'supervision' else b'\n')
            except ValueError as error:
                raise ContractError('campaign record cannot fit declared parent envelope') from error
            expected = hashlib.sha256(data).hexdigest()
            if path.exists() or path.is_symlink():
                _, actual = self.get(name)
                if actual != expected:
                    raise ContractError('immutable campaign record conflicts with existing evidence')
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                _sync(path.parent)
                return actual
            _directories(path.parent)
            temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
            try:
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, 'wb') as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                os.link(temporary, path, follow_symlinks=False)
                temporary.unlink()
                _sync(path.parent)
            finally:
                if temporary.exists():
                    temporary.unlink()
            return expected
        except BaseException:
            self._poisoned = True
            raise

    def update_supervision_cursor(self, value: Json, *, expected_sha256: str | None) -> str:
        """Sole mutable record: canonical supervision/current.json under actual custody.

        The application independently verifies closed seed/disposition evidence;
        this CAS never infers completion or makes an orphan dispatchable.
        """
        from gf_common.rust_player import hex_identity, validate_campaign_cursor

        path = self._path('supervision/current.json')
        validate_campaign_cursor(value)
        if expected_sha256 is not None:
            hex_identity(expected_sha256)
        actual = self.get('supervision/current.json')[1] if path.exists() or path.is_symlink() else None
        if actual != expected_sha256:
            raise ContractError('supervision cursor compare-and-swap conflict')
        try:
            data = canonical_json(value, limit=self._limit)
            checksum = hashlib.sha256(data).hexdigest()
            _directories(path.parent)
            temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
            try:
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, 'wb') as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                # The genuine same-store owner excludes competing writers; expected
                # SHA detects stale application state before the only replacement.
                os.replace(temporary, path)
                _sync(path.parent)
            finally:
                if temporary.exists():
                    temporary.unlink()
            return checksum
        except BaseException:
            self._poisoned = True
            raise

    def read_acknowledged_supervision_disposition(self, name: str) -> tuple[Json, Json]:
        """Verify ALREADY closed terminal DATA under current actual store custody.

        This performs no tell, release, dispatch, old deadline renewal or capability
        reconstruction. The application must independently resolve the CURRENT
        namespace/resources and actual parent custody before reconciling a cursor.
        """
        from gf_common.domain import clone, digest
        from gf_common.rust_player import (
            record,
            safe_component,
            validate_campaign_awaiting,
            validate_campaign_bootstrap,
            validate_campaign_clock,
            validate_campaign_disposition,
            validate_campaign_disposition_ack,
            validate_exit,
            validate_global_admission,
            validate_global_termination,
            validate_loaded_campaign,
            validate_local_admission,
            validate_operation,
            validate_prepared,
            validate_seed_settings,
            validate_training_failure,
        )
        from gf_training.rust_player_supervision import read_control_record

        self._require()
        value, checksum = self.get(name)
        validate_campaign_disposition(value)
        root = self.directory / 'supervision' / value['supervision_id']
        operation, operation_sha = read_control_record(Path(value['operation_path']), root)
        validate_operation(operation)
        operation_id = safe_component(operation['operation_id'])
        if self._path(name) != root / 'terminal' / (operation_id + '.json'):
            raise ContractError('canonical original terminal disposition path required')
        ack_path = root / ('disposition-ack-' + value['invocation_id'] + '.json')
        try:
            ack, _ = read_control_record(ack_path, root)
        except FileNotFoundError as error:
            raise ContractError('terminal disposition lacks durable native ACK; interrupted') from error
        validate_campaign_disposition_ack(ack)
        if (ack['supervision_id'] != value['supervision_id'] or ack['invocation_id'] != value['invocation_id']
                or ack['operation_sha256'] != operation_sha or value['operation_sha256'] != operation_sha
                or ack['disposition_sha256'] != checksum or ack['evidence_sha256'] != digest(value['evidence'])):
            raise ContractError('historical native disposition ACK binding differs')
        bootstrap, _ = read_control_record(root / 'bootstrap.json', root)
        validate_campaign_bootstrap(bootstrap)
        clock, clock_sha = read_control_record(root / 'session-clock.json', root)
        validate_campaign_clock(clock)
        if (Path(bootstrap['campaign_directory']) != self.directory
                or bootstrap['supervision_id'] != value['supervision_id']
                or clock['supervision_id'] != value['supervision_id']
                or clock_sha != value['session_clock_sha256'] or clock_sha != bootstrap['session_clock_sha256']):
            raise ContractError('historical campaign/bootstrap/session clock differs')
        for prefix, validate in (('loaded_campaign', validate_loaded_campaign), ('seed_settings', validate_seed_settings)):
            data, actual = read_control_record(Path(bootstrap[prefix + '_path']), root)
            validate(data)
            if actual != bootstrap[prefix + '_sha256']:
                raise ContractError('historical immutable campaign CONFIG changed')
        validate_operation(operation, session_finish=clock['hard_session_finish_bits'])
        awaiting_path = root / 'awaiting' / (operation_id + '.json')
        awaiting, awaiting_sha = read_control_record(awaiting_path, root)
        validate_campaign_awaiting(awaiting)
        if str(awaiting_path) != value['awaiting_path'] or awaiting_sha != value['awaiting_sha256']:
            raise ContractError('historical terminal awaiting original differs')
        for key in ('campaign_id', 'proposal_no', 'learning_seed', 'supervision_id',
                    'session_clock_sha256', 'operation_path', 'operation_sha256'):
            if awaiting[key] != value[key]:
                raise ContractError('historical disposition/awaiting identity differs')
        for key in ('run_id', 'config_path', 'config_sha256', 'runtime_binding_sha256'):
            if awaiting[key] != operation[key]:
                raise ContractError('historical awaiting/operation identity differs')
        if awaiting['run_path'] != operation['parent_directory']:
            raise ContractError('historical awaiting parent differs')
        admission, _ = read_control_record(root / ('admission-' + value['invocation_id'] + '.json'), root)
        validate_local_admission(admission)
        if (admission['supervision_id'] != value['supervision_id']
                or admission['invocation_id'] != value['invocation_id']
                or admission['global_ticket'] is None or admission['context'] is None
                or admission['context']['campaign_id'] != value['campaign_id']
                or any(admission[key] != operation[key] for key in
                    ('operation_id', 'run_id', 'config_sha256', 'runtime_binding_sha256', 'deadlines'))):
            raise ContractError('historical actual admission/operation identity differs')
        evidence = value['evidence']
        exit_path = root / ('exit-' + value['invocation_id'] + '.json')
        exit_value, exit_sha = read_control_record(exit_path, root)
        validate_exit(exit_value, admission)
        if value['classification'] == 'training_failed':
            failure_path = root / ('failure-' + value['invocation_id'] + '.json')
            failure, failure_sha = read_control_record(failure_path, root)
            validate_training_failure(failure, admission, operation_sha)
            if (evidence['failure_path'] != str(failure_path) or evidence['failure_sha256'] != failure_sha
                    or evidence['exit_path'] != str(exit_path) or evidence['exit_sha256'] != exit_sha
                    or exit_value['cooperative'] is not False or exit_value['prepared_sha256'] is not None
                    or exit_value['wait']['signal'] is not None or exit_value['signals']
                    or any(child['wait']['signal'] is not None for child in exit_value['descendants'])):
                raise ContractError('historical failed return evidence differs')
        else:
            completion_path = root / ('completion-' + value['invocation_id'] + '.json')
            completion, completion_sha = read_control_record(completion_path, root)
            record(completion, 'rust-player-training-completion-v1', {'supervision_id', 'invocation_id',
                'operation_id', 'run_id', 'config_sha256', 'runtime_binding_sha256', 'operation_sha256',
                'prepared_sha256', 'exit_sha256', 'state_sha256', 'generation_sha256', 'source_manifest_sha256',
                'state', 'committed_update', 'audit_path', 'audit_sha256', 'eligible', 'parent_metadata_validation'})
            prepared_path = root / ('prepared-' + value['invocation_id'] + '.json')
            prepared, prepared_sha = read_control_record(prepared_path, root)
            validate_prepared(prepared, admission)
            if (evidence['completion_path'] != str(completion_path) or evidence['completion_sha256'] != completion_sha
                    or prepared['state'] not in ('paused', 'incomplete') or completion['state'] != prepared['state']
                    or completion['eligible'] is not False or completion['operation_sha256'] != operation_sha
                    or completion['prepared_sha256'] != prepared_sha or exit_value['prepared_sha256'] != prepared_sha
                    or completion['exit_sha256'] != exit_sha or exit_value['cooperative'] is not True
                    or any(completion[key] != admission[key] for key in ('supervision_id', 'invocation_id',
                        'operation_id', 'run_id', 'config_sha256', 'runtime_binding_sha256'))):
                raise ContractError('historical genuine incomplete Cold DATA differs')
        physical_path = Path(evidence['physical_path'])
        namespace = physical_path.parent.parent.parent
        if physical_path != namespace / 'jobs' / admission['global_ticket'] / 'termination.json':
            raise ContractError('canonical historical namespace termination path required')
        global_admission, _ = read_control_record(physical_path.parent / 'admission.json', namespace)
        physical, physical_sha = read_control_record(physical_path, namespace)
        validate_global_admission(global_admission, admission['global_ticket'])
        validate_global_termination(physical, global_admission)
        if (physical_sha != evidence['physical_sha256'] or physical['exit_record'] != exit_value
                or physical['exit_sha256'] != exit_sha or global_admission['host_admission'] != admission):
            raise ContractError('historical SAME original physical retirement DATA differs')
        proposal_root = f"proposals/{value['proposal_no']:06}"
        _, proposal_sha = self.get(proposal_root + '/proposal.json')
        if proposal_sha != value['proposal_sha256']:
            raise ContractError('historical original proposal differs')
        rows = []
        for prefix, file in (('proposal_failure', 'failure.json'), ('tell_intent', 'tell.intent.json'), ('tell_receipt', 'tell.receipt.json')):
            path = self.directory / proposal_root / file
            row, actual = self.get(proposal_root + '/' + file)
            if (value[prefix + '_path'] != str(path) or value[prefix + '_sha256'] != actual
                    or row.get('proposal_sha256') != proposal_sha):
                raise ContractError('historical exact closed FAIL evidence differs')
            rows.append(row)
        record(rows[0], 'hpo-proposal-failure', {'proposal_sha256', 'reason', 'boundary'})
        for tell in rows[1:]:
            record(tell, 'hpo-tell', {'proposal_sha256', 'trial_number', 'params', 'distributions',
                'state', 'value', 'evaluation_receipt_sha256', 'reason'})
            if (tell['state'] != 'FAIL' or tell['value'] is not None or tell['evaluation_receipt_sha256'] is not None
                    or tell['reason'] != rows[0]['reason']):
                raise ContractError('historical exact FAIL/value=None tell required')
        if rows[1] != rows[2]:
            raise ContractError('historical FAIL tell intent/receipt differs')
        return clone(value), clone(ack)

    def records(self) -> tuple[str, ...]:
        self._require()
        names = []
        # Exclude this sole mutable diagnostic subtree BEFORE traversal. All
        # other entries retain the original immutable inventory guards.
        pending = [self.directory]
        while pending:
            directory = pending.pop()
            for path in directory.iterdir():
                if path == self.directory / 'telemetry' or path == self.directory / 'lock':
                    continue
                info = path.lstat()
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                    raise ContractError('unowned campaign namespace entry')
                names.append(path.relative_to(self.directory).as_posix())
        return tuple(sorted(names))

    def close(self) -> None:
        if os.getpid() != self._pid or threading.get_ident() != self._thread:
            raise ContractError('foreign process/thread cannot close campaign storage')
        if self._control is not None and self._borrower is not None:
            self._control._close_campaign_borrower(self._borrower, self)
            self._borrower = None
            self._lock = None
        elif self._lock is not None:
            self._lock.close()
            self._lock = None

    def __enter__(self) -> Self:
        self._require()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _free(path: Path) -> int:
    info = os.statvfs(path)
    return info.f_bavail * info.f_frsize


def _usage(path: Path, deadline: float, *, excluded: Path | None = None) -> int:
    if not path.exists():
        return 0
    total = 0
    pending = [path]
    while pending:
        _deadline(deadline)
        current = pending.pop()
        if current == excluded:
            continue
        info = current.lstat()
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
        elif stat.S_ISDIR(info.st_mode):
            with os.scandir(current) as entries:
                pending.extend(Path(entry.path) for entry in entries)
        else:
            raise ContractError('nonregular destination entry cannot establish managed storage usage')
    return total


def _overlap(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def storage_destinations(storage: HPOStorageConfig, *, campaign_dir: Path | None, originals_root: Path,
                         run_directory: Path | None = None, operation_dir: Path | None = None) -> Json:
    """Resolve the declared layout without a live filesystem admission."""
    if not isinstance(storage, HPOStorageConfig):
        raise ContractError('closed HPO storage settings required')
    parent, numeric, optuna = map(Path, (storage.parent_root, storage.numeric_root, storage.optuna_path))
    windows = Path(storage.windows_c_root)
    paths = (parent, numeric, optuna, originals_root, windows)
    for path in paths:
        _safe(path)
    if parent != numeric:
        raise ContractError('HPO numeric controls must colocate with parent and campaign namespace')
    control_dir = campaign_dir if campaign_dir is not None else operation_dir
    if campaign_dir is not None and operation_dir is not None:
        raise ContractError('supply either the HPO campaign or operation directory')
    if control_dir is not None:
        _safe(control_dir)
    if campaign_dir is not None and (campaign_dir == parent or not campaign_dir.is_relative_to(parent)):
        raise ContractError('HPO campaign namespace must be beneath parent_root')
    if run_directory is not None:
        _safe(run_directory)
        if run_directory == parent or not run_directory.is_relative_to(parent):
            raise ContractError('HPO run directory must be beneath parent_root')
    if storage.ext4_control_required is not True:
        raise ContractError('HPO parent/numeric/Optuna must remain ext4')
    result: Json = dict(zip(('parent_root', 'numeric_root', 'optuna_path', 'originals_root',
                            'windows_c_root'), map(str, paths), strict=True))
    result['operation_dir' if operation_dir is not None else 'campaign_dir'] = (
        None if control_dir is None else str(control_dir))
    if not isinstance(storage, HPODiagnosticStorageConfig):
        return result
    diagnostics = storage.hpo_diagnostics
    archive, ram = Path(diagnostics.http_log_archive_root), Path(diagnostics.http_log_ram_root)
    for path in (archive, ram):
        _safe(path)
        if (parent.is_relative_to(path) or any(_overlap(path, other)
                for other in (originals_root, control_dir, optuna, run_directory) if other is not None)):
            raise ContractError('HPO HTTP logs must be separate from original and control directories')
    return result | {'hpo_diagnostics': asdict(diagnostics)}


def check_run_destination(storage: HPOStorageConfig, *, campaign_dir: Path, originals_root: Path,
                          directory: Path) -> None:
    storage_destinations(storage, campaign_dir=campaign_dir, originals_root=originals_root, run_directory=directory)
    _destination_mount(directory, _mounts(), fs_type='ext4', writable=True)


def check_evaluation_destination(output: Path, *, originals_root: Path | None = None,
                                 diagnostics: HPODiagnosticsConfig | None = None) -> None:
    mounts = _mounts()
    _destination_mount(output, mounts, fs_type='ext4', writable=True)
    if originals_root is not None:
        _destination_mount(originals_root, mounts, writable=True)
    if diagnostics is not None:
        if diagnostics.http_log_compression == 'zstd':
            require_zstd()
        for path, fs_type in ((Path(diagnostics.http_log_archive_root), 'ext4'),
                              (Path(diagnostics.http_log_ram_root), 'tmpfs')):
            if _overlap(path, output) or originals_root is not None and _overlap(path, originals_root):
                raise ContractError('HPO HTTP logs must be separate from evaluation and original destinations')
            _destination_mount(path, mounts, fs_type=fs_type, writable=True)


def check_comparison_input(directory: Path, optuna: Path) -> None:
    """Verify destinations opened writable by an existing comparison audit."""
    mounts = _mounts()
    _destination_mount(directory, mounts, fs_type='ext4', writable=True)
    if optuna.exists() or optuna.is_symlink():
        check_sqlite_mounts(optuna, mounts, fs_type='ext4', writable=True)


def check_storage(storage: HPOStorageConfig, *, originals_root: Path, deadline: float,
                  campaign_dir: Path | None = None, operation_dir: Path | None = None,
                  writable: bool = True, run_directory: Path | None = None) -> Json:
    """Check physical destinations before work and account for declared storage reserve."""
    _deadline(deadline)
    if campaign_dir is None and operation_dir is None:
        raise ContractError('an actual HPO campaign or operation directory is required')
    paths = storage_destinations(storage, campaign_dir=campaign_dir, originals_root=originals_root,
                                 run_directory=run_directory, operation_dir=operation_dir)
    parent, optuna = Path(paths['parent_root']), Path(paths['optuna_path'])
    for value in (storage.destination_capacity_bytes, storage.admission_reservation_bytes, storage.windows_c_min_free_bytes):
        integer(value, 1, 2**63 - 1)
    mounts = _mounts()
    targets = ((parent, 'ext4'), (optuna.parent, 'ext4'), (originals_root, None))
    measured = {path: _destination_mount(path, mounts, fs_type=fs_type, writable=writable)
                for path, fs_type in targets}
    control_dir = paths['campaign_dir' if campaign_dir is not None else 'operation_dir']
    _destination_mount(Path(control_dir), mounts, fs_type='ext4', writable=writable)
    check_sqlite_mounts(optuna, mounts, fs_type='ext4', writable=writable)
    if run_directory is not None:
        _destination_mount(run_directory, mounts, fs_type='ext4', writable=writable)
    http_mounts = {}
    archive = None
    if isinstance(storage, HPODiagnosticStorageConfig):
        diagnostics = storage.hpo_diagnostics
        if 'zstd' in (diagnostics.http_log_compression, diagnostics.update_compression):
            require_zstd()
        archive = Path(diagnostics.http_log_archive_root)
        for path, fs_type in ((archive, 'ext4'), (Path(diagnostics.http_log_ram_root), 'tmpfs')):
            mount = _destination_mount(path, mounts, fs_type=fs_type, writable=writable)
            existing = _existing(path)
            free = _free(existing)
            if free == 0:
                raise ContractError(f'HPO HTTP diagnostic destination is full: {path}')
            http_mounts[str(path)] = {'mount': asdict(mount), 'free_bytes': free}
    rows = []
    for path, mount in measured.items():
        _deadline(deadline)
        used, free = _usage(path, deadline, excluded=archive), _free(_existing(path))
        if used + storage.admission_reservation_bytes > storage.destination_capacity_bytes or free < storage.admission_reservation_bytes:
            raise ContractError('actual destination usage/free space cannot admit declared storage reservation')
        rows.append({'path': str(path), 'mount': asdict(mount), 'used_bytes': used, 'free_bytes': free,
            'capacity_bytes': storage.destination_capacity_bytes, 'reservation_bytes': storage.admission_reservation_bytes})
    windows = Path(storage.windows_c_root)
    _safe(windows)
    if not windows.is_dir():
        raise ContractError('actual Windows C path must exist; no ancestor free-space fallback')
    mount = _destination_mount(windows, mounts, writable=False)
    if windows_drive(mount) != 'C':
        raise ContractError('actual mount facts cannot prove configured path is Windows C')
    free = _free(windows)
    if free < storage.windows_c_min_free_bytes:
        raise ContractError('actual Windows C free space is below declared minimum')
    _deadline(deadline)
    receipt: Json = {'schema_version': 1, 'kind': 'hpo-storage-admission',
        'usage_scope': 'regular-file-size-sum' if archive is None else 'regular-file-size-sum-excluding-http-archive',
        'destinations': rows, 'windows_c': {'path': str(windows), 'drive': 'C', 'mount': asdict(mount),
            'free_bytes': free, 'minimum_free_bytes': storage.windows_c_min_free_bytes}}
    if http_mounts:
        receipt['http_diagnostics_destinations'] = http_mounts
    return receipt
