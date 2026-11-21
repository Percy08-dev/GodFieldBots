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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Self

from gf_cli.hpo_config import HPOStorageConfig
from gf_common.domain import ContractError, Json, integer
from gf_common.group_pack import canonical_json, parse_json


def _deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError('HPO storage admission deadline')


def _safe(path: Path) -> None:
    if not path.is_absolute() or '..' in path.parts:
        raise ContractError('absolute HPO owned storage path required')
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ContractError('symlink in HPO owned storage ancestry')


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
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                raise ContractError('campaign record must be an actual owned regular file')
            if info.st_size > self._limit + 1:
                raise ContractError('campaign record exceeds declared parent envelope')
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as source:
                opened = os.fstat(source.fileno())
                if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                    raise ContractError('campaign record changed before read')
                data = source.read(self._limit + 2)
                after = os.fstat(source.fileno())
                current = path.lstat()
                before_identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                if any((item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns) != before_identity
                       for item in (after, current)):
                    raise ContractError('campaign record changed during read')
            if name.split('/')[0] == 'supervision':
                encoded = data
            else:
                if not data.endswith(b'\n'):
                    raise ContractError('canonical completed campaign record required')
                encoded = data[:-1]
            value = parse_json(encoded, canonical=True, limit=self._limit)
            if type(value) is not dict:
                raise ContractError('closed campaign record object required')
            return value, hashlib.sha256(data).hexdigest()
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


@dataclass(frozen=True)
class Mount:
    mount_id: int
    device: str
    root: str
    mount_point: str
    fs_type: str
    source: str
    options: tuple[str, ...]


def _unescape(value: str) -> str:
    return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), value)


def mount_facts(text: str) -> tuple[Mount, ...]:
    result = []
    for line in text.splitlines():
        try:
            before, after = line.split(' - ', 1)
            fields, filesystem = before.split(), after.split()
            if len(fields) < 6 or len(filesystem) != 3:
                raise ValueError
            mount = Mount(int(fields[0]), fields[2], _unescape(fields[3]), _unescape(fields[4]),
                filesystem[0], _unescape(filesystem[1]),
                tuple(_unescape(value) for value in (fields[5] + ',' + filesystem[2]).split(',')))
            if not Path(mount.mount_point).is_absolute():
                raise ValueError
            result.append(mount)
        except (ValueError, IndexError) as error:
            raise ContractError('unreadable actual Linux mount identity') from error
    if not result:
        raise ContractError('actual Linux mount table required')
    return tuple(result)


def _mounts() -> tuple[Mount, ...]:
    return mount_facts(Path('/proc/self/mountinfo').read_text())


def windows_drive(mount: Mount) -> str | None:
    if mount.fs_type not in ('9p', 'drvfs', 'fuseblk'):
        return None
    match = re.match(r'^([A-Za-z]):[\\/]', mount.source)
    if match is not None:
        return match[1].upper()
    if mount.fs_type in ('9p', 'drvfs'):
        for option in mount.options:
            match = re.search(r'(?:^|;)path=([A-Za-z]):[\\/]', option)
            if match is not None and ('drvfs' in option or mount.fs_type == 'drvfs'):
                return match[1].upper()
    return None


def _existing(path: Path) -> Path:
    _safe(path)
    while not path.exists():
        path = path.parent
    return path


def _mount(path: Path, mounts: tuple[Mount, ...]) -> Mount:
    actual = _existing(path)
    candidates = [mount for mount in mounts if actual.is_relative_to(Path(mount.mount_point))]
    if not candidates:
        raise ContractError('destination has no actual mounted filesystem identity')
    return max(candidates, key=lambda mount: (len(Path(mount.mount_point).parts), mount.mount_id))


def _free(path: Path) -> int:
    info = os.statvfs(path)
    return info.f_bavail * info.f_frsize


def _usage(path: Path, deadline: float) -> int:
    if not path.exists():
        return 0
    total = 0
    pending = [path]
    while pending:
        _deadline(deadline)
        current = pending.pop()
        info = current.lstat()
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
        elif stat.S_ISDIR(info.st_mode):
            with os.scandir(current) as entries:
                pending.extend(Path(entry.path) for entry in entries)
        else:
            raise ContractError('nonregular destination entry cannot establish managed storage usage')
    return total


def check_storage(storage: HPOStorageConfig, *, campaign_dir: Path, originals_root: Path,
                  deadline: float) -> Json:
    """Measure actual destinations before work; this is not an OS quota promise."""
    if type(storage) is not HPOStorageConfig:
        raise ContractError('closed HPO storage settings required')
    _deadline(deadline)
    parent, numeric, optuna = map(Path, (storage.parent_root, storage.numeric_root, storage.optuna_path))
    for path in (parent, numeric, optuna, campaign_dir, originals_root):
        _safe(path)
    if parent != numeric or not campaign_dir.is_relative_to(parent):
        raise ContractError('HPO numeric controls must colocate with parent and campaign namespace')
    if storage.ext4_control_required is not True:
        raise ContractError('HPO parent/numeric/Optuna must remain ext4')
    for value in (storage.destination_capacity_bytes, storage.admission_reservation_bytes, storage.windows_c_min_free_bytes):
        integer(value, 1, 2**63 - 1)
    mounts = _mounts()
    rows = []
    seen = set()
    for path, ext4 in ((parent, True), (numeric, True), (optuna.parent, True), (originals_root, False)):
        if path in seen:
            continue
        seen.add(path)
        _deadline(deadline)
        mount = _mount(path, mounts)
        if ext4 and mount.fs_type != 'ext4':
            raise ContractError('HPO parent/model/numeric/Optuna destination is not actual ext4')
        if path.is_relative_to(Path('/mnt/d')) and windows_drive(mount) != 'D':
            raise ContractError('configured D-drive destination cannot fall back to an SSD ancestor')
        used, free = _usage(path, deadline), _free(_existing(path))
        if used + storage.admission_reservation_bytes > storage.destination_capacity_bytes or free < storage.admission_reservation_bytes:
            raise ContractError('actual destination usage/free space cannot admit declared storage reservation')
        rows.append({'path': str(path), 'mount': asdict(mount), 'used_bytes': used, 'free_bytes': free,
            'capacity_bytes': storage.destination_capacity_bytes, 'reservation_bytes': storage.admission_reservation_bytes})
    windows = Path(storage.windows_c_root)
    _safe(windows)
    if not windows.is_dir():
        raise ContractError('actual Windows C path must exist; no ancestor free-space fallback')
    mount = _mount(windows, mounts)
    if windows_drive(mount) != 'C':
        raise ContractError('actual mount facts cannot prove configured path is Windows C')
    free = _free(windows)
    if free < storage.windows_c_min_free_bytes:
        raise ContractError('actual Windows C free space is below declared minimum')
    _deadline(deadline)
    return {'schema_version': 1, 'kind': 'hpo-storage-admission', 'usage_scope': 'regular-file-size-sum',
        'destinations': rows, 'windows_c': {'path': str(windows), 'drive': 'C', 'mount': asdict(mount),
            'free_bytes': free, 'minimum_free_bytes': storage.windows_c_min_free_bytes}}
