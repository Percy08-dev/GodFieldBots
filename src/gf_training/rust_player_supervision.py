"""Authenticated single-host control and original descriptor custody.

Rust owns admission/physical retirement. These live capabilities cannot be
reconstituted from durable JSON. Source qualification belongs to the cold learner
finalizer; this module never labels a parent eligible.
"""

from __future__ import annotations

import array
import fcntl
import hashlib
import importlib.util
import json
import os
import socket
import stat
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer
from gf_common.rust_player import (
    BUILD_KEYS,
    CONTROL_RECORD_LIMIT,
    LINUX_F_GET_SEALS,
    LINUX_REQUIRED_LAUNCH_SEALS,
    deadline_seconds,
    hex_identity,
    text,
    validate_campaign_bootstrap,
    validate_campaign_clock,
    validate_campaign_disposition,
    validate_campaign_disposition_ack,
    validate_control_completion,
    validate_exit,
    validate_global_admission,
    validate_global_termination,
    validate_launch,
    validate_loaded_campaign,
    validate_local_admission,
    validate_operation,
    validate_prepared,
    validate_producer,
    validate_seed_settings,
    validate_training_failure,
)

_LIMIT = CONTROL_RECORD_LIMIT
_HEADER = {"v", "kind", "supervision_id", "invocation_id", "nonce", "seq"}
_ISSUER = object()


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_control_record(path: Path, root: Path) -> tuple[Json, str]:
    path, root = Path(path), Path(root)
    if not path.is_absolute() or not path.is_relative_to(root) or ".." in path.parts:
        raise ContractError("control original outside owned record namespace")
    parent = path.parent
    while True:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ContractError("symlink in control original ancestry")
        if parent == root:
            break
        parent = parent.parent
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_nlink != 1:
            raise ContractError("owned regular control original required")
        with os.fdopen(fd, "rb", closefd=False) as source:
            data = source.read()
        after, current = os.fstat(fd), path.lstat()
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in keys) or (
            (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise ContractError("control original changed")
        value = json.loads(data)
        if type(value) is not dict or data != canonical(value).encode():
            raise ContractError("new protocol requires exact canonical JSON without newline")
        if value.get("kind") == "rust-player-training-failure-v1" and len(data) > _LIMIT:
            raise ContractError("whole canonical training failure record exceeds control envelope")
        return value, hashlib.sha256(data).hexdigest()
    finally:
        os.close(fd)


def write_control_record(path: Path, value: Json) -> str:
    """Only NEW protocol files; never use for existing lineage/checkpoint formats."""
    path = Path(path)
    data = canonical(value).encode()
    if value.get("kind") == "rust-player-training-failure-v1" and len(data) > _LIMIT:
        raise ContractError("whole canonical training failure record exceeds control envelope")
    checksum = hashlib.sha256(data).hexdigest()
    if path.exists() or path.is_symlink():
        previous, actual = read_control_record(path, path.parent)
        if previous != value or actual != checksum:
            raise ContractError("immutable protocol original differs")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        _sync_directory(path.parent)
        return checksum
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        _sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return checksum


def _launch_original(fd: int) -> Json:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > _LIMIT:
        raise ContractError("owned compact launch descriptor required")
    access = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
    try:
        seals = fcntl.fcntl(fd, LINUX_F_GET_SEALS)
    except OSError:
        seals = 0
    required = LINUX_REQUIRED_LAUNCH_SEALS
    if access != os.O_RDONLY and seals & required != required:
        raise ContractError("launch must be readonly or sealed")
    data = os.pread(fd, info.st_size + 1, 0)
    after = os.fstat(fd)
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ContractError("launch original changed")
    value = validate_launch(json.loads(data))
    if data != canonical(value).encode():
        raise ContractError("canonical launch original required")
    return value


class _Capability:
    def __copy__(self) -> Any:
        raise TypeError("supervised live capabilities cannot be copied")

    def __deepcopy__(self, memo: Any) -> Any:
        raise TypeError("supervised live capabilities cannot be copied")

    def __reduce__(self) -> Any:
        raise TypeError("supervised live capabilities cannot be serialized")


class _ParentBorrower(_Capability):
    """One original local FD lifetime; not a receipt or transfer success."""

    def __init__(self, issuer: object, control: Any, file: Any, owner: Any,
                 *, authenticated: bool) -> None:
        if issuer is not _ISSUER:
            raise ContractError("actual parent borrower registration required")
        self.control, self.file, self.owner = control, file, owner
        self.fd = -1 if file is None else file.fileno()
        self.identity = (0, 0)
        if file is not None:
            info = os.fstat(self.fd)
            self.identity = info.st_dev, info.st_ino
        self.authenticated = authenticated
        self._campaign_directory: Path | None = None
        self._campaign_raw_close_uncertain = False


class HostAdmission(_Capability):
    def __init__(self, issuer: object, control: HostControl, record: Json, path: Path) -> None:
        if issuer is not _ISSUER:
            raise ContractError("authenticated original host admission required")
        self._control, self._record, self.path = control, clone(record), path

    @property
    def observation(self) -> Json:
        self.require()
        return clone(self._record)

    def require(self) -> None:
        self._control._require()
        if self._control._prepared or self._control._terminal:
            raise ContractError("prepared host admission is sealed")
        if self._control._admission is not self:
            raise ContractError("SAME actual host admission capability required")
        current, actual = read_control_record(self.path, self._control.root)
        if current != self._record or actual != digest(self._record):
            raise ContractError("original host admission changed")


class PreparedResult(_Capability):
    def __init__(self, issuer: object, admission: HostAdmission, path: Path, checksum: str) -> None:
        if issuer is not _ISSUER:
            raise ContractError("actual prepared acknowledgement required")
        self.admission, self.path, self.sha256 = admission, path, checksum


class OriginalHostExit(_Capability):
    """Actual authenticated cold handoff's positive original-host exit owner."""

    def __init__(self, issuer: object, cold: ColdContinuation) -> None:
        if issuer is not _ISSUER:
            raise ContractError("actual supervised original exit required")
        self._cold = cold

    def require(self) -> Json:
        _, _, exit_record = self._cold.require_pair()
        if self._cold.original_exit is not self:
            raise ContractError("SAME actual original exit capability required")
        return exit_record


class RustPhysicalTermination(_Capability):
    """Physical receipt authority from Rust's actual reap, never an old GpuJob."""

    def __init__(self, issuer: object, cold: ColdContinuation, original: OriginalHostExit) -> None:
        if issuer is not _ISSUER:
            raise ContractError("actual supervised physical retirement required")
        self._cold, self._original = cold, original

    def require(self) -> Json:
        if self._cold.physical_termination is not self:
            raise ContractError("SAME actual physical receipt capability required")
        return self._original.require()

    def global_observation(self) -> Json:
        """Reopen exact HPO3 record; the live authority is this original cold owner."""
        import pwd

        exit_record = self.require()
        admission, _, _ = self._cold.require_pair()
        ticket = admission["global_ticket"]
        if ticket is None:
            raise ContractError("ordinary invocation has no global GPU receipt")
        namespace = (
            Path(pwd.getpwuid(os.getuid()).pw_dir)
            / ".local/state/godfieldbots/hpo-gpu-admission-v1"
        )
        directory = namespace / "jobs" / ticket
        global_admission, _ = read_control_record(directory / "admission.json", namespace)
        physical, _ = read_control_record(directory / "termination.json", namespace)
        validate_global_admission(global_admission, ticket)
        validate_global_termination(physical, global_admission)
        if (
            global_admission["host_admission"] != admission
            or physical["exit_record"] != exit_record
        ):
            raise ContractError("global physical receipt differs from SAME cold original")
        return physical


class ColdContinuation(_Capability):
    def __init__(self, issuer: object, control: HostControl, payload: Json) -> None:
        if issuer is not _ISSUER:
            raise ContractError("authenticated cold continuation required")
        self._control, self._payload = control, clone(payload)
        self._finalized_sha256: str | None = None
        self.original_exit = OriginalHostExit(_ISSUER, self)
        self.physical_termination = RustPhysicalTermination(_ISSUER, self, self.original_exit)

    def require_pair(self) -> tuple[Json, Json, Json]:
        self._control._require()
        if self._control._cold is not self or self._control._phase == "training":
            raise ContractError("SAME original cold control capability required")
        if (self._control.operation_sha256 != self._payload["operation_sha256"]
                or str(self._control.operation_path) != self._payload["operation_path"]):
            raise ContractError("cold pair differs from actual CURRENT bound operation")
        admission = validate_local_admission(self._payload["admission"])
        prepared, prepared_sha = read_control_record(
            Path(self._payload["prepared_path"]), self._control.root
        )
        exit_record, exit_sha = read_control_record(
            Path(self._payload["exit_path"]), self._control.root
        )
        validate_prepared(prepared, admission)
        validate_exit(exit_record, admission)
        if (
            prepared_sha != self._payload["prepared_sha256"]
            or exit_sha != self._payload["exit_sha256"]
            or exit_record["prepared_sha256"] != prepared_sha
            or not exit_record["cooperative"]
        ):
            raise ContractError("cold original prepared/exit pair mismatch")
        if time.monotonic() >= deadline_seconds(admission["deadlines"]["finish_bits"]):
            raise TimeoutError("original cold operation finish expired")
        return clone(admission), prepared, exit_record

    def acknowledge_finalizer_output(self, path: Path) -> str:
        """5c MUST cold-validate Source/saved state before issuing this output."""
        self.require_pair()
        value, checksum = read_control_record(path, self._control.root)
        if (
            value.get("schema_version") != 1
            or value.get("kind") != "rust-player-training-completion-v1"
            or value.get("prepared_sha256") != self._payload["prepared_sha256"]
            or value.get("exit_sha256") != self._payload["exit_sha256"]
        ):
            raise ContractError("fixed cold finalizer output/pair mismatch")
        if self._finalized_sha256 is not None and checksum != self._finalized_sha256:
            raise ContractError("competing/different finalizer output")
        response = self._control._request(
            "finalized",
            {
                "prepared_sha256": self._payload["prepared_sha256"],
                "exit_sha256": self._payload["exit_sha256"],
                "completion_path": str(path),
                "completion_sha256": checksum,
            },
            "finalized_ack",
            {"completion_sha256"},
        )
        if response["completion_sha256"] != checksum:
            raise ContractError("finalizer acknowledgement differs")
        self._finalized_sha256 = checksum
        return checksum


class HostControl(_Capability):
    """Fixed host's sole control owner, bound to original PID and thread."""

    def __init__(self, control_fd: int, launch_fd: int, *, phase: str) -> None:
        if phase not in ("host", "cold", "campaign-return"):
            raise ContractError("unknown fixed host phase")
        self.launch = _launch_original(launch_fd)
        self.root = Path(self.launch["record_directory"])
        self._socket = socket.socket(fileno=control_fd)
        if self._socket.family != socket.AF_UNIX or self._socket.type != socket.SOCK_SEQPACKET:
            raise ContractError("private inherited seqpacket control required")
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        os.set_inheritable(control_fd, False)
        os.set_inheritable(launch_fd, False)
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._parent_pid = os.getppid()
        self._admission: HostAdmission | None = None
        self._prepared = False
        self._terminal = False
        self._cold: ColdContinuation | None = None
        self._closed = False
        self._parent_borrowers: set[_ParentBorrower] = set()
        self._campaign_borrowers: set[_ParentBorrower] = set()
        self._campaign_return: Json | None = None
        self._disposition_ack: Json | None = None
        self._next_sequence = 1
        self._server_sequence = 1
        self._stopped = False
        self._phase = phase if phase != "host" else "training"
        self._borrowed = {item["lock_kind"]: clone(item) for item in self.launch["locks"]}
        for item in self._borrowed.values():
            os.set_inheritable(item["fd"], False)
        hello = self._receive(
            deadline_seconds(self.launch["session_deadlines"]["finish_bits"]), initial=True
        )
        closed(hello, _HEADER | {"supervisor_pid", "host_pid", "host_start_ticks", "build"})
        if (
            hello["kind"] != "hello"
            or hello["seq"] != 0
            or hello["supervisor_pid"] != self._parent_pid
            or hello["host_pid"] != self._pid
            or hello["supervision_id"] != self.launch["supervision_id"]
            or hello["build"] != self.launch["binding"]["build"]
        ):
            raise ContractError("original supervisor/host launch identity mismatch")
        self._envelope = {key: hello[key] for key in ("supervision_id", "invocation_id", "nonce")}
        for key in ("invocation_id", "nonce"):
            hex_identity(self._envelope[key], 32)
        start = int(Path(f"/proc/{self._pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        if start != hello["host_start_ticks"]:
            raise ContractError("actual original host start identity differs")
        from gf_training.native_backend import binary_identity

        identity = binary_identity()
        self.native_identity = identity
        self._build = {key: identity["build"][key] for key in BUILD_KEYS}
        if (
            identity["extension_sha256"] != self.launch["binding"]["extension_sha256"]
            or self._build != self.launch["binding"]["build"]
            or identity["python_abi"] != self.launch["binding"]["python_abi"]
        ):
            raise ContractError("actual loaded Python/native host binding differs")
        self._send(
            "ready",
            0,
            {
                "pid": self._pid,
                "start_ticks": start,
                "python_abi": identity["python_abi"],
                "extension_sha256": identity["extension_sha256"],
                "build": self._build,
            },
        )
        self.operation_path = Path(self.launch["operation_path"])
        self.operation: Json = {}
        initial, initial_sha = read_control_record(self.operation_path, self.root)
        if initial_sha != self.launch['operation_sha256']:
            raise ContractError('authenticated INITIAL launch operation changed')
        validate_operation(initial, session_finish=self.launch['session_deadlines']['finish_bits'])
        self._initial_operation = clone(initial)
        self._initial_operation_path = self.operation_path
        if phase == "host":
            self.operation, actual = read_control_record(self.operation_path, self.root)
            if actual != self.launch["operation_sha256"]:
                raise ContractError("initial operation original hash mismatch")
            self._operation_sha256 = actual
            validate_operation(
                self.operation, session_finish=self.launch["session_deadlines"]["finish_bits"]
            )
        if phase in ("cold", "campaign-return"):
            payload = self._receive(
                deadline_seconds(self.launch["session_deadlines"]["finish_bits"])
            )
            keys = {
                "operation_path",
                "operation_sha256",
                "prepared_path",
                "prepared_sha256",
                "exit_path",
                "exit_sha256",
                "admission",
                "locks",
                "deadlines",
            }
            if phase == 'campaign-return':
                keys = (keys - {'prepared_path', 'prepared_sha256'}) | {'failure_path',
                    'failure_sha256', 'physical_path', 'physical_sha256'}
            closed(payload, _HEADER | keys)
            if payload["kind"] != ('cold' if phase == 'cold' else 'campaign_return') or payload["seq"] != 1:
                raise ContractError("exact cold handoff required")
            self._server_sequence = 2
            self.operation_path = Path(payload["operation_path"])
            self.operation, actual = read_control_record(self.operation_path, self.root)
            if (
                actual != payload["operation_sha256"]
                or self.operation["deadlines"] != payload["deadlines"]
            ):
                raise ContractError("cold current operation original differs")
            self._operation_sha256 = actual
            validate_operation(
                self.operation, session_finish=self.launch["session_deadlines"]["finish_bits"]
            )
            self._borrowed = {item["lock_kind"]: clone(item) for item in payload["locks"]}
            for item in self._borrowed.values():
                os.set_inheritable(item["fd"], False)
            if phase == 'cold':
                self._cold = ColdContinuation(_ISSUER, self, {key: payload[key] for key in keys})
                self._cold.require_pair()
                self._send('cold_ready', 1, {'prepared_sha256': payload['prepared_sha256'],
                    'exit_sha256': payload['exit_sha256']})
            else:
                self._campaign_return = {key: payload[key] for key in keys}
                _ = self.campaign_return
                self._send('campaign_return_ready', 1, {key: payload[key]
                    for key in ('failure_sha256', 'exit_sha256', 'physical_sha256')})
                self._next_sequence = 2
        if phase == 'host' and self._initial_operation['operation_type'] == 'campaign-continue':
            _ = self.campaign_bootstrap
            self._phase = 'session'


    def _require(self) -> None:
        if (
            self._closed
            or os.getpid() != self._pid
            or threading.get_ident() != self._thread
            or self._socket.fileno() < 0
            or os.getppid() != self._parent_pid
        ):
            raise ContractError("SAME original host control owner required")

    @property
    def operation_sha256(self) -> str:
        self._require()
        value, actual = read_control_record(self.operation_path, self.root)
        if actual != self._operation_sha256 or value != self.operation:
            raise ContractError("actual CURRENT bound operation original changed")
        return actual

    @property
    def initial_operation(self) -> Json:
        self._require()
        value, actual = read_control_record(self._initial_operation_path, self.root)
        if actual != self.launch['operation_sha256'] or value != self._initial_operation:
            raise ContractError('authenticated INITIAL launch operation changed')
        return clone(value)

    @property
    def campaign_bootstrap(self) -> Json:
        initial = self.initial_operation
        if initial['operation_type'] != 'campaign-continue' or initial['mode'] != 'hpo':
            raise ContractError('authenticated INITIAL campaign purpose required')
        path = Path(initial['config_path'])
        value, actual = read_control_record(path, self.root)
        validate_campaign_bootstrap(value)
        if actual != initial['config_sha256'] or value['supervision_id'] != self.launch['supervision_id']:
            raise ContractError('initial campaign bootstrap identity differs')
        canonical_path = Path(value['campaign_directory']) / 'supervision' / value['supervision_id'] / 'bootstrap.json'
        if path != canonical_path or self.root != canonical_path.parent:
            raise ContractError('canonical original campaign bootstrap path required')
        clock, clock_sha = read_control_record(Path(value['session_clock_path']), self.root)
        validate_campaign_clock(clock)
        if (clock_sha != value['session_clock_sha256']
                or clock['supervision_id'] != self.launch['supervision_id']
                or clock['hard_session_finish_bits'] != self.launch['session_deadlines']['finish_bits']
                or any(clock[key] != anchor for key, anchor in self.launch['clock_binding'].items())):
            raise ContractError('original campaign/session clock binding differs')
        for name, validate in (('loaded_campaign', validate_loaded_campaign), ('seed_settings', validate_seed_settings)):
            data, actual = read_control_record(Path(value[name + '_path']), self.root)
            validate(data)
            if actual != value[name + '_sha256']:
                raise ContractError('initial immutable campaign CONFIG DATA changed')
        item = self._borrowed.get('campaign')
        info = (Path(value['campaign_directory']) / 'lock').lstat()
        if item is None or (info.st_dev, info.st_ino) != (item['device'], item['inode']):
            raise ContractError('actual initial campaign lock identity differs')
        mapped = os.fstat(item['fd'])
        if (mapped.st_dev, mapped.st_ino) != (item['device'], item['inode']):
            raise ContractError('authenticated original campaign descriptor changed')
        return clone(value)

    @property
    def campaign_return(self) -> Json:
        self._require()
        payload = self._campaign_return
        if payload is None or self._phase != 'campaign-return':
            raise ContractError('actual live failed-operation campaign return DATA required')
        _ = self.campaign_bootstrap
        admission = validate_local_admission(payload['admission'])
        if (admission['invocation_id'] == self._envelope['invocation_id']
                or admission['supervision_id'] != self._envelope['supervision_id']
                or payload['operation_sha256'] != self.operation_sha256
                or any(admission[key] != self.operation[key] for key in
                    ('operation_id', 'run_id', 'config_sha256', 'runtime_binding_sha256', 'deadlines'))):
            raise ContractError('old failed training/new control identities differ')
        failure, failure_sha = read_control_record(Path(payload['failure_path']), self.root)
        validate_training_failure(failure, admission, self.operation_sha256)
        exit_value, exit_sha = read_control_record(Path(payload['exit_path']), self.root)
        validate_exit(exit_value, admission)
        physical_path = Path(payload['physical_path'])
        namespace = physical_path.parent.parent.parent
        global_admission, _ = read_control_record(physical_path.parent / 'admission.json', namespace)
        physical, physical_sha = read_control_record(physical_path, namespace)
        validate_global_admission(global_admission, admission['global_ticket'])
        validate_global_termination(physical, global_admission)
        if (failure_sha != payload['failure_sha256'] or exit_sha != payload['exit_sha256']
                or physical_sha != payload['physical_sha256'] or physical['exit_record'] != exit_value
                or global_admission['host_admission'] != admission
                or exit_value['cooperative'] is not False or exit_value['prepared_sha256'] is not None
                or exit_value['wait']['signal'] is not None or exit_value['signals']):
            raise ContractError('failed return original retirement DATA differs')
        if time.monotonic() >= self._finish():
            raise TimeoutError('OLD failed operation finish expired')
        return clone(payload)

    def _borrow_campaign_store(self, owner: Any, directory: Path) -> tuple[Any, _ParentBorrower]:
        self._require()
        if self._campaign_borrowers or directory != Path(self.campaign_bootstrap['campaign_directory']):
            raise ContractError('sole actual campaign store borrower required')
        mapped = self._borrowed['campaign']
        self._require_campaign_ofd(mapped['fd'])
        # Register the concrete actual owner BEFORE dup/wrapping/validation.
        # A failed constructor must not rely on a traceback/GC to retain its OFD.
        token = _ParentBorrower(_ISSUER, self, None, owner, authenticated=False)
        token._campaign_directory = directory
        token.identity = mapped['device'], mapped['inode']
        self._campaign_borrowers.add(token)
        owner._borrower = token
        try:
            token.fd = os.dup(mapped['fd'])
            # The wrapper never owns raw descriptor retirement, including when
            # wrapping itself raises. Only this registered raw owner may close it.
            token.file = os.fdopen(token.fd, 'r+b', closefd=False)
            owner._lock = token.file
            info = os.fstat(token.fd)
            if (info.st_dev, info.st_ino) != token.identity:
                raise ContractError('campaign duplicate differs from original OFD')
            token.authenticated = True
            return token.file, token
        except BaseException as error:
            try:
                owner.close()
            except BaseException as close_error:
                # The live registration retains BOTH the original actual owner
                # and raw/wrapped descriptor even if outer exception data clears.
                vars(error)['cleanup_owner'] = owner
                raise error from close_error
            raise

    @staticmethod
    def _require_campaign_ofd(fd: int) -> None:
        # Linux fdinfo describes THIS open file description. A separate unlocked
        # open of the same inode has no FLOCK entry even while native retains its
        # real alias; inode equality alone cannot detect accidental FD reuse.
        try:
            lines = Path(f'/proc/self/fdinfo/{fd}').read_text().splitlines()
        except OSError as error:
            raise ContractError('original campaign OFD no longer observable') from error
        if not any(line.startswith('lock:') and {'FLOCK', 'WRITE'} <= set(line.split()) for line in lines):
            raise ContractError('original authenticated campaign OFD must retain its exclusive flock')

    def _require_campaign_borrower(self, token: _ParentBorrower, owner: Any) -> None:
        self._require()
        if (type(token) is not _ParentBorrower or token not in self._campaign_borrowers
                or token.control is not self or token.owner is not owner or not token.authenticated
                or token.file is None or token.file.closed
                or token.file.fileno() != token.fd):
            raise ContractError('SAME actual campaign control-store borrower required')
        if token._campaign_directory != owner.directory:
            raise ContractError('original campaign borrower pathname changed')
        current = (owner.directory / 'lock').lstat()
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != token.identity:
            raise ContractError('original campaign lock pathname changed')
        self._require_campaign_ofd(token.fd)
        info = os.fstat(token.fd)
        mapped = self._borrowed.get('campaign')
        if mapped is None or (info.st_dev, info.st_ino) != token.identity or token.identity != (mapped['device'], mapped['inode']):
            raise ContractError('campaign borrowed original descriptor changed')

    def _close_campaign_borrower(self, token: _ParentBorrower, owner: Any) -> None:
        self._require()
        if (type(token) is not _ParentBorrower or token not in self._campaign_borrowers
                or token.control is not self or token.owner is not owner):
            raise ContractError('SAME actual campaign borrower close required')
        if token._campaign_raw_close_uncertain:
            # A failed close may already have retired this FD lifetime. Even a
            # SAME-number/SAME-inode/SAME-OFD replacement belongs to another
            # actual owner. No numeric probe/retry can reconstruct our ownership.
            raise ContractError('uncertain original raw campaign close requires actual host exit')
        if token.authenticated and token.file is not None and not token.file.closed:
            self._require_campaign_borrower(token, owner)
        if token.file is not None:
            # closefd=False: this is the retained actual wrapper, not retirement
            # of the raw FD. Retrying its own close remains safe after an error.
            token.file.close()
            if not token.file.closed:
                raise ContractError('campaign wrapper close unconfirmed')
        if token.fd >= 0:
            # Known ownership BEFORE the FIRST raw-close attempt follows from
            # dup + closefd=False. Mark uncertainty BEFORE entering fallible I/O.
            token._campaign_raw_close_uncertain = True
            os.close(token.fd)
            token.fd = -1
            token._campaign_raw_close_uncertain = False
        self._campaign_borrowers.remove(token)

    def training_failed(self, admission: HostAdmission, reason: str) -> str:
        admission.require()
        if self._prepared or self._terminal or self._phase != 'training':
            raise ContractError('mutually exclusive actual training terminal arm required')
        value = dict(schema_version=1, kind='rust-player-training-failure-v1',
            **self._envelope, operation_sha256=self.operation_sha256,
            admission_sha256=digest(admission._record), failure_class='application_error', reason=reason)
        validate_training_failure(value, admission._record, self.operation_sha256)
        path = self.root / ('failure-' + self._envelope['invocation_id'] + '.json')
        checksum = write_control_record(path, value)
        response = self._request('training_failed', {'failure_path': str(path),
            'failure_sha256': checksum}, 'training_failed_ack', {'failure_sha256'})
        if response['failure_sha256'] != checksum:
            raise ContractError('training failure ACK differs')
        self._terminal = True
        return checksum

    def campaign_disposition(self, path: Path) -> Json:
        self._require()
        if self._disposition_ack is not None:
            raise ContractError('terminal disposition already acknowledged')
        bootstrap = self.campaign_bootstrap
        value, checksum = read_control_record(path, Path(bootstrap['campaign_directory']))
        validate_campaign_disposition(value)
        if self._phase == 'campaign-return':
            _ = self.campaign_return
        elif self._phase != 'cold' or self.cold._finalized_sha256 is None:
            raise ContractError('actual failed return or finalized incomplete Cold required')
        response = self._request('campaign_disposition', {'disposition_path': str(path),
            'disposition_sha256': checksum}, 'campaign_disposition_ack',
            {'disposition_sha256', 'ack_path', 'ack_sha256'})
        ack_path = self.root / ('disposition-ack-' + value['invocation_id'] + '.json')
        if response['ack_path'] != str(ack_path):
            raise ContractError('canonical native disposition ACK path differs')
        ack, actual = read_control_record(ack_path, self.root)
        validate_campaign_disposition_ack(ack)
        if (actual != response['ack_sha256'] or response['disposition_sha256'] != checksum
                or ack['disposition_sha256'] != checksum or ack['evidence_sha256'] != digest(value['evidence'])
                or ack['control_invocation_id'] != self._envelope['invocation_id']
                or ack['invocation_id'] != value['invocation_id']
                or ack['supervision_id'] != self.launch['supervision_id']
                or ack['operation_sha256'] != self.operation_sha256):
            raise ContractError('actual native disposition ACK identity differs')
        self._disposition_ack = clone(ack)
        return clone(ack)

    def control_complete(self, path: Path) -> str:
        bootstrap = self.campaign_bootstrap
        if (self._phase != 'session' or self._admission is not None
                or 'parent' in self._borrowed or self._parent_borrowers
                or (self.operation['operation_type'] != 'campaign-continue' and self._disposition_ack is None)):
            raise ContractError('actual parent-free terminal campaign Session required')
        value, checksum = read_control_record(path, self.root)
        validate_control_completion(value)
        if (value['supervision_id'] != self.launch['supervision_id']
                or value['operation_sha256'] != self.operation_sha256
                or value['session_clock_sha256'] != bootstrap['session_clock_sha256']):
            raise ContractError('terminal control result binding differs')
        response = self._request('control_complete', {'operation_sha256': self.operation_sha256,
            'result_path': str(path), 'result_sha256': checksum}, 'control_completed',
            {'operation_sha256', 'result_path', 'result_sha256'})
        if any(response[key] != expected for key, expected in {"operation_sha256": self.operation_sha256,
                "result_path": str(path), "result_sha256": checksum}.items()):
            raise ContractError('terminal control ACK differs')
        self._terminal = True
        return checksum

    def _finish(self) -> float:
        if self._phase == "session":
            return deadline_seconds(self.launch["session_deadlines"]["finish_bits"])
        return deadline_seconds(self.operation["deadlines"]["finish_bits"])

    def _send(self, kind: str, sequence: int, payload: Json, *, fd: int | None = None) -> None:
        frame = {"v": 1, "kind": kind, **self._envelope, "seq": sequence, **payload}
        data = canonical(frame).encode()
        if len(data) > _LIMIT:
            raise ContractError("control packet exceeds compact bound")
        ancillary = (
            [] if fd is None else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
        )
        if self._socket.sendmsg(
            [data], ancillary, socket.MSG_DONTWAIT | socket.MSG_NOSIGNAL
        ) != len(data):
            raise ContractError("partial control packet")

    def _receive(self, deadline: float, *, initial: bool = False) -> Json:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("original control deadline expired")
        self._socket.settimeout(remaining)
        data, ancillary, flags, _ = self._socket.recvmsg(
            _LIMIT, socket.CMSG_SPACE(12) + socket.CMSG_SPACE(16), socket.MSG_CMSG_CLOEXEC
        )
        descriptors: list[int] = []
        credentials = []
        unknown = False
        for level, kind, body in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                values = array.array("i")
                values.frombytes(body[: len(body) - len(body) % values.itemsize])
                descriptors.extend(values)
            elif level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS and len(body) == 12:
                credentials.append(struct.unpack("3i", body))
            else:
                unknown = True
        try:
            if (
                not data
                or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
                or unknown
                or descriptors
                or len(credentials) != 1
                or credentials[0][:2] != (self._parent_pid, os.getuid())
            ):
                raise ContractError("control EOF/truncation/FD/original kernel credential failure")
            value = json.loads(data)
            if type(value) is not dict or value.get("v") != 1:
                raise ContractError("invalid control packet")
            if not initial and any(
                value.get(key) != expected for key, expected in self._envelope.items()
            ):
                raise ContractError("original control nonce/invocation differs")
            integer(value.get("seq"))
            return value
        finally:
            for descriptor in descriptors:
                os.close(descriptor)

    def _request(
        self, kind: str, payload: Json, reply: str, keys: set[str], *, fd: int | None = None
    ) -> Json:
        self._require()
        if self._terminal:
            raise ContractError("accepted terminal host may only exit")
        sequence = self._next_sequence
        self._next_sequence += 1
        self._send(kind, sequence, payload, fd=fd)
        while True:
            response = self._receive(self._finish())
            if response["kind"] == "stop":
                self._accept_stop(response)
                continue
            closed(response, _HEADER | keys)
            if response["kind"] != reply or response["seq"] != sequence:
                raise ContractError("out-of-order control acknowledgement")
            return response

    def _accept_stop(self, frame: Json) -> None:
        closed(frame, _HEADER | {"reason"})
        text(frame["reason"])
        if frame["seq"] != self._server_sequence:
            raise ContractError("out-of-order original supervisor stop")
        self._server_sequence += 1
        self._stopped = True
        self._send("stop_ack", frame["seq"], {})

    def poll_stop(self) -> bool:
        self._require()
        import select

        if select.select([self._socket], [], [], 0)[0]:
            frame = self._receive(self._finish())
            if frame["kind"] != "stop":
                raise ContractError("unexpected supervisor packet between requests")
            self._accept_stop(frame)
        return self._stopped

    def transfer_lock(self, kind: str, fd: int) -> Json:
        """After ACK, caller must detach its legacy owner WITHOUT LOCK_UN."""
        if kind not in ("parent", "campaign") or kind in self._borrowed:
            raise ContractError("duplicate/unknown lock custody")
        info = os.fstat(fd)
        response = self._request(
            "lock",
            {"lock_kind": kind, "device": info.st_dev, "inode": info.st_ino},
            "lock_ack",
            {"lock_kind", "device", "inode"},
            fd=fd,
        )
        if response["lock_kind"] != kind or (response["device"], response["inode"]) != (
            info.st_dev,
            info.st_ino,
        ):
            raise ContractError("actual lock acknowledgement differs")
        return {key: response[key] for key in ("lock_kind", "device", "inode")}

    def _register_parent_borrower(self, file: Any, owner: Any, *, inherited: bool) -> _ParentBorrower:
        self._require()
        info = os.fstat(file.fileno())
        path = Path(self.operation["parent_directory"]) / ".lock"
        original = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or not stat.S_ISREG(original.st_mode)
                or (info.st_dev, info.st_ino) != (original.st_dev, original.st_ino)
                or any(item.owner is owner for item in self._parent_borrowers)):
            raise ContractError("original parent borrower FD/path required")
        if inherited:
            mapped = self._borrowed.get("parent")
            if mapped is None or (mapped["device"], mapped["inode"]) != (info.st_dev, info.st_ino):
                raise ContractError("authenticated inherited parent mapping required")
            held = os.fstat(mapped["fd"])
            if (held.st_dev, held.st_ino) != (info.st_dev, info.st_ino):
                raise ContractError("original inherited parent descriptor changed")
        token = _ParentBorrower(_ISSUER, self, file, owner, authenticated=inherited)
        self._parent_borrowers.add(token)
        return token

    def _require_parent_borrower(self, token: _ParentBorrower, owner: Any) -> None:
        self._require()
        if (type(token) is not _ParentBorrower or token not in self._parent_borrowers
                or token.control is not self or token.owner is not owner
                or not token.authenticated or token.file.closed or token.file.fileno() != token.fd):
            raise ContractError("original authenticated parent mutation borrower required")
        info = os.fstat(token.fd)
        if (info.st_dev, info.st_ino) != token.identity:
            raise ContractError("original parent borrower descriptor changed")

    def _confirm_parent_borrower(self, token: _ParentBorrower, owner: Any) -> None:
        self._require()
        if token not in self._parent_borrowers or token.control is not self or token.owner is not owner:
            raise ContractError("SAME original transfer borrower required")
        token.authenticated = True
        self._require_parent_borrower(token, owner)

    def _close_parent_borrower(self, token: _ParentBorrower, owner: Any) -> None:
        self._require()
        if (type(token) is not _ParentBorrower or token not in self._parent_borrowers
                or token.control is not self or token.owner is not owner):
            raise ContractError("SAME actual parent borrower close required")
        # The original high-level FD owner closes itself. A raised/uncertain close
        # retains both registration and actual owner; never close a guessed FD.
        token.file.close()
        if not token.file.closed:
            raise ContractError("original parent borrower close unconfirmed")
        self._parent_borrowers.remove(token)

    def bind_operation(self, path: Path) -> None:
        if self._parent_borrowers:
            raise ContractError('actual parent mutation borrowers remain live or uncertain')
        value, checksum = read_control_record(path, self.root)
        validate_operation(value, session_finish=self.launch["session_deadlines"]["finish_bits"])
        payload = {
            key: value[key]
            for key in (
                "operation_id",
                "run_id",
                "config_sha256",
                "runtime_binding_sha256",
                "deadlines",
            )
        }
        response = self._request(
            "bind_operation",
            {"operation_path": str(path), "operation_sha256": checksum, **payload},
            "operation_bound",
            {"operation_id", "operation_sha256"},
        )
        if (
            response["operation_id"] != value["operation_id"]
            or response["operation_sha256"] != checksum
        ):
            raise ContractError("operation bind acknowledgement differs")
        self.operation, self.operation_path = value, Path(path)
        self._operation_sha256 = checksum
        self._phase = "training"
        self._stopped = False
        self._admission = None
        self._prepared = False
        self._cold = None
        self._campaign_return = None
        self._disposition_ack = None

    def admit(self, budget: Json, producer: Json) -> HostAdmission:
        validate_producer(producer)
        if self._admission is not None:
            raise ContractError("host already owns actual admission")
        payload = {
            key: self.operation[key]
            for key in ("operation_id", "run_id", "config_sha256", "runtime_binding_sha256")
        }
        response = self._request(
            "admit",
            {
                **payload,
                "context": self.operation["expected_context"],
                "budget": budget,
                "producer": producer,
            },
            "admitted",
            {"admission_path", "admission_sha256", "global_ticket"},
        )
        path = Path(response["admission_path"])
        record, actual = read_control_record(path, self.root)
        validate_local_admission(record)
        if (
            actual != response["admission_sha256"]
            or record["global_ticket"] != response["global_ticket"]
            or any(record[key] != value for key, value in self._envelope.items())
        ):
            raise ContractError("actual admission record/ACK differs")
        self._admission = HostAdmission(_ISSUER, self, record, path)
        return self._admission

    def prepare(self, admission: HostAdmission, path: Path) -> PreparedResult:
        admission.require()
        value, checksum = read_control_record(path, self.root)
        validate_prepared(value, admission._record)
        response = self._request(
            "prepared",
            {"prepared_path": str(path), "prepared_sha256": checksum},
            "prepared_ack",
            {"prepared_sha256"},
        )
        if response["prepared_sha256"] != checksum:
            raise ContractError("prepared acknowledgement differs")
        self._prepared = True
        return PreparedResult(_ISSUER, admission, path, checksum)

    @property
    def cold(self) -> ColdContinuation:
        self._require()
        if self._cold is None or self._phase == "training":
            raise ContractError("actual cold continuation required")
        return self._cold

    def release_parent(self) -> None:
        if self._parent_borrowers:
            raise ContractError("actual parent mutation borrowers remain live or uncertain")
        failure_release = self._phase == 'campaign-return' and self._campaign_return is not None and self._disposition_ack is not None
        if not failure_release and (self._phase != 'cold' or self.cold._finalized_sha256 is None):
            raise ContractError('parent release requires acknowledged actual terminal output')
        item = self._borrowed.get("parent")
        if item is None:
            raise ContractError("no inherited actual parent lock")
        response = self._request(
            "release_parent",
            {"device": item["device"], "inode": item["inode"]},
            "parent_released",
            {"device", "inode"},
        )
        if (response["device"], response["inode"]) != (item["device"], item["inode"]):
            raise ContractError("parent release acknowledgement differs")
        os.close(item["fd"])
        del self._borrowed["parent"]
        self._phase = "session"

    def close(self) -> None:
        self._require()
        if self._campaign_borrowers:
            raise ContractError("actual campaign control-store borrower remains live or uncertain")
        if self._parent_borrowers:
            raise ContractError("actual parent mutation borrowers remain live or uncertain")
        self._closed = True
        self._socket.close()  # EOF requests cleanup; this never claims process exit.
        for item in self._borrowed.values():
            os.close(item["fd"])  # close-only, NEVER LOCK_UN
        self._borrowed.clear()


def runtime_producer(companion: Any, *, sampling_runtime_sha256: str) -> Json:
    """Actual loaded source/build identity; sampling metadata is computed before GPU by caller."""
    from gf_training.native_backend import binary_identity

    identity = binary_identity()
    packages = (
        "gf_common",
        "gf_models",
        "gf_training",
        "gf_cli",
        "gf_evaluation",
        "gf_simulator_client",
        "godfieldbots",
    )
    roots = {}
    for package in packages:
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None or Path(spec.origin).name != "__init__.py":
            raise ContractError("actual runtime package sources unavailable")
        roots[package] = Path(spec.origin).resolve().parent

    def source_digest(names: tuple[str, ...]) -> str:
        checksum = hashlib.sha256()
        for package in names:
            root = roots[package]
            for path in sorted(root.rglob("*.py")):
                checksum.update(
                    f"{package}/{path.relative_to(root).as_posix()}".encode()
                    + b"\0"
                    + path.read_bytes()
                    + b"\0"
                )
        return checksum.hexdigest()

    result = {
        "schema_version": 1,
        "kind": "rust-player-producer-v1",
        "python_source_digest": source_digest(("gf_common", "gf_models", "gf_training")),
        "python_runtime_digest": source_digest(tuple(sorted(packages))),
        "extension_sha256": identity["extension_sha256"],
        "build": {key: identity["build"][key] for key in BUILD_KEYS},
        "supervisor_sha256": companion.manifest["executable_sha256"],
        "sampling_runtime_sha256": sampling_runtime_sha256,
    }
    return validate_producer(result)
