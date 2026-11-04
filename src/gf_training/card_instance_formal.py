"""Owned formal declaration, normal-stop and original-parent controls.

This module never starts another trainer, creates policy inputs or qualifies a
Tree Dataset. Formal controls are additional evidence around the SAME coordinator.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.resources
import json
import os
import stat
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, Self

from gf_common.card_instance_formal import (
    ALGORITHM_SHA256,
    DECLARATION_FIELDS,
    NATIVE_FORMAL_MARKER,
    RECIPE_ID,
    RECIPE_SHA256,
    RUN_VARIANT,
    checked_config,
    declaration_reference,
    planned_execution,
    recipe_bytes,
    stop_receipt,
    validate_declaration,
    validate_formal_state,
)
from gf_common.domain import ContractError, Json, canonical, clone, closed, digest
from gf_common.owned_completion import _close_owned_descriptor, _invalidate_owned_completion
from gf_models.structured_sealed import SealedScan


def load_recipe() -> Json:
    asset = importlib.resources.files('gf_common').joinpath('contracts/card-instance-formal-ppo-v1.json')
    with asset.open('rb') as source:
        raw = source.read(65537)
    if len(raw) > 65536:
        raise ContractError('bounded formal recipe asset required')
    return recipe_bytes(raw)


def verify_kernel_ast(recipe: Json) -> None:
    roots = Path(__file__).resolve().parent.parent
    for name, expected in recipe['algorithm_recipe']['kernel_sha256'].items():
        pieces = name.split('.')
        if pieces[:2] not in (['gf_training', 'collection'], ['gf_training', 'ppo'], ['gf_training', 'ppo_math'],
                             ['gf_training', 'ppo_run'], ['gf_training', 'structured_numeric']):
            raise ContractError('unknown formal AST kernel location')
        node: Any = ast.parse((roots / pieces[0] / (pieces[1] + '.py')).read_bytes())
        for identifier in pieces[2:]:
            matches = [child for child in node.body if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == identifier]
            if len(matches) != 1:
                raise ContractError('formal frozen kernel cannot be uniquely located')
            node = matches[0]
        actual = hashlib.sha256(ast.dump(node, annotate_fields=True, include_attributes=False).encode()).hexdigest()
        if actual != expected:
            raise ContractError('formal frozen algorithm kernel differs: ' + name)


def formal_preflight(loaded: Any) -> Json:
    if sys.version_info[:2] != (3, 14) or not sys._is_gil_enabled():
        raise ContractError('formal frozen AST/runtime requires Python3.14 with GIL enabled')
    recipe = load_recipe()
    expected = checked_config(loaded.effective, recipe)
    if canonical(loaded.effective) != canonical(asdict(loaded.config)) or digest(loaded.effective) != loaded.sha256:
        raise ContractError('formal loaded effective Config/hash differs from original caller')
    verify_kernel_ast(recipe)
    from gf_models.structured_artifacts import ADAM_RECIPE
    if canonical(ADAM_RECIPE) != canonical(recipe['algorithm_recipe']['adam']):
        raise ContractError('formal planned named Adam options differ from frozen algorithm')
    from gf_common.card_instance import action_config_v3
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.structured_binding import input_binding_game_v3
    from gf_models.structured_config import model_config_v3
    from gf_models.structured_sealed import static_semantics_v3
    actions = action_config_v3(expected['actions'])
    model = model_config_v3({key: value for key, value in expected['model'].items() if key not in ('checkpoint', 'inference', 'cuda_visible_devices')}, layout=layout)
    if (canonical(input_binding_game_v3(actions.action_contract, card_instance_rules())) != canonical(recipe['input_binding'])
            or canonical(static_semantics_v3(model, layout=layout, action_contract=actions.action_contract, rules=card_instance_rules()))
               != canonical(recipe['static_semantics'][model.model_id])):
        raise ContractError('formal planned input/static semantics differ from literal')
    return recipe


def verify_actual_optimizer_metadata(resume: Json) -> None:
    """Called only on the actual retained paired artifact, before G0/commit."""
    if canonical(resume.get('optimizer_recipe')) != canonical(load_recipe()['algorithm_recipe']['adam']):
        raise ContractError('formal actual named Adam metadata differs from literal algorithm')


def require_native_formal_controls(native: Any) -> None:
    info = dict(native.build_info())
    if info.get('card_instance_formal_controls') != NATIVE_FORMAL_MARKER:
        raise ContractError('Rust formal controls unavailable until reviewed Run3/declaration1/stop1 adoption; no fallback')


def read_formal_control(directory: Path, state: Json, scan: SealedScan, recipe: Json | None = None) -> Json:
    recipe = load_recipe() if recipe is None else recipe
    validate_formal_state(state)
    reference = declaration_reference(state['formal_declaration'])
    declaration = scan.object(directory / reference['relative_path'], reference['sha256'])
    asset = importlib.resources.files('gf_common').joinpath('contracts/card-instance-formal-ppo-v1.json')
    with scan.file(Path(str(asset)), RECIPE_SHA256):
        pass
    for entry in state['config_inputs']:
        closed(entry, {'kind', 'path', 'sha256'})
        path = Path(entry['path'])
        if not path.is_absolute():
            raise ContractError('formal original input paths must be absolute')
        with scan.file(path, entry['sha256']):
            pass
    with scan.file(Path(state['effective_config']['server']['config']), recipe['server_config_sha256']):
        pass
    validate_declaration(declaration, state, recipe)
    return declaration


class FormalCompletion:
    """One retained formal run.json inode, revocable across outer run-lock exit."""
    def __init__(self) -> None:
        self.directory: Path | None = None
        self.directory_fd: int | None = None
        self.fd: int | None = None
        self.snapshot: tuple[int, ...] | None = None
        self._completion_fd_identity: tuple[int, int] | None = None
        self._directory_identity: tuple[int, int] | None = None
        self._uncertain_releases: list[Any] = []
        self.exposed = False
        self.revoked = False
        self.committed = False
        self.started = False
        self.deadline = 0.
        self.max_storage = 0

    @staticmethod
    def _identity(info: os.stat_result) -> tuple[int, ...]:
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def _namespace(self) -> None:
        assert self.directory is not None and self.directory_fd is not None
        info = self.directory.lstat()
        held = os.fstat(self.directory_fd)
        if (not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != self._directory_identity
                or (held.st_dev, held.st_ino) != self._directory_identity):
            raise ContractError('formal completion original directory namespace replaced')

    def _verify(self) -> None:
        assert self.directory is not None and self.snapshot is not None and self.fd is not None
        self._namespace()
        current = (self.directory / 'run.json').lstat()
        held = os.fstat(self.fd)
        if self._identity(current) != self.snapshot or self._identity(held) != self.snapshot:
            raise ContractError('formal completion held inode/current path changed')

    def publish(self, directory: Path, state: Json, *, deadline: float, max_owned_bytes: int) -> None:
        if self.started or self._uncertain_releases:
            raise ContractError('formal complete publication is one-use; later work is cleanup-only')
        if time.monotonic() >= deadline:
            raise TimeoutError('formal completion original wall deadline')
        self.started = True
        self.deadline = deadline
        self.max_storage = state['effective_config']['training']['max_owned_bytes']
        self.directory = directory
        check = SealedScan(max_owned_bytes=max_owned_bytes, deadline=deadline)
        previous = check.object(directory / 'run.json')
        for key in ('run_id', 'identity', 'formal_declaration', 'generation_sha256', 'updates', 'optimizer_steps', 'accepted_decisions', 'config_sha256'):
            if canonical(previous[key]) != canonical(state[key]):
                raise ContractError('formal complete differs from original committed control: ' + key)
        check.finalize()
        self.directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(self.directory_fd)
        self._directory_identity = (info.st_dev, info.st_ino)
        self._namespace()
        name = 'run.json.formal-complete.tmp'
        self.fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.directory_fd)
        initial = os.fstat(self.fd)
        if not stat.S_ISREG(initial.st_mode):
            raise ContractError('formal completion requires its own regular temporary inode')
        self._completion_fd_identity = (initial.st_dev, initial.st_ino)
        self.snapshot = self._identity(initial)
        raw = (canonical(state) + '\n').encode()
        check.charge(32 * len(raw) + 65536)
        offset = 0
        while offset < len(raw):
            amount = os.write(self.fd, raw[offset:])
            if amount <= 0:
                raise OSError('formal completion write did not advance')
            offset += amount
        os.fsync(self.fd)
        check.finalize()
        self._namespace()
        self.exposed = True  # Rename may take effect before reporting an error.
        os.replace(name, 'run.json', src_dir_fd=self.directory_fd, dst_dir_fd=self.directory_fd)
        self.snapshot = self._identity(os.fstat(self.fd))
        self._verify()
        os.lseek(self.fd, 0, os.SEEK_SET)
        if os.read(self.fd, len(raw) + 1) != raw or time.monotonic() >= deadline:
            raise ContractError('formal complete bytes/deadline changed after atomic publication')
        self._verify()

    def _release(self) -> None:
        if self.directory_fd is not None:
            descriptor, self.directory_fd = self.directory_fd, None
            _close_owned_descriptor(self, descriptor, self._directory_identity, 'formal-directory')
        if self.fd is not None:
            descriptor, self.fd = self.fd, None
            _close_owned_descriptor(self, descriptor, self._completion_fd_identity, 'formal-completion')
        if self._uncertain_releases:
            raise ContractError('formal terminal release is uncertain; stale descriptor numbers are quarantined')

    def revoke(self) -> None:
        if self.exposed and not self.revoked:
            assert self.directory is not None
            _invalidate_owned_completion(self, self.directory / 'run.json')
            self.revoked = True
        self._release()

    def _final_guard(self) -> None:
        from gf_training.ppo_run import owned_bytes
        assert self.directory is not None and self.snapshot is not None
        if (time.monotonic() >= self.deadline or owned_bytes(self.directory, self.deadline) > self.max_storage
                or self._identity((self.directory / 'run.json').lstat()) != self.snapshot):
            raise ContractError('formal outer completion wall/storage/original inode guard failed')

    def finish(self) -> None:
        if self.exposed:
            self._verify()
            self._final_guard()
        self._release()
        if self.exposed:
            # Close is reportable too. Snapshot ownership permits fresh known-
            # inode invalidation if this last guard fails after successful close.
            self._final_guard()
        self.committed = self.exposed and not self.revoked


class FormalControl:
    """Closed explicit request; serializable references never construct this owner."""
    def __init__(self, loaded: Any) -> None:
        self.loaded = loaded
        self.recipe: Json | None = None
        self.declaration: Json | None = None
        self.reference: Json | None = None
        self.originals: SealedScan | None = None
        self.completion = FormalCompletion()
        self.directory: Path | None = None
        self.ownership: Any = None

    def preflight(self, expected_loaded: Any = None) -> None:
        if expected_loaded is not None and expected_loaded is not self.loaded:
            raise ContractError('formal control requires the SAME original loaded caller')
        self.recipe = formal_preflight(self.loaded)
        if self.loaded.config.preprocessing.backend == 'rust':
            from gf_training.native_backend import load_native
            require_native_formal_controls(load_native())

    def require_lock(self) -> None:
        from gf_training.lineage import _OWNED_LOCKS
        if self.directory is None or self.ownership is None or _OWNED_LOCKS.get(self.directory) != self.ownership.token:
            raise ContractError('formal control requires the SAME actual coordinator lock')

    def attach_identity(self, directory: Path, identity: Json, execution: Json, *, mode: str, deadline: float, ownership: Any) -> None:
        from gf_training.lineage import RunOwnership
        if type(ownership) is not RunOwnership or ownership.directory != directory or mode not in ('fresh', 'resume'):
            raise ContractError('closed formal coordinator lock/mode required')
        self.directory, self.ownership = directory, ownership
        self.require_lock()
        assert self.recipe is not None
        if canonical(execution) != canonical(planned_execution(self.loaded.effective)):
            raise ContractError('formal planned execution differs from exact literal bindings')
        self.directory = directory
        self.originals = SealedScan(max_owned_bytes=self.loaded.config.training.max_owned_bytes, deadline=deadline)
        if mode == 'fresh':
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.name != '.lock':
                        raise ContractError('formal declaration cannot backfill an existing parent/output')
            declaration = {'schema_version': 1, 'kind': 'card-instance-formal-parent-declaration',
                'recipe_id': RECIPE_ID, 'recipe_sha256': RECIPE_SHA256, 'algorithm_recipe_sha256': ALGORITHM_SHA256,
                'seed_slot': self.loaded.config.training.seeds[0], 'config_sha256': self.loaded.sha256,
                'config_inputs_sha256': digest(self.loaded.inputs), 'input_binding': self.recipe['input_binding'],
                'execution_identity_sha256': digest(execution), 'identity_sha256': digest(identity)}
            closed(declaration, DECLARATION_FIELDS)
            from gf_training.structured_lineage import _ChainScan, _write
            scan = _ChainScan(directory=directory, max_owned_bytes=self.loaded.config.training.max_owned_bytes, deadline=deadline)
            sha = _write(directory / 'formal-declaration.json', declaration, scan=scan, immutable=True)
            self.reference = {'relative_path': 'formal-declaration.json', 'sha256': sha, 'recipe_id': RECIPE_ID,
                'recipe_sha256': RECIPE_SHA256, 'algorithm_recipe_sha256': ALGORITHM_SHA256,
                'seed_slot': declaration['seed_slot']}
        else:
            state = self.originals.object(directory / 'run.json')
            if state.get('run_variant') != RUN_VARIANT:
                raise ContractError('ordinary/short records cannot resume through formal entry')
            self.reference = declaration_reference(state['formal_declaration'])
            declaration = self.originals.object(directory / self.reference['relative_path'], self.reference['sha256'])
            # Current base identity must equal the immutable original declaration,
            # before adding its reference and recomputing ordinary resume identity.
            if declaration['identity_sha256'] != digest(identity) or declaration['execution_identity_sha256'] != digest(execution):
                raise ContractError('formal resume base code/runtime/execution differs')
        self.declaration = self.originals.object(directory / self.reference['relative_path'], self.reference['sha256'])
        # Mutable run control is owned by the existing coordinator scan. Keep
        # only this immutable declaration in the outer formal input owner.
        self.originals.files.pop(directory / 'run.json', None)
        for entry in self.loaded.inputs:
            closed(entry, {'kind', 'path', 'sha256'})
            path = Path(entry['path'])
            if not path.is_absolute():
                raise ContractError('formal original configuration input paths must be absolute')
            with self.originals.file(path, entry['sha256']):
                pass
        with self.originals.file(Path(self.loaded.effective['server']['config']), self.recipe['server_config_sha256']):
            pass
        asset = importlib.resources.files('gf_common').joinpath('contracts/card-instance-formal-ppo-v1.json')
        with self.originals.file(Path(str(asset)), RECIPE_SHA256):
            pass
        # Keep the original code files represented by the actual coordinator's
        # pre-formal identity. No path in this map becomes a policy feature.
        from importlib.util import find_spec
        packages = {'godfieldbots', 'gf_common', 'gf_models', 'gf_training', 'gf_cli', 'gf_evaluation', 'gf_simulator_client'}
        roots = {}
        for relative, sha in identity.get('module_sha256', {}).items():
            parts = Path(relative).parts
            if not parts or parts[0] not in packages or '..' in parts or not relative.endswith('.py'):
                raise ContractError('formal code identity contains a foreign source path')
            if parts[0] not in roots:
                spec = find_spec(parts[0])
                if spec is None or spec.origin is None:
                    raise ContractError('formal original package source unavailable')
                roots[parts[0]] = Path(spec.origin).resolve().parent
            with self.originals.file(roots[parts[0]].joinpath(*parts[1:]), sha):
                pass
        identity['formal_parent'] = clone(self.reference)
        self.originals.finalize()

    def initialize_state(self, state: Json) -> None:
        self.require_lock()
        assert self.reference is not None and self.recipe is not None and self.declaration is not None
        state.update(run_variant=RUN_VARIANT, formal_declaration=clone(self.reference), formal_stop=None)
        validate_declaration(self.declaration, state, self.recipe)

    def observe_stop(self, state: Json, invocation_unix: float, invocation_monotonic: float, observed: float, usage: int) -> bool:
        self.require_lock()
        receipt = stop_receipt(state, invocation_unix=invocation_unix, invocation_monotonic=invocation_monotonic,
            observed_monotonic=observed, owned_bytes=usage)
        if receipt is not None:
            state['formal_stop'] = receipt
            return True
        return False

    def persist(self, path: Path, state: Json, *, deadline: float, max_owned_bytes: int) -> bool:
        self.require_lock()
        if self.originals is not None:
            self.originals.deadline = deadline
        if state['status'] == 'complete':
            assert self.recipe is not None and self.declaration is not None and self.originals is not None
            validate_formal_state(state)
            validate_declaration(self.declaration, state, self.recipe)
            self.originals.deadline = deadline
            self.originals.finalize()
            self.completion.publish(path.parent, state, deadline=deadline, max_owned_bytes=max_owned_bytes)
            return True
        if self.completion.exposed:
            # Invalidate FIRST, then leave the owned completion visibly invalid.
            # Parsing/replacing that invalid JSON could hide the initiating late
            # failure or overwrite a foreign current path. Outer error/logs keep
            # the failure; this file can no longer qualify any formal Source.
            self.completion.revoke()
            return True

        return False

    def finish(self) -> None:
        if self.originals is not None:
            self.originals.finalize()
        self.completion.finish()

    def fail(self, initiating: BaseException) -> None:
        try:
            self.completion.revoke()
        except BaseException as cleanup:  # noqa: BLE001 - primary failure and uncertain capability stay reachable
            vars(initiating)['_formal_completion_cleanup_owner'] = self
            try:
                initiating.add_note('formal completion retirement unconfirmed: ' + str(cleanup))
            except BaseException:  # noqa: BLE001, S110 - notes cannot replace the initiating failure
                pass


class FormalGameParent:
    """Complete original formal parent owner; not Dataset or online permission."""
    def __init__(self, directory: Path, *, max_owned_bytes: int, deadline: float, ownership: Any = None) -> None:
        self.directory = directory.resolve()
        self.limit, self.deadline = max_owned_bytes, deadline
        self.borrowed = ownership
        self._scope = ExitStack()
        self._inputs_scope = ExitStack()
        self._inputs_retired = False
        self._owns_lock = False
        self._lock_retired = False
        self._lock_retirement_failure: BaseException | None = None
        self._external_bytes = 0
        self.scan: Any = None
        self._state_bytes: bytes | None = None
        self._records_bytes: bytes | None = None
        self._active = False
        self._entered = False
        self._closed = False
        self._poisoned = False
        self._failure: BaseException | None = None
        self._token: str | None = None
        self._native_inputs: Any = None
        self._native_scan_base = 0

    @property
    def canonical_control_sizes(self) -> tuple[int, int]:
        """Guarded protected byte lengths only; no parse, copies or permission."""
        self.require_active()
        assert self._state_bytes is not None and self._records_bytes is not None
        return len(self._state_bytes), len(self._records_bytes)

    @property
    def state(self) -> Json | None:
        return None if self._state_bytes is None else json.loads(self._state_bytes)

    @property
    def records(self) -> list[Json] | None:
        return None if self._records_bytes is None else json.loads(self._records_bytes)

    def _poison(self, error: BaseException) -> None:
        self._poisoned = True
        if self._failure is None:
            self._failure = error

    @staticmethod
    def _note(primary: BaseException, label: str, secondary: BaseException) -> None:
        try:
            primary.add_note(label+str(secondary))
        except BaseException:  # noqa: BLE001, S110 - diagnostics cannot replace first error or cleanup ownership
            pass

    def __enter__(self) -> Self:
        if self._entered:
            error = ContractError('formal parent owner has exactly one lifetime; reentry refused')
            self._poison(error)
            raise error
        self._entered = True
        from gf_training.card_instance_source import _open_game_parent_inputs
        from gf_training.lineage import _OWNED_LOCKS, run_lock
        from gf_training.structured_lineage import _ChainScan
        try:
            if self.directory in _OWNED_LOCKS:
                if (self.borrowed is None or self.borrowed.directory != self.directory
                        or self.borrowed.token != _OWNED_LOCKS[self.directory]):
                    raise ContractError('formal parent can borrow only the SAME explicitly supplied lock token')
                self._token = self.borrowed.token
            else:
                if self.borrowed is not None:
                    raise ContractError('stale formal borrowed lock')
                owned = self._scope.enter_context(run_lock(self.directory))
                self._token = owned.token
                self._owns_lock = True
            self.scan = _ChainScan(directory=self.directory, max_owned_bytes=self.limit, deadline=self.deadline)
            initial = self.scan.object(self.directory / 'run.json')
            if initial.get('run_variant') != RUN_VARIANT:
                raise ContractError('ordinary/fixture/short parent has no original formal declaration')
            verify_kernel_ast(load_recipe())
            read_formal_control(self.directory, initial, self.scan)
            if initial['effective_config']['preprocessing']['backend'] == 'rust':
                from gf_training.native_backend import load_native
                require_native_formal_controls(load_native())
            state, records, self._native_inputs = self._inputs_scope.enter_context(_open_game_parent_inputs(
                self.directory, require_complete=True, max_owned_bytes=self.limit, deadline=self.deadline, _scan=self.scan))
            self._native_scan_base = self.scan.retained
            if state['status'] != 'complete' or state['updates'] <= 0:
                raise ContractError('healthy positive complete formal parent required')
            from gf_training.structured_rollout import _owned_bytes
            self.scan.charge(8*_owned_bytes((state, records))+65536)
            # Reserve proposed protected Python copies in the SAME native
            # caller lease and joint total before either serializer allocates.
            self._sync_native(self._external_bytes)
            joint = (self.scan.retained+65536+self._external_bytes if self._native_inputs is None else
                int(self._native_inputs.joint_owned_bytes))
            if joint+self._guard_scratch() > self.limit:
                raise ContractError('formal parent joint canonical materialization exhausted before allocation')
            self._state_bytes = canonical(state).encode()
            self._records_bytes = canonical(records).encode()
            self._active = True
            self.recheck()
            return self
        except BaseException as error:
            self._poison(error)
            self._active = False
            candidate = getattr(error, '_structured_source_cleanup_owner', None)
            if candidate is not None:
                self._native_inputs = candidate
            try:
                cleanup = self._retire_inputs(error)
                if cleanup is not None:
                    vars(error)['_formal_parent_cleanup_owner'] = self
                    self._note(error,'formal parent input close: ',cleanup)
            except BaseException as cleanup:  # noqa: BLE001 - unconfirmed inputs retain SAME lock owner
                vars(error)['_formal_parent_cleanup_owner'] = self
                self._note(error,'formal parent retained native cleanup: ',cleanup)
                raise error
            self._closed = True
            try:
                self._retire_lock()
            except BaseException as cleanup:  # noqa: BLE001 - preserve original and retain uncertain lock owner
                vars(error)['_formal_parent_cleanup_owner'] = self
                self._note(error,'formal parent lock cleanup: ',cleanup)
            raise

    def _retire_inputs(self, initiating: BaseException | None = None) -> BaseException | None:
        failure: BaseException | None = None
        try:
            self._inputs_scope.__exit__(None if initiating is None else type(initiating),
                initiating, None if initiating is None else initiating.__traceback__)
        except BaseException as error:  # noqa: BLE001 - final guard may fail after confirmed retirement
            failure = error
            candidate = getattr(error, '_structured_source_cleanup_owner', None)
            if candidate is not None:
                self._native_inputs = candidate
        if self._native_inputs is not None and not self._native_inputs.cleanup_complete:
            try:
                self._native_inputs.close()
            except BaseException as cleanup:  # noqa: BLE001 - same reachable owner remains for retry
                if failure is None:
                    failure = cleanup
                else:
                    vars(failure)['_formal_parent_cleanup_owner'] = self
                    self._note(failure,'formal native cleanup retry: ',cleanup)
            if not self._native_inputs.cleanup_complete:
                retained_failure = failure if failure is not None else ContractError('formal native cleanup remains unconfirmed')
                vars(retained_failure)['_formal_parent_cleanup_owner'] = self
                raise retained_failure
        self._inputs_retired = True
        return failure

    def _retire_lock(self) -> None:
        # ExitStack consumes callbacks even when close fails. An empty retry
        # cannot prove retirement of the original lock/descriptor; never retry
        # an uncertain descriptor or infer success from registry membership.
        if self._lock_retirement_failure is not None:
            raise self._lock_retirement_failure
        if self._lock_retired:
            return
        try:
            self._scope.close()
        except BaseException as error:
            self._lock_retirement_failure = error
            vars(error)['_formal_parent_cleanup_owner'] = self
            raise
        self._lock_retired = True

    @property
    def cleanup_complete(self) -> bool:
        """No active-use permission; safe to query after poisoning/deadline."""
        native = self._native_inputs is None or self._native_inputs.cleanup_complete
        lock = not self._owns_lock or self._lock_retired
        return self._closed and self._inputs_retired and native and lock

    def _guard_scratch(self) -> int:
        return 1024**2+1024*(len(self.scan.files)+len(self.scan.inventories))

    def _sync_native(self, external: int) -> None:
        if self._native_inputs is not None:
            self._native_inputs.caller_growth(self.scan.retained-self._native_scan_base+external)
            self._native_inputs.recheck()

    @property
    def resident_bytes(self) -> int:
        self.require_active()
        try:
            self._sync_native(self._external_bytes)
            if self._native_inputs is not None:
                return int(self._native_inputs.joint_owned_bytes)-self._external_bytes
            return self.scan.retained+65536
        except BaseException as error:
            self._poison(error)
            raise

    def admit_external(self, extra_owned_bytes: int) -> None:
        from gf_common.domain import integer
        integer(extra_owned_bytes, 0, 2**63-1)
        if extra_owned_bytes == 0:
            # Caller buffers must already be retired. This releases only the
            # logical reservation; failed native leases stay conservatively
            # charged until actual close. No guard or operational revival.
            self._external_bytes = 0
            return
        self.require_active()
        try:
            self._sync_native(extra_owned_bytes)
            intrinsic = (self.scan.retained+65536 if self._native_inputs is None else
                int(self._native_inputs.joint_owned_bytes)-extra_owned_bytes)
            if intrinsic+extra_owned_bytes+self._guard_scratch() > self.limit:
                raise ContractError('formal parent joint intrinsic/external ownership exhausted before allocation')
            self._external_bytes = extra_owned_bytes
            self.recheck()
        except BaseException as error:
            self._poison(error)
            raise

    def require_active(self) -> None:
        from gf_training.lineage import _OWNED_LOCKS
        if self._poisoned:
            raise ContractError('formal parent owner is permanently poisoned after an admitted error') from self._failure
        if not self._active or self._closed or not self._entered:
            raise ContractError('SAME active original formal parent required')
        if time.monotonic() >= self.deadline:
            deadline_error = TimeoutError('formal parent original deadline exhausted')
            self._poison(deadline_error)
            raise deadline_error
        if _OWNED_LOCKS.get(self.directory) != self._token:
            error = ContractError('SAME active original formal parent lock required')
            self._poison(error)
            raise error

    def recheck(self) -> None:
        self.require_active()
        from gf_training.card_instance_source import recheck_game_parent_snapshot
        try:
            self._sync_native(self._external_bytes)
            intrinsic = (self.scan.retained+65536 if self._native_inputs is None else
                int(self._native_inputs.joint_owned_bytes)-self._external_bytes)
            if intrinsic+self._external_bytes+self._guard_scratch() > self.limit:
                raise ContractError('formal parent joint guard ownership exhausted')
            recheck_game_parent_snapshot(self.scan)
        except BaseException as error:
            self._poison(error)
            raise

    def final_binding(self) -> Any:
        self.recheck()
        from gf_training.card_instance_source import _binding
        # Derive exclusively from protected canonical originals, not mutable
        # public views. The binding helper receives private disposable copies.
        try:
            assert self._state_bytes is not None and self._records_bytes is not None
            state, records = json.loads(self._state_bytes), json.loads(self._records_bytes)
            return _binding(self.directory, state, records[-1], state['updates'])
        except BaseException as error:
            self._poison(error)
            raise

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        original = value if isinstance(value, BaseException) else None
        failure = original if original is not None else self._failure
        if original is not None:
            self._poison(original)
        # A poisoned owner can still finish reportable cleanup/guards, but this
        # cannot re-admit it or erase the first failure even after file repair.
        from gf_training.card_instance_source import recheck_game_parent_snapshot
        try:
            if self._active:
                if not self._poisoned:
                    self.require_active()
                recheck_game_parent_snapshot(self.scan)
        except BaseException as error:  # noqa: BLE001 - source mutation cannot replace initiating failure
            self._poison(error)
            if failure is None:
                failure = error
            elif failure is not error:
                try:
                    failure.add_note('formal parent original recheck: ' + str(error))
                except BaseException:  # noqa: BLE001, S110 - preserve first error
                    pass
        self._active = False
        try:
            input_failure = self._retire_inputs(failure)
            if input_failure is not None:
                self._poison(input_failure)
                if failure is None:
                    failure = input_failure
                elif failure is not input_failure:
                    vars(failure)['_formal_parent_cleanup_owner'] = self
                    self._note(failure,'formal parent input retirement: ',input_failure)
        except BaseException as cleanup:  # noqa: BLE001 - do not release lock before native/executor confirmation
            self._poison(cleanup)
            if failure is None:
                failure = cleanup
            vars(failure)['_formal_parent_cleanup_owner'] = self
            if failure is not cleanup:
                self._note(failure,'formal parent inputs still retain cleanup: ',cleanup)
            if original is None:
                raise failure
            return
        self._closed = True
        try:
            self._retire_lock()
        except BaseException as error:  # noqa: BLE001 - retain original and reachable lock uncertainty
            self._poison(error)
            if failure is None:
                failure = error
            vars(failure)['_formal_parent_cleanup_owner'] = self
            if failure is not error:
                self._note(failure,'formal parent lock close: ',error)
        if original is None and failure is not None:
            raise failure
