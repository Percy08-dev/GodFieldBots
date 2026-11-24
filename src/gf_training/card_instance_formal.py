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
    NATIVE_PACKED_FORMAL_MARKER,
    PACKED_DECLARATION_FIELDS,
    PACKED_RECIPE_ID,
    PACKED_RECIPE_SHA256,
    PACKED_RUN_VARIANT,
    RECIPE_ID,
    RECIPE_SHA256,
    RUN_VARIANT,
    checked_config,
    checked_packed_config,
    declaration_reference,
    packed_recipe_bytes,
    planned_execution,
    recipe_bytes,
    stop_receipt,
    validate_declaration,
    validate_formal_state,
)
from gf_common.domain import ContractError, Json, canonical, clone, closed, digest
from gf_common.owned_completion import _close_owned_descriptor, _invalidate_owned_completion
from gf_models.structured_sealed import SealedScan
from gf_training.group_pack_resources import OwnedDescriptor


def load_recipe() -> Json:
    asset = importlib.resources.files('gf_common').joinpath('contracts/card-instance-formal-ppo-v1.json')
    with asset.open('rb') as source:
        raw = source.read(65537)
    if len(raw) > 65536:
        raise ContractError('bounded formal recipe asset required')
    return recipe_bytes(raw)


def load_packed_recipe() -> Json:
    asset = importlib.resources.files('gf_common').joinpath('contracts/card-instance-formal-ppo-group-pack-v1.json')
    with asset.open('rb') as source:
        raw = source.read(65537)
    return packed_recipe_bytes(raw)


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
    packed = hasattr(loaded.config, 'storage')
    if packed:
        load_packed_recipe()
    if sys.version_info[:2] != (3, 14) or not sys._is_gil_enabled():
        raise ContractError('formal frozen AST/runtime requires Python3.14 with GIL enabled')
    recipe = load_recipe()
    expected = (checked_packed_config if packed else checked_config)(loaded.effective, recipe)
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


def require_native_formal_controls(native: Any, *, packed: bool = False) -> None:
    if packed:
        from gf_training.native_backend import require_pack_checksum
        require_pack_checksum(native)
    info = dict(native.build_info())
    if packed and (info.get('card_instance_packed_formal_controls') != NATIVE_PACKED_FORMAL_MARKER
            or info.get('card_instance_group_pack') != 'provider1;proof2;plan4'):
        raise ContractError('Rust packed formal controls/provider unavailable; no fallback')
    if info.get('card_instance_formal_controls') != NATIVE_FORMAL_MARKER:
        raise ContractError('Rust formal controls unavailable until reviewed Run3/declaration1/stop1 adoption; no fallback')


def read_formal_control(directory: Path, state: Json, scan: SealedScan, recipe: Json | None = None) -> Json:
    recipe = load_recipe() if recipe is None else recipe
    validate_formal_state(state)
    packed = state.get('run_variant') == PACKED_RUN_VARIANT
    reference = declaration_reference(state['formal_declaration'], packed=packed)
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
    if packed:
        load_packed_recipe()
        asset = importlib.resources.files('gf_common').joinpath('contracts/card-instance-formal-ppo-group-pack-v1.json')
        with scan.file(Path(str(asset)), PACKED_RECIPE_SHA256):
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
        self.external_roots: tuple[Path, ...] = ()
        self._writer = OwnedDescriptor(-1)
        self._opened_stages: list[OwnedDescriptor] = []
        self._recovery_names: set[str] = set()
        self._original: os.stat_result | None = None
        self._original_sha256 = ''
        self._original_mode: int | None = None

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
        if (self._identity(current) != self.snapshot or self._identity(held) != self.snapshot
                or current.st_mode != self._original_mode or held.st_mode != self._original_mode):
            raise ContractError('formal completion held inode/current path changed')

    @staticmethod
    def _stable_identity(info: os.stat_result) -> tuple[int, ...]:
        return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns

    @staticmethod
    def _full_identity(info: os.stat_result) -> tuple[int, ...]:
        return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def _hash(self, descriptor: int) -> str:
        result, offset = hashlib.sha256(), 0
        while True:
            if time.monotonic() >= self.deadline:
                raise TimeoutError('formal completion original hash deadline')
            block = os.pread(descriptor, 65536, offset)
            if not block:
                return result.hexdigest()
            result.update(block)
            offset += len(block)

    def publish(self, directory: Path, state: Json, *, deadline: float, max_owned_bytes: int) -> None:
        if self.started or self._uncertain_releases:
            raise ContractError('formal complete publication is one-use; later work is cleanup-only')
        if time.monotonic() >= deadline:
            raise TimeoutError('formal completion original wall deadline')
        self.started = True
        self.deadline = deadline
        self.max_storage = state['effective_config']['training']['max_owned_bytes']
        self.directory = directory
        if state.get('run_variant') == PACKED_RUN_VARIANT:
            from gf_training.group_pack_game_source import parent_external_roots
            self.external_roots = parent_external_roots(state)
        check = SealedScan(max_owned_bytes=max_owned_bytes, deadline=deadline)
        previous = check.object(directory / 'run.json')
        for key in ('run_id', 'identity', 'formal_declaration', 'generation_sha256', 'updates', 'optimizer_steps', 'accepted_decisions', 'config_sha256'):
            if canonical(previous[key]) != canonical(state[key]):
                raise ContractError('formal complete differs from original committed control: ' + key)
        check.finalize()
        raw = (canonical(state) + '\n').encode()
        check.charge(32*len(raw)+2*65536+4*4096)
        self._original_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            self.directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(self.directory_fd)
            self._directory_identity = (info.st_dev, info.st_ino)
            self._namespace()
            name = 'run.json.formal-complete.tmp'
            self._writer.fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.directory_fd)
            initial = os.fstat(self._writer.fd)
            if not stat.S_ISREG(initial.st_mode):
                raise ContractError('formal completion requires its own regular temporary inode')
            self._completion_fd_identity = (initial.st_dev, initial.st_ino)
            self._original_mode = initial.st_mode
            self.snapshot = self._identity(initial)
            offset = 0
            while offset < len(raw):
                check.check()
                amount = os.write(self._writer.fd, memoryview(raw)[offset:])
                if amount <= 0:
                    raise OSError('formal completion write did not advance')
                offset += amount
            os.fsync(self._writer.fd)
            original = os.fstat(self._writer.fd)
            if ((original.st_dev, original.st_ino) != self._completion_fd_identity
                    or original.st_mode != self._original_mode or original.st_size != len(raw)):
                raise ContractError('formal complete writer differs from its independently owned original')
            self._original = original
            current = os.stat(name, dir_fd=self.directory_fd, follow_symlinks=False)
            self.snapshot = self._identity(original)
            if (self._full_identity(current) != self._full_identity(self._original)
                    or self._original.st_size != len(raw)
                    or self._original.st_mode != self._original_mode
                    or self._hash(self._writer.fd) != self._original_sha256
                    or self._full_identity(os.fstat(self._writer.fd)) != self._full_identity(self._original)
                    or self._full_identity(os.stat(name, dir_fd=self.directory_fd, follow_symlinks=False)) != self._full_identity(self._original)):
                raise ContractError('formal original temporary identity/bytes differ')
            check.finalize()
            self._namespace()
            self._writer.close()  # This consumed stage is NEVER reset for the retained capability.
            current = os.stat(name, dir_fd=self.directory_fd, follow_symlinks=False)
            if self._stable_identity(current) != self._stable_identity(self._original):
                raise ContractError('formal original temporary changed after writer close')
            self._namespace()
            self.exposed = True  # Replace may take effect before reporting failure.
            os.replace(name, 'run.json', src_dir_fd=self.directory_fd, dst_dir_fd=self.directory_fd)
            self._namespace()
            retained = OwnedDescriptor(-1)
            self._opened_stages.append(retained)  # BEFORE open/stat can partially fail.
            retained.fd = os.open('run.json', os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=self.directory_fd)
            actual = os.fstat(retained.fd)
            current = os.stat('run.json', dir_fd=self.directory_fd, follow_symlinks=False)
            if (self._stable_identity(actual) != self._stable_identity(self._original)
                    or self._full_identity(current) != self._full_identity(actual)
                    or self._hash(retained.fd) != self._original_sha256
                    or self._full_identity(os.fstat(retained.fd)) != self._full_identity(actual)
                    or self._full_identity(os.stat('run.json', dir_fd=self.directory_fd, follow_symlinks=False)) != self._full_identity(actual)):
                raise ContractError('formal fresh retained capability differs from controlled original')
            # Only this verified original may enter the established revocation stage.
            self.fd, retained.fd = retained.fd, -1
            self.snapshot = self._identity(actual)
            self._verify()
        except BaseException as error:
            vars(error)['_formal_completion_publication_owner'] = self
            raise

    def _recover_original(self) -> None:
        assert self.directory_fd is not None and self.snapshot is not None
        held = os.fstat(self.directory_fd)
        if (held.st_dev, held.st_ino) != self._directory_identity:
            raise ContractError('formal retained recovery directory identity changed')
        for name in ('run.json', 'run.json.formal-complete.tmp'):
            if name in self._recovery_names:
                continue
            self._recovery_names.add(name)
            candidate = OwnedDescriptor(-1)
            self._opened_stages.append(candidate)
            try:
                current = os.stat(name, dir_fd=self.directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != self.snapshot[:2]:
                continue
            try:
                candidate.fd = os.open(name, os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=self.directory_fd)
            except FileNotFoundError:
                continue
            actual = os.fstat(candidate.fd)
            if not stat.S_ISREG(actual.st_mode) or (actual.st_dev, actual.st_ino) != self.snapshot[:2]:
                candidate.close()
                continue
            self.fd, candidate.fd = candidate.fd, -1
            return
        raise ContractError('formal actual original cannot be recovered through its fixed owned names')

    def _release(self) -> None:
        failure: BaseException | None = None
        for stage in (self._writer, *self._opened_stages):
            try:
                stage.close()
            except BaseException as error:  # noqa: BLE001 - release independent certain stages too.
                failure = failure or error
        if self.directory_fd is not None:
            descriptor, self.directory_fd = self.directory_fd, None
            try:
                _close_owned_descriptor(self, descriptor, self._directory_identity, 'formal-directory')
            except BaseException as error:  # noqa: BLE001 - preserve the first actual close uncertainty.
                failure = failure or error
        # Keep the last actual revoker until earlier stages are certain, or its
        # SAME known inode has already been invalidated through this capability.
        if self.fd is not None and (self.revoked or (failure is None and not self._uncertain_releases)):
            descriptor, self.fd = self.fd, None
            try:
                _close_owned_descriptor(self, descriptor, self._completion_fd_identity, 'formal-completion')
            except BaseException as error:  # noqa: BLE001 - retain the initiating stage failure.
                failure = failure or error
        if failure is not None:
            raise failure
        if self._uncertain_releases:
            raise ContractError('formal terminal release is uncertain; stale descriptor numbers are quarantined')

    def revoke(self) -> None:
        failure: BaseException | None = None
        try:
            if not self.revoked and self.snapshot is not None:
                assert self.directory is not None
                if self.fd is None and self.directory_fd is not None:
                    self._recover_original()
                _invalidate_owned_completion(self, self.directory/'run.json')
                self.revoked = True
        except BaseException as error:  # noqa: BLE001 - retirement remains independent of revocation.
            failure = error
        try:
            self._release()
        except BaseException as error:  # noqa: BLE001 - first failure and all consumed stages survive.
            failure = failure or error
        if failure is not None:
            vars(failure)['_formal_completion_publication_owner'] = self
            raise failure

    def _final_guard(self) -> None:
        from gf_training.ppo_run import owned_bytes
        assert self.directory is not None and self.snapshot is not None
        if time.monotonic() >= self.deadline:
            raise ContractError('formal outer completion wall/storage/original inode guard failed')
        usage = (owned_bytes(self.directory, self.deadline, external_roots=self.external_roots)
            if self.external_roots else owned_bytes(self.directory, self.deadline))
        current = (self.directory/'run.json').lstat()
        if (time.monotonic() >= self.deadline or usage > self.max_storage
                or self._identity(current) != self.snapshot or current.st_mode != self._original_mode):
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
        self.packed = hasattr(loaded.config, 'storage')
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
            require_native_formal_controls(load_native(), packed=self.packed)

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
            declaration = {'schema_version': 2 if self.packed else 1,
                'kind': 'card-instance-formal-packed-parent-declaration' if self.packed else 'card-instance-formal-parent-declaration',
                'recipe_id': PACKED_RECIPE_ID if self.packed else RECIPE_ID,
                'recipe_sha256': PACKED_RECIPE_SHA256 if self.packed else RECIPE_SHA256, 'algorithm_recipe_sha256': ALGORITHM_SHA256,
                'seed_slot': self.loaded.config.training.seeds[0], 'config_sha256': self.loaded.sha256,
                'config_inputs_sha256': digest(self.loaded.inputs), 'input_binding': self.recipe['input_binding'],
                'execution_identity_sha256': digest(execution), 'identity_sha256': digest(identity)}
            if self.packed:
                from gf_common.group_pack_game import storage_binding
                declaration.update(storage_binding=storage_binding(),
                    logical_config_sha256=digest(checked_packed_config(self.loaded.effective, self.recipe)))
            closed(declaration, PACKED_DECLARATION_FIELDS if self.packed else DECLARATION_FIELDS)
            from gf_training.structured_lineage import _ChainScan, _write
            scan = _ChainScan(directory=directory, max_owned_bytes=self.loaded.config.training.max_owned_bytes, deadline=deadline)
            sha = _write(directory / 'formal-declaration.json', declaration, scan=scan, immutable=True)
            self.reference = {'relative_path': 'formal-declaration.json', 'sha256': sha,
                'recipe_id': PACKED_RECIPE_ID if self.packed else RECIPE_ID,
                'recipe_sha256': PACKED_RECIPE_SHA256 if self.packed else RECIPE_SHA256, 'algorithm_recipe_sha256': ALGORITHM_SHA256,
                'seed_slot': declaration['seed_slot']}
        else:
            state = self.originals.object(directory / 'run.json')
            if state.get('run_variant') != (PACKED_RUN_VARIANT if self.packed else RUN_VARIANT):
                raise ContractError('ordinary/short records cannot resume through formal entry')
            self.reference = declaration_reference(state['formal_declaration'], packed=self.packed)
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
        if self.packed:
            asset = importlib.resources.files('gf_common').joinpath('contracts/card-instance-formal-ppo-group-pack-v1.json')
            with self.originals.file(Path(str(asset)), PACKED_RECIPE_SHA256):
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
        state.update(run_variant=PACKED_RUN_VARIANT if self.packed else RUN_VARIANT, formal_declaration=clone(self.reference), formal_stop=None)
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


class FinalSourceBudgetLoan:
    """Actual-custodian B/L resource ledger, never Source or fitting authority."""
    def __init__(self, parent: FormalGameParent, *, custodian: Any, base_bytes: int, deadline: float) -> None:
        from gf_common.domain import integer
        from gf_training.card_instance_source_locks import GameSourceCustody
        if type(parent) is not FormalGameParent or type(custodian) is not GameSourceCustody:
            raise ContractError('actual parent and complete game custody required for resource loan')
        from math import isfinite
        if type(deadline) not in (int,float) or not isfinite(deadline) or deadline<=time.monotonic():
            raise ContractError('finite active resource loan deadline required')
        parent.require_active()
        custodian.require()
        if custodian.ownership_for(parent.directory) is not parent.borrowed or parent._resource_loan is not None:
            raise ContractError('one resource loan on SAME actual custodian-borrowed parent required')
        integer(base_bytes, 0, parent.limit)
        self._parent, self._custodian = parent, custodian
        self.deadline = min(deadline, parent.deadline, custodian.deadline)
        self._base, self._local = base_bytes, 0
        self._source: Any = None
        self._opening: list[Any] = []
        self._opening_retired = False
        self._retired = False
        self._failure: BaseException | None = None
        parent._resource_loan = self  # Reachable before the first reservation.
        try:
            parent.admit_external(base_bytes)
        except BaseException as error:
            self._failure = error
            vars(error)['_final_source_budget_cleanup_owner'] = self
            raise

    def require(self, parent: FormalGameParent) -> None:
        if self._failure is not None:
            raise ContractError('final Source resource loan is permanently failed') from self._failure
        if self._retired or parent is not self._parent or parent._resource_loan is not self:
            raise ContractError('SAME active actual-parent resource loan required')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('original resource loan deadline')
        parent.require_active()
        self._custodian.require()
        if self._custodian.ownership_for(parent.directory) is not parent.borrowed:
            raise ContractError('resource loan custody/parent changed')

    @property
    def custodian(self) -> Any:
        self.require(self._parent)
        return self._custodian

    @property
    def base_bytes(self) -> int:
        self.require(self._parent)
        return self._base

    @property
    def source_bytes(self) -> int:
        self.require(self._parent)
        return self._local

    @property
    def declared_external_bytes(self) -> int:
        self.require(self._parent)
        if self._parent._native_inputs is None:return self._parent._external_bytes
        return getattr(self._parent, '_declared_external_bytes', self._parent._external_bytes)

    def _admit(self, base: int, local: int) -> None:
        self.require(self._parent)
        try:
            self._parent.admit_external(base+local)
        except BaseException as error:
            self._failure = error
            vars(error)['_final_source_budget_cleanup_owner'] = self
            raise
        self._base, self._local = base, local

    def update_base(self, n: int, *, custodian: Any) -> None:
        from gf_common.domain import integer
        integer(n, 0, self._parent.limit)
        if custodian is not self._custodian:
            raise ContractError('SAME actual Dataset custodian required for base replacement')
        self._admit(n, self._local)

    def admit_local(self, n: int) -> None:
        from gf_common.domain import integer
        integer(n, 0, self._parent.limit)
        # Local high-water never retires data through a numeric lowering claim.
        self._admit(self._base, max(self._local, n))

    def _keep_opening(self, value: Any) -> None:
        self.require(self._parent)
        if self._source is not None or self._opening_retired:
            raise ContractError('opening objects belong only to SAME unbound resource loan')
        self._opening.append(value)

    def _bind_source(self, source: Any) -> None:
        from gf_training.card_instance_final_validation import GameFinalValidationSource
        if (type(source) is not GameFinalValidationSource or source._parent is not self._parent
                or self._source is not None):
            raise ContractError('one SAME concrete physical Source owner required')
        self._source = source
        self._opening.clear()
        self._opening_retired = True

    def retire_local(self, actual_read_owner: Any) -> None:
        if self._retired:
            if actual_read_owner is not self._source:
                raise ContractError('SAME confirmed retired Source required')
            return
        if actual_read_owner is not self._source or self._source is None or not self._source.cleanup_complete:
            raise ContractError('SAME physically retired Source required before releasing local bytes')
        if self._parent._unit_borrow is not None:
            raise ContractError('actual original-episode borrow must retire before local bytes')
        self._opening.clear()
        self._opening_retired = True
        # Failed/expired owner never gains new nonzero cleanup permission. Keep
        # actual leases conservative; Dataset must fail and close actual owners.
        if self._parent._poisoned or not self._parent._active or time.monotonic() >= self.deadline:
            self._retired = True
            return
        try:
            self._parent.admit_external(self._base)
        except BaseException as error:
            self._failure = error
            vars(error)['_final_source_budget_cleanup_owner'] = self
            raise
        self._local = 0
        self._retired = True

    def _retire_failed_opening(self) -> None:
        if self._source is not None:
            raise ContractError('actual bound Source must retire its own local data')
        self._opening.clear()
        self._opening_retired = self._retired = True
        # Conservative old declared reservation stays until actual parent close.

    def _retire_from_parent(self) -> None:
        if self._source is None:
            self._retire_failed_opening()
        elif not self._source.cleanup_complete:
            self._source.close()
        if self._source is not None and self._source.cleanup_complete and not self._retired:
            self.retire_local(self._source)

    @property
    def cleanup_complete(self) -> bool:
        return (self._retired and self._opening_retired and not self._opening
            and (self._source is None or self._source.cleanup_complete))


class _PackedFinalOriginals:
    """One preowned final-original slot on the SAME retained formal parent."""
    def __init__(self, parent: FormalGameParent) -> None:
        self.parent = parent
        self.owner: Any = None
        self.provider: Any = None
        self.pool: Any = None
        self.started = self._closed = False
        self._controls = 0
        self._control_uncertain = False
        self._release_attempted = False
        self._release_before = 0

    def open(self, directory: Path, settings: Any, *, deadline: float,
             pending: Any = None) -> Any:
        from gf_common.artifacts import json_bound
        from gf_training.card_instance_final_validation import _require_borrow
        from gf_training.group_pack_game_source import PackedSourceOwner
        from gf_training.native_group_pack import OriginalArtifactProvider
        parent = self.parent
        parent.require_active()
        _require_borrow(parent)
        if (parent._packed_final_originals is not self or self.started
                or parent.scan._packed_state is None):
            raise ContractError('one actual packed parent final-original admission required')
        native = parent._native_inputs
        if native is not None:
            native._executor.prepare()  # BEFORE opening/querying any final originals.
        self.started = True
        self.pool = parent.scan.ensure_packed_pool()
        self.owner = PackedSourceOwner.__new__(PackedSourceOwner)
        PackedSourceOwner.__init__(self.owner, directory, settings, deadline,
            storage_binding=parent.scan._packed_state['storage_binding'],
            max_owned_bytes=parent.limit, pool=self.pool, _issued_pending=pending, _final_parent=parent)
        original = self.owner.artifacts
        assert original is not None
        # The physical constructor has already admitted the SAME SQL index.
        # This allocation-free bound precedes any new proof/custody/registry JSON.
        wire = (json_bound(original._index) + 4096 * len(self.owner._entries)
            + 64 * len(str(directory)) + 8 * original._originals.descriptor_file.identity.st_size + 65536)
        amount = 8 * (32 * wire + 65536)
        before = self.pool.budget.used
        try:
            self.pool.budget.acquire(amount, deadline)
        finally:
            if self.pool.budget.used == before + amount:
                self._controls = amount
            elif self.pool.budget.used != before:
                self._control_uncertain = True
        if self._controls != amount:
            raise ContractError('packed final metadata admission unconfirmed')
        if native is not None:
            self.provider = OriginalArtifactProvider.__new__(OriginalArtifactProvider)
            OriginalArtifactProvider.__init__(self.provider, (self.owner,))
            native._attach_final_artifacts(self.provider)
        return self.owner

    @property
    def cleanup_complete(self) -> bool:
        return (self._closed and not self._controls and not self._control_uncertain
            and (self.owner is None or getattr(self.owner, '_closed', False))
            and (self.provider is None or getattr(self.provider, '_closed', False)))

    def close(self) -> None:
        if self.cleanup_complete:
            return
        # The actual executor hold refuses this BEFORE originals/metadata refund.
        if self.pool is not None:
            self.pool.require_terminal_close()
        if self.provider is not None:
            # This concrete constructor creates acquisition state BEFORE its
            # first reservation/IO. An earlier failed prefix owns no resources.
            if not hasattr(self.provider, '_acquisition_states'):
                self.provider = None
            else:
                self.provider.close()
        if self.owner is not None:
            if hasattr(self.owner, 'pool'):
                self.owner.close()
            else:
                self.owner._closed = True  # Validation failed before any external effect.
        if self._control_uncertain:
            raise ContractError('packed final metadata admission remains uncertain')
        if self._controls:
            amount = self._controls
            if self._release_attempted:
                if self.pool.budget.used == self._release_before - amount:
                    self._controls = 0
                else:
                    raise ContractError('packed final metadata retirement remains uncertain')
            else:
                self._release_before = self.pool.budget.used
                self._release_attempted = True
                try:
                    self.pool.budget.release(amount)
                finally:
                    if self.pool.budget.used == self._release_before - amount:
                        self._controls = 0
        self._closed = True


class FormalGameParent:
    """Complete original formal parent owner; not Dataset or online permission."""
    def __init__(self, directory: Path, *, max_owned_bytes: int, deadline: float, ownership: Any = None,
                 extra_owned_bytes: int = 0) -> None:
        from gf_common.domain import integer
        integer(extra_owned_bytes, 0, max_owned_bytes-1)
        self._entry_extra = extra_owned_bytes
        self._declared_external_bytes = extra_owned_bytes
        self._resource_loan: Any = None
        self._unit_borrow: Any = None
        self.directory = directory.resolve()
        self.limit, self.deadline = max_owned_bytes, deadline
        self.borrowed = ownership
        self._scope = ExitStack()
        self._inputs_scope = ExitStack()
        self._inputs_retired = False
        self._owns_lock = False
        self._lock_retired = False
        self._lock_retirement_failure: BaseException | None = None
        self._external_bytes = extra_owned_bytes
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
        self._native_final_inputs: Any = None
        self._packed_final_originals: _PackedFinalOriginals | None = None
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
            self.scan = _ChainScan(directory=self.directory, max_owned_bytes=self.limit-self._entry_extra, deadline=self.deadline)
            initial = self.scan.object(self.directory / 'run.json')
            if initial.get('run_variant') not in (RUN_VARIANT, PACKED_RUN_VARIANT):
                raise ContractError('ordinary/fixture/short parent has no original formal declaration')
            verify_kernel_ast(load_recipe())
            read_formal_control(self.directory, initial, self.scan)
            if initial['effective_config']['preprocessing']['backend'] == 'rust':
                from gf_training.native_backend import load_native
                require_native_formal_controls(load_native(), packed=initial['run_variant'] == PACKED_RUN_VARIANT)
            state, records, self._native_inputs = self._inputs_scope.enter_context(_open_game_parent_inputs(
                self.directory, require_complete=True, max_owned_bytes=self.limit, deadline=self.deadline,
                extra_owned_bytes=self._entry_extra, _scan=self.scan))
            self._native_scan_base = self.scan.retained
            if state['status'] != 'complete' or state['updates'] <= 0:
                raise ContractError('healthy positive complete formal parent required')
            from gf_training.structured_rollout import _owned_bytes
            self.scan.charge(8*_owned_bytes((state, records))+65536)
            # Reserve proposed protected Python copies in the SAME native
            # caller lease and joint total before either serializer allocates.
            self._sync_native(self._external_bytes)
            joint = (self._python_owned_bytes()+self._external_bytes if self._native_inputs is None else
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
        resource_loan=getattr(self,'_resource_loan',None)
        if resource_loan is not None and not resource_loan.cleanup_complete:
            try:
                resource_loan._retire_from_parent()
            except BaseException as error:  # noqa: BLE001 - parent retains actual Source/resource cleanup
                failure=error
                vars(error)['_formal_parent_cleanup_owner']=self
            if not resource_loan.cleanup_complete:
                failure=failure or ContractError('actual Source/resource retirement remains unconfirmed')
                vars(failure)['_formal_parent_cleanup_owner']=self
                raise failure
        final_inputs = getattr(self, '_native_final_inputs', None)
        if final_inputs is not None and not final_inputs.parent_cleanup_complete:
            try:
                final_inputs.retire_from_parent(initiating)
            except BaseException as error:  # noqa: BLE001 - SAME child/parent owner survives retirement failure
                if failure is None:failure=error
                elif failure is not error:self._note(failure,'formal child retirement: ',error)
                vars(failure)['_formal_parent_cleanup_owner'] = self
            if not final_inputs.parent_cleanup_complete:
                failure = failure or ContractError('native final64 collection cleanup remains unconfirmed')
                vars(failure)['_formal_parent_cleanup_owner'] = self
                raise failure
        try:
            retirement_error = initiating or failure
            self._inputs_scope.__exit__(None if retirement_error is None else type(retirement_error),
                retirement_error, None if retirement_error is None else retirement_error.__traceback__)
        except BaseException as error:  # noqa: BLE001 - final guard may fail after confirmed retirement
            if failure is None:
                failure = error
            elif failure is not error:
                vars(failure)['_formal_parent_cleanup_owner'] = self
                self._note(failure, 'formal parent original inputs retirement: ', error)
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
        originals = self._packed_final_originals
        if originals is not None:
            try:
                originals.close()
            except BaseException as error:  # noqa: BLE001 - keep actual final originals after uncertain retirement.
                if failure is None:
                    failure = error
                elif failure is not error:
                    self._note(failure, 'formal final originals retirement: ', error)
                vars(failure)['_formal_parent_cleanup_owner'] = self
            if not originals.cleanup_complete:
                failure = failure or ContractError('formal final originals retirement unconfirmed')
                vars(failure)['_formal_parent_cleanup_owner'] = self
                raise failure
        scan = self.scan
        if scan is not None and (getattr(scan, '_packed_pool', None) is not None or getattr(scan, '_packed_state', None) is not None):
            try:
                scan.close()
            except BaseException as error:  # noqa: BLE001 - actual originals/accounts stay parent-owned.
                if failure is None:
                    failure = error
                elif failure is not error:
                    self._note(failure, 'formal packed scan retirement: ', error)
                vars(failure)['_formal_parent_cleanup_owner'] = self
            if not scan._closed:
                failure = failure or ContractError('formal packed scan originals/account retirement unconfirmed')
                vars(failure)['_formal_parent_cleanup_owner'] = self
                raise failure
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
        final_inputs = getattr(self, '_native_final_inputs', None)
        final = final_inputs is None or final_inputs.parent_cleanup_complete
        lock = not self._owns_lock or self._lock_retired
        scan = self.scan
        final_originals = self._packed_final_originals is None or self._packed_final_originals.cleanup_complete
        originals = (scan is None or getattr(scan, '_packed_pool', None) is None and getattr(scan, '_packed_state', None) is None or scan._closed)
        resource=getattr(self,'_resource_loan',None)
        return self._closed and self._inputs_retired and native and final and final_originals and originals and lock and (resource is None or resource.cleanup_complete)

    def _open_packed_final_originals(self, directory: Path, settings: Any, *,
            deadline: float, pending: Any = None) -> Any:
        self.require_active()
        if self._packed_final_originals is not None:
            raise ContractError('parent final-original slot has exactly one lifetime')
        self._packed_final_originals = _PackedFinalOriginals(self)
        try:
            return self._packed_final_originals.open(directory, settings, deadline=deadline, pending=pending)
        except BaseException as error:
            self._poison(error)
            vars(error)['_formal_parent_cleanup_owner'] = self
            raise

    def _borrow_native_final_inputs(self, *, deadline: float) -> Any:
        """Private actual-owner forwarding; never reopen/construct from a summary."""
        self.require_active()
        from gf_common.card_instance_rules import card_instance_rules
        from gf_models.structured_binding import input_binding_game_v3
        from gf_training.card_instance_final_validation import _require_borrow
        from gf_training.native_backend import binary_identity, recorded_card_instance_preprocessing
        from gf_training.native_card_instance import (
            _NativeFinalValidationInputs,
            _NativeGameInputs,
            _require_final_validation_native,
        )
        token = _require_borrow(self)
        owner = self._native_inputs
        if (type(owner) is not _NativeGameInputs or owner._explicit_candidate
                or owner.plan.get('mode') != 'historical_parent'
                or getattr(self, '_native_final_inputs', None) is not None):
            raise ContractError('one SAME actual retained native parent/collection loan required')
        module = _require_final_validation_native(owner.settings, owner.native,
            packed=owner._artifact_provider is not None, recorded_build=owner._recorded_build)
        self.recheck()
        # Existing Actor control admission covers these protected state/identity
        # copies; the caller never supplies a native module, session or flag.
        assert self._state_bytes is not None
        state = json.loads(self._state_bytes)
        identity = binary_identity(module)
        recorded = state['identity']['preprocessing']
        recorded_card_instance_preprocessing(recorded,
            binding=input_binding_game_v3(owner.actions.action_contract, card_instance_rules()))
        if (canonical(recorded['binary']) != canonical(identity)
                or owner._binary_identity != canonical(identity).encode()
                or owner.summary is None or Path(owner.summary['parent']['directory']) != self.directory):
            raise ContractError('final64 original backend/binary/source differs from SAME native parent')
        loan = _NativeFinalValidationInputs(owner, self, token, deadline=deadline)
        self._native_final_inputs = loan  # Reachable BEFORE inspection/packet work.
        return loan

    def _native_final_failure(self, loan: Any, error: BaseException) -> None:
        # Cleanup-only association; no guard/I/O or operational revival.
        if getattr(self, '_native_final_inputs', None) is not loan:
            raise ContractError('SAME retained native final64 cleanup owner required')
        self._poison(error)

    def _original_history_native_owner(self) -> Any:
        """Private SAME actual native owner; never create/reopen from summaries."""
        from gf_common.card_instance_rules import card_instance_rules
        from gf_models.structured_binding import input_binding_game_v3
        from gf_training.native_backend import (
            binary_identity,
            original_history_reader_settings,
            recorded_card_instance_preprocessing,
        )
        from gf_training.native_card_instance import _NativeGameInputs
        self.recheck()
        owner=self._native_inputs
        if type(owner) is not _NativeGameInputs or owner._explicit_candidate or owner.plan.get('mode')!='historical_parent':
            raise ContractError('SAME actual retained native parent required for original history')
        original_history_reader_settings(owner.settings, owner.native, recorded_build=owner._recorded_build)
        assert self._state_bytes is not None
        state=json.loads(self._state_bytes)
        actual=binary_identity(owner.native)
        recorded=state['identity']['preprocessing']
        recorded_card_instance_preprocessing(recorded,binding=input_binding_game_v3(owner.actions.action_contract,card_instance_rules()))
        if canonical(recorded['binary'])!=canonical(actual) or owner._binary_identity!=canonical(actual).encode():
            raise ContractError('original history must retain SAME recorded backend/binary/source')
        return owner

    def borrow_original_training_episode(self, generation: int, episode_ordinal: int, *, budget_loan: FinalSourceBudgetLoan) -> Any:
        from gf_training.card_instance_source import _borrow_original_training_episode
        return _borrow_original_training_episode(self,generation,episode_ordinal,budget_loan=budget_loan)

    def _guard_scratch(self) -> int:
        return 1024**2+1024*(len(self.scan.files)+len(self.scan.inventories))

    def _python_owned_bytes(self) -> int:
        pool = getattr(self.scan, '_packed_pool', None)
        return (self.scan.retained if pool is None else pool.budget.used) + 65536

    def _sync_native(self, external: int) -> None:
        if self._native_inputs is not None:
            initial=getattr(self,'_entry_extra',0)
            controls = self.scan.retained-self._native_scan_base
            growth = max(0,external-initial)
            if getattr(self._native_inputs, '_artifact_provider', None) is None:
                self._native_inputs.caller_growth(controls+growth)
            else:
                self._native_inputs.caller_growth(controls, external_growth=growth)
            self._native_inputs.recheck()
            self._declared_external_bytes=max(initial,external)

    @property
    def resident_bytes(self) -> int:
        self.require_active()
        try:
            self._sync_native(self._external_bytes)
            if self._native_inputs is not None:
                return int(self._native_inputs.joint_owned_bytes)-getattr(self,'_declared_external_bytes',self._external_bytes)
            return self._python_owned_bytes()
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
            intrinsic = (self._python_owned_bytes() if self._native_inputs is None else
                int(self._native_inputs.joint_owned_bytes)-getattr(self,'_declared_external_bytes',extra_owned_bytes))
            declared = extra_owned_bytes if self._native_inputs is None else self._declared_external_bytes
            if intrinsic+declared+self._guard_scratch() > self.limit:
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
            intrinsic = (self._python_owned_bytes() if self._native_inputs is None else
                int(self._native_inputs.joint_owned_bytes)-getattr(self,'_declared_external_bytes',self._external_bytes))
            declared = self._external_bytes if self._native_inputs is None else self._declared_external_bytes
            if intrinsic+declared+self._guard_scratch() > self.limit:
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
