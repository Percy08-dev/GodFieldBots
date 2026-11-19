"""Actual retained Rust-writer originals and canonical data correspondence.

Opening data never grants collection, update, actor or final-parent authority.
The online issuer must additionally own the actual native/service/plan lifetime.
"""

from __future__ import annotations

import fnmatch
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.artifacts import json_bound
from gf_common.domain import ContractError, Json, canonical, digest
from gf_training.group_pack_pool import OriginalArtifactReader, PackedArtifactPool
from gf_training.group_pack_resources import close_resources, retain_cleanup
from gf_training.group_pack_source import OriginalFile, PackedSource
from gf_training.rust_player_storage import RustPendingDescriptor
from gf_training.structured_source import _json

_SOURCE_ISSUER = object()


class RustPlayerOriginals:
    """Borrow SAME pending-descriptor custody; share the actual Source account.

    Decoded records/metadata returned here are borrowed until close. A caller
    retires all such views before closing; canonical numeric banks have their
    own explicit owners in this same account.
    """

    def __init__(self, pending: RustPendingDescriptor, *, pool: PackedArtifactPool,
                 settings: Any, deadline: float) -> None:
        if type(pending) is not RustPendingDescriptor or type(pool) is not PackedArtifactPool:
            raise ContractError("actual Rust pending descriptor and shared original pool required")
        pending.recheck()
        self.pending: Any = pending
        self.pool, self.budget = pool, pool.budget
        self._cold: Any = None
        self._resume: Any = None
        self._cold_descriptor: Any = None
        self._descriptor_path, self._descriptor_sha256 = pending.path, pending.descriptor_sha256
        self.root, self.settings, self.deadline = pending.directory, settings, deadline
        self.source: PackedSource | None = None
        self.files: Json = {}
        self._entries: Json = {}
        self._reservations: list[Any] = []
        self._closed = False
        self._cleanup_children: list[Any] = []
        try:
            self.source = PackedSource(self._descriptor_path, deadline=deadline,
                max_owned_bytes=pool.budget.limit, pool=pool, _allow_pending=True)
            original = self.source._originals
            if (original.descriptor_file.sha256 != self._descriptor_sha256
                    or original.descriptor_file.path != self._descriptor_path):
                raise ContractError("reopened originals differ from SAME retained writer descriptor")
            self._entries = {row["name"]: row for row in
                (*self.source._index["members"], *self.source._index["controls"])}
            self.charge(sys.getsizeof(self._entries))
            self.reader = OriginalArtifactReader(self.source, self.read, self.charge, self.root)
            self.finalize()
        except BaseException as error:
            retain_cleanup(self, error)
            raise

    def _open_sealed_originals(self) -> None:
        """File I/O only, after each constructor's actual authority checks."""
        try:
            self._cold_descriptor = OriginalFile(self._descriptor_path, self.deadline)
            if self._cold_descriptor.sha256 != self._descriptor_sha256:
                raise ContractError('actual retained descriptor differs from prior Source proof')
            self.source = PackedSource(self._descriptor_path, deadline=self.deadline,
                max_owned_bytes=self.pool.budget.limit, pool=self.pool, _allow_pending=True)
            if self.source._originals.descriptor_file.sha256 != self._descriptor_sha256:
                raise ContractError('actual retained descriptor changed during original opening')
            self._entries = {row['name']: row for row in
                (*self.source._index['members'], *self.source._index['controls'])}
            self.charge(sys.getsizeof(self._entries))
            self.reader = OriginalArtifactReader(self.source, self.read, self.charge, self.root)
            self.finalize()
        except BaseException as error:
            retain_cleanup(self, error)
            raise

    def check(self) -> None:
        if self._closed or self.source is None:
            raise ContractError("Rust original data owner is closed")
        if self.pending is not None:
            self.pending.recheck()
        else:
            if self._cold is not None:
                self._cold.require_pair()
            else:
                self._resume.require()
            self._cold_descriptor.recheck()
        self.source.recheck()

    def charge(self, size: int) -> None:
        self._reservations.append(self.budget.reserve(size))

    def _name(self, path: Path) -> str:
        try:
            name = path.absolute().relative_to(self.root).as_posix()
        except ValueError as error:
            raise ContractError("original logical member belongs to another collection") from error
        if name not in self._entries:
            raise ContractError("unregistered original logical member")
        return name

    def member_identity(self, name: str) -> Json:
        self.check()
        row = self._entries[name]
        assert self.source is not None
        original = self.source._originals
        return {"recipe": "rust-player-packed-member-v1",
            "descriptor_sha256": self._descriptor_sha256,
            "database_sha256": original.descriptor["database_sha256"],
            "name": name, "length": row.get("length", row.get("bytes")),
            "sha256": row["sha256"]}

    def _remember(self, name: str) -> None:
        if name not in self.files:
            identity = self.member_identity(name)
            self.charge(32 * json_bound(identity) + 65536)
            self.files[name] = {"sha256": identity["sha256"],
                "bytes": identity["length"], "identity": identity}

    def read(self, path: Path) -> bytes:
        self.check()
        name = self._name(path)
        assert self.source is not None
        row = self._entries[name]
        if row.get("length", row.get("bytes")) > self.settings.max_source_file_mib * 1024**2:
            raise ContractError("bounded original logical file required")
        raw = self.source.read(name)
        self._remember(name)
        return raw

    def object(self, path: Path) -> Json:
        name = self._name(path)
        row = self._entries[name]
        self.charge(32 * row.get("length", row.get("bytes")) + 65536)
        return _json(self.read(path))

    def contains(self, path: Path) -> bool:
        self.check()
        return path.absolute().is_relative_to(self.root) and path.relative_to(self.root).as_posix() in self._entries

    def inventory(self, pattern: str) -> tuple[str, ...]:
        self.check()
        return tuple(name for name in sorted(self._entries)
            if pattern == "**/*" or fnmatch.fnmatchcase(name, pattern))

    def remember_inventory_entry(self, name: str, size: int, checksum: str) -> None:
        self.check()
        row = self._entries.get(name)
        if row is None or row.get("length", row.get("bytes")) != size or row["sha256"] != checksum:
            raise ContractError("inventory differs from actual sealed original index")
        self._remember(name)

    def remember(self, path: Path, identity: Json, checksum: str) -> None:
        name = self._name(path)
        if canonical(identity) != canonical(self.member_identity(name)) or self._entries[name]["sha256"] != checksum:
            raise ContractError("logical member identity differs from actual original")
        self._remember(name)

    def finalize(self) -> None:
        self.check()

    def validate_index(self, manifest: Json) -> None:
        assert self.source is not None
        index = self.source._index
        groups = {row['group_id']: row for row in index['groups']}
        episodes = {row['episode_id']: row for row in index['episodes']}
        results = {row['episode_id']: row for row in manifest['episodes']}
        if (set(groups) != {group['episodes'][0]['episode_id'] for group in manifest['schedule']}
                or set(episodes) != set(results)):
            raise ContractError('actual Rust writer index differs from complete plan')
        membership = {}
        for group in manifest['schedule']:
            names = [row['episode_id'] for row in group['episodes']]
            gid = names[0]
            # Rust storage fixes these data-only group controls before admission;
            # actual executed process health is sealed separately after stop.
            if groups[gid]['record'] != {'seed': group['seed'], 'episode_ids': names}:
                raise ContractError('actual Rust writer group controls differ from plan')
            for name in names:
                row, result = episodes[name], results[name]
                terminal = self.object(self.root / name / 'native-terminal.json')['report']
                if (row['group_id'] != gid or row['game_id'] != result['game_id']
                        or row['record'] != terminal or terminal['episode'] != name
                        or terminal['game'] != result['game_id']
                        or terminal['end_kind'] != result['end_kind']
                        or terminal['error'] != result['failure']
                        or terminal['outcome'] != result['outcome']):
                    raise ContractError('actual Rust writer episode differs from final report')
                membership[name] = gid
        if any(row['group_id'] != membership[row['name'].split('/')[0]] for row in index['members']):
            raise ContractError('actual Rust writer member belongs to another group')

    def close(self) -> None:
        if self._closed:
            return
        if getattr(self, '_cleanup_children', None):
            close_resources(self, [])
        if self.source is not None:
            self.source.close()
        if self._cold_descriptor is not None:
            self._cold_descriptor.close()
        self._entries.clear()
        self.files.clear()
        for reservation in self._reservations:
            reservation.close()
        self._reservations.clear()
        self._closed = True


class RustColdOriginals(RustPlayerOriginals):
    """Cold DATA reader from actual handoff/lock/held files; never online Source."""

    def __init__(self, cold: Any, ownership: Any, path: Path, checksum: str, *,
                 pool: PackedArtifactPool, settings: Any, deadline: float) -> None:
        from gf_training.lineage import require_hpo_run_owner
        from gf_training.rust_player_supervision import ColdContinuation
        if type(cold) is not ColdContinuation or type(pool) is not PackedArtifactPool:
            raise ContractError('actual cold continuation and SAME original pool required')
        cold.require_pair()
        require_hpo_run_owner(ownership.directory, ownership)
        if ownership.directory != Path(cold._control.operation['parent_directory']):
            raise ContractError('actual cold parent mutation owner differs')
        self.pending, self._cold = None, cold
        self._resume = None
        self.pool, self.budget = pool, pool.budget
        self._descriptor_path, self._descriptor_sha256 = path, checksum
        self.root, self.settings, self.deadline = path.parent, settings, deadline
        self.source, self._cold_descriptor = None, None
        self.files, self._entries, self._reservations = {}, {}, []
        self._closed = False
        self._cleanup_children = []
        self._open_sealed_originals()



class RustResumeOriginals(RustPlayerOriginals):
    """Current-host resume DATA arm; never a recreated Cold/Pending capability."""

    def __init__(self, audit: Any, path: Path, checksum: str, *, settings: Any, deadline: float) -> None:
        from gf_training.rust_player_restore import RustResumeAudit
        if type(audit) is not RustResumeAudit:
            raise ContractError('actual current-host resume audit required')
        audit.require()
        self.pending = self._cold = None
        self._resume = audit
        self.pool, self.budget = audit.pool, audit.pool.budget
        self._descriptor_path, self._descriptor_sha256 = path, checksum
        self.root, self.settings, self.deadline = path.parent, settings, deadline
        self.source, self._cold_descriptor = None, None
        self.files, self._entries, self._reservations = {}, {}, []
        self._closed = False
        self._cleanup_children = []
        self._open_sealed_originals()



class RustPlayerSource:
    """Private same-host projection of actual native capture and sealed originals."""

    collection: Any
    registry: Any
    plan: Any
    parent: Any
    state: str
    _rollout: Any
    _records: Any
    _charges: list[Any]
    _numeric_originals: dict[str, OriginalFile]
    _parent_lease: Any
    _pool_hold: Any
    originals: RustPlayerOriginals
    manifest: Any
    accepted_decisions: int
    closure_counts: Json
    missing_replays: list[str]
    _numeric: Any
    selection: Json

    def __init__(self, issuer: object = None) -> None:
        if issuer is not _SOURCE_ISSUER:
            raise ContractError('actual native/writer Source issuer required')

    def __reduce__(self) -> Any:
        raise TypeError('actual same-host Source authority cannot be serialized')

    @classmethod
    def project(cls, collection: Any) -> RustPlayerSource:
        from gf_common.card_instance_collection import PPO_BATCH, hpo_batch_role
        from gf_training.card_instance_collection_source import inspect_rust_game_manifest
        from gf_training.card_instance_source import _check_collection_parent
        from gf_training.rust_player_collection import RustCollectedBatch
        from gf_training.structured_source import _inspect_collection_content
        if type(collection) is not RustCollectedBatch:
            raise ContractError('actual same-host native/writer collection required')
        registry, plan = collection.registry, collection.plan
        collection.require(registry, plan)
        if registry.pending is not None:
            raise ContractError('actual unused Source transaction required')
        result = cls(_SOURCE_ISSUER)
        result.collection, result.registry, result.plan = collection, registry, plan
        result.parent = registry.current
        result.state, result._rollout = 'INSPECTING', None
        result._numeric = None
        result._charges = []
        result._numeric_originals = {}
        registry.pending = result  # Retain actual partially acquired owner on error.
        result._parent_lease = result.parent.retain(result)
        result._pool_hold = registry.pool.hold_close(result)
        owner = collection.open()
        result.originals = owner
        try:
            manifest = collection.manifest
            actual = registry.admission.observation
            if (digest(manifest) != collection.manifest_sha256
                    or manifest['runtime_binding_sha256'] != registry.state['runtime_binding_sha256']
                    or manifest['player_producer'] != actual['producer']
                    or digest(manifest['capture_authority']) != collection.capture_sha256
                    or digest(manifest['logical_retirement']) != collection.retirement_sha256):
                raise ContractError('actual collection identity/runtime/capture differs')
            _check_collection_parent(manifest, registry.state, plan)
            role = hpo_batch_role(registry.config.hpo_spec['training_recipe']['collection_games']) if registry.hpo else PPO_BATCH
            episodes, specs = inspect_rust_game_manifest(owner, manifest, registry.config.actions,
                registry.config.run.split, role=role)
            outputs, accepted, closure, missing = _inspect_collection_content(owner,
                registry.config.actions, manifest, episodes, specs, _game=True, packed=True)
            seats = plan.linkage['learner_seats']
            records = tuple(item.records for item in outputs if item.viewer == seats[item.episode_id])
            if missing or closure != {'natural': role.games}:
                raise ContractError('actual complete natural curriculum required')
            result.manifest, result._records = manifest, records
            result.accepted_decisions = accepted
            result.closure_counts, result.missing_replays = closure, missing
            if registry.hpo:
                from gf_training.hpo_collection import selection_for_counts
                result.selection = selection_for_counts(tuple(len(record.frames) for record in records),
                    learning_seed=registry.state['seed'], batch=registry.state['updates'],
                    decision_budget=registry.config.hpo_spec['decision_budget'], used_before=registry.state['used_decisions'])
            result.state = 'SEALED'
            return result
        except BaseException as error:
            result.state = 'FAILED'
            vars(error)['cleanup_owner'] = result
            raise

    def require(self) -> None:
        from gf_training.group_pack_resources import remaining
        remaining(self.plan.execution_deadline)
        self.collection.require(self.registry, self.plan)
        self.parent.require()
        if self.registry.pending is not self or self.registry.current is not self.parent or self.state in ('FAILED', 'CLOSED'):
            raise ContractError('actual current single-use Rust Source required')
        if self.registry.state['updates'] != self.parent.generation or self.parent.generation != self.plan.linkage['collection_generation']:
            raise ContractError('actual Source generation differs from current parent')

    def persist(self, directory: Path) -> Json:
        from gf_models.arrays import sha256
        from gf_training.structured_rollout import (
            _owned_bytes,
            _predicted_token_bytes,
            _write_rollout,
            game_rollout,
        )
        self.require()
        if self.state != 'SEALED':
            raise ContractError('actual Source numeric persistence is single use')
        if self.registry.hpo:
            return self._persist_hpo(directory)
        budget = self.registry.budget
        # Originals already fund the parsed graphs. Only new token/rollout and
        # planning owners are admitted here, sharing that same Source account.
        token_charge = budget.reserve(_predicted_token_bytes(self._records))
        self._charges.append(token_charge)
        planning = budget.reserve(0)
        self._charges.append(planning)
        try:
            local = self.registry.local
            self._rollout = game_rollout(self._records, local.learner.model,
                actions=self.registry.config.actions, rules=local.learner._context.rules)
            seen: set[int] = set()
            _owned_bytes(self._records, seen)
            token_charge.resize(_owned_bytes((self._rollout.model, self._rollout.episodes,
                self._rollout.locations, self._rollout.action_config_hash, self._rollout.policy_version,
                self._rollout.input_binding), seen))
            metadata = _write_rollout(directory, self._rollout, check=self.require,
                max_owned_bytes=local.learner.model.resources.max_owned_bytes,
                reserve=lambda size: planning.resize(max(0, size - self._rollout.resident_bytes)))
            self.numeric_directory = directory
            self.numeric_files = {'rollout.json': sha256(directory / 'rollout.json'), **metadata['data']['files']}
            for name, expected in self.numeric_files.items():
                original = OriginalFile(directory / name, self.plan.execution_deadline)
                self._numeric_originals[name] = original
                if original.sha256 != expected:
                    raise ContractError('actual numeric publication differs from completed writer')
            self._numeric_metadata = canonical(metadata)
            self._charges.append(budget.reserve(32 * len(self._numeric_metadata.encode()) + 65536))
            self._episode_digests = self._digests()
            self._proof_json = self._build_proof()
            self.state = 'PERSISTED'
            return metadata
        except BaseException as error:
            self.state = 'FAILED'
            vars(error)['cleanup_owner'] = self
            raise
        finally:
            if self.state == 'PERSISTED':
                planning.close()

    @property
    def decision_count(self) -> int:
        if self.registry.hpo:
            return self.selection['used_decisions']
        return sum(len(record.frames) for record in self._records)

    @property
    def collected_decisions(self) -> int:
        return sum(len(record.frames) for record in self._records)

    def _persist_hpo(self, directory: Path) -> Json:
        from gf_common.hpo import selected_locations
        from gf_training.hpo_owned_rollout import encode_rust_batch
        try:
            self._numeric = encode_rust_batch(self, selected_locations=selected_locations(self.selection),
                selection=self.selection, numeric_budget_bytes=self.registry.config.hpo_spec['operating']['numeric_budget_bytes'])
            written = self._numeric.persist(directory)
            metadata = written['manifest']
            self.numeric_directory, self.numeric_files = directory, written['files']
            for name, expected in self.numeric_files.items():
                original = OriginalFile(directory / name, self.plan.execution_deadline)
                self._numeric_originals[name] = original
                if original.sha256 != expected:
                    raise ContractError('actual owned numeric6 publication checksum differs')
            self._numeric_metadata = canonical(metadata)
            self._charges.append(self.registry.budget.reserve(32 * len(self._numeric_metadata.encode()) + 65536))
            self._episode_digests = self._digests()
            self._proof_json = self._build_proof()
            self.state = 'PERSISTED'
            return metadata
        except BaseException as error:
            self.state = 'FAILED'
            vars(error)['cleanup_owner'] = self
            raise

    def _digests(self) -> tuple[str, ...]:
        from gf_training.structured_rollout import _episode_correspondence_digest
        return tuple(_episode_correspondence_digest(record) for record in self._records)

    def _build_proof(self) -> str:
        from gf_common.artifacts import json_bound
        from gf_models.structured_binding import input_binding_game_v3
        from gf_training.rust_player_lineage import validate_record_shape
        owner, collection, registry = self.originals, self.collection, self.registry
        actual = registry.admission.observation
        assert owner.source is not None
        original = owner.source._originals
        descriptor = original.descriptor
        correspondence = {'recipe': 'rust-player-source-numeric-v1',
            'manifest_sha256': collection.manifest_sha256, 'episode_digests': self._episode_digests,
            'snapshot_identity': self.parent.identity, 'numeric_files': self.numeric_files}
        proof = {'schema_version': 4, 'kind': 'rust-player-source-proof-v1',
            **input_binding_game_v3(registry.config.actions.action_contract, registry.local.learner._context.rules),
            'qualified': False, 'parent_qualification': 'not_performed', 'run_id': registry.state['run_id'],
            'generation': registry.state['updates'], 'batch_id': self.plan.linkage['batch_id'],
            'manifest_sha256': collection.manifest_sha256, 'source_set_sha256': digest(owner.files),
            'files': owner.files, 'games': len(self.manifest['episodes']),
            'accepted_decisions': self.accepted_decisions, 'closure_counts': self.closure_counts,
            'missing_replays': self.missing_replays, 'runtime_binding_sha256': registry.state['runtime_binding_sha256'],
            'player_producer': actual['producer'], 'capture_authority_sha256': collection.capture_sha256,
            'collection_retirement_sha256': collection.retirement_sha256,
            'writer_retirement': {'schema_version': 1, 'kind': 'rust-player-writer-retirement-v1',
                'run_id': registry.state['run_id'], 'batch_id': self.plan.linkage['batch_id'],
                'descriptor_sha256': collection.pending.descriptor_sha256,
                'writer_thread_identity': collection.writer.thread_identity, 'writer_joined': True,
                'cleanup_errors': list(collection.writer.cleanup_errors)},
            'storage': {'format': 'sqlite-group-pack-v1', 'descriptor_path': str(collection.pending.path),
                'descriptor_sha256': collection.pending.descriptor_sha256,
                'database_sha256': descriptor['database_sha256'], 'seal_id': descriptor['seal_id'],
                'logical_sha256': original.validated['seal']['logical_sha256']},
            'snapshot_identity': self.parent.identity, 'episode_digests': self._episode_digests,
            'source_correspondence': digest(correspondence), 'service_usage': self.manifest['service_usage']}
        if registry.hpo:
            from gf_common.hpo import selected_locations
            training = registry.local.learner.training
            locations = selected_locations(self.selection)
            proof.update(schema_version=7, kind='rust-player-hpo-source-proof-v1',
                origin='producer-owned-v1', encoder='canonical-python', selection=self.selection,
                selection_sha256=digest(self.selection), replay_reconciled=True,
                expected_optimizer_step_delta=self._numeric.metadata['expected_optimizer_step_delta'],
                context_costs={'context_decisions': training.epochs * sum(position + 1 for _, position in locations),
                    'suffix_decisions': training.epochs * sum(min(position + 1, training.tbptt_length) for _, position in locations)})
        validate_record_shape(proof, 'proof', hpo=registry.hpo)
        self._charges.append(registry.budget.reserve(32 * json_bound(proof) + 65536))
        return canonical(proof)

    @property
    def proof(self) -> Json:
        """Borrowed metadata, valid only until this actual Source closes."""
        self.require()
        if self.state not in ('PERSISTED', 'UPDATED', 'SAVED'):
            raise ContractError('actual completed numeric persistence required')
        return json.loads(self._proof_json)

    def consume(self, output: Path) -> tuple[Json, Any]:
        """Consume the same privately owned rollout once, then actual local save."""
        self.require()
        if self.state != 'PERSISTED':
            raise ContractError('actual persisted Source is single use')
        self.state = 'VALIDATING'
        try:
            return self._consume(output)
        except BaseException as error:
            self.state, self.registry.failed = 'FAILED', True
            vars(error)['cleanup_owner'] = self
            raise

    def _consume(self, output: Path) -> tuple[Json, Any]:
        registry, local = self.registry, self.registry.local
        numeric = self._numeric if registry.hpo else self._rollout
        if (local.learner.steps != self.parent.optimizer_steps
                or local.learner.model != numeric.model
                or numeric.action_config_hash != digest(asdict(registry.config.actions))
                or self._digests() != self._episode_digests
                or output != registry.directory / 'artifacts' / f'{registry.state["updates"] + 1:06}'):
            raise ContractError('actual unchanged single-use persisted Source/local generation required')
        # OriginalFile.hash uses two alternating 64KiB read buffers at assignment.
        with registry.budget.reserve(128 * 1024):
            for name, original in self._numeric_originals.items():
                original.recheck()
                if original.hash() != self.numeric_files[name]:
                    raise ContractError('actual persisted numeric original hash changed')
                original.recheck()
            if canonical(json.loads(self._numeric_originals['rollout.json'].read(
                    registry.config.preprocessing.max_source_file_mib * 1024**2))) != self._numeric_metadata:
                raise ContractError('actual persisted numeric metadata changed')
        if not registry.hpo:
            self._rollout.require_learning()
        before = local.learner.steps
        self.state = 'UPDATING'  # Failures are terminal; no retry resets Adam/RNG.
        try:
            metrics = local.update_owned(self._numeric) if registry.hpo else local.update(self._rollout)
            from gf_training.group_pack_resources import remaining
            remaining(self.plan.execution_deadline)
            delta = self._numeric.metadata['expected_optimizer_step_delta'] if registry.hpo else 4 * ((self.decision_count + 127) // 128)
            if local.learner.steps != before + delta:
                raise ContractError('actual local optimizer delta differs from unchanged PPO recipe')
            self.state = 'UPDATED'
            pair = local.save(output, **({'save_budget_bytes': registry.config.hpo_spec['operating']['save_budget_bytes']} if registry.hpo else {}))
            pair.require(local)
            self.result_pair, self.metrics = pair, canonical(metrics)
            self.state = 'SAVED'
            return metrics, pair
        except BaseException as error:
            self.state, registry.failed = 'FAILED', True
            vars(error)['cleanup_owner'] = self
            raise

    def require_saved_result(self, registry: Any, pair: Any, update: Json, policy: Any) -> None:
        self.require()
        pair.require(registry.local)
        if (registry is not self.registry or self.state != 'SAVED' or pair is not self.result_pair
                or policy._pair is not pair or policy._budget is not registry.budget
                or policy.optimizer_steps != registry.local.learner.steps
                or policy.generation != registry.state['updates'] + 1
                or update['run_id'] != registry.state['run_id']
                or update['generation'] != policy.generation
                or update['parent_sha256'] != self.parent.generation_sha256
                or update['parent_snapshot_identity'] != self.parent.identity
                or update['result_snapshot_identity'] != policy.identity
                or update['accepted_decisions'] != self.decision_count
                or canonical(update['numeric_files']) != canonical(self.numeric_files)
                or update['numeric_sha256'] != self.numeric_files['rollout.json']
                or update['collection_manifest_sha256'] != self.collection.manifest_sha256
                or canonical(update['source_proof']) != self._proof_json
                or canonical(update['metrics']) != self.metrics
                or update['runtime_binding_sha256'] != registry.state['runtime_binding_sha256']
                or canonical(update['policy_bindings']) != canonical(self.manifest['policy_bindings'])
                or canonical(update['learner_seats']) != canonical(self.plan.linkage['learner_seats'])
                or update['batch_id'] != self.plan.linkage['batch_id']
                or update['actual_games'] != (registry.config.hpo_spec['training_recipe']['collection_games'] if registry.hpo else 16)
                or update['planned_games'] != (registry.config.hpo_spec['training_recipe']['collection_games'] if registry.hpo else 16)
                or update['collection_retirement_sha256'] != self.collection.retirement_sha256):
            raise ContractError('actual saved result differs from SAME consumed Source transaction')

    def commit_completed(self, policy: Any) -> None:
        if self.state != 'SAVED' or policy._pair is not self.result_pair:
            raise ContractError('actual generation completion differs from SAME saved result')
        policy.require()
        self.state = 'COMMITTED'

    def close(self) -> None:
        if self.state == 'CLOSED':
            return
        registry = getattr(self, 'registry', None)
        local = getattr(registry, 'local', None)
        account = getattr(local, 'device_account', None)
        operation = account._active if account is not None else None
        if operation is not None and operation.state == 'FAILED_HELD':
            # The numerical hold can still own copied rollout/token/raw aliases.
            # Keep this SAME actual Source grant and owner until positive device
            # retirement; poison alone does not fund retained CPU graphs.
            raise BufferError('actual failed operation still owns Source aliases/funding')
        self._rollout, self._records, self.manifest = None, (), None
        if self._numeric is not None:
            self._numeric.close()
            self._numeric.require_retired()
            self._numeric = None
        self._proof_json = self._numeric_metadata = ''
        self._episode_digests = ()
        for original in self._numeric_originals.values():
            original.close()
        self._numeric_originals.clear()
        self.collection.close()
        self._parent_lease.close(self)
        for charge in self._charges:
            charge.close()
        self._charges.clear()
        self._pool_hold.retire(self)
        if self.registry.pending is self:
            self.registry.pending = None
        self.state = 'CLOSED'
