"""Explicit offline Raw/Replay/numeric audit under a held parent lock."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json
from gf_training.group_pack_pool import PackedArtifactPool
from gf_training.group_pack_resources import retain_cleanup
from gf_training.lineage import require_hpo_run_owner
from gf_training.rust_player_source import RustPlayerOriginals


class RustDiagnosticOriginals(RustPlayerOriginals):
    """Read-only inspection of originals. Holds no live Source or update capability."""

    def __init__(self, ownership: Any, path: Path, checksum: str, *, pool: Any,
                 settings: Any, deadline: float) -> None:
        self.ownership = require_hpo_run_owner(ownership.directory, ownership)
        self.pending = self._cold = self._resume = None
        self.pool, self.budget = pool, pool.budget
        self._descriptor_path, self._descriptor_sha256 = path, checksum
        self.root, self.settings, self.deadline = path.parent, settings, deadline
        self.source, self._cold_descriptor = None, None
        self.files, self._entries, self._reservations = {}, {}, []
        self._closed = False
        self._cleanup_children = []
        self._open_sealed_originals()

    def _check_full(self) -> None:
        require_hpo_run_owner(self.ownership.directory, self.ownership)
        if self._closed or self.source is None:
            raise ContractError('Rust PPO diagnostic originals have retired')
        self._cold_descriptor.recheck()
        self.source.recheck()


def audit_source_update(ownership: Any, loaded: Any, state: Json, previous: Json, current: Json,
                        update: Json, *, full: bool, max_owned_bytes: int, deadline: float) -> Json | None:
    """Audit a committed update under the verifier's remaining scan allowance."""
    from gf_common.rust_ppo_source import source_mode
    from gf_training.rust_player_completion import _audit_sealed_update, _audit_update_data
    from gf_training.rust_player_lineage import validate_record_shape

    require_hpo_run_owner(ownership.directory, ownership)
    mode = source_mode(state['effective_config'])
    if not full and mode is None:
        return None
    hpo = getattr(loaded.config, 'hpo_spec', {}).get('schema_version') == 6
    proof = validate_record_shape(update['source_proof'], 'proof', hpo=hpo)
    generation = current['generation']
    path = Path(loaded.config.storage.db_root) / state['run_id'] / 'collections' / f'batch-{generation - 1:06}' / 'collection.pending.json'
    if (proof['storage']['descriptor_path'] != str(path) or proof.get('ppo_source') != mode
            or update['generation'] != generation or update['parent_sha256'] != current['parent_sha256']
            or update['parent_snapshot_identity'] != previous['snapshot_identity']
            or update['result_snapshot_identity'] != current['snapshot_identity']
            or proof['snapshot_identity'] != previous['snapshot_identity']
            or proof['runtime_binding_sha256'] != state['runtime_binding_sha256']
            or proof['player_producer'] != state['identity']['player_runtime']['producer']):
        raise ContractError('diagnostic Source evidence differs from committed generation')
    entry = state['invocations'][proof['service_usage']['invocation_id']]
    record_root = Path(entry['operation_path']).parent
    pool = PackedArtifactPool(max_owned_bytes=max_owned_bytes, deadline=deadline)
    loans: list[Any] = []
    try:
        observed = None
        if mode is not None:
            fact = _audit_sealed_update(ownership, pool, loaded, state, previous, current, update, deadline,
                expected=path, record_root=record_root, control_loans=loans)
            observed = fact['pack_verification']
        if full:
            originals = RustDiagnosticOriginals(ownership, path, proof['storage']['descriptor_sha256'],
                pool=pool, settings=loaded.config.preprocessing, deadline=deadline)
            _audit_update_data(originals, ownership, pool, loaded, state, previous, current, update,
                deadline, record_root=record_root, control_loans=loans)
        return observed
    except BaseException as error:
        retain_cleanup(pool, error)
        raise
    finally:
        for loan in loans:
            loan.close()
        pool.close()


def audit_empty_source(ownership: Any, loaded: Any, state: Json, records: list[Json],
                       marker: Json, admission: Json, *, full: bool, max_owned_bytes: int,
                       deadline: float) -> Json | None:
    from gf_common.rust_ppo_source import source_mode
    from gf_training.rust_player_completion import _audit_empty_originals, _audit_sealed_empty
    from gf_training.rust_player_lineage import validate_empty_collection

    require_hpo_run_owner(ownership.directory, ownership)
    validate_empty_collection(marker, state=state, config=loaded.config)
    if (ownership.directory / 'numeric' / marker['batch_id']).exists():
        raise ContractError('empty batch must not have a numeric payload')
    current = records[-1]
    policies = {generation: records[generation]['inference_sha256']
        for generation in set(current['parent_controls']['snapshot_pool']) | {current['generation']}}
    pool = PackedArtifactPool(max_owned_bytes=max_owned_bytes, deadline=deadline)
    try:
        observed = None
        if source_mode(state['effective_config']) is not None:
            observed = _audit_sealed_empty(pool, loaded, state, current, marker, admission,
                parent=ownership.directory, inference_files=policies, deadline=deadline)
        if full:
            originals = RustDiagnosticOriginals(ownership, Path(marker['descriptor_path']), marker['descriptor_sha256'],
                pool=pool, settings=loaded.config.preprocessing, deadline=deadline)
            _audit_empty_originals(originals, loaded=loaded, state=state, current=current, marker=marker,
                admission=admission, parent=ownership.directory, inference_files=policies)
        return observed
    except BaseException as error:
        retain_cleanup(pool, error)
        raise
    finally:
        pool.close()
