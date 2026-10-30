"""Owned native PPO ingestion, with Python learning and pre-update finalization.

The live PPO coordinator owns its lock and numerical child. This adapter owns
only its source snapshot, numeric staging and temporary native resources, all of
which must pass final checks before the coordinator may submit an update.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from gf_common.domain import ContractError, Json, integer
from gf_common.preprocessing import PreparationConfig
from gf_models.config import MLPModelConfig
from gf_training.collection import OnlinePPOContext
from gf_training.lineage import _OWNED_LOCKS
from gf_training.native_authorization import NativePreparation
from gf_training.native_backend import binary_identity, load_native, reader_settings
from gf_training.native_learning import PPOLearning
from gf_training.q_data import check_deadline


def preprocessing_identity(config: PreparationConfig, *, deadline: float) -> Json:
    """Fail before collection and identify the actual loaded native binary."""
    check_deadline(deadline)
    if config.backend == 'python':
        if config.reader != 'legacy':
            raise ContractError('typed preprocessing reader requires the Rust backend')
        return {'backend': 'python', 'reader_contract': config.reader}
    if config.backend != 'rust':
        raise ContractError('unsupported PPO preprocessing backend')
    native = load_native()
    reader_settings(config, native)
    session = getattr(native, 'NativeSession', None)
    if session is None or any(not callable(getattr(session, name, None)) for name in (
            'cancel_handle', 'control_summary', 'authorize', 'next_batch', 'proof', 'finalize', 'close')):
        raise ContractError('native PPO preprocessing implementation is incomplete')
    handle_type = getattr(native, 'CancelHandle', None)
    if not callable(handle_type):
        raise ContractError('native PPO preprocessing cancellation support is incomplete')
    try:
        handle = handle_type(deadline)
    except (TypeError, ValueError) as error:
        raise ContractError('native PPO preprocessing cancellation support is incompatible') from error
    if not callable(getattr(handle, 'cancel', None)):
        raise ContractError('independent native PPO cancellation handle required')
    identity = binary_identity(native)
    check_deadline(deadline)
    return {'backend': 'rust', 'native': identity, 'reader_contract': config.reader}


def _context(directory: Path, linkage: Json, context: OnlinePPOContext) -> Path:
    if not isinstance(context, OnlinePPOContext):
        raise ContractError('owned online PPO context required')
    parent = context.parent_directory.resolve()
    if (_OWNED_LOCKS.get(parent) != context.ownership_token
            or directory.resolve() != parent / 'collections' / context.batch_id
            or linkage.get('parent_run_id') != context.parent_run_id
            or linkage.get('parent_run_path') != str(parent)
            or linkage.get('batch_id') != context.batch_id
            or type(linkage.get('collection_generation')) is not int
            or linkage['collection_generation'] != context.generation):
        raise ContractError('owned online PPO context does not authorize this exact frozen batch')
    return parent


def _source_proof(owner: NativePreparation) -> Json:
    proof = owner.proof()
    if proof.get('structural_valid') is not True or proof.get('learning_valid') is not False:
        raise ContractError('native complete-source structural proof required')
    return proof


async def stage_rollout(directory: Path, numeric: Path, model: MLPModelConfig, linkage: Json,
                  config: PreparationConfig, *, deadline: float, online_context: OnlinePPOContext,
                  expected_identity: Json, wall_deadline: float | None = None) -> tuple[int, Json]:
    """Stage the existing rollout format and close its source owner before return.

    No native error dispatches to the Python reference reader. A successful
    return permits the caller's optimizer request, subject to its original
    deadline/storage checks and unchanged numerical parity/update/save gates.
    """
    check_deadline(deadline)
    if (config.backend != 'rust' or expected_identity.get('backend') != 'rust'
            or expected_identity.get('reader_contract') != config.reader):
        raise ContractError('explicit Rust PPO preprocessing identity required')
    parent = _context(directory, linkage, online_context)
    plan = {'kind': 'online', 'encoder': asdict(model.encoder),
            'collection_path': str(directory.resolve()), 'parent_path': str(parent)}
    learning = PPOLearning(model.encoder)
    layout = learning.consumer_layout()
    async with await NativePreparation.create_async(plan, config, deadline=deadline,
                                                     online_context=online_context, wall_deadline=wall_deadline) as owner:
        if owner.identity != expected_identity.get('native'):
            raise ContractError('native PPO preprocessing binary changed since preflight')
        while True:
            check_deadline(deadline)
            batch = await owner.anext_batch(learning.retained_python_bytes(), layout)
            if batch is None:
                break
            try:
                _source_proof(owner)
                learning.append(batch, global_ready=True)
            finally:
                # One complete-episode chunk at a time. The next admission counts
                # owned NumPy learning arrays; borrowed native arrays stay native.
                del batch
        source = _source_proof(owner)
        rollout = learning.finalize(global_ready=owner.exhausted)
        decisions = len(rollout)
        games = sum(len(chunk.episode_offsets) - 1 for chunk in rollout.chunks)
        integer(source.get('train_decisions'), 1)
        integer(source.get('validation_decisions'), 0, 0)
        integer(source.get('games'), 1)
        if source['train_decisions'] != decisions or source['games'] != games:
            raise ContractError('native PPO source/learning episode or decision count mismatch')
        for name in ('source_set_sha256', 'manifest_sha256'):
            value = source.get(name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(character not in '0123456789abcdef' for character in value)):
                raise ContractError('native PPO source digest proof missing')
        # Import lazily: ppo retains learning authority and the existing shard
        # boundary, and can explicitly use this adapter without an import cycle.
        from gf_training.ppo import write_rollout
        write_rollout(numeric, rollout, deadline=deadline)
        check_deadline(deadline)
        finalized = await owner.afinalize()
        if finalized != source:
            raise ContractError('native PPO final source proof changed')
        if preprocessing_identity(config, deadline=deadline) != expected_identity:
            raise ContractError('native PPO preprocessing binary changed during ingestion')
    return decisions, {'source_set_sha256': source['source_set_sha256'],
                       'manifest_sha256': source['manifest_sha256'],
                       'decisions': decisions, 'games': games}
