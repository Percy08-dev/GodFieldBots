"""Wire6 closed numerical facts; shared worker/lifecycle owns execution."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.hpo import checked_selection, optimizer_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.structured_binding import execution_binding_v4
from gf_training.hpo_config import HPOTrainingConfig, hpo_training_config

HPO_PROJECTION_KEYS = {'recipe', 'parent_run_id', 'batch_id', 'collection_generation', 'expected_optimizer_steps',
    'expected_policy_version', 'collection_manifest_sha256', 'numeric_manifest_sha256', 'numeric_files_sha256',
    'source_correspondence', 'training_recipe', 'selection', 'collected_decisions', 'used_decisions',
    'expected_optimizer_step_delta'}


def wire_identity(model: Any, training: HPOTrainingConfig, binding: Any, action_hash: str,
                  resume_identity: str, *, actions: Any, cuda_visible_devices: str | None,
                  storage_binding: Json | None = None, phase_limits: Json | None = None) -> Json:
    from gf_training.hpo_artifacts import checked_phase_limits
    from gf_training.structured_numeric_worker import wire_identity as legacy_identity
    # The shared identity establishes all public action/input/visibility fields;
    # the exact new model version selects its new execution descriptor explicitly.
    raw = legacy_identity(model, binding, action_hash, resume_identity, action_contract=actions.action_contract,
        purpose='qualified_game', synthetic_capacity=False, cuda_visible_devices=cuda_visible_devices,
        _actions=actions, storage_binding=storage_binding, _hpo=True)
    recipe = hpo_training_config(asdict(training))
    raw.update(training_recipe=asdict(recipe), phase_limits=checked_phase_limits(phase_limits), execution_digest=execution_binding_v4(model, binding,
        layout=LAYOUT, cuda_visible_devices=raw['execution']['cuda_visible_devices']))
    return raw


def digest_projection(value: Json, **_: Any) -> str:
    return digest({'recipe': 'card-instance-hpo-update-transaction-v1', 'projection': value})


def validate_projection(value: Json, *, expected_steps: int, policy: str, batch_digest: str,
                         training: HPOTrainingConfig, storage_binding: Json | None = None) -> Json:
    from gf_training.structured_numeric_worker import checksum
    closed(value, HPO_PROJECTION_KEYS)
    selection = checked_selection(value['selection'])
    recipe = hpo_training_config(value['training_recipe'])
    if canonical(asdict(recipe)) != canonical(asdict(training)):
        raise ContractError('wire6 exact training recipe mismatch')
    used = integer(value['used_decisions'], 1)
    collected = integer(value['collected_decisions'], used)
    delta = optimizer_delta(used, epochs=recipe.epochs, minibatch_size=recipe.minibatch_size)
    if (value['recipe'] != 'card-instance-hpo-source-update-v1'
            or type(value['expected_optimizer_steps']) is not int or value['expected_optimizer_steps'] != expected_steps
            or value['expected_policy_version'] != policy or value['numeric_manifest_sha256'] != checksum(batch_digest)
            or value['used_decisions'] != selection['used_decisions'] or collected != selection['collected_decisions']
            or type(value['expected_optimizer_step_delta']) is not int or value['expected_optimizer_step_delta'] != delta):
        raise ContractError('wire6 frozen Source counts/steps/policy mismatch')
    batch = integer(value['collection_generation'], 0, 2**63 - 1)
    if (value['batch_id'] != f'batch-{batch:06}' or selection['context']['batch'] != batch
            or type(value['parent_run_id']) is not str or not value['parent_run_id']
            or len(selection['episode_counts']) != recipe.collection_games):
        raise ContractError('wire6 complete original batch correspondence mismatch')
    for key in ('collection_manifest_sha256', 'numeric_files_sha256'):
        checksum(value[key])
    correspondence = closed(value['source_correspondence'], {'recipe', 'control_fingerprint', 'semantics_sha256',
        'episode_count', 'episode_digests', 'binding_sha256'})
    expected_recipe = ('card-instance-hpo-packed-source-correspondence-v1' if storage_binding is not None else
        'card-instance-hpo-source-correspondence-v1')
    if (correspondence['recipe'] != expected_recipe or type(correspondence['episode_count']) is not int
            or correspondence['episode_count'] != recipe.collection_games):
        raise ContractError('wire6 original complete episode correspondence mismatch')
    return value
