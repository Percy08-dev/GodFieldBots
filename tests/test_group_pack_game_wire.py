from dataclasses import asdict

import pytest

from gf_common.domain import ContractError, digest
from tests.test_card_instance_learning import ACTIONS, learner_for


def test_wire5_identity_and_paired_projection_recipes() -> None:
    from gf_common.group_pack_game import storage_binding
    from gf_training.structured_numeric_worker import (
        digest_projection,
        validate_projection,
        wire_identity,
    )
    learner = learner_for()
    args = {'action_contract': ACTIONS.action_contract, 'purpose': 'qualified_game', 'synthetic_capacity': False,
        'cuda_visible_devices': None, '_actions': ACTIONS}
    legacy = wire_identity(learner.model, learner.binding, digest(asdict(ACTIONS)), 'a' * 64, **args)
    actual = wire_identity(learner.model, learner.binding, digest(asdict(ACTIONS)), 'a' * 64,
        storage_binding=storage_binding(), **args)
    assert actual == {**legacy, 'storage_binding': storage_binding()}
    correspondence = {'recipe': 'card-instance-packed-source-correspondence-v1',
        'control_fingerprint': 'b' * 64, 'semantics_sha256': 'c' * 64, 'episode_count': 16,
        'episode_digests': ['d' * 64] * 16}
    correspondence['binding_sha256'] = digest(correspondence)
    projection = {'recipe': 'card-instance-packed-source-update-v1', 'parent_run_id': 'fixture',
        'batch_id': 'batch-000000', 'collection_generation': 0, 'expected_optimizer_steps': 0,
        'expected_policy_version': 'fixture-policy', 'collection_manifest_sha256': 'e' * 64,
        'numeric_manifest_sha256': 'f' * 64, 'numeric_files_sha256': '0' * 64,
        'source_correspondence': correspondence}
    options = {'expected_steps': 0, 'policy': 'fixture-policy', 'batch_digest': 'f' * 64}
    assert validate_projection(projection, storage_binding=storage_binding(), **options) == projection
    with pytest.raises(ContractError):
        validate_projection(projection, **options)
    assert digest_projection(projection, storage_binding=storage_binding()) == digest({
        'recipe': 'card-instance-packed-update-transaction-v1', 'projection': projection})
    mixed = {**projection, 'source_correspondence': {**correspondence, 'recipe': 'structured-source-cohort-correspondence-v2'}}
    with pytest.raises(ContractError):
        validate_projection(mixed, storage_binding=storage_binding(), **options)


def test_resume_variant_binds_storage_without_changing_old_recipe() -> None:
    from gf_common.group_pack_game import storage_binding
    from gf_training.structured_lineage import resume_identity
    args = ('a' * 64, {'code': 'fixture'}, {'learner': 'fixture'}, {'purpose': 'qualified_game'})
    assert resume_identity(*args) == digest({'recipe': 'structured-ppo-resume-identity-v1',
        'config_sha256': args[0], 'identity': args[1], 'execution': args[2], 'input_binding': args[3]})
    assert resume_identity(*args, storage_binding=storage_binding()) == digest({
        'recipe': 'card-instance-packed-ppo-resume-identity-v1', 'config_sha256': args[0],
        'identity': args[1], 'execution': args[2], 'input_binding': args[3], 'storage_binding': storage_binding()})
