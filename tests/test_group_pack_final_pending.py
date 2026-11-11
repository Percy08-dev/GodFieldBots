"""Private pending physical diagnostics and refusals, with no qualified Source."""
import time

import pytest

from gf_common.domain import ContractError
from gf_common.group_pack_game import storage_binding
from gf_common.preprocessing import PreparationConfig
from gf_training.group_pack_game_source import PackedSourceOwner
from gf_training.group_pack_source import PackedSource, open_artifact_source
from tests.test_group_pack_storage import submit, writer


def test_private_physical_pending_diagnostic_does_not_expand_public_locator_authority(tmp_path):
    owned = writer(tmp_path)
    submit(owned, 'first')
    owned.store_control('manifest.json', b'{}\n')
    locator = owned.seal(publication='pending')
    deadline = time.monotonic() + 30
    with pytest.raises(ValueError, match='pending'):
        open_artifact_source(locator, deadline=deadline, max_owned_bytes=512 * 1024**2)
    source = PackedSource(locator.path, deadline=deadline, max_owned_bytes=512 * 1024**2, _allow_pending=True)
    try:
        assert source.read('manifest.json') == b'{}\n'
        source.recheck()
        assert not (locator.path.parent / 'collection.published.json').exists()
    finally:
        source.close()
    with pytest.raises(ValueError, match='pending'):
        open_artifact_source(locator, deadline=deadline, max_owned_bytes=512 * 1024**2)


def test_flag_or_arbitrary_pending_token_cannot_admit_learning_source_before_effects(tmp_path):
    output = tmp_path / 'unopened'
    with pytest.raises(ContractError):
        PackedSourceOwner(output, PreparationConfig(), time.monotonic() + 30,
            storage_binding=storage_binding(), max_owned_bytes=512 * 1024**2, _issued_pending=object())
    assert not output.exists()


def test_packed_python_backend_check_is_eligibility_neutral():
    from types import SimpleNamespace

    from gf_training.card_instance_final_validation import require_actor_backend
    loaded = SimpleNamespace(config=SimpleNamespace(storage=storage_binding(),
        preprocessing=PreparationConfig(reader='participant-structured-v2')))
    assert require_actor_backend(loaded) is None


def test_packed_final_facts_are_closed_and_keep_false_qualification():
    from gf_common.card_instance_collection import PACKED_FINAL_FACTS_FIELDS, packed_final_facts
    from tests.test_group_pack_dataset_contract import packed_dto
    row = packed_dto()['collections'][1]
    base = {'directory': row['directory'], 'manifest_sha256': 'a' * 64, 'parent_run_id': 'UNADMITTED',
        'final_generation': 1, 'final_inference_sha256': 'b' * 64, 'config_projection': {},
        'validation_invocation': {}, 'elapsed_seconds': 1., 'games': 64, 'selected_decisions': 0,
        'zero_endpoint_games': 64, 'qualified': False, 'parent_qualification': 'not_performed'}
    facts = packed_final_facts(base, storage=row['storage'], custody=row['custody'], source_set_sha256='c' * 64)
    assert set(facts) == PACKED_FINAL_FACTS_FIELDS
    assert facts['schema_version'] == 2 and facts['kind'] == 'card-instance-packed-final-validation-facts'
    assert facts['qualified'] is False and facts['games'] == facts['zero_endpoint_games'] == 64
    with pytest.raises(ContractError):
        packed_final_facts({**base, 'extra': True}, storage=row['storage'], custody=row['custody'], source_set_sha256='c' * 64)
    with pytest.raises(ContractError):
        packed_final_facts({**base, 'qualified': True}, storage=row['storage'], custody=row['custody'], source_set_sha256='c' * 64)
