"""Shared read-only DATA core: real pool/parent, no fabricated Cold capability."""
import time
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from gf_training.group_pack_pool import PackedArtifactPool
from gf_training.lineage import run_lock

MIB = 1024**2


def isolated_core(tmp_path, monkeypatch, *, no_remaining=False, hpo=True):
    from gf_models.config import PPOTrainingConfig
    from gf_training import hpo_rollout, structured_rollout
    from gf_training import rust_player_completion as module
    from tests.test_card_instance_learning import ACTIONS
    pool = PackedArtifactPool(max_owned_bytes=128 * MIB, deadline=time.monotonic()+30)
    retained = pool.budget.reserve(4 * MIB)
    raw = pool.budget.reserve(8 * MIB)
    filled = []
    def retire_raw(*args, **kwargs):
        raw.close()
        if no_remaining:
            filled.append(pool.budget.reserve(pool.budget.limit-pool.budget.used))
        return 3, 3, ('a' * 64,), '2' * 32
    def device_plan(*args, loans, **kwargs):
        if not no_remaining:
            loans.append(pool.budget.reserve(MIB))
    monkeypatch.setattr(module, '_audit_raw_originals', retire_raw)
    monkeypatch.setattr(module, '_audit_source_device_plan', device_plan)
    monkeypatch.setattr(structured_rollout, '_hpo_semantics', lambda *a, **k: {})
    monkeypatch.setattr(structured_rollout, '_source_correspondence', lambda *a, **k: {})
    if not hpo:
        monkeypatch.setattr('gf_models.structured_artifacts.semantics_v3', lambda *a, **k: {})
    monkeypatch.setattr('gf_training.hpo_config.hpo_training_config', lambda value: None)
    loaded = SimpleNamespace(config=SimpleNamespace(storage=SimpleNamespace(db_root=tmp_path),
        actions=ACTIONS, model=SimpleNamespace(architecture=lambda **kwargs: None), training=PPOTrainingConfig(),
        hpo_spec={'schema_version': 6, 'training_recipe': {}, 'operating': {'numeric_budget_bytes': MIB}} if hpo else {}))
    update = {'source_proof': {'selection': {}, 'snapshot_identity': 'snapshot',
        'manifest_sha256': 'a' * 64, 'source_set_sha256': 'b' * 64},
        'numeric': 'numeric/batch-000000', 'numeric_files': {}, 'numeric_sha256': 'c' * 64}
    if not hpo:
        update.update(planned_games=16, actual_games=16, metrics={'optimizer_steps': 4})
    def invoke(owner):
        return module._audit_update_data(object(), owner, pool, loaded, {'run_id': 'run'},
            {'optimizer_steps': 0, 'accepted_decisions': 0},
            {'generation': 1, 'optimizer_steps': 4, 'accepted_decisions': 3}, update,
            pool.deadline, record_root=tmp_path, control_loans=[])
    def close():
        raw.close()
        for loan in filled: loan.close()
        retained.close()
        pool.close()
    return invoke, pool, retained, raw, close, hpo_rollout, structured_rollout


def test_hpo_reader_gets_actual_remaining_pool_after_raw_and_control_retirement(tmp_path, monkeypatch):
    invoke, pool, retained, raw, close, hpo_reader, _ = isolated_core(tmp_path, monkeypatch)
    baseline = pool.budget.used - 8 * MIB
    class ReadbackData:
        def require_learning(self):
            assert pool.budget.used == baseline + 2 * MIB
        def __len__(self): return 3
    def read(*args, numeric_budget_bytes, max_owned_bytes, reserve, **kwargs):
        assert raw.closed and not retained.closed
        assert pool.budget.used == baseline
        assert numeric_budget_bytes == pool.budget.limit - baseline
        assert numeric_budget_bytes > MIB  # Recorded live bank cap is not this reconstruction cap.
        assert max_owned_bytes == pool.budget.limit
        reserve(2 * MIB)
        return ReadbackData()
    monkeypatch.setattr(hpo_reader, 'read_hpo_rollout', read)
    try:
        with run_lock(tmp_path) as owner:
            result = invoke(owner)
        assert result['learner_decisions'] == 3
        assert pool.budget.used == baseline
    finally:
        close()
    assert pool.budget.used == 0


def test_hpo_no_remaining_workspace_refuses_before_reader_or_new_refund(tmp_path, monkeypatch):
    invoke, pool, retained, raw, close, hpo_reader, _ = isolated_core(tmp_path, monkeypatch, no_remaining=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('no remaining Source still entered decoder')
    monkeypatch.setattr(hpo_reader, 'read_hpo_rollout', forbidden)
    try:
        with run_lock(tmp_path) as owner, pytest.raises(ContractError, match='remaining'):
            invoke(owner)
        assert raw.closed and not retained.closed
        assert pool.budget.used == pool.budget.limit
    finally:
        close()
    assert pool.budget.used == 0


def test_hpo_remaining_cap_keeps_actual_resize_authoritative_and_failed_loan_owned(tmp_path, monkeypatch):
    invoke, pool, retained, raw, close, hpo_reader, _ = isolated_core(tmp_path, monkeypatch)
    baseline = pool.budget.used - 8 * MIB
    def read(*args, numeric_budget_bytes, reserve, **kwargs):
        assert numeric_budget_bytes == pool.budget.limit-pool.budget.used
        reserve(numeric_budget_bytes+1)
        raise AssertionError('oversized actual reservation accepted')
    monkeypatch.setattr(hpo_reader, 'read_hpo_rollout', read)
    try:
        with run_lock(tmp_path) as owner, pytest.raises(ValueError, match='storage budget exhausted') as failed:
            invoke(owner)
        assert raw.closed and not retained.closed
        assert pool.budget.used == baseline
        failed.value.cleanup_owner.close()
        failed.value.cleanup_owner.close()
        assert pool.budget.used == baseline
    finally:
        close()
    assert pool.budget.used == 0


def test_ordinary_reader_keeps_existing_arguments_and_actual_resize(tmp_path, monkeypatch):
    invoke, pool, retained, raw, close, _, reader = isolated_core(tmp_path, monkeypatch, hpo=False)
    baseline = pool.budget.used - 8 * MIB
    class ReadbackData:
        def require_learning(self):
            assert pool.budget.used == baseline + 4096
        def __len__(self): return 3
    def read(*args, max_owned_bytes, reserve, **kwargs):
        assert raw.closed and not retained.closed
        assert 'numeric_budget_bytes' not in kwargs
        assert max_owned_bytes == pool.budget.limit
        reserve(4096)
        return ReadbackData()
    monkeypatch.setattr(reader, 'read_game_rollout', read)
    try:
        with run_lock(tmp_path) as owner:
            assert invoke(owner)['learner_decisions'] == 3
        assert pool.budget.used == baseline
    finally:
        close()
    assert pool.budget.used == 0
