"""CPU numeric DATA and real reservations; fixtures grant no Source authority."""
from contextlib import contextmanager

import numpy as np
import pytest

from gf_common.domain import ContractError
from gf_models.arrays import MAX_ARRAY_BYTES
from gf_models.sharded_arrays import read_bundle


@contextmanager
def numeric_data():
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import captured_record
    learner = learner_for()
    batch, owner, plan, snapshot, account, _record = captured_record(learner)
    selection = selection_for_counts((3,), learning_seed=17, batch=0, decision_budget=3, used_before=0)
    numeric = encode_owned_batch(batch, owner=owner, plan=plan, snapshot_owner=snapshot,
        model=learner.model, training=learner.training, actions=ACTIONS, rules=RULES,
        selected_locations=((0, 0), (0, 1), (0, 2)), selection=selection, budget=account)
    try:
        yield numeric, account
    finally:
        numeric.close()
        numeric.require_retired()
        batch.close()
        snapshot.close()


def geometry(numeric):
    largest = max(d.nbytes for bank in numeric.banks.banks for d in bank.descriptors)
    base = 32 * len(numeric._metadata_bytes) + 65536
    assert 0 < largest < numeric.banks.size < MAX_ARRAY_BYTES
    return largest, base


def test_source_quota_keeps_existing_default_when_it_fits(tmp_path, monkeypatch):
    from gf_training.hpo_owned_rollout import OwnedNumericBatch
    original = OwnedNumericBatch.persist
    calls = []
    def persist(self, directory, **kwargs):
        calls.append(kwargs.get('shard_bytes', MAX_ARRAY_BYTES))
        return original(self, directory, **kwargs)
    monkeypatch.setattr(OwnedNumericBatch, 'persist', persist)
    with numeric_data() as (numeric, account):
        _, base = geometry(numeric)
        needed = base + 35 * numeric.banks.size
        occupancy = account.reserve(account.limit - account.used - needed)
        try:
            output = numeric._persist_within_source_quota(tmp_path / 'numeric',
                source_remaining_bytes=account.limit-account.used)
        finally:
            occupancy.close()
        assert calls == [MAX_ARRAY_BYTES]
        assert len(output['manifest']['data']['shards']) == 1
        assert numeric._state == 'PERSISTED'


@pytest.mark.parametrize('binding', ['source', 'numeric'])
def test_narrowed_cap_fits_both_actual_envelopes_and_decodes_identical_arrays(tmp_path, monkeypatch, binding):
    from gf_training.hpo_owned_rollout import OwnedNumericBatch
    original = OwnedNumericBatch.persist
    calls = []
    def persist(self, directory, **kwargs):
        calls.append(kwargs['shard_bytes'])
        return original(self, directory, **kwargs)
    monkeypatch.setattr(OwnedNumericBatch, 'persist', persist)
    with numeric_data() as (numeric, account):
        largest, base = geometry(numeric)
        remaining = base + 35 * largest + 34  # Fractional final byte cannot fund a larger cap.
        occupancy = account.reserve(account.limit - account.used - remaining) if binding == 'source' else None
        if binding == 'numeric':
            numeric._numeric_limit = numeric.resident_bytes + remaining
        before = account.used
        original_array = numeric.banks.array
        array_states = []
        def view(name):
            array_states.append(numeric._state)
            return original_array(name)
        monkeypatch.setattr(numeric.banks, 'array', view)
        try:
            output = numeric._persist_within_source_quota(tmp_path / 'numeric',
                source_remaining_bytes=account.limit-account.used)
            assert calls == [largest]
            assert array_states and set(array_states) == {'FAILED'}  # No planning-time array access/copy.
            assert account.used == before  # Only the ordinary scratch interval was reserved/refunded.
            assert len(output['manifest']['data']['shards']) > 1
            monkeypatch.setattr(numeric.banks, 'array', original_array)
            decoded = read_bundle(tmp_path / 'numeric', output['manifest']['data'])
            assert set(decoded) == set(numeric.banks.names)
            for name, actual in decoded.items():
                np.testing.assert_array_equal(actual, numeric.banks.array(name))
            assert output['manifest']['schema_version'] == 6
            with pytest.raises(ContractError, match='single use'):
                numeric._persist_within_source_quota(tmp_path / 'again', source_remaining_bytes=remaining)
        finally:
            if occupancy is not None:
                occupancy.close()


@pytest.mark.parametrize('binding', ['source', 'numeric'])
def test_impossible_single_array_refuses_before_any_output_or_persist_attempt(tmp_path, monkeypatch, binding):
    from gf_training.hpo_owned_rollout import OwnedNumericBatch
    def forbidden(*args, **kwargs):
        raise AssertionError('impossible plan attempted persistence')
    monkeypatch.setattr(OwnedNumericBatch, 'persist', forbidden)
    with numeric_data() as (numeric, account):
        largest, base = geometry(numeric)
        remaining = base + 35 * largest - 1
        occupancy = account.reserve(account.limit-account.used-remaining) if binding == 'source' else None
        if binding == 'numeric':
            numeric._numeric_limit = numeric.resident_bytes + remaining
        before = account.used
        try:
            with pytest.raises(ContractError, match='persistence'):
                numeric._persist_within_source_quota(tmp_path / 'impossible',
                    source_remaining_bytes=account.limit-account.used)
            assert numeric._state == 'FAILED'
            assert not (tmp_path / 'impossible').exists()
            assert account.used == before
            with pytest.raises(ContractError, match='single use'):
                numeric._persist_within_source_quota(tmp_path / 'retry', source_remaining_bytes=account.limit)
        finally:
            if occupancy is not None:
                occupancy.close()


def test_existing_legacy_persist_still_refuses_shared_source_pressure(tmp_path):
    with numeric_data() as (numeric, account):
        largest, base = geometry(numeric)
        remaining = base + 35 * largest
        occupancy = account.reserve(account.limit-account.used-remaining)
        try:
            with pytest.raises(ValueError, match='storage budget exhausted'):
                numeric.persist(tmp_path / 'legacy')
            assert numeric._state == 'FAILED'
            assert not (tmp_path / 'legacy').exists()
        finally:
            occupancy.close()


def test_capacity_loss_after_planning_fails_actual_reserve_without_retry(tmp_path, monkeypatch):
    from gf_training.hpo_owned_rollout import OwnedNumericBatch
    original = OwnedNumericBatch.persist
    attempts = []
    competing = []
    def persist(self, directory, **kwargs):
        attempts.append(kwargs['shard_bytes'])
        competing.append(self._budget.reserve(1))
        return original(self, directory, **kwargs)
    monkeypatch.setattr(OwnedNumericBatch, 'persist', persist)
    with numeric_data() as (numeric, account):
        largest, base = geometry(numeric)
        remaining = base + 35 * largest
        occupancy = account.reserve(account.limit-account.used-remaining)
        try:
            with pytest.raises(ValueError, match='storage budget exhausted'):
                numeric._persist_within_source_quota(tmp_path / 'lost', source_remaining_bytes=remaining)
            assert attempts == [largest]
            assert numeric._state == 'FAILED'
            assert not (tmp_path / 'lost').exists()
        finally:
            for reservation in competing:
                reservation.close()
            occupancy.close()
