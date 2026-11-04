"""Logical admission and real typed feature buffers; no Source or worker authority."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gf_common.domain import ContractError
from gf_models.structured_tree_config import ConfiguredTreeModelV3, resource_estimate
from gf_training.card_instance_tree_fixture import FixtureSelectedInput
from gf_training.native_card_instance_fixture import (
    LIMIT,
    _admit_feature_work,
    _admit_joint_work,
    _arrays,
    _FacadeLoop,
    _feature_parity,
    _reserve_facade_sink,
    _tokens,
)

GIB = 1024**3
JOINT = 8*GIB
MODEL = ConfiguredTreeModelV3()
FIXED = resource_estimate(MODEL)['total_bytes']


class InertOwnership:
    """Accounting spy only: cannot be passed as a concrete production Source."""
    def __init__(self, intrinsic: int, caller: int = 0, sink: int = 0) -> None:
        self.intrinsic, self.caller, self.sink = intrinsic, caller, sink
        self.external = 0
        self.reservations: list[int] = []

    @property
    def retained_bytes(self) -> int:
        return self.intrinsic+self.caller+self.sink+self.external

    def retained_without_sink(self) -> int:
        return self.retained_bytes-self.sink

    def reserve_sink(self, amount: int) -> None:
        assert amount >= self.sink
        self.reservations.append(amount)
        self.sink = amount

    def admit_external(self, amount: int) -> None:
        self.external = amount


def test_full_original_runtime_fits_exact_joint_boundary_preserving_caller_and_existing_sink():
    existing, caller = 4096, 346456948
    reserved = 2*MODEL.resources.max_owned_bytes
    base = JOINT-FIXED-16*1024**2-reserved
    source = InertOwnership(base-caller-existing,caller,existing)
    result = _reserve_facade_sink(source,MODEL,0,JOINT,0,FIXED)
    assert MODEL.resources.max_owned_bytes == 2*GIB and reserved == 4*GIB
    assert result['controller_bytes'] == result['child_bytes'] == 2*GIB
    assert result['existing_source_sink_bytes'] == existing
    assert result['source_sink_high_water_bytes'] == source.sink == existing+reserved
    assert source.caller == caller and source.retained_bytes+FIXED+16*1024**2 == JOINT
    assert source.reservations == [existing+reserved]


def test_one_byte_over_joint_refuses_full_facade_before_any_reservation_or_constructor():
    reserved = 2*MODEL.resources.max_owned_bytes
    source = InertOwnership(JOINT-FIXED-16*1024**2-reserved+1)
    before = source.retained_bytes
    with pytest.raises(ContractError,match='before construction'):
        _reserve_facade_sink(source,MODEL,0,JOINT,0,FIXED)
    assert source.retained_bytes == before and not source.reservations
    assert MODEL.resources.max_owned_bytes == 2*GIB


def test_separately_supplied_caller_bytes_are_not_lost_or_confused_with_already_charged_ownership():
    reserved = 2*MODEL.resources.max_owned_bytes
    extra = 123456
    source = InertOwnership(JOINT-FIXED-16*1024**2-reserved-extra,caller=extra)
    with pytest.raises(ContractError,match='before construction'):
        _reserve_facade_sink(source,MODEL,0,JOINT,extra,FIXED)
    assert not source.reservations
    # The actual outer wrapper charges its caller in Source and passes extra=0.
    _reserve_facade_sink(source,MODEL,0,JOINT,0,FIXED)
    assert source.caller == extra and source.sink == reserved


def test_joint_fitting_and_exact_one_byte_over_are_additional_to_local_runtime():
    local = 1024
    external = 23456+local+2*LIMIT
    source = InertOwnership(JOINT-FIXED-external)
    _admit_feature_work(MODEL,0,local,23456,
        lambda value:_admit_joint_work(source,FIXED,JOINT,value))
    assert source.retained_bytes+FIXED == JOINT
    source.intrinsic += 1
    with pytest.raises(ContractError,match='joint diagnostic'):
        _admit_feature_work(MODEL,0,local,23456,
            lambda value:_admit_joint_work(source,FIXED,JOINT,value))


def test_local_fitting_and_exact_one_byte_over_cannot_bypass_local_runtime_with_joint_headroom():
    maximum = MODEL.resources.max_owned_bytes-FIXED
    source = InertOwnership(4*GIB)
    _admit_feature_work(MODEL,0,maximum,0,
        lambda value:_admit_joint_work(source,FIXED,JOINT,value))
    assert source.retained_bytes+FIXED < JOINT
    previous = source.external
    with pytest.raises(ContractError,match='Tree3 array/joint'):
        _admit_feature_work(MODEL,0,maximum+1,0,
            lambda _:pytest.fail('local overbudget must refuse before global work or allocation'))
    assert source.external == previous


def test_unreaped_worker_failed_borrow_exit_retains_full_source_sink_and_same_cleanup_owner():
    source = InertOwnership(123456,caller=346456948,sink=1024)
    reservation = _reserve_facade_sink(source,MODEL,0,JOINT,0,FIXED)
    loop = _FacadeLoop(source,reservation['added_source_sink_bytes'])
    async def close():
        raise OSError('unconfirmed inert child reap')
    worker = SimpleNamespace(close=close,health={'returncode':None})
    loop.worker = worker
    primary = OSError('initiating feature comparison failed')
    try:
        source.external = 1234567
        loop.__exit__(type(primary),primary,None)
        source.external = 0  # Failed borrow exit discards only its transient charge.
        assert primary._fixture_scoring_cleanup_owner is loop
        assert loop.source is source and loop.reserved_bytes == 4*GIB and loop.worker is worker
        assert source.sink == 1024+4*GIB and source.caller == 346456948
        assert source.retained_bytes == 123456+346456948+1024+4*GIB
        async def confirmed():
            return None
        worker.close = confirmed
        worker.health = {'returncode':-15}
        loop.close()
        assert loop.worker is None and loop.loop.is_closed()
        assert source.sink == 1024+4*GIB  # Named conservative Source-lifetime quota.
    finally:
        if not loop.loop.is_closed():loop.loop.close()


@pytest.mark.parametrize('case_index',[0,-1])
def test_typed_feature_parity_uses_relevant_local_inputs_and_jointly_counts_them_without_source_permission(case_index):
    literal = json.loads((Path(__file__).parents[1]/'native/tests/fixtures/card_instance_component_tokens_python.json').read_text())
    case = literal['cases'][case_index]
    metadata = case['metadata']
    model = replace(MODEL,encoder=replace(MODEL.encoder,aggregate_window=metadata['aggregate_window']))
    banks = tuple(tuple(np.frombuffer(bytes.fromhex(value),dtype=dtype) for value in case['banks_hex'][dtype])
        for dtype in ('uint8','int64','float32'))
    arrays = _arrays(metadata,banks)
    references = []
    for frame in metadata['decisions']:
        tokens,candidates = _tokens(metadata,frame,arrays)
        sample = FixtureSelectedInput(tokens,candidates,frame['selected_index'],metadata['label'],metadata['population_id'],
            metadata['cohort_id'],metadata['episode'],metadata['split'],metadata['viewer'],frame['decision'],metadata['selected_policy_id'])
        references.append(SimpleNamespace(sample=sample))
    source = InertOwnership(5*GIB)
    borrow = SimpleNamespace(retained_bytes=source.retained_bytes,reference_decisions=lambda:iter(references))
    peaks = []
    def admit(value):
        _admit_joint_work(source,FIXED,JOINT,value)
        peaks.append(source.retained_bytes+FIXED)
    result = _feature_parity(borrow,metadata,arrays,model,None,123456,admit)
    assert result == (sum(len(reference.sample.candidates.kinds) for reference in references),0,0)
    assert peaks and 5*GIB < max(peaks) < JOINT
    assert source.retained_bytes > MODEL.resources.max_owned_bytes  # Not a local feature charge.
