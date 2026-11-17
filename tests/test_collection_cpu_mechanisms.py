"""Bounded tools-only ownership and transfer mechanics."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gf_common.domain import ContractError


def test_proof_requires_success_exact_objects_task_thread_and_revision():
    from benchmarks.collection_cpu.parent import DecisionProof
    async def check():
        frame = SimpleNamespace(actor=1, revision=3)
        table = object()
        calls = []
        proof = DecisionProof()
        def original(t, f):
            calls.append((t, f))
        with proof.decision():
            proof.validate(original, table, frame)
            proof.validate(original, table, frame)
            assert len(calls) == 1
            with proof.suspended():
                proof.validate(original, table, frame)
            assert len(calls) == 2
            await asyncio.create_task(async_call(lambda: proof.validate(original, table, frame)))
            assert len(calls) == 3
            context = copy_context()
            with ThreadPoolExecutor(1) as pool:
                pool.submit(context.run, proof.validate, original, table, frame).result()
            assert len(calls) == 4
            proof.validate(original, object(), frame)
            proof.validate(original, table, SimpleNamespace(actor=1, revision=3))
            assert len(calls) == 6
            frame.revision = 4
            proof.validate(original, table, frame)
            assert len(calls) == 7
        assert proof.current.get() is None
    asyncio.run(check())


async def async_call(fn):
    return fn()


def test_failed_validation_and_cancellation_never_leave_proof():
    from benchmarks.collection_cpu.parent import DecisionProof
    proof = DecisionProof()
    frame = SimpleNamespace(actor=1, revision=3)
    count = 0
    def fail(*args):
        nonlocal count
        count += 1
        raise ContractError('bad table')
    with pytest.raises(asyncio.CancelledError), proof.decision():
        for _ in range(2):
            with pytest.raises(ContractError):
                proof.validate(fail, object(), frame)
        raise asyncio.CancelledError
    assert count == 2 and proof.current.get() is None


def test_http_two_reads_exact_original_observation_and_no_refetch():
    from benchmarks.collection_cpu.http_overlap import ReadOverlap
    async def check():
        seen = []
        observation = {'actionable': True, 'current_player': 1, 'viewer_player_id': 1, 'state_revision': 7, 'result': None}
        client = object()
        overlap = ReadOverlap()
        async def events(c, game, actor, cursor):
            seen.append(('events', asyncio.current_task(), cursor))
            await asyncio.sleep(0)
            return [], 'next'
        async def options(c, game, actor, obs):
            assert obs is observation
            seen.append(('options', asyncio.current_task(), obs))
            await asyncio.sleep(0)
            return {'options': 1}
        with overlap.episode(True):
            overlap.observed(client, 'game', 1, observation)
            assert await overlap.events(events, options, client, 'game', 1, None) == ([], 'next')
            assert len({row[1] for row in seen}) == 2
            assert await overlap.options(options, client, 'game', 1, observation) == {'options': 1}
            assert len(seen) == 2
            with pytest.raises(ContractError, match='identity'):
                await overlap.options(options, client, 'game', 1, dict(observation))
        assert overlap.current.get() is None
    asyncio.run(check())


@pytest.mark.parametrize('cancel', [False, True])
def test_http_failure_cancels_and_joins_sibling(cancel):
    from benchmarks.collection_cpu.http_overlap import ReadOverlap
    async def check():
        overlap = ReadOverlap()
        client = object()
        obs = {'actionable': True, 'current_player': 1, 'viewer_player_id': 1, 'state_revision': 1, 'result': None}
        started, retired = asyncio.Event(), asyncio.Event()
        async def events(*args):
            await started.wait()
            if cancel:
                raise asyncio.CancelledError
            raise RuntimeError('events failed')
        async def options(*args):
            started.set()
            try:
                await asyncio.Future()
            finally:
                retired.set()
        with overlap.episode(True):
            overlap.observed(client, 'g', 1, obs)
            with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
                await overlap.events(events, options, client, 'g', 1, None)
            assert retired.is_set()
            assert not overlap.current.get().tasks
    asyncio.run(check())


@pytest.mark.parametrize('mode', ['T1', 'T2', 'T3'])
def test_typed_and_arena_values_alignment_padding_and_scrub(mode):
    torch = pytest.importorskip('torch')
    from benchmarks.collection_cpu.transfer import TransferPacker
    packer = TransferPacker('cpu', mode, max_bytes=8192, max_tensor_bytes=4096)
    values = {'floats': np.array([0., -0., np.nan, np.inf], np.float32),
              'refs': ((4, 1), (18, 0)), 'mask': (True, False)}
    kinds = {'floats': np.float32, 'refs': np.int64, 'mask': np.bool_}
    packed = packer.pack(values, kinds, group='frame')
    for name, value in values.items():
        assert packed[name].numpy().tobytes() == np.asarray(value, dtype=kinds[name]).tobytes()
        assert packed[name].is_contiguous()
        if mode != 'T1':
            assert packed[name].data_ptr() % 8 == 0
    retained = [tensor for pair in packer.buffers.values() for tensor in pair]
    packer.consumed()
    packer.pack({'mask': (True,)}, {'mask': np.bool_}, group='frame')
    packer.consumed()
    packer.reset()
    assert packer.private_empty()
    assert all(not torch.count_nonzero(tensor) for tensor in retained)


def test_pack_capacity_and_inflight_fail_before_mutation():
    from benchmarks.collection_cpu.transfer import TransferPacker
    packer = TransferPacker('cpu', 'T2', max_bytes=64, max_tensor_bytes=64)
    first = packer.pack({'x': (1, 2, 3)}, {'x': np.int64}, group='frame')
    with pytest.raises(ContractError, match='in.flight'):
        packer.pack({'x': (7,)}, {'x': np.int64}, group='frame')
    assert first['x'].tolist() == [1, 2, 3]
    packer.consumed()
    with pytest.raises(ContractError, match='cap'):
        packer.pack({'x': tuple(range(7))}, {'x': np.int64}, group='frame')
    assert first['x'].tolist() == [1, 2, 3]
    packer.reset()


@pytest.mark.parametrize('mode', ['T1', 'T2', 'T3'])
def test_transfer_uses_one_candidate_pack_and_gpu_views_for_tail(make_input, mode):
    torch = pytest.importorskip('torch')
    from benchmarks.collection_cpu.transfer import install
    from gf_models.structured_network import make_network
    from tests.test_structured_network import inputs, small_model
    network = make_network(small_model(), 9)
    frames, candidates = inputs(make_input, count=1)
    many = replace(candidates, kinds=(candidates.kinds[0],)*257, targets=(candidates.targets[0],)*257,
        numerics=(candidates.numerics[0],)*257, present=(candidates.present[0],)*257,
        offsets=tuple(range(258)), slots=(0,)*257)
    experiment = install(network, mode)
    seen = []
    original = network._candidate_chunk_calculate
    def chunk(shared, projection, features, refs, weights, positions):
        seen.append((refs.shape, refs.untyped_storage().data_ptr()))
        return original(shared, projection, features, refs, weights, positions)
    network._candidate_chunk_calculate = chunk
    with torch.no_grad():
        network(frames, many)
    assert seen[0][0] == (256,18) and seen[1][0] == (1,18)
    assert seen[0][1] == seen[1][1]
    assert experiment.packer.pack_counts['candidates'] == 1
    experiment.reset()
    assert experiment.private_empty()


def test_parent_installer_excludes_inherited_journal_and_memory_override():
    from benchmarks.collection_cpu.parent import patches
    from gf_training.card_instance_journal import CardInstanceJournal
    from gf_training.group_pack_memory import MemoryCardInstanceJournal
    from gf_training.hpo_owned_capture import _CaptureContext
    replacements, proof = patches()
    wrappers = {(owner, name): fn for owner, name, fn in replacements}
    assert (CardInstanceJournal, 'begin_frame') in wrappers
    assert (MemoryCardInstanceJournal, 'begin_frame') in wrappers
    assert (_CaptureContext, 'begin_decision') in wrappers
    # The inherited Journal method must be wrapped, not just overrides.
    frame, table = SimpleNamespace(actor=1, revision=1), object()
    counts = []
    def validate(t, f):
        counts.append(1)
    with proof.decision():
        proof.validate(validate, table, frame)
        with proof.suspended():
            proof.validate(validate, table, frame)
        proof.validate(validate, table, frame)
    assert len(counts) == 2
