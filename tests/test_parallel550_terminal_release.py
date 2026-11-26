"""Checkpoint hold ownership at the staged native terminal ACK boundary."""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from gf_common.domain import ContractError
from gf_models.torch_runtime import serialized_torch
from gf_simulator_client.rust_player import RustPlayerRuntime


@pytest.fixture
def owner() -> RustPlayerRuntime:
    runtime = RustPlayerRuntime.__new__(RustPlayerRuntime)
    runtime._lock = threading.RLock()
    runtime._settled = threading.Condition()
    runtime._native = Mock()
    runtime._service_error = None
    runtime._active_routes = {}
    runtime._holds = {}
    runtime.service = Mock()
    return runtime


def test_native_ack_precedes_inspection_and_leaves_checkpoint_holds(
    owner: RustPlayerRuntime,
) -> None:
    key = ('run', 'trial', 'episode')
    owner._holds[key] = ['checkpoint']
    native = cast(Mock, owner._native)

    class Bundle:
        game = ''

        @property
        def release_key(self) -> tuple[str, str, str]:
            native.storage_ack.assert_called_once_with(self, None)
            return key

    bundle = Bundle()
    native.storage_ack.side_effect = ContractError('expired native ACK')
    with pytest.raises(ContractError, match='expired native ACK'):
        owner._storage_ack_defer_release(bundle)
    assert owner._holds == {key: ['checkpoint']}
    owner.service.release.assert_not_called()

    native.storage_ack.reset_mock(side_effect=True)
    assert owner._storage_ack_defer_release(bundle) == (key, '')
    assert owner._holds == {key: ['checkpoint']}
    owner.service.release.assert_not_called()


def test_only_exact_retired_routes_allow_release(owner: RustPlayerRuntime) -> None:
    key = ('run', 'trial', 'episode')
    owner._holds[key] = ['checkpoint']
    route = ('run', 'trial', '')
    unrelated_route = ('run', 'trial', 'other')
    owner._active_routes = {route: 1, unrelated_route: 1}
    with pytest.raises(ContractError, match='deadline'):
        owner._settle_deferred_storage_releases([(key, '')], time.monotonic() + 0.02)
    assert owner._holds == {key: ['checkpoint']}
    owner.service.release.assert_not_called()

    del owner._active_routes[route]
    owner._settle_deferred_storage_releases([(key, '')], time.monotonic() + 2)
    assert owner._holds == {}
    assert owner._active_routes == {unrelated_route: 1}
    owner.service.release.assert_called_once_with('checkpoint')


def test_service_failure_keeps_checkpoint_holds_for_join(owner: RustPlayerRuntime) -> None:
    key = ('run', 'trial', 'episode')
    owner._holds[key] = ['checkpoint']
    owner._service_error = 'inference failed'

    owner._settle_deferred_storage_releases([(key, 'game')], time.monotonic() + 2)

    assert owner._holds == {key: ['checkpoint']}
    owner.service.release.assert_not_called()


def test_contested_torch_lock_keeps_checkpoint_holds_for_join(owner: RustPlayerRuntime) -> None:
    key = ('run', 'trial', 'episode')
    owner._holds[key] = ['checkpoint']
    locked = threading.Event()
    unlock = threading.Event()

    def hold_torch_lock() -> None:
        with serialized_torch():
            locked.set()
            unlock.wait()

    with ThreadPoolExecutor(max_workers=2) as workers:
        holder = workers.submit(hold_torch_lock)
        try:
            assert locked.wait(timeout=2)
            release = workers.submit(
                owner._settle_deferred_storage_releases,
                [(key, 'game')],
                time.monotonic() + 0.05,
            )
            with pytest.raises(ContractError, match='deadline'):
                release.result(timeout=1)
            assert not holder.done()
        finally:
            unlock.set()
        holder.result(timeout=2)

    assert owner._holds == {key: ['checkpoint']}
    owner.service.release.assert_not_called()


def test_partial_release_preserves_only_unreleased_refs_for_join(
    owner: RustPlayerRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = ('run', 'trial', 'first')
    second = ('run', 'trial', 'second')
    third = ('run', 'trial', 'third')
    owner._holds = {
        first: ['A'],
        second: ['B', 'C', 'D'],
        third: ['E'],
    }
    owner.service.release.side_effect = [None, None, ContractError('release C')]
    with pytest.raises(ContractError, match='release C'):
        owner._settle_deferred_storage_releases(
            [(first, 'one'), (second, 'two'), (third, 'three')],
            time.monotonic() + 2,
        )
    assert owner._holds == {second: ['C', 'D'], third: ['E']}
    assert [call.args for call in owner.service.release.call_args_list] == [
        ('A',), ('B',), ('C',),
    ]
    cast(Mock, owner._native).join.return_value = '{"player_threads": 1}'
    owner._join_done = threading.Condition(owner._lock)
    owner._joining = False
    owner._join_error = None
    owner._report = None
    owner._closing = threading.Event()
    owner._service_thread = None
    owner._sequence = False
    owner.diagnostics = None
    owner.service.release.side_effect = None
    owner.service.release.reset_mock()
    monkeypatch.setitem(sys.modules, 'torch', Mock(get_num_threads=lambda: 1,
                                                 get_num_interop_threads=lambda: 1))
    owner.join()
    assert owner._holds == {}
    assert [call.args for call in owner.service.release.call_args_list] == [
        ('C',), ('D',), ('E',),
    ]


def test_releasing_same_checkpoint_twice_consumes_actual_episode_holds(
    owner: RustPlayerRuntime,
) -> None:
    first = ('run', 'trial', 'first')
    second = ('run', 'trial', 'second')
    owner._holds = {first: ['same'], second: ['same']}
    items = []
    for key, game in ((first, 'one'), (second, 'two')):
        receipt = owner._storage_ack_defer_release(SimpleNamespace(release_key=key, game=game))
        assert receipt is not None
        items.append(receipt)
    owner._settle_deferred_storage_releases(items, time.monotonic() + 2)
    assert owner._holds == {}
    assert [call.args for call in owner.service.release.call_args_list] == [
        ('same',), ('same',),
    ]
