"""CPU/data controls for same-host CUDA accounting; no allocation authority."""
import subprocess
import sys

import pytest


def test_one_shared_torch_mutex_is_reexported_without_torch_import():
    code = """
import sys
from gf_models.torch_runtime import serialized_torch
from gf_models.rust_player import serialized_torch as old
from gf_models.rust_player_device import RustPlayerDeviceAccount
assert old is serialized_torch
assert serialized_torch.__module__ == 'gf_models.torch_runtime'
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', code], check=True)


def checkpoint_profile(*, context=128, workspace=256):
    from dataclasses import replace

    from gf_models.rust_player_device import CudaCheckpointProfile
    from gf_models.structured_config import InferenceBindingV2
    from tests.test_rust_player_cuda import LAYOUT, model4
    model = model4()
    model = replace(model, resources=replace(model.resources,
        cuda_context_bytes=context, workspace_bytes=workspace))
    return CudaCheckpointProfile(model, InferenceBindingV2(device='cuda:0'), layout=LAYOUT)


def test_heterogeneous_host_fixed_and_separate_update_publication_checks():
    from gf_models.rust_player_device import RustPlayerDeviceAccount
    learner, external = checkpoint_profile(), checkpoint_profile(context=512, workspace=1024)
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cuda:0',
        serving_device='cuda:0', learner_profile=learner, admitted_profiles=(learner, external))
    account.observe_learner({('cuda', 0, 1, 100): 100})
    account.observe_cache('external', external, {('cuda', 0, 2, 200): 200})
    operation = account.begin_update(logical_bytes=10**9, operation_id='op', generation=1)
    plan = operation.plan
    assert plan.host_fixed_bytes == 1536 and plan.learner_fixed_bytes == 384
    assert plan.retained_cache_tensor_bytes == 200
    assert plan.device_bytes == 10**9 - 200 - (1536 - 384)
    assert plan.publication_network_bytes == learner.network_bytes
    # Prospective publication is separate; it was NOT subtracted from update D.
    assert plan.post_update_learner_tensor_bytes == 4 * learner.parameter_bytes + learner.position_bytes
    assert operation.state == 'ACTIVE'
    operation.complete()
    assert operation.state == 'SETTLED'


@pytest.mark.parametrize('learner_device,serving_device', [('cpu', 'cuda:0'), ('cuda:0', 'cpu')])
def test_mixed_role_plan_has_only_actual_gpu_roles(learner_device, serving_device):
    from gf_models.rust_player_device import RustPlayerDeviceAccount
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device=learner_device,
        serving_device=serving_device, learner_profile=profile, admitted_profiles=(profile,))
    operation = account.begin_update(logical_bytes=10**9, operation_id='op', generation=1)
    plan = operation.plan
    assert (plan.device_bytes > 0) == (learner_device != 'cpu')
    assert (plan.learner_fixed_bytes > 0) == (learner_device != 'cpu')
    assert (plan.post_update_learner_tensor_bytes > 0) == (learner_device != 'cpu')
    assert (plan.publication_network_bytes > 0) == (serving_device != 'cpu')
    operation.complete()


def test_cache_aliases_and_positive_eviction_change_next_immutable_limit():
    from dataclasses import FrozenInstanceError

    from gf_common.domain import ContractError
    from gf_models.rust_player_device import RustPlayerDeviceAccount
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cuda:0',
        serving_device='cuda:0', learner_profile=profile, admitted_profiles=(profile,))
    storage = {('cuda', 0, 20, 200): 200}
    account.observe_cache('first', profile, storage)
    account.observe_cache('alias', profile, storage)
    assert account.retained_cache_tensor_bytes == 200
    first = account.begin_update(logical_bytes=10**9, operation_id='op', generation=1)
    original = first.device_bytes
    first.complete()
    with pytest.raises(FrozenInstanceError):
        first.plan.device_bytes = 1
    with pytest.raises(ContractError):
        account.evict_cache('first', gpu_settled=False)
    assert account.retained_cache_tensor_bytes == 200
    account.evict_cache('first', gpu_settled=True)
    assert account.retained_cache_tensor_bytes == 200
    account.evict_cache('alias', gpu_settled=True)
    second = account.begin_update(logical_bytes=10**9, operation_id='op', generation=2)
    assert second.device_bytes == original + 200
    assert first.device_bytes == original
    second.complete()


def test_row_geometry_is_checkpoint_specific_and_failed_hold_keeps_real_aliases():
    import weakref

    import numpy as np

    from gf_common.domain import ContractError
    from gf_models.rust_player_device import RustPlayerDeviceAccount
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cpu',
        serving_device='cuda:0', learner_profile=profile, admitted_profiles=(profile,))
    account.observe_cache('checkpoint', profile, {})
    hold = account.begin_row('checkpoint', interval_events=0, candidates=1, input_leaf_bytes=4096)
    assert hold.storage_bytes == profile.row.estimate(interval_events=0, candidates=1, input_leaf_bytes=4096).peak_bytes
    def fail():
        held_array = np.arange(512)
        reference = weakref.ref(held_array)
        try:
            raise ValueError('numerical failure')
        except ValueError as error:
            hold.fail(error, references={'input': held_array})
        return reference
    reference = fail()
    assert reference() is not None
    with pytest.raises(ContractError):
        account.begin_row('checkpoint', interval_events=0, candidates=1, input_leaf_bytes=0)
    with pytest.raises(ContractError):
        hold.retire_failed(gpu_settled=False)
    assert hold.state == 'FAILED_HELD' and reference() is not None
    hold.retire_failed(gpu_settled=True)
    assert reference() is None and hold.state == 'FAILED_RETIRED'
    with pytest.raises(ContractError):
        account.begin_update(logical_bytes=10**9, operation_id='op', generation=1)


def test_failed_operation_keeps_actual_graph_when_external_handler_clears_tb():
    import traceback
    import weakref

    import numpy as np

    from gf_common.domain import ContractError
    from gf_models.rust_player_device import RustPlayerDeviceAccount
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cuda:0', serving_device='cpu',
        learner_profile=profile, admitted_profiles=(profile,))
    hold = account.begin_update(logical_bytes=10**9, operation_id='op', generation=1)
    def fail():
        actual_graph = np.arange(512)
        reference = weakref.ref(actual_graph)
        try:
            raise ValueError('actual stopped graph')
        except ValueError as error:
            hold.fail(error)
            with pytest.raises(ContractError, match='executing'):
                hold.retire_failed(gpu_settled=True)
        return reference
    reference = fail()
    traceback.clear_frames(hold.error.__traceback__)
    assert reference() is not None and hold.state == 'FAILED_HELD'
    hold.retire_failed(gpu_settled=True)
    assert reference() is None and hold.state == 'FAILED_RETIRED'


def test_failed_operation_positive_fence_retires_actual_nested_bank_before_credit(monkeypatch):
    import numpy as np

    from gf_common.ram_bank import RamBankBuilder
    from gf_models.rust_player_device import RustPlayerDeviceAccount
    from tests.test_hpo_owned_numeric import budget
    source = budget()
    bank = RamBankBuilder({'value': np.arange(512, dtype=np.float32)}, source).seal()
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cuda:0', serving_device='cpu',
        learner_profile=profile, admitted_profiles=(profile,))
    hold = account.begin_update(logical_bytes=10**9, operation_id='component', generation=1)
    def fail():
        view = bank.array('value')
        try:
            raise ValueError('actual nested CPU bank')
        except ValueError as error:
            error.cleanup_owner = bank
            hold.fail(error, references={'view': view})
    fail()
    actual_close = bank.close
    def uncertain():
        raise OSError('bank close uncertain')
    monkeypatch.setattr(bank, 'close', uncertain)
    with pytest.raises(OSError, match='uncertain'):
        hold.retire_failed(gpu_settled=True)
    assert hold.state == 'FAILED_HELD' and account._active is hold and source.used > 0
    monkeypatch.setattr(bank, 'close', actual_close)
    hold.retire_failed(gpu_settled=True)
    assert source.used == 0 and hold.state == 'FAILED_RETIRED'


def test_failed_worker_must_stop_before_hold_can_retire():
    import threading

    from gf_common.domain import ContractError
    from gf_models.rust_player_device import RustPlayerDeviceAccount
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cpu', serving_device='cuda:0',
        learner_profile=profile, admitted_profiles=(profile,))
    account.observe_cache('checkpoint', profile, {})
    hold = account.begin_row('checkpoint', interval_events=0, candidates=1, input_leaf_bytes=0)
    failed, release = threading.Event(), threading.Event()
    def worker():
        try:
            raise ValueError('stopped producer required')
        except ValueError as error:
            hold.fail(error)
        failed.set()
        release.wait(2)
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert failed.wait(2)
        with pytest.raises(ContractError, match='worker is still running'):
            hold.retire_failed(gpu_settled=True)
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    hold.retire_failed(gpu_settled=True)
    assert hold.state == 'FAILED_RETIRED' and account._poisoned


@pytest.mark.parametrize('link', ['cause', 'context'])
@pytest.mark.parametrize('uncertain', [False, True])
def test_failed_exception_chain_keeps_real_bank_funding_until_all_nested_cleanup(link, uncertain, monkeypatch):
    import weakref

    import numpy as np

    from gf_common.ram_bank import RamBankBuilder
    from gf_models.rust_player_device import RustPlayerDeviceAccount
    from tests.test_hpo_owned_numeric import budget
    source = budget()
    bank = RamBankBuilder({'value': np.arange(512, dtype=np.float32)}, source).seal()
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cpu', serving_device='cuda:0',
        learner_profile=profile, admitted_profiles=(profile,))
    account.observe_cache('checkpoint', profile, {})
    hold = account.begin_row('checkpoint', interval_events=0, candidates=1, input_leaf_bytes=0)
    references = []
    def inner():
        actual_view = bank.array('value')
        references.append(weakref.ref(actual_view))
        error = ValueError('inner actual owner')
        error.cleanup_owner = bank
        raise error
    def invoke():
        try:
            inner()
        except ValueError as error:
            try:
                if link == 'cause':
                    raise RuntimeError('outer diagnostic') from error
                raise RuntimeError('outer diagnostic')
            except RuntimeError as outer:
                outer.cleanup_owner = bank  # Same actual owner through both links.
                hold.fail(outer)
    invoke()
    original = hold.error
    actual_close = bank.close
    calls = []
    def close():
        calls.append('close')
        if uncertain and len(calls) == 1:
            raise OSError('nested close uncertain')
        actual_close()
    monkeypatch.setattr(bank, 'close', close)
    if uncertain:
        with pytest.raises(OSError, match='uncertain'):
            hold.retire_failed(gpu_settled=True)
        assert hold.state == 'FAILED_HELD' and account._active is hold and source.used > 0
    hold.retire_failed(gpu_settled=True)
    assert len(calls) == (2 if uncertain else 1) and references[0]() is None and source.used == 0
    assert hold.error is original and str(original) == 'outer diagnostic'
    assert (original.__cause__ if link == 'cause' else original.__context__) is not None
    assert hold.state == 'FAILED_RETIRED' and account._poisoned


@pytest.mark.parametrize('link', ['cause', 'context'])
def test_active_inner_exception_frame_refuses_failed_retirement(link):
    from gf_common.domain import ContractError
    from gf_models.rust_player_device import RustPlayerDeviceAccount
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cpu', serving_device='cuda:0',
        learner_profile=profile, admitted_profiles=(profile,))
    account.observe_cache('checkpoint', profile, {})
    hold = account.begin_row('checkpoint', interval_events=0, candidates=1, input_leaf_bytes=0)
    def capture(error):
        try:
            if link == 'cause':
                raise RuntimeError('outer') from error
            raise RuntimeError('outer')
        except RuntimeError as outer:
            hold.fail(outer)
    def active_inner():
        try:
            raise ValueError('inner still executing')
        except ValueError as error:
            capture(error)
            with pytest.raises(ContractError, match='executing'):
                hold.retire_failed(gpu_settled=True)
    active_inner()
    hold.retire_failed(gpu_settled=True)
    assert hold.state == 'FAILED_RETIRED'


def test_nested_close_failure_owner_is_captured_before_caller_overwrite():
    import traceback
    import weakref

    import numpy as np

    from gf_common.ram_bank import RamBankBuilder
    from gf_models.rust_player_device import RustPlayerDeviceAccount
    from tests.test_hpo_owned_numeric import budget
    source = budget()
    bank = RamBankBuilder({'value': np.arange(16, dtype=np.float32)}, source).seal()
    reference = weakref.ref(bank)
    pending = [bank]
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cpu', serving_device='cuda:0',
        learner_profile=profile, admitted_profiles=(profile,))
    hold = account.begin_update(logical_bytes=10**9, operation_id='component', generation=1)
    calls = []
    class Cleanup:
        def close(self):
            calls.append('outer')
            if pending:
                inner_owner = pending.pop()
                error = OSError('actual nested close owner')
                error.cleanup_owner = inner_owner
                raise error
            assert reference() is None or reference()._fd == -1
    outer = Cleanup()
    def fail():
        try:
            raise ValueError('original numerical error')
        except ValueError as error:
            error.cleanup_owner = outer
            hold.fail(error)
    fail()
    del bank
    with pytest.raises(OSError, match='nested close owner') as failed:
        hold.retire_failed(gpu_settled=True)
    original = failed.value
    traceback.clear_frames(original.__traceback__)
    original.cleanup_owner = object()  # Caller overwrite must not lose the actual bank.
    assert reference() is not None and source.used > 0 and hold.state == 'FAILED_HELD'
    assert hold.retirement_errors[-1] is original
    hold.retire_failed(gpu_settled=True)
    assert source.used == 0 and hold.state == 'FAILED_RETIRED' and calls == ['outer', 'outer']


def test_publication_construction_counts_new_network_before_actual_insertion():
    from gf_common.domain import ContractError
    from gf_models.rust_player_device import RustPlayerDeviceAccount
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=profile.fixed_bytes + profile.network_bytes,
        learner_device='cpu', serving_device='cuda:0', learner_profile=profile, admitted_profiles=(profile,))
    hold = account.begin_construction('first', profile)
    assert hold.storage_bytes == profile.network_bytes
    account.observe_cache('first', profile, {('cuda', 0, 1, profile.network_bytes): profile.network_bytes})
    hold.complete()
    with pytest.raises(ContractError):
        account.begin_construction('second', profile)


def test_storage_enumeration_deduplicates_views_and_excludes_cpu_step():
    from types import SimpleNamespace

    from gf_models.rust_player_device import cuda_learner_storage
    class Tensor:
        def __init__(self, pointer, size, device='cuda', grad=None):
            self.device, self.grad = SimpleNamespace(type=device, index=0), grad
            self.pointer, self.size = pointer, size
        def untyped_storage(self):
            return SimpleNamespace(data_ptr=lambda: self.pointer, nbytes=lambda: self.size)
        def numel(self):
            return self.size // 4
    parameter = Tensor(1, 100, grad=Tensor(2, 100))
    moment = Tensor(3, 100)
    network = SimpleNamespace(parameters=lambda: [parameter], buffers=lambda: [Tensor(1, 100), Tensor(4, 20)])
    optimizer = SimpleNamespace(state={'p': {'step': Tensor(5, 4, 'cpu'), 'exp_avg': moment,
        'exp_avg_sq': Tensor(6, 100)}})
    storage, largest = cuda_learner_storage(network, optimizer)
    assert sum(storage.values()) == 420 and len(storage) == 5 and largest == 25


def test_model3_export_actual_moments_and_fresh_restore_are_separate_phases():
    from dataclasses import replace

    from gf_common.domain import ContractError
    from gf_models.rust_player_device import CudaCheckpointProfile, RustPlayerDeviceAccount
    from gf_models.structured_config import InferenceBindingV2, ModelConfigV3
    from tests.test_rust_player_cuda import LAYOUT
    model = ModelConfigV3()
    model = replace(model, resources=replace(model.resources, cuda_context_bytes=128, workspace_bytes=256))
    profile = CudaCheckpointProfile(model, InferenceBindingV2(device='cuda:0'), layout=LAYOUT)
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cuda:0', serving_device='cpu',
        learner_profile=profile, admitted_profiles=(profile,))
    account.observe_learner({('cuda', 0, 1, 100): 100}, moment_max_numel=25)
    export = account.begin_export()
    assert export.required_bytes == profile.fixed_bytes + 100 + 125
    export.complete()
    restore = account.begin_restore(moment_bytes=2 * profile.parameter_bytes)
    assert restore.required_bytes == profile.fixed_bytes + 100 + profile.network_bytes + 2 * profile.parameter_bytes
    restore.complete()
    account.global_bytes = profile.fixed_bytes + 100 + 124
    with pytest.raises(ContractError):
        account.begin_export()


@pytest.mark.parametrize('profile_enabled', [False, True])
def test_serving_row_inputs_retire_before_next_transfer_and_lock_exit(monkeypatch, profile_enabled):
    import json
    from types import SimpleNamespace

    import numpy as np
    import torch

    from gf_models import rust_player, shared_numeric, torch_runtime
    from gf_models.rust_player_device import RustPlayerDeviceAccount
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cpu', serving_device='cuda:0',
        learner_profile=profile, admitted_profiles=(profile,))
    account.observe_cache('checkpoint', profile, {})
    service = rust_player.RustPlayerInference.__new__(rust_player.RustPlayerInference)
    service.device, service.runtime, service.device_account = 'cuda:0', 'fixture', account
    service.performance_callback = None
    service.registry = SimpleNamespace(networks={'checkpoint': object()})
    service.timings = dict.fromkeys(('assembly', 'cpu_share', 'h2d', 'forward', 'sampling', 'return'), 0.)
    service.calls = {'rows': 0}
    alive, retired_under_lock = [0], []
    class Input:
        def __init__(self):
            alive[0] += 1
        def __del__(self):
            retired_under_lock.append(torch_runtime._TORCH_LOCK._is_owned())
            alive[0] -= 1
    class CpuView:
        def __init__(self, name):
            self.name = name
        def to(self, device, **kwargs):
            if self.name == 'interval_numerics':
                assert alive[0] == 0, 'previous row inputs overlap next H2D'
            return Input()
    callbacks, ranges = [], []
    def after_forward(**facts):
        assert torch_runtime._TORCH_LOCK._is_owned()
        assert account._active.state == 'ACTIVE' and alive[0] == 2
        callbacks.append(facts)
    class ForwardRange:
        def __init__(self, name):
            assert profile_enabled, 'disabled profiling must not create a range'
            assert name == 'gf-native-row-forward'
        def __enter__(self):
            assert torch_runtime._TORCH_LOCK._is_owned()
            ranges.append('begin')
        def __exit__(self, *args):
            assert torch_runtime._TORCH_LOCK._is_owned()
            ranges.append('end')
    if profile_enabled:
        service.performance_callback = SimpleNamespace(after_forward=after_forward)
    monkeypatch.setattr(torch.profiler, 'record_function', ForwardRange)
    monkeypatch.setattr(rust_player, 'tensor_views', lambda arrays, **kwargs: {name: CpuView(name) for name in arrays})
    monkeypatch.setattr(shared_numeric, 'score_numeric_row', lambda *args: (torch.tensor([0.]), torch.tensor(1.)))
    monkeypatch.setattr(rust_player, 'provisional_sample', lambda *args, **kwargs: (0, b'state'))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    row = {'identity': json.dumps({'checkpoint': 'checkpoint'}), 'assembly_seconds': 0.,
        'rng_state': None, 'sampling_seed': 17,
        'inputs': {'interval_numerics': np.empty((0, 1), np.float32), 'kinds': np.zeros(1, np.int64)}}
    # CUDA I/O doubles are CPU objects, never unadmitted CUDA allocations.
    answers = service.evaluate({'runtime': 'fixture', 'rows': [row, row]})
    assert len(answers) == 2 and alive[0] == 0
    assert len(retired_under_lock) == 4 and all(retired_under_lock)
    assert account._active is None
    assert callbacks == ([{'scored_rows': 1, 'boundary': 'native-post-row-forward'}] * 2 if profile_enabled else [])
    assert ranges == (['begin', 'end'] * 2 if profile_enabled else [])


def test_cpu_serving_accepts_shared_account_without_gpu_cache_charge():
    from dataclasses import replace
    from types import SimpleNamespace

    from gf_models.rust_player import RustPlayerInference
    from gf_models.rust_player_device import CudaCheckpointProfile, RustPlayerDeviceAccount
    from tests.test_hpo_numeric import learner_for
    learner = learner_for()
    profile = CudaCheckpointProfile(learner.model, replace(learner.binding, device='cuda:0'),
        layout=learner.network.layout)
    account = RustPlayerDeviceAccount(global_bytes=8 * 1024**3, learner_device='cuda:0', serving_device='cpu',
        learner_profile=profile, admitted_profiles=(profile,))
    service = RustPlayerInference(device='cpu', cache_entries=2, cache_weight_bytes=128 * 1024**2,
        device_account=account)
    service.publish(learner.network, 'a' * 64)
    assert account.retained_cache_tensor_bytes == 0 and not account._cache
    import json

    from gf_models.shared_numeric import numeric_row, tensor_row
    from tests.model3_support import raw_inputs
    frames, candidates, *_ = raw_inputs(count=1)
    raw = tensor_row(numeric_row(frames[0], candidates, learner.model, layout=learner.network.layout))
    row = {'identity': json.dumps({'checkpoint': 'a' * 64}), 'assembly_seconds': 0.,
        'rng_state': None, 'sampling_seed': 17, 'inputs': {name: value.numpy() for name, value in raw.items()}}
    assert len(service.evaluate({'runtime': service.runtime, 'rows': [row]})) == 1
    service.evict('a' * 64, dispatcher=SimpleNamespace(checkpoint_references=dict))


def test_serving_role_mismatch_refuses_before_registry_creation(monkeypatch):
    from gf_common.domain import ContractError
    from gf_models import shared_registry
    from gf_models.rust_player import RustPlayerInference
    from gf_models.rust_player_device import RustPlayerDeviceAccount
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cpu', serving_device='cuda:0',
        learner_profile=profile, admitted_profiles=(profile,))
    def forbidden(**kwargs):
        raise AssertionError('registry created for mismatched actual role')
    monkeypatch.setattr(shared_registry, 'make_registry', forbidden)
    with pytest.raises(ContractError, match='role'):
        RustPlayerInference(device='cpu', cache_entries=1, cache_weight_bytes=1024, device_account=account)


def test_failed_publication_keeps_provisional_network_until_positive_cleanup(monkeypatch):
    import weakref
    from types import SimpleNamespace

    import torch

    from gf_models.rust_player import RustPlayerInference
    from gf_models.rust_player_device import RustPlayerDeviceAccount
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cpu', serving_device='cuda:0',
        learner_profile=profile, admitted_profiles=(profile,))
    service = RustPlayerInference.__new__(RustPlayerInference)
    service.device, service.device_account, service.references = 'cuda:0', account, {}
    references = []
    class Network:
        pass
    class Registry:
        def __init__(self):
            self.networks = {}
        def load_policy_weights(self, payload):
            if isinstance(payload, str) and payload == 'incoming':
                network = Network()
                references.append(weakref.ref(network))
                self.networks['checkpoint'] = network
                raise ValueError('publication failed after insertion')
            self.networks.pop(payload.get_non_tensor('publication')['checkpoint'])
    service.registry = Registry()
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    with pytest.raises(ValueError, match='after insertion'):
        service._publish_payload(profile.model, profile.binding, 'incoming', 'checkpoint')
    hold = account._active
    assert hold.state == 'FAILED_HELD' and references[0]() is not None
    service.retire_device_failure(dispatcher=SimpleNamespace(checkpoint_references=dict))
    assert not service.registry.networks and references[0]() is None
    assert hold.state == 'FAILED_RETIRED' and account._poisoned


def test_embedded_plan_closed_fields_roles_and_invocation_are_data_only():
    from gf_common.domain import ContractError
    from gf_models.rust_player_device import RustPlayerDeviceAccount, validate_device_plan
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cuda:0', serving_device='cpu',
        learner_profile=profile, admitted_profiles=(profile,))
    operation = account.begin_update(logical_bytes=10**9, operation_id='source-op', generation=2)
    record = operation.plan.record()
    assert len(record) == 15
    assert validate_device_plan(record, operation_id='source-op', generation=2) is record
    for bad in (record | {'extra': 0}, record | {'device_bytes': True}, record | {'operation_id': 'resume-op'},
                record | {'generation': 3}, record | {'learner_device': 'cpu'}):
        with pytest.raises(ContractError):
            validate_device_plan(bad, operation_id='source-op', generation=2)
    operation.complete()


def test_plan_presence_and_budget_bind_actual_source_invocation_not_resume():
    from gf_common.domain import ContractError
    from gf_models.rust_player_device import RustPlayerDeviceAccount, bind_update_device_plan
    profile = checkpoint_profile()
    account = RustPlayerDeviceAccount(global_bytes=10**9, learner_device='cpu', serving_device='cuda:0',
        learner_profile=profile, admitted_profiles=(profile,))
    operation = account.begin_update(logical_bytes=10**9, operation_id='source', generation=1)
    update = {'generation': 1, 'device_plan': operation.plan.record()}
    context = {'operation_id': 'source', 'learner_device': 'cpu', 'serving_device': 'cuda:0', 'global_bytes': 10**9}
    assert bind_update_device_plan(update, **context) is update['device_plan']
    for changed in ({'operation_id': 'resume'}, {'global_bytes': 10**9 + 1}):
        with pytest.raises(ContractError):
            bind_update_device_plan(update, **(context | changed))
    with pytest.raises(ContractError):
        bind_update_device_plan({'generation': 1}, **context)
    cpu = context | {'serving_device': 'cpu', 'global_bytes': 0}
    assert bind_update_device_plan({'generation': 1}, **cpu) is None
    with pytest.raises(ContractError):
        bind_update_device_plan({'generation': 1, 'device_plan': None}, **cpu)
    operation.complete()


@pytest.mark.parametrize('serving', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('retained', [1, 6])
def test_supported_default_model3_endpoint_guard_has_no_new_false_refusal(serving, retained):
    from dataclasses import replace

    from gf_cli.config import load_config
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.structured_config import resource_estimate
    from gf_training.rust_player_learning import _configured_device_account
    from tests.test_rust_player_config import NORMAL_BASE
    loaded = load_config(NORMAL_BASE, {'training': {'device': 'cuda:0'}, 'model': {'inference': {'device': serving}}})
    config = loaded.config
    account = _configured_device_account(config, global_bytes=config.run.global_cuda_budget_bytes)
    profile = account.learner_profile
    if serving != 'cpu':
        for index in range(retained):
            account.observe_cache(str(index), profile, {('cuda', 0, index + 1, profile.network_bytes): profile.network_bytes})
    logical = min(profile.model.resources.max_owned_bytes, profile.binding.managed_budget_bytes)
    operation = account.begin_update(logical_bytes=logical, operation_id='data-profile', generation=8)
    # This maximum endpoint is a DATA scenario only, never a new unconditional
    # global admission minimum. Runtime uses each actual endpoint's own geometry.
    actual = resource_estimate(profile.model, replace(profile.binding, device='cuda:0'), layout=layout)['total_bytes']
    assert actual + profile.position_bytes <= operation.device_bytes
    operation.complete()
