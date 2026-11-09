"""Fresh-process checks of pinned CPU eager and deferred compiler determinism."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

requires_torch = pytest.mark.skipif(importlib.util.find_spec('torch') is None,
                                    reason='optional pinned Torch is not installed')


def fresh_process(source):
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'TORCHINDUCTOR_DETERMINISTIC': '0', 'TORCHINDUCTOR_BATCH_INVARIANT': '0'}
    result = subprocess.run([sys.executable, '-E', '-c', textwrap.dedent(source)],
        env=environment, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_runtime_helper_keeps_torch_optional():
    fresh_process("""
        import sys
        import gf_models.torch_runtime
        assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
    """)


@requires_torch
def test_cpu_eager_late_import_and_repeated_policy_match_original_api():
    fresh_process("""
        import importlib
        import os
        import sys
        from gf_models.torch_runtime import runtime_metadata, torch_runtime

        metadata = runtime_metadata()
        assert metadata['torch'] in ('2.14.1+cpu', '2.14.1+cu132')
        assert 'torch' not in sys.modules
        torch = torch_runtime('cpu')
        assert os.environ['TORCHINDUCTOR_DETERMINISTIC'] == '1'

        def flags():
            return {
                'algorithms': torch.are_deterministic_algorithms_enabled(),
                'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
                'debug': torch.get_deterministic_debug_mode(),
                'threads': torch.get_num_threads(),
                'fp32': torch.backends.fp32_precision,
                'matmul': torch.backends.cuda.matmul.fp32_precision,
                'cudnn': torch.backends.cudnn.fp32_precision,
                'benchmark': torch.backends.cudnn.benchmark,
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cpu_amp': torch.is_autocast_enabled('cpu'),
                'cuda_amp': torch.is_autocast_enabled('cuda'),
            }

        expected = {'algorithms': True, 'warn_only': False, 'debug': 2, 'threads': 1,
            'fp32': 'ieee', 'matmul': 'ieee', 'cudnn': 'ieee', 'benchmark': False,
            'cudnn_deterministic': True, 'cpu_amp': False, 'cuda_amp': False}
        assert flags() == expected
        torch.manual_seed(17)
        network = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)).eval()
        inputs = torch.tensor([[1., 2., 3.], [-2., 0., 1.]])
        with torch.inference_mode():
            actual = network(inputs).clone()
        assert actual.shape == (2, 2) and torch.isfinite(actual).all()
        assert not any(name == prefix or name.startswith(prefix + '.')
            for name in sys.modules for prefix in ('torch._dynamo', 'torch._inductor'))

        config = importlib.import_module('torch._inductor.config')
        assert config.deterministic is True
        for loaded_flag in (False, True):
            config.deterministic = loaded_flag
            os.environ['TORCHINDUCTOR_DETERMINISTIC'] = '0'
            torch.set_deterministic_debug_mode('warn')
            torch.set_num_threads(2)
            torch.backends.fp32_precision = 'tf32'
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            torch.backends.cudnn.fp32_precision = 'tf32'
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            assert torch_runtime('cpu') is torch
            assert config.deterministic is True
            assert os.environ['TORCHINDUCTOR_DETERMINISTIC'] == '1'
            assert flags() == expected

        # The original ordinary API establishes exactly the same observable flags.
        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.use_deterministic_algorithms(True)
        assert flags() == expected and config.deterministic is True
        with torch.inference_mode():
            assert torch.equal(network(inputs), actual)
        assert not torch.cuda.is_initialized()
    """)


@requires_torch
def test_refusal_before_cpu_side_effects_failed_import_and_original_cuda_dispatch():
    fresh_process("""
        import importlib
        import os
        import sys
        from contextlib import ExitStack
        from unittest import mock
        import torch
        from gf_common.domain import ContractError
        from gf_models import torch_runtime as runtime

        def refused(device, message):
            os.environ['TORCHINDUCTOR_DETERMINISTIC'] = '0'
            try:
                runtime.torch_runtime(device)
            except ContractError as error:
                assert message in str(error)
            else:
                raise AssertionError('unsupported runtime was accepted')
            assert os.environ['TORCHINDUCTOR_DETERMINISTIC'] == '0'
            assert not torch.are_deterministic_algorithms_enabled()

        with mock.patch.object(importlib.metadata, 'version', return_value='unverified'):
            refused('cpu', 'unverified PPO native runtime')
        with mock.patch.object(torch, '__version__', 'unverified'):
            refused('cpu', 'module/distribution/CUDA runtime mismatch')
        with mock.patch.object(torch.version, 'cuda', 'unverified'):
            refused('cpu', 'module/distribution/CUDA runtime mismatch')
        for device in ('cpu:1', 'cuda:1', 'mps', ''):
            refused(device, 'configured learner device unavailable')
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(importlib.metadata, 'version', return_value='2.14.1+cpu'))
            stack.enter_context(mock.patch.object(torch, '__version__', '2.14.1+cpu'))
            stack.enter_context(mock.patch.object(torch.version, 'cuda', None))
            refused('cuda:0', 'CUDA learner requires the pinned cu132 build')
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            refused('cuda:0', 'configured learner device unavailable' if torch.__version__.endswith('+cu132')
                else 'CUDA learner requires the pinned cu132 build')

        assert 'torch._inductor.config' not in sys.modules
        sys.modules['torch._inductor.config'] = None
        try:
            try:
                runtime.torch_runtime('cpu')
            except ModuleNotFoundError as error:
                assert error.name == 'torch._inductor.config'
            else:
                raise AssertionError('failed import sentinel was accepted')
            assert not torch.are_deterministic_algorithms_enabled()
            assert os.environ['TORCHINDUCTOR_DETERMINISTIC'] == '1'
        finally:
            del sys.modules['torch._inductor.config']

        # Check dispatch only: actual hardware execution belongs to the CUDA gates.
        if torch.__version__ == '2.14.1+cu132':
            original = torch.use_deterministic_algorithms
            with mock.patch.object(torch.cuda, 'is_available', return_value=True), \\
                 mock.patch.object(torch, 'use_deterministic_algorithms', wraps=original) as ordinary, \\
                 mock.patch.object(runtime, '_cpu_determinism', side_effect=AssertionError('CPU helper on CUDA')):
                assert runtime.torch_runtime('cuda:0') is torch
                ordinary.assert_called_once_with(True)
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert importlib.import_module('torch._inductor.config').deterministic is True
        assert not torch.cuda.is_initialized()
    """)


@requires_torch
def test_cpu_setup_joins_actual_in_progress_config_import():
    fresh_process("""
        import importlib
        import os
        import sys
        import threading
        from pathlib import Path
        from unittest import mock
        import torch
        from gf_models import torch_runtime as runtime

        key = 'torch._inductor.config'
        assert key not in sys.modules
        filename = str(Path(torch.__file__).parent / '_inductor' / 'config.py')
        paused, release, completed = threading.Event(), threading.Event(), threading.Event()
        joining, helper_done = threading.Event(), threading.Event()
        errors, imported = [], []
        real_import = importlib.import_module

        def trace(frame, event, arg):
            if frame.f_code.co_filename == filename and frame.f_code.co_name == '<module>':
                if event == 'line' and frame.f_globals.get('deterministic') is False and not paused.is_set():
                    paused.set()
                    assert release.wait(20), 'test did not release actual config execution'
                if event == 'return':
                    completed.set()
            return trace

        def load():
            sys.settrace(trace)
            try:
                imported.append(real_import(key))
            except BaseException as error:
                errors.append(error)
            finally:
                sys.settrace(None)

        def join_import(name, package=None):
            if name == key:
                assert sys.modules[key].__spec__._initializing
                assert sys.modules[key].deterministic is False
                joining.set()
            value = real_import(name, package)
            if name == key:
                assert completed.is_set(), 'helper received a partially initialized config'
                assert not value.__spec__._initializing
            return value

        def initialize():
            try:
                assert runtime.torch_runtime('cpu') is torch
                # The actual config override belongs to this caller's ContextVar.
                assert completed.is_set() and not sys.modules[key].__spec__._initializing
                assert sys.modules[key].deterministic is True
            except BaseException as error:
                errors.append(error)
            finally:
                helper_done.set()

        loader = threading.Thread(target=load)
        helper = threading.Thread(target=initialize)
        loader.start()
        try:
            assert paused.wait(20), 'actual config did not reach the controlled import barrier'
            assert sys.modules[key].__spec__._initializing
            assert sys.modules[key].deterministic is False
            with mock.patch.object(importlib, 'import_module', side_effect=join_import):
                helper.start()
                try:
                    assert joining.wait(20), 'helper did not join the actual config import'
                    assert os.environ['TORCHINDUCTOR_DETERMINISTIC'] == '1'
                    assert not helper_done.is_set() and not completed.is_set()
                finally:
                    release.set()
                    helper.join(20)
        finally:
            release.set()
            loader.join(20)
        assert not loader.is_alive() and not helper.is_alive()
        assert not errors, errors
        assert completed.is_set() and helper_done.is_set() and len(imported) == 1
        assert imported[0] is sys.modules[key], (imported[0], sys.modules[key])
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.get_deterministic_debug_mode() == 2
    """)
