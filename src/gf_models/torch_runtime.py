"""Lazy, pinned Torch runtime and numeric state helpers shared by PPO models."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from gf_common.domain import ContractError, Json

NUMERICS = {"precision": "ieee-fp32", "tf32": False, "amp": False, "deterministic": True,
            "cublas_workspace_config": ":4096:8", "threads": 1}

_TORCH_LOCK = threading.RLock()


@contextmanager
def serialized_torch(*, deadline: float | None = None) -> Iterator[None]:
    """Serialize shared Torch work, optionally bounding entry by a monotonic deadline."""
    if deadline is None:
        with _TORCH_LOCK:
            yield
        return

    while (remaining := deadline - time.monotonic()) > 0:
        if _TORCH_LOCK.acquire(timeout=min(remaining, threading.TIMEOUT_MAX)):
            try:
                if time.monotonic() >= deadline:
                    raise ContractError("Torch serialization deadline expired")
                yield
            finally:
                _TORCH_LOCK.release()
            return
    raise ContractError("Torch serialization deadline expired")


def runtime_metadata() -> Json:
    versions = {"torch": importlib.metadata.version('torch'), "numpy": np.__version__,
                "python": platform.python_version(), "gil_enabled": sys._is_gil_enabled()}
    if versions['torch'] not in ('2.14.1+cpu', '2.14.1+cu132') or versions['numpy'] != '2.5.4' or sys.version_info[:2] != (3, 14) or not versions['gil_enabled']:
        raise ContractError('unverified PPO native runtime')
    return versions


def _cpu_determinism(torch: Any) -> None:
    # The pinned Inductor config consumes this value on a later ordinary import.
    os.environ['TORCHINDUCTOR_DETERMINISTIC'] = '1'
    if 'torch._inductor.config' in sys.modules:
        # Join real import completion before touching an existing config module.
        config: Any = importlib.import_module('torch._inductor.config')
        config.deterministic = True
    torch.set_deterministic_debug_mode('error')


def torch_runtime(device: str) -> Any:
    versions = runtime_metadata()
    import torch
    if torch.__version__ != versions['torch'] or torch.version.cuda != ('13.2' if versions['torch'].endswith('+cu132') else None):
        raise ContractError('PPO module/distribution/CUDA runtime mismatch')
    if device == 'cuda:0' and versions['torch'] != '2.14.1+cu132':
        raise ContractError('CUDA learner requires the pinned cu132 build')
    if device not in ('cpu', 'cuda:0') or device == 'cuda:0' and not torch.cuda.is_available():
        raise ContractError("configured learner device unavailable")
    torch.set_num_threads(1)
    if device == 'cpu':
        _cpu_determinism(torch)
    else:
        torch.use_deterministic_algorithms(True)
    backends: Any = torch.backends
    backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch


def weights(network: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().numpy().astype(np.float32, copy=True) for name, tensor in network.state_dict().items()}


def restore_weights(network: Any, arrays: dict[str, Any]) -> None:
    import torch
    expected = network.state_dict()
    if set(arrays) != set(expected):
        raise ContractError("PPO weight keys mismatch")
    for key, tensor in expected.items():
        value = arrays[key]
        if value.dtype != np.float32 or tuple(value.shape) != tuple(tensor.shape) or not np.isfinite(value).all():
            raise ContractError("PPO weight shape/type mismatch")
    network.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in arrays.items()}, strict=True)


def hardware(device: str) -> Json:
    torch = torch_runtime(device)
    name = torch.cuda.get_device_name(0) if device != 'cpu' else 'cpu'
    driver_api = None
    if device != 'cpu':
        # CUDA is initialized before this query. This is the supported CUDA
        # API version, not an attested NVIDIA driver build number.
        import ctypes
        driver = ctypes.CDLL('libcuda.so.1')
        query = driver.cuDriverGetVersion
        query.argtypes = [ctypes.POINTER(ctypes.c_int)]
        query.restype = ctypes.c_int
        actual = ctypes.c_int()
        if query(ctypes.byref(actual)) != 0 or actual.value <= 0:
            raise ContractError('CUDA driver API identity query failed')
        driver_api = actual.value
    return {'device': device, 'name': name, 'cuda_runtime': torch.version.cuda,
            'device_count': torch.cuda.device_count() if device != 'cpu' else 0,
            'cuda_driver_api_version': driver_api, 'exact_driver_build': 'unverified'}
