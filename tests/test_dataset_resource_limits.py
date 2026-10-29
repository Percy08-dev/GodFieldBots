import resource
import subprocess
import sys

import pytest

from gf_cli.resource_limits import dataset_address_limit
from gf_common.domain import ContractError


def test_address_limit_scopes_soft_only_and_restores_on_failure(monkeypatch):
    from gf_cli import resource_limits as limits
    changes = []
    monkeypatch.setattr(limits, 'virtual_bytes', lambda: 16 * 1024 * 1024)
    monkeypatch.setattr(resource, 'getrlimit', lambda key: (128 * 1024 * 1024, 256 * 1024 * 1024))
    monkeypatch.setattr(resource, 'setrlimit', lambda key, value: changes.append(value))
    with pytest.raises(RuntimeError), dataset_address_limit(192):
        raise RuntimeError('source failure')
    assert changes == [(128 * 1024 * 1024, 256 * 1024 * 1024)] * 2


@pytest.mark.parametrize('value', [True, 0, -1, 2**64, float('nan'), float('inf')])
def test_address_limit_invalid_values_never_change_limits(monkeypatch, value):
    monkeypatch.setattr(resource, 'setrlimit', lambda *args: pytest.fail('unexpected limit change'))
    with pytest.raises(ContractError), dataset_address_limit(value):
        pass


def test_address_limit_rejects_current_vm_and_existing_hard_limit(monkeypatch):
    from gf_cli import resource_limits as limits
    monkeypatch.setattr(resource, 'getrlimit', lambda key: (-1, 64 * 1024 * 1024))
    monkeypatch.setattr(resource, 'setrlimit', lambda *args: pytest.fail('unexpected limit change'))
    monkeypatch.setattr(limits, 'virtual_bytes', lambda: 32 * 1024 * 1024)
    with pytest.raises(ContractError, match='current virtual'), dataset_address_limit(16):
        pass
    with pytest.raises(ContractError, match='hard limit'), dataset_address_limit(128):
        pass


def test_no_address_limit_leaves_runtime_untouched(monkeypatch):
    monkeypatch.setattr(resource, 'getrlimit', lambda *args: pytest.fail('unexpected resource access'))
    with dataset_address_limit(None):
        pass


def test_real_address_limit_rejects_allocation_in_owned_subprocess():
    code = '''
import resource
from gf_cli.resource_limits import dataset_address_limit, virtual_bytes
original = resource.getrlimit(resource.RLIMIT_AS)
limit = (virtual_bytes() + 1024*1024 - 1) // (1024*1024) + 16
with dataset_address_limit(limit):
    try:
        allocation = bytearray(64*1024*1024)
    except MemoryError:
        pass
    else:
        raise AssertionError("allocation exceeded declared virtual-address limit")
assert resource.getrlimit(resource.RLIMIT_AS) == original
print("bounded allocation and restoration PASS")
'''
    result = subprocess.run([sys.executable, '-c', code], check=False, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'restoration PASS' in result.stdout


def test_training_cli_does_not_accept_address_limit():
    result = subprocess.run([sys.executable, '-m', 'gf_cli', 'train', '--config',
                             '/nonexistent-config.toml', '--address-space-limit-mib', '64'],
                            check=False, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert 'unrecognized arguments: --address-space-limit-mib' in result.stderr
