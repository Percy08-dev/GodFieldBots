from __future__ import annotations

import json

import pytest

from gf_common.domain import ContractError, digest
from tests.test_group_pack_game_source import owner, packed


def test_registry_description_and_exact_scoped_original_lease(tmp_path):
    from gf_training.native_group_pack import OriginalArtifactProvider
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    registry = json.loads(provider.registry())
    assert registry == {'format': 'python-original-bytes-v1', 'roots': [
        {'root': str(root), 'storage': original.storage, 'custody': original.custody}]}
    assert provider.registry_sha256 == digest(registry)
    description = json.loads(provider.describe(str(root)))
    assert set(description) == {'format', 'root', 'storage', 'custody', 'files', 'index'}
    assert description['index'] == {'groups': [{'group_id': row['group_id'], 'record': row['record']}
        for row in original.artifacts._index['groups']], 'episodes': [{key: row[key]
        for key in ('episode_id', 'group_id', 'game_id', 'record')} for row in original.artifacts._index['episodes']]}
    assert [row['name'] for row in description['files']] == sorted(original._entries)
    lease = provider.borrow(str(root), 'first/e0')
    assert lease.data() == b'\xff\x00'
    assert original.artifacts._loans == 1
    with pytest.raises(ContractError, match='lease'):
        provider.close()
    lease.close()
    assert original.artifacts._loans == 0
    newer = provider.borrow(str(root), 'second/e1')
    lease.close()
    assert original.artifacts._loans == 1
    newer.close()
    with pytest.raises(ContractError, match='retired'):
        lease.data()
    provider.close()
    original.close()


@pytest.mark.parametrize('root_name,member', [('foreign', 'first/e0'), ('root', '../e0'), ('root', 'missing')])
def test_unknown_root_member_rejected_before_group_io(tmp_path, root_name, member):
    from gf_training.native_group_pack import OriginalArtifactProvider
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    with pytest.raises((ContractError, KeyError, ValueError)):
        provider.borrow(str(root) if root_name == 'root' else root_name, member)
    assert original.pool.active is None
    provider.close()
    original.close()


def test_provider_refuses_duplicate_or_different_pool_owners(tmp_path):
    from gf_training.native_group_pack import OriginalArtifactProvider
    a, _ = packed(tmp_path / 'a')
    b, _ = packed(tmp_path / 'b')
    first, second = owner(a), owner(b)
    for originals in ((first, first), (first, second)):
        with pytest.raises(ContractError):
            OriginalArtifactProvider(originals)
    first.close()
    second.close()


def test_provider_file_cap_includes_opaque_members(tmp_path):
    from gf_training.native_group_pack import OriginalArtifactProvider
    root, _ = packed(tmp_path)
    original = owner(root)
    original._entries['first/e0']['length'] = original.settings.max_source_file_mib * 1024**2 + 1
    with pytest.raises(ContractError, match='file'):
        OriginalArtifactProvider((original,))
    original.close()


def test_control_lease_retains_and_releases_actual_copy_charge(tmp_path):
    from gf_training.native_group_pack import OriginalArtifactProvider
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    before = original.pool.budget.used
    loan = provider.borrow(str(root), 'manifest.json')
    assert loan.data() == b'{"hello":"original"}\n'
    assert original.pool.budget.used == before + 2 * len(loan.data())
    loan.close()
    assert original.pool.budget.used == before
    provider.close()
    original.close()


def test_callback_reentry_is_sticky_even_when_inner_error_is_caught(tmp_path):
    from gf_training.native_group_pack import OriginalArtifactProvider
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    with pytest.raises(ContractError, match='cleanup-only'), provider._callback():  # noqa: SIM117 -- callback must observe the caught inner failure.
        with pytest.raises(ContractError, match='reentry'):
            provider.registry()
    with pytest.raises(ContractError, match='cleanup-only'):
        provider.borrow(str(root), 'first/e0')
    provider.close()
    original.close()


def test_lease_close_uncertainty_keeps_concrete_context_and_pool(tmp_path, monkeypatch):
    from gf_training.native_group_pack import OriginalArtifactProvider
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    loan = provider.borrow(str(root), 'first/e0')
    context = loan._context
    class FailingClose:
        def __exit__(self, *args):
            raise OSError('close not confirmed')
    loan._context = FailingClose()
    with pytest.raises(OSError, match='not confirmed'):
        loan.close()
    assert provider._leases[loan._generation] is loan
    assert loan._data == b'\xff\x00' and original.artifacts._loans == 1
    with pytest.raises(ContractError, match='leases'):
        provider.close()
    with pytest.raises(ContractError, match='cleanup-only'):
        loan.data()
    loan._context = context
    loan.close()
    provider.close()
    original.close()


def test_provider_metadata_admits_before_retention(tmp_path, monkeypatch):
    from gf_training.native_group_pack import OriginalArtifactProvider
    root, _ = packed(tmp_path)
    original = owner(root)
    budget = original.pool.budget
    before = budget.used
    budget.limit = before
    with pytest.raises(ValueError, match='capacity|exhausted'):
        OriginalArtifactProvider((original,))
    assert budget.used == before and original.pool.active is None
    original.close()


def test_whole_description_eight_mib_cap_includes_original_index_records(tmp_path):
    from gf_common.domain import canonical
    from gf_training.native_group_pack import METADATA_LIMIT, OriginalArtifactProvider
    assert METADATA_LIMIT == 8 * 1024**2
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    description = json.loads(provider.describe(str(root)))
    provider.close()
    description['index']['groups'][0]['record']['padding'] = ''
    count = METADATA_LIMIT - len(canonical(description).encode())
    # Controlled metadata fault injection into the SAME concrete index. This
    # exercises admission/copy closure, not Source semantics or fresh SQL trust.
    group = original.artifacts._index['groups'][0]
    group['record']['padding'] = 'x' * count
    before = original.pool.budget.used
    provider = OriginalArtifactProvider((original,))
    assert len(provider.describe(str(root))) == METADATA_LIMIT
    provider.close()
    assert original.pool.budget.used == before
    group['record']['padding'] += 'x'
    with pytest.raises(ValueError, match='byte bound'):
        OriginalArtifactProvider((original,))
    assert original.pool.budget.used == before and original.pool.active is None
    original.close()
    assert original.pool.budget.used == 0


@pytest.mark.parametrize('after', [False, True])
def test_control_copy_release_stages_preserve_exact_actual_account(tmp_path, monkeypatch, after):
    from gf_training.native_group_pack import OriginalArtifactProvider
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    before = original.pool.budget.used
    lease = provider.borrow(str(root), 'manifest.json')
    release = original.pool.budget.release
    def fault(amount):
        if after:
            release(amount)
        raise OSError('copy release fault')
    monkeypatch.setattr(original.pool.budget, 'release', fault)
    with pytest.raises(OSError):
        lease.close()
    assert provider._leases[lease._generation] is lease
    assert original.pool.budget.used == (before if after else before + lease._charge)
    monkeypatch.setattr(original.pool.budget, 'release', release)
    lease.close()
    assert original.pool.budget.used == before
    provider.close()
    original.close()


def test_generator_exit_after_effect_is_confirmed_without_empty_generator_retry(tmp_path, monkeypatch):
    from gf_training.native_group_pack import OriginalArtifactProvider
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    lease = provider.borrow(str(root), 'first/e0')
    exit_original = lease._context.__exit__
    count = []
    def fault(*args):
        count.append(True)
        exit_original(*args)
        raise OSError('exit after actual retirement')
    monkeypatch.setattr(lease._context, '__exit__', fault)
    with pytest.raises(OSError):
        lease.close()
    assert original.artifacts._loans == 0 and lease._context_done
    lease.close()
    assert count == [True] and not provider._leases
    provider.close()
    original.close()


@pytest.mark.parametrize('after', [False, True])
def test_metadata_release_stages_never_double_release(tmp_path, monkeypatch, after):
    from gf_training.native_group_pack import OriginalArtifactProvider
    root, _ = packed(tmp_path)
    original = owner(root)
    before = original.pool.budget.used
    provider = OriginalArtifactProvider((original,))
    release = original.pool.budget.release
    def fault(amount):
        if after:
            release(amount)
        raise OSError('metadata close fault')
    monkeypatch.setattr(original.pool.budget, 'release', fault)
    with pytest.raises(OSError):
        provider.close()
    with pytest.raises(ContractError, match='cleanup-only'):
        provider.registry()
    monkeypatch.setattr(original.pool.budget, 'release', release)
    provider.close()
    assert original.pool.budget.used == before
    original.close()
