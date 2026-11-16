"""Bounded immutable inference instances, separate from outer artifact/pin custody."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from gf_common.domain import ContractError, integer
from gf_common.hpo_progress import allocator_facts
from gf_common.hpo_resident import CACHE_ENTRIES, CACHE_WEIGHT_BYTES, _sha


def policy_private_state_empty(policy: Any) -> bool:
    experiment = getattr(getattr(policy, 'network', None), '_inference_experiment', None)
    if experiment is not None and not experiment.private_empty():
        return False
    cache = getattr(policy, 'cache', None)
    if cache is not None:
        if (cache.position != 0 or cache.owner is not None or cache.revision != -1 or cache.event_end != 0):
            return False
        for value in (cache.keys, cache.values, cache.past):
            if value is not None and bool(value.count_nonzero().item()):
                return False
    if hasattr(policy,'_fallback') and (policy._last is not None or policy._selection is not None
            or policy._own_turn is not None or policy._decision != 0 or policy._samples or policy._pending is not None):
        return False
    return not getattr(policy, 'last_timing', {})


def reset_policy_for_residency(policy: Any) -> None:
    policy.reset(0)
    cache = getattr(policy,'cache',None)
    if cache is not None:
        # Inactive identities retain weights, never a second activation/history
        # allocation. Acquisition recreates the original network cache before the next lease.
        cache.keys = cache.values = cache.past = None
    fallback = getattr(policy,'_fallback',None)
    if fallback is not None:
        fallback.reset(0)


class ResidentPolicyCache:
    """At most six identities/128MiB; never mutate an existing network's weights."""
    def __init__(self) -> None:
        self._entries: OrderedDict[str, tuple[int, Any]] = OrderedDict()
        self.active: str | None = None
        self.selected: str | None = None
        self.loads = self.reuses = 0
        self.cpu_weight_restore_bytes = self.cuda_weight_restore_bytes = 0
        self.cuda_allocator: Any = None

    @property
    def identities(self) -> tuple[str, ...]:
        return tuple(self._entries)

    @property
    def weight_bytes(self) -> int:
        return sum(size for size, _ in self._entries.values())

    def acquire(self, identity: str, *, weight_bytes: int, load: Callable[[], Any], validate: Callable[[], object]) -> Any:
        _sha(identity)
        integer(weight_bytes, 0, 2**63-1)
        if self.active is not None:
            raise ContractError('resident cache already active')
        if weight_bytes > CACHE_WEIGHT_BYTES:
            raise ContractError('resident cache weight capacity exceeded before allocation')
        validate()  # Even a hit must check the admitted immutable metadata/file hashes.
        if identity in self._entries:
            size, policy = self._entries[identity]
            if size != weight_bytes:
                raise ContractError('resident immutable cache weight identity changed')
            self._entries.move_to_end(identity)
            self.reuses += 1
        else:
            # Outer custody must confirm these exact evictions before releasing pins.
            if len(self._entries) >= CACHE_ENTRIES or self.weight_bytes + weight_bytes > CACHE_WEIGHT_BYTES:
                raise ContractError('resident cache capacity requires explicit confirmed eviction')
            policy = load()
            self._entries[identity] = weight_bytes, policy
            self.loads += 1
            # Completed original restore_weights copies the validated float32
            # manifest arrays to this device. Count payload bytes, not bus overhead.
            if getattr(getattr(policy,'binding',None),'device','cpu')=='cpu':
                self.cpu_weight_restore_bytes += weight_bytes
            else:
                self.cuda_weight_restore_bytes += weight_bytes
        temporal = getattr(policy,'cache',None)
        if temporal is not None and temporal.keys is temporal.values is temporal.past is None:
            policy.cache = policy.network.new_cache()
        self.active = self.selected = identity
        return policy

    def eviction_candidates(self, weight_bytes: int) -> tuple[str, ...]:
        integer(weight_bytes, 0, CACHE_WEIGHT_BYTES)
        count, size, result = len(self._entries), self.weight_bytes, []
        for identity, (entry_bytes, _) in self._entries.items():
            if count < CACHE_ENTRIES and size + weight_bytes <= CACHE_WEIGHT_BYTES:
                break
            if identity not in (self.active,self.selected):
                result.append(identity)
                count, size = count-1, size-entry_bytes
        return tuple(result)

    def evict(self, identity: str) -> None:
        if identity == self.active:
            raise ContractError('resident active cache cannot be evicted')
        if identity == self.selected:
            raise ContractError('resident current cache cannot be evicted')
        if identity not in self._entries:
            raise ContractError('resident cache eviction identity absent')
        _, policy = self._entries[identity]
        reset_policy_for_residency(policy)
        if not policy_private_state_empty(policy):
            raise ContractError('resident evicted policy retains private state')
        del self._entries[identity]

    def release(self) -> None:
        if self.active is not None:
            reset_policy_for_residency(self._entries[self.active][1])
            self.active = None

    def reset_all(self) -> None:
        for _, policy in self._entries.values():
            reset_policy_for_residency(policy)
        if not self.private_state_empty():
            raise ContractError('resident cache reset left private state')

    def synchronize(self) -> None:
        for _, policy in self._entries.values():
            if getattr(getattr(policy, 'binding', None), 'device', 'cpu') != 'cpu':
                policy.torch.cuda.synchronize(policy.network.device)
                self.cuda_allocator = allocator_facts(policy)['cuda_allocator']
                break

    def diagnostic(self) -> dict[str,Any]:
        return {'schema_version':1,'loads':self.loads,'cache_reuses':self.reuses,
            'cpu_weight_restore_bytes':self.cpu_weight_restore_bytes,
            'cuda_weight_restore_bytes':self.cuda_weight_restore_bytes,'cuda_allocator':self.cuda_allocator}

    def private_state_empty(self) -> bool:
        return all(policy_private_state_empty(policy) for _, policy in self._entries.values())


def resident_execution_profile(options: dict[str,Any]) -> dict[str,Any]:
    """Only process-wide execution invariants; never model features or cache labels."""
    from gf_models.structured_binding import child_cuda_visibility
    from gf_models.structured_config import InferenceBindingV2
    from gf_models.torch_runtime import NUMERICS
    binding=InferenceBindingV2(**options['inference'])
    visibility=child_cuda_visibility(binding.device,options['cuda_visible_devices'])
    if visibility!=options['cuda_visible_devices']:
        raise ContractError('resident execution requires canonical CUDA visibility')
    return {'device':binding.device,'cuda_visible_devices':visibility,'numerics':dict(NUMERICS)}


def resident_registry_execution(bindings: dict[str,Any]) -> dict[str,Any] | None:
    from gf_common.domain import canonical
    profile=None
    for binding in bindings.values():
        if binding.model_id not in ('mlp','gru','transformer'):
            continue
        current=resident_execution_profile(binding.policy_options)
        if profile is not None and canonical(current)!=canonical(profile):
            raise ContractError('resident registry has incompatible process execution settings')
        profile=current
    return profile
