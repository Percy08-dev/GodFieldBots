"""Private numerical token views tied to live sealed bank provenance.

These are not public FrameTokens and never relax the public token validators.
"""
from __future__ import annotations

import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from gf_common.domain import ContractError

if TYPE_CHECKING:
    from gf_common.ram_bank import ReadonlyRamBankSet

FRAME_ARRAY_FIELDS = ('state_numerics', 'state_present', 'state_categories',
    'state_category_present', 'card_numerics', 'card_present', 'card_categories',
    'card_mask', 'opponent_learned', 'history', 'interval_numerics',
    'interval_present', 'interval_categories')
CANDIDATE_ARRAY_FIELDS = ('kinds', 'targets', 'numerics', 'present', 'offsets', 'slots')


@dataclass(frozen=True,slots=True,init=False)
class _OwnedTokenView:
    array_fields: ClassVar[tuple[str,...]]
    _banks: ReadonlyRamBankSet
    _retention: object
    _scalars: Mapping[str,Any]
    _keys: Mapping[str,str]

    def __init__(self, banks: ReadonlyRamBankSet, scalars: dict[str, Any], keys: dict[str, str], retention: object) -> None:
        from gf_common.ram_bank import ReadonlyRamBankSet
        if type(banks) is not ReadonlyRamBankSet or set(keys) != set(self.array_fields):
            raise ContractError('owned token view requires actual sealed bankset/layout')
        object.__setattr__(self,'_banks',banks)
        object.__setattr__(self,'_retention',retention)
        object.__setattr__(self,'_scalars',MappingProxyType(dict(scalars)))
        object.__setattr__(self,'_keys',MappingProxyType(dict(keys)))
        self._require_live()

    def _require_live(self) -> None:
        for key in self._keys.values():
            owner = self._banks._index.get(key)
            if owner is None:
                raise ContractError('owned token array missing')
            owner.fileno()

    def __getattr__(self, name: str) -> Any:
        if name in self._keys:
            values = self._banks.array(self._keys[name])
            # Every ordinary NumPy derived/base alias keeps the frombuffer root.
            # Retain the storage owner there, beyond this particular reshape.
            import numpy as np
            root = values
            while isinstance(root.base, np.ndarray):
                root = root.base
            weakref.finalize(root, lambda owner: None, self._retention)
            return values
        if name in self._scalars:
            return self._scalars[name]
        raise AttributeError(name)


class _OwnedFrameTokens(_OwnedTokenView):
    __slots__=()
    array_fields = FRAME_ARRAY_FIELDS


class _OwnedCandidateTokens(_OwnedTokenView):
    __slots__=()
    array_fields = CANDIDATE_ARRAY_FIELDS


def _require_owned_endpoint(frames: Any, candidates: Any) -> None:
    if (type(frames) is not tuple or not frames
            or type(candidates) is not _OwnedCandidateTokens
            or any(type(frame) is not _OwnedFrameTokens or frame._banks is not candidates._banks for frame in frames)):
        raise ContractError('private sealed owned endpoint required')
    candidates._require_live()
