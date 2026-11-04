"""Private immutable tuple reconstruction shared by bounded raw-token readers."""
from __future__ import annotations

from typing import Any

from gf_common.domain import ContractError


def raw_token_tuple(array: Any, *, boolean: bool = False) -> tuple[Any, ...]:
    def convert(values: Any) -> Any:
        if isinstance(values, list):
            return tuple(convert(value) for value in values)
        if boolean:
            if type(values) is not int or values not in (0, 1):
                raise ContractError('TreeDataset mask/presence is not binary')
            return bool(values)
        if type(values) is not int:
            raise ContractError('TreeDataset int64 token required')
        return values
    return convert(array.tolist())
