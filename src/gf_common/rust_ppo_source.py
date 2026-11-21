"""Versioned training data and saved verification for Rust-player PPO."""

from __future__ import annotations

from typing import Any

from gf_common.domain import ContractError, Json, closed

TRAINING_DATA = 'rust-ppo-training-v1'
PACK_AUDIT = 'sealed-pack-sha256-v1'


def validate_source_mode(value: Any) -> Json:
    mode = closed(value, {'training_data', 'audit'})
    if mode['training_data'] != TRAINING_DATA or mode['audit'] != PACK_AUDIT:
        raise ContractError('unsupported Rust PPO source recipe')
    return mode


def source_mode(effective: Json) -> Json | None:
    run = effective.get('run', {})
    return validate_source_mode(run['ppo_source']) if 'ppo_source' in run else None


def source_encoder(effective: Json) -> str:
    return 'canonical-rust' if source_mode(effective) is not None else 'canonical-python'
