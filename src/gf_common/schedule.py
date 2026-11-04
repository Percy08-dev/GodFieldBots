"""Shared pure fixed episode scheduling; no evaluation/training dependency."""
from __future__ import annotations

import time

from gf_common.domain import EpisodeGroup, EpisodeSpec, digest, integer


def baseline_policy_labels(actions_version: int = 1) -> tuple[str, str]:
    integer(actions_version, 1, 3)
    if actions_version == 3:
        return 'random-card-instance-v2', 'heuristic-card-instance-v2'
    return (("random", "heuristic") if actions_version == 1
            else ("random-structured-v1", "heuristic-structured-v1"))


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("schedule deadline")


def schedule(seeds: tuple[str, ...], sampling_seeds: tuple[int, ...], pairs_per_seed: int,
             model: str, opponent: str, *, deadline: float = float("inf")) -> tuple[EpisodeGroup, ...]:
    """Sampling RNG derives solely from public experiment sampling/schedule indices."""
    _check_deadline(deadline)
    groups = []
    for seed_index, seed in enumerate(seeds):
        _check_deadline(deadline)
        episodes = []
        for sampling_index, sampling_seed in enumerate(sampling_seeds):
            for repeat in range(pairs_per_seed):
                _check_deadline(deadline)
                pair_id = f"s{seed_index}-m{sampling_index}-r{repeat}"
                for swap in range(2):
                    # Keep each model's policy stream fixed across its seat swap.
                    sampling = (int(digest({"sampling_seed": sampling_seed, "repeat": repeat, "policy_stream": 0})[:16], 16),
                                int(digest({"sampling_seed": sampling_seed, "repeat": repeat, "policy_stream": 1})[:16], 16))
                    policies = (model, opponent) if swap == 0 else (opponent, model)
                    if swap:
                        sampling = (sampling[1], sampling[0])
                    episodes.append(EpisodeSpec(pair_id + f"-seat{swap}", policies, sampling, pair_id))
        groups.append(EpisodeGroup(seed, tuple(episodes)))
    return tuple(groups)


def serialize_schedule(groups: tuple[EpisodeGroup, ...], schema: int) -> list[dict[str, object]]:
    """Explicit historical serialization; schema 3 never gains new seat semantics."""
    from dataclasses import asdict

    from gf_common.domain import ContractError, clone
    if schema not in (3, 4):
        raise ContractError("unsupported schedule schema")
    result = [asdict(group) for group in groups]
    if schema == 3:
        for group in result:
            for spec in group["episodes"]:
                spec.pop("learner_viewer")
    return clone(result)
