"""Predetermined paired schedules, explicit failure denominators and clustered CI."""
from __future__ import annotations

import math
import time
from dataclasses import asdict

import numpy as np

from gf_common.domain import (
    EpisodeGroup,
    EpisodeResult,
    Json,
    group_health_failures,
)
from gf_common.schedule import schedule as schedule  # noqa: PLC0414 - compatibility re-export


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("evaluation computation deadline")


def _mean(values: list[float] | list[int], deadline: float) -> float:
    _check_deadline(deadline)
    value = float(np.mean(values))
    _check_deadline(deadline)
    return value


def _quantile(values: list[float], quantile: float, deadline: float) -> float:
    _check_deadline(deadline)
    value = float(np.quantile(values, quantile))
    _check_deadline(deadline)
    return value



def _score(episode: EpisodeResult, seat: int) -> float:
    if episode.outcome == "draw":
        return 0.5
    if isinstance(episode.outcome, dict) and "win" in episode.outcome:
        return 1.0 if episode.outcome["win"]["winner"] == seat else 0.0
    raise ValueError("natural episode missing result")


def clustered_interval(clusters: dict[str, list[float]], samples: int = 2000, seed: int = 0, *, deadline: float = float("inf")) -> list[float] | None:
    _check_deadline(deadline)
    if len(clusters) < 2:
        return None  # One independent seed cannot estimate seed variability.
    values = []
    for _, scores in sorted(clusters.items()):
        _check_deadline(deadline)
        values.append(np.asarray(scores, dtype=float))
    _check_deadline(deadline)
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        _check_deadline(deadline)
        sampled = generator.integers(0, len(values), len(values))
        estimates.append(float(np.mean(np.concatenate([values[i] for i in sampled]))))
    _check_deadline(deadline)
    bounds = [float(q) for q in np.quantile(estimates, [0.025, 0.975])]
    _check_deadline(deadline)
    return bounds


def report(episodes: list[EpisodeResult], planned: tuple[EpisodeGroup, ...], *, bootstrap_samples: int = 2000,
           bootstrap_seed: int = 0, wall_seconds: float = 0, groups_health: list[Json] | None = None, deadline: float = float("inf")) -> Json:
    _check_deadline(deadline)
    specs = {}
    for group in planned:
        for schedule_position, spec in enumerate(group.episodes):
            _check_deadline(deadline)
            specs[spec.episode_id] = (group.seed, spec, spec.learner_viewer - 1 if spec.learner_viewer is not None else schedule_position % 2)
    by_id = {episode.episode_id: episode for episode in episodes}
    pairs: dict[str, list[str]] = {}
    for episode_id, (_, spec, _) in specs.items():
        pairs.setdefault(spec.pair_id, []).append(episode_id)
    complete_pairs = []
    for ids in pairs.values():
        _check_deadline(deadline)
        if len(ids) == 2 and all(i in by_id and by_id[i].end_kind == "natural" for i in ids):
            complete_pairs.append(ids)
    clusters: dict[str, list[float]] = {}
    wdl = {"wins": 0, "draws": 0, "losses": 0}
    first_positions = {"first": {"natural": 0, "score_sum": 0.0}, "second": {"natural": 0, "score_sum": 0.0}}
    seats = {"1": {"natural": 0, "score_sum": 0.0}, "2": {"natural": 0, "score_sum": 0.0}}
    opponents: dict[str, dict[str, int]] = {}
    end_counts: dict[str, int] = {}
    failures: dict[str, int] = {}
    health_failures: dict[str, int] = {}
    for episode in episodes:
        _check_deadline(deadline)
        end_counts[episode.end_kind] = end_counts.get(episode.end_kind, 0) + 1
        if episode.failure:
            failures[episode.failure] = failures.get(episode.failure, 0) + 1
        for key, value in episode.health.items():
            if value:
                health_failures[key] = health_failures.get(key, 0) + 1
        if episode.end_kind == "natural":
            _, _, swapped = specs[episode.episode_id]
            seat = 1 if swapped == 0 else 2
            score = _score(episode, seat)
            key = "draws" if score == 0.5 else "wins" if score == 1 else "losses"
            wdl[key] += 1
            opposing_label = specs[episode.episode_id][1].policies[2 - seat]
            opponents.setdefault(opposing_label, {"wins": 0, "draws": 0, "losses": 0})[key] += 1
            seats[str(seat)]["natural"] += 1
            seats[str(seat)]["score_sum"] += score
            if episode.first_player in (1, 2):
                position = "first" if seat == episode.first_player else "second"
                first_positions[position]["natural"] += 1
                first_positions[position]["score_sum"] += score
    for ids in complete_pairs:
        _check_deadline(deadline)
        for identity in ids:
            seed, _, swapped = specs[identity]
            clusters.setdefault(seed, []).append(_score(by_id[identity], 1 if swapped == 0 else 2))
    candidate_counts = []
    latencies = []
    inference_latencies = []
    inference_by_policy: dict[str, list[float]] = {}
    attempts = []
    episode_records = []
    for episode in episodes:
        _check_deadline(deadline)
        candidate_counts.extend(episode.candidate_counts)
        latencies.extend(episode.search_latency_ms)
        inference_latencies.extend(episode.inference_latency_ms)
        for policy, policy_latencies in episode.inference_latency_by_policy.items():
            inference_by_policy.setdefault(policy, []).extend(policy_latencies)
        attempts.extend(episode.search_attempts)
        episode_records.append(asdict(episode))
    _check_deadline(deadline)
    decisions = sum(e.decisions for e in episodes)
    group_failures: dict[str, int] = {}
    for health in groups_health or []:
        for failure in group_health_failures(health):
            group_failures[failure] = group_failures.get(failure, 0) + 1
    comparison_valid = not group_failures and len(complete_pairs) == len(pairs) and bool(pairs) and not health_failures and not failures
    summary = {"schema_version": 1, "planned_games": len(specs), "recorded_games": len(episodes),
            "missing_games": len(specs) - len(by_id), "natural_denominator": sum(wdl.values()), **wdl,
            "end_counts": end_counts, "failures": failures, "health_failures": health_failures, "group_health_failures": group_failures,
            "planned_pairs": len(pairs), "complete_pairs": len(complete_pairs),
            "comparison_valid": comparison_valid,
            "paired_score": _mean([s for v in clusters.values() for s in v], deadline) if comparison_valid else None,
            "seed_cluster_ci95": clustered_interval(clusters, bootstrap_samples, bootstrap_seed, deadline=deadline) if comparison_valid else None,
            "independent_seed_clusters": len(clusters), "seat_breakdown": seats, "first_player_breakdown": first_positions,
            "opponent_breakdown": opponents,
            "decisions": decisions, "wall_seconds": wall_seconds,
            "decisions_per_second": decisions / wall_seconds if wall_seconds else None,
            "candidate_count_mean": _mean(candidate_counts, deadline) if candidate_counts else None,
            "search_attempts_mean": _mean(attempts, deadline) if attempts else None,
            "search_latency_ms_p50": _quantile(latencies, 0.5, deadline) if latencies else None,
            "search_latency_ms_p95": _quantile(latencies, 0.95, deadline) if latencies else None,
            "inference_latency_ms_p50": _quantile(inference_latencies, .5, deadline) if inference_latencies else None,
            "inference_latency_ms_p95": _quantile(inference_latencies, .95, deadline) if inference_latencies else None,
            "inference_latency_scope": "policy.select only after candidate generation; global mixes all acting policies",
            "inference_by_policy": {policy: {"count": len(values), "p50_ms": _quantile(values, .5, deadline), "p95_ms": _quantile(values, .95, deadline)} for policy, values in inference_by_policy.items()},
            "peak_memory_bytes": None, "artifact_bytes": None,
            "episodes": episode_records}
    _check_deadline(deadline)
    return summary


def final_test_pairs(validation_half_width: float, pilot_pairs: int = 64) -> int:
    if not math.isfinite(validation_half_width) or validation_half_width < 0 or pilot_pairs != 64:
        raise ValueError("invalid fixed validation pilot")
    return max(256, min(2048, math.ceil(pilot_pairs * (validation_half_width / 0.05) ** 2)))
