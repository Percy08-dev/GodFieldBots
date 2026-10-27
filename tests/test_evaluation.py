from gf_common.domain import EpisodeResult
from gf_evaluation.evaluation import clustered_interval, final_test_pairs, report, schedule


def test_schedule_seat_swap_sampling_not_correlated_with_game_seed():
    first = schedule(("seed-a",), (17,), 2, "heuristic", "random")
    other = schedule(("seed-b",), (17,), 2, "heuristic", "random")
    assert first[0].episodes == other[0].episodes
    a, b = first[0].episodes[:2]
    assert a.policies == ("heuristic", "random") and b.policies == ("random", "heuristic")
    assert a.sampling_seeds == tuple(reversed(b.sampling_seeds))
    assert len(first[0].episodes) == 4


def test_natural_outcome_separate_from_health_incomplete_invalidates_pair():
    planned = schedule(("seed-a",), (17,), 1, "heuristic", "random")
    a, b = planned[0].episodes
    one = EpisodeResult(a.episode_id, a.pair_id, "seed-a", a.policies, outcome={"win": {"winner": 1}}, end_kind="natural", first_player=2)
    two = EpisodeResult(b.episode_id, b.pair_id, "seed-a", b.policies, end_kind="truncated", failure="deadline")
    summary = report([one, two], planned)
    assert summary["wins"] == 1 and summary["natural_denominator"] == 1
    assert not summary["comparison_valid"] and summary["paired_score"] is None
    assert summary["complete_pairs"] == 0
    assert summary["first_player_breakdown"]["second"]["natural"] == 1
    two.end_kind = "natural"
    two.outcome = {"win": {"winner": 2}}
    two.first_player = 2
    two.failure = None
    summary = report([one, two], planned)
    assert summary["comparison_valid"] and summary["paired_score"] == 1.0
    assert summary["seed_cluster_ci95"] is None
    two.health["save_failure"] = "disk"
    summary = report([one, two], planned)
    assert summary["natural_denominator"] == 2
    assert not summary["comparison_valid"]


def test_seed_cluster_bootstrap_groups_repeats_and_fixed_pilot():
    clusters = {"seed-a": [1.0] * 6, "seed-b": [0.0] * 2}
    assert clustered_interval(clusters, 100, 13) == clustered_interval(clusters, 100, 13)
    assert clustered_interval(clusters, 100, 13) == [0.0, 1.0]
    assert final_test_pairs(0.1) == 256
    assert final_test_pairs(1.0) == 2048


def test_natural_pair_with_group_cleanup_failure_invalidates_comparison():
    planned = schedule(("seed",), (17,), 1, "heuristic", "random")
    episodes = [EpisodeResult(s.episode_id, s.pair_id, "seed", s.policies, outcome="draw", end_kind="natural") for s in planned[0].episodes]
    for failure in ("client_close_failure", "finalization_cancelled"):
        summary = report(episodes, planned, groups_health=[{failure: "failure"}])
        assert summary["natural_denominator"] == 2 and summary["draws"] == 2
        assert not summary["comparison_valid"]
        assert summary["group_health_failures"] == {failure: 1}


def test_statistics_expired_budget_prevents_work_and_bootstrap_stops_cooperatively(monkeypatch):
    import pytest

    from gf_evaluation import evaluation
    with pytest.raises(TimeoutError):
        report([], (), deadline=0)
    with pytest.raises(TimeoutError):
        clustered_interval({"a": [1.0], "b": [0.0]}, deadline=0)
    calls = 0
    def clock():
        nonlocal calls
        calls += 1
        return 0.0 if calls < 8 else 2.0
    monkeypatch.setattr(evaluation.time, "monotonic", clock)
    with pytest.raises(TimeoutError):
        clustered_interval({"a": [1.0], "b": [0.0]}, samples=1000000, deadline=1)
    assert calls == 8
