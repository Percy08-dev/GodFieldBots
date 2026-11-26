"""Collection startup reuses validated settings for each used policy label."""

from dataclasses import asdict
from types import SimpleNamespace

from gf_common.card_instance import ActionConfigV3, sacrifice_card_instance_contract
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import EpisodeGroup, EpisodeSpec, RuntimePolicyBinding
from gf_training import rust_player_collection as collection


def test_contexts_validate_each_used_policy_once_with_distinct_seat_data(monkeypatch):
    actions = ActionConfigV3(action_contract=sacrifice_card_instance_contract())
    rules = card_instance_rules()
    bindings = {
        "random": RuntimePolicyBinding("random"),
        "refresh": RuntimePolicyBinding("heuristic_refresh", {"version": "1", "attack_threshold": 7}),
        "unused": RuntimePolicyBinding("random", {}),
    }
    groups = (EpisodeGroup("seed", (
        EpisodeSpec("first", ("random", "refresh"), (17, 19), "pair"),
        EpisodeSpec("second", ("refresh", "random"), (23, 29), "pair"),
    )),)
    registry = SimpleNamespace(journal_inputs=lambda bindings: {}, state={"run_id": "run"},
        local=SimpleNamespace(learner=SimpleNamespace(_context=SimpleNamespace(rules=rules))))
    collector = object.__new__(collection.RustPlayerCollector)
    collector.registry, collector.config, collector.mode = registry, SimpleNamespace(actions=actions), None
    plan = SimpleNamespace(bindings=bindings, groups=groups, linkage={"batch_id": "batch"},
        execution_deadline=60.0)
    original = collection.own_game_worker_settings_v3
    calls = []

    def measured(model_id, actions, options, *, rules):
        calls.append(model_id)
        return original(model_id, actions, options, rules=rules)

    monkeypatch.setattr(collection, "own_game_worker_settings_v3", measured)
    contexts = collector._contexts(plan)

    assert calls == ["random", "heuristic_refresh"]
    first_random, first_refresh = contexts["first"]["policies"]
    second_refresh, second_random = contexts["second"]["policies"]
    for policy, seed in zip((first_random, first_refresh, second_refresh, second_random),
                            (17, 19, 23, 29), strict=True):
        assert policy.native["seed"] == policy.journal["sampling_seed"] == seed
    assert first_refresh.native["attack_threshold"] == second_refresh.native["attack_threshold"] == 7
    caps, costs, _, _ = original("random", actions, None, rules=rules)
    first_random.journal["caps"]["max_positions"] = 1
    first_random.native["numeric_costs"]["position_capacity"] = 1
    assert second_random.journal["caps"] == asdict(caps)
    assert first_random.journal["numeric_costs"] == second_random.native["numeric_costs"] == asdict(costs)

    collector._contexts(plan)
    assert calls == ["random", "heuristic_refresh", "random", "heuristic_refresh"]
