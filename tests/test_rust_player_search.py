"""Own-player native v5 compatibility, independent of Python implementation calls."""

import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import gf_native
import pytest

from gf_common import card_instance
from gf_common.card_instance import (
    ActionConfigV3,
    shared_card_instance_contract,
    strength_powder_card_instance_contract,
)
from gf_common.card_instance_rules import card_instance_rules
from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV3
from gf_common.domain import ContractError, EmptyCandidates, digest
from gf_common.refresh_heuristic import RefreshOptions
from gf_common.structured import DecisionFrame, EventBank, event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_wire import frame_dict, game_generated_dict_v3
from gf_models.card_instance_baselines import CardInstanceBaseline
from gf_models.refresh_heuristic import RefreshHeuristicPolicy
from tests.card_instance_action_support import inputs

ORIGINALS = Path(__file__).parent / "fixtures/rust_player_v5/originals.json"


def invoke(name, payload):
    return json.loads(getattr(gf_native, name)(json.dumps(payload).encode()))


@pytest.mark.parametrize(
    "episode", json.loads(ORIGINALS.read_text())["episodes"], ids=lambda e: e["name"]
)
def test_native_original_v5(episode):
    events = []
    state = None
    for decision in episode["decisions"]:
        events.extend(decision["delta"]["events"])
        result = invoke(
            "player_search_v5",
            {
                "frame": decision["frame"],
                "events": events,
                "actions": episode["actions"],
                "rules": episode["rules"],
                "numeric_costs": episode["numeric_costs"],
                "rng_state": decision["search_rng_before"],
                "clock": [row["seconds"] for row in decision["clock_trace"]],
            },
        )
        assert result["generated"] == decision["generated"]
        assert result["rng_state"] == decision["search_rng_after"]
        assert result["clock_calls"] == len(decision["clock_trace"])
        for kind in ("random", "heuristic", "refresh"):
            expected = decision["policies"][kind]
            selected = invoke(
                "player_select_v5",
                {
                    "frame": decision["frame"],
                    "events": events,
                    "actions": episode["actions"],
                    "rules": episode["rules"],
                    "generated": result["generated"],
                    "kind": kind,
                    "rng_state": expected["rng_before"],
                    "state": state if kind == "refresh" else None,
                    "refresh_options": episode["refresh_options"],
                },
            )
            assert selected["selection"] == expected["selection"]
            assert selected["rng_state"] == expected["rng_after"]
            if kind == "refresh":
                state = selected["state"]
                for repeated in expected["repeated_selections"]:
                    again = invoke(
                        "player_select_v5",
                        {
                            "frame": decision["frame"],
                            "events": events,
                            "actions": episode["actions"],
                            "rules": episode["rules"],
                            "generated": result["generated"],
                            "kind": kind,
                            "rng_state": None,
                            "state": state,
                            "refresh_options": episode["refresh_options"],
                        },
                    )
                    assert again["selection"] == repeated
                    assert again["state"] == state


@pytest.mark.parametrize("seed", [0, 1, 17, 2**32 - 1, 2**32, 2**63, 2**64 - 1])
def test_native_python_rng(seed):
    expected = random.Random(seed)
    bits = [0, 1, 2, 31, 32, 33, 63, 64]
    draws = [expected.getrandbits(width) for width in bits]
    ranges = [1, 2, 3, 17, 65536, 2**32 + 1, 2**63]
    choices = [expected.randrange(size) for size in ranges]
    shuffled = list(range(33))
    expected.shuffle(shuffled)
    result = invoke("player_rng_v5", {"seed": seed, "bits": bits, "ranges": ranges, "shuffle": 33})
    assert result == {
        "bits": draws,
        "ranges": choices,
        "shuffle": shuffled,
        "state": json.loads(json.dumps(expected.getstate())),
        "search_seed": int(
            digest({"recipe": "structured-search-stream-v1", "sampling_seed": seed})[:16], 16
        ),
    }


# Differential tests deliberately build fresh own views through the unchanged
# Python implementation. They do not derive expected rows from native output.

RULES = card_instance_rules()
CATALOG = json.loads((ORIGINALS.parents[1] / "card_instance_v2_catalog.json").read_text())[
    "definitions"
]
OWNED = [d for d in CATALOG if d["classification"]["definition_class"] == "owned_card"]


class TraceClock:
    def __init__(self, step=0.0, ceiling=None):
        self.values = []
        self.step, self.ceiling = step, ceiling

    def __call__(self):
        value = len(self.values) * self.step
        if self.ceiling is not None:
            value = min(value, self.ceiling)
        self.values.append(value)
        return value


def own_frame(value, *, episode="native-differential"):
    bank = EventBank(1, episode, layout=LAYOUT)
    bank.append(event_delta(1, episode, 0, bank.boundary.prefix_hash, value.events, layout=LAYOUT))
    return DecisionFrame.from_input(value, bank, 0, layout=LAYOUT), bank


def differential(value, *, seed=17, costs=None, clock=None, contract=None, **settings):
    frame, bank = own_frame(value)
    config = ActionConfigV3(
        action_contract=shared_card_instance_contract() if contract is None else contract,
        **settings,
    )
    costs = costs or NumericCosts()
    py_rng = random.Random(seed)
    clock = clock or TraceClock()
    payload = {
        "frame": frame_dict(frame, layout=LAYOUT),
        "events": list(value.events),
        "actions": asdict(config),
        "rules": asdict(RULES),
        "numeric_costs": asdict(costs),
        "rng_state": py_rng.getstate(),
        "clock": [],
    }
    try:
        expected = PublicCandidateGeneratorV3(config, rules=RULES, clock=clock).generate_game(
            frame, bank.view(), py_rng, costs
        )
    except ContractError, EmptyCandidates:
        # The native diagnostic fails closed too, without inventing a fallback.
        payload["clock"] = clock.values
        with pytest.raises(ValueError):
            invoke("player_search_v5", payload)
        return None
    payload["clock"] = clock.values
    native = invoke("player_search_v5", payload)
    assert native["generated"] == json.loads(
        json.dumps(
            game_generated_dict_v3(expected, action_contract=config.action_contract, rules=RULES)
        )
    )
    assert native["rng_state"] == json.loads(json.dumps(py_rng.getstate()))
    assert native["clock_calls"] == len(clock.values)
    for kind in ("random", "heuristic", "refresh"):
        policy = (
            RefreshHeuristicPolicy(config, rules=RULES, options=RefreshOptions())
            if kind == "refresh"
            else CardInstanceBaseline(kind, config, rules=RULES)
        )
        policy.reset(seed)
        rng_before = None if kind == "refresh" else policy.rng.getstate()
        selected = policy.select_frame(frame, bank.view(), expected)
        actual = invoke(
            "player_select_v5",
            {
                "frame": payload["frame"],
                "events": payload["events"],
                "actions": payload["actions"],
                "rules": payload["rules"],
                "generated": native["generated"],
                "kind": kind,
                "rng_state": rng_before,
                "state": None,
                "refresh_options": asdict(RefreshOptions()),
            },
        )
        assert actual["selection"] == asdict(selected)
        if kind != "refresh":
            assert actual["rng_state"] == json.loads(json.dumps(policy.rng.getstate()))
    return native


@pytest.mark.parametrize("definition", OWNED, ids=lambda d: d["definition_id"])
def test_every_owned_perceived_definition(definition):
    name = definition["definition_id"]
    state = "learned" if definition["kind"] == "miracles" else None
    for mp, choice, flash in [
        (0, "turn_action", False),
        (50, "turn_action", False),
        (50, "defense", False),
        (50, "defense", True),
    ]:
        differential(inputs([(name, state)], mp=mp, choice=choice, flash=flash, can_pray=True))


@pytest.mark.parametrize("seed", range(40))
def test_mixed_instances_and_bounded_breadth(seed):
    recipe = random.Random(seed)
    definitions = recipe.choices(OWNED, k=recipe.randrange(2, 8))
    spec = [
        (
            d["definition_id"],
            recipe.choice(["learned", "unused"]) if d["kind"] == "miracles" else None,
        )
        for d in definitions
    ]
    value = inputs(
        spec,
        mp=recipe.choice([0, 5, 50]),
        can_pray=True,
        end_times=bool(seed % 3 == 0),
        choice="defense" if seed % 4 == 0 else "turn_action",
        usable=range(len(spec)) if seed % 5 else range(0, len(spec), 2),
    )
    differential(
        value,
        seed=seed,
        count_states=recipe.choice([1, 30, 16384]),
        max_candidates=recipe.choice([32, 128, 32768]),
        max_expansions=recipe.choice([40, 200, 262144]),
        channel_weights=tuple(recipe.randrange(1, 5) for _ in range(7)),
    )


@pytest.mark.parametrize("kind", ["buy", "sell", "exchange", "callPhenomenon"])
def test_trade_and_allocation_order(kind):
    definitions = [d for d in OWNED if d.get("ability") == kind]
    assert definitions
    d = definitions[0]
    spec = [
        (d["definition_id"], None),
        ("flame", "unused"),
        ("steel-armor", None),
        ("bronze-club", None),
    ]
    for length in [1, 4]:
        differential(
            inputs(spec, mp=10, can_pray=True),
            max_sequence_length=length,
            max_candidates=256,
            count_states=100,
            max_expansions=3000,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_candidates", 1),
        ("max_temporary_candidates", 1),
        ("max_expansions", 1),
        ("count_states", 1),
    ],
)
def test_mandatory_and_search_limits(field, value):
    differential(
        inputs([("flame", "learned"), ("spiritual-socks", None)], mp=0, can_pray=True),
        **{field: value},
    )


def test_numeric_model_cost_and_resource_cutoff():
    value = inputs([("abyss-dart", None)] * 4, mp=0, can_pray=True)
    neural = NumericCosts(
        context_width=16,
        card_width=8,
        actor_input_width=1050,
        actor_hidden_width=32,
        tail_operations_per_candidate=500,
        tail_working_bytes_per_candidate=256,
        tail_tensor_bytes_per_candidate=128,
    )
    differential(value, costs=neural)
    for field in [
        "max_preparation_bytes",
        "max_scoring_operations",
        "max_storage_bytes",
        "max_tensor_bytes",
    ]:
        differential(value, costs=replace(neural, **{field: 10}))


def test_empty_geometry_and_full18_possession():
    differential(inputs([], discard=False))
    differential(inputs([], choice="buy_accept_reject"))
    native = differential(
        inputs([("steel-armor", None)] * 18, usable=(), can_pray=True),
        max_sequence_length=2,
        max_candidates=512,
        count_states=100,
    )
    assert len(native["generated"]["table"]["kinds"]) > 100


def test_clock_cutoffs_with_fresh_combination():
    differential(
        inputs([("abyss-dart", None)] * 7, can_pray=True),
        max_candidates=512,
        count_milliseconds=1,
        search_milliseconds=10,
        clock=TraceClock(step=0.00001, ceiling=0.009),
    )


def test_frame_layout_binding_is_distinct_from_shared_action_binding():
    episode = json.loads(ORIGINALS.read_text())["episodes"][3]
    decision = episode["decisions"][0]
    frame = decision["frame"]
    binding = {k: v for k, v in frame.items() if k != "frame_hash"}
    binding.update(
        input_recipe=LAYOUT.input_recipe,
        policy_input_version=LAYOUT.policy_input_version,
        action_contract=asdict(LAYOUT.action_contract),
    )
    assert (
        binding["action_contract"]["simulator_revision"]
        == "91d5926a7572db327eaea3751e7c24f428120996"
    )
    assert (
        episode["actions"]["action_contract"]["simulator_revision"]
        == "072dabbca1458c9526897781d2fd64edeeb4f9ca"
    )
    assert digest(binding) == frame["frame_hash"]
    binding["action_contract"] = episode["actions"]["action_contract"]
    assert digest(binding) != frame["frame_hash"]


@pytest.mark.parametrize(
    "mutation",
    ["foreign_hint", "hint_revision", "future_event", "perceived_catalog", "hidden_field"],
)
def test_own_input_boundary_refuses_mismatch(mutation):
    episode = json.loads(ORIGINALS.read_text())["episodes"][2]
    d = episode["decisions"][0]
    payload = {
        "frame": d["frame"],
        "events": d["delta"]["events"],
        "actions": episode["actions"],
        "rules": episode["rules"],
        "numeric_costs": episode["numeric_costs"],
        "rng_state": d["search_rng_before"],
        "clock": [r["seconds"] for r in d["clock_trace"]],
    }
    raw = payload["frame"]["payload"]
    if mutation == "foreign_hint":
        raw["action_options"]["action_options"]["usable_cards"][0]["card_ref"] = "c1_" + "f" * 32
    elif mutation == "hint_revision":
        raw["action_options"]["action_options"]["state_revision"] += 1
    elif mutation == "future_event":
        payload["events"].append({"type": "turn_started", "turn": 2, "player": 2})
    elif mutation == "perceived_catalog":
        raw["catalog"]["abyss-dart"]["attack"] += 1
    else:
        raw["observation"]["game_seed"] = 17
    with pytest.raises(ValueError):
        invoke("player_search_v5", payload)


def test_large_table_reaches65536_without_dense_candidate_features():
    result = differential(
        inputs([("abyss-dart", None)] * 18, mp=0),
        max_sequence_length=4,
        max_candidates=65536,
        max_temporary_candidates=80000,
        max_expansions=250000,
        count_states=1,
    )
    assert len(result["generated"]["table"]["kinds"]) == 65536


@pytest.mark.parametrize(
    "attack,threshold",
    [
        ("gravity-mace", 11),
        ("gravity-mace", 12),
        ("direct-smash-axe", 13),
        ("direct-smash-axe", 2**100),
    ],
)
def test_refresh_threshold_identity_and_inclusive_boundary(attack, threshold):
    episode = RefreshEpisode(threshold=threshold)
    episode.step([(attack, None)], can_pray=True)


class RefreshEpisode:
    def __init__(self, name="native-refresh", threshold=12):
        self.bank = EventBank(1, name, layout=LAYOUT)
        self.actions = ActionConfigV3(
            action_contract=shared_card_instance_contract(),
            max_candidates=512,
            max_expansions=10000,
            count_states=200,
        )
        self.options = RefreshOptions(attack_threshold=threshold)
        self.policy = RefreshHeuristicPolicy(self.actions, rules=RULES, options=self.options)
        self.state = None
        self.revision = 0
        self.last = None

    def step(self, spec, *, turn=1, events=(), **options):
        value = inputs(spec, **options)
        value.observation.update(turn=turn, state_revision=self.revision)
        value.action_options["state_revision"] = self.revision
        value.action_options["action_options"]["state_revision"] = self.revision
        start = self.bank.boundary.end
        self.bank.append(
            event_delta(
                1, self.bank.episode, start, self.bank.boundary.prefix_hash, events, layout=LAYOUT
            )
        )
        value = replace(value, events=self.bank.view().events())
        frame = DecisionFrame.from_input(value, self.bank, start, layout=LAYOUT)
        generated = PublicCandidateGeneratorV3(
            self.actions, rules=RULES, clock=lambda: 0.0
        ).generate_game(frame, self.bank.view(), random.Random(17), NumericCosts())
        selected = self.policy.select_frame(frame, self.bank.view(), generated)
        payload = {
            "frame": frame_dict(frame, layout=LAYOUT),
            "events": list(self.bank.view().events()),
            "actions": asdict(self.actions),
            "rules": asdict(RULES),
            "generated": game_generated_dict_v3(
                generated, action_contract=self.actions.action_contract, rules=RULES
            ),
            "kind": "refresh",
            "rng_state": None,
            "state": self.state,
            "refresh_options": asdict(self.options),
        }
        result = invoke("player_select_v5", payload)
        assert result["selection"] == asdict(selected)
        self.state = result["state"]
        payload["state"] = self.state
        assert invoke("player_select_v5", payload) == result
        self.revision += 1
        self.last = payload
        return selected.index


@pytest.mark.parametrize("damage", [0, 3, 4, 10, 12, 99])
def test_refresh_observed_windows_age_only_on_distinct_own_turns(damage):
    episode = RefreshEpisode()
    episode.step([("direct-smash-axe", None)])
    episode.step(
        [("steel-armor", None)],
        choice="defense",
        events=(
            {"type": "damage_applied", "player": 2, "amount": damage},
            {"type": "damage_applied", "player": 1, "amount": 99},
        ),
    )
    episode.step(
        [("steel-armor", None)],
        choice="defense",
        events=(
            {"type": "turn_started", "turn": 2, "active_player": 2},
            {"type": "damage_applied", "player": 2, "amount": 99},
        ),
    )
    for age in [1, 2, 3, 4]:
        episode.step([("bronze-club", None)], turn=1 + age * 2, can_pray=True)
        episode.step([], turn=1 + age * 2, choice="buy_accept_reject")


@pytest.mark.parametrize("case", ["foreign", "prefix", "frontier", "repeated_table"])
def test_refresh_rejects_foreign_or_divergent_history(case):
    episode = RefreshEpisode()
    episode.step(
        [("direct-smash-axe", None)], events=({"type": "damage_applied", "player": 2, "amount": 1},)
    )
    payload = json.loads(json.dumps(episode.last))
    if case == "foreign":
        foreign = RefreshEpisode(name="another-participant")
        foreign.step([("direct-smash-axe", None)])
        payload = foreign.last
        payload["state"] = episode.state
    elif case == "prefix":
        foreign = RefreshEpisode()
        foreign.step(
            [("direct-smash-axe", None)],
            events=({"type": "damage_applied", "player": 2, "amount": 2},),
        )
        payload = foreign.last
        payload["state"] = episode.state
    elif case == "frontier":
        episode.step([("bronze-club", None)], turn=3, can_pray=True)
        payload["state"] = episode.state
    else:
        table = payload["generated"]["table"]
        table["targets"][0] = "self"
        table["table_hash"] = digest({k: v for k, v in table.items() if k != "table_hash"})
        payload["generated"]["report"]["table_hash"] = table["table_hash"]
    with pytest.raises(ValueError):
        invoke("player_select_v5", payload)


def test_refresh_two_participant_states_remain_independent_when_interleaved():
    a, b = RefreshEpisode("own-a"), RefreshEpisode("own-b")
    a.step([("direct-smash-axe", None)])
    b.step([("bronze-club", None)], can_pray=True)
    a.step(
        [("steel-armor", None)],
        choice="defense",
        events=(
            {"type": "damage_applied", "player": 2, "amount": 12},
            {"type": "turn_started", "turn": 2, "active_player": 2},
        ),
    )
    a.step([("bronze-club", None)], turn=3, can_pray=True)
    b.step([("bronze-club", None)], turn=3, can_pray=True)
    assert a.last["generated"]["table"]["kinds"][a.state["last"]["index"]] == "play_sequence"
    assert b.last["generated"]["table"]["kinds"][b.state["last"]["index"]] == "pray"


@pytest.mark.parametrize("seed", [0, 1, 17, 2**32, 2**64 - 1])
def test_search_stream_seed_recipe(seed):
    actual = invoke("player_rng_v5", {"seed": seed, "bits": [], "ranges": [], "shuffle": 0})
    assert actual["search_seed"] == int(
        digest({"recipe": "structured-search-stream-v1", "sampling_seed": seed})[:16], 16
    )


@pytest.mark.parametrize(
    "spec,options",
    [
        ([("flame", "learned"), ("spiritual-socks", None)], {"mp": 0}),
        ([("wall", "learned"), ("spiritual-socks", None)], {"mp": 0, "choice": "defense"}),
        (
            [("wall", "learned"), ("spiritual-socks", None)],
            {"mp": 0, "choice": "defense", "flash": True},
        ),
        (
            [
                ("fireball", "unused"),
                ("fireball", "learned"),
                ("aura", "learned"),
                ("mirage", "unused"),
            ],
            {},
        ),
        ([("saw-boom-boom", None), ("angel-bow", None), ("battle-ball", None)], {}),
        ([("magical-stick", None), ("aura", "learned"), ("spiritual-socks", None)], {"mp": 0}),
        ([("magical-stick", None), ("aura", "learned"), ("spiritual-socks", None)], {"mp": 12}),
        ([("wind", "learned"), ("spring", "unused"), ("spiritual-socks", None)], {"mp": 0}),
        (
            [("sun-amulet", None), ("dangerous-mortar", None), ("bronze-club", None)],
            {"end_times": True},
        ),
        (
            [("rainbow-curtain", None), ("angel-armor", None), ("steel-armor", None)],
            {"choice": "defense"},
        ),
    ],
)
def test_sequence_payment_roles_rounds_and_uncertain_defense(spec, options):
    for states in [1, 16384]:
        differential(
            inputs(spec, can_pray=True, **options), count_states=states, max_candidates=128
        )


@pytest.mark.parametrize("mutation", ["kind", "target", "carrier", "presence", "negative_mp"])
def test_supplied_table_boundary_validates_typed_syntax(mutation):
    episode = json.loads(ORIGINALS.read_text())["episodes"][2]
    d = episode["decisions"][0]
    generated = d["generated"]
    table = generated["table"]
    if mutation == "kind":
        table["kinds"][0] = "hidden_action"
    elif mutation == "target":
        table["targets"][0] = "null"
    elif mutation == "carrier":
        table["kinds"][0] = "sell"
    elif mutation == "presence":
        table["presence"][0][0] = True
    else:
        table["kinds"][0] = "exchange"
        table["targets"][0] = "omitted"
        table["presence"][0] = [True, True, True, False]
        table["numerics"][0] = [0, -1, 0, 0]
    table["table_hash"] = digest({k: v for k, v in table.items() if k != "table_hash"})
    generated["report"]["table_hash"] = table["table_hash"]
    with pytest.raises(ValueError):
        invoke(
            "player_select_v5",
            {
                "frame": d["frame"],
                "events": d["delta"]["events"],
                "actions": episode["actions"],
                "rules": episode["rules"],
                "generated": generated,
                "kind": "refresh",
                "rng_state": None,
                "state": None,
                "refresh_options": episode["refresh_options"],
            },
        )


@pytest.mark.parametrize(
    "deadline_stage,first_expired_read", [("selection", 12), ("construction", 26)]
)
def test_discretionary_only_presentation_deadline_refuses_empty_table(
    deadline_stage, first_expired_read
):
    value = inputs([("sell", None), ("bronze-club", None)], usable=(), discard=False)
    frame, bank = own_frame(value)
    config = ActionConfigV3(action_contract=shared_card_instance_contract())
    trace = TraceClock()
    expected = PublicCandidateGeneratorV3(config, rules=RULES, clock=trace).generate_game(
        frame, bank.view(), random.Random(17), NumericCosts()
    )
    assert expected.report.mandatory == 0 and len(expected.table.kinds) == 2
    assert len(trace.values) == 29
    payload = {
        "frame": frame_dict(frame, layout=LAYOUT),
        "events": list(value.events),
        "actions": asdict(config),
        "rules": asdict(RULES),
        "numeric_costs": asdict(NumericCosts()),
        "rng_state": random.Random(17).getstate(),
        "clock": trace.values,
    }
    control = invoke("player_search_v5", payload)
    assert control["generated"] == json.loads(
        json.dumps(
            game_generated_dict_v3(expected, action_contract=config.action_contract, rules=RULES)
        )
    )
    assert control["clock_calls"] == 29
    readings = [0.0] * first_expired_read + [1.0] * (29 - first_expired_read)
    replay = iter(readings)
    with pytest.raises(ContractError, match="invalid compact candidate columns"):
        PublicCandidateGeneratorV3(config, rules=RULES, clock=lambda: next(replay)).generate_game(
            frame, bank.view(), random.Random(17), NumericCosts()
        )
    payload["clock"] = readings
    with pytest.raises(ValueError, match="invalid compact candidate columns"):
        invoke("player_search_v5", payload)


@pytest.mark.parametrize("maximum", [18, 19, 128])
@pytest.mark.parametrize("kind", ["random", "heuristic", "refresh"])
def test_baseline_capacity_is18_while_generator_capacity_remains128(maximum, kind):
    value = inputs([("bronze-club", None)], can_pray=True)
    frame, bank = own_frame(value)
    config = ActionConfigV3(
        action_contract=shared_card_instance_contract(), max_sequence_length=maximum
    )
    trace = TraceClock()
    expected = PublicCandidateGeneratorV3(config, rules=RULES, clock=trace).generate_game(
        frame, bank.view(), random.Random(17), NumericCosts()
    )
    search = invoke(
        "player_search_v5",
        {
            "frame": frame_dict(frame, layout=LAYOUT),
            "events": list(value.events),
            "actions": asdict(config),
            "rules": asdict(RULES),
            "numeric_costs": asdict(NumericCosts()),
            "rng_state": random.Random(17).getstate(),
            "clock": trace.values,
        },
    )
    assert search["generated"] == json.loads(
        json.dumps(
            game_generated_dict_v3(expected, action_contract=config.action_contract, rules=RULES)
        )
    )

    def make_policy():
        return (
            RefreshHeuristicPolicy(config, rules=RULES, options=RefreshOptions())
            if kind == "refresh"
            else CardInstanceBaseline(kind, config, rules=RULES)
        )

    payload = {
        "frame": frame_dict(frame, layout=LAYOUT),
        "events": list(value.events),
        "actions": asdict(config),
        "rules": asdict(RULES),
        "generated": search["generated"],
        "kind": kind,
        "rng_state": None if kind == "refresh" else random.Random(17).getstate(),
        "state": None,
        "refresh_options": asdict(RefreshOptions()),
    }
    if maximum == 18:
        policy = make_policy()
        policy.reset(17)
        selected = policy.select_frame(frame, bank.view(), expected)
        assert invoke("player_select_v5", payload)["selection"] == asdict(selected)
    else:
        with pytest.raises(ContractError, match="corrected baseline kind/game capacity required"):
            make_policy()
        with pytest.raises(ValueError, match="corrected baseline kind/game capacity required"):
            invoke("player_select_v5", payload)


@pytest.mark.parametrize(
    "spec,powder_attacks",
    [
        ([("strength-powder", None)], set()),
        ([("strength-powder", None)] * 2, set()),
        ([("strength-powder", None), ("fireball", "unused")], set()),
        ([("strength-powder", None), ("fireball", "learned")], set()),
        ([("strength-powder", None), ("fireball", "learned"), ("spiritual-staff", None)],
         {(2, 0), (2, 0, 1), (2, 1, 0)}),
        ([("strength-powder", None), ("ogre-s-armor", None)], set()),
        ([("strength-powder", None), ("bronze-club", None)], {(1, 0)}),
        ([("strength-powder", None), ("wand-of-mystic-water", None)], {(0, 1), (1, 0)}),
    ],
)
@pytest.mark.parametrize("use_new", [False, True], ids=["powder-release", "dream-release"])
def test_strength_powder_public_candidates(spec, powder_attacks, use_new):
    value = inputs(spec, can_pray=True)
    value.observation["own"]["status"]["dream"] = True
    contract = (
        card_instance.dream_compat_card_instance_contract() if use_new
        else strength_powder_card_instance_contract()
    )
    native = differential(value, contract=contract)
    assert native is not None
    table = native["generated"]["table"]
    attacks = set()
    for index, kind in enumerate(table["kinds"]):
        slots = tuple(table["slots"][table["offsets"][index]:table["offsets"][index + 1]])
        if kind == "play_sequence" and any(spec[slot][0] == "strength-powder" for slot in slots):
            attacks.add(slots)
        if kind == "discard":
            assert all(spec[slot][0] != "wand-of-mystic-water" for slot in slots)
    assert attacks == powder_attacks


@pytest.mark.parametrize(
    "choice,miracle,visible_card,kind,expected",
    [
        ("turn_action", "ice-age", "saw-boom-boom", "play_sequence", False),
        ("turn_action", "ice-age", "spiritual-staff", "play_sequence", True),
        ("defense", "wall", "leather-cap", "respond", False),
        ("defense", "wall", "spiritual-socks", "respond", True),
        ("defense", "wall", "spiritual-hood", "respond", True),
        ("defense", "wall", "spiritual-sash", "respond", True),
    ],
)
def test_dream_compat_costcut_candidates_use_visible_roles_only(choice, miracle, visible_card, kind, expected):
    value = inputs([(miracle, "unused"), (visible_card, None)], mp=0, choice=choice, can_pray=True)
    value.observation["own"]["status"]["dream"] = True
    native = differential(value, contract=card_instance.dream_compat_card_instance_contract())
    assert native is not None
    table = native["generated"]["table"]
    has_combo = any(
        action_kind == kind and table["slots"][table["offsets"][i]:table["offsets"][i + 1]] == [0, 1]
        for i, action_kind in enumerate(table["kinds"])
    )
    assert has_combo is expected


def test_historical_shared_strength_powder_remains_a_completed_singleton():
    native = differential(inputs([("strength-powder", None)], can_pray=True))
    assert native is not None
    assert "play_sequence" in native["generated"]["table"]["kinds"]
