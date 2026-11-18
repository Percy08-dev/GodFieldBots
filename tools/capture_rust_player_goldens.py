"""Explicitly capture original Python v5 parity cases, without a Simulator or port.

Run from the repository root: .venv/bin/python -m tools.capture_rust_player_goldens
Only this command writes the new rust_player_v5 fixture directory. Existing
fixtures are inputs and remain untouched. These synthetic participant-visible
cases are comparison originals, never admitted learning or gameplay evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from gf_common.card_instance import ActionConfigV3, shared_card_instance_contract
from gf_common.card_instance_rules import card_instance_rules
from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV3
from gf_common.domain import canonical, digest
from gf_common.refresh_heuristic import RefreshOptions
from gf_common.structured import DecisionFrame, EventBank, event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_wire import delta_dict, frame_dict, game_generated_dict_v3
from gf_models.card_instance_baselines import CardInstanceBaseline
from gf_models.refresh_heuristic import RefreshHeuristicPolicy
from tests.card_instance_action_support import inputs

ROOT = Path(__file__).resolve().parents[1]
REVISION = "9680f06ad8ba61d65d142863742908678638b02b"
OUTPUT = ROOT / "tests/fixtures/rust_player_v5/originals.json"
SPEC07 = "8e5223b45a6e8c4c80e272736ed97b9e271136c939eca74427a035ddfdbfcac2"
RULES = card_instance_rules()
COSTS = NumericCosts()
OPTIONS = RefreshOptions()
SAMPLING_SEED = 17
SEARCH_SEED = int(
    digest(
        {
            "recipe": "structured-search-stream-v1",
            "sampling_seed": SAMPLING_SEED,
        }
    )[:16],
    16,
)


class CaptureClock:
    """Exact original readings and call-site trace; no timing normalization."""

    def __init__(self, step=0.0, ceiling=None):
        self.step, self.ceiling, self.trace = step, ceiling, []

    def __call__(self):
        value = len(self.trace) * self.step
        if self.ceiling is not None:
            value = min(value, self.ceiling)
        self.trace.append({"seconds": value, "caller": sys._getframe(1).f_code.co_name})
        return value


def source_pins():
    paths = sorted((ROOT / "src/gf_common").glob("*.py")) + [
        ROOT / "src/gf_models/card_instance_baselines.py",
        ROOT / "src/gf_models/refresh_heuristic.py",
        ROOT / "tests/card_instance_action_support.py",
        ROOT / "tests/test_card_instance_structured.py",
        ROOT / "tests/test_card_instance_contract.py",
        ROOT / "tests/fixtures/card_instance_v2_catalog.json",
        ROOT / "tests/fixtures/card_instance_v2_runtime_observation.json",
        ROOT / "tests/fixtures/card_instance_v2_runtime_options.json",
    ]
    pins = {}
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        original = subprocess.run(
            ["git", "show", f"{REVISION}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        actual = path.read_bytes()
        if actual != original:
            raise RuntimeError(f"capture requires original {REVISION} source: {relative}")
        pins[relative] = hashlib.sha256(actual).hexdigest()
    return pins


def capture_episode(name, steps, *, settings=None, clock_step=0.0, clock_ceiling=None):
    actions = ActionConfigV3(action_contract=shared_card_instance_contract(), **(settings or {}))
    bank = EventBank(1, name, layout=LAYOUT)
    search = random.Random(SEARCH_SEED)
    policies = {
        kind: CardInstanceBaseline(kind, actions, rules=RULES) for kind in ("random", "heuristic")
    }
    policies["refresh"] = RefreshHeuristicPolicy(actions, rules=RULES, options=OPTIONS)
    for policy in policies.values():
        policy.reset(SAMPLING_SEED)
    decisions, end = [], 0
    for revision, step in enumerate(steps):
        value = inputs(step["cards"], **step.get("input", {}))
        value.observation.update(state_revision=revision, turn=step.get("turn", 1))
        value.action_options["state_revision"] = revision
        value.action_options["action_options"]["state_revision"] = revision
        delta = event_delta(
            1,
            name,
            bank.boundary.end,
            bank.boundary.prefix_hash,
            step.get("events", ()),
            layout=LAYOUT,
        )
        bank.append(delta)
        value = replace(value, events=bank.view().events())
        frame = DecisionFrame.from_input(value, bank, end, layout=LAYOUT)
        end = frame.boundary.end
        before = search.getstate()
        clock = CaptureClock(clock_step, clock_ceiling)
        generated = PublicCandidateGeneratorV3(actions, rules=RULES, clock=clock).generate_game(
            frame, bank.view(), search, COSTS
        )
        generated.table.validate(frame)
        selections = {}
        for kind, policy in policies.items():
            rng_before = None if kind == "refresh" else policy.rng.getstate()
            selected = policy.select_frame(frame, bank.view(), generated)
            action = selected.validate(generated.table, frame)
            selections[kind] = {
                "selection": asdict(selected),
                "action": action,
                "rng_before": rng_before,
                "rng_after": None if kind == "refresh" else policy.rng.getstate(),
            }
            if kind == "refresh":
                selections[kind]["repeated_selections"] = [
                    asdict(policy.select_frame(frame, bank.view(), generated)) for _ in range(2)
                ]
        decisions.append(
            {
                "delta": delta_dict(delta, layout=LAYOUT),
                "frame": frame_dict(frame, layout=LAYOUT),
                "references": frame.references,
                "search_rng_before": before,
                "search_rng_after": search.getstate(),
                "clock_trace": clock.trace,
                "generated": game_generated_dict_v3(
                    generated, action_contract=actions.action_contract, rules=RULES
                ),
                "ordered_actions": [
                    generated.table.to_action(i, frame) for i in range(len(generated.table.kinds))
                ],
                "policies": selections,
            }
        )
    return {
        "name": name,
        "actor": 1,
        "actions": asdict(actions),
        "rules": asdict(RULES),
        "numeric_costs": asdict(COSTS),
        "refresh_options": asdict(OPTIONS),
        "sampling_seed": SAMPLING_SEED,
        "search_seed": SEARCH_SEED,
        "clock_recipe": {
            "start_seconds": 0.0,
            "step_seconds": clock_step,
            "ceiling_seconds": clock_ceiling,
        },
        "decisions": decisions,
    }


def capture():
    # Own-view public context only. No external server, rejection, Replay or RNG
    # from a Simulator is read. Observed public damage is an approximate proxy.
    miracle = (
        {"type": "action_accepted", "player": 2},
        {"type": "artifact_used", "player": 2, "definition_id": "mudflow"},
        {"type": "miracle_used", "player": 2, "definition_id": "mudflow", "mp_spent": 6},
        {"type": "attack_declared", "attacker": 2, "target": 1, "attack": 12},
    )
    weapon = (
        {"type": "action_accepted", "player": 2},
        {"type": "artifact_used", "player": 2, "definition_id": "bronze-club"},
        {"type": "attack_declared", "attacker": 2, "target": 1, "attack": 12},
    )
    pins = source_pins()  # Refuse accidental recapture after editing original code.
    episodes = [
        capture_episode(
            "defense-miracle-priority",
            [
                {
                    "cards": [("angel-armor", None), ("rainbow-curtain", None)],
                    "input": {"choice": "defense", "mp": 10},
                    "events": miracle,
                }
            ],
        ),
        capture_episode(
            "defense-rainbow-first",
            [
                {
                    "cards": [
                        ("rainbow-curtain", None),
                        ("steel-armor", None),
                        ("bronze-club", None),
                    ],
                    "input": {"choice": "defense"},
                    "events": weapon,
                }
            ],
        ),
        capture_episode(
            "duplicate-instances",
            [
                {
                    "cards": [("abyss-dart", None)] * 3,
                    "input": {"mp": 0},
                }
            ],
        ),
        capture_episode("empty-events", [{"cards": [], "input": {"can_pray": True}}]),
        *[
            capture_episode(
                name,
                [
                    {
                        "cards": [("magical-stick", None)],
                        "input": {"mp": mp, "discard": False},
                    }
                ],
            )
            for name, mp in (("dynamic-weapon-zero-mp", 0), ("dynamic-weapon-positive-mp", 10))
        ],
        capture_episode(
            "large-candidate-table",
            [
                {
                    "cards": [("abyss-dart", None)] * 12,
                    "input": {"mp": 0, "discard": False},
                }
            ],
            settings={"count_states": 1, "max_candidates": 2048, "max_expansions": 20000},
        ),
        capture_episode(
            "clock-cutoffs",
            [
                {
                    "cards": [("abyss-dart", None)] * 8,
                    "input": {"mp": 0, "discard": False},
                }
            ],
            settings={"count_milliseconds": 1, "search_milliseconds": 10},
            clock_step=0.00001,
            clock_ceiling=0.009,
        ),
        capture_episode(
            "refresh-history",
            [
                {"cards": [("direct-smash-axe", None)], "input": {"can_pray": True}},
                {
                    "cards": [("steel-armor", None)],
                    "input": {"choice": "defense"},
                    "events": [{"type": "damage_applied", "player": 2, "amount": 5}],
                },
                {
                    "cards": [("steel-armor", None)],
                    "input": {"choice": "defense"},
                    "events": [
                        {"type": "damage_applied", "player": 2, "amount": 7},
                        {"type": "turn_started", "turn": 2, "active_player": 2},
                    ],
                },
                {"cards": [("bronze-club", None)], "input": {"can_pray": True}, "turn": 3},
            ],
        ),
    ]
    return {
        "schema": "rust-player-python-v5-originals-v1",
        "source_revision": REVISION,
        "source_sha256": pins,
        "capture_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "simulator_revision": shared_card_instance_contract().simulator_revision,
        "spec07_sha256": SPEC07,
        "python": platform.python_version(),
        "random_state_version": random.Random().getstate()[0],
        "scope": "synthetic own-view Python-v5 comparison originals; no gameplay/learning qualification",
        "report_milliseconds_normalization": "none; exact clock transcript",
        "rng_scope": "CPython participant search and Random baseline only; Refresh has no random stream; NumPy control RNG is not input",
        "episodes": episodes,
        "episodes_sha256": digest(episodes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace", action="store_true", help="explicitly replace this new fixture only"
    )
    args = parser.parse_args()
    if OUTPUT.exists() and not args.replace:
        parser.error("capture already exists; use --replace for explicit regeneration")
    value = capture()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(canonical(value) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "episodes": len(value["episodes"]),
                "decisions": sum(len(row["decisions"]) for row in value["episodes"]),
                "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
