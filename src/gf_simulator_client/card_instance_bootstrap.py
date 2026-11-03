"""Clean fixed protocol3 bootstrap: no experiment/management code enters policies."""

from __future__ import annotations

import os
import sys

sys.argv[:] = ["gf-card-instance-policy-worker"]
os.environ.clear()
os.environ.update(
    {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
)

import random
import time
from dataclasses import asdict

from gf_common.card_instance import action_config_v3
from gf_common.card_instance_rules import parse_card_instance_rules, rule_signature
from gf_common.card_instance_runtime import check_game_binding, game_binding
from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV3
from gf_common.domain import ContractError, closed, digest, integer
from gf_common.structured import EventBank
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_common.structured_wire import game_generated_dict_v3, parse_delta, parse_frame
from gf_models.card_instance_baselines import CardInstanceBaseline
from gf_models.structured_runtime import WorkerCaps
from gf_simulator_client.framing import read_frame, write_frame


def main() -> None:
    output, source = sys.stdout.buffer, sys.stdin.buffer
    try:
        initial = read_frame(source)
        closed(
            initial,
            {
                "protocol_version",
                "model_id",
                "sampling_seed",
                "actions",
                "rules",
                "caps",
                "numeric_costs",
                "action_contract",
                "rule_signature",
                "layout_signature",
                "purpose",
                "synthetic_capacity",
            },
        )
        if type(initial["protocol_version"]) is not int or initial["protocol_version"] != 3:
            raise ContractError("explicit protocol3 initialization required")
        actions, rules = (
            action_config_v3(initial["actions"]),
            parse_card_instance_rules(initial["rules"]),
        )
        binding = game_binding(
            actions.action_contract, rule_signature(actions.action_contract, rules)
        )
        check_game_binding(
            {key: initial[key] for key in binding},
            actions.action_contract,
            binding["rule_signature"],
        )
        integer(initial["sampling_seed"])
        caps = WorkerCaps(
            max_positions=actions.max_sequence_length, max_candidates=actions.max_candidates
        )
        costs = NumericCosts()
        if initial["caps"] != asdict(caps) or initial["numeric_costs"] != asdict(costs):
            raise ContractError("protocol3 baseline limits/cost mismatch")
        policy = CardInstanceBaseline(initial["model_id"], actions, rules=rules)
        policy.reset(initial["sampling_seed"])
        search_rng = random.Random(
            int(
                digest(
                    {
                        "recipe": "structured-search-stream-v1",
                        "sampling_seed": initial["sampling_seed"],
                    }
                )[:16],
                16,
            )
        )
        generator = PublicCandidateGeneratorV3(actions, rules=rules)
        write_frame(
            output,
            {
                "protocol_version": 3,
                "kind": "ready",
                "policy_version": policy.version,
                "model_load_ms": 0.0,
                **binding,
            },
        )
        bank = previous = None
        count = 0
        while True:
            message = read_frame(source)
            started = time.perf_counter()
            closed(message, {"protocol_version", "kind", "frame", "delta"} | set(binding))
            if (
                type(message["protocol_version"]) is not int
                or message["protocol_version"] != 3
                or message["kind"] != "select"
            ):
                raise ContractError("protocol3 select message required")
            check_game_binding(
                {key: message[key] for key in binding},
                actions.action_contract,
                binding["rule_signature"],
            )
            frame = parse_frame(message["frame"], layout=CORRECTED_CARD_INSTANCE_LAYOUT)
            delta = parse_delta(message["delta"], layout=CORRECTED_CARD_INSTANCE_LAYOUT)
            integer(frame.boundary.end, 0, caps.max_episode_events)
            integer(frame.boundary.end - frame.interval_start, 0, caps.max_interval_events)
            if caps.max_decisions is not None:
                integer(count + 1, 1, caps.max_decisions)
            if (
                delta.actor != frame.actor
                or delta.episode != frame.episode
                or delta.start != frame.interval_start
                or frame.interval_start != (0 if previous is None else previous.boundary.end)
                or previous is not None
                and frame.revision <= previous.revision
            ):
                raise ContractError("protocol3 child participant continuity mismatch")
            if bank is None:
                bank = EventBank(
                    frame.actor,
                    frame.episode,
                    caps.max_episode_events,
                    layout=CORRECTED_CARD_INSTANCE_LAYOUT,
                )
            bank.append(delta)
            if bank.boundary != frame.boundary:
                raise ContractError("protocol3 child exact prefix boundary mismatch")
            view = frame.bank_view(bank.view())
            generation_started = time.perf_counter()
            generated = generator.generate_game(frame, view, search_rng, costs)
            policy_started = time.perf_counter()
            selected = policy.select_frame(frame, view, generated)
            selected.validate(generated.table, frame)
            finished = time.perf_counter()
            write_frame(
                output,
                {
                    "protocol_version": 3,
                    "kind": "selection",
                    "generated": game_generated_dict_v3(
                        generated, action_contract=actions.action_contract, rules=rules
                    ),
                    "selection": asdict(selected),
                    "timings": {
                        "generation_ms": (policy_started - generation_started) * 1000,
                        "policy_ms": (finished - policy_started) * 1000,
                        "worker_decision_ms": (finished - started) * 1000,
                    },
                    **binding,
                },
            )
            previous, count = frame, count + 1
    except EOFError:
        return
    except BaseException as error:  # noqa: BLE001 - framed owned process boundary.
        try:
            write_frame(output, {"worker_error": type(error).__name__, "message": str(error)})
        except OSError, ValueError:
            pass


if __name__ == "__main__":
    main()
