"""Clean fixed protocol3 bootstrap: no experiment/management code enters policies."""

from __future__ import annotations

import os
import re
import sys

_CONTROL = tuple(sys.argv[1:])
_CONTROL_ERROR = None
_MANAGED = None
if _CONTROL and (len(_CONTROL) != 1 or re.fullmatch('[1-9][0-9]{0,18}', _CONTROL[0]) is None):
    _CONTROL_ERROR = 'Tree3 accepts one sanitized decimal managed limit'
elif _CONTROL:
    _MANAGED = int(_CONTROL[0])
    if not 1 <= _MANAGED <= 2**63-1:
        _CONTROL_ERROR = 'Tree3 managed limit outside1..2^63-1'
del _CONTROL
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
from pathlib import Path
from typing import Any

from gf_common.card_instance import action_config_v3
from gf_common.card_instance_rules import parse_card_instance_rules, rule_signature
from gf_common.card_instance_runtime import check_game_binding, game_binding
from gf_common.diverse_actions import PublicCandidateGeneratorV3
from gf_common.domain import ContractError, closed, digest, integer
from gf_common.structured import EventBank
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_common.structured_wire import game_generated_dict_v3, parse_delta, parse_frame
from gf_models.card_instance_baselines import CardInstanceBaseline
from gf_models.structured_binding import (
    child_cuda_visibility,
    safe_options_game_v3,
    safe_options_game_v4,
)
from gf_models.structured_runtime import own_game_worker_settings_v3
from gf_simulator_client.framing import (
    frame_decode_bytes,
    read_frame,
    read_frame_admitted,
    write_frame,
)


def main() -> None:
    output, source = sys.stdout.buffer, sys.stdin.buffer
    try:
        if _CONTROL_ERROR is not None:
            raise ContractError(_CONTROL_ERROR)
        if _MANAGED is not None and _MANAGED < frame_decode_bytes(1):
            raise ContractError("Tree3 initial header/body admission exceeded before read")
        initial = (read_frame(source) if _MANAGED is None else
            read_frame_admitted(source, max_owned_bytes=_MANAGED))
        model_id = initial.get("model_id")
        neural = model_id in ("mlp", "gru", "transformer")
        tree = model_id == "lightgbm"
        if tree != (_MANAGED is not None):
            raise ContractError("Tree3 alone requires the sanitized managed argv control")
        if model_id not in ("random", "heuristic", "mlp", "gru", "transformer", "lightgbm"):
            raise ContractError("unknown fixed protocol3 model ID")
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
            } | ({"policy_options"} if neural or tree else set()),
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
        sampling_seed = initial["sampling_seed"]
        from gf_models.card_instance_game_tree_runtime import safe_tree_options_game_v3
        raw_options = initial.get("policy_options")
        v4 = type(raw_options) is dict and raw_options.get("version") == "4"
        parser = safe_tree_options_game_v3 if tree else safe_options_game_v4 if v4 else safe_options_game_v3
        options = parser(initial["policy_options"], actions=actions, rules=rules) if neural or tree else None
        if tree and options is not None and options["resources"]["max_owned_bytes"] != _MANAGED:
            raise ContractError("Tree3 argv/decoded resources must match before model allocation")
        if neural and options is not None and options["cuda_visible_devices"] != child_cuda_visibility(
                options["inference"]["device"], options["cuda_visible_devices"]):
            raise ContractError("protocol3 inference CUDA visibility must match canonical child execution")
        caps, costs, _, _ = own_game_worker_settings_v3(model_id, actions, options, rules=rules)
        if initial["caps"] != asdict(caps) or initial["numeric_costs"] != asdict(costs):
            raise ContractError("protocol3 baseline limits/cost mismatch")
        policy: Any
        load_started = time.perf_counter()
        if options is None:
            policy = CardInstanceBaseline(model_id, actions, rules=rules)
        elif tree:
            from gf_models.card_instance_game_tree_runtime import _OwnGameTreePolicyV3
            assert _MANAGED is not None
            policy = _OwnGameTreePolicyV3(options, actions=actions, rules=rules,
                managed_limit=_MANAGED, deadline=time.monotonic()+10)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            visibility = options["cuda_visible_devices"]
            if visibility is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = visibility
            from gf_common.structured_layout import layout_for_contract
            from gf_models.hpo_config import model_config_v4
            from gf_models.structured_config import InferenceBindingV2, model_config_v3
            from gf_models.structured_policy import GameStructuredPolicyV3, GameStructuredPolicyV4
            if v4:
                policy = GameStructuredPolicyV4.from_artifact(
                    Path(options["checkpoint"]), options["expected_manifest_digest"],
                    model_config_v4(options["model"], layout=layout_for_contract(actions.action_contract)),
                    InferenceBindingV2(**options["inference"]), options["action_config_hash"],
                    action_contract=actions.action_contract, rules=rules)
            else:
                policy = GameStructuredPolicyV3.from_artifact(
                    Path(options["checkpoint"]), options["expected_manifest_digest"],
                    model_config_v3(options["model"], layout=layout_for_contract(actions.action_contract)),
                    InferenceBindingV2(**options["inference"]), options["action_config_hash"],
                    action_contract=actions.action_contract, rules=rules)
        model_load_ms = (time.perf_counter() - load_started) * 1000 if neural or tree else 0.0
        policy.reset(sampling_seed)
        del initial, options
        search_rng = random.Random(
            int(
                digest(
                    {
                        "recipe": "structured-search-stream-v1",
                        "sampling_seed": sampling_seed,
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
                "model_load_ms": model_load_ms,
                **binding,
            },
        )
        bank = previous = None
        count = 0
        while True:
            retained = policy.retained_bytes((None if bank is None else vars(bank), previous)) if tree else 0
            message = (read_frame_admitted(source, max_owned_bytes=_MANAGED, retained_bytes=retained)
                if _MANAGED is not None else read_frame(source))
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
            if tree:
                policy.retained_bytes((None if bank is None else vars(bank), previous, message))
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
            if tree:
                # Account raw parsed copies and the complete active bank before mutation.
                from gf_models.structured_tree_encoder import _owned
                event_bytes = sum(len(value) for value in delta._events)
                if bank is not None:
                    event_bytes += sum(len(value) for value in bank.view()._events)
                policy.retained_bytes((None if bank is None else vars(bank), previous, message),
                    extra_bytes=4*_owned((frame, delta))+64*len(frame.payload)+32*event_bytes+65536)
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
