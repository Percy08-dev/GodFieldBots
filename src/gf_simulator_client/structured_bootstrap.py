"""Clean protocol2 child. CUDA visibility is chosen before any factory/Torch import."""
from __future__ import annotations

import os
import sys

sys.argv[:] = ["gf-structured-policy-worker"]
os.environ.clear()
os.environ.update({"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_VISIBLE_DEVICES": ""})

import contextlib
import importlib
import random
import time
from dataclasses import asdict

from gf_common.diverse_actions import PublicCandidateGeneratorV2
from gf_common.domain import ContractError, canonical, closed, digest, integer
from gf_common.structured import EventBank, action_config_v2
from gf_common.structured_wire import generated_dict, parse_delta, parse_frame
from gf_models.structured_binding import check_input_binding, input_binding, safe_options_v2
from gf_models.structured_runtime import own_worker_settings
from gf_simulator_client.framing import read_frame, write_frame


def main() -> None:
    output, source = sys.stdout.buffer, sys.stdin.buffer
    try:
        initial = read_frame(source)
        closed(initial, {"protocol_version", "factory", "model_id", "sampling_seed", "actions", "policy_options", "caps", "numeric_costs", "action_contract", "purpose", "synthetic_capacity"})
        if type(initial["protocol_version"]) is not int or initial["protocol_version"] != 2:
            raise ContractError("protocol2 initialization required")
        descriptor = closed(initial["factory"], {"module", "name"})
        integer(initial["sampling_seed"])
        actions = action_config_v2(initial["actions"])
        classification = input_binding(actions.action_contract, "component", False)
        check_input_binding({key: initial[key] for key in classification}, action_contract=actions.action_contract,
                            purpose="component", synthetic_capacity=False)
        from gf_models.structured_tree_encoder import safe_tree_options_v2
        parser = safe_tree_options_v2 if initial['model_id'] == 'lightgbm' else safe_options_v2
        options = parser(initial["policy_options"], action_contract=actions.action_contract, purpose="component", synthetic_capacity=False) if initial["policy_options"] is not None else None
        if options is not None and initial["model_id"] != "lightgbm":
            if options["inference"]["device"] == "cuda:0":
                if options["cuda_visible_devices"] is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = options["cuda_visible_devices"]
            elif options["cuda_visible_devices"] != "":
                raise ContractError("CPU child binding must declare actual empty CUDA visibility")
        caps, costs, _, _ = own_worker_settings(initial["model_id"], actions, options)
        if canonical(asdict(caps)) != canonical(initial["caps"]) or canonical(asdict(costs)) != canonical(initial["numeric_costs"]):
            raise ContractError("untrusted own worker caps/numeric costs")
        load_started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            factory = getattr(importlib.import_module(descriptor["module"]), descriptor["name"])
            policy = factory(initial["model_id"], options, digest(asdict(actions)), action_contract=actions.action_contract,
                             purpose="component", synthetic_capacity=False)
            policy.reset(initial["sampling_seed"])
            generator = PublicCandidateGeneratorV2(actions)
        search_seed = int(digest({"recipe": "structured-search-stream-v1", "sampling_seed": initial["sampling_seed"]})[:16], 16)
        search_rng = random.Random(search_seed)
        write_frame(output, {"protocol_version": 2, "kind": "ready", "policy_version": policy.version,
                             "model_load_ms": (time.perf_counter() - load_started) * 1000, **classification})
        bank = None
        previous = None
        count = 0
        while True:
            payload = read_frame(source)
            started = time.perf_counter()
            closed(payload, {"protocol_version", "kind", "frame", "delta", "action_contract", "purpose", "synthetic_capacity"})
            check_input_binding({key: payload[key] for key in classification}, action_contract=actions.action_contract,
                                purpose="component", synthetic_capacity=False)
            if type(payload["protocol_version"]) is not int or payload["protocol_version"] != 2 or payload["kind"] != "select":
                raise ContractError("protocol2 own-frame request required")
            raw_frame, raw_delta = payload["frame"], payload["delta"]
            integer(raw_frame.get("boundary", {}).get("end"), 0, caps.max_episode_events)
            if not isinstance(raw_delta.get("events"), list) or len(raw_delta["events"]) > caps.max_interval_events:
                raise ContractError("own event interval exceeds configured cap")
            if caps.max_decisions is not None:
                integer(count + 1, 1, caps.max_decisions)
            frame, delta = parse_frame(raw_frame), parse_delta(raw_delta)
            if bank is None:
                if frame.interval_start != 0:
                    raise ContractError("first actor frame must begin at event zero")
                bank = EventBank(frame.actor, frame.episode, caps.max_episode_events)
            if previous is not None and (frame.interval_start != previous.boundary.end or frame.revision <= previous.revision):
                raise ContractError("own actor history interval/revision continuity mismatch")
            if delta.start != frame.interval_start:
                raise ContractError("delta is not this decision interval")
            bank.append(delta)
            if bank.boundary != frame.boundary:
                raise ContractError("delta must end at exact current actor frame boundary")
            view = frame.bank_view(bank.view())
            with contextlib.redirect_stdout(sys.stderr):
                generation_started = time.perf_counter()
                generated = generator.generate(frame, view, search_rng, costs)
                policy_started = time.perf_counter()
                selection = policy.select_frame(frame, view, generated.table)
                policy_ms = (time.perf_counter() - policy_started) * 1000
            selection.validate(generated.table, frame)
            timings = {"generation_ms": (policy_started - generation_started) * 1000, "policy_ms": policy_ms,
                       "worker_decision_ms": (time.perf_counter() - started) * 1000,
                       **getattr(policy, "last_timing", {})}
            write_frame(output, {"protocol_version": 2, "kind": "selection", "generated": generated_dict(generated),
                                 "selection": asdict(selection), "timings": timings, **classification})
            previous, count = frame, count + 1
    except EOFError:
        return
    except BaseException as exc:  # noqa: BLE001 - framed process boundary preserves failure kind
        try:
            write_frame(output, {"worker_error": type(exc).__name__, "message": str(exc)})
        except (OSError, ValueError):
            pass


if __name__ == "__main__":
    main()
