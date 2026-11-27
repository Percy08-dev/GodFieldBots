"""Participant-only protocol3 session shared by fresh and resident transports."""
from __future__ import annotations

import os
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
from gf_common.refresh_heuristic import parse_refresh_options
from gf_common.structured import DecisionFrame, EventBank
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_common.structured_wire import game_generated_dict_v3, parse_delta, parse_frame
from gf_models.card_instance_baselines import CardInstanceBaseline
from gf_models.structured_binding import (
    child_cuda_visibility,
    safe_options_game_v3,
    safe_options_game_v4,
)
from gf_models.structured_runtime import own_game_worker_settings_v3


class CardInstanceEpisode:
    """Original protocol3 initialization/generation/selection with explicit retirement."""
    def __init__(self, initial: dict[str, Any], *, managed: int | None = None, policy_loader: Any = None) -> None:
        _MANAGED = managed
        model_id = initial.get("model_id")
        refresh = model_id == "heuristic_refresh"
        neural = model_id in ("mlp", "gru", "transformer")
        tree = model_id == "lightgbm"
        if tree != (_MANAGED is not None):
            raise ContractError("Tree3 alone requires the sanitized managed argv control")
        if model_id not in ("random", "heuristic", "heuristic_refresh", "mlp", "gru", "transformer", "lightgbm"):
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
            } | ({"policy_options"} if neural or tree or refresh else set()),
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
        options = (asdict(parse_refresh_options(initial["policy_options"])) if refresh else
            parser(initial["policy_options"], actions=actions, rules=rules) if neural or tree else None)
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
        def load_policy() -> Any:
            policy: Any
            if refresh:
                from gf_models.refresh_heuristic import RefreshHeuristicPolicy
                policy = RefreshHeuristicPolicy(actions, rules=rules, options=parse_refresh_options(options if options is not None else {}))
            elif options is None:
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
                from gf_models.structured_policy import (
                    GameStructuredPolicyV3,
                    GameStructuredPolicyV4,
                )
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
            return policy
        policy = load_policy() if policy_loader is None else policy_loader(model_id, options, actions, rules, load_policy)
        model_load_ms = (time.perf_counter() - load_started) * 1000 if neural or tree else 0.0
        policy.reset(sampling_seed)
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
        ready = {
                "protocol_version": 3,
                "kind": "ready",
                "policy_version": policy.version,
                "model_load_ms": model_load_ms,
                **binding,
        }
        self.actions, self.rules, self.binding = actions, rules, binding
        self.caps, self.costs, self.policy, self.tree = caps, costs, policy, tree
        self.generator: Any = generator
        self.search_rng: Any = search_rng
        self.ready = ready
        self.managed = managed
        self.bank: EventBank | None = None
        self.previous: DecisionFrame | None = None
        self.count, self.retired = 0, False

    def select(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.retired:
            raise ContractError("protocol3 episode retired")
        actions, rules, binding = self.actions, self.rules, self.binding
        caps, costs, policy, tree = self.caps, self.costs, self.policy, self.tree
        bank, previous, count = self.bank, self.previous, self.count
        generator, search_rng = self.generator, self.search_rng
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
        frame = parse_frame(message["frame"], layout=CORRECTED_CARD_INSTANCE_LAYOUT,
                            action_contract=actions.action_contract)
        delta = parse_delta(message["delta"], layout=CORRECTED_CARD_INSTANCE_LAYOUT,
                            action_contract=actions.action_contract)
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
                action_contract=actions.action_contract,
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
        reply = {
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
                    **getattr(policy, "last_timing", {}),
                },
                **binding,
        }
        previous, count = frame, count + 1
        self.bank, self.previous, self.count = bank, previous, count
        return reply

    def reset(self) -> None:
        from gf_models.resident_cache import reset_policy_for_residency
        reset_policy_for_residency(self.policy)
        self.bank = self.previous = self.search_rng = self.generator = None
        self.count, self.retired = 0, True

    def private_state_empty(self) -> bool:
        from gf_models.resident_cache import policy_private_state_empty
        return (self.retired and self.bank is self.previous is self.search_rng is self.generator is None
            and self.count == 0 and policy_private_state_empty(self.policy))


def episode_initial(binding: Any, sampling_seed: int, actions: Any, rules: Any) -> dict[str, Any]:
    """The same protocol3 initializer for an explicitly selected inference binding."""
    caps, costs, _, _ = own_game_worker_settings_v3(binding.model_id,actions,binding.policy_options,rules=rules)
    return {'protocol_version':3,'model_id':binding.model_id,'sampling_seed':sampling_seed,
        'actions':asdict(actions),'rules':asdict(rules),'caps':asdict(caps),'numeric_costs':asdict(costs),
        **game_binding(actions.action_contract,rule_signature(actions.action_contract,rules)),
        **({} if binding.policy_options is None else {'policy_options':binding.policy_options})}
