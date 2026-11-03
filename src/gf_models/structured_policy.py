"""Participant-only structured inference, bounded cache and CPU sampling."""
from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.action_contract import ActionContract
from gf_common.card_instance import CardInstanceContract
from gf_common.card_instance_rules import CardInstanceRules
from gf_common.diverse_actions import GeneratedCandidatesV3
from gf_common.domain import ContractError, canonical, integer
from gf_common.structured import CandidateTable, DecisionFrame, EventBankView, Selection
from gf_common.structured_layout import LEGACY_LAYOUT, StructuredLayout, layout_for_contract
from gf_common.structured_tokens import tokenize_frame, tokenize_table, validate_candidate_tokens
from gf_models.structured_artifacts import load_inference, load_inference_v3
from gf_models.structured_binding import (
    _bind_network_input,
    _bind_network_input_v3,
    check_input_binding,
    check_input_binding_v3,
    input_binding,
    input_binding_v3,
    policy_label,
    policy_label_v3,
    validate_cuda_visibility,
)
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV2,
    ModelConfigV3,
    architecture_identity,
    resource_estimate,
)


class StructuredPolicy:
    def __init__(self, network: Any, model: ModelConfigV2, binding: InferenceBindingV2,
                 manifest_digest: str, *, action_contract: ActionContract, purpose: str, synthetic_capacity: bool) -> None:
        self.input_binding = input_binding(action_contract, purpose, synthetic_capacity)
        self.action_contract, self.purpose, self.synthetic_capacity = action_contract, purpose, synthetic_capacity
        if (not isinstance(manifest_digest, str) or len(manifest_digest) != 64
                or set(manifest_digest) - set("0123456789abcdef")):
            raise ContractError("full inference manifest identity required")
        if type(binding) is not InferenceBindingV2 or type(network.binding) is not InferenceBindingV2:
            raise ContractError("closed policy/network inference bindings required")
        if architecture_identity(network.config) != architecture_identity(model) or str(network.device) != binding.device:
            raise ContractError("policy network architecture/device mismatch")
        # The network's scoring closures capture their construction settings.
        # A wrapper must not relabel or reserve a smaller/different execution
        # envelope while distribution() continues using the original binding.
        if (canonical(asdict(network.config)) != canonical(asdict(model))
                or canonical(asdict(network.binding)) != canonical(asdict(binding))):
            raise ContractError("policy network active config/runtime binding mismatch")
        _bind_network_input(network, action_contract=action_contract, purpose=purpose,
                            synthetic_capacity=synthetic_capacity, fresh=True)
        import torch
        self.torch, self.network, self.model, self.binding = torch, network, model, binding
        self.reserved_bytes = resource_estimate(model, binding)["total_bytes"]
        self.cuda_visible_devices = validate_cuda_visibility(os.environ.get("CUDA_VISIBLE_DEVICES"))
        self.version = policy_label(model, binding, manifest_digest, cuda_visible_devices=self.cuda_visible_devices)
        self.cache = network.new_cache()
        self.last_timing: dict[str, float] = {}
        self.rng = torch.Generator(device="cpu")
        self.reset(0)

    @classmethod
    def from_artifact(cls, directory: Path, expected_digest: str, model: ModelConfigV2,
                      binding: InferenceBindingV2, action_config_hash: str, *, action_contract: ActionContract,
                      purpose: str, synthetic_capacity: bool) -> StructuredPolicy:
        network, _ = load_inference(directory, expected_digest=expected_digest, model=model,
                                     binding=binding, action_config_hash=action_config_hash, action_contract=action_contract,
                                     purpose=purpose, synthetic_capacity=synthetic_capacity)
        return cls(network, model, binding, expected_digest, action_contract=action_contract,
                   purpose=purpose, synthetic_capacity=synthetic_capacity)

    def reset(self, sampling_seed: int) -> None:
        _reset_policy(self, sampling_seed)

    def select_frame(self, frame: DecisionFrame, bank_view: EventBankView,
                     table: CandidateTable) -> Selection:
        check_input_binding(self.input_binding, action_contract=self.action_contract,
                            purpose=self.purpose, synthetic_capacity=self.synthetic_capacity)
        _bind_network_input(self.network, action_contract=self.action_contract,
                            purpose=self.purpose, synthetic_capacity=self.synthetic_capacity)
        check_input_binding({"action_contract": asdict(table.action_contract), "purpose": table.purpose,
                             "synthetic_capacity": table.synthetic_capacity}, action_contract=self.action_contract,
                            purpose=self.purpose, synthetic_capacity=self.synthetic_capacity)
        return _select_table(self, frame, bank_view, table)


def _select_table(self: Any, frame: DecisionFrame, bank_view: EventBankView,
                  table: CandidateTable, *, layout: StructuredLayout = LEGACY_LAYOUT) -> Selection:
    torch = self.torch
    self.last_timing = {}
    token_started = time.perf_counter()
    # Reject all known scalar caps before tokenization/array ownership work.
    integer(frame.boundary.end, 0, self.model.event.max_episode)
    integer(frame.boundary.end - frame.interval_start, 0, self.model.event.max_interval)
    integer(len(table.kinds), 1, self.binding.max_candidates)
    integer(self.cache.position + 1, 1, self.model.temporal.max_decisions)
    if any(stop - start > self.model.max_positions for start, stop in zip(table.offsets[:-1], table.offsets[1:], strict=True)):
        raise ContractError("policy candidate position capacity exceeded before tokenization")
    tokens = tokenize_frame(frame, frame.bank_view(bank_view), aggregate_window=self.model.event.aggregate_window, layout=layout)
    candidates = tokenize_table(table, frame, self.model.max_positions,
                                 aggregate_window=self.model.event.aggregate_window, layout=layout)
    validate_candidate_tokens(candidates, tokens, self.model.max_positions, layout=layout)
    resource_estimate(self.model, self.binding, decisions=self.cache.position + 1,
                      interval_events=len(tokens.interval_numerics), candidates=len(candidates.kinds), suffix_decisions=1, layout=layout)
    token_ms = (time.perf_counter() - token_started) * 1000
    with torch.no_grad():
        self.network._begin_packing_timing()
        forward_started = time.perf_counter()
        try:
            context, cards = self.network.step(tokens, self.cache)
            logs, _ = self.network.distribution(context, cards, candidates)
            value = self.network.value(context)
        finally:
            packing_ms = self.network._end_packing_timing()
        forward_ms = max(0., (time.perf_counter() - forward_started) * 1000 - packing_ms)
        sync_started = time.perf_counter()
        if self.binding.device != "cpu":
            torch.cuda.synchronize(self.binding.device)
        if not torch.isfinite(logs).all() or not torch.isfinite(value):
            raise ContractError("nonfinite structured inference output")
        sync_ms = (time.perf_counter() - sync_started) * 1000
        return_started = time.perf_counter()
        probabilities = logs.exp().cpu()
        if not torch.isfinite(probabilities).all():
            raise ContractError("nonfinite CPU sampling probabilities")
        return_ms = (time.perf_counter() - return_started) * 1000
        sampling_started = time.perf_counter()
        selected = int(torch.multinomial(probabilities, 1, generator=self.rng).item())
        sampling_ms = (time.perf_counter() - sampling_started) * 1000
        return_started = time.perf_counter()
        result = Selection(table.table_hash, selected, self.version,
                           float(logs[selected].cpu()), float(value.cpu()))
        return_ms += (time.perf_counter() - return_started) * 1000
        self.last_timing = {"tokenization_ms": token_ms, "input_pack_transfer_host_ms": packing_ms,
                            "forward_enqueue_host_ms": forward_ms, "sync_host_ms": sync_ms,
                            "return_ms": return_ms, "sampling_ms": sampling_ms}
        return result


def _reset_policy(self: Any, sampling_seed: int) -> None:
    integer(sampling_seed, 0, 2**64 - 1)
    self.cache.reset()
    self.last_timing = {}
    self.rng.manual_seed(sampling_seed)


class StructuredPolicyV3:
    def reset(self, sampling_seed: int) -> None:
        _reset_policy(self, sampling_seed)
    def __init__(self, network: Any, model: ModelConfigV3, binding: InferenceBindingV2, manifest_digest: str, *,
                 layout: StructuredLayout, action_contract: CardInstanceContract, rules: CardInstanceRules,
                 purpose: str, synthetic_capacity: bool, action_config_hash: str) -> None:
        self.input_binding = input_binding_v3(action_contract, rules, purpose, synthetic_capacity)
        if layout is not layout_for_contract(action_contract) or network.layout is not layout:
            raise ContractError("policy3 exact profile/layout mismatch")
        if (type(binding) is not InferenceBindingV2 or type(network.binding) is not InferenceBindingV2
                or canonical(asdict(network.config)) != canonical(asdict(model))
                or canonical(asdict(network.binding)) != canonical(asdict(binding))
                or architecture_identity(network.config, layout=layout) != architecture_identity(model, layout=layout)
                or str(network.device) != binding.device):
            raise ContractError("policy3 network/model/execution mismatch")
        _bind_network_input_v3(network, action_contract=action_contract, rules=rules, purpose=purpose,
            synthetic_capacity=synthetic_capacity, action_config_hash=action_config_hash, fresh=True)
        import torch
        self.torch, self.network, self.model, self.binding = torch, network, model, binding
        self.layout, self.action_contract, self.rules = layout, action_contract, rules
        self.purpose, self.synthetic_capacity, self.action_config_hash = purpose, synthetic_capacity, action_config_hash
        self.reserved_bytes = resource_estimate(model, binding, layout=layout)["total_bytes"]
        self.cuda_visible_devices = validate_cuda_visibility(os.environ.get("CUDA_VISIBLE_DEVICES"))
        self.version = policy_label_v3(model, binding, manifest_digest, layout=layout, action_contract=action_contract,
            rules=rules, purpose=purpose, synthetic_capacity=synthetic_capacity, action_config_hash=action_config_hash,
            cuda_visible_devices=self.cuda_visible_devices)
        self.cache = network.new_cache()
        self.last_timing: dict[str, float] = {}
        self.rng = torch.Generator(device="cpu")
        self.reset(0)

    @classmethod
    def from_artifact(cls, directory: Path, expected_digest: str, model: ModelConfigV3,
                      binding: InferenceBindingV2, action_config_hash: str, *, layout: StructuredLayout,
                      action_contract: CardInstanceContract, rules: CardInstanceRules,
                      purpose: str, synthetic_capacity: bool) -> StructuredPolicyV3:
        network, _ = load_inference_v3(directory, expected_digest=expected_digest, model=model, binding=binding,
            action_config_hash=action_config_hash, layout=layout, action_contract=action_contract, rules=rules,
            purpose=purpose, synthetic_capacity=synthetic_capacity)
        return cls(network, model, binding, expected_digest, layout=layout, action_contract=action_contract,
            rules=rules, purpose=purpose, synthetic_capacity=synthetic_capacity, action_config_hash=action_config_hash)

    def select_frame(self, frame: DecisionFrame, bank_view: EventBankView,
                     generated: GeneratedCandidatesV3) -> Selection:
        if type(generated) is not GeneratedCandidatesV3 or generated.report.config_hash != self.action_config_hash:
            raise ContractError("policy3 requires generated candidates from its immutable action Config")
        check_input_binding_v3(self.input_binding, action_contract=self.action_contract, rules=self.rules,
            purpose=self.purpose, synthetic_capacity=self.synthetic_capacity)
        _bind_network_input_v3(self.network, action_contract=self.action_contract, rules=self.rules, purpose=self.purpose,
            synthetic_capacity=self.synthetic_capacity, action_config_hash=self.action_config_hash)
        table, report = generated.table, generated.report
        if (self.synthetic_capacity or table.synthetic_capacity or table.purpose != "component"
                or table.action_contract != self.action_contract or report.action_contract != self.action_contract
                or report.rule_signature != self.input_binding["rule_signature"]
                or frame.layout is not self.layout or bank_view.layout is not self.layout):
            raise ContractError("policy3 generated/input/rule/purpose/exact profile mismatch")
        GeneratedCandidatesV3(table, report)
        table.validate(frame)
        return _select_table(self, frame, bank_view, table, layout=self.layout)
