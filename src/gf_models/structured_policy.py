"""Participant-only structured inference, bounded cache and CPU sampling."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, canonical, digest, integer
from gf_common.structured import CandidateTable, DecisionFrame, EventBankView, Selection
from gf_common.structured_tokens import tokenize_frame, tokenize_table, validate_candidate_tokens
from gf_models.structured_artifacts import load_inference
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV2,
    architecture_identity,
    resource_estimate,
)


class StructuredPolicy:
    def __init__(self, network: Any, model: ModelConfigV2, binding: InferenceBindingV2,
                 manifest_digest: str) -> None:
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
        import torch
        self.torch, self.network, self.model, self.binding = torch, network, model, binding
        self.reserved_bytes = resource_estimate(model, binding)["total_bytes"]
        self.version = f"{model.model_id}-ppo-v2-{manifest_digest}-{digest(asdict(binding))}"
        self.cache = network.new_cache()
        self.rng = torch.Generator(device="cpu")
        self.reset(0)

    @classmethod
    def from_artifact(cls, directory: Path, expected_digest: str, model: ModelConfigV2,
                      binding: InferenceBindingV2, action_config_hash: str) -> StructuredPolicy:
        network, _ = load_inference(directory, expected_digest=expected_digest, model=model,
                                     binding=binding, action_config_hash=action_config_hash)
        return cls(network, model, binding, expected_digest)

    def reset(self, sampling_seed: int) -> None:
        integer(sampling_seed, 0, 2**63 - 1)
        self.cache.reset()
        self.rng.manual_seed(sampling_seed)

    def select_frame(self, frame: DecisionFrame, bank_view: EventBankView,
                     table: CandidateTable) -> Selection:
        torch = self.torch
        # Reject all known scalar caps before tokenization/array ownership work.
        integer(frame.boundary.end, 0, self.model.event.max_episode)
        integer(frame.boundary.end - frame.interval_start, 0, self.model.event.max_interval)
        integer(len(table.kinds), 1, self.binding.max_candidates)
        integer(self.cache.position + 1, 1, self.model.temporal.max_decisions)
        if any(stop - start > self.model.max_positions for start, stop in zip(table.offsets[:-1], table.offsets[1:], strict=True)):
            raise ContractError("policy candidate position capacity exceeded before tokenization")
        tokens = tokenize_frame(frame, frame.bank_view(bank_view), aggregate_window=self.model.event.aggregate_window)
        candidates = tokenize_table(table, frame, self.model.max_positions,
                                     aggregate_window=self.model.event.aggregate_window)
        validate_candidate_tokens(candidates, tokens, self.model.max_positions)
        resource_estimate(self.model, self.binding, decisions=self.cache.position + 1,
                          interval_events=len(tokens.interval_numerics), candidates=len(candidates.kinds), suffix_decisions=1)
        with torch.no_grad():
            context, cards = self.network.step(tokens, self.cache)
            logs, _ = self.network.distribution(context, cards, candidates)
            value = self.network.value(context)
            if self.binding.device != "cpu":
                torch.cuda.synchronize(self.binding.device)
            if not torch.isfinite(logs).all() or not torch.isfinite(value):
                raise ContractError("nonfinite structured inference output")
            probabilities = logs.exp().cpu()
            if not torch.isfinite(probabilities).all():
                raise ContractError("nonfinite CPU sampling probabilities")
            selected = int(torch.multinomial(probabilities, 1, generator=self.rng).item())
            return Selection(table.table_hash, selected, self.version,
                             float(logs[selected].cpu()), float(value.cpu()))
