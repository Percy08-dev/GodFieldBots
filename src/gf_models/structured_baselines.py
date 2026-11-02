"""Participant-local compact-table baselines and model2 inference factory."""
from __future__ import annotations

import os
import random
import re
from dataclasses import asdict
from typing import Any

from gf_common.action_contract import ActionContract, validate_game_references
from gf_common.domain import ContractError, Json, integer
from gf_common.structured import CandidateTable, DecisionFrame, EventBankView, Selection
from gf_models.structured_binding import check_input_binding, input_binding, safe_options_v2


class StructuredBaseline:
    def __init__(self, kind: str, *, action_contract: ActionContract, purpose: str, synthetic_capacity: bool) -> None:
        self.input_binding = input_binding(action_contract, purpose, synthetic_capacity)
        self.action_contract = action_contract
        if synthetic_capacity:
            raise ContractError("baseline game proposals cannot use synthetic capacity")
        if kind not in ("random", "heuristic"):
            raise ContractError("unknown structured baseline")
        self.kind = kind
        self.version = kind + "-structured-v1"
        self.rng = random.Random(0)

    def reset(self, sampling_seed: int) -> None:
        integer(sampling_seed)
        self.rng.seed(sampling_seed)

    def select_frame(self, frame: DecisionFrame, bank_view: EventBankView, table: CandidateTable) -> Selection:
        check_input_binding({"action_contract": asdict(table.action_contract), "purpose": table.purpose,
                             "synthetic_capacity": table.synthetic_capacity}, action_contract=self.action_contract,
                            purpose="component", synthetic_capacity=False)
        validate_game_references(table.slots[start:end] for start, end in zip(table.offsets[:-1], table.offsets[1:], strict=True))
        frame.bank_view(bank_view)
        table.validate(frame)
        if self.kind == "random":
            selected = self.rng.randrange(len(table.kinds))
        else:
            raw = frame.raw
            own = raw["observation"]["own"]
            cards = [raw["catalog"][card["definition_id"]] for card in own["hand"]]
            cards += [raw["catalog"][name] for name in own["learned_miracles"]]
            scores = []
            for index, kind in enumerate(table.kinds):
                refs = table.slots[table.offsets[index]:table.offsets[index + 1]]
                chosen = [cards[slot] for slot in refs]
                if kind == "play_sequence":
                    free = any(card["kind"] == "weapons" and card.get("ability") == "cutCost" for card in chosen)
                    cost = 0 if free else sum(card.get("mp_cost", 0) for card in chosen if card["kind"] == "miracles")
                    score = 0.
                    for card in chosen:
                        if card.get("ability") == "doubleAtk":
                            score *= 2
                        else:
                            attack = 2 * max(0, own["mp"] - cost) if card.get("ability") == "atkBy2xMP" else max(0, card.get("attack", 0))
                            score += attack * card.get("hit_rate", 100) / 100
                        if card.get("attribute") == "darkness":
                            score += 6
                    if table.targets[index] == "self":
                        score = -score
                    score -= .05 * len(refs)
                elif kind in ("respond", "respond_sequence"):
                    score = sum(max(0, card.get("defense", 0)) for card in chosen) - .05 * len(refs)
                elif kind == "pray":
                    score = 2.
                elif kind == "sacrifice":
                    score = 1 + .1 * len(refs)
                elif kind == "discard":
                    score = -1 - .05 * sum(card.get("price", 0) for card in chosen)
                elif kind == "buy_decision":
                    score = float(table.numerics[index][3])
                elif kind == "exchange":
                    score = .2 * (table.numerics[index][0] - own["hp"])
                else:
                    score = 0.
                scores.append(score)
            # Fixed first-in-table tie rule is part of this baseline recipe.
            selected = max(range(len(scores)), key=scores.__getitem__)
        return Selection(table.table_hash, selected, self.version)


def make_structured_policy(model_id: str, options: Json | None = None, action_config_hash: str | None = None, *,
                           action_contract: ActionContract, purpose: str, synthetic_capacity: bool) -> Any:
    input_binding(action_contract, purpose, synthetic_capacity)
    if synthetic_capacity:
        raise ContractError("participant game factory rejects synthetic capacity artifacts")
    if model_id in ("random", "heuristic"):
        if options is not None:
            raise ContractError("structured baseline has no checkpoint binding")
        return StructuredBaseline(model_id, action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
    if model_id == 'lightgbm' and options is not None:
        from pathlib import Path

        from gf_models.structured_tree_config import (
            TreeDescriptor,
            configured_tree_v2,
        )
        from gf_models.structured_tree_encoder import safe_tree_options_v2
        from gf_models.structured_tree_pairing import TrainingBinding
        from gf_models.structured_tree_policy import StructuredTreePolicy
        safe = safe_tree_options_v2(options, action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
        if type(action_config_hash) is not str or re.fullmatch(r'[0-9a-f]{64}', action_config_hash) is None:
            raise ContractError('Tree2 own action configuration hash required')
        tree_model = configured_tree_v2({**safe['model'], 'checkpoint': safe['checkpoint'],
            'inference': safe['inference'], 'resources': safe['resources']})
        policy = StructuredTreePolicy.from_artifact(Path(safe['checkpoint']), expected_digest=safe['expected_manifest_digest'],
            model=tree_model, action_config_hash=action_config_hash, action_contract=action_contract,
            purpose=purpose, synthetic_capacity=synthetic_capacity,
            expected_tree_descriptor=TreeDescriptor(**safe['tree_descriptor']),
            expected_feature_signature=safe['feature_signature'],
            **({} if 'training_binding' not in safe else {'training_directory':safe['training_binding']['directory'],
                'training_binding':TrainingBinding(**safe['training_binding'])}))
        return policy
    if model_id not in ("mlp", "gru", "transformer") or options is None:
        raise ContractError("model2 neural checkpoint binding required; LightGBM2 is not implemented")
    binding = safe_options_v2(options, action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
    if type(action_config_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", action_config_hash) is None:
        raise ContractError("own action configuration hash required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != binding["cuda_visible_devices"]:
        raise ContractError("actual child CUDA visibility does not match owned binding")
    from pathlib import Path

    from gf_models.structured_config import InferenceBindingV2, model_config_v2
    from gf_models.structured_policy import StructuredPolicy
    model = model_config_v2(binding["model"])
    inference = InferenceBindingV2(**binding["inference"])
    if model.model_id != model_id:
        raise ContractError("factory/checkpoint model kind mismatch")
    # Parent qualification owns formal-vs-owned/toy authorization. This factory
    # receives only this policy's closed numerical and checkpoint settings.
    return StructuredPolicy.from_artifact(Path(binding["checkpoint"]), binding["expected_manifest_digest"],
                                          model, inference, action_config_hash=action_config_hash,
                                          action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
