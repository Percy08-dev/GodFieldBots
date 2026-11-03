"""Journal6 corrected baseline wrapper around the shared journal transaction owner."""

from __future__ import annotations

import math
import os
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.card_instance import (
    CATALOG_META,
    GENERATOR_VERSION,
    ActionConfigV3,
    CardInstanceContract,
    action_config_v3,
)
from gf_common.card_instance_rules import CardInstanceRules, card_instance_rules, rule_signature
from gf_common.card_instance_runtime import (
    game_binding,
    require_corrected_contract,
    validate_runtime_receipt,
)
from gf_common.card_instance_validation import observation
from gf_common.diverse_actions import GameGeneratedCandidatesV3, NumericCosts
from gf_common.domain import (
    END_PROJECTION_VERSION,
    ContractError,
    Json,
    clone,
    closed,
    digest,
    integer,
)
from gf_common.structured import EventBank, Selection
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_common.structured_tokens import signature
from gf_common.structured_wire import parse_report_v3
from gf_models.card_instance_artifacts import validate_neural_metadata
from gf_models.structured_runtime import WorkerCaps, own_game_worker_settings_v3
from gf_training.candidate_storage import (
    publish_candidates_v2,
    read_candidates_v2,
    sidecar_inventory_v2,
)
from gf_training.journal import REWARD_VERSION
from gf_training.structured_journal import (
    StructuredJournal,
    StructuredJournalData,
    _read_structured_journal,
)

CARD_INSTANCE_JOURNAL_SCHEMA = 6


class _Context:
    layout = CORRECTED_CARD_INSTANCE_LAYOUT
    observation = staticmethod(observation)
    publish_candidates = staticmethod(publish_candidates_v2)
    read_candidates = staticmethod(read_candidates_v2)
    sidecar_inventory = staticmethod(sidecar_inventory_v2)
    generated = staticmethod(GameGeneratedCandidatesV3)

    def __init__(self, actions: ActionConfigV3, rules: CardInstanceRules) -> None:
        require_corrected_contract(actions.action_contract)
        self.actions, self.rules = actions, rules
        self.neural: Json | None = None
        self.policy = ""
        self.numeric_costs_hash = ""
        self.binding = game_binding(
            actions.action_contract, rule_signature(actions.action_contract, rules)
        )

    def check_binding(
        self, raw: Json, *, action_contract: Any, purpose: str, synthetic_capacity: bool
    ) -> None:
        expected = {
            key: self.binding[key] for key in ("action_contract", "purpose", "synthetic_capacity")
        }
        if (
            digest(raw) != digest(expected)
            or action_contract != self.actions.action_contract
            or purpose != "qualified_game"
            or type(synthetic_capacity) is not bool
            or synthetic_capacity
        ):
            raise ContractError("journal6 game context mismatch")

    def parse_report(self, raw: Json) -> Any:
        return parse_report_v3(raw, action_contract=self.actions.action_contract, rules=self.rules)

    def validate_generated(self, value: Any, selection: Selection) -> None:
        if (
            type(value) is not GameGeneratedCandidatesV3
            or value.report.rule_signature != self.binding["rule_signature"]
            or value.report.config_hash != digest(asdict(self.actions))
            or value.report.attempts > self.actions.max_expansions
            or value.report.discovered > self.actions.max_temporary_candidates
            or value.report.numeric_costs_hash != self.numeric_costs_hash
            or selection.policy_version != self.policy
            or self.neural is None and (selection.logprob is not None or selection.value is not None)
            or self.neural is not None and any(type(number) not in (int, float)
                or number is None or not math.isfinite(number) for number in (selection.logprob, selection.value))
        ):
            raise ContractError("journal6 exact baseline game report/config/rules required")

    def validate_header(self, header: Json) -> tuple[WorkerCaps, EventBank]:
        neural = header.get("model_version") == "3"
        closed(
            header,
            {
                "kind",
                "schema_version",
                "feature_version",
                "input_recipe",
                "reward_version",
                "policy_input_version",
                "end_projection_version",
                "catalog",
                "episode_id",
                "viewer",
                "policy",
                "run_id",
                "game_id",
                "action_config_hash",
                "generator_version",
                "bank_episode",
                "caps",
                "numeric_costs",
                "model_id",
                "model_version",
                "generation_schema",
                "aggregate_window",
                "token_signature",
                "sampling_seed",
                "use_action_options",
                "action_contract",
                "rule_signature",
                "layout_signature",
                "purpose",
                "synthetic_capacity",
                "protocol_version",
                "actions",
                "runtime_evidence",
            } | ({"neural"} if neural else set()),
        )
        if (
            type(header["schema_version"]) is not int
            or header["schema_version"] != 6
            or type(header["protocol_version"]) is not int
            or header["protocol_version"] != 3
            or header["kind"] != "header"
            or header["feature_version"] != self.layout.input_recipe
            or header["input_recipe"] != self.layout.input_recipe
            or header["policy_input_version"] != self.actions.action_contract.policy_input_version
            or header["reward_version"] != REWARD_VERSION
            or header["end_projection_version"] != END_PROJECTION_VERSION
            or header["catalog"] != CATALOG_META
            or header["generator_version"] != GENERATOR_VERSION
            or not neural and (header["model_id"] not in ("random", "heuristic") or header["model_version"] != "1")
            or header["generation_schema"] is not None
            or header["use_action_options"] is not True
            or not neural and header["policy"] != header["model_id"] + "-card-instance-v2"
            or action_config_v3(header["actions"]) != self.actions
            or header["action_config_hash"] != digest(asdict(self.actions))
        ):
            raise ContractError("journal6 closed semantic/config identity mismatch")
        if digest({key: header[key] for key in self.binding}) != digest(self.binding):
            raise ContractError("journal6 binding mismatch")
        validate_runtime_receipt(header["runtime_evidence"], require_executed=True)
        caps = WorkerCaps(**closed(header["caps"], set(WorkerCaps.__dataclass_fields__)))
        if (
            caps.max_positions != self.actions.max_sequence_length
            or caps.max_candidates != self.actions.max_candidates
        ):
            raise ContractError("journal6 action/capacity mismatch")
        NumericCosts(**closed(header["numeric_costs"], set(NumericCosts.__dataclass_fields__)))
        if neural:
            safe, label = validate_neural_metadata(header["neural"], actions=self.actions, rules=self.rules)
            expected_caps, expected_costs, _, _ = own_game_worker_settings_v3(
                header["model_id"], self.actions, safe, rules=self.rules)
            if (header["policy"] != label or header["caps"] != asdict(expected_caps)
                    or header["numeric_costs"] != asdict(expected_costs)
                    or header["aggregate_window"] != safe["model"]["event"]["aggregate_window"]):
                raise ContractError("journal6 neural execution/label/capacity mismatch")
            self.neural = clone(header["neural"])
        else:
            self.neural = None
        self.policy = header["policy"]
        self.numeric_costs_hash = digest(header["numeric_costs"])
        integer(header["viewer"], 1, 2)
        integer(header["sampling_seed"])
        integer(header["aggregate_window"], 1, min(65536, caps.max_episode_events))
        if header["token_signature"] != signature(header["aggregate_window"], layout=self.layout):
            raise ContractError("journal6 token identity mismatch")
        for key in ("episode_id", "run_id", "game_id", "bank_episode"):
            if type(header[key]) is not str or not header[key]:
                raise ContractError("journal6 identity string required")
        if re.fullmatch(r"[0-9a-f]{64}", header["action_config_hash"]) is None:
            raise ContractError("journal6 config hash required")
        return caps, EventBank(
            header["viewer"], header["bank_episode"], caps.max_episode_events, layout=self.layout
        )


class CardInstanceJournal(StructuredJournal):
    action_contract: CardInstanceContract  # type: ignore[assignment]

    def __init__(
        self,
        directory: Path,
        episode_id: str,
        viewer: int,
        policy: str,
        *,
        run_id: str,
        game_id: str,
        actions: ActionConfigV3,
        bank_episode: str,
        caps: Json,
        numeric_costs: Json,
        model_id: str,
        runtime_evidence: Json,
        sampling_seed: int = 0,
        rules: CardInstanceRules | None = None,
        neural: Json | None = None,
    ) -> None:
        context = _Context(actions, card_instance_rules() if rules is None else rules)
        aggregate_window = 128
        if neural is not None:
            safe, _ = validate_neural_metadata(neural, actions=actions, rules=context.rules)
            aggregate_window = safe["model"]["event"]["aggregate_window"]
        header = {
            **context.binding,
            "kind": "header",
            "schema_version": 6,
            "feature_version": context.layout.input_recipe,
            "input_recipe": context.layout.input_recipe,
            "reward_version": REWARD_VERSION,
            "policy_input_version": actions.action_contract.policy_input_version,
            "end_projection_version": END_PROJECTION_VERSION,
            "catalog": CATALOG_META,
            "episode_id": episode_id,
            "viewer": viewer,
            "policy": policy,
            "run_id": run_id,
            "game_id": game_id,
            "action_config_hash": digest(asdict(actions)),
            "generator_version": GENERATOR_VERSION,
            "bank_episode": bank_episode,
            "caps": caps,
            "numeric_costs": numeric_costs,
            "model_id": model_id,
            "model_version": "3" if neural is not None else "1",
            "generation_schema": None,
            "aggregate_window": aggregate_window,
            "token_signature": signature(aggregate_window, layout=context.layout),
            "sampling_seed": sampling_seed,
            "use_action_options": True,
            "protocol_version": 3,
            "actions": asdict(actions),
            "runtime_evidence": runtime_evidence,
        }
        if neural is not None:
            header["neural"] = clone(neural)
        _, bank = context.validate_header(header)  # before output creation.
        self._context = context
        self.action_contract, self.purpose = actions.action_contract, "qualified_game"
        self.final_path, self.path = (
            directory / f"player-{viewer}.jsonl",
            directory / f"player-{viewer}.partial.jsonl",
        )
        if self.final_path.exists() or self.final_path.is_symlink():
            raise ContractError("journal6 final evidence already exists")
        self.output = self.path.open("x+", encoding="utf-8")
        info = os.fstat(self.output.fileno())
        self._owned_inode = info.st_dev, info.st_ino
        self.viewer, self.count, self.use_action_options = viewer, 0, True
        self.pending = set()
        self.commands = {}
        self.lifecycle_ack = self.last_accepted = self.last_frame = None
        self.bank, self.caps, self.policy, self.header = bank, caps, policy, header
        self.last_storage_ms = 0.0
        self.last_search_state = random.Random(
            int(
                digest({"recipe": "structured-search-stream-v1", "sampling_seed": sampling_seed})[
                    :16
                ],
                16,
            )
        ).getstate()
        self.write(header)


def read_card_instance_journal(
    path: Path,
    *,
    actions: ActionConfigV3,
    rules: CardInstanceRules | None = None,
    deadline: float = float("inf"),
    records: list[Json] | None = None,
    max_jsonl_bytes: int = 64 * 1024**2,
    max_record_bytes: int = 8 * 1024**2,
) -> StructuredJournalData:
    context = _Context(actions, card_instance_rules() if rules is None else rules)
    return _read_structured_journal(
        path,
        deadline=deadline,
        records=records,
        max_jsonl_bytes=max_jsonl_bytes,
        max_record_bytes=max_record_bytes,
        action_contract=actions.action_contract,
        purpose="qualified_game",
        synthetic_capacity=False,
        _context=context,
    )
