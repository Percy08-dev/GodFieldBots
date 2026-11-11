"""Complete-episode numeric HPO component; never a qualified Source capability.

Callers supply sorted selected endpoint locations. Online Source independently
reconstructs these from complete originals; this class does not authorize them.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import CardInstanceRules
from gf_common.domain import ContractError, Json, canonical, digest, integer
from gf_common.hpo import optimizer_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_models.hpo_config import validate_model_v4
from gf_models.structured_binding import input_binding_game_v3
from gf_models.structured_config import ModelConfigV4, aggregate_recipe
from gf_training.hpo_config import HPOTrainingConfig, hpo_training_config
from gf_training.ppo import gae, normalize_advantages
from gf_training.structured_rollout import (
    EpisodeRecords,
    EpisodeTokens,
    StructuredRollout,
    _owned_bytes,
    _predicted_token_bytes,
    _record_binding,
    _RolloutContext,
)


class HpoRollout(StructuredRollout):
    def __init__(self, records: tuple[EpisodeRecords, ...], model: ModelConfigV4,
                 training: HPOTrainingConfig, *, actions: ActionConfigV3, rules: CardInstanceRules,
                 selected_locations: tuple[tuple[int, int], ...], _reserved_bytes: int = 0) -> None:
        self.layout = CORRECTED_CARD_INSTANCE_LAYOUT
        self._context = _RolloutContext(actions, rules)
        validate_model_v4(model, layout=self.layout)
        if type(training) is not HPOTrainingConfig:
            raise ContractError("closed HPO training recipe required")
        training = hpo_training_config(asdict(training))
        if type(actions) is not ActionConfigV3 or actions.max_sequence_length > model.max_positions:
            raise ContractError("numeric5 requires real18 actions")
        self.input_binding = input_binding_game_v3(actions.action_contract, rules)
        self.action_contract, self.purpose, self.synthetic_capacity = actions.action_contract, "qualified_game", False
        self.model, self.training, self.action_config_hash = model, training, digest(asdict(actions))
        if type(records) is not tuple or not records or any(type(record) is not EpisodeRecords for record in records):
            raise ContractError("complete owned episode tuple required")
        integer(_reserved_bytes, 0, model.resources.max_owned_bytes)
        owners = set()
        labels = set()
        all_locations = []
        for episode_index, record in enumerate(records):
            if (_record_binding(record.action_contract, record.purpose, record.synthetic_capacity) != self.input_binding
                    or record.end_kind != "natural"):
                raise ContractError("numeric5 requires complete natural own-player originals")
            owner = record.bank.actor, record.bank.episode
            if owner in owners:
                raise ContractError("duplicate participant episode bank")
            owners.add(owner)
            integer(len(record.frames), 0, model.temporal.max_decisions)
            integer(record.bank.boundary.end, 0, model.event.max_episode)
            for position, (frame, table, selection) in enumerate(zip(record.frames, record.tables, record.selections, strict=True)):
                if frame.layout is not self.layout or record.bank.layout is not self.layout:
                    raise ContractError("numeric5 requires exact corrected layout")
                integer(frame.boundary.end, 0, model.event.max_episode)
                integer(frame.boundary.end - frame.interval_start, 0, model.event.max_interval)
                integer(len(table.kinds), 1, actions.max_candidates)
                if any(stop - start > model.max_positions for start, stop in zip(table.offsets[:-1], table.offsets[1:], strict=True)):
                    raise ContractError("candidate capacity exceeded before tokenization")
                if selection.logprob is None or selection.value is None:
                    raise ContractError("numeric5 requires saved frozen probability/value")
                labels.add(selection.policy_version)
                all_locations.append((episode_index, position))
        if len(labels) != 1 or type(selected_locations) is not tuple or not selected_locations:
            raise ContractError("one frozen policy and nonempty selection required")
        previous = None
        for location in selected_locations:
            if type(location) is not tuple or len(location) != 2:
                raise ContractError("closed endpoint location required")
            episode_index, position = location
            integer(episode_index, 0, len(records) - 1)
            integer(position, 0, len(records[episode_index].frames) - 1)
            if previous is not None and location <= previous:
                raise ContractError("selected endpoint locations must be unique and sorted")
            previous = location
        if _owned_bytes(records) + _predicted_token_bytes(records) + _reserved_bytes > model.resources.max_owned_bytes:
            raise ContractError("numeric5 token ownership reservation exceeded before encoding")
        episodes = []
        for record in records:
            frames = tuple(tokenize_frame(frame, record.bank, aggregate_window=model.event.aggregate_window,
                aggregate_recipe=aggregate_recipe(model), layout=self.layout) for frame in record.frames)
            tables = tuple(tokenize_table(table, frame, model.max_positions,
                aggregate_window=model.event.aggregate_window, aggregate_recipe=aggregate_recipe(model), layout=self.layout)
                for frame, table in zip(record.frames, record.tables, strict=True))
            if frames:
                reward = 0 if record.outcome == "draw" else 1 if record.outcome["win"]["winner"] == record.bank.actor else -1
                advantage, target = gae([selection.value for selection in record.selections], reward, lam=training.gae_lambda)
                target = target.astype(np.float32)
            else:
                advantage, target = np.empty(0, np.float64), np.empty(0, np.float32)
            episodes.append(EpisodeTokens(record, frames, tables, advantage, target))
        normalized = normalize_advantages([episodes[episode].advantage[position] for episode, position in selected_locations])
        advantages = [np.zeros(len(episode.frames), np.float32) for episode in episodes]
        for (episode_index, position), value in zip(selected_locations, normalized, strict=True):
            advantages[episode_index][position] = value
        output = []
        for episode, advantage in zip(episodes, advantages, strict=True):
            advantage.flags.writeable = False
            episode.target.flags.writeable = False
            output.append(EpisodeTokens(episode.records, episode.frames, episode.tables, advantage, episode.target))
        self.episodes = tuple(output)
        self.selected_locations = selected_locations
        self.locations = list(selected_locations)
        self.collected_decisions, self.used_decisions = len(all_locations), len(selected_locations)
        self.policy_version = next(iter(labels))
        self.eligible = True
        self.resident_bytes = _owned_bytes((model, training, self.episodes, self.locations,
            self.selected_locations, self.action_config_hash, self.policy_version, self.input_binding))
        if self.resident_bytes + _reserved_bytes > model.resources.max_owned_bytes:
            raise ContractError("numeric5 owned memory reservation exceeded")

    def require_learning(self) -> None:
        if (not self.eligible or not self.used_decisions or tuple(self.locations) != self.selected_locations
                or self.used_decisions != len(self.locations)
                or self.collected_decisions != sum(len(episode.frames) for episode in self.episodes)
                or any(episode.records.end_kind != "natural" for episode in self.episodes)
                or canonical(asdict(self.training)) != canonical(asdict(hpo_training_config(asdict(self.training))))):
            raise ContractError("invalid complete HPO numeric component")


def _numeric_context(rollout: HpoRollout, selection: Json) -> Json:
    from gf_training.hpo_collection import replay_selection
    locations = replay_selection(selection, tuple(len(episode.records.frames) for episode in rollout.episodes))
    if locations != rollout.selected_locations:
        raise ContractError('numeric5 selection differs from complete rollout endpoints')
    return {'training_recipe': asdict(rollout.training), 'selection': selection,
        'expected_optimizer_step_delta': optimizer_delta(rollout.used_decisions,
            epochs=rollout.training.epochs, minibatch_size=rollout.training.minibatch_size)}


def write_hpo_rollout(directory: Path, rollout: HpoRollout, *, selection: Json, **options: Any) -> Json:
    from gf_training.structured_rollout import _write_rollout
    if type(rollout) is not HpoRollout:
        raise ContractError('exact HPO numeric component required')
    rollout.require_learning()
    return _write_rollout(directory, rollout, _hpo=_numeric_context(rollout, selection), **options)


def read_hpo_rollout(directory: Path, expected_digest: str, model: ModelConfigV4,
                     training: HPOTrainingConfig, *, actions: ActionConfigV3, rules: CardInstanceRules,
                     expected_selection: Json, expected_records: tuple[EpisodeRecords, ...] | None = None,
                     expected_correspondence: Json | None = None, **options: Any) -> HpoRollout:
    from gf_common.hpo import checked_selection
    from gf_training.structured_rollout import _hpo_semantics, _read_rollout
    validate_model_v4(model, layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    training = hpo_training_config(asdict(training))
    selection = checked_selection(expected_selection)
    context = _RolloutContext(actions, rules)
    hpo = {'training_recipe': asdict(training), 'selection': selection,
        'expected_optimizer_step_delta': optimizer_delta(selection['used_decisions'],
            epochs=training.epochs, minibatch_size=training.minibatch_size)}
    output = _read_rollout(directory, expected_digest, model, action_contract=actions.action_contract,
        purpose='qualified_game', synthetic_capacity=False, expected_semantics=_hpo_semantics(model, context),
        expected_records=expected_records, expected_correspondence=expected_correspondence,
        _context=context, _hpo=hpo, **options)
    assert type(output) is HpoRollout
    return output
