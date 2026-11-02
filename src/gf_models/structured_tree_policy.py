"""CPU native Tree2 inference over deterministic admitted participant chunks."""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.action_contract import validate_game_references
from gf_common.domain import ContractError, integer
from gf_common.structured import CandidateTable, DecisionFrame, EventBankView, Selection
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_models.structured_binding import check_input_binding
from gf_models.structured_tree_artifacts import (
    LoadedTree,
    _bind_origin,
    _check,
    _validate_booster,
    load_artifact,
)
from gf_models.structured_tree_config import policy_label, resource_estimate
from gf_models.structured_tree_encoder import PreparedTreeFrame, _owned


class StructuredTreePolicy:
    def __init__(self, tree: LoadedTree, manifest_digest: str, *, deadline:float=float('inf')) -> None:
        _check(deadline)
        if type(tree) is not LoadedTree:
            raise ContractError('actual loaded Tree2 wrapper required')
        _bind_origin(tree.booster, tree.origin)
        pair=None
        if tree.origin.fit_origin=='trained':
            if tree.training_binding is None or tree.origin.provenance is None:raise ContractError('trained Tree2 policy requires complete explicit paired evidence')
            from gf_models.structured_tree_pairing import inspect_training
            pair=inspect_training(Path(tree.training_binding.directory),expected_binding=tree.training_binding,max_owned_bytes=tree.model.resources.max_owned_bytes,deadline=deadline)
            if pair.inference['tree_descriptor']!=asdict(tree.descriptor) or pair.inference['provenance']!=tree.origin.provenance.to_dict():
                raise ContractError('actual trained Tree2 wrapper differs from paired evidence')
        elif tree.training_binding is not None:raise ContractError('toy Tree2 cannot carry trained evidence')
        _, descriptor = _validate_booster(tree.booster, tree.model,deadline=deadline)
        if pair is not None:pair.finalize()
        if descriptor != tree.descriptor:
            raise ContractError('Tree2 runtime descriptor differs from actual native model')
        self.tree, self.model = tree, tree.model
        self.version = policy_label(tree.model, tree.descriptor, manifest_digest, tree.training_binding)
        self.reserved_bytes = resource_estimate(tree.model, model_bytes=descriptor.model_bytes)['total_bytes']
        self.last_timing: dict[str, float] = {}
        _check(deadline)

    @classmethod
    def from_artifact(cls, directory: Path, **options: Any) -> StructuredTreePolicy:
        return cls(load_artifact(directory, **options), options['expected_digest'],deadline=options.get('deadline',float('inf')))

    def reset(self, sampling_seed: int) -> None:
        integer(sampling_seed, 0, 2**64-1)
        self.last_timing = {}

    def select_frame(self, frame: DecisionFrame, bank_view: EventBankView, table: CandidateTable) -> Selection:
        origin = self.tree.origin
        _bind_origin(self.tree.booster, origin)
        check_input_binding({'action_contract': asdict(table.action_contract), 'purpose': table.purpose,
                             'synthetic_capacity': table.synthetic_capacity}, action_contract=origin.action_contract,
                             purpose=origin.purpose, synthetic_capacity=origin.synthetic_capacity)
        if table.synthetic_capacity:
            raise ContractError('Tree2 game factory/policy rejects synthetic-capacity tables')
        validate_game_references(table.slots[a:b] for a, b in zip(table.offsets[:-1], table.offsets[1:], strict=True))
        table.validate(frame)
        integer(len(table.kinds), 1, self.model.inference.max_candidates)
        integer(frame.boundary.end, 0, 262144)
        interval = frame.boundary.end - frame.interval_start
        integer(interval, 0, 4096)
        view = frame.bank_view(bank_view)
        raw_owned = _owned((frame, bank_view, table, self.tree.origin))
        # Bound the existing model-neutral tokenizer's transient JSON owners
        # before parsing. Two frame views may coexist; history and interval
        # decode their bounded, own-prefix event bytes separately.
        event_bytes = sum(len(value) for value in view._events[max(0, frame.boundary.end-self.model.encoder.aggregate_window):])
        event_bytes += sum(len(value) for value in view._events[frame.interval_start:])
        resource_estimate(self.model, model_bytes=self.tree.descriptor.model_bytes,
            extra_bytes=raw_owned + 64 * len(frame.payload) + 32 * event_bytes + 2048 * interval + 65536)
        self.last_timing = {}
        started = time.perf_counter()
        tokens = tokenize_frame(frame, view, aggregate_window=self.model.encoder.aggregate_window)
        candidates = tokenize_table(table, frame, self.model.encoder.max_positions,
                                    aggregate_window=self.model.encoder.aggregate_window)
        resource_estimate(self.model, model_bytes=self.tree.descriptor.model_bytes,
            extra_bytes=_owned((frame, bank_view, table, self.tree.origin, tokens, candidates)))
        prepared = PreparedTreeFrame.create(tokens, self.model, model_bytes=self.tree.descriptor.model_bytes,
                                            extra_owned_bytes=raw_owned)
        token_ms = (time.perf_counter()-started)*1000
        best, selected = -float('inf'), -1
        for start, rows in prepared.iter_chunks(candidates):
            scores = np.asarray(self.tree.booster.predict(rows, num_threads=self.model.inference.num_threads), dtype=np.float64)
            if scores.shape != (len(rows),) or not np.isfinite(scores).all():
                raise ContractError('Tree2 scores must be finite scalar native predictions')
            local = int(np.argmax(scores))
            if float(scores[local]) > best:
                best, selected = float(scores[local]), start+local
        if selected < 0:
            raise ContractError('Tree2 cannot select an empty candidate table')
        # Protocol2 already measures the entire policy call. Keep its closed
        # host timing vocabulary; no unreviewed Tree-only wire field is added.
        self.last_timing = {'tokenization_ms': token_ms}
        return Selection(table.table_hash, selected, self.version)
