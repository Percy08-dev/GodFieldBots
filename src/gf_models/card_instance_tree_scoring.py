"""Private scores-only Tree3 protocol kernels; no selection or Source permission."""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

from gf_common.card_instance import ActionConfigV3, action_config_v3
from gf_common.card_instance_fixture import CONTRACT
from gf_common.card_instance_fixture import origin as fixture_origin
from gf_common.card_instance_rules import card_instance_rules, parse_card_instance_rules
from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.structured import CandidateTable, DecisionFrame, EventBank, EventBankView, EventDelta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.structured_runtime import WorkerCaps
from gf_models.structured_sealed import checksum
from gf_models.structured_tree_artifacts import _bind_model_v3, _load_scoring_tree_v3
from gf_models.structured_tree_config import (
    ConfiguredTreeModelV3,
    TreeDescriptor,
    architecture_identity_v3,
    configured_tree_v3,
    numeric_costs_v3,
    resource_estimate,
)
from gf_models.structured_tree_encoder import _owned, feature_layout_v3
from gf_models.structured_tree_pairing import TREE3_RECORDS
from gf_models.structured_tree_policy import _prepare_policy, _score_chunks

COMPONENT = 'component-originals'
DIAGNOSTIC = 'actual-own-input-diagnostic'
INIT_FIELDS = {'protocol_version', 'kind', 'binding', 'model', 'inference_metadata', 'model_text',
               'actions', 'rules', 'caps', 'numeric_costs', 'sampling_seed', 'actor', 'episode'}
BINDING_FIELDS = {'schema_version', 'kind', 'input_scope', 'actor', 'episode', 'inference_sha256',
    'model_sha256', 'model', 'architecture_identity', 'feature_signature', 'token_signature',
    'layout_signature', 'rule_signature', 'artifact_action_config_hash', 'caller_action_config_hash',
    'inference', 'resources', 'tree_descriptor', 'numeric_costs_hash', 'caps_hash', 'source_kind', 'source_origin'}


def search_state(seed: int) -> Any:
    integer(seed)
    return random.Random(int(digest({'recipe': 'structured-search-stream-v1', 'sampling_seed': seed})[:16], 16)).getstate()


def scoring_settings(model: ConfiguredTreeModelV3, actions: ActionConfigV3, scope: str,
                     artifact_action_hash: str, descriptor: TreeDescriptor) -> tuple[WorkerCaps, Any]:
    if type(model) is not ConfiguredTreeModelV3 or type(actions) is not ActionConfigV3:
        raise ContractError('closed corrected Tree scoring configuration required')
    if (scope not in (COMPONENT, DIAGNOSTIC) or actions.action_contract != CONTRACT
            or model.encoder.max_positions != 18 or model.inference.candidate_chunk != 256
            or actions.max_sequence_length != 18 or actions.max_candidates > model.inference.max_candidates):
        raise ContractError('Tree scoring scope/real18/chunk256/caller capacity mismatch')
    checksum(artifact_action_hash)
    if scope == COMPONENT and digest(asdict(actions)) != artifact_action_hash:
        raise ContractError('fixture caller action configuration differs from artifact')
    if scope == DIAGNOSTIC and actions != ActionConfigV3(action_contract=CONTRACT):
        raise ContractError('diagnostic generation requires unchanged full action budgets')
    caps = WorkerCaps(max_positions=18, max_candidates=actions.max_candidates)
    return caps, numeric_costs_v3(model, descriptor)


def binding_for(model: ConfiguredTreeModelV3, actions: ActionConfigV3, scope: str, artifact: Json,
                inference_sha256: str, actor: int, episode: str) -> Json:
    integer(actor, 1, 2)
    if type(episode) is not str or not episode or len(episode.encode()) > 4096:
        raise ContractError('bounded scoring episode identity required')
    descriptor = TreeDescriptor(**closed(artifact['tree_descriptor'], set(TreeDescriptor.__dataclass_fields__)))
    caps, costs = scoring_settings(model, actions, scope, artifact['action_config_hash'], descriptor)
    layout = feature_layout_v3(model.encoder)
    return {'schema_version': 1, 'kind': 'tree-scoring-binding', 'input_scope': scope, 'actor': actor, 'episode': episode,
        'inference_sha256': checksum(inference_sha256), 'model_sha256': descriptor.model_sha256,
        'model': asdict(model.architecture()), 'architecture_identity': architecture_identity_v3(model.architecture()),
        **{key: layout[key] for key in ('feature_signature', 'token_signature', 'layout_signature', 'rule_signature')},
        'artifact_action_config_hash': artifact['action_config_hash'], 'caller_action_config_hash': digest(asdict(actions)),
        'inference': asdict(model.inference), 'resources': asdict(model.resources), 'tree_descriptor': asdict(descriptor),
        'numeric_costs_hash': digest(asdict(costs)), 'caps_hash': digest(asdict(caps)),
        'source_kind': 'card-instance-tree-fixture-originals', 'source_origin': fixture_origin()}


def check_binding(value: Any, expected: Json) -> None:
    closed(value, BINDING_FIELDS)
    if type(value['schema_version']) is not int or canonical(value) != canonical(expected):
        raise ContractError('Tree scoring full artifact/model/feature/scope/execution binding mismatch')


def timings(value: Any, *, generated: bool = False, ready: bool = False) -> None:
    fields = {'model_load_ms'} if ready else {'tokenization_ms', 'prediction_ms', 'worker_total_ms'}
    if generated:
        fields |= {'generation_ms', 'conversion_ms'}
    closed(value, fields)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in value.values()):
        raise ContractError('finite nonnegative Tree scoring timings required')


def scores(value: Any, count: int) -> tuple[float, ...]:
    if type(value) is not list or len(value) != count or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ContractError('finite full original-order Tree scores required')
    return tuple(float(v) for v in value)


def stage_bank(previous: EventBankView | None, last: DecisionFrame | None, frame: DecisionFrame,
               delta: EventDelta, binding: Json, caps: WorkerCaps) -> EventBank:
    if frame.layout is not LAYOUT or delta.layout is not LAYOUT:
        raise ContractError('scoring raw corrected layout mismatch')
    if ((frame.actor, frame.episode) != (binding['actor'], binding['episode'])
            or (delta.actor, delta.episode, delta.start) != (frame.actor, frame.episode, frame.interval_start)
            or frame.interval_start != (0 if last is None else last.boundary.end)
            or last is not None and frame.revision <= last.revision):
        raise ContractError('scoring participant/episode/revision/interval mismatch')
    integer(frame.boundary.end, 0, caps.max_episode_events)
    integer(frame.boundary.end-frame.interval_start, 0, caps.max_interval_events)
    staged = EventBank(frame.actor, frame.episode, caps.max_episode_events, layout=LAYOUT)
    if previous is not None:
        # Immutable bytes/prefix tuples remain validated by the original view.
        # Copy only containers; no committed parent state is mutated before reply.
        staged._events = list(previous._events)
        staged._prefixes = list(previous._prefixes)
    staged.append(delta)
    if staged.boundary != frame.boundary:
        raise ContractError('scoring full own-prefix boundary mismatch')
    frame.bank_view(staged.view())
    return staged


@dataclass(frozen=True)
class ScoringResult:
    frame_hash: str
    table_hash: str
    scores: tuple[float, ...]
    timings: bytes
    generated: bytes | None = None


class _ScoringKernel:
    """Private own-model handle: scores only, no paired policy or game factory."""
    def __init__(self, initial: Json, managed_limit: int) -> None:
        closed(initial, INIT_FIELDS)
        if type(initial['protocol_version']) is not int or initial['protocol_version'] != 3 or initial['kind'] != 'tree-scoring-init':
            raise ContractError('closed Tree scoring initialization required')
        model = configured_tree_v3(initial['model'])
        if model.checkpoint is not None or model.resources.max_owned_bytes != managed_limit:
            raise ContractError('Tree scoring managed argv/model resource equality required')
        self.actions = action_config_v3(initial['actions'])
        self.rules = parse_card_instance_rules(initial['rules'])
        if self.rules != card_instance_rules() or type(initial['inference_metadata']) is not str:
            raise ContractError('closed scoring rules/inference bytes required')
        for key in ('inference_metadata', 'model_text'):
            if type(initial[key]) is not str or len(initial[key].encode()) > 64*1024**2:
                raise ContractError('scoring metadata/text exceeds64MiB')
        resource_estimate(model, model_bytes=len(initial['model_text'].encode()), extra_bytes=_owned(initial))
        artifact = json.loads(initial['inference_metadata'])
        self.binding = binding_for(model, self.actions, initial['binding']['input_scope'], artifact,
            initial['binding']['inference_sha256'], initial['actor'], initial['episode'])
        check_binding(initial['binding'], self.binding)
        self.caps, self.costs = scoring_settings(model, self.actions, self.binding['input_scope'],
                                                artifact['action_config_hash'], TreeDescriptor(**artifact['tree_descriptor']))
        if canonical(initial['caps']) != canonical(asdict(self.caps)) or canonical(initial['numeric_costs']) != canonical(asdict(self.costs)):
            raise ContractError('scoring exact caps/numeric-cost declaration mismatch')
        integer(initial['sampling_seed'])
        self.rng = random.Random()
        self.rng.setstate(search_state(initial['sampling_seed']))
        self.tree = _load_scoring_tree_v3(initial['inference_metadata'], initial['model_text'],
            expected_digest=self.binding['inference_sha256'], model=model, action_config_hash=artifact['action_config_hash'],
            retained_bytes=_owned(initial), deadline=time.monotonic()+10)
        self.managed_limit = managed_limit
        self.previous: EventBankView | None = None
        self.last: DecisionFrame | None = None
        self.count = 0

    def retained_bytes(self) -> int:
        return resource_estimate(self.tree.model, model_bytes=self.tree.descriptor.model_bytes,
                                 extra_bytes=_owned((self.binding, self.previous, self.last)))['total_bytes']

    def score(self, frame: DecisionFrame, bank: EventBankView, table: CandidateTable) -> tuple[list[float], Json]:
        import numpy as np
        purpose = 'component' if self.binding['input_scope'] == COMPONENT else 'qualified_game'
        if table.purpose != purpose or table.synthetic_capacity or table.layout is not LAYOUT:
            raise ContractError('scoring input scope/table purpose mismatch')
        table.validate(frame)
        integer(len(table.kinds), 1, self.caps.max_candidates)
        _bind_model_v3(self.tree.booster, self.tree.model, self.tree.origin)
        prepared, candidates, token_ms = _prepare_policy(self.tree, frame, bank, table, TREE3_RECORDS)
        count = len(candidates.kinds)
        resource_estimate(self.tree.model, model_bytes=self.tree.descriptor.model_bytes,
            extra_bytes=prepared.resident_bytes+_owned(candidates)+64*count+65536)
        started = time.perf_counter()
        result = np.concatenate([values for _, values in _score_chunks(self.tree, prepared, candidates)])
        prediction_ms = (time.perf_counter()-started)*1000
        return result.tolist(), {'tokenization_ms': token_ms, 'prediction_ms': prediction_ms}
