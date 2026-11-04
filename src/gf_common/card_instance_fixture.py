"""Closed declared component originals; no simulated outcomes or Source authority."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from gf_common.card_instance import action_config_v3, corrected_card_instance_contract
from gf_common.card_instance_rules import card_instance_rules, rule_signature
from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer
from gf_common.structured import DecisionFrame, EventBank, event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import signature
from gf_common.structured_wire import parse_frame

LABEL_RECIPE = 'card-instance-declared-terminal-v1'
CONSTRUCTION_RECIPE = 'card-instance-tree-fixture-construction-v1'
POLICY_RECIPE = 'card-instance-declared-component-policy-v1'
OUTCOME_ORIGIN = 'declared-component-fixture'
SEEDS = (17, 29, 43)
CONTRACT = corrected_card_instance_contract()


def origin() -> Json:
    return {'purpose': 'component', 'qualified': False, 'parent_qualification': 'not_performed',
            'synthetic_capacity': False, 'outcome_origin': OUTCOME_ORIGIN}


def input_binding() -> Json:
    return {'action_contract': asdict(LAYOUT.action_contract), 'purpose': 'component',
            'synthetic_capacity': False,
            'rule_signature': rule_signature(CONTRACT, card_instance_rules())}


def name(value: Any) -> str:
    if type(value) is not str or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}', value) is None:
        raise ContractError('closed fixture identifier required')
    return value


def sha256(value: Any) -> str:
    if type(value) is not str or re.fullmatch('[a-f0-9]{64}', value) is None:
        raise ContractError('fixture exact SHA256 required')
    return value


def label(outcome: Any, viewer: int) -> float:
    integer(viewer, 1, 2)
    if type(outcome) is not dict:
        raise ContractError('declared participant outcome required')
    if outcome.get('kind') == 'draw':
        closed(outcome, {'kind'})
        return 0.
    closed(outcome, {'kind', 'winner'})
    if outcome['kind'] != 'win':
        raise ContractError('unknown declared outcome')
    integer(outcome['winner'], 1, 2)
    return 1. if outcome['winner'] == viewer else -1.


def policy_id(fixture_id: str, population_id: str, generation: int, inference_sha256: str) -> str:
    name(fixture_id); name(population_id); integer(generation, 0, 2); sha256(inference_sha256)
    return digest({'recipe': POLICY_RECIPE, 'fixture_id': fixture_id, 'population_id': population_id,
                   'generation': generation, 'inference_sha256': inference_sha256})


def declaration(value: Any) -> Json:
    raw = closed(value, {'schema_version', 'kind', 'fixture_id', 'neural_model', 'actions', 'execution',
        'input_binding', 'layout_signature', 'label_recipe', 'construction_recipe', 'populations', 'cohorts', 'content_sha256'})
    if (type(raw['schema_version']) is not int or raw['schema_version'] != 1
            or raw['kind'] != 'card-instance-tree-fixture-declaration'
            or raw['construction_recipe'] != CONSTRUCTION_RECIPE or raw['label_recipe'] != LABEL_RECIPE
            or raw['layout_signature'] != signature(layout=LAYOUT)
            or canonical(raw['input_binding']) != canonical(input_binding())
            or digest({k: v for k, v in raw.items() if k != 'content_sha256'}) != raw['content_sha256']):
        raise ContractError('fixture declaration recipe/content identity mismatch')
    name(raw['fixture_id'])
    actions = action_config_v3(raw['actions'])
    if actions.action_contract != LAYOUT.action_contract or actions.max_sequence_length != 18:
        raise ContractError('fixture requires corrected real18 action configuration')
    if type(raw['neural_model']) is not dict or raw['neural_model'].get('model_id') != 'mlp':
        raise ContractError('fixture populations must be homogeneous MLP')
    if type(raw['populations']) is not list or len(raw['populations']) != 3:
        raise ContractError('all three fixture populations required')
    populations = []
    for row, seed in zip(raw['populations'], SEEDS, strict=True):
        closed(row, {'population_id', 'construction_seed'})
        if type(row['construction_seed']) is not int or row['construction_seed'] != seed:
            raise ContractError('fixture construction seeds must be ordered17/29/43')
        populations.append(name(row['population_id']))
    if len(set(populations)) != 3 or type(raw['cohorts']) is not list or len(raw['cohorts']) != 9:
        raise ContractError('nine complete fixture cohorts required')
    episodes, cohorts = set(), set()
    for i, cohort in enumerate(raw['cohorts']):
        closed(cohort, {'cohort_id', 'population_id', 'split', 'generation', 'episodes'})
        generation = i % 3
        expected_split, count = ('validation', 64) if generation == 2 else ('train', 16)
        if (cohort['population_id'] != populations[i//3] or type(cohort['generation']) is not int
                or cohort['generation'] != generation or cohort['split'] != expected_split
                or type(cohort['episodes']) is not list or len(cohort['episodes']) != count):
            raise ContractError('fixture cohort population/generation/split/complete membership mismatch')
        cid = name(cohort['cohort_id'])
        if cid in cohorts:
            raise ContractError('duplicate fixture cohort')
        cohorts.add(cid)
        for item in cohort['episodes']:
            closed(item, {'episode', 'viewer', 'roles', 'decision_count', 'outcome'})
            eid = name(item['episode'])
            if eid in episodes:
                raise ContractError('duplicate fixture episode')
            episodes.add(eid)
            integer(item['viewer'], 1, 2); integer(item['decision_count'], 0, 1024)
            label(item['outcome'], item['viewer'])
            roles = item['roles']
            if type(roles) is not list or len(roles) != 2:
                raise ContractError('fixture must retain both original participant roles')
            for seat, role in enumerate(roles, 1):
                closed(role, {'participant', 'role', 'policy_generation'})
                selected = seat == item['viewer']
                integer(role['policy_generation'], 0, 2 if selected else 1)
                if (type(role['participant']) is not int or role['participant'] != seat
                        or role['role'] != ('declared-selected' if selected else 'declared-opponent')
                        or selected and role['policy_generation'] != generation):
                    raise ContractError('fixture selected viewer/seat/policy generation mismatch')
    return clone(raw)


def freeze_declaration(value: Json) -> Json:
    """Pure freeze for publication BEFORE artifact/episode construction."""
    return declaration({**value, 'content_sha256': digest(value)})


def envelope(value: Any, *, frozen: Json, declaration_sha256: str, cohort: Json, spec: Json,
             policies: dict[tuple[str, int], str], check: Callable[[], None] | None = None
             ) -> tuple[Json, tuple[DecisionFrame, ...], EventBank]:
    raw = closed(value, {'schema_version', 'kind', 'fixture_id', 'declaration_sha256', 'population_id',
        'cohort_id', 'episode', 'viewer', 'split', 'generation', 'roles', 'selected_policy_id',
        'outcome', 'outcome_origin', 'events', 'decisions'})
    if (type(raw['schema_version']) is not int or raw['schema_version'] != 1
            or raw['kind'] != 'card-instance-component-original'
            or raw['declaration_sha256'] != sha256(declaration_sha256)
            or raw['fixture_id'] != frozen['fixture_id'] or raw['outcome_origin'] != OUTCOME_ORIGIN):
        raise ContractError('component envelope declaration/origin mismatch')
    for key in ('population_id', 'cohort_id', 'split', 'generation'):
        if canonical(raw[key]) != canonical(cohort[key]):
            raise ContractError('fixture envelope cohort membership mismatch')
    for key in ('episode', 'viewer', 'outcome'):
        if canonical(raw[key]) != canonical(spec[key]):
            raise ContractError('fixture envelope viewer/outcome membership mismatch')
    roles = [{**role, 'policy_id': policies[cohort['population_id'], role['policy_generation']]}
             for role in spec['roles']]
    if (canonical(raw['roles']) != canonical(roles)
            or raw['selected_policy_id'] != policies[cohort['population_id'], cohort['generation']]
            or type(raw['decisions']) is not list or len(raw['decisions']) != spec['decision_count']
            or type(raw['events']) is not list or len(raw['events']) > 262144):
        raise ContractError('fixture original roles/policies/decisions mismatch')
    bank = EventBank(spec['viewer'], spec['episode'], layout=LAYOUT)
    if check is not None:
        check()
    bank.append(event_delta(bank.actor, bank.episode, 0, bank.boundary.prefix_hash, raw['events'], layout=LAYOUT))
    view = bank.view()
    frames = []
    previous_end, previous_revision = 0, -1
    for decision in raw['decisions']:
        if check is not None:
            check()
        closed(decision, {'frame', 'columns', 'selected_index'})
        frame = parse_frame(decision['frame'], layout=LAYOUT)
        if ((frame.actor, frame.episode) != (spec['viewer'], spec['episode'])
                or frame.interval_start != previous_end or frame.revision <= previous_revision):
            raise ContractError('fixture original actor/revision/interval continuity mismatch')
        frame.bank_view(view)
        previous_end, previous_revision = frame.boundary.end, frame.revision
        frames.append(frame)
    if previous_end != bank.boundary.end:
        raise ContractError('fixture contains future/unbound own event prefix')
    return raw, tuple(frames), bank
