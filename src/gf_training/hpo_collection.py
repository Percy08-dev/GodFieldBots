"""Exploratory schedules and independently seeded endpoint selection."""
from __future__ import annotations

import uuid

import numpy as np

from gf_common.domain import (
    ContractError,
    EpisodeGroup,
    EpisodeSpec,
    Json,
    canonical,
    digest,
    integer,
)
from gf_common.hpo import checked_selection, selected_locations, selection_context, selection_seed

HPO_TRAIN_NAMESPACE = uuid.UUID('2de8b1c1-6db1-5cc9-a2e7-933ce3d10003')


def training_groups_hpo(seed: int, batch: int, learner: str, snapshots: tuple[str, ...],
                        rng: object, *, collection_games: int) -> tuple[EpisodeGroup, ...]:
    integer(seed, 0, 2**63 - 1)
    integer(batch, 0, 2**63 - 1)
    integer(collection_games, 8, 65536)
    if (collection_games % 8 or not isinstance(rng, np.random.Generator)
            or type(snapshots) is not tuple or not snapshots or any(type(label) is not str or not label for label in snapshots)
            or type(learner) is not str or not learner):
        raise ContractError('explicit HPO paired schedule required')
    opponents = (['random-card-instance-v2'] * (collection_games // 8)
        + ['heuristic-card-instance-v2'] * (collection_games // 8)
        + [snapshots[int(rng.integers(len(snapshots)))] for _ in range(collection_games // 4)])
    groups = []
    for pair, opponent in enumerate(opponents):
        game_seed = str(uuid.uuid5(HPO_TRAIN_NAMESPACE, f'hpo-ppo-train-v1/{seed}/{batch}/{pair}'))
        name = f'b{batch:06}-p{pair}'
        streams = tuple(int(digest({'recipe': 'hpo-ppo-policy-stream-v1', 'seed': seed,
            'batch': batch, 'pair': pair, 'role': role})[:16], 16) for role in ('learner', 'opponent'))
        paired_streams = (streams[0], streams[1])
        groups.append(EpisodeGroup(game_seed, (
            EpisodeSpec(name + '-seat0', (learner, opponent), paired_streams, name, 1),
            EpisodeSpec(name + '-seat1', (opponent, learner), paired_streams[::-1], name, 2))))
    return tuple(groups)


def selection_for_counts(episode_counts: tuple[int, ...], *, learning_seed: int, batch: int,
                         decision_budget: int, used_before: int) -> Json:
    if type(episode_counts) is not tuple:
        raise ContractError('complete ordered original episode counts required')
    for count in episode_counts:
        integer(count, 0, 1024)
    context = selection_context(decision_budget=decision_budget, used_before=used_before,
        learning_seed=learning_seed, batch=batch)
    seed = selection_seed(context)
    collected = sum(episode_counts)
    used = min(context['remaining'], collected)
    indices = (list(range(collected)) if used == collected else [] if not used else
        sorted(np.random.Generator(np.random.PCG64(seed)).choice(collected, size=used, replace=False).tolist()))
    return checked_selection({'context': context, 'derived_seed': seed, 'episode_counts': list(episode_counts),
        'collected_decisions': collected, 'used_decisions': used, 'indices': indices, 'indices_sha256': digest(indices)})


def replay_selection(value: Json, episode_counts: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    raw = checked_selection(value)
    context = raw['context']
    expected = selection_for_counts(episode_counts, **{key: context[key]
        for key in ('learning_seed', 'batch', 'decision_budget', 'used_before')})
    if canonical(raw) != canonical(expected):
        raise ContractError('HPO selection differs from independently replayed ORIGINAL decisions')
    return selected_locations(expected)


def check_hpo_collection(manifest: Json) -> None:
    from gf_common.card_instance_collection import checked_linkage, hpo_batch_role
    from gf_training.hpo_config import hpo_training_config
    config = manifest.get('config', {})
    spec = config.get('hpo_spec', {})
    from gf_common.hpo_resident import checked_retirement
    owned=type(spec.get('schema_version')) is int and spec['schema_version'] in (4,5)
    resident = type(spec.get('schema_version')) is int and spec['schema_version'] in (3,4,5)
    if resident:
        from gf_common.shared_inference import runtime_for_spec
        runtime_for_spec(spec)
    if owned and (spec.get('origin')!='producer-owned-v1' or spec.get('encoder')!='canonical-python'
            or manifest.get('origin')!='producer-owned-v1' or manifest.get('encoder')!='canonical-python'
            or manifest.get('actor_protocol_version')!=(10 if spec['schema_version']==5 else 9) or manifest.get('replay_reconciled') is not False):
        raise ContractError('Collection8 requires explicit truthful producer/encoder/actor provenance')
    recipe = hpo_training_config(spec.get('training_recipe'))
    if (type(manifest.get('schema_version')) is not int or manifest['schema_version'] != (9 if spec['schema_version']==5 else 8 if owned else 7 if resident else 6)
            or manifest.get('model_version') != '4' or manifest.get('collection_role') != 'hpo-ppo-batch'
            or manifest.get('config_hash') != digest(config)):
        raise ContractError('explicit HPO Collection6 required')
    link = checked_linkage(manifest.get('linkage'), role=hpo_batch_role(recipe.collection_games),
        schedule=manifest.get('schedule', []), registry=manifest.get('policy_bindings'))
    if resident:
        max_workers = integer(spec.get('operating', {}).get('max_participant_workers'), 1, 2**63-1)
        if integer(config.get('run', {}).get('max_participant_workers'), 1, 2**63-1) != max_workers:
            raise ContractError('resident collection worker capacity differs from HPO operating projection')
        from gf_common.shared_inference import checked_cpu_retirement
        retirement = (checked_cpu_retirement if spec['schema_version']==5 else checked_retirement)(manifest.get('logical_retirement'), max_workers=max_workers)
        if (retirement['run_id'] != link['parent_run_id'] or retirement['collection_id'] != link['batch_id']
                or retirement['generation'] != link['collection_generation']):
            raise ContractError('Collection7 logical retirement differs from original collection/run/generation')
    elif 'logical_retirement' in manifest:
        raise ContractError('Collection6 forbids resident logical retirement')
    context = link['selection_context']
    if (context['decision_budget'] != spec.get('decision_budget')
            or context['selection_version'] != spec.get('selection_version')
            or list(config.get('training', {}).get('seeds', ())) != [context['learning_seed']]):
        raise ContractError('Collection6 selection context differs from sole run spec')


def hpo_collection_metadata(*, linkage: Json, logical_retirement: Json | None = None, **options: object) -> Json:
    from gf_training.card_instance_collection import collection_metadata
    value = collection_metadata(**options, _hpo=True)  # type: ignore[arg-type]
    value.update(collection_role='hpo-ppo-batch', linkage=linkage)
    if logical_retirement is not None:
        value.update(schema_version=7, logical_retirement=logical_retirement)
    if value['config']['hpo_spec']['schema_version'] in (4,5):
        value.update(schema_version=9 if value['config']['hpo_spec']['schema_version']==5 else 8,origin='producer-owned-v1',encoder='canonical-python',
            actor_protocol_version=10 if value['config']['hpo_spec']['schema_version']==5 else 9,replay_reconciled=False)
    check_hpo_collection(value)
    return value


def training_groups_hpo_refresh(seed: int, batch: int, learner: str, snapshots: tuple[str, ...], rng: object,
                                *, collection_games: int, actions: object,
                                training_opponents: Json) -> tuple[EpisodeGroup, ...]:
    """The concrete HPO mixed16 variant preserves the original HPO streams."""
    from dataclasses import replace

    from gf_common.card_instance import ActionConfigV3
    from gf_common.training_opponents import refresh_opponent
    if type(collection_games) is not int or collection_games != 16:
        raise ContractError('mixed-refresh-16-v1 requires collection_games=16')
    if type(actions) is not ActionConfigV3:
        raise ContractError('mixed opponents require corrected actions3')
    label, _ = refresh_opponent(actions, training_opponents)
    groups = list(training_groups_hpo(seed, batch, learner, snapshots, rng, collection_games=16))
    group = groups[3]
    groups[3] = replace(group, episodes=tuple(replace(spec, policies=(spec.policies[0] if spec.learner_viewer == 1 else label,
        label if spec.learner_viewer == 1 else spec.policies[1])) for spec in group.episodes))
    return tuple(groups)
