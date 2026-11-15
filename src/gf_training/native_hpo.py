"""Closed Python authorization of SAME native HPO original facts, never an issuer."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.hpo import PROOF_KINDS, optimizer_delta
from gf_common.hpo_resident import checked_retirement, is_resident
from gf_common.schedule import serialize_schedule
from gf_models.structured_sealed import SEMANTIC_KEYS_V3, checksum
from gf_training.hpo_collection import selection_for_counts
from gf_training.hpo_lineage import hpo_spec, hpo_state, validate_hpo_chain
from gf_training.hpo_source import _initial_schedule, recorded_hpo_plan
from gf_training.structured_lineage import _ChainScan
from gf_training.structured_rollout import _source_correspondence

_CLASS = {'action_contract', 'purpose', 'synthetic_capacity', 'rule_signature', 'layout_signature'}
_COLLECTION = {'collection_index', 'directory', 'generation', 'manifest_sha256', 'config_sha256', 'linkage',
    'schedule_sha256', 'bindings_sha256', 'original_proof', 'semantics', 'learner_label', 'learner_decisions',
    'episode_counts', 'episode_digests', 'control', 'numeric'}


def _same(actual: Any, expected: Any, message: str) -> None:
    if canonical(actual) != canonical(expected):
        raise ContractError(message)


def authorize_controls(value: Json, native: Any, actions: Any, *, directory: Path,
                       max_owned_bytes: int, deadline: float, artifact_provider: Any = None
                       ) -> tuple[Json, _ChainScan, dict[int, Any], Json]:
    packed = artifact_provider is not None
    closed(value, {'schema_version', 'kind', 'reader_contract', 'mode', 'parent', 'collections',
        'authorization_max_bytes', 'qualified', 'parent_qualification'} | _CLASS
        | ({'storage_binding', 'artifact_registry_sha256'} if packed else set()))
    integer(value['schema_version'], 5, 5)
    if (value['kind'] != 'card-instance-hpo-pending-controls' or value['reader_contract'] != 'participant-structured-v2'
            or value['mode'] not in ('historical_parent', 'live_cohort') or value['qualified'] is not False
            or value['parent_qualification'] != 'not_performed'):
        raise ContractError('closed unqualified pending native HPO facts required')
    parent = closed(value['parent'], {'schema_version', 'kind', 'validation_scope', 'npz_validated',
        'journal6_validated', 'qualified', 'parent_qualification', 'directory', 'state', 'model', 'execution',
        'semantics', 'actions_sha256', 'identity_sha256', 'generations', 'registry', 'control_fingerprint'} | _CLASS
        | ({'storage_binding'} if packed else set()))
    if (type(parent['schema_version']) is not int or parent['schema_version'] != 5
            or parent['kind'] != 'card-instance-hpo-parent-controls'
            or parent['validation_scope'] != 'metadata-maps-filehashes-only'
            or any(parent[key] is not False for key in ('npz_validated', 'journal6_validated', 'qualified'))
            or parent['parent_qualification'] != 'not_performed' or Path(parent['directory']).resolve() != directory.resolve()):
        raise ContractError('pending native HPO facts differ from SAME original parent/scope')
    remaining = max_owned_bytes - int(native.used_bytes())
    if remaining <= 0:
        raise ContractError('HPO joint native/sealed admission exhausted before arrays')
    scan = _ChainScan(directory=directory.resolve(), max_owned_bytes=remaining, deadline=deadline)
    try:
        state, records = validate_hpo_chain(directory, max_owned_bytes=remaining, deadline=deadline, scan=scan)
        _initial_schedule(state, records)
        own_actions, model, _ = hpo_state(state)
        _same(asdict(own_actions), asdict(actions), 'HPO actual original action configuration differs')
        for actual, expected in ((parent['state'], state), (parent['generations'], records),
                (parent['registry'], scan.object(directory / 'artifact-registry.json')),
                (parent['model'], asdict(model)), (parent['execution'], state['execution'])):
            _same(actual, expected, 'native HPO recorded parent differs from independently sealed original chain')
        if parent['actions_sha256'] != digest(asdict(actions)) or parent['identity_sha256'] != digest(state['identity']):
            raise ContractError('native HPO parent action/source identity differs')
        for key in _CLASS:
            _same(value[key], state[key], 'native HPO pending input boundary differs')
            _same(parent[key], state[key], 'native HPO parent input boundary differs')
        spec = hpo_spec(state)
        resident = is_resident(state)
        recipe = spec['training_recipe']
        if packed:
            from gf_common.group_pack_game import checked_storage_binding
            checked_storage_binding(value['storage_binding'])
            _same(value['storage_binding'], state['storage_binding'], 'HPO physical storage binding differs')
            if value['artifact_registry_sha256'] != artifact_provider.registry_sha256:
                raise ContractError('HPO pending facts differ from SAME actual original provider')
        first = scan.object(directory / 'artifacts/000000/inference.json', records[0]['inference_sha256'])
        semantics = {key: first[key] for key in SEMANTIC_KEYS_V3 - {'reward_policy_version'}}
        _same(parent['semantics'], semantics, 'native HPO protected public semantics differ')
        if type(value['collections']) is not list:
            raise ContractError('ordered native HPO collection facts required')
        authorization = []
        final = json.loads(canonical(value))
        final['kind'] = 'card-instance-hpo-source-controls'
        final['pending_sha256'] = digest(value)
        generations = set()
        for index, raw in enumerate(value['collections']):
            closed(raw, _COLLECTION | ({'logical_retirement'} if resident else set()))
            if resident:
                retirement = checked_retirement(raw['logical_retirement'])
                if (retirement['run_id'] != state['run_id'] or retirement['generation'] != raw['generation']
                        or retirement['collection_id'] != raw['linkage']['batch_id']):
                    raise ContractError('native logical retirement has foreign original run/collection/generation')
            integer(raw['collection_index'], index, index)
            generation = integer(raw['generation'], 0, state['updates'])
            if generation in generations:
                raise ContractError('duplicate HPO native collection generation')
            generations.add(generation)
            plan, label, after = recorded_hpo_plan(directory, state, records, generation, deadline)
            if (Path(raw['directory']).absolute() != plan.output or raw['config_sha256'] != state['config_sha256']
                    or raw['learner_label'] != label or raw['schedule_sha256'] != digest(serialize_schedule(plan.groups, 4))
                    or raw['bindings_sha256'] != digest({key: asdict(binding) for key, binding in plan.bindings.items()})):
                raise ContractError('native HPO facts differ from original schedule/registry/model')
            _same(raw['linkage'], plan.linkage, 'native HPO original learner seats/selection context differ')
            _same(raw['semantics'], semantics, 'native HPO collection public semantics differ')
            counts = raw['episode_counts']
            digests = raw['episode_digests']
            if type(counts) is not list or len(counts) != recipe['collection_games'] or type(digests) is not list or len(digests) != len(counts):
                raise ContractError('native HPO facts must preserve every ordered learner episode')
            for count in counts:
                integer(count, 0, model.temporal.max_decisions)
            for hashed in digests:
                checksum(hashed)
            integer(raw['learner_decisions'], sum(counts), sum(counts))
            context = plan.linkage['selection_context']
            selection = selection_for_counts(tuple(counts), **{key: context[key]
                for key in ('learning_seed', 'batch', 'decision_budget', 'used_before')})
            delta = optimizer_delta(selection['used_decisions'], epochs=recipe['epochs'], minibatch_size=recipe['minibatch_size'])
            proof = raw['original_proof']
            if (proof.get('qualified') is not False or proof.get('parent_qualification') != 'not_performed'
                    or proof.get('manifest_sha256') != raw['manifest_sha256']
                    or proof.get('source_set_sha256') != digest(proof.get('files'))):
                raise ContractError('native HPO original facts have invalid qualification/hash context')
            stored = None
            if generation < state['updates']:
                stored = scan.object(directory / 'updates' / f'{generation + 1:06}.json', records[generation + 1]['update_sha256'])
                _same(after, records[generation + 1]['parent_controls']['schedule_opponent_rng'], 'HPO original schedule RNG advance differs')
                _same(selection, stored['source_proof']['selection'], 'HPO saved subset differs from independent original-count RNG replay')
                _same(raw['numeric'], {'directory': str(directory / stored['numeric']), 'sha256': stored['numeric_sha256'],
                    'files': stored['numeric_files']}, 'HPO numeric original inventory differs')
                expected_proof = {key: item for key, item in stored['source_proof'].items()
                    if key not in ('selection', 'expected_optimizer_step_delta', 'logical_retirement')}
                expected_proof.update(schema_version=2 if packed else 1,
                    kind='card-instance-packed-original-source-proof' if packed else 'card-instance-original-source-proof')
                if packed:
                    from gf_common.group_pack_game import same_packed_proof
                    if not same_packed_proof(proof, expected_proof):
                        raise ContractError('HPO native packed facts differ from actual committed originals')
                else:
                    _same(proof, expected_proof, 'HPO native facts differ from actual committed originals')
            elif value['mode'] != 'live_cohort' or raw['numeric'] is not None:
                raise ContractError('uncommitted HPO facts require the actual live collection')
            if resident and stored is not None:
                _same(raw['logical_retirement'],stored['source_proof']['logical_retirement'],'native Proof4 retirement differs from committed originals')
            control = {'run_id': state['run_id'], 'config_sha256': state['config_sha256'],
                'identity': state['identity'], 'execution': state['execution'], 'plan': plan.linkage,
                'files': {f'generations/{generation:06}.json': scan.verified_digest(directory / 'generations' / f'{generation:06}.json'),
                    f'artifacts/{generation:06}/inference.json': records[generation]['inference_sha256']},
                'source_set_sha256': proof['source_set_sha256'], 'recipe': 'card-instance-hpo-source-controls-v1',
                'hpo_spec_sha256': state['hpo_spec_sha256']}
            if packed:
                frozen = proof if stored is None else stored['source_proof']
                control.update(storage=frozen['storage'], custody=frozen['custody'])
            _same(raw['control'], control, 'HPO native immutable control/original fingerprint differs')
            control['selection'] = selection
            correspondence = _source_correspondence(digest(control), semantics, tuple(digests), game=True,
                hpo=True, storage_binding=state.get('storage_binding'))
            authorization.append({'collection_index': index, 'selection': selection, 'training_recipe': recipe,
                'expected_optimizer_step_delta': delta, 'expected_correspondence': correspondence})
            item = final['collections'][index]
            for key in ('original_proof', 'episode_counts', 'episode_digests', 'control', 'logical_retirement'):
                item.pop(key, None)
            item.update(selection=selection, correspondence=correspondence,
                source_proof={**proof, **({'logical_retirement':raw['logical_retirement']} if resident else {}),
                    'schema_version': 4 if resident else 3, 'kind': PROOF_KINDS[int(packed)],
                    'selection': selection, 'expected_optimizer_step_delta': delta})
        expected_generations = set(range(state['updates'])) | ({state['updates']} if value['mode'] == 'live_cohort' else set())
        if generations != expected_generations:
            raise ContractError('native HPO pending facts omit or add original collections')
        generation_files = {f'generations/{index:06}.json': scan.verified_digest(directory / 'generations' / f'{index:06}.json')
            for index in range(len(records))}
        request = {'schema_version': 5, 'pending_sha256': digest(value), 'parent': {
            'run_sha256': scan.verified_digest(directory / 'run.json'), 'config_sha256': state['config_sha256'],
            'resume_identity': state['resume_identity'], 'generation_files_sha256': digest(generation_files),
            'registry_sha256': scan.verified_digest(directory / 'artifact-registry.json')}, 'collections': authorization}
        scan.finalize()
        return request, scan, {0: model}, final
    except BaseException as error:
        try:
            scan.close(recheck=False)
        except BaseException as cleanup:  # noqa: BLE001 - retain actual cleanup owner.
            vars(error)["_hpo_authorization_cleanup_scan"] = scan
            error.add_note("HPO sealed authorization cleanup: " + type(cleanup).__name__)
        raise


def attach_gae(metadata: Json, arrays: dict[str, Any]) -> list[Any]:
    """Full original GAE and selected-only normalization; native never labels."""
    import numpy as np

    from gf_training.hpo_collection import replay_selection
    from gf_training.hpo_config import hpo_training_config
    from gf_training.ppo import gae, normalize_advantages
    recipe = hpo_training_config(metadata['training_recipe'])
    counts = tuple(len(episode['frames']) for episode in metadata['episodes'])
    locations = replay_selection(metadata['selection'], counts)
    if len(counts) != recipe.collection_games or metadata['decision_count'] != len(locations):
        raise ContractError('HPO native label input omits original episodes or selected endpoints')
    computed = []
    for episode, count in zip(metadata['episodes'], counts, strict=True):
        if episode['end_kind'] != 'natural':
            raise ContractError('HPO requires all original episodes to end naturally')
        episode['has_targets'] = bool(count)
        if count:
            reward = 0 if episode['outcome'] == 'draw' else 1 if episode['outcome']['win']['winner'] == episode['actor'] else -1
            advantage, target = gae([selection['value'] for selection in episode['selections']], reward, lam=recipe.gae_lambda)
            computed.append((advantage, target.astype(np.float32)))
        else:
            computed.append((np.empty(0, np.float64), np.empty(0, np.float32)))
    normalized = normalize_advantages([computed[episode][0][position] for episode, position in locations])
    advantages = [np.zeros(count, np.float32) for count in counts]
    for (episode, position), value in zip(locations, normalized, strict=True):
        advantages[episode][position] = value
    owned = [normalized]
    for index, (advantage, (_, target)) in enumerate(zip(advantages, computed, strict=True)):
        if counts[index]:
            advantage.flags.writeable = target.flags.writeable = False
            arrays[f'e.{index}.advantage'], arrays[f'e.{index}.target'] = advantage, target
            owned.extend((advantage, target))
    return owned
