"""Real C1 numeric artifacts plus hand-authored structural Gen2 scalar links.

Collection proof fields here test only S2 structure, not original Source authority.
"""
from dataclasses import asdict

import numpy as np

from gf_cli.config import ExperimentConfig, RunConfigV2
from gf_common.action_contract import engineering_contract
from gf_common.domain import digest
from gf_common.structured import INPUT_RECIPE, ActionConfigV2
from gf_models.arrays import atomic_json, sha256
from gf_models.config import PPOTrainingConfig
from gf_models.structured_binding import ConfiguredNeuralModelV2, input_binding, policy_label
from gf_models.torch_runtime import hardware
from gf_training.lineage import checkpoint_path, run_lock
from gf_training.structured_lineage import commit_component_generation, resume_identity
from gf_training.structured_numeric import Learner
from gf_training.structured_rollout import EpisodeRecords, StructuredRollout, write_rollout
from tests.test_structured_learning import learner_for, records_for


def make_chain(tmp_path, make_input, *, kind='mlp', updates=2):
    directory = tmp_path / 'parent'
    classification = input_binding(engineering_contract(), 'component', False)
    base = learner_for(kind, synthetic_capacity=False)
    actions = ActionConfigV2(action_contract=engineering_contract(), max_sequence_length=2)
    config = asdict(ExperimentConfig())
    config.update(actions=asdict(actions), run=asdict(RunConfigV2(split='train')),
        model=asdict(ConfiguredNeuralModelV2(**{key: getattr(base.model, key) for key in base.model.__dataclass_fields__}, inference=base.binding)),
        training=asdict(PPOTrainingConfig(seeds=(17,), max_updates=max(2, updates), device="cpu")))
    execution = {key: {'binding': asdict(base.binding), 'cuda_visible_devices': ''} for key in ('learner', 'inference')}
    identity = {'code': {'revision': None, 'dirty': False}, 'runtime': {'historical': True}, 'input': INPUT_RECIPE}
    resume = resume_identity(digest(config), identity, execution, classification)
    learner = Learner(base.model, 17, base.binding, digest(asdict(actions)), resume,
                      action_contract=actions.action_contract, purpose='component', synthetic_capacity=False)
    state = {'schema_version': 2, 'algorithm': 'ppo', 'run_id': 'structural-component', 'status': 'running',
        'config_sha256': digest(config), 'effective_config': config, 'config_inputs': [], 'identity': identity,
        'seed': 17, 'device': 'cpu', 'created_unix': 1000000., 'last_unix': 1000001.,
        'wall_deadline_unix': 1007200., 'admission_deadline_unix': 1005400., 'execution_deadline_unix': 1007140.,
        'updates': 0, 'accepted_decisions': 0, 'optimizer_steps': 0, 'generation_sha256': None,
        'collection_seconds': 0., 'update_seconds': 0., 'invocations': [], 'completed_updates': None,
        'model_id': kind, 'model_version': '2', 'input_recipe': INPUT_RECIPE, 'resume_identity': resume,
        'execution': execution, 'hardware': hardware('cpu'), **classification}
    rng = np.random.Generator(np.random.PCG64(17))
    def controls(number):
        return {'schedule_opponent_rng': rng.bit_generator.state, 'snapshot_pool': [n for n in range(number + 1) if n % 10 == 0][-5:], 'collection_position': number}
    with run_lock(directory, create=True):
        for name in ('artifacts', 'generations', 'updates', 'collections', 'numeric'):
            (directory / name).mkdir()
        saved = learner.save(checkpoint_path(directory, 0), shard_bytes=400000)
        record = commit_component_generation(directory, state, generation=0, update=None, controls=controls(0),
            artifact_files=saved['files'], max_owned_bytes=512 * 1024**2, deadline=float('inf'))
        for number in range(1, updates + 1):
            label = policy_label(learner.model, learner.binding, record['inference_sha256'], cuda_visible_devices='')
            records = []
            for index in range(16):
                raw = records_for(make_input, learner, count=1, episode=f'b{number}-episode-{index}')
                selections = tuple(type(item)(item.table_hash, item.index, label, item.logprob, item.value) for item in raw.selections)
                records.append(EpisodeRecords(raw.bank, raw.frames, raw.tables, selections, raw.end_kind, raw.outcome,
                    action_contract=actions.action_contract, purpose='component', synthetic_capacity=False))
            rollout = StructuredRollout(tuple(records), learner.model, learner.action_config_hash,
                action_contract=actions.action_contract, purpose='component', synthetic_capacity=False)
            numeric = directory / 'numeric' / f'batch-{number - 1:06}'
            written = write_rollout(numeric, rollout)
            metrics = learner.update(rollout)
            saved = learner.save(checkpoint_path(directory, number), shard_bytes=400000)
            collection = directory / 'collections' / f'batch-{number - 1:06}'
            collection.mkdir()
            atomic_json(collection / 'manifest.json', {'schema_version': 4, 'journal_schema_version': 5, **classification})
            manifest_sha, numeric_sha = sha256(collection / 'manifest.json'), sha256(numeric / 'rollout.json')
            seats = {f'episode-{index}': 1 for index in range(16)}
            update = {'schema_version': 2, 'status': 'complete', 'run_id': state['run_id'], 'generation': number,
                'parent_sha256': state['generation_sha256'], 'batch_id': collection.name,
                'collection': str(collection.relative_to(directory)), 'collection_manifest_sha256': manifest_sha,
                'numeric': str(numeric.relative_to(directory)), 'numeric_sha256': numeric_sha, 'accepted_decisions': len(rollout),
                'source_proof': {'schema_version': 1, 'kind': 'structured-component-source-proof', 'qualified': False, 'parent_qualification': 'not_performed',
                    **classification, 'games': 16, 'manifest_sha256': manifest_sha, 'source_set_sha256': '1' * 64, 'accepted_decisions': 32},
                'metrics': metrics, 'planned_games': 16, 'actual_games': 16, 'health': {}, 'policy_bindings': {}, 'learner_seats': seats,
                'collection_seconds': 1., 'update_seconds': 1., 'model_id': kind, 'model_version': '2', 'input_recipe': INPUT_RECIPE,
                'numeric_files': {'rollout.json': numeric_sha, **written['data']['files']}, **classification}
            state.update(accepted_decisions=state['accepted_decisions'] + len(rollout), optimizer_steps=learner.steps,
                         collection_seconds=float(number), update_seconds=float(number))
            record = commit_component_generation(directory, state, generation=number, update=update, controls=controls(number),
                artifact_files=saved['files'], max_owned_bytes=512 * 1024**2, deadline=float('inf'))
        state.update(status='complete', completed_updates=updates, elapsed_seconds=1., artifact_bytes=10000000,
                     invocations=[{'child_health': {'exit_code': 0, 'stderr_bytes': 0}}])
        atomic_json(directory / 'run.json', state)
    return directory, state
