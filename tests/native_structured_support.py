"""Explicit opaque-NPZ metadata fixtures; never numerical/source verification."""
from dataclasses import asdict

from gf_cli.config import ExperimentConfig, RunConfigV2
from gf_common.action_contract import engineering_contract
from gf_common.domain import (
    CATALOG_META,
    END_PROJECTION_VERSION,
    POLICY_INPUT_VERSION,
    REVIEWED_SPEC_HASHES,
    SIMULATOR_REVISION,
    digest,
)
from gf_common.structured import GENERATOR_V2, INPUT_RECIPE, ActionConfigV2
from gf_common.structured_tokens import signature
from gf_models.arrays import atomic_json, sha256
from gf_models.config import PPOTrainingConfig
from gf_models.sharded_arrays import plan_bundle
from gf_models.structured_binding import ConfiguredNeuralModelV2, input_binding
from gf_models.structured_config import InferenceBindingV2, architecture_identity, parameter_shapes
from gf_models.structured_sealed import ADAM, NUMERICS
from gf_training.structured_lineage import resume_identity


def metadata_parent(root, model):
    import numpy as np
    for name in ('generations', 'updates', 'artifacts/000000', 'collections', 'numeric'):
        (root / name).mkdir(parents=True, exist_ok=True)
    artifact = root / 'artifacts/000000'
    actions = ActionConfigV2(action_contract=engineering_contract())
    binding = InferenceBindingV2()
    classification = input_binding(actions.action_contract, 'component', False)
    semantic = {'input_recipe': INPUT_RECIPE, 'generator_version': GENERATOR_V2, 'policy_input_version': POLICY_INPUT_VERSION,
        'end_projection_version': END_PROJECTION_VERSION, 'reward_policy_version': 'natural-terminal-v1',
        'reviewed_spec_hashes': REVIEWED_SPEC_HASHES, 'simulator_revision': SIMULATOR_REVISION, 'catalog': CATALOG_META,
        'token_signature': signature(model.event.aggregate_window), 'numeric_transform': 'signed-log1p-fp32-v1',
        'runtime': {'torch': 'recorded-only', 'numpy': 'recorded-only', 'python': 'recorded-only', 'gil_enabled': True},
        'numerics': NUMERICS}
    config = asdict(ExperimentConfig())
    config.update(actions=asdict(actions), run=asdict(RunConfigV2(split='train')),
        model=asdict(ConfiguredNeuralModelV2(**{key: getattr(model, key) for key in model.__dataclass_fields__}, inference=binding)),
        training=asdict(PPOTrainingConfig(seeds=(17,), device='cpu', max_updates=2)))
    execution = {key: {'binding': asdict(binding), 'cuda_visible_devices': ''} for key in ('learner', 'inference')}
    identity = {'recorded': 'metadata-only-component'}
    resume = resume_identity(digest(config), identity, execution, classification)
    def opaque_bundle(prefix, descriptors):
        manifest = plan_bundle(prefix, descriptors)
        for shard in manifest['shards']:
            path = artifact / shard['file']
            path.write_bytes(b'EXPLICIT OPAQUE NPZ; NOT A NUMERICAL ARTIFACT')
            shard['metadata']['sha256'] = sha256(path)
            atomic_json(path.with_suffix('.json'), shard['metadata'])
            manifest['files'][path.name] = sha256(path)
            manifest['files'][path.with_suffix('.json').name] = sha256(path.with_suffix('.json'))
        return manifest
    weights = opaque_bundle('weights', {name: {'dtype': 'float32', 'shape': list(shape),
        'bytes': int(np.prod(shape, dtype=object)) * 4} for name, shape in parameter_shapes(model).items()})
    producer = {'source_digest': 'a' * 64, 'code_revision': None}
    inference = {'schema_version': 3, 'kind': 'structured-inference', 'model': asdict(model),
        'architecture_identity': architecture_identity(model), 'action_config_hash': digest(asdict(actions)),
        'producer': producer, 'weights': weights, **semantic, **classification}
    atomic_json(artifact / 'inference.json', inference)
    state_bundle = opaque_bundle('resume', {'rng.torch_cpu': {'dtype': 'uint8', 'shape': [8], 'bytes': 8}})
    rng = np.random.Generator(np.random.PCG64(17))
    saved = {'schema_version': 3, 'kind': 'structured-resume', 'inference_digest': sha256(artifact / 'inference.json'),
        'model': asdict(model), 'binding': asdict(binding), 'resume_identity': resume, 'producer': producer,
        'optimizer_steps': 0, 'optimizer_recipe': ADAM, 'parameter_states': {name: {'present': False} for name in parameter_shapes(model)},
        'cuda_rng_count': 0, 'shuffle_rng': rng.bit_generator.state, 'state': state_bundle, **semantic, **classification}
    atomic_json(artifact / 'resume.json', saved)
    files = {**weights['files'], **state_bundle['files'], 'inference.json': sha256(artifact / 'inference.json'),
             'resume.json': sha256(artifact / 'resume.json')}
    record = {'schema_version': 2, 'run_id': 'opaque-component', 'generation': 0, 'parent_sha256': None,
        'model_id': model.model_id, 'model_version': '2', 'input_recipe': INPUT_RECIPE,
        'inference_sha256': files['inference.json'], 'resume_sha256': files['resume.json'], 'artifact_files': files,
        'accepted_decisions': 0, 'optimizer_steps': 0, 'update_sha256': None,
        'parent_controls': {'schedule_opponent_rng': rng.bit_generator.state, 'snapshot_pool': [0], 'collection_position': 0}, **classification}
    atomic_json(root / 'generations/000000.json', record)
    generation_sha = sha256(root / 'generations/000000.json')
    state = {'schema_version': 2, 'algorithm': 'ppo', 'run_id': 'opaque-component', 'status': 'running',
        'config_sha256': digest(config), 'effective_config': config, 'config_inputs': [], 'identity': identity,
        'seed': 17, 'device': 'cpu', 'created_unix': 1000000., 'last_unix': 1000001.,
        'wall_deadline_unix': 1007200., 'admission_deadline_unix': 1005400., 'execution_deadline_unix': 1007140.,
        'updates': 0, 'accepted_decisions': 0, 'optimizer_steps': 0, 'generation_sha256': generation_sha,
        'collection_seconds': 0., 'update_seconds': 0., 'invocations': [], 'completed_updates': None,
        'model_id': model.model_id, 'model_version': '2', 'input_recipe': INPUT_RECIPE, 'resume_identity': resume,
        'execution': execution, 'hardware': {'device': 'cpu', 'name': 'cpu', 'cuda_runtime': None, 'device_count': 0,
            'cuda_driver_api_version': None, 'exact_driver_build': 'unverified'}, **classification}
    atomic_json(root / 'run.json', state)
    atomic_json(root / 'latest.json', {'generation': 0, 'sha256': generation_sha})
    atomic_json(root / 'artifact-registry.json', {'schema_version': 2, 'generations': {'0': {'status': 'retained',
        'inference_sha256': files['inference.json'], 'original_files': files, 'deletion_intent': None, 'deletion_receipt': None}}})
    return state, record, inference
