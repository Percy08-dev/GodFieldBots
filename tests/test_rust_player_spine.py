"""Genuine supervised CPU collection components; never final-parent authority."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def prepare_cpu_source(root: Path, supervisor: Path, *, common: bool = False) -> dict:
    from gf_cli.config import load_config
    from gf_common.domain import canonical, digest
    from gf_common.rust_player import runtime_binding
    from gf_models.rust_player import sampling_runtime_identity
    from gf_training.rust_player_supervision import runtime_producer

    declaration = {"mode": "rust-player-runtime-v1", "scheduler": "tokio",
        "game_concurrency": 1, "player_worker_threads": 1, "max_batch_size": 1,
        "formation_wait_ms": 0.0}
    base = Path(__file__).parents[1] / "configs/experiments/mlp-shared-inference-scaling-base.toml"
    text = base.read_text().replace('budget_mib = 4096', 'budget_mib = 1024')
    text = re.sub(r'(?m)^(actions|model|opponent|preprocessing|config) = "([^"]+)"$',
        lambda match: f'{match[1]} = "{(base.parent / match[2]).resolve()}"', text)
    text += f'\ndb_root = "{root / "database"}"\nram_root = "{root / "ram"}"\n'
    input_path = root / 'component-config.toml'
    input_path.write_text(text)
    loaded = load_config(input_path, {
        "run": {"concurrency": 1, "player_runtime": declaration, "output_dir": str(root / "parent")},
        "server": {"source_dir": "/tmp/gf-issue72-shared-inference",
            "binary": "/tmp/gfb-shared-simulator-072dabb/target/release/gf-test-server"},
        "model": {"inference": {"device": "cpu"}}, "training": {"device": "cpu", **({'max_updates': 1} if common else {})},
        "preprocessing": {"memory_budget_mib": 2048}})
    (root / ('parent/config.json' if common else 'config.json')).write_bytes(canonical(loaded.effective).encode())
    companion = SimpleNamespace(manifest={"executable_sha256": hashlib.sha256(supervisor.read_bytes()).hexdigest()})
    producer = runtime_producer(companion, sampling_runtime_sha256=sampling_runtime_identity())
    binding = runtime_binding(declaration, producer)
    (root / "component-runtime.json").write_text(canonical(binding))
    return {"config_sha256": loaded.sha256, "runtime_binding_sha256": digest(binding)}


def prepare_cpu_common(root: Path, supervisor: Path) -> dict:
    facts = prepare_cpu_source(root, supervisor, common=True)
    archive_producer(root, supervisor, Path('/tmp/task5c1b-hpo-runtime-wheels'))
    return facts


def archive_producer(root: Path, supervisor: Path, wheel_root: Path) -> None:
    import importlib.util
    import shutil

    from gf_simulator_client.rust_player_companion import discover_companion
    from gf_training.native_backend import binary_identity
    archive = root / 'producer-artifacts'
    archive.mkdir()
    repo = Path(__file__).parents[1]
    for package in ('gf_common', 'gf_models', 'gf_training', 'gf_cli',
                    'gf_evaluation', 'gf_simulator_client', 'godfieldbots'):
        origin = Path(importlib.util.find_spec(package).origin).parent
        shutil.copytree(origin, archive / 'python' / package, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('src', 'assets'):
        shutil.copytree(repo / 'native' / name, archive / 'native' / name)
    for name in ('Cargo.toml', 'Cargo.lock', 'build.rs', 'gf_native.pyi', 'pyproject.toml'):
        shutil.copyfile(repo / 'native' / name, archive / 'native' / name)
    (archive / 'fixture').mkdir()
    for name in ('test_rust_player_spine.py', 'test_rust_player_supervisor.py', 'test_rust_player_config.py'):
        shutil.copyfile(repo / 'tests' / name, archive / 'fixture' / name)
    identity = binary_identity()
    companion = discover_companion(identity)
    shutil.copyfile(supervisor, archive / 'private-test-supervisor')
    shutil.copyfile(companion.executable, archive / 'installed-companion')
    shutil.copyfile(Path(identity['extension_path']), archive / 'loaded-extension.so')
    wheel = wheel_root / 'godfieldbots_native-0.1.0-cp314-cp314-manylinux_2_34_x86_64.whl'
    shutil.copyfile(wheel, archive / wheel.name)
    hashes = {p.relative_to(archive).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(archive.rglob('*')) if p.is_file()}
    (archive / 'hashes.json').write_text(json.dumps(hashes, sort_keys=True))
    (archive / 'identity.json').write_text(json.dumps({'native': identity,
        'installed_companion': companion.manifest, 'runtime_binding': json.loads((root / 'component-runtime.json').read_bytes())}, sort_keys=True))


def prepare_exec_hpo(supervisor: Path, *, config_only: bool = False):
    """Actual CampaignStore + parent OFDs survive exec, never LOCK_UN finally."""
    import fcntl
    import os
    import tempfile
    import uuid

    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_cli.hpo_storage import CampaignStore
    from gf_common.domain import canonical, digest
    from gf_common.rust_player import BUILD_KEYS, deadline_bits, runtime_binding
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.rust_player import sampling_runtime_identity
    from gf_simulator_client.rust_player_companion import sealed_launch_fd
    from gf_training.hpo_config import effective_settings
    from gf_training.native_backend import binary_identity
    from gf_training.rust_player_supervision import runtime_producer, write_control_record
    from tests.test_rust_player_config import BASE, campaign_raw

    root = Path(tempfile.mkdtemp(prefix='task5c1b-hpo-resume-'))
    start = time.monotonic()
    unix_anchor = time.time()
    anchor_after = time.monotonic()
    base = BASE
    text = re.sub(r'(?m)^(actions|model|opponent|preprocessing|config) = "([^"]+)"$',
        lambda match: f'{match[1]} = "{(base.parent / match[2]).resolve()}"', base.read_text())
    text = text.replace('wall_seconds = 600', 'wall_seconds = 630').replace('finalization_seconds = 60', 'finalization_seconds = 90')
    text = text.replace('source_dir = "../../../GodFieldSimurator"', 'source_dir = "/tmp/gf-issue72-shared-inference"')
    text = text.replace('binary = "../../../GodFieldSimurator/target/release/gf-test-server"', 'binary = "/tmp/gfb-shared-simulator-072dabb/target/release/gf-test-server"')
    if '[storage]' in text:
        text = re.sub(r'budget_mib = \d+', 'budget_mib = 1024', text)
    else:
        text += '\n[storage]\nbudget_mib = 1024\n'
    base_path = root / 'component-base.toml'
    base_path.write_text(text)
    raw = campaign_raw()
    raw.update(base_experiment=str(base_path), decision_budget=64, run_wall_seconds=900, session_seconds=900)
    raw['runtime'].update(game_concurrency=1, player_worker_threads=1, max_batch_size=1)
    raw['operating'].update(game_concurrency=1, max_updates=2, source_budget_bytes=2 * 1024**3,
        collector_budget_bytes=1024**3)
    raw['storage'].update(parent_root=str(root / 'parents'), numeric_root=str(root / 'parents'),
        optuna_path=str(root / 'unused-study.sqlite3'), originals_root=str(root / 'originals'))
    loaded_campaign = campaign_config(raw, origin=root)
    settings = effective_settings(loaded_campaign.config.model, loaded_campaign.config.training, {}, layout=layout)
    run_id = str(uuid.uuid4())
    loaded = materialize_hpo_run(loaded_campaign, settings, learning_seed=17, run_id=run_id)
    if config_only:
        return root, loaded_campaign, loaded
    parent = root / 'parents' / run_id
    parent.mkdir(parents=True)
    config_path = parent / 'config.json'
    config_path.write_bytes(canonical(loaded.effective).encode())
    store = CampaignStore(root / 'component-campaign', mode='fresh', max_owned_bytes=256 * 1024**2)
    campaign_id = str(uuid.uuid4())
    store.put('component-campaign.json', {'schema_version': 1, 'kind': 'rust-player-component-campaign-v1',
        'campaign_id': campaign_id, 'campaign_config_sha256': loaded_campaign.sha256,
        'study_identity': loaded_campaign.config.study_identity, 'base_config_sha256': loaded_campaign.base.sha256,
        'effective_campaign': loaded_campaign.effective, 'run_id': run_id, 'learning_seed': 17,
        'scope': {'games': 16, 'game_concurrency': 1, 'player_worker_threads': 1, 'max_batch_size': 1,
            'decision_budget': 64, 'max_updates': 2, 'fresh_pause': 1, 'resume_additional_updates': 0}})
    assert store._lock is not None
    parent_file = (parent / '.lock').open('w+b')
    fcntl.flock(parent_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.fchmod(parent_file.fileno(), 0o600)
    producer = runtime_producer(SimpleNamespace(manifest={
        'executable_sha256': hashlib.sha256(supervisor.read_bytes()).hexdigest()}), sampling_runtime_sha256=sampling_runtime_identity())
    binding = runtime_binding(loaded.config.hpo_spec['runtime'], producer)
    (root / 'component-runtime.json').write_text(canonical(binding))
    operation_id = 'hpo-fresh-' + uuid.uuid4().hex
    deadlines = {key: deadline_bits(start + seconds) for key, seconds in (
        ('admission_bits', 720), ('execution_bits', 810), ('term_bits', 855), ('kill_bits', 885), ('finish_bits', 900))}
    operation = {'schema_version': 2, 'kind': 'rust-player-operation-v2', 'operation_id': operation_id,
        'run_id': run_id, 'mode': 'hpo', 'operation_type': 'hpo-train', 'config_path': str(config_path),
        'config_sha256': loaded.sha256, 'runtime_binding_sha256': digest(binding), 'deadlines': deadlines,
        'parent_directory': str(parent), 'expected_context': {'campaign_id': campaign_id, 'operation_id': operation_id,
            'kind': 'training', 'run_id': run_id, 'evaluation_id': None},
        'execution_request': {'mode': 'fresh', 'pause_after_update': 1}}
    operation_path = root / 'operation-fresh.json'
    operation_sha = write_control_record(operation_path, operation)
    native = binary_identity()
    locks = []
    for kind, file in (('campaign', store._lock), ('parent', parent_file)):
        info = os.fstat(file.fileno())
        os.set_inheritable(file.fileno(), True)
        locks.append({'lock_kind': kind, 'fd': file.fileno(), 'device': info.st_dev, 'inode': info.st_ino})
    launch = {'schema_version': 1, 'kind': 'rust-player-launch-v1', 'supervision_id': uuid.uuid4().hex,
        'python_executable': str(Path(sys.executable)), 'operation_path': str(operation_path),
        'operation_sha256': operation_sha, 'record_directory': str(root), 'mode': 'hpo',
        'session_deadlines': {'finish_bits': deadlines['finish_bits']},
        'clock_binding': {'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'monotonic_before_bits': deadline_bits(start), 'monotonic_after_bits': deadline_bits(anchor_after),
            'unix_anchor_bits': deadline_bits(unix_anchor)},
        'binding': {'python_abi': native['python_abi'], 'extension_sha256': native['extension_sha256'],
            'build': {key: native['build'][key] for key in BUILD_KEYS}}, 'locks': locks}
    launch_fd = sealed_launch_fd(launch)
    os.set_inheritable(launch_fd, True)
    archive_producer(root, supervisor, Path('/tmp/task5c1b-hpo-runtime-wheels'))
    os.environ['GFB_HPO_COMPONENT_LAUNCH_FD'] = str(launch_fd)
    print('HPO COMPONENT MANAGEMENT READY ' + str(root), flush=True)
    os.execv(supervisor, [str(supervisor), '--ignored', '--exact',
        'tests::actual_hpo_resume_inherited_child', '--nocapture'])


def run_cpu_source(control, ownership) -> None:
    import numpy as np

    from gf_cli.config import load_rust_player_effective
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.domain import RuntimePolicyBinding, digest
    from gf_common.rust_player import deadline_seconds
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.arrays import sha256
    from gf_models.rust_player import RustPlayerInference
    from gf_models.structured_binding import input_binding_game_v3
    from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
    from gf_training.card_instance_source import frozen_linkage
    from gf_training.collection import CollectionPlan, training_groups
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.hpo_owned_lineage import OwnedControlWriter
    from gf_training.ppo_run import source_identity
    from gf_training.rust_player_collection import RustPlayerCollector
    from gf_training.rust_player_learning import LocalGameLearner
    from gf_training.rust_player_lineage import resume_identity
    from gf_training.rust_player_registry import RustRunRegistry
    from gf_training.rust_player_source import RustPlayerSource
    from gf_training.structured_ppo import execution

    root, parent = control.root, ownership.directory
    loaded = load_rust_player_effective(Path(control.operation['config_path']),
        expected_sha256=control.operation['config_sha256'])
    config = loaded.config
    bound = json.loads((root / 'component-runtime.json').read_bytes())
    assert digest(bound) == control.operation['runtime_binding_sha256']
    budget = {'managed_ram_bytes': 24 * 1024**3, 'managed_cuda_bytes': 0,
        'source_workspace_bytes': 2 * 1024**3, 'storage_bytes': 1024**3,
        'game_concurrency': 1, 'player_worker_threads': 1, 'max_batch_size': 1}
    admission = control.admit(budget, bound['producer'])
    deadline = deadline_seconds(control.operation['deadlines']['execution_bits'])
    pool = PackedArtifactPool(max_owned_bytes=2 * 1024**3, deadline=deadline)
    writer = OwnedControlWriter(ownership, budget=pool.budget, limit=1024**3)
    rules = card_instance_rules()
    classification = input_binding_game_v3(config.actions.action_contract, rules)
    identity = source_identity(deadline)
    identity['player_runtime'] = bound
    runtime = CardInstanceRuntimeEvidence(Path(config.server.binary), Path(config.server.source_dir),
        contract=config.actions.action_contract, deadline=deadline)
    receipt = runtime.preflight()
    identity['integration'] = {'binary_sha256': receipt['binary']['sha256']}
    state = {'schema_version': 5, 'run_id': control.operation['run_id'], 'seed': 17,
        'updates': 0, 'generation_sha256': None, 'optimizer_steps': 0, 'accepted_decisions': 0,
        'effective_config': loaded.effective, 'config_sha256': loaded.sha256,
        'config_inputs': list(loaded.inputs), 'identity': identity,
        'runtime_binding_sha256': digest(bound), 'execution': execution(config),
        'model_id': 'mlp', 'model_version': '3', 'input_recipe': layout.input_recipe,
        **classification}
    state['resume_identity'] = resume_identity(effective_config=loaded.effective, identity=identity,
        execution=state['execution'], classification=classification, storage_binding={})
    for name in ('artifacts', 'generations', 'updates', 'numeric', 'collections'):
        (parent / name).mkdir()
    local = LocalGameLearner(config.model.architecture(layout=layout), 17, config.model.inference,
        state['resume_identity'], actions=config.actions, rules=rules, admission=admission)
    service = RustPlayerInference(device='cpu', cache_entries=1, cache_weight_bytes=2 * 1024**3)
    registry = RustRunRegistry(ownership, state, config=config, admission=admission,
        pool=pool, writer=writer, local=local, service=service)
    rng = np.random.Generator(np.random.PCG64(17))
    controls = {'snapshot_pool': [0], 'schedule_opponent_rng': rng.bit_generator.state}
    pair = local.save(parent / 'artifacts' / '000000')
    registry.commit(pair, controls=controls, update=None, deadline=deadline)
    label, binding = registry.binding(0)
    bindings = {'random-card-instance-v2': RuntimePolicyBinding('random'),
        'heuristic-card-instance-v2': RuntimePolicyBinding('heuristic'), label: binding}
    groups = training_groups(17, 0, label, (label,), rng, actions_version=3)
    plan = CollectionPlan(groups, bindings, parent / 'collections' / 'batch-000000',
        frozen_linkage(state, parent, 0, groups, bindings), deadline, deadline)
    collector = RustPlayerCollector(registry)
    started = time.monotonic()
    collection = asyncio.run(collector.collect(plan))
    workspace = collection.writer._workspace_lease
    checkpoint = {'schema_version': 1, 'kind': 'rust-player-cpu-source-accounting',
        'phase': 'sealed-before-source', 'source_limit': pool.budget.limit,
        'source_used': pool.budget.used, 'source_peak': pool.budget.peak,
        'pool_base': pool._base_charge,
        'writer_holder_absent': workspace is None,
        'inventory_holder_absent': collector._inventory is None,
        'writer_workspace_bytes': 0 if workspace is None or workspace.reservation is None else workspace.reservation.size,
        'static_context_bytes': 0 if collector._source_control_charge is None else collector._source_control_charge.size,
        'snapshot_metadata_bytes': sum(charge.size for charge in registry.current._charges),
        'inference_bank_bytes': sum(bank.size for bank in registry.current.inference.banks)}
    (root / 'cpu-source-accounting.json').write_text(json.dumps(checkpoint, sort_keys=True))
    assert checkpoint['writer_holder_absent'] and checkpoint['inventory_holder_absent']
    assert checkpoint['writer_workspace_bytes'] == checkpoint['static_context_bytes'] == 0
    assert collection.originals is None and not pool.sources
    assert collection.report['healthy'] is True
    try:
        source = RustPlayerSource.project(collection)
    except BaseException:
        failed = registry.pending
        originals = failed.originals
        packed = originals.source
        checkpoint.update(phase='source-failed', source_used=pool.budget.used,
            source_peak=pool.budget.peak,
            source_graph_metadata_charges=sum(charge.size for charge in originals._reservations),
            packed_total=packed._budget.used, packed_group_cache=packed._cache_charge,
            packed_detached_bytes=packed._detached_charge,
            packed_original_metadata=packed._originals.metadata_charge)
        (root / 'cpu-source-accounting.json').write_text(json.dumps(checkpoint, sort_keys=True))
        raise
    assert source.state == 'SEALED' and source.closure_counts == {'natural': 16}
    numeric = source.persist(parent / 'numeric' / '000000')
    assert source.state == 'PERSISTED' and numeric['schema_version'] == 4 and numeric['decision_count'] > 0
    try:
        source.persist(parent / 'numeric' / 'duplicate')
    except ValueError:
        pass
    else:
        raise AssertionError('single-use numeric persistence was bypassed')
    result = {'schema_version': 1, 'kind': 'rust-player-cpu-source-component',
        'games': 16, 'accepted_learner_decisions': numeric['decision_count'],
        'manifest_sha256': collection.manifest_sha256, 'descriptor_sha256': collection.pending.descriptor_sha256,
        'numeric_sha256': sha256(parent / 'numeric/000000/rollout.json'),
        'generation_sha256': registry.current.generation_sha256, 'admission_sha256': digest(admission.observation),
        'producer': bound['producer'], 'seconds': time.monotonic() - started,
        'source_peak_bytes': pool.budget.peak, 'qualified': False}
    source.close()
    collector.runtime.quiesce()
    collector.runtime.join()
    policy = registry.current
    service.evict(policy.files['inference.json'], dispatcher=collector.runtime.dispatcher)
    cache = registry.cache[0]
    cache['lease'].close(cache)
    policy.close()
    pool.close()
    assert pool.budget.used == 0
    (root / 'cpu-source-result.json').write_text(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '_prepare_hpo_exec':
        prepare_exec_hpo(Path(sys.argv[2]))
    else:
        raise SystemExit(2)
