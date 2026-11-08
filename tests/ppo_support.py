"""Synthetic collector integration identities; never executable Simulator code."""
import json
from dataclasses import asdict, replace
from pathlib import Path

from gf_cli.config import LoadedConfig, load_config
from gf_common.domain import digest
from gf_common.schedule import serialize_schedule
from gf_training.collection import FinalizedCollection
from gf_training.journal import atomic_json
from gf_training.lineage import read_json
from tests.lightgbm_support import synthetic_run


def synthetic_server_identity(tmp_path: Path) -> dict[str, str | None]:
    """Keep synthetic coordinators independent of mutable external source/docs."""
    binary = tmp_path / 'synthetic-simulator.identity'
    binary.write_bytes(b'synthetic collector identity only; not an executable engine')
    return {'source_dir': None, 'binary': str(binary)}


def configuration(tmp_path, max_updates=2, seed=17):
    # Synthetic collectors never execute a Simulator. Bind only an inert local
    # binary identity, without claiming verification of mutable external specs.
    loaded = load_config('configs/experiments/ppo-cpu.toml', {
        'preprocessing': {'backend': 'python'},
        'server': synthetic_server_identity(tmp_path),
    })
    config = replace(loaded.config, training=replace(loaded.config.training, seeds=(seed,), max_updates=max_updates))
    return LoadedConfig(config, asdict(config), digest(asdict(config)), ())


def synthetic_collector(make_input, *, both_viewers=False, outcomes=None):
    async def collect(plan):
        from gf_models.mlp.policy import MLPPolicy
        manifest = synthetic_run(plan.output, make_input, seeds=tuple(group.seed for group in plan.groups), planned=plan.groups,
                                 split='train' if plan.kind == 'ppo_batch' else 'validation', both_viewers=both_viewers, outcomes=outcomes)
        # Use exactly the frozen stored candidates; policy-only evidence receives no parent IDs.
        for path in plan.output.glob('*/player-*.jsonl'):
            records = [json.loads(line) for line in path.read_text().splitlines()]
            header = records[0]
            policy_options = plan.bindings[header['policy']].policy_options
            if policy_options is None:
                continue
            policy = MLPPolicy(policy_options['checkpoint'], policy_options['expected_manifest_digest'])
            policy.reset(123)
            for record in records:
                if record['kind'] != 'decision':
                    continue
                from gf_common.domain import CandidateSet, DecisionInput
                raw = record['input']
                value = DecisionInput(raw['observation'], tuple(raw['events']), raw['history_end'], tuple(raw['typed_metadata']), raw['catalog'], raw['action_options'])
                raw_candidates = record['candidates'].copy()
                candidates = CandidateSet(tuple(raw_candidates.pop('actions')), **raw_candidates)
                selection = policy.select(value, candidates)
                from gf_models.mlp.features import encode_decision
                actor, state = encode_decision(value, candidates.actions, policy.config.encoder)
                index = record['selection']['index']
                torch = policy.torch
                with torch.no_grad():
                    logprobs, values = policy.network(torch.from_numpy(actor)[None], torch.from_numpy(state)[None], torch.ones((1, len(actor)), dtype=torch.bool))
                record['selection'] = asdict(replace(selection, index=index, logprob=float(logprobs[0, index]), value=float(values[0])))
            path.write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records))
        parent = read_json(Path(plan.linkage['parent_run_path']) / 'run.json')
        integration = parent['identity']['integration']
        manifest.update(binary_sha256=integration['binary_sha256'], server_config_sha256=integration['server_config_sha256'],
                        simulator={'revision': integration['source_revision'], 'spec_sha256': integration['source_spec_sha256']}, code=parent['identity']['code'])
        manifest.update(schema_version=4, collection_kind=plan.kind, ppo=plan.linkage,
                        schedule=serialize_schedule(plan.groups, 4), policy_bindings={key: asdict(value) for key, value in plan.bindings.items()})
        atomic_json(plan.output / 'manifest.json', manifest)
        return FinalizedCollection(plan.output, read_json(plan.output / 'report.json'))
    return collect
