"""Fixed actual HPO training operation for the predeclared experiment protocol.

Run with the installed project venv. The observer measures this whole command;
configuration/materialization, actual admission and required retirement belong
inside it. Incomplete parents stay incomplete and produce no objective.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, canonical, integer


def _canonical_id(value: str) -> str:
    try:
        if str(uuid.UUID(value)) != value:
            raise ValueError('noncanonical UUID')
    except (ValueError, TypeError, AttributeError) as error:
        raise ContractError('declared canonical run/experiment UUID required') from error
    return value


def operation_inputs(config_path: Path, *, run_id: str, experiment_id: str,
                     learning_seed: int) -> tuple[Any, Any, Any, Path]:
    """Load and canonically materialize the configured settings, with no I/O owner."""
    from gf_cli.hpo_config import load_hpo_config, materialize_hpo_run
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_training.hpo_config import effective_settings
    from gf_training.hpo_gpu_lease import JobContext
    _canonical_id(run_id)
    _canonical_id(experiment_id)
    loaded = load_hpo_config(config_path)
    cfg = loaded.config
    if cfg.schema_version not in (5, 6) or cfg.model_id != 'mlp':
        raise ContractError('fixed operation supports only shared5/native6 MLP')
    settings = effective_settings(cfg.model, cfg.training, {}, layout=layout)
    materialized = materialize_hpo_run(loaded, settings, learning_seed=learning_seed, run_id=run_id)
    if cfg.schema_version == 6:
        from gf_cli.rust_player_entry import startup_preview
        startup_preview(materialized)  # Requires explicit P; no capacity/host admission.
    context = JobContext(experiment_id, uuid.uuid4().hex, 'training', run_id, None)
    namespace = Path(cfg.storage.parent_root) / '.operation-control' / experiment_id / run_id
    return loaded, materialized, context, namespace


def _invocation_data(loaded: Any, materialized: Any, context: Any, *, mode: str,
                     pause_after_update: int | None) -> Json:
    return {'schema_version': 1, 'kind': 'hpo-operation-invocation-data-v1',
        'context': asdict(context), 'campaign_config_sha256': loaded.sha256,
        'config_sha256': materialized.sha256, 'config_inputs': list(loaded.inputs),
        'mode': mode, 'pause_after_update': pause_after_update,
        'launcher_path': str(Path(__file__).absolute()),
        'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python_executable': str(Path(sys.executable).absolute())}


async def _legacy_operation(loaded: Any, materialized: Any, context: Any, namespace: Path, *,
                            mode: str, pause_after_update: int | None) -> Json:
    from gf_cli.card_instance import collect_hpo_ppo
    from gf_cli.rust_player_entry import _operation_namespace
    from gf_training.hpo_gpu_lease import admit_gpu_job
    from gf_training.hpo_ppo import HpoRunResult, train_hpo_ppo
    cfg = loaded.config
    with _operation_namespace(materialized, context, namespace, mode=mode) as store:
        store.put(f'invocations/{context.operation_id}.json', _invocation_data(loaded, materialized,
            context, mode=mode, pause_after_update=pause_after_update))
        with admit_gpu_job(context, global_cuda_bytes=cfg.operating.managed_cuda_bytes,
                max_workers=cfg.operating.max_participant_workers) as job:
            async def collector(plan: Any, ledger: Any) -> Any:
                if ledger is not job.ledger:
                    raise ContractError('HPO collector requires the exact caller job ledger')
                return await collect_hpo_ppo(materialized, plan, ledger)
            result = await train_hpo_ppo(materialized, collector,
                Path(cfg.storage.parent_root) / context.run_id, gpu_job=job, mode=mode,
                pause_after_update=pause_after_update)
            if type(result) is not HpoRunResult:
                raise ContractError('actual HPO operation result required')
            try:
                # Diagnostic DATA only, never a reconstructed completion/Source.
                output = {**result.report, 'operation_context': asdict(context),
                    'gpu_termination': asdict(result.gpu_termination)}
                canonical(output)
            finally:
                if result.completion is not None:
                    try:
                        result.completion.close()
                    except BaseException as error:
                        vars(error)['hpo_completion_owner'] = result.completion
                        raise
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--learning-seed', type=int, required=True)
    parser.add_argument('--mode', choices=('fresh', 'resume'), required=True)
    parser.add_argument('--pause-after-update', type=int)
    args = parser.parse_args(argv)
    if args.pause_after_update is not None:
        integer(args.pause_after_update, 1)
    loaded, materialized, context, namespace = operation_inputs(args.config, run_id=args.run_id,
        experiment_id=args.experiment_id, learning_seed=args.learning_seed)
    if loaded.config.schema_version == 6:
        from gf_cli.rust_player_entry import prepare_hpo_operation
        with prepare_hpo_operation(materialized, context, namespace, mode=args.mode,
                pause_after_update=args.pause_after_update) as prepared:
            prepared.store.put(f'invocations/{context.operation_id}.json',
                _invocation_data(loaded, materialized, context, mode=args.mode,
                    pause_after_update=args.pause_after_update))
            prepared.execute()  # Actual exec replaces this process; no Python parent remains.
        raise ContractError('fixed operation supervisor exec unexpectedly returned')
    output = asyncio.run(_legacy_operation(loaded, materialized, context, namespace,
        mode=args.mode, pause_after_update=args.pause_after_update))
    print(canonical(output))  # All actual legacy completion/job/Store scopes have closed.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
