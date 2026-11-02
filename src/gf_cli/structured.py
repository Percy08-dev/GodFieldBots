"""Actions2 composition and precollection guards; no eager Torch imports."""
from __future__ import annotations

from typing import Any

from gf_common.domain import ContractError, RuntimePolicyBinding
from gf_common.preprocessing import PreparationConfig
from gf_models.config import ModelConfig
from gf_models.structured_baselines import make_structured_policy
from gf_models.structured_binding import ConfiguredNeuralModelV2
from gf_models.structured_tree_config import ConfiguredTreeModelV2
from gf_simulator_client.environment import EnvironmentSettings
from gf_simulator_client.structured_environment import StructuredSimulatorEnvironment
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.collection import CollectionPlan
from gf_training.structured_journal import StructuredJournal


def structured_bindings(model: object, opponent: object) -> tuple[dict[str, RuntimePolicyBinding], tuple[str, str]]:
    registry = {}
    labels = []
    for config in (model, opponent):
        if isinstance(config, ConfiguredTreeModelV2):
            raise ContractError("formal Tree2 inference requires C3b trained provenance; toy components have no Source authority")
        if isinstance(config, ConfiguredNeuralModelV2):
            raise ContractError("formal model2 inference requires completed healthy Generation2 parent qualification; integration pending")
        if not isinstance(config, ModelConfig) or config.model_id not in ("random", "heuristic"):
            raise ContractError("actions2 collection currently requires structured baseline roles")
        label = config.model_id + "-structured-v1"
        registry[label] = RuntimePolicyBinding(config.model_id)
        labels.append(label)
    return registry, (labels[0], labels[1])


def structured_backend(config: PreparationConfig) -> str:
    if config.reader != "structured-v1":
        raise ContractError("actions2 requires explicit structured-v1 reader")
    if config.backend == "python":
        return "python-reference-structured-v1"
    if config.backend != "rust":
        raise ContractError("unknown structured preparation backend")
    from gf_training.native_backend import load_native, reader_settings
    reader_settings(config, load_native())  # exact capability/signature before any Simulator I/O.
    return "rust-structured-v1"


def structured_environment(settings: EnvironmentSettings, ledger: ReservationLedger) -> StructuredSimulatorEnvironment:
    return StructuredSimulatorEnvironment(settings, make_structured_policy, StructuredJournal, ledger=ledger)


def validate_owned_collection(plan: CollectionPlan, ledger: ReservationLedger, config: Any, *, deadline: float) -> None:
    if config.preprocessing.backend == 'rust':
        structured_backend(config.preprocessing)  # Normal/public path always requires real markers first.
        from gf_training.native_backend import load_native
        from gf_training.structured_ppo import _NativeComponentOps
        _NativeComponentOps(load_native()).validate_owned_collection(plan, ledger, config=config, deadline=deadline)
        return
    from gf_training.structured_ppo import cap
    from gf_training.structured_ppo_source import validate_owned_collection as validate
    validate(plan, ledger, max_owned_bytes=cap(config), deadline=deadline)


async def collect_final_validation(loaded: Any, parent: Any, output: Any) -> Any:
    import time
    from pathlib import Path

    from gf_common.action_contract import require_qualified_contract
    from gf_common.structured import ActionConfigV2
    from gf_training.structured_tree_validation import FinalValidationContext
    if not isinstance(loaded.config.actions,ActionConfigV2):
        raise ContractError('actor-only model2 final validation requires actions2')
    require_qualified_contract(loaded.config.actions.action_contract)
    ledger=ReservationLedger(loaded.config.run.global_cuda_budget_bytes,loaded.config.run.max_participant_workers)
    from dataclasses import asdict, replace

    from gf_common.preprocessing import preparation_config
    source_settings=preparation_config(asdict(replace(loaded.config.preprocessing,backend='rust',reader='structured-v1')))
    async with FinalValidationContext(loaded,Path(parent),Path(output),ledger,source_settings=source_settings,deadline=time.monotonic()+loaded.config.run.wall_seconds) as context:
        assert context.projected is not None and context.plan is not None
        from gf_cli import _run_experiment
        directory,report=await _run_experiment(context.projected,'collect',collection=context.plan,ledger=ledger,
                                             _structured_validation_context=context)
        return context.accept_collection(directory,report)


class _FinalActorPreflightOwner:
    """CLI-owned transport probe; retains the exact source context until reap."""
    def __init__(self, context: Any) -> None:
        self.context=context
        self.worker: Any=None

    async def close(self) -> None:
        if self.worker is None:return
        await self.worker.close()


async def preflight_final_validation_actor(context: Any, config: Any, *, deadline: float) -> None:
    from gf_simulator_client.structured_worker import (
        StructuredParticipantWorker,
        _startup_cleanup_owner,
    )
    from gf_training.structured_tree_validation import FinalValidationContext
    if type(context) is not FinalValidationContext or context.plan is None:
        raise ContractError('actual concrete final validation context required')
    context.validate_collection(context.plan,context.ledger,config,deadline=deadline)
    plan=context.plan
    first=plan.groups[0].episodes[0]
    assert first.learner_viewer is not None
    binding=plan.bindings[first.policies[first.learner_viewer-1]]
    owner=_FinalActorPreflightOwner(context)
    initiating=None
    try:
        try:
            owner.worker=StructuredParticipantWorker(make_structured_policy,binding.model_id,
                first.sampling_seeds[first.learner_viewer-1],config.actions,config.server.worker_seconds,
                ledger=context.ledger,policy_options=binding.policy_options)
        except BaseException as error:
            owner.worker=_startup_cleanup_owner(error,context.ledger)
            raise
        await owner.worker.initialize()
    except BaseException as error:
        initiating=error
        raise
    finally:
        try:
            await owner.close()
        except BaseException as cleanup:
            if owner.worker is not None and owner.worker.process.poll() is None:
                vars(cleanup)['_structured_validation_cleanup_owner']=owner
                if initiating is not None:
                    initiating.add_note('final actor preflight cleanup: '+type(cleanup).__name__)
                    vars(initiating)['_structured_validation_cleanup_owner']=owner
                else:raise
            elif initiating is None:raise
    context.validate_collection(plan,context.ledger,config,deadline=deadline)
