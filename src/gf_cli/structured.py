"""Actions2 composition and precollection guards; no eager Torch imports."""
from __future__ import annotations

from typing import Any

from gf_common.domain import ContractError, RuntimePolicyBinding
from gf_common.preprocessing import PreparationConfig
from gf_models.config import ModelConfig
from gf_models.structured_baselines import make_structured_policy
from gf_models.structured_binding import ConfiguredNeuralModelV2
from gf_simulator_client.environment import EnvironmentSettings
from gf_simulator_client.structured_environment import StructuredSimulatorEnvironment
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.collection import CollectionPlan
from gf_training.structured_journal import StructuredJournal


def structured_bindings(model: object, opponent: object) -> tuple[dict[str, RuntimePolicyBinding], tuple[str, str]]:
    registry = {}
    labels = []
    for config in (model, opponent):
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
