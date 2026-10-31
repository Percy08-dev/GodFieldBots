"""Compose safe policy identities without exposing experiment configuration."""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from gf_common.domain import ActionConfig, ContractError, Json, RuntimePolicyBinding, digest
from gf_models.config import GRUModelConfig, MLPModelConfig, ModelConfig, TransformerModelConfig


def bindings(model: ModelConfig | MLPModelConfig | GRUModelConfig | TransformerModelConfig, opponent: ModelConfig | MLPModelConfig | GRUModelConfig | TransformerModelConfig, actions: ActionConfig, *, deadline: float = float("inf")) -> tuple[dict[str, RuntimePolicyBinding], tuple[str, str]]:
    registry: dict[str, RuntimePolicyBinding] = {}
    labels = []
    for config in (model, opponent):
        if config.model_id in ("random", "heuristic"):
            label = config.model_id
            binding = RuntimePolicyBinding(label)
        elif isinstance(config, (MLPModelConfig, GRUModelConfig, TransformerModelConfig)):
            if config.checkpoint is None:
                raise ContractError("formal PPO inference requires a completed parent checkpoint")
            from gf_models.arrays import sha256
            from gf_models.ppo_artifacts import inspect_artifact as inspect_mlp
            from gf_training.lineage import qualify_ppo
            sha = sha256(Path(config.checkpoint) / "inference.json")
            inspect_mlp(Path(config.checkpoint), expected_digest=sha, action_config_hash=digest(asdict(actions)), model=config, deadline=deadline)
            qualify_ppo(Path(config.checkpoint), sha)
            label = config.model_id + "-" + sha
            binding = RuntimePolicyBinding(config.model_id, {"checkpoint": config.checkpoint, "expected_manifest_digest": sha})
        else:
            if config.checkpoint is None:
                raise ContractError("formal inference requires a ready trained checkpoint")
            from gf_models.lightgbm.model import inspect_artifact
            from gf_models.lightgbm.policy import manifest_digest
            sha = manifest_digest(config.checkpoint)
            inspect_artifact(Path(config.checkpoint), expected_digest=sha, action_config_hash=digest(asdict(actions)), model=config, deadline=deadline)
            qualify_training_bundle(Path(config.checkpoint), sha)
            label = "lightgbm-" + sha
            binding = RuntimePolicyBinding("lightgbm", {"checkpoint": config.checkpoint, "expected_manifest_digest": sha})
        if label in registry and registry[label] != binding:
            raise ContractError("ambiguous policy binding")
        registry[label] = binding
        labels.append(label)
    return registry, (labels[0], labels[1])


def qualify_training_bundle(checkpoint: Path, expected_digest: str) -> Json:
    """Parent-only formal qualification; no training controls enter a worker."""
    try:
        record = json.loads((checkpoint.parent / "training.json").read_text())
        if type(record.get("schema_version")) is not int or record["schema_version"] not in (1, 2) or record.get("status") != "complete" or record.get("purpose") != "trained":
            raise ContractError("checkpoint requires a completed trained run bundle")
        if record['schema_version'] == 2:
            source = record.get('source_provenance', {})
            if record.get('cost_mode') != 'fixed_data' or not isinstance(source, dict) or any(not isinstance(source.get(key), str) or len(source[key]) != 64 for key in ('fixed_dataset_content_sha256', 'shared_generation_cost_key')):
                raise ContractError('schema-2 completion requires fixed-data cost provenance')
        elif record.get('cost_mode', 'inclusive_generation') != 'inclusive_generation':
            raise ContractError('schema-1 completion requires inclusive-generation cost mode')
        if record.get("checkpoint") != checkpoint.name or checkpoint.name != "checkpoint" or record.get("inference_manifest_sha256") != expected_digest:
            raise ContractError("training/checkpoint identity mismatch")
        if any(record.get(key) for key in ("failure", "message", "quarantine", "invalidation_failures", "correction_save_failures")):
            raise ContractError("training run contains durable failure evidence")
        health = record.get("health")
        if not isinstance(health, dict) or set(health) - {"exit_code", "stdout.log_bytes", "stderr.log_bytes"} or type(health.get("exit_code")) is not int or health["exit_code"] != 0:
            raise ContractError("training run health does not qualify")
        for key in ("stdout.log_bytes", "stderr.log_bytes"):
            if key in health and (type(health[key]) is not int or health[key] < 0):
                raise ContractError("invalid training log health")
        elapsed, charged, budget = (record.get(key) for key in ("elapsed_seconds", "charged_total_seconds", "wall_seconds"))
        if any(type(number) not in (int, float) or not math.isfinite(number) for number in (elapsed, charged, budget)) or not 0 <= elapsed <= charged <= budget:
            raise ContractError("training completion budget does not qualify")
        if hashlib.sha256((checkpoint / "inference.json").read_bytes()).hexdigest() != expected_digest:
            raise ContractError("qualified artifact changed")
        return record
    except (OSError, ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError("missing/malformed training completion record") from exc


async def native_preflight(registry: dict[str, RuntimePolicyBinding], actions: ActionConfig, *,
                           deadline: float, worker_seconds: float) -> None:
    """Reuse the clean worker READY exchange before any Simulator listener."""
    from gf_models.baselines import make_policy
    from gf_simulator_client.workers import ParticipantWorker
    for binding in registry.values():
        if binding.model_id not in ("lightgbm", "mlp", "gru", "transformer"):
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("native preflight deadline")
        worker = ParticipantWorker(make_policy, binding.model_id, 0, actions, min(worker_seconds, remaining), policy_options=binding.policy_options)
        original_error: BaseException | None = None
        try:
            await worker.initialize()
        except BaseException as exc:  # noqa: BLE001 - retain original error through owned cleanup
            original_error = exc
        finally:
            try:
                # close owns a shielded shared task and joins it through repeated
                # cancellation. A cleanup cancellation cannot replace init error.
                await worker.close(timeout=max(0, min(2, deadline - time.monotonic())))
            except BaseException as cleanup_error:  # noqa: BLE001 - rethrow only after owned cleanup
                original_error = original_error or cleanup_error
        if original_error is not None:
            raise original_error
        if worker.health or time.monotonic() >= deadline:
            raise RuntimeError("native preflight cleanup/deadline failure")
