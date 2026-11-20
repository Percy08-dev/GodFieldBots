"""Retained external inference inputs; no Torch or learning/Source authority."""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import CardInstanceRules
from gf_common.domain import (
    ContractError,
    Json,
    RuntimePolicyBinding,
    canonical,
    clone,
    closed,
    digest,
    integer,
)
from gf_common.structured_layout import layout_for_contract
from gf_models.structured_binding import (
    ConfiguredNeuralModelV3,
    ConfiguredNeuralModelV4,
    child_cuda_visibility,
    configured_model_v3,
    configured_model_v4,
    inference_model,
    input_binding_game_v3,
    policy_label_game_v3,
    policy_label_game_v4,
    safe_options_game_v3,
    safe_options_game_v4,
)
from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_runtime import own_game_worker_settings_v3
from gf_models.structured_sealed import (
    SealedScan,
    basename,
    checksum,
    inspect_retained_game_inference_v3,
    inspect_retained_game_inference_v4,
    validate_game_inference_metadata,
    validate_game_inference_metadata_v4,
)

EXTERNAL_INPUT_CONTRACT = "retained-external-inference-v1"


def validate_neural_metadata(raw: Json, *, actions: ActionConfigV3, rules: CardInstanceRules) -> tuple[Json, str]:
    """Validate a recorded own-model declaration; never read current external files."""
    closed(raw, {"input_contract", "policy_options", "inference_manifest", "files"})
    if raw["input_contract"] != EXTERNAL_INPUT_CONTRACT:
        raise ContractError("retained external inference contract required")
    v4 = type(raw["policy_options"]) is dict and raw["policy_options"].get("version") == "4"
    safe_parser = safe_options_game_v4 if v4 else safe_options_game_v3
    safe = safe_parser(raw["policy_options"], actions=actions, rules=rules)
    manifest = raw["inference_manifest"]
    metadata_parser = validate_game_inference_metadata_v4 if v4 else validate_game_inference_metadata
    model = metadata_parser(manifest, actions=actions, rules=rules)
    expected_model = {key: value for key, value in safe["model"].items() if key != "resources"} if v4 else safe["model"]
    if canonical(manifest["model"]) != canonical(expected_model):
        raise ContractError("sealed model differs from own execution model")
    expected = {"inference.json": safe["expected_manifest_digest"], **manifest["weights"]["files"]}
    files = raw["files"]
    if type(files) is not list or len(files) != len(expected):
        raise ContractError("external inference file inventory mismatch")
    seen: set[str] = set()
    for row in files:
        closed(row, {"path", "sha256", "identity"})
        name = basename(row["path"])
        if name in seen or name not in expected or checksum(row["sha256"]) != expected[name]:
            raise ContractError("external inference file hash/name mismatch")
        seen.add(name)
        if type(row["identity"]) not in (list, tuple) or len(row["identity"]) != 5:
            raise ContractError("external inference original stat identity required")
        for number in row["identity"]:
            integer(number, 0)
    binding = InferenceBindingV2(**safe["inference"])
    if safe["cuda_visible_devices"] != child_cuda_visibility(binding.device, safe["cuda_visible_devices"]):
        raise ContractError("own inference CUDA visibility differs from child execution")
    own_game_worker_settings_v3(model.model_id, actions, safe, rules=rules)
    if v4:
        from gf_models.hpo_config import model_config_v4
        model = model_config_v4(safe["model"], layout=layout_for_contract(actions.action_contract))
    labeler = policy_label_game_v4 if v4 else policy_label_game_v3
    label = labeler(model, binding, safe["expected_manifest_digest"],
        action_contract=actions.action_contract, rules=rules, action_config_hash=digest(asdict(actions)),
        cuda_visible_devices=safe["cuda_visible_devices"])
    return safe, label


class RetainedGameArtifacts:
    def __init__(self, models: tuple[ConfiguredNeuralModelV3 | ConfiguredNeuralModelV4, ...], actions: ActionConfigV3, *,
                 rules: CardInstanceRules, max_owned_bytes: int, preflight_deadline: float,
                 finalization_deadline: float) -> None:
        input_binding_game_v3(actions.action_contract, rules)
        if (type(models) is not tuple or not 1 <= len(models) <= 2
                or any(type(model) not in (ConfiguredNeuralModelV3, ConfiguredNeuralModelV4) for model in models)
                or actions.max_sequence_length > 18):
            raise ContractError("one or two explicit real18 neural roles required")
        self._initialize(models, actions, rules=rules, max_owned_bytes=max_owned_bytes,
            preflight_deadline=preflight_deadline, finalization_deadline=finalization_deadline)

    @classmethod
    def for_registry(cls, frozenbindings: dict[str, RuntimePolicyBinding], actions: ActionConfigV3, *,
                     rules: CardInstanceRules, max_owned_bytes: int, preflight_deadline: float,
                     finalization_deadline: float) -> RetainedGameArtifacts:
        return cls._for_registry(frozenbindings, actions, rules=rules, max_owned_bytes=max_owned_bytes,
            preflight_deadline=preflight_deadline, finalization_deadline=finalization_deadline)

    @classmethod
    def _for_registry(cls, frozenbindings: dict[str, RuntimePolicyBinding], actions: ActionConfigV3, *,
                     rules: CardInstanceRules, max_owned_bytes: int, preflight_deadline: float,
                     finalization_deadline: float, _hpo_pool_size: int | None = None,
                     _scan: SealedScan | None = None) -> RetainedGameArtifacts:
        """Own the exact current/pool registry; grants no live run or Source authority."""
        if _scan is not None and not isinstance(_scan, SealedScan):
            raise ContractError('actual private SealedScan input required')
        input_binding_game_v3(actions.action_contract, rules)
        if _hpo_pool_size is not None:
            integer(_hpo_pool_size, 1, 2**31 - 1)
        from gf_common.refresh_heuristic import parse_refresh_options, refresh_policy_label
        limit = 6 if _hpo_pool_size is None else _hpo_pool_size + 1
        if type(frozenbindings) is not dict or not frozenbindings or len(frozenbindings) > limit + 3:
            raise ContractError('closed frozen game policy registry required')
        if any(type(value) is not RuntimePolicyBinding for value in frozenbindings.values()):
            raise ContractError('closed frozen runtime policy bindings required')
        neural = [value for value in frozenbindings.values() if value.policy_options is not None and value.model_id != "heuristic_refresh"]
        if not 1 <= len(neural) <= limit or len(frozenbindings) > len(neural) + 3:
            raise ContractError('registry requires one to six exact neural identities and canonical baselines')
        models = []
        expected: dict[str, tuple[str, Json]] = {}
        layout = layout_for_contract(actions.action_contract)
        for label, value in frozenbindings.items():
            if value.model_id == "heuristic_refresh":
                if label != refresh_policy_label(actions, rules, parse_refresh_options(value.policy_options if value.policy_options is not None else {})):
                    raise ContractError("configured baseline registry label mismatch")
                continue
            if value.policy_options is None:
                if value.model_id not in ('random', 'heuristic') or label != value.model_id + '-card-instance-v2':
                    raise ContractError('noncanonical registry baseline binding')
                continue
            v4 = type(value.policy_options) is dict and value.policy_options.get("version") == "4"
            if _hpo_pool_size is not None and not v4:
                raise ContractError('HPO registry requires explicit model4 policies')
            safe_parser = safe_options_game_v4 if v4 else safe_options_game_v3
            configure = configured_model_v4 if v4 else configured_model_v3
            safe = safe_parser(value.policy_options, actions=actions, rules=rules)
            configured = configure({**safe['model'], 'checkpoint': safe['checkpoint'],
                'inference': safe['inference'], 'cuda_visible_devices': safe['cuda_visible_devices']}, layout=layout)
            model = configured.architecture(layout=layout)
            label_options = {"action_contract": actions.action_contract, "rules": rules,
                "action_config_hash": digest(asdict(actions)), "cuda_visible_devices": safe["cuda_visible_devices"]}
            if type(configured) is ConfiguredNeuralModelV4:
                actual_label = policy_label_game_v4(configured.architecture(layout=layout), configured.inference,
                    safe["expected_manifest_digest"], **label_options)
            else:
                assert type(configured) is ConfiguredNeuralModelV3
                actual_label = policy_label_game_v3(configured.architecture(layout=layout), configured.inference,
                    safe["expected_manifest_digest"], **label_options)
            if label != actual_label or value.model_id != model.model_id:
                raise ContractError('frozen registry label/model differs from exact game options')
            key = digest(asdict(configured))
            if key in expected:
                raise ContractError('duplicate frozen registry neural identity')
            expected[key] = (label, safe)
            models.append(configured)
        owner = cls.__new__(cls)
        try:
            owner._initialize(tuple(models), actions, rules=rules, max_owned_bytes=max_owned_bytes,
                preflight_deadline=preflight_deadline, finalization_deadline=finalization_deadline,
                expected=expected, **({} if _scan is None else {'_scan': _scan}))
            return owner
        except BaseException:
            if hasattr(owner, '_scan'):
                owner.release()  # In-memory retirement only; preserve the initiating failure.
            raise

    @classmethod
    def for_hpo_registry(cls, frozenbindings: dict[str, RuntimePolicyBinding], actions: ActionConfigV3, *,
                         pool_size: int, rules: CardInstanceRules, max_owned_bytes: int,
                         preflight_deadline: float, finalization_deadline: float,
                         _scan: SealedScan | None = None) -> RetainedGameArtifacts:
        return cls._for_registry(frozenbindings, actions, rules=rules, max_owned_bytes=max_owned_bytes,
            preflight_deadline=preflight_deadline, finalization_deadline=finalization_deadline,
            _hpo_pool_size=pool_size, **({} if _scan is None else {'_scan': _scan}))

    def _initialize(self, models: tuple[ConfiguredNeuralModelV3 | ConfiguredNeuralModelV4, ...], actions: ActionConfigV3, *,
                    rules: CardInstanceRules, max_owned_bytes: int, preflight_deadline: float,
                    finalization_deadline: float, expected: dict[str, tuple[str, Json]] | None = None,
                    _scan: SealedScan | None = None) -> None:
        for deadline in (preflight_deadline, finalization_deadline):
            if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0:
                raise ContractError("original finite artifact deadlines required")
        if preflight_deadline > finalization_deadline:
            raise ContractError("artifact preflight exceeds original finalization deadline")
        if _scan is not None and (not isinstance(_scan, SealedScan) or _scan.limit != max_owned_bytes
                or _scan.deadline != preflight_deadline):
            raise ContractError('private scan must have the same declared limits/deadline')
        self._scan = _scan if _scan is not None else SealedScan(max_owned_bytes=max_owned_bytes, deadline=preflight_deadline)
        self._finalization_deadline = finalization_deadline
        self._entries: dict[str, Json] = {}
        self._models: dict[str, str] = {}
        self._released = False
        self._failed = False
        layout = layout_for_contract(actions.action_contract)
        for configured in models:
            if configured.checkpoint is None:
                raise ContractError("neural game inference requires an explicit game checkpoint")
            directory = Path(configured.checkpoint).absolute()
            model = configured.architecture(layout=layout)
            # The expected digest is taken from the original manifest read and
            # retained under the same scan; children receive that exact digest.
            frozen = None if expected is None else expected[digest(asdict(configured))]
            manifest = self._scan.object(directory / "inference.json",
                None if frozen is None else frozen[1]['expected_manifest_digest'])
            manifest_digest = self._scan.files[directory / "inference.json"][1]
            v4 = type(configured) is ConfiguredNeuralModelV4
            inspect = inspect_retained_game_inference_v4 if v4 else inspect_retained_game_inference_v3
            artifact = inspect(directory, expected_inference_digest=manifest_digest,
                actions=actions, rules=rules, scan=self._scan)
            if canonical(manifest["model"]) != canonical(inference_model(model)):
                raise ContractError("external artifact differs from configured Model3")
            options = {"version": "4" if v4 else "3", "checkpoint": str(directory), "expected_manifest_digest": manifest_digest,
                "model": asdict(model), "inference": asdict(configured.inference),
                "cuda_visible_devices": child_cuda_visibility(configured.inference.device, configured.cuda_visible_devices),
                **input_binding_game_v3(actions.action_contract, rules), "action_config_hash": digest(asdict(actions))}
            entry = {"input_contract": EXTERNAL_INPUT_CONTRACT, "policy_options": options,
                "inference_manifest": artifact.inference, "files": [
                    {"path": name, "sha256": checksum, "identity": list(self._scan.files[directory / name][0])}
                    for name, checksum in sorted(artifact.files.items())]}
            _, label = validate_neural_metadata(entry, actions=actions, rules=rules)
            if frozen is not None and (label != frozen[0] or canonical(options) != canonical(frozen[1])):
                raise ContractError('retained artifact differs from exact frozen registry label/options/digest')
            self._scan.charge(32 * len(canonical(entry).encode()) + 65536)
            if label in self._entries and canonical(self._entries[label]) != canonical(entry):
                raise ContractError("one policy label cannot name distinct external originals")
            self._entries[label] = clone(entry)
            self._models[digest(asdict(configured))] = label
        self._scan.finalize()

    def _live(self) -> None:
        if self._released or self._failed:
            raise ContractError("external artifact input owner has been released or failed")

    @property
    def retained_bytes(self) -> int:
        """Actual held scan custody only; grants no new artifact/Source permission."""
        self._live()
        return self._scan.retained

    def binding(self, model: ConfiguredNeuralModelV3 | ConfiguredNeuralModelV4) -> tuple[str, RuntimePolicyBinding]:
        self._live()
        label = self._models.get(digest(asdict(model)))
        if label is None:
            raise ContractError("model is not an owned external inference input")
        options = clone(self._entries[label]["policy_options"])
        return label, RuntimePolicyBinding(model.model_id, options)

    def require_options(self, model_id: str, options: Json) -> None:
        self._live()
        if not any(entry["policy_options"]["model"]["model_id"] == model_id
            and canonical(entry["policy_options"]) == canonical(options) for entry in self._entries.values()):
            raise ContractError("worker options differ from retained original inference input")

    def journal_metadata(self, policy_label: str) -> Json:
        self._live()
        if policy_label not in self._entries:
            raise ContractError("unknown retained neural policy label")
        return clone(self._entries[policy_label])

    def finalize(self, *, deadline: float) -> Json:
        self._live()
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ContractError("finite original artifact recheck deadline required")
        try:
            self._scan.deadline = min(deadline, self._finalization_deadline)
            self._scan.finalize()
            return {"input_contract": EXTERNAL_INPUT_CONTRACT, "policies": clone(self._entries)}
        except BaseException:
            self._failed = True
            raise

    def release(self) -> None:
        # No filesystem operations after final publication/reap.
        self._released = True
        self._entries.clear()
        self._models.clear()
        self._scan.files.clear()
        self._scan.inventories.clear()
