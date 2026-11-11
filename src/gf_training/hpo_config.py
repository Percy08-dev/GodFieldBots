"""Closed HPO-only recipes and conditional joint search; no numerical/Optuna I/O."""
from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT, StructuredLayout
from gf_models.hpo_config import model_config_v4, model_defaults_v4
from gf_models.structured_config import ModelConfigV4, architecture_identity


@dataclass(frozen=True)
class HPOTrainingConfig:
    version: str = "hpo-ppo-v1"
    algorithm: str = "ppo"
    optimizer: str = "adam"
    reward: str = "natural-terminal-v1"
    gamma: float = 1.0
    collection_games: int = 16
    adam_learning_rate: float = 0.0003
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    epochs: int = 4
    minibatch_size: int = 128
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    tbptt_length: int = 32


def finite_number(value: Any, low: float, high: float, *, strict_low: bool = False,
                  strict_high: bool = False) -> float:
    if (type(value) not in (int, float) or value < low or value > high
            or not math.isfinite(value) or strict_low and value == low
            or strict_high and value == high):
        raise ContractError("finite numeric value outside admitted range")
    return float(value)


def hpo_training_config(value: Json) -> HPOTrainingConfig:
    closed(value, set(asdict(HPOTrainingConfig())))
    defaults = HPOTrainingConfig()
    for name in ("version", "algorithm", "optimizer", "reward"):
        if type(value[name]) is not str or value[name] != getattr(defaults, name):
            raise ContractError("unsupported fixed HPO training contract")
    finite_number(value["gamma"], 1, 1)
    integer(value["collection_games"], 8, 65536)
    if value["collection_games"] % 8:
        raise ContractError("collection_games must be a positive multiple of eight")
    integer(value["epochs"], 1, 1000)
    integer(value["minibatch_size"], 1, 2**20)
    integer(value["tbptt_length"], 1, 1024)
    converted = dict(value)
    for name, low, high, strict_low, strict_high in (
        ("gamma", 1, 1, False, False), ("adam_learning_rate", 0, 1, True, False),
        ("adam_beta1", 0, 1, False, True), ("adam_beta2", 0, 1, False, True),
        ("adam_epsilon", 0, 1, True, False), ("clip_range", 0, 1, True, True),
        ("value_coefficient", 0, 1e6, False, False), ("entropy_coefficient", 0, 1e6, False, False),
        ("gae_lambda", 0, 1, False, False), ("max_grad_norm", 0, 1e6, True, False)):
        converted[name] = finite_number(value[name], low, high, strict_low=strict_low, strict_high=strict_high)
    return HPOTrainingConfig(**converted)


@dataclass(frozen=True)
class EffectiveSettings:
    model: ModelConfigV4
    training: HPOTrainingConfig
    identity: str


def _leaf_paths(value: Json, prefix: str) -> set[str]:
    result: set[str] = set()
    for key, item in value.items():
        path = f"{prefix}.{key}"
        result.update(_leaf_paths(item, path) if type(item) is dict else {path})
    return result


_FIXED_MODEL = {"model_id", "version", "input_recipe", "max_positions", "initialization", "dropout", "precision"}
_FIXED_TRAINING = {"version", "algorithm", "optimizer", "reward", "gamma"}


def searchable_paths(model_id: str) -> set[str]:
    model = asdict(model_defaults_v4(model_id))
    return (_leaf_paths({key: item for key, item in model.items() if key not in _FIXED_MODEL | {"resources"}}, "model")
            - {"model.event.aggregate_recipe", "model.temporal.kind", "model.temporal.max_decisions"}
            | {"training." + name for name in HPOTrainingConfig.__dataclass_fields__ if name not in _FIXED_TRAINING})


def _set_path(root: Json, path: str, value: Any) -> None:
    parts = path.split(".")
    target = root
    for name in parts[:-1]:
        target = target[name]
    target[parts[-1]] = clone(value)


def _get_path(root: Json, path: str) -> Any:
    value: Any = root
    for name in path.split("."):
        value = value[name]
    return value


def effective_settings(model: ModelConfigV4, training: HPOTrainingConfig, values: Json, *,
                       layout: StructuredLayout) -> EffectiveSettings:
    if type(values) is not dict or set(values) - searchable_paths(model.model_id):
        raise ContractError("unknown or operating-only search setting")
    root = {"model": asdict(model), "training": asdict(training)}
    for path, value in values.items():
        _set_path(root, path, value)
    effective_model = model_config_v4(root["model"], layout=layout)
    effective_training = hpo_training_config(root["training"])
    identity = digest({"model": architecture_identity(effective_model, layout=layout),
                       "training": asdict(effective_training)})
    return EffectiveSettings(effective_model, effective_training, identity)


def _freeze(value: Any) -> Any:
    if type(value) is list or type(value) is tuple:
        return tuple(_freeze(item) for item in value)
    if type(value) not in (str, int, float, bool, type(None)):
        raise ContractError("search choice must be a scalar or scalar array")
    if type(value) is float and not math.isfinite(value):
        raise ContractError("finite search choice required")
    return value


def _unique(values: Any) -> tuple[Any, ...]:
    if type(values) not in (list, tuple) or not values:
        raise ContractError("nonempty search choices required")
    result = tuple(_freeze(item) for item in values)
    if len({canonical(item) for item in result}) != len(result):
        raise ContractError("duplicate search choices")
    return result


def _conditions(raw: Any, allowed: set[str]) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    if type(raw) is not dict or set(raw) - allowed:
        raise ContractError("unknown condition setting")
    return tuple((path, _unique(raw[path])) for path in sorted(raw))


def _active(conditions: tuple[tuple[str, tuple[Any, ...]], ...], values: Json) -> bool:
    return all(path in values and any(canonical(values[path]) == canonical(choice) for choice in choices)
               for path, choices in conditions)


@dataclass(frozen=True)
class SearchParameter:
    path: str
    kind: str
    choices: tuple[Any, ...] = ()
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    value: Any = None
    when: tuple[tuple[str, tuple[Any, ...]], ...] = ()

    def decode(self, value: Any) -> Any:
        """Categoricals use stable choice indices, including tuple-valued widths."""
        if self.kind == "fixed":
            return self.value
        if self.kind == "categorical":
            return self.choices[integer(value, 0, len(self.choices) - 1)]
        assert self.low is not None and self.high is not None
        if self.kind == "int":
            integer(value, int(self.low), int(self.high))
        else:
            finite_number(value, self.low, self.high)
        if self.step is not None:
            steps = (value - self.low) / self.step
            if not math.isclose(steps, round(steps), rel_tol=1e-12, abs_tol=1e-9):
                raise ContractError("search value is off the declared step grid")
        return value


@dataclass(frozen=True)
class CoupledParameter:
    name: str
    paths: tuple[str, ...]
    choices: tuple[tuple[Any, ...], ...]
    when: tuple[tuple[str, tuple[Any, ...]], ...] = ()


@dataclass(frozen=True)
class SearchSpace:
    model_id: str
    parameters: tuple[SearchParameter, ...]
    coupled: tuple[CoupledParameter, ...]
    control_values: tuple[tuple[str, Any], ...]

    def resolve(self, proposal: Json, *, base_values: Json | None = None) -> Json:
        if type(proposal) is not dict:
            raise ContractError("closed proposal mapping required")
        values = dict(self.control_values)
        if base_values is not None:
            if type(base_values) is not dict or set(base_values) - values.keys():
                raise ContractError("closed base search settings required")
            if any(canonical(item) != canonical(values[path]) for path, item in base_values.items()):
                raise ContractError("base differs from validated search control; revalidate the space")
        result: Json = {}
        consumed: set[str] = set()
        for parameter in self.parameters:
            if not _active(parameter.when, values):
                continue
            if parameter.kind != "fixed" and parameter.path not in proposal:
                raise ContractError("missing active search proposal")
            value = parameter.decode(proposal.get(parameter.path))
            if parameter.kind != "fixed":
                consumed.add(parameter.path)
            values[parameter.path] = result[parameter.path] = value
        for coupled in self.coupled:
            if not _active(coupled.when, values):
                continue
            if coupled.name not in proposal:
                raise ContractError("missing active coupled proposal")
            choice = coupled.choices[integer(proposal[coupled.name], 0, len(coupled.choices) - 1)]
            consumed.add(coupled.name)
            for path, value in zip(coupled.paths, choice, strict=True):
                values[path] = result[path] = value
        if set(proposal) != consumed:
            raise ContractError("unknown, fixed or inactive proposal fields")
        return result

    @property
    def identity(self) -> str:
        return digest(asdict(self))


def valid_attention_choices(widths: Any, heads: Any) -> tuple[tuple[int, int], ...]:
    left, right = _unique(widths), _unique(heads)
    for width in left:
        integer(width, 1, 1024)
    for head in right:
        integer(head, 1, 16)
    return tuple((width, head) for width in left for head in right if width % head == 0)


def _validate_setting(path: str, value: Any, model_id: str) -> None:
    """Validate field domains independently; coupled attention is checked jointly."""
    root = {"model": asdict(model_defaults_v4(model_id)), "training": asdict(HPOTrainingConfig())}
    _set_path(root, path, value)
    # A width/head value can be valid although incompatible with default peers.
    if path in ("model.card.output_size", "model.hand.attention_heads"):
        root["model"]["hand"]["kind"] = "deepsets"
    if path in ("model.temporal.hidden_size", "model.temporal.num_heads"):
        root["model"]["temporal"]["num_heads"] = 1
        if path == "model.temporal.num_heads":
            root["model"]["temporal"]["num_heads"] = value
            root["model"]["temporal"]["hidden_size"] = integer(value, 1, 16) * 64
    if path == "model.event.max_episode":
        integer(value, 1, 2**31 - 1)
        root["model"]["event"].update(max_interval=1, aggregate_window=1)
    model_config_v4(root["model"], layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    hpo_training_config(root["training"])


def search_space(value: Json, *, model_id: str, model: ModelConfigV4 | None = None,
                 training: HPOTrainingConfig | None = None) -> SearchSpace:
    closed(value, {"parameters", "coupled"})
    allowed = searchable_paths(model_id)
    control = effective_settings(model_defaults_v4(model_id) if model is None else model,
                                 HPOTrainingConfig() if training is None else training, {},
                                 layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    if control.model.model_id != model_id:
        raise ContractError("search/control model kind mismatch")
    control_root = {"model": asdict(control.model), "training": asdict(control.training)}
    control_values = tuple((path, _freeze(_get_path(control_root, path))) for path in sorted(allowed))
    if type(value["parameters"]) is not list or type(value["coupled"]) is not list:
        raise ContractError("search parameters/coupled must be arrays")
    parameters: list[SearchParameter] = []
    couples: list[CoupledParameter] = []
    seen: set[str] = set()
    declared = {raw["path"] for raw in value["parameters"]
                if type(raw) is dict and type(raw.get("path")) is str}
    declared.update(path for raw in value["coupled"] if type(raw) is dict
                    and type(raw.get("paths")) is list for path in raw["paths"] if type(path) is str)
    available = allowed - declared
    for raw in value["parameters"]:
        closed(raw, {"path", "kind"}, {"choices", "low", "high", "step", "value", "when"})
        path, kind = raw["path"], raw["kind"]
        if type(path) is not str or path not in allowed or path in seen:
            raise ContractError("duplicate/unknown search path")
        if type(kind) is not str or kind not in ("categorical", "int", "float", "log", "fixed"):
            raise ContractError("unsupported search domain")
        default_root = {"model": asdict(model_defaults_v4(model_id)), "training": asdict(HPOTrainingConfig())}
        if type(_get_path(default_root, path)) is int and kind in ("float", "log"):
            raise ContractError("integer settings require integer or categorical domains")
        when = _conditions(raw.get("when", {}), available)
        for condition_path, condition_values in when:
            for condition_value in condition_values:
                _validate_setting(condition_path, condition_value, model_id)
        choices: tuple[Any, ...] = ()
        low = high = step = fixed = None
        keys = {"path", "kind"} | ({"when"} if "when" in raw else set())
        if kind == "categorical":
            keys.add("choices")
            choices = _unique(raw.get("choices"))
            samples = choices
        elif kind == "fixed":
            keys.add("value")
            if "value" not in raw:
                raise ContractError("fixed search setting requires value")
            fixed = _freeze(raw["value"])
            samples = (fixed,)
        else:
            keys.update(("low", "high"))
            low, high = raw.get("low"), raw.get("high")
            if kind == "int":
                integer(low, 0)
                integer(high, low)
                step = raw.get("step", 1)
                integer(step, 1)
            else:
                finite_number(low, -1e100, 1e100)
                finite_number(high, low, 1e100)
                if kind == "log" and low <= 0:
                    raise ContractError("log domain requires positive bounds")
                step = raw.get("step")
                if step is not None:
                    finite_number(step, 0, 1e100, strict_low=True)
                    if kind == "log":
                        raise ContractError("log domain cannot have step")
            if "step" in raw:
                keys.add("step")
            samples = (low, high)
            # Collection ratios remain invariant throughout an integer grid.
            if path == "training.collection_games" and (kind != "int" or low % 8 or step % 8):
                raise ContractError("collection search grid must preserve multiples of eight")
        if set(raw) != keys:
            raise ContractError("fields do not match search domain")
        for sample in samples:
            _validate_setting(path, sample, model_id)
        parameters.append(SearchParameter(path, kind, choices, low, high, step, fixed, when))
        seen.add(path)
        available.add(path)
    names: set[str] = set()
    for raw in value["coupled"]:
        closed(raw, {"name", "paths", "choices"}, {"when"})
        name, paths = raw["name"], raw["paths"]
        if (type(name) is not str or not name or name in names or name in allowed
                or type(paths) is not list or not paths or any(type(path) is not str for path in paths)
                or len(set(paths)) != len(paths) or set(paths) - allowed or set(paths) & seen):
            raise ContractError("invalid coupled search name/paths")
        when = _conditions(raw.get("when", {}), available)
        for condition_path, condition_values in when:
            for condition_value in condition_values:
                _validate_setting(condition_path, condition_value, model_id)
        choices = _unique(raw["choices"])
        for choice in choices:
            if type(choice) is not tuple or len(choice) != len(paths):
                raise ContractError("coupled choices must match declared paths")
            fields = dict(zip(paths, choice, strict=True))
            for path, item in fields.items():
                _validate_setting(path, item, model_id)
        couples.append(CoupledParameter(name, tuple(paths), choices, when))
        names.add(name)
        seen.update(paths)
        available.update(paths)
    result = SearchSpace(model_id, tuple(parameters), tuple(couples), control_values)
    _validate_reachable_domains(result)
    return result


MAX_VALIDATION_STATES = 65536


def _entry_paths(entry: SearchParameter | CoupledParameter) -> tuple[str, ...]:
    return (entry.path,) if isinstance(entry, SearchParameter) else entry.paths


def _representatives(parameter: SearchParameter, condition_values: tuple[Any, ...], *,
                     exact: bool) -> tuple[Any, ...]:
    """Cover numeric equality branches and their monotone extrema.

    Attention dimensions are small integer domains and enumerate exactly. Other
    numeric paths need endpoints plus every condition equality and its immediate
    grid/float neighbours: those include extrema of each remaining interval.
    """
    if parameter.kind == "categorical":
        return parameter.choices
    if parameter.kind == "fixed":
        return (parameter.value,)
    assert parameter.low is not None and parameter.high is not None
    low, high, step = parameter.low, parameter.high, parameter.step
    if parameter.kind == "int":
        assert step is not None
        high = low + ((high - low) // step) * step
        if exact:
            return tuple(range(int(low), int(high) + 1, int(step)))
    elif step is not None:
        count = (high - low) / step
        if not math.isfinite(count):
            raise ContractError("search step grid is not representable")
        high = min(high, low + math.floor(count + 1e-9) * step)
    candidates = [low, high]
    for item in condition_values:
        if type(item) not in (int, float):
            continue
        candidates.append(item)
        if step is not None:
            candidates.extend((item - step, item + step))
        else:
            candidates.extend((math.nextafter(item, -math.inf), math.nextafter(item, math.inf)))
    admitted: dict[str, Any] = {}
    for item in candidates:
        try:
            parameter.decode(item)
        except ContractError:
            continue
        admitted[canonical(item)] = item
    return tuple(admitted.values())


def _reachable_values(space: SearchSpace, targets: tuple[str, ...], *, exact: set[str]) -> Iterator[Json]:
    """Traverse only an invariant's transitive condition inputs, without mutation."""
    entries: tuple[SearchParameter | CoupledParameter, ...] = (*space.parameters, *space.coupled)
    needed = set(targets)
    previous: set[str] = set()
    while previous != needed:
        previous = set(needed)
        for entry in entries:
            if needed.intersection(_entry_paths(entry)):
                needed.update(path for path, _ in entry.when)
    relevant = tuple(entry for entry in entries if needed.intersection(_entry_paths(entry)))
    conditions: dict[str, tuple[Any, ...]] = {}
    for entry in relevant:
        for path, choices in entry.when:
            conditions[path] = (*conditions.get(path, ()), *choices)
    visits = 0

    def walk(index: int, values: Json) -> Iterator[Json]:
        nonlocal visits
        visits += 1
        if visits > MAX_VALIDATION_STATES:
            raise ContractError("search validation exceeds 65536 states; simplify conditions or use coupled choices")
        if index == len(relevant):
            yield values
            return
        entry = relevant[index]
        if not _active(entry.when, values):
            yield from walk(index + 1, values)
        elif isinstance(entry, SearchParameter):
            for item in _representatives(entry, conditions.get(entry.path, ()), exact=entry.path in exact):
                yield from walk(index + 1, values | {entry.path: item})
        else:
            for choice in entry.choices:
                yield from walk(index + 1, values | dict(zip(entry.paths, choice, strict=True)))

    yield from walk(0, dict(space.control_values))


def _validate_reachable_domains(space: SearchSpace) -> None:
    hand = ("model.hand.kind", "model.card.output_size", "model.hand.attention_heads")
    for values in _reachable_values(space, hand, exact=set(hand[1:])):
        if values[hand[0]] == "attention" and values[hand[1]] % values[hand[2]]:
            raise ContractError("reachable hand attention width must divide heads; use valid coupled choices")
    if space.model_id == "transformer":
        temporal = ("model.temporal.hidden_size", "model.temporal.num_heads")
        for values in _reachable_values(space, temporal, exact=set(temporal)):
            if values[temporal[0]] % values[temporal[1]]:
                raise ContractError("reachable temporal attention width must divide heads; use valid coupled choices")
    events = ("model.event.max_interval", "model.event.aggregate_window", "model.event.max_episode")
    for values in _reachable_values(space, events, exact=set()):
        if max(values[events[0]], values[events[1]]) > values[events[2]]:
            raise ContractError("reachable event limits exceed episode capacity; use valid coupled choices")
