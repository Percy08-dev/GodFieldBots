"""Closed protocol2 public DTO conversions. Pure owned records, no transport I/O."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from gf_common.diverse_actions import (
    GeneratedCandidates,
    NumericCosts,
    PrecountReport,
    SearchReport,
)
from gf_common.domain import ContractError, Json, canonical, closed
from gf_common.structured import CandidateTable, DecisionFrame, EventBoundary, EventDelta, Selection


def frame_dict(frame: DecisionFrame) -> Json:
    return {"actor": frame.actor, "episode": frame.episode, "revision": frame.revision,
            "interval_start": frame.interval_start, "boundary": asdict(frame.boundary),
            "interval_start_hash": frame.interval_start_hash, "payload": frame.raw, "frame_hash": frame.frame_hash}


def parse_frame(raw: Json) -> DecisionFrame:
    closed(raw, {"actor", "episode", "revision", "interval_start", "boundary", "interval_start_hash", "payload", "frame_hash"})
    boundary = EventBoundary(**closed(raw["boundary"], {"end", "prefix_hash"}))
    return DecisionFrame(raw["actor"], raw["episode"], raw["revision"], raw["interval_start"], boundary,
                         raw["interval_start_hash"], canonical(raw["payload"]).encode(), raw["frame_hash"])


def delta_dict(delta: EventDelta) -> Json:
    return {"actor": delta.actor, "episode": delta.episode, "start": delta.start,
            "previous_hash": delta.previous_hash, "events": delta.events}


def parse_delta(raw: Json) -> EventDelta:
    closed(raw, {"actor", "episode", "start", "previous_hash", "events"})
    return EventDelta(**raw)


def parse_table(raw: Json) -> CandidateTable:
    closed(raw, set(CandidateTable.__dataclass_fields__))
    for key in ("kinds", "targets", "numerics", "presence", "offsets", "slots"):
        if not isinstance(raw[key], (list, tuple)):
            raise ContractError("compact table column arrays required")
    values = dict(raw)
    for key in ("kinds", "targets", "offsets", "slots"):
        values[key] = tuple(values[key])
    for key in ("numerics", "presence"):
        if any(not isinstance(row, (tuple, list)) for row in values[key]):
            raise ContractError("compact numeric rows required")
        values[key] = tuple(tuple(row) for row in values[key])
    return CandidateTable(**values)


def _state(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not isinstance(value[1], (list, tuple)):
        raise ContractError("closed search RNG state required")
    return value[0], tuple(value[1]), value[2]


def parse_report(raw: Json) -> SearchReport:
    closed(raw, set(SearchReport.__dataclass_fields__))
    costs = NumericCosts(**closed(raw["numeric_costs"], set(NumericCosts.__dataclass_fields__)))
    count = PrecountReport(**closed(raw["precount"], set(PrecountReport.__dataclass_fields__)))
    values = dict(raw)
    values.update(numeric_costs=costs, precount=count)
    for key in ("channel_counts", "cutoff_reasons", "omissions"):
        if not isinstance(values[key], (tuple, list)):
            raise ContractError("closed report array required")
        values[key] = tuple(values[key])
    for key in ("rng_before", "rng_after"):
        values[key] = _state(values[key])
    return SearchReport(**values)


def generated_dict(value: GeneratedCandidates) -> Json:
    return {"table": asdict(value.table), "report": asdict(value.report)}


def parse_generated(raw: Json) -> GeneratedCandidates:
    closed(raw, {"table", "report"})
    return GeneratedCandidates(parse_table(raw["table"]), parse_report(raw["report"]))


def parse_selection(raw: Json) -> Selection:
    return Selection(**closed(raw, set(Selection.__dataclass_fields__)))
