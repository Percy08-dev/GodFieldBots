"""Closed protocol2 public DTO conversions. Pure owned records, no transport I/O."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from gf_common.action_contract import ActionContract, parse_action_contract
from gf_common.card_instance import CardInstanceContract, parse_card_instance_contract
from gf_common.card_instance_rules import CardInstanceRules, rule_signature
from gf_common.diverse_actions import (
    GameGeneratedCandidatesV3,
    GeneratedCandidates,
    GeneratedCandidatesV3,
    NumericCosts,
    PrecountReport,
    SearchReport,
    SearchReportV3,
)
from gf_common.domain import ContractError, Json, canonical, closed
from gf_common.structured import (
    CandidateTable,
    DecisionFrame,
    EventBoundary,
    EventDelta,
    Selection,
    _new_frame,
    _raw_action_contract,
    _validate_purpose,
    _VerifiedJournalFrame,
    event_delta,
)
from gf_common.structured_layout import (
    LEGACY_LAYOUT,
    StructuredLayout,
    layout_for_contract,
    require_layout,
)


def frame_dict(
    frame: DecisionFrame, *, layout: StructuredLayout = LEGACY_LAYOUT,
    action_contract: ActionContract | CardInstanceContract | None = None,
) -> Json:
    return _frame_dict(frame, layout=layout, action_contract=action_contract)


def _frame_dict(
    frame: DecisionFrame | _VerifiedJournalFrame, *, layout: StructuredLayout = LEGACY_LAYOUT,
    action_contract: ActionContract | CardInstanceContract | None = None,
) -> Json:
    if frame.layout is not layout or frame.raw_action_contract != _raw_action_contract(layout, action_contract):
        raise ContractError("frame wire layout/action contract mismatch")
    return {"actor": frame.actor, "episode": frame.episode, "revision": frame.revision,
            "interval_start": frame.interval_start, "boundary": asdict(frame.boundary),
            "interval_start_hash": frame.interval_start_hash, "payload": frame.raw, "frame_hash": frame.frame_hash}


def parse_frame(
    raw: Json, *, layout: StructuredLayout = LEGACY_LAYOUT,
    action_contract: ActionContract | CardInstanceContract | None = None,
) -> DecisionFrame:
    expected = _raw_action_contract(layout, action_contract)
    closed(raw, {"actor", "episode", "revision", "interval_start", "boundary", "interval_start_hash", "payload", "frame_hash"})
    boundary = EventBoundary(**closed(raw["boundary"], {"end", "prefix_hash"}))
    return _new_frame(raw["actor"], raw["episode"], raw["revision"], raw["interval_start"], boundary,
                      raw["interval_start_hash"], canonical(raw["payload"]).encode(), raw["frame_hash"],
                      layout=layout, action_contract=expected)


def delta_dict(
    delta: EventDelta, *, layout: StructuredLayout = LEGACY_LAYOUT,
    action_contract: ActionContract | CardInstanceContract | None = None,
) -> Json:
    if delta.layout is not layout or delta.raw_action_contract != _raw_action_contract(layout, action_contract):
        raise ContractError("delta wire layout/action contract mismatch")
    return {"actor": delta.actor, "episode": delta.episode, "start": delta.start,
            "previous_hash": delta.previous_hash, "events": delta.events}


def parse_delta(
    raw: Json, *, layout: StructuredLayout = LEGACY_LAYOUT,
    action_contract: ActionContract | CardInstanceContract | None = None,
) -> EventDelta:
    closed(raw, {"actor", "episode", "start", "previous_hash", "events"})
    return event_delta(**raw, layout=layout, action_contract=action_contract)


def parse_table(raw: Json, *, action_contract: ActionContract | CardInstanceContract, purpose: str,
                synthetic_capacity: bool, layout: StructuredLayout = LEGACY_LAYOUT) -> CandidateTable:
    _validate_purpose(purpose, action_contract)
    if layout_for_contract(action_contract) is not require_layout(layout):
        raise ContractError("candidate wire layout mismatch")
    closed(raw, set(CandidateTable.__dataclass_fields__))
    for key in ("kinds", "targets", "numerics", "presence", "offsets", "slots"):
        if not isinstance(raw[key], (list, tuple)):
            raise ContractError("compact table column arrays required")
    values = dict(raw)
    if layout is LEGACY_LAYOUT:
        assert type(action_contract) is ActionContract
        values["action_contract"] = parse_action_contract(raw["action_contract"], expected=action_contract)
    else:
        values["action_contract"] = parse_card_instance_contract(raw["action_contract"])
        if values["action_contract"] != action_contract:
            raise ContractError("candidate wire action contract mismatch")
    if (raw["purpose"] != purpose or type(synthetic_capacity) is not bool
            or type(raw["synthetic_capacity"]) is not bool or raw["synthetic_capacity"] != synthetic_capacity):
        raise ContractError("wire candidate purpose/synthetic classification mismatch")
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


def parse_report(raw: Json, *, action_contract: ActionContract) -> SearchReport:
    if type(action_contract) is not ActionContract:
        raise ContractError("expected Config action contract required")
    closed(raw, set(SearchReport.__dataclass_fields__))
    costs = NumericCosts(**closed(raw["numeric_costs"], set(NumericCosts.__dataclass_fields__)))
    count = PrecountReport(**closed(raw["precount"], set(PrecountReport.__dataclass_fields__)))
    values = dict(raw)
    values["action_contract"] = parse_action_contract(raw["action_contract"], expected=action_contract)
    values.update(numeric_costs=costs, precount=count)
    for key in ("channel_counts", "cutoff_reasons", "omissions"):
        if not isinstance(values[key], (tuple, list)):
            raise ContractError("closed report array required")
        values[key] = tuple(values[key])
    for key in ("rng_before", "rng_after"):
        values[key] = _state(values[key])
    return SearchReport(**values)


def generated_dict(value: GeneratedCandidates) -> Json:
    if type(value) is not GeneratedCandidates or type(value.report) is not SearchReport or value.table.layout is not LEGACY_LAYOUT:
        raise ContractError("protocol2 generated candidates require legacy layout")
    return {"table": asdict(value.table), "report": asdict(value.report)}


def parse_generated(raw: Json, *, action_contract: ActionContract) -> GeneratedCandidates:
    closed(raw, {"table", "report"})
    return GeneratedCandidates(parse_table(raw["table"], action_contract=action_contract, purpose="component", synthetic_capacity=False),
                               parse_report(raw["report"], action_contract=action_contract))


def parse_selection(raw: Json) -> Selection:
    return Selection(**closed(raw, set(Selection.__dataclass_fields__)))


def report_dict_v3(report: SearchReportV3, *, action_contract: CardInstanceContract,
                   rules: CardInstanceRules) -> Json:
    expected = rule_signature(action_contract, rules)
    if (type(report) is not SearchReportV3 or report.action_contract != action_contract
            or report.rule_signature != expected):
        raise ContractError("expected explicit card-instance report/rule binding")
    return asdict(report)


def parse_report_v3(raw: Json, *, action_contract: CardInstanceContract,
                    rules: CardInstanceRules) -> SearchReportV3:
    expected = rule_signature(action_contract, rules)
    closed(raw, set(SearchReportV3.__dataclass_fields__))
    if raw["rule_signature"] != expected:
        raise ContractError("wire search rule signature mismatch")
    contract = parse_card_instance_contract(raw["action_contract"])
    if contract != action_contract:
        raise ContractError("wire report action contract mismatch")
    costs = NumericCosts(**closed(raw["numeric_costs"], set(NumericCosts.__dataclass_fields__)))
    count = PrecountReport(**closed(raw["precount"], set(PrecountReport.__dataclass_fields__)))
    values = {**raw, "action_contract": contract, "numeric_costs": costs, "precount": count}
    for key in ("channel_counts", "cutoff_reasons", "omissions"):
        if not isinstance(values[key], (tuple, list)):
            raise ContractError("closed report array required")
        values[key] = tuple(values[key])
    for key in ("rng_before", "rng_after"):
        values[key] = _state(values[key])
    return SearchReportV3(**values)


def generated_dict_v3(value: GeneratedCandidatesV3, *, action_contract: CardInstanceContract,
                      rules: CardInstanceRules) -> Json:
    if type(value) is not GeneratedCandidatesV3 or value.table.layout is not layout_for_contract(action_contract):
        raise ContractError("explicit card-instance generated candidates required")
    report = report_dict_v3(value.report, action_contract=action_contract, rules=rules)
    if value.table.action_contract != action_contract:
        raise ContractError("generated table action contract mismatch")
    return {"table": asdict(value.table), "report": report}


def parse_generated_v3(raw: Json, *, action_contract: CardInstanceContract,
                       rules: CardInstanceRules) -> GeneratedCandidatesV3:
    closed(raw, {"table", "report"})
    report = parse_report_v3(raw["report"], action_contract=action_contract, rules=rules)
    table = parse_table(raw["table"], action_contract=action_contract, purpose="component",
                        synthetic_capacity=False, layout=layout_for_contract(action_contract))
    return GeneratedCandidatesV3(table, report)


def game_generated_dict_v3(value: GameGeneratedCandidatesV3, *, action_contract: CardInstanceContract,
                           rules: CardInstanceRules) -> Json:
    if type(value) is not GameGeneratedCandidatesV3 or value.table.layout is not layout_for_contract(action_contract):
        raise ContractError("explicit corrected game candidates required")
    if value.table.action_contract != action_contract:
        raise ContractError("game table contract mismatch")
    return {"table": asdict(value.table), "report": report_dict_v3(value.report, action_contract=action_contract, rules=rules)}


def parse_game_generated_v3(raw: Json, *, action_contract: CardInstanceContract,
                            rules: CardInstanceRules) -> GameGeneratedCandidatesV3:
    closed(raw, {"table", "report"})
    return GameGeneratedCandidatesV3(
        parse_table(raw["table"], action_contract=action_contract, purpose="qualified_game", synthetic_capacity=False,
                    layout=layout_for_contract(action_contract)),
        parse_report_v3(raw["report"], action_contract=action_contract, rules=rules))
