"""Original v2 records share algorithms while preserving old serialized fields."""

import copy
import json
from dataclasses import FrozenInstanceError, asdict, fields, replace
from pathlib import Path

import pytest

from gf_common.card_instance import ActionConfigV3, card_instance_contract
from gf_common.card_instance_vocab import DEFINITION_SHA256
from gf_common.domain import ContractError, DecisionInput, canonical, digest
from gf_common.structured import (
    CandidateTable,
    DecisionFrame,
    EventBank,
    EventBankView,
    EventDelta,
    Selection,
    event_delta,
    layout_for_actions,
)
from gf_common.structured_layout import CARD_INSTANCE_LAYOUT as V2
from gf_common.structured_layout import LEGACY_LAYOUT, layout_for_model_identity, require_layout
from gf_common.structured_tokens import signature, tokenize_frame, tokenize_table
from gf_common.structured_wire import delta_dict, frame_dict, parse_delta, parse_frame, parse_table
from tests.test_card_instance_contract import fixture


def make_input(cards=None, *, events=(), revision=0):
    obs = fixture("runtime_observation")
    obs["state_revision"] = revision
    if cards is not None:
        obs["own"]["cards"] = [
            {
                "kind": "miracle" if state is not None else "artifact",
                "card_ref": "c1_" + format(index + 1, "032x"),
                "definition_id": name,
                **({"state": state} if state is not None else {}),
            }
            for index, (name, state) in enumerate(cards)
        ]
    catalog = {row["definition_id"]: row for row in fixture("catalog")["definitions"]}
    own_catalog = {
        card["definition_id"]: catalog[card["definition_id"]] for card in obs["own"]["cards"]
    }
    hints = fixture("runtime_options")
    hints["state_revision"] = revision
    hints["viewer_player_id"] = obs["viewer_player_id"]
    body = hints["action_options"]
    body["state_revision"] = revision
    body["usable_cards"] = [
        {"card_ref": card["card_ref"], "definition_id": card["definition_id"]}
        for card in obs["own"]["cards"]
    ]
    body["candidate_groups"] = {"uninterpreted-future-role": copy.deepcopy(body["usable_cards"])}
    for name in ("exchange_cards", "sell_cards", "buy_cards", "phenomenon_cards"):
        body[name] = []
    return DecisionInput(
        obs, tuple(events), None, tuple(obs["typed_action_schema"]), own_catalog, hints
    )


def frame_for(value=None, *, episode="card-instance-test"):
    value = make_input() if value is None else value
    bank = EventBank(1, episode, layout=V2)
    delta = event_delta(1, episode, 0, bank.boundary.prefix_hash, value.events, layout=V2)
    bank.append(delta)
    return DecisionFrame.from_input(value, bank, 0, layout=V2), bank, delta


def table_for(frame, actions, *, synthetic=False):
    return CandidateTable.from_actions(
        frame,
        actions,
        action_contract=card_instance_contract(),
        purpose="component",
        synthetic_capacity=synthetic,
    )


def test_layout_widths_closed_identity_and_actual_digest_inventory():
    from gf_common import structured_vocab as vocab

    public = fixture("catalog")["definitions"]
    assert tuple(sorted(row["definition_id"] for row in public)) == vocab.DEFINITIONS
    assert tuple(sorted({row["ability"] for row in public if "ability" in row})) == vocab.ABILITIES
    assert tuple(sorted({row["kind"] for row in public})) == vocab.KINDS
    assert (
        V2.state_width,
        V2.card_direct_width,
        V2.event_width,
        V2.history_width,
        V2.candidate_width,
    ) == (5499, 478, 395, 377, 26)
    assert layout_for_actions(ActionConfigV3(action_contract=card_instance_contract())) is V2
    assert layout_for_model_identity("3", V2.input_recipe, card_instance_contract()) is V2
    with pytest.raises(ContractError):
        layout_for_model_identity("2", V2.input_recipe, card_instance_contract())
    with pytest.raises(ContractError):
        require_layout(replace(V2, history_width=1))
    with pytest.raises(TypeError):
        DEFINITION_SHA256["bronze-club"] = "0" * 64
    assert dict(DEFINITION_SHA256) == {
        row["definition_id"]: digest(row) for row in fixture("catalog")["definitions"]
    }


def test_frozen_subclasses_fields_serialization_replace_and_delta_constructor():
    frame, bank, delta = frame_for()
    view = bank.view()
    for current, legacy_type in (
        (frame, DecisionFrame),
        (view, EventBankView),
        (delta, EventDelta),
    ):
        assert type(current) is not legacy_type
        assert [field.name for field in fields(current)] == [
            field.name for field in fields(legacy_type)
        ]
        assert set(current.__dataclass_fields__) == set(legacy_type.__dataclass_fields__)
        assert "layout" not in asdict(current)
        with pytest.raises(FrozenInstanceError):
            current.layout = LEGACY_LAYOUT
        with pytest.raises(FrozenInstanceError):
            current.extra_field = True
    assert replace(frame).layout is V2
    assert replace(view).layout is V2
    assert replace(frame) == frame and replace(view) == view
    assert view.prefix(view.boundary).layout is V2
    with pytest.raises(TypeError):
        replace(delta)
    with pytest.raises(TypeError):
        type(delta)(actor=1, episode="x", start=0, previous_hash="0" * 64, _events=())


def test_wire_original_roundtrip_requires_explicit_layout():
    frame, bank, delta = frame_for(make_input(events=({"type": "safe", "player": 1},)))
    raw_frame = frame_dict(frame, layout=V2)
    assert parse_frame(raw_frame, layout=V2) == frame
    assert parse_delta(delta_dict(delta, layout=V2), layout=V2) == delta
    for operation in (
        lambda: frame_dict(frame),
        lambda: delta_dict(delta),
        lambda: parse_frame(raw_frame),
        lambda: parse_delta(delta_dict(delta, layout=V2)),
    ):
        with pytest.raises(ContractError):
            operation()
    table = table_for(frame, [{"type": "pray"}])
    assert (
        parse_table(
            asdict(table),
            action_contract=card_instance_contract(),
            purpose="component",
            synthetic_capacity=False,
            layout=V2,
        )
        == table
    )
    with pytest.raises(ContractError):
        parse_table(
            asdict(table),
            action_contract=card_instance_contract(),
            purpose="component",
            synthetic_capacity=False,
        )
    old = EventBank(1, bank.episode)
    with pytest.raises(ContractError):
        old.append(delta)
    with pytest.raises(ContractError):
        bank.append(EventDelta(1, bank.episode, 1, bank.boundary.prefix_hash, []))


@pytest.mark.parametrize("learned", [7, 12])
def test_eighteen_original_slots_duplicate_learned_definitions_and_opponent_order(learned):
    value = make_input(
        [("bronze-club", None)] * (18 - learned) + [("fireball", "learned")] * learned
    )
    value.observation["opponent"]["learned_miracles"] = ["fireball", "aura", "fireball"]
    frame, bank, _ = frame_for(value)
    assert len(frame.references) == len(set(frame.references)) == 18
    assert all(type(ref) is str for ref in frame.references)
    action = {"type": "play_sequence", "ordered_card_refs": list(frame.references), "target": 2}
    table = table_for(frame, [action])
    assert table.to_action(0, frame) == action
    tokens = tokenize_frame(frame, bank.view(), layout=V2)
    assert tokens.opponent_learned[0] == tokens.opponent_learned[2] != tokens.opponent_learned[1]
    assert tokens.card_mask == (True,) * 18


def test_same_ref_unused_to_learned_reordering_and_immutable_previous_frame():
    value = make_input([("fireball", "unused"), ("bronze-club", None)])
    first, bank, _ = frame_for(value)
    previous_tokens = tokenize_frame(first, bank.view(), layout=V2)
    new = copy.deepcopy(value)
    new.observation["state_revision"] = 1
    new.observation["own"]["cards"][0]["state"] = "learned"
    new.observation["own"]["cards"].reverse()
    new.action_options["state_revision"] = new.action_options["action_options"][
        "state_revision"
    ] = 1
    event = {"type": "miracle_used", "player": 1, "definition_id": "fireball", "mp_spent": 0}
    bank.append(event_delta(1, bank.episode, 0, bank.boundary.prefix_hash, [event], layout=V2))
    new = replace(new, events=(event,))
    second = DecisionFrame.from_input(new, bank, 0, layout=V2)
    assert second.references == first.references[::-1]
    after = tokenize_frame(second, bank.view(), layout=V2)
    assert after.card_numerics[1][8] == 2
    assert previous_tokens == tokenize_frame(first, bank.view(), layout=V2)
    assert first.observation["own"]["cards"][0]["state"] == "unused"
    with pytest.raises(ContractError):
        DecisionFrame.from_input(new, bank, second.boundary.end, layout=V2)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_definition",
        "missing_definition",
        "attack",
        "classification",
        "kind",
        "stale_hints",
        "foreign_hints",
        "future_events",
    ],
)
def test_original_frame_rejects_tampered_subset_or_binding(mutation):
    value = make_input([("bronze-club", None), ("fireball", "learned")])
    if mutation == "extra_definition":
        value.catalog["aura"] = next(
            row for row in fixture("catalog")["definitions"] if row["definition_id"] == "aura"
        )
    elif mutation == "missing_definition":
        value.catalog.pop("fireball")
    elif mutation in ("attack", "classification"):
        value.catalog["bronze-club"][mutation] = (
            999 if mutation == "attack" else {"definition_class": "owned_card", "use_roles": []}
        )
    elif mutation == "kind":
        value.observation["own"]["cards"][0].update(kind="miracle", state="unused")
    elif mutation == "stale_hints":
        value.action_options["action_options"]["state_revision"] = 1
    elif mutation == "foreign_hints":
        value.action_options["action_options"]["usable_cards"][0]["card_ref"] = "c1_" + "ff" * 16
    else:
        value = replace(value, events=({"type": "safe", "player": 1},))
    bank = EventBank(1, "tampered", layout=V2)
    with pytest.raises(ContractError):
        DecisionFrame.from_input(value, bank, 0, layout=V2)


def test_event_prefix_owner_unknown_and_future_views():
    frame, bank, _ = frame_for(make_input(events=({"type": "safe", "player": 1},)))
    previous = frame.bank_view(bank.view())
    bank.append(
        event_delta(
            1,
            bank.episode,
            1,
            bank.boundary.prefix_hash,
            [{"type": "safe", "player": 2}],
            layout=V2,
        )
    )
    assert frame.bank_view(bank.view()) == previous
    with pytest.raises(ContractError):
        frame.bank_view(EventBank(2, bank.episode, layout=V2).view())
    with pytest.raises(ContractError):
        event_delta(
            1,
            "x",
            0,
            "0" * 64,
            [{"type": "artifact_used", "player": 1, "definition_id": "unknown"}],
            layout=V2,
        )
    with pytest.raises(ContractError):
        event_delta(1, "x", 0, "0" * 64, [{"type": "hidden_rng", "player": 1}], layout=V2)
    with pytest.raises(ContractError):
        replace(bank.view(), _prefixes=("0" * 64,) * 3)


@pytest.mark.parametrize(
    "kind",
    [
        "play_sequence",
        "respond",
        "discard",
        "sacrifice",
        "exchange",
        "sell",
        "buy",
        "buy_decision",
        "invoke_phenomenon",
        "pray",
    ],
)
def test_all_new_action_families_roundtrip_and_order(kind):
    value = make_input([("bronze-club", None), ("fireball", "learned")])
    if kind in ("respond", "buy_decision"):
        from gf_common.card_instance_validation import action_descriptors

        choice = "defense" if kind == "respond" else "buy_accept_reject"
        value.observation.update(
            required_choice=choice,
            typed_action_schema=action_descriptors(choice),
            phase="awaiting_response" if kind == "respond" else "resolving",
        )
        value = replace(value, typed_metadata=tuple(value.observation["typed_action_schema"]))
        value.action_options["action_options"]["required_choice"] = choice
    frame, _, _ = frame_for(value)
    a, b = frame.references
    extras = {
        "play_sequence": {"ordered_card_refs": [b, a], "target": 2},
        "respond": {"ordered_card_refs": [b, a]},
        "discard": {"card_refs": [b, a]},
        "sacrifice": {"card_refs": [a, b]},
        "exchange": {"card_ref": a, "hp": -1, "mp": 0, "money": 1},
        "sell": {"card_ref": a, "sold_card_ref": b, "target": 1},
        "buy": {"card_ref": b, "target": 2},
        "buy_decision": {"accept": False},
        "invoke_phenomenon": {"card_ref": a},
        "pray": {},
    }
    action = {"type": kind, **extras[kind]}
    table = table_for(frame, [action])
    assert table.to_action(0, frame) == action
    assert Selection(table.table_hash, 0, "pure-test").validate(table, frame) == action


def test_new_synthetic_capacity_never_emits_submit_ready_action_and_codec1_rejects(monkeypatch):
    from gf_common import candidate_columns

    frame, _, _ = frame_for(make_input([("fireball", "learned")]))
    action = {
        "type": "play_sequence",
        "ordered_card_refs": [frame.references[0]] * 128,
        "target": 2,
    }
    table = table_for(frame, [action], synthetic=True)
    table.validate(frame)
    assert len(table.slots) == 128
    for validate in (False, True):
        with pytest.raises(ContractError):
            table.to_action(0, frame, validate=validate)
    monkeypatch.setattr(
        candidate_columns,
        "column_layout",
        lambda *args, **kwargs: pytest.fail("codec1 must reject before admission/allocation"),
    )
    with pytest.raises(ContractError):
        candidate_columns.encode_candidate_columns(table, "new.bin")
    with pytest.raises(ContractError):
        candidate_columns.encode_candidate_columns(
            table_for(frame, [{"type": "pray"}]), "new-game.bin"
        )


def test_old_numeric_writer_rejects_new_contract_before_output_or_reservation(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from gf_models.structured_config import ModelConfigV2
    from gf_training import structured_rollout as writer

    frame, _, _ = frame_for()
    table = table_for(frame, [{"type": "pray"}])
    # Exercise the writer's defensive boundary independently of the already
    # stricter EpisodeRecords/StructuredRollout public constructors.
    control = SimpleNamespace(
        model=ModelConfigV2(), episodes=(SimpleNamespace(records=SimpleNamespace(tables=(table,))),)
    )
    monkeypatch.setattr(
        writer, "create_output_directory", lambda *args: pytest.fail("must reject before output")
    )
    with pytest.raises(ContractError, match="rollout3"):
        writer.write_rollout(tmp_path / "unsupported", control)
    assert not (tmp_path / "unsupported").exists()


def test_frozen_legacy_record_token_and_column_bytes():
    from gf_common.action_contract import engineering_contract
    from gf_common.candidate_columns import encode_candidate_columns
    from gf_common.structured_wire import parse_table

    path = Path(__file__).parent / "fixtures/card_instance_legacy_structured_golden.json"
    golden = json.loads(path.read_text())
    raw = golden["input"]
    value = DecisionInput(
        raw["observation"],
        tuple(raw["events"]),
        raw["history_end"],
        tuple(raw["typed_metadata"]),
        raw["catalog"],
        raw["action_options"],
    )
    bank = EventBank(1, "stage2-old-golden")
    delta = parse_delta(golden["delta"])
    bank.append(delta)
    frame = DecisionFrame.from_input(value, bank, 0)
    assert canonical({**asdict(frame), "payload": frame.payload.decode()}) == canonical(
        golden["frame"]
    )
    assert frame.frame_hash == golden["frame"]["frame_hash"]
    assert canonical(bank.view()._prefixes) == canonical(golden["bank_prefixes"])
    table = parse_table(
        golden["table"],
        action_contract=engineering_contract(),
        purpose="component",
        synthetic_capacity=False,
    )
    assert canonical(asdict(tokenize_frame(frame, bank.view()))) == canonical(
        golden["frame_tokens"]
    )
    assert canonical(asdict(tokenize_table(table, frame))) == canonical(golden["candidate_tokens"])
    payload, descriptor = encode_candidate_columns(table, "table.bin")
    assert payload.hex() == golden["column_hex"] and canonical(descriptor) == canonical(
        golden["columns"]
    )
    assert signature() == golden["signature"]
