"""Rebuild explicitly synthetic native schema oracles for current defense rules.

Run with ``python -m tests.generate_defense_rule_fixtures``. Old fixture bytes
remain untouched. This tool cannot read or migrate actual parent/Source evidence.
All generated records remain unadmitted structural fixtures, never learning data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

from gf_common.card_instance import action_config_v3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import canonical, digest
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.card_instance_artifacts import validate_neural_metadata
from gf_models.card_instance_game_tree_runtime import validate_tree_metadata
from gf_models.structured_artifacts import semantics_v3
from gf_models.structured_binding import (
    input_binding_game_v3,
    input_binding_v3,
    policy_label_game_v3,
)
from gf_models.structured_config import InferenceBindingV2, model_config_v3
from gf_models.structured_tree_config import TreeEncoderConfigV3, configured_tree_v3
from gf_models.structured_tree_encoder import feature_layout_v3
from gf_training.card_instance_journal import read_card_instance_journal

ROOT = Path(__file__).parents[1] / "native/tests/fixtures"
RULES = card_instance_rules()


def load(name):
    return json.loads((ROOT / (name + ".json")).read_bytes())


def save(name, value):
    (ROOT / (name + "_rules_v2.json")).write_text(canonical(value) + "\n")


def file_hash(value):
    return hashlib.sha256((canonical(value) + "\n").encode()).hexdigest()


def neural(raw, actions):
    """Rebuild synthetic metadata, its manifest hash, options, inventory and label."""
    raw = copy.deepcopy(raw)
    manifest, options = raw["inference_manifest"], raw["policy_options"]
    model = model_config_v3(options["model"], layout=LAYOUT)
    manifest.update(semantics_v3(model, layout=LAYOUT, action_contract=actions.action_contract, rules=RULES))
    manifest.update(input_binding_game_v3(actions.action_contract, RULES))
    options.update(input_binding_game_v3(actions.action_contract, RULES), expected_manifest_digest=file_hash(manifest))
    for row in raw["files"]:
        if Path(row["path"]).name == "inference.json":
            row["sha256"] = options["expected_manifest_digest"]
    safe, label = validate_neural_metadata(raw, actions=actions, rules=RULES)
    return raw, safe, label


def tree_fixture():
    value = load("card_instance_tree3_raw_metadata")
    actions = action_config_v3(value["actions"])
    value["rules"] = asdict(RULES)
    raw = value["tree"]
    manifest, options = raw["inference_manifest"], raw["policy_options"]
    layout = feature_layout_v3(TreeEncoderConfigV3(**manifest["feature_layout"]["encoder"]))
    manifest.update(input_binding_game_v3(actions.action_contract, RULES))
    manifest.update(feature_layout=layout, feature_signature=layout["feature_signature"])
    from gf_models.structured_tree_pairing import TREE3_GAME_RECORDS

    model = configured_tree_v3({**options["model"], "checkpoint": options["checkpoint"],
        "inference": options["inference"], "resources": options["resources"]})
    manifest["architecture_identity"] = TREE3_GAME_RECORDS.identity(model.architecture())
    manifest["semantics"]["rule_signature"] = layout["rule_signature"]
    manifest["provenance"]["input_binding"] = input_binding_game_v3(actions.action_contract, RULES)
    options.update(feature_signature=layout["feature_signature"], expected_manifest_digest=file_hash(manifest))
    for row in raw["files"]:
        if Path(row["path"]).name == "inference.json":
            row["sha256"] = options["expected_manifest_digest"]
    _, value["policy_label"] = validate_tree_metadata(raw, actions=actions, rules=RULES)
    save("card_instance_tree3_raw_metadata", value)
    return value


def journal(case, *, tree=None):
    """Regenerate candidates/report using the original synthetic frame and RNG."""
    from gf_common.candidate_columns import encode_candidate_columns_v2
    from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV3
    from gf_common.structured import EventBank
    from gf_common.structured_tokens import tokenize_frame, tokenize_table
    from gf_common.structured_wire import parse_delta, parse_frame, report_dict_v3

    case = copy.deepcopy(case)
    header = case["rows"][0]
    actions = action_config_v3(header["actions"])
    header.update(input_binding_game_v3(actions.action_contract, RULES))
    if "neural" in header:
        header["neural"], _, header["policy"] = neural(header["neural"], actions)
    if tree is not None:
        header["tree"], header["policy"] = tree["tree"], tree["policy_label"]
    bank = EventBank(header["viewer"], header["bank_episode"], layout=LAYOUT)
    for row in case["rows"]:
        if row["kind"] == "events":
            bank.append(parse_delta(row["delta"], layout=LAYOUT))
        elif row["kind"] == "decision":
            frame = parse_frame(row["frame"], layout=LAYOUT)
            rng = random.Random()
            before = row["search_report"]["rng_before"]
            rng.setstate((before[0], tuple(before[1]), before[2]))
            generated = PublicCandidateGeneratorV3(actions, rules=RULES, clock=lambda: 0).generate_game(
                frame, bank.view(), rng, NumericCosts(**header["numeric_costs"]))
            payload, descriptor = encode_candidate_columns_v2(generated.table, row["table"]["file"], layout=LAYOUT)
            selected = 0
            row["command"]["action"] = generated.table.to_action(selected, frame)
            row["table"] = descriptor
            case["bytes_hex"] = payload.hex()
            row["search_report"] = report_dict_v3(generated.report, action_contract=actions.action_contract, rules=RULES)
            row["selection"]["policy_version"] = header["policy"]
            row["selection"].update(index=selected, table_hash=generated.table.table_hash)
            if "frame_tokens" in case:
                case["frame_tokens"] = asdict(tokenize_frame(frame, bank.view(), aggregate_window=header["aggregate_window"], layout=LAYOUT))
                case["candidate_tokens"] = asdict(tokenize_table(generated.table, frame, aggregate_window=header["aggregate_window"], layout=LAYOUT))
    with tempfile.TemporaryDirectory(prefix="gfb-defense-synthetic-") as directory:
        path = Path(directory) / "player-1.jsonl"
        path.write_text("".join(canonical(row) + "\n" for row in case["rows"]))
        for row in case["rows"]:
            if row["kind"] == "decision":
                (path.parent / row["table"]["file"]).write_bytes(bytes.fromhex(case["bytes_hex"]))
        read_card_instance_journal(path, actions=actions)
    return case


def main():
    models = load("card_instance_models3")
    for case in models["cases"]:
        actions = action_config_v3(case["actions"])
        case["neural"], case["options"], case["policy_label"] = neural(case["neural"], actions)
        case["manifest"] = case["neural"]["inference_manifest"]
        case["cuda_options"].update(input_binding_game_v3(actions.action_contract, RULES),
            expected_manifest_digest=case["options"]["expected_manifest_digest"])
        case["cuda_label"] = policy_label_game_v3(model_config_v3(case["model"], layout=LAYOUT),
            InferenceBindingV2(**case["cuda_options"]["inference"]), case["options"]["expected_manifest_digest"],
            action_contract=actions.action_contract, rules=RULES, action_config_hash=digest(asdict(actions)),
            cuda_visible_devices=case["cuda_options"]["cuda_visible_devices"])
    save("card_instance_models3", models)
    tree = tree_fixture()
    for name, key in (("card_instance_journal6", "journals"),
                      ("card_instance_journal6_rewards", "cases"),
                      ("card_instance_tree3_raw_journal", "journals")):
        value = load(name)
        value["scope"] += "; regenerated current rules-v2 synthetic candidates/metadata; no historical acceptance"
        value[key] = [journal(case, tree=tree if "tree3" in name else None) for case in value[key]]
        save(name, value)
    layouts = []
    for case in load("card_instance_tree3_layout_python"):
        layout = feature_layout_v3(TreeEncoderConfigV3(**case["encoder"]))
        layouts.append({**case, "feature_signature": layout["feature_signature"], "sha256": digest(layout)})
    save("card_instance_tree3_layout_python", layouts)

    from gf_models.structured_artifacts import _game_context_v4
    from gf_training.hpo_lineage import hpo_state

    hpo = load("hpo_metadata5")
    for case in hpo["cases"]:
        state = case["state"]
        actions = action_config_v3(state["effective_config"]["actions"])
        classification = input_binding_game_v3(actions.action_contract, RULES)
        state.update(classification)
        state["identity"]["preprocessing"].update(classification)
        state["resume_identity"] = digest({"version": "hpo-ppo-resume-v1", "config_sha256": state["config_sha256"],
            "identity": state["identity"], "execution": state["execution"], "classification": classification,
            "storage_binding": state.get("storage_binding")})
        _, model, _ = hpo_state(state)
        semantics = _game_context_v4(actions.action_contract, RULES, digest(asdict(actions))).semantics(model)
        for key in ("inference", "resume"):
            case[key].update(classification)
            case[key].update(semantics)
        case["resume"]["resume_identity"] = state["resume_identity"]
        case["resume"]["inference_digest"] = file_hash(case["inference"])
        generation = case["generation"]
        generation.update(classification)
        generation["inference_sha256"] = file_hash(case["inference"])
        generation["resume_sha256"] = file_hash(case["resume"])
        generation["artifact_files"].update({"inference.json": generation["inference_sha256"], "resume.json": generation["resume_sha256"]})
        state["generation_sha256"] = file_hash(generation)
        for entry in case["registry"]["generations"].values():
            for name, metadata in entry["files"].items():
                metadata["sha256"] = generation["artifact_files"][name]
    save("hpo_metadata5", hpo)

    # Pure token DTO fixture: its independently checked numeric arrays and
    # syntax-only candidate columns are invariant; only derived output binding
    # metadata changes. This fixture confers no candidate/Source authority.
    tokens = load("card_instance_component_tokens_python")
    for case in tokens["cases"]:
        case["metadata"]["input_binding"] = input_binding_v3(LAYOUT.action_contract, RULES, "component", False)
    tokens["scope"] += "; current rules-v2 output binding; numerical recipes unchanged"
    save("card_instance_component_tokens_python", tokens)


if __name__ == "__main__":
    main()
