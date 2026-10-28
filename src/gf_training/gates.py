"""Recomputed original feasibility proof; an editable success flag is insufficient."""
from __future__ import annotations

from pathlib import Path

from gf_common.domain import ContractError, Json, canonical
from gf_common.schedule import schedule, serialize_schedule
from gf_training.q_data import RunBundle, check_deadline, load_run, validate_bundles


def feasibility(pilot: Path, *, deadline: float = float("inf")) -> tuple[RunBundle, Json]:
    bundle = load_run(pilot, deadline=deadline)
    config = bundle.manifest["effective_config"]
    run = config["run"]
    sampling = run.get("sampling_seeds", [])
    if len(sampling) != 3 or len(set(sampling)) != 3:
        raise ContractError("pilot requires three distinct policy sampling seeds")
    if len(run.get("seeds", [])) * run.get("pairs_per_seed", 0) * 2 != 64:
        raise ContractError("pilot requires exactly 64 games per sampling seed")
    bindings = bundle.manifest.get("policy_bindings", {})
    def label(role: str) -> str:
        model = config[role]
        if model["model_id"] in ("random", "heuristic"):
            return model["model_id"]
        matches = [name for name, binding in bindings.items() if binding.get("model_id") == model["model_id"] and binding.get("policy_options", {}).get("checkpoint") == model.get("checkpoint")]
        if len(matches) != 1:
            raise ContractError("pilot policy binding is ambiguous")
        return matches[0]
    expected = schedule(tuple(run["seeds"]), tuple(sampling), run["pairs_per_seed"], label("model"), label("opponent"), deadline=deadline)
    if canonical(serialize_schedule(expected, 3)) != canonical(bundle.manifest["schedule"]):
        raise ContractError("pilot fixed schedule mismatch")
    if len(bundle.manifest["episodes"]) != 192:
        raise ContractError("pilot requires all 192 original games")
    _, evidence = validate_bundles([bundle], deadline=deadline)
    hashes = {source["header"]["action_config_hash"] for source in evidence["sources"]}
    if len(hashes) != 1:
        raise ContractError("pilot action conditions must match")
    check_deadline(deadline)
    return bundle, {"passed": True, "games": 192, "games_per_sampling_seed": 64,
                    "sampling_seeds": sampling, "source_set_sha256": evidence["source_set_sha256"],
                    "action_config_hash": next(iter(hashes)), "split": bundle.manifest["split"]}
