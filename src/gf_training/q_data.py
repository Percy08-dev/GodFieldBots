"""Complete original-run ingestion and behavior-selected Monte Carlo returns."""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, DecisionInput, Json, canonical, digest
from gf_common.validation import validate_worker_input
from gf_training.journal import project_journals


@dataclass(frozen=True)
class Sample:
    value: DecisionInput
    action: Json
    target: float
    identity: tuple[str, str, int, int]


@dataclass(frozen=True)
class RunBundle:
    directory: Path
    manifest: Json
    journals: tuple[Path, ...]
    collection_seconds: float


def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("training ingestion deadline")


def load_run(directory: Path, expected_split: str | None = None, *, deadline: float = float("inf"), online_context: Any = None) -> RunBundle:
    check_deadline(deadline)
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") not in (3, 4) or type(manifest.get("schema_version")) is not int or manifest.get("status") != "complete":
        raise ContractError("training requires a finalized complete healthy run")
    if manifest["schema_version"] == 4:
        from gf_training.lineage import validate_collection_parent
        validate_collection_parent(manifest, directory, online_context=online_context, deadline=deadline)
    split = manifest.get("split")
    if split not in ("train", "validation") or expected_split is not None and split != expected_split:
        raise ContractError("source split is not permitted for this training role")
    config = manifest.get("effective_config", {})
    if digest(config) != manifest.get("config_sha256") or config.get("run", {}).get("split") != split:
        raise ContractError("run config provenance mismatch")
    elapsed = manifest.get("elapsed_seconds")
    budget = config.get("run", {}).get("wall_seconds")
    if type(elapsed) not in (int, float) or type(budget) not in (int, float) or not math.isfinite(elapsed) or not math.isfinite(budget) or not 0 <= elapsed <= budget:
        raise ContractError("source collection exceeded/misses original budget")
    schedule = [spec for group in manifest.get("schedule", []) for spec in group.get("episodes", [])]
    episodes = manifest.get("episodes", [])
    ids = [spec.get("episode_id") for spec in schedule]
    if not ids or len(set(ids)) != len(ids) or len(episodes) != len(ids) or {episode.get("episode_id") for episode in episodes} != set(ids):
        raise ContractError("complete original schedule required")
    journals = []
    for episode in episodes:
        check_deadline(deadline)
        if episode.get("end_kind") != "natural" or episode.get("outcome") is None or episode.get("failure") or any(episode.get("health", {}).values()) or episode.get("replay_reconciled") is not True:
            raise ContractError("all original episodes must be healthy natural outcomes")
        name = episode["episode_id"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ContractError("unsafe episode source path")
        for viewer in (1, 2):
            path = directory / name / f"player-{viewer}.jsonl"
            if not path.is_file():
                raise ContractError("all original participant journals required")
            journals.append(path)
    actual = set(directory.glob("*/player-*.jsonl"))
    if actual != set(journals):
        raise ContractError("missing/extra original participant journals")
    report = json.loads((directory / "report.json").read_text())
    if report.get("planned_games") != len(ids) or report.get("natural_denominator") != len(ids):
        raise ContractError("source report schedule count mismatch")
    return RunBundle(directory, manifest, tuple(journals), float(elapsed))


def validate_bundles(bundles: list[RunBundle], *, deadline: float = float("inf"), online_context: Any = None) -> tuple[list[Json], Json]:
    check_deadline(deadline)
    if len({bundle.manifest["run_id"] for bundle in bundles}) != len(bundles):
        raise ContractError("duplicate source run identity")
    rows, metadata = project_journals([path for bundle in bundles for path in bundle.journals], deadline=deadline, online_context=online_context)
    check_deadline(deadline)
    if not all(source["eligible"] for source in metadata["sources"]):
        raise ContractError("source bundle failed journal/report/replay/health verification")
    if any(not row["eligible"] for row in rows):
        raise ContractError("ineligible training row")
    return rows, metadata


def labeled_samples(rows: list[Json], metadata: Json, split: str, *, deadline: float = float("inf")) -> tuple[Sample, ...]:
    returns = {(source["header"]["run_id"], source["header"]["game_id"], source["header"]["viewer"]): source["end"]["terminal_reward"] for source in metadata["sources"]}
    authorized = {(source["header"]["run_id"], source["header"]["game_id"], source["header"]["viewer"]) for source in metadata["sources"] if source["lineage"].get("learner_viewer", source["header"]["viewer"]) == source["header"]["viewer"]}
    result = []
    for row in rows:
        check_deadline(deadline)
        if row["split"] != split or (row["run_id"], row["game_id"], row["viewer"]) not in authorized:
            continue
        if row["acknowledgement_json"] is None or not row["eligible"]:
            raise ContractError("unexecuted/ineligible decision has no training label")
        raw = json.loads(row["input_json"])
        value = validate_worker_input(DecisionInput(raw["observation"], tuple(raw["events"]), raw["history_end"],
                                                  tuple(raw["typed_metadata"]), raw["catalog"], raw["action_options"]))
        action = json.loads(row["command_json"])["action"]
        identity = (row["run_id"], row["game_id"], row["viewer"], row["decision"])
        target = returns[identity[:3]]
        if type(target) is not int or target not in (-1, 0, 1):
            raise ContractError("validated natural actor return required")
        result.append(Sample(value, action, float(target), identity))
    return tuple(result)


def provenance(bundle: RunBundle, *, deadline: float = float("inf")) -> Json:
    check_deadline(deadline)
    manifest_hash = hashlib.sha256()
    with (bundle.directory / "manifest.json").open("rb") as source:
        while chunk := source.read(1024 * 1024):
            check_deadline(deadline)
            manifest_hash.update(chunk)
    result = {"directory": str(bundle.directory), "run_id": bundle.manifest["run_id"],
              "manifest_sha256": manifest_hash.hexdigest(),
              "collection_seconds": bundle.collection_seconds, "split": bundle.manifest["split"],
              "config_sha256": digest(bundle.manifest["effective_config"]), "semantic_config": canonical(bundle.manifest["effective_config"]["actions"])}
    check_deadline(deadline)
    return result
