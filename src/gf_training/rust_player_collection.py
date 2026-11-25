"""Concrete native game/real writer collection, independent of PPO numerics."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.diverse_actions import NumericCosts
from gf_common.domain import ContractError, Json, digest, group_health_failures
from gf_common.hpo_progress import ProgressCallback
from gf_common.hpo_storage import (
    check_collection_destinations,
    check_staged_collection_destinations,
)
from gf_common.rust_player import LEARNED_POLICY_KINDS, SEQUENCE_MODEL_KINDS, deadline_seconds
from gf_common.rust_ppo_source import source_encoder, source_mode
from gf_common.schedule import serialize_schedule
from gf_models.structured_runtime import own_game_worker_settings_v3
from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
from gf_simulator_client.rust_player import RustPlayerRuntime
from gf_training.rust_player_lineage import validate_record_shape
from gf_training.rust_player_matches import RustMatchPump
from gf_training.rust_player_registry import RustRunRegistry
from gf_training.rust_player_source import RustPlayerOriginals
from gf_training.rust_player_storage import (
    EpisodeEncodingPlan,
    PolicyJournalContext,
    RecordingArtifacts,
    RecordingBudget,
    RustPlayerStorage,
    StorageGroup,
    WriterTrainingEpisode,
)

_ISSUER = object()


class RustCollectedBatch:
    def __init__(self, issuer: object, collector: RustPlayerCollector, plan: Any,
                 writer: RustPlayerStorage, pending: Any, manifest_sha256: str,
                 capture_sha256: str, retirement_sha256: str, *,
                 training: tuple[WriterTrainingEpisode, ...] = (), pack_verification: Json | None = None,
                 manifest: Json | None = None, report: Json | None = None, metadata_lease: Any = None) -> None:
        if issuer is not _ISSUER:
            raise ContractError("actual native/writer collection owner required")
        self.collector, self.registry, self.plan = collector, collector.registry, plan
        self.writer, self.pending = writer, pending
        self.directory, self.manifest_sha256 = plan.output, manifest_sha256
        self.capture_sha256, self.retirement_sha256 = capture_sha256, retirement_sha256
        self.originals: RustPlayerOriginals | None = None
        self.training = training
        self.pack_verification = pack_verification
        self._manifest, self._report, self._metadata_lease = manifest, report, metadata_lease
        self._closed = False
        self._cleanup_children: list[Any] = []

    def require(self, registry: RustRunRegistry, plan: Any) -> None:
        registry.require()
        if (self._closed or self.registry is not registry or self.plan is not plan
                or self.collector.current is not self or self.writer.runtime is not self.collector.runtime
                or not self.writer._sealed or self.writer._thread.is_alive()
                or self.writer.cleanup_errors):
            raise ContractError("collection differs from SAME actual registry/runtime/writer/plan")
        self.pending.recheck()

    def open(self) -> RustPlayerOriginals:
        self.require(self.registry, self.plan)
        if self.pack_verification is not None:
            raise ContractError("live native training collection has no decoded Pack owner")
        if self.originals is None:
            try:
                self.originals = RustPlayerOriginals(self.pending, pool=self.registry.pool,
                    settings=self.registry.config.preprocessing, deadline=self.plan.execution_deadline)
            except BaseException as error:
                from gf_training.group_pack_resources import retain_cleanup
                retain_cleanup(self, error)
                raise
        return self.originals

    @property
    def manifest(self) -> Json:
        if self.pack_verification is not None:
            self.require(self.registry, self.plan)
            assert self._manifest is not None
            return self._manifest
        return self.open().object(self.open().root / "manifest.json")

    @property
    def report(self) -> Json:
        if self.pack_verification is not None:
            self.require(self.registry, self.plan)
            assert self._report is not None
            return self._report
        return self.open().object(self.open().root / "report.json")

    def close(self) -> None:
        if self._closed:
            return
        from gf_training.group_pack_resources import close_resources
        close_resources(self, [self.originals, self.pending])
        self._manifest = self._report = None
        for episode in self.training:
            episode.close()
        self.training = ()
        if self._metadata_lease is not None:
            self._metadata_lease.close()
            self._metadata_lease = None
        self._closed = True


class RustPlayerCollector:
    def __init__(self, registry: RustRunRegistry) -> None:
        if type(registry) is not RustRunRegistry:
            raise ContractError("actual Rust run registry required")
        registry.require()
        self.registry, self.config = registry, registry.config
        self.declaration = registry.state["identity"]["player_runtime"]["runtime"]
        self.mode = source_mode(registry.state["effective_config"])
        self.settings = self.config.storage.settings()
        self.cleanup_deadline = deadline_seconds(registry.admission.observation["deadlines"]["term_bits"])
        numeric_limit = ({"training_numeric_bytes": self.config.hpo_spec["operating"]["numeric_budget_bytes"]}
            if self.mode is not None and registry.hpo else {})
        self.runtime = RustPlayerRuntime({key: self.declaration[key] for key in (
            "scheduler", "player_worker_threads", "game_concurrency", "max_batch_size", "formation_wait_ms")} |
            {"storage_bytes": self.settings.budget_bytes,
             "storage_queue": self.declaration["game_concurrency"],
             "runtime_identity": registry.service.runtime} | numeric_limit, service=registry.service)
        self.current: RustCollectedBatch | None = None
        self._active_writer: RustPlayerStorage | None = None
        self._inventory: Any = None
        self._controls: RecordingArtifacts | None = None
        self._matches = RustMatchPump(self.runtime)
        self._servers = self._matches.servers
        self.server_history = self._matches.server_history
        self.writer_history: list[Json] = []
        self._source_control_charge: Any = None
        self._work: tuple[Path, int, int] | None = None

    def _create_work(self, root: Path, device: int) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="collection-", dir=root))
        info = work.lstat()
        self._work = work, info.st_dev, info.st_ino
        self._active_writer = None
        self._controls = None
        if info.st_dev != device:
            raise ContractError("collection workspace moved off the admitted tmpfs")
        return work

    def retire_work(self, failure: BaseException | None = None) -> None:
        if self._work is None:
            return
        writer = self._active_writer
        if (writer is not None and not writer.staging_input_retired()
                or self._matches._pending_http
                or any(not server.stopped or any(not task.done() for task in server.drainers)
                       for server in self._servers.values())):
            raise ContractError("collection tmpfs still has original activity")
        if failure is not None:
            traceback.clear_frames(failure.__traceback__)
            if writer is not None:
                writer.close_unclaimed_training()
            if self._controls is not None:
                self._controls.close()
            if self._inventory is not None:
                self._inventory.close()
                self._inventory = None
            if self._source_control_charge is not None:
                self._source_control_charge.close()
                self._source_control_charge = None
        directory, device, inode = self._work
        current = directory.lstat()
        if (current.st_dev, current.st_ino) != (device, inode):
            raise ContractError("collection tmpfs workspace identity changed")
        shutil.rmtree(directory)
        self._work = None

    def _counters(self) -> Json:
        calls = self.registry.service.calls
        return {"requests": calls["rows"], "publications": self.registry.service.registry.publications,
            "accepted": calls["accepted"], "expired": calls["expired"]}

    def _contexts(self, plan: Any) -> dict[str, Json]:
        inputs = self.registry.journal_inputs(plan.bindings)
        contexts = {}
        for group in plan.groups:
            for episode in group.episodes:
                policies = []
                for viewer, label in enumerate(episode.policies, 1):
                    binding = plan.bindings[label]
                    caps, costs, _, _ = own_game_worker_settings_v3(binding.model_id,
                        self.config.actions, binding.policy_options, rules=self.registry.local.learner._context.rules)
                    native: Json = {"kind": "refresh" if binding.model_id == "heuristic_refresh" else binding.model_id,
                        "seed": episode.sampling_seeds[viewer - 1], "checkpoint": None, "epoch": 0,
                        "limits": None, "attack_threshold": 12, "numeric_costs": asdict(costs)}
                    journal = {"policy": label, "model_id": binding.model_id, "caps": asdict(caps),
                        "numeric_costs": asdict(costs), "sampling_seed": episode.sampling_seeds[viewer - 1]}
                    if binding.model_id in LEARNED_POLICY_KINDS:
                        generation = int(Path(binding.policy_options["checkpoint"]).name)
                        policy = self.registry.policies[generation]
                        native.update(checkpoint=policy.files["inference.json"], epoch=generation,
                            limits={"max_decisions": policy.model.temporal.max_decisions,
                                "max_episode_events": policy.model.event.max_episode,
                                "max_interval_events": policy.model.event.max_interval,
                                "max_candidates": policy.binding.max_candidates,
                                "max_positions": policy.model.max_positions,
                                "aggregate_window": policy.model.event.aggregate_window})
                        journal["neural"] = inputs["policies"][label]
                    elif binding.model_id == "heuristic_refresh":
                        native["attack_threshold"] = binding.policy_options["attack_threshold"]
                        journal["baseline_options"] = binding.policy_options
                    policies.append(PolicyJournalContext(native, journal))
                contexts[episode.episode_id] = {"run_id": self.registry.state["run_id"],
                    "trial": plan.linkage["batch_id"], "episode_id": episode.episode_id,
                    "actions": self.config.actions, "rules": self.registry.local.learner._context.rules,
                    "numeric_costs": asdict(NumericCosts()), "policies": (policies[0], policies[1]),
                    "deadline": plan.execution_deadline, "clock": None}
                if self.mode is not None:
                    viewer = episode.learner_viewer
                    if viewer != plan.linkage['learner_seats'][episode.episode_id] or viewer not in (1, 2):
                        raise ContractError("original plan requires explicit matching learner seat")
                    learner = policies[viewer - 1].native
                    if learner['kind'] != 'mlp' or learner['epoch'] != plan.linkage['collection_generation']:
                        raise ContractError("learner seat requires the actual parent checkpoint")
                    contexts[episode.episode_id]['learner_viewer'] = viewer
        return contexts

    async def collect(self, plan: Any, resources: Any = None, *,
                      progress: ProgressCallback | None = None) -> RustCollectedBatch:
        try:
            result = await self._collect_owned(plan, progress=progress)
        except BaseException as failure:
            if self._work is not None:
                try:
                    self.retire_work(failure)
                except BaseException as cleanup:  # noqa: BLE001 - original owners may need host retirement.
                    failure.add_note("collection tmpfs retirement unconfirmed: " + str(cleanup))
            raise
        # The inner frame and all its borrowed metadata aliases have retired.
        # Inventory retains the shared workspace through writer join and until
        # this concrete final consumer has dropped the encoder graph.
        self.current = result
        try:
            assert self._active_writer is not None and self._controls is not None
            self._active_writer.encoders.clear()
            self._controls.close()
            self._inventory.close()
            self._inventory = None
            self._source_control_charge.close()
            self._source_control_charge = None
            self.retire_work()
        except BaseException as failure:
            try:
                result.close()
            except BaseException as cleanup:  # noqa: BLE001 - retain the actual pending owner.
                failure.add_note("durable collection retirement unconfirmed: " + str(cleanup))
            raise
        self.runtime.measurement_end()
        if self.runtime.diagnostics is not None:
            self.runtime.measurement_begin(self.runtime.diagnostics.scope(run_id=self.registry.state["run_id"],
                operation_id=self.registry.admission.observation["operation_id"],
                invocation_id=self.registry.admission.observation["invocation_id"], phase="idle"))
        return result

    async def _collect_owned(self, plan: Any, *,
                             progress: ProgressCallback | None = None) -> RustCollectedBatch:
        from gf_common.card_instance_rules import rule_signature
        from gf_common.card_instance_runtime import game_binding
        from gf_common.domain import EpisodeResult
        from gf_common.hpo_storage import hpo_diagnostics
        from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
        from gf_common.structured_tokens import signature
        from gf_training.structured_rollout import _owned_bytes

        self.registry.require()
        actual = self.registry.admission.observation
        http_diagnostics = hpo_diagnostics(getattr(self.config, "hpo_spec", {}).get("storage", {}))
        if http_diagnostics is not None:
            from importlib import import_module

            extension = import_module("gf_native")
            if not hasattr(extension.StorageBundle, "_http_diagnostics"):
                raise ContractError("Rust player HTTP diagnostics require the terminal bundle capability")
        configured_work = getattr(self.config.storage, "collection_work_root", None)
        if (self.current is not None and not self.current._closed
                or configured_work is not None and self._work is not None
                or plan.linkage["collection_generation"] != self.registry.state["updates"]
                or plan.execution_deadline != deadline_seconds(actual["deadlines"]["execution_bits"])):
            raise ContractError("collection differs from actual current generation/original deadline")
        originals = Path(self.config.storage.db_root)
        db_directory = originals / self.registry.state["run_id"] / "collections" / plan.linkage["batch_id"]
        work_mount_config = None
        durable_mount = None
        if configured_work is not None:
            work_root = Path(configured_work)
            work_mount, mount = check_staged_collection_destinations(
                work_root, plan.output, catalog_root=originals, originals=db_directory)
            major, minor = work_mount.device.split(":")
            work_mount_config = work_root, os.makedev(int(major), int(minor))
            major, minor = mount.device.split(":")
            durable_mount = Path(mount.mount_point), os.makedev(int(major), int(minor))
        checkpoints: dict[str, None] = {}
        for binding in plan.bindings.values():
            if binding.model_id in LEARNED_POLICY_KINDS:
                published = self.registry.publish(int(Path(binding.policy_options["checkpoint"]).name),
                                                  deadline=plan.execution_deadline)
                checkpoints[published["checkpoint"]] = None
        sequence_seats = (sum(plan.bindings[label].model_id in SEQUENCE_MODEL_KINDS
            for group in plan.groups for episode in group.episodes for label in episode.policies)
            if self.registry.service.sequence_enabled else None)
        self.registry.service.bind_collection(tuple(checkpoints),
            game_concurrency=self.declaration["game_concurrency"], max_batch_size=self.declaration["max_batch_size"],
            learner_checkpoint=self.registry.current.files["inference.json"],
            **({} if sequence_seats is None else {"scheduled_sequence_seats": sequence_seats}))
        diagnostics = self.runtime.diagnostics
        if diagnostics is not None:
            scope = diagnostics.scope(run_id=self.registry.state['run_id'],
                operation_id=actual['operation_id'], invocation_id=actual['invocation_id'],
                batch_id=plan.linkage['batch_id'], phase='collection')
            diagnostics.current_scope = scope
            self.runtime.measurement_begin(scope)
        start_counters = self._counters()
        contexts = self._contexts(plan)
        self._source_control_charge = self.registry.budget.reserve(_owned_bytes(contexts))
        # Static controls deliberately exclude the future executed process receipt.
        plans = tuple(EpisodeEncodingPlan(name, digest({key: asdict(value[key]) if key in ("actions", "rules")
            else [asdict(policy) for policy in value[key]] if key == "policies" else value[key]
            for key in value})) for name, value in contexts.items())
        groups = tuple(StorageGroup(group.episodes[0].episode_id,
            tuple(episode.episode_id for episode in group.episodes),
            {"seed": group.seed, "episode_ids": [episode.episode_id for episode in group.episodes]})
            for group in plan.groups)
        if configured_work is None and hasattr(self.config, 'hpo_spec'):
            check_collection_destinations(plan.output, db_directory, catalog_root=originals)
        plan.output.mkdir(parents=True, exist_ok=False)
        work = None if work_mount_config is None else self._create_work(*work_mount_config)
        grant = self.registry.budget.limit - self.registry.budget.used
        try:
            writer = RustPlayerStorage(self.runtime, db_directory, run_id=self.registry.state["run_id"],
                contexts=(), groups=groups, settings=self.settings, workspace=self.registry.budget,
                workspace_bytes=grant, deadline=plan.execution_deadline, cleanup_deadline=self.cleanup_deadline,
                deferred_contexts=True, context_plans=plans, training=self.mode is not None,
                work_directory=None if work is None else work / "originals", durable_mount=durable_mount)
        except BaseException as failure:
            owner = vars(failure).get("cleanup_owner")
            if work is not None and isinstance(owner, RustPlayerStorage):
                self._active_writer = owner
                self.writer_history.append({"writer": owner, "descriptor_sha256": None})
            raise
        self._active_writer = writer
        writer_history: Json = {'writer': writer, 'descriptor_sha256': None}
        self.writer_history.append(writer_history)
        controls = RecordingArtifacts(RecordingBudget(self.runtime.recording_account(), plan.execution_deadline))
        self._controls, self._inventory = controls, None
        inventory: Any = None
        evidence = CardInstanceRuntimeEvidence(Path(self.config.server.binary), Path(self.config.server.source_dir),
            contract=self.config.actions.action_contract)
        if work is None:
            evidence.preflight()

        def store(name: str, value: Json) -> tuple[str, int]:
            controls.json(name, value)
            data = controls.chunks[name][0][0]
            writer.store_control(name, data)
            return hashlib.sha256(data).hexdigest(), len(data)

        try:
            if work is not None:
                evidence.preflight()
            execution = await self._matches.execute(plan.groups, contexts, writer,
                server=self.config.server, actions=self.config.actions, max_decisions=self.config.run.max_decisions,
                output=plan.output if work is None else work,
                execution_deadline=plan.execution_deadline, cleanup_deadline=self.cleanup_deadline,
                game_concurrency=self.declaration["game_concurrency"], run_id=self.registry.state["run_id"],
                batch_id=plan.linkage["batch_id"], evidence=evidence, progress=progress,
                **({} if http_diagnostics is None else {"http_diagnostics": http_diagnostics,
                    "operation_id": actual["operation_id"]}))
            accepted, bound_contexts, completed_groups = execution.accepted, execution.bound_contexts, execution.completed_groups
            receipts, keys, reports, total = execution.receipts, execution.keys, execution.reports, execution.total
            del execution
            if work is not None and self._matches._pending_http:
                raise ContractError("staged collection simulator HTTP settlement remains unconfirmed")
            if self.runtime._service_error is not None or self.runtime._active_routes or self.runtime._holds:
                raise ContractError("actual inference consumers remain unsettled")
            if self.runtime.dispatcher.checkpoint_references() or any(self.registry.service.references.values()):
                raise ContractError("actual native/service policy references remain live")
            self.registry.service.end_collection()
            end_counters = self._counters()
            settlement = {"schema_version": 1, "kind": "rust-player-service-consumer-settlement-v1",
                "run_id": self.registry.state["run_id"], "batch_id": plan.linkage["batch_id"],
                "generation": self.registry.state["updates"], "invocation_id": actual["invocation_id"],
                "service_id": self.registry.service_id, "active_policy_references": 0, "native_pending": 0,
                "started_producers": 0, "gpu_fenced": True, "counters": end_counters}
            retirement = {"schema_version": 1, "kind": "rust-player-collection-retirement-v1",
                "run_id": self.registry.state["run_id"], "batch_id": plan.linkage["batch_id"],
                "generation": self.registry.state["updates"], "invocation_id": actual["invocation_id"],
                "admission_sha256": digest(actual), "runtime_binding_sha256": self.registry.state["runtime_binding_sha256"],
                "service_id": self.registry.service_id, "episode_ids": [key[2] for key in keys],
                "completed_reports": reports, "accepted_records_sha256": digest(accepted),
                "service_settlement": settlement, "service_settlement_sha256": digest(settlement), "gpu_fenced": True}
            capture = {"schema_version": 1, "kind": "rust-player-capture-authority-v1",
                "actor_protocol_version": 11, "run_id": self.registry.state["run_id"],
                "batch_id": plan.linkage["batch_id"], "generation": self.registry.state["updates"],
                "invocation_id": actual["invocation_id"], "admission_sha256": digest(actual),
                "runtime_binding_sha256": self.registry.state["runtime_binding_sha256"],
                "encoding_contexts_sha256": digest([bound_contexts[i] for i in range(len(plan.groups))]),
                "accepted_records": accepted, "completed_reports_sha256": digest(reports)}
            rows, result_files = [], []
            for (group, episode), native in zip(
                    ((item, spec) for item in plan.groups for spec in item.episodes), reports, strict=True):
                report = native["report"]
                health = {"cleanup_failure": ";".join(report["cleanup_errors"]) or None,
                    "save_failure": None if report["storage_accepted"] else "terminal publication failed"}
                result = asdict(EpisodeResult(episode.episode_id, episode.pair_id, group.seed, episode.policies,
                    game_id=report["game"], outcome=report["outcome"], end_kind=report["end_kind"],
                    simulator_abort_reason=report["simulator_abort_reason"], failure=report["error"],
                    health=health, decisions=report["decisions"]))
                rows.append(result)
                name = episode.episode_id + "/result.json"
                checksum, size = store(name, result)
                result_files.append({"path": name, "sha256": checksum, "bytes": size})
            healthy = (all(row["end_kind"] == "natural" and row["failure"] is None and not any(row["health"].values())
                for row in rows) and all(not group_health_failures(completed_groups[i]) for i in completed_groups)
                and all(row["report"]["inference_released"] and row["report"]["storage_accepted"] for row in reports))
            report = {"schema_version": 1, "healthy": healthy, "comparison_valid": healthy,
                "planned_games": total, "recorded_games": total,
                "natural_denominator": sum(row["end_kind"] == "natural" for row in rows),
                "episodes": rows, "failures": {} if healthy else {"unqualified_collection": 1},
                "health_failures": {}, "group_health_failures": {}}
            report_sha, _ = store("report.json", report)
            inventory = writer.inventory()
            self._inventory = inventory
            files = [{"path": name, "bytes": size, "sha256": checksum} for name, size, checksum in inventory.rows] + result_files
            usage = {"schema_version": 1, "kind": "rust-player-inference-usage-v1",
                "run_id": self.registry.state["run_id"], "invocation_id": actual["invocation_id"],
                "service_id": self.registry.service_id, "admission_sha256": digest(actual),
                "runtime_binding_sha256": self.registry.state["runtime_binding_sha256"],
                "config_sha256": self.registry.state["config_sha256"], "generation": self.registry.state["updates"],
                "batch_id": plan.linkage["batch_id"], "start": start_counters, "end": end_counters,
                "bindings": {label: {"checkpoint": binding.policy_options["expected_manifest_digest"],
                    "token": self.registry.cache[int(Path(binding.policy_options["checkpoint"]).name)]["binding"]["token"]}
                    for label, binding in plan.bindings.items() if binding.model_id in LEARNED_POLICY_KINDS},
                "collection_retirement_sha256": digest(retirement)}
            manifest = {"schema_version": 11 if self.registry.hpo else 10, "kind": "card-instance-collection",
                "command": "collect", "status": "complete" if healthy else "incomplete",
                "run_id": self.registry.state["run_id"], "config": self.registry.state["effective_config"],
                "config_hash": self.registry.state["config_sha256"], "inputs": self.registry.state["config_inputs"],
                "code": self.registry.state["identity"]["code"], "action_config_hash": digest(asdict(self.config.actions)),
                "bots_revision": self.registry.state["identity"]["code"]["revision"],
                "simulator_revision": self.config.actions.action_contract.simulator_revision,
                "runtime_evidence": [receipts[i] for i in range(len(plan.groups))],
                "schedule": serialize_schedule(plan.groups, 4), "schedule_schema_version": 4,
                "files": sorted(files, key=lambda row: row["path"]), "policies": list(plan.bindings),
                "journal_schema_version": 6, "derived_schema_version": 6, "participant_protocol_version": 3,
                "candidate_codec": "candidate-columns-v2", "model_version": "4" if self.registry.hpo else "3",
                "input_recipe": layout.input_recipe, "token_signature": signature(128, layout=layout),
                "source_eligibility": "not-authorized-by-collection-metadata",
                "policy_bindings": {label: asdict(binding) for label, binding in plan.bindings.items()},
                "neural_inputs": self.registry.journal_inputs(plan.bindings),
                "collection_role": "hpo-ppo-batch" if self.registry.hpo else "ppo-batch",
                "linkage": plan.linkage, "episodes": rows, "groups": [completed_groups[i] for i in range(len(plan.groups))],
                "report_sha256": report_sha,
                **game_binding(self.config.actions.action_contract, rule_signature(
                    self.config.actions.action_contract, self.registry.local.learner._context.rules)),
                "runtime_binding_sha256": self.registry.state["runtime_binding_sha256"],
                "player_producer": actual["producer"], "actor_protocol_version": 11,
                "capture_authority": capture, "logical_retirement": retirement, "service_usage": usage}
            if self.registry.hpo:
                manifest.update(origin="producer-owned-v1", encoder=source_encoder(self.registry.state['effective_config']),
                    snapshot_identity=self.registry.current.identity, replay_reconciled=False)
            planned_names = tuple(key[2] for key in keys)
            if self.mode is not None:
                if not healthy:
                    raise ContractError("native training requires healthy natural completion of every game")
                training_facts = writer.training_evidence(inventory, planned_names)
                for fact, row in zip(training_facts, rows, strict=True):
                    if (fact['episode_id'] != row['episode_id'] or fact['game_id'] != row['game_id']
                            or fact['viewer'] != plan.linkage['learner_seats'][row['episode_id']]
                            or fact['decisions'] > row['decisions']):
                        raise ContractError("committed training evidence differs from planned game/report")
                manifest.update(schema_version=13 if self.registry.hpo else 12, training_episodes=training_facts)
            validate_record_shape(manifest, "collection", hpo=self.registry.hpo)
            manifest_sha, _ = store("manifest.json", manifest)
            capture_sha, retirement_sha = digest(capture), digest(retirement)
            await asyncio.to_thread(writer.seal)
            await asyncio.to_thread(writer.join)
            extras: Json = {}
            if self.mode is not None:
                pack = await asyncio.to_thread(writer.verify_training_seal, manifest_sha)
                metadata_lease = writer.budget.reserve(_owned_bytes((manifest, report, pack)))
                extras.update(pack_verification=pack, manifest=manifest, report=report, metadata_lease=metadata_lease,
                    training=writer.take_training_episodes(planned_names))
            pending = writer.take_pending_owner()
            writer_history['descriptor_sha256'] = pending.descriptor_sha256
            return RustCollectedBatch(_ISSUER, self, plan, writer, pending, manifest_sha,
                capture_sha, retirement_sha, **extras)
        except BaseException as error:
            if diagnostics is not None:
                diagnostics.error("collection_interrupted")
                if self.runtime._profile_status is not None:
                    self.runtime._profile_status.end()
            vars(error)["cleanup_owner"] = self
            self.runtime.cancel("collection failed")
            writer.cancel("collection failed")
            for server in self._servers.values():
                if not server.stopped:
                    try:
                        await server.stop()
                    except BaseException as cleanup:  # noqa: BLE001 -- retain primary failure and actual owner.
                        error.add_note("actual server remains owned: " + type(cleanup).__name__)
                if http_diagnostics is not None:
                    try:
                        diagnostic = await self._matches.settle_http(server, deadline=self.cleanup_deadline,
                                                                     abnormal=True)
                        if diagnostic is not None:
                            error.add_note("HTTP pair diagnostics " + str(diagnostic))
                    except BaseException as cleanup:  # noqa: BLE001 -- retain original cancellation and raw owner.
                        error.add_note("HTTP pair diagnostics remain owned: " + type(cleanup).__name__)
            try:
                await asyncio.to_thread(writer.close)
            except BaseException as cleanup:  # noqa: BLE001 -- retain primary failure and actual owner.
                error.add_note("actual writer remains owned: " + type(cleanup).__name__)
            if self.mode is not None:
                try:
                    await asyncio.to_thread(writer.close_unclaimed_training)
                except BaseException as cleanup:  # noqa: BLE001 -- retain primary failure and actual owner.
                    error.add_note("actual sealed writer originals remain owned: " + type(cleanup).__name__)
            # Error tracebacks can still own metadata/serialized rows. Preserve
            # their concrete loan holders on cleanup_owner rather than refunding
            # memory still reachable through the failed transaction.
            raise
