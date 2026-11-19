"""Local learning ownership and normalized input contracts; no learning-quality claim."""

from dataclasses import asdict

import pytest

from gf_common.domain import ContractError, canonical, digest
from tests.test_rust_player_config import (
    NORMAL_BASE,
    RUN_ID,
    campaign_raw,
    runtime,
)


def test_normal_effective_input_is_exact_typed_and_actual_provenance(tmp_path):
    from gf_cli import config

    original = config.load_config(NORMAL_BASE, {"run": {"player_runtime": runtime()}})
    path = tmp_path / "effective-config.json"
    path.write_text(canonical(original.effective))
    loaded = config.load_rust_player_effective(path, expected_sha256=original.sha256)
    assert canonical(asdict(loaded.config)) == canonical(original.effective)
    assert loaded.sha256 == original.sha256
    assert loaded.inputs == ({"path": str(path), "sha256": original.sha256},)


def test_hpo_effective_input_preserves_own_projection_bounds(tmp_path):
    from gf_cli import config
    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_training.hpo_config import effective_settings

    raw = campaign_raw()
    raw["operating"]["max_updates"] = 10001
    raw["storage"]["destination_capacity_bytes"] = 32 * 1024**3
    campaign = campaign_config(raw, origin=tmp_path)
    settings = effective_settings(campaign.config.model, campaign.config.training, {}, layout=layout)
    original = materialize_hpo_run(campaign, settings, learning_seed=17, run_id=RUN_ID)
    path = tmp_path / "effective-config.json"
    path.write_text(canonical(original.effective))
    loaded = config.load_rust_player_effective(path, expected_sha256=original.sha256)
    assert canonical(asdict(loaded.config)) == canonical(original.effective)
    assert loaded.config.training.max_updates == 10001
    assert loaded.config.training.max_owned_bytes == 32 * 1024**3


@pytest.mark.parametrize("source_mib,accepted", [(12288, True), (12289, False), (16384, False)])
def test_fix2_hpo_full_file_obeys_existing_source_codec_boundary(tmp_path, source_mib, accepted):
    from gf_cli.config import load_rust_player_effective
    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_training.hpo_config import effective_settings

    campaign = campaign_config(campaign_raw(), origin=tmp_path)
    settings = effective_settings(campaign.config.model, campaign.config.training, {}, layout=layout)
    value = materialize_hpo_run(campaign, settings, learning_seed=17, run_id=RUN_ID).effective
    value["preprocessing"]["memory_budget_mib"] = source_mib
    value["hpo_spec"]["operating"]["source_budget_bytes"] = 16 * 1024**3
    path = tmp_path / "effective-config.json"
    path.write_text(canonical(value))
    if accepted:
        loaded = load_rust_player_effective(path, expected_sha256=digest(value))
        assert loaded.config.preprocessing.memory_budget_mib == source_mib
    else:
        with pytest.raises(ContractError, match="12GiB"):
            load_rust_player_effective(path, expected_sha256=digest(value))


def test_fix2_hpo_full_file_requires_positive_evaluation_bootstrap(tmp_path):
    from gf_cli.config import load_rust_player_effective
    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_training.hpo_config import effective_settings

    campaign = campaign_config(campaign_raw(), origin=tmp_path)
    settings = effective_settings(campaign.config.model, campaign.config.training, {}, layout=layout)
    value = materialize_hpo_run(campaign, settings, learning_seed=17, run_id=RUN_ID).effective
    value["evaluation"]["bootstrap_samples"] = 0
    path = tmp_path / "effective-config.json"
    path.write_text(canonical(value))
    with pytest.raises(ContractError):
        load_rust_player_effective(path, expected_sha256=digest(value))


@pytest.mark.parametrize("full_file", [False, True])
def test_fix2_unsupported_actions2_rust_profile_is_controlled_error(tmp_path, full_file):
    from gf_cli.config import (
        GroupPackExperimentConfig,
        GroupPackStorageConfig,
        ServerConfig,
        _normalized_config,
        load_config,
        load_rust_player_effective,
    )
    from gf_common.action_contract import engineering_contract
    from gf_common.preprocessing import PreparationConfig
    from gf_common.structured import ActionConfigV2
    from gf_models.config import ModelConfig
    from gf_models.structured_binding import model_defaults_v2

    base = load_config(NORMAL_BASE, {"run": {"player_runtime": runtime()}}).config
    unsupported = GroupPackExperimentConfig(run=base.run,
        server=ServerConfig(binary=base.server.binary, config=base.server.config,
            fixed_seed_experiment=True), actions=ActionConfigV2(action_contract=engineering_contract()),
        model=model_defaults_v2(), opponent=ModelConfig(model_id="random"), training=base.training,
        preprocessing=PreparationConfig(reader="structured-v1"), storage=GroupPackStorageConfig())
    value = asdict(unsupported)
    with pytest.raises(ContractError):
        if full_file:
            path = tmp_path / "effective-config.json"
            path.write_text(canonical(value))
            load_rust_player_effective(path, expected_sha256=digest(value))
        else:
            _normalized_config(value, data=value, inputs=())


@pytest.mark.parametrize("field,value", [
    ("server.simulator_revision", "unreviewed-simulator-revision"),
    ("server.gameplay_api", "v1"), ("server.fixed_seed_experiment", False),
    ("server.external_url", "http://127.0.0.1:1234"),
    ("server.transport_retries", 3), ("server.expected_fingerprint", "wrong"),
    ("preprocessing.reader", "legacy"), ("run.name", "bad/path"),
    ("run.seeds", ["not-uuid"]), ("run.sampling_seeds", [-1]),
    ("run.pairs_per_seed", 0), ("run.max_decisions", 0),
    ("model.checkpoint", "/tmp/preexisting"), ("model.inference.max_candidates", 1),
    ("opponent", "gru"),
    ("run.name", "not-a-run-uuid"), ("run.output_dir", "/tmp/foreign-output"),
    ("hpo_spec.storage.numeric_root", "/tmp/foreign-numeric"),
    ("hpo_spec.storage.originals_root", "/tmp/foreign-originals"),
    ("hpo_spec.operating.collector_budget_bytes", 1),
    ("hpo_spec.operating.source_budget_bytes", 1),
    ("hpo_spec.operating.save_budget_bytes", 1),
])
def test_fix1_hpo_canonical_file_rejects_unsupported_crossfield_profiles(tmp_path, field, value):
    from gf_cli.config import load_rust_player_effective
    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.structured_binding import model_defaults_v3
    from gf_training.hpo_config import effective_settings

    campaign = campaign_config(campaign_raw(), origin=tmp_path)
    settings = effective_settings(campaign.config.model, campaign.config.training, {}, layout=layout)
    effective = materialize_hpo_run(campaign, settings, learning_seed=17, run_id=RUN_ID).effective
    if field == "opponent":
        effective[field] = asdict(model_defaults_v3(value))
    else:
        table, name = field.split(".", 1)
        if "." in name:
            parent, name = name.split(".")
            effective[table][parent][name] = value
        else:
            effective[table][name] = value
    path = tmp_path / "effective-config.json"
    path.write_text(canonical(effective))
    with pytest.raises(ContractError):
        load_rust_player_effective(path, expected_sha256=digest(effective))


@pytest.mark.parametrize("change", ["newline", "missing", "relative", "wrong_hash"])
def test_effective_input_refuses_changed_or_noncanonical_document(tmp_path, change):
    from gf_cli import config

    original = config.load_config(NORMAL_BASE, {"run": {"player_runtime": runtime()}})
    value = original.effective
    if change == "missing":
        del value["run"]["max_decisions"]
    elif change == "relative":
        value["run"]["output_dir"] = "relative"
    path = tmp_path / "effective-config.json"
    path.write_text(canonical(value) + ("\n" if change == "newline" else ""))
    with pytest.raises(ContractError):
        config.load_rust_player_effective(
            path, expected_sha256="0" * 64 if change == "wrong_hash" else digest(value)
        )


@pytest.mark.parametrize("malformed", ["nan", "invalid_hpo", "unknown_hpo"])
def test_effective_input_rejects_malformed_runtime_and_unknown_hpo_before_optional_use(tmp_path, malformed):
    import hashlib

    from gf_cli import config

    original = config.load_config(NORMAL_BASE, {"run": {"player_runtime": runtime()}})
    value = original.effective
    if malformed == "invalid_hpo":
        value["hpo_spec"] = []
    elif malformed == "unknown_hpo":
        value["hpo_spec"] = {"schema_version": 7}
    raw = canonical(value).encode()
    if malformed == "nan":
        raw = raw.replace(b'"formation_wait_ms":0.0', b'"formation_wait_ms":NaN')
    path = tmp_path / "effective-config.json"
    path.write_bytes(raw)
    with pytest.raises(ContractError):
        config.load_rust_player_effective(path, expected_sha256=hashlib.sha256(raw).hexdigest())


def test_run_transfer_requires_actual_control_and_preserves_original_owner(tmp_path):
    from gf_training import lineage

    with lineage.run_lock(tmp_path) as ownership:
        with pytest.raises(ContractError, match="authenticated"):
            lineage.transfer_run_lock(ownership, object())
        assert lineage.require_hpo_run_owner(tmp_path, ownership) is ownership
    with lineage.run_lock(tmp_path):
        pass


def test_borrowed_run_owner_requires_actual_control_not_fd_or_receipt(tmp_path):
    from gf_training import lineage

    with (
        lineage.run_lock(tmp_path) as ownership,
        pytest.raises(ContractError, match="authenticated"),
        lineage.borrow_run_lock(tmp_path, {"token": ownership.token}),
    ):
        pass


def test_fixed_bootstrap_passes_real_loaded_configuration_by_keyword(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gf_cli import config, rust_player_host

    original = config.load_config(NORMAL_BASE, {"run": {"player_runtime": runtime()}})
    path = tmp_path / "effective-config.json"
    path.write_text(canonical(original.effective))
    received = []

    def handler(control, operation, *, loaded):
        received.append(loaded)

    monkeypatch.setattr(rust_player_host.importlib, "import_module",
        lambda _name: SimpleNamespace(run_ordinary=handler))
    operation = {"operation_type": "ordinary-train", "config_path": str(path),
        "config_sha256": original.sha256}
    rust_player_host._dispatch(SimpleNamespace(operation=operation), "host")
    assert len(received) == 1 and received[0].config == original.config
    assert received[0].inputs == ({"path": str(path), "sha256": original.sha256},)


def test_local_model3_update_save_restore_matches_current_numeric(tmp_path):
    import torch

    from gf_training import rust_player_learning as local
    from tests.test_card_instance_learning import ACTIONS, RESUME, RULES, learner_for, rollout_for
    from tests.test_structured_learning import assert_same_learners

    current = learner_for()
    owner = local.LocalGameLearner(current.model, 17, current.binding, RESUME,
        actions=ACTIONS, rules=RULES)
    batch = rollout_for(current)
    expected = current.update(batch)
    expected_rng = torch.get_rng_state().clone()
    actual = owner.update(batch)
    assert actual == expected
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert_same_learners(current, owner.learner)
    saved = owner.save(tmp_path / "pair")
    assert saved.directory == tmp_path / "pair"
    saved.require(owner)
    restored = local.LocalGameLearner(current.model, 19, current.binding, RESUME,
        actions=ACTIONS, rules=RULES)
    restored.restore(saved.directory)
    assert_same_learners(current, restored.learner)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    with pytest.raises(ContractError):
        saved.require(restored)


def test_local_cuda_requires_actual_admission_before_torch_allocation():
    from dataclasses import replace

    from gf_training import rust_player_learning as local
    from tests.test_card_instance_learning import ACTIONS, RESUME, RULES, learner_for

    current = learner_for()
    with pytest.raises(ContractError, match="HostAdmission"):
        local.LocalGameLearner(current.model, 17, replace(current.binding, device="cuda:0"),
            RESUME, actions=ACTIONS, rules=RULES)


def test_local_hpo_saved_snapshot_restores_without_actor9_metadata(tmp_path, monkeypatch):
    import json

    import torch

    from gf_training import rust_player_learning as local
    from gf_training.hpo_owned_snapshot import OwnedModelSnapshot
    from tests.test_card_instance_learning import ACTIONS, RESUME, RULES
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import budget
    from tests.test_structured_learning import assert_same_learners

    current = learner_for()
    account = budget()
    owner = local.LocalHpoLearner(current.model, current.training, 17, current.binding,
        RESUME, actions=ACTIONS, rules=RULES, budget=account)
    monkeypatch.setattr(OwnedModelSnapshot, "actor_metadata",
        property(lambda _self: pytest.fail("local restore constructed actor9 metadata")))
    pair = owner.save(tmp_path / "pair")
    expected_rng = torch.get_rng_state().clone()
    restored = local.LocalHpoLearner(current.model, current.training, 19, current.binding,
        RESUME, actions=ACTIONS, rules=RULES, budget=account)
    restored.restore_snapshot(pair.snapshot,
        resume_manifest=json.loads((pair.directory / "resume.json").read_text()))
    assert_same_learners(current, restored.learner)
    assert torch.equal(expected_rng, torch.get_rng_state())
    pair.snapshot.close()
    assert account.used == 0


@pytest.mark.parametrize("field", ["shuffle_rng", "optimizer_steps"])
def test_fix1_restore_rejects_unbound_saved_metadata_before_any_state_change(tmp_path, field):
    import copy
    import json

    import numpy as np
    import torch

    from gf_training.rust_player_learning import LocalHpoLearner
    from tests.test_card_instance_learning import ACTIONS, RESUME, RULES
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import budget

    current = learner_for()
    account = budget()
    owner = LocalHpoLearner(current.model, current.training, 17, current.binding,
        RESUME, actions=ACTIONS, rules=RULES, budget=account)
    pair = owner.save(tmp_path / "pair")
    restored = LocalHpoLearner(current.model, current.training, 19, current.binding,
        RESUME, actions=ACTIONS, rules=RULES, budget=account)
    metadata = json.loads((pair.directory / "resume.json").read_text())
    metadata[field] = (np.random.Generator(np.random.PCG64(123)).bit_generator.state
        if field == "shuffle_rng" else 1)
    learner = restored.learner
    weights = {name: value.clone() for name, value in learner.network.state_dict().items()}
    optimizer = copy.deepcopy(learner.optimizer.state_dict())
    shuffle = copy.deepcopy(learner.shuffle.bit_generator.state)
    rng = torch.get_rng_state().clone()
    identity = (learner.network, learner.optimizer, learner.shuffle,
        learner.steps, learner._owned_generation, learner._owned_snapshot, learner._owned_source_budget)
    with pytest.raises(ContractError, match="saved resume"):
        restored.restore_snapshot(pair.snapshot, resume_manifest=metadata)
    assert identity == (learner.network, learner.optimizer, learner.shuffle,
        learner.steps, learner._owned_generation, learner._owned_snapshot, learner._owned_source_budget)
    assert learner.optimizer.state_dict() == optimizer
    assert canonical(learner.shuffle.bit_generator.state) == canonical(shuffle)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(weights[name], value) for name, value in learner.network.state_dict().items())
    pair.snapshot.close()


def test_local_hpo_same_synthetic_numeric_update_and_saved_adam(tmp_path, monkeypatch):
    """Numerical control only: the fixture is historical capture, never new Source."""
    import json

    import torch

    from gf_training import rust_player_learning as local
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from gf_training.hpo_rollout import HpoRollout
    from tests import test_hpo_owned_numeric as fixtures
    from tests.test_card_instance_learning import ACTIONS, RESUME, RULES
    from tests.test_hpo_numeric import learner_for
    from tests.test_structured_learning import assert_same_learners

    reference = learner_for()
    account = fixtures.budget()
    owner = local.LocalHpoLearner(reference.model, reference.training, 17, reference.binding,
        RESUME, actions=ACTIONS, rules=RULES, budget=account)
    monkeypatch.setattr(fixtures, "budget", lambda: account)
    captured, issuer, plan, frozen, _, records = fixtures.captured_record(owner.learner)
    selection = selection_for_counts((3,), learning_seed=17, batch=0,
        decision_budget=3, used_before=0)
    numeric = encode_owned_batch(captured, owner=issuer, plan=plan, snapshot_owner=frozen,
        model=reference.model, training=reference.training, actions=ACTIONS, rules=RULES,
        selected_locations=((0, 0), (0, 1), (0, 2)), selection=selection, budget=account)
    numeric.persist(tmp_path / "numeric")
    strict = HpoRollout((records,), reference.model, reference.training, actions=ACTIONS,
        rules=RULES, selected_locations=((0, 0), (0, 1), (0, 2)))
    reference.update(strict)
    expected_rng = torch.get_rng_state().clone()
    owner.update_owned(numeric)
    assert_same_learners(reference, owner.learner)
    assert torch.equal(expected_rng, torch.get_rng_state())
    with pytest.raises(ContractError, match="single use"):
        owner.learner.update_owned(numeric)
    pair = owner.save(tmp_path / "pair")
    restored = local.LocalHpoLearner(reference.model, reference.training, 19, reference.binding,
        RESUME, actions=ACTIONS, rules=RULES, budget=account)
    restored.restore_snapshot(pair.snapshot,
        resume_manifest=json.loads((pair.directory / "resume.json").read_text()))
    assert_same_learners(reference, restored.learner)
    assert torch.equal(expected_rng, torch.get_rng_state())
    numeric.close()
    captured.close()
    frozen.close()
    pair.snapshot.close()
    del records, strict, captured, numeric
    assert account.used == 0


def run_cuda_numerical_component(admission, directory):
    """Private actual-supervisor fixture ONLY, under Root's 2GiB/1GiB/120s grant."""
    import gc
    import json
    import os
    from dataclasses import replace

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    from gf_common.group_pack import StorageSettings
    from gf_models.structured_config import InferenceBindingV2
    from gf_training import rust_player_learning as local
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_numeric import HpoLearner
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from gf_training.hpo_rollout import HpoRollout
    from gf_training.structured_numeric import GameLearner
    from tests import test_hpo_owned_numeric as fixtures
    from tests.model3_support import model3
    from tests.test_card_instance_learning import ACTIONS, RESUME, RULES, rollout_for
    from tests.test_hpo_network import small_model4
    from tests.test_structured_learning import assert_same_learners

    admission.require()
    assert admission.observation["budget"]["managed_cuda_bytes"] == 2 * 1024**3
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    binding = InferenceBindingV2(device="cuda:0", managed_budget_bytes=2 * 1024**3,
        max_candidates=64, candidate_chunk=2)
    result = {"schema_version": 1, "kind": "rust-player-local-cuda-numerical-component",
        "qualified": False, "source_qualification": "not_performed",
        "actions": asdict(ACTIONS), "rules": asdict(RULES), "pairs": []}

    def rng():
        return torch.get_rng_state().clone(), [state.clone() for state in torch.cuda.get_rng_state_all()]

    def same_rng(expected):
        actual = rng()
        assert torch.equal(expected[0], actual[0])
        assert len(expected[1]) == len(actual[1])
        assert all(torch.equal(a, b) for a, b in zip(expected[1], actual[1], strict=True))

    reference = GameLearner(model3(), 17, binding, RESUME, actions=ACTIONS, rules=RULES)
    owner = local.LocalGameLearner(reference.model, 17, binding, RESUME,
        actions=ACTIONS, rules=RULES, admission=admission)
    rollout = rollout_for(reference)
    reference.update(rollout)
    expected = rng()
    owner.update(rollout)
    assert_same_learners(reference, owner.learner)
    same_rng(expected)
    pair = owner.save(directory / "model3")
    restored = local.LocalGameLearner(reference.model, 19, binding, RESUME,
        actions=ACTIONS, rules=RULES, admission=admission)
    restored.restore(pair.directory)
    assert_same_learners(reference, restored.learner)
    same_rng(expected)
    result["pairs"].append({"model_version": "3", "directory": str(pair.directory),
        "inference_sha256": pair.inference_digest, "resume_sha256": pair.resume_digest})
    del reference, owner, restored, pair, rollout, expected
    gc.collect()

    from gf_training.hpo_config import HPOTrainingConfig
    recipe = replace(HPOTrainingConfig(), epochs=2, minibatch_size=2)
    account = StorageBudget(StorageSettings(budget_bytes=1024**3), admission._control._finish())
    reference = HpoLearner(small_model4(), recipe, 17, binding, RESUME, actions=ACTIONS, rules=RULES)
    owner = local.LocalHpoLearner(reference.model, recipe, 17, binding, RESUME,
        actions=ACTIONS, rules=RULES, budget=account, admission=admission)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fixtures, "budget", lambda: account)
        captured, issuer, plan, frozen, _, records = fixtures.captured_record(owner.learner)
    selection = selection_for_counts((3,), learning_seed=17, batch=0,
        decision_budget=3, used_before=0)
    numeric = encode_owned_batch(captured, owner=issuer, plan=plan, snapshot_owner=frozen,
        model=reference.model, training=recipe, actions=ACTIONS, rules=RULES,
        selected_locations=((0, 0), (0, 1), (0, 2)), selection=selection, budget=account)
    numeric.persist(directory / "numeric-component-only")
    strict = HpoRollout((records,), reference.model, recipe, actions=ACTIONS, rules=RULES,
        selected_locations=((0, 0), (0, 1), (0, 2)))
    reference.update(strict)
    expected = rng()
    owner.update_owned(numeric)
    assert_same_learners(reference, owner.learner)
    same_rng(expected)
    pair = owner.save(directory / "model4")
    restored = local.LocalHpoLearner(reference.model, recipe, 19, binding, RESUME,
        actions=ACTIONS, rules=RULES, budget=account, admission=admission)
    restored.restore_snapshot(pair.snapshot,
        resume_manifest=json.loads((pair.directory / "resume.json").read_text()))
    assert_same_learners(reference, restored.learner)
    same_rng(expected)
    result["pairs"].append({"model_version": "4", "directory": str(pair.directory),
        "inference_sha256": pair.inference_digest, "resume_sha256": pair.resume_digest,
        "training_recipe": asdict(recipe)})
    numeric.close()
    captured.close()
    frozen.close()
    pair.snapshot.close()
    del reference, owner, restored, pair, numeric, captured, frozen, records, strict, expected
    gc.collect()
    assert account.used == 0
    torch.cuda.synchronize()
    result["memory"] = {"source_managed_peak_bytes": account.peak,
        "source_managed_remaining_bytes": account.used,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    assert result["memory"]["cuda_peak_reserved_bytes"] <= 2 * 1024**3
    (directory / "result.json").write_text(canonical(result))


def audit_cuda_saved_pairs_cpu(path):
    """Fresh CPU data inspection forbids importing Torch, initializing/querying CUDA."""
    import importlib.abc
    import json
    import sys
    import time
    from pathlib import Path

    class NoTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch" or fullname.startswith("torch."):
                raise AssertionError("cold CPU data audit attempted Torch/CUDA import")

    assert "torch" not in sys.modules
    sys.meta_path.insert(0, NoTorch())
    from gf_common.card_instance import action_config_v3
    from gf_common.card_instance_rules import CardInstanceRules
    from gf_models.arrays import sha256
    from gf_models.structured_artifacts import adam_recipe
    from gf_models.structured_sealed import (
        SealedScan,
        inspect_retained_game_artifact_v3,
        inspect_retained_game_artifact_v4,
    )

    result = json.loads(path.read_text())
    actions, rules = action_config_v3(result["actions"]), CardInstanceRules(**result["rules"])
    for row in result["pairs"]:
        directory = Path(row["directory"])
        assert sha256(directory / "resume.json") == row["resume_sha256"]
        scan = SealedScan(max_owned_bytes=1024**3, deadline=time.monotonic() + 20)
        options = {"expected_inference_digest": row["inference_sha256"],
            "actions": actions, "rules": rules, "scan": scan}
        if row["model_version"] == "4":
            recipe = row["training_recipe"]
            actual = inspect_retained_game_artifact_v4(directory, **options,
                training_recipe=recipe, optimizer_recipe=adam_recipe(recipe["adam_learning_rate"],
                    recipe["adam_beta1"], recipe["adam_beta2"], recipe["adam_epsilon"]))
        else:
            actual = inspect_retained_game_artifact_v3(directory, **options)
        assert actual.resume["binding"]["device"] == "cuda:0"
        assert actual.resume["cuda_rng_count"] >= 1
        assert actual.resume["optimizer_steps"] > 0
        scan.finalize()
    assert "torch" not in sys.modules
    path.with_name("cpu-data-audit.json").write_text(canonical({
        "kind": "rust-player-cuda-saved-state-component-audit", "qualified": False,
        "pairs": 2, "torch_imported": False, "source_qualification": "not_performed"}))
