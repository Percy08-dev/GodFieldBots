"""Same-process numerical owners for the fixed supervised training composition.

Numerical kernels and checkpoint producers remain the existing Model3/4 code.
Source/collection authorization belongs to the concrete coordinator, not these
component facades. CPU components do not fabricate GPU or process authority.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json
from gf_models.arrays import sha256
from gf_models.rust_player import serialized_torch
from gf_training.rust_player_supervision import HostAdmission

_SAVED = object()


def _admit_device(binding: Any, admission: HostAdmission | None) -> None:
    if admission is not None:
        if type(admission) is not HostAdmission:
            raise ContractError("actual same-host admission required")
        admission.require()
    if binding.device != "cpu" and admission is None:
        raise ContractError("CUDA allocation requires actual HostAdmission")


class LocalSavedPair:
    """Actual completed local writer and unchanged inference/resume files."""

    def __init__(self, token: object, owner: Any, directory: Path, result: Json,
                 *, snapshot: Any = None) -> None:
        if token is not _SAVED:
            raise ContractError("actual local completed save required")
        self._owner, self.directory, self.snapshot = owner, directory, snapshot
        self._files = dict(result["files"])
        self.inference_digest = self._files["inference.json"]
        self.resume_digest = self._files["resume.json"]

    @property
    def files(self) -> Json:
        return dict(self._files)

    def require(self, owner: Any) -> None:
        owner.require()
        if owner is not self._owner or owner._saved.get(self.directory) is not self:
            raise ContractError("saved pair differs from original local writer")
        if self.snapshot is not None:
            self.snapshot.save_ack.require(self.snapshot)
        for name, expected in self._files.items():
            if sha256(self.directory / name) != expected:
                raise ContractError("actual saved pair original changed")

    def __reduce__(self) -> Any:
        raise TypeError("local saved writer authority cannot be serialized")


class _LocalLearner:
    def _own(self, admission: HostAdmission | None) -> None:
        self.admission = admission
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._failed = False
        self._saved: dict[Path, LocalSavedPair] = {}

    def require(self) -> None:
        if self._failed or (self._pid, self._thread) != (os.getpid(), threading.get_ident()):
            raise ContractError("original healthy local learner owner required")
        if self.admission is not None:
            self.admission.require()

    def _record_save(self, directory: Path, result: Json, *, snapshot: Any = None) -> LocalSavedPair:
        pair = LocalSavedPair(_SAVED, self, directory, result, snapshot=snapshot)
        self._saved[directory] = pair
        pair.require(self)
        return pair

    def __reduce__(self) -> Any:
        raise TypeError("local numerical owner cannot be serialized")


class LocalGameLearner(_LocalLearner):
    def __init__(self, model: Any, seed: int, binding: Any, resume_identity: str,
                 *, actions: Any, rules: Any, admission: HostAdmission | None = None) -> None:
        _admit_device(binding, admission)
        from gf_training.structured_numeric import GameLearner

        self._own(admission)
        with serialized_torch():
            self.learner = GameLearner(model, seed, binding, resume_identity,
                actions=actions, rules=rules)

    def update(self, rollout: Any) -> Json:
        self.require()
        with serialized_torch():
            try:
                return self.learner.update(rollout)
            except BaseException:
                self._failed = True
                raise

    def save(self, directory: Path, **options: Any) -> LocalSavedPair:
        self.require()
        directory = directory.absolute()
        with serialized_torch():
            try:
                result = self.learner.save(directory, **options)
                return self._record_save(directory, result)
            except BaseException:
                self._failed = True
                raise

    def restore(self, directory: Path) -> None:
        self.require()
        with serialized_torch():
            self.learner.restore(directory)


class LocalHpoLearner(_LocalLearner):
    def __init__(self, model: Any, training: Any, seed: int, binding: Any,
                 resume_identity: str, *, actions: Any, rules: Any, budget: Any,
                 admission: HostAdmission | None = None,
                 device_admission: Json | None = None) -> None:
        _admit_device(binding, admission)
        from gf_training.hpo_numeric import HpoLearner

        self._own(admission)
        self.budget, self.actions = budget, actions
        with serialized_torch():
            self.learner = HpoLearner(model, training, seed, binding, resume_identity,
                actions=actions, rules=rules, _owned_device_admission=device_admission)
            self.learner._owned_source_budget = budget

    def update_owned(self, rollout: Any) -> Json:
        self.require()
        with serialized_torch():
            try:
                return self.learner.update_owned(rollout)
            except BaseException:
                self._failed = True
                raise

    def snapshot_owned(self, **options: Any) -> Any:
        self.require()
        with serialized_torch():
            return self.learner.snapshot_owned(budget=self.budget, **options)

    def restore_snapshot(self, snapshot: Any, *, resume_manifest: Json) -> None:
        self.require()
        from gf_training.hpo_artifacts import _restore_owned_learner

        with serialized_torch():
            _restore_owned_learner(self.learner, inference=snapshot.inference,
                resume=snapshot.resume, actor_metadata=None, resume_manifest=resume_manifest,
                generation=snapshot.generation, budget=self.budget, actions=self.actions,
                local_snapshot=snapshot)

    def save(self, directory: Path, **options: Any) -> LocalSavedPair:
        self.require()
        directory = directory.absolute()
        with serialized_torch():
            try:
                snapshot = self.snapshot_owned()
                ack = snapshot.save(directory, **options)
                ack.require(snapshot)
                return self._record_save(directory, {"files": ack.files}, snapshot=snapshot)
            except BaseException:
                self._failed = True
                raise
