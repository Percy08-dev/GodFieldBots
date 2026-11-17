"""Decision-local full-validation proof, never durable or hash authority."""
from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


def task() -> Any:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


@dataclass
class _Scope:
    owner: Any
    thread: int
    proofs: list[tuple[Any, Any, int, int]] = field(default_factory=list)


class DecisionProof:
    def __init__(self) -> None:
        self.current: ContextVar[_Scope | None] = ContextVar('collection_validation_proof', default=None)
        self.full = self.reused = 0

    @contextmanager
    def decision(self) -> Any:
        scope = _Scope(task(), threading.get_ident())
        token = self.current.set(scope)
        try:
            yield
        finally:
            scope.proofs.clear()
            self.current.reset(token)

    @contextmanager
    def suspended(self) -> Any:
        # A separate binding also protects an inherited child Context from
        # mutating the original Task's proof or suspension depth.
        token = self.current.set(None)
        try:
            yield
        finally:
            self.current.reset(token)

    def validate(self, original: Any, table: Any, frame: Any) -> Any:
        scope = self.current.get()
        own = (scope is not None and scope.owner is task() and scope.thread == threading.get_ident())
        if own:
            assert scope is not None
            for previous_table, previous_frame, actor, revision in scope.proofs:
                if (table is previous_table and frame is previous_frame
                        and frame.actor == actor and frame.revision == revision):
                    self.reused += 1
                    return None
        self.full += 1
        result = original(table, frame)  # Record only after ALL original checks.
        if own:
            assert scope is not None
            scope.proofs.append((table, frame, frame.actor, frame.revision))
        return result


def patches() -> tuple[list[Any], DecisionProof]:
    from gf_common.structured import CandidateTable
    from gf_simulator_client.card_instance_environment import CardInstanceSimulatorEnvironment
    from gf_training.card_instance_journal import CardInstanceJournal
    from gf_training.hpo_owned_capture import _CaptureContext

    proof = DecisionProof()
    validate = CandidateTable.validate
    select = CardInstanceSimulatorEnvironment._select_and_journal

    def validated(table: Any, frame: Any) -> Any:
        return proof.validate(validate, table, frame)

    async def scoped(*args: Any, **kwargs: Any) -> Any:
        with proof.decision():
            return await select(*args, **kwargs)

    def excluded(original: Any) -> Any:
        def full(*args: Any, **kwargs: Any) -> Any:
            with proof.suspended():
                return original(*args, **kwargs)
        return full

    # Include overrides: their own work must also run outside the proof.
    def descendants(cls: Any) -> list[Any]:
        return [cls, *(child for sub in cls.__subclasses__() for child in descendants(sub))]

    result = [(CandidateTable, 'validate', validated),
              (CardInstanceSimulatorEnvironment, '_select_and_journal', scoped)]
    from gf_training.group_pack_memory import MemoryCardInstanceJournal
    assert issubclass(MemoryCardInstanceJournal, CardInstanceJournal)
    result.append((CardInstanceJournal, 'begin_frame', excluded(CardInstanceJournal.begin_frame)))
    result.extend((cls, 'begin_frame', excluded(cls.__dict__['begin_frame']))
                  for cls in descendants(CardInstanceJournal)[1:] if 'begin_frame' in cls.__dict__)
    result.append((_CaptureContext, 'begin_decision', excluded(_CaptureContext.begin_decision)))
    return result, proof
