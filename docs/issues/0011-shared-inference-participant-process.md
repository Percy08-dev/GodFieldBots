# Simulator handoff: participant-scoped inference in a shared tensor executor

Status: open. Affects the proposed optional shared MLP inference path, for both
ordinary PPO and HPO. The existing per-viewer participant runtime can run.

## Question for the Simulator contract owner

May one trusted inference process execute a batch whose rows derive from the
private views of distinct participants, including the two seats, when each
row's result depends only on that participant's authorized inputs and the
bound inference-only checkpoint? This would change where private data may
coexist in memory. A row mask by itself does not authorize that data flow.

In Simulator revision `91d5926a7572db327eaea3751e7c24f428120996`,
`docs/spec/07-external-interfaces-adapters.md` §21.1 explicitly restricts
delivery of another participant's view to the policy process. Section 22
allows an orchestrator to manage both seats and the same model **version**
to be used for both. The current separate participant processes meet both
conditions. The proposed TorchRL process computes action probabilities and
values, and receives rows derived from multiple private observations. Neither
section grants an exception for this execution topology. A search of the
provided Simulator docs found no separate shared-inference exception.

## Requested clarification

If the owner permits batching, specify the allowed executor, ownership of
inputs/outputs, isolation across rows, participants, games and checkpoints,
and when retained private buffers or state must be cleared. Participant
workers would keep candidate generation, action sampling and command creation.
Only the requesting participant would receive that row's outputs. Keep other
participants' views, hidden canonical history, rejection detail, seeds, future
information, optimizer and resume state out of that participant's policy
features. Source, audit, control and training remain separate responsibilities.

If permission is declined, record that decision and select a different design.
A separately checked Simulator specification revision and hash are needed
before adopting a changed contract in Bots. This document is a handoff;
it grants no permission and makes no changes to the Simulator repository.
