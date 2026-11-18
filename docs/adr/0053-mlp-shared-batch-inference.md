# 0053: Shared MLP inference under the adopted Simulator contract

Status: design accepted; implementation and physical acceptance pending.
Date: 2026-11-18. Owner: root. Baseline: c78f8a0.
Preimplementation architecture review: /root/shared_architecture, READY for
design SHA-256 ef888d3e40679480531821b528bee7f37677bcd3f809ff952afd81afd128c8e6.

## Context

ADR0050–0052 improve existing resident GPU admission and diagnostics. Four
concurrent games run within the original12GiB managed budget, but each neural
participant retains its own CUDA context/weights. Simulator issue72 previously
blocked shared execution across participant views. Revision
072dabbca1458c9526897781d2fd64edeeb4f9ca now explicitly permits trusted shared
batch inference subject to independence, checkpoint and response-lifecycle tests.
Historical91d5926 sources keep their original contract.

## Decision

Implement [the bounded MLP design](../design/mlp-shared-batch-inference.md).
Add lazy optional TorchRL0.15.0/TensorDict0.15.2 and an explicit Python-only
shared mode for producer-owned Model4 HPO and ordinary Model3 PPO. One clean CPU
broker bridges dedicated participant channels to official InferenceServer and
MPTransport; one GPU process holds immutable inference-only checkpoint modules.
CPU participants generate candidates, sample using their own RNG and submit Act.

The broker owns the original multiprocessing child, independent of TorchRL's
shutdown return. Service reservations, cache references and retirement proofs
are explicit service resources rather than participant-shaped records. HPO
retains its durable global admission/quarantine; ordinary mode retains local
invocation custody and requires actual service cleanup before healthy completion
or pause. New identities/versions do not reinterpret old config/Source receipts.

Checkpoint acceptance is immutable; postcommit official update_model_weights
publishes a fresh registry entry while referenced older versions remain. Match
execution/game/participant/request/state_revision/checkpoint before delivering
only the requester's ordered scores/value. Expired responses produce no action
or learning transition. Private lengths/values/order must not affect another
row; use row-local arithmetic where necessary and report actual vectorization.

## Alternatives and consequences

Passing MPTransport clients directly to existing exec bootstraps is unsupported:
their queues/events require multiprocessing inheritance. Changing all actors to
parent-main multiprocessing would break the clean participant bootstrap. A
custom replacement batch server duplicates the approved official transport.
The CPU broker retains clean actors/coordinator while adding explicit nested
process custody and IPC cost, which the bounded comparison must measure.

Native shared formats, GRU hidden-state serving and Transformer KV serving are
deferred. Existing native readers reject unsupported shared/profile records.
Revisit model-specific serving only after interleaved own-history equivalence.
No full-RAM physical limit or learning-quality result is presumed. Broker loss
before child-reap proof can leave HPO admission requiring existing explicit
operator recovery; no new administrative recovery authority is introduced.

## Verification

Require adversarial private-row independence/routing/version/retirement checks,
actual service process cleanup, focused and whole/static/package checks and two
independent final review roles. Then execute the owner's seven-HPO-run384-game
protocol plus one short ordinary Model3 run. Decide preset adoption from matched
three-game concurrency and select concurrency from measured throughput/headroom.
Keep first/second collections, estimates and physical samples distinct. Record
final evidence in [the verification record](../verification/ppo-shared-inference.md).
