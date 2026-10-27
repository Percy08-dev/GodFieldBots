# 0002: Configured public-information baseline foundation

- Status: Accepted
- Earlier ActionOptions prohibition/input boundary: superseded by
  [ADR0005](0005-authorized-atomic-action-options.md). Its historical rationale
  below does not override the updated authorized actor atomic-hint contract.
- Persisted end-projection/schema portion: superseded by [ADR0003](0003-episode-closure-projection.md);
  the other foundation decisions remain current. Earlier schema-2 rationale below is historical.
- Date: 2026-10-27
- Decision owner: root
- Review: `bot_architecture_reviewer`, READY, 2026-10-27; configured public-space
  completeness clarified, ordinary transport recovery distinguished from failed
  attestation. Exact supported action/effect coverage must be frozen with tests.
  Provenance amendment also READY: explicit contract pin and optional local
  source directory; no implicit binary-layout or source-checkout requirement.
  Persisted-provenance amendment READY: journal/derived schema 2 links run and
  actual Game identity; export requires finalized manifest, preserves sidecar
  provenance/end metadata, rejects cross-run split conflicts, and keeps seed
  grouping metadata outside policy features. Development schema 1 is rejected.
  Clean-bootstrap amendment READY: fixed child interpreter argv/environment and
  framed JSON prevent Python multiprocessing pre-entry control-data inheritance.
  Deadline amendment READY: configurable report/cleanup reserves, cooperative
  checks and bounded waits; emergency OS-reap and local filesystem limitations
  recorded as debt with failed-run invalidation and explicit revisit conditions.
  Advisor review after repeated deadline fixes requires zero abort grace, linked
  report validation and independent correction writes. If both correction writes
  fail, durable state may remain stale; failed directories must be quarantined.
- Implementation: Phase 0–3 foundation in `src/`, presets in `configs/`;
  final correctness/conformance READY. [Verification](../verification/phase0-3.md).

## Context

The user's plan compares Random/heuristic, LightGBM and three PPO architectures.
The initial repository is a scaffold. The pinned simulator already exposes
a public catalog, participant typed metadata and repeated-seed TestServer.
An evidence-producing Phase 0–3 foundation must precede learning claims.

## Decision

Adopt the responsibilities, interfaces, config precedence, visibility, lifecycle
and journal/evaluation contracts in [architecture](../architecture.md).
Use CPython 3.14 GIL, uv, httpx/NumPy and pytest/ruff/mypy. Optional PyArrow,
LightGBM and Torch load only in their consumers. Version public candidate spaces;
enumerate is the reference and beam is bounded approximation. Spawn separate
participant workers, attested TestServer per seed, serial Create and immutable
transport retries. JSONL journals are primary Bots evidence, optional Parquet is
reproducible. Replay stays diagnostic. Unknown versions/poisoned connections fail
closed. Game outcome and recording/transport/process health remain separate.
Stop dependent experiments at a failed learning gate.

## Alternatives

- Engine action masks violate the learning-input contract in Dream.
- Engine copying violates ADR 0001 and causes semantic drift.
- Shared policy process permits accidental cross-participant retention.
- Implementing all models first risks learning from incorrect episode labels.
- Mandatory ML libraries complicate independent CPU baseline installation.

## Consequences

Initial candidate coverage is explicit and may omit complex effects. Revisit
when pilot finds empty candidates, rejections or poor completion, using public
rule evidence. No full-engine legality claim is made. JSONL costs space but
survives partial failures. Spawn overhead must be measured before optimization.
Truncated episodes lack final observations; simulator extensions are separate.

## Verification

Architecture acceptance checks, phase evidence, independent design review before
implementation, and separate correctness/conformance reviews after implementation.
Acceptance is not evidence of training quality.

Final verification: 123 ordinary tests and two opt-in actual TestServer tests
pass; Ruff/mypy pass, isolated wheel baseline and schema-2 deterministic export
pass. Expanded evaluation retains four rejections out of 12; the learning gate
is unfulfilled and subsequent model/training phases remain unimplemented.
