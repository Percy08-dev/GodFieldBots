# 0046: Standard bounded Model4 PPO updates

- Status: Accepted design; implementation and verification pending
- Date: 2026-11-17
- Owner: root
- Baseline: `b3831a0`
- Preimplementation review: bot_architecture_reviewer READY after the user's
  explicit extreme-capacity risk acceptance removed the universal-fit promise

## Context

ADR0045 measures clear short-protocol improvements for MLP microbatch and sequence
shared-prefix execution while retaining original defaults. The user now explicitly
authorizes those normal defaults and practical subdivision, keeping the baseline
internal for regression. The user then explicitly accepts extreme-setting OOM
risk; universal capacity preservation is not required. The existing single1024-frame,
handFF1024 packed intermediate would be72MiB even though scalar reference frames
fit; endpoint-only splitting cannot solve that case.

## Decision

Implement [the bounded adoption design](../design/ppo-update-adoption.md). Normal
Model4 chooses MLP microbatch and GRU/Transformer shared-prefix automatically.
Preplanned immutable frame/candidate staging refines practical capacity without
enlarging caps or falling back on failed execution. Full-gradient extreme graphs
may still fail conservative admission; no new activation rematerialization/VJP
mechanism is introduced. Both original numerical worker routes
receive their true process allowance through existing creator/module ownership.
Keep older frozen learners and private reference/expt modes intact.

This supersedes ADR0045 only for normal dispatch and its associated launch-cap
plumbing. Previous comparison/telemetry data remains version-pinned historical
evidence. No normal config/checkpoint/Wire6/7 field, dependency or Simulator change.
Ordinary resume source matching remains strict; no legacy paused data is migrated.

## Alternatives and consequences

Raising caps or silently returning to the old loop would not meet the approved
single normal-route choice. A second trainer duplicates PPO responsibility.
Always streaming would add avoidable overhead on the measured common shapes;
retain the admitted packed fast path. Additional lifetime-aware planning is the
necessary cost of preserving long-input capacity. Revisit complexity if storage
strategies proliferate or later evidence shows an avoidable common-path regression.

## Verification

Preimplementation architecture gate, focused capacity/default/ownership/gradient
tests, existing frozen/reference/resume/native-source regressions, normal CUDA
fresh/pause-resume and strict original Source/receipt checks, both final independent
reviews. No new learning-quality/HPO/final-test conclusion follows from adoption.
