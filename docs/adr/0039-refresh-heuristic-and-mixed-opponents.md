# 0039: Selectable refresh heuristic and mixed training opponents

- Status: Accepted design; implementation/acceptance pending.
- Date: 2026-11-15. Owner: root.
- Authority: user's accepted plan and explicit implementation request.
- Architecture review: /root/refresh_heuristic_design, READY on bounded design
  and persisted-design recheck; final implementation-conformance remains required.

## Context

Existing heuristic prioritizes fixed immediate content scores. The user requests
an independently selectable strategy that renews possession until attack is useful,
using configurable initial threshold12 and recent public outcome observations.
Both ordinary PPO and HPO should optionally train against old and new heuristics.

## Decision

Adopt [the design](../design/heuristic-refresh.md) and
[implementation plan](../design/heuristic-refresh-implementation.md).
Participant-local models own behavior; common owns closed shared identity; native
validates records only. The user authorizes an explicitly approximate observed
turn-window damage statistic, not causal attack attribution. Mix exactly16 games
as Random4/old heuristic2/refresh2/snapshot8 via a new opt-in recipe. Historical
fixed kernels/defaults/formal qualification and canonical evaluation remain exact.

## Alternatives and rationale

Raw damage averages are not comparable across different attacks. Exact causal
attribution needs a separate public Simulator contract. A separate policy identity
avoids changing the existing comparator and lets thresholds coexist in records.
Concrete new schedule wrappers avoid a general opponent framework and preserve
frozen historical functions.

## Consequences and accepted debt

Outcome/retention/acquisition values are proxies and do not prove hidden defense,
future legality or tactical quality. Revisit with measured experiment evidence or
a new versioned public contract. Configured baseline options require explicit
Python/native record branches and early native-capability refusal. New mixed
parents never acquire eligibility under the old formal recipe.

## Verification

Behavior, identity, old-data, live/reconstructed schedule, Source/native and
functional game/update/resume evidence and independent reviews are required.
Current implementation evidence: NOT RUN.
