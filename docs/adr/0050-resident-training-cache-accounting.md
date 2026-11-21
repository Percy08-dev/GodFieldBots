# 0050: Account for cached Model4 networks and in-flight resident imports

Status: accepted for the existing HPO training pool. Baseline: dad129f.
See the [design](../design/ppo-shared-inference.md) and
[measurements](../verification/ppo-shared-inference.md).

## Context and decision

ADR0049 preserved the previous admission calculation when making game counts
configurable. Its per-worker cache capacity and simultaneous postcommit
transition estimate treated configured ceilings as actual allocations even for
a model whose six networks occupy much less than the weight-byte limit.

The hpo-resident-cache-envelope-v2 admission result derives the maximum number
of homogeneous networks that fit by entries and FP32 parameter bytes. The
existing active-slot estimate includes active weights; add remaining weights
and one actual position buffer per network. Retain these allocations while
parked. During collection, multiple BEGINs can load concurrently. Start an
import reservation against the original worker and BEGIN request before framing;
release it after complete READY validation and successful transfer retirement.
A failed or cancelled request retains its reservation through original process
reap and outstanding I/O/FD retirement. No such pending imports can certify
PARK, optimization or successful termination. The postcommit SYNC/PARK loop
loads one worker at a time and reserves one transfer allowance. The evaluation
pool has different snapshot/participant ownership and keeps its own admission.

A permanent extra import allowance on every physical worker would charge
parked imports during update/save and could increase managed RAM need even
after correcting weight storage. Simply changing the CLI arithmetic would
make admission and the live ledger disagree. Track only actual outstanding
BEGIN charges and retain the existing failure/receipt owners. The admission
recipe describes new derived results; no historical records are rewritten.

## Consequences and remaining work

The retained 128MiB import allowance is explicitly conservative. The inherited
learner/active actor envelopes still mix host/device estimates; a lower physical
learner lease needs separate optimizer/restore/export and group-planner
accounting. Revisit these when per-domain activation, CPU native overhead and
workspace bounds cover the supported live phases. See the measured original
and maximum-cache cases for the extent of current calibration. This decision
does not introduce a shared policy service; its Simulator boundary is open.
