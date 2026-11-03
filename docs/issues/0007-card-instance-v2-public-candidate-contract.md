# Card-instance v2: missing public candidate predicates

- Status: Open; blocks complete generator adoption and dependent real-game,
  PPO/pilot/performance acceptance under ADR0024. Independent frame/token work
  may continue. No Simulator change is made by this Bots task.
- Simulator pin: `07e8185f98c503aa41200a80ea554b4fd21088ad`.
- Normative inventory SHA:
  `8381079eae8a68378d8419c7f7245c62358833f0dd9045667021e112a1e9bad5`.
- Decision: root accepts independent bot_advisor's public-spec review (2026-11-03).
  Do not derive these predicates from Engine handlers, SequenceOptions, rejection
  probes, global-target Cartesian products or a successful pilot.

## Required separate Simulator public-contract task

One normative clarification can resolve all three predicates. Keep the current
pin and historical evidence until an explicit new revision/content-hash adoption.
An owner statement alone must not silently change a pinned Simulator contract.

### 1. Flash counting after role binding

Spec09§14/§18.3 says one artifact. Classification§4 allows a defense miracle plus
adjacent CostCut. Spec03§34 exempts bound CostCut from weapon count and defense
exclusivity, but does not mention Flash. Counting every selected reference,
only artifact-kind cards, or only effective defense roles each adds an unstated
rule under the new artifact/miracle tagged union.

Specify the Flash predicate over selected instances and bound roles, including
unused/learned defense miracles and bound CostCut. Decide at least:

- Flash + `wall, spiritual-staff`;
- Flash + `wall, spiritual-socks`;
- otherwise-compatible multiple defenses distinguishing artifact-kind from
  effective-role counting.

State whether the generic CostCut combination permission is subject to Flash
or creates an explicit exception.

### 2. Discard/Sacrifice kind, state and subclass eligibility

Spec09§11 establishes nonweapon selection/refill rules, and §16.1 establishes
unified possession plus trade/Broom exclusions. Neither gives a complete
Discard/Sacrifice predicate for miracle states. Full-capacity Pray/Buy's explicit
inclusion of learned cards is operation-specific. `can_discard` and
`can_sacrifice` establish availability, not an eligible-reference list;
`usable_cards` cannot be repurposed as a discard mask.

Specify eligibility for artifact cards and all six miracle subclasses, separately
for unused/learned state. Define whether weapon exclusion includes normal/additional
miracle weapons and how overlapping use roles affect eligibility. The predicate
must be implementable from the permitted actor view under Dream; otherwise add
an appropriate public compatibility guarantee or actor-scoped atomic hint.

### 3. Utility singleton accepted target and effective recipient

Spec09§17.5.2 establishes scoped singleton completion (except Doll). §17.5.4
permits self/opponent for targetable cards but explicitly declines to determine
each definition's target-selection behavior and does not authorize a general
Cartesian product. Spec07§47's mandatory target is syntax, not either-target
legality. Tear and Heaven Herb explicitly apply to the selected player; the
remaining targetable/user-only distinction is not uniformly specified.

For every listed scoped singleton, state accepted wire targets and effective
recipient. A universal normative statement suffices if all are intended to
accept either target. Choosing self universally is also an unsupported domain
reduction and is not an adoption workaround.

## Reopening and verification

After the separate clarification, explicitly adopt its revision and normative
hashes rather than rewrite current evidence. Review the resulting matrix before
generator implementation. Add pure table-driven tests for all discriminating
cases and independently compare generated candidates to the declared matrix.
Runtime acceptance checks implementation conformance; it cannot supply missing
public rules. User-plan acceptance seeds, candidate scope and budgets remain.

Investigation was read-only against normative public specs. No runtime/rejection
experiment, Engine-handler inspection or Simulator edits were performed to
resolve these questions.
