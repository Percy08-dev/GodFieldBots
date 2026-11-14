# 0040: Preserve both public GuardianActed identifier representations

- Status: Accepted; scoped implementation and both independent reviews READY.
  Fresh actuala02 and whole-project acceptance remain pending.
- Date: 2026-11-15. Owner: root.
- Supersedes only ADR0024§14.2's exclusive card-ID partition for corrected
  guardian_acted. Its historical evidence and other decisions remain intact.
- Advisor: /root/guardian_acted_advisor recommends this bounded representation
  repair; no Simulator change/further user authorization needed on public evidence.
- Architecture: /root/refresh_heuristic_design DESIGN READY, no blockers.

## Context and authority

The first actual mixed16 ordinary PPO attempt at revision55e6410 stops before any
update:14natural/2incomplete, both selfplay pair5 episodes fail with unknown public
event-time definition namespace. The whole parent is preserved and ineligible:
`runs/heuristic-refresh/ppo-python-a01/parent-a01`.

Its retained participant-visible HTTP/Replay output contains guardian_acted with
definition_id earth and moon. ADR0024§14.2 already records seafood-soup as an
observed GuardianActed action definition. The public transport DTO field is a
generic String; the pinned normative specifications do not require an exclusive
card-ID domain. Spec07§21.1 permits these participant-visible events. No engine
resolution internals, hidden state or rejection-derived legality were used.

The corrected layout already represents296card IDs plus10Guardian IDs distinctly.
Python/native validation imposes the narrower partition; encoding capacity and
the meaning of previously accepted inputs do not change.

## Decision and boundary

Only for the corrected91profile, accept pinned card IDs OR the known public
Guardian IDs in guardian_acted. Set/Left remain Guardian-only, other card-ID
events remain card-only, unknown identifiers still reject. This is a bounded
supported representation, not a claim that every allowed ID/event pair occurs.
Keep raw values/order/viewer ownership; no conversion, inferred effect, new
candidate/legality rule or unseen catalog values. Legacy/original07 remain exact.

Preserve corrected306vocabulary, indices, dimensions, token/layout/rule/Config/
schema signatures and numerical kernels. This is acceptance coverage, not a new
representation. Producer Python/native source/build identities change and must
be recorded. Preserve all prior failed/historical populations; never rebuild
journals or promote them under the repaired reader.

Add native build_info capability card_instance_guardian_acted=card-or-guardian-v1.
Require it in active corrected native reader preflight and configured refresh
native preflight, before allocation/GPU/Source work. Historical recorded native
identity validation does NOT retroactively require this marker; existing older
records keep their recorded producer meaning. Pure parsing/help remain ungated.
Matching deployed package/source verification is required; equal layout hashes
alone never prove older validators support the new inputs.

## Responsibilities and flow

Common and native validators own closed public representation validation; no
policy math, game rules or inference encoders change. Existing native-backend
preflight owns the additive feature check. Client keeps actor/view/revision/
history ownership; Source retains ordinary immutable original and failure gates.

## Verification

Anchor minimal public-event fixtures to retained evidence without reconstructing
failed journals. Test all10Guardian IDs for Acted/Set/Left, all296card Acted cases,
unknown/cross-namespace refusals, both viewers, exact categories/counts/order/
prefixes, Python/native agreement, and legacy/original07 unchanged behavior.
Old token/layout/Config goldens and eight frozen kernels stay exact. Build/test
source-matched wheels, independent correctness/conformance review, then rerun the
same actual protocol with fresh a02 parent and unchanged experimental conditions.
No broad numerical matrix or unrelated pilot is required by this validator fix.

## Alternatives and accepted limits

Unrestricted String acceptance loses closed unknown-ID validation. Restricting
Acted to either namespace alone contradicts retained public output. A separate
Simulator documentation clarification is useful but not a prerequisite to consume
already authorized public data; it becomes necessary before inferring which
effect occurred, unknown IDs or candidate legality. No such inference is made.
