# Zero-MP dynamic weapon singleton target contract

Update 2026-10-28: Simulator `ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6`
spec09 section13.1.1 resolves the missing singleton semantics and also changes
full MP payment. Bots adoption is tracked in ADR0009 and
[the adoption record](../verification/catalog-v2-adoption.md). Original evidence
and the failed parent below remain unchanged and ineligible.

- Status: Historical blocker resolved by ADR0009 adoption; old failed parent
  remains ineligible. New-contract experiments have separate evidence.
- Date: 2026-10-28
- Owner of clarification: separate Simulator task, not a Bots repository change.
- Decision: root accepts advisor recommendation C. Independent architecture
  investigation confirms the missing public rule. No legality patch or probe
  has been added to the policy.

## Observed failure

Code `9faeac5d14b279b3ea4be5c57dc0c8b958b5ad37`, clean worktree; Simulator
contract/source revision `860d8b92db4ce82012e040510f221b52241f28aa`, the eight
ADR0007 normative hashes, catalog `public-ja-v1` and journal schema 4.
Prebuilt binary SHA-256
`608805f07c59164cf1cac18ee9e7a73a13f02b5aa0a5b31edb216070903be41c`;
its build revision remains unverified.

The first prescribed seed-17 PPO parent completed 24 updates and 3,800 accepted
learner actions, then failed while collecting batch 24. Of 400 attempted games,
399 finished naturally and one was interrupted by a rejected action. The whole
parent is failed/quarantined; all its data is ineligible. Seeds 29/43, final
validation, frozen dataset construction and real LightGBM fits were not run.

Original evidence, retained unchanged:

```text
runs/ppo-comparison/train-seed17/run.json
runs/ppo-comparison/train-seed17/collections/batch-000024/
  b000024-p3-seat0/failure.json
  b000024-p3-seat0/player-2.jsonl
  b000024-p3-seat0/replay-player-2.json
```

Diagnostic GameSeed: `ffe41bd1-f234-5e4f-b5fd-4d6e7dcf3954`.
Game ID: `01M604J1YSKYVZAPV85Z53MSDK`. These are management/reproduction
identities, not policy inputs.

At revision 15, viewer/current player 2 had MP 0 and Dream false. Its own visible
instance 13 was `magical-stick`, kind weapons, ability `atkBy2xMP`, no printed
attack. The public effect text makes attack twice current MP. Same-revision
actor ActionOptions listed instance 13 as usable and global targets `[2,1]`.
The heuristic opponent submitted this stored candidate:

```json
{"type":"play_sequence","ordered_elements":[{"source":"hand_artifact","instance_id":13}],"target":2}
```

The response was HTTP 422, `INVALID_ACTION`, `rejection_code=invalid_target`,
with one invalid attempt and revision still 15. The client stopped; it did not
select another candidate, probe target forms, or manufacture a terminal label.
The rejected participant decision has no positive acknowledgement.

[Frozen participant-visible input](0004-zero-mp-public-input.json) preserves the
exact delivered input and submitted action, with rejection diagnostics outside
the policy input. It contains only that participant's permitted view/catalog/
events/options, and is explicitly diagnostic, not training data. SHA-256:
`3ab588cce1fba47b3be9f3638ffa22c4acbef9bb5810e00fd3861b0df27559e7`.

## What is known and what is missing

Simulator09 §13.1 says single-target attacks have targets and weapons may
self-target in some cases. Simulator03 §11 retains authoritative action-specific
target/resource validation. Simulator07 §§13/21.1 explicitly say typed syntax
and atomic availability/global targets do not establish complete legality;
rejection details must not become policy features or masks.

Own permitted catalog and MP establish perceived attack `2 * 0 = 0`; they do
not specify whether a zero-MP singleton requires target absence/null/self/
opponent, or whether it is unsupported altogether. A self-target rejection
does not prove anything about the other forms. ADR0005's v3 singleton amendment
was verified at MP 10 and extrapolated targeting beyond that evidence.

This is not evidence that ActionOptions violates its atomic-hint contract, nor
proof of an engine defect. Normative clarification and regression evidence may
be sufficient without a wire, schema or engine change.

## Separate Simulator task

Publish the intended legal singleton forms for MP 0 and positive MP, including
omitted, null, own and opponent target; specify consumption, turn progression,
and whether/when a Response is created. Preserve the distinction between
perceived definitions and actual behavior under Dream-compatible appearances.
State which guarantees atomic hints provide in this case. Verify accepted forms
and unchanged State/RNG on rejection in Simulator-owned regression tests.
Do not infer intended live-game behavior solely from the observed rejection.

Then Bots can review the published contract, pin its actual revision/spec hashes,
derive candidate tests (including the sole-supported-card case and existing
MP-10 behavior), perform one bounded normal-path integration reproduction, and
start wholly fresh formal parents under the prescribed budgets.

## Alternatives and compatibility

A deliberately narrower capability subset could omit **perceived zero-attack
dynamic singletons** using only permitted public inputs. That is not a claim of
illegality. It can leave an empty candidate set when no other supported action
exists, and Dream's perceived zero attack need not equal the actual effect.
Such a fresh restricted experiment is permissible but would not establish the
missing semantics. Root declines to substitute it for the planned experiment.

Adding a guessed targetless form or a rejection-derived mask is unsupported.
No SequenceOptions or adaptive rejection retry is authorized.

No generator change is made, so v3 and the original 192-game pilot retain their
identities. A later omission/completion change needs generator v4; unchanged
JSON shapes alone do not require a journal-schema bump. New Simulator normative
snapshots must be pinned even if wire versions stay unchanged. Historical v3
readability, if required, needs explicit reviewed compatibility, not relabeling
or regeneration; the old pilot remains evidence for v3 conditions only.

The failed parent must remain failed. It cannot resume after changed identities,
be promoted, or contribute healthy earlier batches to offline fitting.
Any later formal attempt needs new directories and an explicitly frozen new
three-parent source set; do not overwrite this first prescribed attempt.

Failure-boundary assessment:
`runs/assessments/ppo-seed17-failure-20261028.json`, SHA-256
`83d40a2e8ed2bd6d63608d37d4e670b7d79c3d54ee04300af4b69c8a5d312e98`.
Root verified early-batch ingestion, generic export projection, formal checkpoint
qualification and failed-run resume all reject this actual parent. No update or
generation 25 or numeric input for the failed batch exists.
