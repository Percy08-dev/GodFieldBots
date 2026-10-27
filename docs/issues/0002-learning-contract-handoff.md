# Separate Simulator tasks: complete the public action-construction contract

Status: proposed handoff; no Simulator change is authorized inside this Bots task.
Reference revision: `44a26e3a16b808e826deb332fa72acfc5037d3ef`.
[Diagnosis and reproduction](0001-public-observation-legality-investigation.md)
contain exact binary/source hashes, public trace and observed results.

Update: task B is resolved for atomic availability by the user's updated
specification07 §§13/21.1 and Bots ADR0005. Actor/same-revision ActionOptions
are now authorized and used; no extra opponent catalog lookup or new pending
attack field was required. The original alternatives below are retained as
history, not current blockers. Task A's normative text synchronization remains.
The later [pending-buy metadata task](0003-buy-choice-control-metadata.md) is
separate and still blocks a clean feasibility pilot.

## A. Clarify protected discard/sacrifice items

The engine excludes Sun Amulet and Attract Danger in addition to weapons, but
normative gameplay spec 09 §11 names only the weapon exclusion. The generic
`WeaponCannotBeDiscarded` error also describes protected nonweapons as weapons.

Primary evidence now supplied by the user: [developer announcement,
2022-10-31](https://x.com/guuji/status/1586989576039825410), confirmed through
Twitter's public oEmbed. It explicitly prohibits selecting Sun Amulet and
Dangerous Mortar for **Discard**. The intended Discard rule is therefore
established; synchronize the public normative text and clarify the diagnostic
error. The announcement does not explicitly mention Sacrifice: verify that
separate behavior from authoritative evidence rather than extrapolating.
No Discard engine change is justified merely by the earlier misleading error.
Do not derive a rule from rejection diagnostics or change it to make a run pass.

Acceptance: the fixed saved public prefix has documented, unambiguous expected
behavior; tests cover protected/nonprotected items, mixed selection and unchanged
state/RNG on rejection. No GameSeed/private state/action options enter policy
inputs. Apply Simulator's rule/catalog/schema compatibility policies as required
by the chosen actual semantic change; do not silently change an immutable version.

## B. Specify permitted incoming-defense context

Status: superseded for atomic candidate construction by authorized ActionOptions
and ADR0005. The restrictions below describe the previous contract and remain
relevant only to direct catalog expansion, which was not introduced.

Public defense compatibility depends on incoming attribute/kind. Current
Observation lacks this context; visible `attack_declared` lacks attribute.
Opponent `artifact_used` events carry a definition ID, but spec 07 §21.1 only
permits catalog lookup for own-hand/learned observation IDs. The three diagnosed
incompatible defenses cannot be corrected by expanding policy catalog access
under the current contract.

Possible owner decisions (alternatives, not invented requirements):

1. Expose explicitly reviewed player-visible incoming attack context in a new
   versioned Observation/event contract, after checking official visibility,
   mixed/order-sensitive effects, Dream/Fog and trade/non-damage responses.
2. Explicitly permit interpreting relevant already-visible event definition IDs
   through the pinned public catalog, with scope/visibility rules. Determine
   whether those events suffice for composed/modified attacks; do not claim a
   single-card lookup solves every response context.

Neither alternative may expose actual unrevealed identities, opponent hand,
RNG/future outcomes, ActionOptions/SequenceOptions, or engine legal-action masks.
Typed metadata remains syntax, never a legality oracle.

Acceptance: indistinguishable permitted player inputs do not acquire hidden
legality information; tests cover visible versus concealed events, natural
attribute combinations and unsupported/missing context. Publish the exact
learning-input/catalog scope and version compatibility. Bots then pins the new
revision/schema, obtains its own design review, implements public-rule candidate
construction and reruns the fixed evaluation and clean feasibility pilot.

## Bots work deliberately not performed from diagnostics

No opponent-card attribute table, extra catalog definition, protected-item mask,
or rejection-driven retry has been added from diagnostic findings. The current
implementation uses newly authorized ActionOptions, and normal protected Discard
is based on the user's primary developer announcement, not the error body.
Existing candidate-space limitations and failed-run eligibility remain explicit.
Diagnostic reproduction success is not a passed learning gate. Independent model/toy verification may
proceed separately under the user's plan, but these issues prevent claiming
that ordinary Simulator learning/comparison prerequisites are established.
