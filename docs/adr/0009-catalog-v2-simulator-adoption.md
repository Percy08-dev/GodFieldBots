# 0009: Adopt corrected Magical Stick semantics and catalog v2

- Status: Accepted
- Date: 2026-10-28
- Owner: root
- Review: independent design, correctness and final conformance READY
- Implementation: complete; 501 ordinary tests and actual eight-case contract
  matrix pass. Fresh pilot/PPO/fitting evidence tracked separately.

## Decision and scope

Implement the user's approved server-adoption plan. Pin Simulator
`ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6`, its current nine normative snapshots
(the prior eight plus spec01), and catalog `public-ja-v2`, schema 1, SHA-256
`481ddfb0c068a1d2aaaf2dbcb6a5b4bd04396dfcc5c995e14e47b71222b6a51d`.
The worktree is clean. Use the existing prebuilt binary after behavioral
verification; its hash is provenance, not proof of the source used to build it.
Do not modify/build Simulator or repair its Python environment in this task.

Spec09 section13.1.1 now authorizes existing explicit self/opponent singleton
forms at zero and positive MP. Omitted/null defaults to opponent. All MP is
consumed after any miracle costs; opponent attack zero still waits for response.
Simulator owns payment, effect ordering, refill and RNG. Bots retains generator
v3, scorer, encoder features/signature and existing schemas. Do not add mixed
sequences, hidden-identity inference or rejection-driven masks. Dream uses own
perceived definitions and same-revision scoped atomic hints as before.

Common owns revised compatibility constants. HTTP retrieves the catalog using
the pinned catalog version rather than a v1 URL literal, retaining strict info,
catalog hash and field validation. Model artifacts already bind revision/catalog:
old LightGBM schema2 and MLP schema1 artifacts are rejected by those identities,
without inventing a format migration or changing their JSON shapes.

AttestedClient owns its sticky connection failure. After an asynchronous probe
returns and before publishing cached identity/readiness, recheck poisoned state.
A competing failure/cancellation must defeat a late valid reply. No new client,
lock framework, retry policy or Simulator-client dependency is introduced.

## Compatibility and historical evidence

Archive the old pinned reader wheel with SHA-256 and verify the old complete
192-game pilot in an isolated old-reader process. Its original manifests,
catalog v1 and source hashes remain unchanged; this is historical verification,
not current training eligibility. New readers reject old source semantics,
checkpoints and resume identities. Never resume or harvest the failed seed17
parent. Catalog v1 availability does not restore old engine semantics.

Update current presets, documentation and issue0004 status to adoption-verification
pending; retain the issue's original failure and old ADR reasoning. Root owns
configuration/docs/commits; implementer owns Python/tests. Independent correctness
and conformance reviews precede acceptance.

## Verification and experiment rollout

Add public-fixture and real-server coverage for MP0/positive MP target forms,
response/no-response boundaries, full payment/event ordering, refill and replay.
Reproduce the saved failed public action prefix on a fresh server via normal
attested HTTP and journals, without adaptive retries. Verify catalog v2 and
preserved v1 endpoint bytes, old source/model/resume rejection, unchanged encoder
signature, and Dream/perceived input boundaries. Gate-race tests use barriers;
check no cache refresh or Create/Act after concurrent failure/cancellation.
Cover repeated-seed exhaustion, disabled idle and normal stdin-EOF cleanup.

Run pytest, Ruff, mypy, opt-in integration and isolated wheel/base/native checks.
Then commit frozen code/config, run fresh fixed12-game evaluation and192-game
pilot, recompute original-source feasibility, and run continuous two-update
CUDA training plus explicit pause/resume equality. If all pass, run fresh
`runs/ppo-comparison-v2/train-seed17`, `train-seed29`, `train-seed43` sequentially.
Keep original7200s wall/5400s admission/7140s execution,1000update,10GiB soft
storage/1GiB reserve and global2game limits. No outcome-selected replacement.
Only three complete parents authorize final64games perpolicy validation and
the frozen dataset, followed by observation/history LightGBM each seeds17/29/43.
Generation cost remains separately deduplicated; finaltest is not run.

Any new failure preserves all evidence and stops dependent work. Successful
component tests do not imply training quality or successful formal experiments.
Historical prefix reproduction does not require identical old intermediate
MP/Event/RNG values: payment semantics changed. If the old prefix no longer
reaches the case, retain that diagnostic and construct a fresh bounded public
fixture; never alter the original evidence or pretend it follows old semantics.

## Alternatives and consequences

Restricting zero-MP candidates is unnecessary under the newly published rule.
Changing generator/schema versions would misrepresent unchanged candidate/wire
forms. Broad cross-version readers would create avoidable migration complexity;
the archived reader preserves historical checks without authorizing old data
under new rules. Existing OS/filesystem timing and measured-memory debt remains.
