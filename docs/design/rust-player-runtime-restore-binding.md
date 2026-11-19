# Held Model3 resume metadata binding

Status: accepted narrow preimplementation design; implementation/conformance pending.
Independent architecture READY on original proposal SHA
3ae69c1317623b01743a2af937ceb8f9e0ef246eb4e749fd53df74302416aa49. Owner: root.
Extends the accepted [operation/restored-owner design](rust-player-runtime-operation-v2.md)
and [learning integration](rust-player-runtime-learning.md), without changing
checkpoint formats, PPO mathematics, resource estimates or publication order.

## Concrete gap

RustRestoredPolicy holds the actual previous saved files and resume.json SHA.
Model3 LocalGameLearner currently calls unchanged Learner.restore(directory),
which validates and stages a complete resume, but _read_resume reads resume.json
without an expected_digest. A separate pre/post stat or hash check does not
bind the actual parsed metadata bytes to that already-held F1 digest.

## Bounded change

Add keyword-only expected_resume_sha256: str | None = None to
src/gf_training/structured_numeric.py::Learner.restore and the private
src/gf_models/structured_artifacts.py::_read_resume/_restore_resume path.
Learner.restore forwards it to BOTH its initial complete read_resume validation
and its final restore_resume call. _restore_resume forwards it to _read_resume;
_read_resume supplies expected_digest=expected_resume_sha256 to the existing
read_json for resume.json, so the actual parsed bytes are hash-bound before
resume header interpretation and _apply_resume_state. Do not replace the
existing read_json validator, duplicate its hash reader or loosen its limits.
Other wrappers/callers retain their current defaults and format meanings.

LocalGameLearner forwards the genuine RustRestoredPolicy's already-held resume
SHA at the actual Model3 restore call. Preserve actual file/bank owner references
through final validation and state publication, and retain the previous saved
snapshot lineage identity separately from new read-only inference-bank identity.
A caller-supplied digest alone creates no restored-policy, Source, save-ACK or
publication authority. Do not fabricate OwnedSaveAck or resave the old generation.

All existing complete-shard, weights/optimizer/RNG/PCG validation, staging,
resource bounds, inference digest checks and final assignment remain unchanged.
Mismatch raises the existing controlled ContractError before applying parsed
resume state to network/Adam/RNG; no fallback or second unbound parse. Initial
fresh learner/staging allocation behavior remains the existing implementation.
This supplement does not claim to eliminate unrelated staging RNG side effects
or rewrite existing restore semantics. A failed new invocation remains failed.

## Ownership and verification

After independent architecture READY, the existing sole bot_implementer owns the
two named existing modules in addition to its already-owned rust_player_learning/
registry code and necessary focused tests. Root owns docs, reviews and commits.
No other writer may edit those modules concurrently.

Before implementation, add a failing direct-read/real saved Model3 regression
that changes an otherwise valid resume manifest while keeping its structural
validity and demonstrates rejection against the original held SHA. Verify the
same expected digest reaches the second materializing read, not only the first
precheck; a mutation injected between reads must not publish restored state.
Verify the matching digest still restores named Adam state, optimizer steps,
CPU/CUDA RNG bytes and PCG64 under the existing tests, and omitted kwarg preserves
old callers. Use CPU-focused controls for this seam; actual CUDA live resume
remains mandatory Task6 work. HPO Model4's already accepted snapshot/manifest
binding remains separate and unchanged. Run scoped static and artifact/restore/
common-PPO regressions, then include this seam in final independent B reviews.
