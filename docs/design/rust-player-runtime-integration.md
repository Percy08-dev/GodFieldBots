# Rust player learning integration seams

Status: accepted bounded integration refinement for ADR0054 after independent
architecture READY. Implementation conformance remains pending. The main design,
record inventory, storage and supervision contracts remain authoritative.

## Configuration and identity

Ordinary Model3 declares the new mode in `run.player_runtime`; HPO campaign6 and
materialized spec6 use `runtime`. Each is the same closed declaration with
`mode=rust-player-runtime-v1`, `scheduler=tokio|fixed`, `game_concurrency`,
`player_worker_threads`, `max_batch_size` and `formation_wait_ms`. Materialize the
same declaration before hashing. Initial opt-in input supplies all six fields
explicitly; no provisional best scheduler/W is implied before measurement.
Validate positive integral counts and finite nonnegative formation wait, and
require C to equal normal `run.concurrency` or
HPO `operating.game_concurrency`. There is no W<=C restriction. The old
`run.inference_service` selector and new declaration are mutually exclusive.

The effective configuration records the complete materialized declaration.
The parent identity records the actual producer from supervision.md, including
its full Python/runtime/native/companion/sampling identities. The stable runtime
binding combines those two facts; all new outer records bind its digest. Raw
configuration cannot supply a producer or an already-admitted host capability.
Fresh host PID, nonce and invocation ID are observations under that stable
binding, excluded from the resume compatibility identity. Preprocessing selects
persisted feature encoding independently; it does not select the online player.

Config6 training uses Rust C/W/B semantics. Existing physical participant-worker
limits continue to mean what their legacy/evaluation consumers require; they
must not become a new-mode Rust-thread count or a training process reservation.
HPO6 evaluation explicitly selects the existing fresh runtime and independent
legacy GPU admission after the original training host has been reaped. The
existing evaluation eligibility and capacity rules remain applicable there.

## Original control and lock ownership

The existing common PPO coordinator remains the sole update/commit/schedule
owner. Add explicit concrete new-mode operations; do not let callers register
arbitrary operations or replace the training loop with a second implementation.
Historical operations and readers keep their exact closed-type dispatch.

Legacy coordinator timeout calculation uses UNIX observations and a fresh
invocation anchor. New-mode operations instead consume the authenticated
supervisor operation's original CLOCK_MONOTONIC bits and boot binding. Persist
that authority in the parent and require the same original milestones on resume;
UNIX fields are observations for this mode. All admission, execution, cleanup,
source, numeric, save and continuation work consumes the original authority.
A subsequent distinct HPO operation may bind its own declared milestones only
after the previous operation has finalized, within the original session finish.

The native per-game input must carry an absolute `cleanup_deadline`, derived
from the authenticated operation cooperative-close cutoff (no later than
`term_bits`, or an already earlier original cutoff). Both HTTP/replay cleanup and
terminal storage publication consume that same original cutoff. The component's current
`monotonic()+cleanup_seconds` calculation at each cleanup phase must be replaced
before supervised production integration: sequential cleanup phases cannot each
start a new allowance. Existing HTTP request_seconds still caps an individual
request within the unchanged absolute cutoff. This is an explicit GameSpec/facade
interface refinement; update component callers/tests together, without changing
historical persisted formats or giving new-mode production a fallback grace.

Common coordinator lock entry must distinguish a newly acquired original parent
lock from a supervisor-borrowed original OFD. The actual authenticated lock_ack
permits transferring the local descriptor by closing it WITHOUT LOCK_UN.
Historical run_lock finally remains unchanged for historical owners. New live
ownership derives from the real authenticated descriptor/stat/lock, not a
serialized record or an old owner reconstructed from a PID. Hold campaign and
parent custody through the exact prepared/exit/cold-finalization sequence.

## Same-process numerics and immutable publication

Local numerical facades call unchanged GameLearner and HpoLearner APIs under the
same serialized_torch lock used by inference. Initialization/update/restore
retain their intended Torch RNG changes. Inference construction/publication
preserves the caller's CPU and already initialized CUDA RNG; startup visibility
and Torch configuration occur once before admitted production GPU allocation.

Use the actual local OwnedModelSnapshot inference/resume banks and OwnedSaveAck
for Model4. Their protocol9 actor_metadata/capability remains legacy authority;
new runtime publication derives weights and provenance from the actual saved
snapshot without relabeling protocol9 as native capture11. Add explicit local
owner branches where existing registries/imports assume a remote process. Do not
manufacture RemoteModelSnapshot, remote ACK, GpuJob or actor retirement records.
Shared bank aliases count once through their actual last consumer lifetime.

The native runtime, real writer seal and new Source owner establish the new
capture authority. Existing Journal6/DTO3 and Python canonical persisted numeric
encoders keep their content contracts. Only matching accepted ACKs contribute
experience. The exact new collection/proof/update/registry kinds and versions
remain those in rust-player-runtime-records.md.

The live host can persist committed/paused work awaiting supervised finalization;
it cannot grant final parent or HPO evaluation eligibility. Actual actor/service/
writer/GPU settlement precedes Prepared; original host reap produces Exit; the
cold host independently audits originals, Replay, numerics and saved state before
issuing the new completion and, for exact final-D, final parent capability.
Healthy paused/incomplete work stays distinct from complete eligible HPO data.

## Verification

Require new/legacy selector and schema rejection tests, C mismatch and bool/
nonfinite/unknown-field refusal before I/O, exact producer/runtime resume binding,
and unchanged historical routes. Verify separate-open lock contention through
host exit/reentry and unchanged original deadline bits. Compare local and current
numerical owners on identical CPU/CUDA input, named weights/Adam/CPU+CUDA RNG/
PCG64/control state. Then run the approved real Model3/4 continuous2 and
pause1/resume2 originals/Replay/numeric/saved-state/process-tree checks and timing
matrix. Component tests do not close the learning or deployment evidence gates.
