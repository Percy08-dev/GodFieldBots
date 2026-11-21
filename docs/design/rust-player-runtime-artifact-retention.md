# Rust-runtime artifact retention and cold compatibility

Status: scoped engineering implementation accepted after two repair rounds.
Independent correctness and architecture conformance are READY. Genuine integrated
retired-checkpoint resume remains a Task6 requirement; the separate empty-engine
case has a reviewed conditional reachability proof.
Architecture READY: a475ae642873bc9cfa20ce9eac6f7f137c1bfa2943f736697e745b171a3d1927. Owner: root. Required before
public activation. Root delegated only the accepted bounded implementation;
actual deletion, independent reviews and final activation remain separate gates.
Raw collection originals and persisted numeric batches KEEP their existing
preservation policy. This concerns only files declared in generation.artifact_files.

## Existing policy, preserved exactly

Ordinary structured PPO uses the actual component recipe, not obsolete plain
lineage.retain: after commit it considers the predecessor; G0/every tenth/current
and acquiring service references remain protected. Eligible predecessor artifact
files other than inference.json may retire under the existing intent/receipt
contract. Use the actual declared artifact map rather than assuming an unsharded
weights/resume filename. Do not invent a new retention interval or pool policy.

Owned HPO uses existing pure protected_hpo_files_for_operating. Keep generation0,
resume_latest, inference_latest, decision-period boundaries, snapshot pool,
evaluation/cache/service/pending references and existing resume-generation rules.
Adapt only a truthful PURE data summary of new registry references to that pure
predicate; never manufacture an OwnedRunRegistry/actor9/old live receipt. Existing
legacy callers/policy and format meanings remain unchanged.

## Responsibilities, data and actual ownership

RustRunRegistry retains policy decisions, actual cache/reference owners and new
registry status writes under its actual current RunOwnership/HostAdmission.
A new private file-retirement owner in gf_training/rust_player_retention.py owns
concrete directory/file descriptors through staging/move/unlink. It depends on
existing safe file/scan/control primitives and the actual parent owner, not on
old learner processes or simulator logic. Factor existing data-only I/O routines
only when their complete ownership preconditions remain identical; do not invoke
_OwnedRetirement with a fake registry or blindly duplicate its authority layer.

Retire inference cache/banks and unused resume-file borrowers positively before
pruning their original files. Current/pool/in-flight/evaluation inference remains
protected. RustSavedPolicy and RustRestoredPolicy must retain ONLY the originals
needed for current serving after genuine save/restore validation; positively
retire unused resume originals before considering them deletable. Inference use
must not later recheck an intentionally retired full save ACK or obsolete resume
file. Actual Gprev lineage IDs remain history, not current backing identity.

The owner opens/fstats the exact owned parent/generation directories and original
file, verifies expected SHA and identity, and retains those handles BEFORE any
fallible transfer. Its original expected identity includes device/inode/size/
mtime/ctime. No JSON deletion intent alone ever authorizes filesystem mutation.
Use private staging in the SAME verified artifacts filesystem and dirfd-relative
rename/unlink, no symlink traversal or path reopening as authority. The complete success barrier is: actual rename/unlink, fsync the HELD source
generation and staging directories, verify the HELD staging
directory is empty, fsync the artifacts root, remove that SAME staging directory,
then fsync the artifacts root AGAIN,
then positively retire original-file/stage/source/artifacts directory handles
before publishing receipt and status=deleted. A close failure after destructive
work remains deleting/quarantined, never published success. Cold requires BOTH
the artifact pathname and that staging pathname absent, after verifying their
recorded identities and history.
All work uses the invocation's original remaining milestones; no cleanup grace.

## New strict data records, not live authority

Keep existing new registry5/8 top-level fields. Per-file rows retain
{sha256,status,deletion_intent,deletion_receipt}; statuses retained/deleting/deleted.
The new runtime's deletion_intent and deletion_receipt are SHA/path references to
immutable records, not old owned-executor authority. Proposed new family:

intent1/kind rust-player-artifact-retirement-intent-v1 exact
{schema_version,kind,run_id,operation_id,invocation_id,retirement_id,generation,file,sha256,original_identity,
staging_basename,parent_identity,source_directory_identity,stage_directory_identity,
runtime_binding_sha256}.
receipt1/kind rust-player-artifact-retirement-receipt-v1 exact
{schema_version,kind,run_id,operation_id,invocation_id,retirement_id,generation,file,sha256,intent_sha256,
original_identity,runtime_binding_sha256,completed_unix}.

Records live at parent-relative retirement/artifacts/{retirement_id}/intent.json
and receipt.json. retirement_id is a fresh32-hex ID; invocation_id and operation_id
are the ACTUAL current admission IDs, distinct from that file-operation ID.
staging_basename=.prune-{retirement_id} is an actual private directory under the
held artifacts directory, created/opened/fstat'ed before intent. original_identity
is [device,inode,size,mtime_ns,ctime_ns]; directory identities are [device,inode].
All counters/identity numbers are nonnegative integers, never bool; SHAs are64
lowercase hex; completed_unix is finite nonnegative. file is one exact declared
generation artifact basename. parent_identity denotes the actual artifacts root,
source_directory_identity its generation directory, stage_directory_identity the
held staging directory. Registry references are exact {path,sha256}. Write intent
and registry status=deleting before DESTRUCTIVE file mutation (stage-directory
creation precedes that intent); then original-handle move/unlink/fsync,
immutable receipt, then status=deleted. Record fields are derived from actual
owners/operations. Same encoded bytes/expected hashes bind the cold reader.
No backward immutable-hash cycle: registry is the mutable current ledger;
generation/update/source manifests remain immutable and retain original file SHAs.
Exact paths and field/type checks above are mandatory before issuance.

Failure before movement may close the original owner positively without deletion;
after a move/unlink or uncertain close, retain actual staging/directory/file
owners and registry deleting state, mark whole parent ineligible/quarantined.
Once a deleting intent is published, even an unmoved failure does NOT silently
reset it to retained. Do not replay an intent from JSON, restore a guessed file, report deleted from
absence alone, unlock an uncertain original OFD or release encompassing loans
early. Existing explicit operator recovery remains separate.

## Reader and resume compatibility

Extend new-runtime cold/resume registry validation to accept deleted files ONLY
when the immutable intent/receipt hashes, run/generation/file/SHA/original identity,
matching operation and expected physical absence all agree. Matching operation
means the invocation that ACTUALLY deleted the file, not necessarily the current
reader. Resolve retirement invocation_id in retained Run/frozen invocation history,
check its stored operation/admission paths and exact SHA bindings and the matching
Prepared/Exit/completion chain where historical. Match operation_id/run/runtime
to that record. Historical receipts are DATA only and never recreate a live old
HostAdmission/borrower/deletion capability. Current-invocation retirements bind to
the actual current frozen state/Prepared/Exit pair. Missing/foreign/failed history
refuses import rather than rebinding to today's operation. Reject deleting or
uncertain entries, foreign/missing receipts and replaced objects. Validate only
retained inventories plus exact retirement records. Use held directory identities;
absence is a consequence check, never original mutation authority.

Final required inference/resume files and protected pool/evaluation files must
still exist. Full cold Raw/Replay/numeric auditing still covers EVERY committed
update; no raw/numeric deletion or invented numeric-pruned branch is introduced.
Old unneeded checkpoint payloads are validated through their recorded retirement
facts, not reconstructed as live snapshots. Current saved-pair/optimizer/RNG checks
remain full for the retained final checkpoint. Prior B retained-only records keep
their existing meaning; no old record version is relabeled or globally incremented.

## Verification and acceptance

TDD with actual owned temporary files/dirfds/RunOwnership, no fake Source/Parent/
actor capability. Test ordinary generation boundaries and HPO latest/periodic/
pool/current/in-flight/evaluation protection, sharded maps, inference-only owners,
real move/unlink+receipt, replacement and failure windows before/after intent/move/
unlink/fsync/receipt, uncertain close/positive retry and competitor exclusion.
Use multiple generations (at least enough to retire an old checkpoint), verify
cold import of retained/deleted inventories and corrupt receipt/deleting refusal.
Prove no Raw/numeric path is ever a retirement target. Keep budgets/funders alive
through their actual last data/FD consumer; no additional whole Source allowance.

Root architecture review must close this proposal before dependent code. After
implementation, both correctness and conformance reviews and appropriate scoped
checks are required. Public entry activation, actual Task6/7 and learning quality
are not accepted by this retention design.
