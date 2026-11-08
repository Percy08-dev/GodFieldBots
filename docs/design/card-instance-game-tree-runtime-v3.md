# Corrected qualified GameTree3 runtime — Root bounded design v3

Proposed for architecture review before implementation. This implements the
remaining ADR0024§18.2 runtime interfaces; genuine formal training/real fitting,
comparisons/HPO/final-test execution remain deferred. MAINc96252a preserves
accepted NN3/baseline gameplay and Tree3 component scoring. No Simulator change,
new dependency, second game loop or inference/training stack is introduced.

## Requirements, dependency and ownership

Public corrected actions/model3 LightGBM configuration must dispatch before
legacy/default NN parsing. Reuse ConfiguredTreeModelV3, real18 positions,
participant-structured-v2 and current Tree3 feature/category signatures, CPU
chunk256/thread1 defaults, existing numeric-cost/resource calculations and
corrected public generator. Model3 is input-contract identity, not a new learner.

Root owns config presets/docs/integration. One implementer owns the explicit
Config/worker/runtime/Journal/Collection hooks and necessary tests after review;
the Dataset implementer owns shared qualified artifact/record context first.
Native owner separately updates corrected RAW parsing under an accepted schema.
Shared files have selective handoffs, no concurrent overwrite. Keep legacy
Tree2, NN3, baseline, fixture and score-only shapes/hashes exact.

Dependencies: pure own-policy validators depend on common contract/layout and
Tree config/record validators; model/controller artifact custody depends on
paired artifact inspection; runtime/transport consumes only safe own-policy
metadata and kernels. Training/journal composition records metadata. Neither
models nor simulator-client imports Dataset/parent Source machinery. No reverse
CLI import from training. Native RAW implements the same pure header/metadata
contract without granting source/fitting eligibility.

## Public Config and controller pair ownership

Route model.version3/model_id lightgbm to ConfiguredTreeModelV3 for both model
and opponent. Require corrected actions3/server profile/current rule signature
and no synthetic capacity; real max_positions<=18. Existing default fields stay
exact. Fit permits checkpoint=None through the qualified fresh Dataset workflow;
game/evaluation requires an explicit checkpoint resolving to a healthy trained
GAME record-context pair. Component/toy/diagnostic/legacy/ordinary parent origins
reject before model/worker/Simulator allocation. NN/PPO learner type checks remain
explicitly NN; Tree is never an accepted PPO learner or converted teacher.

Add a retained game Tree artifact owner analogous to RetainedGameArtifacts,
composing the SAME qualified Tree3 pair inspector/resolver/SealedScan. Validate
full architecture/feature/input/action/rule bindings, trained real source kind
card-instance-formal-ppo-originals and observed-natural source origin, actual
request/completion/inference/model hashes and descriptor before allocation.
Retain the actual pair/snapshots through episode reap and collection publication.
Immutable execution-only inference/resource overrides follow existing Tree3
rules; actual-Booster architecture/feature/origin binding cannot be relabeled.

The controller keeps complete request/completion/Source references. Export only
a bounded own-policy declaration for child/journal. Its proposed exact keys:
version, checkpoint, expected_manifest_digest, model, inference, resources,
action_config_hash, tree_descriptor, feature_signature, source_kind,
source_origin, training_pair.

version is string3. checkpoint names OWN artifact files, never training/Dataset
directories. training_pair is exactly {request_sha256,completion_sha256}; no
directory or original Source controls. Model is the full existing architecture
record; inference/resources/descriptor use existing pure schemas. Inference4
keeps the existing immutable request/provenance reference and trained real
origin. Completion outputs hash inference.json/model files; inference NEVER
embeds completion_sha256. The completion digest appears only in the controller-
validated safe training_pair declaration and subsequent journal/collection
metadata. The child treats completion_sha256 as an opaque bound reference,
not independently attested pair authority, and cross-checks the request hash,
source, architecture and features against its own local inference files.
The inference file's own safe provenance fields may be validated locally; do not
send raw Dataset manifests, parents, labels, numeric training features, GameSeed,
canonical history or other-private input. This declaration grants no Source or
fit authority, and child success cannot replace the controller's retained pair.

Policy label: lightgbm-tree-v3-<inferenceSHA>-<executionDigest>. executionDigest
uses this exact safe own-policy declaration except checkpoint path, permitting
the existing sealed private own-artifact projection without relabeling weights.
Controller/child/journal recompute the same label. Use a new closed validator
for GameTree3; never reinterpret component scoring binding as game permission.

## Shared participant and private child handle

Extend SAME corrected CardInstanceParticipantWorker/Environment/factory path,
not the scores-only facade's public API. Protocol3 init retains existing fields
and adds policy_options only in this distinct lightgbm3 branch, as for NN.
NN/baseline init bytes stay exact. Safe options discriminator is lightgbm+version3
with the exact game binding, caps, descriptor-derived numeric costs and policy
label. Source qualifier checks remain controller-side before spawn.

Child loads only sealed own inference/model files under the sanitized declared
managed cap, validating full metadata/architecture/descriptor/hash/pair digest
agreement before Booster construction. It uses a PRIVATE own-model handle,
never LoadedGameTree/TrainingBinding or a class implying paired authority.
Reuse the existing component child own-model validation/score kernel and shared
PreparedTreeFrameV3/_score_chunks/_select mechanisms by explicit closed context;
keep component-origin validation exact. There is no source-validation callback.

Participant state retains the SAME own EventBank/revision/interval continuity,
generator RNG stream and model resource admission. Generate the full admitted
ordered table, tokenize/score all rows in chunk256 and choose first argmax tie.
Return existing Selection shape with table_hash/index/policy_version and BOTH
logprob/value None. The controller restores the Action using original own frame
references; no inferred IDs/definition references, new action or retry payload.
No separate Tree gameplay transport or engine behavior is implemented.

Before raw/request/reply/model/token allocations, admit all controller/child
copies/JSON header sizes, active bank, full candidates and model workspace under
unchanged caps (arrays/shards64MiB, workers<=4/streams<=2/GPU12GiB; Tree usesCPU).
Keep one operation in flight, original finite session and clipped10s operation
deadline, sticky admitted failures, no late success and actual owner retention
until confirmed reap/IO-thread retirement. Use existing close health semantics;
uncertain cleanup prevents collection success and releases no live lease.

## Journal6, Collection5 and native RAW

Journal6 has a distinct closed Tree own-policy metadata branch. Header model_id
lightgbm, model_version3, generation_schema=None, corrected input/layout/rules,
aggregate_window/max_positions from Tree encoder and safe declaration above.
Selection requires logprob/value None; finite NN3 and baseline branches retain
their existing conditions/bytes. Store safe own model options, own inference
file identities/hashes and pair digest references only, not raw Source controls.
Use one pure validator for worker/controller/journal metadata agreement.

Collection5 adds tree_inputs ONLY when Tree roles exist, exactly
{input_contract:'player-card-refs-v2',policies:{label:safe_tree_metadata}}.
Existing neural_inputs remains NN-only and absent/present exactly as before for
old populations; baselines unchanged. model_version3 when either NN3 or Tree3
is present, otherwise1. Validate every registry role through its typed branch,
including mixed NN/Tree evaluation. No evaluation-source or formal qualification
comes from collection flags. Formal NN16/final64 Sources keep their prescribed
NN learner/final policy+opponent bindings and reject Tree learner substitution.

Native OriginalHeader::new_corrected dispatches lightgbm3 through a closed Tree3
model/options/own-metadata parser. Reuse Tree2 parser structure with explicit
current Tree3 encoder/layout recipe; old defaults/bytes remain exact. Typed RAW
and selection validation recognize model_id lightgbm/model_version3/null stats,
without accepting a neural metadata record or fixture-origin game declaration.
Raw episode/bank/token/correspondence/export formats remain the shared current
ones; no numeric/PPO authority or hidden fields are added. A separately gated
native capability/method/signature check rejects old native before processing
new Tree3 metadata. Exact marker and final tree_metadata field placement must be
pinned with native owner before that parser implementation; no silent fallback.

## Verification, compatibility and limits

Meaningful checks: pure options/schema/origin/label cross-rejection before any
optional/native/model allocation; trained fixture/toy refuses public factory;
safe child packet excludes Source/labels/parents; SAME pair changed during
startup/episode/close invalidates; whole18/32768cost/candidate admission; own
view/prefix/order/opaque reference restoration; null Tree vs finite NN stats;
mixed role collection; cancel/timeout/late close/lease/reap; native Tree3 RAW
Python parity and old Tree2/NN3/baseline literal regressions. Fixed-table scores
use existing accepted kernels, not a second algorithm. Current component
288episode fit/worker/native pipeline and fixed512 actual-input score panels
remain their explicitly unqualified evidence.

Controlled metadata/kernel/lifecycle primitives may exercise this code, but no
fabricated complete formal parents, runtime attestations, qualified artifact
pair or Source-positive fit is allowed. Genuine game Tree3 positive execution
requires future qualified real data/trained pair and remains NOT RUN in this
acceptance, as formal experiments/real fitting are user-deferred. Fail-closed
public gates and complete guarded interfaces are required now; a raising stub
or fixture promotion is not completion. Both independent correctness and final
architecture-conformance reviews plus project/package checks follow code.

## V2 hash dependency decision

Root accepts the architecture finding: request/provenance → inference/model →
completion → runtime declaration/journal/collection is acyclic. Controller pair
custody validates completion; no child declaration or inference field supplies
Source or fitting authority. V1 is preserved byte-exact as historical proposal.

## V3 bootstrap and exact metadata decisions

Tree alone supplies one sanitized positive decimal managed-byte argv value,
validated under the existing TreeResources integer range BEFORE the first
protocol3 frame header/body. Use shared admitted framing and complete parent
encoded/decode/copy/model preflight. The decoded init must be lightgbm/version3,
and its resources.max_owned_bytes must equal the argument before native/Booster
construction. Baseline/NN carrying that argument reject; their zero-argument
paths/defaults/bytes remain exact. The argument is resource control, not pair,
Source or game permission; keep shared transport/reply-hook ownership.

Journal6 uses optional key tree ONLY for model_id=lightgbm/model_version=3,
mutually exclusive with neural. Its payload is the same safe Tree external
metadata record used by controller/Collection5: exact keys input_contract,
policy_options,inference_manifest,files. input_contract is player-card-refs-v2.
policy_options is the exact own-policy declaration defined above.
inference_manifest is the actual bounded own inference4 document, including its
request-only trained provenance and real source classification. files is an
exact sorted list of own artifact entries {path,sha256,identity}, using existing
SealedScan basename/checksum/five-integer stat identity semantics. It contains
only inference.json and model.txt. No raw Dataset/parent/training request or
completion body is exported. Actual completion health remains controller-owned;
training_pair digests are opaque binding references in the child.

Protocol3 init keeps its existing top-level shape plus policy_options in this
Tree branch; no raw tree metadata or source controls are added to init. The child
reads/validates bounded own files and hashes under its resource cap. Journal/C5
safe metadata carries their sealed source identities. Collection5 tree_inputs
contains {input_contract,policies} with those exact metadata records. No new
keys appear in existing baseline/NN collections or headers.

Native/J6 parser implementation is gated on this exact discriminator/keyset,
source-kind/origin/request/model/feature agreement and null Tree statistics.
It adds no completion digest to inference and no source permission to metadata.

## Final design dispositions

Architecture reviewed v3's metadata delta READY. The own-policy enumeration is
exactly12 fields (the explicit keyset above governs). Root pins additive native
build capability card_instance_tree_raw=journal6;lightgbm3;inference4;codec2,
together with existing corrected signature/ABI/API checks, before native Tree3
RAW processing. Python gameplay does not gain a native runtime dependency.
Capability publication still requires the new RAW checks and both final reviews;
metadata never supplies Source/fit authority. This closes the schema/literal
pre-code choice only, not implementation or activation acceptance.

## Proposed retained-input publication guard

The existing collection publisher gains an optional retained-input guard,
default None with the existing baseline/NN path unchanged. Tree composition
admits and retains the actual qualified pair owner first; its bound guard is a
final integrity check, never callback-based Source or pair permission. Run it
before commit and inside the existing publish/revoke try boundary after staged
output verification AND successful descriptor release, before committed=True.
This covers mutation during release as well as writing. A guard/release failure
revokes known owned outputs, using existing fresh-descriptor inode recovery when
release already completed, and retains primary/quarantined owners. Subsequent
owner release is memory-only. Original deadlines/caps remain. This narrow
cross-module handoff passes architecture DESIGN READY before its code change.

## Proposed exact controller custody counter

A narrow readonly RetainedGameArtifacts.retained_bytes accessor checks owner
liveness and returns actual sealed-scan retained bytes without I/O, copying or
new authority. The Tree composite controller sums its actual paired-scan and NN
scan custody, and charges that retained ownership plus explicit guard-sweep
scratch before guards, init/request/reply decode and model/worker allocation.
This preserves the existing2GiB Tree local/controller limit and avoids guessed
NN bounds or uncharged sibling custody. External reservations are not added
again. Existing NN/baseline behavior, hashes and defaults remain unchanged when
this accessor is unused. Architecture DESIGN READY precedes the accessor change.

## Runtime review findings to close before acceptance

Independent correctness review confirms two blocking implementation findings.
The shared participant superclass releases its reservation after child reap even
when an I/O thread remains alive; this violates the existing ownership boundary
for NN/baseline/protocol2 as well as Tree. Correct the shared failure path while
preserving healthy packet/config bytes, actual cleanup owner and retry safety.
The composite retained-input guard also omits its NN owner, allowing a mixed
NN/Tree file mutation during output release to escape the final integrity check.
Guard every retained input owner with admitted scratch and original deadlines.
Both controlled counterexamples use actual file/cleanup primitives and grant no
qualified Source/pair evidence. These findings remain blocking until fixed and
independently rechecked; Tree-only workarounds or overall failure flags do not
waive individual ownership and publication requirements.
