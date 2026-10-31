# 0021: Measured local improvements to single-worker preprocessing

- Status: Stage 1 accepted, 2026-10-31, after independent design/source/result
  correctness and conformance READY. Stage 2 local design READY; not yet run.
- Owner: root. The user authorizes sequential implementation and measurements,
  reports no competing heavy task, and requires notice before any major rewrite
  or work equivalent to creating a custom library.
- Baseline: fdb2e1d, native source faf71b33, release/mimalloc ELF 2ed57a31.
  See [the allocation analysis](../verification/single-worker-bottleneck-analysis.md).

## Requirements and scope

Improve the measured one-worker path using existing Serde and standard library
facilities. Apply one candidate at a time and compare with the immediately
preceding accepted version. Preserve simulator revision ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6,
catalog v2, source and numeric schemas, eligibility, all source/sample/index and
array identities, output order, NumPy ownership, Config memory/cache limits,
pool ownership and fail-closed publication. No games, fits or optimizer updates.
No new dependencies, parser, allocator, arena, file format or learning behavior.

The allocation profile is evidence for hypotheses, not a speed forecast. Replay
has 71.28% of allocator requests, not 71.28% of CPU time. Record/Event capacity
growth does not establish that preallocation improves total time or peak memory.

## Stage 1: remove the typed Journal's redundant reserved-key scan

The generic Value decoder still requires its reserved-number-member guard under
serde_json arbitrary_precision. Keep that guard and its acceptance unchanged.
Only typed Journal decoding is a candidate for removing the separate byte scan:
every subtree has a concrete DTO; there are no arbitrary Value/ignored subtrees.

Fixed structs use deny_unknown_fields. Custom Outcome and Defense object visitors
also have closed key domains. Audit these and the generator before implementation.
The remaining dynamic string-key objects are UniqueMap catalog/schema fields.
Reject the decoded reserved key `$serde_json::private::Number` inside their
existing key visitor, before consuming its value, while preserving duplicate
rejection. Use ordinary Serde string decoding, including escaped spellings;
ordinary values containing that text remain valid. A string-key bound on this
private DTO helper is sufficient if all actual callers support it. Do not add
a generic deserializer adapter, another lexical scan or a custom JSON grammar.

The pure DTO layer remains independent of native I/O, resource Control and JSON
Value helpers. Shared owned/borrowed standalone DTO users receive the same key
restriction; their prior direct acceptance of this reserved dynamic key is
intentionally aligned with the production Journal guard. Production input
acceptance stays identical, apart from error wording/order on already-invalid
inputs. Record framing, UTF-8, standard Serde depth, numeric bounds, unknown and
duplicate field contracts remain intact. Preserve cancellation before and after
each record, explicitly after framing immediately before decode and immediately
after decode before retaining it; Serde itself has never offered cancellation during a Deserialize.
Removing the preceding guard must not introduce an uncancellable operation
beyond that existing record-sized decode boundary.

Tests cover all object forms, nested and escaped dynamic reserved keys, reserved
value strings, duplicates after escape decoding, malformed Unicode, numeric and
depth boundaries. Compare the old guard-plus-typed decoder with the new typed
decoder as an independent acceptance oracle on these fixtures, freezing the old
DTO decoder or independent expected outcomes rather than combining the old guard
with the modified DTOs alone. Include derived struct sequence forms and both
custom deserialize_any paths with large integer/decimal tokens. Generic Value
parser tests must still reject the reserved key. Full frozen real-source outputs
and saved PPO ingestion must agree before adoption.

## Later candidates and escalation boundary

After deciding Stage 1, separately investigate duplicate Schema construction,
Replay Value construction, and bounded Record/Event capacity hints, in that
order. Each needs its own bounded design and, where structural, architecture
review. Do not use RawValue to bypass existing recursion semantics. Do not
implement a custom schema-aware parser, allocator or generic arena to pursue an
unmeasured hypothesis. If a candidate needs a large representation/validator
rewrite or library-level machinery, explain its concrete scope, expected benefit
and maintenance cost to the user before starting. Smaller standard-library
capacity experiments may proceed while that larger decision remains open.

The bounded feasibility audit found that allocation-free comparison of the
second Schema requires its own recursive deserializer, and typed Replay affects
source ownership plus parallel/pipeline/projection/view/validation consumers.
Root notified the user before either implementation; both are deferred. Serde
JSON SeqAccess has no useful Event array size hint. A Record-only capacity
experiment is the remaining local candidate; its extra line-count pass and
memory on malformed input must be justified separately. No Event custom visitor
or capacity heuristic is approved by this record.

### Stage 2 local experiment: bounded Record capacity

Only after Stage 1 is accepted or rejected, compare a Record Vec reservation
candidate against that accepted baseline. Count existing CR/LF/CRLF-framed
segments once from admitted raw bytes using ordinary slice operations, with
periodic cancellation. Preserve empty-line/trailing-separator parse behavior.
After the first record has decoded successfully, use try_reserve_exact with
capacity min(segment_count, raw.len() / size_of::<Record>()). Thus the initial
requested slot storage is bounded by raw byte length, already covered by the
unchanged coarse JSON admission; malformed short-line files cannot trigger an
unbounded slot reservation. Do not reserve or scan twice for empty input, and
retain normal Vec growth when the hint is capped. Checked/fallible reservation
failures fail closed as resources; existing malformed inputs still fail.

This adds a byte pass and may be slower: observed Record push CPU was small and
not a complete upper bound. It is a disposable local experiment, not an arena
or an adopted optimization. Test framing, capped hints, invalid records and
resource/cancellation behavior; measure the same whole-path and memory outcomes.
If inconclusive or slower, preserve the private candidate/evidence and remove
only its scoped source changes, retaining the preceding accepted version.

## Verification, measurements and adoption

Root owns this design, documentation, benchmark orchestration and acceptance;
one bot_implementer owns scoped native/DTO/tests after design READY. Independent
bot_reviewer and bot_architecture_reviewer check correctness and conformance.
Preserve the baseline wheel, build each candidate separately in release opt3,
debug0, mimalloc, and keep the main installation fixed throughout comparison.

Run relevant Rust/fmt checks, meaningful Python parser/lifecycle regression,
complete frozen history/observation/index outputs and saved current-v2 PPO
ingestion. Use the existing independent source/numeric oracles. Source jobs run
sequentially. Notify the user before performance work, then run one warmup and
five alternating before/after fresh-process trials per candidate at one worker,
one in-flight episode, unchanged 2 GiB / cache512 MiB / chunk16 MiB policy.
Record whole/stage elapsed, reported CPU, peak/post RSS and sampled PSS, managed
and scratch peaks, logical I/O, build/source/output identities and campaign cost.
No concurrent own builds, tests, edits or source jobs during measurements.

Compare whole-path medians, ranges and paired results; phase sums and sampled
inclusive CPU are not exclusive elapsed partitions. Report CPU and memory costs
alongside any elapsed gain. Inconclusive or regressing candidates are not adopted
as speed improvements. Keep evidence for rejected experiments. Commit coherent
verified stages locally; no push. Architecture and verification documents must
describe what was actually retained and which later stages remain unimplemented.

## Stage 1 disposition

Root accepts the bounded decoded-key change. All92 native tests,11 standalone
DTO tests,106 relevant Python tests, fixed history/observation/index and saved
current-v2 PPO1/4worker proofs pass. All12 same-campaign commands pass; five
paired wins, whole median22.554416->21.609089s (4.191318%), reported CPU
22.418075->21.369977s. Journal job-phase median8.167892->6.983809s. Whole
ranges overlap; this is measured corpus evidence, not a significance or other-
worker claim. Peak RSS median rises9.92MiB, maximum1.77MiB; managed/scratch
peaks and Config remain unchanged. Root explicitly accepts that small memory
cost for the elapsed/CPU reduction. Main installation is deferred until the
remaining local capacity experiment is decided. Full metrics and limitations:
[verification](../verification/single-worker-optimization.md).
