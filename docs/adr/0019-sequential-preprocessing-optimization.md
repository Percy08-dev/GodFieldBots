# 0019: Measured sequential preprocessing optimization

- Status: Accepted2026-10-30; Stage1/2/3a/5a implementation, proofs and measured
  adoption accepted after independent reviews. Stage4 current-format migration is
  conditionally skipped. Further alternatives remain separately gated.
- Owner: root. Baseline`b71ad31`/native source`e604697d...` and accepted release
  mimalloc/typed wheel`5eba6710...`. Main32e3 was the unchanged comparison control;
  final adopted97f1wheel is installed after comparisons. User authorizes applying changes sequentially, measuring
  each, and retaining only demonstrated improvements.

## Requirements and staged decision

The root-cause report identifies repeated history traversal/control checks,
duplicate schema/history representation and stage/task supply. Do not combine
optimizations into one before/after observation. Preserve numeric feature bits,
order/names/signatures, current contracts/reader acceptance, source validation,
whole-parent offline eligibility, memory Config and failure/cleanup semantics.
No simulator/game-rule modification, games/fits/learning updates or final test.

Execute the following sequence. Each candidate is measured against the latest
accepted candidate. Record accept/reject/conditional-skip with evidence. A failed
or inconclusive candidate is not silently promoted. Later structural designs
return to architecture review before implementation; this record does not approve
an unspecified parser, format migration or pool framework.

1. Single-pass history aggregation on the typed numeric path, specified below.
2. Reprofile remaining control-check work. Only if still consequential, test
   cancellation/deadline polling changes independently, with explicit bounded
   responsiveness and no global check removal or changed failure publication.
3. Test safe sharing of identical schema payloads without losing current strict
   acceptance. Borrowed RawValue/raw-byte equality plus one typed representation
   is a candidate, with a full semantic fallback for nonidentical encodings.
   Exact interface/lifetimes/Serde feature changes require the next design gate.
4. Evaluate cumulative-history duplication. Persisted delta/history-reference
   formats require explicit versioning, old-reader compatibility and reconstruction
   of identical acting-player inputs. Never rewrite original source datasets or
   infer cursor internals. Producer-generation cost and existing-data preprocessing
   benefit are separate. Do not promote a new format without measured benefit and
   complete provenance/failure verification.
5. Reassess remaining supply/ramp/tail waits; test pool reuse, line batches or
   overlapped stages only as individually scoped candidates, with independent
   read/parse settings, bounded queues and source-safe publication. No automatic
   Rayon/dependency/default change follows from earlier component experiments.

Every stage has a bounded baseline/control and its own adoption decision. If a
conditional stage has no remaining measured target, document that instead of
adding complexity. Any incompatible or unmeasurable candidate remains rejected
or explicitly unresolved, never presented as applied.

## Stage1 design: typed single-pass history

Change the actual typed public-history feature path only. Keep the legacy numeric
encoder and Python encoder as independent reference oracles and retain all their
accepted behavior. Observation-only recipe is unaffected.

Traverse the same configured tail of Input.events once. Use fixed-size stack
accumulators for24event counts, two role counts and two sets of seven i128sums.
Match the original role precedence player->attacker->active_player, optional/null
behavior, unrecognized feature-kind handling and definition-ID SHA bucket math.
Counts/sums are emitted as f64 in the exact old42+hash_buckets history layout.
Preserve window_truncated, count, field order and original arithmetic semantics.

Avoid a heap allocation proportional to history_window or a4096-entry temporary
on every decision. Prefer filling the existing checked caller-owned history block
for definition buckets, with small stack counters. A narrow private NumericSink
block helper or concrete typed numeric sink is permitted if needed; it must keep
checked width, initialized storage and finite output validation. Do not add a
general accumulator/task framework. Metadata/name generation stays unchanged.

Maintain Control checks at entry/exit and during each event traversal. Fusing
loops naturally removes checks associated with redundant passes; do not also
introduce sampled wall-clock polling or alter Control::check in this stage.
No feature cache escapes a decision. No other participant's information or
diagnostic generation/GameSeed fields enter features. Learning math remains Python.

## Compatibility and failure boundaries

No new stage1 dependency, Config key, persisted format, simulator contract,
concurrency ownership or numeric buffer IPC. Memory/file/record/chunk/scratch
settings and reader defaults remain unchanged. Numeric/name/source/sample hashes
must remain original where their contents have not changed. Code/native identities
change normally and ordinary resume must keep rejecting incompatible identities.
Sources remain pinned to simulatorffcffa0966dbf6d7ce1d982f4ee6444e291d42d6,
catalogv2 and stored schema4 semantics.

Quota/cancel/deadline errors propagate through the existing owned cleanup path.
Do not weaken validation or create source eligibility through an index/cache.
For later format changes, old sources stay authoritative and immutable; new
versions must be explicit. Original six fitted models/data remain protected.

## Stage1 verification and measurements

Implementer owns native code/tests; root owns design/config/docs/orchestration.
Test bitwise parity against unchanged legacy and Python outputs for empty, short,
exact/truncated/max-bound histories, all event feature kinds, actor precedence,
optional fields, unknown feature kinds, hash collisions and integer boundaries.
Exercise2/3max_elements,2/default/4096hash buckets and supported history bounds.
Test cancellation/deadline and wrong output width/nonfinite refusal where relevant.
Do not create tests that merely reproduce the new algorithm.

Before timing, verify full frozen history/observation NPZ/array/sample/source/index
identities and actual saved current-v2 PPO inputs, plus relevant Rust/Python/lint/
type/isolated wheel checks. Run real-source jobs sequentially under existing locks.
Main environment is not installed during comparison. Candidate-only checks select
its private site explicitly; never use inherited Simulator VIRTUAL_ENV.

Primary comparison uses the same83,783-row fixed history workload, releaseopt3/
debug0 mimalloc, memory/cache/chunk2048/512/16MiB, read/parse workers4/inflight4,
one warmup plus five rotating fresh-process trials per variant. No heavy jobs or
source edits during timing. Match exact outputs every trial and bracket source/
protected-model/binary/code checks. Record whole/stage elapsed, reported CPU,
kernel peak and retained RSS/PSS, logical/sampled storage/I/O and check counts
where direct evidence exists. Existing CPU API uncertainty remains explicit.

Accept elapsed improvements beyond observed variation without a consequential
resource regression, or a clearly measured resource improvement with no material
time regression; explain tradeoffs rather than introducing an invented threshold.
Micro/function results alone do not establish complete-path improvement. Reject
equivalence failures and retain their evidence. Independent correctness and
architecture conformance reviews resolve blocking findings before commits.

Final accepted changes are committed coherently and the usable native wheel can
be updated after comparisons finish; protect the old baseline artifact and do
not bypass checkpoint/environment identity checks. Report each stage separately.

## Execution ledger

| Stage | State | Evidence / decision |
| --- | --- | --- |
|1single-pass history|ACCEPTED|14.15%wholewall gain, exactoutputs; verificationrecord|
|2polling|ACCEPTED local coalescence|1.43%wholemedian/5pairedwins; no globalpollchange|
|3schema sharing|ACCEPTED Stage3a|3.42%wholemedian under authorized shared load; median peak RSS -34.34MiB|
|4history representation|conditional skip current format migration|original sources/provenance immutable; future producer experiment separate|
|5supply/pool/streaming|ACCEPTED Stage5a|2.73%wholemedian/21.82%parsemedian gain; explicit CPU/medianRSS cost; no pool|

Stage1 evidence:[verification](../verification/sequential-preprocessing-optimization.md).

## Stage2 bounded candidate: adjacent history-check coalescence

Status: ACCEPTED after source and measurement reviews. Accepted Stage1
`2325bcb`/source`9bd697d...` is the comparison baseline; Stage2source`d9413ab...`.
Counts remain18,772,215in feature/spooling and32,926,323overall. Before adding
a configurable/global polling mechanism, test the smallest remaining duplicate:
the post-event full Control::check followed immediately by the next event's
pre-event full check. Retain function entry, every event's pre-check and final
completion check. Remove only the redundant post-event check. Global Control,
all read/parse/validation/other-feature checks and per-event aggregation remain
unchanged. Do not introduce a counter, wall-clock sampling interval or new API.

Cancellation/deadline observation is still bounded by one event plus fixed
history-output emission before the final check. Numeric block cursor must not
commit on any error, and no output is published without the final existing proof/
deadline checks. Verify preexpired and mid-loop cancellation/deadline, one-event
and final-event cases, maximum history/bucket widths and all Stage1bit oracles.
Retain before-each-event checking even when a definition hash is large. This is
a local check-position candidate; it is not approval to remove global safeguards.

Measure latest accepted Stage1 versus Stage2 candidate with the same full4worker
protocol, warmup+5alternating trials, exact source/numeric proofs and resources.
If the gain is within variation or time/resources regress, reject the candidate
and keep Stage1. Check counts alone do not justify adoption. Broader clock-polling
changes stay unapproved unless further evidence makes them consequential.

## Stage3a: accepted post-decode schema sharing

Status: ACCEPTED after independent source, conformance and measurement reviews.
Accepted Stage2
`8ae4235`/source`d9413ab...` is the comparison baseline. Native and standalone
Serde enable matching rc support without dependency-version or AP changes.
Candidate source`89385a4...`/ELF`d876348...` passes826Python tests(5skips), full
both-recipe/index/savedPPO1/4parity and independent source reviews. Following
advance notice, the user explicitly authorized a trial while reporting roughly
20% all-core CPU usage from another task. This comparison is an explicit exception
to the uncontended primary protocol above, not evidence of uncontended speed.
Future performance measurements still require advance notification.

Root chooses a small compatibility-preserving sharing experiment before lazy
RawValue decoding. The pinned serde_json RawValue path uses iterative ignore_value
without the ordinary typed recursion decrement; decoding that subtree later also
changes its nesting context. Do not assume a RawValue shortcut preserves the old
accept/reject set, and do not add a custom JSON/depth validator to make it fit.
Raw/lazy skipping remains unapproved in this candidate.

Decode both schema descriptions through the existing strict typed path, then
share their retained representation only when complete structural equality holds.
Use ordinary Arc-backed schema collections (Serde rc feature if needed), retaining
external raw-text lifetimes. After Observation shape validation has validated a
shared collection, Input validation need not traverse that same immutable tree
again. Unequal descriptions remain separately decoded/validated and retain the
existing semantic mismatch rejection. No cache/intern table across inputs, journal
files, sessions or versions; no guessed three-value whitelist.

The exact constructor/deserializer design must keep derived closed/duplicate/
numeric/recursion behavior. A small generated six-field Input wire adapter plus
normalization is acceptable; no new JSON grammar or Value reconstruction. Update
the native-owned generator and its generated borrowed DTO coherently; standalone
borrowed tests must use the same schema. Owned experimental DTO need not change.
Canonical Serialize must emit both original fields with the same logical values.
Feature/projection/numeric/source hashes and schema4 bytes stay authoritative.

This candidate deliberately does not eliminate the two initial deserializations;
it targets retained AST memory, repeated shape validation and heap reuse. Report
that limit and measure total path/time/resources against the latest accepted
candidate. Retain only on demonstrated elapsed or meaningful memory benefit with
no material time regression. No new format or producer changes follow from it.

Before implementation, architecture must review the concrete Arc/adapter/lifetime/
Serde-feature design. Test equal/distinct raw encodings, unequal metadata, unknown/
duplicate fields, numeric lexical forms, actual recursion boundaries and direct
JSON/Value fixture compatibility. Preserve all invalid rejection and source health
before publication. Run original both-recipe/index/PPO/NumPy lifetime proofs and
appropriate regression on the final isolated wheel before primary measurements.

All12commands pass the original numeric/source/build/lifecycle proofs. The five
alternating measured pairs favor the candidate; whole median19.769560->19.093615s
(3.419123%). Journal-validation ranges separate2.811832–2.869000s versus
2.423769–2.465289s. Median workload-self peak RSS428.73->394.38MiB and maximum
429.92->423.91MiB; the coarse managed peak remains1,605,390,806bytes. Independent
measurement review accepts bounded retention without requiring an uncontended
rerun. Shared-load and CPU-accounting limits remain explicit. Full evidence:
[verification](../verification/sequential-preprocessing-optimization.md#stage3a-accepted-schema-sharing-under-shared-load).

## Stage4 disposition: no current-source format migration

No history-reference/delta format is adopted in this optimization unit. Original
schema4 source bytes, parent/generation/update proofs and the six fitted artifacts
remain immutable. Replay does not independently attest omitted event-feed deltas.
Rewriting original cumulative inputs or trusting a derived compact cache instead
of authoritative source validation would cross that boundary. A producer format
requires explicit versioning, writer/reader reconstruction and new collection
evidence; those are future work outside this unit's no-games/updates constraint.
Existing-data preprocessing and future collection/storage savings are different
experiments. Preserve full original acting-player history and current eligibility.

This is a compatibility-based conditional skip, not a measured claim that delta
history cannot help. No speed or memory benefit is claimed for an unbuilt format.
Independent architecture review accepts this scoped disposition. Reusing already
validated event prefixes after complete strict decoding and exact comparison is
a compatible, unassessed alternative. It would not remove initial decoding and
requires its own design/measurement gate; this disposition does not claim that
all possible in-memory history optimizations have been exhausted.

## Stage5a candidate: bounded participant-file parse tasks

Status: ACCEPTED after independent design, source-correctness, conformance and
measurement reviews. Source d5090c4/ELF97f1b50; full evidence is in the
[verification record](../verification/sequential-preprocessing-optimization.md#stage5a-accepted-participant-file-task-supply).
Baseline is accepted Stage3a source89385a4/ELFd876348. A read-only source audit
identifies a smaller task-supply experiment before persistent workers/Rayon.
Four in-flight episodes expose only four episode tasks, each sequentially parsing
two participant files. Dynamic episode assignment alone cannot improve that
four-job full window. Exposing its eight files as independently claimable jobs
can reduce tails without changing read concurrency, resident episodes or threads.
It is a hypothesis to measure; no improvement is assumed from the source graph.

### Responsibilities and interfaces

Change private parse scheduling in native/src/parallel.rs, corresponding tests,
and explicit parse-job diagnostic scope in telemetry.rs/test_native_parallel_parse.
Keep parse_window's existing borrowed window and ordered ParsedEpisode return
interface. No dependency, unsafe code, persistent thread pool, independent new
Config setting, raw-file streaming, record parser, validator, feature or format
change. Root owns design/config/docs; implementer owns scoped native code/tests.

One checked task ordinal identifies (episode ordinal, participant slot). At most
two tasks per resident episode are eligible. The same configured worker limit
applies; each scoped worker claims from a window-local atomic next ordinal,
parses one journal, then claims again. Use budgeted worker-local results carrying
ordinals and regroup into the original ordered pairs only after every join.
The ordinal identifies the original file slot; downstream header-based viewer
mapping remains authoritative and must not be inferred from file slot alone.
No unbounded work queue or source-wide task list. Raw owners stay coordinator-
owned and immobile; all Records borrow them through the existing scope. Legacy
all-legacy bypass and mixed-path behavior remain compatible. One worker remains
inline and processes episode/participant order without creating parse threads.

Do not silently reinterpret diagnostics: retain scheduled_episodes for window
membership, add scheduled journal-task count and explicit job_unit/version for
per-slot jobs/elapsed/overlap. Older artifacts continue to mean episode jobs.
Read/decode telemetry and coordinator exclusive phase names retain their scopes;
all wall partitions must reconcile. Nested task elapsed is not CPU utilization.

### Ownership, failure and resource boundaries

Reserve every thread's existing8MiB stack and collection capacities through the
existing shared Budget. Active parse threads never exceed configured workers;
the final partial window may use min(workers,2*episodes). No hidden second pool
or prefetch window. Check cancellation/deadline before claiming/starting work,
during unchanged parsing and before success. First failure cancels siblings;
join all started workers on syntax/quota/spawn/panic/cancel/deadline failure,
release raw/DTO/temporary leases, and return no successful partial window.
Order source-evidence merges, semantic checks, numeric/proof emission and final
source revalidation exactly as before. Parsing acceptance does not confer source
eligibility. Keep caller-owned raw/DTO lifetimes and NumPy storage ownership.

### Verification and adoption

Test shuffled completion and correct episode/participant regrouping, borrowed
lifetimes, actual bounded overlap, inline/mixed/legacy behavior, task/episode
telemetry, panic/cancel/deadline/invalid input and quota cleanup. Retain full
independent legacy/Python numeric/source oracles. Run appropriate native/Python
regression, both fixed recipes, Parquet and saved current-v2 PPO1/4 proofs on an
isolated releaseopt3/debug0 mimalloc wheel. No games, fits or optimizer updates.
Independent correctness/conformance reviews precede acceptance.

Then notify the user before performance execution. Compare against Stage3a with
the existing fixed83,783-row4read/4parse/inflight4/2GiB/cache512/chunk16 protocol,
one warmup plus five alternating fresh-process pairs, exact proofs every run.
Record whole/stage wall, reported CPU, peak/post RSS/PSS, managed memory and
storage/I/O with original scopes. If the external task is active, label shared
load and do not generalize to uncontended speed. No concurrent own heavy jobs,
source edits, or multiple source jobs. Retain only demonstrated complete-path
time/resource benefit without consequential regression; reject or leave
inconclusive otherwise. Pool reuse/line batches/overlap remain separate gated
decisions, never automatically included in this candidate.

All12primary commands pass. Whole median19.261019->18.734746s (2.732321%),
four of five paired wins; parse improves every pair with separated ranges,
median4.409453->3.447428s (21.82%). Reported CPU median rises1.542533%and
median peak RSS18.12MiB; maximum peak is similar and post-release RSS falls2.04MiB.
Managed peak1,605,390,806bytes and logical staging/scratch totals stay unchanged.
Root accepts those bounded resource costs for the measured elapsed gain under
authorized shared load. This does not establish uncontended speed, lower CPU
work or per-job speedups across changed job units. Current Stage5a is the final
accepted candidate for this sequence; no pool/line/overlap implementation follows.

## Final operational acceptance

After all comparisons and independent acceptance, back up the exact old32e3ad68
package, distribution metadata and loaded identity, then install accepted
Stage5a97f1b509wheel into Bots.venv offline with no-deps/no-build. Fresh interpreter
identity matches source d5090c41/build/allocator/ABI/capabilities and the exact
tested ELF. Optional-free CLI help starts without importing native or learning
runtimes. All24protected original fit files remain unchanged. Evidence:
`runs/assessments/sequential-optimization-20261030/final-installation.json` and
`main-before-final-install/`. Config budgets/defaults, dependencies and ordinary
resume compatibility checks are unchanged. No games, fits or optimizer updates
were executed. This closes the bounded sequence; unassessed history/pool/streaming
alternatives do not become accepted implementations.
