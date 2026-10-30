# 0019: Measured sequential preprocessing optimization

- Status: Accepted roadmap/Stage1design2026-10-30; independent architecture READY,
  no blocking findings. Later structural stages remain separately gated.
- Owner: root. Baseline`b71ad31`/native source`e604697d...` and accepted release
  mimalloc/typed wheel`5eba6710...`. Main installed32e3 remains comparison control
  until final adoption. User authorizes applying changes sequentially, measuring
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
|2polling|conditional pending|reassess after1|
|3schema sharing|pending gated design|no format/acceptance assumptions approved|
|4history representation|pending gated design|producer/current-data benefit separate|
|5supply/pool/streaming|conditional pending|reassess remaining waiting after prior stages|

Stage1 evidence:[verification](../verification/sequential-preprocessing-optimization.md).
