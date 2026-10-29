# 0012: Serde JSON and coarse resource admission

- Status: Accepted on2026-10-30; user authorized production replacement and an audit of
  unnecessary custom machinery on2026-10-30.
- Owner: root; preimplementation architecture and final independent correctness/
  conformance reviews READY. Runtime/installed-wheel/corpus gates PASS; see
  [adoption verification](../verification/serde-json-adoption.md).
- Supersedes ADR0011's custom JSON AST/parser, per-JSON-allocation admission and
  fine cancellation cadence; keeps its learning/information/lifecycle boundaries.
- Evidence: verification/serde-json-probe.md, commit580d12a. Same52.82MiB sample,
  complete typed check/drop2.846x faster; standalone RSS36.71->45.60MiB. This is
  not an already measured full-preparation speedup.

## Requirement and scope

Adopt serde_json as the production native JSON parser and value representation.
Delete the custom parser/Text/Value implementation rather than retain a fallback.
Keep Python reward/GAE/model/teacher ownership, actor-only features, frozen
candidate order, whole-parent eligibility, source revalidation and NumPy base
owner leases. Simulator remains external and unchanged at the pinned revision.
No recurrent model, formal PPO campaign, refit or final-test work in this unit.

The user's resource request is configurable practical soft admission, not exact
accounting of every JSON node allocation. Earlier implementation promoted that
into an unnecessary constraint and bespoke parser/index machinery. Do not use
that superseded constraint as a reason to keep generic algorithms custom.

## Standard components and narrow compatibility code

Pin measured serde_json1.0.151 and serde1.0.229; enable arbitrary_precision,
float_roundtrip and unbounded_depth. Root Python/NumPy/Torch versions unchanged.
Use serde_json::Value/Map/Number and ordinary String/Vec storage for JSON. A small
checked extension trait uses noncolliding names (string/array/object/boolean,
require/integer/number/closed/eq_python), preserving closed-domain checks and
integer-versus-float distinction. Update native consumers; no duplicate custom AST.

Use serde serialization for string escaping and map ordering. Retain only the
Python-compatible numeric formatter needed by existing canonical SHA values:
float exponents/decimal thresholds, -0.0 and strict finite canonical output;
integer lexemes remain exact including128-bit RNG controls. Canonical object keys
remain sorted. Use std formatting for exact integral-float/decimal equality
instead of the handwritten decimal-doubling algorithm; test beyond2^53 and f64max.
Python truthiness/numeric evidence equality remains explicit domain compatibility,
not serde Value::PartialEq. Replace canonicalize-and-reparse copy with Value::clone
under operation/document admission. No learning metadata becomes a feature.

Replace parser duplicate-key indexing with serde Map; replace source evidence's
custom open-address index with std HashMap<PathBuf,usize>, retaining evidence Vec
order and all replacement/hash checks. Replace both custom heapsort copies with
std in-place sort_unstable_by and boundary cancellation checks. Remove obsolete
per-node decode-profile hooks/module/tests; keep coarse read/decode/validation/
feature/staging metrics and source-kind decode timers for future measurement.
Use std UTF16 decoding/char conversion in the Replay encoding adapter; do not
maintain a second JSON tokenizer for compatibility edge cases.

## Memory and failure contract

Keep memory_budget_mib/cache/chunk/workers/max_in_flight/source-file/record/scratch
configuration and existing presets. Numeric/IPC/exported-owner buffers retain
capacity leases and old/new growth overlap accounting. Those leases prevent
retained NumPy views from silently escaping the shared budget.

JSON moves to coarse estimated document/file admission. Initial explicit policy:
32*raw_input_bytes+64KiB per source document/file, checked overflow and reserved
before serde construction. One journal file shares one reservation across its
record Documents through Arc; do not reserve64KiB independently for every line.
Reservations last through retained source Document lifetimes (parents/collections
remain retained). Generated/cloned compact controls remain covered by existing
operation-level control/proof admission and bounded builders. Audit moves of
Document.root so retained data cannot silently outlive its intended admission.
These are practical soft estimates, not exact serde heap accounting or RSS quotas;
record policy/estimated peaks separately from OS RSS/PSS. Validate the existing
full corpus at2GiB/16MiB; if the initial reservation formula cannot admit it,
return to root with actual retained-size/peak evidence before changing policy.
Do not silently change memory budgets, worker counts or batch sizes to pass.

Cancellation/deadlines become file/record/operation boundaries and bounded scans,
not every map probe/allocation. A serde parse or std sort is allowed to finish;
check cancellation/deadline before and after, then refuse all late publication.
The Python owner still joins native work and keeps locks until cleanup completes.
Existing filesystem/OS blocking debt and soft cleanup deadlines remain explicit.
No successful batch/update/parent may be committed after cancellation/failure.

Preserve native_json_depth1..990 and root-value-depth0. To avoid unbounded serde
recursion, a small byte nesting/quoted-string resource guard checks container
nesting before disabling serde's built-in recursion limit. This is only a depth
resource guard; serde owns grammar/Unicode/number validation. Exact value-depth
validation follows construction with periodic control checks. Existing owned8MiB
entry/worker/retirement stacks remain until depth/error/GC tests prove replacement
safe; do not replace their ownership protocol speculatively.

## Compatibility policy

All genuine frozen data, feature bytes/signatures, NPZ/shards, dataset/sample/index
hashes and source health rules must remain unchanged. No manifest schema bump or
source rewrite. Keep historical Python reference/legacy readers explicit.

With arbitrary_precision, pinned serde Value interprets a leading object member
named "$serde_json::private::Number" as its internal numeric representation.
Reject that member key before Value deserialization at any depth/position. Extend
the small quoted-string/depth preflight: a closed quoted span followed by a colon
is a member-key candidate; compare raw unescaped keys, and use serde's String
decoder only for escaped keys. Never reinterpret a real object as a numeric DTO
field. The same text as a string VALUE stays valid. Test escaped spellings,
nested/non-first keys and actual128-bit number tokens. This is an explicit
reserved-key input restriction, not a custom AST/tokenizer or fallback parser.

New native parser contract serde-json-v1 accepts standard JSON with scalar Unicode.
Nonstandard bare NaN/Infinity and lone Unicode surrogates now fail during parsing;
there is no custom-parser fallback/token rewriting. Valid UTF8/BOM and UTF16/32
Replay handling remains pinned; journals/control files retain their stricter UTF8
rules. Canonical nonfinite rejection, integer digit limits, bool/int/range checks
and duplicate-key last-wins remain explicit. Record this compatibility tightening
and updated parser/resource contract in native build identity; missing/old native
builds fail preflight. Backend/binary/code change continues to reject ordinary
resume. Do not reinterpret old nonstandard inputs as eligible sources.

## Verification and audit disposition

Before acceptance: independent correctness and conformance reviews; scoped and full
Rust/Python tests, lint/types, installed release wheels and optional-free baseline.
Check serde numeric fidelity/canonical Python hashes (randomfloats,2^53,128-bit,
-0.0/exponents), duplicate keys, depth/resource/cancel/GC, strict actor visibility,
source tampering and failed-parent rejection. Full83,783-row both-recipe NPZ and
index content equality; bounded actual PPO batch ingestion and numeric equality;
three alternating full-preparation release pairs against the frozen old native
extension, with wall/CPU/RSS/PSS/storage/I/O and no concurrent builds/training.
No new fit is required when the fit request/numeric archive hashes are identical.

Root owns dependencies/config/docs/decisions/commits. One bot_implementer owns
native/application integration and tests after design READY; independent reviewers
are read-only. Record other bespoke code in an audit table: remove parser/index/
heapsort/fine-profiler now, retain canonical numeric compatibility, NumPy leases,
source/learning validation and safe deep retirement with concrete reasons. Further
unrelated subsystem rewrites need their own evidence and scope.
