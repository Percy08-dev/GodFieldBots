# 0013: Standard Serde limits and explicit sort contracts

- Status: Accepted2026-10-30; user requests audit/removal of unnecessary JSON preflight and
  validation and confirmation that stable sort is used only where required.
- Owner: root; independent architecture/correctness/conformance READY, runtime
  and installed-wheel/corpus/performance gates PASS. See
  [verification](../verification/json-simplification-and-sorts.md).
- Supersedes ADR0012's configurable990-depth compatibility and global4300-digit
  policy. Memory budgets, canonical numeric hashes and all domain/source/learning
  validation remain unchanged.

## Decision

Use serde_json's normal recursion limit and standard numeric acceptance. Remove
unbounded_depth feature/disable_recursion_limit, native_json_depth from current
settings/presets, the custom preparse depth counter and the postparse full-tree
validate_depth_and_numbers walk. Do not replace them with another parser/visitor
or hidden pass. The old990depth was not a Simulator or learning requirement;
the4300digit boundary was inherited from Python's implementation limit. Preserve
integer/float/range validation at actual domain-field consumption and finite
canonical output. Memory/record/file/scratch/chunk/worker config remains intact.

Parser contract becomes serde-json-v2 (or an equally explicit new identity),
with standard pinned Serde recursion behavior. Test the actual accepted/rejected
boundary of1.0.151 rather than assert that every nominal128-level input passes.
Old effective_config dictionaries remain historical source metadata; do not
rewrite old sources or infer that the removed setting still configures current
parsing. Current explicit use of native_json_depth fails closed as an unknown
setting. Ordinary resume across the binary/code/parser change remains refused.
Strict standard Unicode, nonstandard-number rejection and whole-source eligibility
remain. Huge integer tokens can parse under source-size limits, but bounded integer
DTOs still reject out-of-range inputs. This is an explicit acceptance-policy change.

## Narrow remaining guard and helpers

Keep only the actual serde_json arbitrary_precision reserved-number-member guard:
reject "$serde_json::private::Number" keys, including escaped/nested/non-first
spellings; identical value strings remain allowed. Name the function after this
purpose, not generic preflight. Serde owns grammar, key string decoding and depth
errors. Existing byte-size admission and boundary cancellation remain; no fallback
custom AST/parser and no tree-wide numeric validation.

Remove obsolete ignored Budget parameters and infallible Result wrappers in JSON
constructor helpers where this makes call sites simpler, without broad domain
rewrites. Keep small helpers when they avoid repeated map construction. Preserve
checked ValueExt/domain checks and canonical Python float formatting.

## Sorting

Audited native comparators have unique valid keys, interchangeable equal payloads,
or duplicate rejection before publication; no current native site needs stable
sorting. Keep std unstable sort in those sites and record tie assumptions. Python
sort/sorted remain the standard stable built-ins; do not invent an unstable Python
sort solely for uniformity. Catalog already uses Serde's ordered unique-key Map:
remove its redundant materialized sorted vector by borrowing the map/direct lookup
if consumer audit confirms no order dependence. Preserve candidate/journal order.

If a future contract admits meaningful equal-key records, that caller must add an
explicit deterministic tie key or choose stable sorting with a regression. Stability
is a caller requirement, not a global wrapper default.

## Verification

Independent correctness/conformance review, targeted Rust/Python tests for std
Serde depth, key-type confusion, huge numeric tokens and consuming-field bounds;
existing canonical/random-float/health/source/owner tests. Both full frozen recipes
and index hashes plus saved PPO ingestion remain identical under2GiB/16MiB.
Release current-vs-simplified preparation measurement with wall/CPU/RSS/storage,
no new fit/campaign. Root owns config/dependencies/docs/commits, implementer owns
scoped code/tests after design READY. Keep changes bounded; do not refactor unrelated
ownership/retirement or remove source/learning validation.
