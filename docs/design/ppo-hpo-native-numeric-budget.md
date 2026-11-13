# Native HPO incremental numeric allocation guard

Bounded Source3b design under [the runtime phase limits](ppo-hpo-runtime.md).
Root accepts this precise proposal; scoped architecture review is READY before
the dependent allocator change. It changes no selection/GAE authority or old format.

An absolute ceiling `Budget.used <= initial_source_used + numeric_limit` is
insufficient if old Source leases refund during staging: Source100 can release50
then numeric100 fits a50cap. Do not treat freed Source credit as numeric credit.

Keep one scoped guard INSIDE the existing Budget/Lease accounting. A small mutex
owns active guard generation `{id,limit,charged}` and synchronizes guard install
with new reservation/growth. Each existing Lease records generation-tagged bytes
created or grown during that guard. New reserve/growth charges both the existing
global Source/pool accounting and matching guard; reject before actual allocation
or FFI publication if either bound fails. Do not mutate Budget.limit, create an
external ledger or implement Rust selection/reward/GAE/normalization.

Old Source lease refunds reduce global used only. Only matching-generation tagged
bytes refund numeric charged. Mixed old/new aggregate shrink uses conservative
SOURCE-FIRST attribution: retain current-generation charge up to remaining lease
size. Numeric-only temporary drops refund exactly; stale generation refunds can
never credit a newer guard. Check addition/size arithmetic for overflow. Ambiguous
aggregate resizes may retain charge longer and reject early; they never authorize
excess numeric headroom. This is explicit conservative admission, not a physical
RSS guarantee or full-headroom utilization promise.
Failed allocations refund only confirmed reservation effects. Surviving array
owners retain both global and matching phase leases until actual retirement.

HPO pending scan returns facts without numeric Stage allocation. After final
authorization, retire only genuinely expired authorization allowance through the
existing begin_data boundary, retaining required pending/final trees. Install the
guard before restage scratch/bank allocation. Keep it with the SAME CohortSession
until actual payload retirement/close, not merely until returning a NumPy view;
surviving array owners keep actual leases and accounting. Reject nested guards
and prevent a stale handle from clearing a new generation. Old paths install no
guard and retain their existing limits, values and ownership semantics.

Tests cover the explicit Source-refund counterexample, numeric-only refunds,
mixed resize attribution, stale generations, nested guards, race-free concurrent
reservation/guard installation, arithmetic/limit refusal before allocator work,
and actual HPO Source→final authorization→payload/close parity. Required legacy
Rust/source/NumPy-view lifetime tests and old-data proofs remain unchanged. The
scope does not claim an implemented native capability until positive FFI and
independent reviews pass.
