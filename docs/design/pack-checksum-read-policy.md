# Pack checksum read policy and explicit storage audit

Date: 2026-11-11. Owner: root. Status: architecture READY before implementation.
Independent `bot_architecture_reviewer` found no blocking design findings.
This document implements the owner's Group Pack＋SQLite revision. It supersedes
the ordinary whole-database rehash and eager Pack-verification requirements in
ADR0034 and its dependent designs; their other contracts remain applicable.

## Requirements and approach

Keep one zstd level 3 Pack per managed Simulator group, Python Writer, SQLite,
existing admission/queue limits, sealing/publication order and directory default.
Detect accidental corruption, missing records and mixed-up originals. Resistance
to coordinated malicious rewriting is outside this integrity contract.
Apply the new read policy to already persisted storage without changing bytes,
checksums or historical evidence. No migration or old-parent resume bypass.

The selected approach separates metadata admission, demand-verified Pack access,
and an explicit full audit. Keeping eager verification fails the requested read
cost; adding a persistent audit cache would create an unnecessary stale-trust and
compatibility protocol. No new dependency or Rust SQLite/zstd implementation.

## Responsibilities, interfaces and flow

`gf_training.group_pack_db` owns two distinct operations: bounded index admission
and full readonly database verification. Factor the existing validator rather
than duplicate SQL semantics. Normal admission checks exact schema and DELETE
seal/header, run/seal identities, row types, unique safe names, group/episode
references, declared compressed/raw sizes and budgets, member ranges/overlap,
control sizes/checksums and descriptor agreement. Index-only logical seal digest
checking may remain: this hashes already admitted metadata, never Pack BLOBs.
SQLite integrity/FK pragmas and iteration over Pack payloads belong to full audit.
Foreign references must still be rejected by metadata admission without FK scans.

`group_pack_source` owns SAME original FDs, read-only SQLite connection and group
loans. A packed database original binds descriptor's stored expected SHA and
stat6 without calculating that SHA on normal open/recheck/close. Small descriptor,
locator, predecessor and requested controls retain their checksum checks. Preserve
ordinary OriginalFile hashing for directory sources and non-DB controls; use an
explicit DB-specific mode/owner, never a generic global weakening.

On a cache miss, check original identities/namespace at the group boundary,
measure and admit compressed size before fetching, reserve existing conservative
codec/cache charge, fetch exactly the indexed run/group BLOB, verify length and
stored SHA256, decompress within declared cap, and compare decoded members with
the admitted complete index (names, kind, offset, length, SHA). Decode once per
acquisition. Reject corrupt/truncated/oversized BLOBs before publishing cache.
Reacquisition after eviction repeats verification. Unrequested groups remain
unfetched and undecoded. A cache hit and every cached loan member access check
only deadline/liveness, active loan, generation and group identity: no SQL, stat,
namespace walk, hash or decompression. Preserve stale-loan/borrowed-close refusal,
real close, actual creator-thread ownership and uncertain cleanup reservations.
Session/finalize/group transitions retain lightweight FD and namespace checks.
Source semantic verification may naturally consume every required episode; it
must not start an additional whole-dataset audit.

`group_pack_worker` keeps checkpoint, DELETE transition, close, sync and one
whole-DB SHA generation for sealed descriptors. Its already-validated submission
path and seal metadata checks preserve Writer semantics. Remove duplicate full
validation/rehash through parent Writer, PendingDescriptorOwner, publication and
terminal DB handoff. Full readonly audit is never called implicitly from open,
training startup, resume, finalize, close or publication.

`gf_common.group_pack_game` owns pure evidence validation/comparison. New custody
has exactly the old fields with recipe `packed-original-custody-v2`; the database
SHA is a stored expected value. Policy identity is `pack-checksum-v1`, expressed
by custody2 and the native capability, not an extra persisted schema field.
Accept and validate historical custody1 structure/fingerprint. Compare v1/v2
only after each value passes its own closed checker: remove recipe and its
derived fingerprint and compare all remaining stable evidence exactly. v2/v2
requires full equality. Do not normalize other provenance, file identity, hashes,
paths, visibility, roles, outcomes or numerics. Never emit downgraded v1 receipts.

Python Source/proof/lineage, Dataset4/Tree and Rust consumers use these rules.
Historical correspondence/aggregate digests bind historical evidence, not a
newly generated v2 fingerprint; validate the saved historical digest, then verify
actual episode/role/numeric correspondence independently. New evidence uses v2.
No blind digest substitution or blanket omission of correspondence checks.
Run4/Update4/Dataset4/proof2/provider1/plan4/wire5 and SQLite1/Pack1/descriptor1
remain unchanged. Native keeps all existing markers and adds
`card_instance_pack_checksum='pack-checksum-v1;custody2'`; every packed native
entry point refuses missing/wrong support before new-policy work. Rust continues
to verify member length/SHA across the Python boundary. Source identity continues
to reject resuming parents produced by different code; no compatibility override.

## Explicit read-only audit

Add `gf-bots verify-storage` with required mutually exclusive `--locator PATH`
and `--dataset PATH`, `--memory-budget-mib` default 2048 and `--wall-seconds`
default 7200. Strict positive finite values, no silent fallback to other formats.
Locator accepts existing locator files or sealed published/pending descriptors.
Pending may be audited diagnostically and confers no public Source authority.
Dataset accepts only Dataset4 with explicit Pack references; validate its closed
schema/content/source aggregates and each declared parent/collection/final
reference without invoking training/qualification or modifying any file.

A focused `gf_training.group_pack_verify` orchestrates audit, and CLI only parses,
composes and prints JSON. Keep actual original owners live until verified close.
Deduplicate full DB scans by actual inode identity, not unchecked path text;
individually check each reference's expected size/SHA, run/seal and control/locator
hashes. Conflicting expectations fail even for an already scanned inode. Hash the
SAME retained DB FD, check size against descriptor, run full SQLite integrity/FK,
logical seal, all Packs/members/controls and missing/extra records. Process DBs and
Packs serially within one shared memory/deadline budget; account index/codec and
report metadata, avoid accumulating raw payloads. End-of-audit identity checks
detect concurrent modification. No audit cache or subsequent-read trust token.

Emit a versioned JSON result including selected target/policy/scope, pass/fail,
reference/unique DB/Pack/member/control counts, expected and observed size/SHA and
run/seal values, failures, elapsed/limit and cleanup confirmation. Failures may be
fail-fast within a corrupt DB but must state incomplete scope. Corruption,
unsupported input, timeout/cancel and uncertain cleanup return nonzero, including
a deadline observed during/after close. Preserve actual cleanup owner/error,
never claim a failed close succeeded. PASS means only this full audit succeeded;
it never qualifies training or changes a pending/publication state.

## Information and failure boundaries

Only storage custody changes; original policy inputs, outcomes, splits, rewards,
budgets and Simulator authority remain exact. Audit reads diagnostic originals,
does not execute model fitting/gameplay or confer eligibility. No simulator edit
or contract reinterpretation. Memory remains soft managed admission, not OS RSS;
blocking filesystem calls cannot guarantee a hard wall. Detected overrun fails.
Existing DrvFS compound cleanup limitations remain explicitly unexercised; PPO
parent/control/pruning stay on ext4. No new formal learning/HPO/final test/qualified
fit, data deletion or migration is authorized by this work.

## Verification and acceptance

Use failing behavioral tests before implementation, including call counts for
zero ordinary DB hashes/integrity scans/eager decode, one decode per acquired
group, no cached-file I/O and zero implicit audits across publication/Source/
resume/finalize/close. Cover requested and unrequested corruption, truncation,
oversize, range/name mismatches, eviction/reload and borrowed-close ownership.
Explicit audit must detect missing/extra records, conflicting same-DB references,
wrong expectations, pending diagnostic limits, unsupported inputs, deadlines,
cancellation and uncertain cleanup. Preserve old directory/config fixtures.
Historical v1 Source/Dataset4 proof/correspondence tests and Python/Rust candidate,
feature/numeric parity must run with current source-matched native code.

Run scoped Python/static/Rust checks, bounded real two-update pause/resume and
controlled Dataset/Tree smokes under their existing budgets. Compare old/current
Reader on identical real HDD originals, instrument separately DB I/O/hash, index,
Pack checksum and decompression time/calls/bytes; report OS cache and sampling
limits rather than assert cold disk causality. Independent correctness and
architecture-conformance reviews gate acceptance. Record exact commands/results,
revisions/native identity, unchanged-original evidence and every NOT RUN in
`docs/verification/pack-checksum-read-policy.md`.
