# 0035: Demand-verified Packs and explicit full storage audit

- Status: Accepted design; implementation and final acceptance pending
- Date: 2026-11-11
- Decision owner: root
- Review: `bot_architecture_reviewer` READY, 2026-11-11; no blocking findings
- Supersedes: ADR0034's eager full read verification and repeated whole-DB rehash
  guarantee only; all other storage, learning and publication contracts remain.

## Context

ADR0034's public loans repeatedly hash the same entire sealed DB and admission
eagerly decodes every Pack. Its same-original HDD comparison records slower
verified reads despite improved writing/space. The owner explicitly requests
checking acquired Packs during normal reads and moving full integrity auditing
to an explicit operation, including existing persisted SQLite datasets.
The integrity target is accidental corruption, loss and mixed-up originals;
coordinated malicious rewriting is outside this contract.

## Decision

Adopt [the read-policy design](../design/pack-checksum-read-policy.md).
Normal admission validates original identities, exact schemas/seals, admitted
index relationships/sizes and small controls. Verify stored compressed SHA,
bounded decompression and complete indexed member correspondence once per Pack
acquisition. Cache hits check loan/generation/liveness only. Boundary checks
retain lightweight original-FD/namespace inspection, with real cleanup ownership.
Keep seal-time whole-DB checksum generation and storage; never imply normal
readers recalculated the descriptor's expected DB hash.

`gf-bots verify-storage --locator PATH` or `--dataset PATH` runs readonly full
verification within 2048MiB/7200s defaults. Scan actual databases once while
checking every individual reference expectation. PASS attests only that audit,
not Source/learning authority or a reusable trust cache. Pending remains private.

New evidence uses `packed-original-custody-v2` and policy `pack-checksum-v1`.
Validate each historical/current custody structure/fingerprint before comparing
cross-version stable evidence. Keep original v1 bytes and saved digest bindings;
new receipts never claim a historical v1 check was performed. Native support
requires the additive `card_instance_pack_checksum='pack-checksum-v1;custody2'`.
All persisted format/schema versions and existing capability values stay exact.
Different-code paused parents remain nonresumable. Directory remains default.

## Alternatives and consequences

Keeping eager verification retains the measured read cost and fails the owner's
normal-read requirement. A persistent audit cache adds stale-state and format
complexity without a current need. Selected per-acquisition validation bounds
payload work to requested groups and leaves unrequested payload corruption for
later acquisition or an explicit full audit. This delayed detection is deliberate.
Index/control admission and strict expected evidence comparisons continue to
catch wrong identities/references before payload use.

Managed memory remains conservative soft admission, not a physical RSS quota.
Filesystem calls may overrun deadlines; detected overrun or uncertain cleanup
must fail. Existing DrvFS compound cleanup limitations remain. Revisit these
limits if hard real-time operation or parent/control/pruning on DrvFS becomes
required. Revisit the corruption threat model if hostile coordinated rewriting
must be detected. No current learning qualification or formal campaign changes.

## Verification

Independent architecture review precedes code. Required acceptance covers
call-count/I/O regressions, deferred corruption and explicit complete audit,
historical Source/Dataset4 correspondence, Python/Rust parity and early native
refusal, actual bounded pause/resume and controlled Dataset/Tree evidence,
same-original HDD comparison, static/Rust/Python checks and independent final
correctness/conformance. Exact results and NOT RUN limitations belong in
[the verification record](../verification/pack-checksum-read-policy.md).
