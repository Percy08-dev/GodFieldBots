# Reuse verified inputs through ordinary Rust packing

The supplied working-tree patch from
`tmp/rust-player-trusted-candidates-55cd288` is archived at
`runs/rust-player-runtime-v1/components/all-player-rollout/trusted-candidates-handoff/source.patch`.
Its logical packed Journal reader removes a duplicate table validation.
Mandatory selection validation still checks the whole candidate table and the
selected action before issuing each record.

A table hash identifies bytes. For reuse, the parser must first validate those
bytes against their participant view. A reproduction on the supplied patch
showed that an ambient switch could skip this first validation and admit an
out-of-view candidate. The production path takes verified records directly
from their owner.

## Ordinary Source scope

**Candidate conversion.** The actual registered ordinary Rust Model3 Source
passes its immutable `EpisodeRecords` to private token conversion. Every exact
frame and table pair was already fully checked in the records constructor.
Reuse the canonical encoding body, retaining layout and model position-capacity
rules. The source-bound context takes its records directly from the ordinary
Source.
The public tokenizer performs full candidate validation on caller-provided
tables.

**Pack access.** The ordinary live projection borrows its exact acquired
pending Pack owner for the existing synchronous manifest and episode inspection.
Begin with full original verification on that owner and preserve the full
`owner.finalize()` after inspection, before returning records or issuing Source.
Inside that interval, retained index metadata and verified group bytes may
be reused. The underlying Pack reader still validates actual newly loaded
groups and control bytes, holds group loans and enforces the original deadline.
The interval is local to the acquired Pack owner and clears on every exit.
The full original check applies outside that interval, including HPO, Cold and
resume.

**Numeric packing and publication.** The actual Source is established once
before ordinary synchronous packing. Ownership, current generation and data
stay bound to that Source throughout the call. Internal loop and shard
callbacks check elapsed time and actual capacity through the existing bounds.
Run the current full original check immediately before writing the completed
`rollout.json` manifest, then retain the original existing numeric file
ownership checks and optimizer admission at the update transition. A private
before-manifest hook supplies the full check on the ordinary Source path.

## Verification

Both independent reviews accepted `30a8ab1` after the live Pack owner was
restored to the exact reader type required by the Journal parser. The added
regression parses a valid Journal through the real owner constructor.
Forty focused CPU checks cover byte-identical output from identical inputs,
out-of-view candidates, original replacement, failed final checks, deadline
expiry and capacity refusal. The [public CPU verification](../verification/rust-player-runtime.md#verified-input-reuse-in-ordinary-packing)
records the completed training update and independent readback.
