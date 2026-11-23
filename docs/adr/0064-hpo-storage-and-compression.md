# 0064: HPO storage destinations and diagnostic compression

- Status: Accepted
- Date: 2026-11-23
- Decision owner: root
- Review: independent architecture review cleared the bounded design before implementation
- Verification: [HPO storage and compression](../verification/hpo-storage-and-compression.md)

## Context

The approved 256-game probe needs separate destinations for model and
administrative data, original match data, and transient and retained HTTP
diagnostics. Legacy packed collection can include HTTP captures in the
originals, which makes successful pairs consume persistent storage. HPO update
records are immutable JSON linked from each generation. Both Python and Rust
read the lineage.

The review used Simulator repository revision
`59e2e009af0846063d09963dc9da08e91cf6ab42`, spec 07 sections 14.3 and 21.1,
and the managed TestServer interface in spec 10. The probe config pins
`072dabbca1458c9526897781d2fd64edeeb4f9ca`, gameplay API v2 and catalog schema 2.
The Simulator remains the authority for gameplay and its HTTP stream.

## Decision

HPO TOML owns the save destinations. The optional campaign path is part of the
effective campaign config. Live admission checks both control and data
destinations before publication. Under the template's dedicated WSL path, a
different or missing mount fails admission. HTTP staging requires tmpfs. Where
an ancestor already exists, admission checks the actual mount and write
permissions before creating any descendants.

A managed TestServer has one private raw capture per seat-swap pair. The
legacy collector and Rust match pump use the same settlement policy after
confirmed server shutdown. Classification can use client transport and
response status evidence, terminal outcome, server health, and capture
integrity. Native clients pass their transport evidence through the existing
terminal handoff as separate in-memory metadata. Archive level 3 is the
default. Publication is atomic, followed by staging retirement. Incomplete
cleanup or failed publication keeps the relevant diagnostic location available.
The original storage schema and the policy information boundary are preserved.

HPO lineage selects the update file from the checked storage config. New TOML
runs use a single zstd frame by default. Writers hash the actual stored bytes
and publish them through existing immutable control ownership. Readers bound
stored bytes before hashing and bound expansion and window size before applying
the existing JSON checks. Python uses its standard-library zstd codec. The
native scanner uses the pinned Rust zstd 0.13.3 streaming decoder. Ordinary
PPO and older HPO parents continue to use their original JSON records and
bindings. Reconstruct historical config only when the original recorded
campaign identity matches. Fresh comparison outputs get their diagnostic
policy from the live comparison config after authenticating both input parents.

## Alternatives and consequences

Moving all pair logs to the originals drive would still incur the normal
write volume for healthy pairs and duplicate data needed for investigations.
Keeping gzip or raw JSON for all updates would avoid the new Rust dependency.
The reviewed zstd path supports the requested format and retains an explicit
raw mode in both runtimes. Encoding at the existing write boundary avoids a
second persistent write and a later rewrite of generation hashes.

The retained logs contain private transport data. Keep their staging and
archive custody private, and expose diagnostic status separately from
scientific inputs. The HTTP archive uses filesystem free-space admission and
is excluded from recursive managed-capacity scans. The compression and
destination changes alone do not establish capacity for 10,000 updates.
