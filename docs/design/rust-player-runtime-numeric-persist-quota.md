# Model4 numeric persistence within the shared Source quota

Architecture READY on proposal 77ca6742b15702c94eea009220e0dd08b53ba808583fc3a14a419e6aca98009a.

Local persistence refinement for an observed failure on the prescribed
32-game Model4 workload. The actual controlled diagnostic reaches
`OwnedNumericBatch.persist` and fails `StorageBudget.acquire` before creating the
numeric output directory. The existing account recorded 1,861,907,296 bytes
charged, a 328,614,493-byte codec request, and a 2,147,483,648-byte limit. The
independent numeric envelope check had already passed. These are managed charges,
not measurements of each live Python object or physical device memory.

The existing numeric writer supports explicit shard byte limits. Its scratch
estimate is 35 times the lesser of the shard limit and actual owned bank bytes,
plus the existing metadata term of 32 times the exact serialized metadata length
plus 65,536 bytes. Retain that formula, the existing per-array constraint and
both budgets. Keep the existing default when it fits. When that request exceeds
available capacity on the same actual Source or numeric envelope, choose the
largest supported shard byte limit that fits both current allowances after
subtracting the same metadata term. Require it to hold the largest actual owned
array without splitting or copying it. Refuse before output if this is impossible.

Use existing `OwnedNumericBatch.persist` with the selected explicit shard limit
or a narrow internal method that preserves its validation, actual reservation,
one-use state and directory creation boundary. Invoke this fallback only from
the actual Rust Model4 Source. Existing legacy callers and default output stay
unchanged. The established writer still reserves actual scratch after planning;
a concurrent loss of capacity fails the ordinary admission check. No waiting,
retrying a failed publication, increasing quotas, freeing live Raw data,
changing learner endpoints or changing any numerical tensor/metadata schema.
Smaller shards may change the number and digest of numeric artifact files. The
existing reader accepts their recorded shard descriptors and hashes and validates
identical array content; no old dataset is migrated. The producer continues to
record its exact code and files. The comparison explicitly records resulting
numeric shard counts and whole save/update timing with its other diagnostics.

Verify an actual CPU-backed reservation reproduction, unchanged default when
it fits, existing decoder equality for default and reduced shards, and refusal
before output when even the largest array cannot fit. Source concurrency and
legacy shapes retain their existing contracts. Then rerun a fresh actual
32-game Model4 case with a matching approved producer and unchanged budget.
The failed parent remains failed. This decision needs independent architecture
review before implementation because it changes how the existing persisted
numeric shards are grouped. It does not change the record format or any
experiment workload, limit or adoption threshold.
