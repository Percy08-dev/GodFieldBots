# Corrected Journal RAW entry — bounded design

Accepted after architecture DESIGN READY before the additive FFI change. ADR0024's
accepted Tree3 RAW parser uses corrected Journal6, whereas the existing public
StructuredJournalTokens constructor selects the engineering Journal5 parser.
Expose the already implemented corrected parser through the same owned token
reader instead of adding a second parser, transport or Source authority.

Add classmethod StructuredJournalTokens.card_instance(directory, journal_file,
settings, deadline, cancel_handle), with the same argument types/defaults as the
existing constructor. Its context is fixed to Journal::open_corrected; callers
cannot supply arbitrary profile/schema selectors. The existing constructor and
Journal5 semantics, default settings, output layouts and successful bytes remain
exact. Share construction, Budget/Control, parser, Stage, token/bank export,
recheck, cancellation and physical retirement code; avoid duplicated owners.

Python tree_raw_reader_settings must require the callable, existing corrected
signature, ABI/API and card_instance_tree_raw=journal6;lightgbm3;inference4;codec2
before new RAW processing. The compiled marker remains absent until validation,
token/export parity and both implementation reviews pass. Explicit private
candidate diagnostics can test the classmethod while marker publication is held;
this never grants parent, Dataset, fitting, numeric or PPO permission.

The entry accepts only the acting-participant original journal and its verified
codec sidecars. Canonical/server diagnostics remain opaque evidence; no other
private view, GameSeed/RNG, rejection teacher or future observation becomes a
feature. Corrected baseline/NN/Tree header branches remain closed, including
Tree's null selection statistics and safe own-policy metadata. Use the same
participant binding, perceived definitions and prefix boundaries as Python.

Admit request/settings, source/record parse, copies, arrays and export leases
before allocation under existing declared limits, including64MiB record/array
ceilings and the original finite deadline. An admitted failure becomes sticky
cleanup-only. Export exposes no partial result; retained views keep their actual
quota lease. Stage/files/threads retire before healthy close and ownership release.
Uncertain close retains the actual owner and primary error; no late success or
empty retry proves retirement.
Current owner code does not supply this automatically: the corrected-only branch
must latch admitted diagnostic/recheck failures and preserve actual retirement
uncertainty after core removal. The legacy Journal5 branch remains unchanged;
an empty close retry never proves corrected ownership retired successfully.

Verification: source-matched private candidate exposes both constructors; old
Journal5 literal regressions stay exact. Corrected baseline/NN and controlled
UNADMITTED Tree Journal6 inputs match Python frame/table/event tokens, ordering,
selected identity and dtypes. Reject wrong schema/profile/viewer, malformed or
unsafe metadata, finite Tree stats, over-budget input, changed sidecars, late
results and overlapping escaped leases. Include cancellation, failure and close
tests. No fabricated formal parents, qualified pair or runtime attestation;
genuine qualified Tree gameplay and real-data fits remain user-deferred.

Native implementer owns the FFI/parser/preflight changes in its durable isolated
workspace and tests. Root owns integration, capability publication and packaging.
Independent correctness and architecture-conformance reviews precede acceptance.

## RAW-only allocation ownership clarification

Root adopts the advisor's option B, pending narrow architecture confirmation.
Keep all portable declarations exact: typed model/metadata/provenance, execution
resource formula, array ceilings, caps/costs, labels and selected identity.
Native RAW creates no Booster and does not certify Python inference admission.
Its live controls, parsing/copy overlap, Stage, arrays and escaped export leases
are admitted before allocation to the existing Native budget, without increasing
it or subtracting hypothetical model memory. Python's32*_owned(manifest)+65536
guard remains a CPython graph/alias-specific allocation check under its unchanged
local cap; do not reproduce it as a guessed Rust local-model reservation.

Parity means semantic/token/output parity for inputs admitted by both readers.
Physical admission is representation- and role-specific: a boundary input may
pass Native RAW and fail Python's control-allocation guard. Preserve explicit
sticky resource failure, no partial export and actual lease retirement. Old
Journal5/Tree2 behavior stays exact. No persisted field is needed while this new
capability remains unpublished; tests must demonstrate the documented boundary,
portable formula/array limits and native overlap/exhaustion/escaped-lease guards.

The existing StructuredJournalTokens.diagnostic ABI returns an owned PyBytes
JSON copy, not borrowed NumPy banks. Preserve that output. Admit its complete
control/copy overlap before allocation, forbid repeated/partial publication and
retain sticky failure. Native close retires native core/control ownership;
caller-owned bytes remain valid and must stay charged to the consuming caller
until that copy retires. Tests must not claim a NumPy base-owner lease where
none exists. Escaped native-bank quota tests apply to the separate APIs that
actually export borrowed banks; no new diagnostic output wrapper is introduced.
