# Campaign control-record byte encoding

Status: accepted bounded clarification, architecture READY on proposal SHA
d295c841484a525a8ced0019c7b59c6ca4bf80754a07314b6410eb047651c7d4.
C1 implementation passes independent correctness/conformance review.
Owner: root. Public-entry R1 is accepted; this resolves a concrete integration
assumption surfaced before its record I/O was implemented. No policy, Source,
clock, record-authority, dependency or legacy format change is intended.

## Current source and requirement

CampaignStore.put/get in src/gf_cli/hpo_storage.py uses canonical JSON PLUS one
newline and hashes those actual bytes. Python read_control_record requires canonical JSON WITHOUT that newline.
Native read_owned currently owns original-file reading and raw-byte SHA only;
it parses serde_json without a general canonical/no-newline check. Native prepare
checks its own digest separately. Assuming a global native canonical guarantee
would be incorrect. Silently accepting either
would make a declared file's byte SHA and canonical control digest ambiguous.
New campaign bootstrap/clock/awaiting/disposition are read by both owners, while
dispositions also reference existing proposal-failure and FAIL-tell records.
Those existing records must keep their exact bytes and raw-byte SHA meanings.

## Encoding domains and responsibility

Reserve the NEW campaign-relative supervision/ subtree for the explicit new
runtime's control DATA. CampaignStore.put/get select canonical JSON WITHOUT a
trailing newline for files in this subtree. All existing producer-owned records
outside it retain canonical JSON PLUS exactly one newline and their actual-byte
SHA. No old supported campaign producer used this control subtree. Reject the
wrong style in each domain; do not add a tolerant dual-format fallback or migrate
old files. The native actual control root is the authenticated campaign directory's
supervision/{supervision_id}; its own admission/Prepared/Exit/completion/ACK files
therefore share the same no-newline encoding. Store.get can read those originals
without inferring their authority.

This is a narrow path-domain encoding decision in the actual-store I/O layer,
not a registry of semantic validators or arbitrary kind-prefix acceptance.
Existing path/owner/original-file/budget checks remain. Typed callers retain
closed kind/schema/field/identity validation. Do not duplicate those validators
in generic put/get. Existing put remains immutable. Only the already-approved
update_supervision_cursor CAS may mutate supervision/current.json and its exact
cursor kind; ordinary put must not bypass that mutation boundary.

Native adds one private legacy-campaign reader for ONLY the actual campaign's
canonical proposals/{proposal_no:06}/failure.json, tell.intent.json and
tell.receipt.json (the existing minimum-width formatter, no new maximum proposal
number). It uses bounded owned regular-file/original-identity custody, requires
canonical(value)+one newline EXACTLY, and returns the actual raw-byte SHA. The
caller checks the exact existing hpo-proposal-failure/hpo-tell shapes and current
proposal/reason/FAIL/value=None bindings. This helper is used only for the three
legacy evidence paths in disposition validation. Leave generic native read_owned behavior and existing namespace/legacy callers
UNCHANGED; preserve Python read_control_record strict no-newline behavior. Add a
SCOPED native campaign-control reader/check at NEW campaign consumption points
that requires exact canonical Python bytes with NO newline. Keep the separate
three-path legacy checker above for canonical Python bytes PLUS one newline.
Both return raw-byte SHA after the existing original-file ownership checks.
Factor shared private file-reading mechanics without adding an ownership layer.
No JSON shape, pathname or byte checksum becomes a live capability.

Canonical bytes mean the existing Python producer gf_common.domain.canonical
recipe (sorted keys, compact separators, UTF-8/non-ASCII preserved, finite Python
number formatting), not generic serde_json formatting. Reuse the already tested
Python-compatible formatter currently in native/src/json.rs:279–395. A minimal
pure formatter extraction shared by the existing budgeted JSON module and the
standalone supervisor is allowed; retain existing json::canonical budget/output
semantics and bytes. Do not link supervisor execution to a Python runtime or
copy a second divergent formatter. This narrow factor may touch json.rs/lib.rs
and one new pure native formatter module, without dependencies/public Python API
or numerical/policy changes. Check byte parity for exponent floats, negative
zero, Unicode and the existing integer/string corpus. Existing declared metadata
envelopes/publisher contracts apply; no blanket file cap is inferred merely from
the control socket's packet limit and no new resource formula/margin is added.

For the NEW public campaign route, launch.record_directory is exactly the actual
CampaignStore.directory/supervision/{supervision_id}. Ordinary/direct-operation
roots retain their prior meanings. Historical ACK readback derives that exact
old root from actual current Store ownership plus the OLD disposition supervision
identity, then requires report/completion/operation/bootstrap/ACK paths to match
it and verifies their complete original hash/identity chain. A caller's failure
path parent alone is not a namespace or custody authority.

## Seed-settings control snapshot

The canonical bootstrap seed-settings file is also in supervision/ and uses a
new closed schema1/kind rust-player-seed-settings-v1 with exact fields
{schema_version,kind,sources}. sources is the existing validated provenance-row
list returned by src/gf_cli/hpo.py::_settings_inputs, not paths to reread later.
Each row has exact {path,sha256,settings,origin}; settings has the existing
EffectiveSettings.asdict shape {model,training,identity}; origin has the existing
{campaign_id,study_identity,proposal_sha256}. An empty list represents no imports.
Actual input validation/pinning remains the existing checked import boundary.
The later application reconstructs only typed configuration with pure parsers,
recomputes each effective identity/canonical equality and rechecks the declared
space/fresh-study condition. Preserve its actual model resources. New schema6
uses the accepted same-host preview instead of legacy process admission; old
config1–5 import admission is unchanged. Never import old model state, histories,
objective, live EffectiveSettings/study/store object or Source authority.

Part6c1 may implement this strict DATA snapshot parser and encoding seams. Actual
public import/preview/proposal composition remains Part6c2, as already scoped.
The native bootstrap verifies its bounded file/hash/shape; it does not initialize
Torch, perform model work or claim that parsing granted resource admission.

## Verification and compatibility

Check old Store.put/get exact newline bytes/SHA and immutable behavior; new
supervision no-newline bytes/SHA and wrong-style refusal; native reads actual
Store-produced old failure/tell records, rejects changed bytes/symlinks/foreign
paths/identities. Verify NEW scoped native campaign-control and Python control
readers reject trailing newlines, while generic read_owned/legacy namespace
behavior remains unchanged.
Exercise cursor CAS as its sole overwrite path. Verify the settings snapshot
round-trips existing empty/nonempty validated row DATA with strict keys/identities
and no late source read or live-object reconstruction. Preserve old configuration
and eight formal kernels. No new GPU/game/training or actual experiment budget;
these bounded parser/actual-file checks fit the accepted Part6c1 CPU scope.
Architecture review precedes dependent encoding implementation.
