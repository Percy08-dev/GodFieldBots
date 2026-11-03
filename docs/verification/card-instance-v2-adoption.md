# Card-instance Simulator v2 migration verification

The user's complete migration/acceptance scope is governed by
[ADR0024](../adr/0024-card-instance-simulator-v2.md). This ledger distinguishes
partial implementation milestones from complete migration acceptance.

## Initial evidence and design

- Baseline Bots revision: `8283db9`. Initial working tree: clean.
- Simulator HEAD: `07e8185f98c503aa41200a80ea554b4fd21088ad`; its separate
  working tree includes pre-existing untracked files. No Simulator files changed.
- PASS: thirteen-file normative inventory SHA
  `8381079eae8a68378d8419c7f7245c62358833f0dd9045667021e112a1e9bad5`, independently
  recomputed from sorted canonical UTF-8 JSON path→content SHA256 mappings.
- PASS: closed catalog2 / `public-ja-v3` metadata and content SHA
  `f51a3c73feb8b3f01837e65768aafb5a3aa805a33483308130244b33b16a8ad6`.
- PASS: twelve original experiment effective configurations/input hashes and the
  default ExperimentConfig SHA captured before implementation in
  `runs/assessments/card-instance-v2-20261103/before-compatibility.json`, receipt
  SHA `752f978af0c6a26062f285c030a76301f226eb0adb03e7a47c1eb1219d8d4944`.
  Default Config SHA remains
  `bdb2b3663397693baf9bc059e9831c17ea91cbc85a3184e30d3d2f1aa6f0fa33`.
- DESIGN READY: independent architecture review for the overall staged direction
  and bounded stage1 contract/wire/Config/direct HTTP. A blocking assumption that
  all schema versions were integers was corrected to the public representation
  matrix before implementation. Later stage design gates remain explicit.

## Existing-runtime public read experiment

Command:

```sh
.venv/bin/python runs/assessments/card-instance-v2-20261103/read_only_runtime.py
```

PASS using existing debug TestServer executable SHA
`f2fb42e8598705a1c41a087b9210b114bdf91b732512a02d143856059cb656cc`.
Management `/debug/v1` attestation matches startup identity/effective config;
`/v2/info` declares the new contract and the full296-definition catalog hashes
correctly. No game was created. Process exit0, shutdown `stdin_eof`, no log error.
Receipts/raw public payloads/harness are retained under the assessment directory.
The first sandboxed attempt failed with socket PermissionDenied; that failure is
preserved separately. The local-only execution then passed with sandbox escalation.
The executable's build correspondence to source revision is not independently
attested by HEAD; actual binary identity and public wire behavior are separate
evidence.

## Implementation and remaining acceptance

Stage1 contract/wire/Config/direct-HTTP implementation is accepted after both
independent final reviews. No worker/model/native/Source capability is enabled.
Stage2a pure frame/token implementation and both independent source reviews are
accepted after final host regression checks. Stage2b candidate generation
and stages3–5 (model/worker/journal/native, Source/learning/artifacts/evaluation
and full runtime acceptance) remain closed pending issue0007.
No192-game pilot, real PPO resume matrix, real decision performance panel or
complete migration acceptance is claimed.

## Stage1 scoped implementation acceptance

Implemented closed eleven-field contract/actions3; explicit ServerConfigV2
dispatch before defaults, unchanged old default/Config hashes; original public
HTTP/Replay/Dataset validators; explicit `AttestedClient(contract=...)` sharing
debug attestation/retry/poison ownership. New Config validation works. All
unfinished consumers reject before resource allocation or Simulator I/O.

Both independent reviews are READY. Blocking findings resolved:

- Direct legacy environment and participant-worker constructors previously
  accepted actions3 before initialization/Popen. Exact supported-type guards
  now close those routes, preserving the existing private component seam.
- Dataset original rejected attempts can contain duplicate/>18 references.
  Private syntax preservation is distinct from strict submitted/accepted Action
  checks; it grants no Source/label/teacher authority.

PASS evidence (counts overlap; do not sum them): implementer299related tests;
architecture270ownership/contract regressions and later78Dataset rechecks;
correctness186transport/config/runtime/cleanup cases; latest79test/fixture cases
and independent3-case checkout-origin-delta check. The12original hashes/default
hash remain exact. Simulated path-origin relocation passes; actual checkout
relocation is not claimed. New fixtures retain original public DTO/hash origins.

Exact root checks:

```sh
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src
.venv/bin/python -m gf_cli validate --config configs/experiments/card-instance-v2-contract.toml
git diff --check
uv build --wheel --offline --cache-dir /tmp/gf-structured-uv --out-dir runs/assessments/card-instance-v2-20261103/stage1-wheel
.venv/bin/python runs/assessments/card-instance-v2-20261103/contract_runtime_smoke.py
```

All PASS. Mypy covers122source files. Wheel content matches all122current Python
sources; SHA `2c048f836455b2ca36e5fab20d77ad0da6fb666329b96d35b78daba234a94983`.
Final direct-runtime smoke is `contract-smoke-2`: same fixed seed, new directory,
Create/Join2/acting Observation/options/events/Replay/delete, exit0/stdin_eof.
Its Replay is empty and events count0; nonempty/aborted cases are synthetic
validator evidence, not a completed game. No installed-wheel/native/CUDA/new
learning experiment is claimed by this milestone.

The first full CPU pytest command, `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m
pytest -q`, stalled under the sandbox at
`test_native_control_admission.py::test_source_mode_rejects_numeric_alias_with_coherent_hashes[0]`
after678progress markers. It was interrupted with exit130; its complete partial
log is retained as `stage1-full-pytest.log`, not a whole-suite PASS. The identical
case under host execution, bounded90s with `-vv -o faulthandler_timeout=30`, passes
in3.73s (`stage1-native-stall-probe.log`). This does not prove a root cause or
whole-suite success. A bounded full host run follows the next frozen source
increment; no tests or candidate domain are removed to obtain acceptance.

## Public generator contract limitation

Historical initial finding, superseded by the later owner confirmation below:

Independent bot_advisor review confirms three unresolved public predicates at
the pinned revision: Flash counting for defense miracle+CostCut, Discard/
Sacrifice kind–state–subclass eligibility, and per-utility accepted targets.
Root accepts the finding and records the exact separate Simulator clarification
task in [issue0007](../issues/0007-card-instance-v2-public-candidate-contract.md).
No Simulator files changed, rule inference/probing performed, seeds replaced or
candidate scope reduced. Complete generator adoption and dependent real-game
acceptance are BLOCKED pending an explicit amended-contract adoption. Independent
frame/token implementation continues under its own design gate.

## Stage2a source and scoped verification

Shared pure layout/record/token implementation follows ADR0024§9. New immutable
layout descriptors sit below records/tokens; thin frozen record subclasses add
no serialized fields. Exact own-visible definitions match296pinned per-definition
digests. Original refs/kind/state/order, duplicate learned definitions,25events
and historical prefixes are retained. Expected layout defaults strictly to old
semantics. Codec1/old numeric consumers reject new tables before allocation;
new synthetic128tables cannot reconstruct Actions even with validation disabled.
No codec2, worker3, model3, native recipe, journal6 or Source capability is enabled.

Both final source reviews are READY, including a narrow call-local layout reuse
fix. Independent architecture189tests plus85post-fix cases pass; correctness171
plus92post-fix cases pass. Counts overlap. Both reviewers independently execute
`f0c5636` module text read through Git and reproduce every original golden bank/
frame/table/token/signature/codec byte. All296definition digests match the public
catalog; bottom import isolation passes. Correctness independently enumerates
all25event codes/counts and reproduces both recorded live-smoke frames/tokens.
Implementer423related passes include one inadvertently repeated records module;
the final post-fix239-case command has no such repeat. New stage2modules contain
35cases. These are pure compatibility checks, not learning/Source evidence.

A pre-existing fixture packaging mismatch was independently investigated:
`structured_history62_files.json` lists38files including an empty runtime `.lock`,
but the introduction commit and current fixture contain37payloads and no lock.
Do not create/repair a lock in the frozen fixture. The reviewed test-only remedy
permits only a truly absent literal `.lock` with the known empty-file digest;
if present it must be a regular file with that digest. `lstat` rejects dangling
symlinks/non-file entries, and all37payload checks remain mandatory. Six copied
fixture tests cover this. Inventory and frozen payload bytes are unchanged.
Early implementation failures from missing legacy reexports were fixed by
restoring those exports; no new signature is substituted for an old one.

The same32768-row legacy table constructor measures median0.289615→0.058134s
over3samples after avoiding per-row repeated contract validation. Its exact hash
remains `68e46a1c75289d6d21df918d2083cb90bf5634f0668b7c1e4fa30e62d3237ee0`.
This is constructor-only evidence, not a full search/latency acceptance panel.

## Stage2a runtime, old-data and package checks

PASS: `structured_runtime_smoke.py` creates one diagnostic game and builds two
separate participant frames. A public normal-weapon singleton and empty response
round-trip through the new compact table/tokens back to original refs, and both
HTTP Acts are accepted. Actor1has0visible events, actor2has3; nonempty Replay
validates. Process exit0/stdin_eof. Receipt: `structured-smoke-2/report.json`.
This incomplete game is not a pilot, a complete parent or an eligible Source.
The first attempt used raw HTTP options where the harness needed the explicit
viewer-bound participant envelope; its failure is preserved. Only the harness
binding changed, with raw options saved separately; no rule was inferred.

PASS: all five original full-data proof commands and the independent verifier's
semantic audit. Fresh outputs: `legacy-proofs/`. Both corpus encoders retain
83783accepted actions,13568journals/534640records/6784completed episodes and exact
original dataset/sample/arrays/archive hashes. The83783-row Parquet and sidecar
are exact. Saved PPO1/4worker probes retain16episodes/224decisions and all8arrays
across all3readers;72original source hashes/sizes/mtime/inodes recheck exactly.
Only diagnostic clone paths/manifest pointers change, with zero updates and no
new eligibility. `execution.json` and actual output logs are execution evidence;
the copied `bound-plan.json` remains a template and is not a completion receipt.
The24protected fit-file hashes also pass a fresh root check; the historical
baseline has no stat identities, so none are claimed for those24files.

PASS: unchanged Rust sources, `cargo test --manifest-path native/Cargo.toml
--offline`:114passed/2ignored. Installed native ELF remains
`d093ccaa68e8359b9f1a9619754a65dc733747aa31a8a1578c9a299f051b1e5e`, declared
native source `b29d025b2f547440cced90ecdd5448e19e29ea311b3b5d3fe196847ec5185206`;
independent proof identity/direct ELF checks match the prior verified candidate.
No native source or installed binary changed, and no v2card-instance capability
is advertised.

PASS: whole Ruff/mypy124source files/diff checks. Python wheel SHA
`4b1bd6d987e39757702938c0c927d005fc129dc7870d0868536ee56006e9b3eb` matches all124
current package sources exactly. A separate Python3.14environment contains only
the wheel and8mandatory distributions (1110copied/hash-checked dependency files),
passes dependency checking, loads the new Config, and starts/reaps Random and
heuristic workers with all optional Torch/LightGBM/PyArrow/native packages absent.
Expected worker SIGTERM exits-15 have empty health; the initial harness's mistaken
exit0assertion is preserved separately and corrected to the actual worker cleanup
contract. Existing environments/dependencies remain unchanged.

Full frozen-source HOST pytest PASS:2181passed/55skipped in1445.77s, exit0, under
the2100sbound. It is CPU-only with native Tree/frozen Dataset/trained-default
fixtures enabled. The central original-source fixture recognizes only Source17/
29reuse variables, which this command did not set; the Source43variable does not
reuse that central fixture. It therefore creates all3old engineering/FakeHTTP
parents fresh (seeds17/29/43), each complete after2updates and64natural validation
games. `full-host-central-fixture.json` verifies those conditions. These are
old-format component checks, not new-contract or formal learning parents.
The120sfaulthandler emits a diagnostic stack during the slow central fixture;
the case and whole run subsequently complete successfully. Real Simulator and
CUDA opt-ins remain off;55skips are not passed tests. Log SHA
`bd988704dc3ba8def4433c1029cab281ebea875e6e7336bf098ba2e48987e6b4`;
`stage2a-full-host-conditions.json` records exact flags and completion.

After the test process exits/reaps, all124source bytes still match the tested
wheel map and the24protected fit-file hashes recheck exactly. The full suite
leaves an empty runtime `.lock` in the historical fixture; its precise test
creator was not established. After confirmed process completion, root verifies
regular-file/empty bytes/inode, acquires a nonblocking exclusive flock, and removes
only that initially absent transient file. The37payload hashes and committed
inventory bytes remain exact. No source fixture was repaired or data rewritten.

Root accepts only stages1and2a. Actual new-generator/model/worker/journal/native/
Source, CUDA/PPO/pilot/fixed-data/performance acceptance remains NOT RUN/BLOCKED
under issue0007. Reopen those stages after explicit amended revision/hash
adoption; never relabel these component records as qualified Sources.

## Owner clarification and declared-rule runtime conformance

The owner subsequently confirms all3predicates, including an explicit follow-up
that every scoped utility can target self/opponent. The exact
[owner record](../contracts/owner-20261103-card-instance-candidates.md) has SHA
`3f3090f521041293bcb6de2a3cbac7e74ffc635b4acc0f7f02c957bbcf0e44ee`.
Root accepts that authority and supersedes the blanket requirement for another
Simulator revision just to resolve ambiguity. Actual Simulator HEAD, normative
inventory and catalog hashes remain unchanged. A separate candidate-rule
signature will bind the confirmation; Config fields/token shapes/hashes remain.
Section11's pure-component generator design passes architecture review; no
production implementation or real qualification is accepted by that review.

Read-only source inspection finds supporting named utility cases in
`tests/card_rules_v2.rs` (SHA
`b08c2053a3c57e20ce3a9194c97a28e739ce22f08afe0f7a7c5f83420823cdb4`).
Related `card_instance_engine_contract.rs` SHA is
`413d0b9a46e3d46b99ca4eb10f0a1e662aa0b68e466c2bc2966b2f12a0fa6c1f`;
`canonical_v2.rs` SHA is
`ffd90a0cf5c0dbf584ca69f1935d39bdeba9cf54e46c07ccbc84c238bb6b718d`.
Those sources do not directly establish all3cases: existing learned removal is
Pray or Dreaming-Hat response, not an explicit Discard. No fresh Simulator unit
test execution is claimed.

A bounded check of the already declared rule reproduces a different blocker:
both existing debug/release binaries reject Discard of a CURRENT OWN learned
Mudflow ref with HTTP422 `artifact_not_owned`. The same fixed GameSeed1was used,
with no reselection. Play-to-learn and the opponent's unused Avalanche Discard
both succeed first. Independent correctness review validates every current
Observation/ref/state/choice/revision/hint and exact raw HTTP command, with each
UUID sent once and no intervening transition or resend. The rejection is only
diagnostics; no policy predicate/mask/teacher rule is inferred or narrowed.

Both owned servers exit0/stdin_eof without log error. Receipts/commands/Observations/
HTTP logs and the harness are under
`runs/assessments/card-instance-owner-20261103/discard-conformance-{1,2}`.
The debug/release hashes match the recorded existing binaries. Binary-to-source
build correspondence remains unverified, so the confirmed P1finding applies to
those binaries rather than an inspected current-source handler. Flash/Sacrifice
are NOT RUN in this diagnostic. [Issue0008](../issues/0008-learned-miracle-discard-runtime.md)
records the exact separate Simulator correction/source-attested conformance task.
Pure candidate work continues against the owner rules; real I/O/Source/fitting
qualification stays closed pending corrected-runtime adoption.

## Stage2b pure owner-rule candidate acceptance

Implemented `CardInstanceRules`/pure sequence analysis, PublicCandidateGeneratorV3
through the existing shared bounded search engine, distinct rule-bound reports/
generated DTOs and explicit v3 wire helpers. Rule signature is
`1480f832b98c0b78b894a823681848b1dc11093b3a6a2aa462dd2069ed579a36`.
ActionConfigV2/V3 fields/hashes, old report shape, token signatures, budgets and
all runtime/model/worker/native/Source gates remain unchanged. No new dependency.
Discard includes learned miracles under the owner rule despite issue0008.

Both final independent source reviews are READY. Implementer342related tests
pass; reviewer96new+111legacy cases pass, with32independent exhaustive cases
across8hands×4MPlevels matching Actions and exact count/discovery. That oracle
does not use production prefix pruning/sequence analysis. Matrix coverage includes
24miracle subclass/state/EndTimes cells atMP0without offensive availability,
19utilitytarget cases, CostCut salvage/suppression/payment/summary, attack order,
Flash response limit plus unrestricted-offense control, trade/exchange/phenomenon,
mandatory coverage, long bands, resource limits and rule/wire rejection. Counts
overlap and must not be summed.

Before extraction, four fake-clock legacy generator cases were captured from
unchanged214b1fd. Fixture SHA
`ed0c5a377d7db69ce4f3662fbc9bac19a694d88127833259070bd132ac9a2dfd`;
canonical content SHA
`40d67782e419c8d7ce6208adee4d0a37a63ea33972f34e84696865848c5f5317`.
Independent verifier compiles committed generator text read through Git into a
separate namespace and reproduces every table/order/action/count/cutoff/full-RNG/
fake-clock field: exact8rows/12attempts/55clockcalls; trade16/34/108; bounded
60selected/4290discovered/7499attempts/8845clockcalls; upper-bound30/51/149.
All12provenance hashes match the old commit. Active lower imported dependencies
remain byte-identical; generator and wire are intentionally modified, so the
whole provenance-listed file set is not claimed unchanged after implementation.
No original frozen fixture is rewritten.

Root's bounded HOST regression command passes390tests in52.72s, exit0:

```sh
timeout --signal=INT --kill-after=15s 600s env CUDA_VISIBLE_DEVICES='' GFB_TREE_DATASET_FROZEN_FIXTURE=/tmp/gfb-tree-dataset-originals-20261102-d/tree-dataset-originals0 .venv/bin/python -m pytest -q tests/test_card_instance_rules.py tests/test_card_instance_actions.py tests/test_diverse_actions.py tests/test_structured_protocol.py tests/test_card_instance_structured.py tests/test_card_instance_tokens.py tests/test_structured_source.py tests/test_structured_ppo_source.py tests/test_structured_tree_dataset_frozen.py
```

Whole Ruff/mypy125source files/diff checks pass. No source changes follow the
frozen reviewed code. The earlier2181whole-suite run remains stage2a evidence;
it is not represented as a post-generator whole-suite run. Native sources and
binary remain unchanged; no new native recipe is published.

Root's live-input component witness uses only the declared accepted setup
actions, then fetches the actor's dedicated full Events prefix, current own
Observation and same-revision ActionOptions. The new generator selects531rows,
including exactly one singleton learned-Mudflow Discard. Its v3 wire roundtrip
restores that original ref and the same rule signature. No failing Discard is
submitted again, and no rejection diagnostics enter the input. Receipt/input/
generated data: `runs/assessments/card-instance-owner-20261103/discard-conformance-3`.
Owned server exit0/stdin_eof. This is component evidence, not corrected runtime
conformance, a complete game, a Source or a learning result.

Python wheel SHA
`21965448a5c72ae527163361c512f38ea2d78b9421a10a08feb0be883013f49e`
matches all125package source files exactly. A separately installed mandatory-only
environment passes dependency checking, starts/reaps both baseline workers, and
reproduces the rule-bound531-row component result with no Torch/LightGBM/PyArrow/
native package installed. Receipts/harnesses are in the owner assessment directory.
Root accepts pure stage2b only; at that milestone real acceptance was blocked by
issue0008. The subsequent correction verification below resolves that defect.

## Simulator correction verification — 2026-11-04

User supplied Simulator commit
`91d5926a7572db327eaea3751e7c24f428120996` (parent07e8185). Root verifies the
existing corrected binaries and the formerly failing case without changing Bots
production code, Config pins, input signatures or eligibility gates.

PASS: both actual debug/release fixed-seed HTTP runs accept learned Mudflow
Discard at revision2→3. The harness is byte-identical to the earlier conformance
harness; all3actions/before Observations match the previous case, with fresh
command UUIDs and output directories. Selected own ref is removed, the retained
list/state/order is exact, resources are unchanged, no draw occurs, and both
processes exit0/stdin_eof without log error. Independent correctness audit is
READY. Original FAIL/422 evidence remains unchanged.

PASS: prebuilt focused test binaries, with no Simulator rebuild or source writes:

- debug Engine GF-DISC-003:9tests, including30miracles×2states×2operations=120cases
  atMP0; learned-only/empty/full ActionOptions, order/state retention, atomic
  rejection, Sacrifice refill/Devil slots and Snapshot continuation;
- release Engine: the same9tests;
- debug and release HTTP:1test each for exact retry and spent-ref rejection at
  the correct actor/revision;
- debug canonical-v2:1test for strict old-rejection incompatibility in Gameplay
  Replay, Episode Reconstruction and Dataset, plus corrected accepted records.

All5commands exit0 (11debug+10release test executions; do not count the repeated
matrix as240distinct rule cases). No current release canonical-v2 test binary was
available. The live witness covers normal Discard; Sacrifice evidence is from
the focused tests. Flash status and general utility-target tests were not rerun.
No full Simulator/Bots suite or learning acceptance is claimed in this check.

The13-file normative inventory recomputes to
`131890dfe579bc6951f5189e460971d33658339106e219a2e00b1a289a0534d0`.
Catalog/schema/wire versions and catalog SHA remain unchanged. The nine-file
change inventory recomputes to
`e34b7160f956b6510243077d418a0a96f34f6e271077021641b2fd50a9870346`, matching the
Simulator investigation record and all committed changed content. Actual debug
SHA `b3670dac3eb099ba4363058f8d92a44976ebaaf1dfdc187df5a9fe333ec0e4a1` and release
SHA `76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d` match the
handoff artifact receipt. That record's uncommitted/base07e8185 wording is its
historical handoff state; the reviewed bytes are now committed as91d5926. This
establishes content/receipt correspondence and actual behavior, not a fresh
reproducible compiler build.

Evidence root: `runs/assessments/card-instance-server-fix-20261104`.
`verification.json` SHA:
`0e3f0c6ffc38e414d0a1906b718ceae4783c5bd443a156dd51c719f4297b73a0`.
It includes5test-executable hashes, log hashes, before/after identity checks and
both fresh runtime receipts. The unchanged same-wire Bots validator was used
only for diagnostics, explicitly accompanied by the actual new revision/hash;
these are not new-contract policy Sources.

Root resolves issue0008's runtime defect. Before new collection, explicitly adopt
91d5926, its normative inventory and semantic compatibility policy. Spec06§31.1
requires old incompatible rejected Attempts to fail strict revalidation; rebuilt
ActionOptions can also differ, so full old Replay JSON byte identity is not
promised. Preserve old data/pins/signatures until that adoption is implemented.
Worker/model/native/Source integration and planned full acceptance remain pending.

## Explicit corrected-profile adoption — 2026-11-04

ADR0024§12 implementation and both final reviews are READY. Root accepts the
same-class two-full-tuple CardInstanceContract registry, explicit corrected
factory/profile04, separate corrected layout/no-field record variants and exact
profile compatibility. Corrected bank initial/frame hashes bind full contract.
Old07efactories/defaults/constants/presets/record hashes remain unchanged; raw
Delta parsing still supplies context and append verifies the bank chain.

Corrected token signature:
`72db3a64f6d087a5f889c4e3cf6b4eb46a8595fd334e1249ca2d51f76496e1ad`.
Corrected candidate-rule signature:
`3f55f044225fb20c63d683145cdceb397c36f0b7161c7b16793f72210db3e86d`.
Tuple/codebook widths and Rules1/owner record remain unchanged. New corrected
Config/server defaults select the release binary/revision pair before merge and
fresh `runs/card-instance-v2-91d5926` output; old presets stay exact.

Implementer551affected tests PASS17.81s, including60new cases. Independent
correctness271cases and architecture246cases PASS; counts overlap. Mutation
audit accepts only2of8profile/revision/spec tuple combinations and rejects22
single-field mutations. Actual13spec bytes match the corrected map. All13prior
experiment hashes, original Config/contract/actions defaults, old token/rule
signatures and goldens match the pre-edit receipt, SHA
`3f0768e9aed4118c0f18ee6f765b8892f2df9eaa1400b56b035712d9b88789c3`.
Whole Ruff/mypy125source files/diff checks PASS. No model/native/Source gate lift.

Root's corrected-profile witness uses original dedicated Events/current own
Observation/same-revision hints, corrected Frame,531candidate rows, v3 wire and
actual tokenization. Original learned-Mudflow ref restores and submits at2→3;
retained card identity/order/state/resources match exactly, Replay validates and
server exits0/stdin_eof. Independent audit reproduces frame/wire/tokens and checks
actual HTTP action/ack. Receipt `discard-conformance-2` in
`runs/assessments/card-instance-corrected-adoption-20261104`. First witness1had a
harness metadata label storing recipe under token_signature; it is preserved,
then actual tokenization/correct signature is checked in fresh witness2.
This is direct conformance, not192pilot, complete parent or qualified Source.

Wheel SHA `8b66e54bff9650bbd6064c70858b372637ff1c769d8470bf046f79a18d20bc70`
matches all125current Python sources. Separate mandatory-only installed baseline
reproduces531corrected-profile rows/rule signature and starts/reaps both baseline
workers with optional Torch/LightGBM/PyArrow/native absent. Model3/artifact4 design
is independently READY; implementation and remaining end-to-end acceptance follow.

Hardware check limitation resolved: default sandbox cannot access GPU, but HOST
Torch2.14.1+cu132 reports CUDA available/device1 and nvidia-smi identifies RTX5070Ti
with16303MiBtotal VRAM (12749MiBfree at capture). Preserve12GiBmanaged budget;
this availability check is not CUDA numerical/performance acceptance.
