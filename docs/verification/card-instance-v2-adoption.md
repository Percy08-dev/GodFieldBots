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

## Model3 component and corrected baseline integration — 2026-11-04

ADR0024§13 shared-layout neural kernels, standalone component policy and
inference/resume4 implementations pass both independent reviews. The CPU matrix
has434PASS/27CUDA skips; separate HOST execution of
`GF_RUN_CUDA=1 .venv/bin/pytest -q tests/test_model3_network.py tests/test_model3_artifacts.py -k cuda`
has27PASS across hand3×model3 numeric/gradient/cache, empty-input and full
Adam/RNG resume cases. The wrong destination shuffle-RNG regression now rejects
before network binding, optimizer or RNG mutation; independent targeted review
is READY. These are component checks, not real PPO or model-strength acceptance.

Nine pre-Model3 model2 initializer/state/RNG goldens were independently reproduced
from committed code; all match. The fixture SHA is
`0c7ea418f1e624584397dd250befc8f5a65c71dabdd280b8e89950d7e97ab4e9`.
All14pre-Model3 experiment Config hashes are retained in the assessment receipt
`pre-model3-configs.json`, SHA
`44674705ab4a29a36d6169890d89dc3bf9358facc90d735207339546fb169e9e`.

ADR0024§14 corrected baseline protocol3, codec2, journal6, runtime execution
attestation and collection5 implementations pass both independent reviews.
The raw HTTP atomic-hint viewer binding and late publication-close fault findings
were fixed and re-reviewed. Independent correctness has136targeted PASS;
conformance has157PASS. The healthy two-game real-server smoke has4original
journals,44accepted Actions, matching original commands/Replay closures and
all59inventory hashes. Its run is
`runs/card-instance-v2-91d5926/card-instance-v2-91d5926-contract-c6983ca8-9ff8-42e8-acbe-5ac34222e34f`.
The first zero-decision smoke failure remains retained separately.

The first exact fixed192-game pilot FAILS: all192planned games are recorded,
180natural and12incomplete (11rejections,1contract failure),84complete pairs,
285.229s. Config SHA is
`98ce96a3fd39e07c37d511620eea9b947b88dce8164f34921d1c05761582f3e6`;
all134Python source hashes remain unchanged throughout the run. Group/process
health failures are empty, but comparison_valid and healthy are false. The run
publishes only incomplete diagnostics, not healthy final report/manifest names.
The entire population remains ineligible, without replacement seeds or inferred
legality rules. The retained run is
`runs/card-instance-v2-91d5926/pilot/card-instance-v2-91d5926-pilot-c711832a-8a3c-44de-b672-4c6bac4a0426`;
assessment receipts are `pilot-conditions.json` and `pilot-result-1.json` under
`runs/assessments/card-instance-corrected-adoption-20261104`.

The independent public-rule audit confirms a Bots omission of the already
authorized normal Discard exclusions in ADR0005/ADR0023. The shared V3 fix
restores those exclusions only for artifacts outside EndTimes, preserves all
miracles, uses perceived IDs and applies to both card-instance profiles. Eight
new exact-subset cells cover both profiles/phases/Dream views; implementer225
and independent165affected tests PASS. Config/token/rule signatures are unchanged
by that rule-conformance fix; the failed population is not rewritten.

The event diagnostic is also a Bots input omission. The failed delta is absent
from J6 because validation precedes append. All persisted J6 definition IDs are
known catalog cards. Separate public Player Replay DTOs show guardian_set and
guardian_left events with definition_id="earth", a public Guardian identity
outside the card296namespace. Replay diagnoses the failure only and is never
used to rebuild the missing original prefix or infer candidate legality.
Validator-only relaxation is insufficient: interval categories and exact history
counts also assume card IDs. ADR0024§14.1 therefore requires architecture review
of the corrected layout repair, new input signature/derived widths and explicit
historical91receipt handling. Prior Model3 numerical results above describe the
preceding layout and must be rerun for the repaired input. Simulator revision,
catalog and public rules remain unchanged. No further Simulator fix is indicated.

Full-suite/package/new native/Source/PPO/Tree and final acceptance remain pending.

## Repaired Guardian input — 2026-11-04

ADR0024§14.1 design and both independent implementation reviews are READY.
Root accepts the corrected-only namespace/input repair and the historical91
fail-closed disposition. The old07e and engineering/legacy layouts, original
fixtures and configuration identities remain unchanged. The normalDiscard
correction above also passes its independent review.

Current corrected fixed128 token/layout signature is
`ed0c61152b78fa984aad342be22c6b9d6ef49b12b8f0d1c0f0de89724468002f`.
The earlier72db… signature remains historical and cannot load into the repaired
runtime. Current event/history/default-frame widths are405/387/611; actor width
remains1432. The event306-ID vocabulary preserves all296card indices and appends
ten public Guardian IDs; exact history and interval tokens retain those IDs.
Protocol3 and Journal6 require the new layout signature before acceptance.
Default Model3 architecture identity is
`cc49eff39f3dd3dd93dce396ec4353fa72a05efaa2dc22000680c3502d65e5c8`.

The repaired-input CPU suite/full Model3 matrix has803PASS/32deselected.
Actual HOST `GF_RUN_CUDA=1` network/artifact selection has31PASS/65deselected,
including the original27matrix cases and4invalid-shuffle-RNG guards. Guardian
set/acted/left events are included in numerical inputs. Independent correctness
has286CPU PASS/31CUDA-skipped; conformance has331PASS plus8targeted artifact/
signature PASS (overlapping counts). Both reviews report no remaining findings.
Whole Ruff, mypy134source files and diff checks PASS. Whole HOST pytest and
source-matched/optional-free wheel checks are running; no completed result is
claimed yet. The complete same-condition192pilot has not yet been rerun.

Exact commands, source hashes, all nine model identities/resource estimates and
logs are under `runs/assessments/card-instance-guardian-repair-20261104`.
Receipt `after-layout-and-checks.json` SHA is
`74a9d28b6fa3a231c0aea1a5747f410e8e7506516feb488f0981170a51a1a66e`.
The original public diagnostic is separately retained as
`guardian-event-diagnostic-1.json` in the corrected-adoption assessment directory.
No original journal/prefix/artifact was repaired or relabelled.

ADR0024§15 neural game-inference design is independently READY; implementation,
real NN smoke, Source/learner/coordinator/native/Tree and full requested acceptance
remain pending. §16 qualified Python Source/numeric/PPO design is in review.

Follow-up whole-project checks PASS: HOST `.venv/bin/pytest -q` completes
2554PASS/140SKIPPED in1400.08s. Skipped tests are not counted as executed; the
separate31actual CUDA cases above cover the changed numerical/artifact scope.
The log is `whole-host-pytest.log` in the Guardian-repair assessment directory.

Fresh offline Python wheel SHA is
`757f377fe6b48925b2558266aeb6caf56334f14ec4ae119f64834553d889cb5a`.
All134Python modules are byte-identical to current source, with source inventory
SHA `7f9c5871247fb7547ed4fd66fad866457b7ad2e134c2fbac3ea6da7adeaaa1d6`.
The mandatory-only Python3.14.10 environment imports all exercised package/
NumPy/baseline/bootstrap modules from its installed site-packages, with optional
Torch/torchvision/LightGBM/PyArrow/native absent. Corrected protocol3 Random and
Heuristic workers both select from277game-context candidates with Guardian
events/observations and reap with all reservations released. This is synthetic
component packaging/worker evidence, not a Simulator or Source qualification.
Receipts are `wheel/receipt.json`, `protocol3_installed_smoke.json` and
`verification.json` in the same assessment directory. UV's default cache and
pip-unavailable attempts are retained; the offline temporary-cache/local-wheel
fallback succeeds without adding dependencies or changing the active environment.

The second complete192pilot is now running under `pilot-conditions-2.json`,
with exactly the preceding effective Config SHA,32GameSeeds,17/29/43sampling,
seat swaps and7200/5400/7140s caps. All134source hashes are frozen and rechecked
before launch. A result is pending. §16 design is now independently READY;
dependent implementations remain pending.

## Guardian action identity correction — 2026-11-04

The second full unchanged192pilot FAILS with191natural/1incomplete and zero
rejections,298.246s, no group/process health failures. All134source files remain
unchanged during that run. Result: `pilot-result-2.json` in the Guardian-repair
assessment directory; the whole population remains preserved/ineligible.

The remaining failure is introduced by the first repair's overly narrow
GuardianActed namespace. Its public Replays show guardian_acted with card
definition_id=seafood-soup, linked by the pinned catalog to guardian.id=neptune.
The corrected validator now distinguishes Set/Left Guardian IDs from Acted's
generic catalog-definition IDs, preserving all306indices, raw strings, dimensions
and signatures. All42Guardian-kind action definitions and representative other
catalog IDs are covered;54nestedGuardian references include other definition
kinds and are not that action-kind count. No candidate rule changes, source
promotion or reconstructed missing history are involved.

ADR0024§14.2 design and both independent implementation reviews are READY.
Current affected CPU/full Model3 matrix:551PASS/31deselected. Actual HOST CUDA
matrix:31PASS/65deselected. Independent correctness:260PASS/18CUDA-skipped;
conformance:173PASS (overlapping counts). Ruff/mypy/diff checks PASS.
Receipt `after-provenance-and-checks.json` SHA is
`34a176d685d496c23fd0d0b27f9b697e5126ce0cb8dabde2bb72c065f424582a`
under `runs/assessments/card-instance-guardian-action-repair-20261104`.
Only structured.py changes among the134production files; all204preceding
Guardian-repair/coordinator-reference evidence files retain their exact bytes.
The preceding2554whole-test/wheel results remain preceding-validator evidence;
the final complete code still requires whole-project/package acceptance.

The pre-edit legacy/component2 coordinator semantic reference is also frozen:
two inert updates per path, exact control/update/schedule/label/identity/RNG
outputs, no numerical learner/Simulator/Source qualification. Independent archived
replay imports66saved repository modules without current-code fallback and
matches the reference byte-for-byte. Reference SHA is
`586d3c55a4f426450655e84ef15f6be91b853f4021072deecc71a607d8f4669d`;
receipt `baseline-evidence.json` is under
`runs/assessments/card-instance-ppo-preedit-20261104`.

## Corrected baseline provisional feasibility accepted — 2026-11-04

Third full same-condition192pilot PASS:192natural/96complete pairs, zero failures,
empty episode/group health errors,297.091s,3244accepted decisions. Run:
`runs/card-instance-v2-91d5926/pilot/card-instance-v2-91d5926-pilot-dc641b06-42f0-43e9-bceb-1bebacf3a38c`.
Config SHA remains98ce96a3…182f3e6; all134source bytes remain frozen throughout.
Current input ed0c6115…68002f and rule3f55f044…b3e86d remain unchanged.

Independent read-only audit rereads all384J6s and matches all384own accepted-action
streams to their public Player Replays, all192outcomes/rewards/empty health and
32runtime groups. All4492inventory entries/273062500bytes match hashes/sizes;
192opaque server files are hashed only, never parsed as policy input. The full
192schedule exactly matches existing schedule()/serialize_schedule(...,4),
including17/29/43base sampling streams and seat reversal. All384rereads complete
in88.877s. The initial audit's false hard-coded seed expectation is preserved as
a FAIL receipt; the separately corrected audit passes without changing run data.
Current HOST process scan also finds no target run server or protocol3 worker.

Current wheel SHA:
`246a263256f7deb38bb28f0e4992d3b74cf6853c2c8ab492f5e78feaa144d215`.
All134wheel and installed Python modules match the current source, inventory SHA
`ed7d6f0e56a5f0315c41fb7c2dea6bedd9413e24b667f22a3e131c903e48b78d`.
Mandatory-only installed Random/Heuristic protocol3 selection with actual catalog
GuardianActed ID and reservation/reap checks passes. The pilot ran matching source
bytes; its manifest itself does not claim a wheel SHA or per-worker PID receipts.
Those limits and OS-namespace limits are explicit in the audit. Separate installed
checks establish packaging behavior, not that the actual pilot imported a wheel.

Receipts `pilot-result-3.json`, `pilot3-acceptance-audit-corrected.json`,
`wheel/receipt.json`, `verification.json` and `old-payload-integrity.json` are in
`runs/assessments/card-instance-guardian-action-repair-20261104`.
All37historical fixture payloads exactly match Git and24protected fit files match
their original receipts. Only the test-created, unlocked empty fixture lock was
removed after reaping; no historical payload or failed population was rewritten.

Root accepts ADR0024§13component models/artifacts and§14–14.2baseline provisional
feasibility. Source eligibility remains not-authorized-by-collection-metadata.
No learning strength or qualified Source is inferred. §15neural game inference,
§16Source/numeric/learner/PPO, native-v2, Tree/fixed Dataset3, performance and final
combined whole-project/package acceptance remain required. Formal multiple-seed
training/HPO/final-test remain excluded as requested.

## Neural game inference accepted — 2026-11-04

ADR0024§15 implementation passes independent correctness and conformance.
CPU child visibility is now checked against canonical execution before any
factory/Torch import; malformed None/nonempty CPU visibility refuses. Focused
CPU540PASS/44deselected and actual HOST CUDA43PASS/114deselected cover shared
game policies/artifacts, three actual NN workers, old component numerics and
resource/ownership checks. The visibility recheck adds14CPU/3actual CUDA worker
PASS. Full mypy135source files, affected Ruff and diff checks PASS. Independent
correctness has105CPU plus11targeted PASS, and conformance82PASS/12deselected
(overlapping subsets); neither claims its skipped GPU cases executed.

Fresh seed17 initial game-purpose weights are created for MLP/GRU/Transformer;
no old weights are transferred. Real CPU/CUDA inference for all three models
completes two natural games per condition,12total, zero failures,552decisions.
These use default attention and the same per-model artifact across CPU/CUDA.
They are inference-only initial policies, not PPO updates/resume, formal parents,
qualified Sources or model-strength results. All135source hashes stay frozen
through artifact creation and six-condition execution.

Independent audit rereads all24J6s/own public Replays:282NNdecisions have finite
logprob/value,270baseline decisions retain None, all12outcomes/rewards/health,
actual devices, sealed labels/window/caps and retained-original manifest/shard
hash/stat identities agree. The full corrected audit completes in3.562s. Its
initial erroneous expectation that neural_inputs included baseline records is
retained separately; the corrected audit uses NN-only retained inputs plus the
complete policy_bindings role map without changing production/run data.

Wheel SHA `b6d6af071d75e8a5ee54e6370fbf8eac8ffcb83b7cf700e74a855d640e55cff1`
matches all135current and installed Python modules; source inventory SHA is
`ee5ae6bbf6ccdca7559f8f405bf90894b4494ebf96c08c0da567fb45fce16432`.
Mandatory-only installed Random/Heuristic protocol3 checks pass with optional
Torch/LightGBM/PyArrow/native absent. NN itself is not loaded in that environment.
Actual runs use matching source bytes; they do not claim an installed wheel SHA
or no-process guarantee outside the observed OS namespace.

Assessment directory: `runs/assessments/card-instance-neural-inference-20261104`.
Primary receipts are `implementation-and-checks.json` (SHA
`4d30ba7435950edcb394013b947ec6afb9e4fd372c5a963b7f09685238afe8d2`),
`visibility-fix-checks.json`, `initial-artifacts.json`,
`real-six-inference-smokes.json`, `neural-inference-acceptance-audit-corrected.json`
and `wheel/receipt.json`. The current legacy/component2 coordinator trace still
matches the immutable pre-edit reference exactly, SHA586d3c55…f4669d, no masked
fields or Torch execution. Historical fixture/archive bytes remain preserved.

Root accepts §15 inference-only scope. §16Source/numeric4/wire4/registry3/shared
PPO and its bounded design addendum are independently READY but implementation/
real training/resume remain pending. §17native design is READY, gated on stable
§16Python interfaces. Tree/fixed Dataset3, performance, original-data/native
proofs and final combined whole-project/package acceptance remain pending.

## Source/PPO implementation and first real updates — 2026-11-04

The later §16 implementation passes independent correctness and architecture
conformance reviews. The same `_run_ppo` coordinator now selects explicit corrected
Source/rollout4/wire4/lineage3 operations. Original C5/J6 records, complete cohorts,
live parent/learner/lease ownership, numeric correspondence, transaction identity,
pre-commit cleanup and post-commit memory-only release remain required. Registry
retention supports the current policy and five pool policies without relaxing the
two-policy standalone game owner. Full Ruff and mypy137 PASS; focused actual CUDA
numeric25 PASS. The immutable legacy/component coordinator trace still matches
SHA586d3c55…f4669d exactly.

Actual acceptance uses seed17, default attention,16games/update and unchanged
7200/5400/7140second budgets. Frozen conditions and137source hashes are recorded in
`runs/assessments/card-instance-ppo-acceptance-20261104/conditions-2.json`.
The first MLP/CPU-inference parent fails before its first game because an internal
registry producer retained tuples instead of closed JSON lists. It is preserved
with zero updates/decisions and quarantine; it is not resumed or made eligible.
The producer-only repair keeps canonical hashes/labels and strict consumers exact.

MLP/CPU inference with CUDA learner then completes both continuous2 and
pause1→resume2 parents: each32natural games,419learner decisions,16Adam steps.
Complete-parent validation rereads originals, numeric correspondence and lineage.
The independent trajectories do NOT have identical final weights: their first
batch already contains different candidate ordering/counts at the same visible
input. The original comparison remains FAIL. Adaptive wall-clock search can vary;
these observations do not prove timing was the cause and do not authorize a new
legality rule or a smaller search domain.

ADR0024§16.1 separately predeclares each condition's completed pause/resume parent
as the reference for a same-input learner restoration diagnostic. It restores
that parent's G0 and continuously consumes its verified numeric0+1 using the
unchanged learner, then compares against its actual resumed G2. It creates no
PPO parent/generation/Source authority. For MLP/CPU inference this control PASS:
G0's60 saved arrays and G2's234 saved arrays are exactly equal, including weights,
named Adam, counters, Torch CPU/owned CUDA RNG and PCG64 shuffle. The derived
nonpersistent position buffer and final original-file rechecks also pass. The
initial diagnostic's incorrect buffer-inventory expectation is preserved as FAIL;
only the assessment helper was corrected. Independent read-only result audit
revalidates the complete parent,29parent snapshots and1041original C5/J6 snapshots.
Opaque server files are hashed only.

Receipts include `continuous-resume-state-audit.json`,
`continuous-resume-public-divergence.json`, `mlp-cpu-fixed-input-1.json` (FAIL),
`mlp-cpu-fixed-input-2.json` (PASS), `fixed-input-result-audit-final.json` and the
predeclared `fixed-input-reference-rule.json`. The six-condition helper is
independently reviewed; its closed reference map cannot select a parent after
seeing diagnostic results. The original independent-trajectory failure is never
relabelled as equality. These results establish scoped same-input normal-boundary
restoration, not trajectory determinism, formal parent qualification or strength.

MLP/CUDA inference's continuous2 parent also completes32natural games,
419learner decisions and16steps; its separate parent reaches the normal paused
boundary after one update. Its resumed update and same-input comparison, all four
GRU/Transformer conditions, CPU-learner smoke, direct/DeepSets real checks and final
combined acceptance remain pending at this entry.

Subsequent MLP/CUDA resume also completes32natural games/419decisions/16steps.
Its predeclared same-input control PASS in100.944s, again exact G0's60and G2's234
saved arrays plus derived positions/scalars/RNG. Independent read-only audit
matches all29parent/1041collection file snapshots and all saved-bundle hashes;
`validate_game_parent(require_complete=True)` passes. Receipt
`mlp-cuda-fixed-input-result-audit-final2.json` preserves earlier audit-only hash
failures caused by a scalar-array shape conversion. No original or production
data was changed and no additional learner was run by that audit.
GRU/CPU-inference continuous2 also completes32natural games/419decisions/16steps
in359.333s; its pause/resume acceptance is in progress. Remaining actual conditions
run sequentially under the same frozen137source hashes and budgets; the Root
orchestrator stops on the first nonzero child, unexpected boundary or failed
same-input control. Its progress is in `remaining-matrix-1.log`.

GRU/CPU resume completes2updates/32natural games/419learner decisions/16steps.
Its same-input control PASS in231.909s: G0's64and G2's250saved arrays are exactly
equal, all17scalar checks pass, with the additional derived position buffer
verified. Receipt `gru-cpu-fixed-input-1.json` retains original-file snapshots and
complete Source/numeric correspondence. This brings actual execution plus scoped
same-input restoration to3of6conditions; GRU/CUDA and both Transformer inference
conditions remain in progress. No independent-game-trajectory equality or formal
parent qualification is inferred.

## Native corrected DTO diagnostic — implementation in progress

The first §17 increment implements only private closed corrected recipe/DTO and
shared token/journal primitives. Both independent reviews are READY for that
bounded diagnostic scope;11targeted Rust tests PASS. The Python golden frame hash
is4a6ccf18…040c7e, including the full corrected contract/input recipe/policy version.
An earlier partial binding hash is corrected with its failing reproduction kept.
Existing native capability markers and installed ELF remain unchanged. Native
Source/session/staging, exact Python/native token parity and capability activation
have NOT yet passed; this increment cannot process a corrected training Source.

The next bounded pure-validator/token increment is independently accepted:
17corrected Rust tests plus2old token/1old control/1old16-role Source goldens pass.
Independent Python/NumPy regeneration matches all10literal frame/table arrays,
dtypes/shapes/codec bytes; all9model×hand metadata/shape/label/resource cases and
four full synthetic J6 transactions agree with the Python reader. Source/current
PID/actual NPZ values/session/staging/ABI/capability authority is outside this
pure diagnostic scope. Common-only fixture extraction leaves137Python sources
unchanged and imports no Torch.

The reviewer-requested reward coverage adds viewer1/2×win/loss/draw literals.
All six valid J6s pass both readers;18wrong reward and4fully rebound opposite-
viewer cases reach the intended reward refusal. Target Rust count is now19PASS.
Both independent test-only rechecks are READY, with no production-kernel change.
Stable DTO hash ea0bd91e…6008bb and reward fixture01a971cd…77ed4; logs and extraction
helpers are in `/tmp/card-instance-rust-diagnostic-20261104`. Root accepts this
restricted scope. New native capability/installed ELF remain unchanged; full
§17.1 Source/session/staging/provider implementation and acceptance are pending.

The subsequent source investigation reads only the completed actual MLP/CPU
reference parent for genuine historical parity. Its initial Python reference
extraction refuses a conservatively supplied2GiB allowance before decoding;
the recorded Source limit is8GiB, so comparison is rerun under that same recorded
limit. The refusal is preserved and is not a failed game or a raised experiment
budget. Opaque server/canonical files remain hash-only. No original is rewritten
or rebased and no actual record is promoted to formal eligibility.

## Remaining production interfaces — design accepted, implementation pending

ADR0024§§18–18.2 now have independent architecture DESIGN READY. In addition to
the declared component3-population Dataset3/fresh-fit/pair4/worker/RAW and actual
512decision performance panel, implementation must supply reachable corrected
formal-parent, actor-only64final-validation, three-parent qualified Dataset3,
fresh fit and GameTree3 routing. A component-only implementation would leave the
old engineering-only interfaces unconnected.

The new formal recipe is a literal specification, not executed evidence:
`docs/contracts/card-instance-formal-ppo-v1.json`, SHA871fc8d4…ecd67b2, algorithm
content SHA318eab51…c90cc9. It freezes full1000max-update settings and requires a
non-circular pre-G0 declaration plus healthy original-source ownership. Existing
two-update acceptance parents cannot be retrofitted. Separate bounded actor
validation deadlines preserve training's original completed deadlines; normal
admission/storage-reservation completion remains valid without requiring1000
actual updates. Formal runs, Dataset3 qualification, real3-parent fitting,
multi-seed comparisons, HPO and final test remain NOT RUN as requested.

## Python PPO runtime acceptance complete — 2026-11-04

All six seed17/default-attention conditions complete both continuous2 and
pause1→resume2 with CUDA learner,32natural games/419learner decisions/16Adam
steps per parent. All six predeclared same-input restoration controls PASS;
G0/G2 saved-array counts are60/234MLP,64/250GRU and92/362Transformer, all exact.
The CPU-learner MLP smoke and direct/DeepSets MLP cases each complete16natural
games/1update/234decisions/8steps under the original budgets. Actual execution
totals15parents,432natural games,27updates,5730learner decisions and216Adam steps.
No diagnostic replay creates another parent or generation.

The independent verifier completes all12main-parent Source validations at the
recorded8GiB allowance. The following assessment snapshot glue raises NameError
after those successful checks; its pass is preserved separately as
`six-main-pair-source-validation-interim.json`/`.log`. A subsequent unexecuted
replacement script also has harness errors. Those are assessment errors, not
game/data failures. Root escalates the repeated repair, then supplies small frozen
programs and an explicit case manifest for independent review and execution.
The corrected frozen programs pass syntax/Ruff and cheap six-case preflight;
independent review is READY. The verifier executes them without editing them.

All10new receipts PASS: `root-audit-source-evidence-v1.json`, six
`root-audit-report-*-v1.json`, and three `root-audit-extra-*-v1.json`.
The12Source decodes are not repeated: current sealed generation, manifest,
numeric and inventory hashes corroborate the preserved full pass. The historical
pass did not contain a full stat inventory, so that narrower freshness scope is
explicit. Each same-input audit checks29parent/1041collection file snapshots,
exact model-specific parameter names/shapes and every persisted G0/G2 array's
dtype/shape/full-byte SHA against both helper actual/expected hashes. Scalar,
PCG and derived-position equality remain reviewed-helper results; positions are
not independently reconstructed by the NumPy-only final auditor. Opaque
server/canonical files are hashed only. No GPU learner or game is rerun.

Final Root receipt `python-ppo-acceptance-final-1.json`, SHA
`d8b99b5a137517afab74f3d3ca4ea4fd407cc4fdc64dd0ab4d64a088b3540202`,
binds all10receipts. All137Python source hashes stay exact from conditions2
through actual execution and final audits; the freeze is now lifted. Root accepts
ADR0024§16/16.1 Python runtime scope. The original independent-trajectory FAIL,
initial pre-game failed parent, diagnostic buffer-inventory FAIL and assessment
errors remain preserved. These short parents stay permanently nonformal; no
Source promotion or model-strength claim follows.

Native internal historical parity now matches both actual MLP cohorts and all
4744batch0 raw arrays, but independent review finds a blocking failed-phase
lifecycle issue: corrected finalization methods can retain operational states
after reportable errors. The focused fix must poison admitted attempts before
fallible work while preserving cleanup ownership and old component semantics.
Native Source/provider/ABI/capabilities/actual live acceptance, Tree/fitting and
512ordinary-decision performance, final whole-project/native/old-data/package/
installation checks and local final commits remain required.

## Native PPO runtime acceptance complete — 2026-11-04

The failed-phase lifecycle issue and three provider preallocation/binary/query
findings are fixed, with both independent rechecks READY. The shared Rust bridge
and corrected provider retain the original Source/numeric4/wire4/common PPO
coordinator. A separately built private release exposes the reviewed three
corrected-game capability markers. Its ELF SHA is
06054171c358deff4828e7de4896bd66f860e7e11af225ed33c039b5bd6c9019;
the installed main-environment ELF remains unchanged.

Actual native MLP/default-attention CPU and CUDA inference, both with CUDA
learner, each complete continuous2 and pause1→resume2 under the original
7200/5400/7140/60 budgets and16games/update. Four parents complete32natural games,
419learner decisions and16Adam steps each:128games/8updates/1676decisions/64steps
overall. Code executes from a frozen141module snapshot so independent later
Tree changes do not alter child/coordinator imports. Git revision/dirty metadata
is null for that copied runtime; its full inventory is pinned by snapshot SHA
3f794e089a1e5018da9e88c11facf7f112a356334752949b5382bfa44dfbecec.

Both predeclared native same-input controls PASS. They validate each original
Rust parent unchanged, restore G0 and replay its two actual verified batches
with an independent Python reference reader. They capture state before reading
expected G2; no new parent, generation or eligible Source is created. All60G0
and234G2 saved weights/Adam/RNG arrays are exact. The prior independent-game
trajectory failure remains historical and no trajectory determinism is claimed.

Root executes six bounded final checks: four complete original native parent
Source validations at the recorded8GiB allowance and two saved-report audits
reusing the reviewed NumPy reference-array auditor. All PASS. Snapshot/ELF,
predeclared parent/config membership, original stat/hash inventories and saved
G0/G2 arrays are checked. Scalar/PCG/derived-position equality remains the
reviewed helper's result; the final auditor rehashes persisted arrays without
rerunning numerical learning. Opaque server/canonical files are hashed only.
These final executions are Root checks, not an additional independent code review.

Root receipt `runs/assessments/card-instance-native-ppo-acceptance-20261104/
native-ppo-acceptance-final-1.json`, SHA
ee0cc693dcaf905a0660c35a6dfae096da4f52e11b9465f90a7aa9046f39e075,
binds all six receipts. Root accepts§17's native PPO runtime scope; these short
parents remain permanently nonformal. Native fixture-token diagnostic, Tree3
workers/RAW and formal production interfaces,512ordinary-decision performance,
whole-project/package/old-data checks, installation and commits remain pending.
