# Corrected card-instance Tree3 migration evidence

This ledger covers ADR0024§18 components. It does not qualify formal parents,
run a real three-parent fit, establish policy strength or accept the whole
migration. Simulator91d5926/corrected profile, token/layout ed0c6115…68002f and
rule3f55f044…b3e86d remain the input contract.

## Pure configuration and features — accepted

Closed TreeEncoderConfigV3/TreeArchitectureV3/ConfiguredTreeModelV3 and explicit
Tree3 feature helpers share old packing/resource kernels. Old Tree2 types,
defaults, field names, identities and FP32 outputs stay exact. Derived widths:
card64, context1629, candidate-fixed29, feature1658+64*positions (2810at18).
Default feature signature is0a3c136fad2be0a7bca980805bfeec3a689735ecad57dce1918545b350789574;
architecture identity58406cf951dfb6cd90001d6f30e203db4ff438da431e0a1887b9ca3aa79d03bc.
Ordered raw-to-percent-escaped field mapping, category limits, transforms,
window/token/rule/layout signatures bind the new recipe. Opaque refs are never
numerical features.

Independent correctness finds a blocking cache-binding defect in the first
revision: replacing a window128 model with window1 keeps history128; replacing
actor1 tokens with actor2 tokens keeps actor1's private cache. Root accepts the
finding. The corrected Tree3 prepared value is factory-only/sealed, with exact
token/model/cache binding, immutable byte-backed arrays and readonly accessors.
Binding plus dtype/shape/contiguity/base-owner checks precede every iterator
advance. Frozen candidate tuples are validated once when retained. Admission
explicitly counts slots/binding/cache owners and cumulative inputs. Tree2's
dataclass behavior remains unchanged.

Both independent fix rechecks are READY. Scoped new/old encoder64PASS; expanded
related regression227PASS; Ruff/mypy2/diff PASS. Four expected NumPy warnings
come from intentional dtype/shape mutation regressions. Config imports/identity
need no optional Torch/LightGBM/PyArrow runtime. No game, fit or GPU ran.

Accepted hashes:

- structured_tree_config.py:aec0192ccdc970bdbaf060325c89cc9c7acf8a617600792c13eea234929c86e0
- structured_tree_encoder.py:0a69de6889ec6fc1e58b643e5e7a370fea86c219abedcf127fa9dd44d171254d
- test_card_instance_tree_encoder.py:a1235297ce0d891ac2289281158f8635885a1c4abc876ecb13ca130c3d8a8c7d

## Declared component originals and fixed Dataset3 — accepted Source scope

Three new modules implement a pure closed declaration/envelope, concrete
TreeFixtureSource and source-only VerifiedFixtureTreeDataset3. Three MLP
populations construction-seeded17/29/43 each declare train policies0/1 with16
episodes each and final validation policy2 with64episodes:96train/192validation,
288episodes/nine cohorts. Construction seeds are not Simulator GameSeeds.
Nine actual fresh CPU inference-only component artifacts are sealed; no PPO
updates/Adam/runtime acknowledgements/C5/J6 execution evidence is fabricated.
The fixture has6train/3validation zero-endpoint episodes and96/192selected rows;
all episodes remain in population accounting.

Labels are explicitly declared participant1|2 winner or draw, converted to that
selected viewer's±1/0. Source owns exact role/policy/model/action/own-frame/event
prefix/ordered codec2/selected-index bindings and complete original inventories.
Its one Source-local inference helper reuses existing pure sealed metadata and
bundle kernels, without current optional-runtime queries or a fabricated resume.
The helper's private coupling has a revisit condition if another consumer needs
the same API.

Original metadata/policy/index snapshots stay through streamed row projection and
publication checks. Budgets include cumulative owners/sink/arrays and64MiBdecode
limits. Outputs are outside all source/artifact roots. Reportable final guards/
close precede memory-only release; suspended iterators refuse after close.
Repeated Dataset inspection reconstructs originals and has stable content;
cached features or origin flags grant no fitting/online/qualified authority.

Both independent reviews are SCOPED READY. Independent fixture28CPU PASS;
expanded255CPU PASS, Ruff5/mypy3/diff PASS. Ordinary Source/Dataset import and
inspection require no Torch/LightGBM/PyArrow/native/HTTP. Test construction used
Torch CPU initializers only, no GPU/games/fits. The reported pytest temporary
fixture was later removed by pytest retention; it is not durable acceptance
evidence. Root will construct dedicated durable originals once for fit acceptance,
without copying/rebasing those temporary artifacts.

Accepted production hashes:

- card_instance_fixture.py:5d2e45958e2ac9262c62c2a142026694994ff0c9f551d9265edf4dfca030d9a9
- card_instance_tree_fixture.py:79c62a63c60ef8c76869d6344c0b550ebda8bafc0a88e266a197f934936b6bed
- card_instance_tree_data.py:3e6c464fc01a2b1a290b5d2a6305f7a254dfd77eeee17d8370b2d4f9ac36fc23

## Artifact4 and fresh fixture fit — accepted

Artifact/request/completion/pair4 and fresh component fitting pass both independent
reviews after the corrections below. Final changed-area checks:52new Tree3 PASS,
63old trained Tree2 PASS/1native RAW deselection,19fit-control PASS/8explicit old
Dataset2/native-source fixture skips, Ruff/mypy/diff PASS. Root accepts this
bounded§18.3 increment, not workers or qualified fitting.

Correctness review initially finds that final release errors can leave copied/
fitted completion bindable, or lose reachable cleanup ownership. Both publishers
now include all fallible descriptor retirement inside rollback/revocation, with
no postcommit directory close. The first correction still mistakes same-inode
descriptor reuse for ownership and fails independent counterexamples. Fix2
removes active descriptors BEFORE one close attempt; any close error becomes
terminal reachable quarantine, and that uncertain number is never inspected,
used or retried. Revocation uses a fresh O_NOFOLLOW/O_RDWR regular descriptor
matching the original inode. It invalidates completion while preserving both
uncertainty and the primary error. Independent conformance18targeted PASS and
correctness11targeted PASS cover same-inode RO/RW reuse and full component-fit
failure paths. Successful old record bytes remain exact.

The review also finds that a Tree3 wrapper can attach a different history window
to an actual Booster without changing feature width. The corrected actual
Booster retains its full architecture/feature/input binding, checked on wrapping,
saving and scoring, including trained-pair agreement. Both independent rechecks
are READY. Execution-only inference/resource overrides remain permitted.

Root constructs fresh durable originals in
`runs/card-instance-v2-91d5926/tree-component-acceptance/fixture-default-1`.
The accepted constructor freezes the declaration before any weights/episodes.
Construction and Dataset3 inspection pass with source/helper hashes stable;
construction receipt SHA14cb13d93483d98f0ce1efa2eda8fb4a2fb6cda399dbd48671bf6618a78b60d9.
The first durable default fresh fit then PASSes using all96train/192validation
rows, configured500rounds/50early-stop/.05/min20/thread1, consumed80 and retained30.
Healthy child reap, Source/index closure, paired artifact4 load and stable producer
digest are checked. Receipt `default-fit-acceptance-1.json` SHA
0037fdf5c96cf5556662506fb26b8c87a2271a3f0c64275abd3c0c22af143b88;
inference SHAcffe4fb62c5f3f58300a33a9866ad50ce0614787ee216d63848e7273bba843f1,
model SHA19bbc4bdc1e00ddca169b812e4f01fa68f3d0656059abfea7a6b6bb459cd685c.
Declared component origin and qualified false stay exact; no real-data learning
gate is passed. Component worker/RAW remain pending.

## Native component token diagnostic — accepted structural scope

Both independent§18.4 reviews are READY. Each passes110new FFI and84old J5/
metadata cases on the private release candidate. Correctness also regenerates
all68literal cases/1254arrays through common Python kernels, exact dtype/shape/
bytes/labels/signatures/selected identity, and passes5target Rust tests. The
implementer additionally passes54affected Rust tests/2explicit opt-in ignored.
Seven frozen files, candidate ELF and protected actual-PPO ELF0605/runtime141
inventory match. This increment changes no production Python source.

Handoff receipt `/tmp/card-instance-native-fixture-diagnostic-20261104-v3/handoff-receipt.json`
SHA35ea0724c8be1d10ac795ad69cf888111a1db262cc93b4f2f2d779836f3b9e94;
native source401ce8931eaa37e3a2928155c45acd123e55c089e01061067353a066e7331e16,
ELFe390ad8e412bbd86bf7b32250bc0eb7b3f5f0a4e153e3948ba87706b623e3a76.
Constructor failure retains cleanup owner/quarantine; export retires spool/core
before readonly arrays escape, with independent base-owner quota lifetime.
Fixture marker remains absent. SAME concrete FixtureSource adapter, full
nine-cohort pipeline and capability publication remain pending. Forced Python
allocator-failure injection is NOT RUN. This grants no Source or fitting
permission; no game/GPU/timing or formal experiment runs.

## Performance population fixed; measurement pending

Six accepted Python reference parents supply all2514 learner endpoints from12
complete train cohorts. PCG64seed17 selects512without replacement, sorted by
original pool index exactly as predeclared. Manifest
`runs/assessments/card-instance-v2-latency-20261104/panel-endpoints-1.json`, SHA
82f5a594620e2cd24a9a53c6016a6e1da130adcaf775d221fa8e3ea1fb95053e,
binds original frame/input/table/full-own-prefix/event-boundary hashes, selected
endpoint IDs, episode sampling control and three NN G2 artifact hashes.
The extractor retains each original Source during extraction, checks committed
proof and all16natural learner-seat episodes in schedule order, and uses the
frozen141module reader without Torch. Root separately recomputes selection.
The first healthy default Tree pair and three NN G2 artifacts are frozen before
any measurement: `frozen-artifacts-1.json` SHA
2ea047f36daa250e44d6053837e1430e198bec7544a6c673849c3031fbc24cf1.
No timing has happened; no performance pass is claimed.

Component worker/RAW/FixtureSource-native adapter,512actual-input timing,
reachable formal-parent/64final-validation/qualified Dataset3/GameTree3
interfaces, whole-project/package/old-data checks and final installation/commits
remain required. Actual formal three-parent generation/fitting, comparisons,
HPO and final-test remain NOT RUN.
