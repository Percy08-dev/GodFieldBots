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
Stages2–5 (candidate/frame/token, model/worker/journal/native, Source/learning/
artifacts/evaluation and full runtime acceptance): implementation NOT RUN here.
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

Independent bot_advisor review confirms three unresolved public predicates at
the pinned revision: Flash counting for defense miracle+CostCut, Discard/
Sacrifice kind–state–subclass eligibility, and per-utility accepted targets.
Root accepts the finding and records the exact separate Simulator clarification
task in [issue0007](../issues/0007-card-instance-v2-public-candidate-contract.md).
No Simulator files changed, rule inference/probing performed, seeds replaced or
candidate scope reduced. Complete generator adoption and dependent real-game
acceptance are BLOCKED pending an explicit amended-contract adoption. Independent
frame/token implementation continues under its own design gate.
