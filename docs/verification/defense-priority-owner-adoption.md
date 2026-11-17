# Defense priority owner adoption verification

Status: implemented and accepted by root under ADR0048 after independent
correctness and architecture-conformance reviews FINAL READY. Acceptance uses
the whole-suite plus isolated test-harness correction evidence below; it does
not claim a newly green whole-suite invocation. No new learning/HPO/Simulator
gameplay is part of this verification.

## Authority and compatibility

Owner confirmation:
`docs/contracts/owner-20261117-defense-priority.md`, SHA256
`bdd1792cde50eae2338b5f068436e40ca87cfbdf0c276ad096651bc0ff27375c`.
Simulator remains91d5926 and public-ja-v3. Generator/layout identities remain
unchanged; a new supplemental rule signature distinguishes candidate semantics.
Only current rules-v2 is supported after the owner's explicit compatibility
opt-out. Old descriptors/signatures reject without historical readers, executable
legacy predicates or data migration. Exact current binding and mixed/unknown
identity rejection are verified. The corrected rule signature is
`081fa1e0b6c186c563c949e65fbce19b77f07db505516330d3ac377b1b02e2d6`.

## Baseline

PASS before production implementation,148tests in1.31s:

```sh
.venv/bin/python -m pytest -q tests/test_card_instance_rules.py tests/test_card_instance_actions.py tests/test_refresh_heuristic.py
```

This is baseline evidence at f4cdb4a application bytes. It does not test the
owner's amended behavior, which must receive RED/GREEN regressions separately.

## Acceptance checks

- Recorded Mudflow/Angel Armor/Rainbow rejection no longer appears as a completed
  active-v2 candidate; reverse Rainbow order is also rejected.
- Positive-defense stop/reflect/deflect numeric and special roles, unknown
  contexts, atomic singletons, Flash and bound CostCut remain distinct.
- Public-causality tests reject wrong actors/targets, future or foreign prefixes,
  unsupported suffixes, Guardian/reflection/targeted/mixed/obscured histories,
  and a weapon ID absent from the permitted own catalog. No hidden lookup.
- Old data remains unchanged and old descriptors/signatures reject explicitly.
- Python/native exact current binding, nested old/mixed/unknown rejection,
  old-artifact rejection and native capability preflight are verified.
- Relevant Python/static/native/package/project checks and independent final
  correctness/conformance review; exact PASS/FAIL/NOT RUN and limitations.
- Source-matched native activation or explicit remaining manual requirement.
- Original issue0010 failed DB hashes/stat identities unchanged; no parent
  eligibility, budget, deadline or record rewritten.

## Retired historical acceptance inputs

The owner removed rules-v1 compatibility from this change. Native fixture
inspection identifies real captured `actual_mlp_reference.json` and captured
unadmitted `final64_originals.json` as old-rule inputs. Preserve their original
bytes and execute old-rule rejection checks. Three lifecycle cases can exercise
their existing assertions with newly constructed synthetic current bindings;
five captured final64 acceptance cases retain their original bodies with an
explicit old-input ignore reason. Record final names/counts after implementation.
Those five cases do not verify current-v2 final64 acceptance; revisit when fresh
current-rule reference acquisition is separately authorized. No training or new
capture is performed to manufacture that evidence.

Explicitly synthetic/schema-only journal, rewards, model, HPO and Tree fixtures
may be regenerated as current-v2 inputs using current builders and validation,
including dependent digests and any affected candidates. This is test input
construction, not migration, resealing or eligibility repair of captured data.

## Intermediate environment failure

The first wider sandbox Python selection reports250PASS/25SKIP, three old-hash
assertions and20CPU-preload setup/teardown errors. The template stderr records
`PermissionError: [Errno 1] Operation not permitted` at
`tests/cpu_preload_server.py:45`, reopening its inherited socket. The timeout is
secondary to that sandbox restriction. Do not weaken deadlines or change template
behavior; rerun affected process checks with host permissions after code stability.
This failed selection is not a passing regression result.

## Native activation and package check

The current native wheel builds offline with Maturin1.15.0, release profile,
allocator-mimalloc and empty RUSTFLAGS. The installed extension has source SHA
`5f3b3d441da3ab5d503dc64b2ba93f13d35fa4126ebe75e31d246b8e368abd0d`,
ELF SHA `78830a048eb709829d817b7b40a4f068a8c6954fc776b9e907b8006e4a594a5c`,
and capability `card_instance_defense_priority=owner-defense-priority-v2`.
Actual installed source-matched packed/resident HPO preflight passes. Previous
package files/build metadata are backed up in
`/tmp/gfb-defense-native-rollback-i3jv8sfc`.

```sh
env -u RUSTFLAGS CARGO_NET_OFFLINE=true uv run --no-project --offline --with maturin==1.15.0 maturin build --release --manifest-path native/Cargo.toml --features pyo3/extension-module,allocator-mimalloc --interpreter .venv/bin/python --out /tmp/gfb-defense-native-wheel
uv pip install --offline --no-deps --reinstall --python .venv/bin/python /tmp/gfb-defense-native-wheel/godfieldbots_native-0.1.0-cp314-cp314-manylinux_2_34_x86_64.whl
```

NumPy2.5.4, Torch2.14.1+cu132 and Optuna5.0.0 remain unchanged. A separate root
Python wheel contains232exact source-matching Python files. Installed into an
isolated mandatory-dependency-only Python3.14 environment, CLI/HPO help, actual
HPO config validation and Random baseline construction/reset pass without
Torch, Optuna, native, LightGBM or Arrow imports. Record:
`runs/assessments/defense-priority-20261117/package-baseline.json`.

## Review repair and whole suite attempt

Initial architecture finding B1 identified an independent fixture preflight
without the new capability gate. The shared gate now runs before binary identity,
Source access and allocation in both fixture modes. Four missing/wrong-marker
tests are observed RED then GREEN; architecture recheck6PASS and correctness
delta9PASS leave no production blocker.

Nine external captured Tree-pair acceptance tests require a current-rule pair
that is not available. They skip only the exact recognized old-v1 signature,
before copying/loading/process work. Unknown/current/malformed records do not
become skips; the original test bodies remain. An executed rejection test proves
the actual old pair is refused before model/input work. HOST adapter/native/
Guardian selection passes729/SKIP32. No old data is relabeled or fitted again.

Whole Python attempt01 runs HOST with real Simulator/GPU/heavy safety opt-ins
disabled. It stops at475PASS/1FAIL in11.14s:
`test_normal_corrected_collector_uses_one_fresh_plan_actor_ledger_and_cost_order[None]`
expects a formal recipe with an old binding to pass current preflight. The
production rejection is intended under the owner's current-only decision.
All54frozen task files match before/after. Logs and JUnit are in
`runs/assessments/defense-priority-20261117/whole-python-01.*`.

The formal test disposition retains literal/hash/Adam/AST/stop assertions and
changes direct old-positive preflight assertions to exact unsupported-binding
checks. Existing explicitly unadmitted collector/control/lifetime tests use a
function-scoped test-only checked plan-shape seam. It does not admit a parent,
generate Source or claim formal qualification; there is no generic catch-to-pass
or autouse patch. Frozen recipes and captured bytes stay unchanged. A real,
unpatched preflight rejection test remains. HOST eight affected modules pass282
checks after this disposition. Independent test-delta reviews are READY; later
whole-suite results are recorded below.

Whole attempt02 stops at1396PASS/50SKIP/1FAIL in183.95s. The remaining positive
scoring-process test reads the same captured old-rule Tree pair. All63frozen
inputs match before/after. This is another unavailable historical acceptance
input, not a reason to accept old identities in production.

A bounded sweep covers every use of that pair and the old captured final64/MLP
roots. Shared test availability logic examines only the declared outer old rule
signature, before copying, model/input access, DTO borrowing or process work.
It never searches nested records for a reason to skip, catches no production
exception and does not turn current, unknown or malformed identities into skips.
Direct tests cover current root plus an old nested value. Original test bodies
and captured bytes remain. HOST eight-module sweep passes104/SKIP233 in16.06s;
227skips name unavailable old captures:38Tree-pair cases and189final64 cases,
including previously opt-in cases. Counts are reason totals, not an assertion
that227previously passing default cases were removed. Six other skips retain
their pre-existing conditions. Positive current captured-pair/process/final64
acceptance remains unverified; current synthetic Source/guard tests still run.
Log: `runs/assessments/defense-priority-20261117/captured-input-disposition-host.log`.

Both original failed pilot jobs have unchanged exact-context termination
receipts. Read-only `verify_gpu_termination` passes for both without admission,
recovery, resource release or process action. Their successful cleanup does not
change the failed experiment's eligibility.

## Final whole suite and scoped correction

The full HOST03 invocation completes all7595cases:7007PASS,586SKIP,2FAIL,
33warnings,1872.84s. Both failures are the pre-existing parameterizations of
`tests/test_gru_ppo.py::test_worker_releases_each_rollout_before_save_ack_next_load_and_on_failure`.
The test calls the worker in-process without isolating `sys.argv`, so pytest's
own options reach the strict Model4 execution-cap parser before its intended
rollout-lifetime checks. The test and both production worker/parser modules were
unchanged from HEAD before this final correction.

```sh
env GFB_RUN_INTEGRATION=0 GF_RUN_CUDA=0 CUDA_VISIBLE_DEVICES='' GFB_RUN_CPU_PRELOAD_SAFETY=0 .venv/bin/python -m pytest -q --disable-warnings --tb=short -o faulthandler_timeout=60 --junitxml=runs/assessments/defense-priority-20261117/whole-python-03.xml > runs/assessments/defense-priority-20261117/whole-python-03.log 2>&1
```

The verifier's last process poll result was not delivered before its agent turn
was interrupted; the numeric tool exit code is unavailable. Root independently
verifies the completed JUnit/log results, all67frozen input hashes, and absence
of the exact HOST pytest process. A60second faulthandler diagnostic in the Tree
source fixture was followed by continued progress and normal case reporting;
it was not treated as a passing timeout or used to change deadlines.

One test-only line replaces argv with `['gf-ppo-learner']` through monkeypatch,
allowing fixture teardown to restore the original list. All weak-reference,
save, next-load, failure and reply-count assertions remain. RED reproduces both
failures; the whole GRU module plus launch/parser tests then PASS40 in12.18s.
Both independent reviewers approve this isolated correction and whole-plus-delta
acceptance. Production Python/native bytes do not change after the whole run.

Root's fresh final targeted verification also passes68 in11.77s:

```sh
env GFB_RUN_INTEGRATION=0 GF_RUN_CUDA=0 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_defense_priority.py tests/test_gru_ppo.py tests/test_ppo_update_launch.py
.venv/bin/ruff check src tests
.venv/bin/mypy src
git diff --check
```

RuffPASS, mypyPASS232source files, whitespacePASS. Whole03's two failures remain
honestly recorded; no additional fully green whole-suite run is claimed. Logs
are in `runs/assessments/defense-priority-20261117/whole-python-03.*` and
`final-targeted.log`; the isolated RED/GREEN logs are retained with the evidence.

## Rust checks and final scope

PASS174 /IGNORE13 in the native library suite. Five newly ignored captured
final64 acceptance cases require current-rule acquisition; existing case bodies
and original bytes remain. The rest retain their prior explicit opt-in reasons.

```sh
env RUSTFLAGS='-L /home/percy/.local/share/uv/python/cpython-3.14.10-linux-x86_64-gnu/lib -C link-arg=-lpython3.14' LD_LIBRARY_PATH=/home/percy/.local/share/uv/python/cpython-3.14.10-linux-x86_64-gnu/lib cargo test --offline --locked --manifest-path native/Cargo.toml --lib
cargo fmt --manifest-path native/Cargo.toml -- --check
```

The libpython flags are test-only; the installed release build has empty
RUSTFLAGS. Full cargo-format check FAILS on the pre-existing wrapping at
`card_instance_dto.rs:1496`, independently reproduced from HEAD in
`rust-fmt-baseline.log`. This unrelated formatting debt is not bundled into the
defense correction. Current changed blocks are formatted; Rust test/build checks
pass. Native/core/fixture focused selections overlap and are not summed as a
unique test population.

The three HPO presets validate and use fresh `runs/ppo-budget-probe-v2` namespaces.
Root verifies original failed SQLite SHA256/size/inode/mtime and both captured
native fixture files against HEAD; `original-integrity.json` retains the check.
No past data, recipe or checkpoint is migrated, resealed or made eligible.

Root accepts the predicate, identity and preflight correction with both final
reviews READY, the native/package checks and the exact whole-plus-delta evidence.
Actual D10,000 completion, learned-policy quality, GPU execution and current
formal/final64 captured positive-path acceptance remain NOT RUN. The original
diagnostic masking is a separate unresolved item in issue0010.

## Execution restrictions

Training, HPO campaigns, actual Simulator gameplay and GPU recovery are NOT RUN.
Successful unit/Source/native checks do not establish convergence, a complete
D10,000 run, broader context coverage or learned-policy strength. Original failed
experiments retain their original source/rules and remain ineligible.
