# Corrected Magical Stick / catalog v2 adoption

Date: 2026-10-28. Design: ADR0009, independent architecture design READY.
Implementation and final independent correctness/conformance reviews are READY.

## Frozen identities and preservation

Simulator source HEAD `ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6`, clean.
Prebuilt binary SHA-256
`84b7e900c468998ebe2c83e8567f4f47dcf5ecaad96fdc46c13eb2039725204b`;
build revision remains unverified. Catalog v2 schema1 SHA-256
`481ddfb0c068a1d2aaaf2dbcb6a5b4bd04396dfcc5c995e14e47b71222b6a51d`.
No Simulator build/source/environment modification is part of this adoption.

Old reader wheel is retained at
`dist/archive/7057693fe30574a413683852d9099e3a9d272b40da3d01b4517fd3eff1999d3a/`.
Its SHA-256 matches that directory name. Historical verification using the
installed old wheel with `/tmp/gfb-20261028-stageA-baseline/bin/python -I`
recomputed all192games of the original corrected-Buy pilot: PASS, source-set
hash `5ff2af59443bb258c17203791568785e67adf876294835071a6769c680b4bb04`.
This is old-contract evidence only. The failed PPO parent remains quarantined.

## Preregistered new run sequence

Freeze source/presets after independent reviews. Use fresh outputs under
`runs/catalog-v2-validation`, `runs/catalog-v2-pilot` for fixed12/192games.
Then `runs/ppo-smoke-v2/continuous` and `runs/ppo-smoke-v2/resume` each perform
two updates, with explicit pause after1 on the latter. Compare weights, Adam,
CPU/CUDA/shuffle and parent scheduling RNG, plus original-source health/proofs.

Formal parents are `runs/ppo-comparison-v2/train-seed17`, `train-seed29`,
`train-seed43`, sequentially under unchanged original budgets. All must finish
normally before final validation64games each, dataset.json and observation/
history LightGBM fits at seeds17/29/43. No finaltest or outcome-selected source
replacement. Record actual generation/fit costs and scopes separately.

## Acceptance status

- Historical reader/pilot: PASS.
- Ordinary suite: PASS501, five opt-in skipped,121.06s.
- Ruff/mypy: PASS;47sourcefiles.
- Existing real integration: PASS4tests,2.69s.
- New real contract matrix: PASS8freshgames in1test,2.01s; MP0/10 crossed
  omitted/null/self/opponent, fullMPpayment, damage, Response/refill/turn ordering,
  journals for generatedforms and diagnostic traces for omitted/null, Replay.
- Independent review: correctness83focusedchecks,conformance20focusedchecks;
  bothREADY. Source review alone does not claim executed integration.
- Wheel: all47Pythonfiles byte-identical; SHA-256
  `c2001f5248640f63251bb27273563829625d04503ce27be047a10b2a357e4d81`.
  Separate installed baseline optional-import/help/config boundary PASS;
  cu132 CPUinference/420candidates/exactNPZ/isolatedworker PASS with CUDA not
  initialized. Oldcheckpoint and oldpilot currentqualification reject;
  archived oldreader historicalqualification still passes.
- Fresh fixed evaluation: PASS12/12natural,5.813s,
  `runs/catalog-v2-validation/baseline-validation-e90d21e1-546`.
- Fresh pilot: PASS192/192natural,76.000s,
  `runs/catalog-v2-pilot/baseline-feasibility-pilot-e12ee84f-fa1`.
  All original-source feasibility checks pass; source-set
  `9bbec99851c55813f77778ac1f0addcbbfaee93a9fb4391c555583b33be94393`.
  Assessment `runs/assessments/catalog-v2-pilot.json`, SHA-256
  `826689887229459d7a7f128ff73a71a6cf43a721934327d0347ca4a4e8f3ac55`.
- New CUDA smoke: both continuous and explicit pause/resume complete32natural
  games/two updates/379learneractions. All originalcollection proofs revalidated;
  weights/Adam/CPU/CUDA/shuffle/parentRNG exactlyequal. Elapsed71.114s continuous,
  73.897s resumed including pause. Assessment
  `runs/assessments/catalog-v2-ppo-smoke.json`, SHA-256
  `19dd40e2a4343e38d1aabbe863df8b1ea06e0e29b9efa534a8a8175cf771aa4a`.
  All these runs used clean code`f59b0e1`; they are execution evidence, not
  policy-quality claims or sources for the later frozen comparison dataset.
- Formal three parents: complete at code `1df8849`, seeds17/29/43 with
  123/144/145updates,24,439/27,985/29,099learner decisions and
  5438.883/5429.276/5424.430seconds respectively. All6592games finished naturally.
  Finalvalidation:192natural games total. Fixed dataset81,523train/2,260validation
  decisions, content SHA-256
  `ab5d4913fda45795d1cd5aa02a4a9c4c3464268298c4d3ea49d56d4909df770f`.
  Build and index each revalidated original sources before publication; the
  2026-10-29 inventory check additionally matched415/415collection manifest hashes
  but did not repeat journal validation. Shared generation cost16588.399seconds.
- Six real fixed-data fits complete: observation17/29/43 each bestiteration98,
  validation L2 `0.6045529418541776`; history17 bestiteration95,
  validation L2 `0.59091264247396`. Following performance diagnosis and
  [budgeted-preparation acceptance](budgeted-preparation.md), history29/43 also
  complete with the same bestiteration95/validation L2. A separate seed17 repeat
  exactly reproduces inputs, model and inference bytes and is not a seventh cell.
  The original four fits are preserved. No final-test or win-rate inference follows.
- Simulator Python-environment repair remains separately pending; this task
  does not authorize mutating it.
