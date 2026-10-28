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
- Freshpilot/PPO/realfits: NOT RUN yet.
- Simulator Python-environment repair remains separately pending; this task
  does not authorize mutating it.
