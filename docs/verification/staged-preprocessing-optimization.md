# Staged preprocessing optimization experiments

Status: implementation and correctness verification complete; primary measurements pending under accepted
[ADR0016](../adr/0016-staged-preprocessing-optimization.md). Stages0–2 only;
no production allocator/reader adoption, fitting, game collection or model update.
Independent design review and narrow counter/TLS clarifications are READY.

## Frozen reference and current environment

Baseline source e441053; installed native release SHA256
`32e3ad68f883087517a0d9b393764dd4d01a2543a24175d8b5eb912ded7f901f`;
native source `5e18ca4e3f0a371d79667b38c9fcf52b958e4da9907cd796b706bdf8e9f72fc5`.
Actual current Python is3.14.10, NumPy2.5.4, Rust1.99.0. Historical3.14.9 timings
are not the control for this experiment. Main native installation stays unchanged.

Dataset content `ab5d4913fda45795d1cd5aa02a4a9c4c3464268298c4d3ea49d56d4909df770f`:
415collections,6,784games,83,783accepted learner actions (81,523train/2,260validation).
Existing native `dataset inspect` with native-low-memory preset PASS. Parent/source
eligibility was revalidated by the original code; source-set identity remains
`bef26c2bbf98e9240846b8f8d8414d7488240c0470204a14f2c43a05b6be0628`.
This correctness pass overlapped preparation/build work and is not speed evidence.
Original six fit groups retain24file size/mtime/inode identities and18small-file
hashes. Full protected-file hashes are additionally checked by the freeze runner.

## Full Journal census

Independent verifier PASS:13,568declared participant journals,2,267,302,084bytes,
534,640records; recursive duplicate-key rejection and JSON parsing had0failures.
All files/records meet64MiB/8MiB limits. No repair, source rewrite or exclusion.
Artifact: `runs/assessments/staged-optimization-census.json`, with complete path/
hash/size/origin information. Only declared current sources were scanned.

| Journal record | Count |
| --- | ---: |
| header |13,568|
| decision |169,168|
| acknowledgement |169,168|
| transition |169,168|
| end |13,568|

Every decision input has action_options/catalog/events/history_end/observation/
typed_metadata. Selection.value has19,624null and149,544numeric records. All35,647
respond actions have empty ordered_defenses; respond_sequence has22,253typed
elements. Outcomes include13,474win objects and94draw strings. These are census
facts, not proof that an unimplemented typed reader accepts the corpus.

## Progress and interpretation

The standalone Serde gate passes12tests in AP, no-AP and AP+counter builds, covering direct f64,
required-nullable/missing, duplicate/unknown fields, exact bounded integer input,
external Outcome and actual nested borrowed/unescaped vsowned/escaped strings.
Runner controls have30Python tests passing, including actual observer cancellation/
reaping, case-insensitive allocator-environment cleanup, budget refusal and exact
inventory/binary/oracle authority. Ruff, mypy3experiment files, deterministic DTO
regeneration and rustfmt checks PASS. Independent code-conformance and correctness
reviews are READY for execution; reported defects were fixed before primary trials.

The final isolated System/mimalloc/jemalloc wheels each pass40existing native
ownership, source/session, cancellation and learning-input tests. Complete frozen
history preparation at1/1and4/4 for all three has identical dataset/sample/array/NPZ
identities. These correctness runs overlapped oracle work, so their times are not
speed evidence. Main native32e3ad68... and original models remain unchanged.

B and C each pass the full13,568file/534,640record unchanged read_journal + exact
Python canonical value/order oracle with0newrejections; both canonical record hashes
are `c05bfca5cf9319b59cff52022c77441db30ac8607d00fad5ed9e669283c7d90c`.
Artifacts: `runs/assessments/staged-preprocessing-20261030/typed-oracles/{B,C}`.
Observer completion and joined cleanup ownership are part of the acceptance proof.
This does not establish future typed feature-array or learning equivalence.

Review repairs add coarse preallocation admission and exact frozen-inventory
authority to the standalone harness. Interrupt forwarding now keeps the observer
alive until owned cleanup finishes, following ADR0004; setup failures terminate
remaining owned group members even when the direct root exits and closes pipes.
Allocator/compiler controls are sanitized without recording unrelated secret
environment values. Full primary matrices/counter runs and measurement acceptance
remain pending.

All primary cells use2048MiB/cache512MiB/chunk16MiB, workers/inflight1/1or4/4,
release optimization and thread environment1. Each gets warmup+five rotating
trials; no concurrent build/test/heavy work or profiler. Allocator TLS settings
are explicit for dlopened Python extensions; override/unprefixed modes are off.
Representation A/B/C/D keeps System fixed and isolates no-AP Cargo builds.
Wall, CPU, sampled RSS/PSS, post-release retention, scratch/storage/I/O and build/
validation/measurement costs must stay separately labelled. No fixed speedup
target, no automatic candidate adoption and no micro-to-whole speedup inference.
