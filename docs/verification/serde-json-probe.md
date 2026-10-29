# Disposable serde_json comparison

Measured2026-10-30; independent read-only benchmark code review READY.

Scope: user-requested quick parser/build/drop experiment, not production adoption.
No application/native package dependency, interface, eligibility or dataset format
changes. This is the bounded disposable experiment exception in AGENTS.md;
production architecture review is required before any later adoption.

Use a standalone Cargo crate under /tmp; compile the current ordinary custom
parser modules unchanged, and a pinned cached serde_json1.0.151 with
arbitrary_precision/float_roundtrip for real128-bit PPO control numbers. Use
release, same selected input files and framing/whole-file retention. One file
resident at a time; record read, parse/build and tree-drop times separately with
coarse file-level timers. Compare3alternating pairs after warmup; count bytes,
records and semantic digests outside parse timing; measure CPU and maxRSS in
fresh processes. Input sample: fixed hash1/64 declared JSON inventory plus each
kind's largest file, manifest under runs/performance/serde-probe-20261030.

The custom side retains its production budget/deadline checks; serde side tests
the proposed simpler ordinary allocation path. This comparison includes that
policy difference and does not isolate lexer speed. It does not test full source
validation, feature extraction, learning, pathological compatibility or production
memory/failure guarantees. Keep main installed extension and six fits unchanged.

## Results

Input:455files,55,389,000bytes (52.82MiB),8,935records,3,975,461value nodes
per pass. Largest journal2,458,246bytes; largest Replay382,121bytes; largest
other JSON397,272bytes. Each fresh Rust process makes3passes. One warmup process
per variant, then3alternating paired trials; no concurrent builds/tests/training.
Both variants parse all records and retain the complete file tree through typed
digest verification before separately timed destruction. No whole-corpus preload.

| Median metric | Current custom | serde_json |
| --- | ---: | ---: |
| Parse/build, seconds per pass |1.044999|0.264667|
| Typed digest/check, seconds per pass |0.139674|0.127036|
| Drop, seconds per pass |0.077472|0.048287|
| Parse+digest+drop, seconds per pass |1.261712|0.443286|
| Whole process wall,3passes, seconds |3.901146|1.438812|
| Whole process wait4 CPU,3passes, seconds |3.900280|1.437906|
| Linux process high-water RSS, MiB |36.71|45.60|

Per-column medians are computed independently; the median combined interval need
not equal the sum of individual phase medians. Parse/build ratio3.948x;
parse+digest+drop ratio2.846x; whole-process wall ratio2.711x. Median high-water
RSS increases8.89MiB. This supports comparing ordinary library allocation before
retaining the custom per-allocation design for this corpus size, not a claim that
all future source sizes satisfy current production memory/failure boundaries.

Important representation difference: arbitrary_precision preserves floating-point
lexemes in serde Number. The probe converts them to f64 during digesting, outside
parse time; current custom Parser::number converts during parsing. Consequently
3.948x does not compare identical work placement or isolate JSON lexer speed.
The2.846x combined figure includes that deferred conversion and complete typed
content checking. The custom side also keeps its budget/deadline checks, whereas
serde uses ordinary allocation. Neither ratio is a measured whole-preprocessing
speedup; source validation, feature extraction, numeric export and fit did not run.

Every pass in all8processes matches file/record/byte/node counts, raw-input SHA
and typed semantic SHA. The latter tags integer versus float, retains arbitrary
precision integer text, compares float bits and sorts object keys; it is not the
dataset's canonical serialization hash or an offline-eligibility proof.
Semantic SHA:38b3bb315d0e3762104fb9089f1601c1e6a96a130bac8ff6c6fa87d46accc934.
Measured trials have0kernel-accounted read/write bytes (warm page cache; OS I/O
counters are sampled lower bounds, not WSL host physical traffic). The probe
creates no numeric archive or large scratch output. Full CPU/RSS/I/O/log/storage
measurements are retained alongside results.

## Evidence and reproduction

- `runs/assessments/serde-json-probe.json`: summary, medians, source hashes, limits.
- `runs/performance/serde-probe-20261030/results.json`: all8process results.
- Same directory: inputs.tsv/inputs.json and archived source/Cargo.toml,
  source/Cargo.lock and source/src/main.rs.
- Source revision96add031107b468113b61f958c7edd6348cbcab8; Rust1.99.0.
- Release standalone binary SHA:
  49f7f572f32c779f4af18d34b70a62c913362eafb05dd6585f9199f23cbf9835.

Offline release build and5Rust tests PASS. Tests cover shared line framing,
duplicate keys, Unicode,128bit integers, f64-bit agreement and integer/float
distinction, plus included production helper tests. Code review found no skipped
parse, retained-tree lifetime or timed-drop blocker and identified the deferred
float-conversion caveat above. This is successful-input evidence, not exhaustive
Python/Serde compatibility acceptance. No root/native manifest or lock changes;
the installed production extension remains1a0b1746... and no inference/learning
run or existing artifact was changed.

```sh
env -u VIRTUAL_ENV CARGO_HOME=/tmp/gfb-serde-cargo \
  cargo +1.99.0 test --offline --release --manifest-path /tmp/gfb-serde-probe/Cargo.toml
/tmp/gfb-serde-probe/target/release/gfb-serde-probe custom \
  runs/performance/serde-probe-20261030/inputs.tsv 3
/tmp/gfb-serde-probe/target/release/gfb-serde-probe serde \
  runs/performance/serde-probe-20261030/inputs.tsv 3
```

Raw benchmark timings are collected with tools/benchmark_performance.py in a
fresh process for each cell. Future production adoption still needs its own
architecture decision and end-to-end output/resource/compatibility verification.
