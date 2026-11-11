# Group Pack storage and concurrent-workload comparison

Date:2026-11-11. Code acceptance:0178199; default remains directory. This report
combines two explicitly different measurements. It establishes engineering
behavior and observed costs, with no policy-strength or qualified-learning claim.

## Identical real bytes, same HDD

Six counterbalanced AB/BA pairs archive and restore the SAME historical pilot:
32managed groups,192games,4492members/273,062,500bytes plus4,212,418control bytes.
All12trials and independent arithmetic/coverage/immutability audit pass. D: is
Windows disk2, ASMediaASM1153e USBHDD4,000,787,030,016B through WSL9p.

| Six-trial median | Directory archival reference | Production Pack/Writer archival path |
| --- | ---: | ---: |
| Write through publication/retirement | 166.4435s | 19.8487s |
| Open/read/verify/recheck/close | 171.5748s | 734.0756s |
| Final apparent bytes including catalog | 277,274,918 | 23,646,793 |
| Logical write throughput | 1.5887MiB/s | 13.3257MiB/s |
| Logical verified-read throughput | 1.5412MiB/s | 0.3602MiB/s |
| Group write p95 /p99, n=192 | 7.6716 /17.8417s | 0.5226 /1.2767s |
| Group read p95 /p99, n=192 | 3.3907 /3.9635s | 32.0066 /37.4495s |

Packed median actual checkpoint3.6083s, DELETE0.1014s and complete controls/seal
11.4932s are nested phases. Sampled DB/catalog/sidecar peak median49,383,496B.
The directory writer is a documented archival reference, not the live collector.
Integrity work differs: public packed loans repeatedly hash the immutable DB;
directory readback rereads files. This code-derived explanation is not an isolated
causal measurement. Caches are uncontrolled; readings are not cold-device tests.
Some bounded ext4 development work overlaps the historical panel, as its receipt
records. Small empirical tails and reused-coordinator RSS limit inference.
Archives confer no Source rights or migration-derived learning permission.

## Actual PPO concurrent with HDD collection

One revised four-trial block5670221899a348be94452c8f6613b2b7 completes PASS/exit0,
order directory/packed/packed/directory, with independent raw/derived audit.
Both formats put ALL PPO originals, numerics, checkpoints and controls on the
same ext4 filesystem; each separate six-game collector writes to the SAME D:HDD.
Packed PPO and sibling use distinct DB/catalog/RAM roots. Both change format,
so this is a combined configured-workload comparison.

Each permanently nonformal parent performs2updates with pause1/resume2, seed17,
CPUactors/CUDAlearner and original7200/5400/7140/60limits. All4parents complete:
128natural PPO games,1676accepted decisions,64optimizer steps; the independent
collectors complete24natural games. Each trial has2actual updateACKs and one
Source-request/ACK window overlapping collector actionHTTP. All actual children
retire and monitoring is healthy. Four complete-parent original/numeric/retention
reader audits pass, with every owned ext4 file byte/stat unchanged before/after.
These audits use the recorded8GiB preprocessing cap; the earlier artificial2GiB
audit refusal is preserved. Runtime/training/storage budgets were not changed.

| Trial /format | Total trial | g0→g1 | g1→g2 including pause/resume | PPO environment-group p95 /p99 (n=16) |
| --- | ---: | ---: | ---: | ---: |
| 1 /directory | 582.067s | 107.986s | 469.037s | 12.440 /17.521s |
| 2 /packed | 271.209s | 86.888s | 179.691s | 12.328 /16.467s |
| 3 /packed | 270.270s | 86.891s | 178.885s | 12.447 /16.693s |
| 4 /directory | 578.332s | 107.278s | 466.177s | 12.550 /17.655s |

Median trial time is580.200s directory and270.739s packed. Aggregate38game/trial
throughput is0.06550 and0.14036games/s respectively, including32PPO+6independent
games. Generation2 includes the explicit pause/collector wait; its median is
467.607s versus179.288s. The independent collector's single environment-group
return takes331.921/330.628s directory and21.257/21.296s packed. Group boundaries
are actual Simulator-environment run_group entry/return, not SQLite COMMIT or
complete archive publication. Quantiles use type7; n=16tails and n=1collector
samples are descriptive, not reliable population percentiles.

| Sampled observations | Directory trials | Packed trials |
| --- | --- | --- |
| Aggregate process RSS peaks | 3.3112 /3.3679GiB | 3.3816 /3.3882GiB |
| Owned HDD apparent-byte peaks | 14,945,788 /14,945,698B | 1,451,663 /1,447,567B |
| Owned all-filesystem apparent-byte peaks | 103,994,587 /103,997,997B | 54,185,739 /40,312,645B |
| Owned tmpfs observed peaks | 0B | 2,408,869 /2,329,187B |

Requested RSS sampling is250ms, filesystem sampling15s plus final scan; raw and
derived records retain gaps, coverage and scan costs. The inode census spans
current-trial ext4/HDD/RAM roots. RSS may double-count shared pages; all sampled
peaks are lower bounds and filesystem scans are non-atomic. Storage's managed
2GiB reservation peak includes a temporary whole-available validation reservation;
it is neither RSS nor disk size, and is not an OS limit on all model processes.

Packed actual returned collection metrics preserve every producer timing even
though the optional archive observer's pack_metrics array is empty. HDD collector
checkpoint is0.8153/0.7708s, DELETE0.27055/0.27295s, combined seal/publication
2.4080/2.2715s. PPO collection checkpoints on ext4 range0.00742–0.01391s. These
nested producer phases are not added as independent costs. Each packed trial
observes17submitRPC calls totaling0.56617/0.58173s, includingIPC/validation and
provisional NORMAL COMMIT acknowledgement;23operation-lock acquires total
0.000104/0.000098s. Registered condition coverage observes no condition-wait
calls; this does not establish zero uninstrumented SQL/admission latency.

After completion, packed PPO has2DBs totaling4,841,472B plus12,288Bcatalog per
trial; independent HDD DB is1,437,696/1,433,600B plus12,288Bcatalog. Final WAL/SHM/
journal files are absent. Intermediate per-category DB/catalog/WAL/SHM/journal
peaks are NOT RUN because the registration observer produced no valid category
samples. Per-filesystem peaks above remain valid. Logical MiB/s for this mixed
operational workload, pure SQL/admission timing and optimizer-kernel intervals
also remain NOT RUN. Source-request/ACK overlap is not exact optimizer/HDD-I/O
simultaneity. No matching solo runs means no slowdown estimate.

## Failures, compatibility and decision

The original full-parent-on-D condition remains FAIL: initial63f2block failed
open-FD rename visibility; after the reviewed lineage handoff fix, cd82 completed
2ACKs then failed private0700pruning (actual mode0777). Its other3trials remain
NOT RUN. A revised preflight35a8 failed before workloads on escaped mount-source
notation; strict single-pass decoding/exact drive-root normalization passes both
independent checks and real HOST admission. All failed runs remain preserved and
ineligible. No security check, pruning step, budget, mount or ACL was relaxed.
No failed and revised samples are pooled.

Six compound D: completion-recovery fault constructions remain NOT_EXERCISED
because even held-FD metadata can fail after namespace movement. Uncertain
application owners are retained and never declared healthy; process exit is a
separate boundary. Keep PPO parent/control/pruning on ext4. Standard packed DB
placement on D: remains supported by the ordinary runtime evidence.

Packed archives write faster and use less apparent space in this workload, while
public verified reading is substantially slower. The default remains directory;
explicit storage opt-in is available. Formal campaigns, genuine formal/final64
handoff, qualified Dataset4 fitting, HPO and final-test evaluation remain NOT RUN.

Raw pairs, controls, timelines, wait/sidecar/resource observations, producer-returned
metrics, complete-parent immutability audits and independent arithmetic audits
are retained under runs/assessments/sqlite-group-pack-v1-20261111. Detailed test
and scope acceptance is in [the verification ledger](sqlite-group-pack-v1.md).
