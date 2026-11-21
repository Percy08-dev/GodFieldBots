# Predeclared selection and separate GPU profile population

The selection protocol was recorded before formal scaling and deployment
timing. It applies the accepted closed telemetry contract to 24 primary
invocations and a separate 12-run calibration. Actual results are in the
[verification record](../verification/rust-player-runtime.md).

After the prescribed C8 choice of W for EACH scheduler, C16 runs both candidates
in the predeclared forward/reverse orders. Rank candidates that complete BOTH
C16 invocations and their full audits by total completed games divided by the
sum of their four collection durations. Within the owner's3% fastest set, prefer
the lower maximum complete PSS peak across the two invocations, then lower W,
then A (Tokio) for an otherwise exact tie. Record first/second collection and
each run separately; this selection statistic does not average away deployment
acceptance failures. The resulting ONE fixed scheduler/W is the new candidate
for both actual-entrance populations. Each entrance still receives its OWN two
old/new pairs and its OWN conditional default decision. Do not switch candidates
after seeing deployment results.

If neither candidate completes both C16 runs, withhold adoption. For the required
explicit-mode deployment observations, use the best fully audited C8 candidate
with the same3%/PSS/W/A tie order, label the missing C16 qualification and retain
all failures. This fallback does not authorize a default switch or waive any
Task6 correctness/restore/Source/process requirement. Failed observations remain
outside successful rate/PSS statistics and are never assigned zero metrics.

GPU-event data uses a distinct profile population after primary timing
and audits. Its five one-update conditions run in the following order.

1. Legacy Model4 C16/B16,32games.
2. A Model4 C16/B16 with its selected W,32games.
3. B Model4 C16/B16 with its selected W,32games.
4. Legacy ordinary Model3 C2/B2, its actual16-game entrance recipe.
5. Selected native ordinary Model3 C2/B2, SAME16-game entrance recipe.

Each profile uses the frozen producer for its population and the
corresponding primary workload with one update and explicit profiling.
Model4 uses width 192 and the ordinary Model3 preset uses width 128.
The recording declaration was clarified before any deployment or profile data.
C16 keeps 5 GiB, the actual HPO C4 entrance keeps 4 GiB, and the ordinary C2
entrance and its profiles keep 2 GiB. Seed 17, four epochs, minibatch 128
and formation wait 1 ms remain fixed. Model4 keeps the managed 24 GiB RAM and
12 GiB CUDA limits. Each profile remains a bounded observation without
proposal search or final-test evaluation. Each has a fresh owned namespace and full
actual exit/cold/Original/Replay/numeric/saved audit. If a C16 method already has
a reproducible failed capacity/cleanup precondition, report its profile NOT RUN
with that evidence rather than repeating the failure or changing its workload.
Keep the original recipe, callbacks and window for every condition, and
preserve all failed or incomplete originals. The five initial profile files
were empty because the default Kineto serializer failed on the original
file FD. The recorded repair protocol allowed one replacement for each
empty condition after validating the supported synchronous serializer with
actual CUDA activity. The replacement commands kept the same condition
order and workload.

Use GF_BOTS_PERFORMANCE_PROFILE=forward-window-v1 with the validated base
observation pair; record population=gpu-profile and recipe explicitly. The fixed
post1->post3 worker-thread window and three actual scored-row counts remain the
accepted contract. Input/forward portions cover rows2+3; output portions cover
rows1+2. Keep each topology/boundary/Model3-versus4/configuration separate. Tagged
Torch trace GPU-event durations are not synchronized host-wall stage totals,
full-run GPU utilization or same-batch latency. Missing GPU activity or an
incomplete window stays unavailable, never inferred from CPU/RTT/allocator data.

Primary24 and calibration12 use profile_recipe=off. Their throughput/wall/PSS
remain the only adoption comparison populations. Profiled times are diagnostics
with extra profiler/thread/I/O overhead, never substituted into those criteria.
The profile population is bounded verification of the required GPU transfer/
compute measurements, not evidence of learning quality or an additional search.

Both existing public entrances require --config; neither has an implicit
configuration-free learning default. Conditional adoption therefore means shipping
an explicit new Rust-runtime preset and making it the recommended NEW CUDA MLP
entrance in documentation/examples only after that entrance's gate passes.
Existing preset files/selectors and None/legacy parser defaults retain their
recorded meanings; do not transparently reinterpret an old configuration or old
parent/campaign on fresh/resume. New HPO presets retain their distinct schema6
study identity. A failed/missing gate leaves the new preset explicit opt-in and
the existing recommendation in place. No new CLI selector or automatic fallback
is introduced merely to manufacture a default switch.
