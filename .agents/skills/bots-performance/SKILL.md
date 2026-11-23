---
name: bots-performance
description: >-
  Measure and improve runtime performance, throughput, latency,
  memory efficiency, concurrency scaling, or resource guards in
  GodFieldBots. Use for performance-sensitive implementations,
  bottlenecks, regressions, worker scaling, memory-budget
  rejections, and performance reviews.
---

# GodFieldBots Performance Engineering

## Objective

Improve effective system performance while preserving
established correctness, security, and experiment contracts.

Optimize for measured outcomes, not minimum code changes,
maximum defensive checks, or isolated benchmark scores.

## 1. Establish the performance objective

- Identify the relevant workload and its constraints.
- Select end-to-end metrics appropriate to the task:
  games/sec, samples/sec, updates/sec, inference throughput,
  latency, memory use, or scaling efficiency.
- Include learning quality when comparing changes that may
  affect training semantics or convergence.
- Distinguish required performance from optional improvements.
- Identify existing baselines before introducing new tests.

For performance-sensitive implementations, consider efficiency
during design rather than waiting for a regression.

Do not optimize unrelated cold paths.

## 2. Respect established contracts

- Trust verified guarantees of libraries, frameworks,
  type systems, and established internal interfaces.
- Do not independently reimplement dependency validation.
- Avoid redundant guards for conditions guaranteed impossible
  within the applicable contract.
- Validate untrusted inputs and genuinely uncertain boundaries.
- Investigate suspected dependency violations using evidence.

A defensive check requires an identifiable failure mode and
a reason the responsible layer does not already prevent it.

Do not weaken established security or correctness guarantees.

## 3. Measure the actual system

- Prefer representative end-to-end workloads.
- Identify the dominant cost before selecting optimizations.
- Separate startup, warmup, and steady-state measurements.
- Account for workload size, batch size, worker count,
  hardware, and measurement variability.
- Profile CPU, GPU, memory, synchronization, and I/O
  only as relevant to the suspected bottleneck.
- Avoid expensive or redundant benchmark runs.

If evidence is insufficient, perform a bounded measurement
or add minimal instrumentation before making large changes.

Do not claim performance improvement without evidence.

## 4. Evaluate high-impact improvements

Consider alternatives including:

- Eliminating unnecessary computation and repeated work.
- Improving algorithms, data structures, and data layout.
- Reducing allocations, copying, parsing, and serialization.
- Reusing long-lived processes, models, and initialized state.
- Batching inference, I/O, or other repeated operations.
- Improving CPU/GPU scheduling and utilization.
- Reducing process, IPC, and synchronization overhead.
- Improving storage access patterns and locality.

Compare expected system-level benefit, complexity,
correctness risk, and implementation cost.

Prefer simple improvements when benefits are comparable.
Do not reject a high-impact structural optimization solely
because its diff is larger.

Apply the repository's architecture gate before modifying
interfaces, dependencies, concurrency ownership, or other
gated boundaries.

## 5. Evidence-based resource management

For memory and concurrency budgets:

- Identify the actual resource being constrained.
- Distinguish CPU RAM, GPU allocations, GPU reservations,
  process-private memory, shared memory, and page cache.
- Account for simultaneous allocation lifetimes and reuse.
- Distinguish observed peaks from theoretical upper bounds.
- Model concurrency using actual resource ownership.
- Examine whether rejected workloads are genuinely unsafe.

Every restrictive resource guard should explain:

1. The failure it prevents.
2. The accounting model and assumptions.
3. The evidence supporting its threshold.
4. Its treatment of uncertainty and peak demand.
5. The consequences of underestimating resource usage.

Do not introduce arbitrary multipliers or double-counting.
Do not remove necessary protection merely to pass a workload.

Prefer bounded experiments and monitoring when uncertain
resource behavior needs to be established empirically.

## 6. Implement and validate

- Implement the most promising justified improvement.
- Preserve functional behavior and learning semantics.
- Run focused correctness tests for changed behavior.
- Compare performance before and after under equivalent
  representative conditions.
- Check relevant resource peaks and scalability effects.
- Avoid unrelated refactoring and speculative safeguards.
- Do not launch full HPO or expensive formal training solely
  for performance verification without an approved budget.

When an optimization changes a gated architectural boundary,
obtain the required review before dependent implementation.

## 7. Completion criteria

A performance-improvement task is not complete merely because
a bottleneck has been described.

When implementation and measurement are feasible:

- Apply justified improvements.
- Measure the resulting system-level performance.
- Report regressions and tradeoffs.

If no safe improvement is justified, or measurement cannot
be performed, explicitly explain why and identify the
remaining evidence or decision needed.

Do not invent measurements or claim unverified gains.

## 8. Report

Provide a concise summary of:

1. Objective and representative workload.
2. Bottleneck and supporting evidence.
3. Changes and alternatives considered.
4. Before/after results and measurement conditions.
5. Resource implications and remaining limitations.

Stop when the agreed objective is met or further work
requires a new architectural decision, budget, or scope.
