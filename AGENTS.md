# GodFieldBots: development instructions

## Purpose and sources of truth

This repository implements Python bots, learning, and evaluation for GodField.
The simulator is a separate repository at `../GodFieldSimurator` (currently
`/home/percy/develop/GodFieldSimurator`). It owns game rules and authoritative
state transitions. Do not copy its engine or modify that repository as part of
a Bots task; report a required simulator change as a separate task.

Read `docs/architecture.md` before design or implementation, and relevant
accepted records in `docs/adr/`. Check the simulator's current normative specs
when changing its integration. Record the revision and schema versions used;
do not silently reinterpret an older dataset using current semantics.

## Model documentation

Follow the [model catalog guide](docs/models/README.md) when adding a model or
changing its network blocks, default tensor shapes, policy-input contract, or
compatibility. In the same change, update the catalog and affected model pages.
Regenerate the affected neural SVGs when their graph or representative profile
changes and run the catalog check before acceptance. Link existing run records
for trial-specific settings and verification records for claims about results.

## Engineering efficiency and resource discipline

- Trust documented library guarantees, type invariants, and
  established internal contracts. Do not duplicate validation
  already owned by another layer. Validate untrusted inputs,
  integration assumptions, and actual failure boundaries.

- Defensive checks, resource limits, and safety margins must
  address a concrete failure mode or documented requirement.
  Do not introduce speculative guards or unexplained margins.
  Consider both the benefit and runtime cost of safeguards.

- Resource accounting must distinguish measured and estimated
  usage, private and shared allocations, steady-state and peak
  demand, and actual concurrent lifetimes. Do not double-count
  shared resources or reject valid workloads solely because
  of an unnecessarily pessimistic estimate.

- Treat efficiency as a requirement for performance-sensitive
  paths. Consider algorithmic cost, batching, allocations,
  data movement, concurrency, and end-to-end throughput.
  Prefer evidence-based optimization over speculative
  defensive complexity.

- Use `$bots-performance` when implementing or reviewing
  performance-sensitive changes, investigating bottlenecks,
  changing worker concurrency, or designing/reviewing resource
  guards. Root must include the relevant performance objectives
  in delegated work.

- The skill does not override architecture review, simulator
  contracts, information boundaries, correctness requirements,
  or required verification. Performance work must report
  measured results or explicitly identify missing evidence.

## Agent roles and routing

Use these custom agents for delegated work. Root owns requirements, design,
coordination, project configuration, design documentation, and final acceptance.

| Role | Model / effort | Responsibility |
| --- | --- | --- |
| root | `gpt-6.2-sol` / `xhigh` | Design, decomposition, decisions, integration |
| `bot_implementer` | `gpt-6.2-sol` / `high` | Python implementation, relevant tests, fixes |
| `bot_reviewer` | `gpt-6.2-sol` / `xhigh` | Independent correctness and learning/evaluation review |
| `bot_architecture_reviewer` | `gpt-6.2-sol` / `xhigh` | Independent design and implementation-conformance review |
| `bot_verifier` | `gpt-6-luna` / `high` | Bounded investigation, specified checks, evidence summaries |
| `bot_advisor` | `gpt-6.2-astra` / `high` | Difficult decisions and unresolved disagreements |

Definitions live in `.codex/agents/`; the TOML `name` is the routing identifier.
Select it with the spawn tool's `agent_type` field. `task_name` only labels a
thread; setting it alone does not select the custom role. If the current session
does not expose these agent types, start a new session to load the configuration.
Delegate implementation and independent review explicitly. Do not launch every
role for every task. Missing definitions or unavailable models are configuration
problems: report them rather than silently substituting a different model.

At most three child agents may be open concurrently (four agents including
root). Child agents must not spawn further agents. Use one code-writing agent
at a time by default, with explicit file ownership. Root must not edit files
owned by an active implementer. Reviewers, verifier, and advisor do not edit
files. These are role instructions, not separate sandbox permissions; child
agents inherit the parent's permissions.

Prefer Standard speed. Do not enable Fast solely to accelerate routine work.
The generic child default is `gpt-6.2-sol` / `high`. Custom definitions set both
model and effort explicitly. Reuse relevant task threads for fixes, but do not
use the implementer's thread for independent review. Pass a bounded brief rather
than the entire chat or raw logs whenever the runtime supports it.

## Architecture gate: first implementation included

Architecture review is mandatory **before the first Bot implementation**,
including the initial package structure and first end-to-end feature. It is
also mandatory when a change introduces or changes:

- a subsystem or a cross-module responsibility/dependency;
- a public or cross-module interface, or the simulator integration contract;
- a third-party runtime/development dependency;
- a persisted dataset/checkpoint format or its compatibility policy;
- the information boundary, concurrency ownership, or failure boundary.

Root classifies the task before implementation. A typo, local behavior fix, or
private refactoring that preserves those boundaries needs no architecture agent.
If a supposedly local task crosses a boundary, pause the affected implementation
and apply the gate. Do not split a structural change into small tasks to bypass it.

For a gated task:

1. Root writes a bounded design: requirements, responsibilities, dependency
   direction, main interfaces, data flow, information visibility, failure
   handling, compatibility, and verification. Record uncertain decisions as
   open questions, not invented requirements.
2. `bot_architecture_reviewer` reviews the design before dependent implementation.
   Resolve blocking findings first. A bounded disposable experiment may gather
   evidence for an open question, but is not an approved production foundation.
3. `bot_implementer` implements the accepted design and necessary tests. Changes
   to its assumptions return to root for design review.
4. `bot_reviewer` reviews correctness; `bot_architecture_reviewer` separately
   checks the final implementation against the accepted design. Once the code
   is stable, independent read-only reviews may run concurrently.
5. Root accepts only after blocking findings are resolved, relevant checks pass,
   and the architecture document/ADRs reflect the implemented decisions.

A blocking finding is a concrete violation of an agreed boundary or requirement,
or a consequential design decision without enough evidence to proceed safely.
Distinguish these from optional improvements. Root records the disposition and
reasoning; it must not silently ignore an unresolved blocking finding. Escalate
an unresolved substantive disagreement to the advisor.

Architecture review checks responsibility ownership, dependency direction,
circular coupling, replacement cost, I/O isolation and testability, information
flow, compatibility, and failure/concurrency ownership. Prefer the simplest
design that satisfies current needs. Do not create speculative frameworks,
interfaces for every class, or empty layers merely to resemble a pattern.

Maintain `docs/architecture.md` as the current design and `docs/adr/` as the
history of significant decisions. Record context, alternatives, rationale,
consequences, and verification. Any deliberately accepted debt needs its impact
and an explicit condition for revisiting it. Supersede old ADRs rather than
erasing their reasoning.

## Implementation, review, and escalation

Root's delegation brief must include:

- objective, scope, and assigned files;
- relevant source/specification references and accepted design/ADR;
- inputs, outputs, constraints, and acceptance criteria;
- required verification, bounded execution scope, and expected summary.

The implementer owns tests for changed behavior. The verifier runs specified
checks or investigates a narrow question when that reduces noisy root context;
it does not replace independent review or decide learning quality.

Invoke `bot_advisor` for a consequential design/learning decision that root
cannot resolve with available evidence, a substantive review disagreement, or
the same issue surviving two fix-and-verification cycles. Supply the alternatives,
evidence, attempted fixes, and precise decision needed. Do not call Astra for
routine implementation or every review. Root owns the resulting decision and
replans unresolved work rather than repeating an unbounded repair loop.

Return concise findings with severity, file/line or design reference, impact,
and a reproduction/check where possible. Verification reports distinguish PASS,
FAIL, and NOT RUN, with exact commands and relevant limitations. A review with
no findings is not evidence that unexecuted tests passed. Recheck changed areas
after fixes; avoid repeatedly rerunning unrelated checks without a reason.

## Learning and simulator boundaries

The current simulator learning-input contract is in
`../GodFieldSimurator/docs/spec/07-external-interfaces-adapters.md`, section 21.1.
Policy input is the acting participant's Player Observation, Player-visible
Events, typed Action metadata, and optional acting-player ActionOptions atomic
hints under the updated section 21.1. Any public catalog use must be explicitly
permitted by the applicable contract and version-pinned.

- Acting-player `ActionOptions` may be used as policy features, atomic masks,
  search candidates, and teacher signals under the updated simulator contract
  and the user's explicit authorization. Pair them with the same-revision acting
  participant Observation; map hand/learned IDs to that participant's visible
  references and use perceived definitions for display/features. Atomic hints
  do not guarantee complete combinations, ordering, targets or resource legality.
- Do not use `SequenceOptions` or rejection detail as policy features, masks,
  search candidates, or teacher signals. Do not probe rejections to infer hidden
  legality. Typed syntax is not a legal-action oracle; the engine validates the
  submitted action.
- Do not feed another participant's private view, canonical hidden history,
  snapshots, GameSeed/RNG, or future information into the policy. Diagnostic and
  experiment-control data must remain separate from policy-visible inputs.
- Keep learning, inference, and evaluation responsibilities distinct. Review
  train/evaluation separation, reward perspective, leakage, and comparison fairness.
- Preserve the simulator's distinctions between natural outcomes, truncation,
  aborts, unsupported rules, and infrastructure failures. Do not invent wins,
  losses, or action transitions for interrupted runs.
- Record code/simulator revisions, schema and reward-policy versions where
  applicable, seeds, configuration, model/checkpoint identity, and evaluation
  conditions. State reproducibility limitations rather than claiming that a
  seed alone guarantees deterministic model training.

## Repository checks and current scope

Python is 3.14 with the GIL enabled and the project uses uv. The Phase 0–3
baseline foundation uses httpx/NumPy, pytest, ruff and mypy. PyArrow, LightGBM
and mutually exclusive CPU/CUDA Torch builds are optional extras. Phase 4 implements native CPU LightGBM
inference and guarded fresh training. The corrected-contract 192-game baseline
pilot passes provisional feasibility; real-data LightGBM fitting has not run and still
requires the explicit validated source/split/decision-budget conditions.
ADR0008 implements the MLP PPO, generation/pause-resume and fixed-data comparison
subset; current final acceptance/experiment evidence is in
`docs/verification/mlp-ppo.md`. GRU/Transformer and final-test evaluation remain
future work. Legacy inclusive-generation LightGBM retains its explicit pilot/
source/decision-budget gate; fixed-data mode instead requires all three complete
PPO parents and prescribed final-policy validation sources. Do not infer training quality or a passed
learning gate from synthetic fits, dependency smokes or baseline tests.
The real two-update/resume smoke passes; the first formal seed-17 parent failed
after 24 updates on the then-unresolved zero-MP dynamic-weapon target contract
(`docs/issues/0004-zero-mp-dynamic-weapon-target-contract.md`). Preserve that
whole parent as ineligible; later seeds/dataset/real fits have not run. Do not
derive a legality rule from its rejection or shorten budgets to claim acceptance.
Simulator revision ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6 now specifies that
contract and corrected MP payment with catalog v2. ADR0009 governs adoption;
current new-runtime/pilot/PPO/fitting evidence is in
`docs/verification/catalog-v2-adoption.md`. Old evidence remains historical;
new full experiments use fresh `runs/ppo-comparison-v2` directories.
Adding dependencies follows the architecture gate.
Use the actual project's checks and the phase evidence in docs/verification/.

For configuration-only changes, validate TOML, Codex configuration loading,
role discovery, and links/routing instructions. Do not add a learning stack or
application tests just to validate Agent configuration. Respect pre-existing
uncommitted work and do not bundle unrelated edits. The user has authorized
local commits at coherent, verified milestones; root owns staging and commits.
Do not push or publish without an explicit user request.
