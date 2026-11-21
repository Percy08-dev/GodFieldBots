# Independent verification of runtime experiments

Architecture READY on design `ea7183f3cad9fb5aea4d85f83fd77da7574449e4e14436b746027e8785882d42`.
The tooling supports the approved comparison and accepted operation launcher. It does not add a population, change workloads or establish an
objective. Run after the observed original command exits. The launcher must
never be kept alive as a second Python controller.

On shared legacy Model4 use existing `validate_hpo_parent(require_complete=False)`
under a real read-only parent custody interval. It calls `validate_hpo_chain`,
which calls `audit_owned_artifacts` for every retained generation. Its
`inspect_retained_game_artifact_v4` uses `SealedScan.bundle` to decode every
retained tensor, check all archive contents and saved Adam steps and moments,
validate Torch RNG byte vectors and PCG64 state, and recheck original files.
The same chain then runs `audit_owned_samples` for Raw/Replay and learner numeric
correspondence. For ordinary shared Model3 use the existing
`validate_game_parent(require_complete=False)` under the equivalent custody,
with a final full `inspect_retained_game_artifact_v3` for the actual current
paired checkpoint if the chain has not already verified the entire pair.
Retired checkpoint files remain governed by their declared retention policy.
The verifier returns diagnostic counts/hashes and closes actual owners. It
never turns a healthy incomplete parent into exact-D or publishes HPO eligibility.

The single-owner external reader uses the declared effective preprocessing
budget, additionally bounded by the existing HPO Source allowance for Model4.
It receives a finite deadline from its own externally observed audit start plus
the exact effective run wall configuration. These are existing instance budgets,
not extra time on the original training or new memory allocated to a live job.
A capacity/deadline/owner failure disqualifies the measurement with its actual
reason. No implicit retry, broad budget increase or discarded failed artifact.
Tie comparison of numerical continuous and resumed controls uses already accepted
same-input controls and checks saved optimizer/RNG/schedule metadata and lineage.
Arbitrary live trajectories are not expected to agree under different real
search scheduling. Observations and audit elapsed time are separately recorded.

Native training's fresh Cold finalizer already opens original Raw/Replay,
learner numeric, full saved tensors and retirement state under actual native
capabilities. This stays inside the observed command wall. The external reader
validates exact hashes of its reported operation/clock/admission/Prepared/Exit/
completion/audit/current generation and checks saved data immediately after
observed retirement. It must not reconstruct a live Cold, Parent or Source.
For independently checking numerical values without a live control, use the
existing sealed paired decoder on current DATA under real parent ownership. No
new model load, Torch CUDA initialization or saved schema.

Measure phase durations from actual same-boot monotonic phase boundaries and
whole time from original observer Child start and wait. Each immutable
observation ID must match all selected telemetry. Separate first and second
collection. Rates use actual natural games and all participant decision counts
from committed original records. Recover candidate counts and search limits
from both seats of committed Journal6 data, with the same scope for both modes.
Read only the actual packed descriptor linked by the verified update/audit.
Recheck its original hash and inspect the actual PackedSource under the audit
budget. An audited Native descriptor can retain its recorded pending format.
Use that format only for this post-audit data read. The reader projects only
table counts, cutoff reasons and accepted decision totals, without feeding
policy inputs or opening any hidden Simulator history. Process one group under
an actual group loan at a time and release all per-group bytes before eviction. Missing
instrumentation stays null and disqualifies the associated timing criterion
when required. Use the accepted histogram rank/bin bounds and batch inversion
rules. Keep original PSS sweep coverage, observed thread roles and Python
process lifetimes separate from phase quota estimates. Do not treat diagnostics
as process retirement proof. Report no adoption on unavailable or split results.

Independent architecture review pins these boundaries before dependent driver
code. Test pure metrics readbacks/missing or corrupt inputs with compact fixtures,
then verify the actual cases on the final source. No stage may read private game
state into policy inputs or add any network or Simulator action.
