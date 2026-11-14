# Explicit admission recovery without a restart

Independent design and implementation-conformance review READY. Correctness
review READY; implementation verification and actual operational release pass.
Both final whole-foundation reviews are READY. Admission release remains distinct
from cleanup proof and data qualification.

## Intent and revised requirement

The user challenged repeated environment restarts after discussing memory
headroom and process inspection. Root withdrew the request, and the user then
authorized continuing. The current
global ticket4868 has no termination receipt, but neither host ps nor nvidia-smi
shows a known surviving job. Those observations are diagnostics, not historical
proof of complete cleanup. Do not manufacture a termination or restart receipt.

Add a deliberate operator decision to resume GPU admission despite an old job's
unconfirmed termination. This supersedes only the mandatory-restart fallback in
ADR0037; automatic admission remains blocked on unresolved jobs. Failed or
incomplete data, original deadlines, complete-panel evaluation and every existing
qualification check remain unchanged. The operator accepts possible residual
resource contention; no claim of an idle GPU or complete cleanup is made.

## Alternatives and scope

Automatically treating PID absence, an empty process list, elapsed time or free
VRAM as cleanup would misstate the evidence. A new process supervisor/cgroup
subsystem would add substantial scope and cannot retrospectively reconstruct this
ticket's lost ownership. Prefer one explicit, auditable admission-only release.
No process killing, reboot, dependency, new global namespace or GPU allocation is
part of recovery. Existing normal termination and actual-restart recovery remain.

## Interface and records

Training owns `release_unconfirmed_admission(*, expected_ticket, operator, reason,
confirmation) -> AdmissionReleaseReceipt` in the existing hpo_gpu_lease module.
Use a distinct fixed confirmation string, exposed as `RELEASE_CONFIRMATION`:
`I authorize GPU admission to resume without confirming termination of the previous job.`

The immutable `jobs/<ticket>/release.json` has the following closed fields:

```
schema_version: 1
kind: hpo-gpu-admission-release
ticket, admission_sha256, admitted_boot_id, current_boot_id
operator, reason, confirmation, released_unix
```

Operator and reason must be nonempty text, reason must contain a non-whitespace
character. The receipt binds the exact original admission and current boot. It
has no resources, termination result, evaluation outcome or completion authority.
The existing global namespace stays v1. Old readers fail closed on the new file;
old admissions/termination/restart records retain their exact shapes and meaning.

CLI `hpo recover --ticket T --operator O` takes exactly one of
`--confirm-environment-restarted` (existing behavior) and
`--release-unconfirmed --reason TEXT` (new explicit behavior). Reason is required
only for release and rejected for restart. Help explains that release accepts
possible residual resource use and does not certify termination. CLI/help remains
optional-dependency free. Return the distinct release receipt as JSON.

## Concurrency, durability and failure behavior

All recovery uses the SAME existing per-user nonblocking flock and strict scan.
A live lock holder still rejects recovery. Scan validates release records as a
third mutually exclusive resolution alongside termination/restart; its boolean
means admission resolved, never proof of termination. Only the selected exact
ticket is resolved. Other unresolved orphans continue to block all campaigns.
Malformed/conflicting records still fail closed without rewriting them.

Use existing immutable publication/fsync. Exact replay requires the same ticket,
operator, reason, confirmation and current boot; any difference rejects. A retry
resyncs its receipt and may restore only a missing current pointer to the exact
original admission, matching existing orphan recovery. It never overwrites a
newer pointer. Injected file/directory/pointer failures must retain uncertainty
or reconcile the exact durable record. No old admission/data/receipt is deleted.

Normal GpuJob ownership, actual-process reaping and confirm_termination remain
strict. Evaluation verification must reject a release receipt wherever an actual
termination receipt is required. Future jobs acquire their own new admission and
must satisfy normal resource, storage, cleanup and data-completion checks.

## Verification and implementation plan

1. Add RED behavior tests for explicit release, exact retry, live-holder refusal,
   wrong/missing identity/confirmation/reason, conflicting/malformed records,
   normal-termination refusal, unresolved orphan isolation, and fsync/pointer
   faults. Assert old admission bytes survive and release cannot verify as a
   training/evaluation termination. Cover actual CLI flags/help without extras.
2. Implement only lease/scan and recover CLI changes, with existing helpers and
   no collection/training/evaluation numerical changes. Run relevant CPU lease,
   CLI/campaign/evaluation/coordinator checks and project static checks.
3. Obtain independent correctness and architecture-conformance review. Root
   records the superseding ADR/docs, verifies and locally commits. Run the new
   explicit recovery on4868 under the user's authorization with an honest agent
   operator/reason, preserving the admission and absence of termination/restart.
4. Continue all predeclared fresh a03 GPU cases with unchanged scientific inputs;
   audit normal exits, sources, counts, original data and evaluation receipts.
   Refresh source/package/optional-free/legacy proofs and final acceptance checks
   appropriate to the final source. This is functional evidence, not strength.

User confirmation of an actual restart is not present and must not be invented.
Future improvement: revisit process-group ownership only if normal cleanup and
explicit recovery prove operationally insufficient; it is not this increment.
