# Diagnostic processing versus shared-transport uncertainty

Status: accepted bounded repair clarification, independently READY on proposal
SHA d16766b58383c1b55fc19074bcaaa97012661831e11a0dd45a47edbf985d2698. Owner: root. Resolves telemetry review finding T1 against
the accepted existing-control-pipe design. No new transport, thread, process,
dependency, scientific record or policy behavior is proposed.

## Concrete source and ambiguity

SharedInferenceService._exchange uses the existing framed stdin/stdout authority
channel. Actual read/write/EOF/framing/timeout failure makes it uncertain. Replies
are uncorrelated, so a late diagnostic {'accepted':true} could be mistaken for a
later AUTHORIZE response if uncertainty were simply cleared. That existing
quarantine is correct and must remain. The current broker's DIAGNOSTIC_BEGIN/END
processing can also escape into its global infrastructure-shutdown handler;
those diagnostic-only exceptions must instead be contained.

The accepted statement that a diagnostic control/snapshot error cannot invalidate
otherwise valid policy work applies to diagnostic PROCESSING on a healthy,
framing-valid channel. It cannot establish that an actually failed shared
transport remains safe. Preserve the repository's infrastructure/unknown failure
classification rather than clearing uncertainty or claiming cleanup from absence.
This explicitly resolves the prior ambiguous combination of 'same control pipe'
and generic diagnostic-error wording.

## Minimal repair and ownership

Keep _exchange's actual shared-channel failure ownership unchanged. A read/write,
EOF, malformed/truncated frame, deadline timeout or failed broker/service-health
check remains genuine infrastructure uncertainty and prevents new authority until
the existing actual shutdown/recovery path resolves it. Do not clear uncertain,
retry/renew deadlines, discard uncorrelated replies or invent resynchronization.
A diagnostic read timeout cannot be relabeled a successful harmless snapshot.

For a COMPLETELY received, recognized DIAGNOSTIC_BEGIN/END request on a healthy
channel, contain ONLY its diagnostic validation/counter/snapshot/profile-result/
file-publication processing inside the diagnostic branch. A processing failure
invalidates that observation/scope, records missing/incomplete evidence and
returns the existing complete framed {'accepted':false} reply. No new response
family or authority fields are needed. It does not set scientific failure state,
change RequestAuthority, poison a healthy service or abort valid policy actions.
The parent records that valid negative response as diagnostic failure; later
ordinary AUTHORIZE/RETIRE remains usable. Full valid replies are consumed before
another exchange; no late response remains outstanding in this case.

Keep response writing and actual input framing outside this local containment:
a failure there is a real shared-channel problem. Do not hide actual GPU/model/
service-health failures under a diagnostic label. No added cleanup grace or
per-row fence. Profile first-scope END still happens at its actual logical end,
even if snapshot/publication fails, and later callbacks only clean an invalid
window on their actual worker thread. This does not alter the already-closed
profile lifecycle rule.

## Verification and evidence disposition

Test a healthy BEGIN and END processing exception/negative reply followed by a
valid ordinary authorization, preserving original authority state. Cover broker-
side diagnostic validation and snapshot/file failure, plus parent handling of a
fully framed negative reply. Test real shared read/write/EOF/partial-frame/timeout
uncertainty remains quarantined and no subsequent authority exchange succeeds.
Existing timeout stub reproduction remains evidence of correct quarantine under
this clarified transport boundary, not proof that resetting it would be safe.
Do not relabel a malformed or lost reply as healthy diagnostic rejection.

Use deterministic CPU/frame/stream doubles or bounded existing CPU mechanism
checks; no GPU/game/training budget follows from this clarification. Existing
profiler-END, null-outside-collection batch ID and observed completed-retirement
ordering findings are direct repairs to their already-closed contracts. They
need focused regressions and independent rereview, not changed record meanings.
All prior actual CPU6 evidence remains bound to its historical producer. Final
Source/correctness, native companion, profiling, calibration and full approved
performance populations stay separately required.
