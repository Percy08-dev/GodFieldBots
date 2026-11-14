# PPO clock observation without scheduler-induced rollback

Bounded root proposal after CPU regression failures. Independent architecture
review is required before implementation. This clarification changes no persisted
run/deadline format, original deadline field, hpo-deadline-v1 projection, numerical
kernel, reward or policy input. GPU ticket4868 remains unresolved and untouched.

## Evidence and decision

The existing coordinator records monotonic m0 before wall w0. Its invocation and
cleanup guards read current wall before current monotonic. With a constant clock
offset, scheduler gaps between either pair create apparent rollback and can
exceed the implementation's0.1s tolerance. Controlled-clock reproduction confirms
the mechanism; observed regression failures alone do not prove a real wall step.

The bot_advisor recommends bracketed samples and explicit uncertainty failure.
The0.1s tolerance is implementation detail, not a promised measurement precision
in ADR0008. Retain that tolerance; do not fix this by increasing it or replacing
the original anchor with a convenient later one.

## Private sampling contract

Keep operation_started=m0 and operation_unix=w0 exactly as the original first two
reads; immediately capture m1=monotonic after w0. This brackets the wall read.
The initial bracket must have finite values and0<=m1-m0<=0.025s. Do not re-anchor
or retry that original observation. Reject an imprecise anchor as clock sampling
uncertainty, not as a proven wall rollback.

At each existing clock-check boundary, sample mb=monotonic, w=wall, ma=monotonic.
An acceptable bracket has finite values and0<=ma-mb<=0.025s. Make at most three
attempts, without sleeping, replacing any deadline or excluding their elapsed
time from the original budget. For an accepted bracket compute:

```
rollback_min = (w0-m1) - (w-mb)
rollback_max = (w0-m0) - (w-ma)
```

Reject as rollback if rollback_min>0.1s. Accept and return sampled w if
rollback_max<=0.1s. Otherwise retry; unresolved ambiguity or repeated imprecision
fails explicitly as uncertainty. Do not report a definitive clock step where
the observations cannot establish one. The0.025s/three-attempt limits are bounded
engineering constants, not search dimensions or new caller configuration.

## Integration and failure ownership

Use a small private helper in ppo_run.py for invocation and cleanup checks, not a
clock framework. Preserve all existing monotonic budget calculations and stored
absolute wall/admission/execution deadlines, including formal projection order
and resume's original expiry/last_unix checks.

Validate the captured initial bracket inside the coordinator's existing owned
try/cleanup scope before await child.start, so a new sampling error cannot escape
past reserved child ownership without its normal never-started/close path.
Moving the private check definition/validation within this existing scope is
allowed; do not create a new early unowned failure boundary. No physical GPU is
used to test this path. Existing actual-owner cleanup proofs remain compulsory.

After successful checks use their returned wall observation for last_unix;
preserve max(previous_last_unix, checked_wall). Do not immediately replace it with
an unchecked time.time read. Check at the same postcommit boundary before a
pause. In cleanup, catch clock sampling/rollback failure into the existing error
path, preserve any initiating error, and continue existing cleanup/failure
bookkeeping. On failed cleanup sampling retain the last checked high-water value
rather than fabricating a valid wall observation. Keep the documented
pre-final-write timing scope; final report-only time reads confer no eligibility.

## Guarantees and explicit limitations

Constant offset cannot falsely establish rollback or straddle0.1 with accepted
brackets. Persistent net loss>0.1 from the invocation anchor cannot be accepted:
it fails as rollback or uncertainty. Smaller boundary changes may conservatively
fail as uncertainty. Sampling retries consume the unchanged original monotonic
budget. Original absolute deadline values are never enlarged or rewritten.

Transient changes between observations and net backward steps masked by earlier
forward steps remain outside this original net-offset guard. Persisted wall-only
pause records cannot distinguish elapsed pause from a clock rollback that leaves
resumed wall>=last_unix. Neither this fix nor the original implementation proves
zero extra true elapsed time across that unobservable case. Record this limit;
do not silently compensate, invent cross-boot time authority or change budgets.

## Verification

Deterministic fake-clock tests must cover constant offset; anchor/current read
pauses; bounded retry; exhausted uncertainty; nonfinite/decreasing monotonic
samples; persistent backward/forward adjustments; exact/ambiguous0.1 boundary;
and high-water persistence. No system clock writes or hidden time monkeypatch
to make real functional data pass.

Use actual CPU common-coordinator tests to show initial/check/cleanup sampling
failure preserves original errors, closes actual owners and prevents qualified
completion. Repeat the four failed HPO coordinator cases with GPU hidden;
verify normal pause/resume preserves all original deadline fields, existing
rollback/expiry rejection, and all eight fixed AST pins. Independent correctness
and implementation-conformance review plus relevant legacy regressions are
required. Physical GPU acceptance/recovery remains separate.
