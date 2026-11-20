# Independent Model4 Cold decode allowance

Architecture READY on proposal dcd6493f460846ddeeba25ca1df02a7ac16ad408061af914361c577e6b1dc681.

Narrow resource boundary repair. The actual matched 2.5 GiB Model4
run trained and saved both 32-game updates, then its original host retired
cooperatively with exit 0 and all descendants confirmed reaped. Cold validation
failed in `read_hpo_rollout` before arrays because the independent audit was
passed the recorded **live** numeric bank limit of 536,870,912 bytes. Actual
first-batch file descriptors give a 583,721,519-byte existing read estimate
(230,855,727 base plus 352,865,792 scratch). This reconstruction stage is
separate from the bounded live bank writer and learner.

Use the remaining actual Cold/Resume/completed-parent Source budget as the
read-only Model4 decoder's numeric workspace bound. Keep the existing allocation
estimate and `reservation.resize` against the exact already-owned `pool.budget`;
its actual reserve is authoritative. Preserve finite existing deadline,
`max_owned_bytes`, expected configured model/selection/hash/storage/source
correspondence and all role/FD/cleanup checks. If there is no positive remaining
capacity or the actual read does not fit, fail normally without another account
or more time. No online training numeric or Source limits change. No numeric
schema, tensor, candidate, proof, dataset or learning behavior changes.

The existing legacy external owned Model4 audit already distinguishes its
independently admitted reconstruction allowance from the live numeric bank cap
(`hpo_owned_restore._audit_owned_cohort`). The current Rust shared audit reads
after positive Raw retirement and holds existing control/Originals under the
same Source. Use only actual unused capacity at that stage, not the full cap
again on top of retained owners. This is an allowance allocation, not a
measured RSS estimate. Resume and independently reopened completed data follow
the same rule through the existing core.

An isolated CPU read-only probe of the actual failed archive reproduces the
512 MiB refusal, then decodes all 331 first-batch endpoints with the existing
reader under an actual 2.5 GiB Source pool and its remaining admitted space.
That probe does not issue a Cold capability or qualify the failed parent.
Verify the exact dispatch/account rule in focused tests, unchanged legacy
behavior and frozen numerical kernels. Then repeat a fresh actual complete
32-game × two-update operation through original reaping and full Cold audit.
This allocation boundary change requires independent architecture review
before implementation. All original failed attempts remain ineligible.
