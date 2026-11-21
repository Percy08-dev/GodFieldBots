# Initial campaign CPU-control operation clock

Status: accepted narrow public-entry clarification. Independent architecture
READY on proposal SHA90f7f10ef414f0f5e8cfaebab53baa8a7d19012e92aa25ad4e1da1f7b9c58b3b. Owner: root. The accepted session-clock1
and operation2 wire shapes, public training-clock recipe and Native Session
control phase are unchanged. This settles the initial CPU operation's five literal
fields without inventing fractions, a margin or an extra duration.

The actual initial campaign-continue operation has no parent, training admission
or GPU. It points to the authenticated bootstrap SHA, and execution_request is
None as the accepted operation2 validator requires. The original SESSION anchor
t yields soft=t+S and hard=soft+M under the accepted session-clock1 recipe. M is
the maximum declared TOTAL training/evaluation operation wall; current
materialize_hpo_run sets run.wall_seconds=campaign.run_wall_seconds and legacy
evaluation derives that exact run unchanged in wall, so both currently use this
same M. Keep that source contract explicit rather than adding finalization again.

For ONLY this initial CPU-control operation set admission_bits, execution_bits,
term_bits and kill_bits to the EXACT soft-dispatch bit pattern, and finish_bits
to the EXACT hard-session bit pattern. Require finite representable t<soft<hard;
refuse overflow/collapse rather than invent a minimum. Existing A<=E<=TERM<=KILL
<FINISH syntax is satisfied. Native Session uses the accepted hard-cutoff clock;
these coincident first four values do not authorize train admission or redefine
the per-training operation milestones. No initial operation-clock1 record is
issued: that recipe requires a real training effective-config/P, whereas this
CPU operation's config is the bootstrap. session-clock1 supplies its provenance.

Native already independently refuses train bind at soft. The fixed campaign
controller also checks soft before EVERY new train/evaluation dispatch. An initial CPU control process may still consume remaining hard time
to verify state and report an honest no-dispatch boundary after soft. It cannot
bind/admit new work by treating hard as a dispatch deadline. Existing original
operation2 admission/exec/TERM/KILL/finish remains authoritative once work is
bound. Every actual training operation uses its OWN original anchor and explicit
P recipe, with finish<=original hard; legacy evaluation retains its own total W.
No renewal of older dispatched work or copying old clocks into new authority.

Responsibilities/dependencies: CLI owns pure session/bootstrap construction and
its real CampaignStore OFD publication/exec; existing common/native validators
only validate their existing literal syntax. HostControl authenticates initial
purpose and current campaign custody. No Native/Source/model/runtime change.
Compatibility: only new public campaign construction; old literal internal
operations, ordinary factory, legacy1–5, receipt bytes and current record schemas
retain their meanings. Public clocks stay DATA, not admission or proof of exit.

Verification: pure source/config tests for same exact soft bits in all four
fields, hard finish/session equality, finite/representable refusal, no-P CPU
bootstrap versus explicit-P training distinction, no added finalization or grace;
actual Store/sealed-launch inspection with final exec mocked. Source review must
confirm Native Session consumes hard independently and actual dispatch uses soft.
No host/game/GPU process is needed to accept this construction clarification.
Full genuine public Task6/Task7 tests remain mandatory after application wiring.
