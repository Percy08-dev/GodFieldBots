# 0003: Preserve simulator abort cause and dataset truncation

- Status: Accepted
- Date: 2026-10-27
- Decision owner: root
- Review: `bot_architecture_reviewer`, design READY, 2026-10-27; discovered by
  coverage follow-up. Final implementation conformance remains required.
- Implementation: implemented; final conformance/correctness reviews READY.
  [Coverage follow-up](../verification/coverage-followup.md).

## Context and reproduction

The new whole-episode test
`test_turn_limit_replay_reason_is_dataset_truncation_without_terminal_label`
fails: a positively acknowledged action has lifecycle status `aborted`, and the
Player Replay explicitly reports `turn_limit_exceeded`, but EpisodeResult and
journal label the end `aborted` without preserving its projected distinction.
No reward/transition is fabricated; the error is loss of the truncation cause.

Pinned simulator revision remains `44a26e3a16b808e826deb332fa72acfc5037d3ef`.
Normative `06-persistence-history.md` §28.11 says only that simulator abort reason
projects to RL truncation, with no terminal flag or invented final transition.
`01-server-spec.md` §26 keeps the underlying session lifecycle Aborted. These
facts must coexist. No simulator engine/source or policy input changes are needed.

## Decision

- Keep `EpisodeResult.end_kind` as the Bots projected outcome. Add nullable
  diagnostic `simulator_abort_reason`, sourced only from validated Player Replay.
  The raw positive acknowledgement remains `aborted`; preserve raw Replay intact.
- Validate replay Game/viewer/version and the closed episode-end shape before
  interpreting a reason. Both successfully reconciled participant replays must
  agree on end type/cause before classifying a simulator turn-limit truncation.
  Contradiction/missing required evidence is a replay/contract health failure;
  do not guess a cause or promote unknown delivery to acceptance/outcome.
  Compare the full closed closure identity (type, reason, player, decision),
  not just the reason string. Existing known natural results are never overridden.
- The exact reason `turn_limit_exceeded` projects to `end_kind=truncated` and
  `simulator_abort_reason=turn_limit_exceeded`. Other genuine simulator abort
  reasons stay `aborted` with their original reason. Local decision/deadline
  truncations may have a null simulator reason; preserve their local failure.
  When a local interruption coexists with fully reconciled simulator closure
  evidence, the projected simulator kind/cause may be recorded while the local
  failure stays intact. Without that evidence, retain the local classification.
- Journals record the diagnostic abort reason in the end record. Writer/reader
  accept an aborted acknowledgement followed by truncation only with that exact
  reason. Aborted lifecycle acknowledgement still closes the action stream.
  No terminal reward, terminal transition, final observation/bootstrap, resumed
  decision or invented action is added for a simulator truncation.
- Export cross-checks projected kind/reason against linked manifest episode
  metadata and reconciled Replay, preserves the reason in diagnostic sidecar,
  and keeps all non-natural or unhealthy runs ineligible. Reason is not a policy
  feature and is not added to decision-input or feature columns.
- Compatibility: journal schema 3 and derived schema 3, plus explicit
  end-projection version in run/header/derived provenance. Schema 2 and earlier
  are retained unchanged and rejected explicitly by the current reader/exporter.
  No automatic migration or reinterpretation of old `aborted` records. Reward
  policy, input feature, catalog, candidate and simulator schema versions do not
  change because their meanings are unchanged.

## Responsibilities and flow

HTTP validates the public Replay transport. The environment reconciles both
views and projects the closure before journal finalization. Common result types
carry diagnostic cause, journals enforce state/cause consistency, and export/
evaluation preserve the distinction. CLI records provenance versions. Policy
workers, search and model selection remain outside this information path.

## Alternatives

- Relabel every `aborted` acknowledgement as truncation: wrong for shutdown,
  invalid-attempt limit, action timeout and other aborts.
- Leave only the raw Replay as evidence: continues losing the distinction in
  primary summaries and datasets.
- Silently relax/reinterpret schema 2: violates the explicit compatibility rule.
- Architecture review confirmed schema 3 and a separate projection version:
  the former identifies required persisted fields/compatibility and the latter
  names the transformation independently of reward/input semantics. No legacy
  reader or migration framework is required.

## Verification

Failing whole-episode regression becomes passing; distinguish generic abort,
turn limit, local decision limit and natural terminal. Test contradictory/missing
closure evidence, aborted-ack/end consistency and no fabricated labels/actions.
Check schema-2 rejection, schema-3 roundtrip/export provenance and failed-run
ineligibility. Add a bounded real TestServer turn-limit case if public fixtures
permit. Independently review correctness and architecture conformance after the
fix. Broader coverage additions continue without depending on this decision.

Verification completed: schema-3 projection/episode/journal/golden subset passed
69 tests during independent correctness review; final whole suite passes 200
ordinary tests and three opt-in real-server tests. The exact simulator turn-limit
projection is a controlled public trace; real TestServer max-turn=1 is unsupported
(minimum 150), so that live projection case was NOT RUN. A real local decision
cutoff verifies nonterminal/no-label handling. Current wheel/export evidence is
in the linked coverage record. No simulator changes were made.
