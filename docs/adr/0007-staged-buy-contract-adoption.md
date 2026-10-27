# 0007: Adopt staged Buy control and pin model gameplay semantics

- Status: Accepted
- Date: 2026-10-28
- Decision owner: root
- Review: `bot_architecture_reviewer`, design READY, 2026-10-28. All eight
  normative hashes, actor predicate, preserved ownership and schema-2 compatibility
  policy verified; no blocking or open design findings. Independent final
  correctness and architecture-conformance reviews are READY, with no blockers.
- Implementation: complete and verified; fresh evaluation 12/12 natural and pilot
  192/192 natural, recomputed provisional feasibility passed. No real-data fit. See
  [adoption verification](../verification/buy-contract-adoption.md).
- Supersedes: ADR0005's old Simulator pin/spec snapshots and ADR0006's inference
  artifact schema-1 compatibility only; their remaining decisions are retained.

## Context and scope

The user authorizes updating Bots for the corrected Simulator and rerunning the
fixed evaluation/pilot. Simulator HEAD is
`860d8b92db4ce82012e040510f221b52241f28aa`, replacing the original contract pin
`44a26e3a16b808e826deb332fa72acfc5037d3ef`. It changes Buy gameplay as well as
HTTP metadata: target Allow/Mirror response precedes offer selection; the buyer
then acts at `resolving / buy_accept_reject`. See Simulator10 §3.1 and Simulator09
§17.3. The full-possession rule in the latter is explicitly project-provisional;
adoption is of this pinned Simulator, not verified equivalence to the live game.

Disposable public HTTP confirmation already passed both decisions on the
prebuilt TestServer; current Bots correctly rejects that newer contract until
adoption. [Confirmation](../verification/simulator-buy-fix.md) preserves the
failed historical pilot and diagnostic evidence. No Simulator build or edit,
engine copying, new dependency, model training or new action family is in scope.

## Integration and responsibility

1. `gf_common.domain` pins the new full revision and reviewed normative hashes.
   Root updates current experiment presets, current architecture and README;
   historical ADRs/artifacts retain their old identities.
2. In `gf_common.validation.observation`, an input phase is awaiting_action or
   awaiting_response, or **only** resolving with required_choice buy_accept_reject.
   Actionability still requires current_player == viewer_player_id and no result;
   exact typed descriptors and all existing DTO types/ranges remain checked.
   Other resolving states, nonactors, terminal/aborted projections and unsupported
   choices remain false/empty or explicit failures as the current contract
   requires. Do not fabricate schema/flags, silently wait, or weaken all resolving
   checks. Existing client, worker, generator and journal consume this same
   validated Observation rather than duplicating a new predicate.
3. Keep same-revision actor ActionOptions, viewer scoping, own perceived catalog
   and event-API-only history. Purchase choice still has both boolean candidates;
   the preceding defense retains supported empty Allow. No Mirror implementation
   or expanded defense rules are required by this adoption.
4. Existing client/environment/process/retry/deadline/cleanup ownership remains.
   Tests must use normal AttestedClient/DecisionInput/worker paths, including the
   corrected purchase state; the earlier raw diagnostic bypass is not production.

## Persistence and compatibility

Journal/derived schema 4, run manifest 3, policy input, feature, encoder,
generator/scorer, catalog, reward and end-projection identities keep their
existing meanings and shapes. Linked run manifests already require the exact
supported Simulator revision and reviewed spec snapshot. New ingestion rejects
historical runs through those checks; do not relabel old manifests or refill
missing actions. Raw structural reads are not current training eligibility.
Retain archived old readers/checkpoints and failed pilot unchanged.

LightGBM inference artifact schema **2** adds the required public constant
`simulator_contract_revision` bound to the new pin. Schema 1 lacks this gameplay
identity and is explicitly unsupported; never infer it from the current checkout
or edit an existing artifact to upgrade it. A wrong revision also fails metadata
inspection before native loading or Simulator startup, including explicit toy
loads. Native text, encoder/signature, adapter/policy algorithms, runtime pin,
artifact path/digest binding and training completion schema 1 are unchanged.
The existing safe inference metadata is the proper owner of this public contract
identity; no GameSeed, URL, source paths or diagnostic state is added to features
or to native-fit parameters. Native child uses its pinned common constant when
saving; parent and inference child validate it using existing inspection.

Consequence: old synthetic checkpoints cannot load with the updated adapter.
There is no production trained checkpoint to migrate. A new toy fit can verify
schema-2 lifecycle without claiming playing strength. Portable export and broader
cross-contract compatibility remain Phase 9 work.

## Reviewed normative snapshots

Record these actual source snapshots with the new revision, including the user's
existing documentation edits. Source HEAD does not prove binary build identity.

| Simulator file | SHA-256 |
| --- | --- |
| `docs/spec/00-specification-overview.md` | `a2ef3bbf2d8a1e88300d94f75bccfd1956a9b0ea18c14dce2dc0c1c41845a03d` |
| `docs/spec/03-game-engine.md` | `4e6a3ff1eaea998cffbdb29383f2713bbd4efa16876d59ad0e481ca0c72feee9` |
| `docs/spec/05-observation.md` | `8d8326c1eed4a8d2f81c822f1ed6c6b2782dfec309f1efbcc250d3c4f727cda9` |
| `docs/spec/06-persistence-history.md` | `85bbf2461d9445c143dfce304967e36cfb8705580620e01b622134136ae06fdd` |
| `docs/spec/07-external-interfaces-adapters.md` | `057577a3a3678cdc76a427bf578bd50295433fc0f096e50be3352a1f6e38062b` |
| `docs/spec/08-test-requirements.md` | `2d3f41349ff796e27590a6dd70b469fb2d547134ff5d659aba770fb02cb9a731` |
| `docs/spec/09-gameplay-rules.md` | `7d3721fd5b4cd86c6c39b995e3378784542e34fd1be6c4a08707555c8fb66e33` |
| `docs/spec/10-cli-client.md` | `31a0ad7fcc310d987d54bef1e6805aff3fdb31cb28e18b563f715206eb06588b` |

## Verification and acceptance

- Exact Observation actor/nonactor/result/phase/choice/descriptor matrix;
  malformed/stale/foreign ActionOptions and old false buyer contract fail closed.
- A permitted public fixture from the confirmed buyer Observation/options crosses
  client, DecisionInput, candidate, worker and journal/export boundaries.
- Prebuilt real TestServer integration replays the saved six actions, target
  Allow, then both boolean decisions using normal strict client checks. Check
  nonactor metadata, same-revision hints, transfer/payment vs rejection, journal
  roundtrip/Replay reconciliation as applicable, and owned cleanup.
- Schema-2 toy fit/save/inspect/worker lifecycle passes; old schema, missing/wrong
  Simulator identity fail before native import/listener. Old linked dataset/pilot
  revision is rejected; no historical successful-source substitution.
- Run standard pytest/Ruff/mypy, real integration and separate wheel baseline
  imports/native schema-2 load. No optional library is required for baseline.
- After source stabilizes and independent correctness/conformance are READY,
  commit the integration and freeze config before a new live evaluation/pilot.
  Preserve the same two-GameSeed × seats × three-policy-seed 12-game evaluation,
  then the complete 32-GameSeed × seats × three-policy-seed pilot (192 games).
  Same generator/scorer/actions, fixed policies, concurrency two, configured
  original budgets; new run directories and source identities. Changed Simulator
  rules imply different trajectories, not a paired causal quality comparison.
- Recompute gate from all linked journals/report/Replay/health using existing
  feasibility validation. Any nonnatural or unhealthy episode leaves it unmet;
  do not replace games. Record run/config/code/binary/spec hashes, counts, costs
  and the explicit gate result. A clean pilot is provisional execution evidence,
  not a model-quality claim or permission to invent train/validation datasets.

## Alternatives and retained limits

Keeping the old pin prevents using the corrected server. A blanket resolving
relaxation admits automatic/unsupported states. Patching only the live client
leaves worker/journal validation inconsistent. Reusing schema-1 checkpoints
cannot prove which Buy rules their fitted targets came from. Explicit narrow
validation, linked run rejection and one required versioned checkpoint identity
are sufficient; no compatibility registry or generic migration framework is
needed. Existing filesystem/OS timing and failed-publication quarantine debt is
unchanged. Revisit multi-revision compatibility only with actual cross-version
evaluation requirements and separately tested migration semantics.
