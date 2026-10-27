# Simulator Buy correction confirmation

Date: 2026-10-28. Scope: verify the user's Simulator correction for
[issue 0003](../issues/0003-buy-choice-control-metadata.md), using public HTTP and
existing prebuilt test executables. This is diagnostic verification, not Bots
contract adoption, a training collection run or a passed learning gate.

## Source and executable identity

- Simulator HEAD: `860d8b92db4ce82012e040510f221b52241f28aa`.
- Its existing worktree changes remain in README/spec/tool documentation; the
  engine/DTO correction is committed. No Simulator file was edited or built.
- Tested `target/debug/gf-test-server` SHA-256:
  `608805f07c59164cf1cac18ee9e7a73a13f02b5aa0a5b31edb216070903be41c`.
- Independent verifier used the existing prebuilt HTTP test executable
  `target/debug/deps/http_contract-dd29028f618e5246`, SHA-256
  `79a005198f11381f0d70ad15bf4c9dfcede7014c6cbee2c6ffff6c7a8c7fcb9c`.
  It was timestamped before the commit and contains the named regression;
  source HEAD alone proves neither executable's build revision.
- `gf-simulator` is a different executable, not the TestServer used by Bots;
  its different hash is not a TestServer mismatch.

Source inspection found the narrow input predicate in
`src/server/transport/http/dto.rs:724`, pending buyer choice in
`src/engine/resolution/economy.rs:246`, and updated scheduling in
`docs/spec/10-cli-client.md` §3.1. The patch also adds the target's response
before offer selection. Other resolving choices remain non-actionable.

## Actual public HTTP reproduction

Evidence: `runs/diagnostics/buy-fix-7c0c976437c5/` contains `verify.py`, the
unchanged original case, `report.json` and server/HTTP logs. All files are
diagnostic-only and marked `policy_eligible=false`; no policy worker, training
samples, hidden snapshot or rejection-derived signal was used.
Report SHA-256:
`542e654a337c5de631138d2a02cb96fb84f2e9d3943987eb11e09aa3d3674ada`.

Command, using Python 3.14 and the existing project environment:

```console
env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/gfb-uv-cache uv run --no-sync python /tmp/gfb-verify-buy-server-fix.py
```

The copied `verify.py` also reproduces the check and creates a new unique
diagnostic directory. The script starts the prebuilt TestServer with two
same-seed reservations, bounded request/process deadlines and stdin-EOF cleanup.
It replays the same six saved actions for each case, then submits the additional
fixed target Allow response required by the corrected contract.

| Check | Result |
| --- | --- |
| Original six saved actions | PASS: all accepted in both newly created games |
| After Buy at revision 6 | PASS: P1 is actionable `awaiting_response / defense`; P2 is nonactor false/empty |
| Target ActionOptions | PASS: same revision, `can_allow=true` |
| After P1 Allow at revision 7 | PASS: P2 is actionable `resolving / buy_accept_reject`, exact existing `buy_decision` boolean schema |
| Nonactor at purchase choice | PASS: P1 false/empty |
| Buyer ActionOptions | PASS: revision 7 and `buy_accept_reject` |
| Repeated Observation/options reads | PASS: three repeated reads per case preserve public values/revision |
| Buyer accepts | PASS: revision 8; offered Wood Shield moves P1→P2; price 4 is debited/credited |
| Buyer rejects | PASS: revision 8; no transfer, seller cards and both money values unchanged |
| Purchase wait ends | PASS: both decisions continue beyond `buy_accept_reject` |
| Same-seed public pending offer | PASS: buyer Observation and offer identical before the divergent decision |
| Owned cleanup | PASS: both games deleted, TestServer stdin EOF, exit 0, no log error |

Total diagnostic time: 0.193s. This is not a performance benchmark.
Public read stability does not by itself prove hidden RNG state equality.
The existing HTTP regression additionally checks full state/RNG equality across
reads and snapshot/fork continuation.

The first diagnostic attempt reached successful purchase acceptance, then its
verification code incorrectly expected an `instance_id` in public BuyOffered.
That event intentionally omits the ID. The corrected check uses participant-own
card observations plus public transfer/money events; it passed both cases. The
initial attempt remains retained at `runs/diagnostics/buy-fix-bfcc4d201cf0` and is
a diagnostic-script failure, not a failed server purchase.

## Independent bounded verification

`bot_verifier` inspected the code/spec and temporary-directory test fixture,
then ran the existing prebuilt executable from `/tmp`:

```console
/home/percy/develop/GodFieldSimurator/target/debug/deps/http_contract-dd29028f618e5246 buy_response_and_decision_are_actionable_and_snapshot_replayable --exact --nocapture
```

**PASS:** one test, zero failures. It covers the real Buy→target response→buyer
decision path, actor/nonactor metadata, stable state/RNG across reads and accept/
reject through a forked pending offer. Cargo/full Rust suite was **NOT RUN**;
the Simulator was not rebuilt. No complete baseline pilot was run here.

## Bots compatibility and disposition

The existing Bots validator at `src/gf_common/validation.py:139` still permits
actionable input only in `awaiting_action`/`awaiting_response`. Applying it to
the actual corrected revision-7 buyer Observation produces
`UnsupportedContract: typed metadata mismatch` in both diagnostic cases.
The configured Simulator pin remains the original revision, so normal startup
with the current checkout would also fail the source-revision check.

The server defect is **verified corrected**. Bots production integration remains
pending: review/adopt the new revision and normative snapshots, admit only the
current nonterminal buyer's resolving choice, retain other resolving exclusions,
and test worker/journal/checkpoint compatibility before a complete fresh pilot.
Raw diagnostic reads here do not weaken any production validator. Existing
failed runs and their learning ineligibility are unchanged.
