# Collection HTTP bottleneck follow-up

Date: 2026-11-17. Read-only investigation plus one disposable instrumented normal
run. Production implementation is unchanged. This follows the user's question
about the HTTP component of [collection timing](collection-optimization-investigation.md).

## Finding

The large HTTP total is primarily inside HTTPX request execution and waiting,
with substantial overlap between response-header waits and long parent event-loop
callbacks. Explicit response JSON decoding and the measured selected HTTP
validators are a small component. Thousands of requests also accumulate transport
and HTTP-stack work. The measurements do **not** identify network bandwidth or
Simulator game computation as the principal cause.

The strongest next hypothesis is to reduce synchronous parent decision validation
and recording that occupies the event loop, then consider reducing independent
HTTP round trips under the existing participant/revision contract. The exact
counterfactual saving from either change is NOT MEASURED. Header wait includes
server, socket and scheduling time; callback overlap is association, not a proof
that the same duration would disappear after a change.

## Fresh 32-game measurement

Bots `6c55d3a` (production implementation still `dece918e`), Simulator
`91d5926a7572db327eaea3751e7c24f428120996`; unchanged Model4, inference5,
journal6, collection7/proof4/registry5 and Numeric6/Wire7 semantics.
Original normal MLP width192/two-layer, seed17, four resident workers/two games,
16 games per update/two updates, original budgets/deadlines and GPU reservations.
The timing population ran without concurrent tests, audits or reviews.

```sh
.venv/bin/python runs/collection-http-investigation-v1/run-http-diagnostic.py \
  --freeze runs/collection-http-investigation-v1/freeze-current.json \
  --config runs/collection-http-investigation-v1/actual-configs/mlp.toml \
  --run-id 558c9fd6-9806-4ade-9d25-a697e47ba4ac \
  --context-id 570c7f9c-06b0-4c49-92a4-6673f9457448 \
  --output runs/collection-http-investigation-v1/actual
```

PASS exit0; 32 natural games, 611 decisions, two observed normal microbatch
updates, healthy `incomplete/update_cap`. Collection callback wall 27.689s;
coordinator wall 38.799s. These instrumented one-run totals are not a paired
performance comparison. The script preserves original calls, protocol, body,
headers, retry/deadline and cleanup owners; trace extensions contain timing only.

The [354-file freeze](../../runs/collection-http-investigation-v1/freeze-current.json)
also pins actual HTTPX 0.28.1/HTTPcore 1.0.9 and asyncio implementation files.
[Original result](../../runs/collection-http-investigation-v1/actual/result.json),
[bounded HTTP observations](../../runs/collection-http-investigation-v1/actual/http-observations.json),
[hashed aggregation](../../runs/collection-http-investigation-v1/http-analysis.json).
No observer errors/drops, missing traces or dropped per-request trace events.

### Client and transport scopes

All values are **summed call-seconds across this same 32-game cohort**. HTTP
methods contain HTTPX requests, which contain the trace phases. Calls from
different games overlap. These rows are not an additive collection-wall ledger
or predicted wall savings.

| Scope | Summed seconds |
| --- | ---: |
| Six public HTTP client methods | 9.858 |
| HTTPX request execution, nested inside preceding row | 9.068 |
| HTTP/1.1 response-header receive interval, inside HTTPX | 4.965 |
| HTTPX time outside the five recorded HTTP/1.1 trace phases | 2.910 |
| Send request headers | 0.622 |
| Send request body | 0.325 |
| Close response | 0.153 |
| Receive response body | 0.093 |
| Selected HTTP validators, outside HTTPX request | 0.398 |
| Response JSON decoding, outside HTTPX request | 0.061 |

The 2.910s residual includes HTTPX request construction, response handling,
pool/control/inline work, scheduling and diagnostic overhead; no narrower
attribution is justified. Validation timing suppresses nested selected validators
to avoid double counting, but does not cover every inline check or validator.
It includes request-action validation and Observation prechecks as well as
response validation.
JSON plus measured validation totals 0.459s, about 4.7% of method-call time.

There are 3,202 traced requests and 7,562,506 response-body bytes. No connect/TLS
event is present in these scoped calls; the connection is already established
before episode creation. Async clients already reuse connections. Response-body
receive intervals are short; the payload volume alone is not evidence of a
bandwidth bottleneck.

| Request route | Count |
| --- | ---: |
| Observation | 1,145 |
| Events | 611 |
| ActionOptions | 611 |
| Action submission | 611 |
| Info/attestation | 96 |
| Join | 64 |
| Catalog | 32 |
| Create | 32 |

Create includes required attestation, catalog and post-create fences, hence five
requests per create-method call. The decision loop obtains Observation, Events,
optional ActionOptions and submits an action; changing actors requires an
additional own-view Observation fetch. Finalization also reads observations.
The count 1,145 must not be interpreted as 1,145 unnecessary requests.

### Parent event-loop overlap

Passive `asyncio.Handle._run` observations record callbacks lasting at least 1ms.
Across the invocation, 1,528 such callbacks occupy 15.810s, with 15.046s thread CPU.
Their union overlaps HTTPX request intervals by 4.642 summed call-seconds.

Of 4.965s response-header intervals, 2.745s (55.3%) overlap these recorded callbacks.
The eight longest header intervals range from 60.1–213.9ms; each overlaps parent
callback occupancy for essentially its full duration (within approximately 0.2–0.4ms).
This is strong evidence that the long HTTP tails contain parent scheduling delay.
It does not uniquely identify which instruction caused the delay, or establish
that all overlap is removable: the parent can be working while the server works.
Callbacks shorter than 1ms and OS scheduling remain outside this observation.

Existing source/timers identify candidate checking, journal publication and
capture as synchronous parent decision work. Their expensive all-row operations
are distinct from the small HTTP response-validator totals above. Those parent
operations are the next profiling/optimization target; HTTP JSON decoding alone
cannot account for the large total.

## Independent server-side evidence

The existing matching original 32-game server logs supply per-request
`duration_ms`, measured from the middleware's initial `Instant` to construction
of response metadata immediately after `next.run` returns:
`GodFieldSimurator/src/server/transport/http/operational_log.rs:62,92–101,141`.
It includes request logging/copy and downstream response construction, but
excludes subsequent response-body streaming/capture and completion-log emission.
Body completion emits the already-timed metadata without updating its duration
(`operational_log.rs:180–198`). It is **integer milliseconds rounded down**,
not CPU time or handler-only time.
Zero means less than 1ms at that precision, not zero work. No `Server-Timing`
response header was found.

The [explicit timer-description correction](../../runs/collection-http-investigation-v1/server-log-timer-correction.json)
supersedes the inaccurate `timer_semantics` fields in both original summary
artifacts. Their numeric statistics and original hashes remain unchanged.

[Existing-log summary](../../runs/collection-http-investigation-v1/existing-server-log-summary.json):
16 log members/3,346 matched responses, read through immutable SQLite/pack data;
before/after database/manifest SHA/stat identities remain unchanged. All 1,145
Observation, 611 Events and 611 ActionOptions responses record 0ms. For 611 action
responses, 608 record 0ms and three record 1ms. These coarse historical server
durations cannot be subtracted from the new client trace to calculate pure
network time, but do not support attributing the multi-second client sums solely
to Simulator handler computation.

The [fresh run's server logs](../../runs/collection-http-investigation-v1/actual-server-log-summary.json)
confirm the same pattern: 16 logs/3,346 responses, the same route counts, all
Observation/Events/ActionOptions at logged 0ms, and action responses at 610×0ms
plus one×1ms. Their database/publication SHA/stat identities also remain unchanged.

Server GET Observation and ActionOptions project the player view under the
per-game session lock. Action submission applies the action and builds events,
next Observation and optional ActionOptions; subsequent dedicated reads repeat
some projection work. The client validates that Act response, retains only its
summary, and obtains policy inputs through dedicated endpoints under ADR0005.
Changing that contract or introducing a combined endpoint is a separate reviewed
change; the Simulator repository was not modified.

## Limits and verification

This is one instrumented small-model 32-game sample. Larger models can change
actor delays and scheduling. Trace phases include scheduling and observer cost;
the 2.220s header time outside recorded callbacks is not pure network/server time.
No optimization A/B, learning quality, formal HPO or final-test conclusion follows.

PASS [original new-parent audit](../../runs/collection-http-investigation-v1/new-audit.json):
`validate_hpo_parent(require_complete=False,max_owned_bytes=2GiB,
deadline=monotonic()+180)` and exact-context `verify_gpu_termination`; all five
resources reaped. All 354 frozen SHA/stat identities remain unchanged. All
3,202 requests correlate with JSON responses and complete traces; no observation,
trace, transport, measurement or progress errors/drops/orphans. Independent
`bot_reviewer` evidence review is FINAL READY: all request phases, overlap
calculations, both decoded server-log cohorts and immutable hashes pass. Its
server-timer description finding is resolved by the explicit correction above;
its validator-scope clarification is incorporated. No benchmarks, application
tests, training or Source audits were rerun by the reviewer. No production changes
or application-test claim. Next steps should target parent event-loop occupancy and request count;
replacing the JSON decoder is low priority on the measured evidence.
