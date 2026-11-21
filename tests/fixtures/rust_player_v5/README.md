# Original Python v5 comparison cases

These are synthetic participant-visible comparison originals captured with the
unchanged Python `PublicCandidateGeneratorV3.generate_game` and baseline policies
from `9680f06ad8ba61d65d142863742908678638b02b`. They confer no Simulator legality,
learning-data admission, runtime-port acceptance, throughput or training-quality
claim. Existing older generator goldens remain unchanged.

`originals.json` pins the shared Simulator072dabb contract, catalog2/public-ja-v3,
rules-v2, complete action configuration and numeric costs per episode. Its source
inventory contains actual SHA256s for all common modules, both policy modules,
and the original public fixture/support inputs; the capture tool refuses sources
that differ from that original revision. The separate tool SHA256 identifies the
capture recipe. The episodes digest uses the project's canonical JSON encoding.

Each episode owns a separate public event bank, search RNG and three policy
instances. Each decision records an event delta and complete own-input frame,
ordered own references, exact clock readings with original Python caller names,
full compact candidate table/report, ordered actions, search RNG before/after,
and Random/Heuristic/Refresh selections and actions. CPython RNG states serialize
as `[version, 625-word state, gaussian-cache]`. Random and Heuristic policy states
are explicit; Refresh has no random stream and records null RNG fields. No
learning-control NumPy state is supplied to policy or fixture input.

Clock reads start at zero. Most cases freeze there, yielding exact zero report
milliseconds. The cutoff case uses `min(call_index * 0.00001, 0.009)` seconds,
which crosses the 1ms precount and 75%-of-10ms discovery deadlines while leaving
the table-construction reserve. Every actual read is retained. No report timing
is normalized, rounded, or removed. Tests replay the saved readings and require
all readings to be consumed, without generating expected results.

Coverage includes miracle defense priority; physical Rainbow-first ordering;
duplicate definitions with distinct visible card references; an empty event
interval; Magical Stick at MP0/10; a 2048-row bounded search with RNG advancement;
both clock cutoffs; and four sequential Refresh decisions, including partial
public damage, turn-window closure, subthreshold attack and two repeated calls
per frame. The weapon response cause uses a definition also visible in own cards;
the miracle case supplies public Mudflow events while own cards contain Angel
Armor and Rainbow Curtain. Unknown causes remain the original generator's
responsibility.

Regeneration is explicit, from this repository root:

```sh
.venv/bin/python -m tools.capture_rust_player_goldens --replace
```

Without `--replace`, an existing capture is refused. Never update this original
to make a future port pass: compare the port against these pinned outputs and
retain these files with their original source/version semantics. Live wall-clock
trajectory equivalence, broad RNG seeding APIs and actual native execution need
separate port tests.
