# Concealed Dream draw verification

## Release identity

| Item | Value |
| --- | --- |
| Simulator source | `9e47d4c5f8cd3f60437f9aeabc95076c1dc04321` |
| TestServer SHA-256 | `f95fadbc8eaad23fbe63e17ac9bdb0ebeca4b237956dedeab09d86f30cfd94bc` |
| Action profile | `card-instance-simulator-v2-shared-20261124` |
| Gameplay API, public event schema, catalog schema | `v2`, `2`, `2` |
| Saved formats | Journal `6`, `candidate-columns-v2`, `sqlite-group-pack-v1`, HPO6 structured rollout `8` |
| Candidate rule signature | `0334e53e7b7081f1cbb5b0771a9b091b49dac9c9d77d8035b07251916c11ff12` |
| Native capability | `card_instance_concealed_draw = raw-preserved;feature-omitted-v1` |
| Native source inventory SHA-256 | `4cb258ef880ce2e67851d04698191f9911ce862a6b0038b7bc950dd3e9fbfa89` |
| Native wheel SHA-256 | `abaaf0a0ab050860cabd9ba69795a52f4d7587ec6013b4df0e32db72f5c7fca4` |
| Native extension SHA-256 | `ef1bd0deebac5ab85a2d31ffbb1357bb7369c790568a8d1977831235c58c94db` |

The wheel was built offline with `tools/build_rust_player_wheel.py`, CPython
3.14.10, and `--allocator-mimalloc`. It was installed in `venv-final` under
`runs/concealed-dream-draw.2yyZLu`. The installed extension and bundled
supervisor agreed on the source inventory and passed the corrected Native
reader preflight.

## Fixed reproduction and compatibility

- **PASS** The pinned Simulator replayed the seven p65 actions. HTTP 200,
  `/events` and Replay agreed on all eight concealed draws for Player 1.
  Player 2 received none of those draws. Native candidate generation on the
  revision 7 actor feed offered the fixed exchange of HP 17, MP 3 and money 43.
  The Simulator accepted it at revision 8. Both read-back Journals reconciled
  with their Replay projections. [Recorded actions](../issues/cases/0015-concealed-dream-draw.json)
- **PASS** Python and the installed Native wheel matched on mixed and
  all-concealed intervals and at the 128-event boundary. The tests check raw
  event count and prefix hashes, invalid marker placement, and unknown IDs.
  Prepared arrays round-tripped with empty and nonempty filtered intervals.
  Original-backed reads rejected shortened arrays and changed category values.
- **PASS** The marker-free saved token and checkpoint fixtures remained
  compatible. The previous wheel was rejected by current HPO6 preflight.
  The earlier Dream smoke `aa9b754d-ba86-4bcb-93d2-fdc2b12a5037` passed the
  independent full audit using its recorded producer identity.

The final 1,763-pass, 26-skip focused Python run used the wheel from the new
smoke. Older pinned reproduction cases and unavailable optional environments
were among the skips. The separate packed Dataset owner suite passed 10 tests.
Set `GFB_P65_TEST_SOURCE` to `runs/dream-compat/simulator`, `GFB_P65_TEST_SERVER`
to its release `gf-test-server`, and `GFB_HTTP_DIAGNOSTICS_TEST_CONFIG` to
`configs/test-server.toml` to include the pinned reproduction in this command.

```sh
run=runs/concealed-dream-draw.2yyZLu
"$run/venv-final/bin/python" -m pytest -q -n0 \
  tests/test_hpo_dream_discard_reproduction.py tests/test_concealed_dream_draw.py \
  tests/test_structured_history_compat.py tests/test_structured_history.py \
  tests/test_structured_tokens.py tests/test_card_instance_contract.py \
  tests/test_card_instance_tokens.py tests/test_card_instance_structured.py \
  tests/test_card_instance_native_fixture_tokens.py tests/test_guardian_acted_union.py \
  tests/test_guardian_event_namespace.py tests/test_native_card_instance_capabilities.py \
  tests/test_rust_player_search.py tests/test_rust_player_batch_inference.py \
  tests/test_hpo_owned_numeric.py tests/test_rust_player_operation.py \
  tests/test_rust_player_campaign_factory.py tests/test_card_instance_native_provider_guards.py \
  tests/test_native_group_pack_bridge.py tests/test_native_group_pack_review_fix1.py \
  tests/test_card_instance_final_validation_cli.py tests/test_card_instance_game_tree_cli.py
"$run/venv-final/bin/python" -m pytest -q -n0 tests/test_group_pack_dataset_owner.py
```

**PASS** `cargo test --offline --locked --manifest-path native/Cargo.toml --lib --quiet`
with `PYO3_PYTHON` selecting CPython 3.14 and `RUSTFLAGS` and `LD_LIBRARY_PATH`
selecting its shared library. It passed 221 tests with 13 ignored. The private
Rust Act acknowledgement and live serving/learner row tests passed.
`ruff check src tests`, `mypy src` on 288 files, and `git diff --check` passed.
The full repository pytest run was **NOT RUN**.

## Eight-game HPO6 smoke

**PASS** Fresh run `31edf132-841e-4c26-945d-fec91795c902`, learning seed 17,
completed eight natural games with 149 accepted actions. It committed one PPO
update with 77 learner decisions and eight optimizer steps, then paused at the
requested update boundary. The independent `--full-source-audit` matched the
original Journals, Replay and Python-regenerated numeric arrays. The run used
an RTX 5070 Ti with Torch `2.14.1+cu132`, eight concurrent games, tmpfs staging,
and the dedicated ext4 mount for durable output.

Collection took 4.79 seconds and the update took 1.68 seconds. The corresponding
earlier smoke recorded 4.64 and 1.73 seconds. These are single observations from
the acceptance workload. The pinned p65 replay provides the recurrence check.

**FAIL at startup** The first attempt used a `PYTHONPATH` wheel overlay. The
supervised `python -I` process ignored the overlay and loaded the previous
extension. No collection started. The retry used the wheel installed in its
isolated interpreter and a new run ID.

The successful invocation used `tools/rust_player_operation.py --mode fresh
--learning-seed 17 --pause-after-update 1` under the process observer. The
post-run command was `tools/rust_player_verify_operation.py --lane native-hpo
--learning-seed 17 --full-source-audit`. The exact arguments, configuration,
IDs and outputs are in `runs/concealed-dream-draw.2yyZLu/retry`. The full audit
SHA-256 is `296bbbd15936aaff37f5f2a08a22d0bc70f3e76520afa05384b3bd0bfb7fabf3`.
