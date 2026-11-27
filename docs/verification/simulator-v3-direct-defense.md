# Simulator v3 の防御候補と霧イベントの検証

## 固定版

| 項目 | 値 |
| --- | --- |
| Simulator | `997a0de59a456dd84699605c178f12ec74addf84` |
| Action profile | `card-instance-simulator-v3-shared-20261127-defense` |
| TestServer SHA-256 | `808f0641bb6467997bb20a952777aee0c3253629f366159ae7def95e710ff6e6` |
| 仕様インベントリ SHA-256 | `0cc29438a27f3980c93a6bbbf12fd1be762706199437bf3573ba107cf17b9442` |
| 公開契約 | `/v3`、`true-match-1v1-v3`、公開イベント schema 3、`public-ja-v4` |
| Native wheel SHA-256 | `853455f87e89217927d58accf9df88257044d1dd24bd7034e39e1cd23a24bc8a` |
| Native 拡張 SHA-256 | `7d50e0daaf27284e7275ba87e4cd149586e8cd8a1ad8e561d7ef6ad0c86d6022` |

Simulator の配布元は `.tmp/defense-v3-release`。Bots の実行設定はソースを
`runs/defense-v3-compat/simulator`、バイナリを同ディレクトリの
`target/release/gf-test-server` で参照する。再構成時は配布 bundle の
`codex/defense-v3` と配布バイナリを使い、[固定版のハッシュ](../../src/gf_common/contracts/simulator-defense-v3.json)
を照合する。 [ADR0079](../adr/0079-simulator-v3-direct-defense.md)

## 確認した動作

- **PASS** 既定 TOML を読み込み、実行用設定の生成とハッシュ付き再読み込みを実施した。
  256 試合・1,000 更新の予算で、TestServer とソースインベントリの実行前照合を通過した。
  過去の Sacrifice TOML は `mlp-256-convergence-1000-sacrifice.toml` として参照できる。
- **PASS** Python の実クライアントで [p46](../issues/0019-hpo-defense-costcut-role.md) の
  受理済み 32 手を再生した。候補表は見送りと `[乱気流, 精霊の頭巾]` の 2 件。
  正順の応答は消費 MP 0 で revision 33 に進んだ。別の対局で、霧中の受け手向けイベント、
  Replay と直接防御候補を照合した。ログ用の GameSeed や拒否情報は候補生成に渡していない。
- **PASS** `python -m pytest -q -n0 tests/test_rust_player_config.py` は 87 件通過した。
  `python tools/model_docs.py check` は 19 ページと 12 図を検証した。
- **PASS** 実クライアントで取得した p46 と霧の局面を、導入した Native wheel の
  `player_search_v5` に通した。候補表と検索後の乱数状態が Python と一致した。
  保存済み v2 p46 の候補表・乱数状態も旧 Native wheel と同じだった。
- **PASS** `python tools/build_rust_player_wheel.py` でビルドした wheel と supervisor を
  `.venv` に導入し、ロードされた拡張のソース一致を確認した。既定設定の実行前チェックは
  Native capability、TestServer、CUDA、保存先を照合し、ランを作成せずに完了した。
- **PASS** `cargo test --offline --locked --manifest-path native/Cargo.toml --all-targets`
  はローカルソケットと子プロセスを利用できる環境で319件通過し、28件が ignored。
  Ruff、mypy、`git diff --check` も通過した。
- **FAIL** `python -m pytest -q -p no:cacheprovider` の初回は 9,317 件通過、729 件 skip、
  73 件失敗。うち 67 件は変更前のソースと Native wheel を用いた隔離再試験でも同じ
  テストが失敗した。残り 6 件は旧テストのモックに契約引数を追加し、保存・Replay の
  既存アサーションを維持した。ラストランの全体ログと基準再試験ログは
  `runs/defense-v3-compat/python-tests.log` と `python-baseline-failures.log` にある。
- **PASS** 修正後に `test_rust_player_storage.py`、`test_card_instance_cli.py`、
  `test_hpo_owned_import_custody.py`、`test_simulator_defense_v3.py`、
  `test_rust_player_config.py` を `-n0` で実行し、149 件通過、21 件 skip。
  再試験ではローカル Simulator を使う旧 v2 の Journal 保存の確認も通った。
- **FAIL** `cargo fmt --all -- --check` は整形差分を報告した。変更前の Native
  ソースでも同じチェックに整形差分がある。

## 候補探索の局所計測

同じ p46 の actor frame を反復し、Python の候補生成を固定の探索クロックで測った。
1 回あたりの中央値は v2 が 2.176 ms、v3 が 2.162 ms。各版を 192 回ずつ交互に
実行し、初回を除いた 8 回を集計した。実際の候補数は v2 が 3 件、v3 が 2 件。
単一局面の測定で、収集・学習全体の性能を示す値とは分けて扱う。

同じ入出力で、JSON の取り込みと出力を含む Native `player_search_v5` は v2 が
1.501 ms、v3 が 1.495 ms。測定条件と各回の値は
`runs/defense-v3-compat/live-check` に記録した。

## CUDA HPO6 の収集・保存

**PASS** RTX 5070 Ti と CUDA Torch `2.14.1+cu132` で、seed 17 の
`249cfb05-7e6f-4d4d-88f8-0b412ba348d7` を8試合・1更新まで実行した。8試合が
自然終局し、受理済み149手のうち77件の学習判断を使った。収集3.95秒、更新と保存
1.80秒。PPO更新を確定し `paused` で終了した。`rust_player_verify_operation.py
--full-source-audit` で保存した Journal、Replay、Python 学習用数値の照合を通過した。
実行・監査記録は `runs/defense-v3-compat/smoke` にある。監査 JSON の SHA-256 は
`56351e0c6416702b1f3bfd3e8e14e66c5af1ecfbd9536bdd2ee4adbbd85e19ea`。
収束診断の全1,000更新は未実行。
