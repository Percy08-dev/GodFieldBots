# GodFieldBots

公開された本人の観測・可視イベント・typed Action metadataから行動を選ぶPython Botです。
Phase 0〜3のRandom／heuristic、候補生成、TestServer対戦、本人別journal、固定評価を実装しています。
LightGBM／PPOのモデルと学習は後続Phaseです。依存のCPUスモーク成功とBotの学習成立は別に判定します。
現在の検証結果・Gate・制約は [ActionOptions検証記録](docs/verification/action-options.md)、
過去の基盤検証は [初期記録](docs/verification/phase0-3.md) に記録します。

Python **3.14（GIL有効）** とuvを使用します。ゲームルールと最終合法性は外部の
`../GodFieldSimurator`が所有します。Simulatorをビルド・変更する機能は含みません。

```console
uv sync
uv run gf-bots validate --config configs/experiments/smoke.toml
uv run gf-bots smoke --config configs/experiments/smoke.toml
```

既定の配置では、事前ビルドされた `../GodFieldSimurator/target/debug/gf-test-server` を使います。
Simulator契約はrevision `44a26e3a16b808e826deb332fa72acfc5037d3ef`、HTTP `/v1`、
`true-match-1v1-v1`、Observation `"1"`、Action metadata `1`、catalog `public-ja-v1` に固定します。
実行にはlocalhostのlistenerを起動できる環境が必要です。

## 設定と実行

- [experiments](configs/experiments): 設定の選択、seed、split、予算、保存先、接続。
- [actions](configs/actions): enumerate／beam、固定scorer、長さ・候補数・展開数の上限。
- [models](configs/models): Random／heuristicのIDと版。
- [test-server.toml](configs/test-server.toml): Simulatorの実効ライフサイクル設定。

参照・設定内のパスは、そのファイルからの相対パスです。テーブルは項目単位で結合し、配列は置換します。
参照preset→実験設定→明示CLI上書きの順に解決し、未知のキー・不正値・未実装モデルを実行前に拒否します。
学習設定は対応adapterを実装するまで拒否します。
候補生成は既定で`use_action_options = true`です。同じrevisionの本人向けActionOptionsを取得し、
通常使用・防御・取引・基本行動を目的別に絞ります。取得・検証に失敗した場合の自動fallbackはありません。
観測のみの比較は`--set actions.use_action_options=false`で明示し、別の実験条件として記録します。

```console
uv run gf-bots evaluate --config configs/experiments/baseline.toml
uv run gf-bots collect --config configs/experiments/smoke.toml
uv run gf-bots benchmark --config configs/experiments/smoke.toml
uv run gf-bots validate --config configs/experiments/smoke.toml --set actions.strategy='"enumerate"'
```

`collect`、`evaluate`、`benchmark`は同じ固定スケジュール・対戦・保存経路を使用します。
`report.json`には対戦成績と計算コストを保存します。`godfieldbots`も同じCLIへ接続します。
2ゲーム並列、2時間上限、3つの方策乱数seedなどは初期presetであり、設定で変更できます。
smokeは1つのGameSeedと方策seedで先後を交換した2試合、最大180秒です。

`server.source_dir`を指定した場合は起動前にSimulatorのGit revisionを確認します。
単体バイナリや外部URLでは省略でき、宣言した契約revisionと未検証のbuild由来を区別して記録します。
外部URLでは `server.external_url`、明示的な固定seed実験設定、期待fingerprintが必要です。
外部サーバーを停止することはありません。

## 保存・データ変換

実行ごとに保存先へ一意のディレクトリを作り、以下を保存します。

- `manifest.json`: 実効Config・入力hash・コード／Simulator由来・管理情報・run状態。
- `report.json`: 自然終了の勝分敗、失敗の内訳、seed単位の区間、探索数・時間など。
- 試合ディレクトリ: 各本人の`player-N.jsonl`、Player Replay、結果。
- サーバーディレクトリ: stdout／stderr、HTTPログ、Simulatorが生成した永続化記録。

方策は固定引数・限定環境の専用bootstrapから参加者ごとに起動し、モデルをimportする前から
管理情報を除外します。本人の入力と可視定義だけを渡します。
同じrevision・本人向けのActionOptionsを、使用可能な要素のatomic hintsとして方策へ渡します。
カード情報は観測上のperceived identityを使い、手札と習得済み奇跡の参照を分けます。
GameSeed・管理情報・他者の私有情報・Replay・SequenceOptions・拒否詳細は方策へ渡しません。
拒否後に別候補を試して隠れた合法性を調べる処理もありません。

Parquet変換だけがPyArrowを必要とします。

```console
uv sync --extra data
uv run --extra data gf-bots export path/to/player-1.jsonl path/to/player-2.jsonl --output dataset.parquet
```

schema 4の変換には、元のrunディレクトリの確定済み`manifest.json`も必要です。
journalだけを切り離した入力、同じseed／分岐元に異なるsplitを宣言した入力は拒否します。
変換は元journalの入力・候補順序・選択を保持し、版と入力hashを保存します。
run／Game IDとsplitを保持し、seedや管理情報は診断sidecarへ分離します。
失敗runは自然終了の結果を保持しながら学習用として不適格にします。
異なるschemaを現在の意味へ読み替えません。打ち切り・拒否・通信障害に勝敗や推測の最終報酬を付けません。
Simulatorの`turn_limit_exceeded`は、生のAborted応答を保持したまま非終端truncationとして記録します。
終了理由は診断sidecarへ分離し、両Player Replayの完全な終了情報を照合します。
実際に渡したActionOptionsとそのmode／入力契約versionも保存します。
旧schema 1／2／3は自動変換せず、現在のreaderでは明示的に拒否します。

終了処理とレポート用の時間もConfigで予約し、期限切れの比較を成功扱いにしません。
OSの強制回収やローカル保存の停止不能な待ちにはhard real-time保証がありません。
検出した超過は不成立として記録します。制約と再検討条件は[architecture](docs/architecture.md)を参照してください。

## 候補生成の範囲

`public-atomic-subset-v3`は観測・許可されたatomic hintsから構成する候補空間です。
全エンジン合法手の列挙保証はありません。
正の攻撃値の武器・攻撃奇跡・追加攻撃、Aura、数値防御、空Response、非武器のdiscard／sacrifice、
公開tradeと設定した資源配分、祈りの保守的な公開条件、buy decisionを扱います。
順序と同種カードの別個体を保持し、習得済み奇跡は手札と別の参照・MP消費として扱います。

反射・弾き・指輪、精霊の消費軽減、Mirage、雑貨の単独効果、攻撃値のない状態変化奇跡などは
初期候補の対象外です。受ける攻撃の情報が不足する場合の特殊防御も対象外です。
数値防御もActionOptionsの使用可能一覧で絞るため、相手の定義を追加カタログ参照したり、
攻撃属性の相性をBot側で推測したりする必要はありません。組み合わせ・順序・資源配分の
最終検証は引き続きSimulatorが行います。通常のdiscardでは公開された破棄禁止ルールも適用します。
`can_discard`などの全体フラグや通常使用一覧を、個別の破棄対象／売却対象一覧とは解釈しません。
使用可能なマジカルステッキは、固定の攻撃値がなくても単独候補を構成します。未検証のMP効果や
組み合わせは推測しません。購入承諾を要求しながら`actionable=false`となる既存サーバーの問題は、
[別タスク](docs/issues/0003-buy-choice-control-metadata.md)として記録しています。
詳細は [actions.py](src/gf_common/actions.py) の版付きcoverageを参照してください。
探索予算切れの理由・公開枝刈り・beamによる近似除外を別記録し、候補空を祈りで埋めません。

## 開発・検証

```console
uv run pytest
uv run ruff check .
uv run mypy src
uv build
```

実バイナリintegrationは通常テストから分離し、実行条件は検証記録に記載します。
[追加テストとカバレッジの記録](docs/verification/coverage-followup.md)には、異常系・並行実行・
独立した候補集合の期待値・CLI入口の検証と、計測範囲の制約を記載します。
`data`、`lightgbm`、`torch`は用途別extrasで、baselineの起動時にMLライブラリを読み込みません。
Linux／WindowsのTorchはCPU用indexを選びます。GPUは未検証です。

設計と開発Agentの規則は [architecture](docs/architecture.md)、[ADR](docs/adr/README.md)、
[AGENTS.md](AGENTS.md) にあります。Agent設定は `.codex/`、設定読込確認は
`codex --strict-config app-server --print-startup-config` です。実装・独立レビュー・設計適合確認を
専任Agentへ委譲し、レビューと必要な検証を揃えてから完成と判定します。
