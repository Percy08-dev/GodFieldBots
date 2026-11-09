# GodFieldBots

公開された本人の観測・可視イベント・typed Action metadata・許可された本人向けActionOptionsから行動を選ぶPython Botです。
Phase 0〜3のRandom／heuristic、候補生成、TestServer対戦、本人別journal、固定評価を実装しています。
Phase 4のCPU LightGBM、特徴変換、checkpoint推論、学習CLIも実装しています。
実対戦学習に必要なclean pilotの判定は[購入契約更新・再検証記録](docs/verification/buy-contract-adoption.md)を参照してください。
CUDA MLP PPO・世代管理・境界再開・固定データ比較も実装しています。
全履歴GRU／Transformer PPOと、Config指定のTBPTTも実装しています。
両モデルのCUDA実対戦2更新・停止再開一致と測定結果は
[系列モデル検証記録](docs/verification/sequence-ppo.md)にあります。
設計は[ADR0008](docs/adr/0008-mlp-ppo-generations-and-fixed-data.md)、実装・実験の受入状況は
[PPO検証記録](docs/verification/mlp-ppo.md)を参照してください。
実対戦2更新・停止再開の完全一致は確認済みです。対象契約はSimulatorで明確化され、新契約での検証・正式比較を進めています。
[移行記録](docs/verification/catalog-v2-adoption.md)を参照してください。
合成データでの数値検証とBotの学習成立は別に判定します。
現在の検証結果・Gate・制約は [ActionOptions検証記録](docs/verification/action-options.md)、
[LightGBM検証記録](docs/verification/lightgbm-phase4.md)、過去の基盤検証は
[初期記録](docs/verification/phase0-3.md) に記録します。
モデル構造・最適化・学習制御の探索方法と適用条件は
[ハイパーパラメータ探索調査](docs/research/hyperparameter-search.md)に整理しています。
会話で提案された候補範囲拡張・エンコーダーと、その根拠・訂正・未決事項は
[独立した議論整理](docs/research/discussion-review.md)を参照してください。

Python **3.14（GIL有効）** とuvを使用します。ゲームルールと最終合法性は外部の
`../GodFieldSimurator`が所有します。Simulatorをビルド・変更する機能は含みません。

```console
uv sync
uv run gf-bots validate --config configs/experiments/smoke.toml
uv run gf-bots smoke --config configs/experiments/smoke.toml
```

既定の配置では、事前ビルドされた `../GodFieldSimurator/target/debug/gf-test-server` を使います。
Simulator契約はrevision `ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6`、HTTP `/v1`、
`true-match-1v1-v1`、Observation `"1"`、Action metadata `1`、catalog `public-ja-v2` に固定します。
実行にはlocalhostのlistenerを起動できる環境が必要です。

## カード個体モデルのサーバv2経路

新しい経路はSimulator `91d5926a7572db327eaea3751e7c24f428120996`、
Gameplay `/v2`、`true-match-1v1-v2`、catalog schema2／`public-ja-v3`を使います。
[専用pilot設定](configs/experiments/card-instance-pilot.toml)と専用保存先を使用し、
旧Config・記録・重みの意味を維持します。入力は本人の`own.cards`の順序・個体参照・状態と
可視履歴で、送信時にも元のopaque参照を保持します。Flashは1枚まで、奇跡の破棄は
unused／learned両方、utilityは自分・相手の対象を扱います。
受入結果と未実行の範囲は[移行検証記録](docs/verification/card-instance-v2-adoption.md)にあります。

新しいTree3の固定データ作成・検査・fresh fitには、
[専用設定](configs/experiments/card-instance-tree-fixed-data.toml)を一つ渡します。
`P17 P29 P43`は順に正式PPO親、`V17 V29 V43`は各親の完全な最終方策validation、
`SHA`は生成したmanifestファイルのSHA-256です。すべての原記録を検証し、
合計8GiBの資源管理にfit子プロセスの宣言上限も含めます。

```console
gf-bots game-dataset build --config configs/experiments/card-instance-tree-fixed-data.toml --parents P17 P29 P43 --validations V17 V29 V43 --output DATASET
gf-bots game-dataset inspect DATASET/manifest.json --config configs/experiments/card-instance-tree-fixed-data.toml --expected-digest SHA
gf-bots fit-game-tree --config configs/experiments/card-instance-tree-fixed-data.toml --manifest DATASET/manifest.json --expected-digest SHA --output FIT --seed 17
```

fitにはLightGBM extraが必要です。正式3seedの実行・実データfit・比較・HPO・最終testは後続です。
短いPPO受入runとcomponent fixtureは、これらの適格Sourceには昇格しません。
nativeの各capabilityは形式の実装対応を示し、個々のSource適格性は原記録から別途検証します。

## 設定と実行

- [experiments](configs/experiments): 設定の選択、seed、split、予算、保存先、接続。
- [actions](configs/actions): enumerate／beam、固定scorer、長さ・候補数・展開数の上限。
- [models](configs/models): Random／heuristic／LightGBM／MLP／GRU／TransformerのID、構造、encoderと版。
- [training](configs/training): 学習方式、実行device、データ参照、予算と乱数seed。
- [test-server.toml](configs/test-server.toml): Simulatorの実効ライフサイクル設定。

参照・設定内のパスは、そのファイルからの相対パスです。テーブルは項目単位で結合し、配列は置換します。
参照preset→実験設定→明示CLI上書きの順に解決し、未知のキー・不正値・未実装モデルを実行前に拒否します。
学習方式はモデルと対応した`lightgbm_mc`または`ppo`を選びます。
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

## LightGBM

各候補の内容を採点する`Q(o,a)`を、実際に受理された行動の自然終了returnで回帰します。
未実行候補には教師を作りません。`lightgbm-observation.toml`は現在観測、
`lightgbm-history.toml`は本人の可視履歴の集約を加える設定です。
学習・推論は同じ版付きencoderを使い、カードの参照IDの大小を特徴にしません。

```console
uv sync --extra data --extra lightgbm
uv run --extra data --extra lightgbm gf-bots validate --config configs/experiments/lightgbm.toml
uv run --extra data --extra lightgbm gf-bots train --config configs/experiments/lightgbm.toml
```

最後のコマンドは初期presetではエラーになります。完全なtrain／validation run、
64試合×3方策seedのclean pilot、事前に固定したaccepted-decision目標を設定する必要があります。
当初のpilotは[Simulatorの購入待ちmetadata問題](docs/issues/0003-buy-choice-control-metadata.md)を
含むため不適格です。成功した試合だけを抜き出して学習することはできません。
修正の取り込みと新しいpilotは[再検証記録](docs/verification/buy-contract-adoption.md)に残します。
LightGBMの`resume`は未対応として拒否します。

学習runは各seedにつき元の時間予算を持ち、データ収集の記録済みコスト、変換、fit、保存、
子プロセス回収を計上します。native fitとモデル読込は専用子プロセスで実行します。
checkpointの推論JSONはschema 2で、Simulator revision・特徴・候補生成・catalog・報酬などの互換情報を保存し、
学習元のseedや接続情報は別の`training.json`に残します。
正式な対戦には完了した`training.json`と`checkpoint/`を含むrun bundleが必要です。
合成データのモデルは`purpose=toy`となり、通常の対戦コマンドは起動前に拒否します。
schema 1の旧モデルや異なるSimulator版のモデルは自動移行せず拒否します。

## MLP PPO・世代管理

Torchは用途別extraです。`torch`はCPU版、`torch-cuda`はCUDA 13.2版で、同時指定は拒否します。
通常のCLIとbaselineはTorchを必要としません。GPU学習でも対戦中の推論はCPUで行います。

```console
env -u VIRTUAL_ENV uv sync --extra native --extra torch-cuda --extra data --extra lightgbm
env -u VIRTUAL_ENV uv run --no-sync gf-bots train --config configs/experiments/ppo.toml --mode fresh --run-dir runs/ppo/seed17 --pause-after-update 2
env -u VIRTUAL_ENV uv run --no-sync gf-bots train --config configs/experiments/ppo.toml --mode resume --run-dir runs/ppo/seed17
```

CPU実行は`uv sync --extra torch`と[ppo-cpu.toml](configs/experiments/ppo-cpu.toml)で明示します。
CUDAが利用できない場合は収集前に失敗します。OOM時のCPU切替やbatch変更はありません。
actor／criticは128幅2層のMLPで、学習・評価ともcategorical samplingを使います。
criticは候補一覧から独立した本人の公開特徴だけを使います。
1 batchは16試合で、Random 4・heuristic 4・凍結snapshot 8です。
学習席を明示し、両視点を保存して学習側の受理行動だけを更新に使います。

`--pause-after-update`は累積更新数による明示停止です。正常なbatch境界で`paused`になったrunのみ再開できます。
元の実時間期限は停止中も進みます。期限切れ、コード・実行環境・CPU／CUDA設定の変更、二重所有を拒否します。
世代0・10世代ごと・最新の重みと再開状態を保持し、全世代のhash・来歴を残します。
run全体が失敗した場合は、完了済みbatchを含めoffline学習に使えません。

## GRU／Transformer PPO

1ステップは本人の意思決定1回です。推論はゲーム開始からの履歴をGRU hidden／Transformer KV
に引き継ぎ、学習は現在の重みでprefixを再計算してから、最後の`training.sequence_length`回に
勾配を流します。既定値は32、指定範囲は1〜1,024です。窓長は推論履歴を切り詰める設定ではありません。
最大履歴1,024回を超えた場合は、行動送信前に親run全体を失敗にします。

```console
env -u VIRTUAL_ENV uv run --no-sync gf-bots train --config configs/experiments/gru-ppo.toml --set training.sequence_length=64 --set training.max_updates=2 --run-dir runs/gru-smoke-new --mode fresh
env -u VIRTUAL_ENV uv run --no-sync gf-bots train --config configs/experiments/transformer-ppo.toml --set training.sequence_length=32 --set training.max_updates=2 --run-dir runs/transformer-smoke-new --mode fresh
```

CUDA学習・CPU推論・2GiB/16MiBの型付きRust前処理を使います。明示的なCPU学習には
`gru-ppo-cpu.toml`／`transformer-ppo-cpu.toml`を選びます。依存の導入は既存のCUDA/CPU extra手順と同じです。
停止再開はMLPと同じ正常batch境界に限り、窓長・コード・実行環境を変えた通常resumeは拒否します。
今回の実装受入は2更新と資源測定までで、正式な多seed比較・最終testの結果ではありません。

## 固定データでのLightGBM比較

事前指定したPPO seed 17／29／43を順番に正常終了させてから、最終方策のvalidationを収集します。
validationは各64試合で、既存32 seedのうち前半16ペアをRandom、後半16ペアをheuristicへ割り当てます。

```console
env -u VIRTUAL_ENV uv run --no-sync gf-bots validate-policy --config configs/experiments/ppo.toml --parent-run runs/ppo-comparison-v2/train-seed17 --output runs/ppo-comparison-v2/validation17
env -u VIRTUAL_ENV uv run --no-sync gf-bots dataset build --ppo-runs runs/ppo-comparison-v2/train-seed17 runs/ppo-comparison-v2/train-seed29 runs/ppo-comparison-v2/train-seed43 --validation-runs runs/ppo-comparison-v2/validation17 runs/ppo-comparison-v2/validation29 runs/ppo-comparison-v2/validation43 --output runs/ppo-comparison-v2/dataset.json --preprocessing-config configs/preprocessing/native.toml
env -u VIRTUAL_ENV uv run --no-sync gf-bots dataset inspect runs/ppo-comparison-v2/dataset.json --preprocessing-config configs/preprocessing/native.toml
env -u VIRTUAL_ENV uv run --no-sync gf-bots dataset index runs/ppo-comparison-v2/dataset.json --output runs/ppo-comparison-v2/index.parquet --preprocessing-config configs/preprocessing/native.toml
env -u VIRTUAL_ENV uv run --no-sync gf-bots train --config configs/experiments/lightgbm-fixed-observation.toml
env -u VIRTUAL_ENV uv run --no-sync gf-bots train --config configs/experiments/lightgbm-fixed-history.toml
```

validation収集は29／43についても、それぞれ対応する学習seedの設定で実行します。
全世代の学習側train軌跡を成績で選別せず採用し、教師は受理済み行動と本人視点の自然終了returnです。
PPOのvalue／advantageをLightGBMの教師にしません。
datasetの作成・利用時は元journal・Replay・親runを再検証します。

PPOと固定データLightGBMのpresetは共通のRust前処理を使います。
Rustがjournal・Replay・来歴・特徴変換を処理し、Pythonが報酬・教師・GAE・学習を担当します。
JSONの解析・値の保持には`serde_json`を使い、キー索引やソートも標準実装へ統一しています。
`native` extraは独立したMaturin packageです。ソースからの導入にはRust toolchainが必要です。
baselineには追加依存は不要です。

```sh
env -u VIRTUAL_ENV uv sync --extra native --extra data --extra lightgbm --extra torch-cuda
```

前処理は予算内の試合群を繰り返し検証・変換・解放し、メモリと一時領域を管理します。
処理単位あたりの数値データ受け渡しは1回のFFI呼び出しで、NumPyがRustの配列を共有します。
生存中のviewも予算に計上します。Rust内部のコピーやファイル書き出しは別に残ります。
JSONはファイル／文書単位の見積予約（入力の32倍＋64KiB、journalの各行で共有）、
数値配列は所有バッファの容量で管理します。予約量と実際のRSSは別の指標です。
標準設定はメモリ8GiB・処理単位64MiBです。固定83,783行で検証した
2GiB・16MiBの設定もあります。これらは前処理の管理上限であり、OSのhard quotaや
LightGBM native fitのメモリ上限ではありません。

`preprocessing.workers`と`preprocessing.max_in_flight`は1・2・4・8を選べます。
並列数を増やす場合は両方を設定します。既定値は1です。並列になるのは各試合の
journal／Replay読込・デコードで、後段の検証・特徴変換・解放は順番に処理します。
同じ2GiB予算での実測は1／2／4／8 workerで49.21／36.85／32.32／31.18秒、
RSSは406／469／580／749MiBでした。設定の判断には
[内訳・資源使用量・比較条件](docs/verification/wall-attribution-and-worker-scaling.md)を参照してください。

アロケータ・型付きJournalの隔離比較は
[実験結果](docs/verification/staged-preprocessing-optimization.md)にあります。
実験CLIは`benchmarks/native_preprocessing/runner.py`で、既定はdry-runです。
本番reader・アロケータの既定値には、この実験の候補をまだ採用していません。

mimalloc＋借用型Journalの実経路も
[組合せ検証](docs/verification/combined-borrowed-journal.md)で確認済みです。
利用には`allocator-mimalloc`付きnative wheelと、明示的な
`configs/preprocessing/native-borrowed.toml`（または`preprocessing.reader`設定）が必要です。
既存wheel・reader既定値・main環境は自動で切り替わりません。

```sh
env -u VIRTUAL_ENV uv run --no-sync gf-bots dataset inspect runs/ppo-comparison-v2/dataset.json --preprocessing-config configs/preprocessing/native-low-memory.toml
env -u VIRTUAL_ENV uv run --no-sync gf-bots train --config configs/experiments/lightgbm-fixed-history.toml --set preprocessing.memory_budget_mib=2048 --set preprocessing.chunk_target_mib=16
env -u VIRTUAL_ENV uv run --no-sync gf-bots train --config configs/experiments/lightgbm-fixed-history.toml --set preprocessing.workers=4 --set preprocessing.max_in_flight=4
```

実験presetからは `[references] preprocessing="../preprocessing/native-low-memory.toml"`
で参照できます。Python比較実装は学習CLIの`--set preprocessing.backend='"python"'`、
dataset CLIの`configs/preprocessing/low-memory.toml`で明示できます。
dataset CLIは設定を省略すると互換用のPython設定を使います。Rust失敗時の自動fallbackはありません。
standalone datasetの`--address-space-limit-mib`は任意の`RLIMIT_AS`設定です。
RSSではなく仮想アドレス空間を制限し、学習・CUDA子プロセスには適用しません。
`dataset index --output`は出力と`.json` sidecarの両方が未作成の
パスを要求します。再生成時は新しい出力先を指定してください。
設定・互換性は[ADR0013](docs/adr/0013-standard-json-limits-and-sort-contracts.md)、
直近の整理・ソート契約は[JSON簡略化記録](docs/verification/json-simplification-and-sorts.md)、
Serde導入時の時間・CPU・メモリ・ストレージ比較は[移行記録](docs/verification/serde-json-adoption.md)
を参照してください。
新旧のnativeバイナリが混在する場合は事前検証で拒否します。更新後はnative packageを
再ビルドしてください。標準JSONとUnicode scalarを受け付け、非標準の`NaN`／`Infinity`や
単独surrogateなどの扱いの変更は移行記録に明記しています。バイナリ変更後の通常resumeは拒否します。
JSONの再帰制限はSerde標準に従います。旧`native_json_depth`設定とJSON全体の4300桁制限は廃止し、
数値を使う項目の型・範囲は引き続き検証します。メモリ・ファイル・レコードの予算設定は維持します。
固定データ方式では検証・特徴変換・fit・保存・回収を各fitの2時間へ含め、共有データ生成費を別途報告します。
従来の生成費込み方式と記録schemaで区別し、PPOと総費用が同条件だったとは扱いません。
最終test評価はこの段階では実行しません。

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
組み合わせは推測しません。購入承諾待ちは新しい契約に従い、購入者本人の
`resolving / buy_accept_reject`だけを入力可能として扱います。
[旧サーバーの問題と修正確認](docs/issues/0003-buy-choice-control-metadata.md)も保持しています。
詳細は [actions.py](src/gf_common/actions.py) の版付きcoverageを参照してください。
探索予算切れの理由・公開枝刈り・beamによる近似除外を別記録し、候補空を祈りで埋めません。

## 開発・検証

```console
uv run pytest
uv run ruff check .
uv run mypy src
uv build
```

CPU先読み起動の安全性テストは、軽い16件を通常実行に残し、実プロセスの異常系や
モデル一致・学習を確認する38件は明示実行にしています。全54件を実行する場合:

```console
GFB_RUN_CPU_PRELOAD_SAFETY=1 uv run pytest tests/test_cpu_preload.py tests/test_cpu_preload_review.py
```

起動・通信・終了処理、モデルの初期化やRNG、Torchを変更した際の確認に使用します。
通常のPPOテストで使うCPU先読み起動の最適化は引き続き有効です。

実バイナリintegrationは通常テストから分離し、実行条件は検証記録に記載します。
[追加テストとカバレッジの記録](docs/verification/coverage-followup.md)には、異常系・並行実行・
独立した候補集合の期待値・CLI入口の検証と、計測範囲の制約を記載します。
`data`、`lightgbm`、`torch`、`torch-cuda`は用途別extrasで、baselineの起動時にMLライブラリを読み込みません。
`torch`はCPU用、`torch-cuda`はcu132用の明示indexを選びます。
GPUの部品検証と実対戦学習の検証状況は[PPO記録](docs/verification/mlp-ppo.md)で分けて報告します。

設計と開発Agentの規則は [architecture](docs/architecture.md)、[ADR](docs/adr/README.md)、
[AGENTS.md](AGENTS.md) にあります。Agent設定は `.codex/`、設定読込確認は
`codex --strict-config app-server --print-startup-config` です。実装・独立レビュー・設計適合確認を
専任Agentへ委譲し、レビューと必要な検証を揃えてから完成と判定します。
