# PPO HPOの設定と実行

MLP・GRU・Transformerそれぞれのモデル構造とPPO設定を、同じ学習決定数で
比較する経路です。受入状況と実行した検査は
[検証記録](verification/ppo-hpo.md)、契約は
[ADR0037](adr/0037-ppo-hpo-foundation.md)にあります。
機能確認の結果から方策の強さや探索に十分な試行数を判断しません。

## 実行環境

Python3.14のGIL有効版を使います。学習とニューラル推論はCUDAが必須です。
Optunaは`hpo` extraの5.0.0だけを使用し、通常のCLIやbaselineでは読み込みません。
GPU Torchは`torch-cuda` extraです。CPU用`torch`と同時指定しません。
Rust前処理を指定する場合は、現行ソースに一致するnative版と追加HPO capabilityが
必要です。対応しないnative版からPythonへ自動で切り替わりません。
nativeの構築・reader・allocatorの条件は[README](../README.md)も参照してください。

Simulatorは設定に記録した修正版を事前に用意します。HPOはSimulatorを変更・
再ビルドしません。本人に見えるObservation・Events・許可されたActionOptionsの
入力境界、候補生成、自然終局報酬とγ=1は固定です。

## Campaign TOML

モデル種別・探索段階ごとに別campaignを作ります。TOMLの最上位に次を指定します。
実験用の探索範囲・試行数・seed群・学習量には既定値がありません。

| 項目 | 値と意味 |
| --- | --- |
| `schema_version` / `kind` | `1` / `"ppo-hpo"` |
| `base_experiment` | 同じモデル種別の修正済みmodel3・train実験TOMLへのパス |
| `model_id` | `"mlp"`、`"gru"`、`"transformer"`のいずれか |
| `stage` | 探索段階を示す名前 |
| `trial_count` | 探索空間内で提案する件数。重複・失敗も数える |
| `decision_budget` | 各学習seedの損失に使う一意な意思決定数D |
| `learning_seeds` | 重複のない学習seed配列 |
| `evaluation_game_seeds` | 共通評価GameSeedのcanonical UUID文字列配列 |
| `sampler_seed` / `startup_trials` | 提案seedと初期ランダム探索の件数 |
| `session_seconds` | 正常な学習run・評価の完了境界で停止するセッション予算 |
| `run_wall_seconds` | 各runの元の実時間予算。再開時も延長しない |
| `space.parameters` / `space.coupled` | 下記の探索項目・条件付き組合せの配列 |

相対パスはそのcampaign TOMLのディレクトリを基準に解決します。
`[model]`、`[training]`、`[inference]`で探索前の設定を上書きできます。
探索に使わない項目はこの設定で固定されます。MLPのTBPTT長など、動作に影響しない
項目は実効設定の識別前に正規化します。

`[[space.parameters]]`には`path`と`kind`、分布に対応する項目を指定します。
`path`は`model.hand.kind`、`training.adam_learning_rate`などのモデル・学習項目です。
運用上限やEventの最大記録数は探索項目にできません。

| `kind` | 必要な項目 |
| --- | --- |
| `categorical` | `choices`。層幅の配列なども一つの選択肢にできる |
| `int` | `low`、`high`、必要なら`step` |
| `float` | `low`、`high`、必要なら`step` |
| `log` | 正の`low`、`high` |
| `fixed` | `value` |

条件付き項目は`when`に「先に確定する設定パス → 有効にする値の配列」を指定します。
Attention幅とheadsなどは`[[space.coupled]]`の`name`、`paths`、`choices`で成立する
組合せを一緒に提案します。`when`も指定できます。未使用の条件付き項目を
別の実効設定として重複学習しません。収集試合数は8の倍数で指定します。
設定・分布の具体的な閉じた形式は
[config実装](../src/gf_cli/hpo_config.py)と
[探索設定](../src/gf_training/hpo_config.py)に定義されています。

既定モデルと既定PPO recipeの対照を必ず実行します。呼び出し側が指定した固定値で
対照を置き換えません。対照が探索範囲内なら1提案として数え、範囲外なら追加対照として
保存し、samplerの学習履歴には混ぜません。どちらも同じDと学習seed群を使います。

### 次回のPPO設定探索で固定するモデル

2026-11-15の合意では、MLP・GRU・Transformerとも主要部の幅192、主幹2層を
採用します。モデル構造自体のHPOは後段に回し、この段階ではPPOの学習設定を
探索します。既存の動作確認用campaignやモデルの既定値は変更しません。

新しい各モデルのcampaignには、次の固定overrideを設定します。
`model_id`と`base_experiment`はモデル種別ごとに指定します。

```toml
[model.state]
hidden_sizes = [192]

[model.hand]
hidden_sizes = [192]
latent_size = 192

[model.temporal]
hidden_size = 192
num_layers = 2
feedforward_size = 384

[model.heads]
actor_hidden_sizes = [192]
critic_hidden_sizes = [192]
```

MLPの主幹2層はframe projectionと追加の全結合層を数えます。GRUは2層、
Transformerは2ブロック・4heads・FFN幅384です。`feedforward_size`は
Transformerでのみ有効で、他のモデルでは既定値に正規化されます。
状態・手札encoderとActor/Criticの隠れ層数は各1段を維持し、カード埋め込み、
状態encoder出力幅、Event encoderなどは既定値を使います。
総パラメータ数はMLP約174万、GRU約215万、Transformer約230万です。

必須の対照は従来どおり既定モデルと既定PPO recipeを使います。上記の固定値で
対照を置き換えません。探索範囲・D・試行数・seed群・運用予算は別途設定し、
新しいcampaign・study・保存先を使います。ここでは学習や探索の実行を開始しません。

## 手動で行う学習量の予備検証

既定PPO設定で短い学習を実行し、実際の試合数・学習診断・最終方策の評価を
見て次の学習量を決めます。初回用は
`configs/hpo/budget-probe-{mlp,gru,transformer}-d10000.toml`です。
これは探索範囲や収束予算を確定する設定ではありません。

既知の実行失敗: 初回MLP実行では、既定対照と幅192設定の両方が13更新後、
「土石流」への「エンゼルアーマー＋虹のカーテン」という防御で拒否されました。
[調査記録](issues/0010-hpo-stopping-armor-response-contract.md)に公開入力・候補判定・
エラー伝達の問題を記録しています。所有者確認に従うrules-v2で候補判定と虹の先頭制約を
修正し、対応nativeもこの環境へ反映済みです。[検証記録](verification/defense-priority-owner-adoption.md)
は単体・native・全体検査とテスト側修正の結果を示し、実際のD10,000完了は未検証です。
旧署名のデータは互換対象にせず、元の失敗campaignは保持します。新しい保存先
`runs/ppo-budget-probe-v2`でfresh実行し、旧campaignをresumeしません。
公開入力で攻撃元を確定できない場合、条件付き特殊効果を持つ防具の併用は完成候補に
含めません。単独のatomic候補は残します。元エラーが管理エラーに覆われる問題は別件です。

現在の公開CLIには、Model4の試合数による停止、学習途中の定期評価、
完了した親のDを増やして延長する操作はありません。ここで実行するのは
**Dの異なる独立学習による学習量の比較**です。同じ親を1,000試合で評価し、
その重みから10,000試合へ継続する方式には追加実装が必要です。
通常の`train --pause-after-update`はModel3用で、このModel4設定には使えません。

| 設定 | 初回用の値と意味 |
| --- | --- |
| モデル | 幅192・主幹2層を固定 |
| `decision_budget` | 10,000。損失に使う一意なlearner意思決定数で、試合数ではない |
| `learning_seeds` | `[17]`。最初の学習挙動を観察する1seedで、seed間の安定性は未評価 |
| PPO | 既定recipe。epochs4、学習率0.0003、minibatch128など |
| `space.parameters` | `training.epochs=4`の固定提案。ランダムな設定探索はしない |
| `trial_count` | 1。ただし探索空間外の既定対照を追加するため、合計2設定を学習 |
| 評価 | 共通32GameSeed×席交換で、各設定64試合。既存heuristicに対する最終D方策の評価 |
| 収集 | mixed16、1更新16試合、同時2試合・参加者worker4個 |
| `run_wall_seconds` | 7,200秒。各run・評価の運用上限で、収束に十分な時間という意味ではない |
| `session_seconds` | 14,400秒。正常なrun・評価の完了境界で休止 |
| `operating.max_updates` | 1,000。D到達前に上限・期限へ達した親は未完了で、目的値を持たない |
| 保存 | モデル・D別の新規ext4保存先、容量上限100GiB。容量を事前確保する値ではない |

必須の既定対照が`proposals/000000`、幅192の固定設定が`proposals/000001`です。
既定対照はモデル構造も既定値を使います。PPO設定だけの効果を判断するときは、
幅192の設定同士を比較します。D10,000が1,000試合程度になるとは保証しません。
実際の累計試合数を測り、その結果から次のDを調整してください。

### 初回の実行

リポジトリのルートで、GPUを利用できる既存の`.venv`を使います。
以下はBashまたはZsh用です。`pilot_model`を`gru`または`transformer`に
変えると対応する設定を使えます。各モデルは順番に手動実行します。

```sh
cd /home/percy/develop/GodFieldBots
pilot_model=mlp
pilot_config="configs/hpo/budget-probe-${pilot_model}-d10000.toml"
pilot_root="$PWD/runs/ppo-budget-probe-v2/$pilot_model/d10000"
pilot_campaign="$pilot_root/parents/campaign-01"

.venv/bin/gf-bots hpo validate --config "$pilot_config"
mkdir -p "$pilot_root/logs"
set -o pipefail
.venv/bin/gf-bots hpo run \
  --config "$pilot_config" --mode fresh \
  --campaign-dir "$pilot_campaign" --progress plain \
  > "$pilot_root/logs/campaign-01-result.json" \
  2> >(tee "$pilot_root/logs/campaign-01-progress.log" >&2)
```

`fresh`のcampaignディレクトリは存在していない必要があります。
設定を変えずに別の新規実行を行う場合も、`campaign-02`など新しい名前と
ログ名を使います。GPUを利用できない環境ではvalidateとwatchだけが可能です。
validateは設定・資源見積り・保存先を検査し、GPU起動や実学習は検査しません。

別ターミナルでは、変数を引き継がず絶対パスで監視できます。

```sh
cd /home/percy/develop/GodFieldBots
.venv/bin/gf-bots hpo watch \
  --path "$PWD/runs/ppo-budget-probe-v2/mlp/d10000/parents/campaign-01"
```

`--progress plain`は進捗をstderrへ行形式で出し、上のコマンドは画面表示と
ログ保存を両立します。`auto`はstderrがTTYの場合だけ表示するため、
ログへリダイレクトするときは`plain`を明示します。watchはstdoutへ出力し、
既定1秒間隔で読み取ります。watchのCtrl-Cは監視だけを終了します。

表示項目はモデル、対照/提案、seed、処理フェーズ、commit済みD、更新数、
現在の収集/評価単位の暫定試合数、利用可能なら処理速度・ETA・total loss・
entropy・KLです。`collected-games`は累計学習試合数ではなく、収集batchや
評価が切り替わるとリセットされます。ETAは現runだけの予測です。
詳細なvalue MSEなどは更新記録、診断の追加項目は`watch --once --json`で確認します。
更新から5秒超のsnapshotはstaleと表示され、過去のcomplete表示を現在の稼働や
成功判定に使いません。`telemetry/progress.json`は最新値で、履歴全体ではありません。

### 結果と学習診断の確認

run終了後に結果JSONの`failed_in_session`が0であることを確認し、
inspectで両提案の`recorded_state`が`COMPLETE`になっていることを確認します。
campaignの`status=complete`だけでは、内部の提案失敗がないとは判断できません。
inspectは記録済み状態の構造表示で、独立した成績の再認定ではありません。

```sh
cat "$pilot_root/logs/campaign-01-result.json"
.venv/bin/gf-bots hpo inspect --campaign-dir "$pilot_campaign"
```

各提案の`evaluation.json`には最終評価の`value`、
`seeds/17/training.json`には親ディレクトリの`run_path`があります。
親の`updates/000001.json`以降に各更新の`actual_games`、`accepted_decisions`、
`metrics.total_loss`、`metrics.policy_loss`、`metrics.value_mse`、
`metrics.entropy`、`metrics.approx_kl`が残ります。自然終局した収集試合数は
`actual_games`を累積して数えます。評価試合数はこの累計には含めません。
最終batchは全試合を自然終了させるため、収集decision数がDを超える場合があります。

次の読み取り専用の例は、終了済みcampaignの更新記録をTSVへ出します。
これはグラフ化・観察用で、Sourceや評価成績を認定する処理ではありません。

```sh
.venv/bin/python - "$pilot_campaign" > "$pilot_root/logs/learning.tsv" <<'PY'
import json
import sys
from pathlib import Path

campaign = Path(sys.argv[1])
print("proposal\tseed\tupdate\tgames\tused_D\tloss\tpolicy_loss\tvalue_mse\tentropy\tKL")
for receipt_path in sorted(campaign.glob("proposals/*/seeds/*/training.json")):
    receipt = json.loads(receipt_path.read_text())
    parent = Path(receipt["run_path"])
    state = json.loads((parent / "run.json").read_text())
    games = used = 0
    for number in range(1, state["updates"] + 1):
        update = json.loads((parent / "updates" / f"{number:06}.json").read_text())
        games += update["actual_games"]
        used += update["accepted_decisions"]
        metrics = update["metrics"]
        row = [receipt_path.parents[2].name, receipt["training_seed"], number, games, used]
        row += [metrics.get(key) for key in
                ("total_loss", "policy_loss", "value_mse", "entropy", "approx_kl")]
        print(*row, sep="\t")
PY
```

lossの停滞だけで収束とは判断しません。次のDでの最終評価、entropy・KL・
value MSEの推移、実時間と保存量を併せて見ます。初回の1seed・64評価試合だけで
最適設定やseed間の安定性を判断せず、上位設定の確認ではseed群・評価量も見直します。

### 次の学習量とセッション再開

初回の結果を見て、例えばD30,000やD100,000を次の観測点にできます。
値を採用するかは実測後に決めます。元のTOMLを同じ`configs/hpo`内へコピーし、
次の項目を新しい段階用に変更してからvalidateします。

- `stage`と`decision_budget`。
- `[storage]`の`parent_root`、`numeric_root`、`optuna_path`、`originals_root`。
  `numeric_root`と`parent_root`は同じ値にします。
- 初回実測に応じたrunの時間上限・更新上限・保存容量と、新しいcampaign/log名。

モデル・PPO recipe・学習seed・共通評価GameSeedは、学習量だけを比較するときは
揃えます。新しいDは`--mode fresh`で最初から学習し、重みやoptimizer状態は
引き継ぎません。既存studyのTOMLやDを書き換えてresumeしてはいけません。

`session_seconds`で正常に休止し、結果が`status=paused`となったcampaignだけは、
同じTOML・同じcampaignディレクトリに`--mode resume`を指定して残りの処理を
継続できます。元のrun期限は延長されません。実行中の学習へCtrl-Cを送ることは
正常なセッション休止の代用ではなく、未完了操作は失敗として残ります。

```sh
.venv/bin/gf-bots hpo run \
  --config "$pilot_config" --mode resume \
  --campaign-dir "$pilot_campaign" --progress plain
```

## 常駐workerと進捗監視

schema3の機能確認用設定は`configs/hpo/resident-{mlp,gru,transformer}-smoke.toml`
です。幅192・主幹2層、mixed16、D64、学習seed17、固定提案1件と既定対照を
使います。探索量の推奨値・強さの評価ではありません。新規実験では保存先とDBを
新しくしてください。通常PPOと旧schema1/2はfresh方式を維持します。
保存用と親データ検査用の予算は各1GiBです。親データ検査では保存済みoptimizerの
archive検査にも一時メモリを予約するため、旧256MiB設定はこのモデルで不足します。
192/主幹2層の実機検証と測定範囲は[検証記録](verification/hpo-resident-workers.md)を
参照してください。短い機能確認の結果から、正式な探索量やモデルの強さは判断しません。

常駐方式は最大4物理workerをviewerごとに分離し、試合終了時に履歴・乱数・
GRU/KVなどをリセットします。更新中も実プロセスとVRAM予約を保持し、commit済み
checkpointを保護してから新しい重みを同期します。pauseでは実際に回収し、resume
で再構築します。対応するsource-matched native capabilityが必要です。

```console
gf-bots hpo validate --config configs/hpo/resident-mlp-smoke.toml
gf-bots hpo run --config configs/hpo/resident-mlp-smoke.toml --mode fresh --campaign-dir runs/hpo-resident-workers/functional/parents/campaign-mlp-01 --progress plain
gf-bots hpo watch --path runs/hpo-resident-workers/functional/parents/campaign-mlp-01
gf-bots hpo watch --path runs/hpo-resident-workers/functional/parents/campaign-mlp-01 --once --json
```

`--progress auto`が既定で、TTYのstderrに表示します。リダイレクト時のautoは
表示しません。`plain`は行形式、`off`は表示を止めますが、進捗ファイルの保存は
続きます。最終結果のstdout JSONは従来どおりです。compareも同じ表示設定を使えます。

watchは実行lockを取らず、Torch/Optuna/GPUを読み込まない読み取り専用の監視です。
campaignまたはcomparisonディレクトリを指定し、Ctrl-Cで監視だけを終了できます。
`telemetry/progress.json`は最大64KiBの最新観測であり、再開・Source資格・目的値・
終了確認の権威には使いません。欠落・不正・5秒超の更新遅延はそのまま表示します。

commit済み使用件数/D・更新数・loss/entropy/KLと、暫定試合数・両者の操作数は
区別します。ETAは現runの完了した2更新以降、直近最大3区間から計算し、pauseを
含めません。全campaignの残り時間は推測しません。D到達後も評価・回収を確認する
まで完了表示にしません。管理上の予約量、各プロセスの実測allocator値、未計測の
device全体ピークは別に扱います。進捗保存・表示の失敗は学習条件を変えません。

現在の実装受入範囲と実機検証の状態は
[常駐worker検証記録](verification/hpo-resident-workers.md)を参照してください。

## 資源と保存先

`[operating]`でRAM・CUDA・Source・数値配列・保存・収集の各予算、対戦並列数、
更新回数上限、保持数、Opponent Poolへの追加間隔を指定します。
初期の管理上限はRAM24GiB、CUDA12GiBで、同時trial数は1です。
共有encoder、実際のworker数、保持中のSource、更新・保存時の共存を共同で検査します。
管理上限はOSによるhard quotaではなく、使用量の観測とは区別します。

`[storage]`の`parent_root`・`numeric_root`・`optuna_path`はext4側です。
現行実装は`numeric_root == parent_root`の配置を要求します。
`originals_root`は圧縮対戦DBの保存先です。既定は対応済みDドライブ上で、
明示したext4保存先も対応します。
容量上限は`destination_capacity_bytes`、開始前の余裕は
`admission_reservation_bytes`で指定します。Windows側Cドライブも実際のmountと
空き容量を照合します。存在しないD/CパスをSSD上の祖先へ読み替えません。

各更新で推論・再開状態を完全に保存し検証してから保持処理を行います。
G0と最新2個以上の再開状態、Pool・評価が参照中の推論重みを保護します。
推論重みの最新保持数・使用決定数による周期保持は別の運用設定です。

## コマンド

環境の追加依存を整えた後、同じ環境の`gf-bots`を使います。
`python -m gf_cli hpo`も同じ入口です。

```console
gf-bots hpo validate --config campaign.toml
gf-bots hpo run --config campaign.toml --mode fresh --campaign-dir /absolute/parent-root/campaigns/stage-name
gf-bots hpo inspect --campaign-dir /absolute/parent-root/campaigns/stage-name
gf-bots hpo run --config campaign.toml --mode resume --campaign-dir /absolute/parent-root/campaigns/stage-name
```

`validate`は型・次元・探索条件・共同資源・実際の保存先を確認します。
OptunaやTorchを読み込まず、学習・Simulator起動・Source資格検査は行いません。
`inspect`も保存記録の構造と記録済み状態を表示する機能で、成績を再認定しません。
`fresh`には新しいディレクトリ、`resume`には同じ設定と保存先を渡します。

1trialは全学習seedの新規初期化runと、その最終Dチェックポイントの評価です。
最後の収集batchも自然終了まで保存し、残り件数だけを独立した乱数で選びます。
GAEは完全episodeから計算します。評価はGameSeedごとの席交換平均を取り、
さらに学習seedを等重みで平均します。途中の最高成績モデルを自動採用しません。
不足seed・欠落評価・OOM・輸送・保存・cleanup失敗は目的値を持ちません。
途中の成績は、完了した各学習seedの評価receiptに席交換した全自然終局結果と
検証済みprojectionを保存します。全seedが完了するまでtrial目的値にはしません。
世代ごとのlossなどの学習診断は別に記録します。評価頻度の追加、pruning、
小予算からの自動昇格は行いません。

新しい探索段階へは、過去提案の`settings-seed.json`を設定だけ引き継げます。
新しいcampaign・studyと現在の設定を用意し、`fresh`へ
`--seed-settings /old/proposals/000000/settings-seed.json`を追加します。
Adam/PPOの設定はrecipeとして引き継ぎますが、重み・optimizerのstate/moments・
試行成績・旧sampler履歴は引き継ぎません。
探索空間・D・評価条件を変えた実験を既存studyへ混在させません。

完了した提案同士の直接対戦は次で保存します。

```console
gf-bots hpo compare --config left-campaign.toml --left /left/proposals/000000 --right /right/proposals/000000 --output /absolute/parent-root/new-comparison
```

両側の全学習seed組合せについて、共通GameSeedで席を交換した表を保存します。
初期実装ではLEFT側の運用・公開入力・推論条件に合うconfigが必要です。
結果は探索用であり、確認用・最終test・formal/final64/Datasetの資格を与えません。

## メモリ保持の生成元経路（schema4）

新しいresident HPOは、生成したbatchをRAM上で保持し、そのまま学習へ渡す
`schema_version = 4`の明示的な経路です。設定は次を含みます。

```toml
schema_version = 4
origin = "producer-owned-v1"
encoder = "canonical-python"
```

`runtime`のresident設定も必要です。MLP・GRU・Transformerの設定例は
`configs/hpo/owned-mlp-smoke.toml`、`owned-gru-smoke.toml`、
`owned-transformer-smoke.toml`です。元のbaseがRust前処理でも、実行設定は
Python正規encoderを明示して記録します。

ジャーナル・アーカイブ・numeric・チェックポイントの保存を完了させ、
実際に受理されたcommandと両参加者の終端観測から学習batchを構成します。
通常更新では過去sampleの再生や生成物の意味的な再読込照合を行いません。
実行していないparityは`not_run`、Replay照合は`false`と記録します。
Replayの取得・保存は診断用に継続します。

Source2GiB・numeric512MiB・collector2GiBなど既存上限を維持します。
RAM不足時のディスクspoolや暗黙のreader切替はありません。
Model4・Journal6・inference5・resume5と配列layoutは維持し、旧schema3の
strict経路・データ・監査の意味も変更しません。

実装・受入状況と再開・campaign対応範囲は
[ADR0043](adr/0043-hpo-producer-owned-data.md)と
[検証記録](verification/hpo-producer-owned-data.md)を参照してください。
fresh統合・同一コードでのcold再開・campaign/evaluation連携は独立レビュー済みです。
再開時は必要なcontrol・checkpoint・RNG・opponent poolだけを一度読み込み、
次の更新で使わない過去sampleは再生しません。全プロジェクト検証・実GPUでの
新規/再開・campaign/evaluation・18試行の比較と全保存物のstrict監査を実施済みです。
両独立レビューを経て受入済みです。条件・測定範囲は検証記録に従います。短い比較から学習品質や最終testの
成績は判断しません。

## 停止と障害からの回復

セッション予算は実行中の試合やrunを中断しません。閉じた正常終了receiptがある
学習run・評価の境界で休止し、次のセッションで再検証して続けます。
元のrun期限や更新上限でD未達のrunは未完了です。
実行intentだけが残り完了receiptがない操作は、成功らしい親ファイルから復元せず
trial全体を失敗として残します。inner runtimeの正常な更新境界再開とは区別します。

全campaignで一つのGPU admissionを共有します。異常終了後、実際の回収を確認できない
jobが残ると`RECOVERY_REQUIRED`となり、次のGPU処理を拒否します。
flock解放・保存PID・経過時間だけでは回収済みと判断しません。
回復操作には次の二つがあります。

```console
gf-bots hpo recover --ticket TICKET --operator OPERATOR --confirm-environment-restarted
gf-bots hpo recover --ticket TICKET --operator OPERATOR --release-unconfirmed --reason "受付を再開する判断の理由"
```

前者は、実際に環境を再起動して全jobが終了したことを確認した場合に使います。
後者は、終了未確認の事実を残し、残存処理による資源競合の可能性を受け入れて
GPU受付を再開する操作です。再起動を必須にせず、対象ticket・操作者・理由を
別の記録に保存します。実行中の管理処理がlockを保持している間は使えません。
自動解除は行わず、失敗データの資格・評価結果・元の期限も変更しません。
どちらのコマンドも、プロセスの停止や再起動自体は行いません。
