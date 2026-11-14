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
