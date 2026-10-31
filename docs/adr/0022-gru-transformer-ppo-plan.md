# 0022: GRU／Transformer PPOの実装計画

- 状態: 提案。独立した設計レビューと学習/正しさレビューはREADY。
  実装・学習・性能測定は未実施。実装指示後に段階Aから進める。
- 作成日: 2026-10-31。責任者: root。
- 対象: GRU PPO、Transformer PPO、および両者に必要な系列入力・保存・比較基盤。
- この文書の初期値は今回の提案。過去に承認済みの値と混同しない。

## 1. 過去の計画と現在地

[ADR0008](0008-mlp-ppo-generations-and-fixed-data.md) はMLP PPO、世代管理、
境界再開、正常終了した3親runからの固定データLightGBMまでを完成単位とし、
GRU／Transformerと最終testは後続にしていた。GRUの層数、Transformerの幅、
履歴長、系列PPOの方法は未決定だった。Phase0–3の未実装一覧や、旧seed17失敗時の
記録を、現在もMLP／LightGBMが未完成という意味で扱わない。

[catalog v2の記録](../verification/catalog-v2-adoption.md) ではMLPの3 seedが
123／144／145更新で正常終了し、LightGBMの2特徴条件×3 seedも完了している。
[ADR0021](0021-measured-single-worker-optimization.md) の採用版は前処理を改善した。
今回、その実装済み基盤を再利用する。保留したSchema比較デシリアライザ、Replay型付き化、
不採用のRecord容量予約は本計画の前提にしない。

現在の接続先はSimulator revision
`ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6`、catalog `public-ja-v2`。
再確認した仕様§21.1は `docs/spec/07-external-interfaces-adapters.md`、SHA-256
`342a8d59cc33494c9fbc91e565c7f710578cbf8a0fa59ee373fa0cea533d2d62`。
実装開始時にもHEAD・仕様・実バイナリhashを照合し、差異は別の採用判断にする。

## 2. 引き継ぐ条件

- Policyには本人のObservation、Player-visible Events、typed metadata、同一revisionの
  任意ActionOptions、版固定された公開カタログだけを渡す。過去の本人入力の保持は許可された
  情報の記憶に限定する。他者private、GameSeed、RNG、世代、相手モデルID、拒否詳細、
  SequenceOptions、未来の入力・終局結果は特徴にしない。
- 実対戦からのon-policy PPO。16試合/batch、8 seedペア先後交換、Random4／heuristic4／
  自モデルの凍結snapshot8。snapshotは世代0と10世代ごと、直近5個から選択。
  learner_viewerを明示し、同じpolicy IDでも役割を推測しない。
- 自然終了returnは本人視点+1/−1/0、中間0。γ=1、GAE λ=.95、clip=.2、value係数.5、
  entropy係数.01、Adam lr=3e-4/betas(.9,.999)/eps1e-8、4 epoch、論理minibatch128、
  gradient norm上限.5、全batchでadvantageを一度正規化。評価もcategorical sampling。
- CUDA学習子プロセス1個がTorch/GPUを所有。coordinatorはTorchをimportしない。
  推論は本人専用CPU worker。FP32・IEEE・決定的アルゴリズム、AMP/TF32なし。
  現在のTorch2.14.1+cu132／NumPy2.5.4を使用し、CPU presetを別に用意。
- 2時間/run、90分で新batch受付終了、119分で実処理終了、残り60秒を回収等へ。
  1000更新、所有ファイル10GiB、受付予約1GiB、全体同時対戦2ゲーム。
  失敗runの全データをofflineから排除する。期限を延長せず、OOM時に自動変更しない。
- 世代0・10世代ごと・最新の重み/再開状態、全世代の不変来歴、正常batch境界だけの
  paused再開、排他ロック、同じコード/環境/数値設定を要求する条件を維持。

## 3. 初版の系列契約

**直近32回の本人の意思決定時点の特徴を、現在時点を含めて入力する有限窓方式**を提案する。
全ゲームturnや全Eventを32個に切る意味ではない。元の各時点の特徴は、既存MLPの
public_aggregate/window128 encoderから候補に依存しない373次元を選択し、CPUで
signed-log1p→FP32へ変換する。幅はsignatureから検証し、373という数だけで互換としない。
新recipe IDを `own-decision-window-v1` とし、窓長と基底encoder signatureを保存する。

時点tの窓は同じepisode・同じviewerの `[max(0,t-31), …, t]` のみ。
教師return、advantage、old_value、選択確率は特徴に含めない。過去の観測を現在の
catalog解釈や現在の手札で再構成せず、保存された当時の入力から作る。
現在の候補列は保存済み順序・内容のまま利用し、690次元の既存actor特徴を再利用する。

推論policyはreset時に空にする固定長NumPy ringを持つ。各selectで現在のstate特徴を
1回追加し、窓全体を計算する。既存workerはepisode/participantごとに独立し、selectは逐次。
HTTP retryは同じcommandを再送し、selectを再実行しない。拒否・送達不確定・worker失敗では
その収集を失敗にするため、未受理の選択を次の成功軌跡へ持ち越さない。この条件を回帰テストする。
Policy.reset/select、DecisionInput、Environment.run_groupの外部契約は拡張不要。

GRUは**各窓の先頭でhidden=0**、Transformerは各窓で位置0から計算する。
学習可能なhidden/KV cacheを判断間で持ち越さない。したがって学習中に古い重みで生成した
hiddenを再利用せず、burn-inやepisode全体のTBPTTを初版に導入する必要がない。
これは有限窓GRUであり、無限に履歴を保持するGRUとの同等性は主張しない。
より長い履歴や状態/KV cacheは、初版の勝率・遅延・記憶課題の結果を見て別設計とする。

## 4. モデルの提案preset

| 項目 | GRU v1 | Transformer v1 |
| --- | --- | --- |
| 入力 | 同じ最大32 state tokens | 同左 |
| state投影 | Linear(state幅,128)+ReLU | 同左+固定sin/cos位置符号 |
| 系列層 | nn.GRU、1層、hidden128、単方向 | nn.TransformerEncoder、2層、幅128、4heads、FFN256、pre-norm、ReLU |
| dropout | 0 | 0 |
| context | 最後の有効位置の出力 | 最後の有効位置の出力 |
| actor | contextと現在の各候補特徴を連結、128-ReLU→1 logit | 同左 |
| critic | contextと現在stateを連結、128-ReLU→1 value | 同左 |

系列encoderはactor/criticで共有し、入力はstateのみ。候補内容・個数・順序を変えても
同じ重みのcritic値は不変でなければならない。候補順置換にはlogit列だけが追従する。
全窓をright paddingし、lengthsとpadding maskを渡す。GRUはpad後の最終hiddenではなく
最後の有効位置を選ぶ。Transformerはcausal maskとpadding maskを明示し、空窓を拒否する。
paddingの値を変えても有効出力が変わらず、未来tokenが過去位置へ伝わらないことを検証する。

CPU上でseed固定初期化。Linearは直交初期化、ReLU段gain√2、actor出力.01、value出力1、
bias0。GRUの各gate行列はgain1で個別に直交初期化。TransformerのQ/K/Vとattention出力は
gain1、LayerNorm weight1/bias0。複製されたencoder層も独立の乱数drawで再初期化する。
初期化recipeと全パラメータ順序を保存する。幅128だけで同じparameter数とは扱わず実数を報告する。

初版GRUのCUDA経路はcuDNNを明示的に無効化したPyTorch標準経路を使い、Transformerは
標準SDPAのMATH backendに固定する。Transformerのnested tensorとMHA fast pathは無効にして
推論/学習・CPU/CUDAの経路差を抑える。この設定もmanifestへ記録する。
CPU/GPUの1e-5照合が成立するかは実機で確認し、失敗時に許容値を緩めたり自動fallbackしない。
FlashAttention/cuDNN高速経路/compile/AMPは、必要なら個別の同出力比較を行う後続候補とする。

使用APIは既存依存に含まれる
[GRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html)、
[TransformerEncoder](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html)、
[SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。
maskの真偽の意味やkernel選択による数値差はAPIごとに検証する。
独自RNN、attention kernel、外部flash-attn、汎用trainerフレームワークは作らない。

## 5. 学習、保存形式、メモリ

1. Rustの既存全候補PPO経路から、検証済み特徴配列・episode_offsets・候補offsetを受け取る。
   現在のNativeSessionは完全episode単位を保持する。列をPythonオブジェクトへ展開し直さない。
2. Python学習層がstate変換、episode内の窓index、GAE、全batch advantage正規化を行う。
   現在のBulkRollout→RolloutDecisionではepisode境界が失われるため、系列経路はchunksと
   episode_offsetsを直接保持する。候補や履歴を再生成しない。
3. 新しい数値rollout **schema2** を閉じたJSON＋型付きNumPyで定義する。
   state token bank、decisionごとのwindow start/length、元のepisode/decision対応、候補offset、
   selected/old_logprob/old_value/advantage/targetとrecipe/hashを保存する。
   episode IDなどは照合用metadataでありネットワークへ渡さない。
4. 全データを32倍に複製しない。tokenはbankへ1回格納し、64MiB以下のshard境界には必要な
   同episodeの直前最大31 tokenだけをprefixとして複製する。prefixはloss対象にしない。
   windowは同episode内・現在位置まで・最大32、現在token/state一致を全件検証する。
   shardを跨いでも欠落/重複したlossや別episodeへの参照を認めない。
5. 毎更新前に全batchの窓を凍結CPU方策で再計算し、保存selected logprob/valueおよび
   CPU/GPU全有効候補logprob/valueをatol=rtol=1e-5で照合。全件PASS前にAdamを変更しない。
6. シャッフル単位は終点decision。各終点に完全な有限窓を付け、窓内の全32位置へ勾配を流す。
   PPO lossは終点にだけ加える。全episodeのGAEを学習窓で打ち切らない。
   windowとmaskが同じなら、終点を並べ替えても正しい有限窓PPOになる。
7. 論理minibatch128の中で、padding込みcandidate slots≤4096、state token slots≤4096、
   Transformerでは `layers × heads × B × T² ≤ 524288`、数値入力≤64MiBを満たす
   決定的microbatchを構成する。lossを終点件数で重み付けし、clip/Adam stepは論理minibatchに1回。
   1件でも上限に入らない入力は失敗。実行中のOOMを理由にサイズを自動変更しない。

既存のpreprocessing memory/cache/chunk/scratch Configを維持し、token bank・index・shard
prefix・padding一時配列を予約量へ含める。CPU推論ringの固定上限も設定から算出して記録する。
GPUのactivationは数値入力量とは別であり、allocated/reserved peakを測る。予算はsoft limit。
新たな独自allocatorや細粒度オブジェクト台帳は追加しない。

元のjournal/collection schema4は変更不要。既存MLP rollout schema1・特徴signature・
チェックポイントを読み替えず、旧経路でそのまま扱う。系列checkpointは専用version2の
inference/resume manifestを持ち、model ID、窓recipe、encoder、ネットワーク形状、数値backend、
全array key/shape/dtype/hash、Adam、乱数を閉じて検証する。unknown model/versionは拒否する。
世代/更新は既存のhash付き参照を使う。checkpoint loaderだけでなく、現在MLPに限定された
方策identity照合もPython参照/PPO/native source検証で同時に拡張する。
`mlp-ppo-v1-`／`gru-ppo-v1-`／`transformer-ppo-v1-` と対応するmodel ID・inference
manifest digestを閉じて結び付ける。prefixだけで通したり任意の文字列を許可しない。
特に `native/src/projection.rs` のsource_metadataとPython `ppo.py` の保存selection照合は
新モデルの正例、model ID/prefix/digest取り違えの負例、既存MLPの不変性を一緒にテストする。
`native_authorization.py` のbinding再構成、`lineage.owned_binding`、`fixed_data.py` の
最終方策validation用bindingもこの閉じたdispatchへ含める。ただし共有検証器の一般化で
旧dataset build/fitのsource許可を広げない。既存の固定データ契約はMLP親のみと明示的に制限し、
正常終了したGRU/Transformer親でも旧固定データ契約へ渡すと拒否される負例を追加する。
境界pause時には全ゲームが終了しているため、推論ring/hidden/KVの保存は不要。
途中のepisode再開は追加しない。MLP↔GRU↔Transformerの通常resume/重み変換は認めない。
旧形式の互換性は、コード/実行環境を跨いだ通常resumeの許可ではない。既存のidentity検査を維持し、
MLPの連続/停止再開回帰は同じ新コード/環境で作ったfresh対を使う。

## 6. 変更する責務と実装の順序

| 段階 | 実装内容・主な担当ファイル | 完成条件 |
| --- | --- | --- |
| A | 閉じたGRU/Transformer Config・dispatch、Torch runtime/安全な配列保存の共通化、Python/nativeの方策identity照合。gf_models/config、CLI config/bindings、workers、ppo_run/worker、native projection | 既存MLPの数値・保存・学習/再開が回帰しない。baselineは任意依存なしで起動 |
| B | 有限窓feature ringと系列rollout schema2。gf_models/sequence、gf_training/sequence_rollout、native_learning/ppoの限定接続 | Python/Rust経由のtoken/順序/hash一致、shard/padding/予算/失敗テストPASS |
| C | gf_models/gruと系列PPO learner、CPU policy、保存復元。共通PPO loss/GAEを再利用 | CPU/GPU全件照合、勾配/microbatch一致、実16試合×2更新と境界再開一致 |
| D | gf_models/transformer。B/Cの窓・learnerを再利用 | causal/padding漏洩テスト、同じ数値/実対戦/再開ゲートPASS |
| E | 固定入力によるCPU推論・CPU/GPU更新・collect→prepare→update→saveの比較 | 時間・資源・出力同一性を記録。速度や勝率を受入の代用品にしない |
| F | 下記の事前固定した正式比較 | 正常終了、validationと費用報告。最終testは別段階 |

rootがConfig・文書・仕様・実験条件・統合/コミットを所有し、bot_implementerが限定した
実装/テストを所有する。1人の書き手で進め、各完成単位をbot_reviewerと
bot_architecture_reviewerが独立確認する。変更が採用設計の境界を超えたら設計へ戻る。

これは前処理の局所修正より大きい、モデル/rollout/dispatchを跨ぐ実装である。
標準PyTorchを組み合わせる範囲に収めるが、独自デシリアライザ・RNN kernelや汎用学習libraryが
必要になった場合は着手前に目的・測定上の必要性・保守負担をユーザーへ説明する。

## 7. 測定と実験の順序

性能測定前にユーザーへ通知し、他の重い作業の状況を記録する。計測中は自身のbuild/testを止める。
正しさ確認後の固定入力benchmarkはwarmup1回＋交互順5回、release native版と同じデータ・
候補数・worker条件で行う。CPU推論p50/p95、全行動の候補生成込み時間、更新時間、16試合全体、
RSS/PSS、CPU秒、GPU allocated/reserved、モデルbytes、scratch/I/O、期限超過を報告する。
既存の約22秒という全offline前処理を、実対戦推論や新モデルの速度保証として使わない。

全モデル実装後にコード/設定/runtime/実バイナリを固定して、**MLP/GRU/Transformer各3 seed**
（17/29/43、計9run）を順番に実行する。最大枠は9×2時間=18時間で、smoke/測定/validationは別計上。
既存の3MLP runは保存し、正式な同条件比較用MLPは現在の基盤でfresh runする。
GRU/Transformerだけなら6run/最大12時間だが、旧MLPとの時間予算を揃えた比較とは呼ばない。
実行順は事前に、seed17: MLP→GRU→Transformer、seed29: GRU→Transformer→MLP、
seed43: Transformer→MLP→GRUと固定する。各親runのCPU/GPU外部負荷を記録する。
方策sampling seedもcampaign/学習seed/ペア番号/役割から事前固定し、GameSeedから導出せず、
成績を見た再抽選や実行順の変更をしない。

新train namespaceをcampaign単位で固定する。同じlearning seed/batch/pairには各モデルで同じ
GameSeedを割り当て、validationの既存32seedおよびtest namespaceと分離する。
同じtrain seedでも各方策が生成する軌跡やsnapshotは異なる。公平性の主比較は同じ実時間予算、
更新数/行動数も併記する。モデル間のparameter数、履歴入力差は明記し、既存MLPと系列モデルの
差を純粋な層構造だけの効果とは断定しない。GRUとTransformerは同じ32token契約で比較する。

各完成親の最終世代を、32ペア/64試合（半分Random、半分heuristic）の同じvalidationにかける。
9親合計576試合。ハイパーパラメータ/窓長はこの結果を見て選び直さない。追加のモデル間直接対戦を
行う場合は、3組のモデル対×同じ学習seed3個×32ペア×先後交換=576試合を事前割当にする。
勝敗・引分・無効試合・paired区間・seed別結果を報告し、3学習seedを多数の独立runと見せない。
失敗セルを短い予算の再実行や成績選別で埋めない。

LightGBMの既存6fitと固定データはそのまま比較資料として保持する。共有生成費とfit費用は
別に示し、PPOと同一総費用の比較とは呼ばない。この単位では新しい系列親を旧固定データへ
混ぜたり、既存fitsを上書きしない。新親から固定データを作る拡張は別のsource契約を必要とする。
最終testは本計画の実装受入には含めず、モデル/設定/評価手順の凍結後に別計画で実行する。

## 8. 必須の受入と残る実機確認

- 未来token変更、相手private入力、episode/viewer混合、候補順/個数変更、padding変更による
  漏洩がないこと。window1/32/33、同じpolicy ID、0決定episode、途中shard境界を含める。
- 同じ保存入力から推論ringと学習窓が一致し、whole-batch CPU/saved/GPU照合が更新前に完了。
  既存journalは受理順序・単体入力・Replayの行動列を照合するが、全観測がReplayから再生された
  ことまでは証明していない。この保証を拡大したと主張しない。系列は保存された本人入力を正本とする。
- GAEはepisode単位、正規化はbatch単位、lossは終点1回。候補/時間の二重padding、microbatch
  分割、parameter gradient、shuffle、checkpoint roundtrip、pause/resume連続実行一致を確認。
- unavailable CUDA、parity不一致、OOM、非同期CUDA error、quota/期限/保存/回収失敗では
  成功世代を作らない。元run失敗時は全offlineデータ拒否。古い形式は明示的loaderのみで読む。
- pytest、ruff、mypy、native Cargo checks、別環境wheel、baseline任意依存なし起動、
  現在のMLP/LightGBMの数値・元データ・モデルhashを確認する。

未測定なのは、標準GRU/Transformerの実機CPU/GPU誤差、CPU推論遅延、32tokenでの記憶能力、
更新あたり資源、2時間で進む更新数と勝率である。A～Eで証拠を揃えてからFへ進む。
FP32/1e-5を満たせなければ原因を切り分けて設計を再レビューし、実装成功と学習品質を分けて報告する。

## 9. 計画レビューの記録

- bot_verifier: 現在のPolicy、本人workerの寿命、HTTP retry、元journalの順序検証、
  nativeの完全episode配列、MLP固有dispatchを読み取り調査。実行検証はしていない。
- bot_architecture_reviewer: 修正版の設計READY。MLP専用のsource/方策identity照合を
  Python/native/lineage/authorization/最終validationで明示的に拡張する指摘を採用。
  共有検証器の一般化によって旧固定データのsource許可が広がらない負例も追加した。
- bot_reviewer: 学習/正しさの観点でREADY、blockingなし。正式run順の事前固定という
  任意提案を採用し、学習seedごとにモデル順を交代させ、外部負荷を記録する。
- root: 上記指摘の反映を確認。今回は文書/リンク/差分だけを検証し、アプリの変更や
  benchmark、対戦、fitは実行していない。計画のREADYは実機の数値/性能/学習品質のPASSではない。
