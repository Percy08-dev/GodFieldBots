# 0022: GRU／Transformer PPOの実装計画

- 状態: Accepted、2026-11-01。実装・機能/実対戦/性能検証完了。
  独立した正しさ・設計適合・修正済み2GiB typed実対戦証拠の監査はREADY。
- 作成日: 2026-10-31。責任者: root。
- 対象: GRU PPO、Transformer PPO、および両者に必要な系列入力・保存・比較基盤。
- 旧32回入力窓・正式9runまでの案を置き換える。旧案はgit履歴に保存。

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

## 3. 承認済みの系列とConfig

完成単位はGRU/Transformerの実装、各モデル実対戦2更新、正常境界pause/resume、
性能・資源測定まで。正式9runと最終testは後続。

1ステップは本人が行動を選ぶ直前のDecisionInput。既存MLPと同じ公開履歴集約encoder
（直近128visible events）から候補非依存stateを選択し、CPU signed-log1p→FP32へ変換する。
現signatureではstate373次元、候補690次元だが次元だけで互換とせずsignatureも照合する。
相手の1手ごとのtoken化はしない。本人の過去state全体を参照し、未来・別viewer/episode、
GameSeed/RNG・generation・相手model ID・return/advantage/old_valueを特徴にしない。

初版の最大履歴は本人1,024意思決定。全game20,000行動上限とは独立。
1,025回目はforward/送信前に資源エラーとして親run全体を失敗にし、履歴を切らない。
新系列recipeは `own-decision-full-history-tbptt-v1`。

系列PPO専用の閉じたConfigへ `training.sequence_length` を追加する。既定32、
厳密な整数1..1024（bool/float/0/負数/上限超過拒否）。MLP/LightGBMの設定へ黙って適用しない。
Wは勾配区間の長さであり、推論履歴長でも論理minibatchのloss件数でもない。
実効Config・学習記録・再開manifestにWを保存し、通常resumeでの変更を拒否する。

終点tについてs=max(0,t-W+1)とし、prefix[0,s)を現在の重みでno_grad再計算する。
そのhidden/全層KVをdetachし、suffix[s,t]に勾配を流して、lossは終点tに1回だけ加える。
W>=t+1なら全履歴BPTTと一致する。短いWはTBPTT近似であり、全履歴勾配と同等ではない。
optimizer step後に古いprefix状態を再利用しない。窓内の他の時点にもlossを重複加算しない。
GAEは自然終了episode全体、advantage正規化は受理batch全体で一度行う。

## 4. モデルと推論状態

| 項目 | GRU | Transformer |
| --- | --- | --- |
| state投影 | Linear(state幅,128)+ReLU | 同左+固定sin/cos絶対位置符号 |
| 系列層 | nn.GRU、1層、hidden128、単方向 | 2層、幅128、4heads、FFN256、pre-norm、ReLU |
| dropout | 0 | 0 |
| 推論状態 | 前回のhidden | 各層の過去K/V |
| actor | context+現在候補特徴→128-ReLU→1logit | 同左 |
| critic | context+現在state→128-ReLU→1value | 同左 |

state encoderは共有するが候補情報を入力しない。同じ重みのcriticは候補の個数/内容/順序に
依存せず、候補順の変更はlogit列だけに反映する。Policy.reset/selectとDecisionInputの契約を維持。
本人workerはepisode単位で作られ、resetでhidden/cache/位置/sampling RNGを初期化する。
HTTP retryは同じcommandの再送でありselectを再実行しない。拒否/送達不確定/worker失敗は
親収集失敗とし、未受理状態を次の成功軌跡へ持ち越さない。途中episodeの再開は追加しない。

Transformerは標準Linear/LayerNorm/SDPAによる小さなcache対応モデル層とする。
同じparameter/数式を全履歴・prefix/suffix・逐次推論で共有する。絶対位置はepisode開始から
連続させる。逐次query1個がpast/current keysだけを見る場合はis_causal=False。
学習suffixの複数queryにはprefix offset付きcausal maskとpadding maskを明示する。
prefix KVはdetachするがsuffix内KVまでdetachしない。padding queryはloss対象にしない。

CPU seed固定初期化。Linearは直交初期化、ReLU段gain√2、actor出力.01、value出力1、bias0。
GRU各gateはgain1で個別直交初期化。Transformer Q/K/V/attention出力はgain1、LayerNorm
weight1/bias0。各層を独立drawで初期化する。parameter順序と初期化recipeを固定する。
初版GRUはcuDNNを明示的に無効化した標準Torch経路、TransformerはSDPA MATHを使用する。
FP32 IEEE、AMP/TF32なし。backend設定をmanifestに保存し、失敗時の自動切替はしない。

標準APIの根拠:
[GRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html)、
[SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。
独自RNN/attention kernel、外部flash-attn、汎用trainer frameworkは追加しない。

## 5. 学習配列・数値ゲート・保存

Rustの検証済み全候補特徴配列とcomplete-episode offsetsを再利用する。Python学習層が
state変換、prefix/suffix index、GAEを所有する。BulkRolloutの既存decision iteratorで失われる
episode境界は、系列経路ではchunks/episode_offsetsから直接保持する。

新しい閉じた数値rollout schema2は、episode単位のstate token bank、decision/episode対応、
候補offset、selected/old_logprob/old_value/advantage/target、recipe/signature/hashを持つ。
各終点の全prefixを複製しない。候補・decision shardは64MiB以下、token bankも64MiB以下。
episode最大1024*373*4bytesは単独bankに収まる。各shardはbankと終点位置をhash付きで参照し、
元のepisode内の[0,t]だけを入力とする。ID類は照合metadataでありnetworkへ渡さない。
元journal/collection schema4、MLP rollout schema1、旧feature署名を再解釈しない。

各epochで終点をシャッフルして論理minibatch128を選択する。その中だけで
(prefix長,suffix長)をstable groupし、同じ境界の系列をまとめる。epoch全体を先に並べ替えない。
グループ内の順序を維持して資源上限で分割し、候補をpaddingする。lossは終点件数で重み付けし、
clip/Adam stepは論理minibatchに1回。1件でも収まらなければ失敗し、Wを短縮しない。

各batchの最初のAdam変更前に、全終点・全有効候補について次をatol=rtol=1e-5で照合する:
保存selected logprob/value、凍結CPU逐次cache、CPU全prefix、GPU全prefix、そして実際のWで
実行するlearner deviceのsplit forward（prefix no_grad、suffix grad有効）。CPU presetでも
実際のCPU split照合は必須。CUDA時の追加CPU split参照だけを任意とする。
fixtureのsplitテストだけでは代用しない。照合graphはmicrobatchごとに解放し、全件PASS前に
更新しない。更新後はCUDA同期・finite metrics・CPU array保存・検証の順で応答する。

既存の2GiB memory/cache512MiB/chunk16MiB/scratch8GiBの前処理presetを継承する。
推論cache予算はparticipantごと128MiBのConfigとして追加し、model・配列所有・予約量を記録。
microbatchはpadding込み候補slots4096、state tokens32768、attention score slots8388608
（layers*heads*B*T²）、数値入力64MiB以内。prefix再計算/KV/照合も事前受付する。
Transformer1系列1024tokenがattention上限に一致する。GPU activation実量は別に測定し、
soft limitをOS hard quotaやOOM回避保証とは扱わない。容量/期限/OOM/非同期エラーは親run失敗。

系列inference/resumeは専用version2 JSON+型付きNumPy、allow_pickle=Falseとし、model ID、
全履歴recipe/signature/構成/backend、tensor key/shape/dtype/hash、Adam/RNG、Wを閉じて検証する。
世代/更新の既存hash参照・保存順・保持規則を維持する。正常batch境界では全episodeが終了済み
なのでhidden/KVをresumeへ保存しない。モデル種別/W/code/runtime/backend変更の通常resumeを拒否。
旧形式互換はコード/環境を跨いだ通常resumeの許可ではない。

MLP専用だったPython/nativeのsource/selection照合、lineage.owned_binding、native_authorization、
CLI bindings/preflight、最終policy validationも閉じたmlp/gru/transformer選択へ拡張する。
model ID、{mlp,gru,transformer}-ppo-v1-prefix、inference digestを正確に結ぶ。
旧fixed-data build/fitのMLP親限定は明示的に維持し、正常GRU/Transformer親を渡す負例も追加する。

native build_infoの既存string値形式に `ppo_policy_models` capabilityを追加する。
GRU段階は`mlp,gru`、Transformer完成時は`mlp,gru,transformer`。欠落は旧MLP専用として扱い、
unknown/重複/不正な値を拒否する。native backendでは選択modelの対応をlearner/collector開始前に
確認し、旧wheelで16試合を集めてから失敗することを防ぐ。これは元データの検証を代替しない。
既存binary identityに記録し、旧wheelのMLP互換/GRU早期拒否をテストする。

Native Parentのcompact summaryはeffective_config全体を返さない。Python authorizationへは
既にnativeで検証したstate.effective_config.model.model_idから導いた小さなstate.model_idだけを
追加して渡し、閉じた対応モデルを再検証する。親JSONの二重読込やConfig全体のIPCコピーで
補わない。capability対応wheelとPython/native differential testでsummary形状を検証する。
capability markerがないと識別済みの旧wheelだけは欠落model_idをMLPと解釈できる。
markerのあるwheelではmodel_id欠落/未知値をContractErrorとして拒否し、GRUをMLPへ推測しない。

## 6. 段階・所有と受入

| 段階 | 内容 | 完成条件 |
| --- | --- | --- |
| A | 閉じたConfig、共通Torch/安全配列/artifact dispatch、方策identity照合 | MLPの数値/保存/再開回帰、baseline任意依存なし起動 |
| B | token banks、rollout schema2、TBPTT境界、資源予約 | Python/nativeの順序・feature・教師・shard一致 |
| C | GRU hidden/learner/policy/artifact | 全件parity、gradients、実2更新/境界resume一致 |
| D | Transformer KV/offset masks/learner | cache/full/split照合、同じ実対戦/保存ゲート |
| E | 固定入力と実対戦で速度・資源測定、最終review | 記録と制限の報告。勝率改善を実装受入の代用にしない |

rootは設計・Config要件・project依存/preset・文書・統合/commitを所有する。
bot_implementer一人が割当済み実装とテストを行い、rootはその所有ファイルを同時編集しない。
各完成単位はbot_reviewerとbot_architecture_reviewerが独立確認する。
大規模な独自library/kernel/parserが必要になった場合は着手前に目的・証拠・保守負担を説明する。
Simulator sourceや既存のモデル/データを変更しない。

必須テスト:
- W=1/31/32/33/128/1024、短いepisode、0決定episode、shard境界、1024/1025の上限。
- prefix token/hidden/KVへの勾配が切れ、suffixには届く。共有parameterはsuffix勾配を持つ。
- Wが全履歴を覆う場合のfull BPTT勾配一致。同じ重みのforwardはW変更で変わらない。
- 未来・別viewer/episode・private漏洩がない。候補順、critic独立、padding、絶対offset mask。
- cache reset、HTTP retryの非二重更新、W変更resume拒否、snapshot方策/同policyID両席。
- unavailable CUDA、parity失敗、OOM/非同期error、quota/期限/保存/回収失敗で成功世代を作らない。
- 各モデルseed17で連続2更新と1更新後pause/resumeのfresh対を実行する。各run16試合/batch、
  32試合/run、計4run/128試合。元の2時間/90分/119分予算は短縮しない。
  weights/Adam/RNG/rollout一致を確認し、smokeを正式比較/学習品質と呼ばない。
- pytest/ruff/mypy、必要native checks、別環境wheel、optional-free CLI、元モデルhash維持。

## 7. 性能実装と測定

NumPy/Torchの連続配列を単位とし、token/candidateごとのPython dict/list再構築を避ける。
cacheは有界storageへ追記し、全履歴を毎手concat/copyしない。TBPTTのprefix再計算は
optimizer変更を跨いでcacheしない。標準ライブラリで十分な範囲を優先し、未測定の独自arenaや
schedulerを作らない。配列コピー、prefix再構成、suffix/逆伝播、padding、保存を区別して測る。

性能測定前にユーザーへ通知し、外部負荷を記録する。測定中は他のown build/testを止める。
固定された同じ重み・データ・RNGに毎回戻してwarmup1回+5回測定する。推論cache/fullprefix、
W32/64/128の更新時間とRSS/PSS/CPU秒/GPU allocated/reserved/model bytes/scratch/I/Oを報告する。
W間の勾配差は仕様であり、モデル更新結果のbit一致を要求したり速度から学習品質を推測しない。
モデル間で同じ出力は要求せず、各モデル内のreferenceとの照合を行う。
実smokeで16試合collect→prepare→update→saveと推論p50/p95を記録し、固定入力測定と区別する。
上限1024のmemory/parityを確認し、全候補の境界条件と失敗を隠さない。

正式9run・validation576試合・直接対戦・最終testは今回実行しない。既存MLP3親、LightGBM6fit、
固定データは保持する。旧共有データ生成費を重複計上しない。新系列親を旧fixed-data契約へ混ぜない。
実装と実機の数値/再開/性能が成立した時点で今回の完成とし、品質比較は別計画にする。

## 8. 計画レビューと未実施の証拠

旧有限窓案はa2f005bの履歴に残る。会話でユーザーが全履歴・hidden/KV・1024上限、完成範囲A–E、
設定可能TBPTT・本人意思決定単位を選択し、実装を承認した。
変更後のbot_architecture_reviewerとbot_reviewerの書面reviewはREADY。
追加指摘を反映済み: 実際のGPU split forwardも全batch pre-Adam照合、prefix/KVの資源受付、
logical minibatch内だけのgrouping、共有parameterとprefix tokenの勾配検査の区別。
実装後の独立review、実機CPU/GPU誤差・メモリ・遅延・再開一致はまだPASSとしない。

## 9. 実装結果と記録

実装はGRU/Transformerの閉じたConfig、全履歴cache、W指定TBPTT、schema2bank/artifact、
Python/native identity、CLI学習/正常境界resumeまで接続した。標準Torch演算を使い、新しい
第三者依存や汎用library/kernelを追加していない。既存MLP/LightGBMの仕様・元データは保持。
891Python tests、別実行20CUDA cases、93Rust tests、旧データ5proofs、76ファイル一致の
別環境wheel、60固定入力測定を検証。実対戦は予備8GiB/legacy128試合の後、参照presetを
合意済み2GiB/16MiB/typedへ修正してfresh128試合を再検証した。各モデルの連続/停止再開は
weights/Adam/RNG/numericが一致。source/runtime/codeはこの修正中も不変で、Config参照のみ変更。
性能測定の履歴は最大29なのでW32/64/128は全てfull gradientとなり、長期窓の性能/品質効果を
主張しない。CPU更新がこの小batchではCUDAより速いことも記録。最終受入と全指標は
[検証記録](../verification/sequence-ppo.md)を正本とする。正式9run/最終testは未実施。

rootは独立した最終監査後にA〜Eの完成を受け入れる。初期preset参照の誤りと測定helperの
回収問題は証拠を保持して修正・再検証し、未解決のblocking findingはない。
既存のデータ/モデル24ファイル・依存設定・Simulator識別情報は最終照合でも不変。
