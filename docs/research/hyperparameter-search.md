# GodFieldBots ハイパーパラメータ探索調査

調査日: 2026-11-01。対象コード: `fe017cd97866073a7fc2fdca67b50041e059cb05`。
対象はMLP PPO、GRU PPO、Transformer PPO、固定データLightGBM。
本資料は文献・公式仕様・現実装を照合した研究メモであり、探索実行や新しい学習仕様の採用を決定するADRではない。
後続会話の独立した整理・訂正と、候補範囲拡張・エンコーダーの共同設計に関する論点は
[モデル調整と手札エンコーダーの議論整理](discussion-review.md)を参照する。

幅と層数は共同で探索する。さらに学習率、actor／critic構成、系列の勾配長など、相互作用する設定をまとめて比較する。
全項目の総当たりや、各項目を一度ずつ最適化して固定する方法には偏りがある。
今回の出発点は、既存設定を対照にした条件付きの共同探索、複数の学習seed、未使用の確認用対戦条件である。
探索範囲・試行数・途中打切りの採否は、実機の費用と学習曲線を得てから確定する。

本文では、公式APIが保証する動作、他環境での実証結果、このリポジトリの観測事実、GodField向けの提案を区別する。
他タスクの最良設定やライブラリの既定値を、GodFieldで検証済みの設定として扱わない。

## 現在の基準と未調整の範囲

現在の契約pinはSimulator `ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6`、catalog `public-ja-v2`。
入力は本人のObservation、可視Events、typed metadata、同一revisionの本人向けActionOptions、許可された版固定カタログ。
SequenceOptions、拒否詳細、他者private、GameSeed/RNG、未来情報はpolicy入力・候補生成に使わない。
SequenceOptions、拒否詳細、他者private、GameSeed/RNGはmask・教師にも使わない。
自然終了returnを、行動後に学習ラベルとして使う既存の方式は維持する。
criticと系列state encoderは候補内容から独立させる。[現行設計](../architecture.md)、[ADR0022](../adr/0022-gru-transformer-ppo-plan.md)

| 対象 | 現在の基準 | 設定変更の可否 |
| --- | --- | --- |
| MLP | actorとcriticを分離、それぞれ128幅2層ReLU | 幅・層数・encoderはv1検証で固定 |
| GRU | state投影128、単方向GRU 128幅1層、共有系列encoderと別actor／critic head | 構造固定、TBPTTのWのみ1〜1024で設定可能 |
| Transformer | 128幅2層、4 heads、FFN256、pre-norm、ReLU、sin/cos位置符号、dropout 0 | 構造固定、Wは設定可能。cache対応の独自層を標準Torch演算で構成 |
| PPO共通 | Adam lr 3e-4、betas .9/.999、eps 1e-8、4 epochs、論理minibatch 128、clip .2、value .5、entropy .01、grad norm .5、γ1、GAE λ.95 | 主要係数はコード内固定 |
| LightGBM | CPU GBDT回帰、31 leaves、lr .05、最大500 rounds、early stopping 50、min_data_in_leaf 20 | 葉数・lr・rounds・patience・min_data等は設定可能。L1/L2等は未公開 |

実装参照: [Config](../../src/gf_models/config.py)、[MLP](../../src/gf_models/mlp/model.py)、[GRU](../../src/gf_models/gru/model.py)、[Transformer](../../src/gf_models/transformer/model.py)、[PPO数値処理](../../src/gf_training/ppo_numeric.py)、[系列数値処理](../../src/gf_training/sequence_numeric.py)、[LightGBM](../../src/gf_models/lightgbm/model.py)。

現在のpublic aggregate encoderは直近128可視イベント、hash buckets 32、最大3 action elementsを使う。
現signatureのstate幅は373、共通encoderの候補ベクトル幅は690。MLP actorはこの690を入力し、系列actor headは128幅のcontextを連結した818を入力する。
次元数だけでは互換性を判定しない。
MLP／LightGBMの集約履歴と、系列モデルの全本人意思決定履歴は異なる表現なので、モデル比較を「情報表現まで完全同一」とは説明しない。

既存のMLP3 seedは123／144／145更新、LightGBMは2特徴条件×3 seedの6 fitが完了している。
GRU／Transformerは128自然終了試合の2更新・正常境界再開smokeと性能測定までで、正式な強さの比較は後続。
古い失敗親runの記録や未実施の記述は歴史的経緯として残り、現在地は[新契約での実験記録](../verification/catalog-v2-adoption.md)と[系列モデル検証](../verification/sequence-ppo.md)を優先する。

metadataのみで確認した実環境はPython 3.14.10/GIL有効、NumPy 2.5.4、Torch 2.14.1+cu132、torchvision 0.29.1+cu132、LightGBM 4.7.0、PyArrow 26.0.0。
TorchはIEEE FP32、決定的アルゴリズム、AMP/TF32なし、1 thread。GRUはcuDNN無効、TransformerはSDPA MATH。
以下のPyTorch仕様は2.14、LightGBM仕様は4.7.0を参照する。新しい依存の導入やGPU動作確認は行っていない。

## 文献から採用できる探索原則

| 一次資料 | 確認できる知見 | GodFieldへの適用範囲 |
| --- | --- | --- |
| [Andrychowicz et al. 2021](https://arxiv.org/html/2006.05990v1) §2、§3.2、§3.5、§3.7 | 相互作用する項目をグループ化し、学習率を各グループに含めて調べる。構造、初期化、更新設定が結果を左右する | 連続制御での実証。共同探索の設計を参考にし、数値や活性化の順位は移植しない |
| [Engstrom et al. 2020](https://arxiv.org/abs/2005.12729) | PPO/TRPOでは実装上の補助設定も学習挙動を大きく変える | アルゴリズム名だけで公平性を主張せず、正規化・初期化・clippingも記録する |
| [Eimer et al. 2023](https://proceedings.mlr.press/v202/eimer23a.html) | HPO結果はtuning seedに依存し、seedへの過適合が起きる。自動探索とtuning/testing seedの分離を検討する | 1 seedの順位で設定を確定しない。GodFieldの最適試行数や手法の優劣までは示さない |
| [Ni et al. 2022](https://proceedings.mlr.press/v162/ni22a.html) | 部分観測下ではactor／critic構成、学習方式、context lengthの選択が重要 | 主にoff-policy TD3/SACを扱う。PPOで共有encoderを廃止すべきだという結論にはしない |
| [Parisotto et al. 2020](https://proceedings.mlr.press/v119/parisotto20a.html) | RL用Transformerでは構造変更が最適化の安定性を改善し得る | V-MPOで学習するGTrXLの結果。現行の小型pre-norm PPOモデルにgating等を追加する価値は別に測る |
| [Agarwal et al. 2021](https://arxiv.org/abs/2108.13264) | 少数runの点推定だけでは比較が不安定。区間推定や分布を含める | 学習seedの変動と対戦seedの変動を区別する。多数タスク用集約指標を単一ゲームへ無条件に流用しない |
| [Cawley and Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html) | 選定指標自体への過適合が、選定後の性能評価を楽観化する | 探索で使ったvalidationを最終性能の証明に使わない |
| [Patterson et al. 2024](https://jmlr.org/papers/v25/23-0183.html) | RL実験の変動、複数agent比較、hyperparameter選択と実験者の偏りを扱う | 最良seedだけを報告せず、選定手順・探索費用・不確実性を評価条件に含める |

PyTorchのAPI説明は設定の意味を確定する資料であり、幅128や2層が最適である根拠にはならない。
RL Baselines3 Zooも、探索時の設定がSB3の通常既定値と異なることを明記している。
例えばPPO探索ではorthogonal initializationを無効にする設定がある。現行の直交初期化からの変更は明示的な実験とする。
[Zoo公式資料](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html)

## モデル構造を共同で探索する

幅×層数の組合せを一つの構造設定として扱う。「最良の幅を決めてから層数を調整し、幅は戻さない」という一方向の手順は採用しない。
共同探索は全組合せの実行を意味しない。条件付きの候補抽出、少数の対照実験、上位候補周辺の再探索を組み合わせる。

一様幅hのMLPでも、パラメータ数には入力幅dに依存する項がある。
一般に全結合層のパラメータ数は `Σ (n_previous + 1) × n_next`。
したがって「幅×層数が同じ」「パラメータ数が同じ」「同じ壁時計時間」は別の条件である。
入力690のactorと373のcriticで同じ幅を使っても、同じ容量にはならない。
実測parameter数、推論p50/p95、更新時間、収集込みの時間、メモリを構造設定に添える。

| モデル | 共同探索する構造群 | 初期調査用の候補例 | 条件・追加で確認する点 |
| --- | --- | --- | --- |
| MLP | actor幅・層数、critic幅・層数、学習率 | 幅64/128/256、隠れ層1/2/3から有界に抽出 | 最初はactor/critic同形を含め、critic幅をactorの1倍または2倍にする候補も入れる。全組合せを一律実行しない |
| GRU | state投影幅、hidden幅、recurrent層数、head幅、W、学習率 | hidden64/128/256、層1/2、head幅は当初hiddenと連動 | 投影・hidden・headの連動は探索次元を減らす仮説。上位候補で独立化を試す |
| Transformer | d_model、層数、heads、FFN比率、W、学習率 | d64/128/256、層1/2/3、heads2/4/8、FFN比2/4から資源適合組だけ抽出 | d_modelがheadsで割り切れることに加え、head幅、KV、attention受付、実速度を確認 |

この範囲はGodField用の初期仮説であり、論文が保証する推奨値ではない。
すべてのモデルへ同じ12試行を割り当てれば十分、という根拠もない。
候補を削る場合は、理由を「資源不足」「予算配分」「比較対象の限定」と記録し、未試行の構造を劣ると結論しない。

MLPのactor／critic分離と、系列モデルの共有encoder＋別headは既に異なる。
系列encoderの共有／分離を探索する場合、勾配の競合、学習速度、推論負荷を比較する。
共有を外した方がよかった他のPOMDP研究は仮説の根拠になるが、別学習方式・別入力である。[Ni et al.](https://proceedings.mlr.press/v162/ni22a/ni22a.pdf)
criticへの候補特徴流入や、他者privateの利用は構造探索に含めない。

## 系列長とTransformer固有の制約

現在のWは、推論で参照する履歴長ではなく、逆伝播する本人意思決定数である。
終点tに対し `s=max(0,t-W+1)` とし、prefix `[0,s)` を現在の重みでno_grad再計算し、detachした状態からsuffix `[s,t]` を学習する。
lossは終点だけに一度加え、GAEは自然終了episode全体で計算する。
Wを変えると勾配の近似と費用が変わるが、同じ重みでのforwardが参照する過去は変わらない。
[ADR0022](../adr/0022-gru-transformer-ppo-plan.md)

Wの候補例は16/32/64/128。ただし最初に本人意思決定長の分布、Wを超えるepisodeの割合、長いprefixの費用を測る。
既存のW32/64/128測定では履歴が5〜29なので、全条件が全履歴勾配となっていた。
その結果からW32が十分、あるいはW128に価値がないとは判断できない。[既存測定](../verification/sequence-ppo.md)
Niらのcontext lengthは入力文脈の切り方も含み、現行の全履歴forward＋TBPTTと同一の変数ではない。
TBPTTで勾配を打ち切ると短い依存へ偏る問題は[Tallec and Ollivierの2017年preprint](https://arxiv.org/abs/1705.08209)にも分析されている。
同論文はPPOの検証ではなく、特定のWや別の勾配推定器の採用を正当化するものではない。

現在のattention score受付上限8,388,608は `2 layers × 4 heads × 1 batch × 1024²` に等しい。
層数やheadsを増やすと、幅を狭くしても、現在の式と上限では1024手の単一endpointを受付できない場合がある。
KV保存量もおおむね `2 × layers × history × d_model × bytes_per_value` に比例する。
これらは受付用の概算であり、実際のGPU peakを保証する式ではない。
構造可変化ではcache・parity配列・資源見積もりも変更し、1024手契約を維持する。
Wや履歴を黙って短縮して収めない。[受付実装](../../src/gf_training/sequence_numeric.py)

現行Transformerはpre-normのcache対応実装で、`nn.Transformer`の既定構造をそのまま利用していない。
比較候補にはhead幅・FFN比率・活性化・正規化位置・残差構造があるが、gatingや別の位置表現はモデルrecipeの追加として扱う。
[PyTorch Transformer仕様](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Transformer.html)、[GTrXL論文](https://arxiv.org/abs/1910.06764)

最初の探索ではdropout 0を維持する。GRUのdropoutは層間に作用し、1層では作用対象がない。
SDPAは渡されたdropout確率を使うので、親moduleをevalにしただけでは無効化されない。
learnerだけに任意のdropoutを入れると、保存logprobと再計算する方策の対応、cache/full/split照合に影響する。
導入するなら乱数とlikelihoodの扱いまで設計する。[GRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html)、[SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

cached attentionのcausal maskも維持する。prefixを持つ1-queryの推論へ、単純に`is_causal=True`を付ければよいとは限らない。
現在の絶対位置offset、padding、未来参照禁止を、構造ごとのfull/cache/split照合で検証する。

## 最適化と学習制御の探索群

PPOは同じ収集データを複数epoch更新する方式であり、epochs・minibatch・学習率を一緒に考える必要がある。
[PPO原論文](https://arxiv.org/abs/1707.06347)
GAE λはadvantage推定のbiasとvarianceに関わる。[GAE原論文](https://arxiv.org/abs/1506.02438)

以下は予備実験に向けた候補であり、全項目を一度に解放する計画ではない。
探索群を順に扱っても、各群で学習率を再調整し、得られた設定を組み合わせた後に相互作用を再確認する。

| 探索群 | 一緒に動かす項目 | 初期候補例 | 当初の固定・判断理由 |
| --- | --- | --- | --- |
| 容量と最適化 | 幅、層数、actor/critic比、lr | lr 3e-5〜1e-3を対数抽出、現行3e-4を必ず含む | Adam betas .9/.999、eps 1e-8を維持 |
| 更新量 | lr、epochs、論理minibatch、PPO clip | epochs2/4/8、minibatch64/128/256、clip .1/.2/.3 | 収集16自然終了試合/batchを維持し、実decision数を記録 |
| credit assignment | λ、value係数、W、critic構成、lr | λ .9/.95/.98/1、value係数 .25/.5/1 | γ1と終局±1/0を維持。報酬や割引変更は目的の変更として別扱い |
| 探索と更新制約 | entropy係数、clip、lr、更新回数 | entropy 0と1e-4〜3e-2の正値対数範囲、grad norm .5/1 | 初期actor出力gain .01を対照にし、初期化変更と混同しない |
| スケジュール | 初期lr、減衰方式、epochs、停止規則 | constantと事前定義したlinear decay | KL停止や最良checkpoint選択は記録・資格判定の拡張が必要 |
| 数値・正則化 | optimizer、weight decay、活性化、初期化、正規化 | Adamを基準、AdamW等は後続の対照実験 | AMP/TF32/backendは初期探索で固定。速度と数値方式を同時変更しない |

「batch」は区別する。現在は16試合の収集batch、128決定の論理minibatch、4096候補slots以下のmicrobatchがある。
microbatchは資源のための分割であり、損失の重み付けを保って一つの論理minibatchで一度clip/stepする。
これを変えてoptimizer step数まで変われば、単なる省メモリ変更ではない。
系列モデルのWもloss件数や収集batch長ではない。

可変候補数Kのcategorical entropyは最大`log(K)`なので、候補数分布とentropyを一緒に観測する。
`H/log(K)`はK>1で診断用に併記でき、K=1は別扱いにする。
これは数学上の診断であり、現行のentropy lossを自動的に正規化して置き換える提案ではない。
行動候補generatorを変えると探索空間とentropyの基準が同時に変わるので、当初は固定する。

観測する量は対戦成績に加え、policy/value loss、entropy、KL指標、clip fraction、valueのexplained variance、gradient norm、受理決定数、更新回数、工程別時間。
現行の記録にない量は追加が必要。KLの近似式や集計単位を保存し、ノイズを含む単一値だけで停止しない。
value loss clippingとPPO policy clippingを混同しない。現行value lossはunclipped MSEである。

活性化と初期化は一つのrecipeとして扱う。PyTorchの初期化gainは活性化に依存するため、ReLUを変更して同じgainを使えば同等、とはいえない。
[初期化仕様](https://docs.pytorch.org/docs/2.14/nn.init.html)
AdamW等のdecoupled weight decayは通常のAdamのweight decayと区別し、biasやnormalizationへの適用、parameter groupsを記録する。
schedulerには何を1 stepとするか、再開状態、restore順序が必要である。
[Adam仕様](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)、[scheduler仕様](https://docs.pytorch.org/docs/2.14/optim.html#how-to-adjust-learning-rate)

## LightGBMの探索と教師データの限界

最初の対象は既存の凍結datasetで、81,523 train／2,260 validation decisions。
元の3 MLP親run、seed・両席・共通祖先の分割、最終方策validation、元journal/Replayの適格性を維持する。
新しく探索したPPO親やGRU／Transformer親を、既存dataset契約へ自動追加しない。
過去データでPPOを再学習する方法も、現在のon-policy PPO探索には含まれない。
[固定データ契約](../adr/0008-mlp-ppo-generations-and-fixed-data.md)、[現データ](../verification/catalog-v2-adoption.md)

| 探索する関係 | 候補例 | 比較上の条件 |
| --- | --- | --- |
| 木の複雑さ | num_leaves15/31/63/127、min_data_in_leaf20/50/100/200 | 一緒に探索する。必要ならmax_depthを追加し、葉数との整合を取る |
| 学習速度と木の本数 | lr .01〜.1、最大rounds、early stopping patience | 小さいlrだけを低いround上限で不利にしない。上限到達とearly stopを区別 |
| 正則化 | lambda_l1/lambda_l2、min_gain_to_split | 0を独立候補にし、正値は対数範囲。現Config/native契約の拡張が必要 |
| サンプリング | feature_fraction、bagging_fractionとbagging_freq | baggingはfractionだけでは有効にならない。乱数の影響を再確認 |
| 特徴構成 | observationとpublic aggregate | 同じ凍結source・分割で比較。encoder変更時は署名と数値配列を作り直す |

葉数、min_data_in_leaf、depthの関係は[LightGBM調整資料](https://lightgbm.readthedocs.io/en/v4.7.0/Parameters-Tuning.html)に基づく。
samplingや正則化の定義は[パラメータ仕様](https://lightgbm.readthedocs.io/en/v4.7.0/Parameters.html)を使う。
具体的な候補値は本データ向けの仮説である。

early stoppingはvalidationの改善量とpatienceで最良iterationを選ぶ。
`dart`では同じcallbackが機能しないため、GBDTからの変更時に同じ停止条件とみなさない。
[early_stopping仕様](https://lightgbm.readthedocs.io/en/v4.7.0/pythonapi/lightgbm.early_stopping.html)
既存実装は`feature_pre_filter=False`なので、min_data_in_leaf探索時に古いfilterを再利用する問題を避けている。
キャッシュを追加する場合も、binningや特徴設定との互換性、元sourceの再検証を省かない。

現教師は「実際に選択・受理された行動」と「その後の本人の自然終了return」である。
この構造から、validation L2は収集方策のデータ上での回帰精度を表し、未実行行動の優劣や最適Q値を直接検証する指標ではない。
同一episode内の行は独立な試合ではなく、長いepisodeほど行数が多い。行をrandom splitしたり、未選択候補に順位ラベルを作ったりしない。
episode重みを変えるなら学習目的の変更として別実験にする。

L2で一次選別し、上位候補・現行基準・異なる容量/特徴の代表候補を固定相手との対戦で評価する。
この方法にもL2で落とした強い方策を見逃す可能性があるため、L2の僅差を厳しい除外根拠にしない。
既存6 fitは特徴条件ごとに3 seedで同じL2/best iterationを得た。現在の決定的設定でseedだけを増やしても、独立したデータ汎化の証拠にはならない。
sampling設定を導入した場合には、その設定に応じてfit seedの反復を再設計する。

## 探索アルゴリズムとライブラリの選択

| 手法 | 利点と根拠 | 今回の判断 |
| --- | --- | --- |
| 小さなfactorial比較 | 特定の幅×層数などの相互作用を読み取りやすい | 狭い仮説の確認に使う。全項目のgridへ拡張しない |
| Random search | 簡潔で追跡しやすく、gridより効率的な例がある。[原論文](https://www.jmlr.org/papers/v13/bergstra12a.html) | 再現可能な対照手法。GodFieldで最良の探索法と確定しない |
| 多変量TPE | 複数項目・条件付き空間を扱える。[Optuna仕様](https://optuna.readthedocs.io/en/v5.0.0/reference/samplers/generated/optuna.samplers.TPESampler.html) | Random初期化後の候補。試行が少ない・ノイズが大きい場合の優位性は未測定 |
| Hyperband/ASHA | 少予算から配分を増やし、低成績の候補を打ち切る。[Hyperband](https://jmlr.org/papers/v18/16-558.html)、[ASHA](https://arxiv.org/abs/1810.05934) | 部分予算での順位が有用かを先に調べる。2更新smokeで強さを選別しない |
| BOHB/DEHBなど | モデルベース／進化的探索と複数予算を組み合わせる。[BOHB](https://proceedings.mlr.press/v80/falkner18a.html)、[RLでの比較](https://proceedings.mlr.press/v202/eimer23a.html) | 有力な後続比較対象。既存ライフサイクルへ導入できるかを含めて判断 |
| PBT | 重み継承とハイパーパラメータ変異でscheduleを探索する。[原論文](https://arxiv.org/abs/1711.09846) | 今回の最初の導入対象にしない。親子来歴・設定変更・資格判定・費用の設計が別途必要 |

Hyperbandの議論には、構造による収束速度の違いで、最終的には良い候補が早期に不利になる問題もある。
層数や幅を探索する今回では、段階予算と最終順位の相関、上位候補の取り逃しを複数seedで調べる必要がある。
[Hyperband §6](https://jmlr.org/papers/volume18/16-558/16-558.pdf)
将来pruningを使う場合も、最小学習量、評価回数、猶予、比較する資源軸、誤除外の確認方法を事前定義する。

現時点のOptuna公式資料は5.0.0。single-objectiveではmultivariate TPEが既定で有効になる条件があり、古い4.xの説明を混ぜない。
`group=True`による条件付き空間の扱いはexperimentalである。
採用時は版とsampler設定を明示する。PyPIのPython3.14対応表示は依存全体の導入・数値・再開検証を代替しない。
[Optuna metadata](https://pypi.org/project/optuna/)

samplerが失敗trialを無視しても、campaign側は失敗・不適合・途中打切りを保存する。
OOM、期限切れ、契約拒否、輸送失敗を「低い勝率」として学習データ化しない。
中間値を同じstepで比較する[MedianPruner](https://optuna.readthedocs.io/en/v5.0.0/reference/generated/optuna.pruners.MedianPruner.html)も、そのstepが壁時計時間なのか受理決定数なのかを決めてはくれない。

TorchRLやSB3のtrainerへ置換することは、HPO導入の必須条件ではない。
TorchRLの[PPO教材](https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html)は参照実装として照合するが、今回の可変候補、終局return、本人可視境界、endpoint TBPTT、保存形式へそのまま移植しない。
まず現在の数値処理に閉じた設定を追加し、外側でtrialを管理する最小構成を設計候補とする。

## 強さを評価する方法とデータ分離

先に、最適化する目的を選ぶ。候補は「現在と同じ壁時計予算内での強さ」と「同じ受理決定数から得られる強さ」であり、順位が一致する保証はない。
現行運用に近い主指標としては前者が自然だが、これは提案であり、ユーザーの確定要求ではない。
両者の学習曲線を記録できれば、容量が大きくて学習が遅いのか、データ効率が悪いのかを分けて見られる。
同じupdate数でもepisode長・決定数は違うので、update数だけで同一学習量とはしない。

自然終了試合のscoreを `(wins + 0.5 × draws) / natural_games` と定義し、相手ごとのscoreを事前固定した重みで集約する案を推奨する。
必ず完了率・先後別・相手別・失敗内訳を併記する。欠けた対戦ペアを除いて有利な比較を作らない。
学習中のself-play勝率は相手も変化するため、学習診断として保存し、設定間の強さを選ぶ集約scoreへ混ぜない。
Random、heuristic、資格を満たす凍結モデルから相手panelを事前固定し、上位候補間の直接対戦も確認する。
panelと重みは評価者側の設定であり、相手IDをpolicy特徴に渡さない。
対戦相性が循環する場合は単一順位では表しきれないため、上位候補の対戦表と不確実性も残す。
これは一般的なmultiagent評価の論点であり、GodFieldで強い非推移性を観測したという主張ではない。
[Rowland et al. 2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/510f2318f324cf07fce24c3a4b89c771-Paper.pdf)

| 乱数・データ | 役割 | 分離方法 |
| --- | --- | --- |
| HPO sampler seed | どの設定を提案するか | 学習seed・GameSeedと別に保存 |
| 学習seed | 初期化、shuffle、方策sampling、学習scheduleの系列 | 候補間で事前指定した集合を共有。最終確認には探索未使用seedを用意 |
| 対戦GameSeedと先後ペア | 評価シナリオのばらつき | 候補間で共通化し、先後を交換。train/tuning/confirmation/testの重複禁止 |
| 評価policy sampling seed | 同じ状態での確率的行動の変動 | GameSeedからpolicy入力を作らず、独立streamを版付きで管理 |
| episode・seed・Simulator fork祖先group | 元データ内の依存 | 同じgroupをsplit間に分けない。両視点・繰返し試合を保持 |

ここで共通祖先とはSimulatorのgame fork由来であり、モデルcheckpointの来歴とは区別する。
学習済み方策を未使用GameSeedで評価する正常なvalidationまで、同じcheckpointを使うことを理由にtrainへまとめない。

既存の32 validation seedペアとLightGBMのvalidation L2は既に開発で参照されている。
これらはtuning側として扱い、新しいconfirmationと最終testを別に用意する。
新しい対戦seedだけでなく、選定に未使用の学習seedでも再学習して確認する。
confirmationの結果を見て設定を修正した場合、その集合も開発に使ったことを記録し、未使用だったことにしない。

区間推定では、先後・反復の同一GameSeedをまとめる既存のcluster処理を維持する。
複数学習seedの比較では、そのrunごとの変動も必要である。多数の試合を一つにプールして、1学習seedを多数回学習したように見せない。
学習seed別のscoreと差、対戦seedペアの区間を報告し、必要なら学習runと対戦clusterの両方を扱う再標本化を設計する。
「3 seedだから十分」「5 seedなら統計的に確定」とはしない。必要な精度・実用差・費用に応じて増やす。
[RLの不確実性評価](https://arxiv.org/abs/2108.13264)

最終testの試合数はvalidationから事前に決め、test結果で延長・短縮・設定変更を行わない。
既存の64-pair pilotから256〜2048 pairsを決める規則を、新しい複数相手panelへどう適用するかは後続設計で明示する。
学習・評価ともcategorical samplingを基準とし、評価だけgreedyへ変える場合は別の推論方式として比較する。

## 計算予算と段階的な実施案

1. 現行設定を再現できるConfigと来歴の設計を先に行い、探索してよい領域を限定する。
2. 少数の構造で数値・保存・最大履歴・資源の確認を行う。短いsmokeはこの目的に使う。
3. 各モデルの基準設定と代表的な構造候補で、事前指定した複数seedの学習曲線と工程別費用を得る。
4. 資源と曲線を見て共同探索空間、trial予算、評価頻度を凍結する。Randomを対照に、TPE等の採否を決める。
5. 構造・学習率群、更新量群、系列/credit群などを探索し、組合せを再確認する。早期に一つの構造へ固定しない。
6. 上位設定と現行基準を未使用の学習seed・confirmation対戦で比較し、設定を固定して最終testへ進む。

1学習seedで広く探索する案は低費用だが、優劣の誤判定リスクが大きい。
今回の初期学習曲線と採用判断には複数seedを割り当てることを優先する。
pruningを導入する場合、早期に低く見えた候補の一部も全予算で走らせ、誤除外を測るための予算を確保する。
予算が足りなければ探索範囲と結論の範囲を狭め、評価の不確実性を隠さない。

予算は次の項目を別々に計上する。

`campaign cost = 全trialの収集・前処理・更新・保存・回収 + 探索中評価 + confirmation再学習・評価 + 最終test + 共有準備費`

現行PPOの上限は7200秒、batch受付5400秒、実処理7140秒。
同じ予算で比較するなら、構造変更や休止によってこの元の期限を延長しない。
既存のnormal resumeは停止中も期限が進むため、数日後に良い候補だけを再開する一般的なHPO手順へそのまま使えない。
短予算runを新設するなら探索用の別protocolとして設計し、元の正式条件を満たしたrunと同一視しない。

過去の会話の例「12設定×1 seed＋3設定×追加2 seed」は18 run/model、3モデルで54 run、上限枠108時間という算術である。
これは必要な探索量の推定ではなく、評価・確認・実装費を含まない例である。
採用する試行数は、候補一つの費用、必要な学習seed数、評価精度、総予算から逆算する。

LightGBMは元データ生成費を別途一度だけ計上し、各fitの前処理・fit・保存費を記録する。
PPOとの比較に同じ「2時間fit」を使っても、総生成費が等しい比較にはならない。
またモデル種別間の公平性には、単一run予算だけでなく探索全体に使った予算も関係する。
ハードウェア、worker数、外部負荷を固定・記録し、モデル候補の並列実行による資源競合を避ける。

## 実装へ進む前の設計事項

| 項目 | 必要な変更・条件 |
| --- | --- |
| Config | 探索対象だけを型付きで公開。モデル別の条件、相互制約、有限値、資源上限を検証 |
| 数値処理 | ハードコードの幅・heads・loss係数・optimizer設定と、cache/parity/受付計算を整合させる |
| inference manifest | 推論に必要な公開model recipe、構造、初期化、特徴変換、runtime、tensor/hashを保存。学習管理情報をpolicyへ渡さない |
| training/resume記録 | optimizer、parameter groups、schedulerと正規化の状態、RNG、Config identityを型付きで保存。既存schemaを黙って再解釈しない |
| campaign記録 | 探索protocol、評価条件、選定規則、sampler状態とtrial参照を管理。推論manifestに混ぜない |
| resume | 同じ設定の通常再開と、別設定のfresh trialを区別。旧weightsを初期値に使う実験は別のwarm-start protocol |
| campaign管理 | trial設定・提案順・seed集合・出典・status・費用・失敗理由を保存。samplerだけを成功判定のauthorityにしない |
| source資格 | 失敗親の前半checkpointや良かったbatchを正式比較・offline学習に使わない。pruned status導入時も資格を別定義 |
| 依存 | Optuna等の追加はarchitecture gate。ベースラインCLIのoptional dependency分離を維持 |
| 検証 | 初期設定の回帰、構造別shape、候補順とcritic独立、未来参照禁止、full/cache/split、microbatch等価、保存再開、期限・失敗・旧schema拒否 |

MLPはinference schema1、系列はinference/resume/rollout schema2、LightGBMはinference schema2で、それぞれ閉じた検証がある。
設定欄を追加するだけで探索可能にはならない。[MLP保存](../../src/gf_models/mlp/model.py)、[系列保存](../../src/gf_models/sequence.py)、[optimizer再開](../../src/gf_training/ppo_state.py)
必要な構造変更はAGENTS.mdの設計レビューを経て実装し、正しさと設計適合を別々に確認する。

PyTorchは同じseedでも異なるrelease・platform・CPU/GPU間の完全再現を保証していない。
seedだけでなくコード、依存、backend、Simulator binary/spec、Config、source/checkpoint digestを記録する。
[再現性の公式説明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

## 未確定の事項と調査の限界

GodFieldで最適な幅・深さ・学習率・W・探索器は、今回の文献調査では決定できない。
次の設計で確定する事項は、主目的が同一時間か同一データ量か、推論p95/メモリ上限、総探索予算、相手panelと重み、学習seed/対戦seedの分離、実用差と必要な評価精度である。
候補範囲はこれらと予備実験から絞る。2層固定や同じ幅のactor/criticは比較基準として残し、唯一の正解とはしない。

この調査は2026-11-01時点の主要一次文献と公式資料の目的別レビューであり、全論文を網羅したsystematic reviewではない。
MuJoCo等の連続制御、off-policy再帰学習、大規模Transformerの結果は、GodFieldの可変離散候補・疎な終局報酬・自己対戦へ直接一般化できない。
一方、共同探索、seed分離、独立評価、設定と費用の完全な記録は、今回の探索計画へ取り入れる根拠がある。

## 調査の検証記録

- 実装・設定・ADR・実験記録のread-only照合と、native libraryをimportしないpackage metadata確認を実施。
- 一次文献・著者の論文ページ・公式ライブラリ資料を参照し、本文の主張の近くに直接リンクを付した。
- 学習、探索、Simulator対戦、依存追加、アプリケーションテストはNOT RUN。本変更は資料とREADMEのリンクだけを対象とする。
- `bot_reviewer`の独立した学習・評価レビュー: READY、blocking findingなし。自己対戦scoreは診断専用とし、Simulator fork祖先とモデル来歴、終局return教師と禁止情報の区別を明確化した。
- `bot_architecture_reviewer`の独立した境界・ライブラリ仕様レビュー: READY、blocking findingなし。共通候補幅690と系列head入力818、推論manifestと学習/探索管理記録の所有を明確化した。両レビューの受入対象は調査資料のみ。
- 文書検査: `git diff --check` PASS。`.venv/bin/python`による相対リンク存在確認19件、READMEからの導線、code fence対応確認PASS。外部参照先は35 URLで、本文に用いた一次資料の記述を照合した。
