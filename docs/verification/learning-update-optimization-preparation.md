# 学習更新の最適化：適用・測定の準備調査

調査日：2026-11-17。対象ソース：`a06cb18`、学習コードは前回計測の
`17b2c1c` と同一。これは実装前の情報整理であり、実装設計の承認や高速化の
実測結果ではない。アプリケーション、学習条件、Simulator、依存関係は変更していない。

## 1. 判断に使える情報と、まだ測れていない情報

モデル構造・PPO・更新予算を維持した候補は、同期／指標回収の集約、不変入力の
Tensor一括化、論理minibatch内のTensor microbatch化、同一step内のframe計算共有。
現在の数値更新ループには、これらを検討する具体的な処理箇所がある。

一方、frame encoder、時系列層、backward、GPU同期の個別実時間は未計測。
以下の処理件数やpadding量から速度改善率を推定しない。

前回の現行方式3実行の中央値、2更新合計：

| 区間 | MLP | GRU | Transformer |
| --- | ---: | ---: | ---: |
| 更新要求全体 | 6.271秒 | 22.628秒 | 22.188秒 |
| 数値更新ループ | 5.885秒 | 22.236秒 | 21.716秒 |

数値更新時間は[HpoLearner._update_numeric](../../src/gf_training/hpo_numeric.py)の
host経過時間であり、CPU準備・GPU投入・待ち・forward/backward・Adam等を含む。
GPU演算だけの時間ではない。`batch_completion_wait_seconds`も、受信・数値更新・
同期・入力解放ACK待ちを含むため、純粋なIPC時間として扱わない。
基礎データは[前回の比較記録](inference-performance-comparison.md)にある。

## 2. 使用できる固定入力と開始状態

前回18親の厳密な原本・numeric・終了receipt監査済みデータのうち、round1の現行方式を
各モデル1親ずつ、各2batch調査した。入力はtrain側の診断用途とし、親のincomplete状態や
Source資格を変更しない。3モデル6個のNumeric6 bundleを既存`read_json`／`inspect_bundle`
で確認した。独自のSQLite／archiveデコーダは追加していない。

| モデル | 親run |
| --- | --- |
| MLP | `21cfafb7-56f5-4998-be28-670cbf3caba3` |
| GRU | `67e1ee79-c76f-45e4-8511-b607f78def19` |
| Transformer | `5ec28c6e-36ac-4f59-bee7-a1cf32eef03e` |

親は `runs/inference-comparison-v1/parents/<run>/`。
全パス・hash・実効recipe・選択対象・開始時shuffle状態は
[`input-state-inventory.json`](../../runs/learning-update-preparation-20261117/input-state-inventory.json)、
形状・重複・padding集計は
[`workload-summary.json`](../../runs/learning-update-preparation-20261117/workload-summary.json)に保存した。

| 再実行対象 | 必要な開始状態 | 必要な入力 |
| --- | --- | --- |
| 更新1 | `artifacts/000000`のG0、Adam step0 | `numeric/batch-000000` |
| 更新2 | `artifacts/000001`のG1、Adam step2 | `numeric/batch-000001` |

`inference.json`／weightsだけでなく、`resume.json`、Adam moment／parameter別step、
状態の有無、Torch CPU/CUDA RNG、PCG64 shuffle状態を復元する。G0にはAdam状態がなく、
G1には各モデルの使用parameter状態がある。各測定の前に同じ開始状態を戻す。
更新2をG0で測ったり、同じlearnerを復元せず繰り返したりしない。

現在は全G0/G1のproducer source digestが現ソースと一致することを確認した。
候補コードへの変更後は通常restoreが不一致を拒否し得る。旧resumeを新ソースとして
書き換えず、診断用importの互換条件と旧／新ソースを明示する必要がある。

読み取りには[read_hpo_rollout](../../src/gf_training/hpo_rollout.py)、開始状態の検証・復元には
[restore_learner](../../src/gf_training/hpo_artifacts.py)がある。ただし前者が作る公開
`HpoRollout`は、実際の`update_owned`に必要なsingle-use `OwnedNumericBatch`ではない。

## 3. 実際の入力形状

各分布はmin／median／nearest-rank p95／max。

| モデル・batch | 学習対象数 | 本人履歴長 | intervalイベント数 | 候補数 | 候補行合計 |
| --- | ---: | --- | --- | --- | ---: |
| MLP/GRU 0 | 181 | 1/7/20/29 | 0/7/12/21 | 1/11/1015/1552 | 18,589 |
| MLP/GRU 1 | 133 | 1/6/21/27 | 0/7/12/17 | 1/11/1278/2629 | 15,993 |
| Transformer 0 | 175 | 1/7/21/29 | 0/7/13/21 | 1/11/1165/1552 | 19,962 |
| Transformer 1 | 135 | 1/6/21/27 | 0/8/12/17 | 1/11/1278/2629 | 16,013 |

実効設定は全てepoch1、論理minibatch128、TBPTT32、各更新2回のAdam step。
基底設定の`training.epochs=4`と混同せず、実際にmaterializeされたHPO recipeと
保存済み`metrics.epochs=1`を使う。

今回の全履歴は32以下であり、detached prefixを必要とするケースはない。
実装確認には別途、TBPTT境界を跨ぐ履歴が必要。実データの最大29に合わせて一般の
1024上限や32の勾配範囲を短縮してはいけない。

decoded配列全体はMLP/GRUがbatch0=4,612,041B、batch1=3,575,851B、
Transformerが4,604,019B／3,612,842B。raw JSON、派生token、学習用vectorを含む。
array keyは重複しないが、raw記録と派生tokenは意味が重複する。
この容量をGPU activationや保持すべきgraphの容量と見なさない。

## 4. 固定重みの範囲内での重複とpadding

保存済みG0/G1のPCG64状態から実際のpermutationを復元し、元の論理minibatch内で
同じepisode/frameのprefixキーを数えた。

| モデル | 要求されたprefix frame延べ数 | 各step内のunique frame数の合計 | 重複出現数 |
| --- | ---: | ---: | ---: |
| GRU | 2,508 | 481 | 2,027 |
| Transformer | 2,444 | 491 | 1,953 |

これはframe encoderの重複を減らせる件数の上限。時系列処理全体やbackwardの再利用量、
速度改善率を意味しない。MLPは現在frameだけを314件処理するので、このprefix重複数を
MLPの削減余地として使わない。勾配graphを共有しても各endpointの損失寄与は維持する。

履歴長を完全一致group化しただけでは、候補数のpaddingが大きい。
以下は資源による分割前、group内最大候補数への単純paddingの行数見積り：

| モデル・更新/minibatch | 件数 | 候補行：実数→padding後 | padding後に占める追加行 |
| --- | ---: | ---: | ---: |
| MLP/GRU 1/1 | 128 | 10,888→44,203 | 75.37% |
| MLP/GRU 1/2 | 53 | 7,701→23,498 | 67.23% |
| MLP/GRU 2/1 | 128 | 15,950→79,413 | 79.92% |
| MLP/GRU 2/2 | 5 | 43→43 | 0% |
| Transformer 1/1 | 128 | 15,107→86,947 | 82.63% |
| Transformer 1/2 | 47 | 4,855→15,972 | 69.60% |
| Transformer 2/1 | 128 | 15,733→76,352 | 79.39% |
| Transformer 2/2 | 7 | 280→308 | 9.09% |

割合の分母はpadding後の全行数。例えば82.63%は候補行数が約5.76倍になる見積り。
候補参照slotのpaddingやactivationは未算入。履歴長だけでなく候補数と同時メモリを
含めたgroup化・分割が必要となる。

旧[sequence_numeric.microbatches](../../src/gf_training/sequence_numeric.py)には、
論理minibatch内のprefix/suffix長group化、資源分割、microbatch件数での損失重み付けがある。
これは参考実装であり、異なるデータ／モデル契約のHPO Model4へそのまま差し込めない。

## 5. 実装設計で確定する必要がある制約

基準：[ADR0022](../adr/0022-gru-transformer-ppo-plan.md)、
[ADR0023](../adr/0023-structured-encoders-and-diverse-search.md)、
[ADR0037](../adr/0037-ppo-hpo-foundation.md)、
[ADR0043](../adr/0043-hpo-producer-owned-data.md)。

- 元の選択endpoint、PCG64 permutation、論理minibatchの所属と末尾件数を維持する。
  その内部だけでgroup化する。microbatch平均損失は`m/N`で重み付けし、clip／Adamは
  元の論理minibatchにつき1回。epoch全体の再配列や学習対象削減は行わない。
- GAE、advantage正規化、損失係数・本人視点、各endpoint1回の損失を維持する。
  候補の順序・mask・endpointごとの正規化を保持し、paddingは選択・損失へ混入させない。
- endpointごとのcurrent-weight prefix no_grad／detachとsuffix gradを維持する。
  異なる切断位置を単一の時系列graphで代用しない。重み依存の共有はAdam前までとし、
  本人episode／viewer、位置符号、causal mask、criticの候補非依存性を保つ。
- 空イベント／padding分岐で、未使用parameterの`grad=None`をゼロ勾配へ変えない。
  Adamの状態生成・moment・parameter別stepまで意味が変わるため、global step一致だけでは不足。
- 推論用再利用bufferはno_grad限定。学習用inputは最後のbackward／checkpoint再計算まで
  上書きしない。共有graphは集約backwardか明示的な保持・解放設計が必要。
- finite input／output／loss／gradient／parameter／metric検査を維持する。
  明示同期を消すだけではPython float取得やfinite条件の同期は残る。検査の移動先と
  エラーを観測する境界を明示し、失敗を成功ACK／保存／継続に変えない。
- Wire7のendpoint/context/suffix counterは現在の論理的意味を保持する。
  dedup後の実encode数、microbatch数、転送回数は別の診断値とする。
- `OwnedNumericBatch`は同じSource account・世代・snapshotに束縛されたsingle-use値。
  view、bank、FDと予算は実際の最終利用まで保持し、通常の同期・入力解放ACK・snapshot・
  保存・receiptを維持する。状態をPERSISTEDへ巻き戻して反復しない。
- 現在はIEEE FP32、決定的設定、GRUのCuDNN無効、SDPA MATH。
  backend／AMP／TF32／compileは明示的な数値実行方針の別比較とする。

## 6. メモリ上限と不足している見積り

既存上限：RAM24GiB、CUDA12GiB、Source2GiB、numeric512MiB、collector2GiB、
parent/save各1GiB、単一Tensor64MiB。128件を一度にGPUへ載せる保証はない。

[resource_estimate](../../src/gf_models/structured_config.py)は1endpointの推定。
新しい見積りでは共通のparameter／Adam／context等は1回、同時に生きるprefix/suffix／
event／attention／候補／padding／backward／checkpoint／共有frame graphは全件分数える。
packed CPU/GPUコピー、旧bufferと成長先の同時保持、常駐actor／save／SYNCも含める。

現在のeager演算をB件へ拡張した中間gatherだけでも
`B × candidate_chunk × max_positions × actor_hidden × 4B`。
今回の256×18×192ではB16=54MiB、B19=64.125MiBで単一Tensor上限を超える。
またheads4・履歴1024のFP32 attention score `[4,4,1024,1024]`は1層で64MiB。
2層ならこれらのscore相当だけで計128MiBとなり、他の中間値やbackward分は別にある。
これは形状の算術例であり、B16/B4の全体受付やOOM回避を証明しない。
候補が少ない場合は形状が小さく、上限は実際のgroup形状から計算する。

候補数、履歴、event長、保持graphを含むjoint admissionを新設し、実際のlearner予約と
[常駐各phase見積り](../../src/gf_models/structured_runtime.py)へ同じ式を通す必要がある。
旧sequence-PPOの4096candidate slots等を、現HPOの受付済み上限として流用しない。

## 7. 測定入口と既存toolの再利用範囲

二つの測定は役割が異なる。

| 測定 | 分かること | 必要な準備 |
| --- | --- | --- |
| 同じbatch／開始状態の学習core | 準備、encoder、時系列、候補、loss、backward、同期等の差 | 同じowned/public token経路、全状態復元、診断専用adapterの設計 |
| 本来のcoordinatorによるfresh実行 | 収集、FD転送、保存、常駐worker、終了まで含む効果 | 同じ実効設定、真正なGPU job／Source／snapshot／receipt、別出力root |

公開`HpoRollout`の通常forwardには検証／表現構築があり、実経路の`_forward_owned`と
そのまま同じ時間にはならない。異なる経路同士を22秒の比較基準にしない。
同じTensor coreに切り出す場合は、除外した準備・検証の範囲を明示し、実経路でも確認する。

現clean learnerは[固定のoriginal creator](../../src/gf_training/hpo_owned_process.py)から
`-E -m gf_training.hpo_owned_worker`で起動する。original bound factory／creatorの
差し替えはguardが拒否する。起動moduleはそのcreator内に固定されており、診断用module／
bootstrapを選ぶ既存入口はない。guardがstdlib呼出しや引数への全てのglobal patchを
検出する保証ではない。Wire7 workerには既存`--profile`入口がなく、親の診断callbackだけでは
子のGPU kernel／backward内訳は取得できない。original creatorを維持した明示的な
診断hookが必要。stdoutとclosed Wire7 metricsへ自由なtrace項目を混ぜない。

前回の`benchmarks/inference/replay.py`は推論専用。G0 weightsだけの入力JSONでは
学習更新2を再現できない。`actual.py`のactor置換／96MiB actor追加予約も、今回の
学習候補へ無条件に引き継がない。既存の実行・終了所有者とhost observerは再利用できる。

18親はincompleteのまま。診断input/stateの厳密検査は可能だが、通常のowned cold resumeは
paused親を要求する。incomplete親の再開、producer capability偽造、旧記録の書換えはしない。

## 8. 比較・計測条件として用意するもの

以下は既存条件を踏襲する測定候補であり、この資料だけで実行設計を承認したものではない。

1. 最初に現行の学習内訳を、少量の診断windowで取得する。CPU host経過・process CPU、
   CUDA event区間、kernel／copy／同期回数、allocated/reserved peak、PSS/RSSを分ける。
2. 比較variantを、同期／指標集約、不変input一括化、microbatch化、frame共有の順に
   分離する。必要な依存変更は明示し、backend変更まで一度に混ぜない。
3. 各反復前にG0/batch0またはG1/batch1の重み・Adam・全RNG・shuffleを復元する。
   warmupも勾配／RNG／optimizerを汚染させない。同じlearnerにTimerを繰り返すだけでは不十分。
4. 冷起動・初回allocation／compile、通常反復を分け、近接・順序交替の対照を置く。
   profiler有無を比較の二つの方式にしない。採用判断はprofilerなしの時間とメモリで行う。
5. 実経路は192幅・2層、seed17、4worker／同時2試合、16試合×2更新、epoch1、D1M、
   wall600秒と元の上限を維持する。新しい出力rootとcode/config/input/state hashを固定し、
   親は健康なincomplete/update_capとして扱う。前回時間を別ソースの対照へ混ぜない。
6. 終了後に元のSource／numeric／receipt監査を実施する。計測中にテスト・監査・レビューを
   並行しない。学習品質／HPO完了／final testの結果へ転用しない。

必要な計測ラベル：input取得／packing／HtoD、state-card-hand-event encoder、
temporal prefix、temporal suffix、candidate/logprob/value、loss、CPU指標回収・finite検査、
backward、clip/Adam、最終parameter検査／同期、入力退役ACK、snapshot／保存。
checkpointによる候補再計算はbackward内に含め、forwardへ二重計上しない。

GPUイベントの区間はhost投入待ちによるidleも含む。実kernel時間はCUDA activityから別に
扱い、host wall−CPUや別runのkernel時間を引いて純粋な待ち時間を作らない。

推論56decisionのshape付きtraceが約103MiBだったため、学習全体を無条件に連続記録する
設計は避ける。長短履歴・候補数・空イベントを代表する小さいwindowを選び、trace／
profiler保持参照の上限と欠測扱いを決める。shape/stack記録は別診断runとする。
PyTorch公式もshape記録の追加参照・コピー・時間コストを明記している。
[Profiler 2.14](https://docs.pytorch.org/docs/2.14/profiler.html)

## 9. 実装前に閉じる判断と、確認するケース

未決定：Tensor microbatchの表現・group化方針、joint memory式、検査／指標回収の境界、
frame graphの保持・破棄、候補execution identityとresume互換、診断adapter／worker hook。
これらは独立した事前architecture reviewの対象。今回の情報監査はその承認ではない。

確認対象は、通常の損失重みとoptimizer stepに加え、末尾53/5/47/7件、履歴32超のdetach、
候補1件・256境界・大量候補、空event／空hand、padding除外、`grad=None`とAdam状態、
backwardまでのbuffer寿命、有限値失敗時の非公開・終了、OOM／期限／FD退役。
既存の比較・resume・勾配境界テストを再利用し、新規の出力完全一致や行動列一致を
性能比較のゲートとして追加しない。変更された加算順の数値確認方針は設計で明示する。

## 10. 実行環境と調査の検証

Python3.14.10/GIL、NumPy2.5.4、Torch2.14.1+cu132、CPU affinity0–15、
cgroup CPU/memory上限は`max`。hostのread-only `nvidia-smi`でRTX5070Ti16,303MiB、
driver610.88、共有使用2,677MiB／利用率3%を観測した。これは予約や測定中の専有を意味しない。
CUDAはsandbox外のhostで利用する。現状態は各新測定の前に取り直す。

PASS：3モデルのG0/G1 inference/resume bundleとhash、全6numeric metadata／descriptor、
現在のsource digest一致、開始step0/2、実効epoch1/minibatch128/TBPTT32、shuffleに基づく
group集計を確認。アプリコード、原本、設定を変更せず、GPU計算／訓練／テストは実行していない。
公式資料確認とread-only情報監査を行った。新しい速度／品質／容量合格は主張しない。

独立verifierは表・byte合計・重複数・paddingの分母を確認しPASS。
architecture reviewerはcreator guardとattention容量の表現修正後、設計入力として使用可能と
確認した。この情報監査は、実装設計や新しい測定入口の承認を意味しない。

公式根拠：
[CUDA実行とevent計時](https://docs.pytorch.org/docs/2.14/notes/cuda.html#asynchronous-execution)、
[Autogradのbuffer制約](https://docs.pytorch.org/docs/2.14/notes/autograd.html#in-place-operations-with-autograd)、
[GRU batch/packed入力](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html)、
[可変長packing](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.pack_padded_sequence.html)、
[checkpointの再計算](https://docs.pytorch.org/docs/2.14/checkpoint.html)、
[compile graph break](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html)。
