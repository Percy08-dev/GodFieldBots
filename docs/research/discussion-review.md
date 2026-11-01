# モデル調整と手札エンコーダーに関する会話の独立整理

作成日: 2026-11-01。照合したBots revision: `cb15c2f974e36c32faeee6bf3b7940dd37b605b1`。
Simulatorの契約pinと照合時HEAD: `ffcffa0966dbf6d7ce1d982f4ee6444e291d42d6`、catalog `public-ja-v2`。

対象は、モデルのハイパーパラメータ調整と、手札の組合せ探索から入力表現・エンコーダーまでの議論である。
新規の独立エージェント`discussion_hpo_audit`（bot_reviewer）と`discussion_representation_audit`（bot_architecture_reviewer）が、会話の要点を実コード・仕様・一次資料と照合し、Markdown原稿と訂正事項を作成した。Rootが統合と追加診断を担当した。
エージェントへの引継ぎには、ユーザーの重要な発言・訂正とアシスタントの提案を区別した要約を用いた。

これは研究・議論の整理であり、新しい構成や実験条件を採用するADRではない。
本文の「確認済み」は現実装または資料で確かめた事実、「ユーザーの方向性」は会話で示された意図、「候補・仮説」は有効性をまだ実験していない案を指す。
HPOの詳細な文献調査は[既存の調査資料](hyperparameter-search.md)を参照する。同資料以降の訂正と、手札・エンコーダーに関する追加の議論は本資料にまとめる。

## 1 モデルのハイパーパラメータ調整

### 1.1 現在地

体系的なHPOと探索ライブラリの選定は未完了である。
現段階は、固定設定での実装・検証と、一部モデルの正式学習が完了した状態にある。

| 対象 | 確認できた事実 | その結果だけでは判断できないこと |
| --- | --- | --- |
| MLP PPO | seed17/29/43が123/144/145更新で完了。受理決定数24,439/27,985/29,099 | 最適設定、十分な収束、他モデルより強いこと |
| 固定データ | 81,523 train／2,260 validation decisionsを凍結 | 新しい親runや系列モデル親を任意に追加できること |
| LightGBM | observation/history各3 seedの6正式fitが完了 | 6種のhyperparameter設定を探索したこと、独立した6データセットでの汎化 |
| GRU／Transformer | 2更新・正常境界resumeの実対戦smoke、数値・性能検証が完了 | 正式な多seed学習・HPO・強さ比較の完了 |
| 最終test | 未実施 | 既存validationから確定的な最終性能を得られること |

根拠: [catalog-v2実験記録](../verification/catalog-v2-adoption.md)、[系列モデル検証](../verification/sequence-ppo.md)。
独立監査では親`run.json`・fitの`training.json`要約も照合したが、元journal/Replay全件の適格性検証を再実行したわけではない。
古い失敗親runに関する記述を現在の全体状況と混同しない。新契約の完了済み実験も、古い失敗親runを適格にするものではない。

LightGBMのseed17同等性再実行は、正式6条件に加える7番目の比較条件ではない。
同じ特徴条件の3 seedは同じbest iterationとvalidation L2を得ている。
現設定でfit seedだけを増やすことを、データ分布に対する独立した汎化検証と数えない。

### 1.2 ユーザーの方向性と未決事項

ユーザーは、モデル構造・構成・最適化・学習制御の調整を検討し、幅と層数の相互作用、文献・ライブラリ調査、ベースラインの増強を指摘した。
幅を一度選んで固定し、その後は層数だけを調整する方針は、今回の構造探索の意図を十分に満たさない。

幅・層数・学習率を共同で扱うことには根拠がある。共同探索は全組合せの総当たりを意味せず、条件付きの候補抽出や、狭い仮説を確かめる比較も含む。
一部の軸を予算上固定した実験は可能だが、その場合の結論は固定した構造の範囲に限定する。
[Andrychowiczらの実証研究](https://arxiv.org/html/2006.05990v1) §2は、相互作用する項目をグループ化し、各群にAdam学習率を含めている。
これは連続制御の結果であり、具体的な幅・層数・係数の最良値をGodFieldへ移植する根拠ではない。

未決定なのは、総探索予算、試行数、seed数、主目的、相手panel、探索器、停止条件、推論時間・メモリ制約である。
「提案された」「APIが利用できる」「実装された」「実験で有効だった」を区別する。

### 1.3 現行設定とライブラリ

MLPはactor/criticが別々の128幅2隠れ層、GRUはhidden128・1層、Transformerは幅128・2層・4heads・FFN256である。
これらは現在の比較基準であり、調整で選ばれた最良構造ではない。
PPOのlr3e-4、4 epochs、論理minibatch128等はコード内固定で、モデルの構造もConfig検証で固定されている。
設定ファイルの値を変えるだけで探索できる状態ではない。[Config](../../src/gf_models/config.py)、[PPO learner](../../src/gf_training/ppo_numeric.py)、[共通loss](../../src/gf_training/ppo_math.py)

**HPOライブラリは未選定である。** Optunaは候補として調査済みだが、採用・依存追加・互換性試験・探索実績はない。
独立監査のmetadata確認ではOptuna、Ray、Hyperopt、DEHB、hpbandsterは未インストールだった。
現在利用するPyTorch/LightGBMと、今後選定する探索管理ライブラリは役割が異なる。

[Optuna 5.0.0](https://optuna.readthedocs.io/en/v5.0.0/reference/samplers/generated/optuna.samplers.TPESampler.html)は条件付きの探索等を扱えるが、runの適格性、元の期限、失敗理由、評価分離を自動で定義しない。
ライブラリの採用と、Random/TPE等のsampler、途中打切りの採用は別に判断できる。
Python3.14対応metadataも、実環境の導入・再開・失敗処理の検証を代替しない。

### 1.4 会話中の案に対する訂正と留保

| 説明・案 | 独立した評価 | 本資料での扱い |
| --- | --- | --- |
| 幅を探索し、深さは後から調べる | 限定的な予備比較には使えるが、相互作用を見落とす | 幅・深さ・学習率の共同探索を検討し、固定した軸を記録する |
| 初回12設定を1 seedで絞り、上位2設定と基準を追加seedで確認 | GodFieldでの選定精度・費用の実測がない | 仮の配分例。早期に捨てた候補は後段の多seed確認では救えない |
| 54 run・108時間 | 前提付きの算術であり、総費用の見積もりではない | 初回12設定に基準を含む場合のみ18 run/model。評価・確認・準備費は別 |
| 3または5 seedで確認 | 必要な精度を保証する数ではない | 学習run間の変動と実用差を見て反復数を決める |
| 同じ2時間で比較 | 現行の2時間はwall上限。主目的として未決定 | 同一時間での強さと同一データ量での強さを区別する |
| ベースラインを強化してから探索する順序 | 妥当な候補だが、強化を必須条件にする実証はない | 既存対照を保持し、相手追加の価値と学習分布変更を測る |
| 凍結MLP/LightGBMを強い相手にする | 完了・来歴の健全性と強さは別 | 「適格な凍結相手」と呼び、強さと多様性は対戦で確認する |

初回12設定に基準が含まれないなら、`(12+1)+3×2=19 run/model`、3モデルで57 runになる。
候補の重複、基準の再実行条件、追加seed数を定義してから計算する。
正式MLP親の実時間は約90.4〜90.6分で、7200秒wall、5400秒受付、7140秒executionという制限の下で終了した。常に2時間学習したという意味ではない。
normal resumeでも停止中に元の期限が進むため、数日後に有望候補だけ再開する手法へそのまま使えない。

Randomは有用な対照手法だが、最良の探索器と決定したわけではない。[Random search原論文](https://www.jmlr.org/papers/v13/bergstra12a.html)
RLではtuning seedへの過適合があり、探索用と確認用の学習seedを分ける根拠がある。[Eimerら](https://proceedings.mlr.press/v202/eimer23a.html)
構造による収束速度の差は、途中打切りで最終的に良い設定を取り逃す問題にもつながる。[Hyperband](https://jmlr.org/papers/v18/16-558.html)
これらは、1 seed探索が常に禁止、3 seedなら十分、特定手法がGodFieldでも勝つ、といった一律の規則ではない。

### 1.5 ベースラインと評価条件

現在のRandomは、生成された候補間で一様に選ぶ。全合法行動から一様に選ぶ方策ではない。
heuristic-v1も同じ候補集合を使い、その`content_score`は候補削減とbeam順位にも共有される。
この関数を全体で変更すると、対戦相手だけでなく全モデルが選べる行動も変わる。[baseline](../../src/gf_models/baselines.py)、[候補生成](../../src/gf_common/actions.py)

状態を考慮するheuristic-v2は改善仮説であり、強くなる保証はない。
方策の改善を測る比較では候補generatorを揃える。候補生成そのものを改善する比較では、その変更を独立した実験因子として記録する。
テーマ2で要求された共同設計を妨げるためにgeneratorを恒久固定する、という意味ではない。

評価相手に凍結モデルを追加することと、学習中の相手分布を変えることは区別する。
現行PPO収集はRandom2ペア、heuristic2ペア、過去snapshot4ペアであり、snapshot相手は学習に伴って変わる。
その自己対戦returnを、設定間の強さを選ぶ共通scoreへ混ぜない。[収集schedule](../../src/gf_training/collection.py)

独立した学習runのばらつきと、固定方策の対戦seedごとのばらつきは異なる。
試合数を増やしても学習runの独立反復にはならない。[Agarwalら](https://arxiv.org/abs/2108.13264)
既存validationや選定に使う短い対戦は、未使用の最終testではない。
[選定バイアスの研究](https://www.jmlr.org/papers/v11/cawley10a.html)、[RL実験設計](https://jmlr.org/papers/v25/23-0183.html)

LightGBMのvalidation L2は、収集された受理行動と自然終了return上の回帰誤差である。未選択候補の最適順位や対戦勝率を測る値ではない。
この点は会話でも留保しており、「L2改善を強さ改善と断定した」という誤りがあったわけではない。
凍結source・split・親runの適格性は維持する。[固定データ契約](../adr/0008-mlp-ppo-generations-and-fixed-data.md)

## 2 手札の組合せからエンコーダーまで

### 2.1 ユーザーの方向性

以下は、ヒューリスティック専用の改修ではなく、全モデルに関係する要求・検討方向として示された。

- 単体で使える行動を残し、高攻撃力だけでなく、低攻撃力、属性、資源温存、手札更新等の多様な組合せを提示する。
- 「最大要素数は決めないか、10要素程度までは許容」とし、現行の3要素固定を見直す。正確な長さ上限と資源上限は未決定。
- 手札には通常の手札と習得済み奇跡を含める。現在使用不能なものも所持情報として保持し、使用可否とは区別する。
- カード詳細を残し、長い入力を学習するエンコーダーで圧縮する案を検討する。
- State/Hand/Turn encoderの有無・組合せ、Transformerの追加前段encoderの有無、E2E学習を比較する。
- **探索範囲・提示範囲の拡張と、入力表現・エンコーダーは同時に設計する。** 実装責務を分けることと、独立に設計を確定することは別である。

後半ではさらに、組合せを事前に列挙してHand encoderへ渡さず、手札からカード間相互作用の潜在表現を作って後段モデルへ直接入力する案が出た。
この最終意図を、具体的なカード列を生成するdecoderの採用要求へ読み替えない。

### 2.2 現行の候補生成と固定スコア

現行経路は `本人可視入力 → CandidateSet → 固定数値encoder → 候補採点 → 選択したtyped Action` である。
方策は候補indexを返し、Simulatorへ送る具体的Actionに結び付ける。[公開型](../../src/gf_common/domain.py)、[候補生成](../../src/gf_common/actions.py)

`content_score`は、heuristic方策の選択に加えて、候補上限超過時の削減とbeamの枝選択に使われる。
固定スコアが低いために除外された候補は、後段の学習モデルから選べない。
**このスコア自体はモデルの数値入力特徴には含まれていない。** `scorer_version`等の互換性metadataとは区別する。

Git履歴では、採点式は最初のcommit `1719471`に入り、各行も同commitに帰属する。
闇の`+6`、要素ごとの`-0.05`、祈り`2`等の値について、確認した履歴・文書・関連テストに定量的な選定根拠は見つからなかった。
これは調査範囲で根拠を確認できなかったという結論であり、履歴外の根拠まで存在しないことの証明ではない。

現行generatorは公開ルールに基づく明示的な部分集合であり、全合法行動を列挙するものではない。
単独使用できる全カードを生成できているわけではなく、予約して残す対象もPrayや空応答等のbasic actionsに限られる。
したがって、単体行動の網羅性と、生成した単体行動の保持は別々に確認する必要がある。[ADR0005](../adr/0005-authorized-atomic-action-options.md)

既定値は最大3要素、beam64、最大512候補、最大20,000展開である。
加えてencoderの`max_elements`も2〜3に制限され、ニューラルモデルは現行encoder recipeを固定している。
10要素等への拡張は、探索の数字だけの変更では済まない。[設定](../../configs/actions/beam.toml)、[Config検証](../../src/gf_models/config.py)

組合せを多様化する際の案は、攻撃力の高低・属性/効果・消費資源・手札更新・順序ごとに代表候補を残す方法である。
最大と最小だけでは中間の有用候補を取り逃す可能性がある。完成候補の選別だけでなく、探索途中の枝の保持にも同じ問題がある。
具体的な枠数、重複排除、打切り、必須候補が上限に収まらない場合の動作は未決定である。

### 2.3 ヒューリスティック案とルール上の事実

直近の通常攻撃の被害を闇攻撃の優先度に加える案は、ユーザーが**ヒューリスティック限定**で挙げた方策選択の案である。
そのまま共通候補削減へ転用する要件ではない。
過去にダメージが通ったことは、現在相手に防御手段がない証明ではない。防御の温存や、その後の補充等も関係する。
公開イベントから因果関係を確認できる場合の補助情報として扱う案はあるが、加点量や有効性は未検証である。

闇固有のHP0効果の条件は、防御後の最終damageが1以上であること。通常攻撃で防具を先に除くことは必須条件ではない。
暗雲による必中化は攻撃対象側の状態を参照する。該当する全体・確率攻撃の扱いを、あらゆる確率的効果一般へ広げない。
[Simulator gameplay §§13.2,15.1](../../../GodFieldSimurator/docs/spec/09-gameplay-rules.md)、[対象状態の参照](../../../GodFieldSimurator/src/engine/resolution/choice.rs)

「手札から消えた枚数」「補充枚数」「空く所持枠」も同じではない。
通常Discardは補充されず、奇跡の初回使用はhandからlearnedへの移動で、合計所持枠は空かない。
所持枠は両者合わせて18、習得済み奇跡は最大6で、追加習得時に最古のものを忘れる。
[Simulator gameplay §§11,12,16](../../../GodFieldSimurator/docs/spec/09-gameplay-rules.md)

使用順序を説明した「増幅を先に置く」例は、固定採点関数の演算説明と、実行可能な行動例を区別する必要がある。
現行の公開ルールによる生成は、先行attackなしの`doubleAtk`を除外する。
順序を入れ替えれば、どの列でも候補として有効になるという意味ではない。[actions.py](../../src/gf_common/actions.py)

### 2.4 現行入力で保持する情報と落ちる情報

| 情報 | 現行の扱い |
| --- | --- |
| 各候補に使うカードの種類・属性・攻撃/防御・MPコスト・命中率等 | カードごとの数値・カテゴリ特徴として入力 |
| 本人/相手の可視状態、暗雲等 | 状態特徴として入力 |
| 候補内の使用順序 | `element.0/1/2`の位置別特徴として保持 |
| 候補一覧内の候補index | 特徴にしない。行動選択結果の参照に使う |
| hand instance ID | 本人のカードの解決に使い、その数値の大小は特徴にしない |
| カード定義・能力ID | それぞれ32 bucketへhash |
| 手札全体 | 種類別件数、数値合計、定義/能力hash集計 |
| 可視イベント | public aggregate条件で直近128件の種類別件数・数値合計等 |

根拠: [encoder.py](../../src/gf_models/encoder.py) L113–196。
属性・暗雲・命中率がモデル入力から丸ごと欠落しているわけではない。
一方、集約表現がカードごとの対応関係やイベント順序を完全保持するわけでもない。

**hash bucketの衝突と、特徴ベクトル全体の一致は別である。** 補足診断では公開catalog v2の296定義すべてについて、単一`element.*`特徴は区別できた。
IDのhashが同じでも、属性や数値等が異なれば候補内のカードは区別できる。
これは現在のfixtureの単一カード表現の結果であり、将来の定義やすべての入力についての一意性保証ではない。

一方、`hand.*`集約では具体的な一致例があった。
ブーメランといがナッツはどちらも攻撃3・価格3だが、前者は無属性で追加攻撃、後者は木属性である。
現encoderではこの2カードの`hand.*`部分が一致する一方、候補の`element.*`では区別できる。
したがって「手札全体の集約で区別が失われる」という懸念には具体的な裏付けがある。
これは入力の一部分だけの比較であり、候補集合・履歴を含む全入力の一致、到達可能性、実戦での損失を証明したものではない。

public aggregate条件ではstate373次元、共通の候補ベクトル690次元。MLP actorは690、系列actor headはcontext128を加えた818を受ける。
引用された別チャットの「10万次元」は説明用の仮定であり、このリポジトリの現入力の実測値ではない。
現GRU/Transformerの系列は、本人の意思決定時点ごとのstateであり、カード列や生イベント列をそのまま処理していない。
[ADR0022](../adr/0022-gru-transformer-ppo-plan.md)

### 2.5 所持一覧の参照と両替

使用カードを一つのmulti-hotにまとめると、使用順序と反復回数が失われる。回数ベクトルだけでも順序は残らない。
使用位置ごとのone-hot、または所持一覧への順序付き参照列は、それらを表現できる候補である。
カード情報を所持一覧で共有すれば重複を減らせるが、入力全体や計算量が必ず小さくなる保証はない。

所持一覧では通常handとlearnedの区別、同一定義の別instance、許される奇跡の反復、使用可否、習得順序を保持する。
任意の手札表示順を無意味として扱うことと、最古の習得奇跡を決める順序を消すことは異なる。
番号はその時点の本人可視の参照であり、数値としての大小を戦術特徴にしない。

両替は現在、`balanced/hp/mp/money`の設定済み配分を候補生成し、モデルが選ぶ。
重複除去等で候補数は減り得る。自由な整数配分を直接出力する実装ではない。
入力には現在のHP/MP/moneyと、候補の配分後の最終HP/MP/moneyが入っている。
[両替候補](../../src/gf_common/actions.py) L154–192、[数値特徴](../../src/gf_models/encoder.py) L134–173。
将来は配分候補を増やす方式、直接数値を選ぶ方式が考えられるが、各値0〜99・合計保存・HP0の扱いを含む公開仕様を満たす必要がある。
[Simulator §17.1](../../../GodFieldSimurator/docs/spec/09-gameplay-rules.md)

### 2.6 エンコーダー構成と潜在表現の訂正

| 論理的な役割 | 検討された内容 |
| --- | --- |
| State encoder | 本人から見える自分/相手の状態を表現 |
| Hand encoder | 手札・習得済み奇跡とカード間相互作用を表現 |
| Turn encoder | 本人に公開された履歴と時間的関係を表現 |
| 後段モデル | 表現を統合し、行動選択や状態評価に使用 |

これは固定の4モジュールを必ず作る決定ではない。直接入力、共有部品、encoderを省く条件、モデル別の組合せが比較対象である。
引用chat2のTurn Encoderは「1時点の圧縮」、ユーザーの提案では「過去履歴の圧縮」として使われており、正式設計では用語を定義する必要がある。
Transformerで「追加encoderなし」を比較しても、入力射影、型/位置表現、必要なmaskまで不要になるわけではない。

**ユーザーの最終訂正は「潜在表現をそのまま入力するつもり」である。**
手札だけから相互作用を表す潜在ベクトル、カード別の文脈付き表現、少数のlatent slots等を作り、そのまま後段へ渡す案を意図している。

```text
本人視点の状態 ─ State encoder ─┐
手札・習得済み奇跡 ─ Hand encoder ┼─ 後段モデル ─ 方策・状態評価
本人に見える過去 ─ Turn encoder ─┘
```

アシスタントは一度これを学習型の組合せ提案器・直接行動生成器と読み替え、Pointer Networkとdecoderの話へ進んだ。
それは別の研究案であり、ユーザーが求めたエンコーダーに必須の要件ではない。手札latentを後段へ直接入力する構成に、カード列復元用decoderは不要である。

[Deep Sets](https://arxiv.org/abs/1703.06114)と[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)は、集合の表現や要素間相互作用を扱う参考になる。
これらはGodFieldで有力な組合せを自動発見できる証拠ではなく、各latent成分が特定の組合せに一対一対応する保証もない。
Set Transformer論文でいうdecoderも、実行するActionを生成するdecoderと同義ではない。

手札latentは手札の可能性を表し、局面での有用性はState/Turn表現と合わせて後段で判断する案が成立する。
一方、Simulatorへは具体的なtyped Actionを送る必要がある。
既存候補の採点を残すか、別の出力方式にするかは未決定であり、latentという言葉だけでその問題は解決しない。
この未決事項を理由に、行動decoderの採用を既定へ戻すこともしない。

### 2.7 E2Eと事前学習

方策・価値の損失から学習可能なencoderへ勾配を流すE2E構成は可能である。
外部の候補生成が離散探索のままでも、その後のニューラルencoderと評価モデルを共同学習できる。
候補生成やSimulatorそのものを微分可能にすることは、ここでのE2Eの要求ではない。
[PyTorch autograd](https://docs.pytorch.org/docs/2.14/notes/autograd.html)

現行設計では、criticと共有する系列stateは、現在の候補の個数・内容・順序に依存しない。
Hand encoderが所持カードを読むこととは両立するが、提示候補を要約してcriticへ渡す構成は境界変更になる。
別案として候補依存Action encoderを置くなら、その経路を方策側へ限定する。
[ADR0008](../adr/0008-mlp-ppo-generations-and-fixed-data.md)、[ADR0022](../adr/0022-gru-transformer-ppo-plan.md)

「格段に多い事前学習が必要」は仮説であり、latentを直接後段入力する構成の必須条件ではない。
E2Eのみ、事前学習後にE2E、凍結encoderを使う条件を比較できる。データ・目的関数・追加費用は明示する。
現在の最大3要素の行動ログだけでは、10要素以上の行動を扱う性能は検証できない。

更新するencoderのlatentだけを保存して固定入力として使うと、そこからencoderを再学習できない。
本人可視の元入力をどの粒度で保存し、現在の重みでどこまで再計算するかを定める必要がある。
現行TBPTTはprefixを現在の重みでno_grad再計算してdetachし、suffixへ勾配を流す。E2Eであることと全履歴に勾配を流すことは異なる。
小さいlatentのサイズ比を、そのまま学習時peak memoryの削減率にしない。入力・activation・勾配・候補数・padding・optimizer状態も費用になる。

LightGBMでは通常のfitからニューラルencoderへ勾配を戻せない。
詳細な固定長特徴を使う基準と、train側で学習した凍結encoderを使うhybridを分ける。
後者ではencoder学習へのvalidation/test混入を防ぎ、設定選定用のvalidationと最終testも分離し、追加費用を管理する。

### 2.8 次の設計へ引き継ぐ未決事項

1. 単体候補の保持範囲と、対象・数値引数による複数行動の扱い。
2. 長い組合せ・反復・多様性と、探索量・提示量・推論/学習資源の上限。
3. 具体的候補を採点する方式と手札latentを使う方式の関係、および最終Actionとの対応。
4. カード単位・使用順序・習得順序・イベント順序など、保持する情報の明細。
5. State/Hand/Turnの分担、encoderなし条件、E2E/凍結/事前学習の比較方法。
6. 観測スナップショット履歴とイベント履歴の比較。両者に同じ情報があるとは未確認。
7. 長い入力・候補数増加に対応する保存形式、再計算、互換性、失敗時の扱い。

これらは候補生成・モデル入力・学習・評価を横断する共同設計として扱う。
候補生成と表現学習の効果を測る比較は必要だが、比較可能な共通範囲を明示する。現行3要素encoderへ10要素候補を与える対照実験は成立しない。
新仕様を実装する前に、具体的な責務・interface・保存形式・検証条件のarchitecture reviewが必要である。

## 両テーマに共通する境界

本人のObservation、可視Events、typed metadata、同revisionの本人ActionOptions、許可された版固定カタログを使う。
本人のperceivedなカード定義を保持し、他者private・canonical hidden history・GameSeed/RNG・未来情報をpolicyへ渡さない。
ActionOptionsはatomic hintであり、全組合せ・順序・対象・資源の合法性を保証しない。
SequenceOptionsや拒否詳細をpolicy特徴・候補・mask・teacherに使わず、合法性のprobeを行わない。最終判定はSimulatorに残す。
[learning contract §21.1](../../../GodFieldSimurator/docs/spec/07-external-interfaces-adapters.md)

失敗親runの良かった部分だけを学習・正式比較に流用せず、自然終了・truncation・abort・不適合・インフラ失敗を分ける。
自然終了returnを行動後の学習ラベルに使うことは、未来情報をpolicy入力へ混ぜることとは異なる。
探索seed・学習seed・対戦GameSeed・方策sampling seedを区別し、両席・繰返し・Simulator fork祖先をsplit間で分けない。
checkpointの学習来歴が共通というだけで、その方策の新しいvalidationゲームまでtrainへまとめることはしない。

## 補足診断と確認範囲

追加診断に使った公開fixtureは[catalog_public_v2.json](../../tests/fixtures/catalog_public_v2.json)、file SHA-256は`7a111afca3cc4be03a86ec37db7224824006d8ba2f2f985fd3f79604cf85e839`。
実際の`_features`を呼び、比較対象を`element.*`または`hand.*`だけに限定した。
296定義の単一element特徴は296種類。weapons/armor/miracles/sundries/tradeの239定義を2枚multisetにした28,680通りでは、hand集約が28,419種類、同じ集約になる群が260あった。
これは構造上の表現の診断であり、全ゲーム状態の同一性や実戦成績の測定ではない。

簡単な一致例は、リポジトリrootから次で再現できる。

```console
./.venv/bin/python - <<'PY'
from pathlib import Path
from types import SimpleNamespace
import json
from gf_models.config import EncoderConfig
from gf_models.encoder import _features

catalog = {c['definition_id']: c for c in json.loads(
    Path('tests/fixtures/catalog_public_v2.json').read_text())['definitions']}
config = EncoderConfig(history='public_aggregate')
def project(key):
    value = SimpleNamespace(
        observation={'own': {'player_id': 1, 'hand': [
            {'instance_id': 1, 'definition_id': key}], 'learned_miracles': []}},
        catalog={key: catalog[key]}, action_options=None, events=())
    action = {'type': 'play_sequence', 'target': 2, 'ordered_elements': [
        {'source': 'hand_artifact', 'instance_id': 1}]}
    features = _features(value, action, config)
    return [{k: v for k, v in features.items() if k.startswith(prefix)}
            for prefix in ('hand.', 'element.0.')]
a, b = project('boomerang'), project('bur-nut')
print('hand aggregate equal:', a[0] == b[0])
print('element encoding equal:', a[1] == b[1])
PY
```

期待出力は`hand aggregate equal: True`、`element encoding equal: False`。
この小さな診断は入力validation・合法性・到達性を検証するものではない。

独立エージェントはread-onlyで原稿と根拠を作成し、Rootが統合した。
学習、HPO、Simulator対戦、依存追加、アプリケーションテストはNOT RUN。
最終文書は両エージェントが再確認し、資料としてREADY、blocking findingなし。
指摘されたvalidation選定と学習への混入の区別、禁止情報の用途、Simulator参照節を修正した。
`git diff --check`、3文書の相対リンク存在確認（28/20/32件）・code fence対応・空白検査、上記の再現コードはPASS。
これらは文書と限定した特徴診断の確認であり、実装・探索実験の受入ではない。
