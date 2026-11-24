# 〈夢〉で通常武器が精霊の杖に見える問題

修正版の範囲と採用契約は[ADR0067](../adr/0067-dream-simulator-contract.md)に、検証結果は[検証記録](../verification/dream-simulator-contract.md)に記録する。以下の再現と影響範囲は旧Simulatorと精霊の杖を起点とした調査。

修正先は GodFieldSimurator。のこぶんぶんの偽装先は、既存の〈夢〉の互換条件を満たす通常武器の範囲とする。CostCut 武器は別の互換集合に属する。固定カタログで CostCut 武器は精霊の杖 (`spiritual-staff`) だけなので、精霊の杖は偽装元にも偽装先にもならない。精霊の足袋・頭巾・帯は CostCut 防具で、精霊のぬいぐるみは CostCut 専用の雑貨に分類される。[Simulator のカード分類 §3](../../../GodFieldSimurator/docs/spec/card-definition-classification-v2.md#3-multi-role-owned-definitions)

## 再現記録

HPO6の学習seed 17、run `c254e6a6-fd44-41a0-9172-6a6f7ff01894` は、6更新を確定した後、7回目の収集 `batch-000006` のペア `p41` で停止した。Game IDは `01M84WBG72XQVTZBADCX2WEGVY`、GameSeedは `9eff245e-a303-55ec-a17b-a3441aa184e7`。Simulatorは `3172da407e37a4ad57ac82e70a96910c3ac6fbb8`、Gameplay API v2、カタログ schema 2・`public-ja-v3`、Action profile `card-instance-simulator-v2-shared-20261123`。[実行版の検証記録](../verification/strength-powder-simulator-contract.md)

ターン32、revision 46で〈夢〉が有効なPlayer 2のMPは0だった。Player 2は〈氷河期〉 (`ice-age`) → 表示上の精霊の杖の順に選び、Player 1を対象に `play_sequence` を送った。ActionOptionsの `cost_cut` は表示上の分類に従っていた。診断用Dumpで、杖に見えていたカードの実体はのこぶんぶん (`saw-boom-boom`) と確認した。[ActionOptions の投影](../../../GodFieldSimurator/src/server/observation.rs)

`dream_group` は両方を `weapon:single_target:plain` に分類する。行動を解決する際は実体の定義を使うため、のこぶんぶんは直前の〈氷河期〉のコストを免除せず、攻撃に加わる。EngineはMPの支払い可否を調べる前に `MultipleNormalWeapons` で拒否した。HTTP応答は `422 / INVALID_ACTION / illegal_combination` で、revisionは46のままだった。[偽装グループ](../../../GodFieldSimurator/src/engine/catalog/mod.rs)・[コンボ判定](../../../GodFieldSimurator/src/engine/resolution/choice.rs)

同じ TestServer バイナリで受理済み46手をローカルで再生し、行動者のObservationとActionOptions、カードの実体、同じ422応答を照合した。元のNative拡張でも行動者の公開入力から該当Actionが候補に入ることを確認した。GameSeedとDumpは再現・診断にのみ使用した。

保存HTTPログのパスとSHA-256。

```text
/mnt/gf_data/ppo-hpo/mlp-256-convergence-1000/http-logs/fe76ee195c8f46f38f9e9ee16d848c8e/batch-000006/b000006-p41-gk3nka18/http.jsonl.zst
7ee802c490081f579e2805f46b414cbde94ae88c63384b024c6a3383e969c8fe
```

## 影響範囲

### 偽装候補と乱数

`3172da4` の `GameDefinitions::dream_candidates` で固定カタログ296定義を列挙し、精霊の杖を含む偽装候補だけを両方向から除いた場合の差分を計算した。候補表が変わるのは62定義で、削除される偽装の組は方向別に122件。

| 実体 | 定義数 | 1定義あたりの偽装候補 |
| --- | ---: | ---: |
| 精霊の杖 | 1 | 61 → 0 |
| のこぶんぶん、ソードシールドなど同じ互換集合の通常武器 | 61 | 61 → 60 |

この変更が直接効くのは、〈夢〉中の通常取得・補充・引き直しと、磁気嵐による再配布。守護神の手札付与も同じ候補関数を使う。地属性の守護神が武器を引いた場合は直接攻撃に使う。表示の差は行動者のObservation、取得イベント、ActionOptionsの表示・分類を通じて候補生成に伝わる。[通常取得](../../../GodFieldSimurator/src/engine/resolution/turn.rs)・[磁気嵐と守護神](../../../GodFieldSimurator/src/engine/resolution/status.rs)・[手札付与](../../../GodFieldSimurator/src/engine/resolution/economy.rs)

精霊の杖を〈夢〉中に受け取ると、候補が空になるので偽装の50%判定と候補抽選を省略する。他の61種は引き続き50%判定を行う。候補抽選の上限61と60は現行 `next_bounded` の棄却しきい値がともに16なので、同じ乱数状態からの候補抽選が消費する乱数の個数は同じになる。添字と候補表が変わるため、旧版で杖以外に偽装していたケースにも表示差が出る。杖の取得時の乱数差や、表示差による行動の変化は、後続の配札・命中・結果にも波及し得る。[乱数実装](../../../GodFieldSimurator/src/engine/rng/mod.rs)

### 保存済み試合と採用時の境界

元のrunの確定済み6バッチ、計1,536試合の両席の保存Replayを読み取り専用で確認した。第6バッチ `b000005-p14-seat0` で、〈夢〉中の精霊の杖と〈夢〉解除後のソードシールドが同じ `card_ref` だった。確定済み試合にも誤表示があったことを示す。同じ参照を〈夢〉中と解除後の両方で観測できた例を数えているため、1試合は確認できた下限。

旧Journal・Replay・チェックポイントは元のSimulator版と `rule_signature` で扱う。Simulatorの履歴再生は状態・乱数のハッシュまで照合するので、新版で再実行した結果との一致は保証できない。修正版の採用には新しいSimulator revision・仕様ハッシュ・バイナリハッシュをPythonとRustの契約に追加し、HPOの基礎実験設定を切り替える。現行の基礎実験は256試合の更新予算プローブ、そのテンプレート、1,000更新の収束診断で共有している。[Simulatorの履歴照合](../../../GodFieldSimurator/src/server/history/replay/mod.rs)・[基礎実験設定](../../configs/experiments/hpo-rust-mlp-strength-powder-base.toml)・[版固定の先例](../adr/0066-strength-powder-simulator-contract.md)

### CostCut防具で確認した関連事象

足袋・頭巾・帯も現行の〈夢〉では通常防具23種と互換になっている。CostCut防具と通常防具をまたぐ偽装は方向別に138件。確定済みReplayでは `b000005-p113-seat0` で、革の服が精霊の頭巾に見えていたことを同じ参照の〈夢〉解除後の表示で確認した。

同じSimulatorのEngineで、革の帽子が精霊の足袋に見えている防御局面を作った。〈壁〉→足袋の表示で応答すると、MP 0では `InsufficientMp`、MP 6では `ExclusiveDefenseCombination` で拒否された。本物の足袋を使った対照ではMP 0で受理された。CostCut専用の精霊のぬいぐるみの偽装候補は0件だった。[防御判定](../../../GodFieldSimurator/src/engine/resolution/choice.rs)

## Simulatorの受入条件

- のこぶんぶんには互換性のある通常武器の偽装候補があり、精霊の杖は候補に含まれない。
- 固定カタログで精霊の杖の偽装候補は空で、ほかのカードの偽装候補にも精霊の杖は含まれない。〈夢〉中に精霊の杖を取得しても実体のまま表示し、候補が空の場合の既存の乱数消費規則に従う。
- 本物の精霊の杖は単独で武器として使え、奇跡の直後に置いた場合は直前の奇跡のMPを免除する。〈氷河期〉→精霊の杖がMP 0で成立することを制御した局面で確認する。[Simulator のゲームルール §17.5.3・§18.4](../../../GodFieldSimurator/docs/spec/09-gameplay-rules.md#1753-confirmed-completion-cost-and-empty-effect-rules)

偽装候補の変更で後続の抽選結果が変わり得るため、元の46手は旧版での診断証拠として扱う。新しいSimulatorの採用時は、修正版の規範仕様と実行バイナリに対応する契約を版固定する。
