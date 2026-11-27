# lightgbm-v3

## 識別と適用範囲

model_id は lightgbm、モデル版は 3、入力レシピは participant-structured-tree-v2。訂正済み CardInstance 用の Tree3。

## 構造と入出力

~~~mermaid
flowchart LR
    S["本人の状態・カード・履歴集約"] --> F["共通コンテキスト"]
    C["候補表・順序付き参照"] --> R["候補ごとの特徴行"]
    F --> R
    R --> Q["CPU LightGBM をチャンク実行"]
    Q --> A["全チャンクで最大の候補"]
~~~

既定の18参照位置では共通コンテキストが 1629 幅、候補ごとの特徴行が 2810 幅。チャンク単位の入力は [C_chunk, 特徴幅]、出力は候補ごとの予測値。 LightGBM のスコアは終局リターンの推定に使う。

## 順序・マスク・上限

候補参照順とパディングは入力レシピで固定。既定では256件ずつ採点し、チャンクをまたいで同点なら候補表の先頭側を選ぶ。既定の履歴窓は本人可視イベント128件。

## 設定・互換性・検証

[固定データ設定](../../configs/experiments/card-instance-tree-fixed-data.toml) · [GameTree3 実行契約](../design/card-instance-game-tree-runtime-v3.md) · [Tree3 構成・フィクスチャ検証](../verification/card-instance-tree-v3.md) · [訂正済み経路の統合検証](../verification/card-instance-v2-adoption.md)。 [推論実装](../../src/gf_models/structured_tree_policy.py)。 学習済み重みは対応するデータと推論マニフェストに結び付く。検証記録には実行資格と合成フィクスチャの適用範囲がある。
