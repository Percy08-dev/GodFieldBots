# heuristic-v1

## 識別と適用範囲

model_id は heuristic、モデル版は 1。公開情報から候補に規則ベースのスコアを付ける比較用ポリシー。

## 構造と入出力

~~~mermaid
flowchart LR
    O["本人観測・公開ルール"] --> R["候補ごとのルール採点"]
    C["本人の候補列"] --> R
    R --> A["最大スコアの候補"]
~~~

C 件の候補を採点し、攻撃・防御・交換などの行動内容に応じて1件を選ぶ。学習済みパラメーターは持たない。

## 順序・マスク・上限

旧候補セットでは同点から参加者ローカル乱数で選ぶ。構造化候補表と訂正後 CardInstance では固定の先頭優先。CardInstance の採点は当該レイアウトの公開ルールとカード参照を使う。

## 設定・互換性・検証

[代表設定](../../configs/models/heuristic.toml) · [旧実装](../../src/gf_models/baselines.py) · [構造化実装](../../src/gf_models/structured_baselines.py) · [CardInstance 実装](../../src/gf_models/card_instance_baselines.py) · [基盤検証](../verification/phase0-3.md) · [CardInstance 検証](../verification/card-instance-v2-adoption.md)。
