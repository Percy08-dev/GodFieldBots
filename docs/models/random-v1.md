# random-v1

## 識別と適用範囲

model_id は random、モデル版は 1。旧候補セット、構造化候補表、訂正後 CardInstance の各実行経路がある。

## 構造と入出力

~~~mermaid
flowchart LR
    C["本人の候補列"] --> U["参加者ローカル乱数で一様抽選"]
    U --> S["元の候補列のインデックス"]
~~~

C 件の候補から1件を一様に選ぶ。重みや固定幅のテンソルを持たない。旧経路では行動インデックスと一様分布の対数確率を返し、構造化経路では候補表のハッシュと選択インデックスを返す。

## 順序・マスク・上限

C は候補生成の結果に従う。候補の並びは選択したインデックスを行動に結び付ける。乱数は参加者の reset で与えられたサンプリング seed から進む。

## 設定・互換性・検証

[代表設定](../../configs/models/random.toml) · [旧実装](../../src/gf_models/baselines.py) · [構造化実装](../../src/gf_models/structured_baselines.py) · [CardInstance 実装](../../src/gf_models/card_instance_baselines.py) · [基盤検証](../verification/phase0-3.md) · [CardInstance 検証](../verification/card-instance-v2-adoption.md)。
