# lightgbm-v1-history

## 識別と適用範囲

model_id は lightgbm、モデル版は 1、参加者別特徴レシピは participant-action-v1。encoder.history は public_aggregate。

## 構造と入出力

~~~mermaid
flowchart LR
    O["本人観測と候補行動"] --> R["候補別特徴行"]
    R --> Q["CPU LightGBM 回帰"]
    Q --> A["最大スコアの候補"]
    E["本人可視イベントの集約"] --> R
~~~

候補数 C に対する特徴行列は既定で [C, 690]。1 候補につき終局リターンの予測値を1つ出し、全 C 件から選ぶ。学習レシピは lightgbm_mc。

## 順序・マスク・上限

既定では直近128件の本人可視イベントを集約する。 候補列を空にできない。最高スコアが同じ場合は元の候補列で先に現れるものを選ぶ。

## 設定・互換性・検証

[代表設定](../../configs/models/lightgbm-history.toml) · [推論実装](../../src/gf_models/lightgbm/policy.py) · [Phase 4 検証](../verification/lightgbm-phase4.md)。推論マニフェストはエンコーダー設定、特徴順、候補生成と採点の契約を固定する。
