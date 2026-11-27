# mlp-v2-event-gru

## 識別と適用範囲

model_id は mlp、モデル版は 2、入力レシピは participant-structured-v1。旧公開入力の DecisionFrame と CandidateTable を処理する構造化 PPO。

## 構造と入出力

~~~mermaid
flowchart LR
    S["本人の状態"] --> SM["状態 MLP"]
    H["カード 18 スロット"] --> HC["カード埋め込み・符号化"]
    HC --> HM["手札 attention と集約"]
    E["区間イベント列"] --> GR["イベント GRU"]
    SM --> F["意思決定フレーム"]
    HM --> F
    GR --> F
    G["履歴集約"] --> F
    F --> T["現在フレームの MLP"]
    T --> A["候補別 Actor MLP"]
    T --> V["Critic MLP"]
    HC -->|候補が参照したカード| A
    C["候補特徴と順序付きカード参照"] --> A
    A --> P["候補別対数確率"]
    V --> R["状態価値"]
~~~

既定では状態を 64、カードを 64、手札を 128 に符号化する。手札は 2 層・4 heads の attention とマスク付き平均を使う。区間イベントは幅 394 の行列 [E, 394] を 1 層 GRU で処理し、32 幅に要約する。履歴集約 376 と結合したフレームは 600 幅。現在のフレームだけを射影する。フレームには区間イベントの要約と履歴集約が含まれる。 Actor は候補の 26 幅の特徴と参照カードを文脈に合わせて採点する。既定の展開相当幅は 1414。 バッチの出力は候補別対数確率 [B, C] と状態価値 [B]。

## 順序・マスク・上限

手札側は固定 18 スロットで空き枠をマスクする。1 候補内のカード参照は順序を維持し、候補対数確率は同じ決定の候補全体で正規化する。区間イベントは発生順。0 件の区間はゼロ要約で、既定の区間上限は 4,096 件、履歴集約の窓は 128 件。現在行の処理に入る履歴は区間イベントと集約特徴。

## 設定・互換性・検証

[既定構成](../../src/gf_models/structured_binding.py#L114) · [構造化モデルの設計と形式](../adr/0023-structured-encoders-and-diverse-search.md) · [構造化モデルの検証](../verification/structured-encoders-and-search.md)。 歴史的な 62 幅集約は記録に結び付いた互換経路で解釈する。図は既定の 376 幅集約。 [共通ネットワーク実装](../../src/gf_models/structured_network.py)。

## 代表実行の図

パック済みテンソルからフレーム行をインデックスで取得し、Actor と Critic に渡す eager CPU 演算。B=1、意思決定 T=1、区間イベント E=2、カード2枚／18スロット、候補 C=3。

![mlp-v2-event-gru のテンソル演算図](mlp-v2-event-gru.svg)

[SVG を拡大する](mlp-v2-event-gru.svg)
