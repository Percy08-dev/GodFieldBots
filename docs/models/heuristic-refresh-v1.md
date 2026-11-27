# heuristic-refresh-v1

## 識別と適用範囲

model_id は heuristic_refresh、モデル版は 1。訂正後 CardInstance の候補表を使い、攻撃と手札更新を評価する状態付きルールベースポリシー。

## 構造と入出力

~~~mermaid
flowchart LR
    E["今回までの本人可視イベント"] --> H["直近の推定阻止量を更新"]
    C["本人観測と候補表"] --> A["攻撃評価・手札更新"]
    H --> A
    O["自分の通常ターン以外など"] --> B["CardInstance Heuristic"]
    A --> S["候補表の選択インデックス"]
    B --> S
~~~

有効な攻撃候補は推定阻止量と比較する。過去の攻撃の見積もりから、そのターン中に本人が観測できた相手へのダメージを引く。推定値の下限はゼロ。サンプルがない場合の既定の攻撃閾値は12。攻撃条件を満たす候補がなければ手札更新を採点し、該当候補がなければ CardInstance Heuristic を使う。

## 順序・マスク・上限

参加者と対戦を固定してイベントを順に消費し、直近3回の自分の通常ターン（`turn_action`）の推定阻止量を参照する。同じフレームと候補表への再要求では同じ選択を返す。攻撃の同順位は消費量などで解決し、最後は候補順で決まる。

## 設定・互換性・検証

[代表設定](../../configs/models/heuristic-refresh.toml) · [実装](../../src/gf_models/refresh_heuristic.py) · [モデル仕様](../design/heuristic-refresh.md) · [検証](../verification/heuristic-refresh.md)。設定した閾値、Action 契約、公開ルールが実行時のポリシー識別に入る。
