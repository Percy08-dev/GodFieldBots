# モデル一覧

各ページにモデルのデータフロー、代表的な入出力形状、設定、互換性、既存の検証記録を掲載します。構成名は文書内でモデルを識別するためのものです。実行時の識別には既存の `model_id`、モデル版、入力契約、設定と重みの識別情報を使います。

| モデル | Model1 | Model2 | Model3 | Model4 |
| --- | --- | --- | --- | --- |
| MLP | [mlp-v1](mlp-v1.md) | [mlp-v2-event-gru](mlp-v2-event-gru.md) | [mlp-v3-event-gru](mlp-v3-event-gru.md) | [mlp-v4-event-gru](mlp-v4-event-gru.md) |
| GRU | [gru-v1](gru-v1.md) | [gru-v2-event-gru](gru-v2-event-gru.md) | [gru-v3-event-gru](gru-v3-event-gru.md) | [gru-v4-event-gru](gru-v4-event-gru.md) |
| Transformer | [transformer-v1](transformer-v1.md) | [transformer-v2-event-gru](transformer-v2-event-gru.md) | [transformer-v3-event-gru](transformer-v3-event-gru.md) | [transformer-v4-event-gru](transformer-v4-event-gru.md) |

- LightGBM の [v1 観測版](lightgbm-v1-observation.md)、[v1 履歴版](lightgbm-v1-history.md)、[Tree2](lightgbm-v2.md)、[Tree3](lightgbm-v3.md)
- [Random](random-v1.md)、[Heuristic](heuristic-v1.md)、[Heuristic Refresh](heuristic-refresh-v1.md)

## ページの書き方

1. `model_id`、モデル版、入力レシピ、対応する構成・実行経路を記載する。文書名は `<モデル種別>-v<版>` を起点に、同じ系列を区別できる構造名を必要に応じて加える。Model2〜4 の `event-gru` は意思決定部分の方式と独立した区間イベントエンコーダーを表す。
2. Mermaid で入力から意思決定、出力までを描き、固定軸と可変軸を記載する。候補間と行動内部の順序、イベントの範囲、空入力・パディングの扱いのうち、該当する部分を説明する。
3. 実装・代表設定・互換性の根拠・検証記録にリンクする。試行別の上書きは `run.json` の `effective_config` を参照する。過去の契約や経路の記述には対象の版を添える。
4. ニューラルモデルには代表 CPU 実行の SVG を埋め込み、拡大リンクを付ける。キャプションに図の実行経路と代表入力を示す。Model1 は通常の `forward`、Model2〜4 はパック済みテンソルを使う eager CPU 演算の一例を描く。MLP の図はフレーム行をインデックスで取得する経路を描く。実運用で選択されるキャッシュや CUDA 経路は本文の実装リンクから確認できる。

## 図の生成と確認

Python の図生成依存は uv の `model-docs` グループにある。生成にはシステムの Graphviz `dot` も必要。リポジトリルートで実行する。

```console
uv run --locked --group model-docs --extra torch python tools/model_docs.py generate mlp-v4-event-gru
uv run --locked --group model-docs --extra torch python tools/model_docs.py check
```

モデル名を省いた `generate` はニューラルモデルの図をすべて生成する。`check` は一覧・ページ・画像の対応と図のフィンガープリントを確認する。既存環境で CUDA 版の Torch を使っている場合は `--extra torch-cuda` も選べる。図の実行先は CPU。

モデルの追加や図に出る既定構成の変更は、一覧・個別ページ・生成対象と同じ差分にまとめる。SVG は生成コマンドで更新し、Markdown と一緒にレビューする。
