# Architecture Decision Records

重要な判断について、結論だけでなく理由と影響を残します。
現在の設計は [architecture.md](../architecture.md)、開発手順は [AGENTS.md](../../AGENTS.md) を参照してください。

## 運用

- rootが `NNNN-short-title.md` を作成・更新し、原則として1件の判断を1記録にまとめます。
- 公開インターフェース、責任分界、主要な依存、保存形式、情報境界など、将来の変更判断に必要な事項を記録します。
- `Proposed` の段階でアーキテクチャ担当がレビューし、重大な指摘の解消後にrootが `Accepted` にします。
- `Accepted` は判断の承認であり、実装済みを意味しません。実装状態と検証結果は別に記録します。
- 判断を置き換える場合は新しいADRを追加し、旧ADRを `Superseded` にして相互リンクを残します。
- 残す負債は影響と再検討条件を記載します。通常の局所的な修正にはADRを量産しません。

## 一覧

| ADR | 状態 | 判断 |
| --- | --- | --- |
| [0001](0001-simulator-boundary.md) | Accepted | シミュレーターを外部のルール権威とする |
| [0002](0002-configured-public-baseline-foundation.md) | Accepted | Config・公開候補生成・分離された対戦保存評価基盤 |
| [0003](0003-episode-closure-projection.md) | Accepted | Simulator abort理由と非終端truncationを区別しschema 3へ固定 |
| [0004](0004-coordinated-cancellation.md) | Accepted | キャンセル時もCLIが所有グループの後始末完了を待つ |
| [0005](0005-authorized-atomic-action-options.md) | Accepted | 同revision・本人向けActionOptionsをatomic hintsとして使用 |
| [0006](0006-lightgbm-action-value.md) | Accepted | 公開特徴のLightGBM行動価値・checkpoint・Gate付き学習 |
| [0007](0007-staged-buy-contract-adoption.md) | Accepted | 購入応答・承諾の新契約とモデルのSimulator版固定 |
| [0008](0008-mlp-ppo-generations-and-fixed-data.md) | Accepted | CUDA MLP PPO・世代来歴・境界再開・固定データ比較 |

追加: [0009](0009-catalog-v2-simulator-adoption.md) — Accepted:
Magical Stick新契約・catalog v2・遅延認証応答の失敗ラッチと新規実験。

追加: [0020](0020-owned-episode-pipeline-pool.md) — Accepted:
所有する固定poolと試合単位のworker内処理。実装・同条件48実行の前後比較・採用検証済み。

## 記録テンプレート

```markdown
# NNNN: 判断の題名

- Status: Proposed / Accepted / Superseded
- Date: YYYY-MM-DD
- Decision owner: root
- Review: 担当・結果・根拠への参照
- Implementation: 未実装 / 実装への参照

## Context
対象要件、制約、問題、判断に使った証拠。

## Decision
採用する方針と責任分界。

## Alternatives
検討した代替案と採用しなかった理由。

## Consequences
利点、費用、制約、互換性、意図して残す負債と再検討条件。

## Verification
受入条件、設計レビュー結果、実装後の確認方法・結果。
```
