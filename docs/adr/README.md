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

追加: [0044](0044-inference-performance-comparison.md) — Accepted,implemented and verified:
推論packing・再利用・chunk・padding・compile・TF32を既存予算内で比較する。
独立最終レビュー・576自然試合・18親原本監査を通過。単独推論の再利用は有利だが
全体時間の改善は一貫せず、既定値は維持。学習品質評価は後続。

追加: [0042](0042-hpo-precollection-validation-reuse.md) — Accepted design:
resident HPOの収集前3回で実際の検証scanを保持し、元データ・native・所有権を再確認する。
mkdir前に終了し、その後の完全検証は維持。実装レビューと全CPU6557PASS/408SKIP、
静的・配布・保護対象確認は完了。GPU10試行と元データ監査も通過し、MLP3組中央値の
第2収集26.34%・全体9.77%短縮を確認。最終の両独立レビューはFINAL READY、root受入済み。

追加: [0041](0041-hpo-resident-workers-and-progress.md) — Accepted design:
HPO限定の常駐CUDA参加者、更新境界の重み同期、明示的な待機証明・キャッシュ参照、
学習権威から分離した進捗表示と監視CLI。実装・最終受入は未完了。

追加: [0039](0039-refresh-heuristic-and-mixed-opponents.md) — Accepted design:
独立した手札更新型ヒューリスティックと、通常PPO/HPOの明示的な混合16試合相手構成。
実装・最終受入は進行記録と検証証跡で区別する。

追加: [0040](0040-guardian-acted-public-id-union.md) — Accepted design:
実対戦で確認したGuardianActedの既知カードID／守護神ID両表現を、corrected限定で保持する。
語彙・既存索引・スキーマは維持し、旧失敗実行は不適格のまま保存する。

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

追加: [0025](0025-session-shared-pytest-ppo-sources.md) — Accepted:
pytestのセッション内で合成PPO原本を一度生成し、各テストへ独立した検証済みコピーを渡す。
両独立レビュー・全件検証・同一対象での時間比較を完了。

追加: [0026](0026-structured-ppo-test-reference-reuse.md) — Accepted:
構造化PPOテストの完了済みCPU MLP参照実行を共有し、独立した一時停止・再開実行と比較する。
両レビュー・対象42件・関連219件を通過。対象モジュールの経過時間を約17%削減。

追加: [0027](0027-pytest-cpu-participant-pool.md) — Accepted:
pytest内のCPU参加者プロセスを収集バッチ内で再利用する。通常収集・CUDAは新規起動を維持。
対象42件で約47%短縮。両レビュー・全件実行と新規テスト修正後の48件再確認で受入。

追加: [0028](0028-pytest-parallel-participant-readiness.md) — Accepted:
pytestのCPU収集で同じ対局の参加者初期化を並行化し、元の42件を241.87秒から231.79秒へ短縮。
両レビュー・関連279件・6件の既存CUDAスキップ・原本配列と行動記録の比較で受入。

追加: [0029](0029-lazy-cpu-deterministic-runtime.md) — Accepted:
決定論性を保ってCPUのInductor初期化を遅延。対象42件を231.79秒から195.84秒へ短縮。
両レビュー・全4,305件（4,058成功／247スキップ）・実CUDA6件・保存状態と行動記録の比較で受入。

追加: [0030](0030-pytest-preloaded-cpu-launch.md) — Accepted:
pytestのCPU参加者をTorch先読みプロセスから生成。元の42件は195.84秒→161.16秒。
両レビュー・全4,359件（4,112成功／247スキップ）・実CUDA6件・状態と行動の比較で受入。
安全性テスト54件の追加により全体時間は27.47秒増加することを明記。
その後の利用者指示で重い38件を環境変数による明示実行へ変更し、軽い16件は通常実行に残す。

追加: [0031](0031-tree-pytest-preloaded-cpu-launch.md) — Accepted:
Tree Dataset前処理の学習・最終検証収集にもTorch先読みを適用する。
両独立レビュー・原本との一致確認・全4,396件の検証を完了。
対応する対象テスト計測は200.56秒→80.97秒。重い安全性テストの明示実行を維持。

追加: [0032](0032-native-pytest-preloaded-actors.md) — Accepted:
native Sourceの構造化収集テストへTorch先読みを拡張。設計・最終レビューと対象検証を通過。
原本・数値・行動の一致と数値ワーカー対応を含む全4,504件の最終検証を完了。

追加: [0033](0033-pytest-preloaded-numerical-children.md) — Accepted:
対象のCPU数値ワーカーを独立したTorch先読みテンプレートから新規起動する。
両レビュー・保存状態一致・通常起動と実CUDA検証を通過。全体は4,177成功・327スキップ。

追加: [0037](0037-ppo-hpo-foundation.md) — Accepted:
可変PPOモデル・一意な学習決定数・Optuna探索・評価と保存管理。
全体受入は検証記録に従う。異常終了時の受付回復条件は0038で一部更新。

追加: [0038](0038-explicit-gpu-admission-release.md) — Accepted:
終了未確認を保持したまま、明示的な判断を記録してGPU受付を再開する。
終了・再起動・学習評価の成立を証明する記録とは区別する。

## 記録テンプレート

追加: [0034](0034-sqlite-group-pack.md) — Accepted (保存基盤の設計):
明示選択のグループPack＋SQLite、Python Writer、sealと原本所有権。
実装とruntime／Source／native／final64接続のレビューは進行中。既定形式は維持。

追加: [0023](0023-structured-encoders-and-diverse-search.md) — Accepted (独立実装可能部分):
構造化入力・設定可能E2E encoderと予算付き多様候補探索の共同導入。
core・確認済み候補の設計レビューはREADY。公開utility行動契約の不足はissue0005、全体受入は未完了。

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
