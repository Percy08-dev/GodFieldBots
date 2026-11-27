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

[0079](0079-simulator-v3-direct-defense.md) — Simulator v3 の防御候補と霧の攻撃イベントを、版を固定した Action profile で取り込む。

[0078](0078-model-documentation.md) — モデル構造の一覧と個別ページを管理し、
ニューラルモデルの代表 CPU 実行から SVG を生成する。

[0074](0074-rust-player-sequence-hpo.md) — CUDA HPO6 の Model4 系列推論を
Rust の試合進行、要求の受理と退役、既存 Numeric6 学習へ接続する。

[0073](0073-rust-live-journal-frame-validation.md) — 共有 CardInstance の
ライブ Journal で、単一 writer が元のレコードを Rust で検証する。

[0072](0072-sacrifice-eligibility.md) — 通常の「捨てる」と終末の「ささげる」に
同じ対象判定を使うBot候補生成を、版固定の共有Actionプロファイルで採用。

[0071](0071-hpo-tensorboard-watch.md) — 単独 HPO6 の対話監視に、確定済み PPO 更新を
表示する任意の TensorBoard を追加する。

[0066](0066-strength-powder-simulator-contract.md) — ちからの粉の夢と
武器併用条件を訂正したSimulatorを新しい共有Action契約で採用。

[0065](0065-staged-native-collection-storage.md) — ネイティブPPO・HPO6の
収集をtmpfsで行い、HDD上の専用ext4へ検証・同期してからSourceを発行する。

[0064](0064-hpo-storage-and-compression.md) — Accepted。HPOの保存先をTOMLで
指定し、席交換ペアのHTTP診断と更新JSONにzstdを導入する。過去の更新は元の形式で読む。

[0063](0063-hpo-checkpoint-evaluation.md) — Model4の更新ごとの学習診断と
HPOの定期チェックポイント評価を追加する。

[0060](0060-rust-player-journal-handoff.md) — 共通のRust対戦経路でJournalを連続取得し、
全試合が設定上限に収まる収集では、ペアの実行証跡とwriterコンテキストを確定して一括投入する。

[0059](0059-rust-player-cuda-batch-inference.md) — Accepted。Model3・Model4 の Rust
対戦収集で固定形状の CUDA バッチ推論を選択できる。256 試合の実測で両モデルの
収集時間と応答 p95 が改善し、標準は従来の逐次推論とする。

[0058](0058-four-pytest-workers.md) — Accepted。Pytestの標準を4ワーカーの
ファイル単位分配とし、`-n 0` で直列実行を選べるようにする。

[0057](0057-rust-ppo-training-data.md) — Accepted。Model3・Model4 の Rust 対戦経路で
学習配列と GAE を構築し、通常監査では封印済み Pack を SHA-256 で照合する。
同一入力、保存・再開、Model4 評価への引き継ぎと実機比較を確認済み。

[0056](0056-ordinary-rust-ppo-update.md) — Accepted。通常 Rust Model3 に版付きの
PPO 更新設定を追加し、対戦収集、学習、保存後の監査で同じ条件を使う。

[0055](0055-owned-rust-model3-parity-validation.md) — Accepted。通常 Rust Model3 の
所有済み学習入力では、更新前の数値比較を明示検証へ移す。実機で保存・再開を確認済み。


追加: [0053](0053-mlp-shared-batch-inference.md) — Accepted design:
Simulator072dabbの共有推論契約を採用し、MLPの通常PPO/HPOを公式TorchRLへ接続する。
CPU参加者、共有GPUサービス、checkpointと要求の独立性、実際の終了証明を分離する。
事前設計レビューREADY。実装・契約テスト・384試合の実測による受入は進行中。

追加: [0052](0052-owned-learner-device-allowance.md) — 通常のproducer-owned CUDA学習で、
論理的な更新上限と、常駐参加者を考慮した実際のGPU予約を別に扱う。

追加: [0051](0051-resident-actor-cuda-admission.md) — 常駐HPO学習に限り、推論参加者のCUDA予約を
単一意思決定の形状から計算。モデル種別ごとの時系列状態とcacheを実際の台帳へ反映する。

追加: [0050](0050-resident-training-cache-accounting.md) — 常駐HPO学習poolのモデル形状別cache容量と
並行BEGIN・逐次SYNCの予約を実行時ledgerに反映する。共有推論の仕様判断とlearnerのRAM/CUDA計算は別途。

追加: [0049](0049-configurable-hpo-game-concurrency.md) — Accepted, implemented and verified:
HPOの試合並列数とworker枠を設定可能にし、元のledger/config容量を
常駐workerと終了記録のPython/Rust検証へ伝える。既定2/4・同時trial1・
管理予算検査は維持し、実機OOM時の設定調整は利用者が行う。両独立レビューREADY、
8owner/4並列のCPU確認、Python/Rust検証、native更新と3モデルのCLI検証はPASS。
実学習・実機の容量／性能測定は手動実行として未実施。

追加: [0046](0046-standard-bounded-ppo-updates.md) — Accepted design:
利用者承認に基づくModel4通常経路の高速版統一と、長大入力の事前分割。
旧経路は内部検証用として保持。極端設定の容量保証は対象外とし、上限と失敗処理を維持。
事前設計・実装の両独立レビューと対象CUDAチェックはREADY。
通常9実行・6親192自然終了ゲームの厳密監査と固定入力確認はPASS。
両最終証拠レビューはFINAL READY、root受入済み。

追加: [0045](0045-learning-update-optimization.md) — Accepted,implemented and verified:
PPO更新の指標集約・入力一括化・microbatch・履歴共有を既存予算内で比較する。
当時の通常既定値は維持（通常経路の判断は0046で更新）。
84固定更新・18実比較・6別診断と24親の厳密監査を完了。
両最終レビューはFINAL READY、root受入済み。採用案はMLP microbatch、
GRU/Transformer shared-prefix。品質・正式HPO・final testの合格は主張しない。

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
