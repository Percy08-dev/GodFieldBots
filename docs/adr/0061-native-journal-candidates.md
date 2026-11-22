# 0061 Journalの全候補照合と候補列エンコードをRustで実行する

状態はAccepted。独立した設計・実装レビューと実機検証を確認済み。

## 背景

[writerのCPU比較](../verification/journal-writer-cpu.md)では、共有CardInstanceの
全候補照合とcodec2をRustで処理すると、実同時256試合でModel 3が17.2%、
Model 4が19.2%短縮した。Journal v6の書き込みとグループ確定は既存の
Python writerが担当できる。Simulatorは `072dabbca1458c9526897781d2fd64edeeb4f9ca`、
学習入力は同リビジョンの外部インターフェース仕様 §21.1 に従う。

## 決定

- 確定済みのエピソードコンテキストが現行の共有CardInstance契約に一致する場合、
  writerは同じアカウントが所有する元の `StorageBundle` を使う。他の契約は
  従来のPython検証とcodec2経路を使う。
- Rustは同じwriterスレッド上でGILを解放し、元の不変バイト列から
  候補表の構造と正規SHA-256、同じフレームの全候補の種類・参照・引数、
  選択行動と保存する候補列を検証・生成する。入力を借用し、出力のRust
  バッファはPythonの `bytes` に1回コピーする。
- Pythonは元のtransport SHA、観測とフレーム、イベント、Search report、
  選択行動とコマンド、PPO用数値およびJournalの順序を検証する。
  書き込み前に、Rustの結果を同じ元レコードとそのフレームに結合する。
  公開Journal APIは従来どおり全候補を確かめ、検証済み入力を処理する
  privateな書き込み手順を共有する。
- Rustの解析列は既存のwriterのprivate作業領域に含める。
  実際の列容量とサイズを計測し、出力バッファとPythonの `bytes` が
  重なる区間は既存RecordingBudgetで予約する。元の貸与バッファは
  既存の所有者が計上する。
- 不正な共有プロファイルのレコードは既存の障害経路で処理する。
  ACK、terminal、seal、Pack・SQLiteの確定と退役の順序は
  [ADR0060](0060-rust-player-journal-handoff.md) のとおり。
  Journal v6、codec2、checkpointの保存形式とproducerの記録方式を引き継ぐ。
  [ADR0057](0057-rust-ppo-training-data.md) に従い、
  一時停止中のrunは記録済みのproducerで再開する。

## 検証

全候補と元の参加者観測の照合、未選択の異常候補、ハッシュ、
候補列・Journal・Replayのバイト一致、失敗時のACKと退役を確認する。
通常PPOとHPOの保存・再開から追加収集とPPO更新までを検証する。
実同時256試合の反復A/Bと、独立したwriter・GPU推論の区間計測を行う。

収集時間はModel 3で21.6%、Model 4で21.1%短縮した。
詳細は [導入検証](../verification/native-journal-writer.md) に記録する。
