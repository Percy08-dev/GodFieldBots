# 0073 共有 Journal のフレーム検証を Rust に移す

状態は Accepted。

## 背景

共有 CardInstance のライブ writer は原本に結び付けた候補表を
[ADR0061](0061-native-journal-candidates.md) のとおり Rust で照合する。
フレーム DTO は Python が再構築していた。現行 Sacrifice ランの保存済み
フレーム計372件の診断で、Rust の既存検証は正規 payload バイトまで一致し、
フレーム単体では単一ワーカーでも短縮した。

## 決定

共有 CardInstance のライブ pending で、private な StorageBundle 操作が
同じ不変原本のフレーム範囲を検証し、writer に確定フィールドと正規 payload
バイトを渡す。フレームハッシュは保存済みの corrected layout と照合し、
現行実行契約と候補表・コマンドの照合は従来の経路で適用する。
Journal と学習用の保存データは同じ検証済み payload バイトを使う。
writer は試合タスクが独立に作る学習用数値行を、この検証済み原本と照合する。

先に単一 writer 上で GIL を解放して検証する実装を採用する。source
workspace の既存貸与から子 scratch を一つ使い、native 資源制限で
処理できないフレームは、元の期限内に従来の Python 検証を通す。
試合内 EventBank、原本の SHA-256、全候補、選択コマンド、Replay の照合と
FIFO の受領 ACK は既存の確定条件を引き継ぐ。

## 検証と見直し

[実機検証](../verification/journal-native-frame.md) に採否、同じ原本からの
出力、失敗、通常 PPO と HPO の保存・再開、ピークメモリの結果を記録する。
共有経路の旧ランの保存形式と producer による再開の扱いは
[ADR0057](0057-rust-ppo-training-data.md) に従う。

試合ワーカーでの事前検証は、記録待ちの p95 と writer CPU 時間をさらに下げた。
256試合の比較で中央値差が対照の測定幅を超えた場合に採用する取り決めに従い、
この候補は見送った。並列化を再評価する場合は、別の代表的な負荷または
試合を最後まで進める時間に寄与する追加の計測を根拠とする。
