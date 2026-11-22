# 0062 ライブJournal writerのイベントprefix照合

状態はAccepted。独立した設計・実装レビューと実機検証を確認済み。

## 背景

[ADR0061](0061-native-journal-candidates.md) の導入後、ネイティブのライブwriterが
保存する各手で、既存のイベント履歴からビューを作り、フレームが参照する
prefixビューを作る過程で2回の全履歴照合を行っていた。
方式Cの再計測で、この二つの照合はModel 3で3.34 CPU秒、Model 4で3.61 CPU秒だった。
保持済みprefixを使うPoCでは、実同時256試合の収集時間が6.4%と8.1%短縮した。

## 決定

- [ADR0061](0061-native-journal-candidates.md) に従って元の `StorageBundle` を
  処理するライブwriterだけで、所有する `EventBank` の開始prefixを直接確認する。
- `EventDelta` は追加前にイベントのスキーマを検査し、不変の正規化バイト列を保持する。
  `EventBank.append` が所有者・レイアウト・開始位置・直前ハッシュ・上限を確認し、
  各イベントを1件ずつ、そのバイト列から計算したprefixと対応させて追加する。
- ライブwriterは追記後、現在の終端境界とフレームの境界、所有者とレイアウト、
  フレームの開始位置に保存したprefixを照合してから候補列とJournalを発行する。
  保存済みデータの読取、公開ビュー、旧経路、terminalとReplayの独立した照合には
  既存の履歴検査を使う。
- エラー・ACK・terminal・sealの責務と既存の記録形式を維持する。
  Simulatorは `072dabbca1458c9526897781d2fd64edeeb4f9ca`。Journalはv6、
  候補列はcodec2。保存時のproducerによる再開は [ADR0057](0057-rust-ppo-training-data.md)
  のとおり。

Rustで候補表を除いたJSONを返す候補は、別PoCで両モデル約2.9%短縮した。
追加の呼び出しと走査を含めたWriter CPUの削減が小さく、比較結果を
`runs/native-writer-next/` に保存する。

## 検証

同じ実原本から生成するJournal・Replay・候補列のバイト一致を比較する。
Frameおよび候補表のSHA-256を再計算した開始ハッシュの不一致が、
発行・ACKより前に拒否されることを検証する。
固定条件の実同時256試合の反復A/B、通常PPOとHPOでの再開後の追加収集と更新、
別枠のwriterとGPU推論の診断結果で受け入れを判断する。

反復比較ではModel 3が9.6%、Model 4が10.7%短縮した。
保存・再開と再計測を含む詳細は [導入検証](../verification/owned-event-prefix.md) に記録する。
