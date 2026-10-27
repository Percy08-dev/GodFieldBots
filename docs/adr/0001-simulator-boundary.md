# 0001: シミュレーターを外部のルール権威とする

- Status: Accepted
- ActionOptions restriction: superseded by [ADR0005](0005-authorized-atomic-action-options.md)
  under the user's updated contract; SequenceOptions/rejection restrictions remain.
- Date: 2026-10-26
- Decision owner: root
- Review: ユーザーが承認したAgent構成・責任分界の計画に基づく。具体的な初期設計レビューは未実施。
- Implementation: 2026-10-27にPhase 0〜3の公開情報baseline・接続基盤を実装。
  [検証記録](../verification/phase0-3.md)を参照。学習Gateは未成立。

## Context

GodFieldBotsはPythonによるBot・学習・評価のリポジトリです。
ゲームシミュレーターは別のGodFieldSimuratorリポジトリに存在します。
その外部インターフェース仕様は、Training Clientを外部クライアントとし、
行動の合法性と自動的なゲーム効果の解決をGame Engineに置いています。

## Decision

ゲームルール・正当な状態遷移・最終的な行動合法性はシミュレーターが所有します。
Botsは公開されたプレイヤー入力から行動を選び、接続境界を通じて提出します。
データ処理・学習・評価はBots側の責務とし、シミュレーターの永続化と分離します。

方策はObservation、Player-visible Events、typed Action metadataを基礎とします。
ActionOptions / SequenceOptions / 拒否詳細から隠れた合法性を取得し、
方策の入力・mask・探索候補・教師信号に利用することは認めません。
モデル、接続方法、ライブラリ、具体的なPythonインターフェースはこのADRでは選択しません。

## Alternatives

- Bots側にゲームエンジンを複製する：二重管理とルールの不一致を招くため採用しません。
- サーバー内部状態や内部型に方策を直接結合する：情報境界と独立した変更を損なうため採用しません。

## Consequences

Botの実験はシミュレーターのversionと既知の制約に依存します。
結果には利用revisionと評価条件を残し、未確認ルールによる中断を勝敗へ変換しません。
シミュレーターの変更が必要な場合は別タスクとし、接続契約が変わるなら設計を再レビューします。

この判断は具体的な初期設計の承認を代替しません。
最初のBot実装前にインターフェースとデータフローを設計し、独立レビューを受けます。

## Verification

初回実装時に、境界の契約テスト、方策への情報漏洩の検証、
シミュレーターなしで行える方策の検証を具体化します。
学習データの由来と終了・打ち切り・中断の意味が保持されることも確認します。
現段階ではBot実装の検証結果はありません。
