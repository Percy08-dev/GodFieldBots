# Simulator handoff: participant-scoped inference in a shared tensor executor

Status: resolved upstream by Simulator PR73 at
`072dabbca1458c9526897781d2fd64edeeb4f9ca`; Bots adoption/implementation is in
progress under [the MLP design](../design/mlp-shared-batch-inference.md).
The adopted spec07 SHA-256 is
`8e5223b45a6e8c4c80e272736ed97b9e271136c939eca74427a035ddfdbfcac2`.
Its section21.1 permits trusted shared execution subject to participant
independence, immutable request checkpoints, exact response matching and
retirement verification before enabling batching. The text below preserves the
historical91d5926 question and proposal; it does not describe the new contract.

## Question for the Simulator contract owner

May one trusted inference process execute a batch whose rows derive from the
private views of distinct participants, including the two seats, when each
row's result depends only on that participant's authorized inputs and the
bound inference-only checkpoint? This would change where private data may
coexist in memory. A row mask by itself does not authorize that data flow.

In Simulator revision `91d5926a7572db327eaea3751e7c24f428120996`,
`docs/spec/07-external-interfaces-adapters.md` §21.1 explicitly restricts
delivery of another participant's view to the policy process. Section 22
allows an orchestrator to manage both seats and the same model **version**
to be used for both. The current separate participant processes meet both
conditions. The proposed TorchRL process computes action probabilities and
values, and receives rows derived from multiple private observations. Neither
section grants an exception for this execution topology. A search of the
provided Simulator docs found no separate shared-inference exception.

## Requested clarification

If the owner permits batching, specify the allowed executor, ownership of
inputs/outputs, isolation across rows, participants, games and checkpoints,
and when retained private buffers or state must be cleared. Participant
workers would keep candidate generation, action sampling and command creation.
Only the requesting participant would receive that row's outputs. Keep other
participants' views, hidden canonical history, rejection detail, seeds, future
information, optimizer and resume state out of that participant's policy
features. Source, audit, control and training remain separate responsibilities.

If permission is declined, record that decision and select a different design.
A separately checked Simulator specification revision and hash are needed
before adopting a changed contract in Bots. This document is a handoff;
it grants no permission and makes no changes to the Simulator repository.

## Proposed wording for the separate specification decision

The following is a reviewable proposal for §21.1, not an amendment in force:

> 参加者ごとの方策は、同じ参加者に許可された観測・イベント・候補だけを使用する。
> 推論を別プロセスに委譲する場合に限り、信頼された計算サービスは複数参加者の
> 入力テンソルを同じバッチで処理できる。各行の出力は、その参加者の入力・
> 履歴状態および固定したモデルの重みだけで決まること。他の行との正規化・
> attention・状態の共有を禁止する。応答は元の要求を発行した参加者にだけ返し、
> 別参加者の入力・中間値・出力を方策へ渡さない。状態・応答は試合、参加者、
> 世代をまたいで誤用できないようにし、終了時には当該参加者の状態を退役させる。
> この例外は推論計算に限定する。Simulatorの内部状態、GameSeed、rejectionの詳細、
> 他者視点から生成した教師信号を方策に供給する権限を追加しない。

Verification on adoption would trace two requests from different participants
and pinned versions through candidate masking, response delivery and terminal
cleanup. Any accepted wording and spec digest should come from the Simulator
owner; an approved decision may differ from this proposal.
