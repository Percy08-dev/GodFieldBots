# HPO6 の防御候補で CostCut 専用のカードを区別する

Simulator v3 `997a0de` を Bots に取り込み、p46 の候補照合と CUDA 上の収集・保存を確認した。
[採用した Action profile](../adr/0079-simulator-v3-direct-defense.md) ·
[検証記録](../verification/simulator-v3-direct-defense.md)

## 再現と確認済みの挙動

親ラン `0f9a6874-1e26-4290-b34c-8b757884f6d4` は 70 更新、
284,009 学習判断を確定した後、`batch-000070` の `p46` で停止した。
Simulator は `6d50603dab3f6d2df34ea4e433bab7dca96f52c1`、Action profile は
`card-instance-simulator-v2-shared-20261125-sacrifice`。Player 2 の revision 32 は
防御待ちで MP 0。木属性の＜大木＞に精霊の頭巾だけを選び、HTTP 422
`INVALID_ACTION / unavailable_artifact` を受けた。
[保存した再現データ](cases/0019-defense-costcut-role.json)

同じ Simulator バイナリで受理済み 32 手を再生した。保存した Player 2 の
Observation と ActionOptions が再生時の応答と一致した。`［＜乱気流＞、精霊の頭巾］`
を送ると revision 33 まで進み、`miracle_used.mp_spent` は 0 だった。
頭巾単独の拒否は元の HTTP ログに記録されている。

元の Player 2 の公開入力を使った固定クロックの検索で、Python と元ランと同一 SHA-256
の Rust 拡張が同じ候補表を生成した。候補は見送り、頭巾単独、乱気流の直後に頭巾を
置いた応答の 3 件だった。`actor_public` だけを `decision_input` に渡し、固定カタログ
から当人の見えている定義を解決する。`replay_controls` の両席の操作と GameSeed は
隔離した TestServer の再生用。`rejected` は診断用。
[Python の判定](../../src/gf_common/card_instance_rules.py#L251) ·
[Rust の判定](../../native/src/player_runtime/rules.rs#L174)

元ログの SHA-256 は `5bdea00a4762950646e633b7c44ce32a55b33bf159161a8702f73c7ca6735e7c`。
該当ランは `failed` かつ `quarantine=true`。この失敗と既存の成果物はその時点の契約に属する。

## GodFieldSimurator の先行タスク

防御時の `usable_cards` は、防御に直接使える参照と、対応する奇跡の後ろで CostCut に
使える参照をまとめる。`candidate_groups` の既存グループは見えているカードの分類であり、
局面で成立する個別の役割までは示さない。攻撃属性は当該 Player-visible Events に含まれず、
＜大木＞は今回の Player 2 の手札・習得済みカタログの範囲外だった。
[固定版 Simulator の候補生成](../../runs/pestle-mirage-compat/simulator/src/engine/resolution/entry.rs#L278) ·
[固定版 Simulator の公開投影](../../runs/pestle-mirage-compat/simulator/src/server/observation.rs#L292)

候補案は、`candidate_groups` に `defense:direct` を追加し、現在の効果に直接応答できる
カードの公開参照を列挙すること。MP 支払い、物理枚数、順序、他カードの効果による属性変更
に先立つ atomic hint とする。同一 revision の行動者、`usable_cards` の部分集合、
Observation と同じ見かけの定義という制約を課す。新しい契約では防御時に空でも明示し、
防御以外では省く。元の flat list と静的グループは既存の意味を維持する。actor GET と、
同じ公開投影を使う actor の Act / Replay の対応箇所で一貫することを確認する。

この区別は一部の局面で攻撃との適合性を新たに伝える。§27 と §13、§21.1 で公開を明示的に
許可する範囲を決め、Dream と Fog の影響をレビューする。公開する値は当該役割の参照と
見かけの定義に限る。攻撃の内部属性、実際のカード定義、拒否理由は公開しない。
役割の判定には Engine の既存の応答判定を使い、投影は読み取り専用とする。
[Observation 仕様](../../../GodFieldSimurator/docs/spec/05-observation.md#27-observation-and-actionoptions-schema-2) ·
[学習入力契約](../../../GodFieldSimurator/docs/spec/07-external-interfaces-adapters.md#211-standard-learning-input-contract)

受け入れ条件は、p46 で頭巾が flat list に残り `defense:direct` から外れること、
同じ頭巾が無属性攻撃には直接応答の候補になること、正順の応答が MP 0 で受理されること。
違う revision や非行動者の参照を含めず、Dream の表示と Fog の公開制約を満たすこと。
Bots への引き渡しには、公開仕様、Simulator のリビジョン、仕様・ソースのインベントリ、
対応 TestServer とその SHA-256、上記の検証結果が必要になる。
先行リリースは `6d50603d` を基準に、Strength Powder、Dream、満月、
Dangerous Pestle と Mirage の既存修正を引き継ぐ。
[現行 Simulator リリースの構成](../adr/0070-pestle-mirage-simulator.md)

## GodFieldBots の取り込み条件

- 検証済みの Simulator 契約を新しい Action profile として pin する。既存 profile の
  候補表と署名を保存し、新規 HPO は新しい設定と保存先を使う。新 profile では従来の
  Strength Powder の武器要件、Mirage の対象、Discard / Sacrifice の保護も継承する。
  モデルカタログに対応する入力契約を記録する。
- Python と Rust は行動者の同一 revision の参照をフレーム単位でスロットに解決する。
  新 profile の防御でグループが欠けている場合は候補選択の前に契約エラーにする。
  奇跡の直後で CostCut として消費される精霊の防具は引き続き使える。防御効果を担うときに
  直接応答の hint を照合し、先頭の虹のカーテンが公開ルールに従って属性を消す場合を扱う。
  探索ループ内で公開イベントの再走査やカード参照の再変換を行わない。
- p46 で頭巾単独と逆順を除外し、見送りと乱気流から頭巾への正順を残す。無属性攻撃、
  先頭の虹のカーテン、閃光の物理 1 枚制限も、同じ入力で両実装を比較する。
  新旧契約の候補署名を照合した後、対応する Native ビルドで HPO6 の preflight と
  短い収集・保存の確認を行う。
