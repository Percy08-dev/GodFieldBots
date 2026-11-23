# HPO6の〈夢〉で追加武器をDiscard候補に含める問題

旧構成で原因を特定し、拒否を再現しました。訂正版の採用と検証は
[ちからの粉の契約検証記録](../verification/strength-powder-simulator-contract.md)に記録しています。
HDD同期保存の採用判断は [ストレージ検証記録](../verification/staged-native-collection-storage.md) に残します。

## 対象と再現条件

HPO6 Model 4、学習seed 17、G1を使った第2収集 `batch-000001` のペア `p11`。
GameSeedは `70d3a731-c339-505c-9b2d-52ba0e238abe`。直接保存ランのJournalは
失敗した試合を `b000001-p11-seat0`、行動者を学習側Player 1、sampling seedを
`1810318435683261194` と記録しています。Simulatorは
`072dabbca1458c9526897781d2fd64edeeb4f9ca`、実行バイナリSHA-256は
`76aff19c2f720a572bfe422d9b1e089a1467f253038cda66de3d8c35180d0f7d`。
Gameplay API v2、catalog schema 2・public-ja-v3、Journal 6、candidate-columns-v2です。

| 保存方式 | 親ID | 拒否されたGame ID | HTTPログアーカイブのSHA-256 |
| --- | --- | --- | --- |
| HDDへの直接保存 | `69899b52-e549-4129-9cc1-9b47d8cc930f` | `01M837NG0DWZD3300WWY49T95R` | `06cde12a6dd21038e5961c13f363987a3988b030f19d4afbc911b5b6b5b2fcaa` |
| tmpfsからHDD ext4へ同期確定 | `c0db9169-22ed-4e57-96e3-ed66a22fd5fd` | `01M83DFTJ5F9E0F7RG82ZB4FMP` | `ea70bea938574324c14e01f938483cdd4b8220b33cffbddf64671f90e4564ee0` |

両ランでG1の `weights-000000.npz` は
`4e95228a2fbdf2285c87c18d987d3239f4d5387fbd384766b795821ff639c0e9`、
`resume-000000.npz` は
`4723bfc0288cbf5c935952f8ed5f04cb5f58a61a8895fce5d81eb3c68d06528d`
でバイト一致しました。親・作成情報を含む推論manifestのSHA-256は直接保存の
`725c12ec5d9fa1d3e5c46cd655c0ff0f71881b1b8ce7eeee7106b55bb5c312d6` と
tmpfs同期確定の `8fb3936c015f2b7633600b9b02ddbc47671d19f835ea956c8205ce0a9d564b09`
で異なります。

## 行動者に見えた状態と送信内容

両ランの受理済み39手は、行動者・revision・Actionで一致しました。
拒否直前の行動者向けObservation、Player-visible Events、ActionOptionsも一致しました。
revision 39、turn 34、`awaiting_action` / `turn_action`、終末前です。
Player 1はHP 8、MP 10、所持金20、〈夢〉と〈フラッシュ〉が有効でした。
手札の表示は太陽のお守り、力の粉、買う。ActionOptionsの `can_discard` は
`true` で、力の粉の参照は `usable_cards` と `additional_attack`・`utility`、
買うの参照は `buy_cards` にありました。`can_discard` は行動種別のフラグです。

直接保存ランの未確定Packを読み取り専用で開き、Packと候補列のハッシュを照合しました。
Player 1のJournalは個人内のdecision index 18をpendingとして記録しています。
その候補表の全7行と選択は次のとおりです。表は生成された提案を示します。

| 0始まりのindex | Action |
| ---: | --- |
| 0 | 買う、相手 |
| 1 | 買う、自分 |
| 2 | 力の粉を使用、相手 |
| 3 | 力の粉を使用、自分 |
| **4** | **力の粉を1枚捨てる** |
| 5 | 買うを1枚捨てる |
| 6 | 力の粉と買うを捨てる |

選択index 4を復号したActionはJournalのcommandと一致しました。直接保存のHTTP
ログのrequest_idで照合したPOST本文も、そのcommandと完全に一致しました。
送信先は `POST /v2/games/{game_id}/players/1/actions` で、本文の
`command_id` を除いた内容は両ランで同じです。

```json
{"expected_revision":39,"action":{"type":"discard","card_refs":["c1_9141215d2abe88a963a26ad8f84356a5"]}}
```

応答はHTTP 422、`INVALID_ACTION`、詳細コード `invalid_parameter`。
試合状態は `active`、revisionは39のまま、invalid attempt countは1でした。
Journalの対象decisionにACKはなく、episodeの終端は `incomplete` です。

## 原因と入力境界

オーナー確認では、ちからの粉はダメージ追加を持つ雑貨です。単独使用は
できず、武器と同じActionで使います。固定カタログでダメージ追加を持つ
雑貨は `strength-powder` 1種です。〈夢〉の互換候補がないため、
ちからの粉は偽装元にも偽装先にもなりません。
[固定カタログ](../../../GodFieldSimurator/assets/rules/true-match-1v1-v1/official-ja.json)

固定seedのTestServerで受理済み39手を再生した状態を診断専用Dumpで
確認しました。対象参照の実体は追加武器 `wand-of-mystic-water` で、
〈夢〉で `strength-powder` に見えていました。Simulatorの `dream_group` は
通常の追加武器と攻撃効果を持つ追加雑貨の両方に `weapon:additional:plain`
を割り当てます。これにより、互換性のないワンドからちからの粉への偽装が
発生しました。この誤った表示が422につながった発端です。
[Simulatorの偽装グループ](../../../GodFieldSimurator/src/engine/catalog/mod.rs)

Rust候補生成は公開カタログと行動者に見える定義から雑貨と判定し、
`can_discard=true` の局面でこの参照をDiscard候補に含めました。Engineは
実体が武器なので `WeaponCannotBeDiscarded` を返しました。この拒否自体は
Simulator spec 09 §18.4・§11.1の実体による判定と武器のDiscard禁止に従います。
[候補生成](../../native/src/player_runtime/search.rs)・
[SimulatorのDiscard処理](../../../GodFieldSimurator/src/engine/resolution/turn.rs)・
[拒否コードの投影](../../../GodFieldSimurator/src/server/history/record/mod.rs)

表のindex 2・3は、ちからの粉に見える参照を単独使用する提案です。この試合では
どちらも送信されていません。別の仕様差として、Simulatorの攻撃雑貨の分岐は
ダメージを加算する際に併用武器の有無を検査せず、Rustの攻撃候補生成も
追加攻撃を持つ雑貨だけの並びを許します。ちからの粉そのものを単独使用できない
ことの検証は、偽装グループの訂正と区別します。
[Simulatorの攻撃処理](../../../GodFieldSimurator/src/engine/resolution/choice.rs)・
[Rustの攻撃候補判定](../../native/src/player_runtime/rules.rs)

spec 07 §21.1の入力境界を維持します。再現テストは、同じ行動者の公開入力だけから
Native候補を生成し、診断Dumpはその後の原因照合に限ります。Dump、GameSeed、
他者の非公開情報、拒否詳細は方策入力に入れません。
[公開入力契約](../../../GodFieldSimurator/docs/spec/07-external-interfaces-adapters.md)

## 採用した修正方針

1. **Simulatorの〈夢〉**。 `dream_group` で、ダメージ追加の雑貨を追加武器から
   分けます。固定カタログのちからの粉に互換候補は0件となり、追加武器は
   ちからの粉を偽装先に選べなくなります。追加武器同士の既存の互換性を保ちます。
   候補が空の場合は既存の乱数消費規則に従います。この変更で後続の抽選結果が
   変わり得るため、旧p11の39手は固定版の再現証拠として保存します。
2. **Simulatorの使用判定**。Action確定前に、同じ並びで効果を使う実体の
   `ArtifactKind::Weapon` がある場合だけちからの粉を使えます。追加武器は
   この条件を満たし、習得した奇跡やCostCutの補助として扱う武器は条件を
   満たしません。攻撃値や攻撃元の種別で代用すると、ちからの粉自身がこの
   条件を満たしてしまいます。通常武器を先に置く既存の順序を適用します。
3. **Botの候補生成**。訂正版Simulatorと公開仕様を版固定した後、同じ条件を
   Python・Rustの攻撃候補に適用します。方策が参照するのは行動者に見える
   公開カタログの `weapons` 分類です。武器を追加できる探索途中の並びを
   調べる場合も、武器なしの組合せを完成候補としては出しません。訂正後の
   規則識別子を記録し、過去の候補表・チェックポイントは元の識別子で扱います。

Simulatorの受入ケースは、ちからの粉の偽装候補が空、通常の追加武器同士は
互換、粉だけの1枚・2枚は拒否、通常武器→粉は累計ダメージが正しい、
追加武器単独と追加武器＋粉は成立です。粉と攻撃奇跡だけの組合せや、
武器がCostCutの補助に回る並びも確認し、拒否時の状態・乱数を照合します。
Botでは同じ公開入力に対するPython・Rustの完成候補を照合します。
訂正版での受入には、旧p11の再生列とは別に制御された局面を使います。

## 再現手順と検証範囲

[テスト](../../tests/test_hpo_dream_discard_reproduction.py) は
[フィクスチャ](../../tests/fixtures/hpo_dream_discard_p11.json) にある受理済み39手を
ローカルの1試合で再生し、元の1手を送って同じ拒否を確認します。候補に該当Actionが
含まれることを行動者の公開入力だけで検証します。checkpointの同一性は保存済みの実行
記録で照合し、この再現はモデル推論を再実行せずに該当局面を作ります。
実行にはNative拡張を利用できるPython環境と、固定版に対応するTestServerを使います。

```sh
GFB_HTTP_DIAGNOSTICS_TEST_SERVER=../GodFieldSimurator/target/release/gf-test-server \
GFB_HTTP_DIAGNOSTICS_TEST_CONFIG=configs/test-server.toml \
GFB_HTTP_DIAGNOSTICS_TEST_SOURCE=runs/ppo-hpo-256-probe/simulator-072dabb \
  python -m pytest -q -n0 tests/test_hpo_dream_discard_reproduction.py
```

両方式の同一の送信内容と、学習・Pack転送を介さない同一拒否により、
ストレージ経路から独立した問題と確認しました。HPO6の大規模連続実行は第2収集で
停止しており、保存・再開・追加更新が通った8試合検証とは別の結果です。
失敗した第2収集に学習Sourceの資格はありません。
