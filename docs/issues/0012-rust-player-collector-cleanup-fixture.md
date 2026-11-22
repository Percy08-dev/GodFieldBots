# Rust playerのcollector cleanup fixture不足

状態は未修正。Journal連続取り出しの変更と独立した既存問題。

`tests/test_rust_player_matches.py` の
`test_collector_keeps_original_preflight_try_cleanup_order_aliases_and_loans`
は4つのパラメータ条件で、偽のregistryに `service` がないため、
想定しているpreflightやpumpの障害注入より前に失敗する。
後続のassertion後処理でも未作成の `_source_control_charge` を参照する。

Journal変更前の `a8b6a01` で、照合済みのRust拡張をロードして再現した。
実行結果は47件成功、4件失敗、17件スキップだった。
記録は `runs/native-journal-handoff/standard-baseline.log` にある。
対象の4件には、既存の `service.bind_collection` と
収集準備に必要なfixtureを与え、元の障害注入地点まで到達することを
別変更で確認する。
