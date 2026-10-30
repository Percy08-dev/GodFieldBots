use gf_native::{
    budget::{Budget, Control, monotonic},
    error::Error,
    parallel::EpisodeReader,
    session::Reader,
    source::Sources,
    telemetry::Telemetry,
};
use std::{fs, sync::Arc};
#[allow(dead_code, unused_imports)]
#[path = "../src/typed_journal/dto/mod.rs"]
mod dto;

#[test]
fn result_wire_forms_match_legacy_and_reject_draw_objects() {
    for wire in [
        "null",
        r#""draw""#,
        r#"{"win":{"winner":1}}"#,
        r#"{"\u0077in":{"winner":2}}"#,
        r#"{"draw":null}"#,
        r#"{"win":null}"#,
        r#"{"win":{"winner":0}}"#,
        r#"{"win":{"winner":1,"extra":0}}"#,
        r#"{"win":{"winner":1},"extra":null}"#,
        r#"{"win":{"winner":1},"win":{"winner":2}}"#,
        r#""win""#,
        "false",
        "[]",
    ] {
        let value: serde_json::Value = serde_json::from_str(wire).unwrap();
        let legacy = gf_native::validation::result(&value).is_ok();
        let typed = serde_json::from_str::<Option<dto::borrowed::Outcome>>(wire).is_ok();
        if wire.contains(r#""win":{"winner":1},"win""#) {
            assert!(!typed, "duplicate wire keys must reject");
        } else {
            assert_eq!(typed, legacy, "{wire}");
        }
    }
    assert!(serde_json::from_str::<dto::borrowed::Outcome>("null").is_err());
}

#[test]
fn invalid_draw_objects_reject_in_all_journal_result_positions() {
    let mut observation: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/observation_p1_public.json"
    ))
    .unwrap();
    observation["result"] = serde_json::json!({"draw":null});
    let observation_wire = serde_json::to_string(&observation).unwrap();
    assert!(serde_json::from_str::<dto::borrowed::Observation>(&observation_wire).is_err());
    let wire=serde_json::to_string(&serde_json::json!({"kind":"decision","input":{"observation":observation,"events":[],"history_end":null,"typed_metadata":[],"catalog":{},"action_options":null}})).unwrap();
    assert!(serde_json::from_str::<dto::borrowed::Record>(&wire).is_err());
    for value in [
        serde_json::json!({"kind":"end","outcome":{"draw":null}}),
        serde_json::json!({"kind":"end","final_observation":observation}),
        serde_json::json!({"kind":"acknowledgement","response":{"command_id":"x","previous_revision":0,"state_revision":1,"status":"terminal","result":{"draw":null}}}),
    ] {
        let wire = serde_json::to_string(&value).unwrap();
        assert!(serde_json::from_str::<dto::borrowed::Record>(&wire).is_err());
    }
    let wire = r#"{"type":"game_ended","result":{"draw":null}}"#;
    assert!(serde_json::from_str::<dto::borrowed::Event>(wire).is_err());
    let replay = serde_json::json!({"game_id":"g","viewer_player_id":1,"observation_schema_version":"1","episode_end":{"type":"terminal","result":{"draw":null}}});
    assert!(gf_native::validation::replay_closure(&replay, "g", 1).is_err());
    assert!(
        gf_native::validation::reconcile_replay(&replay, 1, &[], &serde_json::json!("draw"))
            .is_err()
    );
}

#[test]
fn raw_document_refusal_releases_previously_queued_episode_without_stack_pressure() {
    let root = std::env::temp_dir().join(format!(
        "gf-raw-doc-quota-{}-{}",
        std::process::id(),
        monotonic().to_bits()
    ));
    fs::create_dir(&root).unwrap();
    let paths = [root.join("small"), root.join("large")];
    for path in &paths {
        fs::create_dir(path).unwrap();
        for name in [
            "player-1.jsonl",
            "player-2.jsonl",
            "replay-player-1.json",
            "replay-player-2.json",
        ] {
            fs::write(path.join(name), b"{}\n").unwrap();
        }
    }
    fs::write(paths[1].join("player-1.jsonl"), vec![b' '; 256 * 1024]).unwrap();
    let budget = Budget::new(2 * 1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    let telemetry = Arc::new(Telemetry::default());
    let mut sources = Sources::with_telemetry(budget.clone(), 1024 * 1024, 1024 * 1024, &telemetry);
    let mut reader = EpisodeReader::new(
        &paths,
        1,
        2,
        &budget,
        &control,
        1024 * 1024,
        1024 * 1024,
        Reader::TypedBorrowedV1,
    );
    assert!(
        matches!(reader.next(&mut sources),Err(Error::ResourceLimit(message)) if message.contains("JSON document estimate"))
    );
    drop(reader);
    drop(sources);
    assert_eq!(budget.used(), 0);
    let metrics: serde_json::Value = serde_json::from_str(&telemetry.json(0, 0, 0)).unwrap();
    assert_eq!(metrics["decode_parallelism"]["spawned_threads"], 0);
    assert_eq!(metrics["decode_parallelism"]["logical_slots"][0]["jobs"], 2);
    assert_eq!(
        metrics["decode_parallelism"]["logical_slots"][0]["failed_jobs"],
        1
    );
    fs::remove_dir_all(root).unwrap();
}
