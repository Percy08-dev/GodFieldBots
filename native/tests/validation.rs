use gf_native::budget::{Budget, Control, monotonic};
use gf_native::json::parse;
use gf_native::validation::{observation, replay_closure, validate_definition, validate_events};

#[test]
fn fixture_observation_preserves_closed_integer_boundary() {
    let fixture = include_str!("../../tests/fixtures/observation_p1_public.json");
    let budget = Budget::new(1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    let valid = parse(fixture.as_bytes(), &budget, &control).unwrap();
    observation(&valid.root, Some(1)).unwrap();
    assert!(observation(&valid.root, Some(2)).is_err());
    for invalid in [
        fixture.replacen("\"state_revision\": 0", "\"state_revision\": true", 1),
        fixture.replacen("\"hp\": 40", "\"hp\": null", 1),
        fixture.replacen(
            "\"additional_fields\": false",
            "\"additional_fields\": 0",
            1,
        ),
        fixture.replacen("\"turn\": 1", "\"turn\": 4294967296", 1),
    ] {
        let value = parse(invalid.as_bytes(), &budget, &control).unwrap();
        assert!(observation(&value.root, None).is_err());
    }
}

#[test]
fn event_definition_and_closure_boundaries() {
    let budget = Budget::new(1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    for (raw, valid) in [
        (
            r#"[{"type":"damage_applied","player":1,"amount":-2147483648}]"#,
            true,
        ),
        (
            r#"[{"type":"damage_applied","player":1,"amount":1,"remaining_hp":null}]"#,
            false,
        ),
        (r#"[{"type":"action_accepted","player":true}]"#, false),
        (r#"[{"type":"rejection","legal":true}]"#, false),
        (
            r#"[{"type":"game_ended","result":{"win":{"winner":2}}}]"#,
            true,
        ),
    ] {
        let value = parse(raw.as_bytes(), &budget, &control).unwrap();
        assert_eq!(validate_events(&value.root).is_ok(), valid, "{raw}");
    }
    for (raw, valid) in [
        (
            r#"{"definition_id":"visible","display_name":"日本語","kind":"weapons","attack":-2147483648}"#,
            true,
        ),
        (
            r#"{"definition_id":"visible","display_name":"x","kind":"weapons","attack":true}"#,
            false,
        ),
        (
            r#"{"definition_id":"visible","display_name":"x","kind":"weapons","hidden":0}"#,
            false,
        ),
    ] {
        let value = parse(raw.as_bytes(), &budget, &control).unwrap();
        assert_eq!(validate_definition(&value.root).is_ok(), valid, "{raw}");
    }
    let raw = br#"{"game_id":"g","viewer_player_id":1,"observation_schema_version":"1","episode_end":{"type":"aborted","reason":"turn_limit_exceeded","player":null,"decision":{"turn":null,"decision_index":0,"invalid_attempt_count":0}}}"#;
    let value = parse(raw, &budget, &control).unwrap();
    let closure = replay_closure(&value.root, "g", 1).unwrap().unwrap();
    assert_eq!(
        closure.require("type").unwrap().as_str().unwrap(),
        "aborted"
    );
    assert!(replay_closure(&value.root, "foreign", 1).is_err());
}

#[test]
fn controlled_validation_releases_scratch_on_budget_and_cancel_failure() {
    use gf_native::error::Error;
    use gf_native::validation::observation_controlled;
    let budget = Budget::new(1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    let value = parse(
        include_bytes!("../../tests/fixtures/observation_p1_public.json"),
        &budget,
        &control,
    )
    .unwrap();
    let original = budget.used();
    let pressure = budget.reserve(budget.limit() - original).unwrap();
    assert!(matches!(
        observation_controlled(&value.root, None, &control),
        Err(Error::ResourceLimit(_))
    ));
    drop(pressure);
    assert_eq!(budget.used(), original);
    observation_controlled(&value.root, None, &control).unwrap();
    assert_eq!(budget.used(), original);
    control.cancel();
    assert!(matches!(
        observation_controlled(&value.root, None, &control),
        Err(Error::Cancelled)
    ));
    assert_eq!(budget.used(), original);
}

#[test]
fn long_event_walk_observes_concurrent_cancellation() {
    use gf_native::error::Error;
    use gf_native::validation::validate_events_controlled;
    let budget = Budget::new(64 * 1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    let mut events = String::from("[");
    for index in 0..10000 {
        if index > 0 {
            events.push(',');
        }
        events.push_str(
            r#"{"type":"resource_changed","player":1,"hp_delta":0,"mp_delta":0,"money_delta":0}"#,
        );
    }
    events.push(']');
    let value = parse(events.as_bytes(), &budget, &control).unwrap();
    let cancellation = control.clone();
    let thread = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(1));
        cancellation.cancel();
    });
    let result = validate_events_controlled(&value.root, &control);
    thread.join().unwrap();
    assert!(matches!(result, Err(Error::Cancelled)));
}

#[test]
fn worker_and_journal_iterable_coercions_remain_distinct() {
    use gf_native::budget::BudgetVec;
    use gf_native::json::{Text, Value, canonical};
    use gf_native::validation::{
        validate_journal_input_controlled, validate_worker_input_controlled,
    };
    let budget = Budget::new(1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    let mut obs = parse(
        include_bytes!("../../tests/fixtures/observation_p1_public.json"),
        &budget,
        &control,
    )
    .unwrap();
    let Value::Object(fields) = &mut obs.root else {
        unreachable!()
    };
    for (key, value) in fields.iter_mut() {
        match key.as_str().unwrap() {
            "actionable" => *value = Value::Bool(false),
            "phase" => *value = Value::String(Text::from_str("terminal", &budget).unwrap()),
            "typed_action_schema" => *value = Value::Array(BudgetVec::new(&budget)),
            "own" => {
                let Value::Object(fields) = value else {
                    unreachable!()
                };
                for (key, value) in fields.iter_mut() {
                    if key == "hand" {
                        *value = Value::Array(BudgetVec::new(&budget));
                    }
                }
            }
            _ => (),
        }
    }
    let obs = canonical(&obs.root, &budget, &control).unwrap();
    let obs = std::str::from_utf8(&obs).unwrap();
    for events in ["[]", "{}", "\"\""] {
        for metadata in ["[]", "{}", "\"\""] {
            let raw = format!(
                r#"{{"observation":{obs},"events":{events},"history_end":null,"typed_metadata":{metadata},"catalog":{{}},"action_options":null}}"#
            );
            let input = parse(raw.as_bytes(), &budget, &control).unwrap();
            assert_eq!(
                validate_worker_input_controlled(&input.root, None, &control).is_ok(),
                metadata == "[]",
                "worker {raw}"
            );
            validate_journal_input_controlled(&input.root, None, &control).unwrap();
        }
    }
}

#[test]
fn replay_reconciliation_keeps_python_equality_and_optional_targets() {
    use gf_native::json::Value;
    use gf_native::validation::reconcile_replay_controlled;
    let budget = Budget::new(1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    let replay = parse(br#"{"viewer_player_id":true,"observation_schema_version":"1","turns":[{"decisions":[{"actor":1.0,"visible_action":{"type":"action","action":{"type":"play_sequence","ordered_elements":[{"source":"hand_artifact","instance_id":true}]}}}]}]}"#, &budget, &control).unwrap();
    let action = parse(br#"{"type":"play_sequence","ordered_elements":[{"source":"hand_artifact","instance_id":1}],"target":null}"#, &budget, &control).unwrap();
    reconcile_replay_controlled(&replay.root, 1, &[&action.root], &Value::Null, &control).unwrap();
    for empty in ["{}", "\"\""] {
        let raw =
            format!(r#"{{"viewer_player_id":1,"observation_schema_version":"1","turns":{empty}}}"#);
        let replay = parse(raw.as_bytes(), &budget, &control).unwrap();
        reconcile_replay_controlled(&replay.root, 1, &[], &Value::Null, &control).unwrap();
    }
    let malformed = parse(br#"{"viewer_player_id":1,"observation_schema_version":"1","turns":[{"decisions":[{"actor":1,"visible_action":true}]}]}"#, &budget, &control).unwrap();
    assert!(reconcile_replay_controlled(&malformed.root, 1, &[], &Value::Null, &control).is_err());
}
