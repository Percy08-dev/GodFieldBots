use crate::{borrowed, contract::Validate, feasibility::Text, owned};
use std::borrow::Cow;

#[test]
fn owned_and_borrowed_observation_contract() {
    for raw in [
        include_str!("../../../../tests/fixtures/observation_p1_public.json"),
        include_str!("../../../../tests/fixtures/actor_options_observation_public.json"),
    ] {
        let a: owned::Observation = serde_json::from_str(raw).unwrap();
        a.validate().unwrap();
        let b: borrowed::Observation = serde_json::from_str(raw).unwrap();
        b.validate().unwrap();
        assert_eq!(
            serde_json::to_vec(&a).unwrap(),
            serde_json::to_vec(&b).unwrap()
        );
        assert!(matches!(
            b.own.hand[0].definition_id,
            Text(Cow::Borrowed(_))
        ));
    }
}

#[test]
fn both_ownerships_reject_missing_duplicate_null_unknown_numeric_domains() {
    for raw in [
        r#"{"index":0,"policy_version":"p","value":null}"#,
        r#"{"index":0,"policy_version":"p","logprob":null,"value":null,"index":1}"#,
        r#"{"index":0,"policy_version":"p","logprob":null,"value":null,"extra":1}"#,
        r#"{"index":0.0,"policy_version":"p","logprob":null,"value":null}"#,
        r#"{"index":-1,"policy_version":"p","logprob":null,"value":null}"#,
        r#"{"index":0,"policy_version":"p","logprob":1e999,"value":null}"#,
    ] {
        assert!(
            serde_json::from_str::<owned::Selection>(raw).is_err(),
            "{raw}"
        );
        assert!(
            serde_json::from_str::<borrowed::Selection>(raw).is_err(),
            "{raw}"
        );
    }
    for raw in [
        r#"{"type":"exchange","artifact":1,"hp":2147483648,"mp":0,"money":0}"#,
        r#"{"type":"buy","artifact":1,"target":3}"#,
    ] {
        assert!(serde_json::from_str::<owned::Action>(raw).is_err());
        assert!(serde_json::from_str::<borrowed::Action>(raw).is_err());
    }
}

#[test]
fn tagged_actions_validate_shape_without_content_or_value() {
    for raw in [
        r#"{"type":"respond","ordered_defenses":[]}"#,
        r#"{"type":"respond","ordered_defenses":[1,2]}"#,
        r#"{"type":"respond_sequence","ordered_defenses":[{"source":"hand_artifact","instance_id":2}]}"#,
        r#"{"type":"play_sequence","ordered_elements":[{"source":"learned_miracle","definition_id":"x"}],"target":null}"#,
    ] {
        let a: owned::Action = serde_json::from_str(raw).unwrap();
        a.validate().unwrap();
        let b: borrowed::Action = serde_json::from_str(raw).unwrap();
        b.validate().unwrap();
        assert_eq!(
            serde_json::to_vec(&a).unwrap(),
            serde_json::to_vec(&b).unwrap()
        );
    }
    for raw in [
        r#"{"type":"respond","ordered_defenses":[{"source":"hand_artifact","instance_id":2}]}"#,
        r#"{"type":"respond_sequence","ordered_defenses":[2]}"#,
        r#"{"type":"pray","artifact":null}"#,
        r#"{"type":"buy","artifact":1,"target":null}"#,
        r#"{"type":"pray","artifact":1}"#,
        r#"{"type":"respond"}"#,
    ] {
        let a: owned::Action = serde_json::from_str(raw).unwrap();
        assert!(a.validate().is_err());
        let b: borrowed::Action = serde_json::from_str(raw).unwrap();
        assert!(b.validate().is_err());
    }
}

#[test]
fn nested_dynamic_keys_reject_duplicates_and_borrow() {
    let raw =
        r#"{"type":"pray","required_fields":{},"optional_fields":{},"additional_fields":false}"#;
    let a: owned::Schema = serde_json::from_str(raw).unwrap();
    a.validate().unwrap();
    let b: borrowed::Schema = serde_json::from_str(raw).unwrap();
    b.validate().unwrap();
    let dup = r#"{"type":"pray","required_fields":{"x":{"type":"null"},"\u0078":{"type":"null"}},"optional_fields":{},"additional_fields":false}"#;
    assert!(serde_json::from_str::<owned::Schema>(dup).is_err());
    assert!(serde_json::from_str::<borrowed::Schema>(dup).is_err());
}

#[test]
fn both_ownerships_reject_decoded_reserved_dynamic_keys_but_allow_values() {
    // The shared helper intentionally aligns direct standalone DTO acceptance
    // with production's existing reserved-key restriction, for String and Text.
    for (raw, expected) in [
        (r#"{"const":"$serde_json::private::Number"}"#, true),
        (
            r#"{"required_fields":{"ordinary":{"const":"$serde_json::private::Number"}}}"#,
            true,
        ),
        (
            r#"{"required_fields":{"$serde_json::private::Number":{}}}"#,
            false,
        ),
        (
            r#"{"required_fields":{"\u0024serde_json::private::Number":{}}}"#,
            false,
        ),
        (
            r#"{"optional_fields":{"ordinary":{},"$serde_json::private::Numbe\u0072":{}}}"#,
            false,
        ),
        (
            r#"{"items":{"one_of":[{"required_fields":{"x":{"optional_fields":{"$serde_json::private::Number":{}}}}}]}}"#,
            false,
        ),
        (r#"{"required_fields":{"x":{},"\u0078":{}}}"#, false),
    ] {
        assert_eq!(
            serde_json::from_str::<owned::Schema>(raw).is_ok(),
            expected,
            "{raw}"
        );
        assert_eq!(
            serde_json::from_str::<borrowed::Schema>(raw).is_ok(),
            expected,
            "{raw}"
        );
    }
    // The key must reject before its value is consumed, including on malformed
    // values. Both actual key representations use the same MapAccess visitor.
    let raw = r#"{"\u0024serde_json::private::Number": [invalid}"#;
    let owned =
        serde_json::from_str::<crate::contract::UniqueMap<String, String>>(raw).unwrap_err();
    let borrowed =
        serde_json::from_str::<crate::contract::UniqueMap<Text, String>>(raw).unwrap_err();
    for error in [owned, borrowed] {
        assert!(
            error
                .to_string()
                .contains("reserved serde numeric member key")
        );
    }
}

#[test]
fn both_custom_visitors_reject_arbitrary_precision_synthetic_number_maps() {
    for raw in [
        "18446744073709551616",
        "9007199254740993.0",
        "1e999",
        r#"{"$serde_json::private::Number":"1"}"#,
        r#"{"\u0024serde_json::private::Number":"1"}"#,
    ] {
        assert!(
            serde_json::from_str::<owned::Outcome>(raw).is_err(),
            "{raw}"
        );
        assert!(
            serde_json::from_str::<borrowed::Outcome>(raw).is_err(),
            "{raw}"
        );
        assert!(
            serde_json::from_str::<owned::Defense>(raw).is_err(),
            "{raw}"
        );
        assert!(
            serde_json::from_str::<borrowed::Defense>(raw).is_err(),
            "{raw}"
        );
    }
    assert!(serde_json::from_str::<owned::Defense>("18446744073709551615").is_ok());
    assert!(serde_json::from_str::<borrowed::Defense>("18446744073709551615").is_ok());
    for raw in [r#""draw""#, r#"{"win":{"winner":1}}"#, r#"{"win":[2]}"#] {
        assert!(serde_json::from_str::<owned::Outcome>(raw).is_ok(), "{raw}");
        assert!(
            serde_json::from_str::<borrowed::Outcome>(raw).is_ok(),
            "{raw}"
        );
    }
}

#[test]
fn required_nullable_presence_is_preserved_in_record_contract() {
    let raw =
        r#"{"kind":"transition","decision":0,"next_decision":null,"reward":1,"terminal":true}"#;
    let a: owned::Record = serde_json::from_str(raw).unwrap();
    a.validate().unwrap();
    let b: borrowed::Record = serde_json::from_str(raw).unwrap();
    b.validate().unwrap();
    let missing = raw.replace("\"next_decision\":null,", "");
    let a: owned::Record = serde_json::from_str(&missing).unwrap();
    assert!(a.validate().is_err());
    let b: borrowed::Record = serde_json::from_str(&missing).unwrap();
    assert!(b.validate().is_err());
}

#[test]
fn no_ap_diagnostic_records_precision_representation_and_acceptance_differences() {
    let integer = "18446744073709551617";
    let decimal = "9007199254740993.0";
    let a: serde_json::Value = serde_json::from_str(integer).unwrap();
    let b: serde_json::Value = serde_json::from_str(decimal).unwrap();
    let overflow = serde_json::from_str::<serde_json::Value>("1e999");
    if cfg!(feature = "arbitrary_precision") {
        assert_eq!(a.to_string(), integer);
        assert_eq!(b.to_string(), decimal);
        assert!(overflow.is_ok());
        assert!(!overflow.unwrap().as_f64().is_some_and(f64::is_finite));
    } else {
        assert_ne!(a.to_string(), integer);
        assert!(a.as_number().unwrap().is_f64());
        assert_ne!(b.to_string(), decimal);
        assert_eq!(b.as_f64(), Some(9007199254740992.0));
        assert!(overflow.is_err());
    }
}

fn sharing_input() -> serde_json::Value {
    let observation: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/observation_p1_public.json"
    ))
    .unwrap();
    serde_json::json!({"observation":observation,"events":[],"history_end":null,
        "typed_metadata":observation["typed_action_schema"],"catalog":{},"action_options":null})
}
fn input_outcome(raw: &str) -> (bool, bool) {
    let owned = serde_json::from_str::<owned::Input>(raw);
    let borrowed = serde_json::from_str::<borrowed::Input>(raw);
    let result = match (&owned, &borrowed) {
        (Ok(left), Ok(right)) => {
            assert_eq!(
                std::sync::Arc::ptr_eq(
                    &right.typed_metadata,
                    &right.observation.typed_action_schema
                ),
                left.typed_metadata == left.observation.typed_action_schema
            );
            assert_eq!(
                serde_json::to_vec(left).unwrap(),
                serde_json::to_vec(right).unwrap()
            );
            assert_eq!(left.validate().is_ok(), right.validate().is_ok());
            (true, left.validate().is_ok())
        }
        (Err(left), Err(right)) => {
            assert_eq!(left.classify(), right.classify());
            (false, false)
        }
        _ => panic!("owned/borrowed decode acceptance differs"),
    };
    result
}

#[test]
fn shared_schema_wire_fields_encodings_value_and_serializers_match_owned_oracle() {
    use serde::Deserialize;
    use std::sync::Arc;
    let value = sharing_input();
    let raw = serde_json::to_string(&value).unwrap();
    assert_eq!(input_outcome(&raw), (true, true));
    // Derived struct sequence form and its extra-element rejection also remain
    // identical; the adapter must not add a new decoding layer/context.
    let mut sequence = vec![
        value["observation"].clone(),
        value["events"].clone(),
        value["history_end"].clone(),
        value["typed_metadata"].clone(),
        value["catalog"].clone(),
        value["action_options"].clone(),
    ];
    input_outcome(&serde_json::to_string(&sequence).unwrap());
    sequence.push(serde_json::Value::Null);
    assert_eq!(
        input_outcome(&serde_json::to_string(&sequence).unwrap()),
        (false, false)
    );
    let parsed: borrowed::Input = serde_json::from_str(&raw).unwrap();
    assert!(Arc::ptr_eq(
        &parsed.typed_metadata,
        &parsed.observation.typed_action_schema
    ));
    let via_value = borrowed::Input::deserialize(&value).unwrap();
    assert!(Arc::ptr_eq(
        &via_value.typed_metadata,
        &via_value.observation.typed_action_schema
    ));
    let owned_value = owned::Input::deserialize(&value).unwrap();
    assert_eq!(
        serde_json::to_vec(&via_value).unwrap(),
        serde_json::to_vec(&owned_value).unwrap()
    );
    let metadata = serde_json::to_string(&value["typed_metadata"]).unwrap();
    let escaped = serde_json::to_string_pretty(&value["typed_metadata"])
        .unwrap()
        .replace("\"type\"", r#""\u0074ype""#)
        .replace("\"pray\"", r#""\u0070ray""#);
    let altered = raw.replace(
        &format!("\"typed_metadata\":{metadata}"),
        &format!("\"typed_metadata\":{escaped}"),
    );
    assert_ne!(altered, raw);
    assert_eq!(input_outcome(&altered), (true, true));
    let equal: borrowed::Input = serde_json::from_str(&altered).unwrap();
    assert!(Arc::ptr_eq(
        &equal.typed_metadata,
        &equal.observation.typed_action_schema
    ));
    assert_eq!(
        serde_json::to_vec(&equal).unwrap(),
        serde_json::to_vec(&parsed).unwrap()
    );
    let mut unequal = value.clone();
    unequal["typed_metadata"] = serde_json::json!([]);
    assert_eq!(
        input_outcome(&serde_json::to_string(&unequal).unwrap()),
        (true, true)
    );
    let unequal = borrowed::Input::deserialize(&unequal).unwrap();
    assert!(!Arc::ptr_eq(
        &unequal.typed_metadata,
        &unequal.observation.typed_action_schema
    ));
    for record in [
        serde_json::json!({"kind":"decision","input":value}),
        serde_json::json!({"kind":"end","end_kind":"natural","outcome":"draw","terminal_reward":0,
            "final_observation":value["observation"],"pending_decisions":[],"simulator_abort_reason":null}),
    ] {
        let raw = serde_json::to_string(&record).unwrap();
        let old: owned::Record = serde_json::from_str(&raw).unwrap();
        let new: borrowed::Record = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            serde_json::to_vec(&old).unwrap(),
            serde_json::to_vec(&new).unwrap()
        );
    }
}

#[test]
fn schema_sharing_preserves_strict_field_scalar_and_shape_rejections() {
    let base = serde_json::to_string(&sharing_input()).unwrap();
    for field in [
        "observation",
        "events",
        "history_end",
        "typed_metadata",
        "catalog",
        "action_options",
    ] {
        let mut missing = sharing_input();
        missing.as_object_mut().unwrap().remove(field);
        assert_eq!(
            input_outcome(&serde_json::to_string(&missing).unwrap()),
            (false, false),
            "missing {field}"
        );
        let duplicate = format!("{{\"{field}\":{},{}", sharing_input()[field], &base[1..]);
        assert_eq!(
            input_outcome(&duplicate),
            (false, false),
            "duplicate {field}"
        );
    }
    assert_eq!(
        input_outcome(&format!("{{\"extra\":true,{}", &base[1..])),
        (false, false)
    );
    assert_eq!(
        input_outcome(&format!("{{\"\\u0074yped_metadata\":[],{}", &base[1..])),
        (false, false)
    );
    for schema in [
        r#"{"type":"integer","minimum":0,"maximum":1,"extra":0}"#,
        r#"{"type":"integer","type":"integer","minimum":0,"maximum":1}"#,
        r#"{"type":"integer","minimum":0,"\u006dinimum":0,"maximum":1}"#,
        r#"{"type":"object","required_fields":{"x":{"type":"null"},"\u0078":{"type":"null"}},"optional_fields":{},"additional_fields":false}"#,
        r#"{"type":"integer","minimum":null,"maximum":1}"#,
        r#"{"type":"integer","minimum":0.0,"maximum":1}"#,
        r#"{"type":"integer","minimum":0e0,"maximum":1}"#,
        r#"{"type":"integer","minimum":true,"maximum":1}"#,
        r#"{"type":"integer","minimum":-170141183460469231731687303715884105729,"maximum":1}"#,
        r#"{"type":"integer","minimum":0,"maximum":170141183460469231731687303715884105728}"#,
        r#"{"type":"integer","minimum":0,"maximum":1,"enum":[3]}"#,
        r#"{"type":"integer"}"#,
    ] {
        let marker = format!("\"typed_metadata\":{}", sharing_input()["typed_metadata"]);
        let raw = base.replace(&marker, &format!("\"typed_metadata\":[{schema}]"));
        assert_ne!(raw, base);
        let (decode, valid) = input_outcome(&raw);
        assert!(!decode || !valid, "invalid schema accepted: {schema}");
    }
    for schema in [
        r#"{"type":"integer","minimum":-170141183460469231731687303715884105728,"maximum":170141183460469231731687303715884105727}"#,
        r#"{"type":"integer","minimum":-0,"maximum":1}"#,
        r#"{"type":"integer","minimum":0,"maximum":18446744073709551616}"#,
    ] {
        let marker = format!("\"typed_metadata\":{}", sharing_input()["typed_metadata"]);
        let raw = base.replace(&marker, &format!("\"typed_metadata\":[{schema}]"));
        assert_eq!(input_outcome(&raw), (true, true));
    }
}

#[test]
fn schema_arc_and_input_wire_preserve_actual_json_recursion_boundaries() {
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let mut saw_accepted = false;
            let mut saw_rejected = false;
            for depth in 0..132 {
                let mut schema = r#"{"type":"boolean"}"#.to_owned();
                for _ in 0..depth {
                    schema = format!(r#"{{"type":"array","items":{schema},"ordered":true}}"#);
                }
                let mut value = sharing_input();
                value["observation"]["typed_action_schema"] = serde_json::json!([]);
                value["typed_metadata"] = serde_json::json!([]);
                let raw = serde_json::to_string(&value).unwrap().replace(
                    "\"typed_metadata\":[]",
                    &format!("\"typed_metadata\":[{schema}]"),
                );
                let (accepted, _) = input_outcome(&raw);
                saw_accepted |= accepted;
                saw_rejected |= !accepted;
                let observation_raw = raw.replace(
                    "\"typed_action_schema\":[]",
                    &format!("\"typed_action_schema\":[{schema}]"),
                );
                input_outcome(&observation_raw);
                let record = format!(r#"{{"kind":"decision","input":{observation_raw}}}"#);
                let old = serde_json::from_str::<owned::Record>(&record);
                let new = serde_json::from_str::<borrowed::Record>(&record);
                assert_eq!(
                    old.is_ok(),
                    new.is_ok(),
                    "record recursion boundary at depth {depth}"
                );
                if let (Ok(old), Ok(new)) = (old, new) {
                    assert_eq!(
                        serde_json::to_vec(&old).unwrap(),
                        serde_json::to_vec(&new).unwrap()
                    );
                }
            }
            assert!(saw_accepted && saw_rejected);
        })
        .unwrap()
        .join()
        .unwrap();
}
