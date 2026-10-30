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
