use gf_native::{
    budget::{Budget, Control, monotonic},
    error::Error,
    json::{self, Value, ValueExt},
};

#[test]
fn document_admission_is_before_construction_and_shared_for_journal_records() {
    let budget = Budget::new(1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    let admission = json::admit(100, &budget).unwrap();
    let reserved = budget.used();
    assert_eq!(reserved, 32 * 100 + 65536);
    let a = json::parse_admitted(b"[1]", &admission, &control).unwrap();
    let b = json::parse_admitted(b"[2]", &admission, &control).unwrap();
    assert_eq!(budget.used(), reserved);
    drop(admission);
    drop(a);
    assert_eq!(budget.used(), reserved);
    drop(b);
    assert_eq!(budget.used(), 0);
    assert!(matches!(
        json::parse(&vec![b' '; 100000], &budget, &control),
        Err(Error::ResourceLimit(_))
    ));
    assert_eq!(budget.used(), 0);
}
#[test]
fn standard_json_contract_rejects_extensions_and_reserved_numeric_keys() {
    let budget = Budget::new(8 * 1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    for raw in [
        r#"NaN"#,
        r#"Infinity"#,
        r#"-Infinity"#,
        r#""\ud800""#,
        r#""\udc00""#,
        r#"{"$serde_json::private::Number":"1"}"#,
        r#"{"x":0,"$serde_json::private::Number":"1"}"#,
        r#"{"nested":{"\u0024serde_json::private::Number":"1"}}"#,
        r#"{"$serde_json::private::Numbe\u0072":"1"}"#,
    ] {
        assert!(
            matches!(
                json::parse(raw.as_bytes(), &budget, &control),
                Err(Error::Contract(_))
            ),
            "{raw}"
        );
        assert_eq!(budget.used(), 0);
    }
    for raw in [
        r#""$serde_json::private::Number""#,
        r#"{"v":"$serde_json::private::Number"}"#,
        r#"340282366920938463463374607431768211455"#,
        r#""\ud83d\ude00""#,
    ] {
        assert!(
            json::parse(raw.as_bytes(), &budget, &control).is_ok(),
            "{raw}"
        );
    }
    assert_eq!(budget.used(), 0);
}
#[test]
fn canonical_numbers_keys_escaping_and_python_equality_stay_exact() {
    let budget = Budget::new(8 * 1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    let document = json::parse(
        br#"{"a":0,"a":2,"z":[-0,1.0,-0.0,1e-5,1e16,0.0001],"s":"line\n\"\\"}"#,
        &budget,
        &control,
    )
    .unwrap();
    let encoded = json::canonical(&document.root, &budget, &control).unwrap();
    assert_eq!(
        &*encoded,
        br#"{"a":2,"s":"line\n\"\\","z":[0,1.0,-0.0,1e-05,1e+16,0.0001]}"#
    );
    for (raw, other, equal) in [
        ("9007199254740993", "9007199254740992.0", false),
        ("9007199254740992", "9007199254740992.0", true),
        ("true", "1.0", true),
        ("false", "0", true),
    ] {
        let a = json::parse(raw.as_bytes(), &budget, &control).unwrap();
        let b = json::parse(other.as_bytes(), &budget, &control).unwrap();
        assert_eq!(a.root.eq_python(&b.root), equal);
    }
    let maximum = f64::MAX;
    let int = format!("{maximum:.0}");
    let a = json::parse(int.as_bytes(), &budget, &control).unwrap();
    assert!(a.root.eq_python(&json::float(maximum).unwrap()));
    assert!(json::float(f64::INFINITY).is_err());
    assert!(
        json::parse(b"1.0", &budget, &control)
            .unwrap()
            .root
            .integer(0, 2)
            .is_err()
    );
}
#[test]
fn python_numeric_truthiness_preserves_overflow_underflow_and_large_integers() {
    let budget = Budget::new(1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    for (raw, expected) in [
        ("1e999", true),
        ("-1e999", true),
        ("1.7976931348623157e308", true),
        ("5e-324", true),
        ("340282366920938463463374607431768211455", true),
        ("-340282366920938463463374607431768211455", true),
        ("0", false),
        ("-0", false),
        ("0.0", false),
        ("-0.0", false),
        ("0e999", false),
        ("-0e999", false),
        ("1e-999", false),
        ("-1e-999", false),
    ] {
        let document = json::parse(raw.as_bytes(), &budget, &control).unwrap();
        assert_eq!(json::truthy(&document.root), expected, "{raw}");
    }
    let positive = json::parse(b"1e999", &budget, &control).unwrap();
    let same = json::parse(b"2e999", &budget, &control).unwrap();
    let negative = json::parse(b"-1e999", &budget, &control).unwrap();
    assert_eq!(positive.root.number().unwrap(), f64::INFINITY);
    assert_eq!(negative.root.number().unwrap(), f64::NEG_INFINITY);
    assert!(positive.root.eq_python(&same.root));
    assert!(!positive.root.eq_python(&negative.root));
    assert!(json::canonical(&positive.root, &budget, &control).is_err());
}
#[test]
fn boundaries_refuse_late_parse_and_canonical_publication() {
    let budget = Budget::new(1024 * 1024 * 1024);
    let encoded = format!("\"{}\"", "x".repeat(16 * 1024 * 1024));
    let control = Control::new(monotonic() + 0.002).unwrap();
    assert!(matches!(
        json::parse(encoded.as_bytes(), &budget, &control),
        Err(Error::Deadline)
    ));
    assert_eq!(budget.used(), 0);
    let value = Value::String("x".repeat(16 * 1024 * 1024));
    let control = Control::new(monotonic() + 0.002).unwrap();
    assert!(matches!(
        json::canonical(&value, &budget, &control),
        Err(Error::Deadline)
    ));
    assert_eq!(budget.used(), 0);
}
