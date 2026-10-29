use gf_native::{
    budget::{Budget, Control, monotonic},
    error::Error,
    json::{self, Text, Value},
};

#[test]
fn escaped_strings_whitespace_and_numbers_poll_without_stride_gaps() {
    let mut escaped = Vec::with_capacity(16 * 1024 * 1024 + 2);
    escaped.push(b'"');
    for _ in 0..(16 * 1024 * 1024 / 6) {
        escaped.extend_from_slice(br"\u0061");
    }
    escaped.push(b'"');
    let mut spaces = vec![b' '; 16 * 1024 * 1024];
    spaces.push(b'0');
    let mut number = vec![b'1'; 16 * 1024 * 1024];
    number.extend_from_slice(b".0");
    for encoded in [&escaped, &spaces, &number] {
        let budget = Budget::new(128 * 1024 * 1024);
        let control = Control::new(monotonic() + 0.002).unwrap();
        assert!(matches!(
            json::parse(encoded, &budget, &control),
            Err(Error::Deadline)
        ));
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn canonical_long_string_checks_deadline_and_releases_failed_output() {
    let budget = Budget::new(128 * 1024 * 1024);
    let value = Value::String(Text::from_str(&"x".repeat(16 * 1024 * 1024), &budget).unwrap());
    let retained = budget.used();
    let control = Control::new(monotonic() + 0.002).unwrap();
    assert!(matches!(
        json::canonical(&value, &budget, &control),
        Err(Error::Deadline)
    ));
    assert_eq!(budget.used(), retained);
    drop(value);
    assert_eq!(budget.used(), 0);
}

#[test]
fn parsed_nan_has_python_container_identity_semantics_but_cannot_be_hashed() {
    let budget = Budget::new(1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    let first = json::parse(br#"{"unused":NaN}"#, &budget, &control).unwrap();
    let second = json::parse(br#"{"unused":NaN}"#, &budget, &control).unwrap();
    assert!(first.root.eq_python(&second.root));
    let scalar_first = json::parse(b"NaN", &budget, &control).unwrap();
    let scalar_second = json::parse(b"NaN", &budget, &control).unwrap();
    assert!(!scalar_first.root.eq_python(&scalar_second.root));
    assert!(json::canonical(&first.root, &budget, &control).is_err());
    drop(first);
    drop(second);
    drop(scalar_first);
    drop(scalar_second);
    assert_eq!(budget.used(), 0);
}
