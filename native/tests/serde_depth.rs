use gf_native::{
    budget::{Budget, Control, monotonic},
    error::Error,
    json::{self, ValueExt},
};

#[test]
fn standard_serde_depth_boundary_and_error_drop_on_owned_eight_mib_stack() {
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            for object in [false, true] {
                // Pinned serde_json's default budget accepts 127 nested
                // containers and rejects entry into the 128th container.
                for depth in [126, 127, 128, 129, 990] {
                    let encoded = if object {
                        format!("{}0{}", "{\"x\":".repeat(depth), "}".repeat(depth))
                    } else {
                        format!("{}0{}", "[".repeat(depth), "]".repeat(depth))
                    };
                    let budget = Budget::new(64 * 1024 * 1024);
                    let control = Control::new(monotonic() + 30.).unwrap();
                    if depth < 128 {
                        let document = json::parse(encoded.as_bytes(), &budget, &control).unwrap();
                        let result = json::canonical(&document.root, &budget, &control).unwrap();
                        assert_eq!(&*result, encoded.as_bytes());
                        drop(result);
                        drop(document);
                    } else {
                        assert!(matches!(
                            json::parse(encoded.as_bytes(), &budget, &control),
                            Err(Error::Contract(message)) if message.contains("recursion limit exceeded")
                        ));
                    }
                    assert_eq!(budget.used(), 0);
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn standard_serde_accepts_huge_integer_tokens_but_bounded_dtos_reject_them() {
    let budget = Budget::new(8 * 1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    // Parser v2 deliberately has no Python implementation-specific digit cap.
    for digits in [4300, 4301, 10000] {
        let raw = "9".repeat(digits);
        let document = json::parse(raw.as_bytes(), &budget, &control).unwrap();
        assert!(document.root.integer(i128::MIN, i128::MAX).is_err());
        let encoded = json::canonical(&document.root, &budget, &control).unwrap();
        assert_eq!(&*encoded, raw.as_bytes());
    }
    assert_eq!(budget.used(), 0);
}

#[test]
fn current_native_settings_reject_removed_depth_key() {
    let control = Control::new(monotonic() + 10.).unwrap();
    let mut settings = serde_json::json!({
        "memory_budget_mib": 128, "cache_budget_mib": 16,
        "chunk_target_mib": 1, "max_source_file_mib": 64,
        "max_record_mib": 8, "max_in_flight": 1, "workers": 1,
        "backend": "rust", "scratch_budget_mib": 128,
        "wall_seconds": 7200, "cleanup_seconds": 5,
    });
    let valid = serde_json::to_vec(&settings).unwrap();
    assert!(gf_native::session::Settings::decode(&valid, &control).is_ok());
    settings["native_json_depth"] = serde_json::json!(990);
    let old = serde_json::to_vec(&settings).unwrap();
    assert!(
        matches!(gf_native::session::Settings::decode(&old, &control),
                     Err(Error::Contract(message)) if message == "invalid closed object fields")
    );
}
