use gf_native::feature::{FeatureConfig, feature_names_controlled};
use sha2::{Digest, Sha256};

#[test]
fn legacy_name_order_hashes_and_widths() {
    for (history, max_elements, hash_buckets, history_window, expected, width) in [
        (
            "none",
            2,
            2,
            128,
            "d2fa5eedd92f8f0b18b7b67c2967faf10c72704f9ac1358db41020d3f1a21f1a",
            218,
        ),
        (
            "public_aggregate",
            3,
            32,
            2,
            "452f0d220d330753fd81a96c3848e76032379e79723b0f6536d987d1274078ee",
            690,
        ),
        (
            "public_aggregate",
            3,
            2,
            1,
            "769cff25b7503ef112103fb12acd1c144e6d50f568e3617a5f1d5cc44a91bb20",
            300,
        ),
    ] {
        let config = FeatureConfig {
            history: history.into(),
            max_elements,
            hash_buckets,
            history_window,
        };
        let budget = gf_native::budget::Budget::new(1024 * 1024);
        let control =
            gf_native::budget::Control::new(gf_native::budget::monotonic() + 10.).unwrap();
        let names = feature_names_controlled(&config, &budget, &control).unwrap();
        assert_eq!(budget.used(), names.reserved_bytes());
        assert_eq!(names.len(), width);
        assert_eq!(config.width(), width);
        assert_eq!(
            format!("{:x}", Sha256::digest(names.join("\n").as_bytes())),
            expected
        );
    }
}

#[test]
fn prepared_features_reuse_base_and_keep_candidate_checks() {
    use gf_native::budget::{Budget, Control, monotonic};
    use gf_native::feature::{PreparedDecision, encode_controlled};
    use gf_native::json::{canonical, parse};
    let budget = Budget::new(2 * 1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    let observation = parse(
        include_bytes!("../../tests/fixtures/observation_p1_public.json"),
        &budget,
        &control,
    )
    .unwrap();
    let metadata = canonical(
        observation.root.require("typed_action_schema").unwrap(),
        &budget,
        &control,
    )
    .unwrap();
    let mut catalog = String::from("{");
    for (index, card) in observation
        .root
        .require("own")
        .unwrap()
        .require("hand")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        if index > 0 {
            catalog.push(',');
        }
        let name = card.require("definition_id").unwrap().as_str().unwrap();
        catalog.push_str(&format!(r#""{name}":{{"definition_id":"same-inner-id","display_name":"x","kind":"armor","defense":{index}}}"#));
    }
    catalog.push('}');
    let input = format!(
        r#"{{"observation":{},"typed_metadata":{},"catalog":{},"events":[{{"type":"damage_applied","player":1,"amount":-7}}],"history_end":null,"action_options":null}}"#,
        include_str!("../../tests/fixtures/observation_p1_public.json"),
        std::str::from_utf8(&metadata).unwrap(),
        catalog
    );
    let input = parse(input.as_bytes(), &budget, &control).unwrap();
    gf_native::validation::validate_worker_input_controlled(&input.root, None, &control).unwrap();
    let config = FeatureConfig {
        history: "public_aggregate".into(),
        max_elements: 3,
        hash_buckets: 2,
        history_window: 128,
    };
    let used = budget.used();
    {
        let prepared = PreparedDecision::new(&input.root, &config, &budget, &control).unwrap();
        assert!(budget.used() > used);
        let mut actual = vec![0.; config.width()];
        let mut expected = vec![0.; config.width()];
        for raw in [
            r#"{"type":"pray"}"#,
            r#"{"type":"discard","instance_ids":[1,2]}"#,
            r#"{"type":"discard","instance_ids":[2,1]}"#,
        ] {
            let action = parse(raw.as_bytes(), &budget, &control).unwrap();
            prepared
                .encode(&action.root, &mut actual, &control)
                .unwrap();
            encode_controlled(&input.root, &action.root, &config, &mut expected, &control).unwrap();
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        let action = parse(
            br#"{"type":"discard","instance_ids":[999]}"#,
            &budget,
            &control,
        )
        .unwrap();
        assert!(
            prepared
                .encode(&action.root, &mut actual, &control)
                .is_err()
        );
        control.cancel();
        assert!(
            prepared
                .encode(&action.root, &mut actual, &control)
                .is_err()
        );
    }
    assert_eq!(budget.used(), used);
}

#[test]
fn feature_metadata_admission_precedes_allocation_and_releases_on_failure() {
    use gf_native::budget::{Budget, Control, monotonic};
    use gf_native::error::Error;
    let config = FeatureConfig {
        history: "public_aggregate".into(),
        max_elements: 3,
        hash_buckets: 4096,
        history_window: 128,
    };
    assert_eq!(config.width(), 53522);
    let small = Budget::new(65536);
    let control = Control::new(monotonic() + 10.).unwrap();
    assert!(matches!(
        feature_names_controlled(&config, &small, &control),
        Err(Error::ResourceLimit(_))
    ));
    // Failed admission never acquired even a name/vector allocation charge.
    assert_eq!(small.used(), 0);
    assert_eq!(small.peak(), 0);
    let expired = Control::new(monotonic() - 1.).unwrap();
    assert!(matches!(
        feature_names_controlled(&config, &small, &expired),
        Err(Error::Deadline)
    ));
    assert_eq!(small.peak(), 0);
    let adequate = Budget::new(32 * 1024 * 1024);
    let names = feature_names_controlled(&config, &adequate, &control).unwrap();
    assert_eq!(names.len(), config.width());
    assert!(
        names
            .iter()
            .all(|name| name.is_ascii() && name.len() <= 64 && name.capacity() == 64)
    );
    let native_capacity = names.len() * (size_of::<String>() + 64);
    assert!(adequate.used() > native_capacity); // Python conversion is precharged too.
    assert_eq!(adequate.used(), names.reserved_bytes());
    drop(names);
    assert_eq!(adequate.used(), 0);
}

#[test]
fn feature_metadata_generation_observes_concurrent_cancel() {
    use gf_native::budget::{Budget, Control, monotonic};
    use gf_native::error::Error;
    let config = FeatureConfig {
        history: "public_aggregate".into(),
        max_elements: 3,
        hash_buckets: 4096,
        history_window: 128,
    };
    let budget = Budget::new(32 * 1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    let cancellation = control.clone();
    let worker = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(1));
        cancellation.cancel();
    });
    let result = feature_names_controlled(&config, &budget, &control);
    worker.join().unwrap();
    assert!(matches!(result, Err(Error::Cancelled)));
    assert_eq!(budget.used(), 0);
}
