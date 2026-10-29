use gf_native::{
    budget::{Budget, Control, monotonic},
    error::Error,
    json,
};

#[test]
fn serde_depth_boundary_and_error_drop_on_owned_eight_mib_stack() {
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let small = Budget::new(1024 * 1024);
            let shallow = Control::new(monotonic() + 30.)
                .unwrap()
                .with_json_depth(1)
                .unwrap();
            assert!(json::parse(b"[[]]", &small, &shallow).is_ok());
            assert!(matches!(
                json::parse(b"[[0]]", &small, &shallow),
                Err(Error::ResourceLimit(_))
            ));
            for object in [false, true] {
                for depth in [989, 990, 991, 992] {
                    let encoded = if object {
                        format!("{}0{}", "{\"x\":".repeat(depth), "}".repeat(depth))
                    } else {
                        format!("{}0{}", "[".repeat(depth), "]".repeat(depth))
                    };
                    let budget = Budget::new(64 * 1024 * 1024);
                    let control = Control::new(monotonic() + 30.).unwrap();
                    if depth <= 990 {
                        let document = json::parse(encoded.as_bytes(), &budget, &control).unwrap();
                        let result = json::canonical(&document.root, &budget, &control).unwrap();
                        assert_eq!(&*result, encoded.as_bytes());
                        drop(result);
                        drop(document);
                    } else {
                        assert!(matches!(
                            json::parse(encoded.as_bytes(), &budget, &control),
                            Err(Error::ResourceLimit(_))
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
