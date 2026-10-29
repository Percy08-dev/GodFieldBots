#![cfg(feature = "decode-profile")]
use gf_native::{
    budget::{Budget, Control, monotonic},
    decode_profile::{DecodeCounters, Kind, capture_identity, selected},
    error::Error,
    json,
    source::Sources,
    telemetry::Telemetry,
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "gfb-decode-test-{}-{}",
            std::process::id(),
            monotonic().to_bits()
        ));
        fs::create_dir(&path).unwrap();
        Self(fs::canonicalize(path).unwrap())
    }
    fn sampled(&self, kind: Kind, label: &str) -> PathBuf {
        (0..10000)
            .map(|n| self.0.join(format!("{label}-{n}")))
            .find(|path| selected(kind, path))
            .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn diagnostics(telemetry: &Telemetry, budget: &Arc<Budget>, control: &Control) -> json::Document {
    json::parse(telemetry.json(0, 0, 0).as_bytes(), budget, control).unwrap()
}
fn kind<'a>(document: &'a json::Document, name: &str) -> &'a json::Value {
    document
        .root
        .require("decode_profile")
        .unwrap()
        .require("kinds")
        .unwrap()
        .require(name)
        .unwrap()
}

#[test]
fn selected_whole_files_report_actual_kind_coverage_and_exclusive_partition() {
    let fixture = Fixture::new();
    let budget = Budget::new(128 * 1024 * 1024);
    let control = Control::new(monotonic() + 20.).unwrap();
    let telemetry = Arc::new(Telemetry::default());
    let mut sources =
        Sources::with_telemetry(budget.clone(), 2 * 1024 * 1024, 1024 * 1024, &telemetry);
    let journal = fixture.sampled(Kind::Journal, "journal");
    fs::write(&journal, b"{}\n{\"n\":2}\r\n").unwrap();
    let records = sources.journal(&journal, &control).unwrap();
    assert_eq!(records.len(), 2);
    drop(records);
    let replay = fixture.sampled(Kind::Replay, "replay");
    let mut utf16 = vec![0xff, 0xfe];
    for code in "{\"n\":3}".encode_utf16() {
        utf16.extend_from_slice(&code.to_le_bytes());
    }
    fs::write(&replay, &utf16).unwrap();
    drop(sources.json(&replay, true, &control).unwrap());
    let other = fixture.sampled(Kind::Json, "json");
    fs::write(&other, b"{\"a\":[1,true,null],\"s\":\"text\"}").unwrap();
    drop(sources.json(&other, false, &control).unwrap());
    let report = diagnostics(&telemetry, &budget, &control);
    for name in ["journal", "replay", "json"] {
        let entry = kind(&report, name);
        assert_eq!(entry.require("files").unwrap().integer(1, 1).unwrap(), 1);
        assert!(
            entry
                .require("input_bytes")
                .unwrap()
                .integer(1, i128::MAX)
                .unwrap()
                > 0
        );
        let sampled = entry
            .require("sampled_files")
            .unwrap()
            .integer(0, 1)
            .unwrap();
        assert_eq!(
            entry
                .require("cohort_files")
                .unwrap()
                .integer(1, 1)
                .unwrap(),
            1
        );
        assert_eq!(
            entry
                .require("cohort_input_bytes")
                .unwrap()
                .integer(1, i128::MAX)
                .unwrap(),
            entry
                .require("input_bytes")
                .unwrap()
                .integer(1, i128::MAX)
                .unwrap()
        );
        if capture_identity() == "on" {
            assert_eq!(sampled, 1);
            let sum = entry
                .require("sampled_exclusive_seconds")
                .unwrap()
                .as_object()
                .unwrap()
                .iter()
                .map(|(_, v)| v.number().unwrap())
                .sum::<f64>();
            let wall = entry
                .require("sampled_decode_wall_seconds")
                .unwrap()
                .number()
                .unwrap();
            assert!((sum - wall).abs() < 1e-9);
            assert!(
                entry
                    .require("sampled_counts")
                    .unwrap()
                    .require("nodes")
                    .unwrap()
                    .integer(1, i128::MAX)
                    .unwrap()
                    > 0
            );
        } else {
            assert_eq!(sampled, 0);
        }
    }
    let expected = fs::metadata(&journal).unwrap().len() as i128;
    assert_eq!(
        kind(&report, "journal")
            .require("sampled_input_bytes")
            .unwrap()
            .integer(0, i128::MAX)
            .unwrap(),
        if capture_identity() == "on" {
            expected
        } else {
            0
        }
    );
}

#[test]
fn feature_on_sampling_preserves_depth_resource_error_and_following_decode() {
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let fixture = Fixture::new();
            let budget = Budget::new(128 * 1024 * 1024);
            let telemetry = Arc::new(Telemetry::default());
            let control = Control::new(monotonic() + 20.).unwrap();
            let mut sources =
                Sources::with_telemetry(budget.clone(), 2 * 1024 * 1024, 1024 * 1024, &telemetry);
            let path = fixture.sampled(Kind::Json, "depth");
            let encoded = format!("{}0{}", "[".repeat(990), "]".repeat(990));
            fs::write(&path, &encoded).unwrap();
            let parsed = sources.json(&path, false, &control).unwrap();
            assert_eq!(
                &*json::canonical(&parsed.root, &budget, &control).unwrap(),
                encoded.as_bytes()
            );
            drop(parsed);
            let limited = control.with_json_depth(989).unwrap();
            assert!(matches!(
                sources.json(&path, false, &limited),
                Err(Error::ResourceLimit(_))
            ));
            let bad = fixture.sampled(Kind::Json, "bad");
            fs::write(&bad, b"[").unwrap();
            assert!(matches!(
                sources.json(&bad, false, &control),
                Err(Error::Contract(_))
            ));
            let good = fixture.sampled(Kind::Json, "after");
            fs::write(&good, b"[2]").unwrap();
            drop(sources.json(&good, false, &control).unwrap());
            let report = diagnostics(&telemetry, &budget, &control);
            let entry = kind(&report, "json");
            assert_eq!(entry.require("files").unwrap().integer(4, 4).unwrap(), 4);
            assert_eq!(
                entry
                    .require("failed_files")
                    .unwrap()
                    .integer(2, 2)
                    .unwrap(),
                2
            );
            drop(report);
            drop(sources);
            assert_eq!(budget.used(), 0);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn sampling_hash_is_sensitive_to_kind_and_path_not_arrival_order() {
    let paths = (0..256)
        .map(|n| PathBuf::from(format!("/canonical/{n}")))
        .collect::<Vec<_>>();
    let direct = paths
        .iter()
        .map(|p| selected(Kind::Journal, p))
        .collect::<Vec<_>>();
    let mut reverse = paths
        .iter()
        .rev()
        .map(|p| selected(Kind::Journal, p))
        .collect::<Vec<_>>();
    reverse.reverse();
    assert_eq!(direct, reverse);
    assert!(
        paths
            .iter()
            .any(|p| selected(Kind::Journal, p) != selected(Kind::Replay, p))
    );
    assert!(
        selected(Kind::Json, Path::new("/canonical/known"))
            == selected(Kind::Json, Path::new("/canonical/known"))
    );
}

#[test]
fn selected_parse_cancellation_keeps_failure_and_next_file_accounting() {
    let fixture = Fixture::new();
    let budget = Budget::new(128 * 1024 * 1024);
    let counters = DecodeCounters::default();
    let path = fixture.sampled(Kind::Json, "cancel");
    fs::write(&path, b"{}").unwrap();
    let control = Control::new(monotonic() + 20.).unwrap();
    let mut encoded = vec![b'x'; 16 * 1024 * 1024];
    encoded.insert(0, b'"');
    encoded.push(b'"');
    let cancellation = control.clone();
    let worker = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(1));
        cancellation.cancel();
    });
    {
        let _file = counters.begin(Kind::Json, Some(&path), encoded.len());
        assert!(matches!(
            json::parse(&encoded, &budget, &control),
            Err(Error::Cancelled)
        ));
    }
    worker.join().unwrap();
    let fresh = Control::new(monotonic() + 20.).unwrap();
    {
        let mut file = counters.begin(Kind::Json, Some(&path), 2);
        drop(json::parse(b"[]", &budget, &fresh).unwrap());
        file.success();
    }
    let report = json::parse(counters.json().as_bytes(), &budget, &fresh).unwrap();
    let entry = report
        .root
        .require("kinds")
        .unwrap()
        .require("json")
        .unwrap();
    assert_eq!(entry.require("files").unwrap().integer(2, 2).unwrap(), 2);
    assert_eq!(
        entry
            .require("failed_files")
            .unwrap()
            .integer(1, 1)
            .unwrap(),
        1
    );
    drop(report);
    assert_eq!(budget.used(), 0);
}
