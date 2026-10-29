use gf_native::json::ValueExt;
use gf_native::{
    budget::{Budget, Control, monotonic},
    json,
    telemetry::Telemetry,
};

#[test]
fn boundary_counters_report_observed_calls_sharing_and_copy_failures() {
    let telemetry = Telemetry::default();
    telemetry.control(17, 5);
    telemetry.output_copy(23);
    telemetry.data_call();
    telemetry.delivered(true);
    telemetry.data_call();
    telemetry.delivered(false);
    telemetry.export(1024, 1024, 64).unwrap();
    assert!(telemetry.export(1024, 2048, 8).is_err());
    telemetry.stage_read(100);
    telemetry.stage_write(200);
    telemetry.scratch_peak(300);
    let encoded = telemetry.json(2, 40, 80);
    let budget = Budget::new(1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    let parsed = json::parse(encoded.as_bytes(), &budget, &control).unwrap();
    assert_eq!(
        parsed
            .root
            .require("ffi_data_calls")
            .unwrap()
            .integer(2, 2)
            .unwrap(),
        2
    );
    assert_eq!(
        parsed
            .root
            .require("ffi_batches_returned")
            .unwrap()
            .integer(1, 1)
            .unwrap(),
        1
    );
    assert_eq!(
        parsed
            .root
            .require("ffi_eof_returns")
            .unwrap()
            .integer(1, 1)
            .unwrap(),
        1
    );
    let copies = parsed.root.require("payload_copy_stats").unwrap();
    assert_eq!(
        copies
            .require("verified_shared_numeric_bytes")
            .unwrap()
            .integer(64, 64)
            .unwrap(),
        64
    );
    assert_eq!(
        copies
            .require("numeric_copy_events")
            .unwrap()
            .integer(1, 1)
            .unwrap(),
        1
    );
    assert_eq!(
        copies
            .require("numeric_copy_bytes")
            .unwrap()
            .integer(8, 8)
            .unwrap(),
        8
    );
}
