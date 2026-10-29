//! Health evidence uses Python truthiness, including standard JSON f64 overflow.
use gf_native::{
    budget::{Budget, Control, monotonic},
    controls::Parent,
    error::Error,
    json, projection,
    source::Sources,
};
use std::{fs, path::PathBuf};

struct ParentFixture(PathBuf);
impl ParentFixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "gf-native-parent-health-{}-{}",
            std::process::id(),
            monotonic().to_bits()
        ));
        for name in ["generations", "updates", "artifacts"] {
            fs::create_dir_all(path.join(name)).unwrap();
        }
        Self(path)
    }
}
impl Drop for ParentFixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn parent_rejects_overflow_failure_quarantine_cleanup_and_budget_evidence() {
    let fixture = ParentFixture::new();
    let budget = Budget::new(8 * 1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    for flag in [
        "failure",
        "quarantine",
        "cleanup_failure",
        "budget_overrun_seconds",
    ] {
        for value in ["1e999", "-1e999", "340282366920938463463374607431768211455"] {
            // Deliberately stop the fixture before config validation: a failure
            // must be rejected by the real parent health gate first.
            fs::write(
                fixture.0.join("run.json"),
                format!(r#"{{"schema_version":1,"algorithm":"ppo","status":"complete","{flag}":{value}}}"#),
            ).unwrap();
            let mut sources = Sources::new(budget.clone(), 1024 * 1024, 1024 * 1024);
            assert!(
                matches!(
                    Parent::open(&fixture.0, true, &mut sources, &budget, &control),
                    Err(Error::Contract(message)) if message == "PPO parent contains failure evidence"
                ),
                "{flag}={value}"
            );
        }
    }
}

#[test]
fn collection_links_reject_overflow_group_and_process_health_evidence() {
    let budget = Budget::new(8 * 1024 * 1024);
    let control = Control::new(monotonic() + 10.).unwrap();
    for process in [false, true] {
        let flags: &[&str] = if process {
            &["log_error", "reap_failure", "budget_overrun_seconds"]
        } else {
            &[
                "failure",
                "client_close_failure",
                "budget_overrun_seconds",
                "finalization_cancelled",
                "cleanup_failure",
            ]
        };
        for flag in flags {
            for (value, rejected) in [
                ("1e999", true),
                ("-1e999", true),
                ("0", false),
                ("-0.0", false),
                ("1e-999", false),
                ("-1e-999", false),
            ] {
                let evidence = if process {
                    format!(
                        r#""process":{{"exit_code":0,"shutdown":"stdin_eof","{flag}":{value}}}"#
                    )
                } else {
                    format!(r#""{flag}":{value}"#)
                };
                let raw = format!(
                    r#"{{"episodes":[{{"episode_id":"e","pair_id":"p","policies":["a","b"],"seed":1}}],"schedule":[{{"seed":1,"episodes":[{{"episode_id":"e","pair_id":"p","policies":["a","b"]}}]}}],"groups":[{{"episode_ids":["e"],{evidence}}}]}}"#
                );
                let document = json::parse(raw.as_bytes(), &budget, &control).unwrap();
                let result = projection::links(&document.root, "e");
                if rejected {
                    let expected = if process {
                        "episode or group process health failure"
                    } else {
                        "episode or group health failure"
                    };
                    assert!(
                        matches!(result, Err(Error::Contract(message)) if message == expected),
                        "process={process}, {flag}={value}"
                    );
                } else {
                    assert!(result.is_ok(), "process={process}, {flag}={value}");
                }
            }
        }
    }
}
