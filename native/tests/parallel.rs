use gf_native::{
    budget::{Budget, Control, monotonic},
    parallel::{EpisodeReader, JournalData},
    session::Reader,
    source::Sources,
};
use std::{fs, path::PathBuf};

struct Fixture {
    directory: PathBuf,
    paths: Vec<PathBuf>,
}
impl Fixture {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "gf-native-worker-test-{}-{}",
            std::process::id(),
            monotonic().to_bits()
        ));
        fs::create_dir(&directory).unwrap();
        let mut paths = Vec::new();
        for number in 0..8 {
            let path = directory.join(format!("episode-{number}"));
            fs::create_dir(&path).unwrap();
            for name in ["player-1.jsonl", "player-2.jsonl"] {
                fs::write(path.join(name), b"{}\n").unwrap();
            }
            for name in ["replay-player-1.json", "replay-player-2.json"] {
                fs::write(path.join(name), b"{}").unwrap();
            }
            paths.push(path);
        }
        fs::write(
            paths[0].join("replay-player-1.json"),
            format!("{{\"payload\":\"{}\"}}", "x".repeat(256 * 1024)),
        )
        .unwrap();
        Self { directory, paths }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.directory).unwrap();
    }
}

#[test]
fn real_workers_preserve_order_and_share_capacity() {
    for (reader_kind, workers) in [Reader::Legacy, Reader::TypedBorrowedV1]
        .into_iter()
        .flat_map(|reader| [1, 2, 4, 8].map(|workers| (reader, workers)))
    {
        let fixture = Fixture::new();
        let budget = Budget::new(128 * 1024 * 1024);
        let control = Control::new(monotonic() + 10.).unwrap();
        let mut sources = Sources::new(budget.clone(), 2 * 1024 * 1024, 1024 * 1024);
        let mut reader = EpisodeReader::new(
            &fixture.paths,
            workers,
            workers,
            &budget,
            &control,
            2 * 1024 * 1024,
            1024 * 1024,
            reader_kind,
        );
        let mut count = 0;
        while let Some((path, data)) = reader.next(&mut sources).unwrap() {
            assert_eq!(path, fixture.paths[count]);
            match &data.records1 {
                JournalData::Legacy(records) => assert_eq!(records.len(), 1),
                JournalData::Typed(_) => assert_eq!(reader_kind, Reader::TypedBorrowedV1),
            };
            count += 1;
        }
        assert_eq!(count, 8);
        assert_eq!(sources.entries.len(), 32);
        if workers > 1 {
            assert!(budget.peak() >= workers * 8 * 1024 * 1024);
        }
        drop(reader);
        drop(sources);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn worker_failure_drains_every_result_and_lease() {
    for reader_kind in [Reader::Legacy, Reader::TypedBorrowedV1] {
        for missing in [false, true] {
            let fixture = Fixture::new();
            if missing {
                fs::remove_file(fixture.paths[1].join("player-1.jsonl")).unwrap();
            } else {
                fs::write(fixture.paths[1].join("replay-player-1.json"), b"[").unwrap();
            }
            let budget = Budget::new(128 * 1024 * 1024);
            let control = Control::new(monotonic() + 10.).unwrap();
            let mut sources = Sources::new(budget.clone(), 2 * 1024 * 1024, 1024 * 1024);
            let mut reader = EpisodeReader::new(
                &fixture.paths,
                4,
                4,
                &budget,
                &control,
                2 * 1024 * 1024,
                1024 * 1024,
                reader_kind,
            );
            assert!(reader.next(&mut sources).is_err());
            drop(reader);
            drop(sources);
            assert_eq!(budget.used(), 0);
        }
    }
}

#[test]
fn source_depth_bound_parse_canonical_and_drop_on_actual_worker_stacks() {
    std::thread::Builder::new()
        .name("depth-executor".to_owned())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            for (reader_kind,depth) in [Reader::Legacy,Reader::TypedBorrowedV1].into_iter().flat_map(|reader|[126,127,128,990].map(|depth|(reader,depth))) {
                let fixture = Fixture::new();
                let encoded = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
                fs::write(fixture.paths[0].join("replay-player-1.json"), &encoded).unwrap();
                let budget = Budget::new(128 * 1024 * 1024);
                let control = Control::new(monotonic() + 10.).unwrap();
                let mut sources = Sources::new(budget.clone(), 2 * 1024 * 1024, 1024 * 1024);
                let mut reader = EpisodeReader::new(
                    &fixture.paths[..2],
                    2,
                    2,
                    &budget,
                    &control,
                    2 * 1024 * 1024,
                    1024 * 1024,
                    reader_kind,
                );
                if depth < 128 {
                    let (_, data) = reader.next(&mut sources).unwrap().unwrap();
                    let canonical =
                        gf_native::json::canonical(&data.replay1.root, &budget, &control)
                            .unwrap();
                    assert_eq!(&*canonical, encoded.as_bytes());
                    drop(canonical);
                    drop(data);
                } else {
                    assert!(matches!(
                        reader.next(&mut sources),
                        Err(gf_native::error::Error::Contract(message)) if message.contains("recursion limit exceeded")
                    ));
                }
                drop(reader);
                drop(sources);
                assert_eq!(budget.used(), 0);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
