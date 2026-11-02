//! Bounded little-endian numeric spooling, with initialized admitted destinations.
use crate::{
    budget::{BudgetVec, Control},
    error::{Error, Result},
};
use std::io::{Read, Write};

const BLOCK_BYTES: usize = 32 * 1024;
const BLOCK_VALUES: usize = BLOCK_BYTES / 8;

pub(crate) fn read_f32(
    input: &mut impl Read,
    values: &mut BudgetVec<f32>,
    count: usize,
    control: &Control,
) -> Result<()> {
    control.check()?;
    let start = values.len();
    let end = start
        .checked_add(count)
        .ok_or_else(|| Error::resource("FP32 array length overflow"))?;
    values.reserve(count)?;
    let result = (|| {
        let mut next = start;
        while next < end {
            control.check()?;
            let stop = next + (end - next).min(BLOCK_BYTES / 4);
            values.resize(stop, 0.0)?;
            let chunk = &mut values[next..stop];
            if cfg!(target_endian = "little") {
                input.read_exact(bytemuck::cast_slice_mut(chunk))?;
            } else {
                let mut bytes = [0u8; BLOCK_BYTES];
                let bytes = &mut bytes[..chunk.len() * 4];
                input.read_exact(bytes)?;
                for (value, raw) in chunk.iter_mut().zip(bytes.chunks_exact(4)) {
                    *value = f32::from_le_bytes(raw.try_into().unwrap());
                }
            }
            control.check()?;
            next = stop;
        }
        control.check()
    })();
    if result.is_err() {
        values.truncate(start);
    }
    result
}

pub(crate) fn read_f64(
    input: &mut impl Read,
    values: &mut BudgetVec<f64>,
    count: usize,
    control: &Control,
) -> Result<()> {
    read_values(input, values, count, control, f64::from_le_bytes, false)
}
pub(crate) fn read_i64(
    input: &mut impl Read,
    values: &mut BudgetVec<i64>,
    count: usize,
    control: &Control,
) -> Result<()> {
    read_values(input, values, count, control, i64::from_le_bytes, false)
}
pub(crate) fn write_f64(output: &mut impl Write, values: &[f64], control: &Control) -> Result<()> {
    write_values(output, values, control, f64::to_le_bytes, false)
}
pub(crate) fn write_i64(output: &mut impl Write, values: &[i64], control: &Control) -> Result<()> {
    write_values(output, values, control, i64::to_le_bytes, false)
}

fn read_values<T: bytemuck::Pod + Default>(
    input: &mut impl Read,
    values: &mut BudgetVec<T>,
    count: usize,
    control: &Control,
    decode: fn([u8; 8]) -> T,
    force_portable: bool,
) -> Result<()> {
    control.check()?;
    let start = values.len();
    let end = start
        .checked_add(count)
        .ok_or_else(|| Error::resource("numeric array length overflow"))?;
    values.reserve(count)?;
    let result = (|| {
        let mut next = start;
        while next < end {
            control.check()?;
            let stop = next + (end - next).min(BLOCK_VALUES);
            // Initialization is bounded too; the full capacity is admitted above.
            values.resize(stop, T::default())?;
            let chunk = &mut values[next..stop];
            if cfg!(target_endian = "little") && !force_portable {
                input.read_exact(bytemuck::cast_slice_mut(chunk))?;
            } else {
                let mut bytes = [0u8; BLOCK_BYTES];
                let bytes = &mut bytes[..chunk.len() * 8];
                input.read_exact(bytes)?;
                for (value, bytes) in chunk.iter_mut().zip(bytes.chunks_exact(8)) {
                    *value = decode(bytes.try_into().expect("numeric element width"));
                }
            }
            control.check()?;
            next = stop;
        }
        control.check()
    })();
    if result.is_err() {
        // Retained capacity stays admitted, but no partial append is exposed.
        values.truncate(start);
    }
    result
}
fn write_values<T: bytemuck::Pod>(
    output: &mut impl Write,
    values: &[T],
    control: &Control,
    encode: fn(T) -> [u8; 8],
    force_portable: bool,
) -> Result<()> {
    control.check()?;
    for chunk in values.chunks(BLOCK_VALUES) {
        control.check()?;
        if cfg!(target_endian = "little") && !force_portable {
            output.write_all(bytemuck::cast_slice(chunk))?;
        } else {
            let mut bytes = [0u8; BLOCK_BYTES];
            for (value, encoded) in chunk.iter().zip(bytes.chunks_exact_mut(8)) {
                encoded.copy_from_slice(&encode(*value));
            }
            output.write_all(&bytes[..chunk.len() * 8])?;
        }
        control.check()?;
    }
    control.check()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::{Budget, monotonic};
    use std::io::{self, Cursor};

    fn control() -> Control {
        Control::new(monotonic() + 30.).unwrap()
    }

    #[test]
    fn bits_empty_chunk_edges_and_portable_endian_fallback() {
        let patterns = [
            0,
            1 << 63,
            0x7ff8000000000042,
            0xfff8000000000137,
            0x7ff0000000000000,
            0xfff0000000000000,
            0x3ff0000000000000,
            1,
        ];
        for count in [
            0,
            1,
            BLOCK_VALUES - 1,
            BLOCK_VALUES,
            BLOCK_VALUES + 1,
            2 * BLOCK_VALUES + 1,
        ] {
            let floats: Vec<_> = (0..count)
                .map(|i| f64::from_bits(patterns[i % patterns.len()]))
                .collect();
            let integers: Vec<_> = (0..count)
                .map(|i| [i64::MIN, -1, 0, 1, i64::MAX][i % 5])
                .collect();
            for portable in [false, true] {
                let budget = Budget::new(2 * 1024 * 1024);
                let control = control();
                let mut bytes = Vec::new();
                write_values(&mut bytes, &floats, &control, f64::to_le_bytes, portable).unwrap();
                let expected: Vec<_> = floats.iter().flat_map(|v| v.to_le_bytes()).collect();
                assert_eq!(bytes, expected);
                let mut output = BudgetVec::new(&budget);
                output.push(42.).unwrap();
                read_values(
                    &mut bytes.as_slice(),
                    &mut output,
                    count,
                    &control,
                    f64::from_le_bytes,
                    portable,
                )
                .unwrap();
                assert_eq!(output[0], 42.);
                assert!(
                    output[1..]
                        .iter()
                        .zip(&floats)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                let mut bytes = Vec::new();
                write_values(&mut bytes, &integers, &control, i64::to_le_bytes, portable).unwrap();
                assert_eq!(
                    bytes,
                    integers
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>()
                );
                let mut output = BudgetVec::new(&budget);
                read_values(
                    &mut bytes.as_slice(),
                    &mut output,
                    count,
                    &control,
                    i64::from_le_bytes,
                    portable,
                )
                .unwrap();
                assert_eq!(&*output, &integers);
            }
        }
    }

    struct ShortReader {
        input: Cursor<Vec<u8>>,
        max: usize,
        fail_at: Option<usize>,
        calls: usize,
        cancel: Option<Control>,
    }
    impl Read for ShortReader {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            self.calls += 1;
            let remaining = self
                .fail_at
                .unwrap_or(usize::MAX)
                .saturating_sub(self.input.position() as usize);
            if remaining == 0 {
                return Err(io::Error::other("injected read failure"));
            }
            let count = bytes.len().min(self.max).min(remaining);
            let read = self.input.read(&mut bytes[..count])?;
            if let Some(control) = self.cancel.take() {
                control.cancel();
            }
            Ok(read)
        }
    }
    fn reader(bytes: Vec<u8>, max: usize, fail_at: Option<usize>) -> ShortReader {
        ShortReader {
            input: Cursor::new(bytes),
            max,
            fail_at,
            calls: 0,
            cancel: None,
        }
    }

    #[test]
    fn short_reads_and_partial_errors_do_not_expose_partial_appends() {
        let budget = Budget::new(2 * 1024 * 1024);
        let control = control();
        let data: Vec<i64> = (0..BLOCK_VALUES as i64 + 3).collect();
        let encoded: Vec<_> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut short = reader(encoded.clone(), 7, None);
        let mut values = BudgetVec::new(&budget);
        read_i64(&mut short, &mut values, data.len(), &control).unwrap();
        assert_eq!(&*values, &data);
        for bytes in [encoded[..encoded.len() - 1].to_vec(), encoded.clone()] {
            let fail = (bytes.len() == encoded.len()).then_some(BLOCK_BYTES + 3);
            let mut input = reader(bytes, 11, fail);
            let mut output = BudgetVec::new(&budget);
            output.push(99).unwrap();
            assert!(matches!(
                read_i64(&mut input, &mut output, data.len(), &control),
                Err(Error::Io(_))
            ));
            assert_eq!(&*output, &[99]);
        }
    }

    #[test]
    fn quota_overflow_and_cancellation_precede_publication() {
        let budget = Budget::new(31);
        let control = control();
        let mut values = BudgetVec::new(&budget);
        let mut input = Cursor::new([0u8; 8]);
        assert!(matches!(
            read_f64(&mut input, &mut values, 1, &control),
            Err(Error::ResourceLimit(_))
        ));
        assert_eq!(input.position(), 0);
        assert_eq!(budget.used(), 0);
        let budget = Budget::new(2 * 1024 * 1024);
        let mut values = BudgetVec::new(&budget);
        values.push(7.).unwrap();
        assert!(matches!(
            read_f64(&mut input, &mut values, usize::MAX, &control),
            Err(Error::ResourceLimit(_))
        ));
        assert_eq!(&*values, &[7.]);
        let mut input = reader(vec![0; 2 * BLOCK_BYTES], usize::MAX, None);
        input.cancel = Some(control.clone());
        assert_eq!(
            read_f64(&mut input, &mut values, 2 * BLOCK_VALUES, &control),
            Err(Error::Cancelled)
        );
        assert_eq!(&*values, &[7.]);
        assert_eq!(input.calls, 1);
        let mut bytes = Vec::new();
        assert_eq!(
            write_f64(&mut bytes, &[1.], &control),
            Err(Error::Cancelled)
        );
        assert!(bytes.is_empty());
        drop(values);
        assert_eq!(budget.used(), 0);
    }

    struct Writer {
        bytes: Vec<u8>,
        calls: usize,
        max: usize,
        fail_at: Option<usize>,
        cancel: Option<Control>,
    }
    impl Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            let remaining = self
                .fail_at
                .unwrap_or(usize::MAX)
                .saturating_sub(self.bytes.len());
            if remaining == 0 {
                return Err(io::Error::other("injected write failure"));
            }
            let length = bytes.len().min(self.max).min(remaining);
            self.bytes.extend_from_slice(&bytes[..length]);
            if let Some(control) = self.cancel.take() {
                control.cancel();
            }
            Ok(length)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    fn writer(max: usize, fail_at: Option<usize>) -> Writer {
        Writer {
            bytes: Vec::new(),
            calls: 0,
            max,
            fail_at,
            cancel: None,
        }
    }

    #[test]
    fn short_writes_failures_and_chunk_call_counts() {
        let control = control();
        let data = vec![i64::MIN; 2 * BLOCK_VALUES + 1];
        let mut output = writer(usize::MAX, None);
        write_i64(&mut output, &data, &control).unwrap();
        assert_eq!(output.calls, 3);
        let budget = Budget::new(2 * 1024 * 1024);
        let mut input = reader(output.bytes.clone(), usize::MAX, None);
        let mut values = BudgetVec::new(&budget);
        read_i64(&mut input, &mut values, data.len(), &control).unwrap();
        assert_eq!(input.calls, 3); // One Rust Read call per <=32KiB block.
        let mut short = writer(7, None);
        write_i64(&mut short, &data, &control).unwrap();
        assert_eq!(short.bytes, output.bytes);
        let mut failure = writer(13, Some(BLOCK_BYTES + 3));
        assert!(matches!(
            write_i64(&mut failure, &data, &control),
            Err(Error::Io(_))
        ));
        let mut cancelled = writer(usize::MAX, None);
        cancelled.cancel = Some(control.clone());
        assert_eq!(
            write_i64(&mut cancelled, &data, &control),
            Err(Error::Cancelled)
        );
        assert_eq!(cancelled.calls, 1);
    }

    #[test]
    #[ignore = "bounded release microbenchmark; run explicitly without concurrent work"]
    fn benchmark_buffered_scalar_vs_bulk_numeric_io() {
        use std::{
            hint::black_box,
            io::{BufReader, BufWriter},
            time::Instant,
        };
        let data: Vec<f64> = (0..262144).map(|i| i as f64 - 12345.5).collect();
        let control = control();
        let mut samples = Vec::new();
        let mut expected = Vec::new();
        for bulk in [false, true] {
            let start = Instant::now();
            for _ in 0..5 {
                let mut output = BufWriter::with_capacity(BLOCK_BYTES, Vec::new());
                if bulk {
                    write_f64(&mut output, &data, &control).unwrap();
                } else {
                    for (i, v) in data.iter().enumerate() {
                        if i % BLOCK_VALUES == 0 {
                            control.check().unwrap();
                        }
                        output.write_all(&v.to_le_bytes()).unwrap();
                    }
                }
                let encoded = output.into_inner().unwrap();
                if expected.is_empty() {
                    expected = encoded.clone();
                } else {
                    assert_eq!(encoded, expected);
                }
                let mut input = BufReader::with_capacity(BLOCK_BYTES, encoded.as_slice());
                let budget = Budget::new(16 * 1024 * 1024);
                let mut values = BudgetVec::with_capacity(&budget, data.len()).unwrap();
                if bulk {
                    read_f64(&mut input, &mut values, data.len(), &control).unwrap();
                } else {
                    for i in 0..data.len() {
                        if i % BLOCK_VALUES == 0 {
                            control.check().unwrap();
                        }
                        let mut bytes = [0u8; 8];
                        input.read_exact(&mut bytes).unwrap();
                        values.push(f64::from_le_bytes(bytes)).unwrap();
                    }
                }
                assert_eq!(&*values, &data);
                black_box(values);
            }
            samples.push(start.elapsed().as_secs_f64());
        }
        println!(
            "numeric_io values=262144 repetitions=5 buffered_scalar_seconds={} bulk_seconds={}",
            samples[0], samples[1]
        );
    }
}
