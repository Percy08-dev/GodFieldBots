//! Shared pure Python-compatible JSON number formatter; no runtime or budget ownership.
use std::io::{self, Write};

fn python_float(value: f64) -> io::Result<String> {
    if !value.is_finite() {
        return Err(io::Error::other("nonfinite canonical float"));
    }
    let mut buffer = ryu::Buffer::new();
    let printed = buffer.format_finite(value);
    let (negative, printed) = if let Some(s) = printed.strip_prefix('-') {
        (true, s)
    } else {
        (false, printed)
    };
    let (mantissa, exponent) = printed
        .split_once('e')
        .map(|(m, e)| (m, e.parse::<i32>().expect("ryu exponent")))
        .unwrap_or((printed, 0));
    let mut digits = mantissa.replace('.', "");
    let mut decimal = mantissa.find('.').unwrap_or(mantissa.len()) as i32 + exponent;
    while digits.len() > 1 && digits.starts_with('0') {
        digits.remove(0);
        decimal -= 1;
    }
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    let scientific = decimal - 1;
    let mut output = if negative {
        "-".to_owned()
    } else {
        String::new()
    };
    if value != 0. && !(-4..16).contains(&scientific) {
        output.push(digits.as_bytes()[0] as char);
        if digits.len() > 1 {
            output.push('.');
            output.push_str(&digits[1..]);
        }
        output.push_str(&format!(
            "e{}{:02}",
            if scientific < 0 { '-' } else { '+' },
            scientific.unsigned_abs()
        ));
    } else if decimal <= 0 {
        output.push_str("0.");
        output.extend(std::iter::repeat_n('0', (-decimal) as usize));
        output.push_str(&digits);
    } else {
        let left = decimal as usize;
        if left >= digits.len() {
            output.push_str(&digits);
            output.extend(std::iter::repeat_n('0', left - digits.len()));
            output.push_str(".0");
        } else {
            output.push_str(&digits[..left]);
            output.push('.');
            output.push_str(&digits[left..]);
        }
    }
    Ok(output)
}
pub(crate) struct PythonFormatter;
impl serde_json::ser::Formatter for PythonFormatter {
    fn write_number_str<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        value: &str,
    ) -> io::Result<()> {
        if value.bytes().any(|c| matches!(c, b'.' | b'e' | b'E')) {
            writer.write_all(python_float(value.parse().map_err(io::Error::other)?)?.as_bytes())
        } else {
            writer.write_all(if value == "-0" {
                b"0"
            } else {
                value.as_bytes()
            })
        }
    }
    fn write_f64<W: Write + ?Sized>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        writer.write_all(python_float(value)?.as_bytes())
    }
}
