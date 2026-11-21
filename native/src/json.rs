//! Standard serde JSON with checked domain access and narrow Python hash formatting.
use crate::{
    budget::{Budget, BudgetVec, Control, JsonAdmission},
    error::{Error, Result},
};
use serde::{Deserialize, Serialize};
use crate::python_formatter::PythonFormatter;
pub use serde_json::{Map, Number, Value};
use std::{
    io::{self, Write},
    sync::Arc,
};
pub const PARSER_CONTRACT: &str = "serde-json-v2";
pub const MEMORY_POLICY: &str = "json-document-32x-plus64kib-v1";

pub trait ValueExt {
    fn require(&self, key: &str) -> Result<&Value>;
    fn string(&self) -> Result<&str>;
    fn array(&self) -> Result<&[Value]>;
    fn object(&self) -> Result<&Map<String, Value>>;
    fn boolean(&self) -> Result<bool>;
    fn integer(&self, low: i128, high: i128) -> Result<i128>;
    fn number(&self) -> Result<f64>;
    fn closed(&self, required: &[&str], optional: &[&str]) -> Result<()>;
    fn eq_python(&self, other: &Self) -> bool;
}
pub fn integer_token(number: &Number) -> bool {
    !number
        .as_str()
        .bytes()
        .any(|c| matches!(c, b'.' | b'e' | b'E'))
}
fn float_value(number: &Number) -> Option<f64> {
    number.as_str().parse::<f64>().ok()
}
fn int_text(number: &Number) -> &str {
    let text = number.as_str();
    if text == "-0" { "0" } else { text }
}
fn int_float_equal(integer: &Number, float: f64) -> bool {
    float.is_finite()
        && float.fract() == 0.0
        && int_text(integer)
            == if float == 0.0 {
                "0".to_owned()
            } else {
                format!("{float:.0}")
            }
}
impl ValueExt for Value {
    fn require(&self, key: &str) -> Result<&Value> {
        self.get(key)
            .ok_or_else(|| Error::contract(format!("missing field {key}")))
    }
    fn string(&self) -> Result<&str> {
        self.as_str()
            .ok_or_else(|| Error::contract("string required"))
    }
    fn array(&self) -> Result<&[Value]> {
        self.as_array()
            .map(Vec::as_slice)
            .ok_or_else(|| Error::contract("array required"))
    }
    fn object(&self) -> Result<&Map<String, Value>> {
        self.as_object()
            .ok_or_else(|| Error::contract("object required"))
    }
    fn boolean(&self) -> Result<bool> {
        self.as_bool()
            .ok_or_else(|| Error::contract("boolean required"))
    }
    fn integer(&self, low: i128, high: i128) -> Result<i128> {
        let Self::Number(number) = self else {
            return Err(Error::contract("integer required"));
        };
        if !integer_token(number) {
            return Err(Error::contract("integer required"));
        }
        let value = int_text(number)
            .parse::<i128>()
            .map_err(|_| Error::contract("integer outside range"))?;
        if value < low || value > high {
            return Err(Error::contract("integer outside range"));
        }
        Ok(value)
    }
    fn number(&self) -> Result<f64> {
        self.as_number()
            .and_then(float_value)
            .ok_or_else(|| Error::contract("number required/outside range"))
    }
    fn closed(&self, required: &[&str], optional: &[&str]) -> Result<()> {
        let values = self.object()?;
        if required.iter().any(|key| !values.contains_key(*key))
            || values.keys().any(|key| {
                !required
                    .iter()
                    .chain(optional)
                    .any(|expected| key == expected)
            })
        {
            return Err(Error::contract("invalid closed object fields"));
        }
        Ok(())
    }
    fn eq_python(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Number(a), Self::Number(b)) => match (integer_token(a), integer_token(b)) {
                (true, true) => int_text(a) == int_text(b),
                (true, false) => float_value(b).is_some_and(|f| int_float_equal(a, f)),
                (false, true) => float_value(a).is_some_and(|f| int_float_equal(b, f)),
                (false, false) => float_value(a)
                    .zip(float_value(b))
                    .is_some_and(|(a, b)| a == b),
            },
            (Self::Bool(a), Self::Number(b)) | (Self::Number(b), Self::Bool(a)) => {
                if integer_token(b) {
                    int_text(b) == if *a { "1" } else { "0" }
                } else {
                    float_value(b) == Some(*a as u8 as f64)
                }
            }
            (Self::Array(a), Self::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.eq_python(b))
            }
            (Self::Object(a), Self::Object(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .all(|(key, value)| b.get(key).is_some_and(|other| value.eq_python(other)))
            }
            _ => self == other,
        }
    }
}
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(v) => *v,
        Value::Number(v) => {
            if integer_token(v) {
                int_text(v) != "0"
            } else {
                // Standard JSON exponents can overflow f64. Python's parsed
                // infinity is truthy; conversion failure must never clear health.
                float_value(v).is_none_or(|f| f != 0.)
            }
        }
        Value::String(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
    }
}

pub struct Document {
    pub root: Value,
    _admission: Arc<JsonAdmission>,
}
pub fn admit(bytes: usize, budget: &Arc<Budget>) -> Result<Arc<JsonAdmission>> {
    let raw = bytes;
    let bytes = bytes
        .checked_mul(32)
        .and_then(|n| n.checked_add(65536))
        .ok_or_else(|| Error::resource("JSON document estimate overflow"))?;
    let admission=budget.reserve_json(bytes).map_err(|error|match error{
        Error::ResourceLimit(_)=>Error::resource(format!("JSON document estimate exceeds memory budget: raw_bytes={raw}, estimated_bytes={bytes}, used_bytes={}, limit_bytes={}",budget.used(),budget.limit())),
        other=>other,
    })?;
    Ok(Arc::new(admission))
}
const RESERVED: &[u8] = b"$serde_json::private::Number";
/// Reject the member name serde uses internally for arbitrary-precision numbers.
pub(crate) fn reject_reserved_number_members(bytes: &[u8], control: &Control) -> Result<()> {
    let mut index = 0usize;
    let mut next_poll = 0usize;
    while index < bytes.len() {
        if index >= next_poll {
            control.check()?;
            next_poll = index.saturating_add(4096);
        }
        match bytes[index] {
            b'"' => {
                let start = index;
                index += 1;
                let mut escaped = false;
                while index < bytes.len() {
                    if index >= next_poll {
                        control.check()?;
                        next_poll = index.saturating_add(4096);
                    }
                    match bytes[index] {
                        b'\\' => {
                            escaped = true;
                            index = index.saturating_add(2);
                        }
                        b'"' => break,
                        _ => index += 1,
                    }
                }
                if index >= bytes.len() {
                    break;
                }
                let end = index;
                index += 1;
                let mut after = index;
                while matches!(bytes.get(after), Some(b' ' | b'\r' | b'\n' | b'\t')) {
                    after += 1;
                    if after >= next_poll {
                        control.check()?;
                        next_poll = after.saturating_add(4096);
                    }
                }
                if bytes.get(after) == Some(&b':') {
                    let reserved = if escaped {
                        serde_json::from_slice::<String>(&bytes[start..=end])
                            .map_err(|e| Error::contract(e.to_string()))?
                            .as_bytes()
                            == RESERVED
                    } else {
                        &bytes[start + 1..end] == RESERVED
                    };
                    if reserved {
                        return Err(Error::contract("reserved serde numeric member key"));
                    }
                }
            }
            _ => index += 1,
        }
    }
    control.check()
}
pub fn parse(bytes: &[u8], budget: &Arc<Budget>, control: &Control) -> Result<Document> {
    let admission = admit(bytes.len(), budget)?;
    parse_admitted(bytes, &admission, control)
}
pub fn parse_admitted(
    bytes: &[u8],
    admission: &Arc<JsonAdmission>,
    control: &Control,
) -> Result<Document> {
    control.check()?;
    reject_reserved_number_members(bytes, control)?;
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    let root = Value::deserialize(&mut decoder).map_err(|e| Error::contract(e.to_string()))?;
    decoder.end().map_err(|e| Error::contract(e.to_string()))?;
    control.check()?;
    Ok(Document {
        root,
        _admission: admission.clone(),
    })
}
pub fn text(value: &str) -> Value {
    Value::String(value.to_owned())
}
pub fn integer(value: i128) -> Value {
    Value::Number(Number::from_i128(value).expect("i128 number"))
}
pub fn float(value: f64) -> Result<Value> {
    Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| Error::contract("nonfinite canonical float"))
}
pub fn object(entries: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    Value::Object(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
    )
}
pub fn array(entries: impl IntoIterator<Item = Value>) -> Value {
    Value::Array(entries.into_iter().collect())
}
pub fn copy(value: &Value, control: &Control) -> Result<Value> {
    control.check()?;
    let result = value.clone();
    control.check()?;
    Ok(result)
}

struct Output<'a> {
    bytes: &'a mut BudgetVec<u8>,
    error: Option<Error>,
}
impl Write for Output<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        match self.bytes.extend_from_slice(bytes) {
            Ok(()) => Ok(bytes.len()),
            Err(error) => {
                self.error = Some(error);
                Err(io::Error::other("canonical output admission failed"))
            }
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub fn canonical(value: &Value, budget: &Arc<Budget>, control: &Control) -> Result<BudgetVec<u8>> {
    control.check()?;
    let mut bytes = BudgetVec::new(budget);
    let mut output = Output {
        bytes: &mut bytes,
        error: None,
    };
    let result = value.serialize(&mut serde_json::Serializer::with_formatter(
        &mut output,
        PythonFormatter,
    ));
    if let Err(error) = result {
        return Err(output
            .error
            .take()
            .unwrap_or_else(|| Error::contract(error.to_string())));
    }
    control.check()?;
    Ok(bytes)
}
