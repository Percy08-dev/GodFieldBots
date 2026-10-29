//! Budget-aware JSON values. No uncharged serde/Python object graph is created.
use crate::budget::{Budget, BudgetVec, Control};
use crate::error::{Error, Result};
use std::sync::Arc;

#[derive(Debug)]
pub struct Text(BudgetVec<u8>);
impl Text {
    pub fn new(budget: &Arc<Budget>) -> Self {
        Self(BudgetVec::new(budget))
    }
    pub fn from_str(value: &str, budget: &Arc<Budget>) -> Result<Self> {
        let mut result = Self::new(budget);
        result.0.extend_from_slice(value.as_bytes())?;
        Ok(result)
    }
    pub fn as_str(&self) -> Result<&str> {
        std::str::from_utf8(&self.0)
            .map_err(|_| Error::contract("unpaired Unicode surrogate cannot be UTF-8 encoded"))
    }
    pub fn bytes(&self) -> &[u8] {
        &self.0
    }
    fn push(&mut self, byte: u8) -> Result<()> {
        self.0.push(byte)
    }
    fn codepoint(&mut self, codepoint: u32) -> Result<()> {
        if let Some(value) = char::from_u32(codepoint) {
            let mut encoded = [0; 4];
            self.0
                .extend_from_slice(value.encode_utf8(&mut encoded).as_bytes())
        } else {
            // Preserve lone surrogates through parsing, like json.loads. UTF-8
            // publication/hashing fails later, like canonical(value).encode().
            self.0.extend_from_slice(&[
                (0xe0 | (codepoint >> 12)) as u8,
                (0x80 | ((codepoint >> 6) & 63)) as u8,
                (0x80 | (codepoint & 63)) as u8,
            ])
        }
    }
}
impl PartialEq for Text {
    fn eq(&self, other: &Self) -> bool {
        self.bytes() == other.bytes()
    }
}
impl PartialEq<str> for Text {
    fn eq(&self, other: &str) -> bool {
        self.bytes() == other.as_bytes()
    }
}

#[derive(Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Int(Text),
    Float(f64),
    String(Text),
    Array(BudgetVec<Value>),
    Object(BudgetVec<(Text, Value)>),
}
impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        if let Self::Object(values) = self {
            values.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        } else {
            None
        }
    }
    pub fn require(&self, key: &str) -> Result<&Value> {
        self.get(key)
            .ok_or_else(|| Error::contract(format!("missing field {key}")))
    }
    pub fn as_str(&self) -> Result<&str> {
        if let Self::String(v) = self {
            v.as_str()
        } else {
            Err(Error::contract("string required"))
        }
    }
    pub fn as_array(&self) -> Result<&[Value]> {
        if let Self::Array(v) = self {
            Ok(v)
        } else {
            Err(Error::contract("array required"))
        }
    }
    pub fn as_object(&self) -> Result<&[(Text, Value)]> {
        if let Self::Object(v) = self {
            Ok(v)
        } else {
            Err(Error::contract("object required"))
        }
    }
    pub fn as_bool(&self) -> Result<bool> {
        if let Self::Bool(v) = self {
            Ok(*v)
        } else {
            Err(Error::contract("boolean required"))
        }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
    pub fn integer(&self, low: i128, high: i128) -> Result<i128> {
        let Self::Int(value) = self else {
            return Err(Error::contract("integer required"));
        };
        let value = value
            .as_str()?
            .parse::<i128>()
            .map_err(|_| Error::contract("integer outside range"))?;
        if value < low || value > high {
            return Err(Error::contract("integer outside range"));
        }
        Ok(value)
    }
    pub fn number(&self) -> Result<f64> {
        match self {
            Self::Int(v) => v
                .as_str()?
                .parse()
                .map_err(|_| Error::contract("number outside range")),
            Self::Float(v) => Ok(*v),
            _ => Err(Error::contract("number required")),
        }
    }
    pub fn closed(&self, required: &[&str], optional: &[&str]) -> Result<()> {
        let entries = self.as_object()?;
        if required.iter().any(|key| self.get(key).is_none())
            || entries.iter().any(|(key, _)| {
                !required
                    .iter()
                    .chain(optional)
                    .any(|expected| key == *expected)
            })
        {
            return Err(Error::contract("invalid closed object fields"));
        }
        Ok(())
    }
    /// Equality used by existing Python dict/list evidence comparisons. Type
    /// validation remains explicit at integer/boolean/canonical boundaries.
    pub fn eq_python(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bool(a), Self::Int(b)) | (Self::Int(b), Self::Bool(a)) => {
                b.bytes() == if *a { b"1" } else { b"0" }
            }
            (Self::Bool(a), Self::Float(b)) | (Self::Float(b), Self::Bool(a)) => {
                (*a as u8 as f64) == *b
            }
            (Self::Int(a), Self::Float(b)) | (Self::Float(b), Self::Int(a)) => {
                int_float_equal(a, *b)
            }
            (Self::Array(a), Self::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(a, b)| a.eq_container_child(b))
            }
            (Self::Object(a), Self::Object(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .all(|(k, v)| b.iter().any(|(bk, bv)| k == bk && v.eq_container_child(bv)))
            }
            _ => self == other,
        }
    }
    fn eq_container_child(&self, other: &Self) -> bool {
        // CPython json.loads shares its NaN token object. Only container child
        // comparisons short-circuit that identity; scalar NaN equality is false.
        if let (Self::Float(a), Self::Float(b)) = (self, other) {
            if a.is_nan() && b.is_nan() {
                return true;
            }
        }
        self.eq_python(other)
    }
}
fn int_float_equal(integer: &Text, float: f64) -> bool {
    if !float.is_finite() || float.fract() != 0.0 {
        return false;
    }
    if float == 0.0 {
        return integer.bytes() == b"0";
    }
    let bits = float.abs().to_bits();
    let mut mantissa = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 - 52;
    let mut decimal = [0u8; 310];
    let mut count = 0;
    if exponent < 0 {
        mantissa >>= -exponent;
    }
    while mantissa > 0 {
        decimal[count] = (mantissa % 10) as u8;
        count += 1;
        mantissa /= 10;
    }
    for _ in 0..exponent.max(0) {
        let mut carry = 0;
        for digit in &mut decimal[..count] {
            let doubled = *digit * 2 + carry;
            *digit = doubled % 10;
            carry = doubled / 10;
        }
        if carry != 0 {
            decimal[count] = carry;
            count += 1;
        }
    }
    let text = integer.bytes();
    let negative = float.is_sign_negative();
    if text.len() != count + negative as usize || negative && text[0] != b'-' {
        return false;
    }
    text[negative as usize..]
        .iter()
        .rev()
        .zip(decimal[..count].iter())
        .all(|(byte, digit)| *byte == b'0' + digit)
}
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Int(a), Self::Int(b)) | (Self::String(a), Self::String(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a == b,
            (Self::Array(a), Self::Array(b)) => **a == **b,
            (Self::Object(a), Self::Object(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .all(|(k, v)| b.iter().any(|(bk, bv)| k == bk && v == bv))
            }
            _ => false,
        }
    }
}

pub struct Document {
    pub root: Value,
}

pub fn text(value: &str, budget: &Arc<Budget>) -> Result<Value> {
    Ok(Value::String(Text::from_str(value, budget)?))
}
pub fn integer(value: i128, budget: &Arc<Budget>) -> Result<Value> {
    Ok(Value::Int(Text::from_str(&value.to_string(), budget)?))
}
pub fn object(
    entries: impl IntoIterator<Item = (&'static str, Value)>,
    budget: &Arc<Budget>,
) -> Result<Value> {
    let mut values = BudgetVec::new(budget);
    for (key, value) in entries {
        values.push((Text::from_str(key, budget)?, value))?;
    }
    Ok(Value::Object(values))
}
pub fn array(entries: impl IntoIterator<Item = Value>, budget: &Arc<Budget>) -> Result<Value> {
    let mut values = BudgetVec::new(budget);
    for value in entries {
        values.push(value)?;
    }
    Ok(Value::Array(values))
}
pub fn copy(value: &Value, budget: &Arc<Budget>, control: &Control) -> Result<Value> {
    // Used only for compact control/proof fields, never full journal input trees.
    let encoded = canonical(value, budget, control)?;
    Ok(parse(&encoded, budget, control)?.root)
}

pub fn parse(bytes: &[u8], budget: &Arc<Budget>, control: &Control) -> Result<Document> {
    control.check()?;
    std::str::from_utf8(bytes).map_err(|_| Error::contract("invalid UTF-8 JSON"))?;
    control.check()?;
    parse_decoded(bytes, budget, control)
}
pub fn parse_surrogatepass(
    bytes: &[u8],
    budget: &Arc<Budget>,
    control: &Control,
) -> Result<Document> {
    let mut index = 0;
    let mut next_poll = 0usize;
    while index < bytes.len() {
        if index >= next_poll {
            control.check()?;
            next_poll = index.saturating_add(4096);
        }
        let first = bytes[index];
        let (count, mut point, minimum) = match first {
            0..=127 => {
                index += 1;
                continue;
            }
            0xc2..=0xdf => (2, (first & 31) as u32, 0x80),
            0xe0..=0xef => (3, (first & 15) as u32, 0x800),
            0xf0..=0xf4 => (4, (first & 7) as u32, 0x10000),
            _ => return Err(Error::contract("invalid UTF-8 JSON")),
        };
        if index + count > bytes.len() {
            return Err(Error::contract("truncated UTF-8 JSON"));
        }
        for byte in &bytes[index + 1..index + count] {
            if byte & 0xc0 != 0x80 {
                return Err(Error::contract("invalid UTF-8 JSON"));
            }
            point = (point << 6) | (byte & 63) as u32;
        }
        if point < minimum || point > 0x10ffff {
            return Err(Error::contract("invalid UTF-8 JSON"));
        }
        index += count;
    }
    parse_decoded(bytes, budget, control)
}
fn parse_decoded(bytes: &[u8], budget: &Arc<Budget>, control: &Control) -> Result<Document> {
    let mut parser = Parser {
        bytes,
        pos: 0,
        budget,
        control,
        next_poll: 4096,
    };
    let root = parser.value(0)?;
    parser.space()?;
    if parser.pos != bytes.len() {
        return Err(Error::contract("trailing JSON data"));
    }
    control.check()?;
    Ok(Document { root })
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    budget: &'a Arc<Budget>,
    control: &'a Control,
    next_poll: usize,
}
impl Parser<'_> {
    fn poll(&mut self) -> Result<()> {
        if self.pos >= self.next_poll {
            self.control.check()?;
            self.next_poll = self.pos.saturating_add(4096);
        }
        Ok(())
    }
    fn space(&mut self) -> Result<()> {
        while self.pos < self.bytes.len()
            && matches!(self.bytes[self.pos], b' ' | b'\n' | b'\r' | b'\t')
        {
            self.poll()?;
            self.pos += 1;
        }
        Ok(())
    }
    fn token(&mut self, token: &[u8]) -> bool {
        if self.bytes[self.pos..].starts_with(token) {
            self.pos += token.len();
            true
        } else {
            false
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value> {
        self.control.check()?;
        if depth > self.control.json_depth() {
            return Err(Error::resource("source exceeds native_json_depth"));
        }
        self.space()?;
        match self.bytes.get(self.pos).copied() {
            Some(b'n') if self.token(b"null") => Ok(Value::Null),
            Some(b't') if self.token(b"true") => Ok(Value::Bool(true)),
            Some(b'f') if self.token(b"false") => Ok(Value::Bool(false)),
            Some(b'N') if self.token(b"NaN") => Ok(Value::Float(f64::NAN)),
            Some(b'I') if self.token(b"Infinity") => Ok(Value::Float(f64::INFINITY)),
            Some(b'-') if self.bytes[self.pos..].starts_with(b"-Infinity") => {
                self.pos += 9;
                Ok(Value::Float(f64::NEG_INFINITY))
            }
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b'[') => {
                self.pos += 1;
                self.space()?;
                let mut items = BudgetVec::new(self.budget);
                if self.token(b"]") {
                    return Ok(Value::Array(items));
                }
                loop {
                    items.reserve(1)?;
                    let value = self.value(depth + 1)?;
                    items.push(value)?;
                    self.space()?;
                    if self.token(b"]") {
                        break;
                    }
                    if !self.token(b",") {
                        return Err(Error::contract("JSON array separator required"));
                    }
                }
                Ok(Value::Array(items))
            }
            Some(b'{') => {
                self.pos += 1;
                self.space()?;
                let mut items: BudgetVec<(Text, Value)> = BudgetVec::new(self.budget);
                let mut lookup = crate::lookup::Index::new(self.budget);
                if self.token(b"}") {
                    return Ok(Value::Object(items));
                }
                loop {
                    self.space()?;
                    let key = self.string()?;
                    self.space()?;
                    if !self.token(b":") {
                        return Err(Error::contract("JSON object colon required"));
                    }
                    let key_hash = crate::lookup::hash(key.bytes(), self.control)?;
                    let found =
                        lookup.find(key_hash, |index| items[index].0 == key, self.control)?;
                    if found.is_none() {
                        items.reserve(1)?;
                    }
                    let value = self.value(depth + 1)?;
                    if let Some(index) = found {
                        items[index].1 = value;
                    } else {
                        lookup.insert(key_hash, items.len(), self.control)?;
                        items.push((key, value))?;
                    }
                    self.space()?;
                    if self.token(b"}") {
                        break;
                    }
                    if !self.token(b",") {
                        return Err(Error::contract("JSON object separator required"));
                    }
                }
                Ok(Value::Object(items))
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(Error::contract("invalid JSON value")),
        }
    }
    fn hex(&mut self) -> Result<u32> {
        let mut value = 0;
        for _ in 0..4 {
            let digit = *self
                .bytes
                .get(self.pos)
                .ok_or_else(|| Error::contract("short Unicode escape"))?;
            self.pos += 1;
            value = value * 16
                + match digit {
                    b'0'..=b'9' => (digit - b'0') as u32,
                    b'a'..=b'f' => (digit - b'a' + 10) as u32,
                    b'A'..=b'F' => (digit - b'A' + 10) as u32,
                    _ => return Err(Error::contract("invalid Unicode escape")),
                };
        }
        Ok(value)
    }
    fn string(&mut self) -> Result<Text> {
        if !self.token(b"\"") {
            return Err(Error::contract("JSON string required"));
        }
        let mut value = Text::new(self.budget);
        loop {
            self.poll()?;
            let byte = *self
                .bytes
                .get(self.pos)
                .ok_or_else(|| Error::contract("unterminated JSON string"))?;
            self.pos += 1;
            match byte {
                b'"' => return Ok(value),
                b'\\' => {
                    let escape = *self
                        .bytes
                        .get(self.pos)
                        .ok_or_else(|| Error::contract("unterminated JSON escape"))?;
                    self.pos += 1;
                    match escape {
                        b'"' | b'\\' | b'/' => value.push(escape)?,
                        b'b' => value.push(8)?,
                        b'f' => value.push(12)?,
                        b'n' => value.push(10)?,
                        b'r' => value.push(13)?,
                        b't' => value.push(9)?,
                        b'u' => {
                            let first = self.hex()?;
                            if (0xd800..=0xdbff).contains(&first)
                                && self.bytes[self.pos..].starts_with(b"\\u")
                            {
                                let saved = self.pos;
                                self.pos += 2;
                                let second = self.hex()?;
                                if (0xdc00..=0xdfff).contains(&second) {
                                    value.codepoint(
                                        0x10000 + (first - 0xd800) * 1024 + second - 0xdc00,
                                    )?;
                                } else {
                                    self.pos = saved;
                                    value.codepoint(first)?;
                                }
                            } else {
                                value.codepoint(first)?;
                            }
                        }
                        _ => return Err(Error::contract("invalid JSON escape")),
                    }
                }
                0..=31 => return Err(Error::contract("unescaped JSON control character")),
                _ => value.push(byte)?,
            }
        }
    }
    fn number(&mut self) -> Result<Value> {
        let start = self.pos;
        self.token(b"-");
        if !self.token(b"0") {
            if !matches!(self.bytes.get(self.pos), Some(b'1'..=b'9')) {
                return Err(Error::contract("invalid JSON number"));
            }
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.poll()?;
                self.pos += 1;
            }
        }
        let mut float = false;
        if self.token(b".") {
            float = true;
            let begin = self.pos;
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.poll()?;
                self.pos += 1;
            }
            if begin == self.pos {
                return Err(Error::contract("missing fractional digits"));
            }
        }
        if self.token(b"e") || self.token(b"E") {
            float = true;
            if !self.token(b"+") {
                self.token(b"-");
            }
            let begin = self.pos;
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.poll()?;
                self.pos += 1;
            }
            if begin == self.pos {
                return Err(Error::contract("missing exponent digits"));
            }
        }
        let number = std::str::from_utf8(&self.bytes[start..self.pos]).expect("ASCII number");
        if float {
            return Ok(Value::Float(
                number
                    .parse()
                    .map_err(|_| Error::contract("invalid JSON float"))?,
            ));
        }
        if number.trim_start_matches('-').len() > 4300 {
            return Err(Error::contract("integer conversion digit limit"));
        }
        Ok(Value::Int(Text::from_str(
            if number == "-0" { "0" } else { number },
            self.budget,
        )?))
    }
}

pub fn canonical(value: &Value, budget: &Arc<Budget>, control: &Control) -> Result<BudgetVec<u8>> {
    let mut output = BudgetVec::new(budget);
    write_canonical(value, &mut output, control)?;
    control.check()?;
    Ok(output)
}
fn quoted(value: &Text, output: &mut BudgetVec<u8>, control: &Control) -> Result<()> {
    value.as_str()?;
    output.push(b'"')?;
    for (index, byte) in value.bytes().iter().enumerate() {
        if index % 4096 == 0 {
            control.check()?;
        }
        match byte {
            b'"' | b'\\' => {
                output.push(b'\\')?;
                output.push(*byte)?;
            }
            8 => output.extend_from_slice(b"\\b")?,
            9 => output.extend_from_slice(b"\\t")?,
            10 => output.extend_from_slice(b"\\n")?,
            12 => output.extend_from_slice(b"\\f")?,
            13 => output.extend_from_slice(b"\\r")?,
            0..=31 => {
                output.extend_from_slice(b"\\u00")?;
                output.push(b"0123456789abcdef"[(*byte / 16) as usize])?;
                output.push(b"0123456789abcdef"[(*byte % 16) as usize])?;
            }
            _ => output.push(*byte)?,
        }
    }
    output.push(b'"')
}
fn write_canonical(value: &Value, out: &mut BudgetVec<u8>, control: &Control) -> Result<()> {
    control.check()?;
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(v) => out.extend_from_slice(if *v { b"true" } else { b"false" }),
        Value::Int(v) => out.extend_from_slice(v.bytes()),
        Value::Float(v) => write_float(*v, out),
        Value::String(v) => quoted(v, out, control),
        Value::Array(values) => {
            out.push(b'[')?;
            for (i, value) in values.iter().enumerate() {
                if i > 0 {
                    out.push(b',')?;
                }
                write_canonical(value, out, control)?;
            }
            out.push(b']')
        }
        Value::Object(values) => {
            let mut sorted = BudgetVec::with_capacity(out.budget(), values.len())?;
            for item in values.iter() {
                sorted.push(item)?;
            }
            crate::budget::sort_by(&mut sorted, |a, b| a.0.bytes().cmp(b.0.bytes()), control)?;
            out.push(b'{')?;
            for (i, (key, value)) in sorted.iter().enumerate() {
                if i > 0 {
                    out.push(b',')?;
                }
                quoted(key, out, control)?;
                out.push(b':')?;
                write_canonical(value, out, control)?;
            }
            out.push(b'}')
        }
    }
}
fn write_float(value: f64, out: &mut BudgetVec<u8>) -> Result<()> {
    if !value.is_finite() {
        return Err(Error::contract("nonfinite canonical float"));
    }
    let mut buffer = ryu::Buffer::new();
    let printed = buffer.format_finite(value);
    let (negative, printed) = if let Some(v) = printed.strip_prefix('-') {
        (true, v)
    } else {
        (false, printed)
    };
    if negative {
        out.push(b'-')?;
    }
    let (mantissa, exponent) = if let Some((a, b)) = printed.split_once('e') {
        (a, b.parse::<i32>().expect("ryu exponent"))
    } else {
        (printed, 0)
    };
    let mut digits = [0u8; 32];
    let mut count = 0;
    let mut dot = None;
    for byte in mantissa.bytes() {
        if byte == b'.' {
            dot = Some(count);
        } else {
            digits[count] = byte;
            count += 1;
        }
    }
    let decimal = dot.unwrap_or(count) as i32 + exponent;
    let mut first = 0;
    while first + 1 < count && digits[first] == b'0' {
        first += 1;
    }
    let scientific = decimal - first as i32 - 1;
    let mut last = count;
    while last > first + 1 && digits[last - 1] == b'0' {
        last -= 1;
    }
    if !(-4..16).contains(&scientific) && value != 0.0 {
        out.push(digits[first])?;
        if last > first + 1 {
            out.push(b'.')?;
            out.extend_from_slice(&digits[first + 1..last])?;
        }
        out.push(b'e')?;
        out.push(if scientific < 0 { b'-' } else { b'+' })?;
        let exponent = scientific.unsigned_abs();
        if exponent < 10 {
            out.push(b'0')?;
        }
        let text = exponent.to_string();
        out.extend_from_slice(text.as_bytes())
    } else {
        let left = decimal - first as i32;
        if left <= 0 {
            out.extend_from_slice(b"0.")?;
            for _ in 0..-left {
                out.push(b'0')?;
            }
            out.extend_from_slice(&digits[first..last])?;
        } else {
            for i in 0..left as usize {
                out.push(if first + i < last {
                    digits[first + i]
                } else {
                    b'0'
                })?;
            }
            out.push(b'.')?;
            if first + (left as usize) < last {
                out.extend_from_slice(&digits[first + left as usize..last])?;
            } else {
                out.push(b'0')?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::monotonic;
    #[test]
    fn duplicate_last_wins_and_canonical_floats() {
        let budget = Budget::new(1024 * 1024);
        let control = Control::new(monotonic() + 10.).unwrap();
        let parsed = parse(
            br#"{"a":1,"a":2,"x":[-0,1e-5,1e16,0.0001,1.0,-0.0]}"#,
            &budget,
            &control,
        )
        .unwrap();
        let result = canonical(&parsed.root, &budget, &control).unwrap();
        assert_eq!(&*result, br#"{"a":2,"x":[0,1e-05,1e+16,0.0001,1.0,-0.0]}"#);
        drop(result);
        drop(parsed);
        assert_eq!(budget.used(), 0);
    }
}
