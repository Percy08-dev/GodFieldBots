//! Unchanged reserved-member scan semantics from native/src/json.rs (e441053).
//! UTF8/size/hash checks are common setup, outside deserialize/drop phase clocks.
use sha2::{Digest, Sha256};
pub const MAX_FILE: usize = 64 * 1024 * 1024;
pub const MAX_RECORD: usize = 8 * 1024 * 1024;
const RESERVED: &[u8] = b"$serde_json::private::Number";
pub fn reserved(bytes: &[u8]) -> Result<(), String> {
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                let start = index;
                index += 1;
                let mut escaped = false;
                while index < bytes.len() {
                    match bytes[index] {
                        b'\\' => {
                            escaped = true;
                            index = index.saturating_add(2)
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
                    after += 1
                }
                if bytes.get(after) == Some(&b':') {
                    let forbidden = if escaped {
                        serde_json::from_slice::<String>(&bytes[start..=end])
                            .map_err(|e| e.to_string())?
                            .as_bytes()
                            == RESERVED
                    } else {
                        &bytes[start + 1..end] == RESERVED
                    };
                    if forbidden {
                        return Err("reserved serde numeric member key".into());
                    }
                }
            }
            _ => index += 1,
        }
    }
    Ok(())
}
pub fn records<'a>(raw: &'a [u8], expected: &str) -> Result<Vec<&'a [u8]>, String> {
    if raw.len() > MAX_FILE {
        return Err("source file limit".into());
    }
    if format!("{:x}", Sha256::digest(raw)) != expected {
        return Err("frozen input hash mismatch".into());
    }
    std::str::from_utf8(raw).map_err(|e| e.to_string())?;
    let mut lines = Vec::new();
    let mut start = 0;
    while start < raw.len() {
        let mut end = start;
        while end < raw.len() && raw[end] != b'\r' && raw[end] != b'\n' {
            end += 1
        }
        let line = &raw[start..end];
        if line.len() > MAX_RECORD {
            return Err("record limit".into());
        }
        reserved(line)?;
        lines.push(line);
        start = end;
        if raw.get(start) == Some(&b'\r') {
            start += 1;
            if raw.get(start) == Some(&b'\n') {
                start += 1
            }
        } else if raw.get(start) == Some(&b'\n') {
            start += 1
        }
    }
    if lines.is_empty() {
        return Err("empty journal".into());
    }
    Ok(lines)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn guard_and_bounds() {
        for s in [
            r#"{"$serde_json::private::Number":"1"}"#,
            r#"{"\u0024serde_json::private::Number":"1"}"#,
        ] {
            assert!(reserved(s.as_bytes()).is_err())
        }
        assert!(reserved(br#"{"x":"$serde_json::private::Number"}"#).is_ok());
        assert!(records(b"{}\n", "wrong").is_err());
        let raw = b"{}\n{}\n";
        assert_eq!(
            records(raw, &format!("{:x}", Sha256::digest(raw)))
                .unwrap()
                .len(),
            2
        );
        for raw in [b"{}\r\n{}\r\n".as_slice(), b"{}\r{}\r".as_slice()] {
            assert_eq!(
                records(raw, &format!("{:x}", Sha256::digest(raw))).unwrap(),
                vec![b"{}".as_slice(), b"{}".as_slice()]
            );
        }
    }
}
