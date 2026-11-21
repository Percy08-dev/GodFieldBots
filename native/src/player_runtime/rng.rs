//! CPython random.Random integer seeding and version-3 MT19937 state.
//! This is a participant-owned search/baseline stream, never Simulator RNG.
use super::{Result, ensure};
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct PythonRng {
    words: [u32; 624],
    index: usize,
    gaussian: Value,
}
impl PythonRng {
    pub fn seed(seed: u64) -> Self {
        let mut rng = Self {
            words: [0; 624],
            index: 624,
            gaussian: Value::Null,
        };
        rng.words[0] = 19650218;
        for i in 1..624 {
            rng.words[i] = 1812433253u32
                .wrapping_mul(rng.words[i - 1] ^ (rng.words[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        let keys = if seed >> 32 == 0 {
            vec![seed as u32]
        } else {
            vec![seed as u32, (seed >> 32) as u32]
        };
        let (mut i, mut j) = (1, 0);
        for _ in 0..624 {
            rng.words[i] = (rng.words[i]
                ^ (rng.words[i - 1] ^ (rng.words[i - 1] >> 30)).wrapping_mul(1664525))
            .wrapping_add(keys[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                rng.words[0] = rng.words[623];
                i = 1;
            }
            if j >= keys.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            rng.words[i] = (rng.words[i]
                ^ (rng.words[i - 1] ^ (rng.words[i - 1] >> 30)).wrapping_mul(1566083941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                rng.words[0] = rng.words[623];
                i = 1;
            }
        }
        rng.words[0] = 0x80000000;
        rng
    }
    pub fn from_state(value: &Value) -> Result<Self> {
        let a = value
            .as_array()
            .ok_or_else(|| super::error("RNG state array required"))?;
        ensure(
            a.len() == 3 && a[0].as_u64() == Some(3),
            "CPython RNG version3 required",
        )?;
        let words = a[1]
            .as_array()
            .ok_or_else(|| super::error("RNG words required"))?;
        ensure(words.len() == 625, "625 RNG state words required")?;
        let mut result = Self::seed(0);
        for (i, w) in words[..624].iter().enumerate() {
            result.words[i] =
                u32::try_from(w.as_u64().ok_or_else(|| super::error("invalid RNG word"))?)
                    .map_err(|_| super::error("invalid RNG word"))?;
        }
        result.index = words[624]
            .as_u64()
            .ok_or_else(|| super::error("invalid RNG index"))? as usize;
        ensure(result.index <= 624, "invalid RNG index")?;
        ensure(
            a[2].is_null() || a[2].as_f64().is_some_and(f64::is_finite),
            "invalid RNG Gaussian cache",
        )?;
        result.gaussian = a[2].clone();
        Ok(result)
    }
    pub fn state(&self) -> Value {
        let mut words: Vec<Value> = self.words.iter().map(|w| json!(w)).collect();
        words.push(json!(self.index));
        json!([3, words, self.gaussian])
    }
    fn word(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.words[i] & 0x80000000) | (self.words[(i + 1) % 624] & 0x7fffffff);
                self.words[i] = self.words[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 != 0 { 0x9908b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.words[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }
    pub fn bits(&mut self, k: u32) -> Result<u64> {
        ensure(k <= 64, "diagnostic getrandbits width exceeds64")?;
        Ok(match k {
            0 => 0,
            1..=32 => (self.word() >> (32 - k)) as u64,
            _ => {
                let low = self.word() as u64;
                low | ((self.word() >> (64 - k)) as u64) << 32
            }
        })
    }
    pub fn range(&mut self, n: u64) -> Result<u64> {
        ensure(n > 0, "randrange empty range")?;
        let k = 64 - n.leading_zeros();
        loop {
            let v = self.bits(k)?;
            if v < n {
                return Ok(v);
            }
        }
    }
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.range((i + 1) as u64).expect("positive range") as usize;
            values.swap(i, j);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_cpython_seed17() {
        let mut r = PythonRng::seed(17);
        assert_eq!(r.bits(32).unwrap(), 2241903809);
        assert_eq!(r.bits(32).unwrap(), 1778837931);
        let state = r.state();
        let mut restored = PythonRng::from_state(&state).unwrap();
        assert_eq!(r.bits(64).unwrap(), restored.bits(64).unwrap());
    }
}

/// Existing structured-search-stream-v1 recipe; no Simulator seed enters it.
pub fn search_seed(sampling_seed: u64) -> Result<u64> {
    let digest = super::hash(
        &json!({"recipe":"structured-search-stream-v1","sampling_seed":sampling_seed}),
    )?;
    u64::from_str_radix(&digest[..16], 16).map_err(|e| super::error(e.to_string()))
}
