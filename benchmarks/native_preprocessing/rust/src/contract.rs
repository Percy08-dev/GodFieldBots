//! Representation-independent scalar/presence checks; no JSON grammar implementation.
use crate::feasibility::Text;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;

pub trait Validate {
    fn validate(&self) -> Result<(), String>;
}
pub fn text(value: &impl AsRef<str>) -> &str {
    value.as_ref()
}
pub fn ensure(ok: bool, message: &str) -> Result<(), String> {
    if ok { Ok(()) } else { Err(message.to_owned()) }
}
pub fn one_of(value: &str, allowed: &[&str]) -> Result<(), String> {
    ensure(allowed.contains(&value), "unknown string domain")
}
pub fn shape(fields: &[(&str, bool)], required: &[&str], optional: &[&str]) -> Result<(), String> {
    for (name, present) in fields {
        ensure(
            !required.contains(name) || *present,
            "missing per-tag field",
        )?;
        ensure(
            !present || required.contains(name) || optional.contains(name),
            "unexpected per-tag field",
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Player(pub u8);
impl<'de> Deserialize<'de> for Player {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = u8::deserialize(d)?;
        if (1..=2).contains(&value) {
            Ok(Self(value))
        } else {
            Err(serde::de::Error::custom("player range"))
        }
    }
}
impl Validate for Player {
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Debug, Default)]
pub enum Presence<T> {
    #[default]
    Missing,
    Null,
    Value(T),
}
impl<T> Presence<T> {
    pub fn missing(&self) -> bool {
        matches!(self, Self::Missing)
    }
    pub fn present(&self) -> bool {
        !self.missing()
    }
    pub fn value(&self) -> Option<&T> {
        if let Self::Value(v) = self {
            Some(v)
        } else {
            None
        }
    }
}
impl<'de, T: Deserialize<'de>> Deserialize<'de> for Presence<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(match Option::<T>::deserialize(d)? {
            Some(v) => Self::Value(v),
            None => Self::Null,
        })
    }
}
impl<T: Serialize> Serialize for Presence<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Value(v) => v.serialize(s),
            Self::Null => s.serialize_none(),
            Self::Missing => Err(serde::ser::Error::custom("missing cannot serialize")),
        }
    }
}
impl<T: Validate> Validate for Presence<T> {
    fn validate(&self) -> Result<(), String> {
        if let Self::Value(v) = self {
            v.validate()
        } else {
            Ok(())
        }
    }
}
impl<T: Validate> Validate for Option<T> {
    fn validate(&self) -> Result<(), String> {
        if let Some(v) = self {
            v.validate()
        } else {
            Ok(())
        }
    }
}
impl<T: Validate> Validate for Vec<T> {
    fn validate(&self) -> Result<(), String> {
        for v in self {
            v.validate()?;
        }
        Ok(())
    }
}
impl<T: Validate> Validate for Box<T> {
    fn validate(&self) -> Result<(), String> {
        self.as_ref().validate()
    }
}
impl<K: Validate + Ord, V: Validate> Validate for BTreeMap<K, V> {
    fn validate(&self) -> Result<(), String> {
        for (k, v) in self {
            k.validate()?;
            v.validate()?;
        }
        Ok(())
    }
}
#[derive(Debug, Serialize)]
#[serde(transparent)]
pub struct UniqueMap<K: Ord, V>(pub BTreeMap<K, V>);
impl<'de, K: Ord + Deserialize<'de>, V: Deserialize<'de>> Deserialize<'de> for UniqueMap<K, V> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Visitor<K, V>(std::marker::PhantomData<(K, V)>);
        impl<'de, K: Ord + Deserialize<'de>, V: Deserialize<'de>> serde::de::Visitor<'de>
            for Visitor<K, V>
        {
            type Value = UniqueMap<K, V>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("unique-key object")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<Self::Value, M::Error> {
                let mut result = BTreeMap::new();
                while let Some((key, value)) = map.next_entry()? {
                    if result.insert(key, value).is_some() {
                        return Err(serde::de::Error::custom("duplicate object key"));
                    }
                }
                Ok(UniqueMap(result))
            }
        }
        d.deserialize_map(Visitor(std::marker::PhantomData))
    }
}
impl<K: Validate + Ord, V: Validate> Validate for UniqueMap<K, V> {
    fn validate(&self) -> Result<(), String> {
        self.0.validate()
    }
}
impl Validate for String {
    fn validate(&self) -> Result<(), String> {
        ensure(!self.is_empty(), "empty string")
    }
}
impl Validate for Text<'_> {
    fn validate(&self) -> Result<(), String> {
        ensure(!self.0.is_empty(), "empty string")
    }
}
impl Validate for f64 {
    fn validate(&self) -> Result<(), String> {
        ensure(self.is_finite(), "non-finite float")
    }
}
macro_rules! scalar { ($($t:ty),*) => { $(impl Validate for $t {
    fn validate(&self)->Result<(),String> { Ok(()) }
})* }; }
scalar!(u64, u32, i32, i128, bool);
