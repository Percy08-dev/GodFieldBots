//! Borrow unescaped JSON strings; standard Serde owns decoded escaped strings.
use serde::{Deserialize, Deserializer, Serialize};
use std::{borrow::Cow, fmt};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Text<'a>(pub Cow<'a, str>);
impl AsRef<str> for Text<'_> {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}
impl<'de: 'a, 'a> Deserialize<'de> for Text<'a> {
    fn deserialize<D: Deserializer<'de>>(decoder: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Text<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON string")
            }
            fn visit_borrowed_str<E: serde::de::Error>(
                self,
                value: &'de str,
            ) -> Result<Self::Value, E> {
                Ok(Text(Cow::Borrowed(value)))
            }
            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(Text(Cow::Owned(value.to_owned())))
            }
            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(Text(Cow::Owned(value)))
            }
        }
        decoder.deserialize_str(Visitor)
    }
}

pub fn required_nullable<'de, D, T>(decoder: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(decoder)
}
