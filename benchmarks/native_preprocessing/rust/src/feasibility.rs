//! Standard Serde wire-form feasibility fixtures.
pub use crate::shared_dto::text::{Text, required_nullable};
use serde::{Deserialize, Deserializer, Serialize};
#[cfg(test)]
use std::borrow::Cow;
use std::fmt;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Selection<'a> {
    pub index: u64,
    #[serde(borrow)]
    pub policy_version: Text<'a>,
    #[serde(deserialize_with = "required_nullable")]
    pub logprob: Option<f64>,
    #[serde(deserialize_with = "required_nullable")]
    pub value: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Outcome {
    Draw,
    Win(Winner),
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Winner {
    pub winner: u8,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Element<'a> {
    #[serde(borrow)]
    pub source: Text<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<u64>,
    #[serde(borrow, skip_serializing_if = "Option::is_none")]
    pub definition_id: Option<Text<'a>>,
}

#[derive(Debug, Serialize)]
pub enum Defense<'a> {
    Instance(u64),
    Element(Element<'a>),
}
impl<'de: 'a, 'a> Deserialize<'de> for Defense<'a> {
    fn deserialize<D: Deserializer<'de>>(decoder: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Defense<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("integer own reference or typed own element")
            }
            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(Defense::Instance(value))
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                map: M,
            ) -> Result<Self::Value, M::Error> {
                Element::deserialize(serde::de::value::MapAccessDeserializer::new(map))
                    .map(Defense::Element)
            }
        }
        decoder.deserialize_any(Visitor)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Response<'a> {
    #[serde(rename = "type", borrow)]
    pub tag: Text<'a>,
    #[serde(borrow)]
    pub ordered_defenses: Vec<Defense<'a>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn floats_null_missing_duplicates_and_exact_integer_domain() {
        let value: Selection = serde_json::from_str(
            r#"{"index":0,"policy_version":"ppo","logprob":-0.125,"value":1.25}"#,
        )
        .unwrap();
        assert_eq!(value.logprob, Some(-0.125));
        assert_eq!(value.value, Some(1.25));
        assert!(matches!(value.policy_version.0, Cow::Borrowed(_)));
        let null: Selection = serde_json::from_str(
            r#"{"index":0,"policy_version":"ppo","logprob":null,"value":null}"#,
        )
        .unwrap();
        assert!(null.logprob.is_none());
        for invalid in [
            r#"{"index":0,"policy_version":"ppo","value":null}"#,
            r#"{"index":0,"index":1,"policy_version":"ppo","logprob":null,"value":null}"#,
            r#"{"index":0,"policy_version":"ppo","logprob":null,"value":null,"extra":1}"#,
            r#"{"index":0.0,"policy_version":"ppo","logprob":null,"value":null}"#,
            r#"{"index":18446744073709551616,"policy_version":"ppo","logprob":null,"value":null}"#,
            r#"{"index":0,"policy_version":"ppo","logprob":1e999,"value":null}"#,
        ] {
            assert!(
                serde_json::from_str::<Selection>(invalid).is_err(),
                "{invalid}"
            );
        }
    }
    #[test]
    fn standard_external_outcome_and_nullable() {
        for raw in ["null", "\"draw\"", r#"{"win":{"winner":2}}"#] {
            let value: Option<Outcome> = serde_json::from_str(raw).unwrap();
            assert_eq!(serde_json::to_string(&value).unwrap(), raw);
        }
    }
    #[test]
    fn both_defense_shapes_empty_and_nested_borrowing() {
        for raw in [
            r#"{"type":"respond","ordered_defenses":[]}"#,
            r#"{"type":"respond","ordered_defenses":[1,18446744073709551615]}"#,
        ] {
            let value: Response = serde_json::from_str(raw).unwrap();
            assert_eq!(value.tag.0, "respond");
        }
        let value: Response = serde_json::from_str(
            r#"{"type":"respond_sequence","ordered_defenses":[{"source":"hand_artifact","instance_id":3},{"source":"learned_miracle","definition_id":"esc\u0061ped"}]}"#).unwrap();
        let Defense::Element(first) = &value.ordered_defenses[0] else {
            panic!()
        };
        assert!(matches!(first.source.0, Cow::Borrowed(_)));
        let Defense::Element(second) = &value.ordered_defenses[1] else {
            panic!()
        };
        assert!(matches!(
            second.definition_id.as_ref().unwrap().0,
            Cow::Owned(_)
        ));
    }
}
