//! Scoped borrowed Journal DTOs; semantic validation and feature traversal are separate.
pub mod dto;
pub mod journal;
pub mod validation;
pub(crate) mod view;

use crate::error::{Error, Result};
use dto::contract::Presence;

pub(crate) fn required<'a, T>(value: &'a Presence<T>) -> Result<&'a T> {
    value
        .value()
        .ok_or_else(|| Error::contract("missing non-null typed field"))
}
pub(crate) fn nullable<T>(value: &Presence<T>) -> Result<Option<&T>> {
    match value {
        Presence::Missing => Err(Error::contract("missing required nullable typed field")),
        Presence::Null => Ok(None),
        Presence::Value(value) => Ok(Some(value)),
    }
}
pub(crate) fn ensure(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::contract(message))
    }
}
