use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Contract(String),
    ResourceLimit(String),
    Cancelled,
    Deadline,
    Closed,
    Io(String),
    Cleanup {
        primary: Box<Error>,
        cleanup: String,
    },
}

impl Error {
    pub fn contract(message: impl Into<String>) -> Self {
        Self::Contract(message.into())
    }
    pub fn resource(message: impl Into<String>) -> Self {
        Self::ResourceLimit(message.into())
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(s) | Self::ResourceLimit(s) | Self::Io(s) => f.write_str(s),
            Self::Cancelled => f.write_str("native preparation cancelled"),
            Self::Deadline => f.write_str("native preparation deadline"),
            Self::Closed => f.write_str("native preparation ownership is closed"),
            Self::Cleanup { primary, .. } => fmt::Display::fmt(primary, f),
        }
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}
