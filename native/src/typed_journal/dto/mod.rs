//! Pure Serde wire types shared with the standalone representation experiment.
//! No engine, Python, allocator, filesystem or native-crate dependency.
pub mod borrowed;
pub mod contract;
pub mod text;
#[allow(unused_imports)] // The standalone consumer names the borrowed module directly.
pub use borrowed::*;
