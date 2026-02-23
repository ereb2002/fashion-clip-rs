pub mod embed;
pub mod config;
pub mod clip_image_processor;

pub mod error;

pub use embed::{EmbedImage, EmbedText};
pub use error::Error;

pub mod prelude {
    pub use crate::embed::{EmbedImage, EmbedText};
}
