use std::{fmt, num::NonZeroUsize, path::PathBuf};

/// Validated knobs that bound temporal-memory resource use.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeConfig {
    pub short_term_capacity: NonZeroUsize,
    pub chunk_size: NonZeroUsize,
    pub max_closed_chunks: NonZeroUsize,
    pub state_directory: PathBuf,
}

impl RuntimeConfig {
    /// Creates a resource-bounded runtime configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] when a capacity is zero or the state directory is empty.
    pub fn new(
        short_term_capacity: usize,
        chunk_size: usize,
        max_closed_chunks: usize,
        state_directory: PathBuf,
    ) -> Result<Self, ConfigError> {
        let short_term_capacity = NonZeroUsize::new(short_term_capacity)
            .ok_or(ConfigError::ZeroValue("short_term_capacity"))?;
        let chunk_size =
            NonZeroUsize::new(chunk_size).ok_or(ConfigError::ZeroValue("chunk_size"))?;
        let max_closed_chunks = NonZeroUsize::new(max_closed_chunks)
            .ok_or(ConfigError::ZeroValue("max_closed_chunks"))?;
        if state_directory.as_os_str().is_empty() {
            return Err(ConfigError::EmptyStateDirectory);
        }
        Ok(Self {
            short_term_capacity,
            chunk_size,
            max_closed_chunks,
            state_directory,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConfigError {
    ZeroValue(&'static str),
    EmptyStateDirectory,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroValue(name) => write!(formatter, "{name} must be greater than zero"),
            Self::EmptyStateDirectory => formatter.write_str("state_directory must not be empty"),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unbounded_zero_values() {
        let result = RuntimeConfig::new(0, 64, 64, PathBuf::from("state"));
        assert_eq!(result, Err(ConfigError::ZeroValue("short_term_capacity")));
    }
}
