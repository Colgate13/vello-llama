//! ANSI color helpers shared across commands and diagnostics.
//!
//! Honors `NO_COLOR` and falls back to plain ASCII when stdout is not a tty
//! so output piped to files or CI is clean.

use crate::schema::Tier;
use std::io::IsTerminal;

pub struct Style {
    enabled: bool,
}

impl Style {
    pub fn new() -> Self {
        let enabled = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        Self { enabled }
    }

    pub fn dim(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[2m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn bold(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[1m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn green(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[32m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn yellow(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[33m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn red(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[31m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn cyan(&self, s: &str) -> String {
        if self.enabled {
            format!("\x1b[36m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    pub fn tier(&self, t: Tier) -> String {
        let label = t.label();
        if !self.enabled {
            return format!("[{label}]");
        }
        match t {
            Tier::S => format!("\x1b[1;32m[{label}]\x1b[0m"),
            Tier::A => format!("\x1b[32m[{label}]\x1b[0m"),
            Tier::B => format!("\x1b[36m[{label}]\x1b[0m"),
            Tier::C => format!("\x1b[33m[{label}]\x1b[0m"),
            Tier::D => format!("\x1b[2;31m[{label}]\x1b[0m"),
        }
    }
}
