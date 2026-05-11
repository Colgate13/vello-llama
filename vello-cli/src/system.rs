//! System-level configuration (`system.toml`).
//!
//! This file holds the user's machine-level preferences: which ports to bind,
//! whether to require auth on the Web UI, and the runtime fallbacks used when
//! a model doesn't declare its own. It's *the* file users edit by hand.
//!
//! Values flow into the generated `.env` via the resolver; the user should
//! never edit `.env` directly.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SystemConfig {
    #[serde(default)]
    pub ports: Ports,
    #[serde(default)]
    pub web_ui: WebUi,
    #[serde(default)]
    pub runtime: RuntimeDefaults,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Ports {
    #[serde(default = "default_llama_port")]
    pub llama: u16,
    #[serde(default = "default_webui_port")]
    pub web_ui: u16,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct WebUi {
    #[serde(default)]
    pub auth: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuntimeDefaults {
    #[serde(default = "default_ctx")]
    pub default_ctx: u32,
    #[serde(default = "default_ngl")]
    pub default_ngl: u32,
    #[serde(default = "default_threads")]
    pub default_threads: u32,
    #[serde(default = "default_batch")]
    pub default_batch: u32,
    #[serde(default = "default_ubatch")]
    pub default_ubatch: u32,
    #[serde(default = "default_kv_cache")]
    pub kv_cache_k: String,
    #[serde(default = "default_kv_cache")]
    pub kv_cache_v: String,
    #[serde(default = "default_flash_attn")]
    pub flash_attn: bool,
}

fn default_llama_port() -> u16 {
    8080
}
fn default_webui_port() -> u16 {
    3000
}
fn default_ctx() -> u32 {
    32768
}
fn default_ngl() -> u32 {
    99
}
fn default_threads() -> u32 {
    6
}
fn default_batch() -> u32 {
    2048
}
fn default_ubatch() -> u32 {
    512
}
fn default_kv_cache() -> String {
    "q8_0".into()
}
fn default_flash_attn() -> bool {
    true
}

impl Default for Ports {
    fn default() -> Self {
        Self {
            llama: default_llama_port(),
            web_ui: default_webui_port(),
        }
    }
}

impl Default for RuntimeDefaults {
    fn default() -> Self {
        Self {
            default_ctx: default_ctx(),
            default_ngl: default_ngl(),
            default_threads: default_threads(),
            default_batch: default_batch(),
            default_ubatch: default_ubatch(),
            kv_cache_k: default_kv_cache(),
            kv_cache_v: default_kv_cache(),
            flash_attn: default_flash_attn(),
        }
    }
}

pub fn load_or_default(path: &Path) -> Result<SystemConfig> {
    if !path.exists() {
        // Bootstrap: write a documented template so the user has something
        // to edit.
        fs::write(path, render_template())
            .with_context(|| format!("writing initial {}", path.display()))?;
        eprintln!(
            "info: created {} with defaults — edit and re-run if you want to change ports/runtime",
            path.display()
        );
        return Ok(SystemConfig::default());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let cfg: SystemConfig =
        toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    Ok(cfg)
}

/// Count physical CPU cores (excluding SMT siblings) by walking
/// `/proc/cpuinfo`. Each `(physical id, core id)` pair is unique per
/// physical core. Returns `None` if `/proc/cpuinfo` is unavailable or
/// can't be parsed — caller should fall back to the configured default.
///
/// Used by the resolver in CPU mode to size the `--threads` flag so we
/// pin to physical cores rather than logical (SMT pairs typically slow
/// llama.cpp down because the bottleneck is the math units, not threads).
pub fn physical_cores() -> Option<u32> {
    let raw = fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut pairs: std::collections::BTreeSet<(String, String)> = std::collections::BTreeSet::new();
    let mut cur_phys: Option<String> = None;
    let mut cur_core: Option<String> = None;
    for line in raw.lines() {
        if line.is_empty() {
            if let (Some(p), Some(c)) = (cur_phys.take(), cur_core.take()) {
                pairs.insert((p, c));
            }
            continue;
        }
        let (k, v) = match line.split_once(':') {
            Some((k, v)) => (k.trim(), v.trim().to_string()),
            None => continue,
        };
        match k {
            "physical id" => cur_phys = Some(v),
            "core id" => cur_core = Some(v),
            _ => {}
        }
    }
    // Flush trailing block if /proc/cpuinfo didn't end with a blank line.
    if let (Some(p), Some(c)) = (cur_phys, cur_core) {
        pairs.insert((p, c));
    }
    if pairs.is_empty() {
        return None;
    }
    Some(pairs.len() as u32)
}

/// Write a fresh system.toml with documented sections. Used for the example
/// file and as a template if the user runs `vello system init`.
pub fn render_template() -> String {
    r#"# vello-llama — system configuration.
# Edit this file. Vello regenerates .env from here on every `vello apply`,
# `vello switch`, or `vello install`.

[ports]
llama  = 8080
web_ui = 3000

[web_ui]
# Set true if exposing Open WebUI to a network.
auth = false

[runtime]
# Fallbacks used when a model's [runtime] block doesn't override.
default_ctx     = 32768   # context window
default_ngl     = 99      # GPU layers (99 = all)
default_threads = 6       # physical CPU cores (not SMT)
default_batch   = 2048
default_ubatch  = 512

# KV cache quantization — q8_0 saves ~50% VRAM with near-zero quality loss.
kv_cache_k = "q8_0"
kv_cache_v = "q8_0"

# Flash attention (faster + less VRAM). Disable only if you hit numeric issues.
flash_attn = true

# Note: the values above are GPU-optimized. In CPU mode (see `vello doctor
# --cpu`), the resolver applies automatic overrides — ngl=0, flash_attn=off,
# ctx capped at 8192, batch≤512, ubatch≤128, and threads=physical CPU
# cores. You can still override per-model via catalogs/*.toml [model.runtime].
"#
    .to_string()
}
