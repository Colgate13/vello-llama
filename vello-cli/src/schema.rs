//! Catalog and profile schema definitions.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
pub struct Catalog {
    pub schema_version: u32,
    pub name: String,
    #[serde(default)]
    pub maintainer: String,
    #[serde(default, rename = "model")]
    pub models: Vec<Model>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Model {
    pub id: String,
    pub repo: String,
    pub default_quant: String,
    pub files: BTreeMap<String, String>,

    pub params_total_b: f32,
    #[serde(default)]
    pub params_active_b: Option<f32>,
    pub architecture: Architecture,

    #[serde(default = "default_modalities")]
    pub modalities: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub experimental: bool,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub runtime: ModelRuntime,
}

/// Per-model runtime knobs. All optional; resolver falls back to system.toml
/// defaults, then to hard-coded sane values.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelRuntime {
    /// Default context window to load this model with.
    #[serde(default)]
    pub ctx_default: Option<u32>,
    /// GPU layer count override (default 99 = all).
    #[serde(default)]
    pub ngl: Option<u32>,
    /// mmproj filename for multimodal models. Lives in models/.
    #[serde(default)]
    pub mmproj: Option<String>,
    /// Free-form extra llama-server CLI args appended verbatim.
    /// Examples: ["--n-cpu-moe", "24"] for MoE offload.
    #[serde(default)]
    pub extra_args: Vec<String>,
    /// Override KV cache K type (default from system.toml).
    #[serde(default)]
    pub kv_cache_k: Option<String>,
    /// Override KV cache V type (default from system.toml).
    #[serde(default)]
    pub kv_cache_v: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Architecture {
    Dense,
    Moe,
}

fn default_modalities() -> Vec<String> {
    vec!["text".into()]
}

/// Hardware profile loaded from profile.toml or auto-detected.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Profile {
    pub vram_gb: f32,
    pub ram_gb: f32,
    #[serde(default)]
    pub gpu_name: String,
    #[serde(default)]
    pub cuda_arch: Option<u32>,
    /// Reserved VRAM for system/desktop (GB). Default 1.0.
    #[serde(default = "default_vram_reserve")]
    pub vram_reserve_gb: f32,
    /// Reserved RAM for system (GB). Default 4.0.
    #[serde(default = "default_ram_reserve")]
    pub ram_reserve_gb: f32,
    /// Default context size used to estimate KV cache. Default 32768.
    #[serde(default = "default_ctx")]
    pub default_ctx: u32,
}

fn default_vram_reserve() -> f32 {
    1.0
}
fn default_ram_reserve() -> f32 {
    4.0
}
fn default_ctx() -> u32 {
    32768
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            vram_gb: 8.0,
            ram_gb: 16.0,
            gpu_name: String::new(),
            cuda_arch: None,
            vram_reserve_gb: default_vram_reserve(),
            ram_reserve_gb: default_ram_reserve(),
            default_ctx: default_ctx(),
        }
    }
}

/// Tier as displayed to the user — calculated, never stored on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Fits 100% in VRAM, fast.
    S,
    /// Mostly in VRAM, light overflow possible.
    A,
    /// Heavy CPU offload to RAM, MoE comfortable here.
    B,
    /// RAM-dominant, slow but works.
    C,
    /// Won't fit at all.
    D,
}

impl Tier {
    pub fn label(&self) -> &'static str {
        match self {
            Tier::S => "S",
            Tier::A => "A",
            Tier::B => "B",
            Tier::C => "C",
            Tier::D => "D",
        }
    }

    /// Used to rank recommendations.
    pub fn factor(&self) -> f32 {
        match self {
            Tier::S => 1.00,
            Tier::A => 0.85,
            Tier::B => 0.55,
            Tier::C => 0.20,
            Tier::D => 0.00,
        }
    }
}
