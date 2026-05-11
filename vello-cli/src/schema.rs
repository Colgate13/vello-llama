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
    /// Acceleration targets the curator says this model is worth running on.
    /// Catalogs from before this field existed default to `["gpu"]` so they
    /// keep their previous behavior. Dense models ≥12B are typically gpu-only
    /// because CPU inference is too slow to be usable. Consumed in PR 4
    /// (recommend filter / list filter) and PR 5 (cmd_install guard).
    #[serde(default = "default_targets")]
    #[allow(dead_code)]
    pub targets: Vec<Target>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub experimental: bool,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub runtime: ModelRuntime,
}

impl Model {
    /// Used by `cmd_install`, `recommend`, and `list` once PR 4/5 land; the
    /// dead-code allow keeps PR 1 (schema-only) compiling without warnings.
    #[allow(dead_code)]
    pub fn supports(&self, t: Target) -> bool {
        self.targets.contains(&t)
    }
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

/// Runtime acceleration mode. Persisted in `profile.toml`; chosen by
/// `vello doctor --cpu` / `--gpu` or auto-detected on first run.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Gpu,
    Cpu,
}

impl Mode {
    /// Map the runtime mode to the catalog Target value so the same vocabulary
    /// works for both the user's machine and what each model declares.
    /// Consumed in PR 4 (recommend) and PR 5 (cmd_install).
    #[allow(dead_code)]
    pub fn as_target(self) -> Target {
        match self {
            Mode::Gpu => Target::Gpu,
            Mode::Cpu => Target::Cpu,
        }
    }
}

/// Acceleration target a model is curated to run on. A model can list more
/// than one (most small models work in both GPU and CPU). Catalogs from
/// before this field existed default to `[gpu]` for safety.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Target {
    Gpu,
    Cpu,
}

fn default_modalities() -> Vec<String> {
    vec!["text".into()]
}

fn default_targets() -> Vec<Target> {
    vec![Target::Gpu]
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
    /// Acceleration mode. Old profile.toml files without this field load as GPU
    /// (preserves current behavior); next `vello doctor` re-writes explicitly.
    #[serde(default = "default_mode")]
    pub mode: Mode,
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
fn default_mode() -> Mode {
    Mode::Gpu
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
            mode: default_mode(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn sample_model(targets: Vec<Target>) -> Model {
        let mut files = BTreeMap::new();
        files.insert("Q4_K_M".into(), "x.gguf".into());
        Model {
            id: "x".into(),
            repo: "r".into(),
            default_quant: "Q4_K_M".into(),
            files,
            params_total_b: 7.0,
            params_active_b: None,
            architecture: Architecture::Dense,
            modalities: vec!["text".into()],
            targets,
            tags: vec![],
            experimental: false,
            description: String::new(),
            runtime: ModelRuntime::default(),
        }
    }

    #[test]
    fn model_supports_returns_true_when_target_listed() {
        let m = sample_model(vec![Target::Gpu, Target::Cpu]);
        assert!(m.supports(Target::Gpu));
        assert!(m.supports(Target::Cpu));
    }

    #[test]
    fn model_supports_returns_false_when_target_absent() {
        let m = sample_model(vec![Target::Gpu]);
        assert!(m.supports(Target::Gpu));
        assert!(!m.supports(Target::Cpu));
    }

    #[test]
    fn mode_as_target_maps_correctly() {
        assert_eq!(Mode::Gpu.as_target(), Target::Gpu);
        assert_eq!(Mode::Cpu.as_target(), Target::Cpu);
    }

    #[test]
    fn default_targets_is_gpu_only_for_back_compat() {
        // Catalogs written before `targets` existed should default to gpu-only
        // so existing GPU users see no behavior change.
        assert_eq!(default_targets(), vec![Target::Gpu]);
    }

    #[test]
    fn default_mode_is_gpu_for_back_compat() {
        assert_eq!(default_mode(), Mode::Gpu);
    }

    #[test]
    fn profile_without_mode_field_loads_as_gpu() {
        // Old profile.toml format had no `mode` field. Verify serde default
        // kicks in and produces Mode::Gpu (preserving previous behavior).
        let raw = "vram_gb = 8.0\nram_gb = 16.0\n";
        let p: Profile = toml::from_str(raw).unwrap();
        assert_eq!(p.mode, Mode::Gpu);
    }

    #[test]
    fn model_without_targets_field_loads_as_gpu_only() {
        // Same back-compat check for the catalog model schema.
        let raw = r#"
id = "x"
repo = "r"
default_quant = "Q4_K_M"
params_total_b = 7.0
architecture = "dense"
[files]
Q4_K_M = "x.gguf"
"#;
        let m: Model = toml::from_str(raw).unwrap();
        assert_eq!(m.targets, vec![Target::Gpu]);
    }

    #[test]
    fn tier_factor_orders_correctly() {
        assert!(Tier::S.factor() > Tier::A.factor());
        assert!(Tier::A.factor() > Tier::B.factor());
        assert!(Tier::B.factor() > Tier::C.factor());
        assert!(Tier::C.factor() > Tier::D.factor());
    }
}
