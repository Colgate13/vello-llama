//! Merge profile + system + active model's runtime → resolved env.
//!
//! Output is a deterministic .env file consumed by docker-compose. Precedence
//! (high → low): catalog [model.runtime] > system.toml > built-in defaults.

use crate::docker;
use crate::paths::Paths;
use crate::schema::{Mode, Model, Profile};
use crate::system::{self, SystemConfig};
use anyhow::{Context, Result};
use std::fs;

/// Final resolved values that get written to .env.
#[derive(Debug, Clone)]
pub struct Resolved {
    pub llama_port: u16,
    pub webui_port: u16,
    pub webui_auth: bool,

    pub model_file: String,
    pub model_alias: String,

    pub ctx: u32,
    pub ngl: u32,
    pub threads: u32,
    pub batch: u32,
    pub ubatch: u32,

    pub kv_cache_k: String,
    pub kv_cache_v: String,
    pub flash_attn: bool,

    pub mmproj: String,     // empty if no vision
    pub extra_args: String, // space-joined, empty if none

    /// Acceleration mode for this run — propagated into the .env as
    /// `LLAMA_RUNTIME` and `COMPOSE_PROFILES` so `docker compose up` (without
    /// an explicit `--profile`) still activates the right service.
    pub mode: Mode,
}

pub struct ResolveInput<'a> {
    pub model: &'a Model,
    pub model_file: &'a str,
    pub system: &'a SystemConfig,
    pub profile: &'a Profile,
}

pub fn resolve(input: &ResolveInput<'_>) -> Resolved {
    let mut r = resolve_gpu_baseline(input);
    if input.profile.mode == Mode::Cpu {
        apply_cpu_overrides(&mut r, input.system);
    }
    r
}

/// CPU mode overrides applied on top of the GPU-flavored baseline.
///
/// llama.cpp's CPU build doesn't support `--flash-attn` and ignores
/// `--n-gpu-layers`. We zero NGL defensively (in case the user is on a host
/// with both CPU and CUDA builds present), flip flash-attn off, and shrink
/// the batch/context defaults to values appropriate for CPU inference where
/// KV cache lives in RAM rather than VRAM.
///
/// Threads default to physical cores only when the user kept the system.toml
/// default (6) — any explicit override is respected.
fn apply_cpu_overrides(r: &mut Resolved, sys: &SystemConfig) {
    r.ngl = 0;
    r.flash_attn = false;
    r.ctx = r.ctx.min(8192);
    r.batch = r.batch.min(512);
    r.ubatch = r.ubatch.min(128);
    // Only override threads if the user hasn't customised it. system.toml's
    // documented default is 6; anything else means an intentional choice.
    if sys.runtime.default_threads == 6 {
        if let Some(physical) = system::physical_cores() {
            r.threads = physical;
        }
    }
}

fn resolve_gpu_baseline(input: &ResolveInput<'_>) -> Resolved {
    let m = input.model;
    let sys = input.system;
    let profile = input.profile;
    let rt = &m.runtime;

    let ctx = rt.ctx_default.unwrap_or(sys.runtime.default_ctx);
    let ngl = rt.ngl.unwrap_or(sys.runtime.default_ngl);
    let kv_cache_k = rt
        .kv_cache_k
        .clone()
        .unwrap_or_else(|| sys.runtime.kv_cache_k.clone());
    let kv_cache_v = rt
        .kv_cache_v
        .clone()
        .unwrap_or_else(|| sys.runtime.kv_cache_v.clone());

    // Compose extra_args: mmproj (if any) prepended to the model's extra args.
    // Both flow through LLAMA_EXTRA_ARGS so docker-compose only needs one
    // substitution slot.
    let mut extras: Vec<String> = Vec::new();
    if let Some(mmproj) = &rt.mmproj {
        if !mmproj.is_empty() {
            extras.push("--mmproj".into());
            extras.push(format!("/models/{mmproj}"));
        }
    }
    extras.extend(rt.extra_args.iter().cloned());

    Resolved {
        llama_port: sys.ports.llama,
        webui_port: sys.ports.web_ui,
        webui_auth: sys.web_ui.auth,

        model_file: input.model_file.to_string(),
        model_alias: format!("{}-local", m.id),

        ctx,
        ngl,
        threads: sys.runtime.default_threads,
        batch: sys.runtime.default_batch,
        ubatch: sys.runtime.default_ubatch,

        kv_cache_k,
        kv_cache_v,
        flash_attn: sys.runtime.flash_attn,

        mmproj: rt.mmproj.clone().unwrap_or_default(),
        extra_args: extras.join(" "),

        mode: profile.mode,
    }
}

/// Write the resolved values to `.env`. Preserves any custom keys the user
/// added (lines we don't recognize are kept as-is).
pub fn write_env(paths: &Paths, r: &Resolved) -> Result<()> {
    let env_path = paths.project_root.join(".env");
    let existing = fs::read_to_string(&env_path).unwrap_or_default();
    let preserved = preserve_unmanaged(&existing);

    let body = render(r, &preserved);
    fs::write(&env_path, body).with_context(|| format!("writing {}", env_path.display()))?;
    Ok(())
}

/// Keys vello manages — anything else in the existing .env is kept verbatim.
const MANAGED_KEYS: &[&str] = &[
    "LLAMA_PORT",
    "WEBUI_PORT",
    "WEBUI_AUTH",
    "LLAMA_MODEL_FILE",
    "LLAMA_MODEL_ALIAS",
    "LLAMA_CTX",
    "LLAMA_NGL",
    "LLAMA_THREADS",
    "LLAMA_BATCH",
    "LLAMA_UBATCH",
    "LLAMA_KV_CACHE_K",
    "LLAMA_KV_CACHE_V",
    "LLAMA_FLASH_ATTN",
    "LLAMA_EXTRA_ARGS",
    "LLAMA_RUNTIME",
    "COMPOSE_PROFILES",
    // Removed in this version — drop on regen so docker-compose doesn't see it.
    "LLAMA_MMPROJ",
    // Legacy keys we no longer write but should drop on regeneration so the
    // user doesn't end up with stale leftovers from old .env.example versions.
    "LOCAL_LLM_DEFAULT_MODEL",
];

fn preserve_unmanaged(existing: &str) -> String {
    let mut kept = String::new();
    for line in existing.lines() {
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some((k, _)) = line.split_once('=') {
            let key = k.trim();
            if MANAGED_KEYS.contains(&key) {
                continue;
            }
            kept.push_str(line);
            kept.push('\n');
        }
    }
    kept
}

fn render(r: &Resolved, preserved: &str) -> String {
    let mut out = String::new();
    out.push_str("# Generated by `vello apply`. Edit system.toml or catalogs/*.toml,\n");
    out.push_str("# then run `vello apply` (or `vello switch <id>`) to regenerate.\n");
    out.push_str("# Lines you add manually are preserved across regenerations.\n\n");

    out.push_str("# Ports\n");
    out.push_str(&format!("LLAMA_PORT={}\n", r.llama_port));
    out.push_str(&format!("WEBUI_PORT={}\n\n", r.webui_port));

    out.push_str("# Active model\n");
    out.push_str(&format!("LLAMA_MODEL_FILE={}\n", r.model_file));
    out.push_str(&format!("LLAMA_MODEL_ALIAS={}\n\n", r.model_alias));

    out.push_str("# Runtime (resolved from system.toml + model [runtime])\n");
    out.push_str(&format!("LLAMA_CTX={}\n", r.ctx));
    out.push_str(&format!("LLAMA_NGL={}\n", r.ngl));
    out.push_str(&format!("LLAMA_THREADS={}\n", r.threads));
    out.push_str(&format!("LLAMA_BATCH={}\n", r.batch));
    out.push_str(&format!("LLAMA_UBATCH={}\n", r.ubatch));
    out.push_str(&format!("LLAMA_KV_CACHE_K={}\n", r.kv_cache_k));
    out.push_str(&format!("LLAMA_KV_CACHE_V={}\n", r.kv_cache_v));
    out.push_str(&format!(
        "LLAMA_FLASH_ATTN={}\n\n",
        if r.flash_attn { "on" } else { "off" }
    ));

    out.push_str("# Model-specific extras (vision mmproj, MoE flags, etc.)\n");
    out.push_str(&format!("LLAMA_EXTRA_ARGS={}\n\n", r.extra_args));

    // docker compose profile selection. The vello CLI always passes
    // `--profile <mode>` explicitly via docker.rs, but emitting these too
    // means a plain `docker compose up` from the project root still picks
    // the right service (handy for debugging or third-party tooling).
    let runtime = docker::profile_flag(r.mode);
    out.push_str("# Runtime selection (docker compose profile)\n");
    out.push_str(&format!("LLAMA_RUNTIME={runtime}\n"));
    out.push_str(&format!("COMPOSE_PROFILES={runtime}\n\n"));

    out.push_str("# Web UI\n");
    out.push_str(&format!(
        "WEBUI_AUTH={}\n",
        if r.webui_auth { "True" } else { "False" }
    ));

    if !preserved.trim().is_empty() {
        out.push_str("\n# --- preserved (user-added) ---\n");
        out.push_str(preserved);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserve_unmanaged_keeps_unknown_keys() {
        let env = "LLAMA_PORT=8080\nMY_CUSTOM_KEY=hello\nLLAMA_CTX=4096\n";
        let kept = preserve_unmanaged(env);
        assert!(kept.contains("MY_CUSTOM_KEY=hello"));
        assert!(!kept.contains("LLAMA_PORT"));
        assert!(!kept.contains("LLAMA_CTX"));
    }

    #[test]
    fn preserve_unmanaged_drops_comments_and_empty() {
        let env = "# this is a comment\n\n  \nFOO=bar\n";
        let kept = preserve_unmanaged(env);
        assert!(kept.contains("FOO=bar"));
        assert!(!kept.contains("comment"));
    }

    fn sample_resolved(mode: Mode) -> Resolved {
        Resolved {
            llama_port: 8080,
            webui_port: 3000,
            webui_auth: false,
            model_file: "x.gguf".into(),
            model_alias: "x-local".into(),
            ctx: 4096,
            ngl: 99,
            threads: 6,
            batch: 1024,
            ubatch: 256,
            kv_cache_k: "q8_0".into(),
            kv_cache_v: "q8_0".into(),
            flash_attn: true,
            mmproj: String::new(),
            extra_args: String::new(),
            mode,
        }
    }

    #[test]
    fn render_includes_managed_keys() {
        let body = render(&sample_resolved(Mode::Gpu), "");
        assert!(body.contains("LLAMA_MODEL_FILE=x.gguf"));
        assert!(body.contains("LLAMA_CTX=4096"));
        assert!(body.contains("LLAMA_FLASH_ATTN=on"));
        assert!(body.contains("WEBUI_AUTH=False"));
    }

    #[test]
    fn render_emits_cuda_profile_in_gpu_mode() {
        let body = render(&sample_resolved(Mode::Gpu), "");
        assert!(body.contains("LLAMA_RUNTIME=cuda"));
        assert!(body.contains("COMPOSE_PROFILES=cuda"));
    }

    #[test]
    fn render_emits_cpu_profile_in_cpu_mode() {
        let body = render(&sample_resolved(Mode::Cpu), "");
        assert!(body.contains("LLAMA_RUNTIME=cpu"));
        assert!(body.contains("COMPOSE_PROFILES=cpu"));
    }

    #[test]
    fn apply_cpu_overrides_zeroes_ngl_and_flash_attn() {
        let mut r = sample_resolved(Mode::Cpu);
        r.ngl = 99;
        r.flash_attn = true;
        let sys = SystemConfig::default();
        apply_cpu_overrides(&mut r, &sys);
        assert_eq!(r.ngl, 0);
        assert!(!r.flash_attn);
    }

    #[test]
    fn apply_cpu_overrides_caps_ctx_batch_ubatch() {
        let mut r = sample_resolved(Mode::Cpu);
        r.ctx = 32768;
        r.batch = 2048;
        r.ubatch = 512;
        let sys = SystemConfig::default();
        apply_cpu_overrides(&mut r, &sys);
        assert!(r.ctx <= 8192);
        assert!(r.batch <= 512);
        assert!(r.ubatch <= 128);
    }

    #[test]
    fn apply_cpu_overrides_keeps_smaller_explicit_values() {
        // If the user explicitly chose tighter values, don't bump them up.
        let mut r = sample_resolved(Mode::Cpu);
        r.ctx = 4096;
        r.batch = 256;
        r.ubatch = 64;
        let sys = SystemConfig::default();
        apply_cpu_overrides(&mut r, &sys);
        assert_eq!(r.ctx, 4096);
        assert_eq!(r.batch, 256);
        assert_eq!(r.ubatch, 64);
    }
}
