//! Wrappers around `docker compose` calls. All run with cwd set to the
//! project root so they pick up the local docker-compose.yml.
//!
//! The compose file defines two mutually exclusive llama-server profiles —
//! `cuda` and `cpu`. Commands that **start** containers (`up`, `restart`,
//! `build`) take a `Mode` and pass `--profile <mode>` so docker compose only
//! activates the matching service. Read-only / teardown commands (`down`,
//! `status`, `logs`, `nuke`) operate on whatever happens to be running and
//! don't filter by profile.

use crate::paths::Paths;
use crate::schema::Mode;
use anyhow::{bail, Context, Result};
use std::path::Path;
use std::process::{Command, Stdio};

fn compose(paths: &Paths, args: &[&str]) -> Result<()> {
    let status = Command::new("docker")
        .arg("compose")
        .args(args)
        .current_dir(&paths.project_root)
        .status()
        .with_context(|| "invoking docker compose — is Docker installed?")?;
    if !status.success() {
        bail!("docker compose {} failed", args.join(" "));
    }
    Ok(())
}

/// `cuda` or `cpu` — the docker compose profile name for the current mode.
pub fn profile_flag(mode: Mode) -> &'static str {
    match mode {
        Mode::Gpu => "cuda",
        Mode::Cpu => "cpu",
    }
}

pub fn up(paths: &Paths, mode: Mode) -> Result<()> {
    require_active_model(paths)?;
    compose(paths, &["--profile", profile_flag(mode), "up", "-d"])
}

pub fn down(paths: &Paths) -> Result<()> {
    compose(paths, &["down"])
}

pub fn restart(paths: &Paths, mode: Mode) -> Result<()> {
    down(paths)?;
    up(paths, mode)
}

pub fn status(paths: &Paths) -> Result<()> {
    compose(
        paths,
        &["ps", "--format", "table {{.Name}}\t{{.Status}}\t{{.Ports}}"],
    )
}

pub fn logs(paths: &Paths, follow: bool, services: &[String]) -> Result<()> {
    let mut args: Vec<&str> = vec!["logs", "--tail=200"];
    if follow {
        args.push("-f");
    }
    for s in services {
        args.push(s);
    }
    compose(paths, &args)
}

/// Build the llama-server image(s).
///
/// - `all=false`: build only the service for `mode` (typical case — saves disk
///   and time on a host that only uses one runtime).
/// - `all=true`: build both `llama-server-cuda` and `llama-server-cpu` (useful
///   for testing or distributing to mixed hosts). The CUDA build still needs
///   nvidia-container-toolkit on the host; failures are visible at build time.
pub fn build(paths: &Paths, mode: Mode, no_cache: bool, all: bool) -> Result<()> {
    let mut args: Vec<&str> = Vec::new();
    if all {
        args.extend_from_slice(&["--profile", "cuda", "--profile", "cpu"]);
    } else {
        args.extend_from_slice(&["--profile", profile_flag(mode)]);
    }
    args.push("build");
    if no_cache {
        args.push("--no-cache");
    }
    // Without a service argument, compose builds every service in the active
    // profile(s). Both `llama-server-cuda` and `llama-server-cpu` have
    // `build:` blocks, `open-webui` does not, so this naturally builds only
    // the llama-server image(s) we care about.
    compose(paths, &args)
}

pub fn nuke(paths: &Paths) -> Result<()> {
    compose(paths, &["down", "-v", "--rmi", "all", "--remove-orphans"])
}

pub fn is_running(_paths: &Paths) -> bool {
    // Each profile has its own service name now (llama-server-cuda /
    // llama-server-cpu) but both publish container_name=llama-server. The
    // simplest reliable check is to inspect the host container by that
    // shared name via plain docker — `docker compose ps llama-server` no
    // longer works because there's no service with that name.
    //
    // `_paths` is kept on the signature so callers can keep their current
    // call sites unchanged; we don't need the project root for `docker
    // inspect`.
    let out = Command::new("docker")
        .args(["inspect", "-f", "{{.State.Running}}", "llama-server"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim() == "true",
        _ => false,
    }
}

fn require_active_model(paths: &Paths) -> Result<()> {
    let env = paths.project_root.join(".env");
    let raw =
        std::fs::read_to_string(&env).with_context(|| "no .env yet — run: vello install <id>")?;
    let mut file: Option<String> = None;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("LLAMA_MODEL_FILE=") {
            file = Some(rest.trim().trim_matches('"').to_string());
            break;
        }
    }
    let file = file
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow::anyhow!("no active model in .env. Run: vello install <id>"))?;
    let path = paths.models_dir.join(&file);
    if !path.is_file() {
        bail!(
            "model file missing: {}\n  Run: vello install <id>",
            path.display()
        );
    }
    Ok(())
}

/// Stream-download an HF GGUF file using curl. Same UX as the previous
/// bash CLI: progress bar, atomic rename via .partial.
pub fn hf_download(repo: &str, file: &str, dest: &Path) -> Result<()> {
    if dest.is_file() {
        return Ok(());
    }
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}?download=true");
    let partial = dest.with_extension("gguf.partial");
    let status = Command::new("curl")
        .args([
            "-L",
            "--fail",
            "--progress-bar",
            "-H",
            &format!("User-Agent: vello/{}", env!("CARGO_PKG_VERSION")),
            "-o",
        ])
        .arg(&partial)
        .arg(&url)
        .status()
        .with_context(|| "invoking curl — install with: apt install curl")?;
    if !status.success() {
        let _ = std::fs::remove_file(&partial);
        bail!("download failed: {}", url);
    }
    std::fs::rename(&partial, dest)
        .with_context(|| format!("rename {} → {}", partial.display(), dest.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_flag_maps_gpu_to_cuda() {
        assert_eq!(profile_flag(Mode::Gpu), "cuda");
    }

    #[test]
    fn profile_flag_maps_cpu_to_cpu() {
        assert_eq!(profile_flag(Mode::Cpu), "cpu");
    }
}
