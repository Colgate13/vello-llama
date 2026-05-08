//! Wrappers around `docker compose` calls. All run with cwd set to the
//! project root so they pick up the local docker-compose.yml.

use crate::paths::Paths;
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

fn compose_capture(paths: &Paths, args: &[&str]) -> Result<String> {
    let out = Command::new("docker")
        .arg("compose")
        .args(args)
        .current_dir(&paths.project_root)
        .stderr(Stdio::inherit())
        .output()
        .with_context(|| "invoking docker compose")?;
    if !out.status.success() {
        bail!("docker compose {} failed", args.join(" "));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

pub fn up(paths: &Paths) -> Result<()> {
    require_active_model(paths)?;
    compose(paths, &["up", "-d"])
}

pub fn down(paths: &Paths) -> Result<()> {
    compose(paths, &["down"])
}

pub fn restart(paths: &Paths) -> Result<()> {
    down(paths)?;
    up(paths)
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

pub fn build(paths: &Paths, no_cache: bool) -> Result<()> {
    let mut args: Vec<&str> = vec!["build"];
    if no_cache {
        args.push("--no-cache");
    }
    args.push("llama-server");
    compose(paths, &args)
}

pub fn nuke(paths: &Paths) -> Result<()> {
    compose(paths, &["down", "-v", "--rmi", "all", "--remove-orphans"])
}

pub fn is_running(paths: &Paths) -> bool {
    compose_capture(paths, &["ps", "llama-server", "--format", "{{.Status}}"])
        .map(|s| s.contains("Up"))
        .unwrap_or(false)
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
