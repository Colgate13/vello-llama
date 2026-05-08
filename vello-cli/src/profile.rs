//! Hardware profile detection and persistence.

use crate::schema::Profile;
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use std::process::Command;

pub fn load_or_detect(path: &Path) -> Result<Profile> {
    if path.exists() {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("reading profile {}", path.display()))?;
        let p: Profile = toml::from_str(&raw).context("parsing profile.toml")?;
        return Ok(p);
    }
    let p = detect()?;
    save(path, &p)?;
    Ok(p)
}

pub fn save(path: &Path, profile: &Profile) -> Result<()> {
    let body = toml::to_string_pretty(profile).context("serializing profile")?;
    let mut header = String::from("# Auto-detected hardware profile. Edit manually to override.\n");
    header.push_str("# vello uses these numbers to pick quantization and rank models.\n\n");
    header.push_str(&body);
    fs::write(path, header).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

pub fn detect() -> Result<Profile> {
    let mut p = Profile::default();
    if let Some((vram, name, arch)) = detect_gpu() {
        p.vram_gb = vram;
        p.gpu_name = name;
        p.cuda_arch = arch;
    }
    if let Some(ram) = detect_ram() {
        p.ram_gb = ram;
    }
    Ok(p)
}

/// Returns (vram_gb, name, cuda_arch).
fn detect_gpu() -> Option<(f32, String, Option<u32>)> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,name,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8(out.stdout).ok()?;
    let first = line.lines().next()?;
    let parts: Vec<&str> = first.split(',').map(str::trim).collect();
    if parts.len() < 3 {
        return None;
    }
    let mib: f32 = parts[0].parse().ok()?;
    let name = parts[1].to_string();
    // compute_cap comes as "8.9" → 89
    let arch = parts[2].replace('.', "").parse::<u32>().ok();
    Some((mib / 1024.0, name, arch))
}

fn detect_ram() -> Option<f32> {
    let raw = fs::read_to_string("/proc/meminfo").ok()?;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: f32 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb / 1024.0 / 1024.0);
        }
    }
    None
}
