//! Filesystem layout. Resolved relative to the project root, which we find
//! by walking up from the binary location until we hit the repo marker
//! (`vello-installer` script). Falls back to env override.

use anyhow::{bail, Result};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Paths {
    pub project_root: PathBuf,
    pub default_catalog: PathBuf,
    pub user_catalog_dir: PathBuf,
    pub models_dir: PathBuf,
    pub profile: PathBuf,
    pub system: PathBuf,
}

pub fn resolve() -> Result<Paths> {
    let root = if let Ok(p) = std::env::var("VELLO_PROJECT_ROOT") {
        PathBuf::from(p)
    } else {
        find_project_root()?
    };
    let catalogs_dir = root.join("catalogs");
    let default_catalog = catalogs_dir.join("default.toml");
    let user_catalog_dir = catalogs_dir.join("user");
    let models_dir = root.join("models");
    let profile = root.join("profile.toml");
    let system = root.join("system.toml");
    Ok(Paths {
        project_root: root,
        default_catalog,
        user_catalog_dir,
        models_dir,
        profile,
        system,
    })
}

fn find_project_root() -> Result<PathBuf> {
    let cwd = std::env::current_dir()?;
    let mut cur: PathBuf = cwd.clone();
    for _ in 0..6 {
        if cur.join("vello-installer").is_file() && cur.join("docker-compose.yml").is_file() {
            return Ok(cur);
        }
        if !cur.pop() {
            break;
        }
    }
    // Also check next to the binary itself, useful when the binary is symlinked
    // from outside the project.
    if let Ok(exe) = std::env::current_exe() {
        let mut p = exe.clone();
        for _ in 0..6 {
            if p.join("vello-installer").is_file() {
                return Ok(p);
            }
            if !p.pop() {
                break;
            }
        }
    }
    bail!(
        "could not locate the project root from {}. \
         Set VELLO_PROJECT_ROOT or run vello from inside the project.",
        cwd.display()
    )
}
