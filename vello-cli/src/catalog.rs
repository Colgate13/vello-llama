//! Multi-source catalog loader and validation.

use crate::schema::{Catalog, Model, SUPPORTED_SCHEMA_VERSION};
use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// One entry in the merged catalog.
#[derive(Debug, Clone)]
pub struct CatalogEntry {
    pub model: Model,
}

pub struct LoadedCatalogs {
    pub entries: Vec<CatalogEntry>,
    pub sources: Vec<CatalogSource>,
}

#[derive(Debug, Clone)]
pub struct CatalogSource {
    pub name: String,
    pub path: PathBuf,
    pub maintainer: String,
    pub model_count: usize,
}

/// Load the default catalog plus every TOML file under `user_dir`.
/// Conflicts on `id`: the default wins; among user catalogs, first loaded wins.
pub fn load_all(default_path: &Path, user_dir: &Path) -> Result<LoadedCatalogs> {
    let mut entries: Vec<CatalogEntry> = Vec::new();
    let mut sources: Vec<CatalogSource> = Vec::new();
    let mut seen_ids: HashMap<String, String> = HashMap::new();

    if default_path.exists() {
        let cat = load_one(default_path)?;
        absorb(
            &cat,
            default_path,
            &mut entries,
            &mut sources,
            &mut seen_ids,
        );
    }

    if user_dir.exists() {
        let mut paths: Vec<PathBuf> = fs::read_dir(user_dir)
            .with_context(|| format!("reading {}", user_dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "toml"))
            .collect();
        paths.sort();
        for path in paths {
            match load_one(&path) {
                Ok(cat) => absorb(&cat, &path, &mut entries, &mut sources, &mut seen_ids),
                Err(e) => eprintln!("warn: failed to load {}: {e:#}", path.display()),
            }
        }
    }

    Ok(LoadedCatalogs { entries, sources })
}

pub fn load_one(path: &Path) -> Result<Catalog> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let cat: Catalog =
        toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    validate(&cat).with_context(|| format!("validating {}", path.display()))?;
    Ok(cat)
}

fn absorb(
    cat: &Catalog,
    path: &Path,
    entries: &mut Vec<CatalogEntry>,
    sources: &mut Vec<CatalogSource>,
    seen_ids: &mut HashMap<String, String>,
) {
    let mut count = 0;
    for m in &cat.models {
        if let Some(existing) = seen_ids.get(&m.id) {
            eprintln!(
                "warn: duplicate model id '{}' in '{}' (already from '{}'), skipped",
                m.id, cat.name, existing
            );
            continue;
        }
        seen_ids.insert(m.id.clone(), cat.name.clone());
        entries.push(CatalogEntry { model: m.clone() });
        count += 1;
    }
    sources.push(CatalogSource {
        name: cat.name.clone(),
        path: path.to_path_buf(),
        maintainer: cat.maintainer.clone(),
        model_count: count,
    });
}

pub fn validate(cat: &Catalog) -> Result<()> {
    if cat.schema_version != SUPPORTED_SCHEMA_VERSION {
        bail!(
            "schema_version {} not supported (this vello supports {})",
            cat.schema_version,
            SUPPORTED_SCHEMA_VERSION
        );
    }
    if cat.name.trim().is_empty() {
        bail!("catalog name is empty");
    }
    let mut ids: HashMap<&str, ()> = HashMap::new();
    for m in &cat.models {
        if m.id.trim().is_empty() {
            bail!("a model has empty id");
        }
        if ids.insert(m.id.as_str(), ()).is_some() {
            bail!("duplicate id within catalog: {}", m.id);
        }
        if m.files.is_empty() {
            bail!("model '{}': files map is empty", m.id);
        }
        if !m.files.contains_key(&m.default_quant) {
            bail!(
                "model '{}': default_quant '{}' not present in files",
                m.id,
                m.default_quant
            );
        }
        if m.params_total_b <= 0.0 {
            bail!("model '{}': params_total_b must be > 0", m.id);
        }
        if matches!(m.architecture, crate::schema::Architecture::Moe) && m.params_active_b.is_none()
        {
            bail!("model '{}': MoE requires params_active_b", m.id);
        }
    }
    Ok(())
}

pub fn find<'a>(entries: &'a [CatalogEntry], id: &str) -> Result<&'a CatalogEntry> {
    entries
        .iter()
        .find(|e| e.model.id == id)
        .ok_or_else(|| anyhow!("model '{}' not in any catalog", id))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(toml_str: &str) -> Result<Catalog> {
        let cat: Catalog = toml::from_str(toml_str)?;
        validate(&cat)?;
        Ok(cat)
    }

    #[test]
    fn minimal_dense_model_validates() {
        let toml_str = r#"
            schema_version = 1
            name = "test"
            [[model]]
            id = "x"
            repo = "u/r"
            default_quant = "Q4_K_M"
            params_total_b = 7.0
            architecture = "dense"
            [model.files]
            Q4_K_M = "x-Q4.gguf"
        "#;
        assert!(parse(toml_str).is_ok());
    }

    #[test]
    fn moe_without_active_params_is_rejected() {
        let toml_str = r#"
            schema_version = 1
            name = "test"
            [[model]]
            id = "x"
            repo = "u/r"
            default_quant = "Q4_K_M"
            params_total_b = 30.0
            architecture = "moe"
            [model.files]
            Q4_K_M = "x.gguf"
        "#;
        let err = parse(toml_str).unwrap_err().to_string();
        assert!(err.contains("MoE requires params_active_b"), "got: {err}");
    }

    #[test]
    fn default_quant_must_exist_in_files() {
        let toml_str = r#"
            schema_version = 1
            name = "test"
            [[model]]
            id = "x"
            repo = "u/r"
            default_quant = "Q9_NOPE"
            params_total_b = 7.0
            architecture = "dense"
            [model.files]
            Q4_K_M = "x.gguf"
        "#;
        let err = parse(toml_str).unwrap_err().to_string();
        assert!(err.contains("not present in files"), "got: {err}");
    }

    #[test]
    fn duplicate_ids_within_catalog_rejected() {
        let toml_str = r#"
            schema_version = 1
            name = "test"
            [[model]]
            id = "dup"
            repo = "a/b"
            default_quant = "Q4_K_M"
            params_total_b = 7.0
            architecture = "dense"
            [model.files]
            Q4_K_M = "1.gguf"

            [[model]]
            id = "dup"
            repo = "c/d"
            default_quant = "Q4_K_M"
            params_total_b = 8.0
            architecture = "dense"
            [model.files]
            Q4_K_M = "2.gguf"
        "#;
        let err = parse(toml_str).unwrap_err().to_string();
        assert!(err.contains("duplicate id"), "got: {err}");
    }

    #[test]
    fn unsupported_schema_version_rejected() {
        let toml_str = r#"
            schema_version = 999
            name = "future"
        "#;
        let err = parse(toml_str).unwrap_err().to_string();
        assert!(err.contains("schema_version"), "got: {err}");
    }

    #[test]
    fn default_catalog_parses() {
        // Sanity: the shipped default.toml must always validate. Path is
        // resolved relative to the crate root (cargo test sets CWD there).
        let path = std::path::Path::new("../catalogs/default.toml");
        if !path.exists() {
            // When running from the repo root the path differs.
            return;
        }
        let raw = std::fs::read_to_string(path).expect("read catalog");
        let cat: Catalog = toml::from_str(&raw).expect("parse default catalog");
        validate(&cat).expect("validate default catalog");
        assert!(
            cat.models.len() >= 20,
            "catalog has {} models",
            cat.models.len()
        );
    }
}
