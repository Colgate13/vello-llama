//! All CLI subcommand handlers. Output is plain ANSI for terminals, plain
//! ASCII when NO_COLOR is set or stdout isn't a tty.

use crate::catalog;
use crate::catalog::CatalogEntry;
use crate::docker;
use crate::fit::{self, Strategy};
use crate::paths::Paths;
use crate::recommend;
use crate::resolver::{self, ResolveInput};
use crate::schema::{Architecture, Mode, Profile, Tier};
use crate::style::Style;
use crate::system;
use anyhow::{anyhow, bail, Context, Result};
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;

fn mode_label(m: Mode) -> &'static str {
    match m {
        Mode::Gpu => "gpu",
        Mode::Cpu => "cpu",
    }
}

fn other_mode_flag(m: Mode) -> &'static str {
    match m {
        Mode::Gpu => "cpu",
        Mode::Cpu => "gpu",
    }
}

// ---------------------------------------------------------------------------
// Status of a model on disk + active flag
// ---------------------------------------------------------------------------

fn active_filename(paths: &Paths) -> Option<String> {
    let env = paths.project_root.join(".env");
    let raw = fs::read_to_string(env).ok()?;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("LLAMA_MODEL_FILE=") {
            return Some(rest.trim().trim_matches('"').to_string());
        }
    }
    None
}

fn status_for(model_file: &str, paths: &Paths, active: Option<&str>) -> ModelStatus {
    let on_disk = paths.models_dir.join(model_file).is_file();
    let is_active = active.map(|a| a == model_file).unwrap_or(false);
    if is_active {
        ModelStatus::Active
    } else if on_disk {
        ModelStatus::Downloaded
    } else {
        ModelStatus::NotInstalled
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelStatus {
    Active,
    Downloaded,
    NotInstalled,
}

impl ModelStatus {
    fn render(&self, st: &Style) -> String {
        match self {
            ModelStatus::Active => st.green("● active"),
            ModelStatus::Downloaded => st.cyan("○ on disk"),
            ModelStatus::NotInstalled => st.dim("· -"),
        }
    }
}

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------

pub struct ListFilters {
    pub tag: Option<String>,
    pub tier: Option<String>,
    pub modality: Option<String>,
    pub installed: bool,
    /// Show models regardless of whether they declare the active mode in
    /// `targets`. Without --all, models that don't support the current mode
    /// (e.g. dense 14B in CPU mode) are hidden.
    pub all: bool,
}

pub fn cmd_list(paths: &Paths, profile: &Profile, filters: &ListFilters) -> Result<()> {
    let st = Style::new();
    let loaded = catalog::load_all(&paths.default_catalog, &paths.user_catalog_dir)?;
    if loaded.entries.is_empty() {
        println!(
            "{}",
            st.yellow(&format!(
                "No catalogs loaded. Expected default at {}",
                paths.default_catalog.display()
            ))
        );
        return Ok(());
    }

    let active = active_filename(paths);

    let mode_str = match profile.mode {
        Mode::Gpu => st.green("gpu"),
        Mode::Cpu => st.cyan("cpu"),
    };
    println!(
        "{} {}    {} {} GB VRAM, {} GB RAM    {} {}",
        st.bold("Profile:"),
        st.cyan(&profile.gpu_name),
        st.dim("→"),
        round1(profile.vram_gb),
        round1(profile.ram_gb),
        st.bold("Mode:"),
        mode_str,
    );
    if profile.mode == Mode::Cpu && !filters.all {
        println!(
            "  {} (use --all to also show gpu-only entries)",
            st.dim("CPU mode: showing models tagged for CPU"),
        );
    }
    println!();

    println!(
        "{:6} {:30} {:6} {:8} {:8} {}",
        st.bold("TIER"),
        st.bold("ID"),
        st.bold("PARAMS"),
        st.bold("QUANT"),
        st.bold("STATUS"),
        st.bold("TAGS"),
    );

    let tier_filter = filters.tier.as_deref().map(|s| s.to_uppercase());
    let mode_target = profile.mode.as_target();

    for e in &loaded.entries {
        // Hide models the curator didn't tag for the current runtime mode
        // unless the user opted in via --all.
        if !filters.all && !e.model.supports(mode_target) {
            continue;
        }
        if let Some(tag) = &filters.tag {
            if !e.model.tags.iter().any(|t| t == tag) {
                continue;
            }
        }
        if let Some(modality) = &filters.modality {
            if !e.model.modalities.iter().any(|m| m == modality) {
                continue;
            }
        }
        let p = fit::pick(&e.model, profile, Strategy::Balanced);
        let tier = p
            .as_ref()
            .map(|x| x.tier)
            .unwrap_or_else(|| fit::forced_tier(&e.model, profile));
        if let Some(tf) = &tier_filter {
            if tier.label() != tf.as_str() {
                continue;
            }
        }
        let file = p.as_ref().map(|x| x.file.as_str()).unwrap_or_else(|| {
            e.model
                .files
                .get(&e.model.default_quant)
                .map(String::as_str)
                .unwrap_or("?")
        });
        let status = status_for(file, paths, active.as_deref());
        if filters.installed && status == ModelStatus::NotInstalled {
            continue;
        }

        let params = format_params(&e.model);
        let quant = p
            .as_ref()
            .map(|x| x.quant.as_str())
            .unwrap_or(&e.model.default_quant);
        let exp = if e.model.experimental {
            st.yellow(" exp")
        } else {
            String::new()
        };
        let tags = if e.model.tags.is_empty() {
            String::new()
        } else {
            st.dim(&e.model.tags.join(","))
        };
        println!(
            "{:6} {:30} {:6} {:8} {:17} {}{}",
            st.tier(tier),
            e.model.id,
            params,
            quant,
            status.render(&st),
            tags,
            exp,
        );
    }

    println!();
    let legend = match profile.mode {
        Mode::Gpu => {
            "S=fits VRAM  A=light overflow  B=MoE/offload  C=slow  D=won't fit"
        }
        Mode::Cpu => {
            "S=small dense or MoE, fast  A=small but tight  B=7-8B, usable  C=≤14B, slow  D=won't fit / gpu-only"
        }
    };
    println!("{}  {}", st.dim("Tier legend:"), st.dim(legend));
    Ok(())
}

fn format_params(m: &crate::schema::Model) -> String {
    if let Some(active) = m.params_active_b {
        format!("{}/{}B", trim_b(active), trim_b(m.params_total_b))
    } else {
        format!("{}B", trim_b(m.params_total_b))
    }
}

fn trim_b(v: f32) -> String {
    if (v - v.round()).abs() < 0.05 {
        format!("{:.0}", v)
    } else {
        format!("{:.1}", v)
    }
}

fn round1(v: f32) -> f32 {
    (v * 10.0).round() / 10.0
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

pub fn cmd_info(paths: &Paths, profile: &Profile, id: &str, quant: Option<&str>) -> Result<()> {
    let st = Style::new();
    let loaded = catalog::load_all(&paths.default_catalog, &paths.user_catalog_dir)?;
    let entry = catalog::find(&loaded.entries, id)?;
    let m = &entry.model;
    let p = match quant {
        Some(q) => fit::pick_explicit(m, profile, q),
        None => fit::pick(m, profile, Strategy::Balanced),
    };
    if let (Some(q), None) = (quant, &p) {
        bail!(
            "model '{}' has no quant '{}'. Available: {}",
            id,
            q,
            m.files.keys().cloned().collect::<Vec<_>>().join(", ")
        );
    }

    println!("{}", st.bold(&m.id));
    if !m.description.is_empty() {
        println!("{}", m.description.trim());
    }
    println!();

    let arch = match m.architecture {
        Architecture::Dense => format!("dense {}B", trim_b(m.params_total_b)),
        Architecture::Moe => format!(
            "MoE {}B total / {}B active",
            trim_b(m.params_total_b),
            trim_b(m.params_active_b.unwrap_or(0.0))
        ),
    };
    println!("  {:14} {}", st.dim("arch:"), arch);
    println!("  {:14} {}", st.dim("modalities:"), m.modalities.join(", "));
    if !m.tags.is_empty() {
        println!("  {:14} {}", st.dim("tags:"), m.tags.join(", "));
    }
    if let Some(ctx) = m.runtime.ctx_default {
        println!("  {:14} {}", st.dim("ctx:"), ctx);
    }
    if let Some(ngl) = m.runtime.ngl {
        if ngl != 99 {
            println!("  {:14} {}", st.dim("ngl:"), ngl);
        }
    }
    if !m.runtime.extra_args.is_empty() {
        println!(
            "  {:14} {}",
            st.dim("extra:"),
            m.runtime.extra_args.join(" ")
        );
    }
    if let Some(mmproj) = &m.runtime.mmproj {
        println!("  {:14} {}", st.dim("mmproj:"), mmproj);
    }
    println!("  {:14} {}", st.dim("repo:"), m.repo);
    println!(
        "  {:14} {}",
        st.dim("quants:"),
        m.files.keys().cloned().collect::<Vec<_>>().join(", ")
    );
    if m.experimental {
        println!("  {:14} {}", st.dim("status:"), st.yellow("experimental"));
    }
    println!();

    match p {
        Some(p) => {
            println!("{}", st.bold("For your hardware:"));
            println!("  {:18} {}", st.dim("Tier:"), st.tier(p.tier));
            println!("  {:18} {}", st.dim("Auto-pick quant:"), p.quant);
            println!(
                "  {:18} ~{} GB",
                st.dim("Estimated size:"),
                round1(p.size_gb)
            );
            println!(
                "  {:18} {} GB VRAM + {} GB RAM",
                st.dim("Memory need:"),
                round1(p.vram_need_gb),
                round1(p.ram_need_gb)
            );
            println!("  {:18} {}", st.dim("File:"), p.file);
            println!();
            let active = active_filename(paths);
            let status = status_for(&p.file, paths, active.as_deref());
            println!("  {:18} {}", st.dim("Local status:"), status.render(&st));
            println!();
            println!(
                "  Install:  {}",
                st.cyan(&format!("vello install {}", m.id))
            );
        }
        None => {
            println!(
                "{}",
                st.red("This model won't fit on your hardware (Tier D)")
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// recommend
// ---------------------------------------------------------------------------

pub fn cmd_recommend(paths: &Paths, profile: &Profile, query: &str, limit: usize) -> Result<()> {
    let st = Style::new();
    let loaded = catalog::load_all(&paths.default_catalog, &paths.user_catalog_dir)?;
    let recs = recommend::rank(&loaded.entries, profile, query, limit);
    if recs.is_empty() {
        println!("No models matched '{query}' for your hardware.");
        println!(
            "{}",
            st.dim("Tip: try `vello list` to see what fits, or use simpler keywords (chat, code, vision).")
        );
        return Ok(());
    }
    println!(
        "{} {}",
        st.bold("Recommendations for"),
        st.cyan(&format!("\"{query}\""))
    );
    println!();
    for (i, r) in recs.iter().enumerate() {
        let m = &r.entry.model;
        let exp = if m.experimental {
            st.yellow(" exp")
        } else {
            String::new()
        };
        println!(
            "  {} {} {}  {}{}",
            st.bold(&format!("{}.", i + 1)),
            st.tier(r.pick.tier),
            st.bold(&m.id),
            format_params(m),
            exp,
        );
        if !m.description.is_empty() {
            println!(
                "     {}",
                st.dim(m.description.trim().lines().next().unwrap_or(""))
            );
        }
        println!(
            "     quant {} · ~{} GB · {} VRAM + {} RAM",
            r.pick.quant,
            round1(r.pick.size_gb),
            round1(r.pick.vram_need_gb),
            round1(r.pick.ram_need_gb)
        );
        println!("     {}", st.cyan(&format!("vello install {}", m.id)));
        println!();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// install / switch / remove / active
// ---------------------------------------------------------------------------

pub fn cmd_install(
    paths: &Paths,
    profile: &Profile,
    id: &str,
    no_switch: bool,
    quant: Option<&str>,
) -> Result<()> {
    let st = Style::new();
    let loaded = catalog::load_all(&paths.default_catalog, &paths.user_catalog_dir)?;
    let entry = catalog::find(&loaded.entries, id)?.clone();
    let m = &entry.model;
    let mode_target = profile.mode.as_target();
    if !m.supports(mode_target) {
        bail!(
            "'{}' isn't tagged for {} mode (targets = {:?}). Either pick another \
             model from `vello list` or switch modes with `vello doctor --{}`.",
            id,
            mode_label(profile.mode),
            m.targets,
            other_mode_flag(profile.mode),
        );
    }
    let p = match quant {
        Some(q) => fit::pick_explicit(m, profile, q).ok_or_else(|| {
            anyhow!(
                "model '{}' has no quant '{}'. Available: {}",
                id,
                q,
                m.files.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        })?,
        None => fit::pick(m, profile, Strategy::Balanced)
            .ok_or_else(|| anyhow!("'{}' won't fit on your hardware", id))?,
    };

    let chosen_label = if quant.is_some() {
        format!("{} (explicit)", p.quant)
    } else {
        format!("{} (auto-pick)", p.quant)
    };
    println!(
        "{} {} [{}] ~{} GB",
        st.bold("Installing"),
        st.cyan(&m.id),
        chosen_label,
        round1(p.size_gb)
    );
    if matches!(p.tier, Tier::C | Tier::D) {
        println!(
            "{}",
            st.yellow(&format!(
                "  warn: tier {} — this quant fits poorly on your hardware",
                p.tier.label()
            ))
        );
    }
    if m.experimental {
        println!(
            "{}",
            st.yellow("  warn: experimental — llama.cpp support may be incomplete or buggy")
        );
    }

    fs::create_dir_all(&paths.models_dir)?;
    let dest = paths.models_dir.join(&p.file);
    if dest.is_file() {
        println!("  {} already on disk: {}", st.dim("·"), p.file);
    } else {
        println!("  {} pulling {}", st.dim("→"), p.file);
        docker::hf_download(&m.repo, &p.file, &dest)?;
    }

    // Multimodal: pull mmproj from the same repo if declared.
    if let Some(mmproj) = &m.runtime.mmproj {
        let mmproj_dest = paths.models_dir.join(mmproj);
        if !mmproj_dest.is_file() {
            println!("  {} pulling mmproj for vision: {}", st.dim("→"), mmproj);
            docker::hf_download(&m.repo, mmproj, &mmproj_dest)?;
        }
    }

    if no_switch {
        return Ok(());
    }

    apply_model(paths, profile, &entry, &p.file, true)
}

pub fn cmd_switch(paths: &Paths, profile: &Profile, id: &str, quant: Option<&str>) -> Result<()> {
    let loaded = catalog::load_all(&paths.default_catalog, &paths.user_catalog_dir)?;
    let entry = catalog::find(&loaded.entries, id)?.clone();
    let p = match quant {
        Some(q) => fit::pick_explicit(&entry.model, profile, q).ok_or_else(|| {
            anyhow!(
                "model '{}' has no quant '{}'. Available: {}",
                id,
                q,
                entry
                    .model
                    .files
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?,
        None => fit::pick(&entry.model, profile, Strategy::Balanced)
            .ok_or_else(|| anyhow!("'{}' won't fit on your hardware", id))?,
    };
    if !paths.models_dir.join(&p.file).is_file() {
        let install_hint = match quant {
            Some(q) => format!("vello install {} --quant {}", id, q),
            None => format!("vello install {}", id),
        };
        bail!(
            "'{}' ({}) is not downloaded. Run: {}",
            id,
            p.quant,
            install_hint
        );
    }
    apply_model(paths, profile, &entry, &p.file, true)
}

pub fn cmd_apply(paths: &Paths, profile: &Profile, restart: bool) -> Result<()> {
    let st = Style::new();
    let active = active_filename(paths)
        .ok_or_else(|| anyhow!("no active model in .env. Use `vello switch <id>` first"))?;
    let loaded = catalog::load_all(&paths.default_catalog, &paths.user_catalog_dir)?;
    let entry = match find_by_file(&loaded.entries, &active) {
        Some(e) => e,
        None => {
            // Active file not in any catalog — apply system+profile defaults
            // with a synthetic minimal model.
            println!(
                "{}",
                st.yellow(&format!(
                    "warn: active file '{}' not in any catalog — applying system defaults only",
                    active
                ))
            );
            return apply_unknown(paths, profile, &active, restart);
        }
    };
    apply_model(paths, profile, &entry, &active, restart)
}

fn apply_model(
    paths: &Paths,
    profile: &Profile,
    entry: &CatalogEntry,
    model_file: &str,
    restart: bool,
) -> Result<()> {
    let st = Style::new();
    let sys = system::load_or_default(&paths.system)?;
    let resolved = resolver::resolve(&ResolveInput {
        model: &entry.model,
        model_file,
        system: &sys,
        profile,
    });
    resolver::write_env(paths, &resolved)?;
    println!(
        "{} {} ({} ctx, ngl {}{})",
        st.green("applied"),
        st.bold(&entry.model.id),
        resolved.ctx,
        resolved.ngl,
        if !resolved.extra_args.is_empty() {
            format!(", extras: {}", resolved.extra_args)
        } else {
            String::new()
        },
    );
    if !resolved.mmproj.is_empty() {
        println!("  mmproj: {}", resolved.mmproj);
    }
    if restart && docker::is_running(paths) {
        println!("  {}", st.dim("restarting stack..."));
        docker::restart(paths, profile.mode)?;
    } else if restart {
        println!(
            "  {}",
            st.dim("stack not running — bring it up with: vello up")
        );
    }
    Ok(())
}

fn apply_unknown(paths: &Paths, profile: &Profile, file: &str, restart: bool) -> Result<()> {
    use crate::schema::{Architecture, Model, Target};
    use std::collections::BTreeMap;
    let mut files = BTreeMap::new();
    files.insert("?".to_string(), file.to_string());
    let synthetic = Model {
        id: file.trim_end_matches(".gguf").to_string(),
        repo: String::new(),
        default_quant: "?".into(),
        files,
        params_total_b: 0.0,
        params_active_b: None,
        architecture: Architecture::Dense,
        modalities: vec!["text".into()],
        // Unknown user-dropped GGUF: assume permissive — works wherever the
        // user is currently set up. The active mode is included explicitly.
        targets: vec![Target::Gpu, Target::Cpu],
        tags: Vec::new(),
        experimental: false,
        description: String::new(),
        runtime: Default::default(),
    };
    let entry = CatalogEntry { model: synthetic };
    apply_model(paths, profile, &entry, file, restart)
}

fn find_by_file(entries: &[CatalogEntry], file: &str) -> Option<CatalogEntry> {
    for e in entries {
        if e.model.files.values().any(|f| f == file) {
            return Some(e.clone());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// nuke (interactive confirmation)
// ---------------------------------------------------------------------------

pub fn cmd_nuke(paths: &Paths) -> Result<()> {
    let st = Style::new();
    println!(
        "{}",
        st.yellow("This will remove containers, volumes, and the llama-server image.")
    );
    println!(
        "{}",
        st.yellow("Models in ./models/ are kept. Press Ctrl-C to abort.")
    );
    if io::stdin().is_terminal() {
        print!("Continue? [y/N] ");
        io::stdout().flush()?;
        let mut buf = String::new();
        io::stdin().read_line(&mut buf)?;
        if !matches!(buf.trim().to_lowercase().as_str(), "y" | "yes") {
            println!("aborted");
            return Ok(());
        }
    }
    docker::nuke(paths)?;
    println!("{} stack removed", st.green("ok"));
    Ok(())
}

pub fn cmd_remove(paths: &Paths, profile: &Profile, id: &str) -> Result<()> {
    let st = Style::new();
    let loaded = catalog::load_all(&paths.default_catalog, &paths.user_catalog_dir)?;
    let entry = catalog::find(&loaded.entries, id)?;
    let p = fit::pick(&entry.model, profile, Strategy::Balanced)
        .ok_or_else(|| anyhow!("'{}' has no installable quant", id))?;
    let target = paths.models_dir.join(&p.file);
    if !target.is_file() {
        bail!("not on disk: {}", target.display());
    }
    if let Some(active) = active_filename(paths) {
        if active == p.file {
            bail!(
                "'{}' is the active model. Switch first: vello switch <other>",
                id
            );
        }
    }
    fs::remove_file(&target).with_context(|| format!("removing {}", target.display()))?;
    println!("{} removed {}", st.green("ok"), p.file);
    Ok(())
}

pub fn cmd_active(paths: &Paths) -> Result<()> {
    let st = Style::new();
    match active_filename(paths) {
        Some(f) => {
            println!("{} {}", st.bold("Active model file:"), f);
            let p = paths.models_dir.join(&f);
            if let Ok(meta) = fs::metadata(&p) {
                println!("  size: {} GB", round1(meta.len() as f32 / 1e9));
            }
        }
        None => println!("No active model set."),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// catalog management
// ---------------------------------------------------------------------------

pub fn cmd_catalog_list(paths: &Paths) -> Result<()> {
    let st = Style::new();
    let loaded = catalog::load_all(&paths.default_catalog, &paths.user_catalog_dir)?;
    println!("{}", st.bold("Loaded catalogs:"));
    for s in &loaded.sources {
        println!(
            "  · {} {} ({} models)  {}",
            st.cyan(&s.name),
            if s.maintainer.is_empty() {
                String::new()
            } else {
                format!("by {}", s.maintainer)
            },
            s.model_count,
            st.dim(&s.path.display().to_string()),
        );
    }
    Ok(())
}

pub fn cmd_catalog_add(paths: &Paths, source: &str) -> Result<()> {
    let st = Style::new();
    let p = PathBuf::from(source);
    if !p.exists() {
        bail!(
            "remote catalogs (URLs) are not supported yet — copy the .toml locally and run again with the path"
        );
    }
    if !p.is_file() {
        bail!("not a file: {}", p.display());
    }
    let cat = catalog::load_one(&p).with_context(|| format!("validating {}", p.display()))?;
    fs::create_dir_all(&paths.user_catalog_dir)?;
    let target = paths
        .user_catalog_dir
        .join(format!("{}.toml", sanitize(&cat.name)));
    if target.exists() {
        bail!(
            "catalog '{}' already installed at {}. Remove first.",
            cat.name,
            target.display()
        );
    }
    fs::copy(&p, &target)?;
    println!(
        "{} {} ({} models) → {}",
        st.green("added"),
        cat.name,
        cat.models.len(),
        target.display()
    );
    Ok(())
}

pub fn cmd_catalog_remove(paths: &Paths, name: &str) -> Result<()> {
    let st = Style::new();
    let target = paths
        .user_catalog_dir
        .join(format!("{}.toml", sanitize(name)));
    if !target.exists() {
        bail!("no user catalog named '{}'", name);
    }
    fs::remove_file(&target)?;
    println!("{} {}", st.green("removed"), target.display());
    Ok(())
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// profile
// ---------------------------------------------------------------------------

pub fn cmd_profile_show(profile: &Profile) -> Result<()> {
    let st = Style::new();
    println!("{}", st.bold("Hardware profile"));
    println!("  GPU:           {}", profile.gpu_name);
    if let Some(arch) = profile.cuda_arch {
        println!("  CUDA arch:     {}", arch);
    }
    println!("  VRAM:          {} GB", round1(profile.vram_gb));
    println!("  RAM:           {} GB", round1(profile.ram_gb));
    println!("  VRAM reserve:  {} GB", round1(profile.vram_reserve_gb));
    println!("  RAM reserve:   {} GB", round1(profile.ram_reserve_gb));
    println!("  Default ctx:   {}", profile.default_ctx);
    Ok(())
}

pub fn cmd_profile_refresh(paths: &Paths) -> Result<()> {
    let p = crate::profile::detect()?;
    crate::profile::save(&paths.profile, &p)?;
    cmd_profile_show(&p)?;
    Ok(())
}

pub fn cmd_update(paths: &Paths, force: bool) -> Result<()> {
    use std::process::{Command, Stdio};
    let st = Style::new();
    let root = &paths.project_root;

    if !root.join(".git").exists() {
        bail!(
            "{} is not a git checkout — `vello update` only works on a git clone",
            root.display()
        );
    }

    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(root)
        .output()
        .with_context(|| "running `git status` — is git installed?")?;
    if !dirty.status.success() {
        bail!("git status failed");
    }
    let dirty_out = String::from_utf8_lossy(&dirty.stdout);
    if !dirty_out.trim().is_empty() && !force {
        eprintln!("{}", st.yellow("uncommitted changes in working tree:"));
        for line in dirty_out.lines().take(10) {
            eprintln!("  {}", line);
        }
        bail!("commit/stash your changes first, or re-run with --force to attempt the pull anyway");
    }

    let head_before = git_head(root)?;
    println!("{} pulling latest...", st.bold("→"));
    let pull = Command::new("git")
        .args(["pull", "--ff-only"])
        .current_dir(root)
        .status()
        .with_context(|| "running `git pull`")?;
    if !pull.success() {
        bail!("git pull failed — resolve manually then re-run `vello update`");
    }
    let head_after = git_head(root)?;

    if head_before == head_after {
        println!(
            "{} already up to date ({})",
            st.green("✓"),
            short_sha(&head_after)
        );
        return Ok(());
    }

    let changed = Command::new("git")
        .args(["diff", "--name-only", &head_before, &head_after])
        .current_dir(root)
        .output()
        .with_context(|| "running `git diff`")?;
    let changed_files: Vec<&str> = std::str::from_utf8(&changed.stdout)
        .unwrap_or("")
        .lines()
        .collect();

    let cli_changed = changed_files
        .iter()
        .any(|f| f.starts_with("vello-cli/") || *f == "Cargo.toml" || *f == "Cargo.lock");
    let docker_changed = changed_files
        .iter()
        .any(|f| f.starts_with("docker/") || *f == "docker-compose.yml");
    let catalog_changed = changed_files
        .iter()
        .any(|f| f.starts_with("catalogs/default"));

    println!(
        "{} {} → {} ({} files changed)",
        st.green("pulled"),
        short_sha(&head_before),
        short_sha(&head_after),
        changed_files.len()
    );

    if cli_changed {
        println!("{} rebuilding vello binary...", st.bold("→"));
        let cargo = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--manifest-path",
                "vello-cli/Cargo.toml",
            ])
            .current_dir(root)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .with_context(|| "running `cargo build` — is the Rust toolchain installed?")?;
        if !cargo.success() {
            bail!("cargo build failed — fix errors above and re-run `vello update`");
        }
        println!("{} vello rebuilt", st.green("✓"));
    } else {
        println!("  {} no Rust changes — binary not rebuilt", st.dim("·"));
    }

    println!();
    println!("{}", st.bold("next steps:"));
    if docker_changed {
        println!(
            "  {} docker/ or compose changed — rebuild image: {}",
            st.yellow("!"),
            st.cyan("./vello build")
        );
        println!("    then restart: {}", st.cyan("./vello restart"));
    } else {
        println!(
            "  {} restart only if you want to pick up new defaults: {}",
            st.dim("·"),
            st.cyan("./vello restart")
        );
    }
    if catalog_changed {
        println!(
            "  {} catalog updated — see new entries: {}",
            st.dim("·"),
            st.cyan("./vello list")
        );
    }
    println!(
        "  {} models, configs, and Open WebUI data are untouched",
        st.green("✓")
    );

    Ok(())
}

fn git_head(root: &std::path::Path) -> Result<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .with_context(|| "git rev-parse HEAD")?;
    if !out.status.success() {
        bail!("git rev-parse failed");
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn short_sha(sha: &str) -> String {
    sha.chars().take(7).collect()
}
