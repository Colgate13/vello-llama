//! `vello` — curated LLM catalog manager + Docker stack control.
//!
//! The single daily-use binary. The sibling `vello-installer` bash script
//! handles one-time host setup (deps, image build, Rust toolchain) and is
//! never invoked directly after that.

mod catalog;
mod commands;
mod diagnostics;
mod docker;
mod fit;
mod paths;
mod profile;
mod recommend;
mod resolver;
mod schema;
mod system;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "vello",
    version,
    about = "Curated LLM catalog for vello-llama-local"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// List models from all catalogs with auto-calculated tier for your hardware.
    List {
        /// Filter by tag (e.g. code, vision, tools).
        #[arg(long)]
        tag: Option<String>,
        /// Filter by tier (S/A/B/C/D).
        #[arg(long)]
        tier: Option<String>,
        /// Filter by modality (text, image, video, audio).
        #[arg(long)]
        modality: Option<String>,
        /// Show only models present on disk.
        #[arg(long)]
        installed: bool,
    },

    /// Show detailed info for a single model (auto-picked quant, fit, status).
    Info {
        /// Model id (e.g. qwen3-8b).
        id: String,
        /// Show fit estimate for an explicit quant instead of the auto-pick.
        #[arg(long)]
        quant: Option<String>,
    },

    /// Recommend models for a free-text use case (e.g. "code", "vision", "raciocínio").
    Recommend {
        /// Use-case query in natural language.
        query: Vec<String>,
        /// Max recommendations.
        #[arg(long, default_value_t = 3)]
        limit: usize,
    },

    /// Download a model from HuggingFace, switch to it, restart the stack.
    Install {
        /// Model id.
        id: String,
        /// Download but don't switch the active model.
        #[arg(long)]
        no_switch: bool,
        /// Override auto-pick. Must be a key in the model's files map (e.g. Q4_K_M, IQ3_M).
        #[arg(long)]
        quant: Option<String>,
    },

    /// Switch the active model without downloading.
    Switch {
        /// Model id.
        id: String,
        /// Use a specific quant if multiple are present on disk.
        #[arg(long)]
        quant: Option<String>,
    },

    /// Delete a downloaded model file.
    Remove {
        /// Model id.
        id: String,
    },

    /// Show the currently active model.
    Active,

    /// Regenerate .env from system.toml + the active model's [runtime] block.
    /// Use after editing system.toml or the catalog's runtime values.
    Apply {
        /// Don't restart the stack after writing .env.
        #[arg(long)]
        no_restart: bool,
    },

    /// Manage catalogs (default + community).
    Catalog {
        #[command(subcommand)]
        sub: CatalogCmd,
    },

    /// Show or refresh the hardware profile.
    Profile {
        #[command(subcommand)]
        sub: ProfileCmd,
    },

    // ---------- stack lifecycle ----------
    /// Start the stack (llama-server + Open WebUI).
    Up,
    /// Stop the stack.
    Down,
    /// Restart the stack.
    Restart,
    /// Show running containers.
    Status,
    /// Stream container logs.
    Logs {
        /// Follow output (-f).
        #[arg(short, long)]
        follow: bool,
        /// Service names (default: all).
        services: Vec<String>,
    },
    /// Rebuild the llama-server Docker image.
    Build {
        /// Pass --no-cache.
        #[arg(long)]
        no_cache: bool,
    },
    /// Stop and remove containers, volumes, and image (keeps models).
    Nuke,

    // ---------- diagnostics ----------
    /// API health probe.
    Health,
    /// Live nvidia-smi (Ctrl-C to exit).
    Gpu,
    /// Throughput benchmark.
    Bench {
        /// Prompt to evaluate.
        #[arg(default_value = "Write a long essay about transformer neural networks.")]
        prompt: String,
        /// Tokens to predict.
        #[arg(default_value_t = 256)]
        n: u32,
    },
    /// Tool-calling smoke test.
    Test,
}

#[derive(Subcommand, Debug)]
enum CatalogCmd {
    /// List loaded catalogs.
    List,
    /// Add a community catalog from a local .toml file.
    Add {
        /// Path to a catalog .toml.
        source: String,
    },
    /// Remove a previously-added user catalog.
    Remove {
        /// Catalog name (as declared in its `name` field).
        name: String,
    },
}

#[derive(Subcommand, Debug)]
enum ProfileCmd {
    /// Show current profile (auto-detects on first run).
    Show,
    /// Re-detect hardware and rewrite profile.toml.
    Refresh,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let paths = paths::resolve()?;
    let profile = profile::load_or_detect(&paths.profile)?;

    match cli.cmd {
        Cmd::List {
            tag,
            tier,
            modality,
            installed,
        } => commands::cmd_list(
            &paths,
            &profile,
            &commands::ListFilters {
                tag,
                tier,
                modality,
                installed,
            },
        ),
        Cmd::Info { id, quant } => commands::cmd_info(&paths, &profile, &id, quant.as_deref()),
        Cmd::Recommend { query, limit } => {
            let q = query.join(" ");
            commands::cmd_recommend(&paths, &profile, &q, limit)
        }
        Cmd::Install {
            id,
            no_switch,
            quant,
        } => commands::cmd_install(&paths, &profile, &id, no_switch, quant.as_deref()),
        Cmd::Switch { id, quant } => commands::cmd_switch(&paths, &profile, &id, quant.as_deref()),
        Cmd::Remove { id } => commands::cmd_remove(&paths, &profile, &id),
        Cmd::Active => commands::cmd_active(&paths),
        Cmd::Apply { no_restart } => commands::cmd_apply(&paths, &profile, !no_restart),
        Cmd::Catalog { sub } => match sub {
            CatalogCmd::List => commands::cmd_catalog_list(&paths),
            CatalogCmd::Add { source } => commands::cmd_catalog_add(&paths, &source),
            CatalogCmd::Remove { name } => commands::cmd_catalog_remove(&paths, &name),
        },
        Cmd::Profile { sub } => match sub {
            ProfileCmd::Show => commands::cmd_profile_show(&profile),
            ProfileCmd::Refresh => commands::cmd_profile_refresh(&paths),
        },

        Cmd::Up => docker::up(&paths),
        Cmd::Down => docker::down(&paths),
        Cmd::Restart => docker::restart(&paths),
        Cmd::Status => docker::status(&paths),
        Cmd::Logs { follow, services } => docker::logs(&paths, follow, &services),
        Cmd::Build { no_cache } => docker::build(&paths, no_cache),
        Cmd::Nuke => commands::cmd_nuke(&paths),

        Cmd::Health => {
            let sys = system::load_or_default(&paths.system)?;
            diagnostics::health(&sys)
        }
        Cmd::Gpu => diagnostics::gpu(),
        Cmd::Bench { prompt, n } => {
            let sys = system::load_or_default(&paths.system)?;
            diagnostics::bench(&sys, &prompt, n)
        }
        Cmd::Test => {
            let sys = system::load_or_default(&paths.system)?;
            diagnostics::tools_test(&paths, &sys)
        }
    }
}
