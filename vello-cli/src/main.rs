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

const AFTER_HELP: &str = "\
COMMANDS BY CATEGORY:
  Discover    list, info, recommend
  Models      install, switch, remove, active
  Config      apply, catalog, profile, update
  Stack       up, down, restart, status (ps), logs, build, nuke
  Diagnostics health, gpu, bench, test

ALIASES:
  ls = list,  rm = remove,  ps = status

EXAMPLES:
  vello recommend chat            # see what fits your GPU
  vello install qwen3-8b          # download + activate
  vello up                        # start the stack (first boot: ~1–3 min)
  vello switch qwen2.5-coder-7b   # change active model (no re-download)
  vello ls -i                     # what's on disk
  vello logs -f                   # follow container logs

Run `vello <command> --help` for the long form of any subcommand.";

#[derive(Parser, Debug)]
#[command(
    name = "vello",
    version,
    about = "Curated LLM catalog + Docker stack control for vello-llama-local",
    long_about = "vello manages a curated catalog of GGUF models, downloads them from \
                  HuggingFace, and drives the local llama.cpp + Open WebUI Docker stack.\n\
                  \n\
                  Daily flow: discover (list / recommend) → install → up → chat at \
                  http://localhost:3000.",
    after_help = AFTER_HELP,
    after_long_help = AFTER_HELP,
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    // ---------- Discover ----------
    /// List models from all catalogs with auto-calculated tier for your hardware.
    #[command(
        alias = "ls",
        after_help = "EXAMPLES:\n  \
                        vello list                       # full catalog\n  \
                        vello list -i                    # only what's on disk\n  \
                        vello list -t S                  # only models that fit 100% in VRAM\n  \
                        vello list -T vision             # filter by tag\n  \
                        vello list -m image              # only multimodal"
    )]
    List {
        /// Filter by tag (e.g. code, vision, tools).
        #[arg(short = 'T', long)]
        tag: Option<String>,
        /// Filter by tier (S/A/B/C/D).
        #[arg(short = 't', long)]
        tier: Option<String>,
        /// Filter by modality (text, image, video, audio).
        #[arg(short = 'm', long)]
        modality: Option<String>,
        /// Show only models present on disk.
        #[arg(short = 'i', long)]
        installed: bool,
    },

    /// Show detailed info for a single model (auto-picked quant, fit, status).
    Info {
        /// Model id (e.g. qwen3-8b).
        id: String,
        /// Show fit estimate for an explicit quant instead of the auto-pick.
        #[arg(short = 'q', long)]
        quant: Option<String>,
    },

    /// Recommend models for a free-text use case (e.g. "code", "vision", "raciocínio").
    #[command(
        long_about = "Ranks the curated catalog against a use-case query and your hardware \
                      profile. Matches across tags, description, and id; understands EN/PT \
                      synonyms (\"código\" → code, \"raciocínio\" → reasoning).",
        after_help = "EXAMPLES:\n  \
                        vello recommend chat               # general chat\n  \
                        vello recommend código             # PT triggers also work\n  \
                        vello recommend raciocínio -l 5    # top 5 reasoning picks\n  \
                        vello recommend \"vision OCR\"       # multi-word query"
    )]
    Recommend {
        /// Use-case query in natural language.
        query: Vec<String>,
        /// Max recommendations.
        #[arg(short = 'l', long, default_value_t = 3)]
        limit: usize,
    },

    // ---------- Models ----------
    /// Download a model from HuggingFace, switch to it, restart the stack.
    #[command(
        long_about = "Looks the model up across loaded catalogs, picks the best quantization \
                      for your VRAM (Q5_K_M → Q4_K_M → IQ3/IQ2), downloads the GGUF (and an \
                      mmproj if the model is multimodal), regenerates .env, and restarts the \
                      stack if it's already running. On the very first install the stack is not \
                      yet running — finish with `vello up`.",
        after_help = "EXAMPLES:\n  \
                        vello install qwen3-8b                  # auto-pick best quant\n  \
                        vello install qwen3-30b-a3b -q Q4_K_M   # force a specific quant\n  \
                        vello install qwen3-8b -n               # download but don't switch"
    )]
    Install {
        /// Model id.
        id: String,
        /// Download but don't switch the active model.
        #[arg(short = 'n', long)]
        no_switch: bool,
        /// Override auto-pick. Must be a key in the model's files map (e.g. Q4_K_M, IQ3_M).
        #[arg(short = 'q', long)]
        quant: Option<String>,
    },

    /// Switch the active model without downloading.
    #[command(
        long_about = "Changes which GGUF llama-server loads. The file must already be on disk \
                      (use `install` first if not). Rewrites .env and restarts the stack so \
                      the new model is mmapped and uploaded to the GPU.",
        after_help = "EXAMPLES:\n  \
                        vello switch qwen3-8b\n  \
                        vello switch qwen3-8b -q Q4_K_M    # pick a specific quant on disk"
    )]
    Switch {
        /// Model id.
        id: String,
        /// Use a specific quant if multiple are present on disk.
        #[arg(short = 'q', long)]
        quant: Option<String>,
    },

    /// Delete a downloaded model file.
    #[command(alias = "rm")]
    Remove {
        /// Model id.
        id: String,
    },

    /// Show the currently active model.
    Active,

    // ---------- Config ----------
    /// Regenerate .env from system.toml + the active model's [runtime] block.
    #[command(
        long_about = "Reads system.toml + profile.toml + the active model's [runtime] from the \
                      catalog and rewrites .env. Run after editing system.toml or after a \
                      catalog runtime change. By default also restarts the stack so the new \
                      flags take effect."
    )]
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

    /// Pull the latest vello-llama-local from git and rebuild the CLI.
    #[command(
        long_about = "Updates vello in-place: refuses to run on a dirty working tree (use \
                      --force to override), runs `git pull --ff-only`, and rebuilds the Rust \
                      binary if vello-cli/ changed. Never touches models/, your TOMLs, .env, or \
                      the Open WebUI data volume. If docker/ or docker-compose.yml changed, you \
                      get a hint to run `vello build` and `vello restart` — image rebuilds are \
                      not done automatically because they're slow.",
        after_help = "EXAMPLES:\n  \
                        vello update            # pull + rebuild CLI if needed\n  \
                        vello update --force    # ignore dirty working tree (git decides)"
    )]
    Update {
        /// Skip the working-tree-clean check.
        #[arg(short = 'f', long)]
        force: bool,
    },

    // ---------- Stack ----------
    /// Start the stack (llama-server + Open WebUI).
    Up,
    /// Stop the stack.
    Down,
    /// Restart the stack.
    Restart,
    /// Show running containers.
    #[command(alias = "ps")]
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
    /// Stop and remove containers + image (host data — models, openwebui-data, configs — kept).
    Nuke,

    // ---------- Diagnostics ----------
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
        Cmd::Update { force } => commands::cmd_update(&paths, force),

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
