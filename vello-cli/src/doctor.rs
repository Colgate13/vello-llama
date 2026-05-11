//! `vello doctor` — host pre-flight: collect every requirement, classify it,
//! and emit either a human table or a JSON object.
//!
//! Design notes:
//! - **One pass, no fail-fast.** Every check runs and is reported; missing
//!   prerequisites are surfaced together so the user can fix everything once
//!   instead of installing → re-running → installing → re-running.
//! - **Mode is explicit.** `--cpu`/`--gpu` write the choice to `profile.toml`
//!   so downstream code (`apply`, `install`) sees the same answer.
//! - **JSON output is stable.** Schema versioned; CI/servers can grep it.
//! - **`--fix` is intentionally a stub in PR 1.** Auto-install lands in PR 2
//!   together with installer integration; this PR only diagnoses.

use crate::paths::Paths;
use crate::profile;
use crate::schema::{Mode, Profile};
use crate::style::Style;
use crate::system;
use anyhow::{Context, Result};
use serde::Serialize;
use std::fs;
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::process::{Command, Stdio};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Ok,
    Warn,
    Fail,
    Skip,
}

#[derive(Debug, Clone, Serialize)]
pub struct CheckResult {
    pub id: String,
    pub category: String,
    pub status: Severity,
    pub value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
    pub auto_installable: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModeInfo {
    pub selected: Mode,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub ok: usize,
    pub warn: usize,
    pub fail: usize,
    pub skip: usize,
    pub mode: Mode,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub schema: u32,
    pub summary: Summary,
    pub mode: ModeInfo,
    pub checks: Vec<CheckResult>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DoctorOpts {
    pub json: bool,
    pub fix: bool,
    pub yes: bool,
    pub force_cpu: bool,
    pub force_gpu: bool,
    pub deep: bool,
    pub installer_mode: bool,
}

const SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run(paths: &Paths, profile: &Profile, opts: DoctorOpts) -> Result<i32> {
    // Step 1: decide mode (may persist a new value to profile.toml).
    let (mode, mode_reason, profile) = decide_mode(paths, profile, &opts)?;

    // Step 2: first pass — read-only diagnosis.
    let report = collect(paths, &profile, mode, mode_reason.clone(), &opts);

    if !opts.json {
        render_human(&report);
    }

    // Step 3: optional --fix. Re-run the catalog after fixing so the rendered /
    //         emitted report reflects the new state.
    let final_report = if opts.fix {
        let plan = build_fix_plan(&report.checks);
        if plan.is_empty() {
            if !opts.json {
                println!();
                println!("nothing to fix — every auto-installable item is already ok.");
            }
            report
        } else {
            apply_fix_plan(&plan, opts.yes, opts.json)?;
            let after = collect(paths, &profile, mode, mode_reason, &opts);
            if !opts.json {
                println!();
                println!("--- after fix ---");
                render_human(&after);
            }
            after
        }
    } else {
        report
    };

    if opts.json {
        render_json(&final_report)?;
    }

    Ok(exit_code(&final_report))
}

fn collect(
    paths: &Paths,
    profile: &Profile,
    mode: Mode,
    mode_reason: String,
    opts: &DoctorOpts,
) -> Report {
    let mut checks = Vec::new();
    checks.extend(checks_os());
    checks.extend(checks_cpu());
    checks.extend(checks_mem(profile));
    checks.extend(checks_disk(paths));
    checks.extend(checks_tools());
    checks.extend(checks_gpu(mode, opts.deep || opts.installer_mode));
    checks.extend(checks_rust());
    checks.extend(checks_ports(paths));
    checks.extend(checks_path(paths));
    checks.extend(checks_project(paths, mode, opts.installer_mode));
    let summary = summarize(&checks, mode);
    Report {
        schema: SCHEMA_VERSION,
        summary,
        mode: ModeInfo {
            selected: mode,
            reason: mode_reason,
        },
        checks,
    }
}

// ---------------------------------------------------------------------------
// Mode resolution
// ---------------------------------------------------------------------------

fn decide_mode(
    paths: &Paths,
    current: &Profile,
    opts: &DoctorOpts,
) -> Result<(Mode, String, Profile)> {
    let gpu = profile::detect_gpu();

    // Explicit overrides win.
    if opts.force_gpu {
        if gpu.is_none() {
            anyhow::bail!(
                "--gpu requested but no NVIDIA GPU detected (nvidia-smi failed or absent)"
            );
        }
        let p = profile::set_mode(&paths.profile, Mode::Gpu)?;
        return Ok((
            Mode::Gpu,
            "forced by --gpu (NVIDIA GPU detected)".into(),
            p,
        ));
    }
    if opts.force_cpu {
        let p = profile::set_mode(&paths.profile, Mode::Cpu)?;
        return Ok((Mode::Cpu, "forced by --cpu".into(), p));
    }

    // No override: derive from current profile + GPU presence.
    match (current.mode, gpu.is_some()) {
        (Mode::Gpu, true) => (
            Mode::Gpu,
            "profile.toml mode = gpu, NVIDIA GPU detected".to_string(),
            current.clone(),
        )
            .pipe(Ok),
        (Mode::Gpu, false) => {
            // GPU mode persisted but no GPU available — ask or fail.
            handle_missing_gpu(paths, current, opts)
        }
        (Mode::Cpu, _) => (
            Mode::Cpu,
            "profile.toml mode = cpu".to_string(),
            current.clone(),
        )
            .pipe(Ok),
    }
}

fn handle_missing_gpu(
    paths: &Paths,
    _current: &Profile,
    opts: &DoctorOpts,
) -> Result<(Mode, String, Profile)> {
    // In auto/CI contexts, refuse to silently flip to CPU; otherwise prompt.
    if opts.yes || opts.installer_mode {
        let p = profile::set_mode(&paths.profile, Mode::Cpu)?;
        return Ok((
            Mode::Cpu,
            "no GPU detected; auto-selected CPU mode (--yes)".into(),
            p,
        ));
    }
    if !std::io::stdout().is_terminal() {
        anyhow::bail!(
            "no NVIDIA GPU detected and stdin is not a TTY. \
             Re-run with --cpu, --gpu, or --yes to choose explicitly."
        );
    }
    if opts.json {
        // JSON mode never prompts; user must decide explicitly.
        anyhow::bail!(
            "no NVIDIA GPU detected. JSON mode does not prompt — pass --cpu, --gpu, or --yes."
        );
    }
    eprintln!("No NVIDIA GPU detected.");
    eprint!("Continue in CPU mode? [y/N] ");
    std::io::stderr().flush().ok();
    let mut buf = String::new();
    std::io::stdin()
        .read_line(&mut buf)
        .context("reading user input")?;
    let yes = matches!(buf.trim(), "y" | "Y" | "yes" | "YES");
    if yes {
        let p = profile::set_mode(&paths.profile, Mode::Cpu)?;
        Ok((Mode::Cpu, "no GPU detected; user confirmed CPU mode".into(), p))
    } else {
        anyhow::bail!("aborted by user (no GPU and CPU mode declined)")
    }
}

// Tiny helper so the match arm above reads top-down without ugly let-bindings.
trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}
impl<T> Pipe for T {}

// ---------------------------------------------------------------------------
// Check implementations
// ---------------------------------------------------------------------------

fn checks_os() -> Vec<CheckResult> {
    let mut out = Vec::new();
    out.push(CheckResult {
        id: "os.kernel".into(),
        category: "os".into(),
        status: Severity::Ok,
        value: run_string("uname", &["-r"]).unwrap_or_else(|| "unknown".into()),
        fix: None,
        auto_installable: false,
    });

    let (distro, supported) = detect_distro();
    out.push(CheckResult {
        id: "os.distro".into(),
        category: "os".into(),
        status: if supported {
            Severity::Ok
        } else {
            Severity::Warn
        },
        value: distro,
        fix: if supported {
            None
        } else {
            Some(
                "Only Debian/Ubuntu are auto-supported. Install prerequisites manually \
                 and proceed."
                    .into(),
            )
        },
        auto_installable: false,
    });

    let pm = detect_package_manager();
    out.push(CheckResult {
        id: "os.package_manager".into(),
        category: "os".into(),
        // apt = full auto-install path; anything else = doctor still runs,
        // but --fix can only print hints (no auto-install for now).
        status: if pm == "apt" {
            Severity::Ok
        } else {
            Severity::Warn
        },
        value: pm.clone(),
        fix: if pm == "apt" {
            None
        } else if pm == "none" {
            Some("No known package manager detected. Install prerequisites manually.".into())
        } else {
            Some(format!(
                "Auto-install only supports apt today; detected {pm}. Install missing items manually."
            ))
        },
        auto_installable: false,
    });
    out
}

fn checks_cpu() -> Vec<CheckResult> {
    let mut out = Vec::new();
    let cores = run_string("nproc", &[]).unwrap_or_else(|| "?".into());
    out.push(CheckResult {
        id: "cpu.cores".into(),
        category: "cpu".into(),
        status: Severity::Ok,
        value: format!("{} threads", cores),
        fix: None,
        auto_installable: false,
    });

    let arch = run_string("uname", &["-m"]).unwrap_or_else(|| "unknown".into());
    let arch_ok = arch == "x86_64";
    out.push(CheckResult {
        id: "cpu.arch".into(),
        category: "cpu".into(),
        status: if arch_ok {
            Severity::Ok
        } else {
            Severity::Warn
        },
        value: arch,
        fix: if arch_ok {
            None
        } else {
            Some("Only x86_64 is tested today.".into())
        },
        auto_installable: false,
    });
    out
}

fn checks_mem(profile: &Profile) -> Vec<CheckResult> {
    let ram = profile::detect_ram().unwrap_or(profile.ram_gb);
    let status = if ram >= 8.0 {
        Severity::Ok
    } else {
        Severity::Warn
    };
    vec![CheckResult {
        id: "mem.ram_total".into(),
        category: "mem".into(),
        status,
        value: format!("{:.1} GB", ram),
        fix: if status == Severity::Warn {
            Some("At least 8 GB RAM is recommended; small models will still run.".into())
        } else {
            None
        },
        auto_installable: false,
    }]
}

fn checks_disk(paths: &Paths) -> Vec<CheckResult> {
    let free = free_gb(&paths.project_root);
    let (status, hint) = match free {
        Some(g) if g >= 17.0 => (Severity::Ok, None),
        Some(g) if g >= 5.0 => (
            Severity::Warn,
            Some(format!(
                "{:.1} GB free; the catalog assumes ~17 GB (image + first model). Pick a small model.",
                g
            )),
        ),
        Some(g) => (
            Severity::Fail,
            Some(format!(
                "{:.1} GB free is too low. Free up space or move the project to a larger disk.",
                g
            )),
        ),
        None => (Severity::Warn, Some("could not determine free space".into())),
    };
    vec![CheckResult {
        id: "disk.project_root_free".into(),
        category: "disk".into(),
        status,
        value: free
            .map(|g| format!("{:.1} GB free in {}", g, paths.project_root.display()))
            .unwrap_or_else(|| "unknown".into()),
        fix: hint,
        auto_installable: false,
    }]
}

fn checks_tools() -> Vec<CheckResult> {
    let mut out = Vec::new();

    // docker — never auto-install (distros diverge, group changes need re-login).
    let docker_v = command_version("docker", &["--version"]);
    out.push(match docker_v {
        Some(v) => CheckResult {
            id: "tools.docker".into(),
            category: "tools".into(),
            status: Severity::Ok,
            value: v,
            fix: None,
            auto_installable: false,
        },
        None => CheckResult {
            id: "tools.docker".into(),
            category: "tools".into(),
            status: Severity::Fail,
            value: "missing".into(),
            fix: Some("Install Docker: https://docs.docker.com/engine/install/".into()),
            auto_installable: false,
        },
    });

    // docker daemon reachable.
    let daemon = Command::new("docker")
        .arg("info")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    out.push(CheckResult {
        id: "tools.docker_daemon".into(),
        category: "tools".into(),
        status: if daemon {
            Severity::Ok
        } else {
            Severity::Fail
        },
        value: if daemon {
            "reachable".into()
        } else {
            "unreachable".into()
        },
        fix: if daemon {
            None
        } else {
            Some(
                "Start the daemon (`sudo systemctl start docker`) and add yourself \
                 to the docker group (`sudo usermod -aG docker $USER`, then re-login)."
                    .into(),
            )
        },
        auto_installable: false,
    });

    // docker compose v2.
    let compose_v = command_version("docker", &["compose", "version"]);
    out.push(match compose_v {
        Some(v) => CheckResult {
            id: "tools.docker_compose".into(),
            category: "tools".into(),
            status: Severity::Ok,
            value: v,
            fix: None,
            auto_installable: false,
        },
        None => CheckResult {
            id: "tools.docker_compose".into(),
            category: "tools".into(),
            status: Severity::Fail,
            value: "missing (need v2 plugin)".into(),
            fix: Some(
                "Update Docker; modern installs ship `docker compose` (v2 plugin)."
                    .into(),
            ),
            auto_installable: false,
        },
    });

    // (command, optional). The apt package name is resolved via
    // `apt_package_for()` below so e.g. `watch` correctly maps to `procps`.
    for (name, optional) in [
        ("curl", false),
        ("jq", false),
        ("git", false),
        ("gpg", false),   // needed by the nvidia-container-toolkit install pipeline
        ("watch", true),
    ] {
        let present = command_exists(name);
        let pkg = apt_package_for(name);
        out.push(CheckResult {
            id: format!("tools.{name}"),
            category: "tools".into(),
            status: if present {
                Severity::Ok
            } else if optional {
                Severity::Warn
            } else {
                Severity::Fail
            },
            value: if present { "found".into() } else { "missing".into() },
            fix: if present {
                None
            } else {
                Some(format!("sudo apt install -y {pkg}"))
            },
            auto_installable: !present,
        });
    }

    out
}

/// Map a command name to its apt package. Several commands live in packages
/// with a different name — `watch` is in `procps`, `gpg` is in `gnupg`, etc.
/// When the mapping is 1:1 the command name is returned unchanged.
fn apt_package_for(cmd: &str) -> String {
    match cmd {
        "watch" => "procps".into(),
        "gpg" => "gnupg".into(),
        "ss" => "iproute2".into(),
        // Default: assume the binary and the package share a name.
        other => other.into(),
    }
}

fn checks_gpu(mode: Mode, deep: bool) -> Vec<CheckResult> {
    if mode == Mode::Cpu {
        return vec![CheckResult {
            id: "gpu".into(),
            category: "gpu".into(),
            status: Severity::Skip,
            value: "skipped (mode = cpu)".into(),
            fix: None,
            auto_installable: false,
        }];
    }

    let mut out = Vec::new();
    let nvidia_smi = command_exists("nvidia-smi");
    out.push(CheckResult {
        id: "gpu.nvidia_smi".into(),
        category: "gpu".into(),
        status: if nvidia_smi {
            Severity::Ok
        } else {
            Severity::Fail
        },
        value: if nvidia_smi { "found".into() } else { "missing".into() },
        fix: if nvidia_smi {
            None
        } else {
            Some(
                "Install NVIDIA driver 550+ for your distro (e.g. \
                 `sudo apt install nvidia-driver-550` then reboot)."
                    .into(),
            )
        },
        auto_installable: false,
    });

    if nvidia_smi {
        // Driver version
        if let Some(drv) = run_string(
            "nvidia-smi",
            &["--query-gpu=driver_version", "--format=csv,noheader"],
        ) {
            let drv_first = drv.lines().next().unwrap_or(&drv).trim().to_string();
            let major = drv_first
                .split('.')
                .next()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0);
            let (status, fix) = if major >= 550 {
                (Severity::Ok, None)
            } else if major >= 535 {
                (
                    Severity::Warn,
                    Some(
                        "Driver < 550 works with CUDA 12.4 but not 12.6+. \
                         Consider upgrading to 550+."
                            .into(),
                    ),
                )
            } else {
                (
                    Severity::Fail,
                    Some(
                        "Driver < 535 cannot run the bundled CUDA image. \
                         Upgrade the NVIDIA driver."
                            .into(),
                    ),
                )
            };
            out.push(CheckResult {
                id: "gpu.driver_version".into(),
                category: "gpu".into(),
                status,
                value: drv_first,
                fix,
                auto_installable: false,
            });
        }

        // VRAM
        if let Some((vram, name, _)) = profile::detect_gpu() {
            out.push(CheckResult {
                id: "gpu.vram".into(),
                category: "gpu".into(),
                status: if vram >= 6.0 {
                    Severity::Ok
                } else {
                    Severity::Warn
                },
                value: format!("{:.1} GB ({})", vram, name),
                fix: if vram < 6.0 {
                    Some(
                        "Below 6 GB VRAM, only the smallest catalog entries (1.5B/3B) \
                         will fit comfortably."
                            .into(),
                    )
                } else {
                    None
                },
                auto_installable: false,
            });
        }
    }

    // nvidia-container-toolkit (apt-based detection; warn if not apt)
    let toolkit = Command::new("dpkg")
        .args(["-l", "nvidia-container-toolkit"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    out.push(CheckResult {
        id: "gpu.container_toolkit".into(),
        category: "gpu".into(),
        status: if toolkit {
            Severity::Ok
        } else {
            Severity::Fail
        },
        value: if toolkit { "installed".into() } else { "missing".into() },
        fix: if toolkit {
            None
        } else {
            Some(
                "Auto-installable via `./vello-installer install` (uses apt + sudo). \
                 Confirmation will be required."
                    .into(),
            )
        },
        auto_installable: !toolkit,
    });

    // Deep: actually exercise GPU passthrough through Docker.
    if deep && nvidia_smi && toolkit {
        let ok = Command::new("docker")
            .args([
                "run",
                "--rm",
                "--gpus",
                "all",
                "nvidia/cuda:12.4.0-base-ubuntu22.04",
                "nvidia-smi",
                "-L",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        out.push(CheckResult {
            id: "gpu.docker_passthrough".into(),
            category: "gpu".into(),
            status: if ok {
                Severity::Ok
            } else {
                Severity::Fail
            },
            value: if ok { "works".into() } else { "fails".into() },
            fix: if ok {
                None
            } else {
                Some(
                    "Docker cannot reach the GPU. Try: `sudo systemctl restart docker`."
                        .into(),
                )
            },
            auto_installable: false,
        });
    }

    out
}

fn checks_rust() -> Vec<CheckResult> {
    let v = command_version("cargo", &["--version"]);
    vec![match v {
        Some(s) => CheckResult {
            id: "rust.cargo".into(),
            category: "rust".into(),
            status: Severity::Ok,
            value: s,
            fix: None,
            auto_installable: false,
        },
        None => CheckResult {
            id: "rust.cargo".into(),
            category: "rust".into(),
            status: Severity::Fail,
            value: "missing".into(),
            fix: Some(
                "Install via rustup (the installer offers to do this for you): \
                 https://rustup.rs"
                    .into(),
            ),
            auto_installable: false,
        },
    }]
}

fn checks_ports(paths: &Paths) -> Vec<CheckResult> {
    // Best-effort: if system.toml is missing/broken, fall back to defaults.
    let sys = system::load_or_default(&paths.system).unwrap_or_default();
    let mut out = Vec::new();
    for (id, port) in [("ports.llama", sys.ports.llama), ("ports.webui", sys.ports.web_ui)] {
        let free = port_is_free(port);
        out.push(CheckResult {
            id: id.into(),
            category: "ports".into(),
            status: if free {
                Severity::Ok
            } else {
                Severity::Warn
            },
            value: if free {
                format!(":{port} free")
            } else {
                format!(":{port} bound by another process")
            },
            fix: if free {
                None
            } else {
                Some(
                    "Stop the conflicting process or change the port in system.toml \
                     [ports], then `vello apply`."
                        .into(),
                )
            },
            auto_installable: false,
        });
    }
    out
}

fn checks_path(paths: &Paths) -> Vec<CheckResult> {
    let mut out = Vec::new();
    let bin_dir = std::env::var_os("HOME")
        .map(|h| Path::new(&h).join(".local/bin"))
        .unwrap_or_else(|| Path::new("/").to_path_buf());

    let on_path = std::env::var("PATH")
        .map(|p| {
            std::env::split_paths(&p).any(|d| d == bin_dir)
        })
        .unwrap_or(false);

    out.push(CheckResult {
        id: "path.local_bin".into(),
        category: "path".into(),
        status: if on_path {
            Severity::Ok
        } else {
            Severity::Warn
        },
        value: if on_path {
            format!("{} on PATH", bin_dir.display())
        } else {
            format!("{} not on PATH", bin_dir.display())
        },
        fix: if on_path {
            None
        } else {
            Some(
                "Add to your shell rc: export PATH=\"$HOME/.local/bin:$PATH\" \
                 (or keep using ./vello from the project root)."
                    .into(),
            )
        },
        auto_installable: false,
    });

    let symlink = bin_dir.join("vello");
    let symlink_target = std::fs::read_link(&symlink).ok();
    let expected = paths.project_root.join("vello");
    let matches_expected = symlink_target
        .as_ref()
        .map(|t| t == &expected)
        .unwrap_or(false);
    out.push(CheckResult {
        id: "path.vello_symlink".into(),
        category: "path".into(),
        // Both "symlink points elsewhere" and "symlink absent" are non-fatal
        // hints: vello still works from the project root in either case.
        status: if matches_expected {
            Severity::Ok
        } else {
            Severity::Warn
        },
        value: match &symlink_target {
            Some(_) if matches_expected => format!("{} → this project", symlink.display()),
            Some(t) => format!("{} → {} (different project)", symlink.display(), t.display()),
            None => format!("{} not present", symlink.display()),
        },
        fix: if matches_expected {
            None
        } else {
            Some(format!(
                "`./vello-installer install` will create this symlink ({} → {}).",
                symlink.display(),
                expected.display()
            ))
        },
        auto_installable: false,
    });
    out
}

fn checks_project(paths: &Paths, mode: Mode, installer_mode: bool) -> Vec<CheckResult> {
    let mut out = Vec::new();
    let profile_ok = paths.profile.is_file();
    out.push(CheckResult {
        id: "project.profile_toml".into(),
        category: "project".into(),
        status: if profile_ok {
            Severity::Ok
        } else {
            Severity::Warn
        },
        value: paths.profile.display().to_string(),
        fix: if profile_ok {
            None
        } else {
            Some("Will be auto-generated on first run.".into())
        },
        auto_installable: false,
    });
    let system_ok = paths.system.is_file();
    out.push(CheckResult {
        id: "project.system_toml".into(),
        category: "project".into(),
        status: if system_ok {
            Severity::Ok
        } else {
            Severity::Warn
        },
        value: paths.system.display().to_string(),
        fix: if system_ok {
            None
        } else {
            Some("Will be auto-generated from the template on `vello apply`.".into())
        },
        auto_installable: false,
    });

    // Docker image is only relevant after the installer has run. In both
    // installer-mode (pre-build) and user-mode (post-build) we surface it
    // as a warn rather than a fail — the installer will build it in its
    // next step, and outside the installer the user can run `vello build`.
    // The image we look for depends on the active runtime mode.
    let _ = installer_mode; // currently informational only
    let image_tag = match mode {
        Mode::Gpu => "vello/llama-server-cuda:12.4",
        Mode::Cpu => "vello/llama-server-cpu:latest",
    };
    let image_present = Command::new("docker")
        .args(["image", "inspect", image_tag])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    let status = if image_present {
        Severity::Ok
    } else {
        Severity::Warn
    };
    out.push(CheckResult {
        id: "project.docker_image".into(),
        category: "project".into(),
        status,
        value: if image_present {
            format!("{image_tag} built")
        } else {
            format!("{image_tag} not built yet")
        },
        fix: if image_present {
            None
        } else {
            Some(
                "`./vello-installer install` (one-shot) or `./vello build` (rebuild) \
                 will build the image."
                    .into(),
            )
        },
        auto_installable: false,
    });

    out
}

// ---------------------------------------------------------------------------
// Summary + rendering
// ---------------------------------------------------------------------------

fn summarize(checks: &[CheckResult], mode: Mode) -> Summary {
    let mut s = Summary {
        ok: 0,
        warn: 0,
        fail: 0,
        skip: 0,
        mode,
    };
    for c in checks {
        match c.status {
            Severity::Ok => s.ok += 1,
            Severity::Warn => s.warn += 1,
            Severity::Fail => s.fail += 1,
            Severity::Skip => s.skip += 1,
        }
    }
    s
}

fn render_human(report: &Report) {
    let st = Style::new();

    println!("{}", st.bold("vello doctor — host pre-flight"));
    println!(
        "  {}: {} ({})",
        st.dim("mode"),
        mode_label(report.mode.selected),
        report.mode.reason
    );
    println!();

    // Compute column widths
    let id_w = report
        .checks
        .iter()
        .map(|c| c.id.len())
        .max()
        .unwrap_or(20)
        .max(20);
    let val_w = report
        .checks
        .iter()
        .map(|c| c.value.len())
        .max()
        .unwrap_or(40)
        .min(60);

    let mut last_cat = String::new();
    for c in &report.checks {
        if c.category != last_cat {
            if !last_cat.is_empty() {
                println!();
            }
            println!("{}", st.dim(&format!("[{}]", c.category)));
            last_cat = c.category.clone();
        }
        let badge = match c.status {
            Severity::Ok => st.green(" ok  "),
            Severity::Warn => st.yellow("warn "),
            Severity::Fail => st.red("fail "),
            Severity::Skip => st.dim("skip "),
        };
        let value = truncate(&c.value, val_w);
        println!("  {} {:<id_w$}  {}", badge, c.id, value, id_w = id_w);
        if let Some(fix) = &c.fix {
            let prefix = "        ↳ ";
            for (i, line) in wrap(fix, 92).into_iter().enumerate() {
                if i == 0 {
                    println!("{}{}", st.dim(prefix), line);
                } else {
                    println!("{}{}", st.dim("          "), line);
                }
            }
        }
    }

    println!();
    println!(
        "{}: {} ok, {} warn, {} fail, {} skip",
        st.bold("summary"),
        st.green(&report.summary.ok.to_string()),
        st.yellow(&report.summary.warn.to_string()),
        st.red(&report.summary.fail.to_string()),
        st.dim(&report.summary.skip.to_string()),
    );

    let auto_count = report
        .checks
        .iter()
        .filter(|c| c.auto_installable && c.status == Severity::Fail)
        .count();
    if auto_count > 0 {
        println!(
            "  {} {} item(s) can be auto-installed with `./vello doctor --fix` (add --yes for unattended).",
            st.cyan("hint:"),
            auto_count
        );
    }
}

fn render_json(report: &Report) -> Result<()> {
    let s = serde_json::to_string_pretty(report).context("serializing doctor report to JSON")?;
    println!("{s}");
    Ok(())
}

fn exit_code(report: &Report) -> i32 {
    if report.summary.fail == 0 {
        return 0;
    }
    let has_unfixable = report
        .checks
        .iter()
        .any(|c| c.status == Severity::Fail && !c.auto_installable);
    if has_unfixable {
        2
    } else {
        1
    }
}

// ---------------------------------------------------------------------------
// --fix: build + apply an auto-install plan
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum FixAction {
    /// Single batched `sudo apt-get install -y <pkgs...>` after an `apt-get update`.
    AptInstall(Vec<String>),
    /// The full add-repo + install + nvidia-ctk + restart-docker dance, mirrored
    /// from vello-installer's `install_nvidia_toolkit`.
    NvidiaContainerToolkit,
}

fn build_fix_plan(checks: &[CheckResult]) -> Vec<FixAction> {
    let mut apt = Vec::new();
    let mut toolkit = false;
    for c in checks {
        if !c.auto_installable || c.status != Severity::Fail {
            continue;
        }
        match c.id.as_str() {
            "gpu.container_toolkit" => toolkit = true,
            id if id.starts_with("tools.") => {
                if let Some(cmd) = id.strip_prefix("tools.") {
                    let pkg = apt_package_for(cmd);
                    // De-dup: `watch` and any future alias may both map to the
                    // same apt package (e.g., procps).
                    if !apt.contains(&pkg) {
                        apt.push(pkg);
                    }
                }
            }
            _ => {}
        }
    }
    let mut plan = Vec::new();
    if !apt.is_empty() {
        plan.push(FixAction::AptInstall(apt));
    }
    if toolkit {
        plan.push(FixAction::NvidiaContainerToolkit);
    }
    plan
}

fn apply_fix_plan(plan: &[FixAction], yes: bool, json: bool) -> Result<()> {
    if !json {
        println!();
        println!("plan:");
        for a in plan {
            for cmd in describe_action(a) {
                println!("  $ {cmd}");
            }
        }
        println!();
    }

    if !yes {
        if json {
            anyhow::bail!(
                "--fix without --yes requires a TTY for confirmation; JSON mode cannot prompt"
            );
        }
        if !std::io::stdout().is_terminal() {
            anyhow::bail!(
                "--fix without --yes requires a TTY for confirmation; pass --yes for non-interactive runs"
            );
        }
        if !confirm("apply this plan? (sudo will be requested)")? {
            anyhow::bail!("aborted by user");
        }
    }

    for action in plan {
        match action {
            FixAction::AptInstall(pkgs) => run_apt_install(pkgs)?,
            FixAction::NvidiaContainerToolkit => run_nvidia_container_toolkit()?,
        }
    }
    Ok(())
}

fn describe_action(action: &FixAction) -> Vec<String> {
    match action {
        FixAction::AptInstall(pkgs) => vec![
            "sudo apt-get update -y".into(),
            format!("sudo apt-get install -y {}", pkgs.join(" ")),
        ],
        FixAction::NvidiaContainerToolkit => vec![
            "curl … | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg".into(),
            "curl … | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list".into(),
            "sudo apt-get update -y".into(),
            "sudo apt-get install -y nvidia-container-toolkit".into(),
            "sudo nvidia-ctk runtime configure --runtime=docker".into(),
            "sudo systemctl restart docker".into(),
        ],
    }
}

fn run_apt_install(pkgs: &[String]) -> Result<()> {
    let update = Command::new("sudo")
        .args(["apt-get", "update", "-y"])
        .status()
        .context("running sudo apt-get update")?;
    if !update.success() {
        anyhow::bail!("apt-get update failed");
    }
    let mut cmd = Command::new("sudo");
    cmd.args(["apt-get", "install", "-y"]);
    for p in pkgs {
        cmd.arg(p);
    }
    let status = cmd.status().context("running sudo apt-get install")?;
    if !status.success() {
        anyhow::bail!("apt-get install failed for: {}", pkgs.join(", "));
    }
    Ok(())
}

fn run_nvidia_container_toolkit() -> Result<()> {
    // Mirror of vello-installer:install_nvidia_toolkit. Pipes need a shell, so
    // we run the whole sequence in a single `bash -c`. `set -e` aborts on the
    // first failure, mirroring the installer's `set -euo pipefail`.
    let script = r#"set -euo pipefail
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --batch --yes --dearmor \
      -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update -y
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
"#;
    let status = Command::new("bash")
        .args(["-c", script])
        .status()
        .context("running nvidia-container-toolkit install sequence")?;
    if !status.success() {
        anyhow::bail!("nvidia-container-toolkit install failed");
    }
    Ok(())
}

fn confirm(prompt: &str) -> Result<bool> {
    eprint!("{prompt} [y/N] ");
    std::io::stderr().flush().ok();
    let mut buf = String::new();
    std::io::stdin()
        .read_line(&mut buf)
        .context("reading confirmation")?;
    Ok(matches!(buf.trim(), "y" | "Y" | "yes" | "YES"))
}

// ---------------------------------------------------------------------------
// Low-level helpers
// ---------------------------------------------------------------------------

fn command_exists(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn command_version(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next().map(|l| l.trim().to_string())
}

fn run_string(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn detect_distro() -> (String, bool) {
    let raw = match fs::read_to_string("/etc/os-release") {
        Ok(s) => s,
        Err(_) => return ("unknown".into(), false),
    };
    let mut id = String::new();
    let mut id_like = String::new();
    let mut name = String::new();
    for line in raw.lines() {
        if let Some(v) = line.strip_prefix("ID=") {
            id = v.trim_matches('"').to_string();
        } else if let Some(v) = line.strip_prefix("ID_LIKE=") {
            id_like = v.trim_matches('"').to_string();
        } else if let Some(v) = line.strip_prefix("PRETTY_NAME=") {
            name = v.trim_matches('"').to_string();
        }
    }
    let supported =
        matches!(id.as_str(), "debian" | "ubuntu") || id_like.contains("debian") || id_like.contains("ubuntu");
    let label = if name.is_empty() {
        id.clone()
    } else {
        name
    };
    (label, supported)
}

fn detect_package_manager() -> String {
    for pm in ["apt", "dnf", "pacman", "zypper", "apk"] {
        if command_exists(pm) {
            return pm.into();
        }
    }
    "none".into()
}

fn free_gb(path: &Path) -> Option<f32> {
    // `df -B1 --output=avail <path>` → bytes available. Linux-specific, which
    // matches the rest of the project (Docker + nvidia-smi).
    let out = Command::new("df")
        .args(["-B1", "--output=avail"])
        .arg(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let bytes: u64 = text.lines().nth(1)?.trim().parse().ok()?;
    Some(bytes as f32 / 1024.0 / 1024.0 / 1024.0)
}

fn port_is_free(port: u16) -> bool {
    // `ss -ltn` lists listening TCP sockets. Match :PORT at end of address.
    let out = match Command::new("ss")
        .args(["-ltn"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return true, // can't tell → assume free, don't false-alarm
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let needle = format!(":{port} ");
    let needle_end = format!(":{port}\n");
    !text
        .lines()
        .any(|l| l.contains(&needle) || l.ends_with(&needle_end[..needle_end.len() - 1]))
}

fn mode_label(m: Mode) -> &'static str {
    match m {
        Mode::Gpu => "gpu",
        Mode::Cpu => "cpu",
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn wrap(s: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        if cur.is_empty() {
            cur.push_str(word);
        } else if cur.len() + 1 + word.len() <= width {
            cur.push(' ');
            cur.push_str(word);
        } else {
            lines.push(std::mem::take(&mut cur));
            cur.push_str(word);
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}
