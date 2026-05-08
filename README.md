# vello-llama-local

> Self-hosted LLMs on your GPU. One installer, one CLI, one Web UI. No cloud,
> no API key, no telemetry.

```bash
git clone https://github.com/Colgate13/vello-llama-local.git
cd vello-llama-local
./vello-installer install
./vello recommend chat
./vello install qwen3-8b
```

Open <http://localhost:3000> and chat.

---

## Table of contents

- [What this project is](#what-this-project-is)
- [How it fits together](#how-it-fits-together)
- [Installation](#installation)
- [Daily use](#daily-use)
- [Configuration](#configuration)
- [The catalog](#the-catalog)
- [Connecting clients](#connecting-clients)
- [Troubleshooting](#troubleshooting)
- [Uninstall](#uninstall)
- [License](#license)

---

## What this project is

**vello-llama-local** is a thin, opinionated stack that runs open-weight LLMs
locally with three guarantees:

1. **One curated path.** A catalog of models that actually run well on consumer
   GPUs (8–24 GB VRAM), with sensible quantization picked automatically based
   on your hardware.
2. **No magic config.** All runtime settings live in human-readable TOML files;
   the `.env` consumed by Docker is generated from them, never hand-edited.
3. **OpenAI-compatible API.** The model is served at
   `http://localhost:8080/v1` so any OpenAI SDK or tool (Cursor, Continue.dev,
   opencode, langchain, your own scripts) works without changes.

It is built on top of:
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — the inference
  engine. Supports `.gguf` weights, runs on CUDA, exposes an HTTP API.
- **[Open WebUI](https://github.com/open-webui/open-webui)** — the chat
  interface in your browser.
- **[bartowski](https://huggingface.co/bartowski)** /
  **[unsloth](https://huggingface.co/unsloth)** GGUFs — the curated catalog
  pulls from these mirrors.

Two tools ship with the project, with very distinct roles:

| Tool | Type | When you run it |
|---|---|---|
| **`vello-installer`** | bash script | **Once.** Bootstraps the host: prerequisites, NVIDIA toolkit, Docker image, Rust toolchain, builds the `vello` binary. |
| **`vello`** | Rust binary | **Daily.** Catalog discovery, install/switch/remove, lifecycle (up/down/logs), diagnostics, runtime config. |

After the installer finishes, you do not run it again. Everything else is
`./vello`.

---

## How it fits together

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser ──:3000──►  Open WebUI (Docker container)                │
│                          │                                        │
│                          ▼ /v1/chat/completions                   │
│                      llama-server (Docker container)              │
│                          │                                        │
│                          ▼  loads (mmap + GPU)                    │
│                      ./models/<model>.gguf                        │
└──────────────────────────────────────────────────────────────────┘
                                ▲
                                │  controlled by
                                │
                          ./vello (Rust)
                                ▲
              reads / generates │
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
   profile.toml            system.toml          catalogs/*.toml
   (auto-detected)         (you edit)           (curated + community)
                                │
                                ▼ vello apply
                              .env  ──► docker compose up
                          (do not edit)
```

- `profile.toml` describes your hardware (VRAM, RAM, GPU). Auto-detected from
  `nvidia-smi` and `/proc/meminfo` on first run.
- `system.toml` holds **your machine preferences**: ports, Web UI auth,
  fallback runtime values used when a model doesn't override them.
- `catalogs/default.toml` is the curated model list. Each model can declare a
  `[model.runtime]` block with model-specific defaults (context size, MoE
  flags, vision projector path).
- `vello apply` (called automatically by `vello switch`/`install`) merges the
  three sources and writes a generated `.env` that `docker compose` consumes.

---

## Installation

### Prerequisites

You need these on the host before running the installer. Each requires `sudo`
or a manual decision; the installer doesn't try to do them for you.

| Requirement | Check | Notes |
|---|---|---|
| Linux (Debian/Ubuntu) | `uname -a` | tested on Debian 13, Ubuntu 22.04+ |
| NVIDIA driver 550+ | `nvidia-smi` | older drivers can't run CUDA 12.4 |
| Docker, user in `docker` group | `docker info` | no `sudo` should be needed |
| Free disk: ~17 GB initial | `df -h` | 5 GB image + 6 GB Open WebUI + 5 GB first model |

`curl` and `jq` are also expected for HTTP diagnostics; both are usually
already present.

### What the installer does

```bash
git clone https://github.com/Colgate13/vello-llama-local.git
cd vello-llama-local
./vello-installer install
```

Six steps, ~12–20 minutes total (most of it is the Docker image build, which
is cached after first run):

| Step | What happens | Why |
|---|---|---|
| 1. Prerequisites | Verifies `docker`, `curl`, `nvidia-smi` are on PATH and Docker daemon is reachable | Fail fast with actionable errors |
| 2. nvidia-container-toolkit | Asks for `sudo`, adds NVIDIA's apt repo, installs the toolkit, restarts Docker | Lets Docker containers see the GPU |
| 3. GPU smoke test | Runs `docker run --gpus all nvidia/cuda nvidia-smi -L` | Confirms passthrough actually works |
| 4. Build llama-server image | Compiles llama.cpp inside Docker, pinned to CUDA 12.4 | Avoids the official image (which needs driver 555+) |
| 5. Rust toolchain | If `cargo` not on PATH, offers `rustup` install in `$HOME` (no sudo) | Required to build `vello` from source |
| 6. Build vello | `cargo build --release` and symlink `./vello` | The catalog CLI you'll use daily |

When it finishes, you'll have:

```
vello-llama-local/
├── vello-installer        # the script you just ran (won't need again)
├── vello                  # symlink to the Rust binary you'll use daily
├── vello-cli/             # Rust source (for rebuilds)
├── catalogs/default.toml  # curated catalog of GGUF models
├── docker-compose.yml     # stack definition
├── docker/                # Dockerfile for llama-server
├── system.toml            # auto-created on first vello run
├── profile.toml           # auto-detected on first vello run
├── models/                # downloaded .gguf files land here (gitignored)
└── .env                   # generated from system.toml (do not edit)
```

No model is installed yet — that is your next decision.

---

## Daily use

The flow is always: **discover → install → control → tune**.

### 1. Discover

```bash
./vello list                       # full catalog with auto-calculated tier
./vello list --tier S              # only models that fit 100% in your VRAM
./vello list --tag vision          # filter by tag
./vello list --modality image      # only multimodal
./vello list --installed           # only what's already on disk

./vello recommend chat             # top picks for "general chat"
./vello recommend "código"         # for code (PT triggers also work)
./vello recommend "raciocínio"     # reasoning models

./vello info qwen3-8b              # full details for a model
./vello info qwen3-30b-a3b --quant Q4_K_M    # what would Q4 look like?
```

**Tiers** are computed from your hardware profile, not stored in the catalog:

| Tier | Means | Speed |
|---|---|---|
| **S** | fits 100% in VRAM | fastest (30–60 tok/s) |
| **A** | mostly VRAM, light overflow | fast (15–35 tok/s) |
| **B** | MoE or significant CPU offload | usable (5–25 tok/s) |
| **C** | RAM-dominant | slow (1–5 tok/s) |
| **D** | won't fit at all | skip |

MoE models like `qwen3-30b-a3b` typically land in **B** but run at near-S
speed because only the active parameters live in VRAM.

### 2. Install and switch

```bash
./vello install qwen3-8b           # auto-pick best quant + download + apply + restart
./vello install qwen3-30b-a3b --quant Q4_K_M   # force a specific quant

./vello switch qwen2.5-7b          # change active model (must be on disk)
./vello switch qwen3-8b --quant Q4_K_M

./vello active                     # what's loaded right now?
./vello remove qwen2.5-coder-7b    # delete the .gguf
```

**What `install` actually does:**

1. Looks up the model in all loaded catalogs.
2. Picks the best quantization for your hardware (Q5_K_M if it fits VRAM, else
   Q4_K_M, then descends to IQ3/IQ2 only as a last resort).
3. Downloads the GGUF from the catalog's HuggingFace repo via `curl`.
4. If the model declares a vision `mmproj`, downloads that too.
5. Calls `vello apply` to regenerate `.env` from `system.toml` + the model's
   `[model.runtime]` block.
6. Restarts the docker stack so the new model is loaded.

### 3. Control the stack

Direct wrappers around `docker compose`. None require sudo.

```bash
./vello up                # start llama-server + open-webui
./vello down              # stop
./vello restart           # down + up

./vello status            # running containers
./vello logs              # last 200 lines
./vello logs -f           # follow
./vello logs -f llama-server   # only one service

./vello build             # rebuild the llama-server image (rare)
./vello build --no-cache  # force fresh build
./vello nuke              # remove containers, volumes, image (keeps models/)
```

### 4. Diagnose

```bash
./vello health            # is the API responding?
./vello gpu               # live nvidia-smi (Ctrl-C to exit)
./vello bench             # throughput benchmark (default 256 tokens)
./vello bench "Custom prompt" 512
./vello test              # tool-calling smoke test on the active model
```

`bench` is the fastest sanity check after `install`. `test` is what you run
when you suspect the active model can't do tool calling — it sends a known
tool-call prompt and verifies the model returns structured output, not free
text.

### 5. Tune

```bash
./vello apply                        # regenerate .env from TOMLs (and restart)
./vello apply --no-restart           # just rewrite .env, leave stack alone

./vello profile show                 # detected hardware
./vello profile refresh              # re-detect (after a hardware change)

./vello catalog list                 # all loaded catalogs
./vello catalog add ./extra.toml     # add a community catalog
./vello catalog remove extra-name    # remove it
```

`vello apply` is the bridge between editing a TOML and the running stack. Edit
`system.toml`, then run `vello apply` to push changes through to the
container.

---

## Configuration

There are three TOML files. **You only edit two**, and both are optional.

### `profile.toml` — your hardware (auto-detected)

Created on first run by parsing `nvidia-smi` and `/proc/meminfo`. You only
edit it if you want to **lie** to vello (e.g. lower `vram_gb` to force smaller
quantizations) or if auto-detection fails.

```toml
vram_gb = 8.0           # your GPU's total VRAM
ram_gb = 31.2           # your system RAM
gpu_name = "NVIDIA GeForce RTX 4060"
cuda_arch = 89
vram_reserve_gb = 1.0   # held back for desktop/system
ram_reserve_gb = 4.0    # held back for OS
default_ctx = 32768     # used to estimate KV cache
```

### `system.toml` — your machine preferences (you edit)

Created on first `vello apply` from a built-in template. This is **the** file
you edit by hand.

```toml
[ports]
llama  = 8080
web_ui = 3000

[web_ui]
auth = false       # true if exposing Open WebUI on a network

[runtime]
default_ctx     = 32768
default_ngl     = 99
default_threads = 6
default_batch   = 2048
default_ubatch  = 512
kv_cache_k      = "q8_0"
kv_cache_v      = "q8_0"
flash_attn      = true
```

After editing, run `./vello apply`.

### `catalogs/*.toml` — the model catalog (curated + community)

`catalogs/default.toml` is the curated catalog (29 models). You don't edit
this directly; it's maintained by the project. To add your own models or use
a community catalog, drop a TOML file under `catalogs/user/` (or use
`vello catalog add`).

Each model entry can include a `[model.runtime]` block with model-specific
defaults. These **override** `system.toml` defaults for that model:

```toml
[[model]]
id = "qwen2.5-vl-7b"
# ... required fields ...
[model.runtime]
mmproj      = "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"  # vision projector
extra_args  = ["--n-cpu-moe", "32"]                      # MoE offload
ctx_default = 16384                                       # ctx override
```

When `vello install` sees an `mmproj`, it pulls that file too. When the stack
starts, the resolver injects `--mmproj /models/<file>` and any `extra_args`
into the llama-server command.

### `.env` — generated, do not edit

This is what docker-compose reads. `vello apply` rewrites it from the three
TOMLs. Keys you add manually that vello doesn't manage (e.g. extras for Open
WebUI) are **preserved** across regenerations.

---

## The catalog

### Format

```toml
schema_version = 1
name = "my-catalog"
maintainer = "your-handle"

[[model]]
id            = "my-model"
repo          = "bartowski/Some-Model-GGUF"
default_quant = "Q4_K_M"
params_total_b = 7.0
architecture  = "dense"     # or "moe" — moe needs params_active_b
modalities    = ["text"]    # or ["text", "image"], etc.
tags          = ["chat", "tools"]
description   = "What this model is good at."

[model.files]
Q5_K_M = "Some-Model-Q5_K_M.gguf"
Q4_K_M = "Some-Model-Q4_K_M.gguf"

[model.runtime]              # optional
ctx_default = 32768
mmproj      = "mmproj-...-f16.gguf"
extra_args  = ["--n-cpu-moe", "32"]
```

**Required**: `id`, `repo`, `default_quant`, `files`, `params_total_b`,
`architecture` (and `params_active_b` for MoE).

### Auto-pick quantization

`vello install` (without `--quant`) tries:

1. `Q5_K_M` if it fits VRAM (best balanced quality)
2. `Q4_K_M` as fallback
3. `Q3_K_M` / `IQ4_XS` / `IQ3_M` / `IQ2_M` as a last resort

For **MoE** models, only the *active* parameters count toward VRAM; the rest
goes to RAM via llama.cpp's `--n-cpu-moe`. That's why a 30B/3B-active MoE
picks Q5 comfortably while a 14B dense gets Q4.

### Adding a community catalog

```bash
./vello catalog add path/to/their-catalog.toml
```

Vello validates the schema, copies the file to `catalogs/user/`, and merges
its models into `vello list`. Conflicts on `id` are resolved in favor of the
default catalog.

---

## Connecting clients

The llama-server exposes an OpenAI-compatible API on `http://localhost:8080/v1`.
Anything that supports OpenAI works.

| Setting | Value |
|---|---|
| `baseURL` | `http://localhost:8080/v1` |
| `apiKey` | any string (`local`, `xxx`, etc.) |
| `model` | shown by `./vello active` (e.g. `qwen3-8b-local`) |

### Browser (default)

Open <http://localhost:3000> — Open WebUI is already wired to llama-server
inside the Docker network.

### Raw curl

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-local",
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

### Cursor / Continue.dev / langchain / OpenAI SDKs

Set `baseURL` to `http://localhost:8080/v1` and any string for the API key.
The model name is what `./vello active` reports.

### opencode / crush

Pre-baked configs ship in `clients/`. Copy them to `~/.config/opencode/` and
`~/.config/crush/` respectively (or symlink) and run `opencode` / `crush`.

---

## Troubleshooting

<details>
<summary><b>Stack won't start</b></summary>

```bash
./vello logs llama-server
```

Common causes:
- **Out of VRAM** → smaller model: `./vello recommend small`
- **Driver < CUDA in image** → in `docker-compose.yml` lower `CUDA_VERSION` to
  match your driver, then `./vello build`
- **Port collision** (8080/3000 used by another service) → edit `[ports]` in
  `system.toml`, run `./vello apply`
</details>

<details>
<summary><b>Tool calling returns plain text instead of structured calls</b></summary>

Some models (notably `qwen2.5-coder-*`) don't follow tool-calling templates
well. Switch to one tagged `tools`:

```bash
./vello list --tag tools
./vello switch qwen3-8b
./vello test
```
</details>

<details>
<summary><b>I edited .env and my changes vanished</b></summary>

`.env` is generated. Edit `system.toml` (machine-level) or the model's
`[model.runtime]` block (per-model), then `./vello apply`.

Custom keys you add to `.env` that vello doesn't manage are preserved across
regenerations.
</details>

<details>
<summary><b>Vision (image input) doesn't work</b></summary>

The catalog declares an `mmproj` file in `[model.runtime]`; `vello install`
auto-pulls it. If you skipped that step:

```bash
./vello install <id>          # pulls the mmproj if missing
./vello apply
```

Then drag-and-drop an image into Open WebUI.
</details>

<details>
<summary><b>"forward compatibility was attempted on non supported HW"</b></summary>

Your NVIDIA driver is older than the CUDA version in the image. In
`docker-compose.yml`, lower `CUDA_VERSION` to match — for driver 550, use
`12.4.1` (the default). Then:

```bash
./vello build
```
</details>

<details>
<summary><b>Non-RTX-40 GPU</b></summary>

Edit `docker-compose.yml`:

```yaml
CUDA_ARCH: "86"   # 30-series Ampere
CUDA_ARCH: "75"   # 20-series Turing
CUDA_ARCH: "89"   # 40-series Ada (default)
```

Then `./vello build`.
</details>

<details>
<summary><b>Auto-pick chose a quant I don't want</b></summary>

Override at install time:

```bash
./vello install qwen3-30b-a3b --quant Q4_K_M
```

Or globally bias vello toward smaller quants by lowering `vram_gb` in
`profile.toml`.
</details>

<details>
<summary><b>vello says "could not locate the project root"</b></summary>

Run it from inside the project directory, or set
`VELLO_PROJECT_ROOT=/path/to/vello-llama-local` in your environment.
</details>

---

## Uninstall

The project is fully self-contained. To remove it:

```bash
./vello nuke                                          # containers, volumes, image
rm -f models/*.gguf                                   # downloaded models
rm -rf vello-cli/target vello profile.toml system.toml .env
sudo apt purge nvidia-container-toolkit               # optional, system-wide
rm -rf /path/to/vello-llama-local                     # the repo
```

The `vello-installer` only writes inside the project directory and inside
`/etc/apt/sources.list.d/` (for the NVIDIA toolkit repo, on first install).
The Rust toolchain installed by rustup lives in `~/.cargo/` and `~/.rustup/`
and can be removed with `rustup self uninstall`.

---

## License

MIT. See [LICENSE](LICENSE).
