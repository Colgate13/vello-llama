# vello-llama-local

**Run LLMs locally on your GPU. One install command, one model swap, one teardown.**

```bash
git clone https://github.com/Colgate13/vello-llama-local.git
cd vello-llama-local
./vello-llama-local install
```

Open **<http://localhost:3000>** and chat. Done.

---

## What you get

| | |
|---|---|
| **Web chat** | <http://localhost:3000> (Open WebUI) |
| **OpenAI API** | <http://localhost:8080/v1> — works with Cursor, opencode, crush, langchain, anything |
| **Default model** | Qwen 2.5 7B Instruct, ~40 tok/s on RTX 4060 |
| **Tool calling** | works out of the box |
| **6 curated models** | one command to swap |

---

## Cheatsheet

```bash
./vello-llama-local up                 # start
./vello-llama-local down               # stop
./vello-llama-local restart            # restart
./vello-llama-local logs -f            # follow logs
./vello-llama-local status             # are containers up?
./vello-llama-local health             # is the API responding?
./vello-llama-local gpu                # live GPU usage
./vello-llama-local config             # show effective config
./vello-llama-local test               # validate tool calling
./vello-llama-local bench              # measure tokens/sec
./vello-llama-local nuke               # remove everything (keeps models)
```

`make X` is the same as `./vello-llama-local X`.

---

## Models

```bash
./vello-llama-local models list                    # see catalog + status
./vello-llama-local models pull qwen3-8b           # download
./vello-llama-local models use  qwen3-8b           # switch (auto-restarts)
./vello-llama-local models rm   qwen3-8b           # delete the file
```

### Curated catalog

| Name | Best for | Tools? |
|---|---|---|
| `qwen2.5-7b` *(default)* | general chat, agents | yes |
| `qwen2.5-coder-7b` | code autocomplete only | **no** |
| `qwen2.5-coder-14b` | bigger coder (slower on 8 GB) | no |
| `qwen3-8b` | reasoning + tools | yes |
| `llama-3.1-8b` | Meta's instruct | yes |
| `hermes-3-8b` | strongest tool calling | yes |

### Pull anything from HuggingFace

```bash
./vello-llama-local models pull <hf-user>/<hf-repo>/<file.gguf>
```

Example:
```bash
./vello-llama-local models pull bartowski/Phi-3.5-mini-instruct-GGUF/Phi-3.5-mini-instruct-Q5_K_M.gguf
./vello-llama-local models use phi-3.5-mini-instruct-q5_k_m
```

<details>
<summary>How to pick the right quantization</summary>

| Your GPU VRAM | Pick |
|---|---|
| 6 GB | `Q4_K_M` for 7B |
| 8 GB *(RTX 4060)* | `Q5_K_M` for 7B–8B / `Q4_K_M` for 14B |
| 12 GB | `Q6_K` for 7B / `Q5_K_M` for 14B |
| 24 GB | `Q4_K_M` for 32B |

`Q5_K_M` is the size/quality sweet spot.

</details>

---

## Connect a client

### Browser
Already running at <http://localhost:3000>.

### opencode / crush (CLI coding agents)
```bash
./vello-llama-local clients install
opencode    # or: crush
```

### Anything else (Cursor, Continue.dev, langchain, custom apps)
Set the OpenAI provider to:

| Setting | Value |
|---|---|
| `baseURL` | `http://localhost:8080/v1` |
| `apiKey` | any string (e.g. `local`) |
| `model` | `qwen2.5-7b-local` (or whatever `./vello-llama-local config` shows) |

### Raw curl
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b-local","messages":[{"role":"user","content":"Hi"}]}'
```

---

## Configuration

Edit `.env` and run `./vello-llama-local restart`. Common knobs:

| Variable | Purpose | Default |
|---|---|---|
| `LLAMA_PORT` | API port | `8080` |
| `WEBUI_PORT` | Web UI port | `3000` |
| `LLAMA_CTX` | context window | `32768` |
| `LLAMA_THREADS` | CPU cores (not SMT) | `6` |
| `WEBUI_AUTH` | require login on Web UI | `False` |

`./vello-llama-local config` shows everything currently active.

---

## Requirements

- Linux + Docker (user in `docker` group)
- NVIDIA GPU + driver 550+ (any recent driver)
- 10 GB free disk for the runtime, plus ~5 GB per model

That's it. `./vello-llama-local install` checks the rest.

---

## Troubleshooting

<details>
<summary><b>Stack won't start / container unhealthy</b></summary>

```bash
./vello-llama-local logs llama-server
```

Most common causes:
- **Out of VRAM** → lower `LLAMA_CTX` in `.env` or use a smaller model
- **Wrong CUDA version** → see "Older driver" below
- **Model file missing** → `./vello-llama-local models pull <name>`
</details>

<details>
<summary><b>Tool calling returns plain text instead of structured calls</b></summary>

Some models (notably Qwen 2.5 **Coder**) don't follow tool-calling templates well. Switch:
```bash
./vello-llama-local models use qwen2.5-7b   # or hermes-3-8b, llama-3.1-8b
./vello-llama-local test
```
</details>

<details>
<summary><b>"forward compatibility was attempted on non supported HW"</b></summary>

Your NVIDIA driver is older than the CUDA version in the image. In `docker-compose.yml` lower `CUDA_VERSION` to match — for driver 550, use `12.4.1` (the default). Then:
```bash
./vello-llama-local build
```
</details>

<details>
<summary><b>"Open WebUI shows my model under 'OpenAI' — is it remote?"</b></summary>

No. That's just how Open WebUI groups providers by **API protocol**. Your model runs on your GPU. Verify:
```bash
./vello-llama-local config        # see active model file
nvidia-smi                # see VRAM used by /app/llama-server
```
</details>

<details>
<summary><b>I want to use a non-RTX-40 GPU</b></summary>

Edit `docker-compose.yml`:
```yaml
CUDA_ARCH: "86"   # 30-series Ampere
CUDA_ARCH: "75"   # 20-series Turing
CUDA_ARCH: "89"   # 40-series Ada (default)
```
Then `./vello-llama-local build`.
</details>

---

## Uninstall

```bash
./vello-llama-local nuke                              # removes containers, volumes, image
rm -f models/*.gguf                           # delete downloaded models
sudo apt purge nvidia-container-toolkit       # optional: remove GPU toolkit
rm -rf /path/to/vello-llama-local                     # remove project
```

The project doesn't write anywhere outside its own directory.

---

## How it works

<details>
<summary>Architecture</summary>

```
Browser ──:3000──► Open WebUI (container)
                       │
                       ▼ /v1/chat/completions
                   llama-server (container, our local CUDA build)
                       │
                       ▼ mmap
                   ./models/*.gguf
                       │
                       ▼ CUDA passthrough
                   Your GPU
```

- **`llama-server`**: a container running `llama.cpp` we build locally and pin to a CUDA version that matches your driver. Reads a GGUF, loads it on the GPU, exposes an OpenAI-compatible HTTP API.
- **`open-webui`**: the upstream image; a stateless frontend that talks to llama-server over the Docker network. Never touches the internet for inference.
- **`vello-llama-local` script**: one bash file. Read it; it's the source of truth.

</details>

<details>
<summary>Why we build llama.cpp ourselves</summary>

The official `ghcr.io/ggml-org/llama.cpp:server-cuda` is built with CUDA 12.6, which requires NVIDIA driver 555+. Driver 550.x (still common) only supports CUDA 12.4, and consumer RTX cards **don't** support CUDA forward compatibility. So we pin the toolkit to whatever your driver supports. Build is one-time and cached.

</details>

<details>
<summary>Project layout</summary>

```
vello-llama-local/
├── vello-llama-local                  # the CLI (a single bash script)
├── catalog.json               # curated model list (edit to add yours)
├── docker-compose.yml         # stack definition
├── docker/Dockerfile.llama-cuda
├── clients/{opencode,crush}.json
├── models/                    # GGUFs (gitignored)
├── .env / .env.example        # runtime config
└── README.md / LICENSE / Makefile
```

</details>

---

## License

MIT. See [LICENSE](LICENSE).

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp), [Open WebUI](https://github.com/open-webui/open-webui), and quants from [bartowski](https://huggingface.co/bartowski).
