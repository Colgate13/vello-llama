#!/usr/bin/env bash
# scripts/e2e-cpu.sh — end-to-end test of the CPU runtime path.
#
# Mirror of scripts/e2e.sh, but exercises the vello/llama-server-cpu image
# instead of the CUDA one. Designed to run in CI on a plain ubuntu-latest
# runner (no GPU required).
#
# What it does:
#   1. Snapshots host state (mode in profile.toml, .env, current installed
#      models) so we can restore at the end.
#   2. Forces CPU mode via `vello doctor --cpu --yes`.
#   3. Builds the CPU Docker image (~3-5 min, cached after first run).
#   4. Installs the smallest tagged model (deepseek-r1-distill-qwen-1.5b,
#      ~1 GB at Q5_K_M) and brings the stack up.
#   5. Polls /health, sends a tiny chat completion, asserts the response is
#      sane (non-empty text, no traceback).
#   6. Tears down: vello down, restore mode, prune anything the test added.
#
# Skipped vs scripts/e2e.sh: nothing GPU-specific (passthrough, nvidia-smi,
# anything with --gpus). Includes the doctor and target-filter checks.
#
# Usage:
#   scripts/e2e-cpu.sh         # asks for confirmation
#   scripts/e2e-cpu.sh -y      # no prompt (CI default)
#   scripts/e2e-cpu.sh --keep  # keep backup dir + image for inspection

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TS="$(date +%Y%m%d-%H%M%S)"
readonly BACKUP_DIR="${REPO_ROOT}/.e2e-cpu-backup-${TS}"
readonly LOG_DIR="${BACKUP_DIR}/logs"

readonly CPU_IMAGE="vello/llama-server-cpu:latest"

# Smallest tagged model in the default catalog — Q5 ~ 1 GB.
readonly E2E_MODEL_ID="deepseek-r1-distill-qwen-1.5b"

# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
  C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'
  C_CYN=$'\033[36m'; C_DIM=$'\033[2m'; C_BLD=$'\033[1m'; C_RST=$'\033[0m'
else
  C_RED=''; C_GRN=''; C_YEL=''; C_CYN=''; C_DIM=''; C_BLD=''; C_RST=''
fi
log()    { printf "%s\n" "$*"; }
info()   { printf "${C_CYN}==>${C_RST} %s\n" "$*"; }
ok()     { printf "${C_GRN} ok ${C_RST} %s\n" "$*"; }
warn()   { printf "${C_YEL}warn${C_RST} %s\n" "$*" >&2; }
err()    { printf "${C_RED}fail${C_RST} %s\n" "$*" >&2; }
fatal()  { err "$*"; exit 2; }
phase()  { printf "\n${C_BLD}── %s ──${C_RST}\n" "$*"; }

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
ASSUME_YES=0
KEEP_BACKUP=0
for arg in "$@"; do
  case "$arg" in
    -y|--yes)  ASSUME_YES=1 ;;
    --keep)    KEEP_BACKUP=1 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         fatal "unknown flag: $arg (try --help)" ;;
  esac
done

# ---------------------------------------------------------------------------
# Result accounting (identical to e2e.sh)
# ---------------------------------------------------------------------------
declare -a RESULTS=()
PASS=0
FAIL=0

run_step() {
  local name="$1"; shift
  local expect_ok="$1"; shift
  [[ "${1:-}" == "--" ]] && shift
  local logfile="${LOG_DIR}/$(printf "%s" "$name" | tr -c 'A-Za-z0-9' '_').log"

  printf "  %-58s " "$name"
  local rc=0
  ( "$@" ) >"$logfile" 2>&1 || rc=$?

  local pass=0
  if [[ "$expect_ok" == "1" && "$rc" -eq 0 ]]; then pass=1
  elif [[ "$expect_ok" == "0" && "$rc" -ne 0 ]]; then pass=1
  fi

  if [[ "$pass" == "1" ]]; then
    printf "${C_GRN}PASS${C_RST}\n"
    RESULTS+=("PASS|${name}|${logfile}")
    PASS=$((PASS + 1))
  else
    printf "${C_RED}FAIL${C_RST}  ${C_DIM}rc=${rc} log=${logfile}${C_RST}\n"
    RESULTS+=("FAIL|${name}|${logfile}")
    FAIL=$((FAIL + 1))
  fi
}

vello() { ( cd "$REPO_ROOT" && ./vello "$@" ); }

# ---------------------------------------------------------------------------
# Pre-flight (dogfood vello doctor; skip GPU-specific image check)
# ---------------------------------------------------------------------------
preflight() {
  phase "Pre-flight (CPU mode)"

  [[ -x "${REPO_ROOT}/vello" ]] || fatal "${REPO_ROOT}/vello not built — run \`make build-vello\`"
  ok "./vello binary present ($(${REPO_ROOT}/vello --version))"

  command -v jq >/dev/null || fatal "jq not in PATH (needed to parse vello doctor --json)"
  command -v docker >/dev/null || fatal "docker not in PATH"
  docker info >/dev/null 2>&1 || fatal "docker daemon not reachable"
  ok "docker daemon reachable"

  for cn in llama-server open-webui; do
    if docker ps --format '{{.Names}}' | grep -qx "$cn"; then
      fatal "container '$cn' is running — \`./vello down\` first"
    fi
  done
  ok "no stack running"

  cat <<EOF

${C_BLD}Plan:${C_RST}
  - backup dir:    ${BACKUP_DIR}
  - test model:    ${E2E_MODEL_ID} (~1 GB at Q5)
  - CPU image:     ${CPU_IMAGE} (will build if absent, ~3-5 min)
  - destructive at end:
      vello down · restore mode and configs from backup
      remove leftover test model file (only if added by this test)

${C_DIM}This runs in the main repo (${REPO_ROOT}).${C_RST}
${C_DIM}Models that existed before the test are kept. Open WebUI data is${C_RST}
${C_DIM}never touched. The CPU image is kept unless --keep is unset (we${C_RST}
${C_DIM}don't remove it because it's reusable and slow to rebuild).${C_RST}
EOF

  if [[ "$ASSUME_YES" != "1" ]]; then
    printf "\nProceed? [y/N] "
    read -r ans
    [[ "$ans" =~ ^[Yy]$ ]] || { warn "aborted by user"; exit 0; }
  fi
}

# ---------------------------------------------------------------------------
# Backup / restore
# ---------------------------------------------------------------------------
backup_state() {
  phase "Backup"
  mkdir -p "$LOG_DIR"

  for f in .env system.toml profile.toml; do
    if [[ -f "${REPO_ROOT}/${f}" ]]; then
      cp "${REPO_ROOT}/${f}" "${BACKUP_DIR}/${f}"
      ok "saved ${f}"
    else
      touch "${BACKUP_DIR}/${f}.absent"
      ok "${f} did not exist (recorded)"
    fi
  done

  ( cd "${REPO_ROOT}/models" 2>/dev/null && ls -1 *.gguf 2>/dev/null || true ) \
    > "${BACKUP_DIR}/pre-models.txt"
  ok "snapshot of models/ ($(wc -l <"${BACKUP_DIR}/pre-models.txt") files)"
}

restore_state() {
  info "stopping stack (idempotent)"
  vello down >>"${LOG_DIR}/cleanup.log" 2>&1 || true

  info "restoring configs"
  for f in .env system.toml profile.toml; do
    if [[ -f "${BACKUP_DIR}/${f}" ]]; then
      cp "${BACKUP_DIR}/${f}" "${REPO_ROOT}/${f}"
    elif [[ -f "${BACKUP_DIR}/${f}.absent" ]]; then
      rm -f "${REPO_ROOT}/${f}"
    fi
  done
  ok "configs restored"

  info "removing leftover test model files"
  if [[ -f "${BACKUP_DIR}/pre-models.txt" ]]; then
    ( cd "${REPO_ROOT}/models" && for f in *.gguf; do
        [[ -f "$f" ]] || continue
        if ! grep -qFx "$f" "${BACKUP_DIR}/pre-models.txt"; then
          rm -f "$f" && echo "  removed $f" >>"${LOG_DIR}/cleanup.log"
        fi
      done )
  fi
  ok "models/ pruned to pre-test state"
}

cleanup() {
  local rc=$?
  phase "Cleanup"
  mkdir -p "$LOG_DIR" 2>/dev/null || true
  restore_state
  if [[ "$KEEP_BACKUP" == "1" ]]; then
    warn "keeping backup at ${BACKUP_DIR}"
  else
    rm -rf "$BACKUP_DIR"
  fi
  print_summary
  exit "$rc"
}

print_summary() {
  phase "Summary"
  printf "  ${C_GRN}%d passed${C_RST}, ${C_RED}%d failed${C_RST}\n\n" "$PASS" "$FAIL"
  for line in "${RESULTS[@]}"; do
    IFS='|' read -r status name logfile <<<"$line"
    case "$status" in
      PASS) printf "  ${C_GRN}PASS${C_RST}  %s\n" "$name" ;;
      FAIL) printf "  ${C_RED}FAIL${C_RST}  %s  ${C_DIM}%s${C_RST}\n" "$name" "$logfile" ;;
    esac
  done
  echo
  if [[ "$FAIL" -gt 0 ]]; then log "Logs (until cleanup): ${LOG_DIR}"; fi
}

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
phase_mode_switch() {
  phase "Phase 1 — switch to CPU mode"
  run_step "vello doctor --cpu --yes (persist)"          1 -- vello doctor --cpu --yes
  run_step "profile.toml has mode = \"cpu\""             1 -- \
    bash -c "grep -q '^mode = \"cpu\"' '$REPO_ROOT/profile.toml'"
  run_step "vello doctor (json) reports mode=cpu"        1 -- \
    bash -c "cd '$REPO_ROOT' && ./vello doctor --json | jq -e '.summary.mode == \"cpu\"' >/dev/null"
}

phase_catalog_filtering() {
  phase "Phase 2 — catalog filter in CPU mode"

  # In CPU mode, vello list hides gpu-only entries by default.
  run_step "vello list hides gpu-only by default"        1 -- \
    bash -c "cd '$REPO_ROOT' && filtered=\$(./vello list 2>/dev/null | grep -cE '^\\[[A-Z]\\]'); all=\$(./vello list --all 2>/dev/null | grep -cE '^\\[[A-Z]\\]'); [[ \$filtered -lt \$all && \$all -ge 20 ]]"

  # Specific gpu-only models should be absent from filtered output and
  # present with --all.
  run_step "qwq-32b absent from default list"            1 -- \
    bash -c "cd '$REPO_ROOT' && ! ./vello list 2>/dev/null | grep -q 'qwq-32b'"
  run_step "qwq-32b present with --all"                  1 -- \
    bash -c "cd '$REPO_ROOT' && ./vello list --all 2>/dev/null | grep -q 'qwq-32b'"

  # cpu-friendly small models should be visible.
  run_step "phi-4-mini visible (cpu-friendly)"           1 -- \
    bash -c "cd '$REPO_ROOT' && ./vello list 2>/dev/null | grep -q 'phi-4-mini'"

  # Install a gpu-only model in CPU mode → must bail with clear error.
  run_step "install qwq-32b in cpu mode bails"           0 -- \
    bash -c "cd '$REPO_ROOT' && ./vello install qwq-32b -n"

  # Recommendation in CPU mode picks a tiny/MoE top entry.
  run_step "recommend chat top result is cpu-friendly"   1 -- \
    bash -c "cd '$REPO_ROOT' && ./vello recommend chat 2>/dev/null | grep -qE '(phi-4-mini|llama-3.2-3b|deepseek-r1-distill-qwen-1.5b|qwen3-30b-a3b)'"
}

phase_env_overrides() {
  phase "Phase 3 — CPU .env overrides via vello apply"

  # Need an active model to call apply. The test model gets installed in Phase 4
  # for the real stack; for now install it -n (download only, sets active in
  # apply_model). To keep this phase fast, do a download but skip the rest.
  run_step "vello install ${E2E_MODEL_ID} (download + apply)" 1 -- \
    vello install "${E2E_MODEL_ID}"

  # Verify the resolver wrote CPU-flavored values into .env.
  run_step ".env: LLAMA_NGL=0"                            1 -- \
    bash -c "grep -q '^LLAMA_NGL=0' '$REPO_ROOT/.env'"
  run_step ".env: LLAMA_FLASH_ATTN=off"                   1 -- \
    bash -c "grep -q '^LLAMA_FLASH_ATTN=off' '$REPO_ROOT/.env'"
  run_step ".env: LLAMA_CTX ≤ 8192"                       1 -- \
    bash -c "ctx=\$(grep '^LLAMA_CTX=' '$REPO_ROOT/.env' | cut -d= -f2); [[ \$ctx -le 8192 ]]"
  run_step ".env: LLAMA_BATCH ≤ 512"                      1 -- \
    bash -c "b=\$(grep '^LLAMA_BATCH=' '$REPO_ROOT/.env' | cut -d= -f2); [[ \$b -le 512 ]]"
  run_step ".env: LLAMA_UBATCH ≤ 128"                     1 -- \
    bash -c "u=\$(grep '^LLAMA_UBATCH=' '$REPO_ROOT/.env' | cut -d= -f2); [[ \$u -le 128 ]]"
  run_step ".env: COMPOSE_PROFILES=cpu"                   1 -- \
    bash -c "grep -q '^COMPOSE_PROFILES=cpu' '$REPO_ROOT/.env'"
  run_step ".env: LLAMA_RUNTIME=cpu"                      1 -- \
    bash -c "grep -q '^LLAMA_RUNTIME=cpu' '$REPO_ROOT/.env'"
}

phase_build_and_run() {
  phase "Phase 4 — build CPU image + up + chat completion"

  # Build the CPU image. If already present, this is a no-op for compose.
  run_step "vello build (cpu image, may take ~3-5 min)"   1 -- vello build
  run_step "image vello/llama-server-cpu:latest exists"   1 -- \
    bash -c "docker image inspect '$CPU_IMAGE' >/dev/null 2>&1"

  # Bring up the stack with the CPU profile.
  run_step "vello up"                                     1 -- vello up

  # Health-poll for up to 3 minutes — CPU model load is slower than GPU.
  info "polling vello health (max 180s)..."
  local deadline=$(( SECONDS + 180 ))
  local healthy=0
  while [[ $SECONDS -lt $deadline ]]; do
    if vello health >/dev/null 2>&1; then healthy=1; break; fi
    sleep 3
  done
  run_step "vello health (after up)"                      1 -- bash -c "[[ $healthy == 1 ]]"

  run_step "container llama-server is Up"                 1 -- \
    bash -c "docker ps --format '{{.Names}} {{.Status}}' | grep -q '^llama-server Up'"

  # Send a tiny chat completion request via curl. The response must contain
  # JSON with content; we don't care about quality.
  run_step "chat completion returns non-empty content"    1 -- \
    bash -c "
      port=\$(grep '^LLAMA_PORT=' '$REPO_ROOT/.env' | cut -d= -f2)
      alias=\$(grep '^LLAMA_MODEL_ALIAS=' '$REPO_ROOT/.env' | cut -d= -f2)
      body=\$(printf '{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":12}' \"\$alias\")
      response=\$(curl -fsS --max-time 90 -H 'Content-Type: application/json' \
        -d \"\$body\" \"http://localhost:\${port:-8080}/v1/chat/completions\")
      echo \"\$response\" | jq -e '.choices[0].message.content | length > 0' >/dev/null
    "

  run_step "vello down"                                   1 -- vello down
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  preflight
  trap cleanup EXIT INT TERM
  backup_state

  phase_mode_switch
  phase_catalog_filtering
  phase_env_overrides
  phase_build_and_run

  if [[ "$FAIL" -gt 0 ]]; then exit 1; fi
  exit 0
}

main "$@"
