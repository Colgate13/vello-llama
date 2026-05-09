#!/usr/bin/env bash
# scripts/e2e.sh — End-to-end test for vello-llama, run in-place.
#
# What it does:
#   1. Snapshots the host state we'll touch:
#        - .env, system.toml, profile.toml (copied to a backup dir)
#        - the list of *.gguf files currently in models/
#        - the current vello/llama-server-cuda image (re-tagged to a backup)
#   2. Runs every non-interactive vello CLI command in phases.
#      Each command is logged with PASS / FAIL.
#   3. Always cleans up via trap: `vello down`, `vello nuke -y`, restore the
#      backed-up configs, restore the image tag, delete any model files the
#      test added.
#
# Skipped commands (with reasons):
#   vello update     mutates git state
#   vello build      would rebuild the CUDA image (~10–15 min)
#   vello gpu        interactive (live nvidia-smi)
#   vello logs -f    interactive (we run logs without -f)
#
# Touched on the host:
#   .env, profile.toml, system.toml          → backed up + restored
#   models/<test-model>.gguf                  → removed only if added by the test
#   vello/llama-server-cuda:12.4              → restored from backup tag
#   compose stack (containers + image)        → torn down at end
#
# Never touched:
#   openwebui-data/ (your conversations)
#   models that existed before the test
#
# Usage:
#   scripts/e2e.sh         # asks confirmation
#   scripts/e2e.sh -y      # no prompt
#   scripts/e2e.sh --keep  # keep the backup dir (for inspection)

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TS="$(date +%Y%m%d-%H%M%S)"
readonly BACKUP_DIR="${REPO_ROOT}/.e2e-backup-${TS}"
readonly LOG_DIR="${BACKUP_DIR}/logs"

readonly IMAGE="vello/llama-server-cuda:12.4"
readonly BACKUP_IMAGE="vello-e2e-backup:${TS}"

# Small model — already in the project's default catalog.
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
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         fatal "unknown flag: $arg (try --help)" ;;
  esac
done

# ---------------------------------------------------------------------------
# Result accounting
# ---------------------------------------------------------------------------
declare -a RESULTS=()
PASS=0
FAIL=0

run_step() {
  # run_step <name> <expect_ok: 0|1> -- <command...>
  local name="$1"; shift
  local expect_ok="$1"; shift
  [[ "${1:-}" == "--" ]] && shift
  local logfile="${LOG_DIR}/$(printf "%s" "$name" | tr -c 'A-Za-z0-9' '_').log"

  printf "  %-52s " "$name"
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
# Pre-flight
# ---------------------------------------------------------------------------
preflight() {
  phase "Pre-flight"

  for c in git docker curl jq nvidia-smi; do
    command -v "$c" >/dev/null || fatal "$c not found in PATH"
  done
  ok "tools present"

  docker info >/dev/null 2>&1 || fatal "docker daemon not reachable"
  ok "docker daemon reachable"

  [[ -x "${REPO_ROOT}/vello" ]] || fatal "${REPO_ROOT}/vello not built — run \`make build-vello\`"
  ok "./vello binary present ($(${REPO_ROOT}/vello --version))"

  docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || fatal "image $IMAGE not present — run \`./vello build\` first"
  ok "image $IMAGE present"

  for cn in llama-server open-webui; do
    if docker ps --format '{{.Names}}' | grep -qx "$cn"; then
      fatal "container '$cn' is running — \`./vello down\` first"
    fi
  done
  ok "no stack running"

  cat <<EOF

${C_BLD}Plan:${C_RST}
  - backup dir:    ${BACKUP_DIR}
  - test model:    ${E2E_MODEL_ID}
  - destructive at end:
      vello down · vello nuke (auto-y) · restore configs from backup
      restore ${IMAGE} from backup tag · remove leftover test model file

${C_DIM}This runs in the main repo (${REPO_ROOT}).${C_RST}
${C_DIM}Models that existed before the test are kept; conversations in${C_RST}
${C_DIM}openwebui-data/ are never touched.${C_RST}
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
normalize_baseline() {
  # Make sure the .env we're about to back up points to a model that will
  # survive the test (i.e., NOT the test model). If it points to the test
  # model — typical when a previous run already installed it — switch to
  # any other installed model first, so the restored state is functional.
  local env_file="${REPO_ROOT}/.env"
  local active=""
  [[ -f "$env_file" ]] && active=$(grep -E "^LLAMA_MODEL_FILE=" "$env_file" 2>/dev/null | cut -d= -f2)

  if [[ -z "$active" ]] || [[ ! -f "${REPO_ROOT}/models/${active}" ]] \
     || [[ "$active" =~ ^DeepSeek-R1-Distill-Qwen-1\.5B ]]; then
    info "normalizing baseline (current active='$active' is missing or is the test model)"
    # vello list -i format: "[S]    <id>    <params> <quant> <status> <tags>"
    local first_id
    first_id=$(vello list -i 2>/dev/null \
      | awk '/^\[[A-Z]\]/ && $2 != "'"${E2E_MODEL_ID}"'" { print $2; exit }')
    if [[ -n "$first_id" ]]; then
      info "  switching to '$first_id' as baseline"
      vello switch "$first_id" >/dev/null 2>&1 \
        || warn "switch failed; backup will preserve current state as-is"
    else
      warn "no other installed model found; backup will preserve current state"
    fi
  else
    ok "current active model is fine: $active"
  fi
}

backup_state() {
  phase "Backup"
  mkdir -p "$LOG_DIR"
  normalize_baseline

  for f in .env system.toml profile.toml; do
    if [[ -f "${REPO_ROOT}/${f}" ]]; then
      cp "${REPO_ROOT}/${f}" "${BACKUP_DIR}/${f}"
      ok "saved ${f}"
    else
      touch "${BACKUP_DIR}/${f}.absent"
      ok "${f} did not exist (recorded)"
    fi
  done

  # Snapshot which model files existed before — anything new is ours to remove.
  ( cd "${REPO_ROOT}/models" && ls -1 *.gguf *.mmproj 2>/dev/null || true ) \
    > "${BACKUP_DIR}/pre-models.txt"
  ok "snapshot of models/ ($(wc -l <"${BACKUP_DIR}/pre-models.txt") files)"

  # Tag the image so we can restore it after `vello nuke` removes it.
  docker tag "$IMAGE" "$BACKUP_IMAGE"
  ok "image tagged $BACKUP_IMAGE"
}

restore_state() {
  info "stopping stack (idempotent)"
  vello down >>"${LOG_DIR}/cleanup.log" 2>&1 || true

  info "vello nuke (auto-y)"
  ( cd "$REPO_ROOT" && yes y | ./vello nuke ) >>"${LOG_DIR}/cleanup.log" 2>&1 || true

  info "restoring configs"
  for f in .env system.toml profile.toml; do
    if [[ -f "${BACKUP_DIR}/${f}" ]]; then
      cp "${BACKUP_DIR}/${f}" "${REPO_ROOT}/${f}"
    elif [[ -f "${BACKUP_DIR}/${f}.absent" ]]; then
      rm -f "${REPO_ROOT}/${f}"
    fi
  done
  ok "configs restored"

  if docker image inspect "$BACKUP_IMAGE" >/dev/null 2>&1; then
    info "restoring image tag"
    docker tag "$BACKUP_IMAGE" "$IMAGE"
    docker rmi "$BACKUP_IMAGE" >>"${LOG_DIR}/cleanup.log" 2>&1 || true
    ok "$IMAGE restored"
  fi

  info "removing leftover test model files"
  if [[ -f "${BACKUP_DIR}/pre-models.txt" ]]; then
    ( cd "${REPO_ROOT}/models" && for f in *.gguf *.mmproj; do
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
  local skipped=0
  for line in "${RESULTS[@]}"; do [[ "$line" == SKIP\|* ]] && skipped=$((skipped + 1)); done
  phase "Summary"
  printf "  ${C_GRN}%d passed${C_RST}, ${C_RED}%d failed${C_RST}, ${C_YEL}%d skipped${C_RST}\n\n" \
    "$PASS" "$FAIL" "$skipped"
  for line in "${RESULTS[@]}"; do
    IFS='|' read -r status name logfile <<<"$line"
    case "$status" in
      PASS) printf "  ${C_GRN}PASS${C_RST}  %s\n" "$name" ;;
      SKIP) printf "  ${C_YEL}SKIP${C_RST}  %s\n" "$name" ;;
      FAIL) printf "  ${C_RED}FAIL${C_RST}  %s  ${C_DIM}%s${C_RST}\n" "$name" "$logfile" ;;
    esac
  done
  echo
  if [[ "$FAIL" -gt 0 ]]; then log "Logs (until cleanup): ${LOG_DIR}"; fi
}

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
phase_meta() {
  phase "Phase 0 — meta"
  run_step "vello --version"          1 -- vello --version
  run_step "vello --help"             1 -- vello --help
  run_step "vello list --help"        1 -- vello list --help
}

phase_profile() {
  phase "Phase 1 — profile"
  run_step "vello profile show"       1 -- vello profile show
  run_step "vello profile refresh"    1 -- vello profile refresh
  run_step "vello profile show again" 1 -- vello profile show
}

phase_catalog() {
  phase "Phase 2 — catalog"
  run_step "vello catalog list (default only)" 1 -- vello catalog list

  # Mirror the real schema (id/repo/default_quant/params_total_b/architecture
  # + [model.files] as a quant→filename map). Picks a tiny GGUF that won't
  # actually be downloaded (we never `install` it).
  local extra="${BACKUP_DIR}/test-extra-catalog.toml"
  cat > "$extra" <<'EOF'
schema_version = 1
name           = "test-extra"
maintainer     = "scripts/e2e.sh"

[[model]]
id              = "e2e-tiny-test"
repo            = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
default_quant   = "Q4_K_M"
params_total_b  = 0.5
architecture    = "dense"
tags            = ["test", "small"]
description     = "Synthetic entry for the e2e test (never downloaded)."

[model.files]
Q4_K_M = "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
EOF
  run_step "vello catalog add ./test-extra-catalog.toml" 1 -- \
    vello catalog add "$extra"
  run_step "vello catalog list shows test-extra" 1 -- \
    bash -c "cd '$REPO_ROOT' && ./vello catalog list | grep -q test-extra"
  run_step "vello list contains e2e-tiny-test" 1 -- \
    bash -c "cd '$REPO_ROOT' && ./vello list | grep -q e2e-tiny-test"
  run_step "vello catalog remove test-extra" 1 -- \
    vello catalog remove test-extra
  run_step "vello catalog list (test-extra gone)" 1 -- \
    bash -c "cd '$REPO_ROOT' && ! ./vello catalog list | grep -q test-extra"
}

phase_discovery() {
  phase "Phase 3 — discovery"
  run_step "vello list"                          1 -- vello list
  run_step "vello list -i"                       1 -- vello list -i
  run_step "vello list -t S"                     1 -- vello list -t S
  run_step "vello list -T code"                  1 -- vello list -T code
  run_step "vello list -m image"                 1 -- vello list -m image
  run_step "vello info qwen3-8b"                 1 -- vello info qwen3-8b
  run_step "vello info qwen3-8b -q Q4_K_M"       1 -- vello info qwen3-8b -q Q4_K_M
  run_step "vello recommend chat"                1 -- vello recommend chat
  run_step "vello recommend código (PT)"         1 -- vello recommend código
  run_step "vello recommend reasoning -l 5"      1 -- vello recommend reasoning -l 5
  run_step "vello info <bogus> (expects fail)"   0 -- vello info no-such-model-id-xyz
}

phase_install_lifecycle() {
  phase "Phase 4 — install + lifecycle (downloads ~1.1 GB if not already on disk)"

  run_step "vello install ${E2E_MODEL_ID} -n (download only)" 1 -- \
    vello install "${E2E_MODEL_ID}" -n
  run_step "vello install ${E2E_MODEL_ID} (sets active)" 1 -- \
    vello install "${E2E_MODEL_ID}"
  run_step "vello active mentions the test model" 1 -- \
    bash -c "cd '$REPO_ROOT' && ./vello active 2>&1 | grep -qiE 'deepseek.*1\.5b|DeepSeek.*1\.5'"
  run_step "vello apply --no-restart (rewrites .env)" 1 -- vello apply --no-restart
  run_step ".env exists after apply" 1 -- test -f "${REPO_ROOT}/.env"

  run_step "vello up" 1 -- vello up

  info "polling vello health (max 180s)..."
  local deadline=$(( SECONDS + 180 ))
  local healthy=0
  while [[ $SECONDS -lt $deadline ]]; do
    if vello health >/dev/null 2>&1; then healthy=1; break; fi
    sleep 3
  done
  run_step "vello health (after up)" 1 -- bash -c "[[ $healthy == 1 ]]"

  run_step "vello status (alias ps)"           1 -- vello status
  run_step "vello logs llama-server (no -f)"   1 -- vello logs llama-server
  run_step "vello bench short prompt"          1 -- vello bench "Hello." 16

  # `vello test` exercises tool-calling, which the 1.5B reasoning model can't
  # do. If a tools-tagged model is installed, switch to it for this single
  # check, then come back. Otherwise mark the step SKIP.
  local tools_id
  tools_id=$(vello list -T tools -i 2>/dev/null \
    | awk '/^\[[A-Z]\]/ && $2 != "'"${E2E_MODEL_ID}"'" { print $2; exit }')
  if [[ -n "$tools_id" ]]; then
    info "switching to '$tools_id' for vello test"
    run_step "vello switch ${tools_id} (for tool test)" 1 -- vello switch "$tools_id"
    deadline=$(( SECONDS + 120 ))
    healthy=0
    while [[ $SECONDS -lt $deadline ]]; do
      if vello health >/dev/null 2>&1; then healthy=1; break; fi
      sleep 3
    done
    run_step "vello health (post-switch to ${tools_id})" 1 -- bash -c "[[ $healthy == 1 ]]"
    run_step "vello test (tool calling on ${tools_id})" 1 -- vello test
    run_step "vello switch ${E2E_MODEL_ID} (back to test model)" 1 -- vello switch "${E2E_MODEL_ID}"
  else
    printf "  %-52s ${C_YEL}SKIP${C_RST}  ${C_DIM}no tools-tagged model installed${C_RST}\n" \
      "vello test (tool calling)"
    RESULTS+=("SKIP|vello test (tool calling)|none")
  fi

  run_step "vello restart"                     1 -- vello restart

  info "polling health after restart (max 120s)..."
  deadline=$(( SECONDS + 120 ))
  healthy=0
  while [[ $SECONDS -lt $deadline ]]; do
    if vello health >/dev/null 2>&1; then healthy=1; break; fi
    sleep 3
  done
  run_step "vello health (after restart)" 1 -- bash -c "[[ $healthy == 1 ]]"

  run_step "vello switch ${E2E_MODEL_ID}" 1 -- vello switch "${E2E_MODEL_ID}"
  run_step "vello down"                   1 -- vello down
}

phase_remove() {
  phase "Phase 5 — remove"

  # Can't remove the active model — switch to any other installed model first.
  local other_id
  other_id=$(vello list -i 2>/dev/null \
    | awk '/^\[[A-Z]\]/ && $2 != "'"${E2E_MODEL_ID}"'" { print $2; exit }')
  if [[ -n "$other_id" ]]; then
    run_step "vello switch ${other_id} (so we can remove test model)" 1 -- \
      vello switch "$other_id"
  else
    warn "no other installed model to switch to — remove will fail"
  fi

  run_step "vello remove ${E2E_MODEL_ID} (alias rm)" 1 -- \
    vello remove "${E2E_MODEL_ID}"
  run_step "test model file is gone" 1 -- \
    bash -c "! ls '${REPO_ROOT}'/models/DeepSeek-R1-Distill-Qwen-1.5B-*.gguf 2>/dev/null"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  preflight
  trap cleanup EXIT INT TERM
  backup_state

  phase_meta
  phase_profile
  phase_catalog
  phase_discovery
  phase_install_lifecycle
  phase_remove

  if [[ "$FAIL" -gt 0 ]]; then exit 1; fi
  exit 0
}

main "$@"
