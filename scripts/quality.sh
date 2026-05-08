#!/usr/bin/env bash
# Master quality gate. Runs every check, captures stdout+stderr to
# quality-reports/<step>.log, prints a summary table, and exits non-zero
# if any check failed.
#
# Designed to be run both locally and in CI (the workflow uploads
# quality-reports/ as an artifact).
#
# Steps:
#   - cargo-fmt        rustfmt --check
#   - cargo-clippy     -D warnings
#   - cargo-test       unit tests
#   - cargo-build      release build of vello
#   - bash-syntax      bash -n vello-installer
#   - shellcheck       static lint of vello-installer (if installed)
#   - catalog-validate ./vello list (parses default.toml)
#   - compose-validate docker compose config
#   - structure        shebang + executable bit on installer/binary

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS="$REPO/quality-reports"
SUMMARY_MD="$REPORTS/summary.md"
SUMMARY_JSON="$REPORTS/summary.json"

mkdir -p "$REPORTS"
# Fresh slate on every run so old artifacts can't mask new failures.
find "$REPORTS" -mindepth 1 -delete 2>/dev/null || true

if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
  C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'
  C_DIM=$'\033[2m'; C_RST=$'\033[0m'
else
  C_RED=''; C_GRN=''; C_YEL=''; C_DIM=''; C_RST=''
fi

# Ensure cargo is available (works in CI with default rustup install too).
if ! command -v cargo >/dev/null 2>&1; then
  if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
  fi
fi

declare -a STEP_NAMES=()
declare -a STEP_STATUSES=()
declare -a STEP_DURATIONS=()
declare -a STEP_SUMMARIES=()

run_step() {
  local name="$1"; shift
  local summary_fn="${1:-}"; shift || true
  local log="$REPORTS/$name.log"

  printf "${C_DIM}▶${C_RST} %s\n" "$name"
  local start
  start=$(date +%s%N)
  local rc=0
  ( "$@" ) >"$log" 2>&1 || rc=$?
  local end
  end=$(date +%s%N)
  local ms=$(( (end - start) / 1000000 ))

  local sum=""
  if [[ -n "$summary_fn" ]] && declare -F "$summary_fn" >/dev/null; then
    sum=$("$summary_fn" "$log" "$rc" 2>/dev/null || echo "")
  fi

  STEP_NAMES+=("$name")
  STEP_STATUSES+=("$rc")
  STEP_DURATIONS+=("$ms")
  STEP_SUMMARIES+=("$sum")

  local tag
  if [[ "$rc" == "0" ]]; then
    tag="${C_GRN}✓${C_RST}"
  else
    tag="${C_RED}✗${C_RST}"
  fi
  printf "  %s %s ${C_DIM}(%dms)${C_RST} %s\n" "$tag" "$name" "$ms" "${sum:+— $sum}"
  return 0
}

skip_step() {
  local name="$1"
  local reason="$2"
  STEP_NAMES+=("$name")
  STEP_STATUSES+=("skip")
  STEP_DURATIONS+=("0")
  STEP_SUMMARIES+=("$reason")
  printf "  ${C_YEL}-${C_RST} %s ${C_DIM}— skipped (%s)${C_RST}\n" "$name" "$reason"
}

# ---------------------------------------------------------------------------
# Per-step summarizers (read log + exit code, print one short line)
# ---------------------------------------------------------------------------
sum_clippy() {
  local log="$1"
  local warnings errors
  warnings=$(grep -cE '^warning:' "$log" 2>/dev/null || true)
  errors=$(grep -cE '^error(\[E[0-9]+\])?:' "$log" 2>/dev/null || true)
  echo "${errors:-0} errors, ${warnings:-0} warnings"
}

sum_tests() {
  local log="$1"
  # cargo test prints "test result: ok. N passed; M failed; ..." per binary
  awk '
    /^test result:/ {
      gsub(",", " ", $0)
      for (i=1; i<=NF; i++) {
        if ($i ~ /^[0-9]+$/) {
          if ($(i+1) ~ /^passed/) p+=$i
          else if ($(i+1) ~ /^failed/) f+=$i
        }
      }
    }
    END {
      if (p+f > 0) printf "%d/%d passing, %d failing", p, p+f, f
      else print "no test output"
    }
  ' "$log"
}

sum_build() {
  local log="$1"
  if grep -q 'Finished `release`' "$log"; then
    echo "release build ok"
  elif grep -q '^error' "$log"; then
    echo "compile errors"
  else
    echo "see log"
  fi
}

sum_passthrough() {
  local log="$1"; local rc="$2"
  if [[ "$rc" == "0" ]]; then echo "ok"; else echo "see log"; fi
}

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
echo "${C_DIM}== quality gate (${REPO}) ==${C_RST}"
echo

run_step cargo-fmt sum_passthrough \
  cargo fmt --manifest-path vello-cli/Cargo.toml --check

run_step cargo-clippy sum_clippy \
  cargo clippy --manifest-path vello-cli/Cargo.toml --release --all-targets -- -D warnings

run_step cargo-test sum_tests \
  cargo test --manifest-path vello-cli/Cargo.toml --release

run_step cargo-build sum_build \
  cargo build --manifest-path vello-cli/Cargo.toml --release

run_step bash-syntax sum_passthrough bash -n vello-installer

if command -v shellcheck >/dev/null 2>&1; then
  run_step shellcheck sum_passthrough shellcheck -x vello-installer scripts/quality.sh
else
  skip_step shellcheck "shellcheck not installed"
fi

# Catalog validation: vello must parse default.toml without error.
run_step catalog-validate sum_passthrough bash -c '
  ./vello-cli/target/release/vello list >/dev/null
'

# Docker compose YAML validation (no daemon needed for `config`).
if command -v docker >/dev/null 2>&1; then
  run_step compose-validate sum_passthrough docker compose config -q
else
  skip_step compose-validate "docker not installed"
fi

run_step structure sum_passthrough bash -c '
  set -e
  test -x vello-installer || { echo "vello-installer not executable" >&2; exit 1; }
  head -1 vello-installer | grep -q "^#!/usr/bin/env bash" || { echo "missing bash shebang" >&2; exit 1; }
  test -f vello || { echo "vello binary symlink missing — run: make build-vello" >&2; exit 1; }
  test -d catalogs || { echo "catalogs/ missing" >&2; exit 1; }
  test -f catalogs/default.toml || { echo "default.toml missing" >&2; exit 1; }
  echo "structure ok"
'

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
total_ok=0
total_fail=0
total_skip=0
for s in "${STEP_STATUSES[@]}"; do
  case "$s" in
    0) total_ok=$((total_ok+1)) ;;
    skip) total_skip=$((total_skip+1)) ;;
    *) total_fail=$((total_fail+1)) ;;
  esac
done

# Markdown summary
{
  echo "# Quality Gate Report"
  echo
  echo "$total_ok passed · $total_fail failed · $total_skip skipped"
  echo
  echo "| Step | Status | Time | Summary |"
  echo "|---|---|---|---|"
  for i in "${!STEP_NAMES[@]}"; do
    local_name="${STEP_NAMES[$i]}"
    local_status="${STEP_STATUSES[$i]}"
    local_ms="${STEP_DURATIONS[$i]}"
    local_sum="${STEP_SUMMARIES[$i]}"
    case "$local_status" in
      0)    icon="✅" ;;
      skip) icon="⏭️" ;;
      *)    icon="❌" ;;
    esac
    echo "| \`$local_name\` | $icon | ${local_ms}ms | ${local_sum:--} |"
  done
} >"$SUMMARY_MD"

# JSON summary (consumed by scripts/pr-comment.sh and external tools)
{
  echo '{'
  echo "  \"ok_count\": $total_ok,"
  echo "  \"fail_count\": $total_fail,"
  echo "  \"skip_count\": $total_skip,"
  echo '  "steps": ['
  last=$((${#STEP_NAMES[@]} - 1))
  for i in "${!STEP_NAMES[@]}"; do
    sep=","
    [[ "$i" == "$last" ]] && sep=""
    name=$(printf '%s' "${STEP_NAMES[$i]}" | sed 's/"/\\"/g')
    status="${STEP_STATUSES[$i]}"
    ms="${STEP_DURATIONS[$i]}"
    summary=$(printf '%s' "${STEP_SUMMARIES[$i]}" | sed 's/"/\\"/g')
    ok="false"
    [[ "$status" == "0" ]] && ok="true"
    skipped="false"
    [[ "$status" == "skip" ]] && skipped="true"
    printf '    {"name":"%s","ok":%s,"skipped":%s,"ms":%s,"summary":"%s","artifact":"%s.log"}%s\n' \
      "$name" "$ok" "$skipped" "$ms" "$summary" "$name" "$sep"
  done
  echo '  ]'
  echo '}'
} >"$SUMMARY_JSON"

echo "$total_ok passed · $total_fail failed · $total_skip skipped"
echo "Reports: $REPORTS/"

if [[ "$total_fail" -gt 0 ]]; then
  exit 1
fi
exit 0
