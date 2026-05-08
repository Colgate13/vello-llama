#!/usr/bin/env bash
# Build a Markdown PR-comment body from quality-reports/summary.md.
# Appends collapsible <details> blocks with the tail of each failed step's log.
# Output goes to stdout (the workflow redirects to a file).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS="$ROOT/quality-reports"
HEADING="${PR_COMMENT_HEADING:-Quality Gate}"
MAX_LINES=80
MAX_CHARS=6000

if [[ ! -f "$REPORTS/summary.md" || ! -f "$REPORTS/summary.json" ]]; then
  echo "Missing quality-reports/summary.{md,json}" >&2
  exit 1
fi

# Replace the canonical heading with the caller's preferred one.
sed "s/^# Quality Gate Report/# ${HEADING//\//\\/}/" "$REPORTS/summary.md"

# Extract failed steps from JSON (no jq dependency: simple grep + sed).
# Each failed step gets a collapsible block with the tail of its log.
fails=$(grep -oE '"name":"[^"]*","ok":false,"skipped":false,"ms":[0-9]*,"summary":"[^"]*","artifact":"[^"]*"' "$REPORTS/summary.json" || true)

if [[ -n "$fails" ]]; then
  echo
  echo
  echo "## Failure logs (tail)"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    name=$(echo "$line" | sed -nE 's/.*"name":"([^"]*)".*/\1/p')
    summary=$(echo "$line" | sed -nE 's/.*"summary":"([^"]*)".*/\1/p')
    artifact=$(echo "$line" | sed -nE 's/.*"artifact":"([^"]*)".*/\1/p')
    log="$REPORTS/$artifact"
    echo
    if [[ ! -f "$log" ]]; then
      echo "_${name}: no log available._"
      continue
    fi
    tail_text=$(tail -n "$MAX_LINES" "$log" | tail -c "$MAX_CHARS")
    echo "<details><summary><code>${name}</code> — ${summary:-failed}</summary>"
    echo
    echo '```'
    echo "$tail_text"
    echo '```'
    echo
    echo "</details>"
  done <<< "$fails"
fi
