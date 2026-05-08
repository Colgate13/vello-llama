#!/usr/bin/env bash
# Fail if non-test code in vello-cli/src/ uses raw `.unwrap()` or `panic!()`.
# Test modules are gated by `#[cfg(test)]` and exempt; we detect them by
# locating the first `#[cfg(test)]` line in each file and ignoring hits below
# that point.
#
# This is a lightweight invariant check — clippy with -D warnings already
# catches most issues, but it's worth being explicit about no panics in
# library/CLI code.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/vello-cli/src"

if [[ ! -d "$SRC" ]]; then
  echo "no such dir: $SRC" >&2
  exit 1
fi

violations=0

while IFS= read -r file; do
  test_gate=$(grep -n '#\[cfg(test)\]' "$file" | head -1 | cut -d: -f1 || true)
  while IFS=: read -r lineno _; do
    [[ -z "$lineno" ]] && continue
    # Skip if the hit is inside a #[cfg(test)] block.
    if [[ -n "$test_gate" ]] && (( lineno > test_gate )); then
      continue
    fi
    line=$(sed -n "${lineno}p" "$file")
    # Skip valid variants: unwrap_or, unwrap_or_else, unwrap_or_default.
    if echo "$line" | grep -qE '\.unwrap_(or|or_else|or_default)\b'; then
      continue
    fi
    # Skip comments.
    if echo "$line" | grep -qE '^\s*//'; then
      continue
    fi
    echo "$file:$lineno: $line"
    violations=$((violations + 1))
  done < <(grep -nE '\.unwrap\(\)|panic!\(' "$file" || true)
done < <(find "$SRC" -name '*.rs' -type f)

if (( violations > 0 )); then
  echo
  echo "Found $violations raw unwrap()/panic!() usage(s) in non-test code." >&2
  echo "Use ? / .ok_or / anyhow::bail / Result instead." >&2
  exit 1
fi

echo "ok: no raw unwrap()/panic!() in non-test code"
