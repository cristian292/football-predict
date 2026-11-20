#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${FOOTBALL_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  printf '%s\n' 'Creează mediul conform README.md sau setează FOOTBALL_PYTHON.' >&2
  exit 1
fi
# Append at runtime, not in PYTHONPATH: venv ML packages keep precedence.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -c 'import sys, runpy; sys.path.append("/usr/lib/python3/dist-packages"); runpy.run_module("football_predictor", run_name="__main__")' "$@"
