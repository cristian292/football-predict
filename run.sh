#!/usr/bin/env bash
set -euo pipefail
# Do not depend on unrelated packages in ~/.local on another computer.
export PYTHONNOUSERSITE=1
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${FOOTBALL_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  printf '%s\n' 'Creează mediul conform README.md sau setează FOOTBALL_PYTHON.' >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m football_predictor "$@"
