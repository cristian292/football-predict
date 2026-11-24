#!/usr/bin/env bash
# Bootstrap the local environment; never install system packages automatically.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECK_ONLY=0
DEV=0
for arg in "$@"; do
  case "$arg" in
    --help|-h)
      cat <<'HELP'
Utilizare: ./install.sh [--check-only] [--dev] [--help]
  --check-only  Verifică cerințele, fără scrieri sau acces la rețea.
  --dev         Instalează și dependențele de testare și scraping.
  --help        Afișează acest ajutor.
FOOTBALL_BOOTSTRAP_PYTHON: interpretul pentru inițializare (implicit python3).
După instalare, pornește aplicația cu ./run.sh din directorul proiectului.
HELP
      exit 0 ;;
    --check-only) CHECK_ONLY=1 ;;
    --dev) DEV=1 ;;
    *) printf 'Argument necunoscut: %s. Folosește --help.\n' "$arg" >&2; exit 2 ;;
  esac
done
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
BOOTSTRAP="${FOOTBALL_BOOTSTRAP_PYTHON:-python3}"
if ! command -v -- "$BOOTSTRAP" >/dev/null 2>&1; then
  printf 'Python indisponibil: %s. Setează FOOTBALL_BOOTSTRAP_PYTHON la un Python >=3.11.\n' "$BOOTSTRAP" >&2
  exit 1
fi
if ! "$BOOTSTRAP" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  printf '%s\n' 'Este necesar Python >=3.11. Setează FOOTBALL_BOOTSTRAP_PYTHON.' >&2
  exit 1
fi
native_help() {
  printf '%s\n' \
    'Lipsesc cerințe Python venv/ensurepip. Instalează manual pe Debian/Ubuntu/Parrot:' \
    '  sudo apt update' \
    '  sudo apt install python3-venv' \
    'Apoi rulează din nou cu FOOTBALL_BOOTSTRAP_PYTHON=/usr/bin/python3.' >&2
}
if ! "$BOOTSTRAP" -c 'import venv, ensurepip'; then
  native_help
  exit 1
fi
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"
validate_venv() {
  if [[ -L "$VENV" || ! -f "$VENV/pyvenv.cfg" || ! -x "$PYTHON" ]]; then
    printf '%s\n' 'Refuz să modific .venv: nu este un mediu virtual local valid. Mută-l manual și reîncearcă.' >&2
    return 1
  fi
  if ! "$PYTHON" -c '
import pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
config = dict(line.split("=", 1) for line in (root / "pyvenv.cfg").read_text().splitlines() if "=" in line)
config = {key.strip(): value.strip() for key, value in config.items()}
if config.get("include-system-site-packages", "").lower() == "true":
    print("Avertisment: .venv moștenește biblioteci de sistem; mediul existent este păstrat, nu este izolat complet.", file=sys.stderr)
sys.exit(0 if sys.version_info >= (3, 11) and sys.prefix != sys.base_prefix
         and pathlib.Path(sys.prefix).resolve() == root and config.get("home")
         and config.get("include-system-site-packages", "").lower() in {"true", "false"} else 1)
' "$VENV"; then
    printf '%s\n' 'Mediul .venv este incompatibil: necesită un mediu virtual Python >=3.11 valid. Mută-l manual; nu va fi suprascris.' >&2
    return 1
  fi
}
if [[ -e "$VENV" || -L "$VENV" ]]; then
  validate_venv || exit 1
fi
if (( CHECK_ONLY )); then
  printf '%s\n' 'Cerințe verificate. Nicio instalare efectuată.'
  exit 0
fi
if [[ ! -e "$VENV" ]]; then
  if ! "$BOOTSTRAP" -m venv "$VENV"; then
    native_help
    printf '%s\n' 'Crearea mediului a eșuat; fișierele existente nu au fost șterse.' >&2
    exit 1
  fi
fi
validate_venv || exit 1
cd -- "$ROOT"
if [[ ! -f "$ROOT/matches.csv" ]]; then
  printf '%s\n' 'Avertisment: matches.csv lipsește. Instalarea continuă; adaugă setul de date înainte de analiză.' >&2
fi
EDITABLE='.'
if (( DEV )); then
  EDITABLE='.[dev,scraping]'
fi
if ! "$PYTHON" -m pip install -e "$EDITABLE"; then
  printf '%s\n' 'Instalarea pip a eșuat. Verifică mesajul de mai sus și conexiunea, apoi rulează din nou ./install.sh.' >&2
  exit 1
fi
printf '\n%s\n' 'Instalare finalizată. Pornește aplicația din directorul proiectului:'
printf '  cd %q && ./run.sh\n' "$ROOT"
