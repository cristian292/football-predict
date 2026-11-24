"""Offline installation diagnostics; importing this module has no side effects."""
import csv
import importlib
import importlib.metadata
from pathlib import Path
import sys


def check_environment(data_path: Path | str | None = None) -> dict:
    """Return JSON-safe checks without installing anything or accessing the network."""
    checks = [{"name": "Python", "ok": sys.version_info >= (3, 11),
               "detail": f"{'.'.join(map(str, sys.version_info[:3]))} (necesar >=3.11)"}]
    for name in ("numpy", "pandas", "scipy", "sklearn", "flask", "waitress"):
        try:
            module = importlib.import_module(name)
            version = (importlib.metadata.version(name) if name in {"flask", "waitress"}
                       else module.__version__)
            checks.append({"name": name, "ok": True,
                           "detail": f"{version} — {module.__file__}"})
        except Exception as error:
            checks.append({"name": name, "ok": False,
                           "detail": f"Import indisponibil: {type(error).__name__}: {error}"})
    if data_path is not None:
        path = Path(data_path)
        try:
            if not path.is_file():
                raise FileNotFoundError("fișier inexistent sau nu este fișier")
            with path.open(encoding="utf-8-sig", newline="") as handle:
                columns = next(csv.reader(handle, strict=True), [])
            required = {"date", "season", "venue", "team", "opponent", "gf", "ga"}
            missing = required.difference(columns)
            if missing:
                raise ValueError(f"Coloane obligatorii lipsă: {', '.join(sorted(missing))}")
            checks.append({"name": "Date CSV", "ok": True,
                           "detail": f"{path}: schemă CSV validă (doar antetul; conținutul nu a fost validat)"})
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            checks.append({"name": "Date CSV", "ok": False,
                           "detail": f"{path}: {type(error).__name__}: {error}"})
    warnings = ["soccerdata este opțional, doar pentru colectarea datelor; "
                "aplicația cu CSV local nu îl necesită (nu este verificat)."]
    return {"ok": all(item["ok"] for item in checks), "checks": checks, "warnings": warnings}


def format_report(report: dict) -> str:
    """Render a Romanian report; the caller decides whether to print or exit."""
    lines = ["Diagnostic mediu și date"]
    for check in report["checks"]:
        status = "OK" if check["ok"] else "EROARE"
        lines.append(f"[{status}] {check['name']}: {check['detail']}")
    lines.extend(f"AVERTISMENT: {warning}" for warning in report["warnings"])
    lines.append("Rezultat: " + ("OK" if report["ok"] else "EROARE"))
    return "\n".join(lines)

