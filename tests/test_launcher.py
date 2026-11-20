"""Launcher appends system GI bindings without shadowing ML packages."""
from pathlib import Path
import os
import subprocess


def test_launcher_keeps_venv_scientific_packages_first():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['bash', '-x', str(root / 'run.sh'), '--help'], capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': ''})
    assert result.returncode == 0, result.stderr
    exports = [line for line in result.stderr.splitlines() if line.startswith('+ export PYTHONPATH=')]
    assert exports
    assert '/usr/lib/python3/dist-packages' not in exports[-1]
