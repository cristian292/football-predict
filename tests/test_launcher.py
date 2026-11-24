"""Launcher runs the web CLI without injecting native GTK bindings."""
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


def test_launcher_disables_personal_python_packages(tmp_path):
    import sys
    root = Path(__file__).resolve().parents[1]
    probe = tmp_path / 'python-probe'
    probe.write_text(f'#!{sys.executable}\nimport sys\nprint(sys.flags.no_user_site)\n')
    probe.chmod(0o755)
    env = {**os.environ, 'FOOTBALL_PYTHON': str(probe)}
    env.pop('PYTHONNOUSERSITE', None)
    result = subprocess.run(['bash', str(root / 'run.sh'), '--help'], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '1'


def test_launcher_executes_module_and_forwards_arguments(tmp_path):
    import json
    import sys
    root = Path(__file__).resolve().parents[1]
    probe = tmp_path / 'python-probe'
    probe.write_text(f'#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
    probe.chmod(0o755)
    args = ['--host', '127.0.0.1', '--port', '9090', '--work-dir', 'directory with spaces']
    result = subprocess.run(['bash', str(root / 'run.sh'), *args], capture_output=True, text=True,
                            env={**os.environ, 'FOOTBALL_PYTHON': str(probe)})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ['-m', 'football_predictor', *args]
