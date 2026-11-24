"""Installer contract tests; never invoke pip or change the real environment."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest


def probe_python(tmp_path, prelude):
    """Run real Python probes with one deliberately broken prerequisite."""
    wrapper = tmp_path / "probe python"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"exec({prelude!r})\n"
        "assert sys.argv[1] == '-c', 'unexpected write/network command'\n"
        "exec(sys.argv[2])\n"
    )
    wrapper.chmod(0o755)
    return wrapper

SOURCE = Path(__file__).resolve().parents[1] / "install.sh"


@pytest.fixture(scope="module")
def native_python():
    python = "/usr/bin/python3"
    if not Path(python).exists():
        pytest.skip("Linux distro Python is not installed")
    result = subprocess.run([
        python, "-c", 'import venv, ensurepip'
    ], env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
       capture_output=True, text=True)
    if result.returncode:
        pytest.skip("Python venv / ensurepip prerequisites unavailable")
    return python



def project(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    if SOURCE.exists():
        shutil.copy2(SOURCE, root / "install.sh")
    return root


def run_install(root, *args, python="/definitely/missing/python", **env):
    return subprocess.run(
        ["bash", str(root / "install.sh"), *args],
        cwd=root.parent,
        env={**os.environ, "FOOTBALL_BOOTSTRAP_PYTHON": str(python), **env},
        capture_output=True, text=True, timeout=10,
    )


def test_help_needs_no_python_or_writes(tmp_path):
    root = project(tmp_path)
    result = run_install(root, "--help")
    assert result.returncode == 0, result.stderr
    for option in ("--help", "--check-only", "--dev", "FOOTBALL_BOOTSTRAP_PYTHON"):
        assert option in result.stdout
    assert "./run.sh" in result.stdout
    assert sorted(p.name for p in root.iterdir()) == ["install.sh"]
    assert os.access(SOURCE, os.X_OK)


@pytest.mark.parametrize("args, message", [
    (("--typo",), "necunoscut"),
    (("--check-only",), "FOOTBALL_BOOTSTRAP_PYTHON"),
])
def test_invalid_invocations_fail_without_writes(tmp_path, args, message):
    root = project(tmp_path)
    result = run_install(root, *args)
    assert result.returncode != 0
    assert message in result.stderr
    assert not (root / ".venv").exists()


def test_python_older_than_311_is_rejected_before_writes(tmp_path):
    root = project(tmp_path)
    python = probe_python(tmp_path, "sys.version_info = (3, 10, 99)")
    result = run_install(root, "--check-only", python=python)
    assert result.returncode != 0
    assert "3.11" in result.stderr
    assert not (root / ".venv").exists()


@pytest.mark.parametrize("missing", ["venv", "ensurepip"])
def test_missing_native_prerequisite_prints_manual_apt_command(tmp_path, missing):
    root = project(tmp_path)
    python = probe_python(tmp_path, f'''
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == {missing!r}:
        raise ImportError("deliberately missing " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
''')
    result = run_install(root, "--check-only", python=python)
    assert result.returncode != 0
    assert "python3-venv" in result.stderr
    assert "GTK" not in result.stderr
    assert "python3-gi" not in result.stderr
    assert "sudo apt install" in result.stderr
    assert not (root / ".venv").exists()


def test_real_native_check_only_is_read_only_from_another_directory(tmp_path, native_python):
    root = project(tmp_path)
    result = run_install(root, "--check-only", "--dev", python="/usr/bin/python3")
    assert result.returncode == 0, result.stderr
    assert "Cerințe verificate" in result.stdout
    assert sorted(p.name for p in root.iterdir()) == ["install.sh"]


def test_bootstrap_does_not_require_gtk(tmp_path):
    root = project(tmp_path)
    python = probe_python(tmp_path, """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'gi':
        raise ImportError('GTK is not a web prerequisite')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
""")
    result = run_install(root, '--check-only', python=python)
    assert result.returncode == 0, result.stderr
    assert not (root / '.venv').exists()


def test_python_probes_disable_user_site_before_running(tmp_path):
    root = project(tmp_path)
    python = probe_python(tmp_path, '''
import os
if os.environ.get("PYTHONNOUSERSITE") != "1":
    sys.exit("User site was not disabled")
sys.version_info = (3, 10, 0)
''')
    result = run_install(root, "--check-only", python=python, PYTHONNOUSERSITE="0")
    assert result.returncode != 0
    assert "User site was not disabled" not in result.stderr
    assert "3.11" in result.stderr


def failing_bootstrap(tmp_path):
    """Delegate read-only probes, but stop before creating a real venv."""
    log = tmp_path / "commands.json"
    python = tmp_path / "bootstrap boundary"
    python.write_text(f'''#!{sys.executable}
import json, os, sys
if sys.argv[1] == "-c":
    os.execv("/usr/bin/python3", ["/usr/bin/python3", *sys.argv[1:]])
with open({str(log)!r}, "w") as stream:
    json.dump(sys.argv[1:], stream)
sys.exit(73)
''')
    python.chmod(0o755)
    return python, log


def test_venv_creation_is_isolated_and_stops_on_failure(tmp_path, native_python):
    root = project(tmp_path)
    python, log = failing_bootstrap(tmp_path)
    result = run_install(root, python=python)
    assert result.returncode != 0
    assert "python3-venv" in result.stderr
    assert json.loads(log.read_text()) == [
        "-m", "venv", str(root / ".venv")
    ]
    assert not (root / ".venv").exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "foreign_python"])
def test_unrelated_venv_is_never_overwritten(tmp_path, kind, native_python):
    root = project(tmp_path)
    target = root / ".venv"
    if kind == "directory":
        target.mkdir()
        (target / "keep.txt").write_text("important")
    elif kind == "file":
        target.write_text("important")
    elif kind == "symlink":
        target.symlink_to(tmp_path / "nonexistent", target_is_directory=True)
    else:
        (target / "bin").mkdir(parents=True)
        (target / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
        (target / "bin/python").symlink_to("/usr/bin/python3")
    before = [(str(p.relative_to(root)), p.read_bytes())
              for p in root.rglob("*") if p.is_file() and not p.is_symlink()]
    python, log = failing_bootstrap(tmp_path)
    result = run_install(root, python=python)
    assert result.returncode != 0
    assert ".venv" in result.stderr
    assert not log.exists(), "must not recreate an existing .venv"
    assert before == [(str(p.relative_to(root)), p.read_bytes())
                      for p in root.rglob("*") if p.is_file() and not p.is_symlink()]


def existing_venv_with_failing_pip(root, tmp_path, prelude="", system_site=True):
    """Create a real venv without pip; intercept pip before any installation."""
    target = root / ".venv"
    subprocess.run([
        "/usr/bin/python3", "-m", "venv", "--without-pip",
        *(["--system-site-packages"] if system_site else []), str(target),
    ], check=True, capture_output=True, text=True)
    python = target / "bin/python"
    python.rename(target / "bin/real-python")
    log = tmp_path / "pip-command.json"
    python.write_text(f'''#!{sys.executable}
import json, os, sys
if sys.argv[1] == "-c":
    os.execv({str(target / "bin/real-python")!r}, [
        {str(target / "bin/real-python")!r}, "-c", {prelude!r} + "\\n" + sys.argv[2], *sys.argv[3:]
    ])
with open({str(log)!r}, "w") as stream:
    json.dump({{"args": sys.argv[1:], "cwd": os.getcwd(), "no_user_site": os.environ.get("PYTHONNOUSERSITE")}}, stream)
sys.exit(79)
''')
    python.chmod(0o755)
    return log


@pytest.mark.parametrize("flags, editable", [((), "."), (("--dev",), ".[dev,scraping]")])
def test_existing_environment_is_reused_with_explicit_local_pip(tmp_path, flags, editable, native_python):
    root = project(tmp_path)
    pip_log = existing_venv_with_failing_pip(root, tmp_path)
    config = (root / ".venv/pyvenv.cfg").read_bytes()
    python, creation_log = failing_bootstrap(tmp_path)
    for _ in range(2):
        result = run_install(root, *flags, python=python)
        assert result.returncode != 0, "pip failure must never report completion"
        assert "pip" in result.stderr
        assert "Avertisment" in result.stderr and "sistem" in result.stderr
        assert "Instalare finalizată" not in result.stdout
        assert not creation_log.exists(), "must reuse the existing environment"
        assert json.loads(pip_log.read_text()) == {
            "args": ["-m", "pip", "install", "-e", editable],
            "cwd": str(root), "no_user_site": "1",
        }
        assert (root / ".venv/pyvenv.cfg").read_bytes() == config


def test_existing_venv_without_gtk_reaches_pip(tmp_path, native_python):
    root = project(tmp_path)
    pip_log = existing_venv_with_failing_pip(root, tmp_path, '''
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "gi":
        raise ImportError("GTK deliberately unavailable inside venv")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
''')
    python, creation_log = failing_bootstrap(tmp_path)
    result = run_install(root, python=python)
    assert result.returncode != 0
    assert "GTK" not in result.stderr
    assert pip_log.exists()
    assert not creation_log.exists()


def test_existing_isolated_environment_is_reused(tmp_path, native_python):
    root = project(tmp_path)
    pip_log = existing_venv_with_failing_pip(root, tmp_path, system_site=False)
    config = (root / '.venv/pyvenv.cfg').read_bytes()
    python, creation_log = failing_bootstrap(tmp_path)
    result = run_install(root, python=python)
    assert result.returncode != 0  # deliberately failing pip
    assert pip_log.exists(), result.stderr
    assert not creation_log.exists()
    assert (root / '.venv/pyvenv.cfg').read_bytes() == config


def test_missing_dataset_warns_but_does_not_prevent_pip(tmp_path, native_python):
    root = project(tmp_path)
    pip_log = existing_venv_with_failing_pip(root, tmp_path)
    python, _ = failing_bootstrap(tmp_path)
    result = run_install(root, python=python)
    assert "matches.csv" in result.stderr
    assert "Avertisment" in result.stderr
    assert pip_log.exists(), "dataset absence must not block environment installation"
    assert result.returncode != 0  # the deliberately failing pip boundary
