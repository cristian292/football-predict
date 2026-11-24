"""Container distribution contracts; no engine, network, or GTK display needed."""
from pathlib import Path
import json
import os
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def instructions(stage):
    text = (ROOT / "Dockerfile").read_text()
    text = re.sub(r"\\\n\s*", " ", text)
    stages = re.split(r"(?im)^FROM\s+", text)[1:]
    return next(part for part in stages if part.splitlines()[0].endswith(" AS " + stage))


def test_runtime_keeps_distro_gtk_and_source_relative_dataset():
    runtime = instructions("runtime")
    assert runtime.startswith("ubuntu:24.04 AS runtime\n")
    for package in ("python3", "python3-venv", "python3-gi", "python3-gi-cairo",
                    "gir1.2-gtk-4.0", "fontconfig", "fonts-dejavu-core", "ca-certificates"):
        assert package in runtime
    assert "--no-install-recommends" in runtime
    assert "python3 -m venv --system-site-packages /opt/venv" in runtime
    assert re.search(r"pip install --no-cache-dir -e /app(?:\s|$)", runtime)
    assert "COPY src/ /app/src/" in runtime
    assert "COPY pyproject.toml README.md matches.csv /app/" in runtime
    for setting in ("PYTHONPATH=/app/src", "PATH=/opt/venv/bin:$PATH", "PYTHONNOUSERSITE=1",
                    "PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1", "HOME=/tmp/football-home",
                    "XDG_RUNTIME_DIR=/tmp/football-runtime", "GSK_RENDERER=cairo"):
        assert setting in runtime
    assert "WORKDIR /reports" in runtime
    assert "USER 1000:1000" in runtime
    assert 'ENTRYPOINT ["/usr/local/bin/football-entrypoint"]' in runtime
    assert 'CMD ["--doctor"]' in runtime
    assert "COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/football-entrypoint" in runtime
    assert "chmod -R a+rX /app /opt/venv" in runtime
    assert "chmod 1777 /reports" in runtime
    for forbidden in ("sudo", "xhost", "EXPOSE", "[gtk]", "[dev", "xvfb", "COPY . ."):
        assert forbidden not in runtime


def test_entrypoint_initializes_private_writable_paths_and_preserves_arguments(tmp_path):
    home = tmp_path / "home with spaces"
    runtime = tmp_path / "runtime with spaces"
    runtime.mkdir(mode=0o755)
    # A process boundary probe records argv/environment without launching GTK.
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$HOME" "$XDG_RUNTIME_DIR" "$@"\n')
    python.chmod(0o755)
    result = subprocess.run(
        ["sh", str(ROOT / "docker/entrypoint.sh"), "--evaluate", "--data", "/input/a b.csv"],
        env={**os.environ, "HOME": str(home), "XDG_RUNTIME_DIR": str(runtime),
             "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert home.is_dir()
    assert runtime.stat().st_mode & 0o777 == 0o700
    assert result.stdout.splitlines() == [str(home), str(runtime), "-m", "football_predictor",
                                         "--evaluate", "--data", "/input/a b.csv"]
    script = (ROOT / "docker/entrypoint.sh").read_text()
    assert 'exec python -m football_predictor "$@"' in script
    assert "sudo" not in script


def test_test_image_runs_native_widgets_in_xvfb_without_runtime_entrypoint():
    test = instructions("test")
    assert test.startswith("runtime AS test\n")
    assert "USER root" in test
    assert "xvfb xauth" in test
    assert "pip install --no-cache-dir -e '/app[dev,scraping]'" in test
    assert "ENV GDK_BACKEND=x11 FOOTBALL_PYTHON=/opt/venv/bin/python" in test
    assert "COPY tests/ /app/tests/" in test
    assert "COPY docker/ /app/docker/" in test
    assert "COPY Dockerfile .dockerignore compose.yaml docker-run.sh install.sh run.sh /app/" in test
    assert "chmod 755 /app/install.sh /app/run.sh /app/docker-run.sh" in test
    assert "USER 1000:1000" in test
    assert "ENTRYPOINT []" in test
    assert 'CMD ["xvfb-run", "-a", "/opt/venv/bin/python", "-m", "pytest", "/app/tests", "-q"]' in test


def test_build_context_is_allowlisted_and_excludes_private_environment_and_outputs():
    rules = [line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    assert rules[0] == "**"
    for included in ("Dockerfile", ".dockerignore", "compose.yaml", "pyproject.toml", "README.md",
                     "matches.csv", "run.sh", "install.sh", "docker-run.sh", "src/", "src/**",
                     "tests/", "tests/*.py", "docker/", "docker/entrypoint.sh"):
        assert "!" + included in rules
    # Last-match exclusions also protect files nested inside allowed source trees.
    for excluded in ("**/__pycache__", "**/*.pyc", "**/.env", "**/.env.*", "**/.git",
                     "**/.venv", "**/.pytest_cache", "**/.cache", "**/downloaded_files",
                     "**/reports", "**/build", "**/*.bak", "**/*.pem", "**/*.key"):
        assert excluded in rules
        assert rules.index(excluded) > max(i for i, rule in enumerate(rules) if rule.startswith("!"))


def test_test_image_places_pytest_cache_on_writable_tmp():
    assert 'PYTEST_ADDOPTS="-o cache_dir=/tmp/pytest-cache"' in instructions("test")


def test_compose_is_secure_headless_and_only_mounts_explicit_data_paths():
    # JSON is a YAML subset; stdlib parsing keeps image tests dependency-free.
    config = json.loads((ROOT / "compose.yaml").read_text())
    assert set(config["services"]) == {"app"}
    app = config["services"]["app"]
    assert app["image"] == "localhost/football-predict:local"
    assert app["build"] == {"context": ".", "target": "runtime"}
    assert app["user"] == "${LOCAL_UID:-1000}:${LOCAL_GID:-1000}"
    assert app["userns_mode"] == "${FOOTBALL_USERNS:-host}"
    assert app["network_mode"] == "none"
    assert app["read_only"] is True
    assert app["cap_drop"] == ["ALL"]
    assert app["security_opt"] == ["no-new-privileges:true"]
    assert app["command"] == ["--doctor"]
    assert app["working_dir"] == "/reports"
    assert len(app["tmpfs"]) == 1
    assert app["tmpfs"][0].startswith("/tmp:rw,nosuid,nodev,")
    assert "mode=1777" in app["tmpfs"][0]
    assert app["volumes"] == [
        {"type": "bind", "source": "./reports", "target": "/reports", "read_only": False,
         "bind": {"create_host_path": False}},
        {"type": "bind", "source": "./data/input", "target": "/input", "read_only": True,
         "bind": {"create_host_path": False}},
    ]
    for forbidden in ("privileged", "devices", "ports", "pid", "ipc", "environment", "entrypoint"):
        assert forbidden not in app
