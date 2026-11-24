"""Host launcher integration tests; engines are fake executables, never containers."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "docker-run.sh"


@pytest.fixture
def host(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    if SOURCE.exists():
        shutil.copy2(SOURCE, root / "docker-run.sh")
    binary = tmp_path / "bin"
    binary.mkdir()
    log = tmp_path / "engine.jsonl"
    engine = binary / "docker"
    engine.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["ENGINE_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
if args == ["--version"]:
    print(os.environ.get("ENGINE_VERSION", "Docker version test"))
if args == ["info"] and os.environ.get("INFO_FAIL"):
    sys.exit(1)
if args[:2] == ["image", "inspect"] and os.environ.get("IMAGE_MISSING"):
    sys.exit(1)
if args and args[0] == "run":
    for arg in args:
        if "target=/tmp/football.xauth" in arg:
            source = arg.split("source=", 1)[1].split(",target=", 1)[0]
            Path(os.environ["COOKIE_CAPTURE"]).write_text(json.dumps({{"mode": os.stat(source).st_mode & 0o777, "path": source, "content": Path(source).read_text()}}))
    sys.exit(int(os.environ.get("RUN_STATUS", "0")))
''')
    engine.chmod(0o755)
    env = dict(os.environ)
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "CONTAINER_ENGINE", "XAUTHORITY"):
        env.pop(name, None)
    env.update(PATH=f"{binary}:/usr/bin:/bin", ENGINE_LOG=str(log), TMPDIR=str(tmp_path), COOKIE_CAPTURE=str(tmp_path / "cookie.json"))

    class Host:
        def run(self, *args, **extra):
            return subprocess.run(["bash", str(root / "docker-run.sh"), *args], env=env | extra, text=True, capture_output=True)

        def calls(self):
            return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

        def run_args(self):
            return next(args for args in self.calls() if args[0] == "run")

    result = Host()
    result.root, result.bin, result.env = root, binary, env
    return result


@pytest.mark.parametrize("arguments", [("--doctor",), ("--evaluate", "--output", "/reports"), ("--benchmark",)])
def test_headless_isolated_run_with_owned_reports(host, arguments):
    result = host.run(*arguments, DISPLAY=":99", WAYLAND_DISPLAY="absent")
    assert result.returncode == 0, result.stderr
    args = host.run_args()
    for flag in ("--rm", "--init", "--read-only"):
        assert flag in args
    for key, value in (("--network", "none"), ("--cap-drop", "ALL"), ("--security-opt", "no-new-privileges"), ("--tmpfs", "/tmp:rw,nosuid,nodev,size=512m"), ("--user", f"{os.getuid()}:{os.getgid()}")):
        assert args[args.index(key) + 1] == value
    assert args[-len(arguments):] == list(arguments)
    assert args[-len(arguments)-1] == "localhost/football-predict:local"
    mounts = [args[i + 1] for i, arg in enumerate(args) if arg == "--mount"]
    assert mounts == [f"type=bind,source={host.root}/reports,target=/reports", f"type=bind,source={host.root}/data/input,target=/input,readonly"]
    assert (host.root / "reports").is_dir()
    assert (host.root / "data/input").is_dir()
    assert "--userns=keep-id" not in args
    assert not any("DISPLAY=" in arg for arg in args)
    assert not any(args[0] == "build" for args in host.calls())


@pytest.mark.parametrize("force,missing", [(False, True), (True, False)])
def test_image_is_built_when_needed(host, force, missing):
    result = host.run(*(["--build"] if force else []), "--doctor", IMAGE_MISSING="1" if missing else "")
    assert result.returncode == 0, result.stderr
    assert ["info"] in host.calls()
    assert ["build", "--target", "runtime", "--tag", "localhost/football-predict:local", "--file", str(host.root / "Dockerfile"), str(host.root)] in host.calls()
    assert "--build" not in host.run_args()


@pytest.mark.parametrize("override", ["missing-engine", None])
def test_engine_preflight_failure_is_actionable(host, override):
    extra = {"INFO_FAIL": "1"}
    if override:
        extra["CONTAINER_ENGINE"] = override
    result = host.run("--doctor", **extra)
    assert result.returncode != 0
    assert "Docker or Podman" in result.stderr
    assert not any(call[0] in ("build", "run") for call in host.calls())


@pytest.mark.parametrize("selection", ["wrapper", "absolute", "fallback"])
def test_podman_preserves_host_user_namespace(host, selection):
    extra = {"ENGINE_VERSION": "podman version test"}
    if selection == "absolute":
        target = host.bin / "my engine"
        shutil.copy2(host.bin / "docker", target)
        extra["CONTAINER_ENGINE"] = str(target)
    if selection == "fallback":
        (host.bin / "docker").rename(host.bin / "podman")
        # Exclude system docker while preserving shell utilities.
        for name in ("bash", "dirname", "id", "mkdir"):
            (host.bin / name).symlink_to(shutil.which(name))
        extra["PATH"] = str(host.bin)
    result = host.run("--doctor", **extra)
    assert result.returncode == 0, result.stderr
    assert "--userns=keep-id" in host.run_args()


@pytest.mark.parametrize("absolute", [False, True])
def test_gui_prefers_only_the_wayland_socket(host, absolute):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"), prefix="wl-") as runtime:
        path = Path(runtime) / "wayland-0"
        with socket.socket(socket.AF_UNIX) as display:
            display.bind(str(path))
            result = host.run(WAYLAND_DISPLAY=str(path) if absolute else path.name, XDG_RUNTIME_DIR=runtime, DISPLAY=":99")
    assert result.returncode == 0, result.stderr
    args = host.run_args()
    assert f"type=bind,source={path},target=/tmp/host-wayland.sock,readonly" in args
    assert "WAYLAND_DISPLAY=/tmp/host-wayland.sock" in args
    assert "GDK_BACKEND=wayland" in args
    assert "GSK_RENDERER=cairo" in args
    assert args[-2:] == ["--data", "/app/matches.csv"]
    assert not any("XAUTHORITY=" in arg or "X11-unix" in arg for arg in args)
    mounts = [args[i + 1] for i, arg in enumerate(args) if arg == "--mount"]
    assert len(mounts) == 3


@pytest.mark.parametrize("fake_socket", [False, True])
def test_gui_without_a_real_display_fails_before_run(host, fake_socket):
    extra = {}
    if fake_socket:
        impostor = host.root / "not-a-socket"
        impostor.write_text("not a socket")
        extra["WAYLAND_DISPLAY"] = str(impostor)
    result = host.run(**extra)
    assert result.returncode != 0
    assert "Wayland" in result.stderr and "--doctor" in result.stderr
    assert not any(call[0] == "run" for call in host.calls())


@pytest.mark.parametrize("status", [0, 23])
def test_x11_uses_private_wildcard_cookie_and_cleans_it(host, status):
    xauth = host.bin / "xauth"
    xauth.write_text(f'''#!{sys.executable}
import sys
from pathlib import Path
if sys.argv[1] == "nlist":
    print("0100 0004 686f7374 0001 30 0012 4d49542d4d414749432d434f4f4b49452d31 0010 00112233445566778899aabbccddeeff")
else:
    assert sys.argv[1] == "-f" and sys.argv[3:] == ["nmerge", "-"]
    Path(sys.argv[2]).write_text(sys.stdin.read())
''')
    xauth.chmod(0o755)
    result = host.run(DISPLAY=":0", WAYLAND_DISPLAY="absent", RUN_STATUS=str(status))
    assert result.returncode == status, result.stderr
    args = host.run_args()
    assert "DISPLAY=:0" in args and "GDK_BACKEND=x11" in args
    assert "GSK_RENDERER=cairo" in args
    assert "XAUTHORITY=/tmp/football.xauth" in args
    assert "type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix,readonly" in args
    capture = json.loads(Path(host.env["COOKIE_CAPTURE"]).read_text())
    assert capture["mode"] == 0o600
    assert capture["content"].startswith("ffff ")
    assert not Path(capture["path"]).exists()
    assert not Path(capture["path"]).parent.exists()
    assert not any("xhost" in arg or "privileged" in arg or "/dev/" in arg for arg in args)


@pytest.mark.parametrize("display", ["example.org:0", "localhost:10.0", "tcp/example.org:1", ":0,evil"])
def test_x11_refuses_remote_or_invalid_displays(host, display):
    result = host.run(DISPLAY=display)
    assert result.returncode != 0
    assert "local X11" in result.stderr
    assert not any(call[0] == "run" for call in host.calls())


def test_help_requires_no_engine(host):
    result = host.run("--help", CONTAINER_ENGINE="missing-engine")
    assert result.returncode == 0, result.stderr
    assert "--build" in result.stdout
    assert "--doctor" in result.stdout
    assert host.calls() == []
