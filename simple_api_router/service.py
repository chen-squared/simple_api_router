"""Service management for simple-api-router (macOS launchd / Linux systemd).

Provides install / uninstall / start / stop / restart / status / log commands
that are exposed as CLI subcommands of the main executable.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

SERVICE_NAME = "simple-api-router"
SERVICE_LABEL = "com.chen-squared.simple-api-router"  # macOS plist label

# Paths that are the same on both platforms
CONFIG_DIR = Path.home() / ".config" / SERVICE_NAME
DEFAULT_CONFIG = CONFIG_DIR / "config.yaml"
ENV_FILE = CONFIG_DIR / "env"

# macOS
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_LABEL}.plist"
LOG_DIR = Path.home() / "Library" / "Logs" / SERVICE_NAME

# Linux
SYSTEMD_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
SYSTEMD_UNIT = SYSTEMD_UNIT_DIR / f"{SERVICE_NAME}.service"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _os() -> str:
    s = platform.system()
    return {"Darwin": "macos", "Linux": "linux"}.get(s, "unknown")


def _bold(msg: str) -> None:
    print(f"\033[1m{msg}\033[0m")


def _info(msg: str) -> None:
    print(f"  \033[34m→\033[0m {msg}")


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"  \033[33m!\033[0m {msg}", file=sys.stderr)


def _die(msg: str) -> None:
    print(f"\033[31merror:\033[0m {msg}", file=sys.stderr)
    sys.exit(1)


def find_own_executable(exe_override: Optional[str] = None) -> str:
    """Return the absolute path to the simple-api-router executable.

    Priority:
    1. Explicit ``--exe`` override from the user.
    2. ``simple-api-router`` resolved through PATH (works when venv is active
       or the package was installed system-wide).
    3. ``sys.argv[0]`` resolved to an absolute path — always works when the
       user is calling the command right now, which is exactly when
       ``install`` is typically run.
    """
    if exe_override:
        p = Path(exe_override).expanduser().resolve()
        if not p.is_file() or not os.access(p, os.X_OK):
            _die(f"Not executable: {p}")
        return str(p)

    found = shutil.which(SERVICE_NAME)
    if found:
        return str(Path(found).resolve())

    argv0 = Path(sys.argv[0]).resolve()
    if argv0.is_file() and os.access(argv0, os.X_OK):
        return str(argv0)

    _die(
        f"{SERVICE_NAME} not found.\n"
        "  Activate your venv before running install, or pass:\n"
        f"  {SERVICE_NAME} install --exe /path/to/venv/bin/{SERVICE_NAME}"
    )
    return ""  # unreachable but makes type checkers happy


def _ensure_env_file() -> None:
    if ENV_FILE.exists():
        return
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text(
        "# Environment variables for simple-api-router\n"
        "# Uncomment and set your API keys:\n"
        "#\n"
        "# ANTHROPIC_API_KEY=sk-ant-...\n"
        "# OPENAI_API_KEY=sk-...\n"
        "# DEEPSEEK_API_KEY=sk-...\n"
    )
    _info(f"Created env file template: {ENV_FILE}")
    _warn(f"Edit {ENV_FILE} and add your API keys before starting")


def resolve_config(explicit: Optional[str]) -> Path:
    """Find or create the config file; return its resolved absolute path."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            _die(f"Config not found: {p}")
        return p

    if DEFAULT_CONFIG.exists():
        return DEFAULT_CONFIG.resolve()

    cwd_config = Path.cwd() / "config.yaml"
    if cwd_config.exists():
        return cwd_config.resolve()

    # Nothing found — copy the package template or create an empty file
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).parent.parent
    template = package_root / "config.yaml"
    if template.exists():
        shutil.copy(template, DEFAULT_CONFIG)
        _info(f"Copied config template to {DEFAULT_CONFIG}")
    else:
        DEFAULT_CONFIG.touch()
        _info(f"Created empty config at {DEFAULT_CONFIG}")

    return DEFAULT_CONFIG.resolve()


# ---------------------------------------------------------------------------
# macOS — launchd
# ---------------------------------------------------------------------------

def _launchd_loaded() -> bool:
    return subprocess.run(
        ["launchctl", "list", SERVICE_LABEL],
        capture_output=True,
    ).returncode == 0


def _install_macos(exe: str, config_path: Path) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _ok(f"Log dir: {LOG_DIR}")

    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"\n'
        f'    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        f'<plist version="1.0">\n'
        f'<dict>\n'
        f'    <key>Label</key>\n'
        f'    <string>{SERVICE_LABEL}</string>\n'
        f'    <key>ProgramArguments</key>\n'
        f'    <array>\n'
        f'        <string>{exe}</string>\n'
        f'        <string>run</string>\n'
        f'        <string>--config</string>\n'
        f'        <string>{config_path}</string>\n'
        f'        <string>--env-file</string>\n'
        f'        <string>{ENV_FILE}</string>\n'
        f'    </array>\n'
        f'    <key>WorkingDirectory</key>\n'
        f'    <string>{config_path.parent}</string>\n'
        f'    <key>RunAtLoad</key>\n'
        f'    <true/>\n'
        f'    <key>KeepAlive</key>\n'
        f'    <true/>\n'
        f'    <key>StandardOutPath</key>\n'
        f'    <string>{LOG_DIR}/stdout.log</string>\n'
        f'    <key>StandardErrorPath</key>\n'
        f'    <string>{LOG_DIR}/stderr.log</string>\n'
        f'    <key>ThrottleInterval</key>\n'
        f'    <integer>5</integer>\n'
        f'</dict>\n'
        f'</plist>\n'
    )
    _ok(f"Plist: {PLIST_PATH}")

    if _launchd_loaded():
        subprocess.run(
            ["launchctl", "unload", str(PLIST_PATH)],
            capture_output=True,
        )
    subprocess.run(["launchctl", "load", "-w", str(PLIST_PATH)], check=True)
    _ok("Service loaded and enabled (auto-starts on login)")


def _uninstall_macos() -> None:
    if _launchd_loaded():
        subprocess.run(
            ["launchctl", "unload", "-w", str(PLIST_PATH)],
            capture_output=True,
        )
        _ok("Service unloaded")
    if PLIST_PATH.exists():
        PLIST_PATH.unlink()
        _ok("Plist removed")


def _start_macos() -> None:
    if not PLIST_PATH.exists():
        _die(f"Service not installed. Run: {SERVICE_NAME} install")
    subprocess.run(["launchctl", "start", SERVICE_LABEL], check=True)
    _ok("Started")


def _stop_macos() -> None:
    r = subprocess.run(["launchctl", "stop", SERVICE_LABEL], capture_output=True)
    (_ok if r.returncode == 0 else _warn)("Stopped" if r.returncode == 0 else "Was not running")


def _restart_macos() -> None:
    _stop_macos()
    time.sleep(1)
    _start_macos()


def _status_macos() -> None:
    if not _launchd_loaded():
        _warn("Service not loaded")
        return
    r = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{SERVICE_LABEL}"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        r = subprocess.run(
            ["launchctl", "list", SERVICE_LABEL],
            capture_output=True, text=True,
        )
    print(r.stdout)
    log_file = CONFIG_DIR / "router.log"
    if log_file.exists():
        print()
        _info("Last 5 log lines:")
        for line in log_file.read_text().splitlines()[-5:]:
            print(f"    {line}")


def _log_macos() -> None:
    log_file = CONFIG_DIR / "router.log"
    if not log_file.exists():
        _die(f"No log file at {log_file} — is the service running?")
    _info(f"Tailing {log_file} (Ctrl+C to stop)…")
    args = ["tail", "-f", str(log_file)]
    try:
        subprocess.run(args)
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Linux — systemd user
# ---------------------------------------------------------------------------

def _systemd(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True, text=True, check=check,
    )


def _install_linux(exe: str, config_path: Path) -> None:
    SYSTEMD_UNIT_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEMD_UNIT.write_text(
        f"[Unit]\n"
        f"Description=Simple API Router\n"
        f"After=network.target\n"
        f"\n"
        f"[Service]\n"
        f"Type=simple\n"
        f"ExecStart={exe} run --config {config_path} --env-file {ENV_FILE}\n"
        f"WorkingDirectory={config_path.parent}\n"
        f"Restart=always\n"
        f"RestartSec=5\n"
        f"\n"
        f"[Install]\n"
        f"WantedBy=default.target\n"
    )
    _ok(f"Unit file: {SYSTEMD_UNIT}")
    _systemd("daemon-reload")
    _systemd("enable", SERVICE_NAME, check=True)
    _systemd("start", SERVICE_NAME, check=True)
    _ok("Service enabled and started")

    user = os.environ.get("USER", "")
    if user and shutil.which("loginctl"):
        r = subprocess.run(["loginctl", "enable-linger", user], capture_output=True)
        if r.returncode == 0:
            _ok("Linger enabled (service survives logout)")


def _uninstall_linux() -> None:
    if _systemd("is-active", "--quiet", SERVICE_NAME).returncode == 0:
        _systemd("stop", SERVICE_NAME)
        _ok("Stopped")
    if _systemd("is-enabled", "--quiet", SERVICE_NAME).returncode == 0:
        _systemd("disable", SERVICE_NAME)
        _ok("Disabled")
    if SYSTEMD_UNIT.exists():
        SYSTEMD_UNIT.unlink()
        _systemd("daemon-reload")
        _ok("Unit file removed")


def _start_linux() -> None:
    _systemd("start", SERVICE_NAME, check=True)
    _ok("Started")


def _stop_linux() -> None:
    _systemd("stop", SERVICE_NAME, check=True)
    _ok("Stopped")


def _restart_linux() -> None:
    _systemd("restart", SERVICE_NAME, check=True)
    _ok("Restarted")


def _status_linux() -> None:
    r = _systemd("status", SERVICE_NAME, "--no-pager")
    if r.stdout:
        print(r.stdout)
    if r.stderr:
        print(r.stderr, file=sys.stderr)


def _log_linux() -> None:
    _info("Tailing logs (Ctrl+C to stop)…")
    try:
        subprocess.run([
            "journalctl", "--user", "-u", SERVICE_NAME, "-f", "--no-pager",
        ])
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Public API — called by cli.py
# ---------------------------------------------------------------------------

def install(config: Optional[str] = None, exe_override: Optional[str] = None) -> None:
    os_name = _os()
    if os_name == "unknown":
        _die(f"Unsupported OS: {platform.system()}")

    _bold(f"Installing {SERVICE_NAME} service ({os_name})")

    exe = find_own_executable(exe_override)
    _ok(f"Executable: {exe}")

    config_path = resolve_config(config)
    _ok(f"Config: {config_path}")

    _ensure_env_file()

    if os_name == "macos":
        _install_macos(exe, config_path)
    else:
        _install_linux(exe, config_path)

    print()
    _bold("Done.  Useful commands:")
    print(f"  {SERVICE_NAME} status / log / stop / start / restart / uninstall")
    _info("Hot reload: just save config.yaml — changes apply automatically.")
    _info(f"API keys:   edit {ENV_FILE}  then restart.")


def uninstall() -> None:
    os_name = _os()
    _bold(f"Uninstalling {SERVICE_NAME} service")
    if os_name == "macos":
        _uninstall_macos()
    elif os_name == "linux":
        _uninstall_linux()
    else:
        _die(f"Unsupported OS: {platform.system()}")
    _ok("Done (logs and env file preserved)")


def start() -> None:
    os_name = _os()
    if os_name == "macos":
        _start_macos()
    elif os_name == "linux":
        _start_linux()
    else:
        _die(f"Unsupported OS: {platform.system()}")


def stop() -> None:
    os_name = _os()
    if os_name == "macos":
        _stop_macos()
    elif os_name == "linux":
        _stop_linux()
    else:
        _die(f"Unsupported OS: {platform.system()}")


def restart() -> None:
    os_name = _os()
    if os_name == "macos":
        _restart_macos()
    elif os_name == "linux":
        _restart_linux()
    else:
        _die(f"Unsupported OS: {platform.system()}")


def status() -> None:
    os_name = _os()
    if os_name == "macos":
        _status_macos()
    elif os_name == "linux":
        _status_linux()
    else:
        _die(f"Unsupported OS: {platform.system()}")


def log() -> None:
    os_name = _os()
    if os_name == "macos":
        _log_macos()
    elif os_name == "linux":
        _log_linux()
    else:
        _die(f"Unsupported OS: {platform.system()}")
