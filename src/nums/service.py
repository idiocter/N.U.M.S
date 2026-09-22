"""macOS LaunchAgent support for the NUMS wake listener."""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

from .config import Settings


LABEL = "com.bipul.nums"


def service_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def service_plist(settings: Settings) -> bytes:
    log = Path.home() / "Library" / "Logs" / "NUMS.log"
    environment = {
        "NUMS_MODEL": settings.model,
        "NUMS_OLLAMA_URL": settings.ollama_url,
        "NUMS_ACTION_MODE": settings.action_mode,
        "NUMS_WAKE_PHRASE": settings.wake_phrase,
        "NUMS_SESSION_TIMEOUT": str(settings.session_timeout_seconds),
        "NUMS_CAPTURE_DEVICE": str(settings.capture_device),
    }
    if settings.history_file:
        environment["NUMS_HISTORY_FILE"] = settings.history_file
    if settings.trace_file:
        environment["NUMS_TRACE_FILE"] = settings.trace_file
    return plistlib.dumps({
        "Label": LABEL,
        "ProgramArguments": [sys.executable, "-m", "nums.cli", "--wake"],
        "EnvironmentVariables": environment,
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "StandardOutPath": str(log),
        "StandardErrorPath": str(log),
    })


def install_service(settings: Settings) -> Path:
    path = service_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(service_plist(settings))
    path.chmod(0o600)
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)], check=False)
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True)
    return path


def uninstall_service() -> Path:
    path = service_path()
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)], check=False)
    path.unlink(missing_ok=True)
    return path
