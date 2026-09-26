"""macOS LaunchAgent support for the NUMS wake listener."""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import Settings


LABEL = "com.bipul.nums"


def service_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def service_plist(settings: Settings) -> bytes:
    log = Path.home() / "Library" / "Logs" / "NUMS.log"
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "NUMS_MODEL": settings.model,
        "NUMS_OLLAMA_URL": settings.ollama_url,
        "NUMS_OLLAMA_TIMEOUT": str(settings.ollama_timeout_seconds),
        "NUMS_MAX_STEPS": str(settings.max_steps),
        "NUMS_MAX_TOOL_CALLS": str(settings.max_tool_calls),
        "NUMS_HISTORY_TURNS": str(settings.history_turns),
        "NUMS_ACTION_MODE": settings.action_mode,
        "NUMS_REPEAT_TOOL_LIMIT": str(settings.repeat_tool_limit),
        "NUMS_WAKE_PHRASE": settings.wake_phrase,
        "NUMS_SESSION_TIMEOUT": str(settings.session_timeout_seconds),
        "NUMS_SLEEP_PHRASES": "|".join(settings.sleep_phrases),
        "NUMS_WHISPER_MODEL": settings.whisper_model,
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


def _write_plist(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def install_service(settings: Settings) -> Path:
    path = service_path()
    previous = path.read_bytes() if path.exists() else None
    _write_plist(path, service_plist(settings))
    domain = f"gui/{os.getuid()}"
    if previous is not None:
        subprocess.run(["launchctl", "bootout", domain, str(path)], check=False)
    try:
        subprocess.run(["launchctl", "bootstrap", domain, str(path)], check=True)
    except (subprocess.CalledProcessError, OSError):
        if previous is None:
            path.unlink(missing_ok=True)
        else:
            _write_plist(path, previous)
            subprocess.run(["launchctl", "bootstrap", domain, str(path)], check=False)
        raise
    return path


def uninstall_service() -> Path:
    path = service_path()
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)], check=False)
    path.unlink(missing_ok=True)
    return path
