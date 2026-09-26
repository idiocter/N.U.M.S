import os
import plistlib
import stat
import subprocess
import sys

import pytest

from nums.config import Settings
from nums.service import LABEL, install_service, service_plist


def test_service_plist_runs_current_environment_and_preserves_settings() -> None:
    data = plistlib.loads(service_plist(Settings(
        model="qwen:test", action_mode="standard", history_file="/tmp/history.json",
        ollama_timeout_seconds=45,
        max_steps=5, max_tool_calls=9, history_turns=3, repeat_tool_limit=1,
        sleep_phrases=("sleep nums", "good night"), whisper_model="/tmp/voice.bin",
    )))

    assert data["Label"] == LABEL
    assert data["ProgramArguments"] == [sys.executable, "-m", "nums.cli", "--wake"]
    assert data["EnvironmentVariables"]["NUMS_MODEL"] == "qwen:test"
    assert data["EnvironmentVariables"]["NUMS_OLLAMA_TIMEOUT"] == "45"
    assert data["EnvironmentVariables"]["NUMS_ACTION_MODE"] == "standard"
    assert data["EnvironmentVariables"]["NUMS_HISTORY_FILE"] == "/tmp/history.json"
    assert data["EnvironmentVariables"]["NUMS_MAX_STEPS"] == "5"
    assert data["EnvironmentVariables"]["NUMS_MAX_TOOL_CALLS"] == "9"
    assert data["EnvironmentVariables"]["NUMS_HISTORY_TURNS"] == "3"
    assert data["EnvironmentVariables"]["NUMS_REPEAT_TOOL_LIMIT"] == "1"
    assert data["EnvironmentVariables"]["NUMS_SLEEP_PHRASES"] == "sleep nums|good night"
    assert data["EnvironmentVariables"]["NUMS_WHISPER_MODEL"] == "/tmp/voice.bin"
    assert data["EnvironmentVariables"]["PATH"] == os.environ.get("PATH", os.defpath)
    assert data["KeepAlive"] is True


def test_failed_service_update_restores_previous_plist(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "LaunchAgents" / "nums.plist"
    path.parent.mkdir()
    previous = service_plist(Settings(model="old:model"))
    path.write_bytes(previous)
    calls = []
    bootstraps = 0

    def fake_run(command: list[str], **kwargs: object) -> None:
        nonlocal bootstraps
        calls.append(command[1])
        if command[1] == "bootstrap":
            bootstraps += 1
            if bootstraps == 1:
                raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("nums.service.service_path", lambda: path)
    monkeypatch.setattr("nums.service.subprocess.run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        install_service(Settings(model="new:model"))

    assert path.read_bytes() == previous
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert calls == ["bootout", "bootstrap", "bootstrap"]


def test_failed_first_service_install_removes_plist(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "LaunchAgents" / "nums.plist"
    monkeypatch.setattr("nums.service.service_path", lambda: path)

    def fail(command: list[str], **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("nums.service.subprocess.run", fail)

    with pytest.raises(subprocess.CalledProcessError):
        install_service(Settings())

    assert not path.exists()
