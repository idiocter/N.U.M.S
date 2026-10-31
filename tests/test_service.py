import os
import plistlib
import stat
import subprocess
import sys

import pytest

from nums.config import Settings
from nums.service import LABEL, install_service, service_path, service_plist, uninstall_service


def test_service_plist_runs_current_environment_and_preserves_settings() -> None:
    data = plistlib.loads(service_plist(Settings(
        model="qwen:test", history_file="/tmp/history.json",
        ollama_timeout_seconds=45,
        history_turns=3, max_context_chars=40000,
        sleep_phrases=("sleep nums", "good night"), whisper_model="/tmp/voice.bin",
    )))

    assert data["Label"] == LABEL
    assert data["ProgramArguments"] == [sys.executable, "-m", "nums.cli", "--wake"]
    assert data["EnvironmentVariables"]["NUMS_MODEL"] == "qwen:test"
    assert data["EnvironmentVariables"]["NUMS_OLLAMA_TIMEOUT"] == "45"
    assert data["EnvironmentVariables"]["NUMS_KEEP_ALIVE"] == "30m"
    assert data["EnvironmentVariables"]["NUMS_HISTORY_FILE"] == "/tmp/history.json"
    assert data["EnvironmentVariables"]["NUMS_HISTORY_TURNS"] == "3"
    assert data["EnvironmentVariables"]["NUMS_MAX_CONTEXT_CHARS"] == "40000"
    assert data["EnvironmentVariables"]["NUMS_SLEEP_PHRASES"] == "sleep nums|good night"
    assert data["EnvironmentVariables"]["NUMS_WHISPER_MODEL"] == "/tmp/voice.bin"
    assert data["EnvironmentVariables"]["PATH"] == os.environ.get("PATH", os.defpath)
    assert "NUMS_TRACE_FILE" not in data["EnvironmentVariables"]
    assert "NUMS_TRACE_MAX_BYTES" not in data["EnvironmentVariables"]
    for removed in ("NUMS_ACTION_MODE", "NUMS_MAX_STEPS", "NUMS_MAX_TOOL_CALLS", "NUMS_REPEAT_TOOL_LIMIT"):
        assert removed not in data["EnvironmentVariables"]
    assert data["KeepAlive"] is True


def test_service_plist_carries_trace_limit_with_trace_file() -> None:
    data = plistlib.loads(service_plist(Settings(
        trace_file="/tmp/trace.jsonl", trace_max_bytes=2048,
    )))

    assert data["EnvironmentVariables"]["NUMS_TRACE_FILE"] == "/tmp/trace.jsonl"
    assert data["EnvironmentVariables"]["NUMS_TRACE_MAX_BYTES"] == "2048"


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


def test_uninstall_stops_the_service_then_removes_the_plist(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "LaunchAgents" / "nums.plist"
    path.parent.mkdir()
    path.write_bytes(service_plist(Settings()))
    events = []

    def fake_run(command: list[str], **kwargs: object) -> None:
        events.append((command[:2], path.exists(), kwargs.get("check")))

    monkeypatch.setattr("nums.service.service_path", lambda: path)
    monkeypatch.setattr("nums.service.subprocess.run", fake_run)

    assert uninstall_service() == path

    assert events == [(["launchctl", "bootout"], True, False)]
    assert not path.exists()


def test_uninstall_tolerates_a_service_that_was_never_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "missing.plist"
    monkeypatch.setattr("nums.service.service_path", lambda: path)
    monkeypatch.setattr("nums.service.subprocess.run", lambda *args, **kwargs: None)

    assert uninstall_service() == path


def test_service_lives_in_the_users_launch_agents(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("nums.service.Path.home", lambda: tmp_path)

    assert service_path() == tmp_path / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def test_first_install_writes_a_private_plist_and_only_bootstraps(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "LaunchAgents" / "nums.plist"
    calls = []
    monkeypatch.setattr("nums.service.service_path", lambda: path)
    monkeypatch.setattr(
        "nums.service.subprocess.run", lambda command, **kwargs: calls.append((command, kwargs))
    )

    assert install_service(Settings(model="qwen:new")) == path

    assert [command[1] for command, _ in calls] == ["bootstrap"]
    command, kwargs = calls[0]
    assert command[2] == f"gui/{os.getuid()}" and command[3] == str(path)
    assert kwargs == {"check": True}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert plistlib.loads(path.read_bytes())["EnvironmentVariables"]["NUMS_MODEL"] == "qwen:new"
    assert list(path.parent.glob(".nums.plist.*")) == []


def test_reinstall_stops_the_old_service_before_loading_the_new_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "nums.plist"
    path.write_bytes(service_plist(Settings(model="qwen:old")))
    events = []

    def fake_run(command: list[str], **kwargs: object) -> None:
        events.append((command[1], plistlib.loads(path.read_bytes())["EnvironmentVariables"]["NUMS_MODEL"]))

    monkeypatch.setattr("nums.service.service_path", lambda: path)
    monkeypatch.setattr("nums.service.subprocess.run", fake_run)

    install_service(Settings(model="qwen:new"))

    assert events == [("bootout", "qwen:new"), ("bootstrap", "qwen:new")]


def test_plist_logs_to_the_users_library_and_restarts_on_exit(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("nums.service.Path.home", lambda: tmp_path)

    data = plistlib.loads(service_plist(Settings()))

    log = str(tmp_path / "Library" / "Logs" / "NUMS.log")
    assert data["StandardOutPath"] == data["StandardErrorPath"] == log
    assert data["RunAtLoad"] is True and data["KeepAlive"] is True
    assert data["ThrottleInterval"] == 10
    assert "NUMS_HISTORY_FILE" not in data["EnvironmentVariables"]
