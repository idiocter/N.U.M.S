import os
import plistlib
import sys

from nums.config import Settings
from nums.service import LABEL, service_plist


def test_service_plist_runs_current_environment_and_preserves_settings() -> None:
    data = plistlib.loads(service_plist(Settings(
        model="qwen:test", action_mode="standard", history_file="/tmp/history.json",
        ollama_timeout_seconds=45,
        max_steps=5, history_turns=3, repeat_tool_limit=1,
        sleep_phrases=("sleep nums", "good night"), whisper_model="/tmp/voice.bin",
    )))

    assert data["Label"] == LABEL
    assert data["ProgramArguments"] == [sys.executable, "-m", "nums.cli", "--wake"]
    assert data["EnvironmentVariables"]["NUMS_MODEL"] == "qwen:test"
    assert data["EnvironmentVariables"]["NUMS_OLLAMA_TIMEOUT"] == "45"
    assert data["EnvironmentVariables"]["NUMS_ACTION_MODE"] == "standard"
    assert data["EnvironmentVariables"]["NUMS_HISTORY_FILE"] == "/tmp/history.json"
    assert data["EnvironmentVariables"]["NUMS_MAX_STEPS"] == "5"
    assert data["EnvironmentVariables"]["NUMS_HISTORY_TURNS"] == "3"
    assert data["EnvironmentVariables"]["NUMS_REPEAT_TOOL_LIMIT"] == "1"
    assert data["EnvironmentVariables"]["NUMS_SLEEP_PHRASES"] == "sleep nums|good night"
    assert data["EnvironmentVariables"]["NUMS_WHISPER_MODEL"] == "/tmp/voice.bin"
    assert data["EnvironmentVariables"]["PATH"] == os.environ.get("PATH", os.defpath)
    assert data["KeepAlive"] is True
