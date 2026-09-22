import plistlib
import sys

from nums.config import Settings
from nums.service import LABEL, service_plist


def test_service_plist_runs_current_environment_and_preserves_settings() -> None:
    data = plistlib.loads(service_plist(Settings(
        model="qwen:test", action_mode="standard", history_file="/tmp/history.json"
    )))

    assert data["Label"] == LABEL
    assert data["ProgramArguments"] == [sys.executable, "-m", "nums.cli", "--wake"]
    assert data["EnvironmentVariables"]["NUMS_MODEL"] == "qwen:test"
    assert data["EnvironmentVariables"]["NUMS_ACTION_MODE"] == "standard"
    assert data["EnvironmentVariables"]["NUMS_HISTORY_FILE"] == "/tmp/history.json"
    assert data["KeepAlive"] is True
