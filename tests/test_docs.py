import re
from pathlib import Path

from nums.config import Settings


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
CONFIG_SOURCE = (ROOT / "src" / "nums" / "config.py").read_text(encoding="utf-8")


def test_every_setting_is_in_the_configuration_table() -> None:
    variables = set(re.findall(r"NUMS_[A-Z_]+", CONFIG_SOURCE))
    table = README.split("## Configuration", 1)[1].split("\n## ", 1)[0]

    assert variables
    assert [name for name in sorted(variables) if f"`{name}`" not in table] == []


def test_documented_defaults_match_the_code() -> None:
    table = README.split("## Configuration", 1)[1].split("\n## ", 1)[0]
    defaults = Settings()
    expected = {
        "NUMS_MODEL": defaults.model,
        "NUMS_OLLAMA_URL": defaults.ollama_url,
        "NUMS_OLLAMA_TIMEOUT": str(defaults.ollama_timeout_seconds),
        "NUMS_ACTION_MODE": defaults.action_mode,
        "NUMS_MAX_STEPS": str(defaults.max_steps),
        "NUMS_MAX_TOOL_CALLS": str(defaults.max_tool_calls),
        "NUMS_REPEAT_TOOL_LIMIT": str(defaults.repeat_tool_limit),
        "NUMS_HISTORY_TURNS": str(defaults.history_turns),
        "NUMS_MAX_CONTEXT_CHARS": str(defaults.max_context_chars),
        "NUMS_TRACE_MAX_BYTES": str(defaults.trace_max_bytes),
        "NUMS_WAKE_PHRASE": defaults.wake_phrase,
        "NUMS_SESSION_TIMEOUT": str(defaults.session_timeout_seconds),
        "NUMS_CAPTURE_DEVICE": str(defaults.capture_device),
    }

    for name, value in expected.items():
        row = next(line for line in table.splitlines() if line.startswith(f"| `{name}`"))
        assert f"`{value}`" in row, f"{name} default {value!r} is not documented"
