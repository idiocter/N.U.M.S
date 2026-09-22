import pytest

from nums.config import Settings


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("NUMS_MAX_STEPS", "0", "at least 1"),
        ("NUMS_MAX_STEPS", "many", "integer"),
        ("NUMS_HISTORY_TURNS", "0", "at least 1"),
        ("NUMS_CAPTURE_DEVICE", "-2", "at least -1"),
        ("NUMS_MODEL", "  ", "cannot be empty"),
        ("NUMS_OLLAMA_URL", "localhost:11434", "HTTP URL"),
        ("NUMS_WAKE_PHRASE", " ", "cannot be empty"),
        ("NUMS_SPEAK", "maybe", "0 or 1"),
        ("NUMS_ACTION_MODE", "danger", "read_only, standard, or unrestricted"),
        ("NUMS_REPEAT_TOOL_LIMIT", "0", "at least 1"),
        ("NUMS_SESSION_TIMEOUT", "0", "at least 1"),
        ("NUMS_SLEEP_PHRASES", "|||", "at least one phrase"),
    ],
)
def test_bad_environment_has_clear_error(monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        Settings.from_env()
