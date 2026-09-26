import pytest

from nums.config import Settings


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("NUMS_MAX_STEPS", "0", "at least 1"),
        ("NUMS_MAX_TOOL_CALLS", "0", "at least 1"),
        ("NUMS_OLLAMA_TIMEOUT", "0", "at least 1"),
        ("NUMS_MAX_STEPS", "many", "integer"),
        ("NUMS_HISTORY_TURNS", "0", "at least 1"),
        ("NUMS_CAPTURE_DEVICE", "-2", "at least -1"),
        ("NUMS_MODEL", "  ", "cannot be empty"),
        ("NUMS_OLLAMA_URL", "localhost:11434", "HTTP URL"),
        ("NUMS_OLLAMA_URL", "http://localhost:11434/api", "host and optional port"),
        ("NUMS_OLLAMA_URL", "http://localhost:bad", "host and optional port"),
        ("NUMS_OLLAMA_URL", "http://user:secret@localhost:11434", "host and optional port"),
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


def test_model_timeout_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUMS_OLLAMA_TIMEOUT", "45")

    settings = Settings.from_env()

    assert settings.ollama_timeout_seconds == 45
