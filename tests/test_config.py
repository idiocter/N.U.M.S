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
        ("NUMS_MAX_CONTEXT_CHARS", "999", "at least 1000"),
        ("NUMS_TRACE_MAX_BYTES", "1023", "at least 1024"),
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


def test_history_and_trace_cannot_share_a_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    shared = tmp_path / "state.json"
    monkeypatch.setenv("NUMS_HISTORY_FILE", str(shared))
    monkeypatch.setenv("NUMS_TRACE_FILE", str(tmp_path / ".." / tmp_path.name / "state.json"))

    with pytest.raises(ValueError, match="must be different files"):
        Settings.from_env()


def test_distinct_history_and_trace_files_are_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("NUMS_HISTORY_FILE", str(tmp_path / "history.json"))
    monkeypatch.setenv("NUMS_TRACE_FILE", str(tmp_path / "trace.jsonl"))

    settings = Settings.from_env()

    assert settings.history_file != settings.trace_file


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_whisper_model_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("NUMS_WHISPER_MODEL", value)

    with pytest.raises(ValueError, match="NUMS_WHISPER_MODEL cannot be empty"):
        Settings.from_env()
