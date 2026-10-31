import os
import pytest

from nums.config import Settings


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("NUMS_MAX_STEPS", "0", "at least 1"),
        ("NUMS_MAX_TOOL_CALLS", "0", "at least 1"),
        ("NUMS_OLLAMA_TIMEOUT", "0", "at least 1"),
        ("NUMS_MAX_STEPS", "many", "integer"),
        ("NUMS_KEEP_ALIVE", "soon", "NUMS_KEEP_ALIVE must be"),
        ("NUMS_KEEP_ALIVE", "-5", "NUMS_KEEP_ALIVE must be"),
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


@pytest.mark.parametrize(
    "url",
    [
        "ftp://127.0.0.1:11434",
        "127.0.0.1:11434",
        "http://",
        "http://user:pass@127.0.0.1:11434",
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11434?x=1",
        "http://127.0.0.1:11434#frag",
        "http://127.0.0.1:0",
        "http://127.0.0.1:99999",
        "http://127.0.0.1:port",
    ],
)
def test_ollama_url_must_be_a_bare_http_host(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("NUMS_OLLAMA_URL", url)

    with pytest.raises(ValueError, match="NUMS_OLLAMA_URL must be an HTTP URL"):
        Settings.from_env()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://localhost:11434", "http://localhost:11434"),
        ("http://localhost:11434/", "http://localhost:11434"),
        ("https://ollama.internal", "https://ollama.internal"),
        ("http://[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_valid_ollama_urls_are_normalized(monkeypatch: pytest.MonkeyPatch, raw: str, expected: str) -> None:
    monkeypatch.setenv("NUMS_OLLAMA_URL", raw)

    assert Settings.from_env().ollama_url == expected


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("NUMS_MODEL", "   ", "NUMS_MODEL cannot be empty"),
        ("NUMS_WAKE_PHRASE", "", "NUMS_WAKE_PHRASE cannot be empty"),
        ("NUMS_SPEAK", "yes", "NUMS_SPEAK must be 0 or 1"),
        ("NUMS_SPEAK", "true", "NUMS_SPEAK must be 0 or 1"),
        ("NUMS_ACTION_MODE", "root", "must be read_only, standard, or unrestricted"),
        ("NUMS_SLEEP_PHRASES", " | |", "must include at least one phrase"),
        ("NUMS_CAPTURE_DEVICE", "-2", "at least -1"),
        ("NUMS_SESSION_TIMEOUT", "0", "at least 1"),
        ("NUMS_REPEAT_TOOL_LIMIT", "0", "at least 1"),
    ],
)
def test_invalid_values_name_the_variable(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        Settings.from_env()


def test_valid_values_are_trimmed_normalized_and_expanded(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NUMS_MODEL", "  qwen:test  ")
    monkeypatch.setenv("NUMS_WAKE_PHRASE", "  Hey Buddy  ")
    monkeypatch.setenv("NUMS_ACTION_MODE", " Read_Only ")
    monkeypatch.setenv("NUMS_SPEAK", "1")
    monkeypatch.setenv("NUMS_SLEEP_PHRASES", " night | stop listening |")
    monkeypatch.setenv("NUMS_HISTORY_FILE", "~/nums/history.json")
    monkeypatch.setenv("NUMS_TRACE_FILE", "~/nums/trace.jsonl")
    monkeypatch.setenv("NUMS_WHISPER_MODEL", "~/models/voice.bin")
    monkeypatch.setenv("NUMS_CAPTURE_DEVICE", "2")

    settings = Settings.from_env()

    assert settings.model == "qwen:test"
    assert settings.wake_phrase == "Hey Buddy"
    assert settings.action_mode == "read_only"
    assert settings.speak is True
    assert settings.sleep_phrases == ("night", "stop listening")
    assert settings.history_file == str(tmp_path / "nums" / "history.json")
    assert settings.trace_file == str(tmp_path / "nums" / "trace.jsonl")
    assert settings.whisper_model == str(tmp_path / "models" / "voice.bin")
    assert settings.capture_device == 2


def test_unset_environment_yields_the_documented_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [key for key in os.environ if key.startswith("NUMS_")]:
        monkeypatch.delenv(name)

    settings = Settings.from_env()

    assert settings.history_file is None and settings.trace_file is None
    assert settings == Settings(whisper_model=settings.whisper_model)


@pytest.mark.parametrize("value", ["30m", "90", "2h", "-1", "0"])
def test_keep_alive_accepts_ollama_durations(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("NUMS_KEEP_ALIVE", value)

    assert Settings.from_env().keep_alive == value


def test_models_stay_loaded_for_half_an_hour_by_default() -> None:
    assert Settings().keep_alive == "30m"
