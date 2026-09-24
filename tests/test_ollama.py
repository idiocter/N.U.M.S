import io
import json
import urllib.error

import pytest

from nums.ollama import OllamaClient, OllamaError


def test_chat_rejects_missing_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.ollama.urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"{}"))

    with pytest.raises(OllamaError, match="without an assistant message"):
        OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])


def test_chat_explains_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("nums.ollama.urllib.request.urlopen", fail)

    with pytest.raises(OllamaError, match="Cannot reach Ollama"):
        OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])


def test_model_check_treats_malformed_json_as_unready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.ollama.urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"not json"))

    assert not OllamaClient("http://127.0.0.1:11434", "qwen").has_model()


@pytest.mark.parametrize("tool_calls", ["wrong", [None], [{"function": {"name": "shell"}}]])
def test_chat_rejects_malformed_tool_calls_before_execution(
    monkeypatch: pytest.MonkeyPatch, tool_calls: object
) -> None:
    payload = {"message": {"role": "assistant", "tool_calls": tool_calls}}
    monkeypatch.setattr(
        "nums.ollama.urllib.request.urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()),
    )

    with pytest.raises(OllamaError, match="malformed tool calls"):
        OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])


def test_model_check_rejects_malformed_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nums.ollama.urllib.request.urlopen",
        lambda *args, **kwargs: io.BytesIO(b'{"models": [null, "bad"]}'),
    )

    assert not OllamaClient("http://127.0.0.1:11434", "qwen").has_model()
