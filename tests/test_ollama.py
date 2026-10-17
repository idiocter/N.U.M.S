import io
import json
import urllib.error

import pytest

from nums.ollama import OllamaClient, OllamaError


def test_chat_rejects_missing_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.ollama.urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"{}"))

    with pytest.raises(OllamaError, match="without an assistant message"):
        OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])


@pytest.mark.parametrize("role", ["user", "tool", None])
def test_chat_rejects_non_assistant_role(monkeypatch: pytest.MonkeyPatch, role: str | None) -> None:
    payload = {"message": {"role": role, "content": "hello"}}
    monkeypatch.setattr(
        "nums.ollama.urllib.request.urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()),
    )

    with pytest.raises(OllamaError, match="invalid role"):
        OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])


def test_chat_explains_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("nums.ollama.urllib.request.urlopen", fail)

    with pytest.raises(OllamaError, match="Cannot reach Ollama"):
        OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])


def test_chat_bounds_http_error_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.ollama.ERROR_DETAIL_MAX_BYTES", 8)
    requested: list[int | None] = []

    class RecordingBody:
        def read(self, size: int | None = None) -> bytes:
            requested.append(size)
            body = b"detail-" * 1000
            return body if size is None else body[:size]

        def close(self) -> None:
            pass

    def raise_http(*args: object, **kwargs: object) -> None:
        raise urllib.error.HTTPError(
            "http://127.0.0.1:11434/api/chat", 500, "Server Error", {}, RecordingBody()
        )

    monkeypatch.setattr("nums.ollama.urllib.request.urlopen", raise_http)

    with pytest.raises(OllamaError, match="Ollama returned HTTP 500: detail-d"):
        OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])

    assert requested == [8]


def test_chat_explains_low_level_socket_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nums.ollama.urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("socket closed")),
    )

    with pytest.raises(OllamaError, match="Cannot reach Ollama"):
        OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])


def test_client_normalizes_trailing_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    def respond(request, **kwargs):
        requests.append(request.full_url)
        return io.BytesIO(b'{"message":{"role":"assistant","content":"ok"}}')

    monkeypatch.setattr("nums.ollama.urllib.request.urlopen", respond)

    OllamaClient("http://127.0.0.1:11434/", "qwen").chat([], [])

    assert requests == ["http://127.0.0.1:11434/api/chat"]


def test_chat_rejects_oversized_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.ollama.CHAT_RESPONSE_MAX_BYTES", 32)
    monkeypatch.setattr(
        "nums.ollama.urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"x" * 33),
    )

    with pytest.raises(OllamaError, match="exceeded the 32-byte limit"):
        OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])


def test_model_check_treats_malformed_json_as_unready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.ollama.urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"not json"))

    assert not OllamaClient("http://127.0.0.1:11434", "qwen").has_model()


def test_model_check_treats_socket_failure_as_unready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nums.ollama.urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("socket closed")),
    )

    assert not OllamaClient("http://127.0.0.1:11434", "qwen").has_model()


def test_model_check_rejects_oversized_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.ollama.TAGS_RESPONSE_MAX_BYTES", 16)
    monkeypatch.setattr(
        "nums.ollama.urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"x" * 17),
    )

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
