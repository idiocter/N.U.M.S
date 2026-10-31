import http.client
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


def test_chat_reports_truncated_response(monkeypatch: pytest.MonkeyPatch) -> None:
    import http.client

    class TruncatedResponse:
        def read(self, *args: object) -> bytes:
            raise http.client.IncompleteRead(b"partial")

        def __enter__(self) -> "TruncatedResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "nums.ollama.urllib.request.urlopen", lambda *args, **kwargs: TruncatedResponse()
    )

    with pytest.raises(OllamaError, match="truncated or malformed"):
        OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])


def test_model_check_treats_truncated_response_as_unready(monkeypatch: pytest.MonkeyPatch) -> None:
    import http.client

    class TruncatedResponse:
        def read(self, *args: object) -> bytes:
            raise http.client.IncompleteRead(b"partial")

        def __enter__(self) -> "TruncatedResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "nums.ollama.urllib.request.urlopen", lambda *args, **kwargs: TruncatedResponse()
    )

    assert not OllamaClient("http://127.0.0.1:11434", "qwen").has_model()


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


def _serve(monkeypatch: pytest.MonkeyPatch, payload: object, seen: list | None = None) -> None:
    def respond(request, **kwargs):
        if seen is not None:
            seen.append(request)
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("nums.ollama.urllib.request.urlopen", respond)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "without an assistant message"),
        ({"message": "text"}, "without an assistant message"),
        ({"message": {"role": "assistant", "content": 5}}, "invalid assistant content"),
        ({"message": {"role": "assistant", "tool_calls": "x"}}, "malformed tool calls"),
        ({"message": {"role": "assistant", "tool_calls": ["x"]}}, "malformed tool calls"),
        ({"message": {"role": "assistant", "tool_calls": [{"function": "x"}]}}, "malformed tool calls"),
        ({"message": {"role": "assistant", "tool_calls": [{"function": {"name": "", "arguments": {}}}]}}, "malformed tool calls"),
        ({"message": {"role": "assistant", "tool_calls": [{"function": {"name": 7, "arguments": {}}}]}}, "malformed tool calls"),
        ({"message": {"role": "assistant", "tool_calls": [{"function": {"name": "t", "arguments": 3}}]}}, "malformed tool calls"),
        ({"message": {"role": "assistant", "tool_calls": [{"function": {"name": "t"}}]}}, "malformed tool calls"),
    ],
)
def test_chat_rejects_malformed_replies(monkeypatch: pytest.MonkeyPatch, payload: object, message: str) -> None:
    _serve(monkeypatch, payload)

    with pytest.raises(OllamaError, match=message):
        OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": "hi", "tool_calls": None},
        {"role": "assistant", "tool_calls": [{"function": {"name": "t", "arguments": {"a": "1"}}}]},
        {"role": "assistant", "tool_calls": [{"function": {"name": "t", "arguments": '{"a": "1"}'}}]},
    ],
)
def test_chat_accepts_well_formed_replies(monkeypatch: pytest.MonkeyPatch, message: dict) -> None:
    _serve(monkeypatch, {"message": message, "done": True})

    result = OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])

    assert result["message"] == message


def test_chat_sends_a_non_streaming_json_request(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    _serve(monkeypatch, {"message": {"role": "assistant", "content": "ok"}}, seen)
    messages = [{"role": "user", "content": "héllo"}]
    tools = [{"type": "function", "function": {"name": "t"}}]

    OllamaClient("http://127.0.0.1:11434", "qwen:test", timeout=7).chat(messages, tools)

    request = seen[0]
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data) == {
        "model": "qwen:test", "messages": messages, "tools": tools, "stream": False,
    }


def test_http_error_reports_status_and_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise urllib.error.HTTPError(
            "http://127.0.0.1:11434/api/chat", 404, "Not Found", {}, io.BytesIO(b'{"error":"model not found"}'),
        )

    monkeypatch.setattr("nums.ollama.urllib.request.urlopen", fail)

    with pytest.raises(OllamaError, match=r"HTTP 404: .*model not found"):
        OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])


@pytest.mark.parametrize(
    ("model", "names", "expected"),
    [
        ("qwen", ["qwen"], True),
        ("qwen", ["qwen:latest"], True),
        ("qwen:1.5b", ["qwen:1.5b-instruct"], False),
        ("qwen:1.5b-instruct", ["qwen:1.5b"], False),
        ("qwen", ["other", "qwen2"], False),
        ("qwen", [], False),
    ],
)
def test_model_presence_requires_an_exact_name(
    monkeypatch: pytest.MonkeyPatch, model: str, names: list, expected: bool
) -> None:
    _serve(monkeypatch, {"models": [{"name": name} for name in names]})

    assert OllamaClient("http://127.0.0.1:11434", model).has_model() is expected


@pytest.mark.parametrize(
    "payload",
    [[], {"models": "qwen"}, {"models": None}, {"models": [{"name": 5}, "qwen", {"model": "qwen"}]}],
)
def test_model_presence_ignores_unexpected_catalog_shapes(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    _serve(monkeypatch, payload)

    assert OllamaClient("http://127.0.0.1:11434", "qwen").has_model() is False


def test_chat_reports_a_malformed_http_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise http.client.BadStatusLine("garbage")

    monkeypatch.setattr("nums.ollama.urllib.request.urlopen", fail)

    with pytest.raises(OllamaError, match="truncated or malformed"):
        OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])


@pytest.mark.parametrize(
    ("configured", "sent"), [("30m", "30m"), ("-1", -1), ("300", 300)]
)
def test_chat_asks_ollama_to_keep_the_model_loaded(
    monkeypatch: pytest.MonkeyPatch, configured: str, sent: object
) -> None:
    seen = []
    _serve(monkeypatch, {"message": {"role": "assistant", "content": "ok"}}, seen)

    OllamaClient("http://127.0.0.1:11434", "qwen", keep_alive=configured).chat([], [])

    assert json.loads(seen[0].data)["keep_alive"] == sent


def test_chat_leaves_keep_alive_to_ollama_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    _serve(monkeypatch, {"message": {"role": "assistant", "content": "ok"}}, seen)

    OllamaClient("http://127.0.0.1:11434", "qwen").chat([], [])

    assert "keep_alive" not in json.loads(seen[0].data)


def test_agent_clients_use_the_configured_keep_alive() -> None:
    from nums.agent import Agent
    from nums.config import Settings

    assert Agent(Settings(keep_alive="2h")).client.keep_alive == "2h"
