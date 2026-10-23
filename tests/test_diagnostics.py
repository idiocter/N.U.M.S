import json
import pytest

from nums.config import Settings
from nums.diagnostics import probe_mac_ui, probe_model_tool_call, self_test, voice_test
from nums.ollama import OllamaError


def test_self_test_checks_local_files_and_model_tool_call(monkeypatch) -> None:
    monkeypatch.setattr("nums.diagnostics.voice_dependencies", lambda _: (True, True))
    monkeypatch.setattr(
        "nums.diagnostics.OllamaClient.chat",
        lambda self, messages, tools: {
            "message": {"role": "assistant", "tool_calls": [
                {"function": {"name": "system_info", "arguments": {}}}
            ]}
        },
    )

    passed, results = self_test(Settings())

    assert passed
    assert results == ["File tools: ok", "Voice dependencies: ok", "Model tool call: ok"]


def test_voice_test_stops_after_first_transcript(monkeypatch) -> None:
    stopped = []

    class Stream:
        def __init__(self, *args) -> None:
            pass

        def transcripts(self, timeout_seconds=None):
            yield "hello NUMS"

        def stop(self) -> None:
            stopped.append(True)

    monkeypatch.setattr("nums.diagnostics.WhisperStream", Stream)

    assert voice_test(Settings(), timeout_seconds=1) == "hello NUMS"
    assert stopped == [True]


class _ProbeClient:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome

    def __call__(self, *args: object, **kwargs: object) -> "_ProbeClient":
        return self

    def chat(self, messages: list, tools: list) -> dict:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return {"message": {"role": "assistant", "tool_calls": self.outcome}}


def _call(name: str) -> dict:
    return {"function": {"name": name, "arguments": {}}}


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ([_call("system_info")], (True, None)),
        ([_call("read_file")], (False, "model did not choose system_info")),
        ([], (False, "model did not choose system_info")),
        ([_call("system_info"), _call("system_info")], (False, "model did not choose system_info")),
        (OllamaError("model not loaded"), (False, "model not loaded")),
    ],
)
def test_model_probe_requires_exactly_the_requested_tool(
    monkeypatch: pytest.MonkeyPatch, outcome: object, expected: tuple[bool, str | None]
) -> None:
    monkeypatch.setattr("nums.diagnostics.OllamaClient", _ProbeClient(outcome))

    assert probe_model_tool_call(Settings()) == expected


def test_mac_ui_probe_requires_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.diagnostics.platform.system", lambda: "Linux")

    assert probe_mac_ui() == (False, "Mac app control requires macOS")


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"exit_code": 0, "output": "Finder"}, (True, None)),
        ({"exit_code": 1, "hint": "grant Accessibility", "output": "x", "error": "y"}, (False, "grant Accessibility")),
        ({"exit_code": 1, "output": "not authorized", "error": "y"}, (False, "not authorized")),
        ({"exit_code": 1, "output": "", "error": "Command exited with status 1"}, (False, "Command exited with status 1")),
        ({"exit_code": 1}, (False, "System Events did not respond")),
    ],
)
def test_mac_ui_probe_reports_the_most_helpful_failure(
    monkeypatch: pytest.MonkeyPatch, result: dict, expected: tuple[bool, str | None]
) -> None:
    monkeypatch.setattr("nums.diagnostics.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "nums.diagnostics.MacTools.execute", lambda self, name, args: json.dumps(result)
    )

    assert probe_mac_ui() == expected
