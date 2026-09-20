from typing import Any

import pytest

from nums.agent import Agent
from nums.config import Settings
from nums.ollama import OllamaError


class FakeClient:
    def __init__(self) -> None:
        self.responses = iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "write_file",
                                    "arguments": {"path": "/tmp/blocked", "content": "no"},
                                }
                            }
                        ],
                    }
                },
                {"message": {"role": "assistant", "content": "The write completed."}},
            ]
        )

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        return next(self.responses)


class RecordingTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(self, name: str, args: dict[str, Any]) -> str:
        self.calls.append((name, args))
        return '{"written": "/tmp/blocked"}'


class EmptyClient:
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        return {"message": {"role": "assistant", "content": ""}}


def test_tool_action_executes_without_confirmation() -> None:
    agent = Agent(Settings())
    agent.client = FakeClient()  # type: ignore[assignment]
    tools = RecordingTools()
    agent.tools = tools  # type: ignore[assignment]

    response = agent.run("write a file")

    assert response == "The write completed."
    assert tools.calls == [("write_file", {"path": "/tmp/blocked", "content": "no"})]
    assert agent.messages[-2]["content"] == '{"written": "/tmp/blocked"}'


def test_empty_model_response_has_a_spoken_fallback() -> None:
    agent = Agent(Settings())
    agent.client = EmptyClient()  # type: ignore[assignment]

    assert agent.run("hello") == "I couldn't produce a response."


def test_failed_model_request_does_not_replay_user_prompt() -> None:
    class FailingClient:
        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            raise OllamaError("Ollama disconnected")

    agent = Agent(Settings())
    agent.client = FailingClient()  # type: ignore[assignment]

    with pytest.raises(OllamaError, match="disconnected"):
        agent.run("open Safari")

    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"


def test_long_session_keeps_only_recent_complete_turns() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.seen: list[list[str]] = []

        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            self.seen.append([m["content"] for m in messages if m["role"] == "user"])
            return {"message": {"role": "assistant", "content": "ok"}}

    agent = Agent(Settings(history_turns=2))
    client = RecordingClient()
    agent.client = client  # type: ignore[assignment]
    for prompt in ("first", "second", "third"):
        agent.run(prompt)

    assert client.seen == [["first"], ["first", "second"], ["second", "third"]]
