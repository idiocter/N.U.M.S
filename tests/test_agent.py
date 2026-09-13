from typing import Any

from nums.agent import Agent
from nums.config import Settings


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
