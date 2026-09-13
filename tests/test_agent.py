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
                {"message": {"role": "assistant", "content": "The write was denied."}},
            ]
        )

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        return next(self.responses)


class FailingTools:
    def execute(self, name: str, args: dict[str, Any]) -> str:
        raise AssertionError("a denied action must not be executed")


class EmptyClient:
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        return {"message": {"role": "assistant", "content": ""}}


def test_denied_action_never_reaches_tool_executor() -> None:
    agent = Agent(Settings(), confirm=lambda name, args, reason: False)
    agent.client = FakeClient()  # type: ignore[assignment]
    agent.tools = FailingTools()  # type: ignore[assignment]

    response = agent.run("write a file")

    assert response == "The write was denied."
    assert agent.messages[-2]["content"] == '{"denied": true}'


def test_empty_model_response_has_a_spoken_fallback() -> None:
    agent = Agent(Settings(), confirm=lambda name, args, reason: False)
    agent.client = EmptyClient()  # type: ignore[assignment]

    assert agent.run("hello") == "I couldn't produce a response."
