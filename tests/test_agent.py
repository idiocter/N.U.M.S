import json
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
    assert agent.messages[-1]["content"] == "I couldn't produce a response."


def test_oversized_prompt_stops_before_model_request() -> None:
    class UnexpectedClient:
        def chat(self, messages, tools):
            pytest.fail("oversized prompt reached the model")

    baseline = Agent(Settings())
    system_size = len(json.dumps(baseline.system_message, ensure_ascii=False))
    agent = Agent(Settings(max_context_chars=system_size + 40))
    agent.client = UnexpectedClient()  # type: ignore[assignment]

    reply = agent.run("x" * 100)

    assert "too large" in reply
    assert agent.last_run == {
        "status": "context_limit", "steps": 0, "tool_calls": 0, "tool_errors": 0
    }
    assert agent.messages == [agent.system_message]


def test_coding_context_includes_working_directory_and_survives_reset(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    agent = Agent(Settings())

    assert str(tmp_path) in agent.messages[0]["content"]
    assert "review the resulting diff" in agent.messages[0]["content"]
    agent.reset()
    assert str(tmp_path) in agent.messages[0]["content"]


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


def test_disconnect_after_tool_call_preserves_executed_action(tmp_path) -> None:
    class DisconnectingClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {"message": {"role": "assistant", "tool_calls": [
                    {"function": {"name": "system_info", "arguments": {}}}
                ]}}
            raise OllamaError("Ollama disconnected")

    history = tmp_path / "history.json"
    agent = Agent(Settings(history_file=str(history)))
    agent.client = DisconnectingClient()  # type: ignore[assignment]
    tools = RecordingTools()
    agent.tools = tools  # type: ignore[assignment]

    with pytest.raises(OllamaError, match="disconnected"):
        agent.run("check my Mac")

    assert tools.calls == [("system_info", {})]
    assert [message["role"] for message in agent.messages] == [
        "system", "user", "assistant", "tool", "assistant"
    ]
    assert "could not confirm" in agent.messages[-1]["content"]
    assert len(json.loads(history.read_text())) == 4
    assert agent.last_run["status"] == "model_error"


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


def test_context_budget_discards_old_complete_turns() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.seen: list[list[str]] = []

        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            self.seen.append([item["content"] for item in messages if item["role"] == "user"])
            return {"message": {"role": "assistant", "content": "x" * 250}}

    system_size = len(json.dumps(Agent(Settings()).messages[0], ensure_ascii=False))
    agent = Agent(Settings(history_turns=8, max_context_chars=system_size + 500))
    client = RecordingClient()
    agent.client = client  # type: ignore[assignment]
    for prompt in ("first", "second", "third"):
        agent.run(prompt)

    assert client.seen[0] == ["first"]
    assert client.seen[-1] == ["second", "third"]










def test_repeated_tool_call_can_continue_when_results_change() -> None:
    class ChangingClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            self.calls += 1
            if self.calls <= 3:
                return {"message": {"role": "assistant", "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "/tmp/status"}}}
                ]}}
            return {"message": {"role": "assistant", "content": "The status changed."}}

    class ChangingTools:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, name: str, args: dict[str, Any]) -> str:
            self.calls += 1
            return f"status {self.calls}"

    agent = Agent(Settings())
    agent.client = ChangingClient()  # type: ignore[assignment]
    tools = ChangingTools()
    agent.tools = tools  # type: ignore[assignment]

    assert agent.run("watch status") == "The status changed."
    assert tools.calls == 3
    assert agent.last_run["status"] == "completed"




def test_run_report_counts_tool_errors() -> None:
    agent = Agent(Settings())
    agent.client = FakeClient()  # type: ignore[assignment]

    class ErrorTools:
        def execute(self, name: str, args: dict[str, Any]) -> str:
            return '{"error": "failed"}'

    agent.tools = ErrorTools()  # type: ignore[assignment]
    assert agent.run("write") == "The write completed."
    assert agent.last_run == {
        "status": "completed", "steps": 2, "tool_calls": 1, "tool_errors": 1
    }




def test_invalid_json_tool_arguments_are_reported_to_model() -> None:
    class MalformedClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {"message": {"role": "assistant", "tool_calls": [
                    {"function": {"name": "read_file", "arguments": "{broken"}}
                ]}}
            assert "must be a JSON object" in messages[-1]["content"]
            return {"message": {"role": "assistant", "content": "I could not read that file."}}

    agent = Agent(Settings())
    agent.client = MalformedClient()  # type: ignore[assignment]

    assert agent.run("read my file") == "I could not read that file."
    assert agent.last_run["tool_errors"] == 1


def test_unsupported_model_message_fields_are_not_kept() -> None:
    class ThinkingClient:
        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            return {"message": {
                "role": "assistant", "content": "Done.",
                "thinking": "long private reasoning " * 100, "images": ["x"],
            }}

    agent = Agent(Settings())
    agent.client = ThinkingClient()  # type: ignore[assignment]

    assert agent.run("hello") == "Done."
    assert agent.messages[-1] == {"role": "assistant", "content": "Done."}


def test_interrupted_request_rolls_back_the_partial_turn() -> None:
    class InterruptingClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {"message": {"role": "assistant", "content": "First answer."}}
            if self.calls == 2:
                return {"message": {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "system_info", "arguments": {}}}
                ]}}
            if self.calls == 3:
                raise KeyboardInterrupt
            return {"message": {"role": "assistant", "content": "Third answer."}}

    agent = Agent(Settings())
    agent.client = InterruptingClient()  # type: ignore[assignment]
    agent.run("first")
    settled = [dict(message) for message in agent.messages]

    with pytest.raises(KeyboardInterrupt):
        agent.run("second")

    assert agent.messages == settled
    assert agent.last_run["status"] == "interrupted"
    assert agent.run("third") == "Third answer."


def test_agent_starts_when_the_working_directory_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    def gone() -> None:
        raise FileNotFoundError("cwd removed")

    monkeypatch.setattr("nums.agent.Path.cwd", gone)

    agent = Agent(Settings())

    assert 'Current working directory: "unavailable"' in agent.system_message["content"]


def test_identical_tool_calls_are_not_cut_off() -> None:
    class PersistentClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            self.calls += 1
            if self.calls <= 25:
                return {"message": {"role": "assistant", "tool_calls": [
                    {"function": {"name": "system_info", "arguments": {}}}
                ]}}
            return {"message": {"role": "assistant", "content": "finished"}}

    agent = Agent(Settings())
    agent.client = PersistentClient()  # type: ignore[assignment]
    tools = RecordingTools()
    agent.tools = tools  # type: ignore[assignment]

    assert agent.run("keep checking") == "finished"
    assert len(tools.calls) == 25
    assert agent.last_run == {
        "status": "completed", "steps": 26, "tool_calls": 25, "tool_errors": 0
    }


def test_a_large_batch_of_tool_calls_all_run() -> None:
    class BatchClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {"message": {"role": "assistant", "tool_calls": [
                    {"function": {"name": "write_file", "arguments": {"path": f"/tmp/{n}", "content": "x"}}}
                    for n in range(40)
                ]}}
            return {"message": {"role": "assistant", "content": "wrote them"}}

    agent = Agent(Settings())
    agent.client = BatchClient()  # type: ignore[assignment]
    tools = RecordingTools()
    agent.tools = tools  # type: ignore[assignment]

    assert agent.run("write many") == "wrote them"
    assert len(tools.calls) == 40


def test_the_model_is_offered_every_tool() -> None:
    from nums.tools import TOOL_SCHEMAS

    class InspectingClient:
        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            assert tools == TOOL_SCHEMAS
            return {"message": {"role": "assistant", "content": "ok"}}

    agent = Agent(Settings())
    agent.client = InspectingClient()  # type: ignore[assignment]

    assert agent.run("anything") == "ok"



def test_system_prompt_sets_a_casual_friendly_tone() -> None:
    from nums.agent import SYSTEM_PROMPT

    assert "close friend" in SYSTEM_PROMPT
    assert "selected action mode" not in SYSTEM_PROMPT
def test_empty_tool_call_list_can_be_saved(tmp_path) -> None:
    class Client:
        def chat(self, messages, tools):
            return {"message": {"role": "assistant", "content": "hello", "tool_calls": []}}

    agent = Agent(Settings(history_file=str(tmp_path / "history.json")))
    agent.client = Client()
    assert agent.run("hi") == "hello"
    assert "tool_calls" not in agent.messages[-1]
    assert Agent(agent.settings).messages[-1]["content"] == "hello"


