import json
import stat
from pathlib import Path
from typing import Any

import pytest

from nums.agent import Agent
from nums.config import Settings
from nums.history import HistoryStore


def test_opt_in_history_survives_restart_and_is_private(tmp_path: Path) -> None:
    class ReplyClient:
        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            return {"message": {"role": "assistant", "content": "hello"}}

    path = tmp_path / "private" / "history.json"
    settings = Settings(history_file=str(path))
    first = Agent(settings)
    first.client = ReplyClient()  # type: ignore[assignment]
    assert first.run("hi") == "hello"

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())[0]["content"] == "hi"
    second = Agent(settings)
    assert [item["role"] for item in second.messages] == ["system", "user", "assistant"]
    second.reset()
    assert json.loads(path.read_text()) == []
    assert len(Agent(settings).messages) == 1


def test_corrupt_history_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text("not JSON")
    with pytest.raises(ValueError, match="Cannot read NUMS history"):
        HistoryStore(path).load()


def test_history_refuses_symlink_input(tmp_path: Path) -> None:
    target = tmp_path / "actual.json"
    target.write_text("[]")
    link = tmp_path / "history.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="not a regular file"):
        HistoryStore(link).load()


def test_history_read_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text(json.dumps([{"role": "user", "content": "x" * 100}]))

    with pytest.raises(ValueError, match="exceeds the 32-byte limit"):
        HistoryStore(path, max_bytes=32).load()


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": 7},
        {"role": "assistant"},
        {"role": "assistant", "content": []},
        {"role": "assistant", "tool_calls": []},
        {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]},
        {"role": "tool", "tool_name": "", "content": "result"},
        {"role": "tool", "tool_name": "read_file", "content": {}},
    ],
)
def test_history_validates_message_fields(tmp_path: Path, message: dict) -> None:
    path = tmp_path / "history.json"
    path.write_text(json.dumps([message]))

    with pytest.raises(ValueError, match="Invalid NUMS history"):
        HistoryStore(path).load()


def test_legacy_history_drops_unfinished_tool_turn(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    complete = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    unfinished = [
        {"role": "user", "content": "read a file"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "/tmp/note"}}}
        ]},
        {"role": "tool", "tool_name": "read_file", "content": "note"},
    ]
    path.write_text(json.dumps(complete + unfinished))

    assert HistoryStore(path).load() == complete
    assert [item["role"] for item in Agent(Settings(history_file=str(path))).messages] == [
        "system", "user", "assistant"
    ]


def test_legacy_history_drops_trailing_user_without_response(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text(json.dumps([{"role": "user", "content": "unanswered"}]))

    assert HistoryStore(path).load() == []


def test_history_rejects_invalid_message_order(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text(json.dumps([
        {"role": "user", "content": "read"},
        {"role": "tool", "tool_name": "read_file", "content": "result"},
        {"role": "assistant", "content": "done"},
    ]))

    with pytest.raises(ValueError, match="Invalid NUMS history sequence"):
        HistoryStore(path).load()


def test_history_rejects_mismatched_tool_result(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text(json.dumps([
        {"role": "user", "content": "read"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "/tmp/a"}}}
        ]},
        {"role": "tool", "tool_name": "write_file", "content": "result"},
        {"role": "assistant", "content": "done"},
    ]))

    with pytest.raises(ValueError, match="Invalid NUMS history sequence"):
        HistoryStore(path).load()
