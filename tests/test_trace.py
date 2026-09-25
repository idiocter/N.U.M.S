import json
import stat
from pathlib import Path
from typing import Any

import pytest

from nums.agent import Agent
from nums.config import Settings
from nums.trace import TraceStore


def test_opt_in_trace_records_decisions_without_tool_results(tmp_path: Path) -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {"message": {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "/tmp/note.txt"}}}
                ]}}
            return {"message": {"role": "assistant", "content": "done"}}

    class Tools:
        def execute(self, name: str, args: dict[str, Any]) -> str:
            return "private file contents"

    trace = tmp_path / "private" / "trace.jsonl"
    agent = Agent(Settings(trace_file=str(trace)))
    agent.client = Client()  # type: ignore[assignment]
    agent.tools = Tools()  # type: ignore[assignment]
    assert agent.run("read my note") == "done"

    entry = json.loads(trace.read_text().strip())
    assert entry["prompt"] == "read my note"
    assert entry["tool_calls"] == [{"tool": "read_file", "arguments": {"path": "/tmp/note.txt"}}]
    assert "private file contents" not in trace.read_text()
    assert stat.S_IMODE(trace.stat().st_mode) == 0o600


def test_trace_store_refuses_symlink_target(tmp_path: Path) -> None:
    target = tmp_path / "other.txt"
    target.write_text("original")
    link = tmp_path / "trace.jsonl"
    link.symlink_to(target)

    with pytest.raises(OSError):
        TraceStore(link).append("private prompt", [], None)

    assert target.read_text() == "original"
