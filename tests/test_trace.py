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


def test_trace_failure_does_not_hide_completed_response(tmp_path: Path) -> None:
    class Client:
        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            return {"message": {"role": "assistant", "content": "done"}}

    target = tmp_path / "other.txt"
    target.write_text("original")
    link = tmp_path / "trace.jsonl"
    link.symlink_to(target)
    agent = Agent(Settings(trace_file=str(link)))
    agent.client = Client()  # type: ignore[assignment]

    assert agent.run("hello") == "done"
    assert agent.last_run["status"] == "completed"
    assert agent.trace_error
    assert target.read_text() == "original"


def test_trace_store_stops_before_size_limit(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("existing\n")
    store = TraceStore(trace, max_bytes=32)

    with pytest.raises(ValueError, match="trace (entry exceeds|reached)"):
        store.append("a prompt too large for the remaining space", [], "done")

    assert trace.read_text() == "existing\n"


def test_trace_limit_must_be_positive(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        TraceStore(tmp_path / "trace.jsonl", max_bytes=0)


def test_oversized_entry_is_rejected_without_creating_the_file(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"

    with pytest.raises(ValueError, match="entry exceeds the 100-byte trace limit"):
        TraceStore(path, max_bytes=100).append("x" * 500, [], None)

    assert not path.exists()


def test_trace_stops_at_its_limit_and_preserves_existing_entries(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    store = TraceStore(path, max_bytes=700)
    store.append("first", [], "ok")
    store.append("second", [], "ok")
    before = path.read_bytes()

    with pytest.raises(ValueError, match="reached the 700-byte limit"):
        for number in range(20):
            store.append(f"later request {number}", [], "ok")

    assert path.read_bytes().startswith(before)
    assert path.stat().st_size <= 700
    assert all(json.loads(line) for line in path.read_text().splitlines())


def test_entries_keep_unicode_and_tool_details(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"

    TraceStore(path).append(
        "ouvre le café ☕", [{"tool": "open_item", "arguments": {"target": "Café"}}], "fait", "none",
    )

    entry = json.loads(path.read_text(encoding="utf-8"))
    assert "café ☕" in path.read_text(encoding="utf-8")
    assert entry["prompt"] == "ouvre le café ☕"
    assert entry["tool_calls"][0]["arguments"] == {"target": "Café"}
    assert entry["reply"] == "fait" and entry["error"] == "none"
    assert entry["at"].endswith("+00:00")


def test_existing_loose_permissions_are_tightened(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text("")
    path.chmod(0o644)

    TraceStore(path).append("hi", [], "hello")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_symlinked_and_directory_trace_paths_are_refused(tmp_path) -> None:
    real = tmp_path / "real.jsonl"
    real.write_text("")
    link = tmp_path / "link.jsonl"
    link.symlink_to(real)

    with pytest.raises(OSError):
        TraceStore(link).append("hi", [], "hello")
    with pytest.raises(OSError):
        TraceStore(tmp_path).append("hi", [], "hello")

    assert real.read_text() == ""


def test_concurrent_appends_never_interleave(tmp_path) -> None:
    import threading

    path = tmp_path / "trace.jsonl"
    store = TraceStore(path)
    errors = []

    def writer(worker: int) -> None:
        try:
            for number in range(25):
                store.append(f"worker {worker} request {number}", [], "x" * 200)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(worker,)) for worker in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = path.read_text().splitlines()
    assert errors == []
    assert len(lines) == 150
    assert len({json.loads(line)["prompt"] for line in lines}) == 150
