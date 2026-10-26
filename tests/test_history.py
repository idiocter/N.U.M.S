import json
import os
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


def test_history_open_uses_no_follow_flag(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "history.json"
    path.write_text("[]")
    original_open = os.open
    flags = []

    def recording_open(requested, requested_flags, *args):
        flags.append(requested_flags)
        return original_open(requested, requested_flags, *args)

    monkeypatch.setattr("nums.history.os.open", recording_open)

    assert HistoryStore(path).load() == []
    assert flags[0] & os.O_NOFOLLOW


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


def test_invalid_save_keeps_previous_history(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    original = json.dumps([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ])
    path.write_text(original)

    with pytest.raises(ValueError, match="incomplete NUMS history"):
        HistoryStore(path).save([{"role": "user", "content": "unfinished"}])

    assert path.read_text() == original


def test_load_reports_unreadable_path_as_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def deny(self: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "lstat", deny)

    with pytest.raises(ValueError, match="Cannot read NUMS history.*denied"):
        HistoryStore(tmp_path / "history.json").load()


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    assert HistoryStore(tmp_path / "missing.json").load() == []


def test_save_syncs_the_directory_after_the_atomic_rename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr("nums.history.os.fsync", lambda fd: events.append("fsync") or real_fsync(fd))
    monkeypatch.setattr(
        "nums.history.os.replace", lambda a, b: events.append("replace") or real_replace(a, b)
    )

    HistoryStore(tmp_path / "history.json").save([
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
    ])

    assert events == ["fsync", "replace", "fsync"]


def _call(name: str = "path_info") -> dict:
    return {"function": {"name": name, "arguments": {"path": "/tmp"}}}


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": 5}],
        [{"role": "system", "content": "x"}],
        [{"role": "tool", "content": "x"}],
        [{"role": "tool", "tool_name": "", "content": "x"}],
        [{"role": "tool", "tool_name": "path_info", "content": 5}],
        [{"role": "assistant"}],
        [{"role": "assistant", "content": 5}],
        [{"role": "assistant", "content": "", "tool_calls": []}],
        [{"role": "assistant", "content": "", "tool_calls": "x"}],
        [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": ""}}]}],
        [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t", "arguments": 3}}]}],
        ["not a message"],
    ],
)
def test_malformed_messages_are_rejected_on_load_and_save(tmp_path: Path, messages: list) -> None:
    path = tmp_path / "history.json"
    path.write_text(json.dumps(messages))

    with pytest.raises(ValueError, match="Invalid NUMS history"):
        HistoryStore(path).load()
    with pytest.raises(ValueError, match="Invalid NUMS history"):
        HistoryStore(tmp_path / "other.json").save(messages)


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "assistant", "content": "orphan reply"}],
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "", "tool_calls": [_call("path_info")]},
            {"role": "tool", "tool_name": "other_tool", "content": "{}"},
            {"role": "assistant", "content": "done"},
        ],
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "", "tool_calls": [_call()]},
            {"role": "assistant", "content": "skipped the tool result"},
        ],
    ],
)
def test_out_of_order_history_is_rejected(tmp_path: Path, messages: list) -> None:
    path = tmp_path / "history.json"
    path.write_text(json.dumps(messages))

    with pytest.raises(ValueError, match="Invalid NUMS history sequence"):
        HistoryStore(path).load()


def test_load_drops_only_the_final_unanswered_turn(tmp_path: Path) -> None:
    complete = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "first"}]
    for dangling in (
        [{"role": "user", "content": "two"}],
        [
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "", "tool_calls": [_call()]},
        ],
        [
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "", "tool_calls": [_call(), _call()]},
            {"role": "tool", "tool_name": "path_info", "content": "{}"},
        ],
    ):
        path = tmp_path / "history.json"
        path.write_text(json.dumps(complete + dangling))

        assert HistoryStore(path).load() == complete


def test_save_refuses_an_unanswered_turn_and_keeps_the_previous_file(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.json")
    complete = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "first"}]
    store.save(complete)
    before = store.path.read_bytes()

    with pytest.raises(ValueError, match="Cannot save incomplete NUMS history"):
        store.save(complete + [{"role": "user", "content": "two"}])

    assert store.path.read_bytes() == before


@pytest.mark.parametrize("content", ["not json", '{"role": "user"}', "\xff\xfe"])
def test_unreadable_history_content_is_a_clean_error(tmp_path: Path, content: str) -> None:
    path = tmp_path / "history.json"
    path.write_bytes(content.encode("latin-1"))

    with pytest.raises(ValueError, match="NUMS history"):
        HistoryStore(path).load()


def test_history_size_limits_and_special_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        HistoryStore(tmp_path / "history.json", max_bytes=0)

    big = tmp_path / "big.json"
    big.write_text(json.dumps([{"role": "user", "content": "x" * 200}]))
    with pytest.raises(ValueError, match="exceeds the 50-byte limit"):
        HistoryStore(big, max_bytes=50).load()

    with pytest.raises(ValueError, match="not a regular file"):
        HistoryStore(tmp_path).load()

    link = tmp_path / "link.json"
    link.symlink_to(big)
    with pytest.raises(ValueError, match="not a regular file"):
        HistoryStore(link).load()
