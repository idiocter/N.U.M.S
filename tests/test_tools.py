import json
import stat
from pathlib import Path

import pytest

from nums.tools import MacTools


def test_read_and_list(tmp_path: Path) -> None:
    file = tmp_path / "hello.txt"
    file.write_text("hello NUMS")
    tools = MacTools()
    assert tools.read_file({"path": str(file)}) == "hello NUMS"
    assert "hello.txt" in tools.list_directory({"path": str(tmp_path)})


def test_write_replaces_content_and_preserves_permissions(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("before")
    path.chmod(0o640)

    MacTools().write_file({"path": str(path), "content": "after"})

    assert path.read_text() == "after"
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_failed_replace_keeps_previous_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("before")

    def fail(*args: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("nums.tools.os.replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        MacTools().write_file({"path": str(path), "content": "after"})

    assert path.read_text() == "before"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["note.txt"]


def test_read_only_mode_blocks_mutating_tools(tmp_path: Path) -> None:
    target = tmp_path / "blocked.txt"

    result = json.loads(MacTools("read_only").execute(
        "write_file", {"path": str(target), "content": "no"}
    ))

    assert result["action_mode"] == "read_only"
    assert not target.exists()


def test_standard_mode_blocks_shell() -> None:
    result = json.loads(MacTools("standard").execute("shell", {"command": "echo no"}))
    assert result["action_mode"] == "standard"


def test_clipboard_write_uses_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Result:
        returncode = 0

    def fake_run(command: list[str], **kwargs: object) -> Result:
        calls.append((command, kwargs))
        return Result()

    monkeypatch.setattr("nums.tools.subprocess.run", fake_run)
    result = json.loads(MacTools("standard").set_clipboard({"text": "hello"}))

    assert result == {"exit_code": 0, "characters": 5}
    assert calls[0][0] == ["pbcopy"]
    assert calls[0][1]["input"] == "hello"


def test_reminder_passes_values_as_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command: commands.append(command) or "ok")

    MacTools("standard").create_reminder({"title": 'Call "Sam"', "notes": "At 4"})

    assert commands[0][-2:] == ['Call "Sam"', "At 4"]


def test_search_treats_dash_prefixed_query_as_text(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("-TODO follow up\n")

    result = json.loads(MacTools().search_files({"query": "-TODO", "path": str(tmp_path)}))

    assert result["exit_code"] == 0
    assert "-TODO follow up" in result["output"]


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("Safari", ["open", "-a", "Safari"]),
        ("https://example.com", ["open", "https://example.com"]),
        ("/tmp/report.pdf", ["open", "/tmp/report.pdf"]),
    ],
)
def test_open_item_dispatches_apps_and_urls(monkeypatch: pytest.MonkeyPatch, target: str, expected: list[str]) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command: commands.append(command) or "ok")

    assert MacTools().open_item({"target": target}) == "ok"
    assert commands == [expected]
