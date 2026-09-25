import json
import stat
from pathlib import Path

import pytest

from nums.tools import MacTools, _run


def test_read_and_list(tmp_path: Path) -> None:
    file = tmp_path / "hello.txt"
    file.write_text("hello NUMS")
    tools = MacTools()
    assert tools.read_file({"path": str(file)}) == "hello NUMS"
    listing = json.loads(tools.list_directory({"path": str(tmp_path)}))
    assert listing == {
        "items": [{"name": "hello.txt", "type": "file"}],
        "total": 1,
        "truncated": False,
    }


def test_large_file_and_directory_outputs_warn_about_truncation(tmp_path: Path) -> None:
    file = tmp_path / "large.txt"
    file.write_text("x" * 12001)
    for number in range(500):
        (tmp_path / f"item-{number:03}").touch()

    text = MacTools().read_file({"path": str(file)})
    listing = json.loads(MacTools().list_directory({"path": str(tmp_path)}))

    assert text.startswith("x" * 12000)
    assert "file output truncated" in text
    assert listing["total"] == 501
    assert len(listing["items"]) == 500
    assert listing["truncated"] is True


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


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "missing required arguments"),
        ({"path": "/tmp/x", "content": 123}, "arguments must be strings"),
        ({"path": "/tmp/x", "content": "a", "surprise": "b"}, "unexpected arguments"),
        (["/tmp/x", "a"], "must be a JSON object"),
    ],
)
def test_invalid_tool_arguments_are_rejected_without_writing(
    tmp_path: Path, arguments: object, message: str
) -> None:
    target = tmp_path / "untouched.txt"
    if isinstance(arguments, dict) and arguments.get("path") == "/tmp/x":
        arguments = {**arguments, "path": str(target)}

    result = json.loads(MacTools().execute("write_file", arguments))  # type: ignore[arg-type]

    assert message in result["error"]
    assert not target.exists()


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


def test_clipboard_failure_does_not_report_a_successful_write(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 1
        stderr = "pasteboard unavailable"

    monkeypatch.setattr("nums.tools.subprocess.run", lambda *args, **kwargs: Result())

    result = json.loads(MacTools().set_clipboard({"text": "hello"}))

    assert result["exit_code"] == 1
    assert "pasteboard unavailable" in result["error"]
    assert "characters" not in result


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


def test_search_with_no_matches_is_not_a_tool_error(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello\n")

    result = json.loads(MacTools().search_files({"query": "absent", "path": str(tmp_path)}))

    assert result["exit_code"] == 1
    assert result["matches"] == 0
    assert "error" not in result


def test_nonzero_process_exit_is_reported_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 7
        stdout = ""
        stderr = "permission denied"

    monkeypatch.setattr("nums.tools.subprocess.run", lambda *args, **kwargs: Result())

    result = json.loads(_run(["open", "missing"]))

    assert result["exit_code"] == 7
    assert "error" in result
    assert "permission denied" in result["output"]


def test_large_command_output_reports_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 0
        stdout = "x" * 12001
        stderr = ""

    monkeypatch.setattr("nums.tools.subprocess.run", lambda *args, **kwargs: Result())

    result = json.loads(_run(["say", "hello"]))

    assert len(result["output"]) == 12000
    assert result["truncated"] is True


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


def test_trash_path_preserves_existing_name_and_moves_symlink_itself(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("nums.tools.Path.home", lambda: tmp_path)
    trash = tmp_path / ".Trash"
    trash.mkdir()
    (trash / "shortcut.txt").write_text("existing")
    target = tmp_path / "target.txt"
    target.write_text("keep")
    link = tmp_path / "shortcut.txt"
    link.symlink_to(target)

    result = json.loads(MacTools().trash_path({"path": str(link)}))

    assert (trash / "shortcut.txt").read_text() == "existing"
    assert Path(result["recoverable_at"]).is_symlink()
    assert result["recoverable_at"] == str(trash / "shortcut-1.txt")
    assert target.read_text() == "keep"
    assert not link.is_symlink()


def test_trash_path_moves_broken_symlink(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nums.tools.Path.home", lambda: tmp_path)
    link = tmp_path / "broken.txt"
    link.symlink_to(tmp_path / "missing.txt")

    result = json.loads(MacTools().trash_path({"path": str(link)}))

    assert Path(result["recoverable_at"]).is_symlink()
    assert not link.is_symlink()
