import io
import json
import stat
import subprocess
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
    assert listing["next_offset"] == 500

    final_page = json.loads(MacTools().list_directory({"path": str(tmp_path), "offset": "500"}))
    assert len(final_page["items"]) == 1
    assert final_page["truncated"] is False


def test_directory_list_rejects_negative_offset(tmp_path: Path) -> None:
    result = json.loads(MacTools().execute("list_directory", {
        "path": str(tmp_path), "offset": "-1",
    }))

    assert "nonnegative" in result["error"]


def test_find_files_recurses_respects_ignores_and_globs(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored.py\n")
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "main.py").write_text("pass\n")
    (nested / "ignored.py").write_text("pass\n")
    (nested / "notes.txt").write_text("notes\n")
    tools = MacTools("read_only")

    result = json.loads(tools.execute("find_files", {
        "path": str(tmp_path), "glob": "*.py",
    }))

    assert result == {
        "files": [str(nested / "main.py")], "total": 1, "truncated": False,
    }


def test_find_files_pages_large_results(tmp_path: Path) -> None:
    for index in range(201):
        (tmp_path / f"file-{index:03}.py").touch()
    tools = MacTools()

    first = json.loads(tools.execute("find_files", {"path": str(tmp_path)}))
    second = json.loads(tools.execute("find_files", {
        "path": str(tmp_path), "offset": "200",
    }))

    assert len(first["files"]) == 200
    assert first["next_offset"] == 200
    assert second["files"] == [str(tmp_path / "file-200.py")]
    assert second["truncated"] is False


def test_file_read_is_bounded_before_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    class GuardedReader(io.StringIO):
        def read(self, size: int = -1) -> str:
            assert size == 12001
            return super().read(size)

    monkeypatch.setattr("nums.tools.Path.open", lambda *args, **kwargs: GuardedReader("x" * 20000))

    result = MacTools().read_file({"path": "/unused/large.txt"})

    assert len(result.split("\n[NUMS:")[0]) == 12000
    assert "truncated" in result


def test_read_file_offset_returns_next_unicode_chunk(tmp_path: Path) -> None:
    path = tmp_path / "long.txt"
    path.write_text("é" * 12000 + "next", encoding="utf-8")

    first = MacTools().read_file({"path": str(path)})
    second = MacTools().read_file({"path": str(path), "offset": "12000"})

    assert first.startswith("é" * 12000)
    assert "continue with offset 12000" in first
    assert second == "next"


def test_read_file_rejects_negative_offset(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("hello")

    result = json.loads(MacTools().execute("read_file", {"path": str(path), "offset": "-1"}))

    assert "nonnegative" in result["error"]


def test_read_lines_returns_numbered_code_and_next_page(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("first\nsecond\nthird\n")
    tools = MacTools("read_only")

    first = json.loads(tools.execute("read_lines", {
        "path": str(path), "start_line": "2", "count": "1",
    }))
    second = json.loads(tools.execute("read_lines", {
        "path": str(path), "start_line": "3",
    }))

    assert first == {
        "lines": [{"number": 2, "text": "second", "truncated": False}],
        "next_line": 3,
    }
    assert second == {"lines": [{"number": 3, "text": "third", "truncated": False}]}


def test_read_lines_clips_very_long_line(tmp_path: Path) -> None:
    path = tmp_path / "long.py"
    path.write_text("x" * 3000 + "\n")

    result = json.loads(MacTools().execute("read_lines", {"path": str(path)}))

    assert len(result["lines"][0]["text"]) == 2000
    assert result["lines"][0]["truncated"] is True


def test_read_lines_rejects_invalid_ranges(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("hello\n")

    for arguments in ({"start_line": "0"}, {"count": "101"}, {"count": "-1"}):
        result = json.loads(MacTools().execute("read_lines", {
            "path": str(path), **arguments,
        }))
        assert "error" in result


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


def test_replace_in_file_preserves_surrounding_text_permissions_and_newlines(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_bytes(b"before\r\nvalue = 1\r\nafter\r\n")
    path.chmod(0o640)

    result = json.loads(MacTools("standard").execute("replace_in_file", {
        "path": str(path), "old_text": "value = 1", "new_text": "value = 2",
    }))

    assert result == {"updated": str(path), "replacements": 1}
    assert path.read_bytes() == b"before\r\nvalue = 2\r\nafter\r\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.parametrize("old_text", ["", "missing", "same"])
def test_replace_in_file_rejects_unsafe_matches_without_writing(tmp_path: Path, old_text: str) -> None:
    path = tmp_path / "module.py"
    path.write_text("same\nsame\n")

    result = json.loads(MacTools().execute("replace_in_file", {
        "path": str(path), "old_text": old_text, "new_text": "changed",
    }))

    assert "error" in result
    assert path.read_text() == "same\nsame\n"


def test_read_only_mode_blocks_precise_edits(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("before")

    result = json.loads(MacTools("read_only").execute("replace_in_file", {
        "path": str(path), "old_text": "before", "new_text": "after",
    }))

    assert result["action_mode"] == "read_only"
    assert path.read_text() == "before"


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


def test_search_glob_limits_matches_to_requested_files(tmp_path: Path) -> None:
    (tmp_path / "note.py").write_text("needle\n")
    (tmp_path / "note.txt").write_text("needle\n")

    result = json.loads(MacTools().search_files({
        "query": "needle", "path": str(tmp_path), "glob": "*.py",
    }))

    assert "note.py" in result["output"]
    assert "note.txt" not in result["output"]


def test_search_rejects_empty_query(tmp_path: Path) -> None:
    result = json.loads(MacTools().execute("search_files", {
        "query": "  ", "path": str(tmp_path),
    }))

    assert "query must not be empty" in result["error"]


def test_git_tools_show_changes_in_read_only_mode(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "module.py"
    tracked.write_text("value = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "module.py"], check=True)
    subprocess.run([
        "git", "-C", str(tmp_path), "-c", "user.name=NUMS Test",
        "-c", "user.email=nums@example.test", "commit", "-qm", "baseline",
    ], check=True)
    tracked.write_text("value = 2\n")
    (tmp_path / "new.py").write_text("new = True\n")
    tools = MacTools("read_only")

    status = json.loads(tools.execute("git_status", {"repo": str(tmp_path)}))
    diff = json.loads(tools.execute("git_diff", {
        "repo": str(tmp_path), "file": "module.py",
    }))

    assert status["exit_code"] == 0
    assert "module.py" in status["output"]
    assert "new.py" in status["output"]
    assert diff["exit_code"] == 0
    assert "+value = 2" in diff["output"]
    assert "new.py" not in diff["output"]


def test_git_diff_rejects_empty_file_path(tmp_path: Path) -> None:
    result = json.loads(MacTools().execute("git_diff", {
        "repo": str(tmp_path), "file": " ",
    }))

    assert "file must not be empty" in result["error"]


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


def test_large_command_output_keeps_failure_context_at_both_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        returncode = 1
        stdout = "first failure\n" + "x" * 20000
        stderr = "\nlast failure"

    monkeypatch.setattr("nums.tools.subprocess.run", lambda *args, **kwargs: Result())

    result = json.loads(_run(["pytest"]))

    assert result["output"].startswith("first failure")
    assert result["output"].endswith("last failure")
    assert "middle of command output omitted" in result["output"]


def test_command_timeout_returns_partial_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("pytest", 5, output=b"test started\n")

    monkeypatch.setattr("nums.tools.subprocess.run", timeout)

    result = json.loads(_run(["pytest"], timeout=5))

    assert result["exit_code"] is None
    assert result["output"] == "test started"
    assert "timed out after 5 seconds" in result["error"]


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
