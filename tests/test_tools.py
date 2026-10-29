import io
import json
import os
import platform
import stat
import subprocess
from pathlib import Path

import pytest

from nums.tools import MacTools, TOOL_SCHEMAS, _run, validate_tool_arguments
from nums.mac_ui import (
    ACTIVATE_APP_SCRIPT, ADJUST_CONTROL_SCRIPT, CLICK_ELEMENT_SCRIPT, CLICK_MENU_ITEM_SCRIPT, CLOSE_WINDOW_SCRIPT,
    FOCUS_WINDOW_SCRIPT, INSPECT_APP_SCRIPT, MOVE_WINDOW_SCRIPT, RESIZE_WINDOW_SCRIPT,
    INSPECT_FOCUSED_ELEMENT_SCRIPT, INSPECT_MENU_SCRIPT, LIST_APPS_SCRIPT, LIST_WINDOWS_SCRIPT,
    NOTIFY_SCRIPT, SELECT_POPUP_ITEM_SCRIPT, SELECT_RADIO_SCRIPT, SET_TOGGLE_SCRIPT, TYPE_TEXT_SCRIPT,
    SET_WINDOW_FULLSCREEN_SCRIPT, SET_WINDOW_MINIMIZED_SCRIPT,
    SHOW_ELEMENT_MENU_SCRIPT, WAIT_ELEMENT_SCRIPT, WAIT_MENU_ITEM_SCRIPT, WAIT_WINDOW_SCRIPT,
    key_script,
)


def test_paging_and_git_schemas_expose_value_constraints() -> None:
    schemas = {item["function"]["name"]: item["function"] for item in TOOL_SCHEMAS}

    assert schemas["read_file"]["parameters"]["properties"]["offset"]["pattern"]
    assert schemas["read_lines"]["parameters"]["properties"]["count"]["pattern"]
    assert schemas["git_log"]["parameters"]["properties"]["count"]["pattern"]
    assert schemas["git_show"]["parameters"]["properties"]["revision"]["pattern"]


def test_window_schemas_expose_exact_states_and_bounds() -> None:
    schemas = {item["function"]["name"]: item["function"] for item in TOOL_SCHEMAS}

    assert schemas["open_item"]["parameters"]["properties"]["kind"]["enum"] == [
        "auto", "app", "path", "url",
    ]
    assert schemas["wait_for_app_window"]["parameters"]["properties"]["state"]["enum"] == [
        "exists", "absent",
    ]
    assert schemas["set_app_window_minimized"]["parameters"]["properties"]["state"]["enum"] == [
        "on", "off",
    ]
    assert schemas["resize_app_window"]["parameters"]["properties"]["width"]["pattern"]


def test_ui_schemas_expose_roles_states_and_shortcut_format() -> None:
    schemas = {item["function"]["name"]: item["function"] for item in TOOL_SCHEMAS}

    assert schemas["wait_for_app_element"]["parameters"]["properties"]["state"]["enum"] == [
        "exists", "enabled", "disabled", "absent",
    ]
    assert schemas["set_app_toggle"]["parameters"]["properties"]["role"]["enum"] == [
        "AXCheckBox", "AXSwitch", "AXDisclosureTriangle",
    ]
    assert schemas["adjust_app_control"]["parameters"]["properties"]["direction"]["enum"] == [
        "increase", "decrease",
    ]
    assert schemas["press_app_key"]["parameters"]["properties"]["modifiers"]["pattern"]


def test_shared_argument_validation_enforces_schema_values() -> None:
    count_error = validate_tool_arguments("read_lines", {"path": "/tmp/a", "count": "101"})
    assert count_error is not None and "between 1 and 100" in count_error
    assert validate_tool_arguments("open_item", {"target": "Safari", "kind": "program"}) == (
        "open_item.kind must be one of: auto, app, path, url"
    )
    assert validate_tool_arguments("wait_for_app_window", {"app": "", "window": "Save"}) == (
        "wait_for_app_window.app must not be empty"
    )


def test_file_mutation_schemas_reject_empty_paths_and_bad_modes() -> None:
    assert validate_tool_arguments("copy_path", {"source": "", "destination": "/tmp/copy"}) == (
        "copy_path.source must not be empty"
    )
    mode_error = validate_tool_arguments(
        "set_path_permissions", {"path": "/tmp/file", "mode": "999"},
    )
    assert mode_error is not None and "three-digit octal" in mode_error.lower()
    assert validate_tool_arguments(
        "replace_in_file", {"path": "/tmp/a", "old_text": "", "new_text": ""},
    ) == "replace_in_file.old_text must not be empty"


def test_exact_ui_selector_schemas_reject_empty_names_and_bad_indexes() -> None:
    assert validate_tool_arguments(
        "click_app_menu_item", {"app": "Safari", "menu": "", "item": "New Tab"},
    ) == "click_app_menu_item.menu must not be empty"
    index_error = validate_tool_arguments("select_app_popup_item", {
        "app": "System Settings", "window": "Sound", "index": "0", "item": "AirPods",
    })
    assert index_error is not None and "positive element number" in index_error
    assert validate_tool_arguments("click_app_element", {
        "app": "System Settings", "window": "Appearance", "role": "", "label": "Dark",
    }) == "click_app_element.role must not be empty"


def test_personal_tool_schemas_require_meaningful_commands_and_titles() -> None:
    assert validate_tool_arguments("shell", {"command": ""}) == "shell.command must not be empty"
    assert validate_tool_arguments("notify", {"title": "NUMS", "message": ""}) == (
        "notify.message must not be empty"
    )
    assert validate_tool_arguments("create_reminder", {"title": ""}) == (
        "create_reminder.title must not be empty"
    )
    assert validate_tool_arguments("set_clipboard", {"text": ""}) is None


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


def test_path_info_reports_metadata_without_following_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("hello")
    target.chmod(0o640)
    link = tmp_path / "link.txt"
    link.symlink_to("target.txt")

    file_info = json.loads(MacTools("read_only").execute("path_info", {"path": str(target)}))
    link_info = json.loads(MacTools("read_only").execute("path_info", {"path": str(link)}))
    assert file_info["type"] == "file"
    assert file_info["bytes"] == 5
    assert file_info["permissions"] == "0o640"
    assert link_info["type"] == "symlink"
    assert link_info["target"] == "target.txt"
    assert "error" in json.loads(MacTools().execute("path_info", {"path": str(tmp_path / "missing")}))


def test_hash_file_streams_sha256_in_read_only_mode(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    result = json.loads(MacTools("read_only").execute("hash_file", {"path": str(path)}))
    assert result == {
        "path": str(path),
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "bytes": 3,
    }
    assert "error" in json.loads(MacTools().execute("hash_file", {"path": str(tmp_path)}))


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

    monkeypatch.setattr("nums.tools._open_regular_text", lambda path: GuardedReader("x" * 20000))

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


def test_append_file_preserves_existing_content_and_supports_unicode(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "notes.txt"
    first = json.loads(MacTools("standard").execute("append_file", {
        "path": str(path), "content": "first\n",
    }))
    second = json.loads(MacTools("standard").execute("append_file", {
        "path": str(path), "content": "café\n",
    }))
    assert first["bytes"] == 6
    assert second["bytes"] == 6
    assert path.read_text() == "first\ncafé\n"
    blocked = json.loads(MacTools("read_only").execute("append_file", {
        "path": str(path), "content": "blocked",
    }))
    assert blocked["action_mode"] == "read_only"


def test_create_directory_is_idempotent_and_standard_mode_safe(tmp_path: Path) -> None:
    path = tmp_path / "parent" / "child"
    first = json.loads(MacTools("standard").execute("create_directory", {"path": str(path)}))
    second = json.loads(MacTools("standard").execute("create_directory", {"path": str(path)}))
    assert first == {"directory": str(path), "created": True}
    assert second == {"directory": str(path), "created": False}
    assert path.is_dir()
    blocked = tmp_path / "blocked"
    result = json.loads(MacTools("read_only").execute("create_directory", {"path": str(blocked)}))
    assert result["action_mode"] == "read_only"
    assert not blocked.exists()


def test_copy_path_preserves_files_trees_and_symlinks_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "note.txt").write_text("hello")
    (source / "link.txt").symlink_to("note.txt")
    destination = tmp_path / "copies" / "source"
    result = json.loads(MacTools("standard").execute("copy_path", {
        "source": str(source), "destination": str(destination),
    }))
    assert result["type"] == "directory"
    assert (destination / "note.txt").read_text() == "hello"
    assert (destination / "link.txt").is_symlink()
    assert "error" in json.loads(MacTools().execute("copy_path", {
        "source": str(source), "destination": str(destination),
    }))
    assert "error" in json.loads(MacTools().execute("copy_path", {
        "source": str(source), "destination": str(source / "nested"),
    }))


def test_move_path_renames_without_overwriting(tmp_path: Path) -> None:
    source = tmp_path / "draft.txt"
    source.write_text("draft")
    destination = tmp_path / "archive" / "final.txt"
    result = json.loads(MacTools("standard").execute("move_path", {
        "source": str(source), "destination": str(destination),
    }))
    assert result == {"moved": str(source), "destination": str(destination)}
    assert not source.exists()
    assert destination.read_text() == "draft"
    other = tmp_path / "other.txt"
    other.write_text("other")
    assert "error" in json.loads(MacTools().execute("move_path", {
        "source": str(other), "destination": str(destination),
    }))
    assert other.read_text() == "other"


def test_set_path_permissions_is_verified_and_refuses_symlinks(tmp_path: Path) -> None:
    path = tmp_path / "script.sh"
    path.write_text("#!/bin/sh\n")
    path.chmod(0o600)
    result = json.loads(MacTools("standard").execute("set_path_permissions", {
        "path": str(path), "mode": "755",
    }))
    assert result == {"path": str(path), "previous": "0o600", "permissions": "0o755"}
    link = tmp_path / "script-link"
    link.symlink_to(path)
    assert "error" in json.loads(MacTools().execute("set_path_permissions", {
        "path": str(link), "mode": "644",
    }))
    assert "error" in json.loads(MacTools().execute("set_path_permissions", {
        "path": str(path), "mode": "u+x",
    }))


def test_create_symlink_preserves_relative_target_and_refuses_overwrite(tmp_path: Path) -> None:
    link = tmp_path / "nested" / "current"
    result = json.loads(MacTools("standard").execute("create_symlink", {
        "target": "../releases/v1", "link": str(link),
    }))
    assert result == {"link": str(link), "target": "../releases/v1"}
    assert link.is_symlink()
    assert link.readlink() == Path("../releases/v1")
    assert "error" in json.loads(MacTools().execute("create_symlink", {
        "target": "elsewhere", "link": str(link),
    }))
    blocked_link = tmp_path / "blocked"
    blocked = json.loads(MacTools("read_only").execute("create_symlink", {
        "target": "target", "link": str(blocked_link),
    }))
    assert blocked["action_mode"] == "read_only"
    assert not blocked_link.exists()


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


def test_replace_in_file_rejects_oversized_file_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("nums.tools.REPLACE_FILE_MAX_CHARS", 16)
    path = tmp_path / "big.txt"
    path.write_text("needle " + "x" * 100)

    result = json.loads(MacTools().execute("replace_in_file", {
        "path": str(path), "old_text": "needle", "new_text": "pin",
    }))

    assert "16-character limit" in result["error"]
    assert path.read_text().startswith("needle ")


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


def test_git_log_is_bounded_read_only_and_path_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"repo": "/project", "count": "12", "file": "src/main.py"}
    assert MacTools("read_only").execute("git_log", args) == "ok"
    assert "--max-count=12" in commands[0]
    assert commands[0][-2:] == ["--", "src/main.py"]
    for update in ({"count": "0"}, {"count": "101"}, {"file": ""}):
        assert "error" in json.loads(MacTools("read_only").execute("git_log", {**args, **update}))


def test_git_show_validates_one_revision_and_optional_path(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"repo": "/project", "revision": "HEAD~1", "file": "src/main.py"}
    assert MacTools("read_only").execute("git_show", args) == "ok"
    assert commands[0][-3:] == ["HEAD~1", "--", "src/main.py"]
    for revision in ("--help", "HEAD..main", "HEAD:secret", "bad ref"):
        assert "error" in json.loads(MacTools().execute("git_show", {
            "repo": "/project", "revision": revision,
        }))


def test_git_branches_is_read_only_and_includes_local_and_remote_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    assert MacTools("read_only").execute("git_branches", {"repo": "/project"}) == "ok"
    assert commands[0][-2:] == ["refs/heads", "refs/remotes"]
    assert any("%(upstream:short)" in argument for argument in commands[0])


def test_git_blame_is_line_bounded_and_revision_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"repo": "/project", "file": "src/main.py", "start_line": "10", "count": "25", "revision": "main"}
    assert MacTools("read_only").execute("git_blame", args) == "ok"
    assert commands[0][-5:] == ["-L", "10,+25", "main", "--", "src/main.py"]
    for update in ({"start_line": "0"}, {"count": "201"}, {"file": ""}, {"revision": "HEAD..main"}):
        assert "error" in json.loads(MacTools().execute("git_blame", {**args, **update}))


def test_app_inspection_is_read_only_but_ui_actions_require_unrestricted() -> None:
    tools = MacTools("standard")
    click = json.loads(tools.execute("click_app_element", {
        "app": "Finder", "window": "Documents", "role": "AXButton", "label": "Back",
    }))
    typed = json.loads(tools.execute("type_in_app", {
        "app": "Finder", "window": "Documents", "text": "hello",
    }))

    assert click["action_mode"] == "standard"
    assert typed["action_mode"] == "standard"


def test_app_inspection_passes_page_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")

    assert MacTools("read_only").execute("inspect_app_ui", {
        "app": "Finder", "offset": "120",
    }) == "ok"
    assert commands[0][-3:] == ["Finder", "120", "40"]
    assert MacTools("read_only").execute("inspect_app_ui", {
        "app": "Finder", "offset": "40", "limit": "80",
    }) == "ok"
    assert commands[1][-3:] == ["Finder", "40", "80"]
    invalid = json.loads(MacTools().execute("inspect_app_ui", {
        "app": "Finder", "offset": "-1",
    }))
    assert "nonnegative" in invalid["error"]
    invalid_limit = json.loads(MacTools().execute("inspect_app_ui", {
        "app": "Finder", "limit": "121",
    }))
    assert "between 1 and 120" in invalid_limit["error"]


def test_app_inspection_reads_interactive_control_values() -> None:
    for role in ("AXRadioButton", "AXSlider", "AXPopUpButton", "AXDisclosureTriangle", "AXIncrementor"):
        assert f'roleText is "{role}"' in INSPECT_APP_SCRIPT


def test_focused_element_inspection_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "focused")
    assert MacTools("read_only").execute("inspect_focused_app_element", {"app": "Notes"}) == "focused"
    assert commands[0][-1] == "Notes"
    assert "error" in json.loads(MacTools().execute("inspect_focused_app_element", {"app": ""}))


def test_list_app_windows_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    assert MacTools("read_only").execute("list_app_windows", {"app": "Finder"}) == "ok"
    assert commands[0][-1] == "Finder"
    assert 'attribute "AXMinimized"' in LIST_WINDOWS_SCRIPT
    assert "position of windowRef" in LIST_WINDOWS_SCRIPT
    assert "size of windowRef" in LIST_WINDOWS_SCRIPT


def test_activate_app_requires_unrestricted_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    assert MacTools().execute("activate_app", {"app": "Safari"}) == "ok"
    assert commands[0][-1] == "Safari"
    assert "error" in json.loads(MacTools().execute("activate_app", {"app": ""}))
    assert json.loads(MacTools("standard").execute("activate_app", {"app": "Safari"}))["action_mode"] == "standard"


def test_wait_for_app_window_is_bounded_and_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        "nums.tools._run",
        lambda command, **kwargs: calls.append((command, kwargs)) or "ready",
    )
    args = {"app": "Preview", "window": "Export", "timeout": "8"}

    assert MacTools("read_only").execute("wait_for_app_window", args) == "ready"
    assert calls[0][0][-4:] == ["Preview", "Export", "exists", "8"]
    assert calls[0][1]["timeout"] == 13
    assert MacTools("read_only").execute("wait_for_app_window", {**args, "state": "absent"}) == "ready"

    for update in ({"timeout": "0"}, {"timeout": "31"}, {"window": ""}, {"state": "closed"}):
        result = json.loads(MacTools("read_only").execute(
            "wait_for_app_window", {**args, **update},
        ))
        assert "error" in result


def test_focus_window_requires_unrestricted_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    assert MacTools().execute("focus_app_window", {"app": "Finder", "window": "Documents"}) == "ok"
    assert commands[0][-3:] == ["Finder", "Documents", ""]
    assert MacTools().execute("focus_app_window", {
        "app": "Finder", "window": "Documents", "index": "2",
    }) == "ok"
    assert commands[1][-3:] == ["Finder", "Documents", "2"]
    blocked = json.loads(MacTools("standard").execute("focus_app_window", {
        "app": "Finder", "window": "Documents",
    }))
    assert blocked["action_mode"] == "standard"


def test_close_window_targets_exact_title_and_optional_index(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"app": "Preview", "window": "Draft.pdf"}
    assert MacTools().execute("close_app_window", args) == "ok"
    assert commands[0][-3:] == ["Preview", "Draft.pdf", ""]
    assert MacTools().execute("close_app_window", {**args, "index": "2"}) == "ok"
    assert commands[1][-1] == "2"
    assert "error" in json.loads(MacTools().execute("close_app_window", {**args, "index": "0"}))
    assert json.loads(MacTools("standard").execute("close_app_window", args))["action_mode"] == "standard"


def test_set_window_minimized_validates_state_and_index(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"app": "Safari", "window": "Downloads", "state": "on"}
    assert MacTools().execute("set_app_window_minimized", args) == "ok"
    assert commands[0][-4:] == ["Safari", "Downloads", "", "on"]
    assert "error" in json.loads(MacTools().execute("set_app_window_minimized", {**args, "state": "toggle"}))
    assert "error" in json.loads(MacTools().execute("set_app_window_minimized", {**args, "index": "0"}))


def test_set_window_fullscreen_validates_state(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"app": "Safari", "window": "Downloads", "state": "on"}
    assert MacTools().execute("set_app_window_fullscreen", args) == "ok"
    assert commands[0][-4:] == ["Safari", "Downloads", "", "on"]
    assert "error" in json.loads(MacTools().execute("set_app_window_fullscreen", {**args, "state": "toggle"}))


def test_move_window_accepts_bounded_signed_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"app": "Preview", "window": "Draft.pdf", "x": "-120", "y": "80"}
    assert MacTools().execute("move_app_window", args) == "ok"
    assert commands[0][-5:] == ["Preview", "Draft.pdf", "", "-120", "80"]
    for update in ({"x": "left"}, {"x": "10001"}, {"index": "0"}):
        assert "error" in json.loads(MacTools().execute("move_app_window", {**args, **update}))


def test_resize_window_accepts_bounded_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"app": "Preview", "window": "Draft.pdf", "width": "1200", "height": "800"}
    assert MacTools().execute("resize_app_window", args) == "ok"
    assert commands[0][-5:] == ["Preview", "Draft.pdf", "", "1200", "800"]
    for update in ({"width": "wide"}, {"width": "99"}, {"height": "10001"}):
        assert "error" in json.loads(MacTools().execute("resize_app_window", {**args, **update}))


def test_notification_passes_user_text_as_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    message = 'Done " now\nwith details'
    assert MacTools("standard").execute("notify", {"message": message, "title": "NUMS"}) == "ok"
    assert commands[0][-2:] == [message, "NUMS"]
    assert message not in commands[0][2]


def test_menu_inspection_lists_menus_or_named_items(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    tools = MacTools("read_only")

    assert tools.execute("inspect_app_menu", {"app": "Safari"}) == "ok"
    assert tools.execute("inspect_app_menu", {"app": "Safari", "menu": "File"}) == "ok"
    assert tools.execute("inspect_app_menu", {"app": "Safari", "menu": "File", "submenu": "Share"}) == "ok"
    assert commands[0][-3:] == ["Safari", "", ""]
    assert commands[1][-3:] == ["Safari", "File", ""]
    assert commands[2][-3:] == ["Safari", "File", "Share"]
    assert "error" in json.loads(tools.execute("inspect_app_menu", {"app": "Safari", "submenu": "Share"}))


def test_menu_click_passes_exact_names_as_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")

    assert MacTools().execute("click_app_menu_item", {
        "app": "Safari", "menu": "File", "item": "Email This Page", "submenu": "Share",
    }) == "ok"
    assert commands[0][-4:] == ["Safari", "File", "Email This Page", "Share"]
    blocked = json.loads(MacTools("standard").execute("click_app_menu_item", {
        "app": "Safari", "menu": "File", "item": "New Window",
    }))
    assert blocked["action_mode"] == "standard"


def test_wait_for_menu_item_is_bounded_and_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: calls.append((command, kwargs)) or "ready")
    args = {"app": "Safari", "menu": "File", "item": "Share", "state": "enabled", "timeout": "7"}
    assert MacTools("read_only").execute("wait_for_app_menu_item", args) == "ready"
    assert calls[0][0][-6:] == ["Safari", "File", "Share", "", "enabled", "7"]
    assert calls[0][1]["timeout"] == 12
    for state in ("exists", "disabled", "absent"):
        assert MacTools("read_only").execute("wait_for_app_menu_item", {**args, "state": state}) == "ready"
    for update in ({"state": "visible"}, {"timeout": "0"}, {"item": ""}):
        assert "error" in json.loads(MacTools("read_only").execute(
            "wait_for_app_menu_item", {**args, **update},
        ))


def test_ui_permission_failure_names_required_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    for diagnostic, setting in (
        ("Not authorized to send Apple events to System Events. (-1743)", "Automation"),
        ("osascript is not allowed assistive access. (-1719)", "Accessibility"),
        ("An error of type -10827 has occurred. (-10827)", "Launch Services"),
    ):
        monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: json.dumps({
            "exit_code": 1, "error": "Command exited with status 1", "output": diagnostic,
        }))
        result = json.loads(MacTools().execute("inspect_app_ui", {"app": "Finder"}))
        assert setting in result["hint"]
        assert diagnostic == result["output"]


def test_ui_actions_pass_app_text_as_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr(
        "nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok",
    )
    app = 'Finder" & do shell script "false'
    tools = MacTools()

    tools.click_app_element({
        "app": app, "window": "Documents", "role": "AXButton", "label": "Back",
    })
    tools.type_in_app({"app": app, "window": "Documents", "text": "hello"})
    tools.press_app_key({
        "app": app, "window": "Documents", "key": "s", "modifiers": "command,shift",
    })

    assert all(command[0:2] == ["osascript", "-e"] for command in commands)
    assert all(command[3] == app and app not in command[2] for command in commands)
    assert commands[0][-4:] == ["Documents", "AXButton", "Back", ""]
    assert "command down, shift down" in commands[2][2]


def test_type_in_exact_text_field(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"app": "Notes", "window": "Notes", "text": "hello", "role": "AXTextArea", "label": "Body"}
    assert MacTools().execute("type_in_app", {**args, "index": "12"}) == "ok"
    assert commands[0][-7:] == ["Notes", "Notes", "hello", "AXTextArea", "Body", "12", "append"]
    assert MacTools().execute("type_in_app", {
        "app": "Notes", "window": "Notes", "text": "hello", "role": "AXTextArea", "index": "13",
    }) == "ok"
    assert commands[1][-4:] == ["AXTextArea", "", "13", "append"]
    for bad in ({"role": "AXButton"}, {"index": "0"}):
        result = json.loads(MacTools().execute("type_in_app", {**args, **bad}))
        assert "error" in result
    assert "error" in json.loads(MacTools().execute("type_in_app", {
        "app": "Notes", "window": "Notes", "text": "hello", "index": "2",
    }))


def test_replace_app_text_requires_an_exact_field(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"app": "Notes", "window": "Notes", "text": "new", "role": "AXTextField", "label": "Title"}
    assert MacTools().execute("replace_app_text", args) == "ok"
    assert commands[0][-7:] == ["Notes", "Notes", "new", "AXTextField", "Title", "", "replace"]
    assert MacTools().execute("replace_app_text", {**args, "text": ""}) == "ok"
    assert "error" in json.loads(MacTools().execute("replace_app_text", {**args, "label": ""}))
    assert "error" in json.loads(MacTools().execute("replace_app_text", {**args, "role": "AXButton"}))
    assert json.loads(MacTools("standard").execute("replace_app_text", args))["action_mode"] == "standard"


def test_indexed_click_passes_inspected_index_and_rejects_bad_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"app": "Finder", "window": "Documents", "role": "AXButton", "label": "Back"}

    assert MacTools().execute("click_app_element", {**args, "index": "7"}) == "ok"
    assert commands[0][-1] == "7"
    assert MacTools().execute("click_app_element", {
        "app": "Finder", "window": "Documents", "role": "AXButton", "index": "8",
    }) == "ok"
    assert commands[1][-2:] == ["", "8"]
    assert "error" in json.loads(MacTools().execute("click_app_element", {
        "app": "Finder", "window": "Documents", "role": "AXButton",
    }))
    for bad in ("0", "-1", "abc"):
        result = json.loads(MacTools().execute("click_app_element", {**args, "index": bad}))
        assert "positive" in result["error"]


def test_show_element_menu_requires_exact_target(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"app": "Finder", "window": "Documents", "role": "AXRow", "label": "report.pdf"}
    assert MacTools().execute("show_app_element_menu", args) == "ok"
    assert commands[0][-5:] == ["Finder", "Documents", "AXRow", "report.pdf", ""]
    assert "error" in json.loads(MacTools().execute("show_app_element_menu", {**args, "label": ""}))
    assert json.loads(MacTools("standard").execute("show_app_element_menu", args))["action_mode"] == "standard"


def test_select_popup_item_uses_exact_target_and_requires_unrestricted_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {
        "app": "System Settings", "window": "Displays", "label": "Refresh Rate",
        "item": "60 Hertz",
    }

    assert MacTools().execute("select_app_popup_item", args) == "ok"
    assert commands[0][-5:] == ["System Settings", "Displays", "Refresh Rate", "", "60 Hertz"]
    assert MacTools().execute("select_app_popup_item", {
        "app": "System Settings", "window": "Displays", "index": "9", "item": "60 Hertz",
    }) == "ok"
    assert commands[1][-3:] == ["", "9", "60 Hertz"]
    assert json.loads(MacTools("standard").execute("select_app_popup_item", args))["action_mode"] == "standard"

    for update, message in (
        ({"label": "", "index": ""}, "label or inspected index"),
        ({"index": "0"}, "positive element number"),
        ({"item": ""}, "must not be empty"),
    ):
        result = json.loads(MacTools().execute("select_app_popup_item", {**args, **update}))
        assert message in result["error"]


def test_wait_for_app_element_is_bounded_and_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        "nums.tools._run",
        lambda command, **kwargs: calls.append((command, kwargs)) or "ready",
    )
    args = {
        "app": "Safari", "window": "Downloads", "role": "AXButton",
        "label": "Clear", "state": "enabled", "timeout": "12",
    }

    assert MacTools("read_only").execute("wait_for_app_element", args) == "ready"
    assert calls[0][0][-6:] == ["Safari", "Downloads", "AXButton", "Clear", "enabled", "12"]
    assert calls[0][1]["timeout"] == 17
    for state in ("exists", "disabled", "absent"):
        assert MacTools("read_only").execute(
            "wait_for_app_element", {**args, "state": state},
        ) == "ready"

    for update, message in (
        ({"state": "visible"}, "state must be"),
        ({"timeout": "0"}, "timeout must be"),
        ({"timeout": "31"}, "timeout must be"),
        ({"label": ""}, "must not be empty"),
    ):
        result = json.loads(MacTools("read_only").execute(
            "wait_for_app_element", {**args, **update},
        ))
        assert message in result["error"]


def test_set_toggle_requires_valid_state_and_unrestricted_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"app": "Settings", "window": "Privacy", "role": "AXSwitch", "index": "4", "state": "on"}
    assert MacTools().execute("set_app_toggle", args) == "ok"
    assert commands[0][-6:] == ["Settings", "Privacy", "AXSwitch", "", "4", "on"]
    assert "error" in json.loads(MacTools().execute("set_app_toggle", {**args, "state": "toggle"}))
    disclosure = {**args, "role": "AXDisclosureTriangle", "state": "off"}
    assert MacTools().execute("set_app_toggle", disclosure) == "ok"
    assert commands[1][-4:] == ["AXDisclosureTriangle", "", "4", "off"]
    assert json.loads(MacTools("standard").execute("set_app_toggle", args))["action_mode"] == "standard"


def test_select_radio_targets_label_or_index(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"app": "System Settings", "window": "Sound", "label": "MacBook Speakers"}
    assert MacTools().execute("select_app_radio", args) == "ok"
    assert commands[0][-4:] == ["System Settings", "Sound", "MacBook Speakers", ""]
    assert MacTools().execute("select_app_radio", {"app": "System Settings", "window": "Sound", "index": "6"}) == "ok"
    assert commands[1][-2:] == ["", "6"]
    assert "error" in json.loads(MacTools().execute("select_app_radio", {"app": "System Settings", "window": "Sound"}))
    assert json.loads(MacTools("standard").execute("select_app_radio", args))["action_mode"] == "standard"


def test_adjust_control_validates_direction_role_and_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")
    args = {"app": "System Settings", "window": "Sound", "role": "AXSlider", "label": "Output volume", "direction": "increase", "steps": "3"}
    assert MacTools().execute("adjust_app_control", args) == "ok"
    assert commands[0][-7:] == ["System Settings", "Sound", "AXSlider", "Output volume", "", "increase", "3"]
    for update in ({"role": "AXButton"}, {"direction": "up"}, {"steps": "0"}, {"steps": "21"}, {"label": ""}):
        assert "error" in json.loads(MacTools().execute("adjust_app_control", {**args, **update}))
    assert json.loads(MacTools("standard").execute("adjust_app_control", args))["action_mode"] == "standard"


def test_press_app_key_rejects_unknown_modifier() -> None:
    result = json.loads(MacTools().execute("press_app_key", {
        "app": "Finder", "window": "Documents", "key": "s", "modifiers": "super",
    }))

    assert "modifiers must be" in result["error"]


def test_extended_navigation_keys_have_mac_key_codes() -> None:
    assert "key code 116" in key_script("page_up")
    assert "key code 121" in key_script("page_down")
    assert "key code 117" in key_script("forward_delete")


@pytest.mark.skipif(platform.system() != "Darwin", reason="AppleScript compiler requires macOS")
def test_accessibility_scripts_compile_on_mac(tmp_path: Path) -> None:
    scripts = [
        ACTIVATE_APP_SCRIPT, ADJUST_CONTROL_SCRIPT, CLOSE_WINDOW_SCRIPT, LIST_APPS_SCRIPT, LIST_WINDOWS_SCRIPT,
        MOVE_WINDOW_SCRIPT, RESIZE_WINDOW_SCRIPT, SET_WINDOW_FULLSCREEN_SCRIPT, SET_WINDOW_MINIMIZED_SCRIPT,
        WAIT_WINDOW_SCRIPT, FOCUS_WINDOW_SCRIPT,
        INSPECT_APP_SCRIPT, INSPECT_FOCUSED_ELEMENT_SCRIPT, INSPECT_MENU_SCRIPT,
        CLICK_ELEMENT_SCRIPT, CLICK_MENU_ITEM_SCRIPT, SELECT_POPUP_ITEM_SCRIPT,
        SELECT_RADIO_SCRIPT, SET_TOGGLE_SCRIPT, SHOW_ELEMENT_MENU_SCRIPT,
        TYPE_TEXT_SCRIPT, WAIT_ELEMENT_SCRIPT, WAIT_MENU_ITEM_SCRIPT, NOTIFY_SCRIPT,
        key_script("s", "command"), key_script("return"),
    ]
    for index, script in enumerate(scripts):
        source = tmp_path / f"ui-{index}.applescript"
        output = tmp_path / f"ui-{index}.scpt"
        source.write_text(script)
        subprocess.run(["osacompile", "-o", str(output), str(source)], check=True)


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
    rejected = json.loads(MacTools("standard").execute("create_reminder", {"title": "  "}))
    assert "title must not be empty" in rejected["error"]
    assert len(commands) == 1


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

    monkeypatch.setattr("nums.tools._bounded_run", lambda *args, **kwargs: Result())

    result = json.loads(_run(["open", "missing"]))

    assert result["exit_code"] == 7
    assert "error" in result
    assert "permission denied" in result["output"]


def test_large_command_output_reports_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 0
        stdout = "x" * 12001
        stderr = ""

    monkeypatch.setattr("nums.tools._bounded_run", lambda *args, **kwargs: Result())

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

    monkeypatch.setattr("nums.tools._bounded_run", lambda *args, **kwargs: Result())

    result = json.loads(_run(["pytest"]))

    assert result["output"].startswith("first failure")
    assert result["output"].endswith("last failure")
    assert "middle of command output omitted" in result["output"]


def test_command_timeout_returns_partial_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("pytest", 5, output=b"test started\n")

    monkeypatch.setattr("nums.tools._bounded_run", timeout)

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


def test_open_item_explicit_kind_resolves_dotted_app_names(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command: commands.append(command) or "ok")
    assert MacTools().execute("open_item", {"target": "Some.App", "kind": "app"}) == "ok"
    assert commands == [["open", "-a", "Some.App"]]
    invalid = json.loads(MacTools().execute("open_item", {"target": "Some.App", "kind": "unknown"}))
    assert "kind must be" in invalid["error"]


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


@pytest.mark.parametrize("tool", ["read_file", "read_lines"])
def test_read_tools_reject_pipes_instead_of_blocking(tmp_path: Path, tool: str) -> None:
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)

    result = json.loads(MacTools().execute(tool, {"path": str(pipe)}))

    assert result["error"] == "ValueError: path must be a regular file"


def test_read_file_rejects_directories(tmp_path: Path) -> None:
    result = json.loads(MacTools().execute("read_file", {"path": str(tmp_path)}))

    assert "error" in result


def test_speak_passes_option_like_text_after_separator(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr(
        "nums.tools._run", lambda command, **kwargs: commands.append(command) or "{}"
    )

    MacTools().speak({"text": "-v Alex hello"})

    assert commands == [["say", "--", "-v Alex hello"]]


def test_runaway_command_output_is_bounded_while_streaming() -> None:
    result = json.loads(_run([
        "/bin/sh", "-c", "head -c 20000000 /dev/zero | tr '\\0' x; echo done-marker 1>&2",
    ]))

    assert result["exit_code"] == 0
    assert result["truncated"] is True
    assert len(result["output"]) <= 12000
    assert result["output"].endswith("done-marker")


def test_timeout_kills_background_children_and_returns_promptly() -> None:
    import time

    started = time.monotonic()
    result = json.loads(_run(["/bin/sh", "-c", "echo started; sleep 30 & sleep 30"], timeout=1))

    assert time.monotonic() - started < 10
    assert result["exit_code"] is None
    assert "started" in result["output"]
    assert "timed out after 1 seconds" in result["error"]


def test_append_file_rejects_pipes_instead_of_blocking(tmp_path: Path) -> None:
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)

    result = json.loads(MacTools().execute("append_file", {"path": str(pipe), "content": "x"}))

    assert "error" in result


def test_append_file_still_appends_to_regular_files(tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    path.write_text("one\n")

    result = json.loads(MacTools().execute("append_file", {"path": str(path), "content": "two\n"}))

    assert result["bytes"] == 4
    assert path.read_text() == "one\ntwo\n"


def test_copy_path_rejects_special_files_without_creating_a_destination(tmp_path: Path) -> None:
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)
    destination = tmp_path / "copy"

    result = json.loads(MacTools().execute("copy_path", {
        "source": str(pipe), "destination": str(destination),
    }))

    assert "source must be a file, directory, or symlink" in result["error"]
    assert not destination.exists()


def test_failed_file_copy_leaves_no_partial_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("data")
    destination = tmp_path / "copy.txt"

    def fail_midway(src: object, dst: object, **kwargs: object) -> None:
        Path(str(dst)).write_text("da")
        raise OSError("disk full")

    monkeypatch.setattr("nums.tools.shutil.copy2", fail_midway)

    result = json.loads(MacTools().execute("copy_path", {
        "source": str(source), "destination": str(destination),
    }))

    assert "disk full" in result["error"]
    assert not destination.exists()
    assert source.read_text() == "data"


def test_failed_directory_copy_removes_the_partial_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "tree"
    source.mkdir()
    (source / "file.txt").write_text("data")
    destination = tmp_path / "tree-copy"

    def fail_midway(src: object, dst: object, **kwargs: object) -> None:
        Path(str(dst)).mkdir()
        (Path(str(dst)) / "half.txt").write_text("x")
        raise OSError("permission denied")

    monkeypatch.setattr("nums.tools.shutil.copytree", fail_midway)

    result = json.loads(MacTools().execute("copy_path", {
        "source": str(source), "destination": str(destination),
    }))

    assert "permission denied" in result["error"]
    assert not destination.exists()
    assert (source / "file.txt").read_text() == "data"


WINDOW_BASE = {"app": "Notes", "window": "Draft"}


@pytest.mark.parametrize(
    ("tool", "extra"),
    [
        ("focus_app_window", {}),
        ("close_app_window", {}),
        ("set_app_window_minimized", {"state": "on"}),
        ("set_app_window_fullscreen", {"state": "off"}),
        ("move_app_window", {"x": "10", "y": "20"}),
        ("resize_app_window", {"width": "400", "height": "300"}),
    ],
)
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"app": "   "}, "must not be empty"),
        ({"window": ""}, "must not be empty"),
        ({"index": "0"}, "positive window number"),
        ({"index": "first"}, "positive window number"),
    ],
)
def test_window_tools_reject_bad_targets_before_running_anything(
    monkeypatch: pytest.MonkeyPatch, tool: str, extra: dict, override: dict, message: str
) -> None:
    monkeypatch.setattr("nums.tools._run", lambda *args, **kwargs: pytest.fail("ran osascript"))
    handler = getattr(MacTools(), tool)

    with pytest.raises(ValueError, match=message):
        handler({**WINDOW_BASE, **extra, **override})


@pytest.mark.parametrize("tool", ["set_app_window_minimized", "set_app_window_fullscreen"])
def test_window_state_tools_require_on_or_off(monkeypatch: pytest.MonkeyPatch, tool: str) -> None:
    monkeypatch.setattr("nums.tools._run", lambda *args, **kwargs: pytest.fail("ran osascript"))

    with pytest.raises(ValueError, match="state must be on or off"):
        getattr(MacTools(), tool)({**WINDOW_BASE, "state": "toggle"})


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("move_app_window", {"x": "10001", "y": "0"}, "x must be between -10000 and 10000"),
        ("move_app_window", {"x": "0", "y": "-10001"}, "y must be between -10000 and 10000"),
        ("move_app_window", {"x": "left", "y": "0"}, "x must be an integer"),
        ("resize_app_window", {"width": "99", "height": "300"}, "width must be between 100 and 10000"),
        ("resize_app_window", {"width": "400", "height": "10001"}, "height must be between 100 and 10000"),
        ("resize_app_window", {"width": "wide", "height": "300"}, "width must be an integer"),
    ],
)
def test_window_geometry_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tool: str, args: dict, message: str
) -> None:
    monkeypatch.setattr("nums.tools._run", lambda *a, **k: pytest.fail("ran osascript"))

    with pytest.raises(ValueError, match=message):
        getattr(MacTools(), tool)({**WINDOW_BASE, **args})


def test_valid_window_geometry_reaches_the_script(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")

    MacTools().move_app_window({**WINDOW_BASE, "x": "-10000", "y": "10000"})
    MacTools().resize_app_window({**WINDOW_BASE, "width": "100", "height": "10000"})

    assert commands[0][-5:] == ["Notes", "Draft", "", "-10000", "10000"]
    assert commands[1][-5:] == ["Notes", "Draft", "", "100", "10000"]


ELEMENT_BASE = {"app": "Notes", "window": "Draft"}


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("click_app_element", {"role": "AXButton"}, "label or inspected index is required"),
        ("click_app_element", {"role": "AXButton", "index": "0"}, "positive element number"),
        ("click_app_element", {"role": " ", "label": "Save"}, "must not be empty"),
        ("show_app_element_menu", {"role": "AXButton"}, "label or inspected index is required"),
        ("show_app_element_menu", {"role": "AXButton", "index": "x"}, "positive element number"),
        ("select_app_popup_item", {"item": "Large"}, "label or inspected index is required"),
        ("select_app_popup_item", {"item": " ", "label": "Size"}, "must not be empty"),
        ("select_app_popup_item", {"item": "Large", "index": "0"}, "positive element number"),
        ("select_app_radio", {}, "label or inspected index is required"),
        ("select_app_radio", {"index": "-1"}, "positive element number"),
        ("set_app_toggle", {"role": "AXButton", "state": "on", "label": "A"}, "role must be"),
        ("set_app_toggle", {"role": "AXCheckBox", "state": "maybe", "label": "A"}, "state must be on or off"),
        ("set_app_toggle", {"role": "AXCheckBox", "state": "on"}, "label or inspected index is required"),
        ("set_app_toggle", {"role": "AXCheckBox", "state": "on", "index": "0"}, "positive element number"),
        ("adjust_app_control", {"role": "AXButton", "direction": "increase", "label": "A"}, "role must be"),
        ("adjust_app_control", {"role": "AXSlider", "direction": "sideways", "label": "A"}, "direction must be"),
        ("adjust_app_control", {"role": "AXSlider", "direction": "increase"}, "label or inspected index is required"),
        ("adjust_app_control", {"role": "AXSlider", "direction": "increase", "label": "A", "steps": "0"}, "steps must be between 1 and 20"),
        ("adjust_app_control", {"role": "AXSlider", "direction": "increase", "label": "A", "steps": "21"}, "steps must be between 1 and 20"),
        ("type_in_app", {"text": ""}, "must not be empty"),
        ("type_in_app", {"text": "hi", "label": "Body"}, "label requires a role"),
        ("type_in_app", {"text": "hi", "role": "AXButton", "label": "Body"}, "role must be"),
        ("type_in_app", {"text": "hi", "index": "1"}, "index requires a text field"),
        ("type_in_app", {"text": "hi", "role": "AXTextField", "index": "0"}, "index requires a text field"),
        ("type_in_app", {"text": "hi", "role": "AXTextArea"}, "requires a label or inspected index"),
        ("replace_app_text", {"text": "hi", "role": "AXButton", "label": "Body"}, "role must be"),
        ("replace_app_text", {"text": "hi", "role": "AXTextField"}, "label or inspected index is required"),
        ("replace_app_text", {"text": "hi", "role": "AXTextField", "index": "0"}, "positive element number"),
    ],
)
def test_element_tools_reject_bad_targets_before_running_anything(
    monkeypatch: pytest.MonkeyPatch, tool: str, args: dict, message: str
) -> None:
    monkeypatch.setattr("nums.tools._run", lambda *a, **k: pytest.fail("ran osascript"))

    with pytest.raises(ValueError, match=message):
        getattr(MacTools(), tool)({**ELEMENT_BASE, **args})


@pytest.mark.parametrize(
    ("tool", "args", "mode"),
    [
        ("type_in_app", {"text": "hello"}, "append"),
        ("type_in_app", {"text": "hello", "role": "AXTextField", "label": "Title"}, "append"),
        ("replace_app_text", {"text": "hello", "role": "AXTextArea", "index": "2"}, "replace"),
    ],
)
def test_text_tools_pick_append_or_replace_mode(
    monkeypatch: pytest.MonkeyPatch, tool: str, args: dict, mode: str
) -> None:
    commands = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: commands.append(command) or "ok")

    getattr(MacTools(), tool)({**ELEMENT_BASE, **args})

    assert commands[0][-1] == mode
    assert commands[0][3:6] == ["Notes", "Draft", "hello"]


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("activate_app", {"app": " "}, "app must not be empty"),
        ("list_app_windows", {"app": ""}, "app must not be empty"),
        ("inspect_focused_app_element", {"app": "  "}, "app must not be empty"),
        ("inspect_app_ui", {"app": " "}, "app must not be empty"),
        ("inspect_app_ui", {"app": "Notes", "offset": "-1"}, "offset must be nonnegative"),
        ("inspect_app_ui", {"app": "Notes", "limit": "0"}, "limit must be between 1 and 120"),
        ("inspect_app_ui", {"app": "Notes", "limit": "121"}, "limit must be between 1 and 120"),
        ("inspect_app_menu", {"app": " "}, "app must not be empty"),
        ("inspect_app_menu", {"app": "Notes", "submenu": "Recent"}, "submenu requires a menu"),
        ("click_app_menu_item", {"app": "Notes", "menu": "File", "item": " "}, "must not be empty"),
        ("wait_for_app_window", {"app": "Notes", "window": " "}, "must not be empty"),
        ("wait_for_app_window", {"app": "Notes", "window": "A", "state": "visible"}, "state must be exists or absent"),
        ("wait_for_app_window", {"app": "Notes", "window": "A", "timeout": "0"}, "between 1 and 30 seconds"),
        ("wait_for_app_window", {"app": "Notes", "window": "A", "timeout": "31"}, "between 1 and 30 seconds"),
        ("wait_for_app_element", {"app": "Notes", "window": "A", "role": "AXButton", "label": " "}, "must not be empty"),
        ("wait_for_app_element", {"app": "Notes", "window": "A", "role": "AXButton", "label": "Save", "state": "hidden"}, "state must be exists, enabled, disabled, or absent"),
        ("wait_for_app_element", {"app": "Notes", "window": "A", "role": "AXButton", "label": "Save", "timeout": "soon"}, "between 1 and 30 seconds"),
        ("wait_for_app_menu_item", {"app": "Notes", "menu": "File", "item": ""}, "must not be empty"),
        ("wait_for_app_menu_item", {"app": "Notes", "menu": "File", "item": "Save", "state": "gone"}, "state must be exists, enabled, disabled, or absent"),
        ("wait_for_app_menu_item", {"app": "Notes", "menu": "File", "item": "Save", "timeout": "99"}, "between 1 and 30 seconds"),
        ("press_app_key", {"app": " ", "window": "A", "key": "a"}, "must not be empty"),
        ("press_app_key", {"app": "Notes", "window": "A", "key": "ab"}, "one character or a supported key name"),
        ("press_app_key", {"app": "Notes", "window": "A", "key": "a", "modifiers": "hyper"}, "modifiers must be unique names"),
        ("press_app_key", {"app": "Notes", "window": "A", "key": "a", "modifiers": "shift,shift"}, "modifiers must be unique names"),
    ],
)
def test_inspection_wait_and_key_tools_reject_invalid_arguments(
    monkeypatch: pytest.MonkeyPatch, tool: str, args: dict, message: str
) -> None:
    monkeypatch.setattr("nums.tools._run", lambda *a, **k: pytest.fail("ran osascript"))

    with pytest.raises(ValueError, match=message):
        getattr(MacTools(), tool)(args)


@pytest.mark.parametrize(
    ("tool", "args", "timeout"),
    [
        ("wait_for_app_window", {"app": "Notes", "window": "A", "timeout": "30"}, 35),
        ("wait_for_app_window", {"app": "Notes", "window": "A"}, 15),
        ("wait_for_app_element", {"app": "Notes", "window": "A", "role": "AXButton", "label": "Save", "timeout": "1"}, 6),
        ("wait_for_app_menu_item", {"app": "Notes", "menu": "File", "item": "Save", "timeout": "20"}, 25),
    ],
)
def test_wait_tools_allow_the_script_its_wait_plus_margin(
    monkeypatch: pytest.MonkeyPatch, tool: str, args: dict, timeout: int
) -> None:
    calls = []
    monkeypatch.setattr("nums.tools._run", lambda command, **kwargs: calls.append(kwargs) or "ok")

    getattr(MacTools(), tool)(args)

    assert calls == [{"timeout": timeout}]


def test_trash_path_reports_a_missing_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nums.tools.Path.home", lambda: tmp_path)

    result = json.loads(MacTools().execute("trash_path", {"path": str(tmp_path / "ghost.txt")}))

    assert result["error"].startswith("FileNotFoundError")
    assert not (tmp_path / ".Trash").exists()


def test_trash_path_refuses_the_trash_and_its_parents(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".Trash").mkdir(parents=True)
    (home / ".Trash" / "kept.txt").write_text("kept")
    monkeypatch.setattr("nums.tools.Path.home", lambda: home)

    for path in (home / ".Trash", home, tmp_path):
        result = json.loads(MacTools().execute("trash_path", {"path": str(path)}))
        assert "Cannot move the Trash or its parent" in result["error"]

    assert (home / ".Trash" / "kept.txt").read_text() == "kept"
    assert home.exists()


def test_trash_path_numbers_repeated_names_and_keeps_dotfile_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("nums.tools.Path.home", lambda: home)
    locations = []

    for round_number in range(3):
        (home / ".zshrc").write_text(f"round {round_number}")
        (home / "report.txt").write_text(f"round {round_number}")
        for name in (".zshrc", "report.txt"):
            result = json.loads(MacTools().trash_path({"path": str(home / name)}))
            locations.append(Path(result["recoverable_at"]).name)

    assert locations == [
        ".zshrc", "report.txt", ".zshrc-1", "report-1.txt", ".zshrc-2", "report-2.txt",
    ]
    assert (home / ".Trash" / "report-2.txt").read_text() == "round 2"
    assert stat.S_IMODE((home / ".Trash").stat().st_mode) == 0o700


def test_trash_path_moves_a_whole_directory_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = home / "project"
    (project / "src").mkdir(parents=True)
    (project / "src" / "main.py").write_text("print('hi')")
    monkeypatch.setattr("nums.tools.Path.home", lambda: home)

    result = json.loads(MacTools().trash_path({"path": str(project)}))

    assert not project.exists()
    assert (Path(result["recoverable_at"]) / "src" / "main.py").read_text() == "print('hi')"


def test_read_file_offsets_and_boundaries(tmp_path: Path) -> None:
    tools = MacTools()
    exact = tmp_path / "exact.txt"
    exact.write_text("a" * 12000)
    over = tmp_path / "over.txt"
    over.write_text("b" * 12001)

    assert tools.read_file({"path": str(exact)}) == "a" * 12000
    truncated = tools.read_file({"path": str(over)})
    assert truncated.startswith("b" * 12000) and "continue with offset 12000" in truncated
    assert tools.read_file({"path": str(over), "offset": "12000"}) == "b"
    assert tools.read_file({"path": str(over), "offset": "99999"}) == ""
    with pytest.raises(ValueError, match="offset must be a nonnegative character position"):
        tools.read_file({"path": str(exact), "offset": "-1"})
    with pytest.raises(FileNotFoundError):
        tools.read_file({"path": str(tmp_path / "missing.txt")})


def test_read_file_replaces_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "binary.txt"
    path.write_bytes(b"ok \xff\xfe end")

    assert MacTools().read_file({"path": str(path)}) == "ok �� end"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"start_line": "0"}, "start_line must be a positive line number"),
        ({"start_line": "first"}, "start_line must be a positive line number"),
        ({"count": "0"}, "count must be between 1 and 100"),
        ({"count": "101"}, "count must be between 1 and 100"),
        ({"count": "many"}, "count must be between 1 and 100"),
    ],
)
def test_read_lines_validates_its_window(tmp_path: Path, args: dict, message: str) -> None:
    path = tmp_path / "lines.txt"
    path.write_text("one\n")

    with pytest.raises(ValueError, match=message):
        MacTools().read_lines({"path": str(path), **args})


def test_read_lines_pages_clips_and_reports_the_next_line(tmp_path: Path) -> None:
    path = tmp_path / "lines.txt"
    path.write_text("".join(f"line {number}\n" for number in range(1, 251)))
    tools = MacTools()

    first = json.loads(tools.read_lines({"path": str(path), "count": "100"}))
    assert [row["number"] for row in first["lines"]] == list(range(1, 101))
    assert first["next_line"] == 101
    last = json.loads(tools.read_lines({"path": str(path), "start_line": "201", "count": "100"}))
    assert len(last["lines"]) == 50 and "next_line" not in last
    beyond = json.loads(tools.read_lines({"path": str(path), "start_line": "999"}))
    assert beyond == {"lines": []}

    long_path = tmp_path / "long.txt"
    long_path.write_text("x" * 5000 + "\nshort\r\n")
    clipped = json.loads(tools.read_lines({"path": str(long_path)}))
    assert clipped["lines"][0]["truncated"] is True and len(clipped["lines"][0]["text"]) == 2000
    assert clipped["lines"][1] == {"number": 2, "text": "short", "truncated": False}


def test_read_lines_stops_at_the_character_budget(tmp_path: Path) -> None:
    path = tmp_path / "wide.txt"
    path.write_text("".join(("y" * 1900) + "\n" for _ in range(20)))

    result = json.loads(MacTools().read_lines({"path": str(path), "count": "100"}))

    assert sum(len(row["text"]) for row in result["lines"]) <= 12000
    assert result["next_line"] == len(result["lines"]) + 1


def test_list_directory_orders_pages_and_validates(tmp_path: Path) -> None:
    for name in ("beta.txt", "Alpha.txt", ".hidden"):
        (tmp_path / name).write_text("x")
    for name in ("zeta", "Docs"):
        (tmp_path / name).mkdir()
    tools = MacTools()

    listing = json.loads(tools.list_directory({"path": str(tmp_path)}))

    assert [item["name"] for item in listing["items"]] == [
        "Docs", "zeta", ".hidden", "Alpha.txt", "beta.txt",
    ]
    assert [item["type"] for item in listing["items"]][:2] == ["directory", "directory"]
    assert listing["total"] == 5 and listing["truncated"] is False

    for number in range(520):
        (tmp_path / f"file-{number:04d}.log").write_text("")
    page = json.loads(tools.list_directory({"path": str(tmp_path)}))
    assert len(page["items"]) == 500 and page["truncated"] and page["next_offset"] == 500
    rest = json.loads(tools.list_directory({"path": str(tmp_path), "offset": "500"}))
    assert len(rest["items"]) == 25 and not rest["truncated"]
    with pytest.raises(ValueError, match="offset must be a nonnegative item position"):
        tools.list_directory({"path": str(tmp_path), "offset": "-3"})
    with pytest.raises(FileNotFoundError):
        tools.list_directory({"path": str(tmp_path / "missing")})


def test_hash_file_returns_known_digest_and_rejects_directories(tmp_path: Path) -> None:
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")

    result = json.loads(MacTools().hash_file({"path": str(path)}))

    assert result["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert result["bytes"] == 3
    with pytest.raises(ValueError, match="path must be a file"):
        MacTools().hash_file({"path": str(tmp_path)})


def test_path_info_describes_each_kind_of_path(tmp_path: Path) -> None:
    tools = MacTools()
    file = tmp_path / "note.txt"
    file.write_text("hello")
    file.chmod(0o640)
    link = tmp_path / "link"
    link.symlink_to(file)
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "gone")

    info = json.loads(tools.path_info({"path": str(file)}))
    assert (info["type"], info["bytes"], info["permissions"]) == ("file", 5, "0o640")
    assert json.loads(tools.path_info({"path": str(tmp_path)}))["type"] == "directory"
    link_info = json.loads(tools.path_info({"path": str(link)}))
    assert link_info["type"] == "symlink"
    assert link_info["target"] == str(file)
    assert json.loads(tools.path_info({"path": str(broken)}))["target"] == str(tmp_path / "gone")
    with pytest.raises(FileNotFoundError):
        tools.path_info({"path": str(tmp_path / "nothing")})


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run([
        "git", "-C", str(repo), "-c", "user.name=NUMS Test",
        "-c", "user.email=nums@example.test", "-c", "commit.gpgsign=false", *arguments,
    ], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    (tmp_path / "module.py").write_text("one\ntwo\nthree\n")
    _git(tmp_path, "add", "module.py")
    _git(tmp_path, "commit", "-qm", "first commit")
    (tmp_path / "module.py").write_text("one\nTWO\nthree\n")
    _git(tmp_path, "commit", "-qam", "second commit")
    _git(tmp_path, "branch", "feature")
    return tmp_path


@pytest.mark.parametrize(
    "revision", ["HEAD", "HEAD~1", "main", "v1.0.0", "HEAD@{1}", "feature/x-1", "abc1234", "HEAD^"],
)
def test_ordinary_git_revisions_are_accepted(revision: str) -> None:
    from nums.tools import _validated_git_revision

    assert _validated_git_revision(revision) == revision


@pytest.mark.parametrize(
    "revision", ["", "--output=/tmp/x", "-p", "a..b", "a b", "HEAD;ls", "$(id)", "a`b`", "x\ny", "ref*", "a:b"],
)
def test_option_ranges_and_shell_syntax_are_rejected_as_revisions(revision: str) -> None:
    from nums.tools import _validated_git_revision

    with pytest.raises(ValueError, match="revision must name one Git commit or ref"):
        _validated_git_revision(revision)


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("git_log", {"count": "0"}, "count must be between 1 and 100"),
        ("git_log", {"count": "101"}, "count must be between 1 and 100"),
        ("git_log", {"count": "ten"}, "count must be between 1 and 100"),
        ("git_log", {"file": "  "}, "file must not be empty"),
        ("git_diff", {"file": " "}, "file must not be empty"),
        ("git_show", {"file": ""}, "file must not be empty"),
        ("git_show", {"revision": "--help"}, "revision must name one Git commit or ref"),
        ("git_blame", {"file": " "}, "file must not be empty"),
        ("git_blame", {"file": "module.py", "start_line": "0"}, "start_line must be a positive line number"),
        ("git_blame", {"file": "module.py", "count": "201"}, "count must be between 1 and 200"),
        ("git_blame", {"file": "module.py", "revision": "a..b"}, "revision must name one Git commit or ref"),
    ],
)
def test_git_tools_reject_bad_arguments_before_running_git(
    monkeypatch: pytest.MonkeyPatch, tool: str, args: dict, message: str
) -> None:
    monkeypatch.setattr("nums.tools._run", lambda *a, **k: pytest.fail("ran git"))

    with pytest.raises(ValueError, match=message):
        getattr(MacTools(), tool)({"repo": "/tmp/unused", **args})


def test_git_history_tools_report_real_repository_state(repo: Path) -> None:
    tools = MacTools("read_only")

    log = json.loads(tools.execute("git_log", {"repo": str(repo), "count": "1"}))
    assert log["exit_code"] == 0
    assert "second commit" in log["output"] and "first commit" not in log["output"]

    scoped = json.loads(tools.execute("git_log", {"repo": str(repo), "file": "module.py"}))
    assert "first commit" in scoped["output"] and "second commit" in scoped["output"]

    show = json.loads(tools.execute("git_show", {"repo": str(repo), "revision": "HEAD~1", "file": "module.py"}))
    assert "first commit" in show["output"] and "+one" in show["output"]

    branches = json.loads(tools.execute("git_branches", {"repo": str(repo)}))
    assert "main" in branches["output"] and "feature" in branches["output"]

    blame = json.loads(tools.execute("git_blame", {
        "repo": str(repo), "file": "module.py", "start_line": "2", "count": "1",
    }))
    assert blame["exit_code"] == 0
    assert "summary second commit" in blame["output"]
    assert "\tTWO" in blame["output"]
    assert "\tone" not in blame["output"] and "\tthree" not in blame["output"]

    status = json.loads(tools.execute("git_status", {"repo": str(repo)}))
    assert status["output"].startswith("## main")


def test_git_errors_are_reported_not_raised(tmp_path: Path) -> None:
    result = json.loads(MacTools("read_only").execute("git_status", {"repo": str(tmp_path)}))

    assert result["exit_code"] != 0
    assert "not a git repository" in result["output"].lower()


def test_search_files_treats_the_query_as_literal_text(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("abc\na.c\n")
    tools = MacTools()

    result = json.loads(tools.search_files({"query": "a.c", "path": str(tmp_path)}))

    assert result["exit_code"] == 0
    assert "a.c" in result["output"] and "abc" not in result["output"].replace("a.c", "")


def test_search_files_reports_no_match_without_an_error(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n")

    result = json.loads(MacTools().search_files({"query": "absent", "path": str(tmp_path)}))

    assert result["matches"] == 0 and "error" not in result


def test_search_files_includes_hidden_files_but_never_dot_git(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("TOKEN=needle\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("needle in git\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\n")
    (tmp_path / "src" / "notes.md").write_text("needle\n")
    tools = MacTools()

    everything = json.loads(tools.search_files({"query": "needle", "path": str(tmp_path)}))["output"]
    only_python = json.loads(tools.search_files({"query": "needle", "path": str(tmp_path), "glob": "*.py"}))["output"]

    assert ".env.example" in everything and "app.py" in everything and "notes.md" in everything
    assert ".git/config" not in everything
    assert "app.py" in only_python and "notes.md" not in only_python


def test_search_files_reports_a_missing_directory(tmp_path: Path) -> None:
    result = json.loads(MacTools().search_files({"query": "x", "path": str(tmp_path / "missing")}))

    assert result["exit_code"] == 2 and result["error"]


@pytest.mark.parametrize("tool", ["find_files", "search_files"])
def test_blank_globs_are_rejected(tmp_path: Path, tool: str) -> None:
    args = {"path": str(tmp_path), "glob": "  "}
    if tool == "search_files":
        args["query"] = "x"

    with pytest.raises(ValueError, match="glob must not be empty"):
        getattr(MacTools(), tool)(args)


def test_find_files_lists_hidden_files_sorted_and_excludes_dot_git(tmp_path: Path) -> None:
    for name in ("b.py", "a.py", ".hidden", "pkg/mod.py", ".git/HEAD"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")

    result = json.loads(MacTools().find_files({"path": str(tmp_path)}))
    names = [str(Path(file).relative_to(tmp_path)) for file in result["files"]]

    assert names == [".hidden", "a.py", "b.py", "pkg/mod.py"]
    assert result["total"] == 4 and result["truncated"] is False

    only_modules = json.loads(MacTools().find_files({"path": str(tmp_path), "glob": "pkg/*.py"}))
    assert [Path(file).name for file in only_modules["files"]] == ["mod.py"]


def test_find_files_offset_must_be_numeric(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="offset must be a nonnegative file position"):
        MacTools().find_files({"path": str(tmp_path), "offset": "-1"})


@pytest.mark.parametrize(
    ("key", "modifiers", "action"),
    [
        ("return", "", "key code 36"),
        ("RETURN", "", "key code 36"),
        ("Page_Down", "", "key code 121"),
        ("left", "command", "key code 123 using {command down}"),
        ("a", "", "keystroke keyValue"),
        ("A", "shift", "keystroke keyValue using {shift down}"),
        ("s", "command, shift", "keystroke keyValue using {command down, shift down}"),
        ("tab", "Option,Control", "key code 48 using {option down, control down}"),
    ],
)
def test_key_scripts_use_whitelisted_actions(key: str, modifiers: str, action: str) -> None:
    from nums.mac_ui import key_script

    script = key_script(key, modifiers)

    assert f"            {action}\n" in script
    assert "item 3 of argv" in script


def test_every_named_key_has_a_script() -> None:
    from nums.mac_ui import KEY_CODES, key_script

    for name, code in KEY_CODES.items():
        assert f"key code {code}\n" in key_script(name)


def test_path_info_reports_pipes_as_other(tmp_path: Path) -> None:
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)

    assert json.loads(MacTools().path_info({"path": str(pipe)}))["type"] == "other"


def test_copy_path_refuses_pipes_and_leaves_no_destination(tmp_path: Path) -> None:
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)
    destination = tmp_path / "copy"

    result = json.loads(MacTools().execute(
        "copy_path", {"source": str(pipe), "destination": str(destination)}
    ))

    assert result["error"] == "ValueError: source must be a file, directory, or symlink"
    assert not destination.exists()


def test_unknown_tool_name_is_reported_not_raised() -> None:
    result = json.loads(MacTools().execute("format_disk", {}))

    assert result == {"error": "Unknown tool: format_disk"}


def test_find_files_reports_search_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2, "", "rg: bad glob\n")

    monkeypatch.setattr("nums.tools.subprocess.run", fake_run)

    result = json.loads(MacTools().execute("find_files", {"path": str(tmp_path)}))

    assert result == {"error": "rg: bad glob", "exit_code": 2}
