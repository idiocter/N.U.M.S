from __future__ import annotations

import json
from fnmatch import fnmatchcase
import os
import platform
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .mac_ui import (
    ACTIVATE_APP_SCRIPT, ADJUST_CONTROL_SCRIPT, CLICK_ELEMENT_SCRIPT, CLICK_MENU_ITEM_SCRIPT, CLOSE_WINDOW_SCRIPT,
    FOCUS_WINDOW_SCRIPT, INSPECT_APP_SCRIPT,
    INSPECT_FOCUSED_ELEMENT_SCRIPT, INSPECT_MENU_SCRIPT, LIST_APPS_SCRIPT, LIST_WINDOWS_SCRIPT, MOVE_WINDOW_SCRIPT,
    NOTIFY_SCRIPT, RESIZE_WINDOW_SCRIPT, SELECT_POPUP_ITEM_SCRIPT, SELECT_RADIO_SCRIPT, SET_TOGGLE_SCRIPT,
    SET_WINDOW_FULLSCREEN_SCRIPT, SET_WINDOW_MINIMIZED_SCRIPT, TYPE_TEXT_SCRIPT,
    SHOW_ELEMENT_MENU_SCRIPT, WAIT_ELEMENT_SCRIPT, WAIT_MENU_ITEM_SCRIPT, WAIT_WINDOW_SCRIPT,
    key_script,
)
from .policy import tool_allowed


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object", "properties": properties,
                "required": required, "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _schema(
        "read_file", "Read up to 12000 characters from a UTF-8 file; use offset for later chunks.",
        {"path": {"type": "string"}, "offset": {"type": "string"}}, ["path"],
    ),
    _schema(
        "read_lines", "Read numbered lines from a UTF-8 file; use start_line and count to inspect code sections.",
        {"path": {"type": "string"}, "start_line": {"type": "string"}, "count": {"type": "string"}},
        ["path"],
    ),
    _schema(
        "list_directory", "List up to 500 files and folders; use offset for later pages.",
        {"path": {"type": "string"}, "offset": {"type": "string"}}, ["path"],
    ),
    _schema(
        "find_files", "Find project files recursively, respecting Git ignores; use glob and offset to narrow or page results.",
        {"path": {"type": "string"}, "glob": {"type": "string"}, "offset": {"type": "string"}},
        ["path"],
    ),
    _schema(
        "search_files",
        "Search file contents for a literal string with ripgrep; optionally narrow by glob.",
        {"query": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}},
        ["query", "path"],
    ),
    _schema(
        "git_status", "Show the branch and changed or untracked files in a Git repository.",
        {"repo": {"type": "string"}}, ["repo"],
    ),
    _schema(
        "git_diff", "Show tracked changes against HEAD, optionally limited to one file; untracked files appear in git_status only.",
        {"repo": {"type": "string"}, "file": {"type": "string"}}, ["repo"],
    ),
    _schema(
        "write_file",
        "Write UTF-8 text to a file, creating parent folders.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    _schema(
        "replace_in_file",
        "Replace one exact, unique UTF-8 text span in an existing file. Fails if the old text is missing or repeated.",
        {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
        ["path", "old_text", "new_text"],
    ),
    _schema(
        "shell",
        "Run a zsh command on this Mac. Use for tasks not covered by a narrower tool.",
        {"command": {"type": "string"}, "cwd": {"type": "string"}},
        ["command"],
    ),
    _schema("open_item", "Open an app, file, folder, or URL; set kind to app, path, or url when the target is ambiguous.", {"target": {"type": "string"}, "kind": {"type": "string"}}, ["target"]),
    _schema("list_running_apps", "List running foreground Mac apps by process name.", {}, []),
    _schema("activate_app", "Bring one exact running Mac app to the foreground without launching another app.", {"app": {"type": "string"}}, ["app"]),
    _schema("list_app_windows", "List window titles, indexes, main/minimized state, position, and size in a running Mac app.", {"app": {"type": "string"}}, ["app"]),
    _schema(
        "wait_for_app_window", "Wait for one exact named window to appear or become absent. Timeout defaults to 10 seconds and is capped at 30.",
        {"app": {"type": "string"}, "window": {"type": "string"}, "state": {"type": "string"}, "timeout": {"type": "string"}},
        ["app", "window"],
    ),
    _schema("focus_app_window", "Raise one exact named window in a running Mac app; use inspected index if titles repeat.", {"app": {"type": "string"}, "window": {"type": "string"}, "index": {"type": "string"}}, ["app", "window"]),
    _schema("close_app_window", "Close one exact named app window; use its listed index when titles repeat and inspect if a confirmation dialog blocks closing.", {"app": {"type": "string"}, "window": {"type": "string"}, "index": {"type": "string"}}, ["app", "window"]),
    _schema("set_app_window_minimized", "Set one exact named window's minimized state on or off; use its listed index when titles repeat.", {"app": {"type": "string"}, "window": {"type": "string"}, "index": {"type": "string"}, "state": {"type": "string"}}, ["app", "window", "state"]),
    _schema("set_app_window_fullscreen", "Set one exact named window's full-screen state on or off when the app exposes it.", {"app": {"type": "string"}, "window": {"type": "string"}, "index": {"type": "string"}, "state": {"type": "string"}}, ["app", "window", "state"]),
    _schema("move_app_window", "Move one exact named window to bounded screen coordinates; use its listed index when titles repeat.", {"app": {"type": "string"}, "window": {"type": "string"}, "index": {"type": "string"}, "x": {"type": "string"}, "y": {"type": "string"}}, ["app", "window", "x", "y"]),
    _schema("resize_app_window", "Resize one exact named window to bounded dimensions; use its listed index when titles repeat.", {"app": {"type": "string"}, "window": {"type": "string"}, "index": {"type": "string"}, "width": {"type": "string"}, "height": {"type": "string"}}, ["app", "window", "width", "height"]),
    _schema(
        "inspect_app_ui", "Inspect a page of front-window accessibility elements with availability and exposed control values; use offset and limit to page.",
        {"app": {"type": "string"}, "offset": {"type": "string"}, "limit": {"type": "string"}}, ["app"],
    ),
    _schema("inspect_focused_app_element", "Inspect the role, label, description, and availability of the app's focused UI element.", {"app": {"type": "string"}}, ["app"]),
    _schema(
        "wait_for_app_element", "Wait for one exact labeled front-window element to exist, become enabled or disabled, or become absent. Timeout defaults to 10 seconds and is capped at 30.",
        {"app": {"type": "string"}, "window": {"type": "string"}, "role": {"type": "string"}, "label": {"type": "string"}, "state": {"type": "string"}, "timeout": {"type": "string"}},
        ["app", "window", "role", "label"],
    ),
    _schema(
        "inspect_app_menu", "List app menus or named menu/submenu items with enabled status.",
        {"app": {"type": "string"}, "menu": {"type": "string"}, "submenu": {"type": "string"}}, ["app"],
    ),
    _schema(
        "wait_for_app_menu_item", "Wait for one exact menu or submenu item to exist, enable, disable, or become absent.",
        {"app": {"type": "string"}, "menu": {"type": "string"}, "item": {"type": "string"}, "submenu": {"type": "string"}, "state": {"type": "string"}, "timeout": {"type": "string"}},
        ["app", "menu", "item"],
    ),
    _schema(
        "click_app_menu_item", "Click an exact menu item in a running app; optionally name its parent submenu.",
        {"app": {"type": "string"}, "menu": {"type": "string"}, "item": {"type": "string"}, "submenu": {"type": "string"}},
        ["app", "menu", "item"],
    ),
    _schema(
        "click_app_element", "Click a front-window element by role and label, or by role and inspected index when unlabeled.",
        {"app": {"type": "string"}, "window": {"type": "string"}, "role": {"type": "string"}, "label": {"type": "string"}, "index": {"type": "string"}},
        ["app", "window", "role"],
    ),
    _schema(
        "show_app_element_menu", "Open the accessibility context menu for one exact front-window element by label or inspected index.",
        {"app": {"type": "string"}, "window": {"type": "string"}, "role": {"type": "string"}, "label": {"type": "string"}, "index": {"type": "string"}},
        ["app", "window", "role"],
    ),
    _schema(
        "select_app_popup_item", "Select one exact item from an inspected AXPopUpButton by label or element index.",
        {"app": {"type": "string"}, "window": {"type": "string"}, "label": {"type": "string"}, "index": {"type": "string"}, "item": {"type": "string"}},
        ["app", "window", "item"],
    ),
    _schema(
        "set_app_toggle", "Set an inspected checkbox, switch, or disclosure triangle on or off; does nothing if already in the requested state.",
        {"app": {"type": "string"}, "window": {"type": "string"}, "role": {"type": "string"}, "label": {"type": "string"}, "index": {"type": "string"}, "state": {"type": "string"}},
        ["app", "window", "role", "state"],
    ),
    _schema(
        "select_app_radio", "Select one inspected AXRadioButton by label or element index; does nothing when already selected.",
        {"app": {"type": "string"}, "window": {"type": "string"}, "label": {"type": "string"}, "index": {"type": "string"}},
        ["app", "window"],
    ),
    _schema(
        "adjust_app_control", "Increase or decrease an inspected AXSlider or AXIncrementor by 1 to 20 accessibility steps.",
        {"app": {"type": "string"}, "window": {"type": "string"}, "role": {"type": "string"}, "label": {"type": "string"}, "index": {"type": "string"}, "direction": {"type": "string"}, "steps": {"type": "string"}},
        ["app", "window", "role", "direction"],
    ),
    _schema(
        "type_in_app", "Type into an inspected text field by role and label or index, or the currently focused control when no target is supplied.",
        {"app": {"type": "string"}, "window": {"type": "string"}, "text": {"type": "string"}, "role": {"type": "string"}, "label": {"type": "string"}, "index": {"type": "string"}},
        ["app", "window", "text"],
    ),
    _schema(
        "replace_app_text", "Replace all text in one inspected text field, text area, or combo box; an empty value clears it.",
        {"app": {"type": "string"}, "window": {"type": "string"}, "text": {"type": "string"}, "role": {"type": "string"}, "label": {"type": "string"}, "index": {"type": "string"}},
        ["app", "window", "text", "role"],
    ),
    _schema(
        "press_app_key", "Press one key or shortcut in the named app and front window. Named keys include arrows, home, end, page_up, page_down, enter, return, delete, forward_delete, tab, space, escape. Modifiers: command, option, control, shift.",
        {"app": {"type": "string"}, "window": {"type": "string"}, "key": {"type": "string"}, "modifiers": {"type": "string"}},
        ["app", "window", "key"],
    ),
    _schema("speak", "Speak text using the macOS voice.", {"text": {"type": "string"}}, ["text"]),
    _schema(
        "notify",
        "Show a macOS notification.",
        {"title": {"type": "string"}, "message": {"type": "string"}},
        ["title", "message"],
    ),
    _schema("system_info", "Get basic local Mac system information.", {}, []),
    _schema("get_clipboard", "Read text from the macOS clipboard.", {}, []),
    _schema("set_clipboard", "Replace text on the macOS clipboard.", {"text": {"type": "string"}}, ["text"]),
    _schema(
        "create_reminder",
        "Create a reminder in the macOS Reminders app.",
        {"title": {"type": "string"}, "notes": {"type": "string"}},
        ["title"],
    ),
    _schema("get_calendar_events", "List today's events from macOS Calendar.", {}, []),
    _schema(
        "run_applescript",
        "Run AppleScript for Mac app automation. This may require Automation permissions.",
        {"script": {"type": "string"}},
        ["script"],
    ),
    _schema("trash_path", "Move a file or folder to the macOS Trash.", {"path": {"type": "string"}}, ["path"]),
]


def available_tool_schemas(action_mode: str) -> list[dict[str, Any]]:
    return [
        schema for schema in TOOL_SCHEMAS
        if tool_allowed(action_mode, schema["function"]["name"])
    ]


def validate_tool_arguments(name: str, args: Any) -> str | None:
    if not isinstance(args, dict):
        return f"{name} arguments must be a JSON object"
    schema = next(item for item in TOOL_SCHEMAS if item["function"]["name"] == name)
    parameters = schema["function"]["parameters"]
    missing = set(parameters["required"]) - args.keys()
    unexpected = args.keys() - parameters["properties"].keys()
    wrong_type = {
        key for key, value in args.items()
        if key in parameters["properties"] and not isinstance(value, str)
    }
    if missing:
        return f"{name} is missing required arguments: {', '.join(sorted(missing))}"
    if unexpected:
        return f"{name} has unexpected arguments: {', '.join(sorted(unexpected))}"
    if wrong_type:
        return f"{name} arguments must be strings: {', '.join(sorted(wrong_type))}"
    return None


def _bounded_integer(raw: str, name: str, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bounded_output(output: str) -> tuple[str, bool]:
    if len(output) <= 12000:
        return output, False
    marker = "\n[NUMS: middle of command output omitted]\n"
    return output[:6000] + marker + output[-(6000 - len(marker)):], True


def _run(command: list[str], cwd: str | None = None, timeout: int = 120) -> str:
    try:
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        output, truncated = _bounded_output((stdout + stderr).strip())
        return json.dumps({
            "exit_code": None, "output": output, "truncated": truncated,
            "error": f"Command timed out after {timeout} seconds",
        })
    output, truncated = _bounded_output((result.stdout + result.stderr).strip())
    response = {
        "exit_code": result.returncode,
        "output": output,
        "truncated": truncated,
    }
    if result.returncode != 0:
        response["error"] = f"Command exited with status {result.returncode}"
    return json.dumps(response)


UI_TOOLS = {
    "list_running_apps", "activate_app", "list_app_windows", "wait_for_app_window", "focus_app_window", "close_app_window", "set_app_window_minimized", "set_app_window_fullscreen", "move_app_window", "resize_app_window", "inspect_app_ui", "inspect_focused_app_element", "wait_for_app_element", "inspect_app_menu", "wait_for_app_menu_item", "click_app_menu_item",
    "click_app_element", "show_app_element_menu", "select_app_popup_item", "set_app_toggle", "select_app_radio", "adjust_app_control", "type_in_app", "replace_app_text", "press_app_key",
}


def _with_ui_permission_hint(result: str) -> str:
    try:
        response = json.loads(result)
    except json.JSONDecodeError:
        return result
    if not isinstance(response, dict) or not response.get("error"):
        return result
    output = str(response.get("output", "")).lower()
    if "-1743" in output or "not authorized to send apple events" in output:
        response["hint"] = "Allow your terminal to control System Events in System Settings > Privacy & Security > Automation."
    elif any(marker in output for marker in ("-25211", "-1719", "not allowed assistive access")):
        response["hint"] = "Allow your terminal in System Settings > Privacy & Security > Accessibility."
    elif "-10827" in output:
        response["hint"] = "Launch Services could not open System Events. Retry from a normal logged-in macOS desktop session."
    return json.dumps(response)


class MacTools:
    def __init__(self, action_mode: str = "unrestricted") -> None:
        self.action_mode = action_mode

    def execute(self, name: str, args: dict[str, Any]) -> str:
        handlers: dict[str, Callable[[dict[str, Any]], str]] = {
            "read_file": self.read_file,
            "read_lines": self.read_lines,
            "list_directory": self.list_directory,
            "find_files": self.find_files,
            "search_files": self.search_files,
            "git_status": self.git_status,
            "git_diff": self.git_diff,
            "write_file": self.write_file,
            "replace_in_file": self.replace_in_file,
            "shell": self.shell,
            "open_item": self.open_item,
            "list_running_apps": self.list_running_apps,
            "activate_app": self.activate_app,
            "list_app_windows": self.list_app_windows,
            "wait_for_app_window": self.wait_for_app_window,
            "focus_app_window": self.focus_app_window,
            "close_app_window": self.close_app_window,
            "set_app_window_minimized": self.set_app_window_minimized,
            "set_app_window_fullscreen": self.set_app_window_fullscreen,
            "move_app_window": self.move_app_window,
            "resize_app_window": self.resize_app_window,
            "inspect_app_ui": self.inspect_app_ui,
            "inspect_focused_app_element": self.inspect_focused_app_element,
            "wait_for_app_element": self.wait_for_app_element,
            "inspect_app_menu": self.inspect_app_menu,
            "wait_for_app_menu_item": self.wait_for_app_menu_item,
            "click_app_menu_item": self.click_app_menu_item,
            "click_app_element": self.click_app_element,
            "show_app_element_menu": self.show_app_element_menu,
            "select_app_popup_item": self.select_app_popup_item,
            "set_app_toggle": self.set_app_toggle,
            "select_app_radio": self.select_app_radio,
            "adjust_app_control": self.adjust_app_control,
            "type_in_app": self.type_in_app,
            "replace_app_text": self.replace_app_text,
            "press_app_key": self.press_app_key,
            "speak": self.speak,
            "notify": self.notify,
            "system_info": self.system_info,
            "get_clipboard": self.get_clipboard,
            "set_clipboard": self.set_clipboard,
            "create_reminder": self.create_reminder,
            "get_calendar_events": self.get_calendar_events,
            "run_applescript": self.run_applescript,
            "trash_path": self.trash_path,
        }
        if name not in handlers:
            return json.dumps({"error": f"Unknown tool: {name}"})
        if not tool_allowed(self.action_mode, name):
            return json.dumps({
                "error": f"Tool {name} is blocked in {self.action_mode} mode",
                "action_mode": self.action_mode,
            })
        validation_error = validate_tool_arguments(name, args)
        if validation_error:
            return json.dumps({"error": validation_error})
        try:
            result = handlers[name](args)
            return _with_ui_permission_hint(result) if name in UI_TOOLS else result
        except Exception as exc:  # tool errors are returned to the model
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def read_file(self, args: dict[str, Any]) -> str:
        raw_offset = args.get("offset", "0")
        if not raw_offset.isdecimal():
            raise ValueError("offset must be a nonnegative character position")
        offset = int(raw_offset)
        with Path(args["path"]).expanduser().open(encoding="utf-8", errors="replace") as handle:
            remaining = offset
            while remaining:
                skipped = handle.read(min(remaining, 8192))
                if not skipped:
                    return ""
                remaining -= len(skipped)
            content = handle.read(12001)
        if len(content) > 12000:
            return content[:12000] + (
                f"\n[NUMS: file output truncated after 12000 characters; "
                f"continue with offset {offset + 12000}]"
            )
        return content

    def read_lines(self, args: dict[str, Any]) -> str:
        raw_start = args.get("start_line", "1")
        raw_count = args.get("count", "80")
        if not raw_start.isdecimal() or int(raw_start) < 1:
            raise ValueError("start_line must be a positive line number")
        if not raw_count.isdecimal() or not 1 <= int(raw_count) <= 100:
            raise ValueError("count must be between 1 and 100")
        start, count = int(raw_start), int(raw_count)
        lines: list[dict[str, Any]] = []
        characters = 0
        next_line = None
        with Path(args["path"]).expanduser().open(encoding="utf-8", errors="replace") as handle:
            for number, raw_line in enumerate(handle, start=1):
                if number < start:
                    continue
                if len(lines) >= count or characters >= 12000:
                    next_line = number
                    break
                line = raw_line.rstrip("\r\n")
                clipped = len(line) > 2000
                if clipped:
                    line = line[:2000]
                if characters + len(line) > 12000:
                    next_line = number
                    break
                lines.append({"number": number, "text": line, "truncated": clipped})
                characters += len(line)
        response: dict[str, Any] = {"lines": lines}
        if next_line is not None:
            response["next_line"] = next_line
        return json.dumps(response)

    def list_directory(self, args: dict[str, Any]) -> str:
        raw_offset = args.get("offset", "0")
        if not raw_offset.isdecimal():
            raise ValueError("offset must be a nonnegative item position")
        offset = int(raw_offset)
        path = Path(args["path"]).expanduser()
        items = [
            {"name": child.name, "type": "directory" if child.is_dir() else "file"}
            for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        ]
        response: dict[str, Any] = {
            "items": items[offset:offset + 500],
            "total": len(items),
            "truncated": len(items) > offset + 500,
        }
        if response["truncated"]:
            response["next_offset"] = offset + 500
        return json.dumps(response)

    def find_files(self, args: dict[str, Any]) -> str:
        raw_offset = args.get("offset", "0")
        if not raw_offset.isdecimal():
            raise ValueError("offset must be a nonnegative file position")
        offset = int(raw_offset)
        root = Path(args["path"]).expanduser()
        command = [
            "rg", "--files", "--hidden", "--no-require-git", "--sort", "path",
            "--glob", "!.git",
        ]
        if "glob" in args:
            if not args["glob"].strip():
                raise ValueError("glob must not be empty")
        command.extend(["--", str(root)])
        result = subprocess.run(command, text=True, capture_output=True, timeout=30)
        if result.returncode not in {0, 1}:
            return json.dumps({"error": result.stderr.strip()[-1000:], "exit_code": result.returncode})
        files = result.stdout.splitlines()
        if "glob" in args:
            files = [
                file for file in files
                if fnmatchcase(os.path.relpath(file, root), args["glob"])
            ]
        response: dict[str, Any] = {
            "files": files[offset:offset + 200],
            "total": len(files),
            "truncated": len(files) > offset + 200,
        }
        if response["truncated"]:
            response["next_offset"] = offset + 200
        return json.dumps(response)

    def search_files(self, args: dict[str, Any]) -> str:
        if not args["query"].strip():
            raise ValueError("query must not be empty")
        command = ["rg", "-n", "-F", "--hidden", "--glob", "!.git"]
        if "glob" in args:
            if not args["glob"].strip():
                raise ValueError("glob must not be empty")
            command.extend(["--glob", args["glob"]])
        command.extend(["--", args["query"], str(Path(args["path"]).expanduser())])
        result = json.loads(_run(command))
        if result["exit_code"] == 1:
            result.pop("error", None)
            result["matches"] = 0
        return json.dumps(result)

    def git_status(self, args: dict[str, Any]) -> str:
        repo = str(Path(args["repo"]).expanduser())
        return _run(["git", "-C", repo, "status", "--short", "--branch"])

    def git_diff(self, args: dict[str, Any]) -> str:
        repo = str(Path(args["repo"]).expanduser())
        command = ["git", "-C", repo, "diff", "--no-ext-diff", "--no-color", "HEAD", "--"]
        if "file" in args:
            if not args["file"].strip():
                raise ValueError("file must not be empty")
            command.append(args["file"])
        return _run(command)

    def _atomic_write(self, requested: Path, content: str) -> None:
        path = requested.resolve() if requested.is_symlink() else requested
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists():
                temporary.chmod(stat.S_IMODE(path.stat().st_mode))
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def write_file(self, args: dict[str, Any]) -> str:
        requested = Path(args["path"]).expanduser()
        self._atomic_write(requested, args["content"])
        return json.dumps({"written": str(requested), "bytes": len(args["content"].encode())})

    def replace_in_file(self, args: dict[str, Any]) -> str:
        requested = Path(args["path"]).expanduser()
        old = args["old_text"]
        if not old:
            raise ValueError("old_text must not be empty")
        path = requested.resolve() if requested.is_symlink() else requested
        with path.open("r", encoding="utf-8", newline="") as handle:
            content = handle.read()
        occurrences = content.count(old)
        if occurrences != 1:
            raise ValueError(f"old_text must occur exactly once; found {occurrences}")
        replacement = content.replace(old, args["new_text"], 1)
        self._atomic_write(requested, replacement)
        return json.dumps({"updated": str(requested), "replacements": 1})

    def shell(self, args: dict[str, Any]) -> str:
        cwd = str(Path(args.get("cwd") or Path.home()).expanduser())
        return _run(["/bin/zsh", "-lc", args["command"]], cwd=cwd)

    def open_item(self, args: dict[str, Any]) -> str:
        target = args["target"]
        if not target.strip():
            raise ValueError("target must not be empty")
        kind = args.get("kind", "auto")
        if kind not in {"auto", "app", "path", "url"}:
            raise ValueError("kind must be auto, app, path, or url")
        path = Path(target).expanduser()
        if kind == "app":
            return _run(["open", "-a", target])
        if kind == "path":
            return _run(["open", str(path)])
        if kind == "url":
            if not urlparse(target).scheme:
                raise ValueError("url target must include a scheme")
            return _run(["open", target])
        if urlparse(target).scheme or path.exists() or path.suffix or "/" in target:
            return _run(["open", str(path) if target.startswith("~") else target])
        return _run(["open", "-a", target])

    def list_running_apps(self, args: dict[str, Any]) -> str:
        return _run(["osascript", "-e", LIST_APPS_SCRIPT], timeout=20)

    def activate_app(self, args: dict[str, Any]) -> str:
        if not args["app"].strip():
            raise ValueError("app must not be empty")
        return _run(["osascript", "-e", ACTIVATE_APP_SCRIPT, args["app"]], timeout=20)

    def list_app_windows(self, args: dict[str, Any]) -> str:
        if not args["app"].strip():
            raise ValueError("app must not be empty")
        return _run(["osascript", "-e", LIST_WINDOWS_SCRIPT, args["app"]], timeout=20)

    def wait_for_app_window(self, args: dict[str, Any]) -> str:
        if not args["app"].strip() or not args["window"].strip():
            raise ValueError("app and window must not be empty")
        state = args.get("state", "exists")
        if state not in {"exists", "absent"}:
            raise ValueError("state must be exists or absent")
        timeout = args.get("timeout", "10")
        if not timeout.isdecimal() or not 1 <= int(timeout) <= 30:
            raise ValueError("timeout must be between 1 and 30 seconds")
        return _run([
            "osascript", "-e", WAIT_WINDOW_SCRIPT,
            args["app"], args["window"], state, timeout,
        ], timeout=int(timeout) + 5)

    def focus_app_window(self, args: dict[str, Any]) -> str:
        if not args["app"].strip() or not args["window"].strip():
            raise ValueError("app and window must not be empty")
        index = args.get("index", "")
        if index and (not index.isdecimal() or int(index) < 1):
            raise ValueError("index must be a positive window number")
        return _run(["osascript", "-e", FOCUS_WINDOW_SCRIPT, args["app"], args["window"], index], timeout=20)

    def close_app_window(self, args: dict[str, Any]) -> str:
        if not args["app"].strip() or not args["window"].strip():
            raise ValueError("app and window must not be empty")
        index = args.get("index", "")
        if index and (not index.isdecimal() or int(index) < 1):
            raise ValueError("index must be a positive window number")
        return _run([
            "osascript", "-e", CLOSE_WINDOW_SCRIPT,
            args["app"], args["window"], index,
        ], timeout=20)

    def set_app_window_minimized(self, args: dict[str, Any]) -> str:
        if not args["app"].strip() or not args["window"].strip():
            raise ValueError("app and window must not be empty")
        if args["state"] not in {"on", "off"}:
            raise ValueError("state must be on or off")
        index = args.get("index", "")
        if index and (not index.isdecimal() or int(index) < 1):
            raise ValueError("index must be a positive window number")
        return _run([
            "osascript", "-e", SET_WINDOW_MINIMIZED_SCRIPT,
            args["app"], args["window"], index, args["state"],
        ], timeout=20)

    def set_app_window_fullscreen(self, args: dict[str, Any]) -> str:
        if not args["app"].strip() or not args["window"].strip():
            raise ValueError("app and window must not be empty")
        if args["state"] not in {"on", "off"}:
            raise ValueError("state must be on or off")
        index = args.get("index", "")
        if index and (not index.isdecimal() or int(index) < 1):
            raise ValueError("index must be a positive window number")
        return _run([
            "osascript", "-e", SET_WINDOW_FULLSCREEN_SCRIPT,
            args["app"], args["window"], index, args["state"],
        ], timeout=20)

    def move_app_window(self, args: dict[str, Any]) -> str:
        if not args["app"].strip() or not args["window"].strip():
            raise ValueError("app and window must not be empty")
        index = args.get("index", "")
        if index and (not index.isdecimal() or int(index) < 1):
            raise ValueError("index must be a positive window number")
        x = _bounded_integer(args["x"], "x", -10000, 10000)
        y = _bounded_integer(args["y"], "y", -10000, 10000)
        return _run([
            "osascript", "-e", MOVE_WINDOW_SCRIPT,
            args["app"], args["window"], index, str(x), str(y),
        ], timeout=20)

    def resize_app_window(self, args: dict[str, Any]) -> str:
        if not args["app"].strip() or not args["window"].strip():
            raise ValueError("app and window must not be empty")
        index = args.get("index", "")
        if index and (not index.isdecimal() or int(index) < 1):
            raise ValueError("index must be a positive window number")
        width = _bounded_integer(args["width"], "width", 100, 10000)
        height = _bounded_integer(args["height"], "height", 100, 10000)
        return _run([
            "osascript", "-e", RESIZE_WINDOW_SCRIPT,
            args["app"], args["window"], index, str(width), str(height),
        ], timeout=20)

    def inspect_app_ui(self, args: dict[str, Any]) -> str:
        offset = args.get("offset", "0")
        if not args["app"].strip() or not offset.isdecimal():
            raise ValueError("app must not be empty and offset must be nonnegative")
        limit = args.get("limit", "40")
        if not limit.isdecimal() or not 1 <= int(limit) <= 120:
            raise ValueError("limit must be between 1 and 120")
        return _run(["osascript", "-e", INSPECT_APP_SCRIPT, args["app"], offset, limit], timeout=20)

    def inspect_focused_app_element(self, args: dict[str, Any]) -> str:
        if not args["app"].strip():
            raise ValueError("app must not be empty")
        return _run(["osascript", "-e", INSPECT_FOCUSED_ELEMENT_SCRIPT, args["app"]], timeout=20)

    def wait_for_app_element(self, args: dict[str, Any]) -> str:
        if not all(args[key].strip() for key in ("app", "window", "role", "label")):
            raise ValueError("app, window, role, and label must not be empty")
        state = args.get("state", "enabled")
        if state not in {"exists", "enabled", "disabled", "absent"}:
            raise ValueError("state must be exists, enabled, disabled, or absent")
        timeout = args.get("timeout", "10")
        if not timeout.isdecimal() or not 1 <= int(timeout) <= 30:
            raise ValueError("timeout must be between 1 and 30 seconds")
        return _run([
            "osascript", "-e", WAIT_ELEMENT_SCRIPT,
            args["app"], args["window"], args["role"], args["label"], state, timeout,
        ], timeout=int(timeout) + 5)

    def inspect_app_menu(self, args: dict[str, Any]) -> str:
        if not args["app"].strip():
            raise ValueError("app must not be empty")
        if args.get("submenu") and not args.get("menu"):
            raise ValueError("submenu requires a menu")
        return _run([
            "osascript", "-e", INSPECT_MENU_SCRIPT,
            args["app"], args.get("menu", ""), args.get("submenu", ""),
        ], timeout=20)

    def wait_for_app_menu_item(self, args: dict[str, Any]) -> str:
        if not all(args[key].strip() for key in ("app", "menu", "item")):
            raise ValueError("app, menu, and item must not be empty")
        state = args.get("state", "enabled")
        if state not in {"exists", "enabled", "disabled", "absent"}:
            raise ValueError("state must be exists, enabled, disabled, or absent")
        timeout = args.get("timeout", "10")
        if not timeout.isdecimal() or not 1 <= int(timeout) <= 30:
            raise ValueError("timeout must be between 1 and 30 seconds")
        return _run([
            "osascript", "-e", WAIT_MENU_ITEM_SCRIPT,
            args["app"], args["menu"], args["item"], args.get("submenu", ""), state, timeout,
        ], timeout=int(timeout) + 5)

    def click_app_menu_item(self, args: dict[str, Any]) -> str:
        if not all(args[key].strip() for key in ("app", "menu", "item")):
            raise ValueError("app, menu, and item must not be empty")
        return _run([
            "osascript", "-e", CLICK_MENU_ITEM_SCRIPT,
            args["app"], args["menu"], args["item"], args.get("submenu", ""),
        ], timeout=20)

    def click_app_element(self, args: dict[str, Any]) -> str:
        if not all(args[key].strip() for key in ("app", "window", "role")):
            raise ValueError("app, window, and role must not be empty")
        index = args.get("index", "")
        if index and (not index.isdecimal() or int(index) < 1):
            raise ValueError("index must be a positive element number")
        label = args.get("label", "")
        if not label and not index:
            raise ValueError("label or inspected index is required")
        return _run([
            "osascript", "-e", CLICK_ELEMENT_SCRIPT,
            args["app"], args["window"], args["role"], label, index,
        ], timeout=20)

    def show_app_element_menu(self, args: dict[str, Any]) -> str:
        if not all(args[key].strip() for key in ("app", "window", "role")):
            raise ValueError("app, window, and role must not be empty")
        label, index = args.get("label", ""), args.get("index", "")
        if index and (not index.isdecimal() or int(index) < 1):
            raise ValueError("index must be a positive element number")
        if not label and not index:
            raise ValueError("label or inspected index is required")
        return _run([
            "osascript", "-e", SHOW_ELEMENT_MENU_SCRIPT,
            args["app"], args["window"], args["role"], label, index,
        ], timeout=20)

    def select_app_popup_item(self, args: dict[str, Any]) -> str:
        if not all(args[key].strip() for key in ("app", "window", "item")):
            raise ValueError("app, window, and item must not be empty")
        label, index = args.get("label", ""), args.get("index", "")
        if index and (not index.isdecimal() or int(index) < 1):
            raise ValueError("index must be a positive element number")
        if not label and not index:
            raise ValueError("label or inspected index is required")
        return _run([
            "osascript", "-e", SELECT_POPUP_ITEM_SCRIPT,
            args["app"], args["window"], label, index, args["item"],
        ], timeout=20)

    def type_in_app(self, args: dict[str, Any]) -> str:
        if not args["app"].strip() or not args["window"].strip() or not args["text"]:
            raise ValueError("app, window, and text must not be empty")
        role, label, index = (args.get(key, "") for key in ("role", "label", "index"))
        if label and not role:
            raise ValueError("label requires a role")
        if role and role not in {"AXTextField", "AXTextArea", "AXComboBox"}:
            raise ValueError("role must be AXTextField, AXTextArea, or AXComboBox")
        if index and (not role or not index.isdecimal() or int(index) < 1):
            raise ValueError("index requires a text field and a positive element number")
        if role and not (label or index):
            raise ValueError("targeted typing requires a label or inspected index")
        return _run([
            "osascript", "-e", TYPE_TEXT_SCRIPT,
            args["app"], args["window"], args["text"], role, label, index, "append",
        ], timeout=20)

    def replace_app_text(self, args: dict[str, Any]) -> str:
        if not args["app"].strip() or not args["window"].strip():
            raise ValueError("app and window must not be empty")
        role, label, index = (args.get(key, "") for key in ("role", "label", "index"))
        if role not in {"AXTextField", "AXTextArea", "AXComboBox"}:
            raise ValueError("role must be AXTextField, AXTextArea, or AXComboBox")
        if index and (not index.isdecimal() or int(index) < 1):
            raise ValueError("index must be a positive element number")
        if not label and not index:
            raise ValueError("label or inspected index is required")
        return _run([
            "osascript", "-e", TYPE_TEXT_SCRIPT,
            args["app"], args["window"], args["text"], role, label, index, "replace",
        ], timeout=20)

    def set_app_toggle(self, args: dict[str, Any]) -> str:
        if not args["app"].strip() or not args["window"].strip():
            raise ValueError("app and window must not be empty")
        if args["role"] not in {"AXCheckBox", "AXSwitch", "AXDisclosureTriangle"}:
            raise ValueError("role must be AXCheckBox, AXSwitch, or AXDisclosureTriangle")
        if args["state"] not in {"on", "off"}:
            raise ValueError("state must be on or off")
        label, index = args.get("label", ""), args.get("index", "")
        if index and (not index.isdecimal() or int(index) < 1):
            raise ValueError("index must be a positive element number")
        if not label and not index:
            raise ValueError("label or inspected index is required")
        return _run([
            "osascript", "-e", SET_TOGGLE_SCRIPT,
            args["app"], args["window"], args["role"], label, index, args["state"],
        ], timeout=20)

    def select_app_radio(self, args: dict[str, Any]) -> str:
        if not args["app"].strip() or not args["window"].strip():
            raise ValueError("app and window must not be empty")
        label, index = args.get("label", ""), args.get("index", "")
        if index and (not index.isdecimal() or int(index) < 1):
            raise ValueError("index must be a positive element number")
        if not label and not index:
            raise ValueError("label or inspected index is required")
        return _run([
            "osascript", "-e", SELECT_RADIO_SCRIPT,
            args["app"], args["window"], label, index,
        ], timeout=20)

    def adjust_app_control(self, args: dict[str, Any]) -> str:
        if not args["app"].strip() or not args["window"].strip():
            raise ValueError("app and window must not be empty")
        if args["role"] not in {"AXSlider", "AXIncrementor"}:
            raise ValueError("role must be AXSlider or AXIncrementor")
        if args["direction"] not in {"increase", "decrease"}:
            raise ValueError("direction must be increase or decrease")
        label, index = args.get("label", ""), args.get("index", "")
        if index and (not index.isdecimal() or int(index) < 1):
            raise ValueError("index must be a positive element number")
        if not label and not index:
            raise ValueError("label or inspected index is required")
        steps = args.get("steps", "1")
        if not steps.isdecimal() or not 1 <= int(steps) <= 20:
            raise ValueError("steps must be between 1 and 20")
        return _run([
            "osascript", "-e", ADJUST_CONTROL_SCRIPT,
            args["app"], args["window"], args["role"], label, index,
            args["direction"], steps,
        ], timeout=20)

    def press_app_key(self, args: dict[str, Any]) -> str:
        if not args["app"].strip() or not args["window"].strip():
            raise ValueError("app and window must not be empty")
        script = key_script(args["key"], args.get("modifiers", ""))
        return _run([
            "osascript", "-e", script, args["app"], args["window"], args["key"],
        ], timeout=20)

    def speak(self, args: dict[str, Any]) -> str:
        return _run(["say", args["text"]])

    def notify(self, args: dict[str, Any]) -> str:
        return _run(["osascript", "-e", NOTIFY_SCRIPT, args["message"], args["title"]])

    def system_info(self, args: dict[str, Any]) -> str:
        return json.dumps(
            {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "hostname": platform.node(),
                "user": os.getenv("USER"),
                "home": str(Path.home()),
            }
        )

    def get_clipboard(self, args: dict[str, Any]) -> str:
        return _run(["pbpaste"])

    def set_clipboard(self, args: dict[str, Any]) -> str:
        result = subprocess.run(
            ["pbcopy"], input=args["text"], text=True, capture_output=True, timeout=10
        )
        if result.returncode != 0:
            return json.dumps({
                "exit_code": result.returncode,
                "error": f"Clipboard write failed: {result.stderr.strip()[-1000:]}",
            })
        return json.dumps({"exit_code": 0, "characters": len(args["text"])})

    def create_reminder(self, args: dict[str, Any]) -> str:
        if not args["title"].strip():
            raise ValueError("title must not be empty")
        script = """on run argv
tell application "Reminders"
  tell default list
    set createdReminder to make new reminder with properties {name:item 1 of argv, body:item 2 of argv}
    return name of createdReminder
  end tell
end tell
end run"""
        return _run(["osascript", "-e", script, args["title"], args.get("notes", "")])

    def get_calendar_events(self, args: dict[str, Any]) -> str:
        script = """set startOfDay to current date
set time of startOfDay to 0
set endOfDay to startOfDay + (1 * days)
tell application "Calendar"
  set rows to {}
  repeat with cal in calendars
    repeat with eventItem in (every event of cal whose start date is greater than or equal to startOfDay and start date is less than endOfDay)
      set end of rows to (summary of eventItem) & " | " & ((start date of eventItem) as string)
    end repeat
  end repeat
  return rows as string
end tell"""
        return _run(["osascript", "-e", script])

    def run_applescript(self, args: dict[str, Any]) -> str:
        return _run(["osascript", "-e", args["script"]])

    def trash_path(self, args: dict[str, Any]) -> str:
        source = Path(args["path"]).expanduser().absolute()
        if not source.exists() and not source.is_symlink():
            raise FileNotFoundError(source)
        trash_directory = (Path.home() / ".Trash").absolute()
        if trash_directory == source or trash_directory.is_relative_to(source):
            raise ValueError("Cannot move the Trash or its parent into the Trash")
        trash_directory.mkdir(mode=0o700, exist_ok=True)
        trash = trash_directory / source.name
        suffix = 1
        while trash.exists() or trash.is_symlink():
            trash = trash_directory / f"{source.stem}-{suffix}{source.suffix}"
            suffix += 1
        shutil.move(str(source), str(trash))
        return json.dumps({"trashed": str(source), "recoverable_at": str(trash)})
