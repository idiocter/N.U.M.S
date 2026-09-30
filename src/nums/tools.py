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
    CLICK_ELEMENT_SCRIPT, INSPECT_APP_SCRIPT, LIST_APPS_SCRIPT, TYPE_TEXT_SCRIPT,
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
    _schema("open_item", "Open an app, file, folder, or URL.", {"target": {"type": "string"}}, ["target"]),
    _schema("list_running_apps", "List running foreground Mac apps by process name.", {}, []),
    _schema(
        "inspect_app_ui", "Inspect 120 accessibility elements in an app's front window; use offset for later pages.",
        {"app": {"type": "string"}, "offset": {"type": "string"}}, ["app"],
    ),
    _schema(
        "click_app_element", "Click exactly one front-window element matching its AX role and name or accessibility description. Inspect the app first.",
        {"app": {"type": "string"}, "window": {"type": "string"}, "role": {"type": "string"}, "label": {"type": "string"}},
        ["app", "window", "role", "label"],
    ),
    _schema(
        "type_in_app", "Type into the focused control of the named app and front window. Click the field first.",
        {"app": {"type": "string"}, "window": {"type": "string"}, "text": {"type": "string"}},
        ["app", "window", "text"],
    ),
    _schema(
        "press_app_key", "Press one key or a shortcut in the named app and front window. Modifiers: command, option, control, shift, comma-separated.",
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
            "inspect_app_ui": self.inspect_app_ui,
            "click_app_element": self.click_app_element,
            "type_in_app": self.type_in_app,
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
            return handlers[name](args)
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
        path = Path(target).expanduser()
        if urlparse(target).scheme or path.exists() or path.suffix or "/" in target:
            return _run(["open", str(path) if target.startswith("~") else target])
        return _run(["open", "-a", target])

    def list_running_apps(self, args: dict[str, Any]) -> str:
        return _run(["osascript", "-e", LIST_APPS_SCRIPT], timeout=20)

    def inspect_app_ui(self, args: dict[str, Any]) -> str:
        offset = args.get("offset", "0")
        if not args["app"].strip() or not offset.isdecimal():
            raise ValueError("app must not be empty and offset must be nonnegative")
        return _run(["osascript", "-e", INSPECT_APP_SCRIPT, args["app"], offset], timeout=20)

    def click_app_element(self, args: dict[str, Any]) -> str:
        if not all(args[key].strip() for key in ("app", "window", "role", "label")):
            raise ValueError("app, window, role, and label must not be empty")
        return _run([
            "osascript", "-e", CLICK_ELEMENT_SCRIPT,
            args["app"], args["window"], args["role"], args["label"],
        ], timeout=20)

    def type_in_app(self, args: dict[str, Any]) -> str:
        if not args["app"].strip() or not args["window"].strip() or not args["text"]:
            raise ValueError("app, window, and text must not be empty")
        return _run([
            "osascript", "-e", TYPE_TEXT_SCRIPT,
            args["app"], args["window"], args["text"],
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
        script = 'display notification ' + json.dumps(args["message"]) + ' with title ' + json.dumps(args["title"])
        return _run(["osascript", "-e", script])

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
        script = """on run argv
tell application "Reminders"
  tell default list
    make new reminder with properties {name:item 1 of argv, body:item 2 of argv}
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
