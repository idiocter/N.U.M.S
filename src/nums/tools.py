from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS = [
    _schema("read_file", "Read a UTF-8 text file.", {"path": {"type": "string"}}, ["path"]),
    _schema("list_directory", "List files and folders.", {"path": {"type": "string"}}, ["path"]),
    _schema(
        "search_files",
        "Search file contents for a literal string with ripgrep.",
        {"query": {"type": "string"}, "path": {"type": "string"}},
        ["query", "path"],
    ),
    _schema(
        "write_file",
        "Write UTF-8 text to a file, creating parent folders.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    _schema(
        "shell",
        "Run a zsh command on this Mac. Use for tasks not covered by a narrower tool.",
        {"command": {"type": "string"}, "cwd": {"type": "string"}},
        ["command"],
    ),
    _schema("open_item", "Open an app, file, folder, or URL.", {"target": {"type": "string"}}, ["target"]),
    _schema("speak", "Speak text using the macOS voice.", {"text": {"type": "string"}}, ["text"]),
    _schema(
        "notify",
        "Show a macOS notification.",
        {"title": {"type": "string"}, "message": {"type": "string"}},
        ["title", "message"],
    ),
    _schema("system_info", "Get basic local Mac system information.", {}, []),
    _schema(
        "run_applescript",
        "Run AppleScript for Mac app automation. This may require Automation permissions.",
        {"script": {"type": "string"}},
        ["script"],
    ),
    _schema("trash_path", "Move a file or folder to the macOS Trash.", {"path": {"type": "string"}}, ["path"]),
]


def _run(command: list[str], cwd: str | None = None, timeout: int = 120) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    output = (result.stdout + result.stderr).strip()
    return json.dumps({"exit_code": result.returncode, "output": output[-12000:]})


class MacTools:
    def execute(self, name: str, args: dict[str, Any]) -> str:
        handlers: dict[str, Callable[[dict[str, Any]], str]] = {
            "read_file": self.read_file,
            "list_directory": self.list_directory,
            "search_files": self.search_files,
            "write_file": self.write_file,
            "shell": self.shell,
            "open_item": self.open_item,
            "speak": self.speak,
            "notify": self.notify,
            "system_info": self.system_info,
            "run_applescript": self.run_applescript,
            "trash_path": self.trash_path,
        }
        if name not in handlers:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return handlers[name](args)
        except Exception as exc:  # tool errors are returned to the model
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def read_file(self, args: dict[str, Any]) -> str:
        return Path(args["path"]).expanduser().read_text(errors="replace")[:12000]

    def list_directory(self, args: dict[str, Any]) -> str:
        path = Path(args["path"]).expanduser()
        items = [
            {"name": child.name, "type": "directory" if child.is_dir() else "file"}
            for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        ]
        return json.dumps(items[:500])

    def search_files(self, args: dict[str, Any]) -> str:
        return _run([
            "rg", "-n", "-F", "--hidden", "--glob", "!.git", "--",
            args["query"], str(Path(args["path"]).expanduser()),
        ])

    def write_file(self, args: dict[str, Any]) -> str:
        path = Path(args["path"]).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"])
        return json.dumps({"written": str(path), "bytes": len(args["content"].encode())})

    def shell(self, args: dict[str, Any]) -> str:
        cwd = str(Path(args.get("cwd") or Path.home()).expanduser())
        return _run(["/bin/zsh", "-lc", args["command"]], cwd=cwd)

    def open_item(self, args: dict[str, Any]) -> str:
        target = args["target"]
        path = Path(target).expanduser()
        if urlparse(target).scheme or path.exists() or path.suffix or "/" in target:
            return _run(["open", str(path) if target.startswith("~") else target])
        return _run(["open", "-a", target])

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

    def run_applescript(self, args: dict[str, Any]) -> str:
        return _run(["osascript", "-e", args["script"]])

    def trash_path(self, args: dict[str, Any]) -> str:
        source = Path(args["path"]).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        trash = Path.home() / ".Trash" / source.name
        if trash.exists():
            trash = trash.with_name(f"{trash.stem}-{os.getpid()}{trash.suffix}")
        shutil.move(str(source), str(trash))
        return json.dumps({"trashed": str(source), "recoverable_at": str(trash)})
