"""Optional local conversation history with private, atomic writes."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


def _valid_message(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    role = message.get("role")
    if role == "user":
        return isinstance(message.get("content"), str)
    if role == "tool":
        return (
            isinstance(message.get("tool_name"), str)
            and bool(message["tool_name"])
            and isinstance(message.get("content"), str)
        )
    if role != "assistant":
        return False
    if "content" in message and not isinstance(message["content"], str):
        return False
    calls = message.get("tool_calls")
    if calls is None:
        return "content" in message
    if not isinstance(calls, list) or not calls:
        return False
    return all(
        isinstance(call, dict)
        and isinstance(call.get("function"), dict)
        and isinstance(call["function"].get("name"), str)
        and bool(call["function"]["name"])
        and isinstance(call["function"].get("arguments"), (dict, str))
        for call in calls
    )


class HistoryStore:
    def __init__(self, path: Path, max_bytes: int = 1_000_000) -> None:
        self.path = path.expanduser()
        if max_bytes < 1:
            raise ValueError("history max_bytes must be positive")
        self.max_bytes = max_bytes

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        details = self.path.lstat()
        if self.path.is_symlink() or not stat.S_ISREG(details.st_mode):
            raise ValueError(f"NUMS history path is not a regular file: {self.path}")
        try:
            with self.path.open("rb") as handle:
                data = handle.read(self.max_bytes + 1)
            if len(data) > self.max_bytes:
                raise ValueError(
                    f"NUMS history at {self.path} exceeds the {self.max_bytes}-byte limit"
                )
            messages = json.loads(data.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read NUMS history at {self.path}: {exc}") from exc
        if not isinstance(messages, list) or any(not _valid_message(item) for item in messages):
            raise ValueError(f"Invalid NUMS history at {self.path}")
        while messages and (
            messages[-1]["role"] != "assistant" or messages[-1].get("tool_calls")
        ):
            starts = [index for index, item in enumerate(messages) if item["role"] == "user"]
            if not starts:
                raise ValueError(f"Invalid NUMS history at {self.path}")
            messages = messages[:starts[-1]]
        return messages

    def save(self, messages: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=f".{self.path.name}.", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(messages, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
