"""Optional local conversation history with private, atomic writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class HistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            messages = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read NUMS history at {self.path}: {exc}") from exc
        if not isinstance(messages, list) or any(
            not isinstance(item, dict) or item.get("role") not in {"user", "assistant", "tool"}
            for item in messages
        ):
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
