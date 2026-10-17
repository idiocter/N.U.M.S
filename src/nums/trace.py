"""Opt-in, private JSONL traces for reviewing tool decisions."""

from __future__ import annotations

import fcntl
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TraceStore:
    def __init__(self, path: Path, max_bytes: int = 10_000_000) -> None:
        self.path = path.expanduser()
        if max_bytes < 1:
            raise ValueError("trace max_bytes must be positive")
        self.max_bytes = max_bytes

    def append(self, prompt: str, calls: list[dict[str, Any]], reply: str | None, error: str | None = None) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        entry = {
            "at": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "tool_calls": calls,
            "reply": reply,
            "error": error,
        }
        data = (json.dumps(entry, ensure_ascii=False) + "\n").encode()
        if len(data) > self.max_bytes:
            raise ValueError(
                f"NUMS trace entry exceeds the {self.max_bytes}-byte trace limit"
            )
        fd = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
        )
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(f"NUMS trace path is not a regular file: {self.path}")
            fcntl.flock(fd, fcntl.LOCK_EX)
            if os.fstat(fd).st_size + len(data) > self.max_bytes:
                raise ValueError(
                    f"NUMS trace reached the {self.max_bytes}-byte limit"
                )
            os.fchmod(fd, 0o600)
            remaining = memoryview(data)
            while remaining:
                remaining = remaining[os.write(fd, remaining):]
            os.fsync(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
