from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    needs_confirmation: bool
    reason: str


def classify(tool: str, arguments: dict[str, object]) -> Decision:
    if tool in {"write_file", "trash_path", "run_applescript", "shell"}:
        return Decision(True, f"{tool} changes your Mac")
    return Decision(False, "read-only or low-impact action")
