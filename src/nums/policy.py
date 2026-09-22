"""Configurable tool access modes for NUMS."""

from __future__ import annotations


READ_ONLY_TOOLS = {
    "read_file", "list_directory", "search_files", "system_info",
    "get_clipboard", "get_calendar_events",
}
STANDARD_TOOLS = READ_ONLY_TOOLS | {
    "write_file", "open_item", "speak", "notify", "set_clipboard", "create_reminder",
}
VALID_MODES = {"read_only", "standard", "unrestricted"}


def tool_allowed(mode: str, name: str) -> bool:
    if mode == "unrestricted":
        return True
    if mode == "standard":
        return name in STANDARD_TOOLS
    return name in READ_ONLY_TOOLS
