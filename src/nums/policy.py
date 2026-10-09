"""Configurable tool access modes for NUMS."""

from __future__ import annotations


READ_ONLY_TOOLS = {
    "read_file", "read_lines", "list_directory", "path_info", "find_files", "search_files",
    "git_status", "git_diff", "system_info",
    "get_clipboard", "get_calendar_events", "list_running_apps", "inspect_app_ui", "inspect_focused_app_element",
    "wait_for_app_element", "inspect_app_menu", "wait_for_app_menu_item",
    "list_app_windows", "wait_for_app_window",
}
STANDARD_TOOLS = READ_ONLY_TOOLS | {
    "write_file", "replace_in_file", "open_item", "speak", "notify", "set_clipboard", "create_reminder",
}
VALID_MODES = {"read_only", "standard", "unrestricted"}


def tool_allowed(mode: str, name: str) -> bool:
    if mode == "unrestricted":
        return True
    if mode == "standard":
        return name in STANDARD_TOOLS
    return name in READ_ONLY_TOOLS
