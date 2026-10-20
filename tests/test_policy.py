import pytest

from nums.policy import READ_ONLY_TOOLS, STANDARD_TOOLS, VALID_MODES, tool_allowed
from nums.tools import TOOL_SCHEMAS, MacTools, available_tool_schemas


SCHEMA_NAMES = {item["function"]["name"] for item in TOOL_SCHEMAS}


def test_policy_sets_only_name_real_tools() -> None:
    assert STANDARD_TOOLS <= SCHEMA_NAMES
    assert READ_ONLY_TOOLS <= STANDARD_TOOLS


def test_every_schema_has_a_handler_method() -> None:
    assert [name for name in SCHEMA_NAMES if not callable(getattr(MacTools, name, None))] == []


def test_modes_widen_access_in_order() -> None:
    for name in SCHEMA_NAMES:
        assert tool_allowed("unrestricted", name)
        if tool_allowed("read_only", name):
            assert tool_allowed("standard", name)


@pytest.mark.parametrize(
    ("mode", "blocked", "allowed"),
    [
        ("read_only", {"write_file", "shell", "set_clipboard"}, {"read_file", "git_status"}),
        ("standard", {"shell", "run_applescript", "trash_path"}, {"write_file", "read_file"}),
        ("unrestricted", set(), {"shell", "run_applescript", "trash_path"}),
    ],
)
def test_mode_boundaries(mode: str, blocked: set[str], allowed: set[str]) -> None:
    assert not any(tool_allowed(mode, name) for name in blocked)
    assert all(tool_allowed(mode, name) for name in allowed)


@pytest.mark.parametrize("mode", sorted(VALID_MODES))
def test_model_only_sees_tools_its_mode_allows(mode: str) -> None:
    offered = {item["function"]["name"] for item in available_tool_schemas(mode)}

    assert offered == {name for name in SCHEMA_NAMES if tool_allowed(mode, name)}
