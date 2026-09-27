import pytest

from training.compare_reports import compare


def _case(identifier: str, tool: str, correct: bool) -> dict:
    return {
        "id": identifier, "expected_tool": tool, "expected_arguments": {},
        "tool_correct": correct, "arguments_correct": correct,
    }


def test_candidate_must_improve_on_same_held_out_cases() -> None:
    baseline = {"cases": [_case("a", "read_file", True), _case("b", "open_item", False)]}
    candidate = {"cases": [_case("a", "read_file", True), _case("b", "open_item", True)]}

    assert compare(baseline, candidate) == (True, [])


def test_per_tool_regression_blocks_candidate_even_with_higher_total() -> None:
    baseline = {"cases": [
        _case("a", "read_file", True),
        _case("b", "open_item", False),
        _case("c", "open_item", False),
    ]}
    candidate = {"cases": [
        _case("a", "read_file", False),
        _case("b", "open_item", True),
        _case("c", "open_item", True),
    ]}

    eligible, reasons = compare(baseline, candidate)

    assert not eligible
    assert "read_file accuracy regressed" in reasons


def test_comparison_rejects_changed_labels() -> None:
    baseline = {"cases": [_case("a", "read_file", True)]}
    candidate = {"cases": [_case("a", "open_item", True)]}

    with pytest.raises(ValueError, match="label changed"):
        compare(baseline, candidate)
