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


def test_new_evaluation_response_errors_block_candidate() -> None:
    baseline = {"cases": [
        _case("a", "read_file", False),
        _case("b", "open_item", False),
    ]}
    candidate = {"cases": [
        _case("a", "read_file", True),
        {**_case("b", "open_item", False), "error": "Evaluation request failed"},
    ]}

    eligible, reasons = compare(baseline, candidate)

    assert not eligible
    assert "evaluation response errors increased" in reasons


def test_comparison_rejects_changed_evaluation_input() -> None:
    baseline = {"cases": [{**_case("a", "read_file", True), "case_fingerprint": "a" * 64}]}
    candidate = {"cases": [{**_case("a", "read_file", True), "case_fingerprint": "b" * 64}]}

    with pytest.raises(ValueError, match="held-out input changed"):
        compare(baseline, candidate)


def test_comparison_rejects_tampered_report_summary() -> None:
    baseline = {
        "total": 1, "tool_correct": 1, "arguments_correct": 1,
        "cases": [_case("a", "read_file", True)],
    }
    candidate = {
        "total": 1, "tool_correct": 1, "arguments_correct": 1,
        "cases": [_case("a", "read_file", False)],
    }

    with pytest.raises(ValueError, match="tool_correct does not match its cases"):
        compare(baseline, candidate)


def test_comparison_rejects_impossible_case_scores() -> None:
    baseline = {"cases": [{
        **_case("a", "read_file", True), "tool_correct": False,
    }]}
    candidate = {"cases": [_case("a", "read_file", True)]}

    with pytest.raises(ValueError, match="correct arguments with the wrong tool"):
        compare(baseline, candidate)


def test_comparison_rejects_correct_score_with_response_error() -> None:
    baseline = {"cases": [{**_case("a", "read_file", True), "error": "timeout"}]}
    candidate = {"cases": [_case("a", "read_file", True)]}

    with pytest.raises(ValueError, match="has an error but is scored correct"):
        compare(baseline, candidate)


def test_comparison_rejects_different_dataset_digests() -> None:
    baseline = {"dataset_digest": "a" * 64, "cases": [_case("a", "read_file", False)]}
    candidate = {"dataset_digest": "b" * 64, "cases": [_case("a", "read_file", True)]}

    with pytest.raises(ValueError, match="different evaluation datasets"):
        compare(baseline, candidate)


def test_comparison_rejects_invalid_report_version() -> None:
    baseline = {"report_version": 2, "cases": [_case("a", "read_file", False)]}
    candidate = {"report_version": 1, "cases": [_case("a", "read_file", True)]}

    with pytest.raises(ValueError, match="unsupported evaluation report version"):
        compare(baseline, candidate)
