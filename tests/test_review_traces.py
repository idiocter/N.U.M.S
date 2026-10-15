import json
import stat
from pathlib import Path

import pytest

from training.review_traces import export, prepare


def test_only_human_reviewed_traces_are_exported(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    review = tmp_path / "review.jsonl"
    output = tmp_path / "examples.jsonl"
    trace.write_text(json.dumps({
        "at": "2026-01-01T00:00:00+00:00",
        "prompt": "open Safari",
        "tool_calls": [{"tool": "open_item", "arguments": {"target": "Safari"}}],
        "reply": "done",
        "error": None,
    }) + "\n")

    assert prepare(trace, review) == 1
    assert export(review, output) == 0
    row = json.loads(review.read_text())
    row["reviewed"] = True
    row["decision"] = "accept"
    review.write_text(json.dumps(row) + "\n")
    assert export(review, output) == 1
    example = json.loads(output.read_text())
    assert example["tool"] == "open_item"
    assert "proposed_tool" not in example
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_prepare_keeps_reviewed_rows_and_private_permissions(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    review = tmp_path / "review.jsonl"
    first = {"at": "2026-01-01", "prompt": "open Safari", "tool_calls": []}
    second = {"at": "2026-01-02", "prompt": "read note", "tool_calls": []}
    trace.write_text(json.dumps(first) + "\n")
    prepare(trace, review)
    row = json.loads(review.read_text())
    row.update({
        "reviewed": True, "decision": "accept", "answer": "Done",
        "review_notes": "Checked",
    })
    review.write_text(json.dumps(row) + "\n")
    trace.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")

    assert prepare(trace, review) == 2
    rows = [json.loads(line) for line in review.read_text().splitlines()]
    assert rows[0] == row
    assert rows[1]["reviewed"] is False
    assert stat.S_IMODE(review.stat().st_mode) == 0o600


def test_trace_reordering_does_not_duplicate_reviewed_row(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    review = tmp_path / "review.jsonl"
    first = {"at": "2026-01-01", "prompt": "open Safari", "tool_calls": []}
    second = {"at": "2026-01-02", "prompt": "read note", "tool_calls": []}
    trace.write_text(json.dumps(first) + "\n")
    prepare(trace, review)
    row = json.loads(review.read_text())
    row["reviewed"] = True
    row["decision"] = "accept"
    review.write_text(json.dumps(row) + "\n")
    trace.write_text(json.dumps(second) + "\n" + json.dumps(first) + "\n")

    assert prepare(trace, review) == 2
    rows = [json.loads(line) for line in review.read_text().splitlines()]
    assert rows[0] == row
    assert len({item["id"] for item in rows}) == 2


def test_prepare_preserves_legacy_review_id(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    review = tmp_path / "review.jsonl"
    trace.write_text(json.dumps({
        "at": "2026-01-01", "prompt": "open Safari", "tool_calls": [],
    }) + "\n")
    legacy = {
        "id": "trace-legacy", "user": "open Safari", "proposed_tool": "none",
        "proposed_arguments": {}, "tool": "none", "arguments": {},
        "answer": "done", "reviewed": True, "decision": "accept",
    }
    review.write_text(json.dumps(legacy) + "\n")

    assert prepare(trace, review) == 1
    updated = json.loads(review.read_text())
    assert updated["id"] == "trace-legacy"
    assert updated["reviewed"] is True
    assert updated["source_fingerprint"]


def test_bad_review_label_does_not_replace_prior_export(tmp_path: Path) -> None:
    review = tmp_path / "review.jsonl"
    output = tmp_path / "examples.jsonl"
    output.write_text("previous export\n")
    review.write_text(json.dumps({
        "id": "bad", "user": "open Safari", "tool": "no_such_tool",
        "arguments": {}, "reviewed": True, "decision": "corrected",
    }) + "\n")

    with pytest.raises(ValueError, match="invalid tool"):
        export(review, output)

    assert output.read_text() == "previous export\n"


def test_reviewed_trace_requires_an_explicit_decision(tmp_path: Path) -> None:
    review = tmp_path / "review.jsonl"
    output = tmp_path / "examples.jsonl"
    review.write_text(json.dumps({
        "id": "trace-one", "user": "open Safari",
        "proposed_tool": "open_item", "proposed_arguments": {"target": "Safari"},
        "tool": "open_item", "arguments": {"target": "Safari"}, "reviewed": True,
        "decision": "",
    }) + "\n")

    with pytest.raises(ValueError, match="decision must be accept, corrected, or reject"):
        export(review, output)

    assert not output.exists()


def test_rejected_trace_is_reviewed_but_not_exported(tmp_path: Path) -> None:
    review = tmp_path / "review.jsonl"
    output = tmp_path / "examples.jsonl"
    review.write_text(json.dumps({
        "id": "unsafe-proposal", "user": "delete everything",
        "proposed_tool": "shell", "proposed_arguments": {"command": "rm -rf ~"},
        "tool": "shell", "arguments": {"command": "rm -rf ~"},
        "reviewed": True, "decision": "reject", "review_notes": "unsafe label",
    }) + "\n")

    assert export(review, output) == 0
    assert output.read_text() == ""


def test_accept_decision_must_match_model_proposal(tmp_path: Path) -> None:
    review = tmp_path / "review.jsonl"
    output = tmp_path / "examples.jsonl"
    review.write_text(json.dumps({
        "id": "trace-two", "user": "open Safari",
        "proposed_tool": "open_item", "proposed_arguments": {"target": "Safari"},
        "tool": "activate_app", "arguments": {"app": "Safari"}, "reviewed": True,
        "decision": "accept",
    }) + "\n")

    with pytest.raises(ValueError, match="accept decision does not match"):
        export(review, output)


def test_failed_trace_is_not_proposed_as_a_no_tool_label(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    review = tmp_path / "review.jsonl"
    output = tmp_path / "examples.jsonl"
    trace.write_text(json.dumps({
        "at": "2026-01-01", "prompt": "open Safari", "tool_calls": [],
        "reply": None, "error": "Ollama unavailable",
    }) + "\n")

    prepare(trace, review)

    row = json.loads(review.read_text())
    assert row["proposed_tool"] == ""
    assert row["source_error"] == "Ollama unavailable"
    row.update({
        "tool": "open_item", "arguments": {"target": "Safari"},
        "reviewed": True, "decision": "corrected",
    })
    review.write_text(json.dumps(row) + "\n")
    assert export(review, output) == 1
    assert json.loads(output.read_text())["tool"] == "open_item"


def test_reviewed_trace_preserves_optional_paraphrase_group(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    review = tmp_path / "review.jsonl"
    output = tmp_path / "examples.jsonl"
    trace.write_text(json.dumps({
        "at": "2026-01-01", "prompt": "open Safari",
        "tool_calls": [{"tool": "open_item", "arguments": {"target": "Safari"}}],
        "reply": "done", "error": None,
    }) + "\n")
    prepare(trace, review)
    row = json.loads(review.read_text())
    row.update({"reviewed": True, "decision": "accept", "group": "open-browser"})
    review.write_text(json.dumps(row) + "\n")

    assert export(review, output) == 1
    assert json.loads(output.read_text())["group"] == "open-browser"


def test_malformed_trace_does_not_replace_existing_review(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    review = tmp_path / "review.jsonl"
    original = json.dumps({
        "id": "kept", "user": "hello", "proposed_tool": "none",
        "proposed_arguments": {}, "tool": "none", "arguments": {},
        "answer": "hello", "reviewed": False,
    }) + "\n"
    review.write_text(original)
    trace.write_text(json.dumps({
        "prompt": "open Safari",
        "tool_calls": [{"tool": "open_item", "arguments": "not-an-object"}],
    }) + "\n")

    with pytest.raises(ValueError, match=r"trace .*:1: tool call arguments must be an object"):
        prepare(trace, review)

    assert review.read_text() == original
