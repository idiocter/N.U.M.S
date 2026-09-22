import json
from pathlib import Path

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
    review.write_text(json.dumps(row) + "\n")
    assert export(review, output) == 1
    example = json.loads(output.read_text())
    assert example["tool"] == "open_item"
    assert "proposed_tool" not in example
