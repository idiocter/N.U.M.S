import json
from contextlib import nullcontext

from training.evaluate import evaluate


def test_evaluate_reports_per_tool_results(monkeypatch, tmp_path) -> None:
    examples = [
        {
            "id": "right",
            "messages": [{"role": "user", "content": "status"}, {"role": "assistant"}],
            "tools": [],
            "expected_tool": "system_info",
            "expected_arguments": {},
        },
        {
            "id": "wrong",
            "messages": [{"role": "user", "content": "open notes"}, {"role": "assistant"}],
            "tools": [],
            "expected_tool": "open_app",
            "expected_arguments": {"name": "Notes"},
        },
    ]
    data = tmp_path / "test.jsonl"
    data.write_text("\n".join(json.dumps(item) for item in examples))
    responses = iter(
        [
            {"message": {"tool_calls": [{"function": {"name": "system_info", "arguments": {}}}]}},
            {"message": {"tool_calls": [{"function": {"name": "open_app", "arguments": {"name": "Mail"}}}]}},
        ]
    )

    monkeypatch.setattr(
        "training.evaluate.urllib.request.urlopen",
        lambda *args, **kwargs: nullcontext(_JsonResponse(next(responses))),
    )

    report = evaluate(data, "test-model", "http://ollama.test")

    assert report["tool_correct"] == 2
    assert report["arguments_correct"] == 1
    assert report["tool_accuracy"] == 1
    assert report["arguments_accuracy"] == 0.5
    assert report["by_expected_tool"]["open_app"] == {
        "total": 1,
        "tool_correct": 1,
        "arguments_correct": 0,
    }
    assert report["cases"][1]["actual_tool"] == "open_app"


class _JsonResponse:
    def __init__(self, value) -> None:
        self.value = value

    def read(self, *args, **kwargs) -> bytes:
        return json.dumps(self.value).encode()
