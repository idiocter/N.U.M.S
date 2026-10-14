import json
import stat
import urllib.error
from contextlib import nullcontext

import pytest

from training.evaluate import evaluate, load_cases, write_report


def test_evaluate_reports_per_tool_results(monkeypatch, tmp_path) -> None:
    examples = [
        {
            "id": "right",
            "case_fingerprint": "a" * 64,
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
    assert report["cases"][0]["case_fingerprint"] == "a" * 64
    assert report["by_expected_tool"]["open_app"] == {
        "total": 1,
        "tool_correct": 1,
        "arguments_correct": 0,
    }
    assert report["cases"][1]["actual_tool"] == "open_app"
    assert report["cases"][1]["actual_arguments"] == {"name": "Mail"}


def test_malformed_model_response_scores_wrong_and_keeps_next_case(monkeypatch, tmp_path) -> None:
    examples = [
        {"id": "invalid", "messages": [{"role": "user"}, {"role": "assistant"}],
         "tools": [], "expected_tool": "none", "expected_arguments": {}},
        {"id": "valid", "messages": [{"role": "user"}, {"role": "assistant"}],
         "tools": [], "expected_tool": "none", "expected_arguments": {}},
    ]
    data = tmp_path / "test.jsonl"
    data.write_text("\n".join(json.dumps(item) for item in examples))
    responses = iter([{}, {"message": {"content": "hello"}}])
    monkeypatch.setattr(
        "training.evaluate.urllib.request.urlopen",
        lambda *args, **kwargs: nullcontext(_JsonResponse(next(responses))),
    )

    report = evaluate(data, "test-model", "http://ollama.test")

    assert report["total"] == 2
    assert report["tool_correct"] == 1
    assert "missing assistant message" in report["cases"][0]["error"]
    assert report["cases"][1]["tool_correct"] is True


def test_request_failure_scores_wrong_and_keeps_next_case(monkeypatch, tmp_path) -> None:
    examples = [
        {"id": "offline", "messages": [{"role": "user"}, {"role": "assistant"}],
         "tools": [], "expected_tool": "none", "expected_arguments": {}},
        {"id": "valid", "messages": [{"role": "user"}, {"role": "assistant"}],
         "tools": [], "expected_tool": "none", "expected_arguments": {}},
    ]
    data = tmp_path / "test.jsonl"
    data.write_text("\n".join(json.dumps(item) for item in examples))
    responses = iter([urllib.error.URLError("Ollama unavailable"), _JsonResponse({
        "message": {"content": "hello"},
    })])

    def open_next(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return nullcontext(response)

    monkeypatch.setattr("training.evaluate.urllib.request.urlopen", open_next)

    report = evaluate(data, "test-model", "http://ollama.test")

    assert report["total"] == 2
    assert report["tool_correct"] == 1
    assert "Evaluation request failed" in report["cases"][0]["error"]
    assert report["cases"][1]["tool_correct"] is True


def test_non_object_tool_arguments_are_a_model_response_error(monkeypatch, tmp_path) -> None:
    example = {
        "id": "list-args", "messages": [{"role": "user"}, {"role": "assistant"}],
        "tools": [], "expected_tool": "system_info", "expected_arguments": {},
    }
    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps(example) + "\n")
    response = _JsonResponse({"message": {"tool_calls": [{
        "function": {"name": "system_info", "arguments": "[]"},
    }]}})
    monkeypatch.setattr(
        "training.evaluate.urllib.request.urlopen",
        lambda *args, **kwargs: nullcontext(response),
    )

    report = evaluate(data, "test-model", "http://ollama.test")

    assert report["tool_correct"] == 0
    assert "not a JSON object" in report["cases"][0]["error"]


def test_evaluation_input_rejects_duplicate_ids_before_requests(monkeypatch, tmp_path) -> None:
    case = {
        "id": "duplicate", "messages": [{"role": "user"}, {"role": "assistant"}],
        "tools": [], "expected_tool": "none", "expected_arguments": {},
    }
    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps(case) + "\n" + json.dumps(case) + "\n")
    monkeypatch.setattr(
        "training.evaluate.urllib.request.urlopen",
        lambda *args, **kwargs: pytest.fail("invalid evaluation data reached Ollama"),
    )

    with pytest.raises(ValueError, match="duplicate id"):
        evaluate(data, "test-model", "http://ollama.test")


def test_evaluation_input_validates_case_fingerprint(tmp_path) -> None:
    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps({
        "id": "bad-fingerprint", "case_fingerprint": "ABC",
        "messages": [{"role": "user"}, {"role": "assistant"}], "tools": [],
        "expected_tool": "none", "expected_arguments": {},
    }) + "\n")

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_cases(data)


def test_evaluation_resumes_completed_cases_and_checkpoints(monkeypatch, tmp_path) -> None:
    examples = [
        {"id": "done", "case_fingerprint": "a" * 64,
         "messages": [{"role": "user"}, {"role": "assistant"}], "tools": [],
         "expected_tool": "none", "expected_arguments": {}},
        {"id": "pending", "case_fingerprint": "b" * 64,
         "messages": [{"role": "user"}, {"role": "assistant"}], "tools": [],
         "expected_tool": "none", "expected_arguments": {}},
    ]
    data = tmp_path / "test.jsonl"
    data.write_text("\n".join(json.dumps(item) for item in examples))
    prior_case = {
        "id": "done", "case_fingerprint": "a" * 64,
        "expected_tool": "none", "actual_tool": "none",
        "expected_arguments": {}, "actual_arguments": {},
        "tool_correct": True, "arguments_correct": True, "error": None,
    }
    requests = []
    monkeypatch.setattr(
        "training.evaluate.urllib.request.urlopen",
        lambda request, **kwargs: requests.append(request) or nullcontext(
            _JsonResponse({"message": {"content": "hello"}})
        ),
    )
    checkpoint = tmp_path / "runs" / "resume.json"

    report = evaluate(
        data, "test-model", "http://ollama.test",
        prior={"model": "test-model", "cases": [prior_case]}, checkpoint=checkpoint,
    )

    assert len(requests) == 1
    assert [case["id"] for case in report["cases"]] == ["done", "pending"]
    assert json.loads(checkpoint.read_text())["total"] == 2
    assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o600


def test_resume_rejects_a_different_model(tmp_path) -> None:
    data = tmp_path / "empty.jsonl"
    data.write_text("")

    with pytest.raises(ValueError, match="model does not match"):
        evaluate(data, "candidate", "http://ollama.test", prior={
            "model": "baseline", "cases": [],
        })


def test_evaluation_report_is_private(tmp_path) -> None:
    output = tmp_path / "runs" / "report.json"
    write_report(output, {"cases": [{"expected_arguments": {"path": "/private"}}]})

    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text())["cases"][0]["expected_arguments"]["path"] == "/private"


class _JsonResponse:
    def __init__(self, value) -> None:
        self.value = value

    def read(self, *args, **kwargs) -> bytes:
        return json.dumps(self.value).encode()
