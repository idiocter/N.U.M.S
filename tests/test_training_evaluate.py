import json
import hashlib
import stat
import urllib.error
from contextlib import nullcontext

import pytest

from training.evaluate import evaluate, load_cases, write_report


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": "test", "parameters": {"type": "object"}},
    }


def _assistant_call(name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [{"function": {"name": name, "arguments": json.dumps(arguments)}}],
    }


def _fingerprint(case: dict) -> str:
    protected = {
        "messages": case["messages"][:-1],
        "tools": case["tools"],
        "expected_tool": case["expected_tool"],
        "expected_arguments": case["expected_arguments"],
    }
    return hashlib.sha256(
        json.dumps(protected, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def test_evaluate_reports_per_tool_results(monkeypatch, tmp_path) -> None:
    examples = [
        {
            "id": "right",
            "messages": [
                {"role": "user", "content": "status"},
                _assistant_call("system_info", {}),
            ],
            "tools": [_tool("system_info")],
            "expected_tool": "system_info",
            "expected_arguments": {},
        },
        {
            "id": "wrong",
            "messages": [
                {"role": "user", "content": "open notes"},
                _assistant_call("open_app", {"name": "Notes"}),
            ],
            "tools": [_tool("open_app")],
            "expected_tool": "open_app",
            "expected_arguments": {"name": "Notes"},
        },
    ]
    examples[0]["case_fingerprint"] = _fingerprint(examples[0])
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
    assert report["report_version"] == 1
    assert len(report["dataset_digest"]) == 64
    assert report["cases"][0]["case_fingerprint"] == examples[0]["case_fingerprint"]
    assert report["by_expected_tool"]["open_app"] == {
        "total": 1,
        "tool_correct": 1,
        "arguments_correct": 0,
    }
    assert report["cases"][1]["actual_tool"] == "open_app"
    assert report["cases"][1]["actual_arguments"] == {"name": "Mail"}
    assert report["cases"][1]["attempts"] == 1


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
        "id": "list-args",
        "messages": [{"role": "user"}, _assistant_call("system_info", {})],
        "tools": [_tool("system_info")],
        "expected_tool": "system_info", "expected_arguments": {},
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


@pytest.mark.parametrize(
    ("tools", "expected_tool", "error"),
    [
        ([{"type": "command", "function": {}}], "none", "must be a function schema"),
        ([{"type": "function", "function": {"name": "read_file"}}], "none", "parameters schema"),
        ([_tool("read_file"), _tool("read_file")], "read_file", "duplicate tool schema"),
        ([_tool("read_file")], "write_file", "not present in the offered tools"),
    ],
)
def test_evaluation_input_validates_offered_tools(
    tmp_path, tools, expected_tool, error
) -> None:
    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps({
        "id": "bad-tools", "messages": [{"role": "user"}, {"role": "assistant"}],
        "tools": tools, "expected_tool": expected_tool, "expected_arguments": {},
    }) + "\n")

    with pytest.raises(ValueError, match=error):
        load_cases(data)


@pytest.mark.parametrize(
    ("expected_tool", "expected_arguments", "assistant", "error"),
    [
        ("none", {}, _assistant_call("read_file", {"path": "/tmp/a"}), "no-tool label"),
        ("read_file", {"path": "/tmp/a"}, {"role": "assistant"}, "contain one tool call"),
        (
            "read_file", {"path": "/tmp/a"},
            _assistant_call("read_file", {"path": "/tmp/b"}), "do not match",
        ),
    ],
)
def test_evaluation_input_validates_expected_assistant_label(
    tmp_path, expected_tool, expected_arguments, assistant, error
) -> None:
    data = tmp_path / "test.jsonl"
    tools = [] if expected_tool == "none" else [_tool(expected_tool)]
    data.write_text(json.dumps({
        "id": "bad-label", "messages": [{"role": "user"}, assistant],
        "tools": tools, "expected_tool": expected_tool,
        "expected_arguments": expected_arguments,
    }) + "\n")

    with pytest.raises(ValueError, match=error):
        load_cases(data)


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        ([{"role": "developer"}, {"role": "assistant"}], "recognized role"),
        ([{"role": "user"}, {"role": "user"}], "final message must be from the assistant"),
        ([{"role": "system"}, {"role": "assistant"}], "include a user prompt"),
        ([{"role": "user", "content": 42}, {"role": "assistant"}], "content must be a string"),
    ],
)
def test_evaluation_input_validates_message_structure(tmp_path, messages, error) -> None:
    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps({
        "id": "bad-messages", "messages": messages, "tools": [],
        "expected_tool": "none", "expected_arguments": {},
    }) + "\n")

    with pytest.raises(ValueError, match=error):
        load_cases(data)


def test_evaluation_input_validates_case_fingerprint(tmp_path) -> None:
    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps({
        "id": "bad-fingerprint", "case_fingerprint": "ABC",
        "messages": [{"role": "user"}, {"role": "assistant"}], "tools": [],
        "expected_tool": "none", "expected_arguments": {},
    }) + "\n")

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_cases(data)


def test_evaluation_input_rejects_stale_case_fingerprint(tmp_path) -> None:
    data = tmp_path / "test.jsonl"
    case = {
        "id": "changed", "messages": [{"role": "user"}, {"role": "assistant"}],
        "tools": [], "expected_tool": "none", "expected_arguments": {},
    }
    case["case_fingerprint"] = _fingerprint(case)
    case["messages"][0]["content"] = "changed after the split was built"
    data.write_text(json.dumps(case) + "\n")

    with pytest.raises(ValueError, match="does not match the evaluation input"):
        load_cases(data)


def test_evaluation_resumes_completed_cases_and_checkpoints(monkeypatch, tmp_path) -> None:
    examples = [
        {"id": "done",
         "messages": [{"role": "user"}, {"role": "assistant"}], "tools": [],
         "expected_tool": "none", "expected_arguments": {}},
        {"id": "pending",
         "messages": [{"role": "user"}, {"role": "assistant"}], "tools": [],
         "expected_tool": "none", "expected_arguments": {}},
    ]
    for example in examples:
        example["case_fingerprint"] = _fingerprint(example)
    data = tmp_path / "test.jsonl"
    data.write_text("\n".join(json.dumps(item) for item in examples))
    prior_case = {
        "id": "done", "case_fingerprint": examples[0]["case_fingerprint"],
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


def test_resume_rejects_cases_removed_from_evaluation_set(tmp_path) -> None:
    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps({
        "id": "current", "messages": [{"role": "user"}, {"role": "assistant"}],
        "tools": [], "expected_tool": "none", "expected_arguments": {},
    }) + "\n")
    prior = {
        "model": "test-model",
        "cases": [{
            "id": "removed", "expected_tool": "none", "expected_arguments": {},
            "tool_correct": True, "arguments_correct": True, "error": None,
        }],
    }

    with pytest.raises(ValueError, match="cases absent.*removed"):
        evaluate(data, "test-model", "http://ollama.test", prior=prior)


def test_resume_rejects_a_different_dataset_digest(tmp_path) -> None:
    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps({
        "id": "current", "messages": [{"role": "user"}, {"role": "assistant"}],
        "tools": [], "expected_tool": "none", "expected_arguments": {},
    }) + "\n")

    with pytest.raises(ValueError, match="dataset does not match"):
        evaluate(data, "test-model", "http://ollama.test", prior={
            "model": "test-model", "dataset_digest": "a" * 64, "cases": [],
        })


def test_transient_request_failure_is_retried(monkeypatch, tmp_path) -> None:
    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps({
        "id": "retry", "messages": [{"role": "user"}, {"role": "assistant"}],
        "tools": [], "expected_tool": "none", "expected_arguments": {},
    }) + "\n")
    responses = iter([
        urllib.error.URLError("temporarily unavailable"),
        _JsonResponse({"message": {"content": "hello"}}),
    ])
    attempts = []

    def open_next(*args, **kwargs):
        attempts.append(1)
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return nullcontext(response)

    monkeypatch.setattr("training.evaluate.urllib.request.urlopen", open_next)

    report = evaluate(data, "test-model", "http://ollama.test", retries=1)

    assert len(attempts) == 2
    assert report["tool_correct"] == 1
    assert report["cases"][0]["error"] is None
    assert report["cases"][0]["attempts"] == 2


@pytest.mark.parametrize("status", [400, 401, 404])
def test_non_transient_http_failure_is_not_retried(monkeypatch, tmp_path, status) -> None:
    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps({
        "id": "bad-request", "messages": [{"role": "user"}, {"role": "assistant"}],
        "tools": [], "expected_tool": "none", "expected_arguments": {},
    }) + "\n")
    attempts = []

    def fail_request(*args, **kwargs):
        attempts.append(1)
        raise urllib.error.HTTPError(
            "http://ollama.test/api/chat", status, "request rejected", None, None
        )

    monkeypatch.setattr("training.evaluate.urllib.request.urlopen", fail_request)

    report = evaluate(data, "test-model", "http://ollama.test", retries=3)

    assert len(attempts) == 1
    assert report["cases"][0]["tool_correct"] is False
    assert report["cases"][0]["attempts"] == 1


def test_retryable_http_failure_is_retried(monkeypatch, tmp_path) -> None:
    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps({
        "id": "busy", "messages": [{"role": "user"}, {"role": "assistant"}],
        "tools": [], "expected_tool": "none", "expected_arguments": {},
    }) + "\n")
    responses = iter([
        urllib.error.HTTPError(
            "http://ollama.test/api/chat", 503, "temporarily unavailable", None, None
        ),
        _JsonResponse({"message": {"content": "hello"}}),
    ])
    attempts = []

    def open_next(*args, **kwargs):
        attempts.append(1)
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return nullcontext(response)

    monkeypatch.setattr("training.evaluate.urllib.request.urlopen", open_next)

    report = evaluate(data, "test-model", "http://ollama.test", retries=1)

    assert len(attempts) == 2
    assert report["tool_correct"] == 1


def test_evaluation_retries_are_bounded(tmp_path) -> None:
    data = tmp_path / "empty.jsonl"
    data.write_text("")

    with pytest.raises(ValueError, match="retries must be between 0 and 3"):
        evaluate(data, "test-model", "http://ollama.test", retries=4)


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


def test_resume_with_corrupt_checkpoint_exits_with_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from training import evaluate as evaluate_module

    report = tmp_path / "report.json"
    report.write_text("{truncated")
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate", str(tmp_path / "data.jsonl"), "--model", "qwen", "--json-out", str(report), "--resume"],
    )
    monkeypatch.setattr(
        evaluate_module, "evaluate", lambda *args, **kwargs: pytest.fail("evaluated")
    )

    with pytest.raises(SystemExit) as excinfo:
        evaluate_module.main()

    assert "Expecting" in str(excinfo.value) or "property name" in str(excinfo.value)


def test_oversized_model_response_is_recorded_as_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    case = {
        "id": "right",
        "messages": [
            {"role": "user", "content": "status"},
            _assistant_call("system_info", {}),
        ],
        "tools": [_tool("system_info")],
        "expected_tool": "system_info",
        "expected_arguments": {},
    }
    data = tmp_path / "cases.jsonl"
    data.write_text(json.dumps(case) + "\n")
    monkeypatch.setattr("training.evaluate.RESPONSE_MAX_BYTES", 16)
    monkeypatch.setattr(
        "training.evaluate.urllib.request.urlopen",
        lambda *args, **kwargs: nullcontext(_JsonResponse({"pad": "x" * 100})),
    )

    report = evaluate(data, "qwen", "http://127.0.0.1:11434")

    assert "byte limit" in report["cases"][0]["error"]
    assert report["tool_correct"] == 0


def test_truncated_http_response_is_retried(monkeypatch, tmp_path) -> None:
    import http.client

    data = tmp_path / "test.jsonl"
    data.write_text(json.dumps({
        "id": "cut", "messages": [{"role": "user"}, {"role": "assistant"}],
        "tools": [], "expected_tool": "none", "expected_arguments": {},
    }) + "\n")
    responses = iter([
        http.client.IncompleteRead(b"partial"),
        _JsonResponse({"message": {"content": "hello"}}),
    ])

    def open_next(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return nullcontext(response)

    monkeypatch.setattr("training.evaluate.urllib.request.urlopen", open_next)

    report = evaluate(data, "test-model", "http://ollama.test", retries=1)

    assert report["cases"][0]["error"] is None
    assert report["cases"][0]["attempts"] == 2


def test_report_path_cannot_overwrite_the_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from training import evaluate as evaluate_module

    data = tmp_path / "test.jsonl"
    data.write_text("original labels\n")
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate", str(data), "--model", "qwen", "--json-out", str(tmp_path / "." / "test.jsonl")],
    )
    monkeypatch.setattr(
        evaluate_module, "evaluate", lambda *args, **kwargs: pytest.fail("evaluated")
    )

    with pytest.raises(SystemExit) as excinfo:
        evaluate_module.main()

    assert excinfo.value.code == 2
    assert data.read_text() == "original labels\n"


def _base_case() -> dict:
    return {
        "id": "c1",
        "messages": [
            {"role": "user", "content": "status"},
            _assistant_call("system_info", {}),
        ],
        "tools": [_tool("system_info")],
        "expected_tool": "system_info",
        "expected_arguments": {},
    }


def _mutated(**changes: object) -> dict:
    case = _base_case()
    case.update(changes)
    return case


def _nested(path: list, value: object) -> dict:
    case = _base_case()
    target = case
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return case


INVALID_CASES = [
    (_mutated(id=""), "id must be a nonempty string"),
    (_mutated(id=5), "id must be a nonempty string"),
    (_mutated(messages=[{"role": "user", "content": "x"}]), "prompt and expected reply"),
    (_mutated(messages=[{"role": "user"}, "oops"]), "every message must be an object"),
    (_mutated(messages=[{"role": "user"}, {"role": "robot"}]), "recognized role"),
    (_mutated(messages=[{"role": "assistant"}, {"role": "user"}]), "must be from the assistant"),
    (_mutated(messages=[{"role": "system"}, _assistant_call("system_info", {})]), "must include a user prompt"),
    (_nested(["messages", 0, "content"], 5), "content must be a string"),
    (_nested(["messages", 0, "tool_calls"], "x"), "tool_calls must be a list"),
    (_mutated(tools="nope"), "tools must be a list"),
    (_mutated(expected_tool=""), "expected_tool must be a nonempty string"),
    (_mutated(expected_arguments=[]), "expected_arguments must be an object"),
    (_mutated(tools=["x"]), "every tool must be a function schema"),
    (_mutated(tools=[{"type": "function", "function": {"name": "", "parameters": {}}}]), "nonempty function name"),
    (_mutated(tools=[{"type": "function", "function": {"name": "system_info"}}]), "object parameters schema"),
    (_mutated(tools=[_tool("system_info"), _tool("system_info")]), "duplicate tool schema: system_info"),
    (_mutated(expected_tool="other"), "not present in the offered tools"),
    (
        _mutated(expected_tool="none", expected_arguments={}),
        "no-tool label cannot have an expected tool call",
    ),
    (
        _nested(["messages", 1], {"role": "assistant", "tool_calls": []}),
        "must contain one tool call",
    ),
    (
        _nested(["messages", 1], {"role": "assistant", "tool_calls": [{"id": "x"}]}),
        "needs a function",
    ),
    (
        _nested(["messages", 1], {"role": "assistant", "tool_calls": [
            {"function": {"name": "system_info", "arguments": "{broken"}}
        ]}),
        "valid JSON",
    ),
    (_mutated(expected_arguments={"extra": "1"}), "expected fields do not match"),
    (_mutated(case_fingerprint="ABC"), "lowercase SHA-256 digest"),
    (_mutated(case_fingerprint="0" * 64), "does not match the evaluation input"),
]


@pytest.mark.parametrize(("case", "message"), INVALID_CASES)
def test_invalid_held_out_cases_are_rejected_with_their_line(tmp_path, case: dict, message: str) -> None:
    data = tmp_path / "cases.jsonl"
    data.write_text(json.dumps(_base_case()) .replace('"c1"', '"first"') + "\n" + json.dumps(case) + "\n")

    with pytest.raises(ValueError, match=rf"cases\.jsonl:2: .*{message}"):
        load_cases(data)


def test_non_object_duplicate_and_blank_lines_are_handled(tmp_path) -> None:
    data = tmp_path / "cases.jsonl"
    data.write_text("\n[1, 2]\n")
    with pytest.raises(ValueError, match=r":2: case must be a JSON object"):
        load_cases(data)

    data.write_text(json.dumps(_base_case()) + "\n\n" + json.dumps(_base_case()) + "\n")
    with pytest.raises(ValueError, match=r":3: duplicate id: c1"):
        load_cases(data)

    data.write_text("\n" + json.dumps(_base_case()) + "\n\n")
    assert [case["id"] for case in load_cases(data)] == ["c1"]


def test_a_no_tool_expectation_is_accepted(tmp_path) -> None:
    case = _mutated(
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello."}],
        expected_tool="none", expected_arguments={},
    )
    data = tmp_path / "cases.jsonl"
    data.write_text(json.dumps(case) + "\n")

    assert load_cases(data)[0]["expected_tool"] == "none"
