"""Compare first tool calls against held-out labels without executing any tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}


def _is_transient_request_error(exc: BaseException) -> bool:
    return not isinstance(exc, urllib.error.HTTPError) or exc.code in RETRYABLE_HTTP_STATUS


def _case_fingerprint(item: dict[str, Any]) -> str:
    protected = {
        "messages": item["messages"][:-1],
        "tools": item["tools"],
        "expected_tool": item["expected_tool"],
        "expected_arguments": item["expected_arguments"],
    }
    return hashlib.sha256(
        json.dumps(protected, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _dataset_digest(cases: list[dict[str, Any]]) -> str:
    identities = sorted(
        (item["id"], item.get("case_fingerprint") or _case_fingerprint(item))
        for item in cases
    )
    return hashlib.sha256(
        json.dumps(identities, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def load_cases(data: Path) -> list[dict[str, Any]]:
    cases = []
    seen = set()
    for number, line in enumerate(data.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("case must be a JSON object")
            identifier = item.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("id must be a nonempty string")
            if identifier in seen:
                raise ValueError(f"duplicate id: {identifier}")
            if not isinstance(item.get("messages"), list) or len(item["messages"]) < 2:
                raise ValueError("messages must contain the prompt and expected reply")
            if not all(isinstance(message, dict) for message in item["messages"]):
                raise ValueError("every message must be an object")
            roles = [message.get("role") for message in item["messages"]]
            if any(role not in {"system", "user", "assistant", "tool"} for role in roles):
                raise ValueError("every message needs a recognized role")
            if roles[-1] != "assistant":
                raise ValueError("the expected final message must be from the assistant")
            if "user" not in roles[:-1]:
                raise ValueError("messages must include a user prompt before the expected reply")
            for message in item["messages"]:
                if "content" in message and not isinstance(message["content"], str):
                    raise ValueError("message content must be a string when provided")
                if "tool_calls" in message and not isinstance(message["tool_calls"], list):
                    raise ValueError("message tool_calls must be a list when provided")
            if not isinstance(item.get("tools"), list):
                raise ValueError("tools must be a list")
            if not isinstance(item.get("expected_tool"), str) or not item["expected_tool"]:
                raise ValueError("expected_tool must be a nonempty string")
            if not isinstance(item.get("expected_arguments"), dict):
                raise ValueError("expected_arguments must be an object")
            tool_names: set[str] = set()
            for tool in item["tools"]:
                if not isinstance(tool, dict) or tool.get("type") != "function":
                    raise ValueError("every tool must be a function schema")
                function = tool.get("function")
                if not isinstance(function, dict) or not isinstance(
                    function.get("name"), str
                ) or not function["name"]:
                    raise ValueError("every tool schema needs a nonempty function name")
                if not isinstance(function.get("parameters"), dict):
                    raise ValueError("every tool schema needs an object parameters schema")
                if function["name"] in tool_names:
                    raise ValueError(f"duplicate tool schema: {function['name']}")
                tool_names.add(function["name"])
            if item["expected_tool"] != "none" and item["expected_tool"] not in tool_names:
                raise ValueError("expected_tool is not present in the offered tools")
            expected_message = item["messages"][-1]
            expected_calls = expected_message.get("tool_calls") or []
            if item["expected_tool"] == "none":
                if expected_calls:
                    raise ValueError("a no-tool label cannot have an expected tool call")
            else:
                if len(expected_calls) != 1 or not isinstance(expected_calls[0], dict):
                    raise ValueError("the expected assistant reply must contain one tool call")
                expected_function = expected_calls[0].get("function")
                if not isinstance(expected_function, dict):
                    raise ValueError("the expected tool call needs a function")
                labeled_arguments = expected_function.get("arguments")
                if isinstance(labeled_arguments, str):
                    try:
                        labeled_arguments = json.loads(labeled_arguments)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "expected tool call arguments must contain valid JSON"
                        ) from exc
                if (
                    expected_function.get("name") != item["expected_tool"]
                    or labeled_arguments != item["expected_arguments"]
                ):
                    raise ValueError("expected fields do not match the assistant tool call")
            fingerprint = item.get("case_fingerprint")
            if fingerprint is not None and (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
            ):
                raise ValueError("case_fingerprint must be a lowercase SHA-256 digest")
            if fingerprint is not None and fingerprint != _case_fingerprint(item):
                raise ValueError("case_fingerprint does not match the evaluation input")
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{data}:{number}: {exc}") from exc
        seen.add(identifier)
        cases.append(item)
    return cases


def _build_report(
    model: str, cases: list[dict[str, Any]], dataset_digest: str
) -> dict[str, Any]:
    per_tool: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "tool_correct": 0, "arguments_correct": 0}
    )
    for case in cases:
        scores = per_tool[case["expected_tool"]]
        scores["total"] += 1
        scores["tool_correct"] += int(case["tool_correct"])
        scores["arguments_correct"] += int(case["arguments_correct"])
    total = len(cases)
    tool_correct = sum(int(case["tool_correct"]) for case in cases)
    arguments_correct = sum(int(case["arguments_correct"]) for case in cases)
    return {
        "report_version": 1,
        "model": model,
        "dataset_digest": dataset_digest,
        "total": total,
        "tool_correct": tool_correct,
        "arguments_correct": arguments_correct,
        "tool_accuracy": tool_correct / total if total else 0,
        "arguments_accuracy": arguments_correct / total if total else 0,
        "by_expected_tool": dict(sorted(per_tool.items())),
        "cases": cases,
    }


def _resume_cases(
    prior: dict[str, Any] | None, model: str, dataset_digest: str
) -> dict[str, dict[str, Any]]:
    if prior is None:
        return {}
    if prior.get("model") != model:
        raise ValueError("resume report model does not match the requested model")
    if prior.get("dataset_digest") not in {None, dataset_digest}:
        raise ValueError("resume report dataset does not match the evaluation set")
    rows = prior.get("cases")
    if not isinstance(rows, list):
        raise ValueError("resume report must contain cases")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("resume report contains an invalid case")
        if row["id"] in result:
            raise ValueError(f"resume report contains duplicate id: {row['id']}")
        result[row["id"]] = row
    return result


def evaluate(
    data: Path, model: str, url: str, prior: dict[str, Any] | None = None,
    checkpoint: Path | None = None, retries: int = 0,
) -> dict[str, Any]:
    if not 0 <= retries <= 3:
        raise ValueError("retries must be between 0 and 3")
    cases: list[dict[str, Any]] = []
    inputs = load_cases(data)
    dataset_digest = _dataset_digest(inputs)
    resumed = _resume_cases(prior, model, dataset_digest)
    orphaned = set(resumed) - {item["id"] for item in inputs}
    if orphaned:
        identifiers = ", ".join(sorted(orphaned))
        raise ValueError(f"resume report contains cases absent from the evaluation set: {identifiers}")
    for item in inputs:
        previous = resumed.get(item["id"])
        if previous is not None and not previous.get("error"):
            if previous.get("case_fingerprint") != item.get("case_fingerprint"):
                raise ValueError(f"resume fingerprint changed for {item['id']}")
            if (
                previous.get("expected_tool") != item["expected_tool"]
                or previous.get("expected_arguments") != item["expected_arguments"]
                or not isinstance(previous.get("tool_correct"), bool)
                or not isinstance(previous.get("arguments_correct"), bool)
            ):
                raise ValueError(f"resume label or scores changed for {item['id']}")
            cases.append(previous)
            print(f"{item['id']}: resumed")
            if checkpoint:
                write_report(checkpoint, _build_report(model, cases, dataset_digest))
            continue
        payload = json.dumps({
            "model": model,
            "messages": item["messages"][:-1],
            "tools": item["tools"],
            "stream": False,
            "options": {"temperature": 0},
        }).encode()
        request = urllib.request.Request(
            f"{url.rstrip('/')}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.load(response)
                    if not isinstance(result, dict) or not isinstance(result.get("message"), dict):
                        raise ValueError("missing assistant message")
                    calls = result["message"].get("tool_calls") or []
                    if not isinstance(calls, list) or any(
                        not isinstance(call, dict) or not isinstance(call.get("function"), dict)
                        for call in calls
                    ):
                        raise ValueError("malformed tool calls")
                    function = calls[0]["function"] if calls else {}
                    actual = function.get("arguments", {})
                    if isinstance(actual, str):
                        actual = json.loads(actual)
                    if not isinstance(actual, dict):
                        raise ValueError("tool arguments are not a JSON object")
                    error = None
                break
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                calls = []
                function = {}
                actual = None
                error = f"Invalid model response: {exc}"
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                calls = []
                function = {}
                actual = None
                error = f"Evaluation request failed: {type(exc).__name__}: {exc}"
                if attempt < retries and _is_transient_request_error(exc):
                    print(f"{item['id']}: transient request failure; retry {attempt + 1}/{retries}")
                    continue
                break
        if error:
            name_ok = args_ok = False
        elif item["expected_tool"] == "none":
            name_ok = len(calls) == 0
            args_ok = name_ok
        else:
            name_ok = len(calls) == 1 and function.get("name") == item["expected_tool"]
            args_ok = name_ok and actual == item["expected_arguments"]
        cases.append(
            {
                "id": item["id"],
                "case_fingerprint": item.get("case_fingerprint"),
                "expected_tool": item["expected_tool"],
                "actual_tool": function.get("name") if calls else (None if error else "none"),
                "expected_arguments": item["expected_arguments"],
                "actual_arguments": actual,
                "tool_correct": name_ok,
                "arguments_correct": args_ok,
                "error": error,
            }
        )
        print(
            f"{item['id']}: tool={'ok' if name_ok else 'wrong'} "
            f"args={'ok' if args_ok else 'wrong'}" + (f" ({error})" if error else "")
        )
        if checkpoint:
            write_report(checkpoint, _build_report(model, cases, dataset_digest))
    return _build_report(model, cases, dataset_digest)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--json-out", type=Path, help="Write a detailed JSON report")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse completed cases in --json-out and checkpoint after every case",
    )
    parser.add_argument(
        "--retries", type=int, choices=range(4), default=0,
        help="Retry each transient Ollama request up to 3 times",
    )
    args = parser.parse_args()
    if args.resume and args.json_out is None:
        parser.error("--resume requires --json-out")
    prior = None
    if args.resume and args.json_out.exists():
        prior = json.loads(args.json_out.read_text())
    report = evaluate(
        args.data, args.model, args.url, prior=prior,
        checkpoint=args.json_out if args.resume else None, retries=args.retries,
    )
    if not report["total"]:
        raise SystemExit("empty evaluation set")
    if args.json_out:
        write_report(args.json_out, report)
    print(
        f"Tool choice: {report['tool_correct']}/{report['total']}; "
        f"tool and arguments: {report['arguments_correct']}/{report['total']}"
    )


if __name__ == "__main__":
    main()
