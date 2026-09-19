"""Compare first tool calls against held-out labels without executing any tools."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def evaluate(data: Path, model: str, url: str) -> tuple[int, int, int]:
    total = tool_correct = arguments_correct = 0
    for line in data.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
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
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        calls = result.get("message", {}).get("tool_calls") or []
        function = calls[0].get("function", {}) if calls else {}
        actual = function.get("arguments", {})
        if isinstance(actual, str):
            try:
                actual = json.loads(actual)
            except json.JSONDecodeError:
                actual = {}
        if item["expected_tool"] == "none":
            name_ok = len(calls) == 0
            args_ok = name_ok
        else:
            name_ok = len(calls) == 1 and function.get("name") == item["expected_tool"]
            args_ok = name_ok and actual == item["expected_arguments"]
        total += 1
        tool_correct += name_ok
        arguments_correct += args_ok
        print(f"{item['id']}: tool={'ok' if name_ok else 'wrong'} args={'ok' if args_ok else 'wrong'}")
    return total, tool_correct, arguments_correct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    total, names, arguments = evaluate(args.data, args.model, args.url)
    if not total:
        raise SystemExit("empty evaluation set")
    print(f"Tool choice: {names}/{total}; tool and arguments: {arguments}/{total}")


if __name__ == "__main__":
    main()
