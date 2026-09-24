from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 180) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload = json.dumps(
            {"model": self.model, "messages": messages, "tools": tools, "stream": False}
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:1000]
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve`."
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OllamaError("Ollama returned an invalid JSON response") from exc
        if not isinstance(result, dict) or not isinstance(result.get("message"), dict):
            raise OllamaError("Ollama returned a response without an assistant message")
        message = result["message"]
        if message.get("content") is not None and not isinstance(message["content"], str):
            raise OllamaError("Ollama returned invalid assistant content")
        calls = message.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list) or any(
                not isinstance(call, dict)
                or not isinstance(call.get("function"), dict)
                or not isinstance(call["function"].get("name"), str)
                or not call["function"]["name"]
                or not isinstance(call["function"].get("arguments"), (dict, str))
                for call in calls
            ):
                raise OllamaError("Ollama returned malformed tool calls")
        return result

    def has_model(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, UnicodeDecodeError):
            return False
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            return False
        names = {
            model.get("name") for model in payload["models"]
            if isinstance(model, dict) and isinstance(model.get("name"), str)
        }
        return self.model in names or f"{self.model}:latest" in names
