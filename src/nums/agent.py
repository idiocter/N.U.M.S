from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings
from .history import HistoryStore
from .ollama import OllamaClient, OllamaError
from .tools import MacTools, available_tool_schemas
from .trace import TraceStore


SYSTEM_PROMPT = """You are NUMS, Bipul's private local macOS assistant.
Be concise, capable, and honest. Use tools when they provide evidence or complete the task.
Use only the provided tools. Execute requested actions directly within the selected action mode.
"""

class Agent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OllamaClient(
            settings.ollama_url, settings.model, timeout=settings.ollama_timeout_seconds
        )
        self.tools = MacTools(settings.action_mode)
        self.history_store = HistoryStore(Path(settings.history_file)) if settings.history_file else None
        self.trace_store = TraceStore(Path(settings.trace_file)) if settings.trace_file else None
        saved = self.history_store.load() if self.history_store else []
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, *saved]
        self.last_run: dict[str, Any] = {
            "status": "idle", "steps": 0, "tool_calls": 0, "tool_errors": 0
        }

    def _save_history(self) -> None:
        if self.history_store:
            self.history_store.save(self.messages[1:])

    def _trace(self, prompt: str, calls: list[dict[str, Any]], reply: str | None, error: str | None = None) -> None:
        if self.trace_store:
            self.trace_store.append(prompt, calls, reply, error)

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._save_history()

    def _trim_history(self) -> None:
        starts = [i for i, message in enumerate(self.messages) if message.get("role") == "user"]
        previous_turns = self.settings.history_turns - 1
        if len(starts) > previous_turns:
            cutoff = starts[-previous_turns] if previous_turns else len(self.messages)
            self.messages = self.messages[:1] + self.messages[cutoff:]

    def run(self, prompt: str) -> str:
        self._trim_history()
        self.last_run = {"status": "running", "steps": 0, "tool_calls": 0, "tool_errors": 0}
        history_length = len(self.messages)
        self.messages.append({"role": "user", "content": prompt})
        trace_calls: list[dict[str, Any]] = []
        last_signature: str | None = None
        last_result: str | None = None
        unchanged_count = 0
        try:
            for step in range(1, self.settings.max_steps + 1):
                self.last_run["steps"] = step
                response = self.client.chat(
                    self.messages, available_tool_schemas(self.settings.action_mode)
                )
                message = response.get("message", {})
                self.messages.append(message)
                calls = message.get("tool_calls") or []
                if not calls:
                    self.last_run["status"] = "completed"
                    reply = message.get("content") or "I couldn't produce a response."
                    message["content"] = reply
                    self._save_history()
                    self._trace(prompt, trace_calls, reply)
                    return reply
                if len(calls) > self.settings.max_tool_calls - self.last_run["tool_calls"]:
                    self.messages.pop()
                    self.last_run["status"] = "tool_limit"
                    reply = "I stopped because this request exceeded the tool-call limit. Try a smaller request."
                    self.messages.append({"role": "assistant", "content": reply})
                    self._save_history()
                    self._trace(prompt, trace_calls, reply, "tool-call limit")
                    return reply
                for call in calls:
                    function = call.get("function", {})
                    name = function.get("name", "")
                    args = function.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    trace_calls.append({"tool": name, "arguments": args})
                    self.last_run["tool_calls"] += 1
                    signature = json.dumps([name, args], sort_keys=True, default=str)
                    if signature == last_signature and unchanged_count >= self.settings.repeat_tool_limit:
                        result = json.dumps({
                            "error": "Repeated tool call stopped because it made no observable progress",
                            "tool": name,
                        })
                        self.messages.append({"role": "tool", "tool_name": name, "content": result})
                        self.last_run["tool_errors"] += 1
                        self.last_run["status"] = "no_progress"
                        reply = f"I stopped after repeating the same {name} action without progress."
                        self.messages.append({"role": "assistant", "content": reply})
                        self._save_history()
                        self._trace(prompt, trace_calls, reply, "repeated tool call")
                        return reply
                    result = self.tools.execute(name, args)
                    if signature == last_signature and result == last_result:
                        unchanged_count += 1
                    else:
                        unchanged_count = 1
                    last_signature, last_result = signature, result
                    try:
                        parsed_result = json.loads(result)
                        if isinstance(parsed_result, dict) and "error" in parsed_result:
                            self.last_run["tool_errors"] += 1
                    except (json.JSONDecodeError, TypeError):
                        pass
                    self.messages.append({"role": "tool", "tool_name": name, "content": result})
        except OllamaError as exc:
            self.last_run["status"] = "model_error"
            del self.messages[history_length:]
            self._save_history()
            self._trace(prompt, trace_calls, None, str(exc))
            raise
        self.last_run["status"] = "step_limit"
        reply = "I reached the tool-step limit. Try splitting the task into a smaller request."
        self.messages.append({"role": "assistant", "content": reply})
        self._save_history()
        self._trace(prompt, trace_calls, reply)
        return reply
