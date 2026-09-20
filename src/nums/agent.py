from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings
from .history import HistoryStore
from .ollama import OllamaClient, OllamaError
from .tools import MacTools, TOOL_SCHEMAS
from .trace import TraceStore


SYSTEM_PROMPT = """You are NUMS, Bipul's private local macOS assistant.
Be concise, capable, and honest. Use tools when they provide evidence or complete the task.
You have unrestricted access to the provided tools. Execute requested actions directly.
"""

class Agent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OllamaClient(settings.ollama_url, settings.model)
        self.tools = MacTools()
        self.history_store = HistoryStore(Path(settings.history_file)) if settings.history_file else None
        self.trace_store = TraceStore(Path(settings.trace_file)) if settings.trace_file else None
        saved = self.history_store.load() if self.history_store else []
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, *saved]

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
        history_length = len(self.messages)
        self.messages.append({"role": "user", "content": prompt})
        trace_calls: list[dict[str, Any]] = []
        try:
            for _ in range(self.settings.max_steps):
                response = self.client.chat(self.messages, TOOL_SCHEMAS)
                message = response.get("message", {})
                self.messages.append(message)
                calls = message.get("tool_calls") or []
                if not calls:
                    self._save_history()
                    reply = message.get("content") or "I couldn't produce a response."
                    self._trace(prompt, trace_calls, reply)
                    return reply
                for call in calls:
                    function = call.get("function", {})
                    name = function.get("name", "")
                    args = function.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    trace_calls.append({"tool": name, "arguments": args})
                    result = self.tools.execute(name, args)
                    self.messages.append({"role": "tool", "tool_name": name, "content": result})
        except OllamaError as exc:
            if len(self.messages) == history_length + 1:
                self.messages.pop()
            self._save_history()
            self._trace(prompt, trace_calls, None, str(exc))
            raise
        self._save_history()
        reply = "I reached the tool-step limit. Try splitting the task into a smaller request."
        self._trace(prompt, trace_calls, reply)
        return reply
