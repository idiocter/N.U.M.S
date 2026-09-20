from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings
from .history import HistoryStore
from .ollama import OllamaClient, OllamaError
from .tools import MacTools, TOOL_SCHEMAS


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
        saved = self.history_store.load() if self.history_store else []
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, *saved]

    def _save_history(self) -> None:
        if self.history_store:
            self.history_store.save(self.messages[1:])

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
        try:
            for _ in range(self.settings.max_steps):
                response = self.client.chat(self.messages, TOOL_SCHEMAS)
                message = response.get("message", {})
                self.messages.append(message)
                calls = message.get("tool_calls") or []
                if not calls:
                    self._save_history()
                    return message.get("content") or "I couldn't produce a response."
                for call in calls:
                    function = call.get("function", {})
                    name = function.get("name", "")
                    args = function.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    result = self.tools.execute(name, args)
                    self.messages.append({"role": "tool", "tool_name": name, "content": result})
        except OllamaError:
            if len(self.messages) == history_length + 1:
                self.messages.pop()
            self._save_history()
            raise
        self._save_history()
        return "I reached the tool-step limit. Try splitting the task into a smaller request."
