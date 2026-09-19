from __future__ import annotations

import json
from typing import Any

from .config import Settings
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
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run(self, prompt: str) -> str:
        history_length = len(self.messages)
        self.messages.append({"role": "user", "content": prompt})
        try:
            for _ in range(self.settings.max_steps):
                response = self.client.chat(self.messages, TOOL_SCHEMAS)
                message = response.get("message", {})
                self.messages.append(message)
                calls = message.get("tool_calls") or []
                if not calls:
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
            raise
        return "I reached the tool-step limit. Try splitting the task into a smaller request."
