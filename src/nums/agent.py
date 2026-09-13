from __future__ import annotations

import json
from typing import Any, Callable

from .config import Settings
from .ollama import OllamaClient
from .policy import classify
from .tools import MacTools, TOOL_SCHEMAS


SYSTEM_PROMPT = """You are NUMS, Bipul's private local macOS assistant.
Be concise, capable, and honest. Use tools when they provide evidence or complete the task.
Never claim a tool action succeeded until its result confirms success.
Prefer narrow tools over shell. Do not seek credentials, browser secrets, keychains, or private tokens.
For multi-step work, inspect first, act second, and verify at the end.
The host application asks the user before consequential actions; do not evade that check.
"""


Confirm = Callable[[str, dict[str, Any], str], bool]


class Agent:
    def __init__(self, settings: Settings, confirm: Confirm) -> None:
        self.settings = settings
        self.confirm = confirm
        self.client = OllamaClient(settings.ollama_url, settings.model)
        self.tools = MacTools()
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
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
                decision = classify(name, args)
                approved = self.settings.auto_approve or not decision.needs_confirmation
                if decision.needs_confirmation and not approved:
                    approved = self.confirm(name, args, decision.reason)
                result = self.tools.execute(name, args) if approved else json.dumps({"denied": True})
                self.messages.append({"role": "tool", "tool_name": name, "content": result})
        return "I reached the tool-step limit. Try splitting the task into a smaller request."
