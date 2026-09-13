from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model: str = "qwen2.5:1.5b-instruct"
    ollama_url: str = "http://127.0.0.1:11434"
    max_steps: int = 8
    speak: bool = False
    auto_approve: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model=os.getenv("NUMS_MODEL", cls.model),
            ollama_url=os.getenv("NUMS_OLLAMA_URL", cls.ollama_url).rstrip("/"),
            max_steps=int(os.getenv("NUMS_MAX_STEPS", str(cls.max_steps))),
            speak=os.getenv("NUMS_SPEAK", "0") == "1",
            auto_approve=os.getenv("NUMS_AUTO_APPROVE", "0") == "1",
        )
