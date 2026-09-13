from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model: str = "qwen2.5:1.5b-instruct"
    ollama_url: str = "http://127.0.0.1:11434"
    max_steps: int = 8
    speak: bool = False
    auto_approve: bool = False
    wake_phrase: str = "hey numnum"
    whisper_model: str = str(Path.home() / ".cache" / "nums" / "ggml-base.en.bin")
    capture_device: int = -1

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model=os.getenv("NUMS_MODEL", cls.model),
            ollama_url=os.getenv("NUMS_OLLAMA_URL", cls.ollama_url).rstrip("/"),
            max_steps=int(os.getenv("NUMS_MAX_STEPS", str(cls.max_steps))),
            speak=os.getenv("NUMS_SPEAK", "0") == "1",
            auto_approve=os.getenv("NUMS_AUTO_APPROVE", "0") == "1",
            wake_phrase=os.getenv("NUMS_WAKE_PHRASE", cls.wake_phrase),
            whisper_model=os.path.expanduser(
                os.getenv("NUMS_WHISPER_MODEL", cls.whisper_model)
            ),
            capture_device=int(os.getenv("NUMS_CAPTURE_DEVICE", str(cls.capture_device))),
        )
