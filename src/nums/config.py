from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


def _integer(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    model: str = "qwen2.5:1.5b-instruct"
    ollama_url: str = "http://127.0.0.1:11434"
    max_steps: int = 8
    history_turns: int = 8
    speak: bool = False
    wake_phrase: str = "hey numnum"
    whisper_model: str = str(Path.home() / ".cache" / "nums" / "ggml-base.en.bin")
    capture_device: int = -1

    @classmethod
    def from_env(cls) -> "Settings":
        model = os.getenv("NUMS_MODEL", cls.model).strip()
        url = os.getenv("NUMS_OLLAMA_URL", cls.ollama_url).rstrip("/")
        phrase = os.getenv("NUMS_WAKE_PHRASE", cls.wake_phrase).strip()
        if not model:
            raise ValueError("NUMS_MODEL cannot be empty")
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("NUMS_OLLAMA_URL must be an HTTP URL")
        if not phrase:
            raise ValueError("NUMS_WAKE_PHRASE cannot be empty")
        speak = os.getenv("NUMS_SPEAK", "0")
        if speak not in {"0", "1"}:
            raise ValueError("NUMS_SPEAK must be 0 or 1")
        return cls(
            model=model,
            ollama_url=url,
            max_steps=_integer("NUMS_MAX_STEPS", cls.max_steps, 1),
            history_turns=_integer("NUMS_HISTORY_TURNS", cls.history_turns, 1),
            speak=speak == "1",
            wake_phrase=phrase,
            whisper_model=os.path.expanduser(
                os.getenv("NUMS_WHISPER_MODEL", cls.whisper_model)
            ),
            capture_device=_integer("NUMS_CAPTURE_DEVICE", cls.capture_device, -1),
        )
