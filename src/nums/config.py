from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .policy import VALID_MODES


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
    ollama_timeout_seconds: int = 180
    max_steps: int = 8
    history_turns: int = 8
    history_file: str | None = None
    trace_file: str | None = None
    action_mode: str = "unrestricted"
    repeat_tool_limit: int = 2
    session_timeout_seconds: int = 300
    sleep_phrases: tuple[str, ...] = ("go to sleep", "good night nums", "aight baby girl lets sleep")
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
        try:
            valid_port = parsed_url.port is None or 1 <= parsed_url.port <= 65535
        except ValueError:
            valid_port = False
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
            or not valid_port
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.path
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError("NUMS_OLLAMA_URL must be an HTTP URL with only a host and optional port")
        if not phrase:
            raise ValueError("NUMS_WAKE_PHRASE cannot be empty")
        speak = os.getenv("NUMS_SPEAK", "0")
        if speak not in {"0", "1"}:
            raise ValueError("NUMS_SPEAK must be 0 or 1")
        action_mode = os.getenv("NUMS_ACTION_MODE", cls.action_mode).strip().lower()
        if action_mode not in VALID_MODES:
            raise ValueError("NUMS_ACTION_MODE must be read_only, standard, or unrestricted")
        sleep_phrases = tuple(
            phrase.strip() for phrase in os.getenv(
                "NUMS_SLEEP_PHRASES", "|".join(cls.sleep_phrases)
            ).split("|") if phrase.strip()
        )
        if not sleep_phrases:
            raise ValueError("NUMS_SLEEP_PHRASES must include at least one phrase")
        return cls(
            model=model,
            ollama_url=url,
            ollama_timeout_seconds=_integer(
                "NUMS_OLLAMA_TIMEOUT", cls.ollama_timeout_seconds, 1
            ),
            max_steps=_integer("NUMS_MAX_STEPS", cls.max_steps, 1),
            history_turns=_integer("NUMS_HISTORY_TURNS", cls.history_turns, 1),
            history_file=os.path.expanduser(os.environ["NUMS_HISTORY_FILE"])
            if os.getenv("NUMS_HISTORY_FILE") else None,
            trace_file=os.path.expanduser(os.environ["NUMS_TRACE_FILE"])
            if os.getenv("NUMS_TRACE_FILE") else None,
            action_mode=action_mode,
            repeat_tool_limit=_integer("NUMS_REPEAT_TOOL_LIMIT", cls.repeat_tool_limit, 1),
            session_timeout_seconds=_integer(
                "NUMS_SESSION_TIMEOUT", cls.session_timeout_seconds, 1
            ),
            sleep_phrases=sleep_phrases,
            speak=speak == "1",
            wake_phrase=phrase,
            whisper_model=os.path.expanduser(
                os.getenv("NUMS_WHISPER_MODEL", cls.whisper_model)
            ),
            capture_device=_integer("NUMS_CAPTURE_DEVICE", cls.capture_device, -1),
        )
