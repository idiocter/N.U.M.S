from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from .agent import Agent
from .config import Settings


LOGO = r"""
 _   _ _   _ __  __ ____
| \ | | | | |  \/  / ___|
|  \| | | | | |\/| \___ \
| |\  | |_| | |  | |___) |
|_| \_|\___/|_|  |_|____/
"""


def confirm(name: str, args: dict[str, object], reason: str) -> bool:
    print(f"\nNUMS wants to run: {name}\nReason: {reason}\nArguments: {args}")
    return input("Allow once? [y/N] ").strip().lower() in {"y", "yes"}


def doctor(settings: Settings) -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Ollama: {shutil.which('ollama') or 'not found'}")
    print(f"Model: {settings.model}")
    client = Agent(settings, confirm).client
    print(f"Model ready: {'yes' if client.has_model() else 'no'}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="NUMS local macOS assistant")
    parser.add_argument("prompt", nargs="*", help="Run one request and exit")
    parser.add_argument("--doctor", action="store_true", help="Check Ollama and model availability")
    parser.add_argument("--pull", action="store_true", help="Download the configured local model")
    parser.add_argument("--speak", action="store_true", help="Read responses aloud")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.pull:
        raise SystemExit(subprocess.call(["ollama", "pull", settings.model]))
    if args.doctor:
        raise SystemExit(doctor(settings))

    agent = Agent(settings, confirm)
    speak = settings.speak or args.speak
    if args.prompt:
        response = agent.run(" ".join(args.prompt))
        print(response)
        if speak:
            subprocess.run(["say", response], check=False)
        return

    print(LOGO)
    print(f"Local model: {settings.model} | Type /quit to exit\n")
    while True:
        try:
            prompt = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt in {"/quit", "/exit"}:
            break
        response = agent.run(prompt)
        print(f"NUMS > {response}\n")
        if speak:
            subprocess.run(["say", response], check=False)


if __name__ == "__main__":
    main()
