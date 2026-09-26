# NUMS

NUMS is a private, local-first macOS assistant powered by Qwen through Ollama. It can inspect files, search the Mac, run commands, open apps and URLs, use the clipboard, read today's calendar, create reminders, speak, show notifications, and automate apps with AppleScript.

The model and chat stay on the Mac through Ollama. Model-requested file writes, Trash operations, AppleScript, shell commands, and app actions execute immediately without confirmation prompts.

## Quick start

```bash
cd ~/Documents/Projects/NUMS
ollama serve  # keep this running, or start the Ollama app
```

In another terminal:

```bash
cd ~/Documents/Projects/NUMS
uv sync
uv run nums --pull
uv run nums
```

One-shot requests work too:

```bash
uv run nums "summarize the files on my Desktop"
uv run nums --speak "tell me the current system information"
```

NUMS normally keeps conversation history in memory for one run. To resume a
conversation after restarting, opt in to a private local history file:

```bash
NUMS_HISTORY_FILE=~/.local/share/nums/history.json uv run nums
```

This file includes prompts, model replies, and tool results. NUMS writes it
atomically with owner-only permissions. `NUMS_HISTORY_TURNS` controls how many
recent turns reach the model on each request (default: 8).
When loading an older history file, NUMS skips an unfinished final turn so a
disconnected tool call does not affect the next request.
`NUMS_REPEAT_TOOL_LIMIT` controls how many identical tool calls a request may
make before NUMS stops the no-progress loop (default: 2).
`NUMS_MAX_TOOL_CALLS` caps the total proposed tool actions in one request
(default: 16), including batches returned in one model response.

In the interactive terminal, `/status` shows the current model and history
location plus the last run's steps, tool calls, and errors. `/reset` clears the
conversation (including the optional history file), and `/help` lists the commands.
If optional trace writing fails, `/status` reports the trace error while the
assistant still returns its response.

## Wake phrase

NUMS can stay asleep until you say **“hey numnum.”** Voice recognition and the Qwen response both run locally.

Install the microphone runtime and download the English base model once:

```bash
brew install whisper-cpp
uv run nums --setup-voice
```

Start the listener:

```bash
uv run nums --wake
```

To start the listener automatically at login and restart it after failures:

```bash
uv run nums --print-service       # inspect the LaunchAgent first
uv run nums --install-service
uv run nums --uninstall-service   # stop it and remove the plist
```

The service writes output to `~/Library/Logs/NUMS.log` and uses the Python
environment and NUMS settings from which it was installed. Reinstall the
service after changing those settings or the microphone runtime path.

Say “hey numnum” and wait for the chime. NUMS stays online after the first command, so you can continue talking to it without repeating the wake phrase. You can also say both together, such as “hey numnum, open Safari.”

When you are finished, say **“aight baby girl, let’s sleep.”** NUMS says good night and returns to wake-only mode. The first launch may trigger a macOS Microphone permission prompt for your terminal.

An active voice session returns to sleep after five idle minutes. Change this
with `NUMS_SESSION_TIMEOUT`, and provide custom sleep phrases separated by `|`
through `NUMS_SLEEP_PHRASES`.

NUMS uses silence-based voice activity detection so it transcribes complete phrases instead of chopping them at fixed intervals. Each spoken turn uses a fresh microphone stream: NUMS closes it before answering and starts a clean one when the response finishes, so its own voice cannot become the next command.

If the wrong microphone is selected, list the capture devices shown when the listener starts and set its number:

```bash
NUMS_CAPTURE_DEVICE=0 uv run nums --wake
```

The default model is `qwen2.5:1.5b-instruct`, a compact model close to the requested 2B size. Override it with `NUMS_MODEL`, for example:

```bash
NUMS_MODEL=qwen2.5:3b-instruct uv run nums --pull
NUMS_MODEL=qwen2.5:3b-instruct uv run nums
```

Set `NUMS_OLLAMA_TIMEOUT` to change the maximum seconds for one model response
(default: 180). The login service keeps the value set when it is installed.

## Mac access

NUMS runs with the permissions of the Terminal app that launches it. macOS may ask for access when NUMS first touches protected locations or automates another app.

Large file, directory, and command results indicate when NUMS has truncated
the output. Ask for a narrower path or search when the omitted part matters.

For broad access, open **System Settings → Privacy & Security** and grant your terminal only the permissions you actually want it to have, such as:

- Full Disk Access for protected files
- Automation for controlling specific apps with AppleScript
- Accessibility for UI automation scripts you choose to run

macOS permissions remain the outer security boundary. NUMS cannot and should not bypass them.

## Unrestricted execution

NUMS applies no application-level action policy or approval step. Its effective access is the access granted to the Terminal process that launches it. macOS privacy permissions remain the operating-system boundary.

You can reduce that access with `NUMS_ACTION_MODE=standard` (blocks shell,
AppleScript, and Trash) or `NUMS_ACTION_MODE=read_only` (only file inspection,
search, and system information). The default remains `unrestricted` for
compatibility with existing NUMS behavior.

## Diagnostics

```bash
uv run nums --doctor
uv run nums --self-test
uv run nums --voice-test
uv run pytest
```

`--self-test` uses a temporary directory and asks the local model for a
`system_info` tool call. It does not execute the model's proposed action or
modify user files.
`--voice-test` listens for one sentence for up to 15 seconds and prints the
transcript, providing the final manual check for microphone selection and
recognition quality.

GitHub Actions runs the test suite on macOS with Python 3.11, 3.12, and 3.13
and verifies that the package builds. Release notes are tracked in
[`CHANGELOG.md`](CHANGELOG.md).

## Tool-use fine tuning

The [training workflow](training/README.md) validates labeled tool calls, creates
train/validation/test splits, and compares model tool choices without executing
the requested actions. The included examples are synthetic pipeline fixtures;
collect representative reviewed examples before choosing a trained model for NUMS.
To collect real tool decisions for review, opt in with
`NUMS_TRACE_FILE=~/.local/share/nums/usage.nums-trace.jsonl`.

## Architecture

- `OllamaClient` talks only to the local Ollama HTTP API.
- `Agent` runs the tool-use loop and keeps conversation context in memory.
- `MacTools` provides files, shell, apps, speech, notifications, and AppleScript.
- Tool calls execute directly through `MacTools`.
- `WhisperStream` listens locally and activates the agent only after the wake phrase.

## Current limits

- Wake listening requires the `whisper-cpp` Homebrew package and Microphone permission.
- A 1.5B model is fast and private but may need simple, explicit requests for long workflows.
- Conversation history persists only when `NUMS_HISTORY_FILE` is set.
