# NUMS

NUMS is a private, local-first macOS assistant powered by Qwen through Ollama. It can inspect files, search the Mac, run commands, open apps and URLs, use the clipboard, read today's calendar, create reminders, speak, show notifications, and automate apps with AppleScript.

The model and chat stay on the Mac through Ollama. Model-requested file writes, Trash operations, AppleScript, shell commands, and app actions execute immediately without confirmation prompts.

## Quick start

Install Ollama once, then run NUMS from the project directory:

```bash
cd ~/Documents/Projects/NUMS
uv run nums --pull       # download Qwen once
uv run nums              # interactive text mode
uv run nums --voice      # wake phrase voice mode
```

`uv run` syncs the Python environment automatically. NUMS starts the local
Ollama server if it is not running and leaves it available after NUMS exits.
Ollama startup errors are written to `~/Library/Logs/NUMS-Ollama.log`.
Text mode checks that the configured model is installed before accepting a
request and points to `uv run nums --pull` when it is missing.

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
`NUMS_MAX_CONTEXT_CHARS` also removes oldest complete turns when the saved
conversation grows beyond its character budget (default: 60000).
When loading an older history file, NUMS skips an unfinished final turn so a
disconnected tool call does not affect the next request.
If the model disconnects after a tool has run, NUMS records the partial work
and reports that the final result could not be confirmed.
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

Install the microphone runtime once:

```bash
brew install whisper-cpp
```

Start voice mode:

```bash
uv run nums --voice
```

The first voice launch downloads the English Whisper model automatically if it
is missing or the cached file is incomplete. Downloads have a time limit and
keep a partial file so the next attempt can resume. `uv run nums --setup-voice`
can download it ahead of time.
`--wake` remains an alias for `--voice`. Plain `uv run nums` starts text mode.
Voice mode checks that Qwen can load before it opens the microphone; a model
load failure appears immediately in the terminal.

To start the listener automatically at login and restart it after failures:

```bash
uv run nums --print-service       # inspect the LaunchAgent first
uv run nums --install-service
uv run nums --uninstall-service   # stop it and remove the plist
```

The service writes output to `~/Library/Logs/NUMS.log` and uses the Python
environment and NUMS settings from which it was installed. Reinstall the
service after changing those settings or the microphone runtime path.
If starting a replacement service fails, NUMS restores the previous plist.

Say “hey numnum” and wait for the chime. NUMS stays online after the first command, so you can continue talking to it without repeating the wake phrase. You can also say both together, such as “hey numnum, open Safari.”
The wake phrase must begin a spoken turn; mentioning it later in a sentence
does not wake NUMS.

When you are finished, say **“aight baby girl, let’s sleep.”** NUMS says good night and returns to wake-only mode. The first launch may trigger a macOS Microphone permission prompt for your terminal.
Say a sleep phrase on its own turn so a request that merely mentions one is
handled as a request.

An active voice session returns to sleep after five idle minutes. Change this
with `NUMS_SESSION_TIMEOUT`, and provide custom sleep phrases separated by `|`
through `NUMS_SLEEP_PHRASES`.

NUMS uses silence-based voice activity detection so it transcribes complete phrases instead of chopping them at fixed intervals. Each spoken turn uses a fresh microphone stream: NUMS closes it before answering and starts a clean one when the response finishes, so its own voice cannot become the next command.

If the wrong microphone is selected, list the capture devices shown when the listener starts and set its number:

```bash
NUMS_CAPTURE_DEVICE=0 uv run nums --voice
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
the output. File reads include the next character offset so NUMS can request
the remaining text. Directory listings include a next item offset when more
entries are available. Ask for a narrower path or search when that is easier.
File searches can use an optional glob, such as `*.py`, to narrow results.

## Coding tasks

NUMS can inspect a project, make small code changes, and review them with Git.
Give it the project path and a specific request, for example:

```bash
uv run nums "In /absolute/path/to/project, fix the failing parser test, run that test, and show the Git diff"
```

For a focused edit, `replace_in_file` replaces one exact text span and refuses
missing or repeated matches. `git_status` shows changed and untracked files;
`git_diff` shows tracked changes against `HEAD`. In the default `unrestricted`
mode NUMS can run project checks through `shell`. In `standard` mode it can
inspect and edit files but cannot run shell checks; `read_only` can inspect
the project and Git state without editing. Its coding instructions call for
reporting which checks ran and which were skipped.

`find_files` discovers project files recursively, honors Git ignore rules, and
supports a glob and result pages. `read_lines` returns numbered code lines and
a next line for longer files. Long individual lines are clipped; use
`read_file` with an offset when their full contents matter. Long command
results retain both the start and end, where failure details often appear.

These tools support coding workflows, but the bundled 1.5B model has not been
validated as a reliable autonomous coding agent. The tool-use training fixtures
are synthetic; no coding-specific model training or live coding benchmark has
been completed. Review important changes and their test results before use.

## App control

Text and voice requests use the same Mac tools. NUMS can list running apps,
wait for, list, raise, and close an exact named window, inspect pages of its accessibility roles and labels, inspect
menus and submenus, click an exact control or menu item, type into an inspected
text field or the focused control, select an exact option from an inspected
popup control, and press keys or shortcuts.
It can replace or clear the complete value of an exact inspected text control
instead of appending at an uncertain cursor position.
It can wait up to 30 seconds for an exact labeled control to appear, disappear,
become enabled, or become disabled, which makes asynchronous workflows more reliable.
Menu workflows can also wait for an exact item to appear or become enabled.
Inspection shows disabled controls and menu items, plus exposed values for toggles,
radio buttons, popup controls, sliders, disclosure controls, and steppers.
Focused-control inspection reports which element currently receives keyboard input.
For checkboxes, switches, and disclosure triangles, `set_app_toggle` requests
an explicit on or off state and checks the result.
`select_app_radio` selects an exact radio option and leaves an already selected
option unchanged.
Sliders and steppers can be adjusted in bounded accessibility steps, with the
before and after values returned for verification.
For example, ask `uv run nums "Inspect the front window of Safari"` before
asking it to use a named control. Use the inspected element index when labels
repeat or when a control has no label. UI inspection returns 40 elements by
default; use `offset` for the next page or `limit` (up to 120) for a larger page.
Clicks fail when the front window changes or a target is ambiguous.
Targeted typing also stops if the selected field does not receive focus.
Window titles are listed with indexes so repeated titles can be selected
precisely. Exact window closing verifies that the window disappeared and reports
when a confirmation dialog blocks it. Keyboard actions support Home, End, Page Up, Page Down, Enter, and
Forward Delete alongside the existing keys and modifiers.

`list_running_apps`, `list_app_windows`, `wait_for_app_window`, `inspect_app_ui`,
`inspect_focused_app_element`, `wait_for_app_element`,
and `inspect_app_menu` are available in read-only mode.
Clicking, typing, and key presses require the default `unrestricted` mode.
Popup selection also requires `unrestricted` mode and rejects ambiguous,
missing, or disabled controls and options.
macOS must allow the process running NUMS to control System Events and use
Accessibility. These tools use the app's accessibility information; controls
with neither a useful role nor a stable inspected index may need app-specific AppleScript.
NUMS does not interpret screenshots or click arbitrary screen coordinates.

For broad access, open **System Settings → Privacy & Security** and grant your terminal only the permissions you actually want it to have, such as:

- Full Disk Access for protected files
- Automation for controlling specific apps with AppleScript
- Accessibility for UI automation scripts you choose to run

macOS permissions remain the outer security boundary. NUMS cannot and should not bypass them.

## Unrestricted execution

NUMS applies no application-level action policy or approval step. Its effective access is the access granted to the Terminal process that launches it. macOS privacy permissions remain the operating-system boundary.

You can reduce that access with `NUMS_ACTION_MODE=standard` (blocks shell,
AppleScript, UI actions, and Trash) or `NUMS_ACTION_MODE=read_only` (inspection
and information tools only). The default remains `unrestricted` for
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
`--doctor` also checks whether Ollama can actually produce that tool call,
so a model listed by Ollama but unable to load is reported as a failure. It
also makes a read-only System Events query to check Mac app UI access, and
returns a failure if that check cannot run.
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
