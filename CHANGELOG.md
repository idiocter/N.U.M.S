# Changelog

## Unreleased

- Added bounded-memory SHA-256 file hashing in read-only mode.
- Added bounded read-only Git history inspection with optional path scoping.
- Added verified non-overwriting moves and renames for files, directories, and symlinks.
- Added non-overwriting file, directory, and symlink copies with recursive-copy protection.
- Added idempotent directory creation with parent creation and symlink refusal.
- Added read-only path metadata inspection with explicit symlink reporting.
- Added exact accessibility context-menu opening for inspected Mac UI elements.
- Added verified full-screen state control for exact Mac windows that expose it.
- Added exact window resizing with bounded dimensions and final-size checks.
- Added exact window movement with bounded multi-display coordinates and final-position checks.
- Added idempotent minimize and restore control for exact Mac windows.
- Added exact activation of running Mac apps with foreground verification.
- Expanded window listings with main/minimized state, position, and size.
- Extended menu waits to confirm items becoming disabled or absent.
- Extended exact window waits to confirm that windows and dialogs disappeared.
- Added precise window closing with repeated-title indexes and confirmation-dialog detection.
- Added bounded, verified adjustments for inspected Mac sliders and steppers.
- Extended explicit state control to expandable Mac disclosure controls.
- Added idempotent selection of exact Mac radio button options.
- Added exact text replacement and clearing for inspected Mac text controls.
- Added read-only inspection of the currently focused Mac UI element.
- Show exposed values for popup, slider, disclosure, and stepper controls during UI inspection.
- Extended UI waits to detect controls becoming absent or disabled.
- Added bounded waits for exact menu and submenu items to appear or become enabled.
- Added exact, verified option selection for inspected Mac popup controls.
- Added bounded waits for exact Mac windows and dialogs before later automation steps.
- Added bounded waits for exact Mac accessibility controls to appear or become enabled during asynchronous app workflows.
- Added window listing and precise window raising, explicit checkbox and switch state control, and more Mac navigation keys.
- Improved UI action checks, app opening, notifications, and reminder feedback.
- Improved voice model validation and download recovery; reduced accidental wake and sleep triggers; checked text model availability at startup.
- Added unlabeled-control targeting by inspected index, smaller configurable UI pages, and a read-only Mac app UI check in `--doctor`.
- Show UI control availability and toggle state, reject ambiguous or disabled menu actions, and verify text-field focus before typing.
- Paginated UI inspection; indexed control clicks; menu and submenu inspection and clicks; targeted text-field typing; actionable Mac permission errors.
- Added general Mac app controls through Accessibility: inspect labeled UI elements, click exact matches, type text, and press keys or shortcuts.
- Added `nums --voice` for wake mode, with automatic Whisper model setup and an Ollama model check; plain `nums` remains text mode.
- NUMS starts a local Ollama server when needed for either mode.
- Added recursive project file discovery and numbered code reads.
- Kept both ends of long command output and partial output from timed-out commands.
- Made `--doctor` probe a real model tool call before reporting the model operational.

## 0.2.0 - 2026-09-22

- Added configurable execution modes and repeated-tool loop detection.
- Added optional persistent history and reviewed tool traces.
- Added clipboard, reminder, and calendar tools.
- Added configurable voice session timeouts and sleep phrases.
- Added macOS LaunchAgent installation and removal.
- Added deterministic training data preparation and trace review workflows.
- Improved atomic writes, diagnostics, Ollama failures, and voice reliability.

## 0.1.0

- Initial local Qwen assistant, Mac tools, CLI, and wake phrase support.
