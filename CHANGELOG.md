# Changelog

## Unreleased

- Bounded the Ollama HTTP error-response body read so a large error payload cannot be pulled into memory.
- Hardened the Ollama reachability probe to treat any socket error (including DNS failures) as unreachable instead of aborting startup.
- Bounded the whisper-stream failure log tail to the final bytes so a large log is not fully read into memory.
- Added bounded retries for transient Ollama failures during held-out evaluation.
- Added private per-case checkpoints and resumable held-out model evaluation.
- Validated complete held-out datasets before sending any evaluation requests to Ollama.
- Validated trace JSON and tool-call structure before updating human review files.
- Preserved optional paraphrase groups from trace review into exported labels.
- Distinguished failed model traces from valid no-tool answers during label review.
- Required an explicit accept-or-correct review decision before traces can become training labels.
- Added optional paraphrase groups that stay together across train, validation, and test splits.
- Fingerprinted complete held-out evaluation inputs and rejected comparisons across changed prompts or schemas.
- Added meaningful command, notification, reminder, speech, script, and trash-path constraints.
- Added exact menu, accessibility role, label, and inspected-index constraints to UI selectors.
- Added nonempty path, safe destination, octal mode, and exact replacement constraints to file tools.
- Kept held-out evaluation running across Ollama request failures and blocked candidates with increased response errors.
- Fingerprinted split assignments so prompt or label edits cannot silently rewrite held-out cases.
- Enforced model-facing enum, pattern, and nonempty constraints in both runtime calls and training labels.
- Added accessibility roles, wait states, text targets, and shortcut formats to Mac UI tool schemas.
- Added exact states, ranges, and index guidance to Mac app and window tool schemas.
- Added model-facing value constraints for file paging and bounded Git inspection arguments.
- Fixed macOS wait scripts to compile without scripting additions, corrected disabled-control waits, and made notifications self-contained.
- Added a training coverage audit and a test requiring at least two labels for every runtime tool.
- Expanded model-adaptation labels for Git, clipboard, reminders, calendar, and no-action contrasts.
- Expanded model-adaptation labels for project inspection and safe file operations.
- Expanded model-adaptation labels for exact Mac menus, controls, text, and keyboard actions.
- Expanded model-adaptation labels for exact Mac app activation and window management.
- Expanded model-adaptation labels for Mac app discovery, inspection, and asynchronous waits.
- Added verified, non-overwriting symbolic link creation with exact relative-target preservation.
- Added verified standard permission changes with strict octal validation and symlink refusal.
- Added exact UTF-8 file appends with parent creation and explicit disk flushes.
- Added bounded line-level Git blame with validated revision targeting.
- Added read-only local and remote Git branch inspection with upstream details.
- Added read-only inspection of one validated Git revision and optional path patch.
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
