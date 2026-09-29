"""AppleScript commands for accessibility-based Mac app interaction."""

from __future__ import annotations


LIST_APPS_SCRIPT = """tell application "System Events"
    get name of every process whose background only is false
end tell"""


INSPECT_APP_SCRIPT = """on run argv
    set appName to item 1 of argv
    tell application "System Events"
        if not (exists process appName) then error "App is not running: " & appName
        tell process appName
            if not (exists window 1) then error "App has no open window: " & appName
            set targetWindow to window 1
            set windowName to name of targetWindow
            set entries to {"Window: " & windowName}
            set elements to entire contents of targetWindow
            set totalCount to count of elements
            set limitCount to totalCount
            if limitCount > 120 then set limitCount to 120
            repeat with elementIndex from 1 to limitCount
                set elementRef to item elementIndex of elements
                set roleText to ""
                set nameText to ""
                set descriptionText to ""
                try
                    set roleText to role of elementRef as text
                end try
                try
                    set nameText to name of elementRef as text
                end try
                try
                    set descriptionText to accessibility description of elementRef as text
                end try
                if roleText is not "" or nameText is not "" or descriptionText is not "" then
                    set end of entries to (elementIndex as text) & tab & roleText & tab & nameText & tab & descriptionText
                end if
            end repeat
            if totalCount > limitCount then set end of entries to "More elements exist; inspect a narrower window or use app-specific scripting."
        end tell
    end tell
    set AppleScript's text item delimiters to linefeed
    return entries as text
end run"""


CLICK_ELEMENT_SCRIPT = """on run argv
    set appName to item 1 of argv
    set targetWindow to item 2 of argv
    set targetRole to item 3 of argv
    set targetLabel to item 4 of argv
    tell application "System Events"
        if not (exists process appName) then error "App is not running: " & appName
        tell process appName
            if not (exists window 1) then error "App has no open window: " & appName
            if (name of window 1) is not targetWindow then error "Front window changed; inspect the app again"
            set frontmost to true
            set matches to {}
            repeat with elementRef in entire contents of window 1
                set roleText to ""
                set nameText to ""
                set descriptionText to ""
                try
                    set roleText to role of elementRef as text
                end try
                try
                    set nameText to name of elementRef as text
                end try
                try
                    set descriptionText to accessibility description of elementRef as text
                end try
                if roleText is targetRole and (nameText is targetLabel or descriptionText is targetLabel) then
                    set end of matches to contents of elementRef
                end if
            end repeat
            if (count of matches) is not 1 then error "Expected one matching UI element; found " & (count of matches)
            click item 1 of matches
        end tell
    end tell
    return "Clicked " & targetRole & " " & targetLabel
end run"""


TYPE_TEXT_SCRIPT = """on run argv
    set appName to item 1 of argv
    set targetWindow to item 2 of argv
    set textValue to item 3 of argv
    tell application "System Events"
        if not (exists process appName) then error "App is not running: " & appName
        tell process appName
            if not (exists window 1) then error "App has no open window: " & appName
            if (name of window 1) is not targetWindow then error "Front window changed; inspect the app again"
            set frontmost to true
            keystroke textValue
        end tell
    end tell
    return "Typed text into " & appName
end run"""


KEY_CODES = {
    "return": 36,
    "tab": 48,
    "space": 49,
    "delete": 51,
    "escape": 53,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
}
MODIFIERS = {
    "command": "command down",
    "option": "option down",
    "control": "control down",
    "shift": "shift down",
}


def key_script(key: str, modifiers: str = "") -> str:
    """Build a key event from whitelisted names; user text is passed as argv."""
    names = [name.strip().lower() for name in modifiers.split(",") if name.strip()]
    if len(names) != len(set(names)) or any(name not in MODIFIERS for name in names):
        raise ValueError("modifiers must be unique names from command, option, control, shift")
    normalized = key.lower()
    if normalized in KEY_CODES:
        action = f"key code {KEY_CODES[normalized]}"
    elif len(key) == 1:
        action = "keystroke keyValue"
    else:
        raise ValueError("key must be one character or a supported key name")
    if names:
        action += " using {" + ", ".join(MODIFIERS[name] for name in names) + "}"
    return f"""on run argv
    set appName to item 1 of argv
    set targetWindow to item 2 of argv
    set keyValue to item 3 of argv
    tell application "System Events"
        if not (exists process appName) then error "App is not running: " & appName
        tell process appName
            if not (exists window 1) then error "App has no open window: " & appName
            if (name of window 1) is not targetWindow then error "Front window changed; inspect the app again"
            set frontmost to true
            {action}
        end tell
    end tell
    return "Pressed key in " & appName
end run"""
