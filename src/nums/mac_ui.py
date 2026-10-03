"""AppleScript commands for accessibility-based Mac app interaction."""

from __future__ import annotations


LIST_APPS_SCRIPT = """tell application "System Events"
    get name of every process whose background only is false
end tell"""


NOTIFY_SCRIPT = """on run argv
    display notification (item 1 of argv) with title (item 2 of argv)
end run"""


LIST_WINDOWS_SCRIPT = """on run argv
    set appName to item 1 of argv
    tell application "System Events"
        if not (exists process appName) then error "App is not running: " & appName
        tell process appName
            set entries to {"Windows for " & appName}
            set allWindows to windows
            repeat with windowIndex from 1 to count of allWindows
                try
                    set end of entries to (windowIndex as text) & tab & (name of item windowIndex of allWindows as text)
                end try
            end repeat
        end tell
    end tell
    set AppleScript's text item delimiters to linefeed
    return entries as text
end run"""


FOCUS_WINDOW_SCRIPT = """on run argv
    set appName to item 1 of argv
    set targetTitle to item 2 of argv
    set selectedIndexText to item 3 of argv
    set selectedIndex to 0
    if selectedIndexText is not "" then set selectedIndex to selectedIndexText as integer
    tell application "System Events"
        if not (exists process appName) then error "App is not running: " & appName
        tell process appName
            set matches to {}
            set allWindows to windows
            repeat with windowIndex from 1 to count of allWindows
                if selectedIndex is 0 or windowIndex is selectedIndex then
                    set windowRef to item windowIndex of allWindows
                    try
                        if (name of windowRef as text) is targetTitle then set end of matches to contents of windowRef
                    end try
                end if
            end repeat
            if (count of matches) is not 1 then error "Expected one matching window; found " & (count of matches)
            set frontmost to true
            perform action "AXRaise" of item 1 of matches
            if not (exists window 1) then error "Window did not become frontmost"
            if (name of window 1) is not targetTitle then error "Window did not become frontmost"
        end tell
    end tell
    return "Focused window " & targetTitle
end run"""


INSPECT_MENU_SCRIPT = """on run argv
    set appName to item 1 of argv
    set menuName to item 2 of argv
    set submenuName to item 3 of argv
    tell application "System Events"
        if not (exists process appName) then error "App is not running: " & appName
        tell process appName
            if menuName is "" then
                set entries to {"Menus:"}
                repeat with menuRef in menu bar items of menu bar 1
                    try
                        set end of entries to name of menuRef as text
                    end try
                end repeat
            else
                if not (exists menu bar item menuName of menu bar 1) then error "Menu not found: " & menuName
                if submenuName is "" then
                    set entries to {"Menu: " & menuName}
                    set itemsToInspect to menu items of menu menuName of menu bar item menuName of menu bar 1
                else
                    set submenuMatches to {}
                    repeat with candidate in menu items of menu menuName of menu bar item menuName of menu bar 1
                        try
                            if (name of candidate as text) is submenuName then set end of submenuMatches to contents of candidate
                        end try
                    end repeat
                    if (count of submenuMatches) is not 1 then error "Expected one matching submenu; found " & (count of submenuMatches)
                    set parentItem to item 1 of submenuMatches
                    set entries to {"Submenu: " & menuName & " > " & submenuName}
                    set itemsToInspect to menu items of menu of parentItem
                end if
                repeat with itemRef in itemsToInspect
                    try
                        set itemName to name of itemRef as text
                        if itemName is not "" then
                            set availability to "disabled"
                            if enabled of itemRef then set availability to "enabled"
                            set end of entries to itemName & tab & availability
                        end if
                    end try
                end repeat
            end if
        end tell
    end tell
    set AppleScript's text item delimiters to linefeed
    return entries as text
end run"""


CLICK_MENU_ITEM_SCRIPT = """on run argv
    set appName to item 1 of argv
    set menuName to item 2 of argv
    set itemName to item 3 of argv
    set submenuName to item 4 of argv
    tell application "System Events"
        if not (exists process appName) then error "App is not running: " & appName
        tell process appName
            set frontmost to true
            if not (exists menu bar item menuName of menu bar 1) then error "Menu not found: " & menuName
            if submenuName is "" then
                set candidateItems to menu items of menu menuName of menu bar item menuName of menu bar 1
            else
                set submenuMatches to {}
                repeat with candidate in menu items of menu menuName of menu bar item menuName of menu bar 1
                    try
                        if (name of candidate as text) is submenuName then set end of submenuMatches to contents of candidate
                    end try
                end repeat
                if (count of submenuMatches) is not 1 then error "Expected one matching submenu; found " & (count of submenuMatches)
                set candidateItems to menu items of menu of item 1 of submenuMatches
            end if
            set itemMatches to {}
            repeat with candidate in candidateItems
                try
                    if (name of candidate as text) is itemName then set end of itemMatches to contents of candidate
                end try
            end repeat
            if (count of itemMatches) is not 1 then error "Expected one matching menu item; found " & (count of itemMatches)
            set targetItem to item 1 of itemMatches
            if not (enabled of targetItem) then error "Menu item is disabled: " & itemName
            click targetItem
        end tell
    end tell
    return "Clicked menu item " & itemName
end run"""


INSPECT_APP_SCRIPT = """on run argv
    set appName to item 1 of argv
    set offsetCount to item 2 of argv as integer
    set pageSize to item 3 of argv as integer
    tell application "System Events"
        if not (exists process appName) then error "App is not running: " & appName
        tell process appName
            if not (exists window 1) then error "App has no open window: " & appName
            set targetWindow to window 1
            set windowName to name of targetWindow
            set entries to {"Window: " & windowName}
            set elements to entire contents of targetWindow
            set totalCount to count of elements
            set end of entries to "Total elements: " & totalCount
            set startIndex to offsetCount + 1
            set lastIndex to totalCount
            if lastIndex > offsetCount + pageSize then set lastIndex to offsetCount + pageSize
            if startIndex <= lastIndex then
                repeat with elementIndex from startIndex to lastIndex
                    set elementRef to item elementIndex of elements
                    set roleText to ""
                    set nameText to ""
                    set descriptionText to ""
                    set availability to "unknown"
                    set stateText to ""
                    try
                        set roleText to role of elementRef as text
                    end try
                    try
                        set nameText to name of elementRef as text
                    end try
                    try
                        set descriptionText to accessibility description of elementRef as text
                    end try
                    try
                        if enabled of elementRef then
                            set availability to "enabled"
                        else
                            set availability to "disabled"
                        end if
                    end try
                    if roleText is "AXCheckBox" or roleText is "AXRadioButton" or roleText is "AXSwitch" then
                        try
                            set stateText to value of elementRef as text
                        end try
                    end if
                    if roleText is not "" or nameText is not "" or descriptionText is not "" then
                        set end of entries to (elementIndex as text) & tab & roleText & tab & nameText & tab & descriptionText & tab & availability & tab & stateText
                    end if
                end repeat
            end if
            if totalCount > lastIndex then set end of entries to "Next offset: " & lastIndex
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
    set selectedIndexText to item 5 of argv
    set selectedIndex to 0
    if selectedIndexText is not "" then set selectedIndex to selectedIndexText as integer
    tell application "System Events"
        if not (exists process appName) then error "App is not running: " & appName
        tell process appName
            if not (exists window 1) then error "App has no open window: " & appName
            if (name of window 1) is not targetWindow then error "Front window changed; inspect the app again"
            set frontmost to true
            if not (exists window 1) or (name of window 1) is not targetWindow then error "Front window changed during activation; inspect the app again"
            set matches to {}
            set elements to entire contents of window 1
            repeat with elementIndex from 1 to count of elements
                if selectedIndex is 0 or elementIndex is selectedIndex then
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
                    if roleText is targetRole and (targetLabel is "" or nameText is targetLabel or descriptionText is targetLabel) then
                        set end of matches to contents of elementRef
                    end if
                end if
            end repeat
            if (count of matches) is not 1 then error "Expected one matching UI element; found " & (count of matches)
            set targetElement to item 1 of matches
            if not (enabled of targetElement) then error "UI element is disabled; inspect the app again"
            click targetElement
        end tell
    end tell
    return "Clicked " & targetRole & " " & targetLabel
end run"""


SET_TOGGLE_SCRIPT = """on run argv
    set appName to item 1 of argv
    set targetWindow to item 2 of argv
    set targetRole to item 3 of argv
    set targetLabel to item 4 of argv
    set selectedIndexText to item 5 of argv
    set desiredState to item 6 of argv
    set selectedIndex to 0
    if selectedIndexText is not "" then set selectedIndex to selectedIndexText as integer
    tell application "System Events"
        if not (exists process appName) then error "App is not running: " & appName
        tell process appName
            if not (exists window 1) then error "App has no open window: " & appName
            if (name of window 1) is not targetWindow then error "Front window changed; inspect the app again"
            set frontmost to true
            if not (exists window 1) or (name of window 1) is not targetWindow then error "Front window changed during activation; inspect the app again"
            set matches to {}
            set elements to entire contents of window 1
            repeat with elementIndex from 1 to count of elements
                if selectedIndex is 0 or elementIndex is selectedIndex then
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
                    if roleText is targetRole and (targetLabel is "" or nameText is targetLabel or descriptionText is targetLabel) then
                        set end of matches to contents of elementRef
                    end if
                end if
            end repeat
            if (count of matches) is not 1 then error "Expected one matching toggle; found " & (count of matches)
            set targetElement to item 1 of matches
            if not (enabled of targetElement) then error "Toggle is disabled"
            set currentState to (value of targetElement) as text
            if currentState is not "0" and currentState is not "1" and currentState is not "false" and currentState is not "true" then error "Toggle state is unavailable"
            set currentOn to currentState is "1" or currentState is "true"
            set desiredOn to desiredState is "on"
            if currentOn is not desiredOn then click targetElement
            set finalState to (value of targetElement) as text
            set finalOn to finalState is "1" or finalState is "true"
            if finalOn is not desiredOn then error "Toggle did not reach requested state"
        end tell
    end tell
    return "Toggle is " & desiredState
end run"""


TYPE_TEXT_SCRIPT = """on run argv
    set appName to item 1 of argv
    set targetWindow to item 2 of argv
    set textValue to item 3 of argv
    set targetRole to item 4 of argv
    set targetLabel to item 5 of argv
    set selectedIndexText to item 6 of argv
    set selectedIndex to 0
    if selectedIndexText is not "" then set selectedIndex to selectedIndexText as integer
    tell application "System Events"
        if not (exists process appName) then error "App is not running: " & appName
        tell process appName
            if not (exists window 1) then error "App has no open window: " & appName
            if (name of window 1) is not targetWindow then error "Front window changed; inspect the app again"
            set frontmost to true
            if not (exists window 1) or (name of window 1) is not targetWindow then error "Front window changed during activation; inspect the app again"
            if targetRole is not "" then
                set matches to {}
                set elements to entire contents of window 1
                repeat with elementIndex from 1 to count of elements
                    if selectedIndex is 0 or elementIndex is selectedIndex then
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
                        if roleText is targetRole and (targetLabel is "" or nameText is targetLabel or descriptionText is targetLabel) then
                            set end of matches to contents of elementRef
                        end if
                    end if
                end repeat
                if (count of matches) is not 1 then error "Expected one matching text field; found " & (count of matches)
                set targetField to item 1 of matches
                click targetField
                if not (exists window 1) then error "Front window changed after focusing the text field"
                if (name of window 1) is not targetWindow then error "Front window changed after focusing the text field"
                if not (focused of targetField) then error "Text field did not receive focus; nothing was typed"
            end if
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
    "enter": 76,
    "home": 115,
    "page_up": 116,
    "forward_delete": 117,
    "end": 119,
    "page_down": 121,
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
            if not (exists window 1) or (name of window 1) is not targetWindow then error "Front window changed during activation; inspect the app again"
            {action}
        end tell
    end tell
    return "Pressed key in " & appName
end run"""
