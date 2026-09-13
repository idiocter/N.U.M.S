from nums.wake import WakePhraseDetector


def test_wake_phrase_with_command() -> None:
    event = WakePhraseDetector().feed("Hey Num Num, open Safari")

    assert event is not None
    assert event.kind == "command"
    assert event.text == "open Safari"


def test_command_can_follow_wake_phrase() -> None:
    detector = WakePhraseDetector()

    wake = detector.feed("Hey numnum")
    command = detector.feed("what is the time")

    assert wake is not None and wake.kind == "wake"
    assert command is not None and command.text == "what is the time"


def test_unrelated_speech_does_not_wake() -> None:
    assert WakePhraseDetector().feed("open Safari") is None


def test_repeated_transcript_is_ignored() -> None:
    detector = WakePhraseDetector()

    assert detector.feed("Hey numnum") is not None
    assert detector.feed("Hey numnum") is None
