
import sys
from pathlib import Path

# Add the src directory to Python's import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emora.safety.safety_guard import SafetyGuard


def test_natural_language_reassurance():
    """Emora should understand a natural-language denial."""
    guard = SafetyGuard()

    result = guard.check("I feel like dying")
    assert result["is_crisis"] is True

    result = guard.check(
        "no. I was just emotional. I am good now"
    )
    assert result["is_crisis"] is True

    # The user denied current suicidal intent.
    # Emora should now check whether they are safe.
    assert guard.last_question == "safe"


def test_natural_language_safe_response():
    """Emora should recognize informal safety confirmation."""
    guard = SafetyGuard()

    guard.check("I feel like dying")
    guard.check("no. I was just emotional. I am good now")

    result = guard.check("I am safe rn")

    assert result["is_crisis"] is True

    # Emora should ask a final safety follow-up.
    assert guard.last_question == "reassess"


def test_complete_safety_recovery():
    """Emora should return to normal after safety confirmation."""
    guard = SafetyGuard()

    guard.check("I feel like dying")
    guard.check("no. I was just emotional. I am good now")
    guard.check("I am safe rn")

    result = guard.check("Yes")

    assert result["is_crisis"] is True
    assert guard.active is False

    result = guard.check("Can you help me with my exam?")

    assert result["is_crisis"] is False


def test_ambiguous_answer_does_not_clear_safety():
    """An unclear response must not automatically clear safety mode."""
    guard = SafetyGuard()

    guard.check("I feel like dying")

    result = guard.check("I don't know")

    assert result["is_crisis"] is True
    assert guard.active is True


def test_immediate_danger_not_cleared_by_reassurance():
    """A recent weapon emergency needs additional safety checks."""
    guard = SafetyGuard()

    guard.check("I feel like dying")
    guard.check("I have a gun on my head")

    result = guard.check("I am okay now")

    assert result["is_crisis"] is True
    assert guard.active is True
