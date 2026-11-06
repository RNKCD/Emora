import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
"""Regression tests for the multi-turn safety sequence observed during Emora testing."""
from emora.safety.safety_guard import SafetyGuard


def test_ambiguous_yes_and_immediate_firearm_escalation():
    guard = SafetyGuard()
    assert guard.check('I feel like dying')['is_crisis']
    assert 'someone' in guard.check('ye')['response'].lower()
    result = guard.check('i have gun on my head')
    assert '911' in result['response']
    assert 'firearm' in result['response'].lower()
    assert guard.immediate_danger


def test_help_request_and_safety_reassessment():
    guard = SafetyGuard()
    guard.check('I feel like dying')
    guard.check('ye')
    guard.check('i have gun on my head')
    assert '911' in guard.check('help me')['response']
    assert guard.check('can you help me in the exam?')['is_crisis']
    assert 'away' in guard.check('I am okay now')['response'].lower()
    assert guard.active
    assert guard.check('yes')['is_crisis']
    assert guard.active
    assert guard.check('I am away from the gun and called 911')['is_crisis']
    assert not guard.active
    assert not guard.check('Can you help me study?')['is_crisis']


def test_noncrisis_and_reset():
    guard = SafetyGuard()
    assert not guard.check('I failed my exam')['is_crisis']
    guard.check('I feel like dying')
    guard.reset()
    assert not guard.check('hello')['is_crisis']


def test_immediate_overdose():
    guard = SafetyGuard()
    result = guard.check('I already overdosed')
    assert result['is_crisis']
    assert 'emergency' in result['response'].lower()
