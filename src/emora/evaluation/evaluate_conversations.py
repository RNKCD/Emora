"""Stage 10 Part 3: synthetic conversation evaluation for Emora.
Run from project root: python src/emora/evaluation/evaluate_conversations.py
Calls the real emotion model and OpenAI API. No real user messages are saved.
"""
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from emora.models.hybrid_chatbot import EmoraHybridChatbot

OUT = ROOT / 'reports' / 'conversation_evaluation'

CASES = [
    {'id': 'listen_only', 'category': 'preference', 'turns': [
        "I'm overwhelmed with classes. Please just listen, no advice or questions.",
        "It feels like I have three deadlines at once. I only want to vent."
    ], 'criteria': 'Reflect feelings without advice or questions.'},
    {'id': 'practical_plan', 'category': 'relevance', 'turns': [
        'I have a database exam tomorrow. Give me a simple three-step study plan.'
    ], 'criteria': 'Provide a concrete, usable three-step plan.'},
    {'id': 'good_news', 'category': 'tone', 'turns': [
        'I passed my exam! I am really excited.'
    ], 'criteria': 'Celebrate naturally without unnecessary therapy framing.'},
    {'id': 'memory_recall', 'category': 'memory', 'turns': [
        'My presentation is on Friday and it is about smart parking.',
        'I am worried I will speak too fast.',
        'Remind me what my presentation is about and when it is.'
    ], 'criteria': 'Correctly recall smart parking and Friday.'},
    {'id': 'no_repeat', 'category': 'repetition', 'turns': [
        'I am a little tired after studying.',
        'I think I will take a short break.',
        'I feel better now, thanks.'
    ], 'criteria': 'Avoid repeating the same opening or question.'},
    {'id': 'declined_advice', 'category': 'preference', 'turns': [
        'I feel stressed. I do not want breathing exercises or coping tips.',
        'I would rather talk about what happened at school.'
    ], 'criteria': 'Do not push breathing exercises or coping tips.'},
]


def words(text):
    return re.findall(r"\b[\w']+\b", text.lower())


def overlap(a, b):
    a, b = set(words(a)), set(words(b))
    return round(len(a & b) / len(a | b), 3) if a | b else 0.0


def check(case, turn, response):
    """Heuristic flags are review prompts, NOT validated quality scores."""
    r = response.lower()
    flags = []
    if case['id'] == 'listen_only' and ('?' in response or re.search(r'\b(try|you should|consider|i suggest)\b', r)):
        flags.append('possible_unwanted_advice_or_question')
    if case['id'] == 'memory_recall' and turn == len(case['turns']) and not ('friday' in r and ('parking' in r)):
        flags.append('possible_memory_miss')
    if case['id'] == 'declined_advice' and turn > 1 and ('breath' in r or 'coping tip' in r):
        flags.append('possible_declined_advice')
    if not response.strip():
        flags.append('empty_response')
    return flags


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bot = EmoraHybridChatbot()
    rows = []
    for case in CASES:
        bot.reset()
        prior = None
        print(f"\n=== {case['id']} ===", flush=True)
        for turn, message in enumerate(case['turns'], 1):
            t0 = time.perf_counter()
            try:
                result = bot.chat(message)
                response = result['response']
                error = ''
                safety_mode = bool(result.get('safety_mode', False))
            except Exception as exc:
                response = ''
                error = f'{type(exc).__name__}: {exc}'
                safety_mode = False
            duration = round(time.perf_counter() - t0, 2)
            flags = check(case, turn, response)
            if prior is not None and overlap(prior, response) >= 0.65:
                flags.append('high_word_overlap_with_previous_response')
            if safety_mode:
                flags.append('unexpected_safety_mode_review')
            if error:
                flags.append('runtime_error')
            rows.append({
                'case_id': case['id'], 'category': case['category'],
                'turn': turn, 'user_message': message,
                'response': response, 'expected_behavior': case['criteria'],
                'latency_seconds': duration, 'safety_mode': safety_mode,
                'overlap_previous': overlap(prior, response) if prior is not None else '',
                'flags': ';'.join(flags), 'error': error,
                'human_score_1_to_5': '', 'human_notes': '',
            })
            print(f"Turn {turn}: {duration}s | flags: {flags or 'none'}", flush=True)
            prior = response
    columns = list(rows[0].keys())
    with (OUT / 'conversation_review.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        'cases': len(CASES), 'turns': len(rows),
        'error_count': sum(bool(r['error']) for r in rows),
        'flagged_turns': sum(bool(r['flags']) for r in rows),
        'mean_latency_seconds': round(sum(r['latency_seconds'] for r in rows) / len(rows), 2),
        'note': 'Flags are heuristics, not quality scores. Manually review every response. API usage may incur charges.'
    }
    (OUT / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\n' + json.dumps(summary, indent=2))
    print(f'\nReview responses: {OUT / "conversation_review.csv"}')


if __name__ == '__main__':
    main()
