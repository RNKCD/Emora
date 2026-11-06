"""Emora Stage 10 Part 4: synthetic performance and API-usage benchmark.

Run: python src/emora/evaluation/evaluate_performance.py
Uses real OpenAI API (billable). Does not save conversation text or API keys.
Set EMORA_INPUT_COST_PER_1M and EMORA_OUTPUT_COST_PER_1M in .env
using current pricing for your configured model; otherwise costs are omitted.
"""
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')
from emora.models.hybrid_chatbot import EmoraHybridChatbot

OUT = ROOT / 'reports' / 'performance_evaluation'
# Synthetic messages only. Keep this suite free of actual user information.
MESSAGES = [
    'I have several assignments this week and feel a bit overwhelmed.',
    'Please just listen for a moment; I do not want advice or questions.',
    'The deadlines seem to arrive at the same time.',
    'Actually, can you help me make a three-step plan?',
    'My first task is a database assignment due Friday.',
    'I also have a short presentation to practice.',
    'Can you remind me what my first task was?',
    'Thanks, that makes the week feel more manageable.',
    'What should I prioritize if I only have one hour tonight?',
    'I finished the first task today.',
    'I would like to keep tomorrow simple.',
    'Can you summarize what we discussed in two sentences?',
]


def optional_price(name):
    raw = os.getenv(name, '').strip()
    if not raw:
        return None
    try:
        value = float(raw)
        if value < 0:
            raise ValueError
        return value
    except ValueError:
        raise SystemExit(f'{name} must be a nonnegative USD price per 1M tokens')


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * pct / 100
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return round(values[low] + (values[high] - values[low]) * (pos - low), 3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bot = EmoraHybridChatbot()
    # Instrument the actual OpenAI client(s) without changing application code.
    # Some projects use separate clients for response generation and summarization.
    observed = []
    wrapped = set()
    for component in (getattr(bot, 'response_generator', None),
                      getattr(bot, 'generator', None),
                      getattr(getattr(bot, 'memory', None), 'client', None)):
        client = component if hasattr(component, 'chat') else getattr(component, 'client', None)
        if client is None:
            continue
        completions = getattr(getattr(client, 'chat', None), 'completions', None)
        if completions is None or id(completions) in wrapped:
            continue
        original = completions.create
        def measured_create(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            usage = getattr(result, 'usage', None)
            observed.append({
                'model': getattr(result, 'model', None),
                'input_tokens': int(getattr(usage, 'prompt_tokens', 0) or 0),
                'output_tokens': int(getattr(usage, 'completion_tokens', 0) or 0),
            })
            return result
        completions.create = measured_create
        wrapped.add(id(completions))

    if not wrapped:
        raise RuntimeError('Could not locate OpenAI clients. Check chatbot attribute names in this script.')

    rows = []
    for i, message in enumerate(MESSAGES, 1):
        start = time.perf_counter()
        before = len(observed)
        error = None
        try:
            result = bot.chat(message)
            response = result.get('response', '')
            safety_mode = bool(result.get('safety_mode', False))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            response, safety_mode = '', False
        elapsed = round(time.perf_counter() - start, 3)
        calls = observed[before:]
        row = {
            'turn': i, 'latency_seconds': elapsed,
            'response_characters': len(response),
            'api_calls': len(calls),
            'input_tokens': sum(x['input_tokens'] for x in calls),
            'output_tokens': sum(x['output_tokens'] for x in calls),
            'safety_mode': safety_mode,
            'error': error,
        }
        rows.append(row)
        print(f"Turn {i:02d}: {elapsed:.2f}s | API calls={row['api_calls']} | "
              f"input={row['input_tokens']} output={row['output_tokens']} | "
              f"{'ERROR' if error else 'OK'}", flush=True)

    input_tokens = sum(r['input_tokens'] for r in rows)
    output_tokens = sum(r['output_tokens'] for r in rows)
    durations = [r['latency_seconds'] for r in rows]
    input_price = optional_price('EMORA_INPUT_COST_PER_1M')
    output_price = optional_price('EMORA_OUTPUT_COST_PER_1M')
    cost = None if input_price is None or output_price is None else round(
        (input_tokens * input_price + output_tokens * output_price) / 1_000_000, 6)
    summary = {
        'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        'model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
        'turns': len(rows),
        'errors': sum(bool(r['error']) for r in rows),
        'mean_latency_seconds': round(statistics.mean(durations), 3),
        'median_latency_seconds': round(statistics.median(durations), 3),
        'p95_latency_seconds': percentile(durations, 95),
        'api_calls': len(observed),
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'avg_input_tokens_per_turn': round(input_tokens / len(rows), 1),
        'avg_output_tokens_per_turn': round(output_tokens / len(rows), 1),
        'configured_input_usd_per_1m': input_price,
        'configured_output_usd_per_1m': output_price,
        'estimated_cost_usd': cost,
        'monthly_scenarios': [
            {'users': users, 'turns_per_user_per_day': turns,
             'estimated_monthly_api_usd': None if cost is None else round(
                 cost / len(rows) * users * turns * 30, 2)}
            for users, turns in [(10, 10), (50, 10), (100, 10), (100, 20)]
        ],
        'limitations': [
            'Small synthetic sample, not a load or concurrency test.',
            'Monthly projections assume similar token usage and no caching discounts.',
            'Pricing must be supplied from current provider rates; hosting is excluded.',
            'Only instrumented OpenAI chat completion clients are counted.',
            'No user text or model responses are stored in output files.',
        ],
    }
    (OUT / 'performance_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    import csv
    with (OUT / 'performance_turns.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('\n' + json.dumps(summary, indent=2))
    print(f'\nSaved results to {OUT}')


if __name__ == '__main__':
    main()
