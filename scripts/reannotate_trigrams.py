#!/usr/bin/env python3
"""
Hexagram Re-annotation Script (2-step trigram method)

Purpose: Re-annotate classical_before/after_hexagram fields using
         separate upper/lower trigram selection to utilize full 64-hexagram space.

Usage:
    # Pilot run (20 cases)
    python3 scripts/reannotate_trigrams.py --pilot

    # Full run (500 cases)
    python3 scripts/reannotate_trigrams.py --full

    # Custom count
    python3 scripts/reannotate_trigrams.py --count 50
"""

import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

# Force unbuffered output
print = partial(print, flush=True)

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
CASES_PATH = BASE_DIR / "data" / "raw" / "cases.jsonl"
OUTPUT_DIR = BASE_DIR / "analysis" / "phase3"
PILOT_OUTPUT = OUTPUT_DIR / "reannotation_pilot.json"
FULL_OUTPUT = OUTPUT_DIR / "reannotation_full.json"

# ── King Wen mapping: (lower, upper) -> number ──
TRIGRAM_TO_KW = {
    ('乾','乾'): 1, ('坤','坤'): 2, ('震','坎'): 3, ('艮','坎'): 4,
    ('乾','坎'): 5, ('坎','乾'): 6, ('坎','坤'): 7, ('坤','坎'): 8,
    ('乾','巽'): 9, ('兌','乾'): 10, ('乾','坤'): 11, ('坤','乾'): 12,
    ('離','乾'): 13, ('乾','離'): 14, ('艮','坤'): 15, ('坤','震'): 16,
    ('震','兌'): 17, ('巽','艮'): 18, ('坤','兌'): 19, ('坤','巽'): 20,
    ('震','離'): 21, ('離','艮'): 22, ('坤','艮'): 23, ('震','坤'): 24,
    ('震','乾'): 25, ('乾','艮'): 26, ('震','艮'): 27, ('巽','兌'): 28,
    ('坎','坎'): 29, ('離','離'): 30, ('艮','兌'): 31, ('巽','震'): 32,
    ('艮','乾'): 33, ('乾','震'): 34, ('坤','離'): 35, ('離','坤'): 36,
    ('離','巽'): 37, ('兌','離'): 38, ('艮','坎'): 39, ('坎','震'): 40,
    ('兌','艮'): 41, ('震','巽'): 42, ('乾','兌'): 43, ('巽','乾'): 44,
    ('坤','兌'): 45, ('巽','坤'): 46, ('坎','兌'): 47, ('巽','坎'): 48,
    ('離','兌'): 49, ('巽','離'): 50, ('震','震'): 51, ('艮','艮'): 52,
    ('艮','巽'): 53, ('兌','震'): 54, ('離','震'): 55, ('艮','離'): 56,
    ('巽','巽'): 57, ('兌','兌'): 58, ('坎','巽'): 59, ('兌','坎'): 60,
    ('兌','巽'): 61, ('艮','震'): 62, ('離','坎'): 63, ('坎','離'): 64,
}

# Reverse lookup: number -> name
KW_TO_NAME = {
    1: '乾為天', 2: '坤為地', 3: '水雷屯', 4: '山水蒙', 5: '水天需',
    6: '天水訟', 7: '地水師', 8: '水地比', 9: '風天小畜', 10: '天沢履',
    11: '地天泰', 12: '天地否', 13: '天火同人', 14: '火天大有', 15: '地山謙',
    16: '雷地豫', 17: '沢雷随', 18: '山風蠱', 19: '沢地臨', 20: '風地観',
    21: '火雷噬嗑', 22: '山火賁', 23: '山地剥', 24: '地雷復', 25: '天雷无妄',
    26: '山天大畜', 27: '山雷頤', 28: '沢風大過', 29: '坎為水', 30: '離為火',
    31: '沢山咸', 32: '雷風恒', 33: '天山遯', 34: '雷天大壮', 35: '火地晋',
    36: '地火明夷', 37: '風火家人', 38: '火沢睽', 39: '水山蹇', 40: '雷水解',
    41: '山沢損', 42: '風雷益', 43: '沢天夬', 44: '天風姤', 45: '沢地萃',
    46: '地風升', 47: '沢水困', 48: '水風井', 49: '沢火革', 50: '火風鼎',
    51: '震為雷', 52: '艮為山', 53: '風山漸', 54: '雷沢帰妹', 55: '雷火豊',
    56: '火山旅', 57: '巽為風', 58: '兌為沢', 59: '風水渙', 60: '水沢節',
    61: '風沢中孚', 62: '雷山小過', 63: '水火既済', 64: '火水未済',
}

VALID_TRIGRAMS = {'乾', '坤', '震', '巽', '坎', '離', '艮', '兌'}

PURE_HEXAGRAM_NAMES = {'乾為天', '坤為地', '震為雷', '巽為風', '坎為水', '離為火', '艮為山', '兌為沢'}


# ── Sampling ──

def load_cases():
    """Load all cases from JSONL."""
    cases = []
    with open(CASES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def stratified_sample(cases, n=500, seed=42):
    """
    Stratified sampling by source_type.
    Target: news 300, book 80, official 80, blog 40.
    Remaining from sns or overflow goes to largest bucket.
    """
    rng = random.Random(seed)

    targets = {
        'news': 300,
        'book': 80,
        'official': 80,
        'blog': 40,
    }

    by_type = {}
    for c in cases:
        st = c.get('source_type', 'news')
        by_type.setdefault(st, []).append(c)

    sampled = []
    for st, target_n in targets.items():
        pool = by_type.get(st, [])
        actual_n = min(target_n, len(pool))
        sampled.extend(rng.sample(pool, actual_n))

    # Fill remaining from sns or news
    remaining = n - len(sampled)
    if remaining > 0:
        sns_pool = by_type.get('sns', [])
        extra = rng.sample(sns_pool, min(remaining, len(sns_pool)))
        sampled.extend(extra)

    rng.shuffle(sampled)
    return sampled


# ── Prompt generation ──

def build_before_prompt(case):
    """Build the before-state annotation prompt for a case."""
    return f"""あなたは易経の専門家です。以下の事例の「変化前」の状態を、八卦の上卦と下卦で表現してください。

【事例情報】
- タイトル: {case.get('target_name', '')}
- 時期: {case.get('period', '')}
- 概要: {case.get('story_summary', '')}
- 変化前の状態: {case.get('before_state', '')}
- トリガー: {case.get('trigger_type', '')}

### Step 1: 下卦（内卦）の選定

事例の「変化前」における内的状態・基盤・本質を最もよく表す八卦を1つ選んでください。

選択肢:
- 乾(☰): 創造力・強い意志・主導性
- 坤(☷): 受容・従順・基盤の安定
- 震(☳): 衝撃・覚醒・新しい動き
- 巽(☴): 浸透・柔軟・じわじわとした変化
- 坎(☵): 困難・危険・流動
- 離(☲): 明晰・注目・輝き
- 艮(☶): 停止・安定・蓄積
- 兌(☱): 喜び・交流・開放

### Step 2: 上卦（外卦）の選定

事例の「変化前」における外的状況・環境・表面的な現れを最もよく表す八卦を1つ選んでください。

選択肢は Step 1 と同じ8つです。

### 重要な指示
- 上卦と下卦は独立に選んでください。同じ八卦を選んでも構いませんが、必ず別々に判断してください
- 下卦は「内面・本質・根底にある力」を表します
- 上卦は「外面・環境・表面に現れている状況」を表します
- 内面と外面が異なる状況は非常に一般的です（例: 内面は強い意志(乾)だが外面は困難(坎)→天水訟）

### 出力形式（JSONのみ出力してください。説明文は不要です）
```json
{{
  "lower_trigram": "<八卦名>",
  "lower_reasoning": "<内的状態の選定理由>",
  "upper_trigram": "<八卦名>",
  "upper_reasoning": "<外的状況の選定理由>",
  "hexagram_number": <1-64>,
  "hexagram_name": "<64卦名>"
}}
```

上記のJSON以外は一切出力しないでください。説明文・前置き・補足は不要です。JSONだけを返してください。"""


def build_after_prompt(case):
    """Build the after-state annotation prompt for a case."""
    return f"""あなたは易経の専門家です。以下の事例の「変化後」の状態を、八卦の上卦と下卦で表現してください。

【事例情報】
- タイトル: {case.get('target_name', '')}
- 時期: {case.get('period', '')}
- 概要: {case.get('story_summary', '')}
- 変化後の状態: {case.get('after_state', '')}
- 取った行動: {case.get('action_type', '')}
- 結果: {case.get('outcome', '')}

### Step 1: 下卦（内卦）の選定

事例の「変化後」における内的状態・基盤・本質を最もよく表す八卦を1つ選んでください。

選択肢:
- 乾(☰): 創造力・強い意志・主導性
- 坤(☷): 受容・従順・基盤の安定
- 震(☳): 衝撃・覚醒・新しい動き
- 巽(☴): 浸透・柔軟・じわじわとした変化
- 坎(☵): 困難・危険・流動
- 離(☲): 明晰・注目・輝き
- 艮(☶): 停止・安定・蓄積
- 兌(☱): 喜び・交流・開放

### Step 2: 上卦（外卦）の選定

事例の「変化後」における外的状況・環境・表面的な現れを最もよく表す八卦を1つ選んでください。

選択肢は Step 1 と同じ8つです。

### 重要な指示
- 上卦と下卦は独立に選んでください。同じ八卦を選んでも構いませんが、必ず別々に判断してください
- 下卦は「内面・本質・根底にある力」を表します
- 上卦は「外面・環境・表面に現れている状況」を表します
- 内面と外面が異なる状況は非常に一般的です（例: 内面は強い意志(乾)だが外面は困難(坎)→天水訟）

### 出力形式（JSONのみ出力してください。説明文は不要です）
```json
{{
  "lower_trigram": "<八卦名>",
  "lower_reasoning": "<内的状態の選定理由>",
  "upper_trigram": "<八卦名>",
  "upper_reasoning": "<外的状況の選定理由>",
  "hexagram_number": <1-64>,
  "hexagram_name": "<64卦名>"
}}
```

上記のJSON以外は一切出力しないでください。説明文・前置き・補足は不要です。JSONだけを返してください。"""


# ── Codex exec ──

def extract_json_from_text(text):
    """Extract JSON object from text that may contain markdown fences, codex headers, etc."""
    if not text or not text.strip():
        return None

    # Strip codex header lines (everything before the first '{')
    # Codex output often starts with "OpenAI Codex v...\n--------\n..."
    lines = text.split('\n')
    cleaned_lines = []
    found_content = False
    for line in lines:
        # Skip codex header/metadata lines
        if not found_content:
            stripped = line.strip()
            if stripped.startswith('{') or stripped.startswith('```'):
                found_content = True
                cleaned_lines.append(line)
            elif stripped.startswith('"lower_trigram"') or stripped.startswith('"upper_trigram"'):
                found_content = True
                cleaned_lines.append('{')
                cleaned_lines.append(line)
            # Skip header lines
            continue
        cleaned_lines.append(line)

    cleaned_text = '\n'.join(cleaned_lines) if cleaned_lines else text

    # Try to find JSON in code fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned_text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try to find a JSON object with nested content (allow inner quotes/braces)
    # Find the first { and the matching last }
    first_brace = cleaned_text.find('{')
    last_brace = cleaned_text.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        candidate = cleaned_text[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Also try on the original text
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def call_codex(prompt, timeout=180, max_retries=3):
    """Call codex exec and return parsed JSON result."""
    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(
                ['codex', 'exec', '--skip-git-repo-check', prompt],
                capture_output=True, text=True, timeout=timeout,
                cwd=str(BASE_DIR)
            )

            # Try stdout first, then stderr, then combined
            output = result.stdout.strip()
            parsed = extract_json_from_text(output) if output else None

            if parsed is None and result.stderr:
                stderr_out = result.stderr.strip()
                parsed = extract_json_from_text(stderr_out)
                if parsed is None and output:
                    # Try combined
                    parsed = extract_json_from_text(output + '\n' + stderr_out)

            if parsed is not None:
                return parsed, None

            if attempt < max_retries:
                wait = 5 + attempt * 3  # 5s, 8s, 11s
                print(f"    Parse failed (attempt {attempt+1}/{max_retries+1}), retrying in {wait}s...")
                time.sleep(wait)
                continue

            # For error reporting, show what we got
            raw = output or result.stderr.strip() or "(empty)"
            return None, f"JSON parse failed. Raw output: {raw[:500]}"

        except subprocess.TimeoutExpired:
            if attempt < max_retries:
                wait = 5 + attempt * 3
                print(f"    Timeout (attempt {attempt+1}/{max_retries+1}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            return None, "Timeout after 180s"

        except Exception as e:
            return None, f"Exception: {str(e)}"

    return None, "Max retries exceeded"


# ── Validation ──

def validate_result(result, phase='before'):
    """Validate a codex result. Returns (is_valid, issues)."""
    issues = []

    if result is None:
        return False, ["null result"]

    lower = result.get('lower_trigram', '')
    upper = result.get('upper_trigram', '')

    if lower not in VALID_TRIGRAMS:
        issues.append(f"invalid lower_trigram: {lower}")
    if upper not in VALID_TRIGRAMS:
        issues.append(f"invalid upper_trigram: {upper}")

    if lower in VALID_TRIGRAMS and upper in VALID_TRIGRAMS:
        expected_kw = TRIGRAM_TO_KW.get((lower, upper))
        reported_kw = result.get('hexagram_number')
        if expected_kw and reported_kw and expected_kw != reported_kw:
            issues.append(f"KW mismatch: ({lower},{upper})={expected_kw} but reported {reported_kw}")
            # Auto-correct
            result['hexagram_number'] = expected_kw
            result['hexagram_name'] = KW_TO_NAME.get(expected_kw, '')

    if not result.get('lower_reasoning'):
        issues.append("missing lower_reasoning")
    if not result.get('upper_reasoning'):
        issues.append("missing upper_reasoning")

    return len(issues) == 0, issues


# ── Analysis ──

def analyze_results(results):
    """Analyze reannotation results and compute statistics."""
    stats = {
        'total_cases': len(results),
        'before': {'success': 0, 'fail': 0, 'pure': 0, 'non_pure': 0, 'unique_hexagrams': set()},
        'after': {'success': 0, 'fail': 0, 'pure': 0, 'non_pure': 0, 'unique_hexagrams': set()},
        'errors': [],
    }

    for r in results:
        for phase in ['before', 'after']:
            res = r.get(f'{phase}_result')
            if res and r.get(f'{phase}_error') is None:
                stats[phase]['success'] += 1
                lower = res.get('lower_trigram', '')
                upper = res.get('upper_trigram', '')
                kw = TRIGRAM_TO_KW.get((lower, upper))
                if kw:
                    name = KW_TO_NAME.get(kw, '')
                    stats[phase]['unique_hexagrams'].add(kw)
                    if name in PURE_HEXAGRAM_NAMES:
                        stats[phase]['pure'] += 1
                    else:
                        stats[phase]['non_pure'] += 1
            else:
                stats[phase]['fail'] += 1
                if r.get(f'{phase}_error'):
                    stats['errors'].append({
                        'case_id': r.get('transition_id', ''),
                        'phase': phase,
                        'error': r.get(f'{phase}_error')
                    })

    # Convert sets to counts
    for phase in ['before', 'after']:
        s = stats[phase]
        s['unique_count'] = len(s['unique_hexagrams'])
        s['unique_hexagrams'] = sorted(s['unique_hexagrams'])
        total_valid = s['success']
        if total_valid > 0:
            s['pure_rate'] = round(s['pure'] / total_valid * 100, 1)
            s['non_pure_rate'] = round(s['non_pure'] / total_valid * 100, 1)
        else:
            s['pure_rate'] = 0
            s['non_pure_rate'] = 0

    return stats


def print_stats(stats):
    """Print analysis statistics."""
    print("\n" + "=" * 60)
    print("REANNOTATION RESULTS")
    print("=" * 60)
    print(f"Total cases processed: {stats['total_cases']}")

    for phase in ['before', 'after']:
        s = stats[phase]
        print(f"\n--- {phase.upper()} ---")
        print(f"  Success: {s['success']}, Fail: {s['fail']}")
        print(f"  Pure hexagrams:     {s['pure']:3d} ({s['pure_rate']}%)")
        print(f"  Non-pure hexagrams: {s['non_pure']:3d} ({s['non_pure_rate']}%)")
        print(f"  Unique hexagrams:   {s['unique_count']}/64")

    if stats['errors']:
        print(f"\n--- ERRORS ({len(stats['errors'])}) ---")
        for e in stats['errors'][:10]:
            print(f"  [{e['phase']}] {e['case_id']}: {e['error'][:100]}")

    # Compare with baseline
    print("\n--- COMPARISON ---")
    print(f"  Baseline pure rate: ~82% (before), ~82% (after)")
    print(f"  New pure rate:      {stats['before']['pure_rate']}% (before), {stats['after']['pure_rate']}% (after)")
    print("=" * 60)


# ── Main ──

def run_reannotation(n_cases=20, output_path=None):
    """Run the reannotation pipeline."""
    if output_path is None:
        output_path = PILOT_OUTPUT

    print(f"Loading cases from {CASES_PATH}...")
    all_cases = load_cases()
    print(f"  Total cases: {len(all_cases)}")

    print(f"Sampling {n_cases} cases (stratified by source_type)...")
    sampled = stratified_sample(all_cases, n=n_cases)[:n_cases]
    print(f"  Sampled: {len(sampled)} cases")

    # Show source_type distribution
    st_dist = Counter(c.get('source_type', '') for c in sampled)
    print(f"  Distribution: {dict(st_dist)}")

    results = []
    for i, case in enumerate(sampled):
        tid = case.get('transition_id', f'case_{i}')
        print(f"\n[{i+1}/{len(sampled)}] {tid}: {case.get('target_name', '')[:50]}")

        entry = {
            'transition_id': tid,
            'target_name': case.get('target_name', ''),
            'source_type': case.get('source_type', ''),
            'original_before': case.get('classical_before_hexagram', ''),
            'original_after': case.get('classical_after_hexagram', ''),
        }

        # Before annotation
        print("  Before annotation...")
        before_prompt = build_before_prompt(case)
        before_result, before_error = call_codex(before_prompt)

        if before_result:
            is_valid, issues = validate_result(before_result, 'before')
            if not is_valid:
                before_error = f"Validation: {'; '.join(issues)}"
                print(f"    Validation issues: {issues}")
            else:
                lower = before_result.get('lower_trigram', '')
                upper = before_result.get('upper_trigram', '')
                kw = TRIGRAM_TO_KW.get((lower, upper))
                name = KW_TO_NAME.get(kw, '?')
                is_pure = "PURE" if name in PURE_HEXAGRAM_NAMES else "non-pure"
                print(f"    Result: {name} (#{kw}) [{is_pure}] lower={lower} upper={upper}")
        else:
            print(f"    Error: {before_error}")

        entry['before_result'] = before_result
        entry['before_error'] = before_error

        # After annotation
        print("  After annotation...")
        after_prompt = build_after_prompt(case)
        after_result, after_error = call_codex(after_prompt)

        if after_result:
            is_valid, issues = validate_result(after_result, 'after')
            if not is_valid:
                after_error = f"Validation: {'; '.join(issues)}"
                print(f"    Validation issues: {issues}")
            else:
                lower = after_result.get('lower_trigram', '')
                upper = after_result.get('upper_trigram', '')
                kw = TRIGRAM_TO_KW.get((lower, upper))
                name = KW_TO_NAME.get(kw, '?')
                is_pure = "PURE" if name in PURE_HEXAGRAM_NAMES else "non-pure"
                print(f"    Result: {name} (#{kw}) [{is_pure}] lower={lower} upper={upper}")
        else:
            print(f"    Error: {after_error}")

        entry['after_result'] = after_result
        entry['after_error'] = after_error

        results.append(entry)

        # Rate limiting: small delay between cases
        if i < len(sampled) - 1:
            time.sleep(1)

    # Analyze
    stats = analyze_results(results)
    print_stats(stats)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_cases': len(results),
            'seed': 42,
            'method': '2-step trigram annotation',
        },
        'stats': {
            k: {kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in stats.items()
        },
        'results': results,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")

    return stats


def retry_failed(input_path, output_path=None):
    """Retry only failed cases from a previous run."""
    if output_path is None:
        output_path = input_path

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_cases = load_cases()
    case_map = {}
    for c in all_cases:
        tid = c.get('transition_id', '')
        if tid:
            case_map[tid] = c

    results = data.get('results', [])
    retry_count = 0
    success_count = 0

    for i, entry in enumerate(results):
        needs_retry = False
        tid = entry.get('transition_id', '')
        case = case_map.get(tid)
        if not case:
            continue

        for phase in ['before', 'after']:
            if entry.get(f'{phase}_error') is not None:
                needs_retry = True
                print(f"\n[Retry] {tid} ({phase}): {entry.get('target_name', '')[:50]}")
                prompt = build_before_prompt(case) if phase == 'before' else build_after_prompt(case)
                result, error = call_codex(prompt)

                if result:
                    is_valid, issues = validate_result(result, phase)
                    if is_valid:
                        entry[f'{phase}_result'] = result
                        entry[f'{phase}_error'] = None
                        lower = result.get('lower_trigram', '')
                        upper = result.get('upper_trigram', '')
                        kw = TRIGRAM_TO_KW.get((lower, upper))
                        name = KW_TO_NAME.get(kw, '?')
                        print(f"  SUCCESS: {name} (#{kw})")
                        success_count += 1
                    else:
                        entry[f'{phase}_error'] = f"Validation: {'; '.join(issues)}"
                        print(f"  Validation failed: {issues}")
                else:
                    entry[f'{phase}_error'] = error
                    print(f"  Still failed: {error[:100]}")

                time.sleep(2)

        if needs_retry:
            retry_count += 1

    print(f"\nRetried {retry_count} cases, {success_count} newly succeeded")

    # Re-analyze and save
    stats = analyze_results(results)
    print_stats(stats)

    data['stats'] = {
        k: {kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v
        for k, v in stats.items()
    }
    data['metadata']['retry_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Updated results saved to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Hexagram re-annotation with 2-step trigram method')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pilot', action='store_true', help='Pilot run with 20 cases')
    group.add_argument('--full', action='store_true', help='Full run with 500 cases')
    group.add_argument('--count', type=int, help='Custom number of cases')
    group.add_argument('--retry', type=str, help='Retry failed cases from a previous result file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    if args.retry:
        retry_failed(args.retry)
    elif args.full:
        run_reannotation(n_cases=500, output_path=FULL_OUTPUT)
    elif args.count:
        run_reannotation(n_cases=args.count, output_path=OUTPUT_DIR / f"reannotation_{args.count}.json")
    else:
        # Default: pilot with 20
        run_reannotation(n_cases=20, output_path=PILOT_OUTPUT)
