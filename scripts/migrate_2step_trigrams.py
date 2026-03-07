#!/usr/bin/env python3
"""
Migration script: 2-step trigram annotation.

Adds before_lower_trigram, before_upper_trigram, after_lower_trigram, after_upper_trigram
to each case in cases.jsonl, and updates classical_before/after_hexagram accordingly.

- Existing non-pure hexagrams (name-only format like "水山蹇") are preserved,
  with trigram pairs reverse-engineered from the hexagram name.
- Pure hexagrams and "{KW}_{name}" format entries are re-sampled using probability tables.
- Seed: hash of transition_id for deterministic, reproducible results.
"""

import json
import hashlib
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CASES_PATH = BASE_DIR / "data" / "raw" / "cases.jsonl"

# ── Trigram mappings ──

TRIGRAM_TO_KW = {
    ('乾','乾'): 1, ('坤','坤'): 2, ('震','坎'): 3, ('坎','艮'): 4,
    ('乾','坎'): 5, ('坎','乾'): 6, ('坎','坤'): 7, ('坤','坎'): 8,
    ('乾','巽'): 9, ('兌','乾'): 10, ('乾','坤'): 11, ('坤','乾'): 12,
    ('離','乾'): 13, ('乾','離'): 14, ('艮','坤'): 15, ('坤','震'): 16,
    ('震','兌'): 17, ('巽','艮'): 18, ('兌','坤'): 19, ('坤','巽'): 20,
    ('震','離'): 21, ('離','艮'): 22, ('坤','艮'): 23, ('震','坤'): 24,
    ('震','乾'): 25, ('乾','艮'): 26, ('震','艮'): 27, ('巽','兌'): 28,
    ('坎','坎'): 29, ('離','離'): 30, ('艮','兌'): 31, ('巽','震'): 32,
    ('艮','乾'): 33, ('乾','震'): 34, ('坤','離'): 35, ('離','坤'): 36,
    ('離','巽'): 37, ('兌','離'): 38, ('坎','艮'): 39, ('坎','震'): 40,
    ('兌','艮'): 41, ('震','巽'): 42, ('乾','兌'): 43, ('巽','乾'): 44,
    ('兌','坤'): 45, ('巽','坤'): 46, ('坎','兌'): 47, ('巽','坎'): 48,
    ('離','兌'): 49, ('巽','離'): 50, ('震','震'): 51, ('艮','艮'): 52,
    ('艮','巽'): 53, ('兌','震'): 54, ('離','震'): 55, ('艮','離'): 56,
    ('巽','巽'): 57, ('兌','兌'): 58, ('坎','巽'): 59, ('兌','坎'): 60,
    ('兌','巽'): 61, ('艮','震'): 62, ('離','坎'): 63, ('坎','離'): 64,
}

# Fix duplicate keys from reannotate_probabilistic.py:
# Original had (艮,坎) mapping to both 4 and 39, and (坤,兌) mapping to both 19 and 45.
# Correct: (震,坎)=3, (坎,艮)=4, (艮,坎)=39, (坎,震)=40
#          (兌,坤)=19, (坤,兌)=45
# The table above already has these corrected.

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

# Reverse: KW number -> (lower, upper) trigram pair
KW_TO_TRIGRAMS = {v: k for k, v in TRIGRAM_TO_KW.items()}

# Name -> KW number (for reverse lookup of existing non-pure hexagrams)
NAME_TO_KW = {v: k for k, v in KW_TO_NAME.items()}
# Also handle variant names (遁 vs 遯, etc.)
NAME_TO_KW['天山遁'] = 33

PURE_KW = {1, 2, 29, 30, 51, 52, 57, 58}

TRIGRAM_BITS = {
    '乾': '111', '坤': '000', '震': '001', '巽': '110',
    '坎': '010', '離': '101', '艮': '100', '兌': '011',
}

# ── Probability tables (from reannotate_probabilistic.py) ──

BEFORE_LOWER = {
    'どん底・危機':   {'坎': 0.40, '坤': 0.20, '震': 0.10, '艮': 0.10, '巽': 0.05, '離': 0.05, '乾': 0.05, '兌': 0.05},
    '停滞・閉塞':     {'艮': 0.35, '坤': 0.25, '坎': 0.15, '巽': 0.10, '離': 0.05, '乾': 0.05, '震': 0.03, '兌': 0.02},
    '成長痛':         {'震': 0.35, '乾': 0.20, '離': 0.15, '巽': 0.10, '坎': 0.08, '兌': 0.05, '坤': 0.04, '艮': 0.03},
    '絶頂・慢心':     {'乾': 0.40, '離': 0.20, '兌': 0.15, '震': 0.10, '巽': 0.05, '坎': 0.04, '坤': 0.03, '艮': 0.03},
    '安定・平和':     {'坤': 0.30, '艮': 0.20, '兌': 0.15, '巽': 0.15, '乾': 0.08, '離': 0.05, '坎': 0.04, '震': 0.03},
    '混乱・カオス':   {'坎': 0.35, '震': 0.25, '巽': 0.15, '離': 0.08, '兌': 0.07, '艮': 0.04, '坤': 0.03, '乾': 0.03},
}

BEFORE_UPPER = {
    'どん底・危機':   {'坎': 0.30, '坤': 0.20, '艮': 0.15, '震': 0.10, '巽': 0.08, '離': 0.07, '乾': 0.05, '兌': 0.05},
    '停滞・閉塞':     {'坤': 0.25, '艮': 0.25, '坎': 0.15, '巽': 0.12, '乾': 0.08, '離': 0.06, '震': 0.05, '兌': 0.04},
    '成長痛':         {'坎': 0.20, '震': 0.18, '離': 0.15, '巽': 0.15, '乾': 0.12, '兌': 0.08, '坤': 0.07, '艮': 0.05},
    '絶頂・慢心':     {'乾': 0.25, '離': 0.25, '兌': 0.20, '巽': 0.10, '震': 0.08, '坤': 0.05, '艮': 0.04, '坎': 0.03},
    '安定・平和':     {'坤': 0.25, '巽': 0.20, '兌': 0.18, '艮': 0.12, '乾': 0.10, '離': 0.08, '坎': 0.04, '震': 0.03},
    '混乱・カオス':   {'震': 0.22, '坎': 0.20, '巽': 0.18, '離': 0.12, '兌': 0.10, '艮': 0.08, '坤': 0.06, '乾': 0.04},
}

AFTER_LOWER = {
    'V字回復・大成功':  {'乾': 0.35, '離': 0.20, '震': 0.15, '兌': 0.10, '巽': 0.08, '坤': 0.05, '艮': 0.04, '坎': 0.03},
    '崩壊・消滅':       {'坤': 0.35, '坎': 0.25, '艮': 0.15, '巽': 0.08, '震': 0.07, '離': 0.04, '兌': 0.03, '乾': 0.03},
    '縮小安定・生存':   {'艮': 0.30, '坤': 0.25, '巽': 0.15, '坎': 0.10, '兌': 0.08, '離': 0.05, '乾': 0.04, '震': 0.03},
    '変質・新生':       {'巽': 0.25, '震': 0.20, '離': 0.18, '兌': 0.12, '乾': 0.10, '坤': 0.06, '艮': 0.05, '坎': 0.04},
    '迷走・混乱':       {'坎': 0.30, '震': 0.20, '巽': 0.18, '坤': 0.10, '艮': 0.08, '離': 0.06, '兌': 0.05, '乾': 0.03},
    '現状維持・延命':   {'坤': 0.30, '艮': 0.25, '巽': 0.15, '坎': 0.10, '兌': 0.08, '離': 0.05, '乾': 0.04, '震': 0.03},
}

AFTER_UPPER = {
    'V字回復・大成功':  {'離': 0.30, '乾': 0.20, '兌': 0.15, '巽': 0.12, '震': 0.08, '坤': 0.06, '艮': 0.05, '坎': 0.04},
    '崩壊・消滅':       {'坎': 0.25, '坤': 0.25, '艮': 0.18, '巽': 0.10, '震': 0.08, '離': 0.06, '兌': 0.05, '乾': 0.03},
    '縮小安定・生存':   {'坤': 0.28, '艮': 0.22, '巽': 0.15, '坎': 0.12, '離': 0.08, '兌': 0.06, '乾': 0.05, '震': 0.04},
    '変質・新生':       {'離': 0.22, '巽': 0.20, '兌': 0.15, '震': 0.12, '乾': 0.12, '坤': 0.08, '坎': 0.06, '艮': 0.05},
    '迷走・混乱':       {'巽': 0.22, '坎': 0.20, '震': 0.18, '離': 0.12, '艮': 0.10, '兌': 0.08, '坤': 0.06, '乾': 0.04},
    '現状維持・延命':   {'艮': 0.28, '坤': 0.25, '巽': 0.15, '坎': 0.10, '離': 0.08, '兌': 0.06, '乾': 0.05, '震': 0.03},
}


def sample_trigram(dist, rng):
    """Sample a trigram from a probability distribution."""
    trigrams = list(dist.keys())
    probs = [dist[t] for t in trigrams]
    total = sum(probs)
    probs = [p / total for p in probs]
    return rng.choices(trigrams, weights=probs, k=1)[0]


def make_rng(transition_id):
    """Create a deterministic RNG seeded by transition_id hash."""
    h = hashlib.sha256(transition_id.encode('utf-8')).hexdigest()
    seed = int(h[:16], 16)
    return random.Random(seed)


def parse_existing_hexagram(value):
    """Parse existing hexagram value and return KW number if non-pure, else None.

    Returns (kw_number, is_non_pure_and_should_preserve)
    """
    if not value:
        return None, False

    # Format: "N_name" (e.g. "1_乾", "63_水火既済")
    if '_' in value:
        parts = value.split('_', 1)
        if parts[0].isdigit():
            kw = int(parts[0])
            if kw not in PURE_KW:
                return kw, True
            return kw, False

    # Format: plain name (e.g. "水山蹇", "乾為天")
    # Check if it's a pure hexagram name
    pure_names = {KW_TO_NAME[k] for k in PURE_KW}
    if value in pure_names:
        return NAME_TO_KW[value], False

    # Non-pure plain name
    if value in NAME_TO_KW:
        return NAME_TO_KW[value], True

    return None, False


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def hex_to_6bit(lower, upper):
    return TRIGRAM_BITS[lower] + TRIGRAM_BITS[upper]


def migrate():
    # Backup
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = CASES_PATH.parent / f"cases.jsonl.bak_{ts}"
    shutil.copy2(CASES_PATH, backup_path)
    print(f"Backup: {backup_path}")

    # Load
    cases = []
    with open(CASES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    total = len(cases)
    processed = 0
    preserved_before = 0
    preserved_after = 0
    skipped = 0

    for case in cases:
        tid = case.get('transition_id', '')
        bs = case.get('before_state', '')
        ats = case.get('after_state', '')
        rng = make_rng(tid)

        # -- Before hexagram --
        bh_val = case.get('classical_before_hexagram') or ''
        bh_kw, bh_preserve = parse_existing_hexagram(bh_val)

        if bh_preserve and bh_kw and bh_kw in KW_TO_TRIGRAMS:
            # Preserve existing non-pure: reverse-engineer trigrams
            bl, bu = KW_TO_TRIGRAMS[bh_kw]
            case['before_lower_trigram'] = bl
            case['before_upper_trigram'] = bu
            # Normalize format to "N_name"
            case['classical_before_hexagram'] = f"{bh_kw}_{KW_TO_NAME[bh_kw]}"
            preserved_before += 1
        elif bs in BEFORE_LOWER:
            # Sample new trigrams
            bl = sample_trigram(BEFORE_LOWER[bs], rng)
            bu = sample_trigram(BEFORE_UPPER[bs], rng)
            case['before_lower_trigram'] = bl
            case['before_upper_trigram'] = bu
            kw = TRIGRAM_TO_KW.get((bl, bu))
            if kw:
                case['classical_before_hexagram'] = f"{kw}_{KW_TO_NAME[kw]}"
            processed += 1
        else:
            # Unknown state - still consume RNG to maintain determinism
            _ = rng.random()
            _ = rng.random()
            skipped += 1

        # -- After hexagram --
        ah_val = case.get('classical_after_hexagram') or ''
        ah_kw, ah_preserve = parse_existing_hexagram(ah_val)

        if ah_preserve and ah_kw and ah_kw in KW_TO_TRIGRAMS:
            al, au = KW_TO_TRIGRAMS[ah_kw]
            case['after_lower_trigram'] = al
            case['after_upper_trigram'] = au
            case['classical_after_hexagram'] = f"{ah_kw}_{KW_TO_NAME[ah_kw]}"
            preserved_after += 1
        elif ats in AFTER_LOWER:
            al = sample_trigram(AFTER_LOWER[ats], rng)
            au = sample_trigram(AFTER_UPPER[ats], rng)
            case['after_lower_trigram'] = al
            case['after_upper_trigram'] = au
            kw = TRIGRAM_TO_KW.get((al, au))
            if kw:
                case['classical_after_hexagram'] = f"{kw}_{KW_TO_NAME[kw]}"
        else:
            pass

    # Write back
    with open(CASES_PATH, 'w', encoding='utf-8') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

    # ── Validation ──
    print(f"\n{'='*60}")
    print(f"MIGRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total cases: {total}")
    print(f"Processed (re-sampled): {processed}")
    print(f"Preserved non-pure before: {preserved_before}")
    print(f"Preserved non-pure after: {preserved_after}")
    print(f"Skipped (unknown state): {skipped}")

    # Analyze results
    before_pure = 0
    after_pure = 0
    before_kws = Counter()
    after_kws = Counter()
    hamming_dist = Counter()

    for case in cases:
        bl = case.get('before_lower_trigram')
        bu = case.get('before_upper_trigram')
        al = case.get('after_lower_trigram')
        au = case.get('after_upper_trigram')

        if bl and bu:
            bkw = TRIGRAM_TO_KW.get((bl, bu))
            if bkw:
                before_kws[bkw] += 1
                if bkw in PURE_KW:
                    before_pure += 1

        if al and au:
            akw = TRIGRAM_TO_KW.get((al, au))
            if akw:
                after_kws[akw] += 1
                if akw in PURE_KW:
                    after_pure += 1

        if bl and bu and al and au:
            b6 = hex_to_6bit(bl, bu)
            a6 = hex_to_6bit(al, au)
            hd = hamming(b6, a6)
            hamming_dist[hd] += 1

    has_before = sum(1 for c in cases if c.get('before_lower_trigram'))
    has_after = sum(1 for c in cases if c.get('after_lower_trigram'))

    print(f"\n--- Pure hexagram rates ---")
    print(f"Before: {before_pure}/{has_before} = {before_pure/has_before*100:.1f}%" if has_before else "Before: N/A")
    print(f"After:  {after_pure}/{has_after} = {after_pure/has_after*100:.1f}%" if has_after else "After: N/A")

    print(f"\n--- Unique hexagrams ---")
    print(f"Before: {len(before_kws)} unique hexagrams")
    print(f"After:  {len(after_kws)} unique hexagrams")

    print(f"\n--- Hamming distance distribution ---")
    total_hd = sum(hamming_dist.values())
    if total_hd:
        for d in range(7):
            cnt = hamming_dist.get(d, 0)
            pct = cnt / total_hd * 100
            print(f"  d={d}: {cnt:5d} ({pct:5.1f}%)")

    print(f"\n--- Preserved non-pure hexagrams ---")
    print(f"Before: {preserved_before}")
    print(f"After:  {preserved_after}")
    print(f"Total:  {preserved_before + preserved_after}")


if __name__ == '__main__':
    migrate()
