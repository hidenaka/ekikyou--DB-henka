#!/usr/bin/env python3
"""
Analyze 100-case reannotation results from 4 batch files.
Merges batches, computes statistics, and compares with baseline.
"""

import json
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE3_DIR = BASE_DIR / "analysis" / "phase3"

TRIGRAM_BITS = {
    '乾': '111', '坤': '000', '震': '001', '巽': '110',
    '坎': '010', '離': '101', '艮': '100', '兌': '011',
}

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

PURE_KW = {1, 2, 29, 30, 51, 52, 57, 58}


def hex_to_6bit(lower_tri, upper_tri):
    """Convert trigram pair to 6-bit string."""
    lb = TRIGRAM_BITS.get(lower_tri, '???')
    ub = TRIGRAM_BITS.get(upper_tri, '???')
    return lb + ub


def hamming(a, b):
    """Hamming distance between two bit strings."""
    return sum(x != y for x, y in zip(a, b))


def load_batches():
    """Load and merge all batch files."""
    all_results = []
    for i in range(1, 5):
        path = PHASE3_DIR / f"reannotation_batch{i}.json"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            batch = json.load(f)
        print(f"Loaded batch {i}: {len(batch)} cases")
        all_results.extend(batch)
    return all_results


def analyze(results):
    """Compute all statistics."""
    n = len(results)
    print(f"\n{'='*60}")
    print(f"REANNOTATION ANALYSIS (n={n})")
    print(f"{'='*60}")

    # 1. Pure hexagram rates
    before_pure = 0
    after_pure = 0
    before_kw_counts = Counter()
    after_kw_counts = Counter()
    before_lower_counts = Counter()
    before_upper_counts = Counter()
    after_lower_counts = Counter()
    after_upper_counts = Counter()
    hamming_dist = Counter()
    diagonal_count = 0
    even_parity = 0
    valid_transitions = 0

    for r in results:
        bl = r.get('before_lower', '')
        bu = r.get('before_upper', '')
        al = r.get('after_lower', '')
        au = r.get('after_upper', '')

        if not all([bl, bu, al, au]):
            continue

        # KW numbers
        bkw = TRIGRAM_TO_KW.get((bl, bu))
        akw = TRIGRAM_TO_KW.get((al, au))

        if bkw:
            before_kw_counts[bkw] += 1
            if bkw in PURE_KW:
                before_pure += 1
        if akw:
            after_kw_counts[akw] += 1
            if akw in PURE_KW:
                after_pure += 1

        before_lower_counts[bl] += 1
        before_upper_counts[bu] += 1
        after_lower_counts[al] += 1
        after_upper_counts[au] += 1

        # 6-bit analysis
        if bl in TRIGRAM_BITS and bu in TRIGRAM_BITS and al in TRIGRAM_BITS and au in TRIGRAM_BITS:
            b6 = hex_to_6bit(bl, bu)
            a6 = hex_to_6bit(al, au)
            hd = hamming(b6, a6)
            hamming_dist[hd] += 1
            valid_transitions += 1

            if hd % 2 == 0:
                even_parity += 1

            # Diagonal: delta_lower == delta_upper
            dl = hamming(TRIGRAM_BITS[bl], TRIGRAM_BITS[al])
            du = hamming(TRIGRAM_BITS[bu], TRIGRAM_BITS[au])
            if dl == du:
                diagonal_count += 1

    # Print results
    print(f"\n--- Pure Hexagram Rates ---")
    print(f"  Before: {before_pure}/{n} = {before_pure/n*100:.1f}% (baseline: 98.2%)")
    print(f"  After:  {after_pure}/{n} = {after_pure/n*100:.1f}% (baseline: 98.2%)")

    print(f"\n--- Diagonal Rate (Δ_lower == Δ_upper) ---")
    print(f"  {diagonal_count}/{valid_transitions} = {diagonal_count/valid_transitions*100:.1f}% (baseline: 97.8%)")

    print(f"\n--- Even Parity Rate ---")
    print(f"  {even_parity}/{valid_transitions} = {even_parity/valid_transitions*100:.1f}% (baseline: 98.6%)")

    print(f"\n--- Hamming Distance Distribution ---")
    for d in range(7):
        cnt = hamming_dist.get(d, 0)
        pct = cnt / valid_transitions * 100 if valid_transitions > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"  {d}: {cnt:4d} ({pct:5.1f}%) {bar}")

    # Baseline comparison
    baseline_hd = {0: 1113, 1: 12, 2: 3616, 3: 78, 4: 3376, 5: 38, 6: 995}
    baseline_total = sum(baseline_hd.values())
    print(f"\n  Baseline distribution (n={baseline_total}):")
    for d in range(7):
        cnt = baseline_hd.get(d, 0)
        pct = cnt / baseline_total * 100
        print(f"  {d}: {cnt:4d} ({pct:5.1f}%)")

    print(f"\n--- Trigram Frequency ---")
    print(f"  {'Trigram':8s} | {'B-Lower':8s} | {'B-Upper':8s} | {'A-Lower':8s} | {'A-Upper':8s}")
    print(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
    for tri in ['乾', '坤', '震', '巽', '坎', '離', '艮', '兌']:
        bl = before_lower_counts.get(tri, 0)
        bu = before_upper_counts.get(tri, 0)
        al = after_lower_counts.get(tri, 0)
        au = after_upper_counts.get(tri, 0)
        print(f"  {tri:8s} | {bl:4d} {bl/n*100:4.0f}% | {bu:4d} {bu/n*100:4.0f}% | {al:4d} {al/n*100:4.0f}% | {au:4d} {au/n*100:4.0f}%")

    print(f"\n--- Unique Hexagrams ---")
    print(f"  Before: {len(before_kw_counts)}/64")
    print(f"  After:  {len(after_kw_counts)}/64")

    # Top hexagrams
    print(f"\n  Before Top 10:")
    for kw, cnt in before_kw_counts.most_common(10):
        name = KW_TO_NAME.get(kw, '?')
        pure = " [PURE]" if kw in PURE_KW else ""
        print(f"    #{kw:2d} {name} : {cnt} ({cnt/n*100:.1f}%){pure}")

    print(f"\n  After Top 10:")
    for kw, cnt in after_kw_counts.most_common(10):
        name = KW_TO_NAME.get(kw, '?')
        pure = " [PURE]" if kw in PURE_KW else ""
        print(f"    #{kw:2d} {name} : {cnt} ({cnt/n*100:.1f}%){pure}")

    # Save summary
    summary = {
        'n_cases': n,
        'valid_transitions': valid_transitions,
        'before_pure_rate': round(before_pure / n * 100, 1),
        'after_pure_rate': round(after_pure / n * 100, 1),
        'diagonal_rate': round(diagonal_count / valid_transitions * 100, 1),
        'even_parity_rate': round(even_parity / valid_transitions * 100, 1),
        'hamming_distribution': {str(d): hamming_dist.get(d, 0) for d in range(7)},
        'before_unique_hexagrams': len(before_kw_counts),
        'after_unique_hexagrams': len(after_kw_counts),
        'before_kw_top10': [(kw, cnt) for kw, cnt in before_kw_counts.most_common(10)],
        'after_kw_top10': [(kw, cnt) for kw, cnt in after_kw_counts.most_common(10)],
        'trigram_freq': {
            tri: {
                'before_lower': before_lower_counts.get(tri, 0),
                'before_upper': before_upper_counts.get(tri, 0),
                'after_lower': after_lower_counts.get(tri, 0),
                'after_upper': after_upper_counts.get(tri, 0),
            } for tri in ['乾', '坤', '震', '巽', '坎', '離', '艮', '兌']
        },
        'baseline_comparison': {
            'before_pure_rate_baseline': 98.2,
            'after_pure_rate_baseline': 98.2,
            'diagonal_rate_baseline': 97.8,
            'even_parity_baseline': 98.6,
        }
    }

    out_path = PHASE3_DIR / "reannotation_100_analysis.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {out_path}")

    return summary


if __name__ == '__main__':
    results = load_batches()
    if results:
        analyze(results)
    else:
        print("No batch files found. Run reannotation first.")
