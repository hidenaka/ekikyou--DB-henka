#!/usr/bin/env python3
"""
Probabilistic 2-step trigram reannotation.

Instead of calling an LLM for each case, uses expert-designed probability
distributions P(lower_trigram|state) and P(upper_trigram|state) to sample
independent upper/lower trigrams. This approach:
1. Is reproducible (seeded)
2. Mirrors how an LLM would choose trigrams given state labels
3. Allows large-scale simulation (thousands of cases)
4. Tests whether independent trigram selection breaks diagonal structure

The probability tables are informed by:
- Traditional I Ching trigram semantics
- Pilot results from LLM-based annotation (n=15)
- State label semantics
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CASES_PATH = BASE_DIR / "data" / "raw" / "cases.jsonl"
OUTPUT_DIR = BASE_DIR / "analysis" / "phase3"

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
TRIGRAMS = ['乾', '坤', '震', '巽', '坎', '離', '艮', '兌']

# ── Expert-designed probability tables ──
# P(lower_trigram | state_label) - inner state
# P(upper_trigram | state_label) - outer situation
# Informed by traditional I Ching semantics + pilot LLM results

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
    # Normalize in case of rounding
    total = sum(probs)
    probs = [p / total for p in probs]
    return rng.choices(trigrams, weights=probs, k=1)[0]


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def hex_to_6bit(lower, upper):
    return TRIGRAM_BITS[lower] + TRIGRAM_BITS[upper]


def run_simulation(n_sims=1000, seed=42):
    """Run probabilistic reannotation on all valid cases."""
    # Load cases
    cases = []
    with open(CASES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    # Filter to valid cases (have both before and after state)
    valid = [c for c in cases if c.get('before_state') and c.get('after_state')
             and c['before_state'] in BEFORE_LOWER and c['after_state'] in AFTER_LOWER]
    print(f"Total cases: {len(cases)}, Valid for simulation: {len(valid)}")

    rng = random.Random(seed)

    # Run multiple simulations
    all_diagonal_rates = []
    all_even_parity_rates = []
    all_pure_before_rates = []
    all_pure_after_rates = []
    all_hamming_dists = Counter()

    for sim in range(n_sims):
        diagonal = 0
        even_parity = 0
        before_pure = 0
        after_pure = 0

        for c in valid:
            bs = c['before_state']
            ats = c['after_state']

            # Sample trigrams independently
            bl = sample_trigram(BEFORE_LOWER[bs], rng)
            bu = sample_trigram(BEFORE_UPPER[bs], rng)
            al = sample_trigram(AFTER_LOWER[ats], rng)
            au = sample_trigram(AFTER_UPPER[ats], rng)

            # Check pure
            bkw = TRIGRAM_TO_KW.get((bl, bu))
            akw = TRIGRAM_TO_KW.get((al, au))
            if bkw in PURE_KW:
                before_pure += 1
            if akw in PURE_KW:
                after_pure += 1

            # 6-bit analysis
            b6 = hex_to_6bit(bl, bu)
            a6 = hex_to_6bit(al, au)
            hd = hamming(b6, a6)

            if sim == 0:
                all_hamming_dists[hd] += 1

            if hd % 2 == 0:
                even_parity += 1

            dl = hamming(TRIGRAM_BITS[bl], TRIGRAM_BITS[al])
            du = hamming(TRIGRAM_BITS[bu], TRIGRAM_BITS[au])
            if dl == du:
                diagonal += 1

        n_valid = len(valid)
        all_diagonal_rates.append(diagonal / n_valid)
        all_even_parity_rates.append(even_parity / n_valid)
        all_pure_before_rates.append(before_pure / n_valid)
        all_pure_after_rates.append(after_pure / n_valid)

    # Statistics
    def stats(values):
        values.sort()
        mean = sum(values) / len(values)
        ci_lo = values[int(len(values) * 0.025)]
        ci_hi = values[int(len(values) * 0.975)]
        return mean, ci_lo, ci_hi

    diag_mean, diag_lo, diag_hi = stats(all_diagonal_rates)
    even_mean, even_lo, even_hi = stats(all_even_parity_rates)
    bpure_mean, bpure_lo, bpure_hi = stats(all_pure_before_rates)
    apure_mean, apure_lo, apure_hi = stats(all_pure_after_rates)

    print(f"\n{'='*60}")
    print(f"PROBABILISTIC REANNOTATION ({n_sims} simulations, n={len(valid)} cases)")
    print(f"{'='*60}")
    print(f"\n--- Key Metrics (mean [95% CI]) ---")
    print(f"  Diagonal rate:      {diag_mean*100:5.1f}% [{diag_lo*100:.1f}%, {diag_hi*100:.1f}%]  (baseline: 97.8%)")
    print(f"  Even parity rate:   {even_mean*100:5.1f}% [{even_lo*100:.1f}%, {even_hi*100:.1f}%]  (baseline: 98.6%)")
    print(f"  Before pure rate:   {bpure_mean*100:5.1f}% [{bpure_lo*100:.1f}%, {bpure_hi*100:.1f}%]  (baseline: 98.2%)")
    print(f"  After pure rate:    {apure_mean*100:5.1f}% [{apure_lo*100:.1f}%, {apure_hi*100:.1f}%]  (baseline: 98.2%)")

    print(f"\n--- Hamming Distance Distribution (1st sim, n={len(valid)}) ---")
    total_hd = sum(all_hamming_dists.values())
    baseline_hd = {0: 1113, 1: 12, 2: 3616, 3: 78, 4: 3376, 5: 38, 6: 995}
    baseline_total = sum(baseline_hd.values())
    print(f"  {'Dist':4s} | {'Reannotation':>12s} | {'Baseline':>12s}")
    print(f"  {'----':4s} | {'------------':>12s} | {'--------':>12s}")
    for d in range(7):
        r_cnt = all_hamming_dists.get(d, 0)
        r_pct = r_cnt / total_hd * 100
        b_cnt = baseline_hd.get(d, 0)
        b_pct = b_cnt / baseline_total * 100
        print(f"  {d:4d} | {r_cnt:5d} ({r_pct:5.1f}%) | {b_cnt:5d} ({b_pct:5.1f}%)")

    # Random baseline for comparison
    print(f"\n--- Random Baseline (uniform trigram selection) ---")
    rng2 = random.Random(99)
    rand_diag = 0
    rand_even = 0
    rand_pure_b = 0
    rand_pure_a = 0
    n_rand = len(valid)
    for _ in range(n_rand):
        bl = rng2.choice(TRIGRAMS)
        bu = rng2.choice(TRIGRAMS)
        al = rng2.choice(TRIGRAMS)
        au = rng2.choice(TRIGRAMS)
        bkw = TRIGRAM_TO_KW.get((bl, bu))
        akw = TRIGRAM_TO_KW.get((al, au))
        if bkw in PURE_KW: rand_pure_b += 1
        if akw in PURE_KW: rand_pure_a += 1
        b6 = hex_to_6bit(bl, bu)
        a6 = hex_to_6bit(al, au)
        hd = hamming(b6, a6)
        if hd % 2 == 0: rand_even += 1
        dl = hamming(TRIGRAM_BITS[bl], TRIGRAM_BITS[al])
        du = hamming(TRIGRAM_BITS[bu], TRIGRAM_BITS[au])
        if dl == du: rand_diag += 1

    print(f"  Diagonal rate:    {rand_diag/n_rand*100:.1f}%")
    print(f"  Even parity:     {rand_even/n_rand*100:.1f}%")
    print(f"  Pure before:     {rand_pure_b/n_rand*100:.1f}%")
    print(f"  Pure after:      {rand_pure_a/n_rand*100:.1f}%")

    # Save results
    results = {
        'metadata': {
            'n_simulations': n_sims,
            'n_cases': len(valid),
            'seed': seed,
            'method': 'probabilistic_2step_trigram',
        },
        'reannotation': {
            'diagonal_rate': {'mean': round(diag_mean, 4), 'ci_lo': round(diag_lo, 4), 'ci_hi': round(diag_hi, 4)},
            'even_parity_rate': {'mean': round(even_mean, 4), 'ci_lo': round(even_lo, 4), 'ci_hi': round(even_hi, 4)},
            'before_pure_rate': {'mean': round(bpure_mean, 4), 'ci_lo': round(bpure_lo, 4), 'ci_hi': round(bpure_hi, 4)},
            'after_pure_rate': {'mean': round(apure_mean, 4), 'ci_lo': round(apure_lo, 4), 'ci_hi': round(apure_hi, 4)},
        },
        'baseline': {
            'diagonal_rate': 0.978,
            'even_parity_rate': 0.986,
            'before_pure_rate': 0.982,
            'after_pure_rate': 0.982,
        },
        'hamming_distribution': {str(d): all_hamming_dists.get(d, 0) for d in range(7)},
    }

    out_path = OUTPUT_DIR / "reannotation_probabilistic.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == '__main__':
    run_simulation(n_sims=1000, seed=42)
