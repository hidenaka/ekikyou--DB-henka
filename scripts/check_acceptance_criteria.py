#!/usr/bin/env python3
"""
check_acceptance_criteria.py — 裁定済みアノテーションの受入基準チェック

基準:
1. ≥60/64 hexagrams が非ゼロ
2. 最小サポート ≥5
3. 平均出次数 ≥3
4. 単一hexagram ≤15%

King Wen sequence で (lower_trigram, upper_trigram) → hexagram number にマッピング
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_SET_DIR = PROJECT_ROOT / "analysis" / "gold_set"

# King Wen mapping: (lower, upper) -> hexagram number
# Trigram order: 乾=0, 坤=1, 震=2, 巽=3, 坎=4, 離=5, 艮=6, 兌=7
TRIGRAM_INDEX = {"乾": 0, "坤": 1, "震": 2, "巽": 3, "坎": 4, "離": 5, "艮": 6, "兌": 7}

# King Wen sequence: KING_WEN[lower][upper] = hexagram number
KING_WEN = [
    # upper:  乾  坤  震  巽  坎  離  艮  兌
    [1,  11, 34, 9,  5,  14, 26, 43],  # lower=乾
    [12, 2,  16, 20, 8,  35, 23, 45],  # lower=坤
    [25, 24, 51, 42, 3,  21, 27, 17],  # lower=震
    [44, 46, 32, 57, 48, 50, 18, 28],  # lower=巽
    [6,  7,  40, 59, 29, 64, 4,  47],  # lower=坎
    [13, 36, 55, 37, 63, 30, 22, 49],  # lower=離
    [33, 15, 62, 53, 39, 56, 52, 31],  # lower=艮
    [10, 19, 54, 61, 60, 38, 41, 58],  # lower=兌
]


def trigram_to_hex(lower: str, upper: str) -> int:
    """Convert trigram pair to King Wen hexagram number."""
    li = TRIGRAM_INDEX.get(lower)
    ui = TRIGRAM_INDEX.get(upper)
    if li is None or ui is None:
        return -1
    return KING_WEN[li][ui]


def main():
    path = GOLD_SET_DIR / "adjudicated_annotations.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data["annotations"]
    print(f"Loaded {len(annotations)} adjudicated annotations\n")

    # Map to hexagrams
    before_hexagrams = []
    after_hexagrams = []
    transitions = []
    errors = 0

    for ann in annotations:
        bl = ann.get("before_lower", "")
        bu = ann.get("before_upper", "")
        al = ann.get("after_lower", "")
        au = ann.get("after_upper", "")

        bh = trigram_to_hex(bl, bu)
        ah = trigram_to_hex(al, au)

        if bh < 0 or ah < 0:
            errors += 1
            continue

        before_hexagrams.append(bh)
        after_hexagrams.append(ah)
        transitions.append((bh, ah))

    print(f"Valid transitions: {len(transitions)}, Errors: {errors}")

    # Criterion 1: Active hexagrams
    all_hex = set(before_hexagrams) | set(after_hexagrams)
    active_count = len(all_hex)
    crit1_pass = active_count >= 60

    # Criterion 2: Min support
    hex_counts = Counter(before_hexagrams) + Counter(after_hexagrams)
    min_support = min(hex_counts.values()) if hex_counts else 0
    crit2_pass = min_support >= 5

    # Criterion 3: Mean out-degree
    out_edges = defaultdict(set)
    for bh, ah in transitions:
        out_edges[bh].add(ah)
    mean_out_degree = sum(len(v) for v in out_edges.values()) / len(out_edges) if out_edges else 0
    crit3_pass = mean_out_degree >= 3

    # Criterion 4: Max concentration
    total_appearances = sum(hex_counts.values())
    max_hex = max(hex_counts.values()) if hex_counts else 0
    max_concentration = max_hex / total_appearances * 100
    crit4_pass = max_concentration <= 15

    # Report
    print(f"\n{'='*60}")
    print("  Acceptance Criteria Check")
    print(f"{'='*60}")
    print(f"  1. Active hexagrams: {active_count}/64 (need ≥60) {'✅' if crit1_pass else '❌'}")
    print(f"  2. Min support: {min_support} (need ≥5) {'✅' if crit2_pass else '❌'}")
    print(f"  3. Mean out-degree: {mean_out_degree:.1f} (need ≥3) {'✅' if crit3_pass else '❌'}")
    print(f"  4. Max concentration: {max_concentration:.1f}% (need ≤15%) {'✅' if crit4_pass else '❌'}")

    all_pass = crit1_pass and crit2_pass and crit3_pass and crit4_pass
    print(f"\n  Overall: {'PASS ✅' if all_pass else 'FAIL ❌'}")
    print(f"{'='*60}")

    # Hexagram distribution details
    print(f"\n  Before hexagram distribution (top 10):")
    bc = Counter(before_hexagrams)
    for h, c in bc.most_common(10):
        print(f"    Hex {h:2d}: {c} ({c/len(before_hexagrams)*100:.1f}%)")

    print(f"\n  After hexagram distribution (top 10):")
    ac = Counter(after_hexagrams)
    for h, c in ac.most_common(10):
        print(f"    Hex {h:2d}: {c} ({c/len(after_hexagrams)*100:.1f}%)")

    # Missing hexagrams
    missing = set(range(1, 65)) - all_hex
    if missing:
        print(f"\n  Missing hexagrams ({len(missing)}): {sorted(missing)}")
    else:
        print(f"\n  All 64 hexagrams represented!")

    # Low-support hexagrams
    low_support = [(h, hex_counts[h]) for h in range(1, 65) if hex_counts.get(h, 0) < 5]
    if low_support:
        print(f"\n  Low-support hexagrams (<5): {len(low_support)}")
        for h, c in sorted(low_support, key=lambda x: x[1]):
            print(f"    Hex {h:2d}: {c}")

    # Save transition data for Phase 3
    transition_data = {
        "metadata": {
            "n_transitions": len(transitions),
            "n_active_hexagrams": active_count,
            "min_support": min_support,
            "mean_out_degree": round(mean_out_degree, 2),
            "max_concentration_pct": round(max_concentration, 2),
            "acceptance_passed": all_pass,
        },
        "transitions": [{"before": bh, "after": ah} for bh, ah in transitions],
        "before_distribution": dict(bc),
        "after_distribution": dict(ac),
    }

    out_path = GOLD_SET_DIR / "hexagram_transitions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transition_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Transition data saved: {out_path}")


if __name__ == "__main__":
    main()
