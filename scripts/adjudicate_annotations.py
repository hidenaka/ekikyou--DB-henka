#!/usr/bin/env python3
"""
adjudicate_annotations.py — Pass 1/2 の不一致を裁定し最終アノテーションを生成

裁定ルール:
1. 両パス一致 → そのまま採用
2. 不一致時:
   a. 信頼度が高い方を採用 (high > medium > low)
   b. 同じ信頼度 → Pass 1（分析的・エビデンスベース）を採用
3. Pass 2のみに存在するケース（140件）→ Pass 2をそのまま採用

出力: analysis/gold_set/adjudicated_annotations.json
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_SET_DIR = PROJECT_ROOT / "analysis" / "gold_set"

FIELDS = ["before_lower", "before_upper", "after_lower", "after_upper"]
CONFIDENCE_ORDER = {"high": 3, "medium": 2, "low": 1, "": 0}


def load_annotations(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_id = {}
    for ann in data.get("annotations", []):
        tid = ann.get("transition_id", "")
        if tid:
            by_id[tid] = ann
    return by_id


def adjudicate_field(p1_val, p1_conf, p2_val, p2_conf):
    """Return (chosen_value, source, was_agreement)."""
    if p1_val == p2_val:
        return p1_val, "agree", True

    p1_score = CONFIDENCE_ORDER.get(p1_conf, 0)
    p2_score = CONFIDENCE_ORDER.get(p2_conf, 0)

    if p1_score > p2_score:
        return p1_val, "pass1_higher_conf", False
    elif p2_score > p1_score:
        return p2_val, "pass2_higher_conf", False
    else:
        return p1_val, "pass1_tiebreak", False


def main():
    pass1_path = GOLD_SET_DIR / "annotations_pass1.json"
    pass2_path = GOLD_SET_DIR / "annotations_pass2.json"

    pass1 = load_annotations(pass1_path)
    pass2 = load_annotations(pass2_path)

    print(f"Pass 1: {len(pass1)} annotations")
    print(f"Pass 2: {len(pass2)} annotations")

    all_ids = sorted(set(pass1.keys()) | set(pass2.keys()))
    common_ids = sorted(set(pass1.keys()) & set(pass2.keys()))
    pass2_only = sorted(set(pass2.keys()) - set(pass1.keys()))

    print(f"Common: {len(common_ids)}, Pass2-only: {len(pass2_only)}, Total: {len(all_ids)}")

    adjudicated = []
    stats = {
        "agree": 0,
        "pass1_higher_conf": 0,
        "pass2_higher_conf": 0,
        "pass1_tiebreak": 0,
        "pass2_only": 0,
    }
    field_stats = {f: Counter() for f in FIELDS}

    # Common cases: adjudicate
    for tid in common_ids:
        p1 = pass1[tid]
        p2 = pass2[tid]
        result = {"transition_id": tid}

        for field in FIELDS:
            p1_val = p1.get(field, "")
            p1_conf = p1.get(f"{field}_confidence", "")
            p2_val = p2.get(field, "")
            p2_conf = p2.get(f"{field}_confidence", "")

            chosen, source, was_agree = adjudicate_field(p1_val, p1_conf, p2_val, p2_conf)
            result[field] = chosen
            result[f"{field}_confidence"] = p1_conf if "pass1" in source or source == "agree" else p2_conf
            result[f"{field}_source"] = source

            stats[source] += 1
            field_stats[field][source] += 1

        adjudicated.append(result)

    # Pass 2-only cases
    for tid in pass2_only:
        p2 = pass2[tid]
        result = {"transition_id": tid}
        for field in FIELDS:
            result[field] = p2.get(field, "")
            result[f"{field}_confidence"] = p2.get(f"{field}_confidence", "")
            result[f"{field}_source"] = "pass2_only"
            stats["pass2_only"] += 1
            field_stats[field]["pass2_only"] += 1
        adjudicated.append(result)

    # Summary
    total_decisions = sum(stats.values())
    print(f"\n{'='*60}")
    print("  Adjudication Summary")
    print(f"{'='*60}")
    print(f"  Total cases:     {len(adjudicated)}")
    print(f"  Total decisions: {total_decisions} (4 fields x {len(adjudicated)} cases)")
    print(f"  Agreed:          {stats['agree']} ({stats['agree']/total_decisions*100:.1f}%)")
    print(f"  Pass1 higher:    {stats['pass1_higher_conf']} ({stats['pass1_higher_conf']/total_decisions*100:.1f}%)")
    print(f"  Pass2 higher:    {stats['pass2_higher_conf']} ({stats['pass2_higher_conf']/total_decisions*100:.1f}%)")
    print(f"  Pass1 tiebreak:  {stats['pass1_tiebreak']} ({stats['pass1_tiebreak']/total_decisions*100:.1f}%)")
    print(f"  Pass2 only:      {stats['pass2_only']} ({stats['pass2_only']/total_decisions*100:.1f}%)")

    print(f"\n  Per-field breakdown:")
    for field in FIELDS:
        fs = field_stats[field]
        n = sum(fs.values())
        agree_pct = fs["agree"] / n * 100 if n else 0
        print(f"    {field}: agree={fs['agree']}({agree_pct:.0f}%) "
              f"p1={fs['pass1_higher_conf']+fs['pass1_tiebreak']} "
              f"p2={fs['pass2_higher_conf']+fs['pass2_only']}")

    # Trigram distribution
    print(f"\n  Final trigram distribution:")
    for field in FIELDS:
        dist = Counter(a[field] for a in adjudicated)
        print(f"    {field}: {dict(sorted(dist.items()))}")

    # Write output
    output = {
        "metadata": {
            "total_cases": len(adjudicated),
            "common_adjudicated": len(common_ids),
            "pass2_only": len(pass2_only),
            "adjudication_stats": stats,
        },
        "annotations": adjudicated,
    }

    out_path = GOLD_SET_DIR / "adjudicated_annotations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  -> {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
