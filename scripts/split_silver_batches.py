#!/usr/bin/env python3
"""
split_silver_batches.py — Silver Set バッチ分割スクリプト

800件の選定事例を25件ずつ32バッチに分割する。
各バッチの先頭に5件のキャリブレーションアンカーを挿入
（アノテーターは30件処理、うち先頭5件がアンカー）。

層別化:
- story_summary長さ（短/中/長を各バッチに混在）
- domain（可能な限り多様なドメインを各バッチに分配）

Usage:
    python3 scripts/split_silver_batches.py

Output:
    analysis/gold_set/batches/batch_001.json ~ batch_032.json
"""

import json
import os
import sys
import math
from collections import defaultdict

# --- Paths ---

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES_PATH = os.path.join(
    PROJECT_ROOT, "analysis", "gold_set", "selected_800_cases.json"
)
ANCHORS_PATH = os.path.join(
    PROJECT_ROOT, "analysis", "gold_set", "calibration_anchors.json"
)
BATCHES_DIR = os.path.join(PROJECT_ROOT, "analysis", "gold_set", "batches")

BATCH_SIZE = 25


def load_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def length_tier(story_length: int) -> str:
    if story_length <= 80:
        return "short"
    elif story_length <= 130:
        return "medium"
    return "long"


def stratified_split(cases: list, n_batches: int) -> list:
    """
    Split cases into n_batches with stratification by domain and length.
    Round-robin by domain (largest first) to maximize diversity per batch.
    """
    # Tag each case with length tier
    for c in cases:
        c["_tier"] = length_tier(c.get("story_length", 100))

    # Group by domain
    domain_groups = defaultdict(list)
    for c in cases:
        domain_groups[c.get("main_domain", "") or "unknown"].append(c)

    # Sort within each domain by tier for interleaving
    for domain in domain_groups:
        domain_groups[domain].sort(key=lambda c: c["_tier"])

    # Round-robin across batches, largest domains first
    batches = [[] for _ in range(n_batches)]
    idx = 0
    for domain in sorted(domain_groups, key=lambda d: -len(domain_groups[d])):
        for c in domain_groups[domain]:
            batches[idx].append(c)
            idx = (idx + 1) % n_batches

    return batches


def make_blind_anchors(anchors: list) -> list:
    """
    Strip expected answers from anchors so the annotator processes
    them blind (like regular cases).
    """
    blind = []
    for a in anchors:
        blind.append({
            "transition_id": a["transition_id"],
            "story_summary": a["story_summary"],
            "before_state": a["before_state"],
            "after_state": a["after_state"],
            "main_domain": a.get("main_domain", ""),
            "scale": a.get("scale", ""),
            "is_calibration": True,
        })
    return blind


def save_batches(batches: list, blind_anchors: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for i, batch in enumerate(batches):
        num = i + 1
        # Clean internal fields
        clean = []
        for c in batch:
            entry = {k: v for k, v in c.items() if not k.startswith("_")}
            entry["is_calibration"] = False
            clean.append(entry)

        all_cases = blind_anchors + clean

        data = {
            "batch_id": f"batch_{num:03d}",
            "batch_number": num,
            "total_batches": len(batches),
            "calibration_count": len(blind_anchors),
            "case_count": len(clean),
            "total_items": len(all_cases),
            "cases": all_cases,
        }
        path = os.path.join(out_dir, f"batch_{num:03d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(batches)} batch files to {out_dir}")


def print_summary(batches: list, anchors: list):
    total_cases = sum(len(b) for b in batches)
    sizes = [len(b) for b in batches]

    print(f"\n{'=' * 60}")
    print("  Silver Set Batch Split Summary")
    print(f"{'=' * 60}")
    print(f"  Total cases:         {total_cases}")
    print(f"  Batches:             {len(batches)}")
    print(f"  Cases per batch:     {BATCH_SIZE} + {len(anchors)} anchors "
          f"= {BATCH_SIZE + len(anchors)} items")
    print(f"  Batch sizes:         min={min(sizes)}, max={max(sizes)}")

    print(f"\n  Length tier distribution (first 5 batches):")
    for i, batch in enumerate(batches[:5]):
        tiers = defaultdict(int)
        for c in batch:
            tiers[c.get("_tier", "?")] += 1
        t_str = ", ".join(f"{t}={n}" for t, n in sorted(tiers.items()))
        print(f"    batch_{i + 1:03d}: {t_str}  (n={len(batch)})")

    print(f"\n  Domain diversity (first 5 batches):")
    for i, batch in enumerate(batches[:5]):
        domains = set(c.get("main_domain", "") for c in batch)
        print(f"    batch_{i + 1:03d}: {len(domains)} unique domains")

    print(f"\n  Output: {BATCHES_DIR}/batch_001.json ~ "
          f"batch_{len(batches):03d}.json")
    print(f"{'=' * 60}")


def main():
    if not os.path.exists(CASES_PATH):
        print(f"Error: {CASES_PATH} not found")
        sys.exit(1)
    if not os.path.exists(ANCHORS_PATH):
        print(f"Error: {ANCHORS_PATH} not found")
        sys.exit(1)

    cases = load_json(CASES_PATH)
    anchors = load_json(ANCHORS_PATH)
    print(f"Loaded {len(cases)} cases, {len(anchors)} calibration anchors")

    n_batches = math.ceil(len(cases) / BATCH_SIZE)
    if len(cases) != 800:
        print(f"Warning: expected 800 cases, got {len(cases)}; "
              f"using {n_batches} batches")

    batches = stratified_split(cases, n_batches)
    blind_anchors = make_blind_anchors(anchors)

    print_summary(batches, anchors)
    save_batches(batches, blind_anchors, BATCHES_DIR)
    print("\nDone. Ready for annotation.")


if __name__ == "__main__":
    main()
