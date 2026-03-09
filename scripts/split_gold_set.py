#!/usr/bin/env python3
"""
ゴールドセット500件をパイロット100件と評価400件に層化分割する。
source_type, country比率を維持する。
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
PILOT_SIZE = 100

BASE = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE / "analysis" / "gold_set" / "gold_set_sample.json"
PILOT_PATH = BASE / "analysis" / "gold_set" / "pilot_100.json"
EVAL_PATH = BASE / "analysis" / "gold_set" / "eval_400.json"


def main():
    random.seed(SEED)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"入力: {total}件")

    # 層化キー: (source_type, country)
    buckets = defaultdict(list)
    for case in data:
        key = (case.get("source_type", "unknown"), case.get("country", "unknown"))
        buckets[key].append(case)

    pilot = []
    eval_set = []

    for key, cases in sorted(buckets.items()):
        n_bucket = len(cases)
        n_pilot = max(1, round(n_bucket * PILOT_SIZE / total))
        # Cap at bucket size
        n_pilot = min(n_pilot, n_bucket)

        random.shuffle(cases)
        pilot.extend(cases[:n_pilot])
        eval_set.extend(cases[n_pilot:])

    # Adjust to exactly PILOT_SIZE
    if len(pilot) > PILOT_SIZE:
        random.shuffle(pilot)
        overflow = pilot[PILOT_SIZE:]
        pilot = pilot[:PILOT_SIZE]
        eval_set.extend(overflow)
    elif len(pilot) < PILOT_SIZE:
        need = PILOT_SIZE - len(pilot)
        random.shuffle(eval_set)
        pilot.extend(eval_set[:need])
        eval_set = eval_set[need:]

    # Sort by transition_id for reproducibility
    pilot.sort(key=lambda x: x.get("transition_id") or "")
    eval_set.sort(key=lambda x: x.get("transition_id") or "")

    # Save
    with open(PILOT_PATH, "w", encoding="utf-8") as f:
        json.dump(pilot, f, ensure_ascii=False, indent=2)

    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)

    # Stats
    print(f"\nパイロット: {len(pilot)}件 → {PILOT_PATH.name}")
    print(f"評価: {len(eval_set)}件 → {EVAL_PATH.name}")

    print("\n--- source_type比率の比較 ---")
    print(f"{'':12s} {'元500':>8s} {'Pilot':>8s} {'Eval':>8s}")
    orig_st = Counter(d.get("source_type") for d in data)
    pilot_st = Counter(d.get("source_type") for d in pilot)
    eval_st = Counter(d.get("source_type") for d in eval_set)
    for st in sorted(orig_st.keys()):
        o = orig_st[st] / total * 100
        p = pilot_st.get(st, 0) / len(pilot) * 100
        e = eval_st.get(st, 0) / len(eval_set) * 100
        print(f"  {st:10s} {o:7.1f}% {p:7.1f}% {e:7.1f}%")

    print("\n--- country比率の比較 ---")
    print(f"{'':12s} {'元500':>8s} {'Pilot':>8s} {'Eval':>8s}")
    orig_co = Counter(d.get("country") for d in data)
    pilot_co = Counter(d.get("country") for d in pilot)
    eval_co = Counter(d.get("country") for d in eval_set)
    for co in sorted(orig_co.keys()):
        o = orig_co[co] / total * 100
        p = pilot_co.get(co, 0) / len(pilot) * 100
        e = eval_co.get(co, 0) / len(eval_set) * 100
        print(f"  {co:10s} {o:7.1f}% {p:7.1f}% {e:7.1f}%")


if __name__ == "__main__":
    main()
