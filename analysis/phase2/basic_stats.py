#!/usr/bin/env python3
"""Phase 2A-1: Basic statistics for I Ching Transition Logic DB."""

import json
import os
from collections import Counter, defaultdict

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw/cases.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "basic_stats.json")

VARIABLES = [
    "before_state",
    "after_state",
    "trigger_type",
    "action_type",
    "pattern_type",
    "scale",
    "outcome",
]

def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def frequency_distributions(records, variables):
    result = {}
    for var in variables:
        counter = Counter()
        for r in records:
            val = r.get(var)
            if val is not None and val != "":
                counter[val] += 1
        # Sort by count descending
        result[var] = dict(sorted(counter.items(), key=lambda x: -x[1]))
    return result

def missing_values(records, variables):
    total = len(records)
    result = {}
    for var in variables:
        missing = sum(1 for r in records if r.get(var) is None or r.get(var) == "")
        result[var] = {
            "count": missing,
            "rate": round(missing / total, 4) if total > 0 else 0,
        }
    return result

def cross_tab_2d(records, var1, var2):
    table = defaultdict(lambda: defaultdict(int))
    for r in records:
        v1 = r.get(var1)
        v2 = r.get(var2)
        if v1 and v2:
            table[v1][v2] += 1
    # Convert to regular dict
    return {k: dict(v) for k, v in table.items()}

def cross_tab_3d(records, var1, var2, var3):
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in records:
        v1 = r.get(var1)
        v2 = r.get(var2)
        v3 = r.get(var3)
        if v1 and v2 and v3:
            table[v1][v2][v3] += 1
    return {k1: {k2: dict(v2) for k2, v2 in v1.items()} for k1, v1 in table.items()}

def main():
    records = load_records(DATA_PATH)
    total = len(records)

    result = {
        "total_records": total,
        "frequency_distributions": frequency_distributions(records, VARIABLES),
        "cross_tabulations": {
            "before_after": cross_tab_2d(records, "before_state", "after_state"),
            "before_action": cross_tab_2d(records, "before_state", "action_type"),
            "action_after": cross_tab_2d(records, "action_type", "after_state"),
            "trigger_action": cross_tab_2d(records, "trigger_type", "action_type"),
            "before_action_after": cross_tab_3d(records, "before_state", "action_type", "after_state"),
        },
        "missing_values": missing_values(records, VARIABLES),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Total records: {total}")
    print(f"Output: {os.path.abspath(OUTPUT_PATH)}")
    print()
    print("=== Frequency Distributions ===")
    for var in VARIABLES:
        print(f"\n{var}:")
        for val, cnt in result["frequency_distributions"][var].items():
            print(f"  {val}: {cnt}")
    print()
    print("=== Missing Values ===")
    for var in VARIABLES:
        mv = result["missing_values"][var]
        print(f"  {var}: {mv['count']} ({mv['rate']*100:.1f}%)")

if __name__ == "__main__":
    main()
