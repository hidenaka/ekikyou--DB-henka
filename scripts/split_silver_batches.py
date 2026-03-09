#!/usr/bin/env python3
"""Split 800 selected cases into batches for agent-based annotation."""

import json
import os
import random

GOLD_DIR = os.path.join(os.path.dirname(__file__), '..', 'analysis', 'gold_set')
BATCH_DIR = os.path.join(GOLD_DIR, 'batches')
CASES_FILE = os.path.join(GOLD_DIR, 'selected_800_cases.json')
ANCHORS_FILE = os.path.join(GOLD_DIR, 'calibration_anchors.json')
BATCH_SIZE = 25

def main():
    with open(CASES_FILE) as f:
        cases = json.load(f)
    with open(ANCHORS_FILE) as f:
        anchors = json.load(f)

    random.seed(42)
    random.shuffle(cases)

    os.makedirs(BATCH_DIR, exist_ok=True)

    n_batches = (len(cases) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(cases))
        batch_cases = cases[start:end]
        batch = {
            "batch_id": i + 1,
            "n_cases": len(batch_cases),
            "calibration_anchors": anchors,
            "cases": batch_cases
        }
        path = os.path.join(BATCH_DIR, f'batch_{i+1:03d}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)

    print(f"Created {n_batches} batches of ~{BATCH_SIZE} cases in {BATCH_DIR}")
    print(f"Each batch includes {len(anchors)} calibration anchors")

if __name__ == '__main__':
    main()
