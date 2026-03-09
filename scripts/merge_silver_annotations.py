#!/usr/bin/env python3
"""
merge_silver_annotations.py — Silver annotation batch files を Pass 単位に統合

32個の silver_pass{N}_batch_NNN.json を読み込み、
1つの annotations_pass{N}.json に統合する。
キャリブレーション事例は除外（CALIBRATION_ prefix）。

出力フォーマットは analyze_annotation_agreement.py と互換:
  - transition_id をキーとした dict
  - before_lower, before_upper, after_lower, after_upper は直接トリグラム値
  - {field}_confidence で信頼度も保持

Usage:
    python3 scripts/merge_silver_annotations.py
"""

import json
import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_SET_DIR = PROJECT_ROOT / "analysis" / "gold_set"

FIELDS = ["before_lower", "before_upper", "after_lower", "after_upper"]


def merge_pass(pass_number: int) -> dict:
    """Merge all batch files for a given pass into a single annotations dict."""
    pattern = str(GOLD_SET_DIR / f"silver_pass{pass_number}_batch_*.json")
    batch_files = sorted(glob.glob(pattern))

    if not batch_files:
        print(f"WARNING: No files found for pass {pass_number} ({pattern})")
        return {}

    annotations = []
    seen_ids = set()
    calibration_count = 0
    duplicate_count = 0

    for bf in batch_files:
        with open(bf, "r", encoding="utf-8") as f:
            data = json.load(f)

        for ann in data.get("annotations", []):
            tid = ann.get("transition_id", "")

            # Skip calibration cases
            if tid.startswith("CALIBRATION_"):
                calibration_count += 1
                continue

            # Skip duplicates (keep first occurrence)
            if tid in seen_ids:
                duplicate_count += 1
                continue
            seen_ids.add(tid)

            # Flatten nested structure for agreement script compatibility
            flat = {"transition_id": tid}
            for field in FIELDS:
                field_data = ann.get(field, {})
                if isinstance(field_data, dict):
                    flat[field] = field_data.get("trigram", "")
                    flat[f"{field}_confidence"] = field_data.get("confidence", "")
                    flat[f"{field}_alternative"] = field_data.get("alternative", "")
                    flat[f"{field}_reasoning"] = field_data.get("reasoning", "")
                else:
                    flat[field] = field_data

            annotations.append(flat)

    print(f"Pass {pass_number}: {len(batch_files)} batch files, "
          f"{len(annotations)} unique cases, {duplicate_count} duplicates skipped, "
          f"{calibration_count} calibration items skipped")

    return {
        "metadata": {
            "pass_number": pass_number,
            "source": "silver_set_agent_annotation",
            "n_batches": len(batch_files),
            "n_annotations": len(annotations),
            "n_calibration_skipped": calibration_count,
        },
        "annotations": annotations,
    }


def main():
    for pass_num in [1, 2]:
        result = merge_pass(pass_num)
        if not result:
            continue

        out_path = GOLD_SET_DIR / f"annotations_pass{pass_num}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  -> {out_path} ({result['metadata']['n_annotations']} annotations)")

    print("\nDone. Run: python3 scripts/analyze_annotation_agreement.py")


if __name__ == "__main__":
    main()
