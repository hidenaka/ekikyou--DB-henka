#!/usr/bin/env python3
"""
validate_silver_annotations.py — Silver Set アノテーション検証スクリプト

エージェントベースのアノテーション結果を検証する:
- スキーマ検証（必須フィールド、Enum値）
- 重複検出
- キャリブレーションアンカーとの照合
- 統計レポート（八卦分布、信頼度分布）
- エラー検出（primary == alternative）

2つのアノテーション形式をサポート:
  フラット形式:  {"before_lower": "坎", "before_lower_confidence": "high", ...}
  ネスト形式:    {"before_lower": {"trigram": "坎", "confidence": "high", ...}, ...}

Usage:
    python3 scripts/validate_silver_annotations.py <annotation_file_or_dir>
    python3 scripts/validate_silver_annotations.py analysis/gold_set/batches/batch_001_annotated.json
    python3 scripts/validate_silver_annotations.py analysis/gold_set/batches/
"""

import json
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path

# --- Constants ---

VALID_TRIGRAMS = {"乾", "坤", "震", "巽", "坎", "離", "艮", "兌"}
VALID_CONFIDENCE = {"high", "medium", "low"}
POSITIONS = ["before_lower", "before_upper", "after_lower", "after_upper"]


# --- Normalize annotation format ---

def normalize_annotation(ann: dict) -> dict:
    """
    Normalize annotation to flat format.
    Supports both flat and nested formats.
    """
    # Check if it's nested format (before_lower is a dict)
    sample = ann.get("before_lower")
    if isinstance(sample, dict):
        normalized = {"transition_id": ann.get("transition_id", "?")}
        for pos in POSITIONS:
            entry = ann.get(pos, {})
            if not isinstance(entry, dict):
                entry = {}
            normalized[pos] = entry.get("trigram", entry.get("value"))
            normalized[f"{pos}_confidence"] = entry.get("confidence")
            normalized[f"{pos}_rationale"] = entry.get("rationale", entry.get("reasoning", ""))
            normalized[f"{pos}_alternative"] = entry.get("alternative")
        return normalized
    # Already flat format
    return ann


# --- Calibration anchor loader ---

def load_calibration_anchors(project_root: str) -> dict:
    """Load calibration anchors and return as {transition_id: anchor_data}."""
    anchor_path = os.path.join(project_root, "analysis", "gold_set", "calibration_anchors.json")
    if not os.path.exists(anchor_path):
        print(f"  [WARN] Calibration anchors not found at {anchor_path}")
        return {}
    with open(anchor_path, encoding="utf-8") as f:
        anchors = json.load(f)
    return {a["transition_id"]: a for a in anchors}


# --- Validation functions ---

def validate_schema(ann: dict, idx: int) -> list:
    """Validate a single annotation against the expected schema."""
    errors = []
    tid = ann.get("transition_id", f"idx={idx}")

    # Must have transition_id
    if "transition_id" not in ann:
        errors.append(f"  [{idx}] Missing field: transition_id")

    for pos in POSITIONS:
        # Trigram value
        val = ann.get(pos)
        if val is None:
            errors.append(f"  [{tid}] Missing field: {pos}")
        elif val not in VALID_TRIGRAMS:
            errors.append(f"  [{tid}] Invalid trigram '{val}' in {pos}")

        # Confidence
        conf = ann.get(f"{pos}_confidence")
        if conf is None:
            errors.append(f"  [{tid}] Missing field: {pos}_confidence")
        elif conf not in VALID_CONFIDENCE:
            errors.append(f"  [{tid}] Invalid confidence '{conf}' in {pos}_confidence")

        # Rationale
        rat = ann.get(f"{pos}_rationale", "")
        if not rat or (isinstance(rat, str) and len(rat.strip()) == 0):
            errors.append(f"  [{tid}] Empty rationale in {pos}_rationale")

        # Alternative
        alt = ann.get(f"{pos}_alternative")
        if alt is not None and alt not in VALID_TRIGRAMS:
            errors.append(f"  [{tid}] Invalid alternative '{alt}' in {pos}_alternative")

    return errors


def check_primary_equals_alternative(ann: dict) -> list:
    """Flag cases where primary trigram == alternative (likely error)."""
    warnings = []
    tid = ann.get("transition_id", "?")
    for pos in POSITIONS:
        primary = ann.get(pos)
        alt = ann.get(f"{pos}_alternative")
        if primary and alt and primary == alt:
            conf = ann.get(f"{pos}_confidence", "?")
            warnings.append(
                f"  [{tid}] primary == alternative in {pos}: "
                f"'{primary}' (confidence: {conf})"
            )
    return warnings


def check_calibration(ann: dict, anchors: dict) -> tuple:
    """
    Check if a calibration anchor annotation matches expected answers.
    Returns (passes: list, issues: list).
    """
    tid = ann.get("transition_id", "")
    if tid not in anchors:
        return [], []

    anchor = anchors[tid]
    passes = []
    issues = []

    for pos in POSITIONS:
        expected = anchor.get(f"expected_{pos}")
        actual = ann.get(pos)
        acceptable = anchor.get("acceptable_alternatives", {}).get(pos, [])

        if expected is None:
            continue

        if actual == expected:
            passes.append(f"  [{tid}] {pos}: MATCH '{actual}'")
        elif actual in acceptable:
            issues.append(
                f"  [{tid}] {pos}: '{actual}' (acceptable alt), expected '{expected}'"
            )
        else:
            issues.append(
                f"  [{tid}] {pos}: MISMATCH '{actual}', "
                f"expected '{expected}' (acceptable: {acceptable})"
            )

    # CALIBRATION_005: ambiguous case should not get high confidence
    if tid == "CALIBRATION_005":
        for pos in POSITIONS:
            conf = ann.get(f"{pos}_confidence")
            if conf == "high":
                issues.append(
                    f"  [{tid}] {pos}_confidence='high' for ambiguous case "
                    f"(expected medium or low)"
                )

    return passes, issues


def check_duplicates(annotations: list) -> list:
    """Check for duplicate transition_ids."""
    id_counts = Counter(a.get("transition_id", "") for a in annotations)
    duplicates = []
    for tid, count in id_counts.items():
        if count > 1:
            duplicates.append(
                f"  Duplicate transition_id: '{tid}' appears {count} times"
            )
    return duplicates


# --- Statistics ---

def compute_statistics(annotations: list) -> dict:
    """Compute trigram and confidence distribution statistics."""
    stats = {
        "total": len(annotations),
        "trigram_distribution": Counter(),
        "confidence_distribution": Counter(),
        "per_position": {},
    }

    for pos in POSITIONS:
        pos_trigrams = Counter()
        pos_confidence = Counter()
        for a in annotations:
            val = a.get(pos)
            if val and val in VALID_TRIGRAMS:
                stats["trigram_distribution"][val] += 1
                pos_trigrams[val] += 1
            conf = a.get(f"{pos}_confidence")
            if conf and conf in VALID_CONFIDENCE:
                stats["confidence_distribution"][conf] += 1
                pos_confidence[conf] += 1
        stats["per_position"][pos] = {
            "trigram": dict(pos_trigrams.most_common()),
            "confidence": dict(pos_confidence.most_common()),
        }

    stats["trigram_distribution"] = dict(stats["trigram_distribution"].most_common())
    stats["confidence_distribution"] = dict(stats["confidence_distribution"].most_common())
    return stats


def detect_distribution_anomalies(stats: dict) -> list:
    """Detect potential distribution bias."""
    warnings = []
    total = sum(stats["trigram_distribution"].values())
    if total == 0:
        return warnings

    for trigram in VALID_TRIGRAMS:
        count = stats["trigram_distribution"].get(trigram, 0)
        share = count / total
        expected = total / 8
        if share < 0.02:
            warnings.append(
                f"  Under-represented '{trigram}': {count} ({share:.1%}), "
                f"expected ~{expected:.0f}"
            )
        elif share > 0.35:
            warnings.append(
                f"  Over-represented '{trigram}': {count} ({share:.1%}), "
                f"expected ~{expected:.0f}"
            )

    return warnings


# --- Load annotations ---

def load_annotations(path: str) -> list:
    """Load annotations from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("annotations", "cases", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Cannot parse annotations from {path}")


def find_annotation_files(path: str) -> list:
    """Find all annotation JSON files."""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted(p.glob("*_annotated.json"))
        if not files:
            files = sorted(p.glob("*_annotations.json"))
        if not files:
            files = sorted(p.glob("*.json"))
        return [str(f) for f in files]
    print(f"Error: {path} not found")
    sys.exit(1)


# --- Report ---

def print_report(all_annotations, schema_errors, dup_errors,
                 primary_alt_warnings, calibration_passes,
                 calibration_issues, stats, dist_warnings):
    """Print the full validation report."""
    total = len(all_annotations)
    print("\n" + "=" * 70)
    print("  Silver Annotation Validation Report")
    print("=" * 70)
    print(f"\n  Total annotations: {total}")

    # Schema errors
    print(f"\n--- Schema Validation ---")
    if schema_errors:
        print(f"  FAIL: {len(schema_errors)} schema errors")
        for e in schema_errors[:20]:
            print(e)
        if len(schema_errors) > 20:
            print(f"  ... and {len(schema_errors) - 20} more")
    else:
        print(f"  PASS: All {total} annotations have valid schema")

    # Duplicates
    print(f"\n--- Duplicate Check ---")
    if dup_errors:
        print(f"  FAIL: {len(dup_errors)} duplicate issues")
        for e in dup_errors:
            print(e)
    else:
        print(f"  PASS: No duplicate transition_ids")

    # Primary == Alternative
    print(f"\n--- Primary == Alternative Check ---")
    if primary_alt_warnings:
        print(f"  WARN: {len(primary_alt_warnings)} cases")
        for w in primary_alt_warnings[:10]:
            print(w)
        if len(primary_alt_warnings) > 10:
            print(f"  ... and {len(primary_alt_warnings) - 10} more")
    else:
        print(f"  PASS: No primary == alternative conflicts")

    # Calibration anchors
    print(f"\n--- Calibration Anchor Check ---")
    if calibration_passes or calibration_issues:
        n_checked = len(calibration_passes) + len(calibration_issues)
        print(f"  Positions checked: {n_checked}")
        for p in calibration_passes:
            print(p)
        if calibration_issues:
            print(f"  ISSUES: {len(calibration_issues)} mismatches")
            for c in calibration_issues:
                print(c)
        else:
            print(f"  PASS: All calibration anchors match")
    else:
        print(f"  SKIP: No calibration anchors found in annotations")

    # Trigram distribution
    print(f"\n--- Trigram Distribution (all positions combined) ---")
    total_t = sum(stats["trigram_distribution"].values())
    for trigram in ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]:
        count = stats["trigram_distribution"].get(trigram, 0)
        pct = (count / total_t * 100) if total_t > 0 else 0
        bar = "#" * int(pct * 2)
        print(f"  {trigram}: {count:4d} ({pct:5.1f}%) {bar}")

    if dist_warnings:
        print(f"\n  Distribution warnings:")
        for w in dist_warnings:
            print(w)

    # Confidence distribution
    print(f"\n--- Confidence Distribution ---")
    total_c = sum(stats["confidence_distribution"].values())
    for conf in ["high", "medium", "low"]:
        count = stats["confidence_distribution"].get(conf, 0)
        pct = (count / total_c * 100) if total_c > 0 else 0
        bar = "#" * int(pct)
        print(f"  {conf:6s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Per-position breakdown
    print(f"\n--- Per-Position Trigram Distribution ---")
    for pos in POSITIONS:
        pos_data = stats["per_position"].get(pos, {})
        trigrams = pos_data.get("trigram", {})
        total_p = sum(trigrams.values())
        top3 = sorted(trigrams.items(), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{t}={c}" for t, c in top3)
        print(f"  {pos:14s}: n={total_p:4d}  top3: {top3_str}")

    # Summary
    print(f"\n{'=' * 70}")
    has_errors = bool(schema_errors or dup_errors)
    has_warnings = bool(primary_alt_warnings or calibration_issues or dist_warnings)
    if has_errors:
        print("  RESULT: FAIL")
    elif has_warnings:
        print("  RESULT: PASS WITH WARNINGS")
    else:
        print("  RESULT: PASS")
    print(f"{'=' * 70}")


# --- Main ---

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/validate_silver_annotations.py "
              "<annotation_file_or_dir> [anchors_file]")
        sys.exit(1)

    target = sys.argv[1]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Optional explicit anchors path
    if len(sys.argv) > 2:
        anchor_path = sys.argv[2]
        with open(anchor_path, encoding="utf-8") as f:
            anchors_list = json.load(f)
        anchors = {a["transition_id"]: a for a in anchors_list}
    else:
        anchors = load_calibration_anchors(project_root)

    # Find and load annotation files
    files = find_annotation_files(target)
    if not files:
        print(f"No annotation files found at {target}")
        sys.exit(1)

    print(f"Loading annotations from {len(files)} file(s)...")
    all_annotations = []
    for fpath in files:
        try:
            raw = load_annotations(fpath)
            normalized = [normalize_annotation(a) for a in raw]
            all_annotations.extend(normalized)
            print(f"  {os.path.basename(fpath)}: {len(normalized)} annotations")
        except Exception as e:
            print(f"  Error loading {fpath}: {e}")

    if not all_annotations:
        print("No annotations loaded.")
        sys.exit(1)

    # Validate
    schema_errors = []
    primary_alt_warnings = []
    calibration_passes = []
    calibration_issues = []

    for i, ann in enumerate(all_annotations):
        schema_errors.extend(validate_schema(ann, i))
        primary_alt_warnings.extend(check_primary_equals_alternative(ann))
        passes, issues = check_calibration(ann, anchors)
        calibration_passes.extend(passes)
        calibration_issues.extend(issues)

    dup_errors = check_duplicates(all_annotations)

    # Statistics (exclude calibration cases)
    real = [a for a in all_annotations
            if not a.get("transition_id", "").startswith("CALIBRATION_")]
    stats = compute_statistics(real if real else all_annotations)
    dist_warnings = detect_distribution_anomalies(stats)

    # Report
    print_report(all_annotations, schema_errors, dup_errors,
                 primary_alt_warnings, calibration_passes,
                 calibration_issues, stats, dist_warnings)

    sys.exit(1 if (schema_errors or dup_errors) else 0)


if __name__ == "__main__":
    main()
