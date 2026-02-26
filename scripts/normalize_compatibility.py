#!/usr/bin/env python3
"""
normalize_compatibility.py
--------------------------
64個の個別hexagram_XX.jsonファイルから互換性データを読み込み、
単一の構造化JSONファイルに正規化する。

入力: HAQEI-Cross-Device-Sync/js/data/compatibility/engine-interface/hexagram_01..64.json
出力:
  - data/reference/hexagram_compatibility.json       (完全版: 4,096組)
  - data/reference/hexagram_compatibility_lookup.json (簡易lookup版)

Usage:
  python3 scripts/normalize_compatibility.py
  python3 scripts/normalize_compatibility.py --source /path/to/source --output /path/to/output
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# --- Constants ---
DEFAULT_SOURCE = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/"
    "HAQEI-Cross-Device-Sync/js/data/compatibility/engine-interface"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "reference"
)
EXPECTED_FILES = 64
EXPECTED_PAIRS_PER_FILE = 64
EXPECTED_TOTAL_PAIRS = EXPECTED_FILES * EXPECTED_PAIRS_PER_FILE  # 4,096


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize 64 hexagram compatibility files into a single JSON"
    )
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE,
        help="Source directory containing hexagram_XX.json files"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR,
        help="Output directory for normalized JSON files"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and validate without writing output files"
    )
    return parser.parse_args()


def extract_evaluation(eval_obj: dict) -> dict:
    """Extract evaluation subscores, handling both nested and flat formats."""
    result = {}
    expected_keys = [
        "functional_efficiency", "growth_potential",
        "stress_resilience", "creativity", "integration_challenge"
    ]
    for key in expected_keys:
        if key in eval_obj:
            val = eval_obj[key]
            if isinstance(val, dict):
                result[key] = {
                    "score": val.get("score", 0.0),
                    "description": val.get("description", "")
                }
            elif isinstance(val, (int, float)):
                result[key] = {"score": float(val), "description": ""}
            else:
                result[key] = {"score": 0.0, "description": str(val)}
        else:
            result[key] = {"score": 0.0, "description": ""}
    return result


def extract_advice(advice_obj: dict) -> dict:
    """Extract advice fields with safe defaults."""
    return {
        "strengths": advice_obj.get("strengths", []),
        "challenges": advice_obj.get("challenges", []),
        "recommendations": advice_obj.get("recommendations", [])
    }


def process_file(filepath: str) -> tuple:
    """
    Process a single hexagram file.
    Returns (hex_id, hex_name, pairs_list, errors_list).
    """
    pairs = []
    errors = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"JSON parse error in {filepath}: {e}")
        return None, None, pairs, errors
    except Exception as e:
        errors.append(f"Read error in {filepath}: {e}")
        return None, None, pairs, errors

    hex_id = data.get("hexagram_id")
    analysis = data.get("internal_team_analysis", {})
    hex_name = analysis.get("engine_os_name", f"卦{hex_id}")
    combinations = analysis.get("interface_combinations", [])

    if not combinations:
        errors.append(f"{filepath}: No interface_combinations found")
        return hex_id, hex_name, pairs, errors

    for combo in combinations:
        try:
            pair = {
                "hex_a": hex_id,
                "hex_b": combo.get("interface_id"),
                "hex_a_name": hex_name,
                "hex_b_name": combo.get("interface_name", ""),
                "type": combo.get("type", "UNKNOWN"),
                "overall_score": float(combo.get("overall_score", 0.0)),
                "summary": combo.get("summary", ""),
                "evaluation": extract_evaluation(combo.get("evaluation", {})),
                "advice": extract_advice(combo.get("advice", {}))
            }
            pairs.append(pair)
        except Exception as e:
            errors.append(
                f"{filepath}, interface_id={combo.get('interface_id', '?')}: {e}"
            )

    return hex_id, hex_name, pairs, errors


def build_lookup(pairs: list) -> dict:
    """Build compact lookup dictionary from pairs list."""
    lookup = {}
    for p in pairs:
        key = f"{p['hex_a']}-{p['hex_b']}"
        lookup[key] = {
            "type": p["type"],
            "score": p["overall_score"],
            "summary": p["summary"]
        }
    return lookup


def print_report(all_pairs, all_errors, type_counts, output_full, output_lookup,
                 duplicates_removed=0):
    """Print processing report to stdout."""
    print("=" * 60)
    print("  Hexagram Compatibility Normalization Report")
    print("=" * 60)
    print(f"\n  Unique pairs:          {len(all_pairs)} / {EXPECTED_TOTAL_PAIRS}")
    if duplicates_removed:
        print(f"  Duplicates removed:    {duplicates_removed}")
    print(f"  Missing pairs:         {EXPECTED_TOTAL_PAIRS - len(all_pairs)}")
    print(f"  Errors encountered:    {len(all_errors)}")

    print(f"\n  Compatibility Type Distribution:")
    print(f"  {'Type':<12} {'Count':>6} {'Pct':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*8}")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_pairs) * 100 if all_pairs else 0
        print(f"  {t:<12} {count:>6} {pct:>7.1f}%")

    if output_full and os.path.exists(output_full):
        size_full = os.path.getsize(output_full)
        size_lookup = os.path.getsize(output_lookup)
        print(f"\n  Output files:")
        print(f"    Full:   {output_full}")
        print(f"            {size_full:,} bytes ({size_full / 1024 / 1024:.1f} MB)")
        print(f"    Lookup: {output_lookup}")
        print(f"            {size_lookup:,} bytes ({size_lookup / 1024:.0f} KB)")

    if all_errors:
        print(f"\n  Errors:")
        for err in all_errors[:20]:
            print(f"    - {err}")
        if len(all_errors) > 20:
            print(f"    ... and {len(all_errors) - 20} more")

    print("\n" + "=" * 60)


def main():
    args = parse_args()
    source_dir = args.source
    output_dir = args.output

    # Validate source directory
    if not os.path.isdir(source_dir):
        print(f"ERROR: Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all files
    source_files = sorted(
        Path(source_dir).glob("hexagram_*.json"),
        key=lambda p: int(p.stem.split("_")[1])
    )
    print(f"Found {len(source_files)} source files in {source_dir}")

    if len(source_files) != EXPECTED_FILES:
        print(f"WARNING: Expected {EXPECTED_FILES} files, found {len(source_files)}")

    # Process all files
    all_pairs = []
    all_errors = []
    type_counts = Counter()
    hex_names = {}  # id -> name mapping

    for filepath in source_files:
        hex_id, hex_name, pairs, errors = process_file(str(filepath))
        all_pairs.extend(pairs)
        all_errors.extend(errors)

        if hex_id is not None:
            hex_names[hex_id] = hex_name

        for p in pairs:
            type_counts[p["type"]] += 1

    # Deduplicate pairs: keep first occurrence of each (hex_a, hex_b)
    seen = set()
    deduped_pairs = []
    duplicates_removed = 0
    for p in all_pairs:
        key = (p["hex_a"], p["hex_b"])
        if key not in seen:
            seen.add(key)
            deduped_pairs.append(p)
        else:
            duplicates_removed += 1

    if duplicates_removed > 0:
        print(f"Deduplicated: removed {duplicates_removed} duplicate pairs")
        # Recount types after dedup
        type_counts = Counter(p["type"] for p in deduped_pairs)

    all_pairs = deduped_pairs

    # Sort pairs by (hex_a, hex_b)
    all_pairs.sort(key=lambda p: (p["hex_a"], p["hex_b"]))

    # Build output structures
    full_output = {
        "version": "1.0",
        "description": "64卦\u00d764卦の互換性マトリクス（4,096組）",
        "source": "HAQEI-Cross-Device-Sync/engine-interface",
        "generated_by": "scripts/normalize_compatibility.py",
        "total_pairs": len(all_pairs),
        "type_distribution": dict(type_counts.most_common()),
        "pairs": all_pairs
    }

    lookup_output = build_lookup(all_pairs)

    # Write output
    output_full_path = None
    output_lookup_path = None

    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)
        output_full_path = os.path.join(output_dir, "hexagram_compatibility.json")
        output_lookup_path = os.path.join(
            output_dir, "hexagram_compatibility_lookup.json"
        )

        with open(output_full_path, "w", encoding="utf-8") as f:
            json.dump(full_output, f, ensure_ascii=False, indent=2)

        with open(output_lookup_path, "w", encoding="utf-8") as f:
            json.dump(lookup_output, f, ensure_ascii=False, indent=2)

        print(f"Written: {output_full_path}")
        print(f"Written: {output_lookup_path}")
    else:
        print("[DRY RUN] No files written.")

    # Print report
    print_report(all_pairs, all_errors, type_counts,
                 output_full_path, output_lookup_path,
                 duplicates_removed)

    # Exit code
    if len(all_pairs) < EXPECTED_TOTAL_PAIRS:
        print(f"\nWARNING: Missing {EXPECTED_TOTAL_PAIRS - len(all_pairs)} pairs")
        sys.exit(2) if len(all_pairs) == 0 else None

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
