#!/usr/bin/env python3
"""
Hexagram Coverage Analysis for the I Ching Isomorphism Database.

Analyzes all 64 hexagrams and 8 trigrams across before/after fields
to identify coverage gaps and estimate enrichment needs.

Usage:
    python3 scripts/analyze_hexagram_coverage.py
    python3 scripts/analyze_hexagram_coverage.py --output analysis/phase3/hexagram_coverage_report.md
"""

import json
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
CASES_PATH = BASE_DIR / "data" / "raw" / "cases.jsonl"
REFERENCE_PATH = BASE_DIR / "data" / "reference" / "iching_texts_ctext_legge_ja.json"
DEFAULT_OUTPUT = BASE_DIR / "analysis" / "phase3" / "hexagram_coverage_report.md"

TRIGRAM_NAMES = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]
TRIGRAM_FIELDS = [
    "before_lower_trigram",
    "before_upper_trigram",
    "after_lower_trigram",
    "after_upper_trigram",
]
MIN_SUPPORT = 5
TARGET_COVERAGE = 60  # out of 64


def load_hexagram_names(path: Path) -> dict[int, str]:
    """Load hexagram number -> name mapping from reference JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    names = {}
    for num_str, h in data["hexagrams"].items():
        names[int(num_str)] = h.get("local_name", f"卦{num_str}")
    return names


def load_cases(path: Path) -> list[dict]:
    """Load all cases from JSONL."""
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def analyze(cases: list[dict], hex_names: dict[int, str]) -> str:
    """Run full analysis and return markdown report."""
    lines = []
    total = len(cases)

    # =========================================================
    # 1. Count hexagram occurrences (before / after / total)
    # =========================================================
    before_counts = Counter()
    after_counts = Counter()
    missing_before = 0
    missing_after = 0

    for c in cases:
        bn = c.get("hexagram_number_before")
        an = c.get("hexagram_number_after")
        if bn is not None and isinstance(bn, (int, float)):
            before_counts[int(bn)] += 1
        else:
            missing_before += 1
        if an is not None and isinstance(an, (int, float)):
            after_counts[int(an)] += 1
        else:
            missing_after += 1

    # Build combined table for all 64
    hex_table = []
    for num in range(1, 65):
        b = before_counts.get(num, 0)
        a = after_counts.get(num, 0)
        t = b + a
        name = hex_names.get(num, f"卦{num}")
        hex_table.append((num, name, b, a, t))

    # Sort by total descending
    hex_table_sorted = sorted(hex_table, key=lambda x: -x[4])

    # =========================================================
    # 2. Trigram field distributions
    # =========================================================
    trigram_counts = {}
    for field in TRIGRAM_FIELDS:
        counter = Counter()
        missing = 0
        for c in cases:
            val = c.get(field)
            if val and isinstance(val, str) and val in TRIGRAM_NAMES:
                counter[val] += 1
            else:
                missing += 1
        trigram_counts[field] = (counter, missing)

    # =========================================================
    # 3. Build report
    # =========================================================
    lines.append("# Hexagram Coverage Report")
    lines.append("")
    lines.append(f"**Generated**: 2026-03-09")
    lines.append(f"**Total cases**: {total:,}")
    lines.append(f"**Cases with hexagram_number_before**: {total - missing_before:,} ({(total - missing_before) / total * 100:.1f}%)")
    lines.append(f"**Cases with hexagram_number_after**: {total - missing_after:,} ({(total - missing_after) / total * 100:.1f}%)")
    lines.append("")

    # --- Summary stats ---
    zero_hex = [h for h in hex_table if h[4] == 0]
    low_hex = [h for h in hex_table if 0 < h[4] < MIN_SUPPORT]
    covered = [h for h in hex_table if h[4] >= MIN_SUPPORT]

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Hexagrams with >= {MIN_SUPPORT} occurrences | {len(covered)} / 64 |")
    lines.append(f"| Hexagrams with 1-{MIN_SUPPORT - 1} occurrences | {len(low_hex)} |")
    lines.append(f"| Hexagrams with 0 occurrences | {len(zero_hex)} |")
    lines.append(f"| Target: >= {TARGET_COVERAGE}/64 with >= {MIN_SUPPORT} support | {'ACHIEVED' if len(covered) >= TARGET_COVERAGE else 'NOT YET'} |")
    lines.append("")

    # --- Section A: Full 64-hexagram frequency table ---
    lines.append("## A. Full 64-Hexagram Frequency Table (sorted by total)")
    lines.append("")
    lines.append("| Rank | # | Hexagram | Before | After | Total | Status |")
    lines.append("|------|---|----------|--------|-------|-------|--------|")
    for rank, (num, name, b, a, t) in enumerate(hex_table_sorted, 1):
        if t == 0:
            status = "ZERO"
        elif t < MIN_SUPPORT:
            status = "LOW"
        else:
            status = "OK"
        lines.append(f"| {rank} | {num} | {name} | {b} | {a} | {t} | {status} |")
    lines.append("")

    # --- Section B: Zero-occurrence hexagrams ---
    lines.append("## B. Zero-Occurrence Hexagrams")
    lines.append("")
    if zero_hex:
        lines.append(f"**{len(zero_hex)} hexagrams have zero occurrences** (neither as before nor after):")
        lines.append("")
        lines.append("| # | Hexagram |")
        lines.append("|---|----------|")
        for num, name, b, a, t in sorted(zero_hex, key=lambda x: x[0]):
            lines.append(f"| {num} | {name} |")
    else:
        lines.append("All 64 hexagrams have at least 1 occurrence.")
    lines.append("")

    # --- Section C: Low-occurrence hexagrams (<5) ---
    lines.append(f"## C. Low-Occurrence Hexagrams (1 to {MIN_SUPPORT - 1})")
    lines.append("")
    if low_hex:
        low_sorted = sorted(low_hex, key=lambda x: x[4])
        lines.append(f"**{len(low_hex)} hexagrams** have fewer than {MIN_SUPPORT} total occurrences:")
        lines.append("")
        lines.append("| # | Hexagram | Before | After | Total |")
        lines.append("|---|----------|--------|-------|-------|")
        for num, name, b, a, t in low_sorted:
            lines.append(f"| {num} | {name} | {b} | {a} | {t} |")
    else:
        lines.append(f"All hexagrams have >= {MIN_SUPPORT} occurrences.")
    lines.append("")

    # --- Section D: Trigram field distributions ---
    lines.append("## D. Trigram Field Distributions")
    lines.append("")

    for field in TRIGRAM_FIELDS:
        counter, missing = trigram_counts[field]
        field_total = sum(counter.values())
        lines.append(f"### {field}")
        lines.append("")
        lines.append(f"Valid entries: {field_total:,} / {total:,} ({field_total / total * 100:.1f}%), Missing: {missing:,}")
        lines.append("")
        lines.append("| Trigram | Count | % of valid |")
        lines.append("|--------|-------|------------|")
        for tg in sorted(TRIGRAM_NAMES, key=lambda t: -counter.get(t, 0)):
            cnt = counter.get(tg, 0)
            pct = cnt / field_total * 100 if field_total > 0 else 0
            lines.append(f"| {tg} | {cnt:,} | {pct:.1f}% |")
        lines.append("")

    # --- Section E: Trigram x field combinations with 0% or <1% ---
    lines.append("## E. Underrepresented Trigram x Field Combinations")
    lines.append("")
    lines.append("Combinations where a trigram has < 1% share in a given field:")
    lines.append("")
    lines.append("| Field | Trigram | Count | % |")
    lines.append("|-------|--------|-------|---|")
    has_underrep = False
    for field in TRIGRAM_FIELDS:
        counter, _ = trigram_counts[field]
        field_total = sum(counter.values())
        for tg in TRIGRAM_NAMES:
            cnt = counter.get(tg, 0)
            pct = cnt / field_total * 100 if field_total > 0 else 0
            if pct < 1.0:
                lines.append(f"| {field} | {tg} | {cnt} | {pct:.2f}% |")
                has_underrep = True
    if not has_underrep:
        lines.append("| (none) | - | - | - |")
    lines.append("")

    # Also add a cross-table: trigram-pair coverage (before_lower x before_upper = 64 combos)
    lines.append("### Trigram Pair Coverage (before: lower x upper)")
    lines.append("")
    lines.append("Each cell = number of cases with that trigram combination in before_lower x before_upper.")
    lines.append("King Wen number shown in parentheses where known.")
    lines.append("")

    # Build trigram pair -> King Wen number mapping
    # We can derive it from the reference data
    kw_lookup = {}
    try:
        with open(REFERENCE_PATH, encoding="utf-8") as f:
            ref = json.load(f)
        for num_str, h in ref["hexagrams"].items():
            # Try to extract trigrams from local_name or other fields
            pass  # We'll build from cases instead
    except Exception:
        pass

    # Build from actual data: (lower, upper) -> set of hexagram_numbers seen
    pair_to_hex = defaultdict(set)
    pair_count_before = Counter()
    pair_count_after = Counter()
    for c in cases:
        bl = c.get("before_lower_trigram")
        bu = c.get("before_upper_trigram")
        if bl and bu and bl in TRIGRAM_NAMES and bu in TRIGRAM_NAMES:
            pair_count_before[(bl, bu)] += 1
            bn = c.get("hexagram_number_before")
            if bn:
                pair_to_hex[(bl, bu)].add(int(bn))

        al = c.get("after_lower_trigram")
        au = c.get("after_upper_trigram")
        if al and au and al in TRIGRAM_NAMES and au in TRIGRAM_NAMES:
            pair_count_after[(al, au)] += 1
            an = c.get("hexagram_number_after")
            if an:
                pair_to_hex[(al, au)].add(int(an))

    # Cross table header
    header = "| lower \\ upper |"
    for tg in TRIGRAM_NAMES:
        header += f" {tg} |"
    lines.append(header)
    lines.append("|" + "---|" * (len(TRIGRAM_NAMES) + 1))

    for lower in TRIGRAM_NAMES:
        row = f"| {lower} |"
        for upper in TRIGRAM_NAMES:
            cnt = pair_count_before.get((lower, upper), 0)
            hexnums = pair_to_hex.get((lower, upper), set())
            hex_label = ",".join(str(h) for h in sorted(hexnums)) if hexnums else "?"
            if cnt == 0:
                row += f" **0** (#{hex_label}) |"
            else:
                row += f" {cnt} (#{hex_label}) |"
        lines.append(row)
    lines.append("")

    lines.append("### Trigram Pair Coverage (after: lower x upper)")
    lines.append("")
    header = "| lower \\ upper |"
    for tg in TRIGRAM_NAMES:
        header += f" {tg} |"
    lines.append(header)
    lines.append("|" + "---|" * (len(TRIGRAM_NAMES) + 1))

    for lower in TRIGRAM_NAMES:
        row = f"| {lower} |"
        for upper in TRIGRAM_NAMES:
            cnt = pair_count_after.get((lower, upper), 0)
            hexnums = pair_to_hex.get((lower, upper), set())
            hex_label = ",".join(str(h) for h in sorted(hexnums)) if hexnums else "?"
            if cnt == 0:
                row += f" **0** (#{hex_label}) |"
            else:
                row += f" {cnt} (#{hex_label}) |"
        lines.append(row)
    lines.append("")

    # --- Section F: Gap estimation ---
    lines.append("## F. Gap Estimation")
    lines.append("")

    gap_hexagrams = [h for h in hex_table if h[4] < MIN_SUPPORT]
    cases_needed = sum(max(0, MIN_SUPPORT - h[4]) for h in gap_hexagrams)
    lines.append(f"### Minimum additional cases to reach >= {TARGET_COVERAGE}/64 hexagrams with >= {MIN_SUPPORT} support")
    lines.append("")
    lines.append(f"- Hexagrams currently below {MIN_SUPPORT} support: **{len(gap_hexagrams)}**")
    lines.append(f"- Hexagrams currently at 0: **{len(zero_hex)}**")
    lines.append(f"- Hexagrams at 1-{MIN_SUPPORT - 1}: **{len(low_hex)}**")
    lines.append(f"- Hexagrams at >= {MIN_SUPPORT}: **{len(covered)}**")
    lines.append("")

    # If we already have >= 60/64 covered, we're done
    if len(covered) >= TARGET_COVERAGE:
        lines.append(f"**Target already achieved**: {len(covered)} / 64 hexagrams have >= {MIN_SUPPORT} support.")
        shortfall = 64 - len(covered)
        if shortfall > 0:
            remaining_needed = sum(max(0, MIN_SUPPORT - h[4]) for h in hex_table if h[4] < MIN_SUPPORT)
            lines.append(f"To reach full 64/64 coverage, {remaining_needed} additional targeted cases are needed.")
    else:
        # Calculate how many of the gap hexagrams we need to fill to reach TARGET_COVERAGE
        needed_to_fill = TARGET_COVERAGE - len(covered)
        # Sort gap hexagrams by how close they are to MIN_SUPPORT (easiest to fill first)
        gap_sorted = sorted(gap_hexagrams, key=lambda x: -x[4])  # highest first = cheapest
        cheapest = gap_sorted[:needed_to_fill]
        min_cases_for_target = sum(max(0, MIN_SUPPORT - h[4]) for h in cheapest)
        full_cases = sum(max(0, MIN_SUPPORT - h[4]) for h in gap_hexagrams)

        lines.append(f"**To reach {TARGET_COVERAGE}/64 (easiest path):**")
        lines.append(f"- Fill the {needed_to_fill} gap hexagrams closest to {MIN_SUPPORT}")
        lines.append(f"- Minimum cases needed: **{min_cases_for_target}**")
        lines.append(f"- These hexagrams:")
        lines.append("")
        lines.append("| # | Hexagram | Current | Needed |")
        lines.append("|---|----------|---------|--------|")
        for num, name, b, a, t in cheapest:
            lines.append(f"| {num} | {name} | {t} | {MIN_SUPPORT - t} |")
        lines.append("")

        lines.append(f"**To reach full 64/64:**")
        lines.append(f"- Total additional cases needed: **{full_cases}**")
        lines.append("")

    # --- Before/After balance ---
    lines.append("## G. Before vs After Balance")
    lines.append("")
    lines.append("Hexagrams that appear heavily skewed (>80% in one role):")
    lines.append("")
    lines.append("| # | Hexagram | Before | After | Total | Before% |")
    lines.append("|---|----------|--------|-------|-------|---------|")
    skewed = []
    for num, name, b, a, t in hex_table:
        if t >= MIN_SUPPORT:
            b_pct = b / t * 100 if t > 0 else 0
            if b_pct > 80 or b_pct < 20:
                skewed.append((num, name, b, a, t, b_pct))
    skewed.sort(key=lambda x: x[5])
    for num, name, b, a, t, b_pct in skewed:
        lines.append(f"| {num} | {name} | {b} | {a} | {t} | {b_pct:.1f}% |")
    if not skewed:
        lines.append("| (none with >= 5 total that are >80% skewed) | - | - | - | - | - |")
    lines.append("")

    # --- Top 10 / Bottom 10 ---
    lines.append("## H. Top 10 and Bottom 10 Hexagrams")
    lines.append("")
    lines.append("### Top 10 (most represented)")
    lines.append("")
    lines.append("| Rank | # | Hexagram | Before | After | Total |")
    lines.append("|------|---|----------|--------|-------|-------|")
    for rank, (num, name, b, a, t) in enumerate(hex_table_sorted[:10], 1):
        lines.append(f"| {rank} | {num} | {name} | {b} | {a} | {t} |")
    lines.append("")

    lines.append("### Bottom 10 (least represented)")
    lines.append("")
    lines.append("| Rank | # | Hexagram | Before | After | Total |")
    lines.append("|------|---|----------|--------|-------|-------|")
    for rank, (num, name, b, a, t) in enumerate(hex_table_sorted[-10:], len(hex_table_sorted) - 9):
        lines.append(f"| {rank} | {num} | {name} | {b} | {a} | {t} |")
    lines.append("")

    return "\n".join(lines)


def main():
    output_path = DEFAULT_OUTPUT
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = Path(sys.argv[idx + 1])

    # Load data
    print(f"Loading cases from {CASES_PATH}...")
    cases = load_cases(CASES_PATH)
    print(f"  Loaded {len(cases):,} cases.")

    print(f"Loading hexagram reference from {REFERENCE_PATH}...")
    hex_names = load_hexagram_names(REFERENCE_PATH)
    print(f"  Loaded {len(hex_names)} hexagram names.")

    # Analyze
    print("Running analysis...")
    report = analyze(cases, hex_names)

    # Output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")

    # Also print to stdout
    print("\n" + "=" * 80)
    print(report)


if __name__ == "__main__":
    main()
