#!/usr/bin/env python3
"""
Dual annotation agreement analysis for trigram gold set.

Reads annotations_pass1.json and annotations_pass2.json, then computes:
- Per-field raw agreement rate
- Per-field Cohen's kappa
- Per-field Gwet's AC1 (prevalence-adjusted)
- Per-trigram agreement matrix (confusion matrix)
- Most confused trigram pairs
- Cases needing adjudication (disagreement on any field)
- Confidence-stratified agreement
- Distribution of trigrams per pass (to detect bias)

Outputs:
  analysis/gold_set/dual_annotation_agreement.md

Usage:
  python3 scripts/analyze_annotation_agreement.py
  python3 scripts/analyze_annotation_agreement.py --pass1 path/to/pass1.json --pass2 path/to/pass2.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
DEFAULT_PASS1 = BASE / "analysis" / "gold_set" / "annotations_pass1.json"
DEFAULT_PASS2 = BASE / "analysis" / "gold_set" / "annotations_pass2.json"
OUTPUT_MD = BASE / "analysis" / "gold_set" / "dual_annotation_agreement.md"
OUTPUT_JSON = BASE / "analysis" / "gold_set" / "dual_annotation_agreement.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRIGRAMS = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]
FIELDS = ["before_lower", "before_upper", "after_lower", "after_upper"]
CONFIDENCE_LEVELS = ["high", "medium", "low"]


def load_annotations(path: Path) -> dict[str, dict]:
    """Load annotations file and return dict keyed by transition_id."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    result = {}
    for ann in annotations:
        tid = ann.get("transition_id")
        if tid and "error" not in ann:
            result[tid] = ann
    return result


def extract_pairs(pass1: dict, pass2: dict, field: str) -> list[tuple[str, str]]:
    """Extract aligned (pass1_value, pass2_value) pairs for a field."""
    pairs = []
    common_ids = set(pass1.keys()) & set(pass2.keys())
    for tid in sorted(common_ids):
        v1 = pass1[tid].get(field)
        v2 = pass2[tid].get(field)
        if v1 in TRIGRAMS and v2 in TRIGRAMS:
            pairs.append((v1, v2))
    return pairs


def raw_agreement(pairs: list[tuple[str, str]]) -> float:
    """Calculate raw agreement rate."""
    if not pairs:
        return 0.0
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    """Calculate Cohen's kappa."""
    n = len(pairs)
    if n == 0:
        return 0.0

    po = sum(1 for a, b in pairs if a == b) / n

    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)

    pe = sum(a_counts.get(t, 0) * b_counts.get(t, 0) for t in TRIGRAMS) / (n * n)

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def gwets_ac1(pairs: list[tuple[str, str]]) -> float:
    """Calculate Gwet's AC1 (prevalence-adjusted agreement)."""
    n = len(pairs)
    if n == 0:
        return 0.0

    po = sum(1 for a, b in pairs if a == b) / n

    # Marginal proportions
    all_labels = [a for a, _ in pairs] + [b for _, b in pairs]
    total = len(all_labels)
    pi = {t: all_labels.count(t) / total for t in TRIGRAMS}

    k = len(TRIGRAMS)
    pe = sum(pk * (1 - pk) for pk in pi.values()) / (k - 1)

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Build confusion matrix: rows=pass1, cols=pass2."""
    matrix = {t1: {t2: 0 for t2 in TRIGRAMS} for t1 in TRIGRAMS}
    for a, b in pairs:
        if a in TRIGRAMS and b in TRIGRAMS:
            matrix[a][b] += 1
    return matrix


def top_confusion_pairs(matrix: dict[str, dict[str, int]], top_n: int = 10) -> list[tuple[str, str, int]]:
    """Find the most frequently confused trigram pairs (off-diagonal)."""
    pairs = []
    for t1 in TRIGRAMS:
        for t2 in TRIGRAMS:
            if t1 != t2:
                count = matrix[t1][t2] + matrix[t2][t1]
                pairs.append((t1, t2, count))

    # Deduplicate (A,B) and (B,A)
    seen = set()
    deduped = []
    for t1, t2, count in pairs:
        key = tuple(sorted([t1, t2]))
        if key not in seen:
            seen.add(key)
            deduped.append((key[0], key[1], count))

    deduped.sort(key=lambda x: -x[2])
    return deduped[:top_n]


def trigram_distribution(annotations: dict[str, dict], field: str) -> dict[str, int]:
    """Count trigram distribution for a field."""
    counts = Counter()
    for ann in annotations.values():
        v = ann.get(field)
        if v in TRIGRAMS:
            counts[v] += 1
    return dict(counts)


def find_disagreement_cases(pass1: dict, pass2: dict) -> list[dict]:
    """Find cases where pass1 and pass2 disagree on any field."""
    disagreements = []
    common_ids = set(pass1.keys()) & set(pass2.keys())

    for tid in sorted(common_ids):
        a1 = pass1[tid]
        a2 = pass2[tid]
        diffs = {}
        for field in FIELDS:
            v1 = a1.get(field)
            v2 = a2.get(field)
            if v1 != v2:
                diffs[field] = {"pass1": v1, "pass2": v2}

        if diffs:
            disagreements.append({
                "transition_id": tid,
                "target_name": a1.get("target_name", ""),
                "disagreements": diffs,
                "num_disagreements": len(diffs),
            })

    return disagreements


def confidence_stratified_agreement(
    pass1: dict, pass2: dict, field: str
) -> dict[str, dict]:
    """Calculate agreement rate stratified by pass1's confidence level."""
    results = {}
    conf_field = f"{field}_confidence"
    common_ids = set(pass1.keys()) & set(pass2.keys())

    for level in CONFIDENCE_LEVELS:
        pairs = []
        for tid in common_ids:
            if pass1[tid].get(conf_field) == level:
                v1 = pass1[tid].get(field)
                v2 = pass2[tid].get(field)
                if v1 in TRIGRAMS and v2 in TRIGRAMS:
                    pairs.append((v1, v2))

        if pairs:
            results[level] = {
                "n": len(pairs),
                "agreement": round(raw_agreement(pairs), 4),
                "kappa": round(cohens_kappa(pairs), 4),
            }
        else:
            results[level] = {"n": 0, "agreement": None, "kappa": None}

    return results


def format_confusion_matrix_md(matrix: dict[str, dict[str, int]], field_name: str) -> str:
    """Format confusion matrix as markdown table."""
    lines = [f"#### {field_name}"]
    lines.append("")
    header = "| P1 \\ P2 | " + " | ".join(TRIGRAMS) + " | Total |"
    lines.append(header)
    lines.append("|" + "---|" * (len(TRIGRAMS) + 2))

    for t1 in TRIGRAMS:
        row_total = sum(matrix[t1].values())
        cells = [str(matrix[t1][t2]) for t2 in TRIGRAMS]
        lines.append(f"| **{t1}** | " + " | ".join(cells) + f" | {row_total} |")

    # Column totals
    col_totals = [sum(matrix[t1][t2] for t1 in TRIGRAMS) for t2 in TRIGRAMS]
    grand_total = sum(col_totals)
    lines.append(
        "| **Total** | " + " | ".join(str(c) for c in col_totals) + f" | {grand_total} |"
    )
    lines.append("")
    return "\n".join(lines)


def generate_report(
    pass1: dict,
    pass2: dict,
) -> tuple[str, dict]:
    """Generate full agreement report. Returns (markdown_text, json_data)."""
    common_ids = set(pass1.keys()) & set(pass2.keys())
    n_common = len(common_ids)
    n_pass1 = len(pass1)
    n_pass2 = len(pass2)

    # --- Per-field metrics ---
    field_metrics = {}
    for field in FIELDS:
        pairs = extract_pairs(pass1, pass2, field)
        field_metrics[field] = {
            "n": len(pairs),
            "raw_agreement": round(raw_agreement(pairs), 4),
            "cohens_kappa": round(cohens_kappa(pairs), 4),
            "gwets_ac1": round(gwets_ac1(pairs), 4),
        }

    # --- Overall (average across fields) ---
    avg_agreement = sum(m["raw_agreement"] for m in field_metrics.values()) / len(FIELDS)
    avg_kappa = sum(m["cohens_kappa"] for m in field_metrics.values()) / len(FIELDS)
    avg_ac1 = sum(m["gwets_ac1"] for m in field_metrics.values()) / len(FIELDS)

    # --- Confusion matrices ---
    confusion_matrices = {}
    top_confusions = {}
    for field in FIELDS:
        pairs = extract_pairs(pass1, pass2, field)
        cm = confusion_matrix(pairs)
        confusion_matrices[field] = cm
        top_confusions[field] = top_confusion_pairs(cm, top_n=5)

    # --- Trigram distributions ---
    distributions = {"pass1": {}, "pass2": {}}
    for field in FIELDS:
        distributions["pass1"][field] = trigram_distribution(pass1, field)
        distributions["pass2"][field] = trigram_distribution(pass2, field)

    # --- Disagreement cases ---
    disagreements = find_disagreement_cases(pass1, pass2)
    n_full_agree = n_common - len(disagreements)

    # --- Confidence-stratified ---
    conf_stratified = {}
    for field in FIELDS:
        conf_stratified[field] = confidence_stratified_agreement(pass1, pass2, field)

    # --- Check if 離/兌 appear as lower (inner) trigrams ---
    li_dui_inner = {"pass1": {"離": 0, "兌": 0}, "pass2": {"離": 0, "兌": 0}}
    for field in ["before_lower", "after_lower"]:
        for tid in common_ids:
            v1 = pass1[tid].get(field)
            v2 = pass2[tid].get(field)
            if v1 == "離":
                li_dui_inner["pass1"]["離"] += 1
            if v1 == "兌":
                li_dui_inner["pass1"]["兌"] += 1
            if v2 == "離":
                li_dui_inner["pass2"]["離"] += 1
            if v2 == "兌":
                li_dui_inner["pass2"]["兌"] += 1

    # ===================================================================
    # Build Markdown report
    # ===================================================================
    md_lines = []
    md_lines.append("# Dual Annotation Agreement Report")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append(f"- **Pass 1 annotations**: {n_pass1}")
    md_lines.append(f"- **Pass 2 annotations**: {n_pass2}")
    md_lines.append(f"- **Common (matched) cases**: {n_common}")
    md_lines.append(f"- **Full agreement (all 4 fields)**: {n_full_agree} ({n_full_agree/n_common:.1%})")
    md_lines.append(f"- **Cases with any disagreement**: {len(disagreements)} ({len(disagreements)/n_common:.1%})")
    md_lines.append("")

    md_lines.append("## Per-Field Agreement")
    md_lines.append("")
    md_lines.append("| Field | N | Raw Agreement | Cohen's kappa | Gwet's AC1 |")
    md_lines.append("|-------|---|---------------|---------------|------------|")
    for field in FIELDS:
        m = field_metrics[field]
        md_lines.append(
            f"| {field} | {m['n']} | {m['raw_agreement']:.1%} | {m['cohens_kappa']:.3f} | {m['gwets_ac1']:.3f} |"
        )
    md_lines.append(
        f"| **Average** | - | {avg_agreement:.1%} | {avg_kappa:.3f} | {avg_ac1:.3f} |"
    )
    md_lines.append("")

    # Kappa interpretation
    md_lines.append("### Kappa Interpretation")
    md_lines.append("")
    md_lines.append("| Range | Interpretation |")
    md_lines.append("|-------|---------------|")
    md_lines.append("| < 0.20 | Poor |")
    md_lines.append("| 0.21-0.40 | Fair |")
    md_lines.append("| 0.41-0.60 | Moderate |")
    md_lines.append("| 0.61-0.80 | Substantial |")
    md_lines.append("| 0.81-1.00 | Almost perfect |")
    md_lines.append("")

    # Quality gate
    pass_fail = "PASS" if avg_kappa >= 0.60 and avg_ac1 >= 0.65 else "FAIL"
    if avg_kappa >= 0.60 and avg_ac1 < 0.65:
        pass_fail = "CONDITIONAL PASS (check prevalence bias)"
    elif avg_kappa < 0.60 and avg_ac1 >= 0.65:
        pass_fail = "CONDITIONAL PASS (check low-frequency categories)"

    md_lines.append(f"**Quality Gate**: {pass_fail}")
    md_lines.append("")

    # --- 離/兌 inner trigram check ---
    md_lines.append("## 離/兌 Inner Trigram Coverage")
    md_lines.append("")
    md_lines.append("Critical check: do 離 and 兌 now appear as lower (inner) trigrams?")
    md_lines.append("")
    md_lines.append("| Trigram | Pass 1 (as inner) | Pass 2 (as inner) |")
    md_lines.append("|---------|-------------------|-------------------|")
    md_lines.append(
        f"| 離 | {li_dui_inner['pass1']['離']} | {li_dui_inner['pass2']['離']} |"
    )
    md_lines.append(
        f"| 兌 | {li_dui_inner['pass1']['兌']} | {li_dui_inner['pass2']['兌']} |"
    )
    md_lines.append("")
    if li_dui_inner["pass1"]["離"] > 0 or li_dui_inner["pass2"]["離"] > 0:
        md_lines.append("離 as inner: **Fixed** (appears in annotations)")
    else:
        md_lines.append("離 as inner: **Still missing** -- prompt may need revision")
    if li_dui_inner["pass1"]["兌"] > 0 or li_dui_inner["pass2"]["兌"] > 0:
        md_lines.append("兌 as inner: **Fixed** (appears in annotations)")
    else:
        md_lines.append("兌 as inner: **Still missing** -- prompt may need revision")
    md_lines.append("")

    # --- Confusion matrices ---
    md_lines.append("## Confusion Matrices (Pass1 rows x Pass2 cols)")
    md_lines.append("")
    for field in FIELDS:
        md_lines.append(format_confusion_matrix_md(confusion_matrices[field], field))

    # --- Top confused pairs ---
    md_lines.append("## Most Confused Trigram Pairs")
    md_lines.append("")
    for field in FIELDS:
        md_lines.append(f"### {field}")
        md_lines.append("")
        md_lines.append("| Pair | Total Confusions |")
        md_lines.append("|------|-----------------|")
        for t1, t2, count in top_confusions[field]:
            if count > 0:
                md_lines.append(f"| {t1} <-> {t2} | {count} |")
        md_lines.append("")

    # --- Trigram distributions ---
    md_lines.append("## Trigram Distribution Comparison")
    md_lines.append("")
    for field in FIELDS:
        md_lines.append(f"### {field}")
        md_lines.append("")
        md_lines.append("| Trigram | Pass 1 | Pass 2 | Diff |")
        md_lines.append("|---------|--------|--------|------|")
        d1 = distributions["pass1"][field]
        d2 = distributions["pass2"][field]
        for t in TRIGRAMS:
            c1 = d1.get(t, 0)
            c2 = d2.get(t, 0)
            diff = c2 - c1
            sign = "+" if diff > 0 else ""
            md_lines.append(f"| {t} | {c1} | {c2} | {sign}{diff} |")
        md_lines.append("")

    # --- Confidence-stratified agreement ---
    md_lines.append("## Confidence-Stratified Agreement (by Pass 1 confidence)")
    md_lines.append("")
    for field in FIELDS:
        md_lines.append(f"### {field}")
        md_lines.append("")
        md_lines.append("| Confidence | N | Agreement | Kappa |")
        md_lines.append("|------------|---|-----------|-------|")
        cs = conf_stratified[field]
        for level in CONFIDENCE_LEVELS:
            data = cs[level]
            if data["n"] > 0:
                md_lines.append(
                    f"| {level} | {data['n']} | {data['agreement']:.1%} | {data['kappa']:.3f} |"
                )
            else:
                md_lines.append(f"| {level} | 0 | - | - |")
        md_lines.append("")

    # --- Cases needing adjudication ---
    md_lines.append("## Cases Needing Adjudication")
    md_lines.append("")
    md_lines.append(f"Total: {len(disagreements)} cases with at least one field disagreement.")
    md_lines.append("")

    # Group by number of disagreements
    by_count = defaultdict(int)
    for d in disagreements:
        by_count[d["num_disagreements"]] += 1

    md_lines.append("| Disagreements per case | Count |")
    md_lines.append("|------------------------|-------|")
    for n_dis in sorted(by_count.keys()):
        md_lines.append(f"| {n_dis} field(s) | {by_count[n_dis]} |")
    md_lines.append("")

    # Show first 20 disagreement cases
    md_lines.append("### Sample Disagreements (first 20)")
    md_lines.append("")
    for d in disagreements[:20]:
        md_lines.append(f"**{d['transition_id']}** ({d['target_name']})")
        for field, vals in d["disagreements"].items():
            md_lines.append(f"  - {field}: Pass1={vals['pass1']} vs Pass2={vals['pass2']}")
        md_lines.append("")

    md_text = "\n".join(md_lines)

    # ===================================================================
    # Build JSON data
    # ===================================================================
    json_data = {
        "summary": {
            "n_pass1": n_pass1,
            "n_pass2": n_pass2,
            "n_common": n_common,
            "n_full_agreement": n_full_agree,
            "n_disagreement": len(disagreements),
            "avg_raw_agreement": round(avg_agreement, 4),
            "avg_cohens_kappa": round(avg_kappa, 4),
            "avg_gwets_ac1": round(avg_ac1, 4),
            "quality_gate": pass_fail,
        },
        "per_field": field_metrics,
        "li_dui_inner_coverage": li_dui_inner,
        "top_confusions": {
            field: [{"pair": [t1, t2], "count": c} for t1, t2, c in top_confusions[field]]
            for field in FIELDS
        },
        "distributions": distributions,
        "confidence_stratified": conf_stratified,
        "disagreement_case_ids": [d["transition_id"] for d in disagreements],
    }

    return md_text, json_data


def main():
    parser = argparse.ArgumentParser(description="Analyze dual annotation agreement")
    parser.add_argument("--pass1", type=str, default=None, help=f"Path to pass 1 annotations (default: {DEFAULT_PASS1})")
    parser.add_argument("--pass2", type=str, default=None, help=f"Path to pass 2 annotations (default: {DEFAULT_PASS2})")
    parser.add_argument("--output-md", type=str, default=None, help=f"Output markdown path (default: {OUTPUT_MD})")
    parser.add_argument("--output-json", type=str, default=None, help=f"Output JSON path (default: {OUTPUT_JSON})")
    args = parser.parse_args()

    pass1_path = Path(args.pass1) if args.pass1 else DEFAULT_PASS1
    pass2_path = Path(args.pass2) if args.pass2 else DEFAULT_PASS2
    output_md_path = Path(args.output_md) if args.output_md else OUTPUT_MD
    output_json_path = Path(args.output_json) if args.output_json else OUTPUT_JSON

    # Check inputs exist
    if not pass1_path.exists():
        print(f"ERROR: Pass 1 file not found: {pass1_path}", file=sys.stderr)
        sys.exit(1)
    if not pass2_path.exists():
        print(f"ERROR: Pass 2 file not found: {pass2_path}", file=sys.stderr)
        sys.exit(1)

    # Load
    print(f"Loading pass 1: {pass1_path}")
    pass1 = load_annotations(pass1_path)
    print(f"  {len(pass1)} valid annotations")

    print(f"Loading pass 2: {pass2_path}")
    pass2 = load_annotations(pass2_path)
    print(f"  {len(pass2)} valid annotations")

    common = set(pass1.keys()) & set(pass2.keys())
    print(f"Common cases: {len(common)}")

    if not common:
        print("ERROR: No common cases between pass 1 and pass 2.", file=sys.stderr)
        sys.exit(1)

    # Generate report
    md_text, json_data = generate_report(pass1, pass2)

    # Save
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Markdown report: {output_md_path}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"JSON report: {output_json_path}")

    # Print summary
    print()
    print("=== SUMMARY ===")
    s = json_data["summary"]
    print(f"  Common cases: {s['n_common']}")
    print(f"  Full agreement: {s['n_full_agreement']} ({s['n_full_agreement']/s['n_common']:.1%})")
    print(f"  Avg raw agreement: {s['avg_raw_agreement']:.1%}")
    print(f"  Avg Cohen's kappa: {s['avg_cohens_kappa']:.3f}")
    print(f"  Avg Gwet's AC1: {s['avg_gwets_ac1']:.3f}")
    print(f"  Quality gate: {s['quality_gate']}")
    print()

    # 離/兌 check
    lid = json_data["li_dui_inner_coverage"]
    p1_li = lid["pass1"]["離"]
    p1_dui = lid["pass1"]["兌"]
    p2_li = lid["pass2"]["離"]
    p2_dui = lid["pass2"]["兌"]
    print(f"  離 as inner: Pass1={p1_li}, Pass2={p2_li}")
    print(f"  兌 as inner: Pass1={p1_dui}, Pass2={p2_dui}")

    if p1_li == 0 and p2_li == 0:
        print("  WARNING: 離 never appears as inner trigram!")
    if p1_dui == 0 and p2_dui == 0:
        print("  WARNING: 兌 never appears as inner trigram!")


if __name__ == "__main__":
    main()
