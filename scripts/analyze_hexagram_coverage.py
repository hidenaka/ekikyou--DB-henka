#!/usr/bin/env python3
"""
Hexagram Coverage Analysis (Step 0-C)

Reads all cases from data/raw/cases.jsonl and produces a comprehensive
hexagram coverage report including:
  - Trigram distribution per field
  - Hexagram frequency table (before/after)
  - Coverage summary
  - Gap analysis (0-occurrence hexagrams)
  - Concentration metrics (Gini coefficient, Shannon entropy)
  - Transition graph statistics

Usage:
    python3 scripts/analyze_hexagram_coverage.py
"""

import json
import sys
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import math

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
CASES_PATH = BASE / "data" / "raw" / "cases.jsonl"
OUTPUT_DIR = BASE / "analysis" / "phase3"
OUTPUT_PATH = OUTPUT_DIR / "hexagram_coverage_report.md"

# ── King Wen Mapping ──────────────────────────────────────────────────────
KING_WEN = {
    ("乾","乾"):1, ("坤","乾"):11, ("震","乾"):34, ("巽","乾"):9,
    ("坎","乾"):5, ("離","乾"):14, ("艮","乾"):26, ("兌","乾"):43,
    ("乾","坤"):12, ("坤","坤"):2, ("震","坤"):16, ("巽","坤"):20,
    ("坎","坤"):8, ("離","坤"):35, ("艮","坤"):23, ("兌","坤"):45,
    ("乾","震"):25, ("坤","震"):24, ("震","震"):51, ("巽","震"):42,
    ("坎","震"):3, ("離","震"):21, ("艮","震"):27, ("兌","震"):17,
    ("乾","巽"):44, ("坤","巽"):46, ("震","巽"):32, ("巽","巽"):57,
    ("坎","巽"):48, ("離","巽"):50, ("艮","巽"):18, ("兌","巽"):28,
    ("乾","坎"):6, ("坤","坎"):7, ("震","坎"):40, ("巽","坎"):59,
    ("坎","坎"):29, ("離","坎"):64, ("艮","坎"):4, ("兌","坎"):47,
    ("乾","離"):13, ("坤","離"):36, ("震","離"):55, ("巽","離"):37,
    ("坎","離"):63, ("離","離"):30, ("艮","離"):22, ("兌","離"):49,
    ("乾","艮"):33, ("坤","艮"):15, ("震","艮"):62, ("巽","艮"):53,
    ("坎","艮"):39, ("離","艮"):56, ("艮","艮"):52, ("兌","艮"):31,
    ("乾","兌"):10, ("坤","兌"):19, ("震","兌"):54, ("巽","兌"):61,
    ("坎","兌"):60, ("離","兌"):38, ("艮","兌"):41, ("兌","兌"):58,
}

# Reverse lookup: hexagram number -> (lower, upper) trigram pair
REV_KING_WEN = {v: k for k, v in KING_WEN.items()}

HEX_NAMES = {
    1: "乾為天", 2: "坤為地", 3: "水雷屯", 4: "山水蒙",
    5: "水天需", 6: "天水訟", 7: "地水師", 8: "水地比",
    9: "風天小畜", 10: "天沢履", 11: "地天泰", 12: "天地否",
    13: "天火同人", 14: "火天大有", 15: "地山謙", 16: "雷地豫",
    17: "沢雷随", 18: "山風蠱", 19: "地沢臨", 20: "風地観",
    21: "火雷噬嗑", 22: "山火賁", 23: "山地剥", 24: "地雷復",
    25: "天雷無妄", 26: "山天大畜", 27: "山雷頤", 28: "沢風大過",
    29: "坎為水", 30: "離為火", 31: "沢山咸", 32: "雷風恒",
    33: "天山遯", 34: "雷天大壮", 35: "火地晋", 36: "地火明夷",
    37: "風火家人", 38: "火沢睽", 39: "水山蹇", 40: "雷水解",
    41: "山沢損", 42: "風雷益", 43: "沢天夬", 44: "天風姤",
    45: "沢地萃", 46: "地風升", 47: "沢水困", 48: "水風井",
    49: "沢火革", 50: "火風鼎", 51: "震為雷", 52: "艮為山",
    53: "風山漸", 54: "雷沢帰妹", 55: "雷火豊", 56: "火山旅",
    57: "巽為風", 58: "兌為沢", 59: "風水渙", 60: "水沢節",
    61: "風沢中孚", 62: "雷山小過", 63: "水火既済", 64: "火水未済",
}

TRIGRAMS = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]
TRIGRAM_FIELDS = [
    "before_lower_trigram", "before_upper_trigram",
    "after_lower_trigram", "after_upper_trigram",
]

MIN_SUPPORT = 5
TARGET_COVERAGE = 60  # out of 64


# ── Utility Functions ─────────────────────────────────────────────────────

def gini_coefficient(values):
    """Compute the Gini coefficient for a list of non-negative values."""
    if not values or sum(values) == 0:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals):
        weighted_sum += (i + 1) * v
    total = sum(sorted_vals)
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def shannon_entropy(values):
    """Compute Shannon entropy in bits for a frequency distribution."""
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log2(p) for p in probs)


def load_cases():
    """Load all cases from JSONL."""
    cases = []
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def analyze_trigram_distribution(cases):
    """Count trigram frequency per field."""
    counts = {}
    for field in TRIGRAM_FIELDS:
        counter = Counter()
        missing = 0
        for c in cases:
            val = c.get(field)
            if val and isinstance(val, str) and val in TRIGRAMS:
                counter[val] += 1
            else:
                missing += 1
        counts[field] = {"counter": counter, "missing": missing}
    return counts


def analyze_hexagram_frequency(cases):
    """Count hexagram frequency in before/after positions."""
    before_counter = Counter()
    after_counter = Counter()
    missing_before = 0
    missing_after = 0

    for c in cases:
        hb = c.get("hexagram_number_before")
        ha = c.get("hexagram_number_after")
        if hb is not None and isinstance(hb, (int, float)) and 1 <= int(hb) <= 64:
            before_counter[int(hb)] += 1
        else:
            missing_before += 1
        if ha is not None and isinstance(ha, (int, float)) and 1 <= int(ha) <= 64:
            after_counter[int(ha)] += 1
        else:
            missing_after += 1

    return before_counter, after_counter, missing_before, missing_after


def analyze_transition_graph(cases):
    """Analyze the before->after transition graph."""
    edges = Counter()
    for c in cases:
        hb = c.get("hexagram_number_before")
        ha = c.get("hexagram_number_after")
        if (hb is not None and ha is not None
                and isinstance(hb, (int, float)) and isinstance(ha, (int, float))):
            hb, ha = int(hb), int(ha)
            if 1 <= hb <= 64 and 1 <= ha <= 64:
                edges[(hb, ha)] += 1

    # Node metrics
    out_degree = Counter()
    in_degree = Counter()
    for (src, dst), w in edges.items():
        out_degree[src] += 1
        in_degree[dst] += 1

    active_nodes = set()
    for (src, dst) in edges:
        active_nodes.add(src)
        active_nodes.add(dst)

    # Edge weight stats
    weights = list(edges.values())
    if weights:
        mean_w = sum(weights) / len(weights)
        sorted_w = sorted(weights)
        n = len(sorted_w)
        median_w = sorted_w[n // 2] if n % 2 == 1 else (sorted_w[n // 2 - 1] + sorted_w[n // 2]) / 2
        max_w = max(weights)
        min_w = min(weights)
    else:
        mean_w = median_w = max_w = min_w = 0

    # Degree stats
    degree = Counter()
    for n in active_nodes:
        degree[n] = out_degree[n] + in_degree[n]
    deg_vals = list(degree.values()) if degree else [0]
    mean_deg = sum(deg_vals) / len(deg_vals) if deg_vals else 0
    max_deg = max(deg_vals) if deg_vals else 0

    # Graph density: actual edges / possible edges (64*64 = 4096 for directed graph with self-loops)
    possible_edges = 64 * 64
    density = len(edges) / possible_edges

    return {
        "edges": edges,
        "unique_edge_count": len(edges),
        "total_transitions": sum(weights),
        "active_nodes": len(active_nodes),
        "mean_edge_weight": mean_w,
        "median_edge_weight": median_w,
        "max_edge_weight": max_w,
        "min_edge_weight": min_w,
        "mean_degree": mean_deg,
        "max_degree": max_deg,
        "graph_density": density,
        "out_degree": out_degree,
        "in_degree": in_degree,
    }


def generate_report(cases, trigram_dist, before_hex, after_hex,
                    missing_before, missing_after, graph_stats):
    """Generate a comprehensive markdown report."""
    total = len(cases)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    def add(s=""):
        lines.append(s)

    # ── Header ─────────────────────────────────────────────────────────
    add("# Hexagram Coverage Analysis Report (Step 0-C)")
    add()
    add(f"**Generated**: {now}")
    add(f"**Total cases**: {total:,}")
    add(f"**Cases with hexagram_number_before**: {total - missing_before:,} ({(total - missing_before)/total*100:.1f}%)")
    add(f"**Cases with hexagram_number_after**: {total - missing_after:,} ({(total - missing_after)/total*100:.1f}%)")
    add()

    # ── 1. Trigram Distribution ────────────────────────────────────────
    add("## 1. Trigram Distribution")
    add()
    add("Frequency of each trigram across the 4 trigram fields.")
    add()

    # Compact cross-table
    header = "| Trigram |"
    sep = "|---------|"
    for field in TRIGRAM_FIELDS:
        short = field.replace("_trigram", "").replace("_", " ")
        header += f" {short} |"
        sep += "--------|"
    add(header)
    add(sep)

    for tg in TRIGRAMS:
        row = f"| **{tg}** |"
        for field in TRIGRAM_FIELDS:
            cnt = trigram_dist[field]["counter"].get(tg, 0)
            pct = cnt / total * 100 if total else 0
            row += f" {cnt:,} ({pct:.1f}%) |"
        add(row)

    row = "| *(missing)* |"
    for field in TRIGRAM_FIELDS:
        m = trigram_dist[field]["missing"]
        row += f" {m:,} |"
    add(row)
    add()

    # Trigram Gini per field
    add("### Trigram Concentration (Gini per field)")
    add()
    add("| Field | Gini | Interpretation |")
    add("|-------|------|----------------|")
    for field in TRIGRAM_FIELDS:
        vals = [trigram_dist[field]["counter"].get(tg, 0) for tg in TRIGRAMS]
        g = gini_coefficient(vals)
        short = field.replace("_trigram", "").replace("_", " ")
        interp = "High" if g > 0.4 else ("Moderate" if g > 0.25 else "Low")
        add(f"| {short} | {g:.4f} | {interp} concentration |")
    add()
    add("> Gini = 0 means perfectly uniform distribution; Gini approaching 1 means extreme concentration.")
    add()

    # Underrepresented trigram x field combos
    add("### Underrepresented Trigram x Field Combinations (< 1%)")
    add()
    add("| Field | Trigram | Count | % |")
    add("|-------|---------|-------|---|")
    has_underrep = False
    for field in TRIGRAM_FIELDS:
        field_total = sum(trigram_dist[field]["counter"].values())
        for tg in TRIGRAMS:
            cnt = trigram_dist[field]["counter"].get(tg, 0)
            pct = cnt / field_total * 100 if field_total > 0 else 0
            if pct < 1.0:
                short = field.replace("_trigram", "").replace("_", " ")
                add(f"| {short} | {tg} | {cnt} | {pct:.2f}% |")
                has_underrep = True
    if not has_underrep:
        add("| (none) | - | - | - |")
    add()

    # ── 2. Hexagram Frequency Table ───────────────────────────────────
    add("## 2. Full 64-Hexagram Frequency Table")
    add()

    combined = {}
    for h in range(1, 65):
        b = before_hex.get(h, 0)
        a = after_hex.get(h, 0)
        combined[h] = {"before": b, "after": a, "total": b + a}

    sorted_hex = sorted(combined.items(), key=lambda x: -x[1]["total"])

    add("| Rank | # | Name | Before | After | Total | Status |")
    add("|------|---|------|--------|-------|-------|--------|")
    for rank, (h, d) in enumerate(sorted_hex, 1):
        name = HEX_NAMES.get(h, "?")
        if d["total"] >= MIN_SUPPORT:
            status = "OK"
        elif d["total"] >= 1:
            status = "LOW"
        else:
            status = "ZERO"
        add(f"| {rank} | {h} | {name} | {d['before']:,} | {d['after']:,} | {d['total']:,} | {status} |")
    add()

    # ── 3. Coverage Summary ───────────────────────────────────────────
    add("## 3. Coverage Summary")
    add()

    ge5 = sum(1 for d in combined.values() if d["total"] >= 5)
    ge1 = sum(1 for d in combined.values() if d["total"] >= 1)
    low = sum(1 for d in combined.values() if 1 <= d["total"] < 5)
    zero = sum(1 for d in combined.values() if d["total"] == 0)

    add("| Metric | Count |")
    add("|--------|-------|")
    add(f"| Hexagrams with >= {MIN_SUPPORT} occurrences | **{ge5}** / 64 |")
    add(f"| Hexagrams with 1-{MIN_SUPPORT-1} occurrences | {low} |")
    add(f"| Hexagrams with 0 occurrences | **{zero}** |")
    add(f"| Total hexagram types active (>= 1) | {ge1} / 64 |")
    add(f"| Target: >= {TARGET_COVERAGE}/64 with >= {MIN_SUPPORT} support | {'**MET**' if ge5 >= TARGET_COVERAGE else '**NOT MET**'} |")
    add()

    before_active = sum(1 for h in range(1, 65) if before_hex.get(h, 0) > 0)
    after_active = sum(1 for h in range(1, 65) if after_hex.get(h, 0) > 0)
    add(f"- Active as **before** hexagram: {before_active} / 64")
    add(f"- Active as **after** hexagram: {after_active} / 64")
    add()

    # ── 4. Gap Analysis ───────────────────────────────────────────────
    add("## 4. Gap Analysis")
    add()

    # 4a. Zero-occurrence hexagrams
    zero_hexs = [(h, HEX_NAMES.get(h, "?")) for h in range(1, 65) if combined[h]["total"] == 0]
    add("### 4a. Hexagrams with 0 Occurrences (need targeted gold set expansion)")
    add()
    if zero_hexs:
        add(f"**{len(zero_hexs)} hexagrams** have zero occurrences:")
        add()
        add("| # | Name | Lower trigram | Upper trigram |")
        add("|---|------|---------------|---------------|")
        for h, name in zero_hexs:
            lo, up = REV_KING_WEN.get(h, ("?", "?"))
            add(f"| {h} | {name} | {lo} | {up} |")
        add()
    else:
        add("All 64 hexagrams have at least 1 occurrence.")
        add()

    # 4b. Low-occurrence hexagrams
    low_hexs = [(h, combined[h]) for h in range(1, 65) if 1 <= combined[h]["total"] < MIN_SUPPORT]
    add(f"### 4b. Low-Frequency Hexagrams (1-{MIN_SUPPORT-1} total)")
    add()
    if low_hexs:
        add(f"**{len(low_hexs)} hexagrams** below minimum threshold:")
        add()
        add("| # | Name | Before | After | Total |")
        add("|---|------|--------|-------|-------|")
        for h, d in sorted(low_hexs, key=lambda x: x[1]["total"]):
            add(f"| {h} | {HEX_NAMES.get(h, '?')} | {d['before']} | {d['after']} | {d['total']} |")
        add()
    else:
        add(f"All active hexagrams have >= {MIN_SUPPORT} occurrences.")
        add()

    # 4c. Before-only / After-only gaps
    before_zero = [h for h in range(1, 65) if before_hex.get(h, 0) == 0 and combined[h]["total"] > 0]
    after_zero = [h for h in range(1, 65) if after_hex.get(h, 0) == 0 and combined[h]["total"] > 0]

    if before_zero:
        add(f"### 4c. Never appears as BEFORE (but active as after): {len(before_zero)} hexagrams")
        add()
        add("| # | Name | After count |")
        add("|---|------|-------------|")
        for h in sorted(before_zero):
            add(f"| {h} | {HEX_NAMES.get(h, '?')} | {after_hex.get(h, 0):,} |")
        add()

    if after_zero:
        add(f"### 4d. Never appears as AFTER (but active as before): {len(after_zero)} hexagrams")
        add()
        add("| # | Name | Before count |")
        add("|---|------|--------------|")
        for h in sorted(after_zero):
            add(f"| {h} | {HEX_NAMES.get(h, '?')} | {before_hex.get(h, 0):,} |")
        add()

    # 4e. Before/After balance (skew analysis)
    add("### 4e. Before/After Skew (>= 5 total, >80% in one role)")
    add()
    add("| # | Name | Before | After | Total | Before% |")
    add("|---|------|--------|-------|-------|---------|")
    skewed = []
    for h in range(1, 65):
        d = combined[h]
        if d["total"] >= MIN_SUPPORT:
            b_pct = d["before"] / d["total"] * 100
            if b_pct > 80 or b_pct < 20:
                skewed.append((h, d, b_pct))
    skewed.sort(key=lambda x: x[2])
    for h, d, b_pct in skewed:
        add(f"| {h} | {HEX_NAMES.get(h, '?')} | {d['before']:,} | {d['after']:,} | {d['total']:,} | {b_pct:.1f}% |")
    if not skewed:
        add("| (none) | - | - | - | - | - |")
    add()

    # ── 5. Concentration Metrics ──────────────────────────────────────
    add("## 5. Concentration Metrics")
    add()

    all_totals = [combined[h]["total"] for h in range(1, 65)]
    before_vals = [before_hex.get(h, 0) for h in range(1, 65)]
    after_vals = [after_hex.get(h, 0) for h in range(1, 65)]

    gini_all = gini_coefficient(all_totals)
    gini_before = gini_coefficient(before_vals)
    gini_after = gini_coefficient(after_vals)

    add("### 5a. Gini Coefficient")
    add()
    add("| Metric | Gini | Interpretation |")
    add("|--------|------|----------------|")
    add(f"| Combined (before+after) | **{gini_all:.4f}** | {'High' if gini_all > 0.5 else 'Moderate' if gini_all > 0.3 else 'Low'} concentration |")
    add(f"| Before position only | **{gini_before:.4f}** | {'High' if gini_before > 0.5 else 'Moderate' if gini_before > 0.3 else 'Low'} concentration |")
    add(f"| After position only | **{gini_after:.4f}** | {'High' if gini_after > 0.5 else 'Moderate' if gini_after > 0.3 else 'Low'} concentration |")
    add()

    # Top-N concentration
    sorted_totals = sorted(all_totals, reverse=True)
    grand_total = sum(all_totals)

    add("### 5b. Top-N Concentration")
    add()
    if grand_total > 0:
        top5_share = sum(sorted_totals[:5]) / grand_total * 100
        top10_share = sum(sorted_totals[:10]) / grand_total * 100
        top20_share = sum(sorted_totals[:20]) / grand_total * 100

        add("| Top-N hexagrams | Share of all occurrences |")
        add("|-----------------|-------------------------|")
        add(f"| Top 5 | {top5_share:.1f}% |")
        add(f"| Top 10 | {top10_share:.1f}% |")
        add(f"| Top 20 | {top20_share:.1f}% |")
        add()

        ideal_per_hex = grand_total / 64
        add(f"- **Ideal uniform count per hexagram**: {ideal_per_hex:.0f}")
        add(f"- **Actual range**: {sorted_totals[-1]} - {sorted_totals[0]}")
        add(f"- **Max/Ideal ratio**: {sorted_totals[0] / ideal_per_hex:.1f}x")
        add()

    # Shannon entropy
    add("### 5c. Shannon Entropy")
    add()
    if grand_total > 0:
        entropy_all = shannon_entropy(all_totals)
        entropy_before = shannon_entropy(before_vals)
        entropy_after = shannon_entropy(after_vals)
        max_entropy = math.log2(64)  # 6.0 bits for uniform over 64

        add("| Metric | Entropy (bits) | Normalized (vs max 6.0) |")
        add("|--------|---------------|------------------------|")
        add(f"| Combined | {entropy_all:.4f} | {entropy_all/max_entropy:.4f} |")
        add(f"| Before only | {entropy_before:.4f} | {entropy_before/max_entropy:.4f} |")
        add(f"| After only | {entropy_after:.4f} | {entropy_after/max_entropy:.4f} |")
        add()
        add(f"> Maximum entropy for uniform distribution over 64 hexagrams = {max_entropy:.4f} bits")
        add(f"> Normalized = 1.0 means perfectly uniform; lower means more concentrated")
        add()

    # ── 6. Transition Graph Statistics ────────────────────────────────
    add("## 6. Transition Graph Statistics")
    add()

    gs = graph_stats
    add("### 6a. Graph Metrics")
    add()
    add("| Metric | Value |")
    add("|--------|-------|")
    add(f"| Total transitions | {gs['total_transitions']:,} |")
    add(f"| Unique edges (before->after pairs) | {gs['unique_edge_count']:,} |")
    add(f"| Active nodes (hexagrams in graph) | {gs['active_nodes']} / 64 |")
    add(f"| Graph density | {gs['graph_density']:.4f} ({gs['graph_density']*100:.2f}%) |")
    add(f"| Mean edge weight | {gs['mean_edge_weight']:.1f} |")
    add(f"| Median edge weight | {gs['median_edge_weight']:.1f} |")
    add(f"| Min edge weight | {gs['min_edge_weight']:,} |")
    add(f"| Max edge weight | {gs['max_edge_weight']:,} |")
    add(f"| Mean degree (in+out) per active node | {gs['mean_degree']:.1f} |")
    add(f"| Max degree | {gs['max_degree']} |")
    add()

    # Top 20 transitions
    top_edges = gs["edges"].most_common(20)
    add("### 6b. Top 20 Transition Edges")
    add()
    add("| Rank | Before (#) | After (#) | Count | % of total |")
    add("|------|-----------|----------|-------|-----------|")
    for rank, ((src, dst), w) in enumerate(top_edges, 1):
        src_name = HEX_NAMES.get(src, "?")
        dst_name = HEX_NAMES.get(dst, "?")
        pct = w / gs["total_transitions"] * 100 if gs["total_transitions"] else 0
        add(f"| {rank} | {src} {src_name} | {dst} {dst_name} | {w:,} | {pct:.1f}% |")
    add()

    # Top 10 hub nodes
    all_nodes_degree = Counter()
    for n in range(1, 65):
        all_nodes_degree[n] = gs["out_degree"].get(n, 0) + gs["in_degree"].get(n, 0)
    top_hubs = all_nodes_degree.most_common(10)

    add("### 6c. Top 10 Hub Nodes (highest degree)")
    add()
    add("| # | Name | Out-degree | In-degree | Total degree |")
    add("|---|------|------------|-----------|-------------|")
    for h, deg in top_hubs:
        od = gs["out_degree"].get(h, 0)
        id_ = gs["in_degree"].get(h, 0)
        add(f"| {h} | {HEX_NAMES.get(h, '?')} | {od} | {id_} | {deg} |")
    add()

    # Isolated nodes
    isolated = [h for h in range(1, 65) if all_nodes_degree[h] == 0]
    if isolated:
        add(f"### 6d. Isolated Nodes (degree 0): {len(isolated)} hexagrams")
        add()
        add("These hexagrams never appear in any before->after transition:")
        add()
        for h in isolated:
            lo, up = REV_KING_WEN.get(h, ("?", "?"))
            add(f"- **{h}. {HEX_NAMES.get(h, '?')}** ({lo}/{up})")
        add()

    # ── 7. Trigram Pair Coverage Matrix ───────────────────────────────
    add("## 7. Trigram Pair Coverage Matrix")
    add()
    add("Number of cases for each (lower x upper) trigram pair.")
    add()

    # Before matrix
    pair_count_before = Counter()
    pair_count_after = Counter()
    for c in cases:
        bl = c.get("before_lower_trigram")
        bu = c.get("before_upper_trigram")
        if bl and bu and bl in TRIGRAMS and bu in TRIGRAMS:
            pair_count_before[(bl, bu)] += 1
        al = c.get("after_lower_trigram")
        au = c.get("after_upper_trigram")
        if al and au and al in TRIGRAMS and au in TRIGRAMS:
            pair_count_after[(al, au)] += 1

    add("### Before position (lower \\ upper)")
    add()
    header = "| lower \\ upper |"
    sep_row = "|---|"
    for tg in TRIGRAMS:
        header += f" {tg} |"
        sep_row += "---|"
    add(header)
    add(sep_row)
    for lower in TRIGRAMS:
        row = f"| **{lower}** |"
        for upper in TRIGRAMS:
            cnt = pair_count_before.get((lower, upper), 0)
            hnum = KING_WEN.get((lower, upper), "?")
            if cnt == 0:
                row += f" **0** (#{hnum}) |"
            else:
                row += f" {cnt:,} (#{hnum}) |"
        add(row)
    add()

    add("### After position (lower \\ upper)")
    add()
    header = "| lower \\ upper |"
    sep_row = "|---|"
    for tg in TRIGRAMS:
        header += f" {tg} |"
        sep_row += "---|"
    add(header)
    add(sep_row)
    for lower in TRIGRAMS:
        row = f"| **{lower}** |"
        for upper in TRIGRAMS:
            cnt = pair_count_after.get((lower, upper), 0)
            hnum = KING_WEN.get((lower, upper), "?")
            if cnt == 0:
                row += f" **0** (#{hnum}) |"
            else:
                row += f" {cnt:,} (#{hnum}) |"
        add(row)
    add()

    # ── 8. Gap Estimation & Recommendations ───────────────────────────
    add("## 8. Gap Estimation & Recommendations")
    add()

    gap_hexagrams = [(h, combined[h]["total"]) for h in range(1, 65) if combined[h]["total"] < MIN_SUPPORT]
    cases_needed_total = sum(max(0, MIN_SUPPORT - t) for _, t in gap_hexagrams)

    add(f"### Current gap: {len(gap_hexagrams)} hexagrams below threshold ({MIN_SUPPORT})")
    add()
    add(f"- Zero occurrences: **{zero}**")
    add(f"- Low (1-{MIN_SUPPORT-1}): **{low}**")
    add(f"- Well-represented (>= {MIN_SUPPORT}): **{ge5}**")
    add()

    if ge5 >= TARGET_COVERAGE:
        add(f"Target of {TARGET_COVERAGE}/64 is **already met** ({ge5}/64).")
        add()
        remaining = [(h, t) for h, t in gap_hexagrams]
        if remaining:
            add(f"To reach full 64/64 coverage, **{cases_needed_total}** additional targeted cases are needed:")
            add()
    else:
        needed_to_fill = TARGET_COVERAGE - ge5
        gap_sorted = sorted(gap_hexagrams, key=lambda x: -x[1])
        cheapest = gap_sorted[:needed_to_fill]
        min_for_target = sum(max(0, MIN_SUPPORT - t) for _, t in cheapest)

        add(f"**To reach {TARGET_COVERAGE}/64 (easiest path):**")
        add(f"- Fill {needed_to_fill} hexagrams closest to threshold")
        add(f"- Minimum cases needed: **{min_for_target}**")
        add()
        add(f"**To reach full 64/64:**")
        add(f"- Total additional cases needed: **{cases_needed_total}**")
        add()

    # Priority table
    priority = []
    for h in range(1, 65):
        t = combined[h]["total"]
        if t == 0:
            priority.append((h, t, "CRITICAL", MIN_SUPPORT - t))
        elif t < MIN_SUPPORT:
            priority.append((h, t, "HIGH", MIN_SUPPORT - t))
        elif t < 20:
            priority.append((h, t, "MEDIUM", 20 - t))

    if priority:
        add(f"### Priority List ({len(priority)} hexagrams need attention)")
        add()
        add("| Priority | # | Name | Lower | Upper | Current | Needed (to {}) |".format(MIN_SUPPORT))
        add("|----------|---|------|-------|-------|---------|----------------|")
        for h, t, label, need in sorted(priority, key=lambda x: x[1]):
            lo, up = REV_KING_WEN.get(h, ("?", "?"))
            add(f"| {label} | {h} | {HEX_NAMES.get(h, '?')} | {lo} | {up} | {t} | +{need} |")
        add()

    add("---")
    add(f"*Report generated by `scripts/analyze_hexagram_coverage.py` on {now}*")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("  Hexagram Coverage Analysis (Step 0-C)")
    print("=" * 60)
    print(f"Reading cases from: {CASES_PATH}")

    cases = load_cases()
    print(f"Loaded {len(cases):,} cases")

    # 1. Trigram distribution
    print("Analyzing trigram distribution...")
    trigram_dist = analyze_trigram_distribution(cases)

    # 2. Hexagram frequency
    print("Analyzing hexagram frequency...")
    before_hex, after_hex, missing_b, missing_a = analyze_hexagram_frequency(cases)

    # 3. Transition graph
    print("Analyzing transition graph...")
    graph_stats = analyze_transition_graph(cases)

    # Generate report
    print("Generating report...")
    report = generate_report(
        cases, trigram_dist, before_hex, after_hex,
        missing_b, missing_a, graph_stats
    )

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved to: {OUTPUT_PATH}")

    # Print quick summary to stdout
    combined = {}
    for h in range(1, 65):
        combined[h] = before_hex.get(h, 0) + after_hex.get(h, 0)
    ge5 = sum(1 for v in combined.values() if v >= 5)
    ge1 = sum(1 for v in combined.values() if v >= 1)
    zero_count = sum(1 for v in combined.values() if v == 0)

    all_totals = list(combined.values())
    g = gini_coefficient(all_totals)
    e = shannon_entropy(all_totals)
    max_e = math.log2(64)

    print()
    print("--- Quick Summary ---")
    print(f"  Active hexagrams (>= 1):  {ge1}/64")
    print(f"  Well-represented (>= 5):  {ge5}/64")
    print(f"  Zero occurrences:         {zero_count}/64")
    print(f"  Unique transition edges:  {graph_stats['unique_edge_count']}")
    print(f"  Graph density:            {graph_stats['graph_density']:.4f} ({graph_stats['graph_density']*100:.2f}%)")
    print(f"  Gini coefficient:         {g:.4f}")
    print(f"  Shannon entropy:          {e:.4f} / {max_e:.4f} (normalized: {e/max_e:.4f})")
    print()


if __name__ == "__main__":
    main()
