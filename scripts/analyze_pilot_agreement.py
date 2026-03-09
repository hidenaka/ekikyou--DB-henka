#!/usr/bin/env python3
"""
パイロット100件の一致率分析

- Cohen's κ（各trigramフィールド別）
- Gwet's AC1
- カテゴリ別一致率
- 混同行列（8x8）
- 95%信頼区間（bootstrap）
- source_type別の一致率
- 不一致パターンの分析
"""

import json
import sys
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ANNOTATIONS_PATH = BASE / "analysis" / "gold_set" / "pilot_annotations.json"
PILOT_PATH = BASE / "analysis" / "gold_set" / "pilot_100.json"
REPORT_JSON_PATH = BASE / "analysis" / "gold_set" / "pilot_agreement_report.json"
REPORT_MD_PATH = BASE / "analysis" / "gold_set" / "pilot_agreement_report.md"

TRIGRAMS = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]
FIELDS = ["before_lower", "before_upper", "after_lower", "after_upper"]

SEED = 42
N_BOOTSTRAP = 2000


def load_data():
    """アノテーションとパイロットデータを読み込み"""
    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        ann_data = json.load(f)

    with open(PILOT_PATH, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)

    # Build source_type lookup
    source_lookup = {}
    for case in pilot_data:
        tid = case.get("transition_id")
        if tid:
            source_lookup[tid] = case.get("source_type", "unknown")

    return ann_data, source_lookup


def extract_pairs(annotations: list, field: str) -> list[tuple[str, str]]:
    """Annotator A, Bのペアを抽出"""
    pairs = []
    for ann in annotations:
        a = ann.get("annotator_a")
        b = ann.get("annotator_b")
        if a and b and field in a and field in b:
            if a[field] in TRIGRAMS and b[field] in TRIGRAMS:
                pairs.append((a[field], b[field]))
    return pairs


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's κを計算"""
    n = len(pairs)
    if n == 0:
        return 0.0

    # Observed agreement
    po = sum(1 for a, b in pairs if a == b) / n

    # Expected agreement
    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)

    pe = sum(a_counts.get(t, 0) * b_counts.get(t, 0) for t in TRIGRAMS) / (n * n)

    if pe == 1.0:
        return 1.0

    return (po - pe) / (1 - pe)


def gwets_ac1(pairs: list[tuple[str, str]]) -> float:
    """Gwet's AC1を計算（偏り耐性）"""
    n = len(pairs)
    if n == 0:
        return 0.0

    # Observed agreement
    po = sum(1 for a, b in pairs if a == b) / n

    # Expected agreement (Gwet's method)
    # pi_k = proportion of items in category k across both raters
    all_labels = [a for a, _ in pairs] + [b for _, b in pairs]
    total = len(all_labels)
    pi = {t: all_labels.count(t) / total for t in TRIGRAMS}

    pe = sum(pi_k * (1 - pi_k) for pi_k in pi.values()) / (len(TRIGRAMS) - 1)

    if pe == 1.0:
        return 1.0

    return (po - pe) / (1 - pe)


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict:
    """8x8混同行列を作成"""
    matrix = {t1: {t2: 0 for t2 in TRIGRAMS} for t1 in TRIGRAMS}
    for a, b in pairs:
        matrix[a][b] += 1
    return matrix


def category_agreement(pairs: list[tuple[str, str]]) -> dict:
    """カテゴリ別一致率"""
    result = {}
    for t in TRIGRAMS:
        a_has = [(a, b) for a, b in pairs if a == t or b == t]
        if len(a_has) == 0:
            result[t] = {"total": 0, "agree": 0, "rate": 0.0}
            continue
        agree = sum(1 for a, b in a_has if a == b)
        result[t] = {"total": len(a_has), "agree": agree, "rate": round(agree / len(a_has), 3)}
    return result


def bootstrap_ci(pairs: list[tuple[str, str]], metric_fn, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap 95%信頼区間"""
    rng = np.random.RandomState(SEED)
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)

    scores = []
    for _ in range(n_boot):
        indices = rng.choice(n, size=n, replace=True)
        boot_pairs = [pairs[i] for i in indices]
        scores.append(metric_fn(boot_pairs))

    lower = float(np.percentile(scores, alpha / 2 * 100))
    upper = float(np.percentile(scores, (1 - alpha / 2) * 100))
    return (round(lower, 4), round(upper, 4))


def disagreement_patterns(pairs: list[tuple[str, str]]) -> list[dict]:
    """不一致パターンを分析（どの卦ペアが混同されやすいか）"""
    disagreements = Counter()
    for a, b in pairs:
        if a != b:
            # Sort to make (坎,艮) and (艮,坎) the same
            pair_key = tuple(sorted([a, b]))
            disagreements[pair_key] += 1

    result = []
    for (t1, t2), count in disagreements.most_common():
        result.append({"trigram_1": t1, "trigram_2": t2, "count": count})
    return result


def source_type_agreement(annotations: list, source_lookup: dict, field: str) -> dict:
    """source_type別の一致率"""
    by_source = defaultdict(list)
    for ann in annotations:
        tid = ann.get("transition_id")
        a = ann.get("annotator_a")
        b = ann.get("annotator_b")
        if a and b and field in a and field in b:
            if a[field] in TRIGRAMS and b[field] in TRIGRAMS:
                st = source_lookup.get(tid, "unknown")
                by_source[st].append((a[field], b[field]))

    result = {}
    for st, pairs in sorted(by_source.items()):
        n = len(pairs)
        agree = sum(1 for a, b in pairs if a == b)
        k = cohens_kappa(pairs) if n >= 5 else None
        result[st] = {
            "n": n,
            "agree": agree,
            "raw_agreement": round(agree / max(1, n), 3),
            "kappa": round(k, 4) if k is not None else None,
        }
    return result


def generate_report(report: dict) -> str:
    """Markdownレポートを生成"""
    lines = []
    lines.append("# パイロット100件 アノテーション一致率レポート")
    lines.append("")
    lines.append(f"**生成日**: 2026-03-09")
    lines.append(f"**対象件数**: {report['meta']['total_pairs']}件（両アノテータ完了）")
    lines.append("")

    # Summary
    lines.append("## サマリー")
    lines.append("")
    lines.append("| フィールド | Cohen's κ | 95% CI | Gwet's AC1 | 95% CI | 生一致率 |")
    lines.append("|-----------|----------|--------|-----------|--------|---------|")
    for field in FIELDS:
        fd = report["fields"][field]
        lines.append(
            f"| {field} | {fd['kappa']:.4f} | [{fd['kappa_ci'][0]:.4f}, {fd['kappa_ci'][1]:.4f}] "
            f"| {fd['ac1']:.4f} | [{fd['ac1_ci'][0]:.4f}, {fd['ac1_ci'][1]:.4f}] "
            f"| {fd['raw_agreement']:.1%} |"
        )

    # Judgment
    lines.append("")
    lines.append("## 総合判定")
    lines.append("")
    judgment = report["judgment"]
    lines.append(f"**判定**: {judgment['result']}")
    lines.append("")
    lines.append(f"- κ平均: {judgment['avg_kappa']:.4f} (閾値: ≥0.60)")
    lines.append(f"- AC1平均: {judgment['avg_ac1']:.4f} (閾値: ≥0.65)")
    lines.append(f"- uncertain率A: {report['meta']['uncertain_rate_a']:.1%}")
    lines.append(f"- uncertain率B: {report['meta']['uncertain_rate_b']:.1%}")
    lines.append("")

    if judgment.get("uncertain_warning"):
        lines.append(f"**WARNING**: {judgment['uncertain_warning']}")
        lines.append("")

    # Confusion matrices
    lines.append("## 混同行列")
    lines.append("")
    for field in FIELDS:
        lines.append(f"### {field}")
        lines.append("")
        cm = report["fields"][field]["confusion_matrix"]
        header = "| A \\ B | " + " | ".join(TRIGRAMS) + " |"
        sep = "|-------|" + "|".join(["---:" for _ in TRIGRAMS]) + "|"
        lines.append(header)
        lines.append(sep)
        for t1 in TRIGRAMS:
            vals = [str(cm[t1][t2]) for t2 in TRIGRAMS]
            lines.append(f"| {t1} | " + " | ".join(vals) + " |")
        lines.append("")

    # Disagreement patterns
    lines.append("## 不一致パターン（混同されやすい卦ペア）")
    lines.append("")
    for field in FIELDS:
        lines.append(f"### {field}")
        lines.append("")
        dp = report["fields"][field]["disagreement_patterns"]
        if dp:
            lines.append("| 卦ペア | 不一致回数 |")
            lines.append("|--------|----------|")
            for p in dp[:10]:
                lines.append(f"| {p['trigram_1']}↔{p['trigram_2']} | {p['count']} |")
        else:
            lines.append("不一致なし")
        lines.append("")

    # Category agreement
    lines.append("## カテゴリ別一致率")
    lines.append("")
    for field in FIELDS:
        lines.append(f"### {field}")
        lines.append("")
        lines.append("| 八卦 | 出現数 | 一致数 | 一致率 |")
        lines.append("|------|--------|--------|--------|")
        ca = report["fields"][field]["category_agreement"]
        for t in TRIGRAMS:
            c = ca[t]
            rate = f"{c['rate']:.1%}" if c['total'] > 0 else "N/A"
            lines.append(f"| {t} | {c['total']} | {c['agree']} | {rate} |")
        lines.append("")

    # Source type agreement
    lines.append("## source_type別一致率")
    lines.append("")
    for field in FIELDS:
        lines.append(f"### {field}")
        lines.append("")
        lines.append("| source_type | N | 一致数 | 生一致率 | κ |")
        lines.append("|-------------|---|--------|---------|---|")
        sa = report["fields"][field]["source_type_agreement"]
        for st, vals in sorted(sa.items()):
            k_str = f"{vals['kappa']:.4f}" if vals['kappa'] is not None else "N/A"
            lines.append(f"| {st} | {vals['n']} | {vals['agree']} | {vals['raw_agreement']:.1%} | {k_str} |")
        lines.append("")

    # Pure hexagram rates
    lines.append("## 純卦率")
    lines.append("")
    lines.append("| | Annotator A | Annotator B | 目標 |")
    lines.append("|---|-----------|-----------|------|")
    lines.append(f"| Before純卦率 | {report['purity']['before_a']:.1%} | {report['purity']['before_b']:.1%} | ≤40% |")
    lines.append(f"| After純卦率 | {report['purity']['after_a']:.1%} | {report['purity']['after_b']:.1%} | ≤15% |")
    lines.append("")

    # Unique hexagram count
    lines.append("## ユニーク卦数")
    lines.append("")
    lines.append("| | Annotator A | Annotator B | 目標 |")
    lines.append("|---|-----------|-----------|------|")
    lines.append(f"| Before | {report['diversity']['before_unique_a']}/64 | {report['diversity']['before_unique_b']}/64 | ≥25 |")
    lines.append(f"| After | {report['diversity']['after_unique_a']}/64 | {report['diversity']['after_unique_b']}/64 | ≥25 |")
    lines.append("")

    return "\n".join(lines)


def main():
    ann_data, source_lookup = load_data()
    annotations = ann_data.get("annotations", [])
    summary = ann_data.get("summary", {})

    # Filter to only cases with both annotations
    valid = [a for a in annotations if a.get("annotator_a") and a.get("annotator_b")]
    print(f"分析対象: {len(valid)}件（両アノテータ完了）")

    report = {"meta": {}, "fields": {}, "judgment": {}, "purity": {}, "diversity": {}}
    report["meta"]["total_pairs"] = len(valid)
    report["meta"]["uncertain_rate_a"] = summary.get("uncertain_rate_a", 0)
    report["meta"]["uncertain_rate_b"] = summary.get("uncertain_rate_b", 0)

    # Per-field analysis
    kappas = []
    ac1s = []
    for field in FIELDS:
        pairs = extract_pairs(valid, field)
        n = len(pairs)
        agree = sum(1 for a, b in pairs if a == b)
        raw_agree = agree / max(1, n)

        k = cohens_kappa(pairs)
        ac1 = gwets_ac1(pairs)
        k_ci = bootstrap_ci(pairs, cohens_kappa)
        ac1_ci = bootstrap_ci(pairs, gwets_ac1)
        cm = confusion_matrix(pairs)
        ca = category_agreement(pairs)
        dp = disagreement_patterns(pairs)
        sa = source_type_agreement(valid, source_lookup, field)

        kappas.append(k)
        ac1s.append(ac1)

        report["fields"][field] = {
            "n": n,
            "agree": agree,
            "raw_agreement": round(raw_agree, 4),
            "kappa": round(k, 4),
            "kappa_ci": k_ci,
            "ac1": round(ac1, 4),
            "ac1_ci": ac1_ci,
            "confusion_matrix": cm,
            "category_agreement": ca,
            "disagreement_patterns": dp,
            "source_type_agreement": sa,
        }

        print(f"  {field}: κ={k:.4f} [{k_ci[0]:.4f},{k_ci[1]:.4f}], AC1={ac1:.4f} [{ac1_ci[0]:.4f},{ac1_ci[1]:.4f}], raw={raw_agree:.1%}")

    # Judgment
    avg_k = np.mean(kappas)
    avg_ac1 = np.mean(ac1s)

    if avg_k >= 0.60 and avg_ac1 >= 0.65:
        result = "PASS"
    elif avg_k >= 0.60 or avg_ac1 >= 0.65:
        result = "CONDITIONAL PASS"
    else:
        result = "FAIL"

    report["judgment"] = {
        "result": result,
        "avg_kappa": round(float(avg_k), 4),
        "avg_ac1": round(float(avg_ac1), 4),
    }

    # Check uncertain rate
    ur_a = summary.get("uncertain_rate_a", 0)
    ur_b = summary.get("uncertain_rate_b", 0)
    if ur_a > 0.10 or ur_b > 0.10:
        report["judgment"]["uncertain_warning"] = (
            f"uncertain率がA={ur_a:.1%}, B={ur_b:.1%}で10%を超えています。規約の見直しが必要です。"
        )

    # Purity analysis
    def pure_rate(annotations_list, lower_field, upper_field, annotator_key):
        n_total = 0
        n_pure = 0
        for ann in annotations_list:
            a = ann.get(annotator_key)
            if a and lower_field in a and upper_field in a:
                n_total += 1
                if a[lower_field] == a[upper_field]:
                    n_pure += 1
        return n_pure / max(1, n_total)

    report["purity"] = {
        "before_a": round(pure_rate(valid, "before_lower", "before_upper", "annotator_a"), 4),
        "before_b": round(pure_rate(valid, "before_lower", "before_upper", "annotator_b"), 4),
        "after_a": round(pure_rate(valid, "after_lower", "after_upper", "annotator_a"), 4),
        "after_b": round(pure_rate(valid, "after_lower", "after_upper", "annotator_b"), 4),
    }

    # Diversity analysis
    def unique_hexagrams(annotations_list, lower_field, upper_field, annotator_key):
        hexagrams = set()
        for ann in annotations_list:
            a = ann.get(annotator_key)
            if a and lower_field in a and upper_field in a:
                hexagrams.add((a[lower_field], a[upper_field]))
        return len(hexagrams)

    report["diversity"] = {
        "before_unique_a": unique_hexagrams(valid, "before_lower", "before_upper", "annotator_a"),
        "before_unique_b": unique_hexagrams(valid, "before_lower", "before_upper", "annotator_b"),
        "after_unique_a": unique_hexagrams(valid, "after_lower", "after_upper", "annotator_a"),
        "after_unique_b": unique_hexagrams(valid, "after_lower", "after_upper", "annotator_b"),
    }

    # Save JSON
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON: {REPORT_JSON_PATH}")

    # Save MD
    md = generate_report(report)
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"MD: {REPORT_MD_PATH}")

    print(f"\n=== 総合判定: {result} ===")
    print(f"  κ平均: {avg_k:.4f}, AC1平均: {avg_ac1:.4f}")


if __name__ == "__main__":
    main()
