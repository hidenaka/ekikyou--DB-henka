#!/usr/bin/env python3
"""
ソース偏重分析スクリプト
source_type別の分布差をχ²検定で統計的に比較する。
GPT-5.2批評: 「newsが10,060/11,336を占める構成は構造を歪める」の定量検証。
"""

import json
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# scipy は必須
try:
    from scipy import stats as scipy_stats
except ImportError:
    print("ERROR: scipy が必要です。 pip install scipy")
    sys.exit(1)

# ─── 設定 ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CASES_PATH = BASE_DIR / "data" / "raw" / "cases.jsonl"
OUTPUT_PATH = BASE_DIR / "analysis" / "quality" / "source_bias_report.json"

# 分析対象フィールド
ANALYSIS_FIELDS = [
    ("before_state", "state_label_before"),
    ("after_state", "state_label_after"),
    ("outcome", "outcome_type"),
    ("pattern_type", "pattern_type"),
    ("country", "country"),
]

# trigram フィールド（八卦分布）
TRIGRAM_FIELDS = [
    ("before_lower_trigram", "before_lower_trigram"),
    ("before_upper_trigram", "before_upper_trigram"),
    ("after_lower_trigram", "after_lower_trigram"),
    ("after_upper_trigram", "after_upper_trigram"),
]

TRIGRAM_ORDER = ["乾", "兌", "離", "震", "巽", "坎", "艮", "坤"]


def load_cases():
    """cases.jsonl を読み込む"""
    cases = []
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_contingency_table(cases, source_field, target_field, source_types):
    """
    source_type × target_field のクロス集計表を構築。
    Returns: (table: np.ndarray, row_labels: list, col_labels: list)
    """
    # まず全カテゴリを収集
    target_values = sorted(set(c.get(target_field) or "unknown" for c in cases))

    # クロス集計
    counts = defaultdict(lambda: defaultdict(int))
    for c in cases:
        st = c.get("source_type") or "unknown"
        tv = c.get(target_field) or "unknown"
        counts[st][tv] += 1

    # numpy配列に変換
    row_labels = source_types
    col_labels = target_values
    table = np.zeros((len(row_labels), len(col_labels)), dtype=int)
    for i, st in enumerate(row_labels):
        for j, tv in enumerate(col_labels):
            table[i, j] = counts[st][tv]

    return table, row_labels, col_labels


def chi2_test(table, row_labels, col_labels):
    """
    χ²検定を実行。
    少数セルがある場合の処理も含む。
    """
    # ゼロ行・ゼロ列を除去
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    valid_rows = row_sums > 0
    valid_cols = col_sums > 0
    filtered_table = table[valid_rows][:, valid_cols]
    filtered_row_labels = [r for r, v in zip(row_labels, valid_rows) if v]
    filtered_col_labels = [c for c, v in zip(col_labels, valid_cols) if v]

    if filtered_table.shape[0] < 2 or filtered_table.shape[1] < 2:
        return {
            "chi2": None,
            "p_value": None,
            "dof": None,
            "cramers_v": None,
            "note": "insufficient_categories",
        }

    chi2, p, dof, expected = scipy_stats.chi2_contingency(filtered_table)

    # Cramer's V
    n = filtered_table.sum()
    min_dim = min(filtered_table.shape[0] - 1, filtered_table.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0

    # 期待度数5未満のセル割合
    low_expected_pct = (expected < 5).sum() / expected.size * 100

    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "cramers_v": float(cramers_v),
        "n": int(n),
        "low_expected_cells_pct": round(low_expected_pct, 1),
        "table_shape": list(filtered_table.shape),
    }


def compute_distribution(cases, field, filter_fn=None):
    """特定フィールドの分布を計算"""
    filtered = cases if filter_fn is None else [c for c in cases if filter_fn(c)]
    counts = Counter(c.get(field) or "unknown" for c in filtered)
    total = sum(counts.values())
    dist = {k: {"count": v, "pct": round(v / total * 100, 2)} for k, v in sorted(counts.items(), key=lambda x: -x[1])}
    return dist, total


def news_vs_non_news_comparison(cases, field):
    """news vs non-news の分布比較"""
    news_dist, news_n = compute_distribution(cases, field, lambda c: c.get("source_type") == "news")
    non_news_dist, non_news_n = compute_distribution(cases, field, lambda c: c.get("source_type") != "news")

    # 全カテゴリ
    all_cats = sorted(set(list(news_dist.keys()) + list(non_news_dist.keys())))

    comparison = {}
    for cat in all_cats:
        n_pct = news_dist.get(cat, {"pct": 0.0})["pct"]
        nn_pct = non_news_dist.get(cat, {"pct": 0.0})["pct"]
        comparison[cat] = {
            "news_pct": n_pct,
            "non_news_pct": nn_pct,
            "delta_pp": round(n_pct - nn_pct, 2),
        }

    return {
        "news_n": news_n,
        "non_news_n": non_news_n,
        "comparison": comparison,
    }


def analyze_hexagram_distribution(cases, source_types):
    """
    八卦(trigram)分布のソースタイプ別分析。
    before/after の upper/lower trigram を使用。
    """
    results = {}
    for field, label in TRIGRAM_FIELDS:
        table, row_labels, col_labels = build_contingency_table(cases, "source_type", field, source_types)
        chi2_result = chi2_test(table, row_labels, col_labels)

        # ソースタイプ別分布
        dist_by_source = {}
        for i, st in enumerate(row_labels):
            row_total = table[i].sum()
            if row_total > 0:
                dist_by_source[st] = {
                    col_labels[j]: {"count": int(table[i, j]), "pct": round(table[i, j] / row_total * 100, 2)}
                    for j in range(len(col_labels))
                    if table[i, j] > 0
                }

        # news vs non-news
        nvnn = news_vs_non_news_comparison(cases, field)

        results[label] = {
            "chi2_test": chi2_result,
            "distribution_by_source": dist_by_source,
            "news_vs_non_news": nvnn,
        }

    return results


def compute_overall_summary(report):
    """全体サマリーを生成"""
    significant_fields = []
    for section in ["categorical_fields", "hexagram_fields"]:
        if section not in report:
            continue
        for field_name, field_data in report[section].items():
            chi2_data = field_data.get("chi2_test", {})
            p = chi2_data.get("p_value")
            v = chi2_data.get("cramers_v")
            if p is not None and p < 0.05:
                significant_fields.append({
                    "field": field_name,
                    "p_value": p,
                    "cramers_v": v,
                    "effect_size": (
                        "large" if v and v >= 0.3 else
                        "medium" if v and v >= 0.1 else
                        "small"
                    ),
                })

    # news dominance
    total = report["source_type_distribution"]["total"]
    news_n = report["source_type_distribution"]["counts"].get("news", {}).get("count", 0)

    return {
        "total_cases": total,
        "news_count": news_n,
        "news_pct": round(news_n / total * 100, 1),
        "non_news_count": total - news_n,
        "non_news_pct": round((total - news_n) / total * 100, 1),
        "fields_tested": len(significant_fields) + (
            len(report.get("categorical_fields", {})) +
            len(report.get("hexagram_fields", {})) -
            len(significant_fields)
        ),
        "fields_significant_p05": len(significant_fields),
        "significant_fields": sorted(significant_fields, key=lambda x: x["cramers_v"] or 0, reverse=True),
        "gpt52_critique_validated": news_n / total > 0.85,
        "interpretation": (
            f"newsが{round(news_n/total*100,1)}%を占める。"
            f"χ²検定で{len(significant_fields)}フィールドが有意(p<0.05)。"
            f"source_typeは分布に統計的に有意な影響を与えている。"
            if significant_fields else
            "有意な偏りは検出されなかった。"
        ),
    }


def main():
    print("=" * 60)
    print("ソース偏重分析 (Source Bias Analysis)")
    print("=" * 60)

    # データ読み込み
    cases = load_cases()
    print(f"\n総事例数: {len(cases)}")

    # source_type分布
    source_counts = Counter(c.get("source_type", "unknown") for c in cases)
    source_types = sorted(source_counts.keys(), key=lambda x: -source_counts[x])
    print(f"source_type分布:")
    for st in source_types:
        print(f"  {st}: {source_counts[st]} ({source_counts[st]/len(cases)*100:.1f}%)")

    report = {
        "metadata": {
            "script": "scripts/analyze_source_bias.py",
            "total_cases": len(cases),
            "analysis_date": "2026-03-09",
            "context": "GPT-5.2批評: newsが構造を歪める可能性の定量検証",
        },
        "source_type_distribution": {
            "total": len(cases),
            "counts": {
                st: {"count": source_counts[st], "pct": round(source_counts[st] / len(cases) * 100, 2)}
                for st in source_types
            },
        },
    }

    # ─── Part 1A: カテゴリカルフィールドのχ²検定 ───
    print("\n--- カテゴリカルフィールド分析 ---")
    categorical_results = {}
    for field, label in ANALYSIS_FIELDS:
        print(f"\n  [{label}] source_type × {field}")
        table, row_labels, col_labels = build_contingency_table(cases, "source_type", field, source_types)
        chi2_result = chi2_test(table, row_labels, col_labels)

        # ソースタイプ別分布
        dist_by_source = {}
        for i, st in enumerate(row_labels):
            row_total = table[i].sum()
            if row_total > 0:
                dist_by_source[st] = {
                    col_labels[j]: {"count": int(table[i, j]), "pct": round(table[i, j] / row_total * 100, 2)}
                    for j in range(len(col_labels))
                    if table[i, j] > 0
                }

        # news vs non-news
        nvnn = news_vs_non_news_comparison(cases, field)

        categorical_results[label] = {
            "chi2_test": chi2_result,
            "distribution_by_source": dist_by_source,
            "news_vs_non_news": nvnn,
        }

        if chi2_result["chi2"] is not None:
            sig = "***" if chi2_result["p_value"] < 0.001 else "**" if chi2_result["p_value"] < 0.01 else "*" if chi2_result["p_value"] < 0.05 else "ns"
            print(f"    χ²={chi2_result['chi2']:.1f}, p={chi2_result['p_value']:.2e}, V={chi2_result['cramers_v']:.4f} {sig}")
            print(f"    期待度数<5のセル: {chi2_result['low_expected_cells_pct']:.1f}%")

            # 最大差のカテゴリを表示
            max_delta = max(nvnn["comparison"].items(), key=lambda x: abs(x[1]["delta_pp"]))
            print(f"    最大差(news-nonews): {max_delta[0]} = {max_delta[1]['delta_pp']:+.1f}pp")

    report["categorical_fields"] = categorical_results

    # ─── Part 1B: 八卦分布分析 ───
    print("\n--- 八卦(trigram)分布分析 ---")
    hexagram_results = analyze_hexagram_distribution(cases, source_types)
    for label, data in hexagram_results.items():
        chi2_r = data["chi2_test"]
        if chi2_r["chi2"] is not None:
            sig = "***" if chi2_r["p_value"] < 0.001 else "**" if chi2_r["p_value"] < 0.01 else "*" if chi2_r["p_value"] < 0.05 else "ns"
            print(f"  [{label}] χ²={chi2_r['chi2']:.1f}, p={chi2_r['p_value']:.2e}, V={chi2_r['cramers_v']:.4f} {sig}")

    report["hexagram_fields"] = hexagram_results

    # ─── サマリー ───
    summary = compute_overall_summary(report)
    report["summary"] = summary

    print("\n" + "=" * 60)
    print("サマリー")
    print("=" * 60)
    print(f"  news: {summary['news_count']} ({summary['news_pct']}%)")
    print(f"  non-news: {summary['non_news_count']} ({summary['non_news_pct']}%)")
    print(f"  有意なフィールド数: {summary['fields_significant_p05']}/{summary['fields_tested']}")
    print(f"  GPT-5.2批評(news>85%): {'確認' if summary['gpt52_critique_validated'] else '否定'}")
    for sf in summary["significant_fields"]:
        print(f"    {sf['field']}: V={sf['cramers_v']:.4f} ({sf['effect_size']})")
    print(f"\n  解釈: {summary['interpretation']}")

    # 保存
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存先: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
