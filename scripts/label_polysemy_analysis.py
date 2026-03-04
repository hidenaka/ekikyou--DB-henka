#!/usr/bin/env python3
"""
ラベル多義性問題の実態調査スクリプト
- action_type × scale クロス集計
- before_state × scale クロス集計
- after_state × scale クロス集計
- outcome × scale クロス集計
- Jensen-Shannon divergence によるラベル多義性スコア算出
- 同一ラベルの具体的事例比較用サンプリング
"""

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# パス設定
BASE_DIR = Path("/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB")
CASES_PATH = BASE_DIR / "data" / "raw" / "cases.jsonl"
OUTPUT_DIR = BASE_DIR / "analysis" / "phase3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_SCALES = ["company", "individual", "family", "country", "other"]

def load_cases():
    """cases.jsonlを読み込む"""
    cases = []
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases

def cross_tabulate(cases, label_field, scale_field="scale"):
    """label_field × scale のクロス集計"""
    cross = defaultdict(lambda: Counter())
    for case in cases:
        label = case.get(label_field, "MISSING")
        scale = case.get(scale_field, "other")
        if scale not in VALID_SCALES:
            scale = "other"
        cross[label][scale] += 1
    return cross

def compute_distribution(cross_tab, label):
    """特定ラベルのscale別分布(確率)を計算"""
    counts = cross_tab[label]
    total = sum(counts.values())
    if total == 0:
        return {s: 0.0 for s in VALID_SCALES}
    return {s: counts.get(s, 0) / total for s in VALID_SCALES}

def kl_divergence(p, q, epsilon=1e-10):
    """KL divergence D(p||q)"""
    result = 0.0
    for s in VALID_SCALES:
        p_val = max(p.get(s, 0), epsilon)
        q_val = max(q.get(s, 0), epsilon)
        result += p_val * math.log2(p_val / q_val)
    return result

def jensen_shannon_divergence(p, q):
    """Jensen-Shannon divergence"""
    m = {}
    for s in VALID_SCALES:
        m[s] = 0.5 * p.get(s, 0) + 0.5 * q.get(s, 0)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def compute_polysemy_score(cross_tab, label):
    """
    ラベルの多義性スコア: 全scaleペア間のJSD平均
    高い = scale間で分布が大きく異なる = 多義性が高い
    """
    dist = compute_distribution(cross_tab, label)

    # 全体の分布（マージナル）
    total_counts = Counter()
    for l in cross_tab:
        for s in VALID_SCALES:
            total_counts[s] += cross_tab[l].get(s, 0)
    grand_total = sum(total_counts.values())
    global_dist = {s: total_counts[s] / grand_total for s in VALID_SCALES}

    # ラベル固有分布 vs 全体分布のJSD
    jsd_vs_global = jensen_shannon_divergence(dist, global_dist)

    # 各scaleペア間のJSD
    scale_dists = {}
    for s in VALID_SCALES:
        # このラベルが出現する他のラベルとのscale内分布
        scale_cases_total = sum(cross_tab[l].get(s, 0) for l in cross_tab)
        if scale_cases_total > 0 and cross_tab[label].get(s, 0) > 0:
            scale_dists[s] = cross_tab[label][s] / scale_cases_total
        else:
            scale_dists[s] = 0.0

    # ペアワイズJSD (scale間の条件付き分布)
    pair_jsds = []
    active_scales = [s for s in VALID_SCALES if cross_tab[label].get(s, 0) >= 5]
    for i, s1 in enumerate(active_scales):
        for s2 in active_scales[i+1:]:
            # このscaleにおけるラベルの相対的重要度は異なるか？
            # → 各scale内でのこのラベルの出現率を比較
            pass

    return {
        "jsd_vs_global": jsd_vs_global,
        "distribution": dist,
        "total_count": sum(cross_tab[label].values()),
        "scale_counts": dict(cross_tab[label])
    }

def compute_pairwise_jsd_for_label(cases, label_field, label_value):
    """
    特定ラベルのscaleペア間JSDを計算。
    各scaleで、このラベルが付いた事例の他フィールドの分布を比較する。
    """
    # 各scale内でのこのラベルの事例を取得
    scale_cases = defaultdict(list)
    for case in cases:
        if case.get(label_field) == label_value:
            scale = case.get("scale", "other")
            if scale not in VALID_SCALES:
                scale = "other"
            scale_cases[scale].append(case)

    # 各scaleでの事例数
    scale_counts = {s: len(scale_cases[s]) for s in VALID_SCALES}

    return scale_counts

def compute_full_polysemy_scores(cross_tab, label_field_name):
    """全ラベルのJSD多義性スコアを計算"""
    # 全体のscale分布
    total_counts = Counter()
    for label in cross_tab:
        for s in VALID_SCALES:
            total_counts[s] += cross_tab[label].get(s, 0)
    grand_total = sum(total_counts.values())
    global_dist = {s: total_counts[s] / grand_total for s in VALID_SCALES}

    results = {}
    for label in sorted(cross_tab.keys()):
        label_total = sum(cross_tab[label].values())
        if label_total < 5:
            continue

        label_dist = compute_distribution(cross_tab, label)
        jsd = jensen_shannon_divergence(label_dist, global_dist)

        # Pairwise JSD between scales for this label
        # Scale内でこのラベルが占める割合
        scale_share = {}
        for s in VALID_SCALES:
            scale_total = sum(cross_tab[l].get(s, 0) for l in cross_tab)
            if scale_total > 0:
                scale_share[s] = cross_tab[label].get(s, 0) / scale_total
            else:
                scale_share[s] = 0.0

        # Scale間のペアワイズJSD（最も乖離が大きいペアを記録）
        active_scales = [s for s in VALID_SCALES if cross_tab[label].get(s, 0) >= 3]
        max_pair_diff = 0.0
        max_pair = None
        for i, s1 in enumerate(active_scales):
            for s2 in active_scales[i+1:]:
                diff = abs(scale_share[s1] - scale_share[s2])
                if diff > max_pair_diff:
                    max_pair_diff = diff
                    max_pair = (s1, s2)

        results[label] = {
            "jsd_vs_global": round(jsd, 6),
            "distribution": {s: round(v, 4) for s, v in label_dist.items()},
            "scale_share": {s: round(v, 4) for s, v in scale_share.items()},
            "total_count": label_total,
            "scale_counts": {s: cross_tab[label].get(s, 0) for s in VALID_SCALES},
            "max_pair_divergence": round(max_pair_diff, 4) if max_pair else 0.0,
            "max_divergent_pair": list(max_pair) if max_pair else None
        }

    return results

def sample_cases_for_label(cases, label_field, label_value, scale, n=5):
    """特定ラベル×scaleの事例をサンプリング"""
    matching = [c for c in cases if c.get(label_field) == label_value and c.get("scale") == scale]
    if len(matching) <= n:
        return matching
    random.seed(42)  # 再現性のため
    return random.sample(matching, n)

def format_case_summary(case):
    """事例の要約をフォーマット"""
    return {
        "transition_id": case.get("transition_id", ""),
        "target_name": case.get("target_name", ""),
        "scale": case.get("scale", ""),
        "main_domain": case.get("main_domain", ""),
        "story_summary": case.get("story_summary", ""),
        "before_state": case.get("before_state", ""),
        "action_type": case.get("action_type", ""),
        "after_state": case.get("after_state", ""),
        "outcome": case.get("outcome", ""),
        "pattern_type": case.get("pattern_type", "")
    }

def main():
    print("Loading cases...")
    cases = load_cases()
    print(f"Loaded {len(cases)} cases")

    # === 1. action_type × scale クロス集計 ===
    print("\n=== action_type × scale ===")
    action_cross = cross_tabulate(cases, "action_type")
    action_polysemy = compute_full_polysemy_scores(action_cross, "action_type")

    for label in sorted(action_polysemy.keys(), key=lambda x: action_polysemy[x]["jsd_vs_global"], reverse=True):
        info = action_polysemy[label]
        print(f"  {label}: JSD={info['jsd_vs_global']:.4f}, total={info['total_count']}, dist={info['distribution']}")

    # === 2. before_state × scale クロス集計 ===
    print("\n=== before_state × scale ===")
    before_cross = cross_tabulate(cases, "before_state")
    before_polysemy = compute_full_polysemy_scores(before_cross, "before_state")

    for label in sorted(before_polysemy.keys(), key=lambda x: before_polysemy[x]["jsd_vs_global"], reverse=True):
        info = before_polysemy[label]
        print(f"  {label}: JSD={info['jsd_vs_global']:.4f}, total={info['total_count']}, dist={info['distribution']}")

    # === 3. after_state × scale クロス集計 ===
    print("\n=== after_state × scale ===")
    after_cross = cross_tabulate(cases, "after_state")
    after_polysemy = compute_full_polysemy_scores(after_cross, "after_state")

    for label in sorted(after_polysemy.keys(), key=lambda x: after_polysemy[x]["jsd_vs_global"], reverse=True):
        info = after_polysemy[label]
        print(f"  {label}: JSD={info['jsd_vs_global']:.4f}, total={info['total_count']}, dist={info['distribution']}")

    # === 4. outcome × scale クロス集計 ===
    print("\n=== outcome × scale ===")
    outcome_cross = cross_tabulate(cases, "outcome")
    outcome_polysemy = compute_full_polysemy_scores(outcome_cross, "outcome")

    for label in sorted(outcome_polysemy.keys(), key=lambda x: outcome_polysemy[x]["jsd_vs_global"], reverse=True):
        info = outcome_polysemy[label]
        print(f"  {label}: JSD={info['jsd_vs_global']:.4f}, total={info['total_count']}, dist={info['distribution']}")

    # === 5. 同一ラベルの具体的事例比較 ===
    print("\n=== Sampling cases for comparison ===")

    comparison_samples = {}

    # 「対話・融合」の企業 vs 個人
    dialogue_company = sample_cases_for_label(cases, "action_type", "対話・融合", "company", 5)
    dialogue_individual = sample_cases_for_label(cases, "action_type", "対話・融合", "individual", 5)
    comparison_samples["対話・融合"] = {
        "company": [format_case_summary(c) for c in dialogue_company],
        "individual": [format_case_summary(c) for c in dialogue_individual],
        "company_count": len([c for c in cases if c.get("action_type") == "対話・融合" and c.get("scale") == "company"]),
        "individual_count": len([c for c in cases if c.get("action_type") == "対話・融合" and c.get("scale") == "individual"])
    }
    print(f"  対話・融合: company={comparison_samples['対話・融合']['company_count']}, individual={comparison_samples['対話・融合']['individual_count']}")

    # 「攻める・挑戦」の企業 vs 家族
    attack_company = sample_cases_for_label(cases, "action_type", "攻める・挑戦", "company", 5)
    attack_family = sample_cases_for_label(cases, "action_type", "攻める・挑戦", "family", 5)
    comparison_samples["攻める・挑戦"] = {
        "company": [format_case_summary(c) for c in attack_company],
        "family": [format_case_summary(c) for c in attack_family],
        "company_count": len([c for c in cases if c.get("action_type") == "攻める・挑戦" and c.get("scale") == "company"]),
        "family_count": len([c for c in cases if c.get("action_type") == "攻める・挑戦" and c.get("scale") == "family"])
    }
    print(f"  攻める・挑戦: company={comparison_samples['攻める・挑戦']['company_count']}, family={comparison_samples['攻める・挑戦']['family_count']}")

    # === 6. スキーマ外ラベルの検出 ===
    print("\n=== Schema deviation check ===")
    schema_action_types = {"攻める・挑戦", "守る・維持", "捨てる・撤退", "耐える・潜伏", "対話・融合", "刷新・破壊", "逃げる・放置", "分散・スピンオフ"}
    schema_before_states = {"絶頂・慢心", "停滞・閉塞", "混乱・カオス", "成長痛", "どん底・危機", "安定・平和"}
    schema_after_states = {"V字回復・大成功", "縮小安定・生存", "変質・新生", "現状維持・延命", "迷走・混乱", "崩壊・消滅"}
    schema_outcomes = {"Success", "PartialSuccess", "Failure", "Mixed"}

    actual_action_types = set(action_cross.keys())
    actual_before_states = set(before_cross.keys())
    actual_after_states = set(after_cross.keys())
    actual_outcomes = set(outcome_cross.keys())

    extra_action = actual_action_types - schema_action_types
    extra_before = actual_before_states - schema_before_states
    extra_after = actual_after_states - schema_after_states
    extra_outcome = actual_outcomes - schema_outcomes

    print(f"  Extra action_types ({len(extra_action)}): {sorted(extra_action)}")
    print(f"  Extra before_states ({len(extra_before)}): {sorted(extra_before)}")
    print(f"  Extra after_states ({len(extra_after)}): {sorted(extra_after)}")
    print(f"  Extra outcomes ({len(extra_outcome)}): {sorted(extra_outcome)}")

    # スキーマ外ラベルの件数
    extra_label_counts = {}
    for label in extra_action:
        extra_label_counts[f"action_type:{label}"] = sum(action_cross[label].values())
    for label in extra_before:
        extra_label_counts[f"before_state:{label}"] = sum(before_cross[label].values())
    for label in extra_after:
        extra_label_counts[f"after_state:{label}"] = sum(after_cross[label].values())
    for label in extra_outcome:
        extra_label_counts[f"outcome:{label}"] = sum(outcome_cross[label].values())

    # === 7. 総合ラベル多義性ランキング ===
    print("\n=== Polysemy Ranking (all fields, by JSD) ===")
    all_polysemy = {}
    for label, info in action_polysemy.items():
        all_polysemy[f"action_type:{label}"] = info
    for label, info in before_polysemy.items():
        all_polysemy[f"before_state:{label}"] = info
    for label, info in after_polysemy.items():
        all_polysemy[f"after_state:{label}"] = info
    for label, info in outcome_polysemy.items():
        all_polysemy[f"outcome:{label}"] = info

    ranked = sorted(all_polysemy.items(), key=lambda x: x[1]["jsd_vs_global"], reverse=True)
    for i, (key, info) in enumerate(ranked[:20]):
        print(f"  {i+1}. {key}: JSD={info['jsd_vs_global']:.4f} (n={info['total_count']})")

    # === 出力 ===
    stats_output = {
        "metadata": {
            "total_cases": len(cases),
            "scale_distribution": dict(Counter(c.get("scale", "other") for c in cases)),
            "analysis_date": "2026-03-04",
            "description": "ラベル多義性問題の定量分析: action_type, before_state, after_state, outcomeの各ラベルについて、scale間のJensen-Shannon divergenceを算出"
        },
        "action_type_polysemy": action_polysemy,
        "before_state_polysemy": before_polysemy,
        "after_state_polysemy": after_polysemy,
        "outcome_polysemy": outcome_polysemy,
        "polysemy_ranking": [
            {"label": key, "field": key.split(":")[0], "value": key.split(":", 1)[1], **info}
            for key, info in ranked
        ],
        "schema_deviations": {
            "extra_action_types": sorted(extra_action),
            "extra_before_states": sorted(extra_before),
            "extra_after_states": sorted(extra_after),
            "extra_outcomes": sorted(extra_outcome),
            "extra_label_counts": extra_label_counts
        },
        "comparison_samples": comparison_samples
    }

    stats_path = OUTPUT_DIR / "phase3_label_polysemy_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_output, f, ensure_ascii=False, indent=2)
    print(f"\nStats saved to {stats_path}")

    # === Markdownレポート生成 ===
    generate_report(stats_output, action_cross, before_cross, after_cross, outcome_cross, comparison_samples)

    print("Done!")

def generate_report(stats, action_cross, before_cross, after_cross, outcome_cross, comparison_samples):
    """Markdownレポートを生成"""
    lines = []

    lines.append("# Phase 3: ラベル多義性問題 実態調査レポート")
    lines.append("")
    lines.append(f"分析日: 2026-03-04")
    lines.append(f"総事例数: {stats['metadata']['total_cases']:,}")
    lines.append("")

    # Scale分布
    lines.append("## 1. Scale別事例数")
    lines.append("")
    lines.append("| Scale | 件数 | 割合 |")
    lines.append("|-------|------|------|")
    total = stats["metadata"]["total_cases"]
    for s in VALID_SCALES:
        count = stats["metadata"]["scale_distribution"].get(s, 0)
        pct = count / total * 100
        lines.append(f"| {s} | {count:,} | {pct:.1f}% |")
    lines.append("")

    # action_type クロス集計
    lines.append("## 2. action_type × scale クロス集計")
    lines.append("")
    lines.append("### 2.1 件数テーブル")
    lines.append("")
    header = "| action_type | " + " | ".join(VALID_SCALES) + " | total |"
    sep = "|" + "---|" * (len(VALID_SCALES) + 2)
    lines.append(header)
    lines.append(sep)
    for label in sorted(action_cross.keys()):
        counts = action_cross[label]
        row_total = sum(counts.values())
        cells = [str(counts.get(s, 0)) for s in VALID_SCALES]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {row_total} |")
    lines.append("")

    lines.append("### 2.2 Scale内比率テーブル（各scale内でのaction_typeの割合）")
    lines.append("")
    # Scale内での各action_typeの割合
    scale_totals = {s: sum(action_cross[l].get(s, 0) for l in action_cross) for s in VALID_SCALES}
    header = "| action_type | " + " | ".join(VALID_SCALES) + " |"
    sep = "|" + "---|" * (len(VALID_SCALES) + 1)
    lines.append(header)
    lines.append(sep)
    for label in sorted(action_cross.keys()):
        counts = action_cross[label]
        cells = []
        for s in VALID_SCALES:
            st = scale_totals[s]
            if st > 0:
                cells.append(f"{counts.get(s, 0)/st*100:.1f}%")
            else:
                cells.append("0.0%")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### 2.3 action_type 多義性スコア (JSD vs Global)")
    lines.append("")
    lines.append("JSD (Jensen-Shannon Divergence) が高いほど、そのラベルのscale間分布が全体分布から乖離しており、多義性のリスクが高い。")
    lines.append("")
    lines.append("| action_type | JSD | 最大乖離ペア | 乖離度 | 件数 |")
    lines.append("|---|---|---|---|---|")
    for label in sorted(stats["action_type_polysemy"].keys(),
                        key=lambda x: stats["action_type_polysemy"][x]["jsd_vs_global"], reverse=True):
        info = stats["action_type_polysemy"][label]
        pair = info.get("max_divergent_pair")
        pair_str = f"{pair[0]} vs {pair[1]}" if pair else "-"
        lines.append(f"| {label} | {info['jsd_vs_global']:.4f} | {pair_str} | {info['max_pair_divergence']:.4f} | {info['total_count']} |")
    lines.append("")

    # before_state クロス集計
    lines.append("## 3. before_state × scale クロス集計")
    lines.append("")
    lines.append("### 3.1 件数テーブル")
    lines.append("")
    header = "| before_state | " + " | ".join(VALID_SCALES) + " | total |"
    sep = "|" + "---|" * (len(VALID_SCALES) + 2)
    lines.append(header)
    lines.append(sep)
    for label in sorted(before_cross.keys()):
        counts = before_cross[label]
        row_total = sum(counts.values())
        cells = [str(counts.get(s, 0)) for s in VALID_SCALES]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {row_total} |")
    lines.append("")

    lines.append("### 3.2 多義性スコア")
    lines.append("")
    lines.append("| before_state | JSD | 最大乖離ペア | 乖離度 | 件数 |")
    lines.append("|---|---|---|---|---|")
    for label in sorted(stats["before_state_polysemy"].keys(),
                        key=lambda x: stats["before_state_polysemy"][x]["jsd_vs_global"], reverse=True):
        info = stats["before_state_polysemy"][label]
        pair = info.get("max_divergent_pair")
        pair_str = f"{pair[0]} vs {pair[1]}" if pair else "-"
        lines.append(f"| {label} | {info['jsd_vs_global']:.4f} | {pair_str} | {info['max_pair_divergence']:.4f} | {info['total_count']} |")
    lines.append("")

    # after_state クロス集計
    lines.append("## 4. after_state × scale クロス集計")
    lines.append("")
    lines.append("### 4.1 件数テーブル")
    lines.append("")
    header = "| after_state | " + " | ".join(VALID_SCALES) + " | total |"
    sep = "|" + "---|" * (len(VALID_SCALES) + 2)
    lines.append(header)
    lines.append(sep)
    for label in sorted(after_cross.keys()):
        counts = after_cross[label]
        row_total = sum(counts.values())
        cells = [str(counts.get(s, 0)) for s in VALID_SCALES]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {row_total} |")
    lines.append("")

    lines.append("### 4.2 多義性スコア")
    lines.append("")
    lines.append("| after_state | JSD | 最大乖離ペア | 乖離度 | 件数 |")
    lines.append("|---|---|---|---|---|")
    for label in sorted(stats["after_state_polysemy"].keys(),
                        key=lambda x: stats["after_state_polysemy"][x]["jsd_vs_global"], reverse=True):
        info = stats["after_state_polysemy"][label]
        pair = info.get("max_divergent_pair")
        pair_str = f"{pair[0]} vs {pair[1]}" if pair else "-"
        lines.append(f"| {label} | {info['jsd_vs_global']:.4f} | {pair_str} | {info['max_pair_divergence']:.4f} | {info['total_count']} |")
    lines.append("")

    # outcome クロス集計
    lines.append("## 5. outcome × scale クロス集計")
    lines.append("")
    lines.append("### 5.1 件数テーブル")
    lines.append("")
    header = "| outcome | " + " | ".join(VALID_SCALES) + " | total |"
    sep = "|" + "---|" * (len(VALID_SCALES) + 2)
    lines.append(header)
    lines.append(sep)
    for label in sorted(outcome_cross.keys()):
        counts = outcome_cross[label]
        row_total = sum(counts.values())
        cells = [str(counts.get(s, 0)) for s in VALID_SCALES]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {row_total} |")
    lines.append("")

    lines.append("### 5.2 多義性スコア")
    lines.append("")
    lines.append("| outcome | JSD | 最大乖離ペア | 乖離度 | 件数 |")
    lines.append("|---|---|---|---|---|")
    for label in sorted(stats["outcome_polysemy"].keys(),
                        key=lambda x: stats["outcome_polysemy"][x]["jsd_vs_global"], reverse=True):
        info = stats["outcome_polysemy"][label]
        pair = info.get("max_divergent_pair")
        pair_str = f"{pair[0]} vs {pair[1]}" if pair else "-"
        lines.append(f"| {label} | {info['jsd_vs_global']:.4f} | {pair_str} | {info['max_pair_divergence']:.4f} | {info['total_count']} |")
    lines.append("")

    # 総合ランキング
    lines.append("## 6. 多義性スコア 総合ランキング（全フィールド・上位20）")
    lines.append("")
    lines.append("全フィールドのラベルをJSD降順で並べた。JSDが高いラベルほど、scale間で出現分布が偏っており、同一ラベルが異なるscaleで異なる概念を表している可能性が高い。")
    lines.append("")
    lines.append("| Rank | フィールド:ラベル | JSD | 件数 | 最大乖離ペア |")
    lines.append("|---|---|---|---|---|")
    for i, entry in enumerate(stats["polysemy_ranking"][:20]):
        pair = entry.get("max_divergent_pair")
        pair_str = f"{pair[0]} vs {pair[1]}" if pair else "-"
        lines.append(f"| {i+1} | {entry['field']}:{entry['value']} | {entry['jsd_vs_global']:.4f} | {entry['total_count']} | {pair_str} |")
    lines.append("")

    # スキーマ逸脱
    lines.append("## 7. スキーマ外ラベル（データ品質問題）")
    lines.append("")
    lines.append("v3スキーマに定義されていないラベル値。Phase 2Aで既に検出済みだが、多義性問題の観点から再掲。")
    lines.append("")

    devs = stats["schema_deviations"]
    if devs["extra_action_types"]:
        lines.append("### action_type スキーマ外")
        lines.append("")
        for label in devs["extra_action_types"]:
            count = devs["extra_label_counts"].get(f"action_type:{label}", 0)
            lines.append(f"- `{label}` ({count}件)")
        lines.append("")

    if devs["extra_before_states"]:
        lines.append("### before_state スキーマ外")
        lines.append("")
        for label in devs["extra_before_states"]:
            count = devs["extra_label_counts"].get(f"before_state:{label}", 0)
            lines.append(f"- `{label}` ({count}件)")
        lines.append("")

    if devs["extra_after_states"]:
        lines.append("### after_state スキーマ外")
        lines.append("")
        for label in devs["extra_after_states"]:
            count = devs["extra_label_counts"].get(f"after_state:{label}", 0)
            lines.append(f"- `{label}` ({count}件)")
        lines.append("")

    if devs["extra_outcomes"]:
        lines.append("### outcome スキーマ外")
        lines.append("")
        for label in devs["extra_outcomes"]:
            count = devs["extra_label_counts"].get(f"outcome:{label}", 0)
            lines.append(f"- `{label}` ({count}件)")
        lines.append("")

    # 事例比較
    lines.append("## 8. 同一ラベルの具体的事例比較")
    lines.append("")
    lines.append("同一のaction_typeラベルが、異なるscaleでどのような意味を持つかを具体的事例で検証する。")
    lines.append("")

    for label_name, samples in comparison_samples.items():
        lines.append(f"### 8.{list(comparison_samples.keys()).index(label_name)+1} 「{label_name}」")
        lines.append("")

        for scale_name in [k for k in samples.keys() if k.endswith("_count") is False]:
            count_key = f"{scale_name}_count"
            count = samples.get(count_key, "?")
            lines.append(f"#### {scale_name} (全{count}件中5件サンプル)")
            lines.append("")

            for case in samples[scale_name]:
                lines.append(f"**{case['target_name']}** (`{case['transition_id']}`)")
                lines.append(f"- ドメイン: {case['main_domain']}")
                lines.append(f"- 前: {case['before_state']} -> 行動: {case['action_type']} -> 後: {case['after_state']}")
                lines.append(f"- 結果: {case['outcome']} / パターン: {case['pattern_type']}")
                lines.append(f"- 要約: {case['story_summary'][:200]}")
                lines.append("")
        lines.append("")

    # 考察
    lines.append("## 9. 考察と提言")
    lines.append("")
    lines.append("### 9.1 ラベル多義性の実態")
    lines.append("")
    lines.append("JSD分析の結果、以下の3段階に分類できる:")
    lines.append("")
    lines.append("1. **高多義性 (JSD > 0.05)**: scale間で分布が大きく偏っている。同一ラベルが実質的に異なる概念を指している可能性が高い")
    lines.append("2. **中程度の多義性 (0.01 < JSD < 0.05)**: 一定の偏りはあるが、核心的意味は共通")
    lines.append("3. **低多義性 (JSD < 0.01)**: scale間でほぼ均一な分布。ラベルの意味がscaleに依存しない")
    lines.append("")
    lines.append("### 9.2 影響範囲")
    lines.append("")
    lines.append("ラベル多義性は以下の2つの下流処理を汚染する:")
    lines.append("")
    lines.append("1. **類似事例検索**: 同一ラベルの事例が「類似」と判定されるが、scaleが異なれば実質的に非類似")
    lines.append("2. **確率推定**: 条件付き確率 P(outcome|action_type) がscaleによって大きく異なる場合、混合分布は意味をなさない")
    lines.append("")
    lines.append("### 9.3 対策の方向性")
    lines.append("")
    lines.append("1. **階層化**: `scale:action_type` の複合キーで確率テーブルを構築（Phase 1で実施済み）")
    lines.append("2. **ラベル拡張**: 高多義性ラベルにscale固有の修飾子を追加（例: `対話・融合[企業]` = M&A/提携, `対話・融合[個人]` = カウンセリング/対人関係修復）")
    lines.append("3. **埋め込みベース類似度**: ラベルではなくstory_summaryのテキスト埋め込みで類似度を計算し、ラベル依存を緩和")
    lines.append("")

    report_path = OUTPUT_DIR / "phase3_label_polysemy_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
