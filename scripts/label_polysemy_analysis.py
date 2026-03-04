#!/usr/bin/env python3
"""
ラベル多義性問題の定量的調査スクリプト (v2)

action_type / before_state / after_state ラベルが
scale（company/individual/family/country/other）間で
同じ名前でも異なる遷移パターン（before_state分布・after_state分布）を持つ
「多義性問題」を Jensen-Shannon divergence で定量化する。

出力:
  - analysis/phase3/label_polysemy_report.md
  - analysis/phase3/label_polysemy_stats.json
"""

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ─── パス設定 ───
BASE_DIR = Path("/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB")
CASES_PATH = BASE_DIR / "data" / "raw" / "cases.jsonl"
OUTPUT_DIR = BASE_DIR / "analysis" / "phase3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "label_polysemy_report.md"
STATS_PATH = OUTPUT_DIR / "label_polysemy_stats.json"

VALID_SCALES = ["company", "individual", "family", "country", "other"]

# 前回の議論で問題視された重点ラベル
PRIORITY_LABELS = [
    "対話・融合",
    "段階的拡大",
    "全面的変革",
    "忍耐・持久",
    "積極的行動",
]


# ─── ユーティリティ ───

def load_cases():
    """cases.jsonlを読み込む"""
    cases = []
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return cases


def normalize_counter(counter):
    """Counterを確率分布dictに正規化"""
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def js_divergence(p, q):
    """
    Jensen-Shannon divergence between two distributions.
    p, q: {label: probability} dicts.
    Returns value in [0, 1] (log base 2).
    """
    all_keys = set(p.keys()) | set(q.keys())
    if not all_keys:
        return 0.0

    m = {}
    for k in all_keys:
        m[k] = 0.5 * p.get(k, 0.0) + 0.5 * q.get(k, 0.0)

    def kl(a, b):
        val = 0.0
        for k in all_keys:
            a_k = a.get(k, 0.0)
            b_k = b.get(k, 0.0)
            if a_k > 0 and b_k > 0:
                val += a_k * math.log2(a_k / b_k)
        return val

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pairwise_avg_jsd(distributions):
    """
    複数スケールの分布間のペアワイズ平均 JS divergence.
    distributions: {scale: {label: prob}}
    """
    scales = list(distributions.keys())
    if len(scales) < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(len(scales)):
        for j in range(i + 1, len(scales)):
            total += js_divergence(distributions[scales[i]], distributions[scales[j]])
            count += 1
    return total / count if count > 0 else 0.0


def pairwise_jsd_details(distributions):
    """ペアワイズJSD詳細を返す"""
    scales = list(distributions.keys())
    details = {}
    for i in range(len(scales)):
        for j in range(i + 1, len(scales)):
            key = f"{scales[i]}_vs_{scales[j]}"
            details[key] = round(js_divergence(distributions[scales[i]], distributions[scales[j]]), 6)
    return details


# ─── メイン分析 ───

def main():
    random.seed(42)
    print("Loading cases...")
    cases = load_cases()
    print(f"  Loaded {len(cases):,} cases")

    # ─── 1. 基礎データ抽出 ───
    scale_counts = Counter()

    # action_type × scale クロス集計
    action_scale_counts = defaultdict(Counter)  # {action_type: {scale: count}}

    # before_state × scale
    before_scale_counts = defaultdict(Counter)

    # after_state × scale
    after_scale_counts = defaultdict(Counter)

    # action_type ごとの、スケール別 before_state / after_state 分布
    # {action_type: {scale: {"before": Counter, "after": Counter}}}
    action_transitions = defaultdict(lambda: defaultdict(lambda: {"before": Counter(), "after": Counter()}))

    # 遷移ペア(before→after)のスケール別分布
    # {action_type: {scale: Counter((before, after))}}
    action_transition_pairs = defaultdict(lambda: defaultdict(Counter))

    # サンプル事例保存: {action_type: {scale: [case_summary, ...]}}
    action_samples = defaultdict(lambda: defaultdict(list))

    for c in cases:
        scale = c.get("scale", "other")
        if scale not in VALID_SCALES:
            scale = "other"

        action_type = c.get("action_type", "")
        before_state = c.get("before_state", "")
        after_state = c.get("after_state", "")

        scale_counts[scale] += 1

        if action_type:
            action_scale_counts[action_type][scale] += 1

            if before_state:
                action_transitions[action_type][scale]["before"][before_state] += 1
            if after_state:
                action_transitions[action_type][scale]["after"][after_state] += 1
            if before_state and after_state:
                action_transition_pairs[action_type][scale][(before_state, after_state)] += 1

            # サンプル事例(各scale最大10件)
            if len(action_samples[action_type][scale]) < 10:
                action_samples[action_type][scale].append({
                    "transition_id": c.get("transition_id", ""),
                    "target_name": c.get("target_name", ""),
                    "scale": scale,
                    "before_state": before_state,
                    "after_state": after_state,
                    "action_type": action_type,
                    "outcome": c.get("outcome", ""),
                    "main_domain": c.get("main_domain", ""),
                    "story_summary": (c.get("story_summary", "") or "")[:200],
                })

        if before_state:
            before_scale_counts[before_state][scale] += 1
        if after_state:
            after_scale_counts[after_state][scale] += 1

    # ─── 2. 共有ラベル特定 ───
    shared_actions = {}
    for at, sc in action_scale_counts.items():
        n_scales = len(sc)
        if n_scales >= 2:
            shared_actions[at] = {
                "n_scales": n_scales,
                "scale_counts": dict(sc),
                "total": sum(sc.values()),
            }

    print(f"  action_type: {len(action_scale_counts)} unique, {len(shared_actions)} shared across 2+ scales")

    # ─── 3. 各共有action_typeの多義性スコア計算 ───
    # 多義性 = スケール間で before_state分布 / after_state分布 がどれだけ異なるか
    jsd_results = {}

    for action_type in shared_actions:
        # before_state分布 (スケール別)
        before_dists = {}
        for scale in action_transitions[action_type]:
            before_dists[scale] = normalize_counter(action_transitions[action_type][scale]["before"])

        # after_state分布 (スケール別)
        after_dists = {}
        for scale in action_transitions[action_type]:
            after_dists[scale] = normalize_counter(action_transitions[action_type][scale]["after"])

        # 遷移ペア分布 (スケール別)
        pair_dists = {}
        for scale in action_transition_pairs[action_type]:
            pair_counter = action_transition_pairs[action_type][scale]
            total = sum(pair_counter.values())
            if total > 0:
                pair_dists[scale] = {f"{b}->{a}": v / total for (b, a), v in pair_counter.items()}

        jsd_before = pairwise_avg_jsd(before_dists)
        jsd_after = pairwise_avg_jsd(after_dists)
        jsd_pairs = pairwise_avg_jsd(pair_dists)
        jsd_combined = (jsd_before + jsd_after) / 2.0

        # ペアワイズ詳細
        pw_before = pairwise_jsd_details(before_dists)
        pw_after = pairwise_jsd_details(after_dists)
        pw_pairs = pairwise_jsd_details(pair_dists)

        jsd_results[action_type] = {
            "jsd_before_avg": round(jsd_before, 6),
            "jsd_after_avg": round(jsd_after, 6),
            "jsd_pairs_avg": round(jsd_pairs, 6),
            "jsd_combined": round(jsd_combined, 6),
            "n_scales": shared_actions[action_type]["n_scales"],
            "total_cases": shared_actions[action_type]["total"],
            "scale_counts": shared_actions[action_type]["scale_counts"],
            "pairwise_before": pw_before,
            "pairwise_after": pw_after,
            "pairwise_pairs": pw_pairs,
            "before_distributions": {
                s: {k: round(v, 4) for k, v in d.items()}
                for s, d in before_dists.items()
            },
            "after_distributions": {
                s: {k: round(v, 4) for k, v in d.items()}
                for s, d in after_dists.items()
            },
        }

    # ─── 4. ランキング ───
    ranked = sorted(jsd_results.items(), key=lambda x: x[1]["jsd_combined"], reverse=True)
    top10 = ranked[:10]

    # 重点ラベル順位
    priority_ranks = {}
    for i, (label, _) in enumerate(ranked):
        if label in PRIORITY_LABELS:
            priority_ranks[label] = i + 1

    # ─── 5. before_state / after_state のscale間JSD (ラベルそのもの) ───
    # action_typeだけでなく、before_state/after_state自体のscale偏り
    # ここでは「action_type × scale のクロス集計」に相当するスケール分布JSD
    global_scale_dist = normalize_counter(scale_counts)

    before_state_jsd = {}
    for bs, sc in before_scale_counts.items():
        total = sum(sc.values())
        if total < 5:
            continue
        label_dist = {s: sc.get(s, 0) / total for s in VALID_SCALES}
        before_state_jsd[bs] = {
            "jsd_vs_global": round(js_divergence(label_dist, global_scale_dist), 6),
            "total": total,
            "scale_counts": {s: sc.get(s, 0) for s in VALID_SCALES},
            "distribution": {s: round(v, 4) for s, v in label_dist.items()},
        }

    after_state_jsd = {}
    for as_, sc in after_scale_counts.items():
        total = sum(sc.values())
        if total < 5:
            continue
        label_dist = {s: sc.get(s, 0) / total for s in VALID_SCALES}
        after_state_jsd[as_] = {
            "jsd_vs_global": round(js_divergence(label_dist, global_scale_dist), 6),
            "total": total,
            "scale_counts": {s: sc.get(s, 0) for s in VALID_SCALES},
            "distribution": {s: round(v, 4) for s, v in label_dist.items()},
        }

    # action_type自体のscale偏り
    action_type_scale_jsd = {}
    for at, sc in action_scale_counts.items():
        total = sum(sc.values())
        if total < 5:
            continue
        label_dist = {s: sc.get(s, 0) / total for s in VALID_SCALES}
        action_type_scale_jsd[at] = {
            "jsd_vs_global": round(js_divergence(label_dist, global_scale_dist), 6),
            "total": total,
            "scale_counts": {s: sc.get(s, 0) for s in VALID_SCALES},
            "distribution": {s: round(v, 4) for s, v in label_dist.items()},
        }

    # ─── 6. サンプル事例取得 (Top10) ───
    top10_samples = {}
    for action_type, _ in top10:
        top10_samples[action_type] = {}
        for scale, sample_list in action_samples[action_type].items():
            top10_samples[action_type][scale] = sample_list[:3]

    # 重点ラベルのサンプルも追加
    for label in PRIORITY_LABELS:
        if label not in top10_samples and label in action_samples:
            top10_samples[label] = {}
            for scale, sample_list in action_samples[label].items():
                top10_samples[label][scale] = sample_list[:3]

    # ─── 7. 多義性帯域分類 ───
    high_jsd = [(l, d) for l, d in ranked if d["jsd_combined"] > 0.1]
    mid_jsd = [(l, d) for l, d in ranked if 0.05 < d["jsd_combined"] <= 0.1]
    low_jsd = [(l, d) for l, d in ranked if 0.01 < d["jsd_combined"] <= 0.05]
    minimal_jsd = [(l, d) for l, d in ranked if d["jsd_combined"] <= 0.01]

    # ─── 8. JSON出力 ───
    stats_output = {
        "metadata": {
            "total_cases": len(cases),
            "scale_distribution": dict(scale_counts),
            "global_scale_distribution": {s: round(v, 4) for s, v in global_scale_dist.items()},
            "n_unique_action_types": len(action_scale_counts),
            "n_shared_action_types": len(shared_actions),
            "n_unique_before_states": len(before_scale_counts),
            "n_unique_after_states": len(after_scale_counts),
            "analysis_date": "2026-03-04",
            "method": "Jensen-Shannon Divergence on per-scale before_state/after_state distributions for each action_type",
        },
        "action_type_x_scale_crosstab": {
            at: dict(sc) for at, sc in action_scale_counts.items()
        },
        "before_state_x_scale_crosstab": {
            bs: dict(sc) for bs, sc in before_scale_counts.items()
        },
        "after_state_x_scale_crosstab": {
            as_: dict(sc) for as_, sc in after_scale_counts.items()
        },
        "jsd_scores": jsd_results,
        "jsd_ranking": [
            {
                "rank": i + 1,
                "action_type": label,
                "jsd_combined": data["jsd_combined"],
                "jsd_before_avg": data["jsd_before_avg"],
                "jsd_after_avg": data["jsd_after_avg"],
                "jsd_pairs_avg": data["jsd_pairs_avg"],
                "n_scales": data["n_scales"],
                "total_cases": data["total_cases"],
            }
            for i, (label, data) in enumerate(ranked)
        ],
        "priority_label_ranks": priority_ranks,
        "polysemy_bands": {
            "high_jsd_gt_0.1": [l for l, _ in high_jsd],
            "mid_jsd_0.05_to_0.1": [l for l, _ in mid_jsd],
            "low_jsd_0.01_to_0.05": [l for l, _ in low_jsd],
            "minimal_jsd_le_0.01": [l for l, _ in minimal_jsd],
        },
        "action_type_scale_bias": action_type_scale_jsd,
        "before_state_scale_bias": before_state_jsd,
        "after_state_scale_bias": after_state_jsd,
        "top10_samples": top10_samples,
    }

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats_output, f, ensure_ascii=False, indent=2)
    print(f"  Stats written to {STATS_PATH}")

    # ─── 9. マークダウンレポート ───
    lines = []

    lines.append("# ラベル多義性分析レポート")
    lines.append("")
    lines.append("分析日: 2026-03-04  ")
    lines.append(f"データ件数: {len(cases):,}  ")
    lines.append(f"分析手法: Jensen-Shannon Divergence (各action_typeのスケール別 before_state/after_state 遷移分布間)")
    lines.append("")

    # ─── 1. エグゼクティブサマリー ───
    lines.append("## 1. エグゼクティブサマリー")
    lines.append("")
    lines.append(f"- 全事例数: **{len(cases):,}**")
    lines.append(f"- ユニーク action_type 数: **{len(action_scale_counts)}**")
    lines.append(f"- 2スケール以上に出現する共有ラベル: **{len(shared_actions)}**")
    lines.append(f"- 高多義性 (JSD > 0.1): **{len(high_jsd)}** ラベル")
    lines.append(f"- 中多義性 (0.05 < JSD <= 0.1): **{len(mid_jsd)}** ラベル")
    lines.append(f"- 低多義性 (0.01 < JSD <= 0.05): **{len(low_jsd)}** ラベル")
    lines.append(f"- 最小多義性 (JSD <= 0.01): **{len(minimal_jsd)}** ラベル")
    lines.append("")

    # 結論
    lines.append("### 主要知見")
    lines.append("")
    if len(high_jsd) > 0:
        high_labels = ", ".join([f"「{l}」" for l, _ in high_jsd])
        lines.append(f"1. **高多義性ラベル**: {high_labels} — スケール別サブラベル化または文脈メタデータ付与が必要")
    if len(mid_jsd) > 0:
        mid_labels = ", ".join([f"「{l}」" for l, _ in mid_jsd])
        lines.append(f"2. **中多義性ラベル**: {mid_labels} — フィードバック生成時のスケール文脈注入を推奨")
    lines.append(f"3. **低・最小多義性**: {len(low_jsd) + len(minimal_jsd)}ラベル — 現状のスケール分離で対応可能")
    lines.append("")

    # ─── 2. スケール別事例数 ───
    lines.append("## 2. スケール別事例数")
    lines.append("")
    lines.append("| scale | 件数 | 割合 |")
    lines.append("|-------|------|------|")
    for s in VALID_SCALES:
        cnt = scale_counts.get(s, 0)
        pct = cnt / len(cases) * 100
        lines.append(f"| {s} | {cnt:,} | {pct:.1f}% |")
    lines.append("")

    # ─── 3. action_type × scale クロス集計 ───
    lines.append("## 3. action_type x scale クロス集計")
    lines.append("")
    lines.append("| action_type | company | individual | country | family | other | total | shared |")
    lines.append("|------------|---------|-----------|---------|--------|-------|-------|--------|")
    for at in sorted(action_scale_counts.keys(), key=lambda x: sum(action_scale_counts[x].values()), reverse=True):
        sc = action_scale_counts[at]
        total = sum(sc.values())
        n_sc = len(sc)
        shared_mark = "Yes" if n_sc >= 2 else ""
        lines.append(
            f"| {at} | {sc.get('company', 0)} | {sc.get('individual', 0)} "
            f"| {sc.get('country', 0)} | {sc.get('family', 0)} "
            f"| {sc.get('other', 0)} | {total} | {shared_mark} |"
        )
    lines.append("")

    # ─── 4. 多義性スコア Top10 ───
    lines.append("## 4. 多義性スコア Top10")
    lines.append("")
    lines.append("Jensen-Shannon Divergence の combined スコア（各action_typeについて、スケール別の before_state 分布と after_state 分布のペアワイズ平均JSDの平均）。")
    lines.append("")
    lines.append("- JSD = 0: 全スケールで同一の遷移分布（多義性なし）")
    lines.append("- JSD = 1: 全スケールで完全に異なる遷移分布（最大多義性）")
    lines.append("")
    lines.append("| 順位 | action_type | JSD(combined) | JSD(before) | JSD(after) | JSD(pairs) | スケール数 | 合計件数 |")
    lines.append("|------|-----------|--------------|------------|-----------|-----------|----------|---------|")
    for i, (label, data) in enumerate(top10):
        priority_mark = " **[重点]**" if label in PRIORITY_LABELS else ""
        lines.append(
            f"| {i+1} | {label}{priority_mark} | {data['jsd_combined']:.4f} "
            f"| {data['jsd_before_avg']:.4f} | {data['jsd_after_avg']:.4f} "
            f"| {data['jsd_pairs_avg']:.4f} "
            f"| {data['n_scales']} | {data['total_cases']:,} |"
        )
    lines.append("")

    # 重点ラベル順位
    lines.append("### 重点ラベルの順位")
    lines.append("")
    for label in PRIORITY_LABELS:
        if label in priority_ranks:
            rank = priority_ranks[label]
            jsd_val = jsd_results[label]["jsd_combined"]
            total = jsd_results[label]["total_cases"]
            lines.append(f"- **{label}**: 第{rank}位 / {len(ranked)}ラベル中 (JSD = {jsd_val:.4f}, n={total})")
        elif label in jsd_results:
            for i, (l, _) in enumerate(ranked):
                if l == label:
                    lines.append(f"- **{label}**: 第{i+1}位 (JSD = {jsd_results[label]['jsd_combined']:.4f})")
                    break
        else:
            # 1スケールのみ or 存在しない
            if label in action_scale_counts:
                sc = action_scale_counts[label]
                total = sum(sc.values())
                scales = list(sc.keys())
                lines.append(f"- **{label}**: 1スケールのみ ({scales[0]}, n={total}) — 多義性計算不可")
            else:
                lines.append(f"- **{label}**: データなし（cases.jsonlに存在しない）")
    lines.append("")

    # ─── 5. Top10 + 重点ラベル詳細 ───
    lines.append("## 5. ラベル別詳細分析")
    lines.append("")

    # Top10と重点ラベルを統合（重複排除）
    detail_labels = []
    seen = set()
    for label, data in top10:
        detail_labels.append((label, data))
        seen.add(label)
    for label in PRIORITY_LABELS:
        if label not in seen and label in jsd_results:
            detail_labels.append((label, jsd_results[label]))
            seen.add(label)

    for idx, (action_type, data) in enumerate(detail_labels):
        is_priority = action_type in PRIORITY_LABELS
        is_top10 = idx < len(top10)
        tags = []
        if is_top10:
            tags.append(f"Top{idx+1}")
        if is_priority:
            tags.append("重点")
        tag_str = f" [{'/'.join(tags)}]" if tags else ""

        lines.append(f"### 5.{idx+1}. {action_type}{tag_str} (JSD = {data['jsd_combined']:.4f})")
        lines.append("")

        # スケール別件数
        lines.append("**スケール別件数:**")
        lines.append("")
        for scale in VALID_SCALES:
            cnt = data["scale_counts"].get(scale, 0)
            if cnt > 0:
                lines.append(f"- {scale}: {cnt}件")
        lines.append("")

        # ペアワイズJSD (before)
        pw_before = data.get("pairwise_before", {})
        pw_after = data.get("pairwise_after", {})
        if pw_before or pw_after:
            lines.append("**ペアワイズ JSD:**")
            lines.append("")
            lines.append("| ペア | JSD(before_state) | JSD(after_state) |")
            lines.append("|------|------------------|-----------------|")
            all_pairs = set(list(pw_before.keys()) + list(pw_after.keys()))
            for pair in sorted(all_pairs, key=lambda x: pw_before.get(x, 0) + pw_after.get(x, 0), reverse=True):
                b_val = pw_before.get(pair, 0)
                a_val = pw_after.get(pair, 0)
                lines.append(f"| {pair} | {b_val:.4f} | {a_val:.4f} |")
            lines.append("")

        # before_state分布 (スケール別Top3)
        before_dists = data.get("before_distributions", {})
        if before_dists:
            lines.append("**before_state分布 (スケール別 Top3):**")
            lines.append("")
            for scale in VALID_SCALES:
                if scale in before_dists:
                    dist = before_dists[scale]
                    top3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
                    top3_str = ", ".join([f"{k}({v:.1%})" for k, v in top3])
                    lines.append(f"- {scale}: {top3_str}")
            lines.append("")

        # after_state分布 (スケール別Top3)
        after_dists = data.get("after_distributions", {})
        if after_dists:
            lines.append("**after_state分布 (スケール別 Top3):**")
            lines.append("")
            for scale in VALID_SCALES:
                if scale in after_dists:
                    dist = after_dists[scale]
                    top3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
                    top3_str = ", ".join([f"{k}({v:.1%})" for k, v in top3])
                    lines.append(f"- {scale}: {top3_str}")
            lines.append("")

        # サンプル事例
        if action_type in top10_samples:
            lines.append("**スケール別サンプル事例:**")
            lines.append("")
            for scale in VALID_SCALES:
                if scale in top10_samples[action_type]:
                    samples = top10_samples[action_type][scale]
                    lines.append(f"*{scale} (3件サンプル):*")
                    lines.append("")
                    for s in samples:
                        summary = s.get("story_summary", "")[:150]
                        lines.append(
                            f"- `{s['transition_id']}` **{s['target_name']}**: "
                            f"{s['before_state']} -> {s['after_state']} ({s.get('outcome', '')})"
                        )
                        if summary:
                            lines.append(f"  > {summary}...")
                    lines.append("")
        lines.append("---")
        lines.append("")

    # ─── 6. before_state / after_state のスケール偏り ───
    lines.append("## 6. before_state / after_state のスケール偏り")
    lines.append("")
    lines.append("action_typeの遷移多義性とは別に、before_state / after_state ラベル自体がスケール間で出現頻度に偏りがあるかを分析。")
    lines.append("JSD vs Global = このラベルのscale分布が、全体のscale分布からどれだけ乖離しているか。")
    lines.append("")

    lines.append("### before_state スケール偏り (JSD降順)")
    lines.append("")
    lines.append("| before_state | JSD vs Global | total | company | individual | country | family | other |")
    lines.append("|-------------|--------------|-------|---------|-----------|---------|--------|-------|")
    for bs in sorted(before_state_jsd.keys(), key=lambda x: before_state_jsd[x]["jsd_vs_global"], reverse=True):
        info = before_state_jsd[bs]
        sc = info["scale_counts"]
        lines.append(
            f"| {bs} | {info['jsd_vs_global']:.4f} | {info['total']} "
            f"| {sc.get('company', 0)} | {sc.get('individual', 0)} "
            f"| {sc.get('country', 0)} | {sc.get('family', 0)} "
            f"| {sc.get('other', 0)} |"
        )
    lines.append("")

    lines.append("### after_state スケール偏り (JSD降順)")
    lines.append("")
    lines.append("| after_state | JSD vs Global | total | company | individual | country | family | other |")
    lines.append("|------------|--------------|-------|---------|-----------|---------|--------|-------|")
    for as_ in sorted(after_state_jsd.keys(), key=lambda x: after_state_jsd[x]["jsd_vs_global"], reverse=True):
        info = after_state_jsd[as_]
        sc = info["scale_counts"]
        lines.append(
            f"| {as_} | {info['jsd_vs_global']:.4f} | {info['total']} "
            f"| {sc.get('company', 0)} | {sc.get('individual', 0)} "
            f"| {sc.get('country', 0)} | {sc.get('family', 0)} "
            f"| {sc.get('other', 0)} |"
        )
    lines.append("")

    # ─── 7. JSD分布 ───
    lines.append("## 7. JSD分布ヒストグラム")
    lines.append("")
    lines.append("全共有action_typeラベルのJSD(combined)分布:")
    lines.append("")
    lines.append("```")
    bins = [
        (0, 0.005, "0.000-0.005"),
        (0.005, 0.01, "0.005-0.010"),
        (0.01, 0.02, "0.010-0.020"),
        (0.02, 0.05, "0.020-0.050"),
        (0.05, 0.1, "0.050-0.100"),
        (0.1, 0.2, "0.100-0.200"),
        (0.2, 0.5, "0.200-0.500"),
        (0.5, 1.01, "0.500-1.000"),
    ]
    for low, high, label in bins:
        cnt = sum(1 for _, d in jsd_results.items() if low <= d["jsd_combined"] < high)
        bar = "#" * cnt
        lines.append(f"  {label}: {cnt:3d} {bar}")
    lines.append("```")
    lines.append("")

    # ─── 8. 全ラベルJSD一覧 ───
    lines.append("## 8. 全action_typeラベル JSD一覧")
    lines.append("")
    lines.append("| 順位 | action_type | JSD(combined) | JSD(before) | JSD(after) | JSD(pairs) | scales | n |")
    lines.append("|------|-----------|--------------|------------|-----------|-----------|--------|---|")
    for i, (label, data) in enumerate(ranked):
        priority_mark = " *" if label in PRIORITY_LABELS else ""
        lines.append(
            f"| {i+1} | {label}{priority_mark} | {data['jsd_combined']:.4f} "
            f"| {data['jsd_before_avg']:.4f} | {data['jsd_after_avg']:.4f} "
            f"| {data['jsd_pairs_avg']:.4f} "
            f"| {data['n_scales']} | {data['total_cases']:,} |"
        )
    lines.append("")
    lines.append("*印: 前回の議論で問題視された重点ラベル")
    lines.append("")

    # ─── 9. 推奨アクション ───
    lines.append("## 9. 推奨アクション")
    lines.append("")

    lines.append("### 9.1 即時対応（高多義性ラベル: JSD > 0.1）")
    lines.append("")
    if high_jsd:
        for label, data in high_jsd:
            lines.append(f"- **{label}** (JSD={data['jsd_combined']:.4f}):")
            # 最も乖離しているスケールペアを特定
            pw = data.get("pairwise_before", {})
            if pw:
                max_pair = max(pw.items(), key=lambda x: x[1])
                lines.append(f"  - 最大乖離: {max_pair[0]} (before JSD={max_pair[1]:.4f})")
            lines.append(f"  - 対策: スケール修飾子付きサブラベル化 or フィードバック生成時の文脈注入")
    else:
        lines.append("- 該当なし")
    lines.append("")

    lines.append("### 9.2 中期対応（中多義性ラベル: 0.05 < JSD <= 0.1）")
    lines.append("")
    if mid_jsd:
        for label, data in mid_jsd:
            lines.append(f"- **{label}** (JSD={data['jsd_combined']:.4f}, n={data['total_cases']})")
    else:
        lines.append("- 該当なし")
    lines.append("")

    lines.append("### 9.3 既存対策で十分（低多義性）")
    lines.append("")
    lines.append(f"- {len(low_jsd) + len(minimal_jsd)}ラベル: 既に実施済みのスケール別確率テーブル + ベイズ平滑化で対応可能")
    lines.append("")

    lines.append("### 9.4 横断的推奨事項")
    lines.append("")
    lines.append("1. **BacktraceEngine**: `reverse_state()` / `reverse_action()` でスケール別インデックスを使用済みだが、")
    lines.append("   高多義性ラベルについてはクロススケール参照時に多義性警告メタデータを付与")
    lines.append("")
    lines.append("2. **フィードバック生成**: `feedback_engine.py` のテンプレートにスケール別解釈分岐を追加")
    lines.append("   - 例: `対話・融合[company]` → 「M&A・提携・顧客対話」")
    lines.append("   - 例: `対話・融合[individual]` → 「対人関係修復・カウンセリング」")
    lines.append("   - 例: `対話・融合[country]` → 「外交交渉・国際協調」")
    lines.append("")
    lines.append("3. **類似事例検索**: `case_search.py` のスケールフィルタ済みだが、")
    lines.append("   高多義性ラベルでのクロススケール検索結果に「多義性注意」フラグを追加")
    lines.append("")
    lines.append("4. **before_state/after_state の偏り**: 一部の状態ラベル（特にスキーマ外ラベル）が")
    lines.append("   特定スケールに集中している。ラベル統合時にスケールコンテキストを保持する設計が必要")
    lines.append("")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report written to {REPORT_PATH}")

    # ─── コンソールサマリー ───
    print("\n=== 分析結果サマリー ===")
    print(f"全事例数: {len(cases):,}")
    print(f"scale分布: {dict(scale_counts)}")
    print(f"共有action_type数: {len(shared_actions)}")

    print(f"\nTop10 多義性ラベル (遷移パターンJSD):")
    for i, (label, data) in enumerate(top10):
        mark = " [重点]" if label in PRIORITY_LABELS else ""
        print(f"  {i+1}. {label}{mark}: JSD={data['jsd_combined']:.4f} (before={data['jsd_before_avg']:.4f}, after={data['jsd_after_avg']:.4f}) [{data['n_scales']}scales, {data['total_cases']}cases]")

    print(f"\n重点ラベル順位:")
    for label in PRIORITY_LABELS:
        if label in priority_ranks:
            print(f"  {label}: 第{priority_ranks[label]}位/{len(ranked)} (JSD={jsd_results[label]['jsd_combined']:.4f})")
        elif label in action_scale_counts:
            sc = action_scale_counts[label]
            print(f"  {label}: 1スケールのみ ({list(sc.keys())}, n={sum(sc.values())})")
        else:
            print(f"  {label}: データなし")

    print(f"\n多義性帯域:")
    print(f"  高 (JSD>0.1): {len(high_jsd)}ラベル {[l for l,_ in high_jsd]}")
    print(f"  中 (0.05<JSD<=0.1): {len(mid_jsd)}ラベル {[l for l,_ in mid_jsd]}")
    print(f"  低 (0.01<JSD<=0.05): {len(low_jsd)}ラベル")
    print(f"  最小 (JSD<=0.01): {len(minimal_jsd)}ラベル")


if __name__ == "__main__":
    main()
