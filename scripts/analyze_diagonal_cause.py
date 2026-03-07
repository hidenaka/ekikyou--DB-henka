#!/usr/bin/env python3
"""
対角率42%の原因分析スクリプト
- 統計的有意性検定
- state_label条件付き対角率
- LLMバイアス検出
- reasoning分析
"""

import json
import os
import re
from collections import Counter, defaultdict
from scipy import stats
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 八卦ビットマッピング ---
TRIGRAM_BITS = {
    "乾": "111", "坤": "000", "震": "001", "巽": "110",
    "坎": "010", "離": "101", "艮": "100", "兌": "011"
}
TRIGRAMS = list(TRIGRAM_BITS.keys())


def load_data():
    """再アノテーション結果(4バッチ)とメタデータ(100件)を読み込み、結合"""
    batches = []
    for i in range(1, 5):
        path = os.path.join(BASE, f"analysis/phase3/reannotation_batch{i}.json")
        batches.extend(json.load(open(path)))

    meta_path = os.path.join(BASE, "analysis/phase3/reannotation_batch_100.json")
    meta_list = json.load(open(meta_path))
    meta_by_idx = {r["idx"]: r for r in meta_list}

    # 結合
    for rec in batches:
        m = meta_by_idx.get(rec["idx"], {})
        rec.update({
            "before_state": m.get("before_state", ""),
            "after_state": m.get("after_state", ""),
            "trigger_type": m.get("trigger_type", ""),
            "action_type": m.get("action_type", ""),
            "outcome": m.get("outcome", ""),
            "story_summary": m.get("story_summary", ""),
            "original_before": m.get("original_before", ""),
            "original_after": m.get("original_after", ""),
            "target_name": m.get("target_name", ""),
        })

    return batches


def compute_delta(rec, phase):
    """phase='before' or 'after' の内卦・外卦を返す"""
    lower = rec[f"{phase}_lower"]
    upper = rec[f"{phase}_upper"]
    return lower, upper


def is_diagonal(rec):
    """Δ_lower == Δ_upper かどうか（対角判定）
    Δは変化の方向: before→afterの八卦名の変化。同一八卦なら変化なし。
    対角 = 内卦の変化パターンと外卦の変化パターンが同じ
    """
    bl, bu = rec["before_lower"], rec["before_upper"]
    al, au = rec["after_lower"], rec["after_upper"]
    # Δ_lower = (before_lower → after_lower), Δ_upper = (before_upper → after_upper)
    # 対角 = 両方の変化が同一
    delta_lower = (bl, al)
    delta_upper = (bu, au)
    return delta_lower == delta_upper


def binomial_test(n_diagonal, n_total, p0=0.309):
    """H0: p = p0, H1: p > p0 の片側二項検定"""
    result = stats.binomtest(n_diagonal, n_total, p0, alternative='greater')
    # 95%信頼区間
    ci = result.proportion_ci(confidence_level=0.95, method='wilson')
    return {
        "observed_rate": n_diagonal / n_total,
        "n_diagonal": n_diagonal,
        "n_total": n_total,
        "p0_random": p0,
        "p_value": result.pvalue,
        "ci_95_lower": ci.low,
        "ci_95_upper": ci.high,
        "significant_at_005": result.pvalue < 0.05,
        "significant_at_001": result.pvalue < 0.01,
    }


def state_label_diagonal_rates(data):
    """before_state別の対角率"""
    groups = defaultdict(list)
    for rec in data:
        groups[rec["before_state"]].append(rec)

    results = {}
    for label, recs in sorted(groups.items(), key=lambda x: -len(x[1])):
        n = len(recs)
        n_diag = sum(1 for r in recs if is_diagonal(r))
        rate = n_diag / n if n > 0 else 0
        results[label] = {
            "n": n,
            "n_diagonal": n_diag,
            "diagonal_rate": round(rate, 4),
        }
    return results


def after_state_diagonal_rates(data):
    """after_state別の対角率"""
    groups = defaultdict(list)
    for rec in data:
        groups[rec["after_state"]].append(rec)

    results = {}
    for label, recs in sorted(groups.items(), key=lambda x: -len(x[1])):
        n = len(recs)
        n_diag = sum(1 for r in recs if is_diagonal(r))
        rate = n_diag / n if n > 0 else 0
        results[label] = {
            "n": n,
            "n_diagonal": n_diag,
            "diagonal_rate": round(rate, 4),
        }
    return results


def llm_bias_analysis(data):
    """LLMバイアスの検出"""
    n = len(data)

    # 内卦不変・外卦不変の割合
    lower_unchanged = sum(1 for r in data if r["before_lower"] == r["after_lower"])
    upper_unchanged = sum(1 for r in data if r["before_upper"] == r["after_upper"])

    # 対角ペアの内訳
    diagonal_cases = [r for r in data if is_diagonal(r)]
    n_diag = len(diagonal_cases)

    # 両方不変 (Δ=0,0)
    both_unchanged = sum(1 for r in diagonal_cases
                         if r["before_lower"] == r["after_lower"]
                         and r["before_upper"] == r["after_upper"])
    # 両方同一変化 (Δ=x,x where x≠0)
    both_same_change = n_diag - both_unchanged

    # 八卦の出現頻度（before/after, lower/upper全て）
    trigram_counts = Counter()
    for r in data:
        for field in ["before_lower", "before_upper", "after_lower", "after_upper"]:
            trigram_counts[r[field]] += 1
    total_trigram_slots = n * 4
    expected_per_trigram = total_trigram_slots / 8

    # 八卦偏りのカイ二乗検定
    observed = [trigram_counts.get(t, 0) for t in TRIGRAMS]
    expected = [expected_per_trigram] * 8
    chi2, chi2_p = stats.chisquare(observed, expected)

    # before側とafter側別の偏り
    before_counts = Counter()
    after_counts = Counter()
    for r in data:
        before_counts[r["before_lower"]] += 1
        before_counts[r["before_upper"]] += 1
        after_counts[r["after_lower"]] += 1
        after_counts[r["after_upper"]] += 1

    return {
        "lower_unchanged_rate": round(lower_unchanged / n, 4),
        "upper_unchanged_rate": round(upper_unchanged / n, 4),
        "lower_unchanged_n": lower_unchanged,
        "upper_unchanged_n": upper_unchanged,
        "diagonal_breakdown": {
            "total_diagonal": n_diag,
            "both_unchanged_00": both_unchanged,
            "both_same_change_xx": both_same_change,
            "both_unchanged_pct_of_diagonal": round(both_unchanged / n_diag, 4) if n_diag > 0 else 0,
        },
        "trigram_distribution": {
            "counts": {t: trigram_counts.get(t, 0) for t in TRIGRAMS},
            "expected_uniform": round(expected_per_trigram, 1),
            "chi2": round(chi2, 4),
            "chi2_p_value": round(chi2_p, 6),
            "significant_bias": chi2_p < 0.05,
        },
        "before_trigram_counts": {t: before_counts.get(t, 0) for t in TRIGRAMS},
        "after_trigram_counts": {t: after_counts.get(t, 0) for t in TRIGRAMS},
    }


def reasoning_analysis(data):
    """reasoning分析: 上下卦が同じロジックで選ばれているケースの検出"""
    # 分析1: before/afterのreasoningで上下卦の説明が同一キーワードを共有するか
    # 分析2: 対角ケースと非対角ケースでreasoningの類似度に差があるか

    diagonal_cases = [r for r in data if is_diagonal(r)]
    non_diagonal_cases = [r for r in data if not is_diagonal(r)]

    def extract_keywords(text):
        """英語キーワード抽出（括弧内の八卦説明を除く）"""
        # 括弧内を除去
        text_clean = re.sub(r'\([^)]*\)', '', text)
        words = set(re.findall(r'[a-zA-Z]{4,}', text_clean.lower()))
        # ストップワード除去
        stop = {'with', 'from', 'into', 'that', 'this', 'through', 'after', 'before',
                'inner', 'outer', 'their', 'have', 'been', 'being', 'other', 'more',
                'than', 'also', 'both', 'each', 'which', 'when', 'what', 'were', 'will'}
        return words - stop

    # 対角ケースのbefore reasoning内で、lower/upperが同一語で説明される率
    def reasoning_overlap_score(rec, phase):
        """1つのreasoningテキスト内の前半(lower)と後半(upper)のキーワード重複率"""
        text = rec[f"{phase}_reasoning"]
        # "with" で前後を分割（inner ... with outer ...）
        parts = text.split(' with ', 1)
        if len(parts) != 2:
            parts = text.split(') ', 1)
        if len(parts) != 2:
            return 0.0
        kw1 = extract_keywords(parts[0])
        kw2 = extract_keywords(parts[1])
        if not kw1 or not kw2:
            return 0.0
        overlap = len(kw1 & kw2)
        return overlap / min(len(kw1), len(kw2))

    # before/after reasoning間のキーワード重複
    def cross_reasoning_overlap(rec):
        """before_reasoningとafter_reasoningのキーワード重複率"""
        kw_b = extract_keywords(rec["before_reasoning"])
        kw_a = extract_keywords(rec["after_reasoning"])
        if not kw_b or not kw_a:
            return 0.0
        return len(kw_b & kw_a) / min(len(kw_b), len(kw_a))

    # 対角 vs 非対角で比較
    diag_before_overlap = [reasoning_overlap_score(r, "before") for r in diagonal_cases]
    diag_after_overlap = [reasoning_overlap_score(r, "after") for r in diagonal_cases]
    nondiag_before_overlap = [reasoning_overlap_score(r, "before") for r in non_diagonal_cases]
    nondiag_after_overlap = [reasoning_overlap_score(r, "after") for r in non_diagonal_cases]

    diag_cross = [cross_reasoning_overlap(r) for r in diagonal_cases]
    nondiag_cross = [cross_reasoning_overlap(r) for r in non_diagonal_cases]

    # 統計検定: 対角ケースの方がreasoning重複率が高いか？
    if diag_cross and nondiag_cross:
        u_stat, u_p = stats.mannwhitneyu(diag_cross, nondiag_cross, alternative='greater')
    else:
        u_stat, u_p = 0, 1.0

    # 特定パターン検出: 同じ形容詞/名詞が上下卦の説明に使われている
    same_logic_count = 0
    for r in data:
        # lower/upperの説明で同一キーワードが3個以上共通
        for phase in ["before", "after"]:
            text = r[f"{phase}_reasoning"]
            parts = text.split(' with ', 1)
            if len(parts) == 2:
                kw1 = extract_keywords(parts[0])
                kw2 = extract_keywords(parts[1])
                if len(kw1 & kw2) >= 3:
                    same_logic_count += 1
                    break

    return {
        "intra_reasoning_overlap": {
            "diagonal_before_mean": round(np.mean(diag_before_overlap), 4) if diag_before_overlap else 0,
            "diagonal_after_mean": round(np.mean(diag_after_overlap), 4) if diag_after_overlap else 0,
            "non_diagonal_before_mean": round(np.mean(nondiag_before_overlap), 4) if nondiag_before_overlap else 0,
            "non_diagonal_after_mean": round(np.mean(nondiag_after_overlap), 4) if nondiag_after_overlap else 0,
        },
        "cross_reasoning_overlap": {
            "diagonal_mean": round(np.mean(diag_cross), 4) if diag_cross else 0,
            "non_diagonal_mean": round(np.mean(nondiag_cross), 4) if nondiag_cross else 0,
            "mann_whitney_u": round(u_stat, 4),
            "mann_whitney_p": round(u_p, 6),
            "diagonal_has_higher_overlap": u_p < 0.05,
        },
        "same_logic_pattern": {
            "cases_with_shared_logic_keywords_gte3": same_logic_count,
            "rate": round(same_logic_count / len(data), 4),
        },
    }


def detailed_diagonal_patterns(data):
    """対角ペアの具体的な変化パターンを集計"""
    diagonal_cases = [r for r in data if is_diagonal(r)]
    pattern_counts = Counter()
    for r in diagonal_cases:
        bl, al = r["before_lower"], r["after_lower"]
        bu, au = r["before_upper"], r["after_upper"]
        if bl == al:
            pattern = f"不変({bl})"
        else:
            pattern = f"{bl}→{al}"
        pattern_counts[pattern] += 1

    return {
        "diagonal_transition_patterns": dict(pattern_counts.most_common()),
        "n_diagonal": len(diagonal_cases),
    }


def main():
    data = load_data()
    print(f"Loaded {len(data)} records")

    # 対角判定
    n_diag = sum(1 for r in data if is_diagonal(r))
    print(f"\n=== 対角率 ===")
    print(f"対角数: {n_diag} / {len(data)} = {n_diag/len(data):.1%}")

    # 1. 統計的有意性検定
    print(f"\n=== 1. 統計的有意性検定 ===")
    binom = binomial_test(n_diag, len(data))
    print(f"観測対角率: {binom['observed_rate']:.3f}")
    print(f"ランダム基準 (p0): {binom['p0_random']}")
    print(f"p値 (片側): {binom['p_value']:.6f}")
    print(f"95% CI: [{binom['ci_95_lower']:.3f}, {binom['ci_95_upper']:.3f}]")
    print(f"有意 (α=0.05): {binom['significant_at_005']}")
    print(f"有意 (α=0.01): {binom['significant_at_001']}")

    # 2. state_label条件付き対角率
    print(f"\n=== 2. before_state別 対角率 ===")
    state_rates = state_label_diagonal_rates(data)
    for label, info in state_rates.items():
        print(f"  {label}: {info['n_diagonal']}/{info['n']} = {info['diagonal_rate']:.1%}")

    print(f"\n=== 2b. after_state別 対角率 ===")
    astate_rates = after_state_diagonal_rates(data)
    for label, info in astate_rates.items():
        print(f"  {label}: {info['n_diagonal']}/{info['n']} = {info['diagonal_rate']:.1%}")

    # 3. LLMバイアス検出
    print(f"\n=== 3. LLMバイアス検出 ===")
    bias = llm_bias_analysis(data)
    print(f"内卦不変率: {bias['lower_unchanged_rate']:.1%} ({bias['lower_unchanged_n']}/{len(data)})")
    print(f"外卦不変率: {bias['upper_unchanged_rate']:.1%} ({bias['upper_unchanged_n']}/{len(data)})")
    db = bias['diagonal_breakdown']
    print(f"対角の内訳:")
    print(f"  両方不変 (0,0): {db['both_unchanged_00']} ({db['both_unchanged_pct_of_diagonal']:.1%} of diagonal)")
    print(f"  両方同一変化 (x,x): {db['both_same_change_xx']}")
    td = bias['trigram_distribution']
    print(f"八卦分布 (期待値: {td['expected_uniform']}):")
    for t in TRIGRAMS:
        print(f"  {t}: {td['counts'][t]}")
    print(f"カイ二乗: χ²={td['chi2']}, p={td['chi2_p_value']:.6f}, 有意偏り: {td['significant_bias']}")

    # 4. reasoning分析
    print(f"\n=== 4. reasoning分析 ===")
    reason = reasoning_analysis(data)
    cr = reason['cross_reasoning_overlap']
    print(f"before↔after reasoning 重複率:")
    print(f"  対角ケース: {cr['diagonal_mean']:.3f}")
    print(f"  非対角ケース: {cr['non_diagonal_mean']:.3f}")
    print(f"  Mann-Whitney U p値: {cr['mann_whitney_p']:.6f}")
    print(f"  対角の方が重複率高い: {cr['diagonal_has_higher_overlap']}")
    sl = reason['same_logic_pattern']
    print(f"同一ロジック（共有キーワード≥3）: {sl['cases_with_shared_logic_keywords_gte3']}/{len(data)} ({sl['rate']:.1%})")

    # 5. 対角パターン詳細
    print(f"\n=== 5. 対角パターン詳細 ===")
    patterns = detailed_diagonal_patterns(data)
    for pat, cnt in patterns['diagonal_transition_patterns'].items():
        print(f"  {pat}: {cnt}")

    # 結果JSONを出力
    result = {
        "summary": {
            "n_total": len(data),
            "n_diagonal": n_diag,
            "diagonal_rate": round(n_diag / len(data), 4),
        },
        "1_statistical_test": binom,
        "2a_before_state_diagonal_rates": state_rates,
        "2b_after_state_diagonal_rates": astate_rates,
        "3_llm_bias": bias,
        "4_reasoning_analysis": reason,
        "5_diagonal_patterns": patterns,
    }

    out_path = os.path.join(BASE, "analysis/phase3/diagonal_cause_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存: {out_path}")


if __name__ == "__main__":
    main()
