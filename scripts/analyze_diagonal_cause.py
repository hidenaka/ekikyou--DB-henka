#!/usr/bin/env python3
"""
対角率42%の原因分析スクリプト

対角の定義:
  Δ_lower = hamming(TRIGRAM_BITS[before_lower], TRIGRAM_BITS[after_lower])
  Δ_upper = hamming(TRIGRAM_BITS[before_upper], TRIGRAM_BITS[after_upper])
  対角 = Δ_lower == Δ_upper

分析内容:
  1. 統計的有意性検定 (二項検定 H0: p=0.309)
  2. state_label条件付き対角率
  3. LLMバイアス検出
  4. reasoning分析
"""

import json
import os
import re
from collections import Counter, defaultdict
from scipy import stats
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRIGRAM_BITS = {
    "乾": "111", "坤": "000", "震": "001", "巽": "110",
    "坎": "010", "離": "101", "艮": "100", "兌": "011"
}
TRIGRAMS = list(TRIGRAM_BITS.keys())


def hamming(a, b):
    """Hamming distance between two bit strings."""
    return sum(x != y for x, y in zip(a, b))


def delta_lower(rec):
    return hamming(TRIGRAM_BITS[rec["before_lower"]], TRIGRAM_BITS[rec["after_lower"]])


def delta_upper(rec):
    return hamming(TRIGRAM_BITS[rec["before_upper"]], TRIGRAM_BITS[rec["after_upper"]])


def is_diagonal(rec):
    return delta_lower(rec) == delta_upper(rec)


def load_data():
    """再アノテーション結果(4バッチ)とメタデータ(100件)を読み込み、結合"""
    batches = []
    for i in range(1, 5):
        path = os.path.join(BASE, f"analysis/phase3/reannotation_batch{i}.json")
        batches.extend(json.load(open(path)))

    meta_path = os.path.join(BASE, "analysis/phase3/reannotation_batch_100.json")
    meta_list = json.load(open(meta_path))
    meta_by_idx = {r["idx"]: r for r in meta_list}

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


# ============================================================
# 1. 統計的有意性検定
# ============================================================
def binomial_test(n_diagonal, n_total, p0=0.309):
    result = stats.binomtest(n_diagonal, n_total, p0, alternative='greater')
    ci = result.proportion_ci(confidence_level=0.95, method='wilson')
    return {
        "observed_rate": round(n_diagonal / n_total, 4),
        "n_diagonal": int(n_diagonal),
        "n_total": int(n_total),
        "p0_random": p0,
        "p_value": float(round(result.pvalue, 8)),
        "ci_95_lower": float(round(ci.low, 4)),
        "ci_95_upper": float(round(ci.high, 4)),
        "significant_at_005": bool(result.pvalue < 0.05),
        "significant_at_001": bool(result.pvalue < 0.01),
    }


# ============================================================
# 1b. ランダム基準の理論的算出
# ============================================================
def compute_random_diagonal_rate():
    """8八卦×8八卦の全ペアでHamming距離分布を算出し、
    ランダムに2つの変化(lower, upper)を独立に選んだときの対角率を計算"""
    # Hamming距離分布: 8×8=64ペアの距離
    dist_counts = Counter()
    for t1 in TRIGRAMS:
        for t2 in TRIGRAMS:
            d = hamming(TRIGRAM_BITS[t1], TRIGRAM_BITS[t2])
            dist_counts[d] += 1

    total_pairs = sum(dist_counts.values())  # 64
    # P(Δ=d) for d in {0,1,2,3}
    probs = {d: dist_counts[d] / total_pairs for d in range(4)}

    # 対角率 = Σ P(Δ_lower=d) × P(Δ_upper=d)
    diagonal_rate = sum(probs[d] ** 2 for d in range(4))

    return {
        "hamming_dist_distribution": {str(d): dist_counts[d] for d in range(4)},
        "hamming_dist_probs": {str(d): round(probs[d], 4) for d in range(4)},
        "theoretical_diagonal_rate": round(diagonal_rate, 4),
    }


# ============================================================
# 2. state_label条件付き対角率
# ============================================================
def grouped_diagonal_rates(data, key):
    groups = defaultdict(list)
    for rec in data:
        groups[rec[key]].append(rec)

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


# ============================================================
# 2c. Hamming距離ペア条件付き対角率
# ============================================================
def delta_pair_analysis(data):
    """(Δ_lower, Δ_upper)のペア分布を詳細に分析"""
    pair_counts = Counter()
    for r in data:
        dl = delta_lower(r)
        du = delta_upper(r)
        pair_counts[(dl, du)] += 1

    results = {}
    for (dl, du), cnt in sorted(pair_counts.items()):
        results[f"({dl},{du})"] = {
            "count": cnt,
            "rate": round(cnt / len(data), 4),
            "is_diagonal": dl == du,
        }

    # 対角ペアだけの内訳
    diag_breakdown = {}
    for d in range(4):
        cnt = pair_counts.get((d, d), 0)
        diag_breakdown[f"({d},{d})"] = cnt

    return {
        "pair_distribution": results,
        "diagonal_breakdown_by_hamming": diag_breakdown,
    }


# ============================================================
# 3. LLMバイアス検出
# ============================================================
def llm_bias_analysis(data):
    n = len(data)

    # 内卦不変・外卦不変の割合
    lower_unchanged = sum(1 for r in data if r["before_lower"] == r["after_lower"])
    upper_unchanged = sum(1 for r in data if r["before_upper"] == r["after_upper"])

    # 対角ペアの内訳(Hamming距離ベース)
    diagonal_cases = [r for r in data if is_diagonal(r)]
    n_diag = len(diagonal_cases)

    # 両方Δ=0 (両方不変)
    both_h0 = sum(1 for r in diagonal_cases
                  if delta_lower(r) == 0 and delta_upper(r) == 0)
    # 両方Δ=1
    both_h1 = sum(1 for r in diagonal_cases
                  if delta_lower(r) == 1 and delta_upper(r) == 1)
    # 両方Δ=2
    both_h2 = sum(1 for r in diagonal_cases
                  if delta_lower(r) == 2 and delta_upper(r) == 2)
    # 両方Δ=3
    both_h3 = sum(1 for r in diagonal_cases
                  if delta_lower(r) == 3 and delta_upper(r) == 3)

    # 八卦の出現頻度
    trigram_counts = Counter()
    for r in data:
        for field in ["before_lower", "before_upper", "after_lower", "after_upper"]:
            trigram_counts[r[field]] += 1
    total_slots = n * 4
    expected = total_slots / 8

    observed = [trigram_counts.get(t, 0) for t in TRIGRAMS]
    chi2, chi2_p = stats.chisquare(observed, [expected] * 8)

    # position別の八卦分布
    position_counts = {}
    for pos in ["before_lower", "before_upper", "after_lower", "after_upper"]:
        pc = Counter()
        for r in data:
            pc[r[pos]] += 1
        position_counts[pos] = {t: pc.get(t, 0) for t in TRIGRAMS}

    # 対角 vs 非対角でのΔ分布比較
    diag_dl = [delta_lower(r) for r in diagonal_cases]
    nondiag = [r for r in data if not is_diagonal(r)]
    nondiag_dl = [delta_lower(r) for r in nondiag]
    nondiag_du = [delta_upper(r) for r in nondiag]

    return {
        "lower_unchanged_n": lower_unchanged,
        "lower_unchanged_rate": round(lower_unchanged / n, 4),
        "upper_unchanged_n": upper_unchanged,
        "upper_unchanged_rate": round(upper_unchanged / n, 4),
        "diagonal_breakdown": {
            "total_diagonal": n_diag,
            "both_delta_0": both_h0,
            "both_delta_1": both_h1,
            "both_delta_2": both_h2,
            "both_delta_3": both_h3,
        },
        "trigram_distribution": {
            "overall_counts": {t: trigram_counts.get(t, 0) for t in TRIGRAMS},
            "expected_uniform": round(expected, 1),
            "chi2": round(float(chi2), 4),
            "chi2_p_value": round(float(chi2_p), 6),
            "significant_bias": bool(chi2_p < 0.05),
        },
        "position_counts": position_counts,
    }


# ============================================================
# 4. reasoning分析
# ============================================================
def reasoning_analysis(data):
    diagonal_cases = [r for r in data if is_diagonal(r)]
    non_diagonal_cases = [r for r in data if not is_diagonal(r)]

    def extract_keywords(text):
        text_clean = re.sub(r'\([^)]*\)', '', text)
        words = set(re.findall(r'[a-zA-Z]{4,}', text_clean.lower()))
        stop = {'with', 'from', 'into', 'that', 'this', 'through', 'after', 'before',
                'inner', 'outer', 'their', 'have', 'been', 'being', 'other', 'more',
                'than', 'also', 'both', 'each', 'which', 'when', 'what', 'were', 'will'}
        return words - stop

    def intra_reasoning_overlap(rec, phase):
        """lower/upper説明間のキーワード重複率"""
        text = rec[f"{phase}_reasoning"]
        parts = text.split(' with ', 1)
        if len(parts) != 2:
            parts = text.split(') ', 1)
        if len(parts) != 2:
            return 0.0
        kw1 = extract_keywords(parts[0])
        kw2 = extract_keywords(parts[1])
        if not kw1 or not kw2:
            return 0.0
        return len(kw1 & kw2) / min(len(kw1), len(kw2))

    def cross_reasoning_overlap(rec):
        """before_reasoning と after_reasoning のキーワード重複率"""
        kw_b = extract_keywords(rec["before_reasoning"])
        kw_a = extract_keywords(rec["after_reasoning"])
        if not kw_b or not kw_a:
            return 0.0
        return len(kw_b & kw_a) / min(len(kw_b), len(kw_a))

    # 対角 vs 非対角 で比較
    diag_intra_before = [intra_reasoning_overlap(r, "before") for r in diagonal_cases]
    diag_intra_after = [intra_reasoning_overlap(r, "after") for r in diagonal_cases]
    nondiag_intra_before = [intra_reasoning_overlap(r, "before") for r in non_diagonal_cases]
    nondiag_intra_after = [intra_reasoning_overlap(r, "after") for r in non_diagonal_cases]

    diag_cross = [cross_reasoning_overlap(r) for r in diagonal_cases]
    nondiag_cross = [cross_reasoning_overlap(r) for r in non_diagonal_cases]

    # Mann-Whitney U: 対角の方が重複率高いか？
    if diag_cross and nondiag_cross:
        u_stat, u_p = stats.mannwhitneyu(diag_cross, nondiag_cross, alternative='greater')
    else:
        u_stat, u_p = 0, 1.0

    # 同一ロジック検出: reasoning内で上下卦説明に共通キーワード3個以上
    same_logic_diag = 0
    same_logic_nondiag = 0
    for r in data:
        found = False
        for phase in ["before", "after"]:
            text = r[f"{phase}_reasoning"]
            parts = text.split(' with ', 1)
            if len(parts) == 2:
                kw1 = extract_keywords(parts[0])
                kw2 = extract_keywords(parts[1])
                if len(kw1 & kw2) >= 3:
                    found = True
                    break
        if found:
            if is_diagonal(r):
                same_logic_diag += 1
            else:
                same_logic_nondiag += 1

    n_diag = len(diagonal_cases)
    n_nondiag = len(non_diagonal_cases)

    # reasoning内でlower/upperに同じ意味的方向性（成長、衰退等）を使っているか
    direction_words = {
        'growth': ['growth', 'growing', 'expansion', 'expanding', 'rise', 'rising', 'success', 'prosperity'],
        'decline': ['decline', 'declining', 'collapse', 'falling', 'crisis', 'danger', 'failure', 'stagnation'],
        'stability': ['stable', 'stability', 'steady', 'maintain', 'foundation', 'grounding'],
        'change': ['change', 'changing', 'transform', 'shift', 'transition', 'renewal', 'disruption'],
    }

    def get_direction(text):
        text_lower = text.lower()
        dirs = set()
        for d, words in direction_words.items():
            if any(w in text_lower for w in words):
                dirs.add(d)
        return dirs

    same_direction_count = 0
    for r in data:
        for phase in ["before", "after"]:
            text = r[f"{phase}_reasoning"]
            parts = text.split(' with ', 1)
            if len(parts) == 2:
                d1 = get_direction(parts[0])
                d2 = get_direction(parts[1])
                if d1 & d2:
                    same_direction_count += 1
                    break

    return {
        "intra_reasoning_overlap": {
            "diagonal_before_mean": round(float(np.mean(diag_intra_before)), 4) if diag_intra_before else 0,
            "diagonal_after_mean": round(float(np.mean(diag_intra_after)), 4) if diag_intra_after else 0,
            "non_diagonal_before_mean": round(float(np.mean(nondiag_intra_before)), 4) if nondiag_intra_before else 0,
            "non_diagonal_after_mean": round(float(np.mean(nondiag_intra_after)), 4) if nondiag_intra_after else 0,
        },
        "cross_reasoning_overlap": {
            "diagonal_mean": round(float(np.mean(diag_cross)), 4) if diag_cross else 0,
            "non_diagonal_mean": round(float(np.mean(nondiag_cross)), 4) if nondiag_cross else 0,
            "mann_whitney_u": round(float(u_stat), 4),
            "mann_whitney_p": round(float(u_p), 6),
            "diagonal_has_higher_overlap": bool(u_p < 0.05),
        },
        "same_logic_keywords_gte3": {
            "diagonal_cases": same_logic_diag,
            "non_diagonal_cases": same_logic_nondiag,
            "diagonal_rate": round(same_logic_diag / n_diag, 4) if n_diag else 0,
            "non_diagonal_rate": round(same_logic_nondiag / n_nondiag, 4) if n_nondiag else 0,
        },
        "same_direction_in_reasoning": {
            "count": same_direction_count,
            "rate": round(same_direction_count / len(data), 4),
        },
    }


# ============================================================
# Main
# ============================================================
def main():
    data = load_data()
    n = len(data)
    print(f"Loaded {n} records")

    n_diag = sum(1 for r in data if is_diagonal(r))
    print(f"\n{'='*60}")
    print(f"対角率: {n_diag}/{n} = {n_diag/n:.1%}")
    print(f"{'='*60}")

    # 1. 統計的有意性検定
    print(f"\n--- 1. 統計的有意性検定 ---")
    random_info = compute_random_diagonal_rate()
    p0 = random_info["theoretical_diagonal_rate"]
    print(f"理論的ランダム対角率: {p0:.4f}")
    print(f"  Hamming距離分布 P(Δ=d): {random_info['hamming_dist_probs']}")

    binom = binomial_test(n_diag, n, p0)
    print(f"観測対角率: {binom['observed_rate']:.4f}")
    print(f"p値 (片側 H1: p > {p0}): {binom['p_value']:.8f}")
    print(f"95% CI: [{binom['ci_95_lower']:.4f}, {binom['ci_95_upper']:.4f}]")
    print(f"有意 (α=0.05): {binom['significant_at_005']}")
    print(f"有意 (α=0.01): {binom['significant_at_001']}")

    # 2. state_label条件付き対角率
    print(f"\n--- 2a. before_state別 対角率 ---")
    bs_rates = grouped_diagonal_rates(data, "before_state")
    for label, info in bs_rates.items():
        print(f"  {label}: {info['n_diagonal']}/{info['n']} = {info['diagonal_rate']:.1%}")

    print(f"\n--- 2b. after_state別 対角率 ---")
    as_rates = grouped_diagonal_rates(data, "after_state")
    for label, info in as_rates.items():
        print(f"  {label}: {info['n_diagonal']}/{info['n']} = {info['diagonal_rate']:.1%}")

    print(f"\n--- 2c. outcome別 対角率 ---")
    oc_rates = grouped_diagonal_rates(data, "outcome")
    for label, info in oc_rates.items():
        print(f"  {label}: {info['n_diagonal']}/{info['n']} = {info['diagonal_rate']:.1%}")

    print(f"\n--- 2d. trigger_type別 対角率 ---")
    tr_rates = grouped_diagonal_rates(data, "trigger_type")
    for label, info in tr_rates.items():
        print(f"  {label}: {info['n_diagonal']}/{info['n']} = {info['diagonal_rate']:.1%}")

    # 2e. Δペア分布
    print(f"\n--- 2e. (Δ_lower, Δ_upper)ペア分布 ---")
    pair_info = delta_pair_analysis(data)
    for pair, info in pair_info["pair_distribution"].items():
        diag_mark = " ← DIAGONAL" if info["is_diagonal"] else ""
        print(f"  {pair}: {info['count']} ({info['rate']:.1%}){diag_mark}")
    print(f"  対角内訳: {pair_info['diagonal_breakdown_by_hamming']}")

    # 3. LLMバイアス検出
    print(f"\n--- 3. LLMバイアス検出 ---")
    bias = llm_bias_analysis(data)
    print(f"内卦不変率: {bias['lower_unchanged_rate']:.1%} ({bias['lower_unchanged_n']}/{n})")
    print(f"外卦不変率: {bias['upper_unchanged_rate']:.1%} ({bias['upper_unchanged_n']}/{n})")
    db = bias['diagonal_breakdown']
    print(f"対角の内訳 (Hamming距離ベース):")
    print(f"  両方Δ=0 (不変): {db['both_delta_0']}")
    print(f"  両方Δ=1: {db['both_delta_1']}")
    print(f"  両方Δ=2: {db['both_delta_2']}")
    print(f"  両方Δ=3: {db['both_delta_3']}")
    td = bias['trigram_distribution']
    print(f"八卦分布 (期待値: {td['expected_uniform']}):")
    for t in TRIGRAMS:
        cnt = td['overall_counts'][t]
        diff = cnt - td['expected_uniform']
        print(f"  {t}: {cnt} ({diff:+.0f})")
    print(f"カイ二乗: χ²={td['chi2']}, p={td['chi2_p_value']:.6f}, 有意偏り: {td['significant_bias']}")

    print(f"\n  position別分布:")
    for pos, counts in bias['position_counts'].items():
        vals = [f"{t}:{counts[t]}" for t in TRIGRAMS]
        print(f"    {pos}: {', '.join(vals)}")

    # 4. reasoning分析
    print(f"\n--- 4. reasoning分析 ---")
    reason = reasoning_analysis(data)
    ir = reason['intra_reasoning_overlap']
    print(f"reasoning内 lower/upper 重複率:")
    print(f"  対角 before: {ir['diagonal_before_mean']:.3f}, after: {ir['diagonal_after_mean']:.3f}")
    print(f"  非対角 before: {ir['non_diagonal_before_mean']:.3f}, after: {ir['non_diagonal_after_mean']:.3f}")
    cr = reason['cross_reasoning_overlap']
    print(f"before↔after reasoning 重複率:")
    print(f"  対角: {cr['diagonal_mean']:.3f}")
    print(f"  非対角: {cr['non_diagonal_mean']:.3f}")
    print(f"  Mann-Whitney U p値: {cr['mann_whitney_p']:.6f}")
    print(f"  対角の方が重複率高い (有意): {cr['diagonal_has_higher_overlap']}")
    sl = reason['same_logic_keywords_gte3']
    print(f"同一ロジック (共有キーワード≥3):")
    print(f"  対角: {sl['diagonal_cases']}/{sum(1 for r in data if is_diagonal(r))} ({sl['diagonal_rate']:.1%})")
    print(f"  非対角: {sl['non_diagonal_cases']}/{sum(1 for r in data if not is_diagonal(r))} ({sl['non_diagonal_rate']:.1%})")
    sd = reason['same_direction_in_reasoning']
    print(f"同一方向性 (成長/衰退等): {sd['count']}/{n} ({sd['rate']:.1%})")

    # 総合判定
    print(f"\n{'='*60}")
    print(f"総合判定")
    print(f"{'='*60}")
    if binom['significant_at_005']:
        print(f"統計的に有意 (p={binom['p_value']:.6f} < 0.05)")
        if cr['diagonal_has_higher_overlap']:
            print(f"→ LLMバイアスの可能性が高い: 対角ケースのreasoning重複率が有意に高い")
        else:
            print(f"→ reasoning重複率に有意差なし: Q6空間の真の同型性シグナルの可能性あり")
    else:
        print(f"統計的に有意でない (p={binom['p_value']:.6f} >= 0.05)")
        print(f"→ ランダム基準との差は偶然の範囲内")

    # 結果JSON保存
    result = {
        "summary": {
            "n_total": n,
            "n_diagonal": n_diag,
            "diagonal_rate": round(n_diag / n, 4),
            "random_baseline": p0,
            "excess_pp": round((n_diag / n - p0) * 100, 1),
        },
        "1_random_baseline": random_info,
        "1_statistical_test": binom,
        "2a_before_state_rates": bs_rates,
        "2b_after_state_rates": as_rates,
        "2c_outcome_rates": oc_rates,
        "2d_trigger_type_rates": tr_rates,
        "2e_delta_pair_distribution": pair_info,
        "3_llm_bias": bias,
        "4_reasoning_analysis": reason,
    }

    out_path = os.path.join(BASE, "analysis/phase3/diagonal_cause_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存: {out_path}")


if __name__ == "__main__":
    main()
