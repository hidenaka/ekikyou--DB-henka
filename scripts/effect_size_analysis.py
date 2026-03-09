#!/usr/bin/env python3
"""
Q6距離1の効果量上限分析
Phase 3 v4.0のデータ（800件遷移）を使い、ハミング距離1遷移の上乗せ効果を数値で確定する。

分析内容:
  1. 距離1遷移の実測割合 vs ランダム期待値
  2. 95%信頼区間（二項分布の正確信頼区間 = Clopper-Pearson）
  3. 効果量上限（信頼区間上端 - ランダム期待値）
  4. Poisson回帰（距離1ダミー変数）
  5. TOST等価性検定（±5%マージン）
"""

import json
import sys
import os
from collections import Counter
from math import comb
import numpy as np
from scipy import stats

# ─────────────────────────────────────────────
# 1. King Wen番号 → 6ビットバイナリ マッピング
# ─────────────────────────────────────────────

# Trigram番号(hexagram_calculator.py準拠) → 3ビットバイナリ
# 下から上へ: 初爻=bit0, 二爻=bit1, 三爻=bit2
# 陽=1, 陰=0
TRIGRAM_TO_BITS = {
    1: 0b111,  # 乾 = 111
    2: 0b011,  # 兌 = 011
    3: 0b101,  # 離 = 101
    4: 0b001,  # 震 = 001
    5: 0b110,  # 巽 = 110
    6: 0b010,  # 坎 = 010
    7: 0b100,  # 艮 = 100
    8: 0b000,  # 坤 = 000
}

# hexagram_calculator.pyのHEXAGRAM_NAMES辞書から
# (上卦trigram番号, 下卦trigram番号) → King Wen番号 のマッピングを構築
TRIGRAM_PAIR_TO_KW = {
    (1, 1): 1,   (1, 2): 10,  (1, 3): 13,  (1, 4): 25,
    (1, 5): 44,  (1, 6): 6,   (1, 7): 33,  (1, 8): 12,
    (2, 1): 43,  (2, 2): 58,  (2, 3): 49,  (2, 4): 17,
    (2, 5): 28,  (2, 6): 47,  (2, 7): 31,  (2, 8): 45,
    (3, 1): 14,  (3, 2): 38,  (3, 3): 30,  (3, 4): 21,
    (3, 5): 50,  (3, 6): 64,  (3, 7): 56,  (3, 8): 35,
    (4, 1): 34,  (4, 2): 54,  (4, 3): 55,  (4, 4): 51,
    (4, 5): 32,  (4, 6): 40,  (4, 7): 62,  (4, 8): 16,
    (5, 1): 9,   (5, 2): 61,  (5, 3): 37,  (5, 4): 42,
    (5, 5): 57,  (5, 6): 59,  (5, 7): 53,  (5, 8): 20,
    (6, 1): 5,   (6, 2): 60,  (6, 3): 63,  (6, 4): 3,
    (6, 5): 48,  (6, 6): 29,  (6, 7): 39,  (6, 8): 8,
    (7, 1): 26,  (7, 2): 41,  (7, 3): 22,  (7, 4): 27,
    (7, 5): 18,  (7, 6): 4,   (7, 7): 52,  (7, 8): 23,
    (8, 1): 11,  (8, 2): 19,  (8, 3): 36,  (8, 4): 24,
    (8, 5): 46,  (8, 6): 7,   (8, 7): 15,  (8, 8): 2,
}

# King Wen番号 → 6ビットバイナリ（下卦=下位3bit, 上卦=上位3bit）
KW_TO_BINARY = {}
for (upper_tri, lower_tri), kw_num in TRIGRAM_PAIR_TO_KW.items():
    lower_bits = TRIGRAM_TO_BITS[lower_tri]
    upper_bits = TRIGRAM_TO_BITS[upper_tri]
    six_bits = (upper_bits << 3) | lower_bits
    KW_TO_BINARY[kw_num] = six_bits


def hamming_distance(a: int, b: int) -> int:
    """2つの6ビット整数のハミング距離"""
    return bin(a ^ b).count('1')


def verify_mapping():
    """マッピングの検証"""
    assert len(KW_TO_BINARY) == 64, f"Expected 64, got {len(KW_TO_BINARY)}"
    assert KW_TO_BINARY[1] == 0b111111, f"卦1(乾為天) should be 111111, got {bin(KW_TO_BINARY[1])}"
    assert KW_TO_BINARY[2] == 0b000000, f"卦2(坤為地) should be 000000, got {bin(KW_TO_BINARY[2])}"
    # 卦29=坎為水: 上坎(010) 下坎(010) = 010_010
    assert KW_TO_BINARY[29] == 0b010010, f"卦29(坎為水) should be 010010, got {bin(KW_TO_BINARY[29])}"
    # 卦30=離為火: 上離(101) 下離(101) = 101_101
    assert KW_TO_BINARY[30] == 0b101101, f"卦30(離為火) should be 101101, got {bin(KW_TO_BINARY[30])}"
    # 全64ビットパターンがユニークであること
    binary_values = list(KW_TO_BINARY.values())
    assert len(set(binary_values)) == 64, "Binary values are not unique"
    print("✓ Mapping verification passed")


def random_expected_d1_proportion():
    """
    ランダム期待値: 64ノード中、任意の2ノード間がハミング距離1である確率
    距離1の隣接ペア数 / 全可能ペア数
    各ノードには6個の距離1隣接 → 64*6/2 = 192 ペア
    全ペア数: 64*63/2 = 2016
    確率: 192/2016 = 6/63 ≈ 0.09524
    """
    # 実際に計算して確認
    d1_pairs = 0
    total_pairs = 0
    for i in range(64):
        for j in range(i + 1, 64):
            total_pairs += 1
            if hamming_distance(i, j) == 1:
                d1_pairs += 1
    prop = d1_pairs / total_pairs
    print(f"  Distance-1 pairs: {d1_pairs} / {total_pairs} = {prop:.6f}")
    return prop


def main():
    # ── データ読み込み ──
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "analysis", "gold_set", "hexagram_transitions.json"
    )
    with open(data_path, 'r') as f:
        data = json.load(f)

    transitions = data["transitions"]
    n = len(transitions)
    print(f"Total transitions: {n}")

    # ── マッピング検証 ──
    verify_mapping()

    # ── 2. ハミング距離計算 ──
    distances = []
    d1_count = 0
    distance_counts = Counter()

    for t in transitions:
        before_kw = t["before"]
        after_kw = t["after"]
        b_bin = KW_TO_BINARY[before_kw]
        a_bin = KW_TO_BINARY[after_kw]
        d = hamming_distance(b_bin, a_bin)
        distances.append(d)
        distance_counts[d] += 1
        if d == 1:
            d1_count += 1

    print(f"\n── Distance Distribution ──")
    for d in sorted(distance_counts.keys()):
        pct = distance_counts[d] / n * 100
        print(f"  d={d}: {distance_counts[d]} ({pct:.1f}%)")

    # ── 3a. 距離1遷移の実測割合 ──
    p_observed = d1_count / n
    print(f"\n── Distance-1 Analysis ──")
    print(f"  Observed d=1: {d1_count}/{n} = {p_observed:.4f} ({p_observed*100:.2f}%)")

    # ── 3b. ランダム期待値 ──
    print(f"\n── Random Expected Proportion ──")
    p_random = random_expected_d1_proportion()
    print(f"  Random expected: {p_random:.6f} ({p_random*100:.2f}%)")

    # ── 3c. 点推定（上乗せ） ──
    delta_point = p_observed - p_random
    print(f"\n── Point Estimate of Distance-1 Surplus ──")
    print(f"  Delta (observed - random): {delta_point:.4f} ({delta_point*100:.2f}%pt)")

    # ── 3d. 95%信頼区間（Clopper-Pearson正確信頼区間） ──
    # 二項分布の正確信頼区間
    alpha = 0.05
    # Lower bound
    if d1_count == 0:
        ci_lower = 0.0
    else:
        ci_lower = stats.beta.ppf(alpha / 2, d1_count, n - d1_count + 1)
    # Upper bound
    if d1_count == n:
        ci_upper = 1.0
    else:
        ci_upper = stats.beta.ppf(1 - alpha / 2, d1_count + 1, n - d1_count)

    print(f"\n── 95% Clopper-Pearson CI for d=1 proportion ──")
    print(f"  CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

    # ── 3e. 効果量上限 ──
    effect_upper = ci_upper - p_random
    effect_lower = ci_lower - p_random
    print(f"\n── Effect Size (surplus over random) ──")
    print(f"  Point estimate: {delta_point*100:+.2f}%pt")
    print(f"  95% CI: [{effect_lower*100:+.2f}%pt, {effect_upper*100:+.2f}%pt]")
    print(f"  Upper bound: {effect_upper*100:+.2f}%pt")
    print(f"  → 距離1上乗せは最大でも {effect_upper*100:.2f}%pt")

    # ── 4. Poisson回帰 ──
    print(f"\n{'='*60}")
    print(f"── Poisson Regression ──")

    # 遷移頻度テーブル: 各(before, after)ペアの出現回数
    pair_counts = Counter()
    for t in transitions:
        pair_counts[(t["before"], t["after"])] += 1

    # 全可能ペア（自己遷移除く）について距離と頻度を計算
    y_values = []  # 遷移頻度
    x_distance = []  # ハミング距離
    x_d1_dummy = []  # 距離1ダミー

    for kw_a in range(1, 65):
        for kw_b in range(1, 65):
            if kw_a == kw_b:
                continue
            freq = pair_counts.get((kw_a, kw_b), 0)
            d = hamming_distance(KW_TO_BINARY[kw_a], KW_TO_BINARY[kw_b])
            y_values.append(freq)
            x_distance.append(d)
            x_d1_dummy.append(1 if d == 1 else 0)

    y = np.array(y_values)
    X_d1 = np.array(x_d1_dummy)
    X_dist = np.array(x_distance)

    # Poisson回帰（距離1ダミー）: GLM
    # log(E[Y]) = beta0 + beta1 * I(d=1)
    try:
        import statsmodels.api as sm
        X_design = sm.add_constant(X_d1)
        poisson_model = sm.GLM(y, X_design, family=sm.families.Poisson())
        poisson_result = poisson_model.fit()

        print(f"  Model: log(E[freq]) = β0 + β1 * I(d=1)")
        print(f"  β0 (intercept): {poisson_result.params[0]:.4f}")
        print(f"  β1 (d=1 dummy): {poisson_result.params[1]:.4f}")
        print(f"  β1 p-value: {poisson_result.pvalues[1]:.4f}")
        ci_beta1 = poisson_result.conf_int()[1]
        print(f"  β1 95% CI: [{ci_beta1[0]:.4f}, {ci_beta1[1]:.4f}]")

        # Rate Ratio (exp(β1))
        rr = np.exp(poisson_result.params[1])
        rr_ci_lower = np.exp(ci_beta1[0])
        rr_ci_upper = np.exp(ci_beta1[1])
        print(f"  Rate Ratio exp(β1): {rr:.4f}")
        print(f"  RR 95% CI: [{rr_ci_lower:.4f}, {rr_ci_upper:.4f}]")

        poisson_significant = poisson_result.pvalues[1] < 0.05
        print(f"  Significant at α=0.05: {poisson_significant}")

        poisson_results = {
            "beta0": round(float(poisson_result.params[0]), 4),
            "beta1_d1_dummy": round(float(poisson_result.params[1]), 4),
            "beta1_pvalue": round(float(poisson_result.pvalues[1]), 4),
            "beta1_ci_lower": round(float(ci_beta1[0]), 4),
            "beta1_ci_upper": round(float(ci_beta1[1]), 4),
            "rate_ratio": round(float(rr), 4),
            "rate_ratio_ci_lower": round(float(rr_ci_lower), 4),
            "rate_ratio_ci_upper": round(float(rr_ci_upper), 4),
            "significant": poisson_significant,
        }

        # 追加: 距離を連続変数として入れたモデル
        print(f"\n  --- Additional: Distance as continuous variable ---")
        X_design2 = sm.add_constant(X_dist)
        poisson_model2 = sm.GLM(y, X_design2, family=sm.families.Poisson())
        poisson_result2 = poisson_model2.fit()
        print(f"  β0 (intercept): {poisson_result2.params[0]:.4f}")
        print(f"  β1 (distance): {poisson_result2.params[1]:.4f}")
        print(f"  β1 p-value: {poisson_result2.pvalues[1]:.4f}")
        ci_dist = poisson_result2.conf_int()[1]
        print(f"  β1 95% CI: [{ci_dist[0]:.4f}, {ci_dist[1]:.4f}]")

        poisson_distance_results = {
            "beta0": round(float(poisson_result2.params[0]), 4),
            "beta1_distance": round(float(poisson_result2.params[1]), 4),
            "beta1_pvalue": round(float(poisson_result2.pvalues[1]), 4),
            "beta1_ci_lower": round(float(ci_dist[0]), 4),
            "beta1_ci_upper": round(float(ci_dist[1]), 4),
        }

    except ImportError:
        print("  [WARNING] statsmodels not available, using manual calculation")
        # 手動でPoisson回帰の近似
        d1_pairs_n = sum(X_d1)
        d1_freq_sum = sum(y[i] for i in range(len(y)) if X_d1[i] == 1)
        other_pairs_n = len(y) - d1_pairs_n
        other_freq_sum = sum(y[i] for i in range(len(y)) if X_d1[i] == 0)

        d1_rate = d1_freq_sum / d1_pairs_n if d1_pairs_n > 0 else 0
        other_rate = other_freq_sum / other_pairs_n if other_pairs_n > 0 else 0

        print(f"  d=1 pairs: {d1_pairs_n}, total freq: {d1_freq_sum}, rate: {d1_rate:.4f}")
        print(f"  d≠1 pairs: {other_pairs_n}, total freq: {other_freq_sum}, rate: {other_rate:.4f}")
        print(f"  Rate ratio (d=1 / d≠1): {d1_rate/other_rate:.4f}" if other_rate > 0 else "  Rate ratio: N/A")

        poisson_results = {
            "d1_pairs": int(d1_pairs_n),
            "d1_freq_sum": int(d1_freq_sum),
            "d1_rate": round(float(d1_rate), 4),
            "other_rate": round(float(other_rate), 4),
            "rate_ratio": round(float(d1_rate / other_rate), 4) if other_rate > 0 else None,
            "note": "statsmodels not available, manual calculation",
        }
        poisson_distance_results = None

    # ── 5. TOST等価性検定 ──
    print(f"\n{'='*60}")
    print(f"── TOST (Two One-Sided Tests) ──")
    equivalence_margin = 0.05  # ±5%pt
    print(f"  Equivalence margin: ±{equivalence_margin*100:.0f}%pt")
    print(f"  H0: |p_observed - p_random| >= {equivalence_margin*100:.0f}%pt")
    print(f"  H1: |p_observed - p_random| < {equivalence_margin*100:.0f}%pt (equivalent)")

    # SE of proportion
    se = np.sqrt(p_observed * (1 - p_observed) / n)
    print(f"  SE: {se:.6f}")

    # TOST: Two one-sided z-tests
    # Test 1: H0: delta <= -margin vs H1: delta > -margin
    z1 = (delta_point - (-equivalence_margin)) / se
    p1 = 1 - stats.norm.cdf(z1)  # one-sided p-value (upper tail)

    # Test 2: H0: delta >= +margin vs H1: delta < +margin
    z2 = (delta_point - equivalence_margin) / se
    p2 = stats.norm.cdf(z2)  # one-sided p-value (lower tail)

    tost_p = max(p1, p2)
    tost_equivalent = tost_p < 0.05

    print(f"\n  Test 1 (delta > -{equivalence_margin*100:.0f}%pt):")
    print(f"    z = {z1:.4f}, p = {p1:.6f}")
    print(f"  Test 2 (delta < +{equivalence_margin*100:.0f}%pt):")
    print(f"    z = {z2:.4f}, p = {p2:.6f}")
    print(f"\n  TOST p-value: {tost_p:.6f}")
    print(f"  Equivalence demonstrated (α=0.05): {tost_equivalent}")

    if tost_equivalent:
        print(f"  → 距離1割合はランダム期待値の±{equivalence_margin*100:.0f}%pt以内と統計的に示された")
    else:
        print(f"  → 等価性は示されなかった（データ不足の可能性）")
        # 等価性が示されるために必要なサンプルサイズを推定
        # 必要なn: z_alpha^2 * p*(1-p) / (margin - |delta|)^2
        if abs(delta_point) < equivalence_margin:
            z_alpha = stats.norm.ppf(0.975)
            needed_n = (z_alpha ** 2 * p_observed * (1 - p_observed)) / \
                       ((equivalence_margin - abs(delta_point)) ** 2)
            print(f"  → 等価性を示すのに推定必要サンプルサイズ: ~{int(needed_n)}")

    # ── 追加: より狭いマージンでのTOST ──
    print(f"\n  --- TOST with ±3%pt margin ---")
    margin_3 = 0.03
    z1_3 = (delta_point - (-margin_3)) / se
    p1_3 = 1 - stats.norm.cdf(z1_3)
    z2_3 = (delta_point - margin_3) / se
    p2_3 = stats.norm.cdf(z2_3)
    tost_p_3 = max(p1_3, p2_3)
    print(f"  TOST p-value (±3%pt): {tost_p_3:.6f}")
    print(f"  Equivalence (±3%pt): {tost_p_3 < 0.05}")

    # ── 結果まとめ ──
    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"── SUMMARY ──")
    print(f"  N = {n}")
    print(f"  距離1遷移: {d1_count}件 ({p_observed*100:.2f}%)")
    print(f"  ランダム期待: {p_random*100:.2f}%")
    print(f"  差分(点推定): {delta_point*100:+.2f}%pt")
    print(f"  95% CI of proportion: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print(f"  効果量上限(CI上端 - ランダム): {effect_upper*100:+.2f}%pt")
    print(f"  TOST等価性(±5%pt): {'示された' if tost_equivalent else '示されなかった'}")

    if effect_upper <= 0:
        conclusion = "距離1への集中は観測されない（むしろ逆方向の傾向）"
    elif effect_upper <= 0.02:
        conclusion = f"距離1上乗せは最大でも{effect_upper*100:.1f}%pt。実質的に無視可能"
    elif effect_upper <= 0.05:
        conclusion = f"距離1上乗せは最大{effect_upper*100:.1f}%pt。小さいが完全にゼロとは言えない"
    else:
        conclusion = f"距離1上乗せの上限は{effect_upper*100:.1f}%pt。無視できない可能性あり"

    print(f"\n  結論: {conclusion}")

    # ── JSON出力 ──
    output = {
        "metadata": {
            "n_transitions": n,
            "analysis": "Q6 distance-1 effect size upper bound",
            "data_source": "analysis/gold_set/hexagram_transitions.json",
            "phase": "Phase 3 v4.0",
        },
        "distance_distribution": {
            str(d): {
                "count": distance_counts[d],
                "proportion": round(distance_counts[d] / n, 4),
            }
            for d in sorted(distance_counts.keys())
        },
        "distance_1_analysis": {
            "observed_count": d1_count,
            "observed_proportion": round(p_observed, 6),
            "random_expected_proportion": round(p_random, 6),
            "delta_point_estimate": round(delta_point, 6),
            "delta_point_estimate_pct": round(delta_point * 100, 2),
            "ci_95_lower": round(ci_lower, 6),
            "ci_95_upper": round(ci_upper, 6),
            "effect_size_lower_bound": round(effect_lower, 6),
            "effect_size_upper_bound": round(effect_upper, 6),
            "effect_size_upper_bound_pct": round(effect_upper * 100, 2),
        },
        "poisson_regression": {
            "d1_dummy_model": poisson_results,
            "distance_continuous_model": poisson_distance_results,
        },
        "tost_equivalence": {
            "margin": equivalence_margin,
            "margin_pct": equivalence_margin * 100,
            "z1": round(float(z1), 4),
            "p1": round(float(p1), 6),
            "z2": round(float(z2), 4),
            "p2": round(float(p2), 6),
            "tost_p_value": round(float(tost_p), 6),
            "equivalence_demonstrated": tost_equivalent,
            "tost_3pct_margin": {
                "p_value": round(float(tost_p_3), 6),
                "equivalence_demonstrated": tost_p_3 < 0.05,
            },
        },
        "conclusion": conclusion,
    }

    # numpy bool / int64 を Python ネイティブ型に変換
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    output = convert_types(output)

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "analysis", "phase3", "effect_size_upper_bound.json"
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
