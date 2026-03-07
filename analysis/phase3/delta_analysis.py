#!/usr/bin/env python3
"""
Phase 3: Delta Analysis — Codex提案「Delta = before XOR after（6bit XOR差分）」の分布分析

背景:
  同型性検証v2.0でハミング距離分布に偶数パリティへの強い偏向(98.6%)が発見された。
  本スクリプトはΔベクトル(6bit XOR差分)を一次データとして扱い、
  偶数重みへの集中・特定Δの過剰出現をモデル化して検定する。

分析内容:
  1. Δベクトルの基本統計（ハミング重み分布、偶数vs奇数）
  2. Δパターンの頻度分析（64通りの出現頻度）
  3. 偶数パリティの検定（二項検定 + カイ二乗適合度）
  4. ビット位置ごとの反転率
  5. 上卦・下卦の独立性検定
  6. 置換テスト（偶数パリティ偏向の構造的原因検証）
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime
from scipy.stats import binomtest, chi2_contingency
from scipy.stats import chisquare

# isomorphism_test.py から共有ユーティリティをインポート
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isomorphism_test import (
    load_json,
    load_cases,
    build_name_to_kw,
    build_kw_to_bits,
    resolve_hexagram_field,
    TRIGRAM_BITS,
    REFERENCE_FILE,
    CASES_FILE,
)

# ---------- パス設定 ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PHASE3_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = PHASE3_DIR / "delta_analysis.json"

# ---------- 定数 ----------
RANDOM_SEED = 42
N_PERMUTATIONS = 1000


# ============================================================
# ユーティリティ
# ============================================================

def xor_bits(bits_a, bits_b):
    """2つの6bitタプルのXOR → タプル"""
    return tuple(a ^ b for a, b in zip(bits_a, bits_b))


def hamming_weight(bits):
    """6bitタプルのハミング重み（1の個数）"""
    return sum(bits)


def bits_to_str(bits):
    """6bitタプルを文字列に変換"""
    return ''.join(str(b) for b in bits)


def extract_transitions(cases, name_to_kw, kw_to_bits):
    """cases からΔベクトルを含む遷移リストを抽出"""
    transitions = []
    n_excluded = 0

    for case in cases:
        before_val = case.get('classical_before_hexagram', '')
        after_val = case.get('classical_after_hexagram', '')

        kw_from = resolve_hexagram_field(before_val, name_to_kw)
        kw_to = resolve_hexagram_field(after_val, name_to_kw)

        if kw_from is None or kw_to is None:
            n_excluded += 1
            continue
        if kw_from not in kw_to_bits or kw_to not in kw_to_bits:
            n_excluded += 1
            continue

        bits_from = kw_to_bits[kw_from]
        bits_to = kw_to_bits[kw_to]
        delta = xor_bits(bits_from, bits_to)

        transitions.append({
            'kw_from': kw_from,
            'kw_to': kw_to,
            'bits_from': bits_from,
            'bits_to': bits_to,
            'delta': delta,
            'hamming_weight': hamming_weight(delta),
        })

    return transitions, n_excluded


# ============================================================
# 1. Δベクトルの基本統計
# ============================================================

def analyze_hamming_weight_distribution(transitions):
    """ハミング重み分布と偶数vs奇数の比率"""
    print("\n[1] Deltaベクトルの基本統計")
    print("-" * 40)

    weights = [t['hamming_weight'] for t in transitions]
    weight_counter = Counter(weights)

    # 0〜6の全重みを含める
    dist = {str(w): weight_counter.get(w, 0) for w in range(7)}
    total = len(weights)

    print(f"  遷移ペア数: {total}")
    print(f"  ハミング重み分布:")
    for w in range(7):
        count = weight_counter.get(w, 0)
        pct = count / total * 100 if total > 0 else 0
        bar = '#' * int(pct)
        print(f"    w={w}: {count:>5} ({pct:>5.1f}%) {bar}")

    even_count = sum(weight_counter.get(w, 0) for w in [0, 2, 4, 6])
    odd_count = sum(weight_counter.get(w, 0) for w in [1, 3, 5])
    even_ratio = even_count / total if total > 0 else 0

    print(f"\n  偶数重み(0,2,4,6): {even_count} ({even_ratio:.4f})")
    print(f"  奇数重み(1,3,5):   {odd_count} ({1 - even_ratio:.4f})")

    # 二項検定: 偶数重みの割合が50%と異なるか
    # scipy.stats.binom_test は両側検定
    binom_result = binomtest(even_count, total, 0.5, alternative='two-sided')
    binom_p = binom_result.pvalue
    effect_size = (even_ratio - 0.5) / 0.5  # 相対的偏差

    print(f"  二項検定 (H0: even_ratio=0.5): p={binom_p:.2e}, effect_size={effect_size:.4f}")

    return {
        'hamming_weight_distribution': dist,
        'even_parity_ratio': float(even_ratio),
        'even_count': int(even_count),
        'odd_count': int(odd_count),
        'even_parity_binomial_test': {
            'p_value': float(binom_p),
            'effect_size': float(effect_size),
            'n': int(total),
            'even_count': int(even_count),
        },
    }


# ============================================================
# 2. Δパターンの頻度分析
# ============================================================

def analyze_delta_patterns(transitions):
    """64通りのΔベクトルの出現頻度"""
    print("\n[2] Deltaパターンの頻度分析")
    print("-" * 40)

    delta_counter = Counter(t['delta'] for t in transitions)
    total = len(transitions)

    # 解釈マッピング
    def interpret_delta(delta):
        interpretations = []
        if delta == (0, 0, 0, 0, 0, 0):
            interpretations.append("同一卦への遷移")
        if delta == (1, 1, 1, 1, 1, 1):
            interpretations.append("錯卦（全反転）への遷移")
        lower = delta[:3]
        upper = delta[3:]
        if lower == (1, 1, 1) and upper == (0, 0, 0):
            interpretations.append("下卦のみ全反転")
        if lower == (0, 0, 0) and upper == (1, 1, 1):
            interpretations.append("上卦のみ全反転")
        if lower == (0, 0, 0) and upper != (0, 0, 0):
            interpretations.append("上卦のみ変化")
        if upper == (0, 0, 0) and lower != (0, 0, 0):
            interpretations.append("下卦のみ変化")
        hw = sum(delta)
        interpretations.append(f"ハミング重み={hw}")
        return "; ".join(interpretations)

    # 上位20パターン
    top_patterns = delta_counter.most_common(20)
    print(f"  ユニークΔパターン数: {len(delta_counter)} / 64")
    print(f"\n  上位20パターン:")
    print(f"  {'Delta':>8}  {'Count':>5}  {'%':>6}  解釈")
    print(f"  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*30}")

    top_patterns_json = []
    for delta, count in top_patterns:
        pct = count / total * 100
        interp = interpret_delta(delta)
        delta_str = bits_to_str(delta)
        print(f"  {delta_str:>8}  {count:>5}  {pct:>5.1f}%  {interp}")
        top_patterns_json.append({
            'delta_bits': delta_str,
            'count': int(count),
            'percentage': round(float(pct), 2),
            'interpretation': interp,
        })

    # 特殊パターンの出現数
    special = {
        'same_hexagram': delta_counter.get((0, 0, 0, 0, 0, 0), 0),
        'cuogua_full_flip': delta_counter.get((1, 1, 1, 1, 1, 1), 0),
        'lower_only_full_flip': delta_counter.get((1, 1, 1, 0, 0, 0), 0),
        'upper_only_full_flip': delta_counter.get((0, 0, 0, 1, 1, 1), 0),
    }
    print(f"\n  特殊パターン:")
    for name, count in special.items():
        print(f"    {name}: {count}")

    return {
        'n_unique_patterns': len(delta_counter),
        'top_delta_patterns': top_patterns_json,
        'special_patterns': special,
    }


# ============================================================
# 3. 偶数パリティの検定（カイ二乗適合度）
# ============================================================

def test_chi2_goodness_of_fit(transitions):
    """カイ二乗適合度検定: 観測分布 vs Bin(6, 0.5)"""
    print("\n[3] 偶数パリティの検定（カイ二乗適合度）")
    print("-" * 40)

    from scipy.stats import binom as binom_dist

    weights = [t['hamming_weight'] for t in transitions]
    total = len(weights)
    weight_counter = Counter(weights)

    # 帰無分布: Bin(6, 0.5)
    expected_probs = {w: binom_dist.pmf(w, 6, 0.5) for w in range(7)}
    expected_counts = {w: expected_probs[w] * total for w in range(7)}

    # Bin(6,0.5)の偶数重み確率
    even_prob_theoretical = sum(expected_probs[w] for w in [0, 2, 4, 6])
    print(f"  Bin(6,0.5)の偶数重み確率: {even_prob_theoretical:.4f}")

    observed = np.array([weight_counter.get(w, 0) for w in range(7)])
    expected = np.array([expected_counts[w] for w in range(7)])

    print(f"\n  {'Weight':>6}  {'Observed':>8}  {'Expected':>8}  {'O/E':>6}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}")
    for w in range(7):
        o = observed[w]
        e = expected[w]
        ratio = o / e if e > 0 else float('inf')
        print(f"  {w:>6}  {o:>8}  {e:>8.1f}  {ratio:>6.2f}")

    # カイ二乗適合度検定
    # 期待度数が小さい場合はビンを統合（w=0とw=6を隣接ビンと統合）
    # ここでは全ビンで実行
    chi2_stat, chi2_p = chisquare(observed, f_exp=expected)

    print(f"\n  カイ二乗統計量: {chi2_stat:.2f}")
    print(f"  p値: {chi2_p:.2e}")

    return {
        'chi2_goodness_of_fit': {
            'statistic': float(chi2_stat),
            'p_value': float(chi2_p),
            'df': 6,
            'expected_dist': {str(w): round(float(expected_counts[w]), 2) for w in range(7)},
            'observed_dist': {str(w): int(weight_counter.get(w, 0)) for w in range(7)},
            'even_prob_theoretical': float(even_prob_theoretical),
        }
    }


# ============================================================
# 4. ビット位置ごとの反転率
# ============================================================

def analyze_bit_flip_rates(transitions):
    """6ビットの各位置の反転率"""
    print("\n[4] ビット位置ごとの反転率")
    print("-" * 40)

    total = len(transitions)
    bit_names = ['bit0_lower1', 'bit1_lower2', 'bit2_lower3',
                 'bit3_upper1', 'bit4_upper2', 'bit5_upper3']

    flip_counts = [0] * 6
    for t in transitions:
        for i in range(6):
            flip_counts[i] += t['delta'][i]

    flip_rates = {}
    print(f"  {'Position':>14}  {'Flips':>6}  {'Rate':>6}")
    print(f"  {'-'*14}  {'-'*6}  {'-'*6}")
    for i in range(6):
        rate = flip_counts[i] / total if total > 0 else 0
        flip_rates[bit_names[i]] = round(float(rate), 4)
        print(f"  {bit_names[i]:>14}  {flip_counts[i]:>6}  {rate:>6.4f}")

    # 上卦vs下卦の反転率比較
    lower_rate = sum(flip_counts[:3]) / (total * 3) if total > 0 else 0
    upper_rate = sum(flip_counts[3:]) / (total * 3) if total > 0 else 0
    print(f"\n  下卦平均反転率: {lower_rate:.4f}")
    print(f"  上卦平均反転率: {upper_rate:.4f}")

    flip_rates['lower_trigram_avg'] = round(float(lower_rate), 4)
    flip_rates['upper_trigram_avg'] = round(float(upper_rate), 4)

    return {'bit_flip_rates': flip_rates}


# ============================================================
# 5. 上卦・下卦の独立性検定
# ============================================================

def test_upper_lower_independence(transitions):
    """上卦変化と下卦変化の独立性検定"""
    print("\n[5] 上卦・下卦の独立性検定")
    print("-" * 40)

    # 各遷移について、下卦が変化したか・上卦が変化したかを判定
    # 下卦変化: delta[:3]に1が含まれる
    # 上卦変化: delta[3:]に1が含まれる
    contingency = np.zeros((2, 2), dtype=int)  # [lower_changed][upper_changed]

    for t in transitions:
        lower_changed = 1 if any(t['delta'][:3]) else 0
        upper_changed = 1 if any(t['delta'][3:]) else 0
        contingency[lower_changed][upper_changed] += 1

    print(f"  分割表 (下卦変化 x 上卦変化):")
    print(f"               上卦不変  上卦変化")
    print(f"  下卦不変     {contingency[0][0]:>6}    {contingency[0][1]:>6}")
    print(f"  下卦変化     {contingency[1][0]:>6}    {contingency[1][1]:>6}")

    # カイ二乗独立性検定
    chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency)

    # Cramer's V
    n = contingency.sum()
    k = min(contingency.shape)
    cramers_v = np.sqrt(chi2_stat / (n * (k - 1))) if n > 0 and k > 1 else 0

    print(f"\n  カイ二乗統計量: {chi2_stat:.4f}")
    print(f"  p値: {chi2_p:.2e}")
    print(f"  Cramer's V: {cramers_v:.4f}")

    # 追加: Δの上3bitと下3bitのパターン相関
    lower_patterns = Counter(t['delta'][:3] for t in transitions)
    upper_patterns = Counter(t['delta'][3:] for t in transitions)

    print(f"\n  下卦Δパターン分布 (上位5):")
    for pattern, count in lower_patterns.most_common(5):
        print(f"    {bits_to_str(pattern)}: {count}")
    print(f"  上卦Δパターン分布 (上位5):")
    for pattern, count in upper_patterns.most_common(5):
        print(f"    {bits_to_str(pattern)}: {count}")

    return {
        'upper_lower_independence': {
            'chi2': float(chi2_stat),
            'p_value': float(chi2_p),
            'cramers_v': float(cramers_v),
            'contingency_table': {
                'lower_unchanged_upper_unchanged': int(contingency[0][0]),
                'lower_unchanged_upper_changed': int(contingency[0][1]),
                'lower_changed_upper_unchanged': int(contingency[1][0]),
                'lower_changed_upper_changed': int(contingency[1][1]),
            },
        }
    }


# ============================================================
# 6. 置換テスト
# ============================================================

def permutation_test(transitions, kw_to_bits, rng):
    """
    H0: Δの偶数パリティ偏向は、before卦とafter卦の周辺分布から生じる
    帰無分布: after列をシャッフル → Δ計算 → 偶数パリティ比率。1,000回
    """
    print("\n[6] 置換テスト")
    print("-" * 40)

    # 観測された偶数パリティ比率
    total = len(transitions)
    observed_even = sum(1 for t in transitions if t['hamming_weight'] % 2 == 0)
    observed_ratio = observed_even / total if total > 0 else 0

    print(f"  観測偶数パリティ比率: {observed_ratio:.4f}")

    # after列シャッフル
    before_bits_list = [t['bits_from'] for t in transitions]
    after_kw_list = [t['kw_to'] for t in transitions]

    perm_even_ratios = []
    for i in range(N_PERMUTATIONS):
        perm_after = list(after_kw_list)
        rng.shuffle(perm_after)

        even_count = 0
        for bits_from, kw_to in zip(before_bits_list, perm_after):
            bits_to = kw_to_bits[kw_to]
            delta = xor_bits(bits_from, bits_to)
            if hamming_weight(delta) % 2 == 0:
                even_count += 1
        perm_even_ratios.append(even_count / total)

        if (i + 1) % 200 == 0:
            print(f"    置換 {i+1}/{N_PERMUTATIONS}")

    perm_even_ratios = np.array(perm_even_ratios)

    null_mean = np.mean(perm_even_ratios)
    null_std = np.std(perm_even_ratios)
    z_score = (observed_ratio - null_mean) / null_std if null_std > 0 else 0.0
    p_value = np.mean(perm_even_ratios >= observed_ratio)

    print(f"  帰無平均: {null_mean:.4f}")
    print(f"  帰無標準偏差: {null_std:.4f}")
    print(f"  z値: {z_score:.3f}")
    print(f"  p値 (片側: 観測 > ランダム): {p_value:.4f}")

    return {
        'permutation_test': {
            'observed_even_ratio': float(observed_ratio),
            'null_mean': float(null_mean),
            'null_std': float(null_std),
            'p_value': float(p_value),
            'z_score': float(z_score),
            'n_permutations': N_PERMUTATIONS,
        }
    }


# ============================================================
# メイン
# ============================================================

def main():
    print("=" * 60)
    print("Phase 3: Delta Analysis (6bit XOR差分)")
    print("  Codex提案: Delta = before XOR after の分布分析")
    print("=" * 60)

    t_start = time.time()
    rng = np.random.default_rng(RANDOM_SEED)

    # --- データ読み込み ---
    print("\n[0] データ読み込み...")
    reference_data = load_json(REFERENCE_FILE)
    cases = load_cases()
    print(f"  事例数: {len(cases)}")

    # --- マッピング構築 ---
    name_to_kw = build_name_to_kw(reference_data)
    kw_to_bits, bits_to_kw = build_kw_to_bits(reference_data)
    print(f"  卦名→番号: {len(name_to_kw)}件")
    print(f"  番号→6bit: {len(kw_to_bits)}件")

    # --- 遷移抽出 ---
    transitions, n_excluded = extract_transitions(cases, name_to_kw, kw_to_bits)
    n_used = len(transitions)
    print(f"  有効遷移ペア: {n_used} (除外: {n_excluded})")

    if n_used == 0:
        print("ERROR: 有効な遷移ペアが0件。終了。")
        return

    # --- 分析実行 ---
    results = {}

    # メタデータ
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'n_cases_used': n_used,
        'n_cases_excluded': n_excluded,
        'seed': RANDOM_SEED,
    }

    # 1. ハミング重み分布
    r1 = analyze_hamming_weight_distribution(transitions)
    results['hamming_weight_distribution'] = r1['hamming_weight_distribution']
    results['even_parity_ratio'] = r1['even_parity_ratio']
    results['even_parity_binomial_test'] = r1['even_parity_binomial_test']

    # 2. Δパターン頻度
    r2 = analyze_delta_patterns(transitions)
    results['top_delta_patterns'] = r2['top_delta_patterns']
    results['n_unique_patterns'] = r2['n_unique_patterns']
    results['special_patterns'] = r2['special_patterns']

    # 3. カイ二乗適合度
    r3 = test_chi2_goodness_of_fit(transitions)
    results['chi2_goodness_of_fit'] = r3['chi2_goodness_of_fit']

    # 4. ビット反転率
    r4 = analyze_bit_flip_rates(transitions)
    results['bit_flip_rates'] = r4['bit_flip_rates']

    # 5. 上卦・下卦独立性
    r5 = test_upper_lower_independence(transitions)
    results['upper_lower_independence'] = r5['upper_lower_independence']

    # 6. 置換テスト
    r6 = permutation_test(transitions, kw_to_bits, rng)
    results['permutation_test'] = r6['permutation_test']

    # --- JSON出力 ---
    print("\n" + "=" * 60)
    print("[出力] 結果をJSONに保存")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {OUTPUT_JSON}")

    # --- サマリー ---
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Delta Analysis 完了 ({elapsed:.1f}秒)")
    print(f"{'=' * 60}")
    print(f"\n  主要な発見:")
    print(f"    偶数パリティ比率:   {results['even_parity_ratio']:.4f}")
    print(f"    二項検定 p値:       {results['even_parity_binomial_test']['p_value']:.2e}")
    print(f"    カイ二乗検定 p値:   {results['chi2_goodness_of_fit']['p_value']:.2e}")
    print(f"    上下独立性 p値:     {results['upper_lower_independence']['p_value']:.2e}")
    print(f"    上下独立性 V:       {results['upper_lower_independence']['cramers_v']:.4f}")
    print(f"    置換テスト p値:     {results['permutation_test']['p_value']:.4f}")
    print(f"    置換テスト z値:     {results['permutation_test']['z_score']:.3f}")


if __name__ == '__main__':
    main()
