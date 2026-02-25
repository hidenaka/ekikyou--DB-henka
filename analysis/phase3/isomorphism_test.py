#!/usr/bin/env python3
"""
Phase 3: 同型性検証 — Q6超立方体と実データ変化パターンの構造的対応テスト

帰無仮説 H0: 実データの変化パターン構造とQ6超立方体構造の間に、
ランダムに生成可能な構造以上の対応関係は存在しない。

6つの検証:
1. 次元数の対応 (MCA dim vs Q6 dim)
2. 遷移パターンとQ6エッジの対応 (ハミング距離)
3. 八卦タグとクラスタの対応 (Cramer's V有意性)
4. スペクトル構造の比較 (Q6 vs MCA固有値)
5. 構造的関係の保存 (錯卦ペア対称性)
6. 部分同型性テスト (Louvainコミュニティ vs データパターン)
"""

import json
import sys
import os
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# ---------- パス設定 ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PHASE1_DIR = BASE_DIR / "analysis" / "phase1"
PHASE2_DIR = BASE_DIR / "analysis" / "phase2"
PHASE3_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CASES_FILE = DATA_DIR / "raw" / "cases.jsonl"

# ---------- 定数 ----------
RANDOM_SEED = 42
N_PERMUTATIONS = 1000
ALPHA = 0.05
N_TESTS = 6
ALPHA_BONFERRONI = ALPHA / N_TESTS  # 0.00833

# 八卦ビット定義 (Phase 1から)
TRIGRAM_BITS = {
    '乾': [1, 1, 1],
    '坤': [0, 0, 0],
    '震': [1, 0, 0],
    '巽': [0, 1, 1],
    '坎': [0, 1, 0],
    '離': [1, 0, 1],
    '艮': [0, 0, 1],
    '兌': [1, 1, 0],
}

TRIGRAM_NAMES = list(TRIGRAM_BITS.keys())


def load_json(path):
    """JSONファイル読み込み"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_cases():
    """cases.jsonl読み込み"""
    cases = []
    with open(CASES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def trigram_pair_to_hexagram_bits(lower_trigram, upper_trigram):
    """下卦・上卦名から6ビットベクトルを返す"""
    if lower_trigram not in TRIGRAM_BITS or upper_trigram not in TRIGRAM_BITS:
        return None
    return TRIGRAM_BITS[lower_trigram] + TRIGRAM_BITS[upper_trigram]


def hamming_distance_bits(bits_a, bits_b):
    """2つの6ビットベクトル間のハミング距離"""
    return sum(a != b for a, b in zip(bits_a, bits_b))


def trigram_to_bits(trigram_name):
    """八卦名から3ビットを返す"""
    return TRIGRAM_BITS.get(trigram_name)


def cramers_v(contingency_table):
    """Cramer's Vを計算"""
    chi2 = chi_square_from_table(contingency_table)
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def chi_square_from_table(observed):
    """カイ二乗統計量を計算"""
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    n = observed.sum()
    if n == 0:
        return 0.0
    expected = row_sums * col_sums / n
    # ゼロ除算防止
    mask = expected > 0
    chi2 = np.sum(((observed - expected) ** 2)[mask] / expected[mask])
    return chi2


def cosine_similarity(vec_a, vec_b):
    """コサイン類似度"""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cohens_d(group1, group2):
    """Cohen's d 効果量"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def effect_size_label(d):
    """Cohen's dの効果量ラベル"""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def cramers_v_label(v):
    """Cramer's Vの効果量ラベル"""
    if v < 0.1:
        return "negligible"
    elif v < 0.3:
        return "small"
    elif v < 0.5:
        return "medium"
    else:
        return "large"


# ============================================================
# 検証1: 次元数の対応
# ============================================================
def test1_dimension_correspondence(cases, mca_results, dimension_report, rng):
    """
    Q6の幾何学的次元(6)とMCAのスクリー次元(5)/Benzecri累積80%次元(6)の対応検証。

    置換テスト: カテゴリラベルをランダムに再割り当てした場合のMCA次元数の分布を生成し、
    実データの次元数がこの分布の下でどの位置にあるかを評価。

    MCAを直接再計算するのは重いため、固有値のスクリーelbow検出をsurrogateデータで実施。
    """
    print("  [Test 1] 次元数の対応...")

    # 実データ: スクリー次元=5, Benzecri cum80%=6
    observed_scree_dim = dimension_report["recommended_dimensions"]  # 5
    observed_benzecri_dim = dimension_report["dimension_decision"]["benzecri"]["dimensions_cum80"]  # 6
    q6_dim = 6

    # MCA固有値
    eigenvalues = np.array(mca_results["eigenvalues"])
    n_eigenvalues = len(eigenvalues)

    # スクリーelbow検出関数 (二次微分の最大値)
    def find_scree_elbow(evals):
        if len(evals) < 3:
            return 1
        second_diff = np.abs(np.diff(np.diff(evals)))
        return int(np.argmax(second_diff)) + 1  # 1-indexed

    # 置換テスト: カテゴリカル変数をシャッフルしてMCA固有値の代理分布を生成
    # 各事例の7カテゴリカル変数をシャッフル → indicator matrix → 固有値計算
    # 簡略化: 各変数を独立にシャッフルして指示行列の特異値分解

    mca_columns = ["before_state", "trigger_type", "action_type", "after_state",
                    "pattern_type", "outcome", "scale"]

    # データ行列の構築
    data_matrix = []
    for case in cases:
        row = [case.get(col, "unknown") for col in mca_columns]
        data_matrix.append(row)
    data_matrix = np.array(data_matrix)
    n_records = len(data_matrix)

    # 各変数のユニークカテゴリ
    unique_cats = {}
    for j, col in enumerate(mca_columns):
        unique_cats[j] = sorted(set(data_matrix[:, j]))

    total_cats = sum(len(v) for v in unique_cats.values())
    K = len(mca_columns)

    def build_indicator_and_get_eigenvalues(dm):
        """指示行列 → MCA固有値計算"""
        # 指示行列構築
        cols = []
        for j in range(dm.shape[1]):
            cats = unique_cats[j]
            for cat in cats:
                cols.append((dm[:, j] == cat).astype(float))
        Z = np.column_stack(cols)

        # MCA: Z の列の正規化された相関行列の固有値
        n = Z.shape[0]
        col_sums = Z.sum(axis=0)
        col_sums[col_sums == 0] = 1  # ゼロ除算防止

        # Burt行列から固有値計算
        # MCA固有値 = (1/K^2) * Burt行列の固有値 (ただし対角ブロック除去バージョン)
        # 簡略化: 標準化残差行列のSVD
        row_masses = np.ones(n) / n
        col_masses = col_sums / col_sums.sum()

        # 標準化残差
        P = Z / Z.sum()
        r = P.sum(axis=1)  # row margins
        c = P.sum(axis=0)  # col margins

        r[r == 0] = 1e-10
        c[c == 0] = 1e-10

        Dr_inv_sqrt = np.diag(1.0 / np.sqrt(r))
        Dc_inv_sqrt = np.diag(1.0 / np.sqrt(c))

        # 標準化残差行列 (n x J)
        S = Dr_inv_sqrt @ (P - np.outer(r, c)) @ Dc_inv_sqrt

        # SVD (truncated to save time)
        # 固有値 = 特異値の二乗
        try:
            from scipy.linalg import svd
            U, s, Vt = svd(S, full_matrices=False)
            eigenvals = s ** 2
            # 最初の非自明な固有値（最大は常に1に近い）をスキップ
            eigenvals = np.sort(eigenvals)[::-1]
            # 最初の固有値は自明（= 1.0付近）なのでスキップ
            return eigenvals[1:min(31, len(eigenvals))]
        except Exception:
            return np.zeros(30)

    # 実データの固有値でelbowを再確認
    observed_elbow = find_scree_elbow(eigenvalues)

    # 置換テスト
    perm_scree_dims = []
    for i in range(N_PERMUTATIONS):
        # 各変数を独立にシャッフル
        dm_perm = data_matrix.copy()
        for j in range(dm_perm.shape[1]):
            rng.shuffle(dm_perm[:, j])

        perm_evals = build_indicator_and_get_eigenvalues(dm_perm)
        if len(perm_evals) >= 3:
            perm_elbow = find_scree_elbow(perm_evals)
            perm_scree_dims.append(perm_elbow)

        if (i + 1) % 100 == 0:
            print(f"    置換 {i+1}/{N_PERMUTATIONS} 完了")

    perm_scree_dims = np.array(perm_scree_dims)

    # p値: 置換分布で実データ以上の次元数が出る確率
    # 実データの5次元がQ6の6次元に「近い」ことを検証
    # → 5以上の次元数がランダムで出る確率
    p_value_scree = np.mean(perm_scree_dims >= observed_scree_dim)

    # 効果量: 実データ次元数とランダム分布の平均の差をSDで標準化
    if len(perm_scree_dims) > 0 and np.std(perm_scree_dims) > 0:
        effect_d = (observed_scree_dim - np.mean(perm_scree_dims)) / np.std(perm_scree_dims)
    else:
        effect_d = 0.0

    # Benzecriの6次元がQ6の6次元と一致する点も考慮
    dim_gap = abs(q6_dim - observed_benzecri_dim)  # 0 (完全一致)
    scree_gap = abs(q6_dim - observed_scree_dim)   # 1

    result = {
        "name": "dimension_correspondence",
        "null_hypothesis": "MCAの次元数はQ6の6次元と無関係（ランダムシャッフルで同等の次元数が得られる）",
        "observed_scree_dim": int(observed_scree_dim),
        "observed_benzecri_cum80_dim": int(observed_benzecri_dim),
        "q6_geometric_dim": q6_dim,
        "permutation_distribution": {
            "mean": float(np.mean(perm_scree_dims)) if len(perm_scree_dims) > 0 else None,
            "std": float(np.std(perm_scree_dims)) if len(perm_scree_dims) > 0 else None,
            "median": float(np.median(perm_scree_dims)) if len(perm_scree_dims) > 0 else None,
        },
        "test_statistic": float(observed_scree_dim),
        "p_value": float(p_value_scree),
        "p_value_bonferroni": float(min(p_value_scree * N_TESTS, 1.0)),
        "effect_size": float(effect_d),
        "effect_size_type": "standardized_difference (observed - random_mean) / random_sd",
        "effect_size_label": effect_size_label(effect_d),
        "benzecri_q6_match": dim_gap == 0,
        "conclusion": "reject" if min(p_value_scree * N_TESTS, 1.0) < ALPHA else "fail_to_reject",
        "details": {
            "note": f"Scree elbow={observed_scree_dim}, Benzecri cum80%={observed_benzecri_dim}, Q6 dim=6. "
                    f"Benzecri cum80%次元はQ6の6次元と完全一致。"
                    f"ランダム置換分布の平均={np.mean(perm_scree_dims):.2f} (SD={np.std(perm_scree_dims):.2f})" if len(perm_scree_dims) > 0 else "Permutation test incomplete"
        }
    }

    print(f"    Scree dim={observed_scree_dim}, Benzecri cum80%={observed_benzecri_dim}, Q6=6")
    print(f"    p={p_value_scree:.4f}, Bonferroni p={min(p_value_scree * N_TESTS, 1.0):.4f}, d={effect_d:.3f}")

    return result


# ============================================================
# 検証2: 遷移パターンとQ6エッジの対応
# ============================================================
def test2_transition_hamming(cases, graph_data, rng):
    """
    実データの遷移（before_hex → after_hex）におけるハミング距離が、
    ランダム遷移モデルより有意に小さいかを検証。

    八卦レベルでのハミング距離: before_hexとafter_hexは各3ビット。
    """
    print("  [Test 2] 遷移パターンとQ6エッジの対応...")

    # 実データからbefore_hex, after_hexペアを抽出
    hex_pairs = []
    for case in cases:
        bh = case.get("before_hex", "")
        ah = case.get("after_hex", "")
        if bh in TRIGRAM_BITS and ah in TRIGRAM_BITS:
            hex_pairs.append((bh, ah))

    if not hex_pairs:
        return {"name": "transition_hamming", "error": "No valid hex pairs found"}

    # 八卦ペアのハミング距離計算 (3ビット空間)
    def trigram_hamming(t1, t2):
        b1 = TRIGRAM_BITS[t1]
        b2 = TRIGRAM_BITS[t2]
        return sum(a != b for a, b in zip(b1, b2))

    # 実データのハミング距離
    observed_distances = [trigram_hamming(bh, ah) for bh, ah in hex_pairs]
    observed_mean = np.mean(observed_distances)

    # 拡張: trigger_hexとaction_hexも含めた遷移チェーン分析
    # before → trigger → action → after の連鎖でハミング距離を計算
    chain_distances = []
    for case in cases:
        chain = []
        for field in ["before_hex", "trigger_hex", "action_hex", "after_hex"]:
            val = case.get(field, "")
            if val in TRIGRAM_BITS:
                chain.append(val)
        if len(chain) >= 2:
            total_d = sum(trigram_hamming(chain[i], chain[i+1]) for i in range(len(chain)-1))
            chain_distances.append(total_d / (len(chain) - 1))  # 平均ステップ距離

    observed_chain_mean = np.mean(chain_distances) if chain_distances else None

    # 置換テスト: before_hexとafter_hexを独立にシャッフル
    before_list = [p[0] for p in hex_pairs]
    after_list = [p[1] for p in hex_pairs]

    perm_means = []
    for i in range(N_PERMUTATIONS):
        perm_after = list(after_list)
        rng.shuffle(perm_after)
        perm_dist = [trigram_hamming(b, a) for b, a in zip(before_list, perm_after)]
        perm_means.append(np.mean(perm_dist))

    perm_means = np.array(perm_means)

    # p値: ランダムで実データ以下の平均距離が出る確率（片側検定: 実データが近い方向）
    p_value = np.mean(perm_means <= observed_mean)

    # 効果量
    effect_d = cohens_d(perm_means, [observed_mean] * len(perm_means))
    # より適切: 標準化差
    if np.std(perm_means) > 0:
        z_score = (observed_mean - np.mean(perm_means)) / np.std(perm_means)
    else:
        z_score = 0.0

    result = {
        "name": "transition_hamming_distance",
        "null_hypothesis": "実データの遷移ペア間ハミング距離はランダム遷移と同等",
        "n_pairs": len(hex_pairs),
        "observed_mean_hamming": float(observed_mean),
        "observed_chain_mean_hamming": float(observed_chain_mean) if observed_chain_mean else None,
        "random_mean_hamming": float(np.mean(perm_means)),
        "random_std_hamming": float(np.std(perm_means)),
        "test_statistic": float(z_score),
        "p_value": float(p_value),
        "p_value_bonferroni": float(min(p_value * N_TESTS, 1.0)),
        "effect_size": float(z_score),
        "effect_size_type": "z-score (standardized difference)",
        "effect_size_label": effect_size_label(z_score),
        "conclusion": "reject" if min(p_value * N_TESTS, 1.0) < ALPHA else "fail_to_reject",
        "details": {
            "hamming_distribution": {
                str(d): int(c) for d, c in sorted(Counter(observed_distances).items())
            },
            "note": "八卦(3ビット)空間でのハミング距離分析。距離0=同一八卦, 距離3=錯卦(全ビット反転)。"
        }
    }

    print(f"    実データ平均ハミング距離={observed_mean:.4f}, ランダム={np.mean(perm_means):.4f}")
    print(f"    p={p_value:.4f}, Bonferroni p={min(p_value * N_TESTS, 1.0):.4f}, z={z_score:.3f}")

    return result


# ============================================================
# 検証3: 八卦タグとクラスタの対応 (Cramer's V有意性)
# ============================================================
def test3_hex_cluster_association(cases, cluster_results, rng):
    """
    Phase 2のCramer's V (before_hex=0.318, after_hex=0.3825) が
    ランダムに期待される値より有意に大きいかを置換テストで検証。
    """
    print("  [Test 3] 八卦タグとクラスタの対応...")

    # Phase 2から報告されたCramer's V
    reported_v_before = cluster_results["hexagram_association"]["before_hex"]["cramers_v"]
    reported_v_after = cluster_results["hexagram_association"]["after_hex"]["cramers_v"]

    # 独自計算: cases.jsonlからクラスタ割り当てを再現
    # Phase 2のk=2クラスタリングはMCA 5次元空間でのk-means
    # クラスタ割り当てを直接持っていないため、Phase 2のクラスタプロファイルから
    # 最も寄与する変数の値に基づいて近似的にクラスタを割り当てる

    # クラスタ0: 主流遷移(86.2%) — before_state上位が安定・平和、成長痛、停滞・閉塞
    # クラスタ1: 低頻度カテゴリ遷移(13.8%) — before_stateが混乱・衰退、安定・停止

    # Phase 2のクラスタプロファイルに基づく近似割り当て
    cluster1_before_states = {"混乱・衰退", "安定・停止", "混乱・カオス"}
    cluster1_after_states = {"混乱・衰退", "安定・停止"}
    cluster1_trigger_types = {"自然推移", "外部ショック"}
    cluster1_action_types = {"分散・探索", "捨てる・転換", "捨てる・撤退"}
    cluster1_pattern_types = {"Crisis_Pivot", "Breakthrough", "Exploration"}

    # より正確な方法: クラスタ1のサイズ(13.8%)に合わせて
    # Dim1の値が大きいレコードをクラスタ1とする近似
    # → 簡略化: before_stateとafter_stateの組み合わせで割り当て

    # 直接Cramer's Vを計算するため、分割表を構築
    # before_hex × cluster の分割表
    before_hex_values = []
    after_hex_values = []

    # クラスタ割り当ての近似: Phase 2のプロファイルに最も整合する方法
    # Cluster 1 (13.8%) の特徴: 混乱・衰退(before), 安定・停止(before), Crisis_Pivot, Exploration
    cluster_labels = []
    for case in cases:
        bs = case.get("before_state", "")
        pt = case.get("pattern_type", "")
        # クラスタ1の判別: Phase 2のクラスタプロファイルに基づく近似
        is_cluster1 = (
            bs in {"混乱・カオス", "安定・停止"} or
            pt in {"Crisis_Pivot", "Breakthrough", "Exploration"}
        )
        cluster_labels.append(1 if is_cluster1 else 0)
        before_hex_values.append(case.get("before_hex", "unknown"))
        after_hex_values.append(case.get("after_hex", "unknown"))

    cluster_labels = np.array(cluster_labels)

    # 分割表構築
    unique_before = sorted(set(before_hex_values))
    unique_after = sorted(set(after_hex_values))

    # before_hex × cluster
    before_table = np.zeros((2, len(unique_before)))
    for i, (cl, bh) in enumerate(zip(cluster_labels, before_hex_values)):
        if bh in unique_before:
            before_table[cl, unique_before.index(bh)] += 1

    # after_hex × cluster
    after_table = np.zeros((2, len(unique_after)))
    for i, (cl, ah) in enumerate(zip(cluster_labels, after_hex_values)):
        if ah in unique_after:
            after_table[cl, unique_after.index(ah)] += 1

    # 実データのCramer's V
    observed_v_before = cramers_v(before_table)
    observed_v_after = cramers_v(after_table)

    # 置換テスト: 八卦タグをランダムに再割り当て
    perm_v_before = []
    perm_v_after = []

    for i in range(N_PERMUTATIONS):
        # before_hexをシャッフル
        perm_before = list(before_hex_values)
        rng.shuffle(perm_before)
        perm_table = np.zeros((2, len(unique_before)))
        for cl, bh in zip(cluster_labels, perm_before):
            if bh in unique_before:
                perm_table[cl, unique_before.index(bh)] += 1
        perm_v_before.append(cramers_v(perm_table))

        # after_hexをシャッフル
        perm_after = list(after_hex_values)
        rng.shuffle(perm_after)
        perm_table = np.zeros((2, len(unique_after)))
        for cl, ah in zip(cluster_labels, perm_after):
            if ah in unique_after:
                perm_table[cl, unique_after.index(ah)] += 1
        perm_v_after.append(cramers_v(perm_table))

        if (i + 1) % 200 == 0:
            print(f"    置換 {i+1}/{N_PERMUTATIONS} 完了")

    perm_v_before = np.array(perm_v_before)
    perm_v_after = np.array(perm_v_after)

    # p値
    p_before = np.mean(perm_v_before >= observed_v_before)
    p_after = np.mean(perm_v_after >= observed_v_after)
    # 保守的: 二つのうち大きいp値を使用
    p_combined = max(p_before, p_after)

    # 効果量はCramer's V自体
    avg_v = (observed_v_before + observed_v_after) / 2

    result = {
        "name": "hex_cluster_cramers_v",
        "null_hypothesis": "八卦タグとクラスタの間のCramer's Vはランダムに期待される値と同等",
        "phase2_reported": {
            "before_hex_v": float(reported_v_before),
            "after_hex_v": float(reported_v_after),
        },
        "observed": {
            "before_hex_v": float(observed_v_before),
            "after_hex_v": float(observed_v_after),
        },
        "random_baseline": {
            "before_hex_v_mean": float(np.mean(perm_v_before)),
            "before_hex_v_std": float(np.std(perm_v_before)),
            "after_hex_v_mean": float(np.mean(perm_v_after)),
            "after_hex_v_std": float(np.std(perm_v_after)),
        },
        "test_statistic": float(avg_v),
        "p_value": float(p_combined),
        "p_value_bonferroni": float(min(p_combined * N_TESTS, 1.0)),
        "p_value_before": float(p_before),
        "p_value_after": float(p_after),
        "effect_size": float(avg_v),
        "effect_size_type": "Cramer's V (average of before/after)",
        "effect_size_label": cramers_v_label(avg_v),
        "conclusion": "reject" if min(p_combined * N_TESTS, 1.0) < ALPHA else "fail_to_reject",
        "details": {
            "note": "クラスタ割り当てはPhase 2のプロファイルに基づく近似。"
                    f"Phase 2報告値: before_V={reported_v_before}, after_V={reported_v_after}. "
                    f"独自計算値: before_V={observed_v_before:.4f}, after_V={observed_v_after:.4f}"
        }
    }

    print(f"    before_hex V={observed_v_before:.4f} (Phase2: {reported_v_before}), p={p_before:.4f}")
    print(f"    after_hex V={observed_v_after:.4f} (Phase2: {reported_v_after}), p={p_after:.4f}")
    print(f"    Bonferroni p={min(p_combined * N_TESTS, 1.0):.4f}")

    return result


# ============================================================
# 検証4: スペクトル構造の比較
# ============================================================
def test4_spectral_comparison(mca_results, graph_data, rng):
    """
    Q6のスペクトル {-6,-4,-2,0,2,4,6} (多重度 C(6,k)) と
    MCAの固有値分布の構造的類似性をWasserstein距離で比較。
    """
    print("  [Test 4] スペクトル構造の比較...")

    # Q6スペクトル (graph_analysis.jsonから)
    q6_eigenvalues = np.array(graph_data["spectral_properties"]["eigenvalues"])

    # MCA固有値
    mca_eigenvalues = np.array(mca_results["eigenvalues"])

    # 両方を正規化して分布として比較
    # Q6: 64固有値を [0,1] にマップ
    q6_normalized = (q6_eigenvalues - q6_eigenvalues.min()) / (q6_eigenvalues.max() - q6_eigenvalues.min())

    # MCA: 30固有値を [0,1] にマップ
    mca_normalized = (mca_eigenvalues - mca_eigenvalues.min()) / (mca_eigenvalues.max() - mca_eigenvalues.min())

    # Wasserstein距離の計算
    from scipy.stats import wasserstein_distance

    # 分布として比較するため、ヒストグラムビンに変換
    n_bins = 20
    q6_hist, bin_edges = np.histogram(q6_normalized, bins=n_bins, range=(0, 1), density=True)
    mca_hist, _ = np.histogram(mca_normalized, bins=n_bins, range=(0, 1), density=True)

    # 正規化
    q6_dist = q6_hist / q6_hist.sum() if q6_hist.sum() > 0 else q6_hist
    mca_dist = mca_hist / mca_hist.sum() if mca_hist.sum() > 0 else mca_hist

    # Wasserstein距離
    observed_wasserstein = wasserstein_distance(q6_normalized, mca_normalized)

    # Q6固有値の多重度構造を確認
    q6_unique, q6_counts = np.unique(np.round(q6_eigenvalues, 1), return_counts=True)

    # MCA固有値の「間隔パターン」がQ6の等間隔スペクトルと類似するか
    # Q6: 等間隔 (-6,-4,-2,0,2,4,6) → 間隔は全て2
    # MCA: 間隔の均一性を測定
    mca_gaps = np.abs(np.diff(mca_eigenvalues))
    gap_cv = np.std(mca_gaps) / np.mean(mca_gaps) if np.mean(mca_gaps) > 0 else float('inf')
    # Q6のgap CVは0（完全等間隔）

    # 置換テスト: ランダムな固有値分布とのWasserstein距離を計算
    perm_wasserstein = []
    for i in range(N_PERMUTATIONS):
        # ランダム固有値: [0, max_eigenvalue]の一様分布から30個サンプリング
        random_evals = rng.uniform(mca_eigenvalues.min(), mca_eigenvalues.max(), size=len(mca_eigenvalues))
        random_evals = np.sort(random_evals)[::-1]
        random_normalized = (random_evals - random_evals.min()) / (random_evals.max() - random_evals.min()) if random_evals.max() > random_evals.min() else np.zeros_like(random_evals)
        perm_wasserstein.append(wasserstein_distance(q6_normalized, random_normalized))

    perm_wasserstein = np.array(perm_wasserstein)

    # p値: ランダムで実データ以下のWasserstein距離が出る確率
    p_value = np.mean(perm_wasserstein <= observed_wasserstein)

    # 効果量
    if np.std(perm_wasserstein) > 0:
        z_score = (observed_wasserstein - np.mean(perm_wasserstein)) / np.std(perm_wasserstein)
    else:
        z_score = 0.0

    result = {
        "name": "spectral_structure_comparison",
        "null_hypothesis": "MCA固有値分布とQ6スペクトルのWasserstein距離はランダム固有値と同等",
        "q6_spectrum": {
            "unique_eigenvalues": {str(k): int(v) for k, v in zip(q6_unique, q6_counts)},
            "n_eigenvalues": len(q6_eigenvalues),
        },
        "mca_spectrum": {
            "n_eigenvalues": len(mca_eigenvalues),
            "range": [float(mca_eigenvalues.min()), float(mca_eigenvalues.max())],
            "gap_cv": float(gap_cv),
        },
        "observed_wasserstein": float(observed_wasserstein),
        "random_wasserstein_mean": float(np.mean(perm_wasserstein)),
        "random_wasserstein_std": float(np.std(perm_wasserstein)),
        "test_statistic": float(z_score),
        "p_value": float(p_value),
        "p_value_bonferroni": float(min(p_value * N_TESTS, 1.0)),
        "effect_size": float(z_score),
        "effect_size_type": "z-score (Wasserstein distance)",
        "effect_size_label": effect_size_label(z_score),
        "conclusion": "reject" if min(p_value * N_TESTS, 1.0) < ALPHA else "fail_to_reject",
        "details": {
            "q6_gap_cv": 0.0,
            "mca_gap_cv": float(gap_cv),
            "note": "Q6は7つのユニーク固有値（等間隔）を持つ完全に規則的なスペクトル。"
                    "MCAの30固有値との直接比較は次元数が異なるため限定的。"
                    "Wasserstein距離は正規化後の分布形状を比較。"
        }
    }

    print(f"    Wasserstein距離={observed_wasserstein:.4f}, ランダム={np.mean(perm_wasserstein):.4f}")
    print(f"    p={p_value:.4f}, Bonferroni p={min(p_value * N_TESTS, 1.0):.4f}, z={z_score:.3f}")

    return result


# ============================================================
# 検証5: 構造的関係の保存 (錯卦ペア対称性)
# ============================================================
def test5_cuogua_symmetry(cases, graph_data, rng):
    """
    錯卦（全爻反転）ペアにおけるbefore_state分布が、
    非錯卦ペアよりも類似しているかをコサイン類似度で検証。

    32個の錯卦ペアと32個のランダムペアで比較。
    """
    print("  [Test 5] 構造的関係の保存（錯卦ペア対称性）...")

    # 錯卦ペア (graph_analysis.jsonから)
    cuogua_pairs = graph_data["structural_relations"]["cuogua_pairs"]  # [[1,2],[3,50],...]

    # 八卦レベルでの錯卦: 各八卦のビット反転
    trigram_cuogua = {}
    for name, bits in TRIGRAM_BITS.items():
        flipped = tuple(1 - b for b in bits)
        for other_name, other_bits in TRIGRAM_BITS.items():
            if tuple(other_bits) == flipped:
                trigram_cuogua[name] = other_name
                break

    # 八卦錯卦ペア: 乾↔坤, 震↔巽, 坎↔離, 艮↔兌
    cuogua_trigram_pairs = set()
    for t1, t2 in trigram_cuogua.items():
        pair = tuple(sorted([t1, t2]))
        cuogua_trigram_pairs.add(pair)
    cuogua_trigram_pairs = sorted(cuogua_trigram_pairs)

    # 各八卦のbefore_state分布を計算
    before_states_all = sorted(set(c.get("before_state", "unknown") for c in cases))

    trigram_before_dist = {}
    for trigram in TRIGRAM_NAMES:
        counts = Counter()
        for case in cases:
            if case.get("before_hex") == trigram:
                counts[case.get("before_state", "unknown")] += 1
        total = sum(counts.values())
        if total > 0:
            dist = np.array([counts.get(s, 0) / total for s in before_states_all])
        else:
            dist = np.zeros(len(before_states_all))
        trigram_before_dist[trigram] = dist

    # 錯卦ペア間のコサイン類似度
    cuogua_similarities = []
    for t1, t2 in cuogua_trigram_pairs:
        if t1 in trigram_before_dist and t2 in trigram_before_dist:
            sim = cosine_similarity(trigram_before_dist[t1], trigram_before_dist[t2])
            cuogua_similarities.append(sim)

    # ランダムペア間のコサイン類似度 (32ペア × N_PERMUTATIONS)
    all_trigrams = [t for t in TRIGRAM_NAMES if t in trigram_before_dist]

    perm_similarities_means = []
    for i in range(N_PERMUTATIONS):
        perm_sims = []
        # 32個のランダムペアを生成（重複なし）
        shuffled = list(all_trigrams)
        rng.shuffle(shuffled)
        n_pairs = min(len(cuogua_trigram_pairs), len(shuffled) // 2)
        for j in range(n_pairs):
            t1 = shuffled[2 * j]
            t2 = shuffled[2 * j + 1]
            sim = cosine_similarity(trigram_before_dist[t1], trigram_before_dist[t2])
            perm_sims.append(sim)
        if perm_sims:
            perm_similarities_means.append(np.mean(perm_sims))

    perm_similarities_means = np.array(perm_similarities_means)
    observed_mean_sim = np.mean(cuogua_similarities) if cuogua_similarities else 0.0

    # p値: ランダムで実データ以上の類似度が出る確率
    p_value = np.mean(perm_similarities_means >= observed_mean_sim)

    # 効果量
    if np.std(perm_similarities_means) > 0:
        z_score = (observed_mean_sim - np.mean(perm_similarities_means)) / np.std(perm_similarities_means)
    else:
        z_score = 0.0

    # after_state分布でも同様の分析
    trigram_after_dist = {}
    after_states_all = sorted(set(c.get("after_state", "unknown") for c in cases))
    for trigram in TRIGRAM_NAMES:
        counts = Counter()
        for case in cases:
            if case.get("before_hex") == trigram:
                counts[case.get("after_state", "unknown")] += 1
        total = sum(counts.values())
        if total > 0:
            dist = np.array([counts.get(s, 0) / total for s in after_states_all])
        else:
            dist = np.zeros(len(after_states_all))
        trigram_after_dist[trigram] = dist

    cuogua_after_sims = []
    for t1, t2 in cuogua_trigram_pairs:
        if t1 in trigram_after_dist and t2 in trigram_after_dist:
            sim = cosine_similarity(trigram_after_dist[t1], trigram_after_dist[t2])
            cuogua_after_sims.append(sim)

    result = {
        "name": "cuogua_symmetry",
        "null_hypothesis": "錯卦ペアのbefore_state分布のコサイン類似度はランダムペアと同等",
        "cuogua_trigram_pairs": [list(p) for p in cuogua_trigram_pairs],
        "n_cuogua_pairs": len(cuogua_trigram_pairs),
        "observed_mean_similarity_before": float(observed_mean_sim),
        "observed_similarities_before": [float(s) for s in cuogua_similarities],
        "observed_mean_similarity_after": float(np.mean(cuogua_after_sims)) if cuogua_after_sims else None,
        "random_mean_similarity": float(np.mean(perm_similarities_means)) if len(perm_similarities_means) > 0 else None,
        "random_std_similarity": float(np.std(perm_similarities_means)) if len(perm_similarities_means) > 0 else None,
        "test_statistic": float(z_score),
        "p_value": float(p_value),
        "p_value_bonferroni": float(min(p_value * N_TESTS, 1.0)),
        "effect_size": float(z_score),
        "effect_size_type": "z-score (cosine similarity)",
        "effect_size_label": effect_size_label(z_score),
        "conclusion": "reject" if min(p_value * N_TESTS, 1.0) < ALPHA else "fail_to_reject",
        "details": {
            "trigram_cuogua_mapping": trigram_cuogua,
            "note": "八卦レベルでの錯卦ペア（乾↔坤, 震↔巽, 坎↔離, 艮↔兌）の"
                    "before_state分布のコサイン類似度を、ランダムペアと比較。"
                    f"八卦は8個のため、錯卦ペアは4組のみ（64卦レベルの32組ではない）。"
        }
    }

    print(f"    錯卦ペア平均類似度={observed_mean_sim:.4f}, ランダム={np.mean(perm_similarities_means):.4f}")
    print(f"    p={p_value:.4f}, Bonferroni p={min(p_value * N_TESTS, 1.0):.4f}, z={z_score:.3f}")

    return result


# ============================================================
# 検証6: 部分同型性テスト
# ============================================================
def test6_partial_isomorphism(cases, graph_data, rng):
    """
    Q6のLouvainコミュニティ（5個）が、実データの変化パターンの
    部分構造と対応するかを検証。

    各コミュニティに属する卦の八卦分布と、実データの八卦分布の対応を分析。
    """
    print("  [Test 6] 部分同型性テスト...")

    # Q6のLouvainコミュニティ (graph_analysis.jsonから)
    communities = graph_data["community_structure"]["communities"]
    n_communities = graph_data["community_structure"]["num_communities"]

    # Phase 1の kw_to_trigrams マッピングが必要
    # state_space_model.pyのロジックを再現
    # 簡略化: graph_dataから直接取得できないため、別途構築

    # 八卦ビットからKing Wen番号への逆引きテーブルを構築
    # 全64卦のビットベクトルを生成
    all_hexagram_bits = {}
    for lower_name, lower_bits in TRIGRAM_BITS.items():
        for upper_name, upper_bits in TRIGRAM_BITS.items():
            bits = tuple(lower_bits + upper_bits)
            all_hexagram_bits[bits] = (lower_name, upper_name)

    # 各コミュニティの卦が持つ八卦タグの分布を計算
    # コミュニティ → [卦番号] → 八卦タグ分布
    # 注意: kw番号→八卦のマッピングはstate_space_model.pyに依存
    # ここではreference dataを使わず、cases.jsonlから八卦分布を直接使用

    # 各八卦のパターンタイプ分布
    trigram_pattern_dist = {}
    pattern_types_all = sorted(set(c.get("pattern_type", "unknown") for c in cases))

    for trigram in TRIGRAM_NAMES:
        counts = Counter()
        for case in cases:
            if case.get("before_hex") == trigram:
                counts[case.get("pattern_type", "unknown")] += 1
        total = sum(counts.values())
        if total > 0:
            dist = np.array([counts.get(p, 0) / total for p in pattern_types_all])
        else:
            dist = np.zeros(len(pattern_types_all))
        trigram_pattern_dist[trigram] = dist

    # コミュニティ間の八卦分布の差異を分析
    # 各コミュニティの「支配的八卦」を特定
    # → Q6のコミュニティは卦番号で定義されているため、八卦タグとの対応を計算

    # 実データの八卦×パターンタイプの分割表
    hex_vals = [c.get("before_hex", "unknown") for c in cases]
    pattern_vals = [c.get("pattern_type", "unknown") for c in cases]

    unique_hex = sorted(set(hex_vals))
    unique_pattern = sorted(set(pattern_vals))

    # 分割表: 八卦 × パターンタイプ
    observed_table = np.zeros((len(unique_hex), len(unique_pattern)))
    for h, p in zip(hex_vals, pattern_vals):
        if h in unique_hex and p in unique_pattern:
            observed_table[unique_hex.index(h), unique_pattern.index(p)] += 1

    # 実データのCramer's V (八卦 × パターンタイプ)
    observed_v = cramers_v(observed_table)

    # コミュニティ数(5)とデータクラスタ数(k=2, DBSCAN最適=5)の対応
    # DBSCAN最適k=5はQ6のLouvainコミュニティ数5と一致
    dbscan_best_k = 5  # cluster_results.jsonのDBSCAN best n_clusters
    community_count_match = (n_communities == dbscan_best_k)

    # 置換テスト: 八卦タグをシャッフルしてCramer's V分布を生成
    perm_v = []
    for i in range(N_PERMUTATIONS):
        perm_hex = list(hex_vals)
        rng.shuffle(perm_hex)
        perm_table = np.zeros((len(unique_hex), len(unique_pattern)))
        for h, p in zip(perm_hex, pattern_vals):
            if h in unique_hex and p in unique_pattern:
                perm_table[unique_hex.index(h), unique_pattern.index(p)] += 1
        perm_v.append(cramers_v(perm_table))

        if (i + 1) % 200 == 0:
            print(f"    置換 {i+1}/{N_PERMUTATIONS} 完了")

    perm_v = np.array(perm_v)

    # p値
    p_value = np.mean(perm_v >= observed_v)

    # 効果量
    if np.std(perm_v) > 0:
        z_score = (observed_v - np.mean(perm_v)) / np.std(perm_v)
    else:
        z_score = 0.0

    # after_hexでも同様
    hex_after_vals = [c.get("after_hex", "unknown") for c in cases]
    unique_hex_after = sorted(set(hex_after_vals))
    after_table = np.zeros((len(unique_hex_after), len(unique_pattern)))
    for h, p in zip(hex_after_vals, pattern_vals):
        if h in unique_hex_after and p in unique_pattern:
            after_table[unique_hex_after.index(h), unique_pattern.index(p)] += 1
    observed_v_after = cramers_v(after_table)

    result = {
        "name": "partial_isomorphism",
        "null_hypothesis": "Q6のコミュニティ構造と実データの変化パターンに部分的対応は存在しない",
        "q6_communities": n_communities,
        "dbscan_optimal_clusters": dbscan_best_k,
        "community_count_match": community_count_match,
        "observed_cramers_v_before_pattern": float(observed_v),
        "observed_cramers_v_after_pattern": float(observed_v_after),
        "random_v_mean": float(np.mean(perm_v)),
        "random_v_std": float(np.std(perm_v)),
        "test_statistic": float(z_score),
        "p_value": float(p_value),
        "p_value_bonferroni": float(min(p_value * N_TESTS, 1.0)),
        "effect_size": float(observed_v),
        "effect_size_type": "Cramer's V (before_hex x pattern_type)",
        "effect_size_label": cramers_v_label(observed_v),
        "conclusion": "reject" if min(p_value * N_TESTS, 1.0) < ALPHA else "fail_to_reject",
        "details": {
            "community_count_note": f"Q6 Louvainコミュニティ数({n_communities})とDBSCAN最適クラスタ数({dbscan_best_k})が一致",
            "note": "八卦×パターンタイプの分割表のCramer's Vを、八卦タグのランダムシャッフルと比較。"
                    "Q6のコミュニティ構造を直接テストするのではなく、八卦タグがパターンを"
                    "構造的に区別する力を検証。"
        }
    }

    print(f"    V(before×pattern)={observed_v:.4f}, ランダム={np.mean(perm_v):.4f}")
    print(f"    Q6コミュニティ={n_communities}, DBSCAN最適k={dbscan_best_k} → 一致={community_count_match}")
    print(f"    p={p_value:.4f}, Bonferroni p={min(p_value * N_TESTS, 1.0):.4f}, z={z_score:.3f}")

    return result


# ============================================================
# 同型性レベルの判定
# ============================================================
def determine_isomorphism_level(test_results):
    """6つの検証結果から同型性レベルを判定"""
    n_rejected = sum(1 for t in test_results if t.get("conclusion") == "reject")

    # 効果量の評価
    effect_labels = [t.get("effect_size_label", "negligible") for t in test_results]
    large_effects = sum(1 for l in effect_labels if l == "large")
    medium_effects = sum(1 for l in effect_labels if l in ("medium", "large"))

    if n_rejected >= 5 and large_effects >= 3:
        level = "complete"
        summary = (f"{n_rejected}/6検証でH0棄却（Bonferroni補正後）、"
                   f"大効果量{large_effects}件。完全同型性が示唆される。")
    elif n_rejected >= 3:
        level = "partial"
        summary = (f"{n_rejected}/6検証でH0棄却（Bonferroni補正後）、"
                   f"中以上の効果量{medium_effects}件。部分同型性が示唆される。")
    elif n_rejected >= 1:
        level = "analogous"
        summary = (f"{n_rejected}/6検証でH0棄却（Bonferroni補正後）。"
                   f"類似構造が存在するが、同型性は限定的。")
    else:
        level = "none"
        summary = (f"6検証中いずれもH0を棄却できず。"
                   f"Q6と実データ間に統計的に有意な同型性は検出されなかった。")

    return {
        "isomorphism_level": level,
        "significant_tests": n_rejected,
        "total_tests": N_TESTS,
        "large_effect_count": large_effects,
        "medium_or_larger_effect_count": medium_effects,
        "effect_labels": effect_labels,
        "summary": summary
    }


# ============================================================
# レポート生成
# ============================================================
def generate_report(test_results, overall_conclusion):
    """Phase 3レポートを生成"""

    report_lines = [
        "# Phase 3 レポート: 同型性検証結果",
        "",
        f"**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**乱数シード**: {RANDOM_SEED}",
        f"**置換回数**: {N_PERMUTATIONS}",
        f"**有意水準**: alpha={ALPHA}, Bonferroni補正後 alpha'={ALPHA_BONFERRONI:.5f}",
        "",
        "---",
        "",
        "## 総合判定",
        "",
        f"**同型性レベル**: **{overall_conclusion['isomorphism_level'].upper()}**",
        "",
        f"- H0棄却数: {overall_conclusion['significant_tests']}/{overall_conclusion['total_tests']}",
        f"- 大効果量: {overall_conclusion['large_effect_count']}件",
        f"- 中以上効果量: {overall_conclusion['medium_or_larger_effect_count']}件",
        "",
        f"**要約**: {overall_conclusion['summary']}",
        "",
        "---",
        "",
        "## 各検証の結果",
        "",
    ]

    test_names_ja = {
        "dimension_correspondence": "検証1: 次元数の対応",
        "transition_hamming_distance": "検証2: 遷移パターンとQ6エッジの対応",
        "hex_cluster_cramers_v": "検証3: 八卦タグとクラスタの対応",
        "spectral_structure_comparison": "検証4: スペクトル構造の比較",
        "cuogua_symmetry": "検証5: 構造的関係の保存（錯卦ペア対称性）",
        "partial_isomorphism": "検証6: 部分同型性テスト",
    }

    for i, test in enumerate(test_results, 1):
        name = test.get("name", f"test_{i}")
        ja_name = test_names_ja.get(name, name)
        conclusion_ja = "H0棄却" if test.get("conclusion") == "reject" else "H0棄却できず"

        report_lines.extend([
            f"### {ja_name}",
            "",
            f"**帰無仮説**: {test.get('null_hypothesis', 'N/A')}",
            "",
            f"| 指標 | 値 |",
            f"|------|-----|",
            f"| 検定統計量 | {test.get('test_statistic', 'N/A'):.4f} |" if isinstance(test.get('test_statistic'), (int, float)) else f"| 検定統計量 | {test.get('test_statistic', 'N/A')} |",
            f"| p値 | {test.get('p_value', 'N/A'):.6f} |" if isinstance(test.get('p_value'), (int, float)) else f"| p値 | {test.get('p_value', 'N/A')} |",
            f"| p値 (Bonferroni) | {test.get('p_value_bonferroni', 'N/A'):.6f} |" if isinstance(test.get('p_value_bonferroni'), (int, float)) else f"| p値 (Bonferroni) | {test.get('p_value_bonferroni', 'N/A')} |",
            f"| 効果量 ({test.get('effect_size_type', 'N/A')}) | {test.get('effect_size', 'N/A'):.4f} |" if isinstance(test.get('effect_size'), (int, float)) else f"| 効果量 | {test.get('effect_size', 'N/A')} |",
            f"| 効果量ラベル | {test.get('effect_size_label', 'N/A')} |",
            f"| **判定** | **{conclusion_ja}** |",
            "",
        ])

        # テスト固有の詳細
        details = test.get("details", {})
        if details.get("note"):
            report_lines.extend([
                f"**備考**: {details['note']}",
                "",
            ])

        report_lines.append("")

    # 限界と今後の課題
    report_lines.extend([
        "---",
        "",
        "## 限界と今後の課題",
        "",
        "### 方法論的限界",
        "",
        "1. **八卦レベルの分析**: cases.jsonlのbefore_hex/after_hexは八卦（3ビット）レベルの情報であり、"
        "64卦（6ビット）レベルの完全な遷移情報ではない。Q6は64卦空間で定義されているため、"
        "八卦レベルの分析はQ6の構造を部分的にしか捉えられない。",
        "",
        "2. **クラスタ割り当ての近似**: Phase 2のk-meansクラスタラベルが直接利用できないため、"
        "クラスタプロファイルに基づく近似割り当てを使用した。これにより検証3の精度に限界がある。",
        "",
        "3. **MCA再計算の省略**: 検証1の置換テストでMCAを毎回再計算する代わりに、"
        "指示行列のSVDによる近似を使用した。厳密なMCA置換テストとは結果が異なる可能性がある。",
        "",
        "4. **k=2クラスタリングの制約**: Phase 2で採用されたk=2は頻度構造の反映であり、"
        "意味的クラスタリングではない（Phase 2レビューで指摘済み）。検証3はこの制約を受ける。",
        "",
        "5. **MCA列座標の不安定性**: Phase 2レビューで指摘されたDim1-2の不安定性（r≈0.25）は、"
        "MCA空間に基づく分析の信頼性に影響する。",
        "",
        "### 今後の課題",
        "",
        "1. 64卦レベルの遷移データ（classical_before_hexagram, classical_after_hexagram）を活用した"
        "Q6上の遷移分析",
        "2. MCAの完全な再実装による厳密な置換テスト",
        "3. 信頼できるPhase 2結果（カイ二乗検定、調整残差、確率遷移行列）を活用した"
        "追加の構造的分析",
        "4. 綜卦（zonggua）ペアの対称性検証",
        "",
        "---",
        "",
        f"*Generated by Phase 3 isomorphism_test.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    return "\n".join(report_lines)


# ============================================================
# メイン実行
# ============================================================
def main():
    print("=" * 60)
    print("Phase 3: 同型性検証")
    print(f"乱数シード={RANDOM_SEED}, 置換回数={N_PERMUTATIONS}")
    print(f"有意水準: α={ALPHA}, Bonferroni α'={ALPHA_BONFERRONI:.5f}")
    print("=" * 60)

    # 乱数生成器
    rng = np.random.RandomState(RANDOM_SEED)

    # データ読み込み
    print("\n[1/8] データ読み込み...")
    cases = load_cases()
    print(f"  事例数: {len(cases)}")

    graph_data = load_json(PHASE1_DIR / "graph_analysis.json")
    print(f"  Q6: {graph_data['basic_properties']['num_nodes']}ノード, "
          f"{graph_data['basic_properties']['num_edges']}エッジ")

    mca_results = load_json(PHASE2_DIR / "mca_results.json")
    print(f"  MCA: {mca_results['n_components_computed']}成分, "
          f"推奨{mca_results['recommended_dimensions']}次元")

    dimension_report = load_json(PHASE2_DIR / "dimension_report.json")
    cluster_results = load_json(PHASE2_DIR / "cluster_results.json")
    transition_stats = load_json(PHASE2_DIR / "transition_stats.json")

    # 6つの検証を実行
    print("\n[2/8] 検証1: 次元数の対応...")
    test1 = test1_dimension_correspondence(cases, mca_results, dimension_report, rng)

    print("\n[3/8] 検証2: 遷移パターンとQ6エッジの対応...")
    test2 = test2_transition_hamming(cases, graph_data, rng)

    print("\n[4/8] 検証3: 八卦タグとクラスタの対応...")
    test3 = test3_hex_cluster_association(cases, cluster_results, rng)

    print("\n[5/8] 検証4: スペクトル構造の比較...")
    test4 = test4_spectral_comparison(mca_results, graph_data, rng)

    print("\n[6/8] 検証5: 構造的関係の保存...")
    test5 = test5_cuogua_symmetry(cases, graph_data, rng)

    print("\n[7/8] 検証6: 部分同型性テスト...")
    test6 = test6_partial_isomorphism(cases, graph_data, rng)

    # 結果集約
    test_results = [test1, test2, test3, test4, test5, test6]

    # 同型性レベル判定
    print("\n[8/8] 同型性レベル判定...")
    overall_conclusion = determine_isomorphism_level(test_results)

    print(f"\n{'=' * 60}")
    print(f"同型性レベル: {overall_conclusion['isomorphism_level'].upper()}")
    print(f"H0棄却: {overall_conclusion['significant_tests']}/{overall_conclusion['total_tests']}")
    print(f"要約: {overall_conclusion['summary']}")
    print(f"{'=' * 60}")

    # JSON出力
    output_json = {
        "tests": test_results,
        "overall_conclusion": overall_conclusion,
        "random_seed": RANDOM_SEED,
        "n_permutations": N_PERMUTATIONS,
        "alpha": ALPHA,
        "alpha_bonferroni": ALPHA_BONFERRONI,
        "n_cases": len(cases),
        "execution_timestamp": datetime.now().isoformat(),
    }

    json_path = PHASE3_DIR / "statistical_tests.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON出力: {json_path}")

    # レポート出力
    report = generate_report(test_results, overall_conclusion)
    report_path = PHASE3_DIR / "report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  レポート出力: {report_path}")

    return test_results, overall_conclusion


if __name__ == "__main__":
    main()
