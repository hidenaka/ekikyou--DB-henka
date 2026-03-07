#!/usr/bin/env python3
"""
Phase 3: 同型性検証 v2.0 — Q6超立方体と観測遷移グラフ G_obs の構造的類似性テスト

「観測遷移グラフ G_obs（64卦の実遷移）と Q6（ハミング距離1の6正則グラフ）の構造的類似性」

5つの統計的検証:
  A. エッジ重複率（Edge Overlap）
  B. ハミング距離分布（6bit空間）
  C. ラプラシアンスペクトル類似性（Wasserstein距離）
  D. 錯卦対称性（64卦レベル）
  E. コミュニティ構造NMI

総合判定: Fisher結合p値
"""

import json
import sys
import time
import numpy as np
import networkx as nx
import community as community_louvain
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from scipy.stats import wasserstein_distance, chi2
from sklearn.metrics import normalized_mutual_info_score

# ---------- パス設定 ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PHASE1_DIR = BASE_DIR / "analysis" / "phase1"
PHASE3_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CASES_FILE = DATA_DIR / "raw" / "cases.jsonl"
REFERENCE_FILE = DATA_DIR / "reference" / "iching_texts_ctext_legge_ja.json"
GRAPH_ANALYSIS_FILE = PHASE1_DIR / "graph_analysis.json"

# ---------- 定数 ----------
RANDOM_SEED = 42
N_PERMUTATIONS = 1000
ALPHA = 0.05

# 八卦ビット定義（Phase 1と同一）
TRIGRAM_BITS = {
    '乾': (1, 1, 1), '兌': (1, 1, 0), '離': (1, 0, 1), '震': (1, 0, 0),
    '巽': (0, 1, 1), '坎': (0, 1, 0), '艮': (0, 0, 1), '坤': (0, 0, 0),
}

# 自然元素→八卦名
ELEMENT_TO_TRIGRAM = {
    '天': '乾', '地': '坤', '雷': '震', '風': '巽',
    '水': '坎', '火': '離', '山': '艮', '沢': '兌',
}

# 異体字マッピング（フォールバック用）
VARIANT_MAP = {
    '遁': '遯',   # 天山遁 → 天山遯
    '無妄': '无妄',  # 天雷無妄 → 天雷无妄
    '習坎': '坎為水',  # 習坎は坎為水の別名
}


# ============================================================
# ユーティリティ
# ============================================================

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_cases():
    cases = []
    with open(CASES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_name_to_kw(reference_data):
    """local_name → King Wen番号のマッピングを構築"""
    mapping = {}
    for kw_str, hdata in reference_data['hexagrams'].items():
        kw = int(kw_str)
        local_name = hdata['local_name']
        mapping[local_name] = kw
    return mapping


def build_kw_to_bits(reference_data):
    """King Wen番号 → 6bitベクトル(tuple) のマッピング"""
    kw_to_bits = {}
    bits_to_kw = {}

    for kw_str, hdata in reference_data['hexagrams'].items():
        kw = int(kw_str)
        local_name = hdata['local_name']
        upper, lower = _parse_trigrams(local_name)
        if upper is None or lower is None:
            print(f"  WARNING: 卦名解析失敗: kw={kw}, name={local_name}")
            continue
        bits = TRIGRAM_BITS[lower] + TRIGRAM_BITS[upper]  # 下卦3bit + 上卦3bit
        kw_to_bits[kw] = bits
        bits_to_kw[bits] = kw

    return kw_to_bits, bits_to_kw


def _parse_trigrams(local_name):
    """卦名から (上卦名, 下卦名) を返す"""
    if '為' in local_name:
        trigram_name = local_name[0]
        if trigram_name in TRIGRAM_BITS:
            return trigram_name, trigram_name
        element = local_name[2]
        trigram_name = ELEMENT_TO_TRIGRAM.get(element)
        if trigram_name:
            return trigram_name, trigram_name
    upper_elem = local_name[0]
    lower_elem = local_name[1]
    upper = ELEMENT_TO_TRIGRAM.get(upper_elem)
    lower = ELEMENT_TO_TRIGRAM.get(lower_elem)
    return upper, lower


def resolve_hexagram_field(value, name_to_kw):
    """
    classical_before/after_hexagram の値を King Wen番号に解決する。

    パターン1: "52_艮" → 番号52
    パターン2: "艮為山" / "地雷復" → local_nameでルックアップ
    """
    if not value or not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    # パターン1: "NUMBER_NAME"
    if '_' in value:
        try:
            num_str = value.split('_')[0]
            kw = int(num_str)
            if 1 <= kw <= 64:
                return kw
        except (ValueError, IndexError):
            pass

    # パターン2: 卦名でルックアップ
    if value in name_to_kw:
        return name_to_kw[value]

    # 異体字フォールバック
    # 1文字置換チェック
    for variant, canonical in VARIANT_MAP.items():
        if variant in value:
            resolved = value.replace(variant, canonical)
            if resolved in name_to_kw:
                return name_to_kw[resolved]

    # 習坎の特別処理
    if value == '習坎':
        return name_to_kw.get('坎為水')

    return None


def hamming_distance_6bit(bits_a, bits_b):
    """2つの6bitタプル間のハミング距離"""
    return sum(a != b for a, b in zip(bits_a, bits_b))


def cosine_similarity(vec_a, vec_b):
    """コサイン類似度"""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def effect_label(value, thresholds=(0.3, 0.5)):
    """効果量ラベル: small / medium / large"""
    av = abs(value)
    if av < thresholds[0]:
        return "small"
    elif av < thresholds[1]:
        return "medium"
    else:
        return "large"


def effect_direction(observed, null_mean, pro_when_less=False):
    """
    方向判定。
    pro_when_less=True: 観測値 < 帰無平均 が pro方向（例: ハミング距離、Wasserstein距離）
    pro_when_less=False: 観測値 > 帰無平均 が pro方向（例: overlap率、NMI、コサイン）
    """
    if pro_when_less:
        return "pro" if observed < null_mean else ("anti" if observed > null_mean else "neutral")
    else:
        return "pro" if observed > null_mean else ("anti" if observed < null_mean else "neutral")


# ============================================================
# G_obs構築
# ============================================================

def build_g_obs(cases, name_to_kw, kw_to_bits):
    """
    観測遷移グラフ G_obs を構築。
    cases から classical_before_hexagram → classical_after_hexagram の遷移を抽出。
    Returns: (G_obs: nx.DiGraph, transitions: list of (kw_from, kw_to), n_excluded: int)
    """
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

        transitions.append((kw_from, kw_to))

    # 有向重み付きグラフ構築
    G = nx.DiGraph()
    for kw in range(1, 65):
        G.add_node(kw)

    edge_weights = defaultdict(int)
    for kw_from, kw_to in transitions:
        edge_weights[(kw_from, kw_to)] += 1

    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w)

    return G, transitions, n_excluded


def build_q6(kw_to_bits):
    """Q6超立方体グラフを構築（ハミング距離1の全ペアにエッジ）"""
    G = nx.Graph()
    for kw in range(1, 65):
        G.add_node(kw)

    kw_list = list(range(1, 65))
    for i, kw_a in enumerate(kw_list):
        bits_a = kw_to_bits[kw_a]
        for kw_b in kw_list[i + 1:]:
            bits_b = kw_to_bits[kw_b]
            if hamming_distance_6bit(bits_a, bits_b) == 1:
                G.add_edge(kw_a, kw_b)

    return G


# ============================================================
# Test A: エッジ重複率（Edge Overlap）
# ============================================================

def test_a_edge_overlap(transitions, kw_to_bits, q6, rng):
    """
    G_obsの無向エッジのうちQ6エッジ（ハミング距離1）と重なる割合。
    帰無分布: after列シャッフル（周辺分布保存）で1,000回。
    p値: 片側（観測 > ランダム）
    """
    print("\n  [Test A] エッジ重複率...")

    q6_edges = set(frozenset(e) for e in q6.edges())

    # G_obsの無向エッジ集合（重複なし）
    obs_edges = set()
    for kw_from, kw_to in transitions:
        obs_edges.add(frozenset((kw_from, kw_to)))

    if len(obs_edges) == 0:
        return {"h0": "skip", "notes": "No edges in G_obs"}

    # 重複率
    overlap_count = sum(1 for e in obs_edges if e in q6_edges)
    observed_overlap = overlap_count / len(obs_edges)
    print(f"    G_obs無向エッジ数: {len(obs_edges)}, Q6エッジ数: {len(q6_edges)}")
    print(f"    重複: {overlap_count}, 重複率: {observed_overlap:.4f}")

    # 帰無分布: after列シャッフル
    before_list = [t[0] for t in transitions]
    after_list = [t[1] for t in transitions]

    perm_overlaps = []
    for i in range(N_PERMUTATIONS):
        perm_after = list(after_list)
        rng.shuffle(perm_after)
        perm_edges = set()
        for b, a in zip(before_list, perm_after):
            perm_edges.add(frozenset((b, a)))
        perm_overlap = sum(1 for e in perm_edges if e in q6_edges) / len(perm_edges) if perm_edges else 0
        perm_overlaps.append(perm_overlap)
        if (i + 1) % 100 == 0:
            print(f"    置換 {i+1}/{N_PERMUTATIONS}")

    perm_overlaps = np.array(perm_overlaps)
    p_value = np.mean(perm_overlaps >= observed_overlap)

    null_mean = np.mean(perm_overlaps)
    null_std = np.std(perm_overlaps)
    z_score = (observed_overlap - null_mean) / null_std if null_std > 0 else 0.0

    direction = effect_direction(observed_overlap, null_mean, pro_when_less=False)
    elabel = effect_label(z_score)

    print(f"    観測={observed_overlap:.4f}, 帰無平均={null_mean:.4f}, z={z_score:.3f}, p={p_value:.4f}")

    return {
        "h0": "G_obsのQ6エッジ重複率はランダム遷移と同等",
        "statistic": float(observed_overlap),
        "p_value": float(p_value),
        "effect_size": float(z_score),
        "effect_label": elabel,
        "effect_direction": direction,
        "notes": f"overlap={overlap_count}/{len(obs_edges)}, null_mean={null_mean:.4f}, null_std={null_std:.4f}"
    }


# ============================================================
# Test B: ハミング距離分布（6bit空間）
# ============================================================

def test_b_hamming_6bit(transitions, kw_to_bits, rng):
    """
    遷移ペアの平均ハミング距離（6bit）。
    帰無分布: after列シャッフルで1,000回。
    p値: 片側（観測 < ランダム = Q6方向 = 近い遷移が多い）
    """
    print("\n  [Test B] ハミング距離分布（6bit空間）...")

    observed_distances = []
    for kw_from, kw_to in transitions:
        d = hamming_distance_6bit(kw_to_bits[kw_from], kw_to_bits[kw_to])
        observed_distances.append(d)

    observed_mean = np.mean(observed_distances)
    print(f"    遷移ペア数: {len(transitions)}")
    print(f"    観測平均ハミング距離: {observed_mean:.4f}")

    # 距離分布
    from collections import Counter
    dist_counter = Counter(observed_distances)
    print(f"    距離分布: {dict(sorted(dist_counter.items()))}")

    # 帰無分布
    before_list = [t[0] for t in transitions]
    after_list = [t[1] for t in transitions]

    perm_means = []
    for i in range(N_PERMUTATIONS):
        perm_after = list(after_list)
        rng.shuffle(perm_after)
        perm_dists = [hamming_distance_6bit(kw_to_bits[b], kw_to_bits[a])
                      for b, a in zip(before_list, perm_after)]
        perm_means.append(np.mean(perm_dists))
        if (i + 1) % 100 == 0:
            print(f"    置換 {i+1}/{N_PERMUTATIONS}")

    perm_means = np.array(perm_means)
    # p値: 観測 < ランダム 方向
    p_value = np.mean(perm_means <= observed_mean)

    null_mean = np.mean(perm_means)
    null_std = np.std(perm_means)
    z_score = (observed_mean - null_mean) / null_std if null_std > 0 else 0.0

    direction = effect_direction(observed_mean, null_mean, pro_when_less=True)
    elabel = effect_label(z_score)

    print(f"    観測={observed_mean:.4f}, 帰無平均={null_mean:.4f}, z={z_score:.3f}, p={p_value:.4f}")

    return {
        "h0": "遷移ペアの平均ハミング距離はランダム遷移と同等",
        "statistic": float(observed_mean),
        "p_value": float(p_value),
        "effect_size": float(z_score),
        "effect_label": elabel,
        "effect_direction": direction,
        "notes": f"null_mean={null_mean:.4f}, null_std={null_std:.4f}, "
                 f"dist_distribution={dict(sorted(dist_counter.items()))}"
    }


# ============================================================
# Test C: ラプラシアンスペクトル類似性
# ============================================================

def test_c_laplacian_spectrum(transitions, kw_to_bits, q6, rng):
    """
    G_obs（無向化重み付き）のラプラシアン固有値 vs Q6のラプラシアン固有値のWasserstein距離。
    帰無分布: 次数保存シャッフルのランダムグラフで1,000回。
    p値: 片側（観測W < ランダムW = Q6に近い方向）
    """
    print("\n  [Test C] ラプラシアンスペクトル類似性...")

    # G_obs無向化（重み付き）
    G_obs_undirected = nx.Graph()
    for kw in range(1, 65):
        G_obs_undirected.add_node(kw)
    edge_weights = defaultdict(int)
    for kw_from, kw_to in transitions:
        pair = tuple(sorted((kw_from, kw_to)))
        edge_weights[pair] += 1
    for (u, v), w in edge_weights.items():
        G_obs_undirected.add_edge(u, v, weight=w)

    nodelist = list(range(1, 65))

    # G_obsのラプラシアン固有値
    L_obs = nx.laplacian_matrix(G_obs_undirected, nodelist=nodelist, weight='weight').toarray().astype(float)
    eig_obs = np.sort(np.linalg.eigvalsh(L_obs))

    # Q6のラプラシアン固有値
    L_q6 = nx.laplacian_matrix(q6, nodelist=nodelist).toarray().astype(float)
    eig_q6 = np.sort(np.linalg.eigvalsh(L_q6))

    # 正規化: [0, 1]
    def normalize_spectrum(eigs):
        mn, mx = eigs.min(), eigs.max()
        if mx - mn == 0:
            return np.zeros_like(eigs)
        return (eigs - mn) / (mx - mn)

    eig_obs_norm = normalize_spectrum(eig_obs)
    eig_q6_norm = normalize_spectrum(eig_q6)

    observed_w = wasserstein_distance(eig_obs_norm, eig_q6_norm)
    print(f"    観測 Wasserstein距離: {observed_w:.6f}")

    # 帰無分布: 次数保存ランダムグラフ（configuration model）
    # G_obs_undirectedの次数列を保存してランダムグラフ生成
    degree_sequence = [d for _, d in G_obs_undirected.degree(weight='weight')]

    perm_ws = []
    for i in range(N_PERMUTATIONS):
        # 次数保存シャッフル: エッジのダブルスワップ
        # 無向グラフのエッジをランダムに再配線
        try:
            G_rand = G_obs_undirected.copy()
            # nx.double_edge_swap でエッジを入れ替え
            n_swaps = max(G_rand.number_of_edges(), 100)
            if G_rand.number_of_edges() >= 2:
                nx.double_edge_swap(G_rand, nswap=n_swaps, max_tries=n_swaps * 10, seed=int(rng.integers(0, 2**31)))
            L_rand = nx.laplacian_matrix(G_rand, nodelist=nodelist, weight='weight').toarray().astype(float)
            eig_rand = np.sort(np.linalg.eigvalsh(L_rand))
            eig_rand_norm = normalize_spectrum(eig_rand)
            perm_ws.append(wasserstein_distance(eig_rand_norm, eig_q6_norm))
        except nx.NetworkXError:
            # スワップ失敗時はスキップ
            # フォールバック: configuration modelで生成
            try:
                deg_seq = [d for _, d in G_obs_undirected.degree()]
                G_rand = nx.configuration_model(deg_seq, seed=int(rng.integers(0, 2**31)))
                G_rand = nx.Graph(G_rand)  # マルチグラフ→単純グラフ
                G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
                # ノードラベルを合わせる
                mapping = {i: nodelist[i] for i in range(min(len(nodelist), G_rand.number_of_nodes()))}
                G_rand = nx.relabel_nodes(G_rand, mapping)
                for kw in nodelist:
                    if kw not in G_rand:
                        G_rand.add_node(kw)
                L_rand = nx.laplacian_matrix(G_rand, nodelist=nodelist).toarray().astype(float)
                eig_rand = np.sort(np.linalg.eigvalsh(L_rand))
                eig_rand_norm = normalize_spectrum(eig_rand)
                perm_ws.append(wasserstein_distance(eig_rand_norm, eig_q6_norm))
            except Exception:
                pass  # skip this permutation

        if (i + 1) % 100 == 0:
            print(f"    置換 {i+1}/{N_PERMUTATIONS}")

    perm_ws = np.array(perm_ws)
    if len(perm_ws) == 0:
        return {"h0": "skip", "notes": "All permutations failed"}

    # p値: 観測W < ランダムW 方向（Q6に近い方向）
    p_value = np.mean(perm_ws <= observed_w)

    null_mean = np.mean(perm_ws)
    null_std = np.std(perm_ws)
    z_score = (observed_w - null_mean) / null_std if null_std > 0 else 0.0

    direction = effect_direction(observed_w, null_mean, pro_when_less=True)
    elabel = effect_label(z_score)

    print(f"    帰無平均={null_mean:.6f}, z={z_score:.3f}, p={p_value:.4f}, 有効置換数={len(perm_ws)}")

    return {
        "h0": "G_obsのラプラシアンスペクトルとQ6のWasserstein距離はランダムグラフと同等",
        "statistic": float(observed_w),
        "p_value": float(p_value),
        "effect_size": float(z_score),
        "effect_label": elabel,
        "effect_direction": direction,
        "notes": f"null_mean={null_mean:.6f}, null_std={null_std:.6f}, n_valid_perms={len(perm_ws)}"
    }


# ============================================================
# Test D: 錯卦対称性（64卦レベル）
# ============================================================

def test_d_cuogua_symmetry(transitions, kw_to_bits, bits_to_kw, cuogua_pairs, rng):
    """
    G_obsの遷移確率行列の行ベクトルについて、32錯卦ペアのコサイン類似度の平均。
    帰無分布: 64ノードからランダム完全マッチング（32ペア）を1,000回生成。
    p値: 片側（観測 > ランダム = 錯卦が類似方向）
    """
    print("\n  [Test D] 錯卦対称性...")

    # 遷移確率行列構築（64x64）
    # 行: from卦（0-indexed: kw-1）、列: to卦
    trans_counts = np.zeros((64, 64))
    for kw_from, kw_to in transitions:
        trans_counts[kw_from - 1][kw_to - 1] += 1

    # 遷移確率行列
    trans_prob = np.zeros((64, 64))
    out_degree_zero = set()
    for i in range(64):
        row_sum = trans_counts[i].sum()
        if row_sum > 0:
            trans_prob[i] = trans_counts[i] / row_sum
        else:
            trans_prob[i] = np.ones(64) / 64  # 一様分布
            out_degree_zero.add(i + 1)  # King Wen番号（1-indexed）

    print(f"    出次数0の卦: {len(out_degree_zero)}個")

    # 錯卦ペアのコサイン類似度（出次数0の卦を含むペアは除外）
    valid_pairs = []
    for a, b in cuogua_pairs:
        if a not in out_degree_zero and b not in out_degree_zero:
            valid_pairs.append((a, b))

    print(f"    有効錯卦ペア数: {len(valid_pairs)}/{len(cuogua_pairs)}")

    if len(valid_pairs) == 0:
        return {"h0": "skip", "notes": "No valid cuogua pairs (all have zero out-degree)"}

    observed_sims = []
    for a, b in valid_pairs:
        sim = cosine_similarity(trans_prob[a - 1], trans_prob[b - 1])
        observed_sims.append(sim)
    observed_mean = np.mean(observed_sims)
    print(f"    観測錯卦コサイン類似度平均: {observed_mean:.4f}")

    # 帰無分布: ランダム完全マッチング
    # 出次数0でないノードのみからペアを作る
    valid_nodes = [kw for kw in range(1, 65) if kw not in out_degree_zero]
    n_valid = len(valid_nodes)
    n_pairs = len(valid_pairs)  # 有効ペアと同数のペアを生成

    perm_means = []
    for i in range(N_PERMUTATIONS):
        # ランダム完全マッチング: validノードをシャッフルして隣接ペアを作る
        shuffled = list(valid_nodes)
        rng.shuffle(shuffled)
        # n_pairs個のペアを作る（残りは捨てる）
        rand_pairs = [(shuffled[2 * j], shuffled[2 * j + 1])
                      for j in range(min(n_pairs, n_valid // 2))]
        rand_sims = [cosine_similarity(trans_prob[a - 1], trans_prob[b - 1])
                     for a, b in rand_pairs]
        perm_means.append(np.mean(rand_sims) if rand_sims else 0.0)

        if (i + 1) % 100 == 0:
            print(f"    置換 {i+1}/{N_PERMUTATIONS}")

    perm_means = np.array(perm_means)
    p_value = np.mean(perm_means >= observed_mean)

    null_mean = np.mean(perm_means)
    null_std = np.std(perm_means)
    z_score = (observed_mean - null_mean) / null_std if null_std > 0 else 0.0

    direction = effect_direction(observed_mean, null_mean, pro_when_less=False)
    elabel = effect_label(z_score)

    print(f"    帰無平均={null_mean:.4f}, z={z_score:.3f}, p={p_value:.4f}")

    return {
        "h0": "錯卦ペアの遷移確率類似度はランダムペアと同等",
        "statistic": float(observed_mean),
        "p_value": float(p_value),
        "effect_size": float(z_score),
        "effect_label": elabel,
        "effect_direction": direction,
        "notes": f"valid_pairs={len(valid_pairs)}/{len(cuogua_pairs)}, "
                 f"zero_outdeg={len(out_degree_zero)}, null_mean={null_mean:.4f}"
    }


# ============================================================
# Test E: コミュニティ構造NMI
# ============================================================

def test_e_community_nmi(transitions, kw_to_bits, q6, graph_data, rng):
    """
    G_obs（無向化）にLouvainを適用 → Q6のLouvainコミュニティとのNMI。
    帰無分布: after列シャッフルで作ったランダムグラフにLouvain → Q6とのNMIを1,000回。
    p値: 片側（観測NMI > ランダムNMI）
    """
    print("\n  [Test E] コミュニティ構造NMI...")

    # Q6のLouvainコミュニティ（graph_analysis.jsonから取得）
    q6_communities = graph_data['community_structure']['communities']
    q6_labels = {}
    for comm_id_str, members in q6_communities.items():
        for m in members:
            q6_labels[m] = int(comm_id_str)
    q6_label_arr = np.array([q6_labels.get(kw, -1) for kw in range(1, 65)])
    print(f"    Q6コミュニティ数: {len(q6_communities)}")

    # G_obs無向化
    def build_undirected_from_transitions(trans_list):
        G = nx.Graph()
        for kw in range(1, 65):
            G.add_node(kw)
        ew = defaultdict(int)
        for kw_from, kw_to in trans_list:
            pair = tuple(sorted((kw_from, kw_to)))
            ew[pair] += 1
        for (u, v), w in ew.items():
            G.add_edge(u, v, weight=w)
        return G

    G_obs_u = build_undirected_from_transitions(transitions)

    # G_obsにLouvain適用
    if G_obs_u.number_of_edges() == 0:
        return {"h0": "skip", "notes": "G_obs has no edges"}

    obs_partition = community_louvain.best_partition(G_obs_u, weight='weight', random_state=RANDOM_SEED)
    obs_label_arr = np.array([obs_partition.get(kw, -1) for kw in range(1, 65)])
    n_comm_obs = len(set(obs_partition.values()))
    print(f"    G_obsコミュニティ数: {n_comm_obs}")

    # NMI
    observed_nmi = normalized_mutual_info_score(q6_label_arr, obs_label_arr)
    print(f"    観測NMI: {observed_nmi:.4f}")

    # 帰無分布: after列シャッフル
    before_list = [t[0] for t in transitions]
    after_list = [t[1] for t in transitions]

    perm_nmis = []
    for i in range(N_PERMUTATIONS):
        perm_after = list(after_list)
        rng.shuffle(perm_after)
        perm_trans = list(zip(before_list, perm_after))
        G_perm = build_undirected_from_transitions(perm_trans)
        if G_perm.number_of_edges() == 0:
            perm_nmis.append(0.0)
            continue
        perm_partition = community_louvain.best_partition(G_perm, weight='weight',
                                                          random_state=RANDOM_SEED)
        perm_label_arr = np.array([perm_partition.get(kw, -1) for kw in range(1, 65)])
        perm_nmi = normalized_mutual_info_score(q6_label_arr, perm_label_arr)
        perm_nmis.append(perm_nmi)

        if (i + 1) % 100 == 0:
            print(f"    置換 {i+1}/{N_PERMUTATIONS}")

    perm_nmis = np.array(perm_nmis)
    p_value = np.mean(perm_nmis >= observed_nmi)

    null_mean = np.mean(perm_nmis)
    null_std = np.std(perm_nmis)
    z_score = (observed_nmi - null_mean) / null_std if null_std > 0 else 0.0

    direction = effect_direction(observed_nmi, null_mean, pro_when_less=False)
    elabel = effect_label(z_score)

    print(f"    帰無平均={null_mean:.4f}, z={z_score:.3f}, p={p_value:.4f}")

    return {
        "h0": "G_obsとQ6のコミュニティ構造のNMIはランダム遷移と同等",
        "statistic": float(observed_nmi),
        "p_value": float(p_value),
        "effect_size": float(z_score),
        "effect_label": elabel,
        "effect_direction": direction,
        "notes": f"G_obs communities={n_comm_obs}, Q6 communities={len(q6_communities)}, "
                 f"null_mean={null_mean:.4f}"
    }


# ============================================================
# 総合判定: Fisher結合p値
# ============================================================

def combined_judgment(test_results):
    """Fisher結合p値による総合判定"""
    p_values = []
    effect_profile = []

    for test_name, result in test_results.items():
        if result.get("h0") == "skip":
            continue
        p = result['p_value']
        p_values.append(p)
        effect_profile.append({
            "test": test_name,
            "effect_size": result['effect_size'],
            "direction": result['effect_direction'],
            "label": result['effect_label'],
        })

    if not p_values:
        return {
            "fisher_statistic": None,
            "fisher_p_value": None,
            "level": "None",
            "summary": "No valid tests completed",
            "effect_profile": effect_profile
        }

    # Fisher結合統計量
    fisher_stat = -2 * sum(np.log(max(p, 1e-300)) for p in p_values)
    fisher_p = 1 - chi2.cdf(fisher_stat, df=2 * len(p_values))

    # 効果量プロファイル集計
    n_pro_medium_plus = sum(1 for ep in effect_profile
                            if ep['direction'] == 'pro'
                            and ep['label'] in ('medium', 'large'))
    n_pro_small_plus = sum(1 for ep in effect_profile
                           if ep['direction'] == 'pro'
                           and ep['label'] in ('small', 'medium', 'large'))

    # 判定レベル
    if fisher_p < 0.01 and n_pro_medium_plus >= 4:
        level = "Strong"
    elif fisher_p < 0.05 and n_pro_small_plus >= 3:
        level = "Moderate"
    elif fisher_p < 0.05:
        level = "Weak"
    else:
        level = "None"

    summary = (
        f"Fisher chi2={fisher_stat:.2f}, p={fisher_p:.6f}, level={level}. "
        f"Pro方向medium+: {n_pro_medium_plus}/{len(effect_profile)}, "
        f"Pro方向small+: {n_pro_small_plus}/{len(effect_profile)}"
    )

    return {
        "fisher_statistic": float(fisher_stat),
        "fisher_p_value": float(fisher_p),
        "level": level,
        "summary": summary,
        "effect_profile": effect_profile
    }


# ============================================================
# レポート生成
# ============================================================

def generate_report(metadata, test_results, judgment):
    """人間可読レポート（Markdown）"""
    lines = []
    lines.append("# Phase 3: 同型性検証レポート v2.0")
    lines.append("")
    lines.append(f"**実行日時**: {metadata['timestamp']}")
    lines.append(f"**使用事例数**: {metadata['n_cases_used']} (除外: {metadata['n_cases_excluded']})")
    lines.append(f"**置換回数**: {metadata['n_permutations']}")
    lines.append(f"**乱数シード**: {metadata['seed']}")
    lines.append("")
    lines.append("## テスト結果")
    lines.append("")

    for test_name, result in test_results.items():
        lines.append(f"### {test_name}")
        if result.get("h0") == "skip":
            lines.append(f"- **スキップ**: {result.get('notes', '')}")
        else:
            lines.append(f"- **H0**: {result['h0']}")
            lines.append(f"- **統計量**: {result['statistic']:.4f}")
            lines.append(f"- **p値**: {result['p_value']:.4f}")
            lines.append(f"- **効果量**: {result['effect_size']:.3f} ({result['effect_label']}, {result['effect_direction']})")
            lines.append(f"- **備考**: {result.get('notes', '')}")
        lines.append("")

    lines.append("## 総合判定")
    lines.append("")
    lines.append(f"- **Fisher統計量**: {judgment['fisher_statistic']}")
    lines.append(f"- **Fisher p値**: {judgment['fisher_p_value']}")
    lines.append(f"- **判定レベル**: {judgment['level']}")
    lines.append(f"- **要約**: {judgment['summary']}")
    lines.append("")
    lines.append("### 効果量プロファイル")
    lines.append("")
    lines.append("| テスト | 効果量 | 方向 | ラベル |")
    lines.append("|--------|--------|------|--------|")
    for ep in judgment['effect_profile']:
        lines.append(f"| {ep['test']} | {ep['effect_size']:.3f} | {ep['direction']} | {ep['label']} |")
    lines.append("")

    lines.append("## 判定基準")
    lines.append("")
    lines.append("- **Strong**: Fisher p < 0.01 かつ 5テスト中4つ以上がpro方向のmedium以上効果")
    lines.append("- **Moderate**: Fisher p < 0.05 かつ 3つ以上がpro方向のsmall以上効果")
    lines.append("- **Weak**: Fisher p < 0.05 だが pro方向効果は2つ以下")
    lines.append("- **None**: Fisher p >= 0.05")
    lines.append("")

    return "\n".join(lines)


# ============================================================
# メイン
# ============================================================

def main():
    print("=" * 60)
    print("Phase 3: 同型性検証 v2.0")
    print("=" * 60)

    t_start = time.time()
    rng = np.random.default_rng(RANDOM_SEED)

    # --- データ読み込み ---
    print("\n[1] データ読み込み...")
    reference_data = load_json(REFERENCE_FILE)
    graph_data = load_json(GRAPH_ANALYSIS_FILE)
    cases = load_cases()
    print(f"  事例数: {len(cases)}")

    # --- マッピング構築 ---
    print("\n[2] マッピング構築...")
    name_to_kw = build_name_to_kw(reference_data)
    kw_to_bits, bits_to_kw = build_kw_to_bits(reference_data)
    print(f"  卦名→番号マッピング: {len(name_to_kw)}件")
    print(f"  番号→6bit: {len(kw_to_bits)}件")

    # 検証: 乾=1=(1,1,1,1,1,1), 坤=2=(0,0,0,0,0,0)
    assert kw_to_bits[1] == (1, 1, 1, 1, 1, 1), f"乾のビット列が不正: {kw_to_bits[1]}"
    assert kw_to_bits[2] == (0, 0, 0, 0, 0, 0), f"坤のビット列が不正: {kw_to_bits[2]}"
    print(f"  乾(1) = {kw_to_bits[1]}  OK")
    print(f"  坤(2) = {kw_to_bits[2]}  OK")

    # --- G_obs構築 ---
    print("\n[3] 観測遷移グラフ G_obs 構築...")
    G_obs, transitions, n_excluded = build_g_obs(cases, name_to_kw, kw_to_bits)
    n_used = len(transitions)
    print(f"  遷移ペア数: {n_used} (除外: {n_excluded})")
    print(f"  G_obsエッジ数: {G_obs.number_of_edges()}")
    print(f"  G_obsノード（出次数>0）: {sum(1 for n in G_obs.nodes() if G_obs.out_degree(n) > 0)}")

    # --- Q6構築 ---
    print("\n[4] Q6超立方体グラフ構築...")
    q6 = build_q6(kw_to_bits)
    print(f"  Q6エッジ数: {q6.number_of_edges()} (理論値: 192)")

    # --- 錯卦ペア取得 ---
    cuogua_pairs = [(a, b) for a, b in graph_data['structural_relations']['cuogua_pairs']]
    print(f"  錯卦ペア: {len(cuogua_pairs)}")

    # --- 5テスト実行 ---
    print("\n[5] 統計的検証...")
    test_results = {}

    test_results['test_a_edge_overlap'] = test_a_edge_overlap(transitions, kw_to_bits, q6, rng)
    test_results['test_b_hamming_6bit'] = test_b_hamming_6bit(transitions, kw_to_bits, rng)
    test_results['test_c_laplacian_spectrum'] = test_c_laplacian_spectrum(transitions, kw_to_bits, q6, rng)
    test_results['test_d_cuogua_symmetry'] = test_d_cuogua_symmetry(transitions, kw_to_bits, bits_to_kw, cuogua_pairs, rng)
    test_results['test_e_community_nmi'] = test_e_community_nmi(transitions, kw_to_bits, q6, graph_data, rng)

    # --- 総合判定 ---
    print("\n[6] 総合判定...")
    judgment = combined_judgment(test_results)
    print(f"  Fisher統計量: {judgment['fisher_statistic']}")
    print(f"  Fisher p値: {judgment['fisher_p_value']}")
    print(f"  判定レベル: {judgment['level']}")
    print(f"  {judgment['summary']}")

    # --- メタデータ ---
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "seed": RANDOM_SEED,
        "n_permutations": N_PERMUTATIONS,
        "n_cases_used": n_used,
        "n_cases_excluded": n_excluded,
        "version": "2.0"
    }

    # --- JSON出力 ---
    print("\n[7] 結果保存...")
    output_json = {
        "metadata": metadata,
        "tests": test_results,
        "combined_judgment": judgment
    }

    json_path = PHASE3_DIR / "statistical_tests.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {json_path}")

    # --- レポート出力 ---
    report_md = generate_report(metadata, test_results, judgment)
    report_path = PHASE3_DIR / "report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"  Saved: {report_path}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Phase 3 完了 ({elapsed:.1f}秒)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
