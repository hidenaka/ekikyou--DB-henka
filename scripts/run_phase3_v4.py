#!/usr/bin/env python3
"""
run_phase3_v4.py — Phase 3 v4.0 再実行（感度分析込み）

measurement_spec.md に準拠した5つの統計検定を実行:
  Test A: Edge Overlap with Q6
  Test B: Hamming Distance Distribution
  Test C: Laplacian Spectral Similarity
  Test D: Cuogua (錯卦) Symmetry
  Test E: Community Structure (NMI)

入力: analysis/gold_set/hexagram_transitions.json
出力: analysis/phase3/report_v4.md, analysis/phase3/phase3_v4_results.json
"""

import json
import numpy as np
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_SET_DIR = PROJECT_ROOT / "analysis" / "gold_set"
PHASE3_DIR = PROJECT_ROOT / "analysis" / "phase3"

# ─── Binary Encoding ───────────────────────────────────────────────
TRIGRAM_BINARY = {
    "乾": (1, 1, 1), "兌": (1, 1, 0), "離": (1, 0, 1), "震": (1, 0, 0),
    "巽": (0, 1, 1), "坎": (0, 1, 0), "艮": (0, 0, 1), "坤": (0, 0, 0),
}
TRIGRAM_INDEX = {"乾": 0, "坤": 1, "震": 2, "巽": 3, "坎": 4, "離": 5, "艮": 6, "兌": 7}

# King Wen: KING_WEN[lower_idx][upper_idx] = hex_number
KING_WEN = [
    [1, 11, 34, 9, 5, 14, 26, 43],    # 乾
    [12, 2, 16, 20, 8, 35, 23, 45],    # 坤
    [25, 24, 51, 42, 3, 21, 27, 17],   # 震
    [44, 46, 32, 57, 48, 50, 18, 28],  # 巽
    [6, 7, 40, 59, 29, 64, 4, 47],     # 坎
    [13, 36, 55, 37, 63, 30, 22, 49],  # 離
    [33, 15, 62, 53, 39, 56, 52, 31],  # 艮
    [10, 19, 54, 61, 60, 38, 41, 58],  # 兌
]

TRIGRAM_NAMES = ["乾", "坤", "震", "巽", "坎", "離", "艮", "兌"]


def hex_to_binary(hex_num):
    """King Wen number -> 6-bit binary vector."""
    for li, row in enumerate(KING_WEN):
        for ui, val in enumerate(row):
            if val == hex_num:
                lower = TRIGRAM_BINARY[TRIGRAM_NAMES[li]]
                upper = TRIGRAM_BINARY[TRIGRAM_NAMES[ui]]
                return lower + upper
    return None


def hamming_distance(a, b):
    """Hamming distance between two 6-bit tuples."""
    return sum(x != y for x, y in zip(a, b))


# Precompute binary encodings and Q6 adjacency
HEX_BINARY = {}
for h in range(1, 65):
    HEX_BINARY[h] = hex_to_binary(h)

Q6_EDGES = set()
for i in range(1, 65):
    for j in range(1, 65):
        if i != j and hamming_distance(HEX_BINARY[i], HEX_BINARY[j]) == 1:
            Q6_EDGES.add((i, j))


# ─── Graph Construction ────────────────────────────────────────────

def build_graph(transitions, threshold=1):
    """Build directed graph from transitions with weight threshold."""
    edge_counts = Counter()
    for t in transitions:
        edge_counts[(t["before"], t["after"])] += 1

    edges = {e: w for e, w in edge_counts.items() if w >= threshold}
    return edges


def graph_to_adjacency(edges, n=64):
    """Convert edge dict to adjacency structures."""
    out_neighbors = defaultdict(set)
    in_neighbors = defaultdict(set)
    for (s, t) in edges:
        out_neighbors[s].add(t)
        in_neighbors[t].add(s)
    return out_neighbors, in_neighbors


# ─── Null Model ────────────────────────────────────────────────────

def generate_null_graph(out_degree_seq, in_degree_seq, n_nodes=64, rng=None):
    """
    Generate a random directed graph preserving degree sequences.
    Simple edge-swap Markov chain approach.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Build initial edge list from degree sequences
    sources = []
    targets = []
    for node, deg in out_degree_seq.items():
        sources.extend([node] * deg)
    for node, deg in in_degree_seq.items():
        targets.extend([node] * deg)

    if len(sources) != len(targets):
        # Pad shorter
        diff = len(sources) - len(targets)
        if diff > 0:
            targets = targets[:len(sources)]
        else:
            sources = sources[:len(targets)]

    rng.shuffle(targets)

    # Build edge set (collapse multi-edges)
    edges = set()
    for s, t in zip(sources, targets):
        edges.add((s, t))

    return edges


def generate_null_graphs(observed_edges, n_permutations=1000, seed=42):
    """Generate n null model graphs preserving degree sequences."""
    rng = np.random.default_rng(seed)

    # Compute degree sequences
    out_deg = defaultdict(int)
    in_deg = defaultdict(int)
    for (s, t) in observed_edges:
        out_deg[s] += 1
        in_deg[t] += 1

    null_graphs = []
    for i in range(n_permutations):
        null_edges = generate_null_graph(out_deg, in_deg, rng=rng)
        null_graphs.append(null_edges)

    return null_graphs


# ─── Test A: Edge Overlap ──────────────────────────────────────────

def test_a_overlap(observed_edges, null_graphs):
    """Test A: Edge overlap with Q6."""
    obs_set = set(observed_edges.keys())
    obs_overlap = len(obs_set & Q6_EDGES) / len(obs_set) if obs_set else 0

    null_overlaps = []
    for null_edges in null_graphs:
        overlap = len(null_edges & Q6_EDGES) / len(null_edges) if null_edges else 0
        null_overlaps.append(overlap)

    null_overlaps = np.array(null_overlaps)
    z = (obs_overlap - null_overlaps.mean()) / null_overlaps.std() if null_overlaps.std() > 0 else 0
    p_raw = np.mean(null_overlaps >= obs_overlap)

    return {
        "test": "A",
        "name": "Edge Overlap with Q6",
        "observed": round(obs_overlap, 4),
        "null_mean": round(float(null_overlaps.mean()), 4),
        "null_std": round(float(null_overlaps.std()), 4),
        "null_p2.5": round(float(np.percentile(null_overlaps, 2.5)), 4),
        "null_p97.5": round(float(np.percentile(null_overlaps, 97.5)), 4),
        "z_score": round(float(z), 3),
        "p_raw": round(float(p_raw), 4),
        "p_bonf": round(min(float(p_raw) * 5, 1.0), 4),
        "direction": "pro" if z > 0 else "anti",
        "decision": "reject" if p_raw < 0.01 and z > 0 else "fail_to_reject",
    }


# ─── Test B: Hamming Distance ─────────────────────────────────────

def mean_hamming(edges):
    """Compute mean Hamming distance of edges."""
    total = 0
    count = 0
    for (s, t) in edges:
        if s in HEX_BINARY and t in HEX_BINARY:
            total += hamming_distance(HEX_BINARY[s], HEX_BINARY[t])
            count += 1
    return total / count if count > 0 else 0


def test_b_hamming(observed_edges, null_graphs):
    """Test B: Hamming distance distribution."""
    # Weighted mean for observed
    total_hd = 0
    total_w = 0
    for (s, t), w in observed_edges.items():
        if s in HEX_BINARY and t in HEX_BINARY:
            total_hd += hamming_distance(HEX_BINARY[s], HEX_BINARY[t]) * w
            total_w += w
    obs_mean = total_hd / total_w if total_w > 0 else 0

    null_means = []
    for null_edges in null_graphs:
        hd = mean_hamming(null_edges)
        null_means.append(hd)

    null_means = np.array(null_means)
    z = (obs_mean - null_means.mean()) / null_means.std() if null_means.std() > 0 else 0
    p_raw = np.mean(null_means <= obs_mean)  # One-sided: smaller = more Q6-like

    return {
        "test": "B",
        "name": "Hamming Distance Distribution",
        "observed": round(obs_mean, 4),
        "null_mean": round(float(null_means.mean()), 4),
        "null_std": round(float(null_means.std()), 4),
        "null_p2.5": round(float(np.percentile(null_means, 2.5)), 4),
        "null_p97.5": round(float(np.percentile(null_means, 97.5)), 4),
        "z_score": round(float(z), 3),
        "p_raw": round(float(p_raw), 4),
        "p_bonf": round(min(float(p_raw) * 5, 1.0), 4),
        "direction": "pro" if z < 0 else "anti",
        "decision": "reject" if p_raw < 0.01 and z < 0 else "fail_to_reject",
    }


# ─── Test C: Laplacian Spectral Similarity ─────────────────────────

def compute_laplacian_spectrum(edges, n=64):
    """Compute normalized Laplacian eigenvalues of undirected version."""
    # Build adjacency matrix (undirected)
    A = np.zeros((n, n))
    for (s, t) in edges:
        if 1 <= s <= n and 1 <= t <= n:
            A[s-1][t-1] = 1
            A[t-1][s-1] = 1

    # Degree matrix
    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.zeros_like(D)
    for i in range(n):
        if D[i, i] > 0:
            D_inv_sqrt[i, i] = 1.0 / np.sqrt(D[i, i])

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L)))
    return eigenvalues


def wasserstein_1d(a, b):
    """1D Wasserstein distance between sorted arrays."""
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    n = min(len(a_sorted), len(b_sorted))
    return np.mean(np.abs(a_sorted[:n] - b_sorted[:n]))


def test_c_spectral(observed_edges, null_graphs):
    """Test C: Laplacian spectral similarity."""
    # Q6 spectrum
    q6_spectrum = compute_laplacian_spectrum(Q6_EDGES)

    # Observed spectrum (use largest connected component conceptually, but compute on full)
    obs_spectrum = compute_laplacian_spectrum(observed_edges.keys())
    obs_w1 = wasserstein_1d(obs_spectrum, q6_spectrum)

    null_w1s = []
    for null_edges in null_graphs:
        null_spectrum = compute_laplacian_spectrum(null_edges)
        null_w1s.append(wasserstein_1d(null_spectrum, q6_spectrum))

    null_w1s = np.array(null_w1s)
    z = (obs_w1 - null_w1s.mean()) / null_w1s.std() if null_w1s.std() > 0 else 0
    p_raw = np.mean(null_w1s <= obs_w1)  # One-sided: smaller = more similar

    return {
        "test": "C",
        "name": "Laplacian Spectral Similarity",
        "observed": round(float(obs_w1), 4),
        "null_mean": round(float(null_w1s.mean()), 4),
        "null_std": round(float(null_w1s.std()), 4),
        "null_p2.5": round(float(np.percentile(null_w1s, 2.5)), 4),
        "null_p97.5": round(float(np.percentile(null_w1s, 97.5)), 4),
        "z_score": round(float(z), 3),
        "p_raw": round(float(p_raw), 4),
        "p_bonf": round(min(float(p_raw) * 5, 1.0), 4),
        "direction": "pro" if z < 0 else "anti",
        "decision": "reject" if p_raw < 0.01 and z < 0 else "fail_to_reject",
    }


# ─── Test D: Cuogua Symmetry ──────────────────────────────────────

def complement_hex(h):
    """Get the cuogua (complement) hexagram: flip all bits."""
    b = HEX_BINARY[h]
    comp = tuple(1 - x for x in b)
    # Find the hexagram with this binary
    for h2, b2 in HEX_BINARY.items():
        if b2 == comp:
            return h2
    return None


def transition_vector(node, edges, n=64):
    """Get outgoing transition probability vector for a node."""
    vec = np.zeros(n)
    total = 0
    for (s, t), w in edges.items():
        if s == node:
            vec[t - 1] += w
            total += w
    if total > 0:
        vec /= total
    return vec


def cosine_sim(a, b):
    """Cosine similarity."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def test_d_cuogua(observed_edges, null_graphs):
    """Test D: Cuogua symmetry."""
    # Build 32 cuogua pairs
    pairs = []
    seen = set()
    for h in range(1, 65):
        comp = complement_hex(h)
        if comp and h not in seen and comp not in seen:
            pairs.append((h, comp))
            seen.add(h)
            seen.add(comp)

    # Observed cuogua similarity
    def cuogua_mean_sim(edges_dict):
        sims = []
        for h1, h2 in pairs:
            v1 = transition_vector(h1, edges_dict)
            v2 = transition_vector(h2, edges_dict)
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                sims.append(cosine_sim(v1, v2))
        return np.mean(sims) if sims else 0, len(sims)

    obs_sim, n_pairs_used = cuogua_mean_sim(observed_edges)

    # For null graphs, we need weighted edges
    null_sims = []
    for null_edges in null_graphs:
        null_dict = {e: 1 for e in null_edges}
        sim, _ = cuogua_mean_sim(null_dict)
        null_sims.append(sim)

    null_sims = np.array(null_sims)
    z = (obs_sim - null_sims.mean()) / null_sims.std() if null_sims.std() > 0 else 0
    p_raw = np.mean(null_sims >= obs_sim)

    return {
        "test": "D",
        "name": "Cuogua Symmetry",
        "observed": round(float(obs_sim), 4),
        "null_mean": round(float(null_sims.mean()), 4),
        "null_std": round(float(null_sims.std()), 4),
        "null_p2.5": round(float(np.percentile(null_sims, 2.5)), 4),
        "null_p97.5": round(float(np.percentile(null_sims, 97.5)), 4),
        "z_score": round(float(z), 3),
        "p_raw": round(float(p_raw), 4),
        "p_bonf": round(min(float(p_raw) * 5, 1.0), 4),
        "n_cuogua_pairs_used": n_pairs_used,
        "direction": "pro" if z > 0 else "anti",
        "decision": "reject" if p_raw < 0.01 and z > 0 else "fail_to_reject",
    }


# ─── Test E: Community Structure (NMI) ─────────────────────────────

def louvain_communities(edges, n=64, rng=None):
    """Simple greedy modularity-based community detection."""
    if rng is None:
        rng = np.random.default_rng()

    # Build adjacency (undirected)
    adj = defaultdict(lambda: defaultdict(float))
    for (s, t) in edges:
        if 1 <= s <= n and 1 <= t <= n:
            adj[s][t] += 1
            adj[t][s] += 1

    # Total weight
    m = sum(sum(adj[i].values()) for i in adj) / 2.0
    if m == 0:
        return {i: 0 for i in range(1, n + 1)}

    # Initialize: each node in its own community
    community = {i: i for i in range(1, n + 1)}

    # Iterative greedy optimization
    improved = True
    max_iter = 50
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        nodes = list(range(1, n + 1))
        rng.shuffle(nodes)

        for node in nodes:
            # Try moving node to each neighbor's community
            best_delta = 0
            best_comm = community[node]
            current_comm = community[node]

            neighbor_comms = set()
            for nb in adj[node]:
                neighbor_comms.add(community[nb])

            for target_comm in neighbor_comms:
                if target_comm == current_comm:
                    continue

                # Simplified modularity gain
                ki = sum(adj[node].values())
                sum_in = sum(adj[node].get(j, 0) for j in range(1, n + 1)
                           if community[j] == target_comm)
                sum_out = sum(adj[node].get(j, 0) for j in range(1, n + 1)
                            if community[j] == current_comm and j != node)

                delta = (sum_in - sum_out) / m if m > 0 else 0

                if delta > best_delta:
                    best_delta = delta
                    best_comm = target_comm

            if best_comm != current_comm:
                community[node] = best_comm
                improved = True

    return community


def nmi(labels_a, labels_b, n=64):
    """Normalized Mutual Information."""
    # Build contingency
    classes_a = set(labels_a.values())
    classes_b = set(labels_b.values())

    N = n
    # Contingency matrix
    cont = defaultdict(int)
    for i in range(1, n + 1):
        a = labels_a.get(i, 0)
        b = labels_b.get(i, 0)
        cont[(a, b)] += 1

    # Marginals
    a_counts = Counter(labels_a.get(i, 0) for i in range(1, n + 1))
    b_counts = Counter(labels_b.get(i, 0) for i in range(1, n + 1))

    # MI
    mi = 0
    for (a, b), nij in cont.items():
        if nij == 0:
            continue
        pi = a_counts[a] / N
        pj = b_counts[b] / N
        pij = nij / N
        mi += pij * np.log(pij / (pi * pj))

    # Entropies
    ha = -sum((c / N) * np.log(c / N) for c in a_counts.values() if c > 0)
    hb = -sum((c / N) * np.log(c / N) for c in b_counts.values() if c > 0)

    if ha + hb == 0:
        return 0.0

    return 2 * mi / (ha + hb)


def test_e_community(observed_edges, null_graphs):
    """Test E: Community structure NMI."""
    # Q6 reference partition: 8 groups by lower trigram
    q6_partition = {}
    for h in range(1, 65):
        b = HEX_BINARY[h]
        lower_tri = b[:3]  # lower trigram bits
        q6_partition[h] = lower_tri

    # Run Louvain 10 times on observed
    obs_nmis = []
    for seed in range(10):
        rng = np.random.default_rng(seed + 100)
        comm = louvain_communities(observed_edges.keys(), rng=rng)
        obs_nmis.append(nmi(comm, q6_partition))

    obs_nmi_mean = np.mean(obs_nmis)
    obs_nmi_std = np.std(obs_nmis)
    stable = obs_nmi_std <= 0.1

    # Null distribution
    null_nmis = []
    for i, null_edges in enumerate(null_graphs):
        rng = np.random.default_rng(i + 200)
        comm = louvain_communities(null_edges, rng=rng)
        null_nmis.append(nmi(comm, q6_partition))

    null_nmis = np.array(null_nmis)
    z = (obs_nmi_mean - null_nmis.mean()) / null_nmis.std() if null_nmis.std() > 0 else 0
    p_raw = np.mean(null_nmis >= obs_nmi_mean)

    return {
        "test": "E",
        "name": "Community Structure (NMI)",
        "observed": round(float(obs_nmi_mean), 4),
        "observed_std": round(float(obs_nmi_std), 4),
        "stable": stable,
        "null_mean": round(float(null_nmis.mean()), 4),
        "null_std": round(float(null_nmis.std()), 4),
        "null_p2.5": round(float(np.percentile(null_nmis, 2.5)), 4),
        "null_p97.5": round(float(np.percentile(null_nmis, 97.5)), 4),
        "z_score": round(float(z), 3),
        "p_raw": round(float(p_raw), 4),
        "p_bonf": round(min(float(p_raw) * 5, 1.0), 4),
        "direction": "pro" if z > 0 else "anti",
        "decision": ("inconclusive" if not stable else
                     "reject" if p_raw < 0.01 and z > 0 else "fail_to_reject"),
    }


# ─── Main ──────────────────────────────────────────────────────────

def run_all_tests(transitions, threshold, n_permutations=1000):
    """Run all 5 tests at a given threshold."""
    edges = build_graph(transitions, threshold)
    print(f"  Threshold w>={threshold}: {len(edges)} unique edges")

    if len(edges) < 10:
        print("  WARNING: Too few edges, skipping tests")
        return None

    null_graphs = generate_null_graphs(edges, n_permutations)

    results = []
    print("  Running Test A (Edge Overlap)...", end=" ", flush=True)
    results.append(test_a_overlap(edges, null_graphs))
    print(f"z={results[-1]['z_score']}")

    print("  Running Test B (Hamming Distance)...", end=" ", flush=True)
    results.append(test_b_hamming(edges, null_graphs))
    print(f"z={results[-1]['z_score']}")

    print("  Running Test C (Spectral)...", end=" ", flush=True)
    results.append(test_c_spectral(edges, null_graphs))
    print(f"z={results[-1]['z_score']}")

    print("  Running Test D (Cuogua)...", end=" ", flush=True)
    results.append(test_d_cuogua(edges, null_graphs))
    print(f"z={results[-1]['z_score']}")

    print("  Running Test E (Community NMI)...", end=" ", flush=True)
    results.append(test_e_community(edges, null_graphs))
    print(f"z={results[-1]['z_score']}")

    return results


def generate_report(all_results, metadata):
    """Generate markdown report."""
    lines = ["# Phase 3 v4.0 Results Report", ""]
    lines.append(f"**Generated**: {datetime.now().isoformat()}")
    lines.append(f"**Cases**: {metadata['n_transitions']}")
    lines.append(f"**Active hexagrams**: {metadata['n_active']}/64")
    lines.append(f"**Null model permutations**: {metadata['n_permutations']}")
    lines.append("")

    for threshold, results in sorted(all_results.items()):
        lines.append(f"## Threshold w >= {threshold}")
        lines.append("")

        if results is None:
            lines.append("Insufficient edges. Tests skipped.")
            lines.append("")
            continue

        lines.append("| Test | Statistic | Observed | Null Mean±SD | z-score | p (Bonf.) | Direction | Decision |")
        lines.append("|------|-----------|----------|-------------|---------|-----------|-----------|----------|")

        for r in results:
            lines.append(
                f"| {r['test']} | {r['name']} | {r['observed']:.4f} | "
                f"{r['null_mean']:.4f}±{r['null_std']:.4f} | "
                f"{r['z_score']:.2f} | {r['p_bonf']:.4f} | "
                f"{r['direction']} | {r['decision']} |"
            )

        lines.append("")

        # Interpretation
        n_reject = sum(1 for r in results if r["decision"] == "reject")
        n_pro = sum(1 for r in results if r["direction"] == "pro")
        n_anti = sum(1 for r in results if r["direction"] == "anti")

        lines.append(f"**Summary**: {n_reject}/5 tests rejected H0 "
                     f"({n_pro} pro-isomorphism, {n_anti} anti-isomorphism)")
        lines.append("")

    # Overall conclusion
    lines.append("## Overall Conclusion")
    lines.append("")

    primary = all_results.get(1)
    if primary:
        rejected = [r for r in primary if r["decision"] == "reject"]
        pro = [r for r in primary if r["direction"] == "pro" and r["decision"] == "reject"]
        anti = [r for r in primary if r["direction"] == "anti" and r["decision"] == "reject"]

        if len(pro) >= 3:
            level = "STRONG"
        elif len(pro) >= 2:
            level = "MODERATE"
        elif len(pro) >= 1:
            level = "WEAK"
        else:
            level = "NONE"

        lines.append(f"- **Isomorphism evidence level**: {level}")
        lines.append(f"- Pro-isomorphism rejections: {len(pro)}/5")
        lines.append(f"- Anti-isomorphism rejections: {len(anti)}/5")
        effect_parts = [f"{r['test']}={r['z_score']}" for r in primary]
        lines.append(f"- Effect sizes (z): {', '.join(effect_parts)}")
        lines.append("")

        # Sensitivity
        thresholds_tested = sorted(all_results.keys())
        if len(thresholds_tested) > 1:
            lines.append("### Sensitivity Analysis")
            lines.append("")
            lines.append("| Test | w≥1 | w≥2 | w≥3 | Stable? |")
            lines.append("|------|-----|-----|-----|---------|")
            for i in range(5):
                test_name = primary[i]["test"]
                decisions = []
                for t in thresholds_tested:
                    if all_results[t] and len(all_results[t]) > i:
                        decisions.append(all_results[t][i]["decision"])
                    else:
                        decisions.append("N/A")
                stable = "Yes" if len(set(decisions)) <= 1 else "**No**"
                lines.append(f"| {test_name} | {' | '.join(decisions)} | {stable} |")
            lines.append("")

    return "\n".join(lines)


def main():
    # Load transitions
    trans_path = GOLD_SET_DIR / "hexagram_transitions.json"
    with open(trans_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transitions = data["transitions"]
    n_active = data["metadata"]["n_active_hexagrams"]
    print(f"Loaded {len(transitions)} transitions, {n_active} active hexagrams")

    N_PERM = 1000
    metadata = {
        "n_transitions": len(transitions),
        "n_active": n_active,
        "n_permutations": N_PERM,
    }

    # Run at three thresholds
    all_results = {}
    for threshold in [1, 2, 3]:
        print(f"\n--- Threshold w >= {threshold} ---")
        all_results[threshold] = run_all_tests(transitions, threshold, N_PERM)

    # Generate report
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)

    report = generate_report(all_results, metadata)
    report_path = PHASE3_DIR / "report_v4.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport: {report_path}")

    # Save JSON results
    json_results = {
        "metadata": metadata,
        "results": {str(k): v for k, v in all_results.items()},
        "generated_at": datetime.now().isoformat(),
    }
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    json_path = PHASE3_DIR / "phase3_v4_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
