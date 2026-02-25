#!/usr/bin/env python3
"""
64卦の状態空間モデル — Phase 1

6次元二値ベクトル空間 {0,1}^6 として64卦をモデル化し、
グラフ構造の数学的性質を分析する。
"""

import json
import os
import sys
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import linalg
from collections import Counter
from pathlib import Path

# ---------- パス設定 ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
REFERENCE_FILE = DATA_DIR / "reference" / "iching_texts_ctext_legge_ja.json"
TRANSITIONS_FILE = DATA_DIR / "mappings" / "yao_transitions.json"
OUTPUT_DIR = Path(__file__).resolve().parent
VIS_DIR = OUTPUT_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- フォント設定 ----------
def setup_japanese_font():
    """日本語フォントを設定。macOSのヒラギノを優先的に使用。"""
    jp_fonts = [
        '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
        '/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
        '/System/Library/Fonts/Hiragino Sans GB.ttc',
    ]
    for fp in jp_fonts:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            prop = fm.FontProperties(fname=fp)
            plt.rcParams['font.family'] = prop.get_name()
            return prop.get_name()
    # フォールバック
    plt.rcParams['font.family'] = 'sans-serif'
    return 'sans-serif'

# ---------- 八卦定義 ----------
# 仕様書に従った三爻の定義（下から上: [初爻, 二爻, 三爻]）
TRIGRAM_BITS = {
    # 爻順序: [初爻(下), 二爻(中), 三爻(上)] — 下から上へ
    # 伝統的な八卦記号の爻構成に基づく
    '乾': [1, 1, 1],  # ☰ 天  ━━━ ━━━ ━━━
    '坤': [0, 0, 0],  # ☷ 地  ━ ━ ━ ━ ━ ━
    '震': [1, 0, 0],  # ☳ 雷  初爻のみ陽
    '巽': [0, 1, 1],  # ☴ 風  初爻のみ陰
    '坎': [0, 1, 0],  # ☵ 水  中爻のみ陽
    '離': [1, 0, 1],  # ☲ 火  中爻のみ陰
    '艮': [0, 0, 1],  # ☶ 山  上爻のみ陽
    '兌': [1, 1, 0],  # ☱ 沢  上爻のみ陰
}

# 卦名から上卦・下卦を判別するための名前マッピング
# local_name形式: "上卦名+下卦名+卦名" (例: "水雷屯" → 上卦=水(坎), 下卦=雷(震))
# 「〜為〜」形式は上下同一 (例: "乾為天" → 上下とも乾)
ELEMENT_TO_TRIGRAM = {
    '天': '乾', '地': '坤', '雷': '震', '風': '巽',
    '水': '坎', '火': '離', '山': '艮', '沢': '兌',
}


class HexagramStateSpace:
    """64卦の状態空間モデル

    6次元二値ベクトル空間として64卦をモデル化し、
    超立方体グラフ Q6 の構造的性質を分析する。
    """

    def __init__(self):
        """64卦を6ビットベクトルとして初期化"""
        self._load_reference_data()
        self._load_transitions()
        self._build_bit_vectors()
        self.graph = None

    def _load_reference_data(self):
        """iching_texts_ctext_legge_ja.json からデータ読み込み"""
        with open(REFERENCE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.hexagram_data = data['hexagrams']

    def _load_transitions(self):
        """yao_transitions.json から遷移マップ読み込み"""
        with open(TRANSITIONS_FILE, 'r', encoding='utf-8') as f:
            self.transitions = json.load(f)

    def _parse_trigrams_from_name(self, local_name: str):
        """卦名から上卦・下卦を解析する

        形式1: "X為Y" → 上下とも同じ八卦 (例: "乾為天" → 乾/乾)
        形式2: "XY+卦名" → X=上卦の元素, Y=下卦の元素 (例: "水雷屯" → 坎/震)

        Returns:
            (upper_trigram_name, lower_trigram_name): 八卦名のタプル
        """
        if '為' in local_name:
            # "X為Y" 形式: Xが八卦名
            trigram_name = local_name[0]
            if trigram_name in TRIGRAM_BITS:
                return trigram_name, trigram_name
            # "X為Y"のYから判定
            element = local_name[2]
            trigram_name = ELEMENT_TO_TRIGRAM.get(element, None)
            if trigram_name:
                return trigram_name, trigram_name

        # "XY+卦名" 形式: 先頭2文字が上卦元素と下卦元素
        upper_elem = local_name[0]
        lower_elem = local_name[1]
        upper = ELEMENT_TO_TRIGRAM.get(upper_elem)
        lower = ELEMENT_TO_TRIGRAM.get(lower_elem)
        return upper, lower

    def _build_bit_vectors(self):
        """King Wen番号⇔6ビットベクトルの対応テーブル構築

        6ビットベクトル: [初爻, 二爻, 三爻, 四爻, 五爻, 上爻]
        下卦(初爻〜三爻) + 上卦(四爻〜上爻)
        """
        self.kw_to_bits = {}  # King Wen番号 → ビット列
        self.bits_to_kw = {}  # ビット列(タプル) → King Wen番号
        self.kw_to_name = {}  # King Wen番号 → 卦名
        self.kw_to_trigrams = {}  # King Wen番号 → (上卦名, 下卦名)

        for kw_str, hdata in self.hexagram_data.items():
            kw = int(kw_str)
            local_name = hdata['local_name']
            self.kw_to_name[kw] = local_name

            upper, lower = self._parse_trigrams_from_name(local_name)
            self.kw_to_trigrams[kw] = (upper, lower)

            # ビット列: 下卦(初〜三爻) + 上卦(四〜上爻)
            lower_bits = TRIGRAM_BITS[lower]
            upper_bits = TRIGRAM_BITS[upper]
            bits = lower_bits + upper_bits  # [初, 二, 三, 四, 五, 上]
            self.kw_to_bits[kw] = bits
            self.bits_to_kw[tuple(bits)] = kw

    def get_bits(self, hex_id: int) -> list:
        """King Wen番号からビット列を返す"""
        return self.kw_to_bits[hex_id]

    def get_kw(self, bits) -> int:
        """ビット列からKing Wen番号を返す"""
        return self.bits_to_kw[tuple(bits)]

    def hamming_distance(self, hex_a: int, hex_b: int) -> int:
        """2つの卦間のハミング距離を計算"""
        bits_a = self.kw_to_bits[hex_a]
        bits_b = self.kw_to_bits[hex_b]
        return sum(a != b for a, b in zip(bits_a, bits_b))

    def build_graph(self) -> nx.Graph:
        """64ノード有向グラフを構築

        ハミング距離1の全ペアに双方向エッジを張る（1爻変化）。
        無向グラフとして表現（双方向なので同等）。
        """
        G = nx.Graph()

        # ノード追加
        for kw in range(1, 65):
            bits = self.kw_to_bits[kw]
            name = self.kw_to_name[kw]
            upper, lower = self.kw_to_trigrams[kw]
            G.add_node(kw,
                       bits=bits,
                       name=name,
                       upper_trigram=upper,
                       lower_trigram=lower,
                       bit_string=''.join(map(str, bits)))

        # ハミング距離1の全ペアにエッジ
        kw_list = list(range(1, 65))
        for i, kw_a in enumerate(kw_list):
            bits_a = self.kw_to_bits[kw_a]
            for kw_b in kw_list[i + 1:]:
                bits_b = self.kw_to_bits[kw_b]
                diff_positions = [pos + 1 for pos, (a, b) in enumerate(zip(bits_a, bits_b)) if a != b]
                if len(diff_positions) == 1:
                    G.add_edge(kw_a, kw_b, changing_line=diff_positions[0])

        self.graph = G
        return G

    def get_cuogua(self, hex_id: int) -> int:
        """錯卦（全爻反転）を返す"""
        bits = self.kw_to_bits[hex_id]
        flipped = [1 - b for b in bits]
        return self.bits_to_kw[tuple(flipped)]

    def get_zonggua(self, hex_id: int) -> int:
        """綜卦（上下反転＝爻順序の反転）を返す"""
        bits = self.kw_to_bits[hex_id]
        reversed_bits = bits[::-1]
        return self.bits_to_kw[tuple(reversed_bits)]

    def get_hugua(self, hex_id: int) -> int:
        """互卦（2-5爻から新しい卦を構成）を返す

        下卦 = 2,3,4爻 (元の爻位置 index 1,2,3)
        上卦 = 3,4,5爻 (元の爻位置 index 2,3,4)
        """
        bits = self.kw_to_bits[hex_id]
        lower_hu = bits[1:4]  # 2,3,4爻
        upper_hu = bits[2:5]  # 3,4,5爻
        hu_bits = lower_hu + upper_hu
        return self.bits_to_kw[tuple(hu_bits)]

    def get_zhigua(self, hex_id: int, changing_lines: list) -> int:
        """之卦（変爻による遷移先）を返す

        Args:
            hex_id: 元の卦のKing Wen番号
            changing_lines: 変化する爻のリスト (1-6)
        """
        bits = list(self.kw_to_bits[hex_id])
        for line in changing_lines:
            idx = line - 1  # 0-indexed
            bits[idx] = 1 - bits[idx]
        return self.bits_to_kw[tuple(bits)]

    def transition_matrix(self) -> np.ndarray:
        """64x64の遷移行列を返す

        1爻変化で到達可能な卦への等確率遷移。
        行i→列j: 卦iから卦jへの遷移確率 (1/6 or 0)。
        """
        if self.graph is None:
            self.build_graph()

        # ノード番号を0-indexに変換
        n = 64
        T = np.zeros((n, n))
        for kw_a in range(1, 65):
            neighbors = list(self.graph.neighbors(kw_a))
            for kw_b in neighbors:
                T[kw_a - 1][kw_b - 1] = 1.0 / len(neighbors)
        return T

    # ---------- 構造的関係の分析 ----------

    def all_cuogua_pairs(self) -> list:
        """全錯卦ペアを返す"""
        pairs = set()
        for kw in range(1, 65):
            partner = self.get_cuogua(kw)
            pair = tuple(sorted([kw, partner]))
            pairs.add(pair)
        return sorted(pairs)

    def all_zonggua_pairs(self) -> dict:
        """全綜卦マッピングを返す

        Returns:
            dict with:
                'pairs': 異なるペアのリスト
                'self_zonggua': 自己綜卦のリスト
        """
        pairs = set()
        self_zong = []
        for kw in range(1, 65):
            partner = self.get_zonggua(kw)
            if partner == kw:
                self_zong.append(kw)
            else:
                pair = tuple(sorted([kw, partner]))
                pairs.add(pair)
        return {
            'pairs': sorted(pairs),
            'self_zonggua': sorted(self_zong),
        }

    def all_hugua_mapping(self) -> dict:
        """全互卦マッピングを返す"""
        return {kw: self.get_hugua(kw) for kw in range(1, 65)}

    # ---------- グラフ分析 ----------

    def analyze_basic_properties(self) -> dict:
        """グラフの基本性質を計算"""
        G = self.graph
        return {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'diameter': nx.diameter(G),
            'average_shortest_path': round(nx.average_shortest_path_length(G), 6),
            'clustering_coefficient': round(nx.average_clustering(G), 6),
            'is_connected': nx.is_connected(G),
        }

    def analyze_degree_distribution(self) -> dict:
        """次数分布を分析"""
        degrees = [d for _, d in self.graph.degree()]
        return {
            'min_degree': min(degrees),
            'max_degree': max(degrees),
            'mean_degree': round(np.mean(degrees), 4),
            'is_regular': min(degrees) == max(degrees),
            'degree_histogram': dict(Counter(degrees)),
        }

    def analyze_community_structure(self) -> dict:
        """コミュニティ構造を分析（Louvainアルゴリズム）"""
        partition = community_louvain.best_partition(self.graph, random_state=42)
        modularity = community_louvain.modularity(partition, self.graph)
        num_communities = len(set(partition.values()))

        # コミュニティごとのメンバー
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        for k in communities:
            communities[k].sort()

        return {
            'method': 'Louvain',
            'num_communities': num_communities,
            'modularity': round(modularity, 6),
            'partition': partition,
            'communities': communities,
        }

    def analyze_spectral_properties(self) -> dict:
        """スペクトル性質を分析"""
        A = nx.adjacency_matrix(self.graph, nodelist=list(range(1, 65))).toarray().astype(float)
        eigenvalues = np.sort(linalg.eigvalsh(A))[::-1]  # 降順
        spectral_gap = eigenvalues[0] - eigenvalues[1]

        return {
            'eigenvalues': [round(float(e), 6) for e in eigenvalues],
            'spectral_gap': round(float(spectral_gap), 6),
            'largest_eigenvalue': round(float(eigenvalues[0]), 6),
            'smallest_eigenvalue': round(float(eigenvalues[-1]), 6),
        }

    def analyze_structural_relations(self) -> dict:
        """構造的関係のサマリー"""
        cuogua_pairs = self.all_cuogua_pairs()
        zonggua_data = self.all_zonggua_pairs()
        hugua_map = self.all_hugua_mapping()

        return {
            'cuogua_pairs_count': len(cuogua_pairs),
            'cuogua_pairs': [[a, b] for a, b in cuogua_pairs],
            'zonggua_pairs_count': len(zonggua_data['pairs']),
            'zonggua_self_count': len(zonggua_data['self_zonggua']),
            'zonggua_self_hexagrams': zonggua_data['self_zonggua'],
            'zonggua_pairs': [[a, b] for a, b in zonggua_data['pairs']],
            'hugua_mapping': {str(k): v for k, v in hugua_map.items()},
        }

    def run_full_analysis(self) -> dict:
        """全分析を実行し、結果を辞書で返す"""
        if self.graph is None:
            self.build_graph()

        print("  [1/5] 基本性質の分析...")
        basic = self.analyze_basic_properties()

        print("  [2/5] 次数分布の分析...")
        degree = self.analyze_degree_distribution()

        print("  [3/5] コミュニティ構造の分析...")
        community_data = self.analyze_community_structure()

        print("  [4/5] スペクトル性質の分析...")
        spectral = self.analyze_spectral_properties()

        print("  [5/5] 構造的関係の分析...")
        structural = self.analyze_structural_relations()

        # コミュニティデータからpartitionを除去 (JSON非対応の内部データ)
        community_json = {k: v for k, v in community_data.items() if k != 'partition'}

        return {
            'basic_properties': basic,
            'degree_distribution': degree,
            'community_structure': community_json,
            'spectral_properties': spectral,
            'structural_relations': structural,
        }

    # ---------- 可視化 ----------

    def visualize_hypercube_projection(self, community_data=None):
        """6次元超立方体の2D射影（spectral layout）"""
        G = self.graph
        font_name = setup_japanese_font()

        fig, axes = plt.subplots(1, 2, figsize=(24, 12))

        # --- 左: Spectral Layout ---
        pos_spectral = nx.spectral_layout(G)
        ax = axes[0]
        ax.set_title('Spectral Layout (Q6 Hypercube)', fontsize=14)

        if community_data and 'partition' in community_data:
            partition = community_data['partition']
            colors = [partition[n] for n in G.nodes()]
            nx.draw_networkx_edges(G, pos_spectral, ax=ax, alpha=0.15, edge_color='gray', width=0.5)
            nx.draw_networkx_nodes(G, pos_spectral, ax=ax, node_color=colors,
                                   cmap=plt.cm.Set3, node_size=200, alpha=0.9)
        else:
            nx.draw_networkx_edges(G, pos_spectral, ax=ax, alpha=0.15, edge_color='gray', width=0.5)
            nx.draw_networkx_nodes(G, pos_spectral, ax=ax, node_color='steelblue',
                                   node_size=200, alpha=0.9)
        labels = {n: str(n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos_spectral, labels, ax=ax, font_size=6)

        # --- 右: Spring Layout ---
        pos_spring = nx.spring_layout(G, seed=42, k=0.8, iterations=100)
        ax = axes[1]
        ax.set_title('Spring Layout (Q6 Hypercube)', fontsize=14)

        if community_data and 'partition' in community_data:
            partition = community_data['partition']
            colors = [partition[n] for n in G.nodes()]
            nx.draw_networkx_edges(G, pos_spring, ax=ax, alpha=0.15, edge_color='gray', width=0.5)
            nx.draw_networkx_nodes(G, pos_spring, ax=ax, node_color=colors,
                                   cmap=plt.cm.Set3, node_size=200, alpha=0.9)
        else:
            nx.draw_networkx_edges(G, pos_spring, ax=ax, alpha=0.15, edge_color='gray', width=0.5)
            nx.draw_networkx_nodes(G, pos_spring, ax=ax, node_color='steelblue',
                                   node_size=200, alpha=0.9)
        nx.draw_networkx_labels(G, pos_spring, labels, ax=ax, font_size=6)

        plt.tight_layout()
        path = VIS_DIR / 'hypercube_projection.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")

    def visualize_community_graph(self, community_data):
        """コミュニティ構造のグラフ"""
        G = self.graph
        font_name = setup_japanese_font()

        partition = community_data['partition']
        communities = community_data['communities']
        n_comm = len(communities)

        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        ax.set_title(f'Community Structure (Louvain, {n_comm} communities, '
                     f'modularity={community_data["modularity"]:.4f})', fontsize=14)

        pos = nx.spring_layout(G, seed=42, k=0.8, iterations=100)
        colors = [partition[n] for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1, edge_color='gray', width=0.5)
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                                        cmap=plt.cm.Set3, node_size=250, alpha=0.9)

        # ラベル: 卦番号＋卦名（短縮）
        labels = {}
        for n in G.nodes():
            name = self.kw_to_name[n]
            short = name[:2] if len(name) > 2 else name
            labels[n] = f'{n}\n{short}'
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=5,
                                font_family=font_name)

        plt.tight_layout()
        path = VIS_DIR / 'community_graph.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")

    def visualize_cuogua_pairs(self):
        """錯卦ペアの可視化"""
        G = self.graph
        font_name = setup_japanese_font()
        pairs = self.all_cuogua_pairs()

        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        ax.set_title(f'Cuogua (Inverse) Pairs — {len(pairs)} pairs', fontsize=14)

        pos = nx.spring_layout(G, seed=42, k=0.8, iterations=100)

        # 通常エッジ（薄く）
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.05, edge_color='gray', width=0.3)

        # 錯卦ペアのエッジ（目立たせる）
        cuogua_edges = [(a, b) for a, b in pairs]
        nx.draw_networkx_edges(G, pos, edgelist=cuogua_edges, ax=ax,
                               alpha=0.7, edge_color='red', width=1.5, style='dashed')

        # ノード描画
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue',
                               node_size=200, alpha=0.9)
        labels = {n: str(n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6)

        plt.tight_layout()
        path = VIS_DIR / 'cuogua_pairs.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")

    def visualize_spectral_embedding(self):
        """スペクトル埋め込み（隣接行列の固有ベクトルを使用）"""
        G = self.graph
        font_name = setup_japanese_font()

        A = nx.adjacency_matrix(G, nodelist=list(range(1, 65))).toarray().astype(float)
        eigenvalues, eigenvectors = linalg.eigh(A)

        # 最大固有値に対応する2つの固有ベクトルを射影軸に使用
        idx = np.argsort(eigenvalues)[::-1]
        ev1 = eigenvectors[:, idx[1]]  # 2番目（最大は定数ベクトル）
        ev2 = eigenvectors[:, idx[2]]  # 3番目

        fig, axes = plt.subplots(1, 2, figsize=(24, 12))

        # 左: 固有ベクトル2,3による射影
        ax = axes[0]
        ax.set_title('Spectral Embedding (Eigenvectors 2-3)', fontsize=14)
        for kw_a in range(1, 65):
            for kw_b in G.neighbors(kw_a):
                if kw_a < kw_b:
                    ax.plot([ev1[kw_a - 1], ev1[kw_b - 1]],
                            [ev2[kw_a - 1], ev2[kw_b - 1]],
                            color='gray', alpha=0.1, linewidth=0.5)
        ax.scatter(ev1, ev2, c='steelblue', s=80, zorder=5, alpha=0.8)
        for kw in range(1, 65):
            ax.annotate(str(kw), (ev1[kw - 1], ev2[kw - 1]),
                        fontsize=5, ha='center', va='center')
        ax.set_xlabel('Eigenvector 2')
        ax.set_ylabel('Eigenvector 3')

        # 右: 固有値スペクトル
        ax = axes[1]
        ax.set_title('Eigenvalue Spectrum of Adjacency Matrix', fontsize=14)
        sorted_evals = np.sort(eigenvalues)[::-1]
        ax.bar(range(64), sorted_evals, color='steelblue', alpha=0.8)
        ax.set_xlabel('Index')
        ax.set_ylabel('Eigenvalue')
        ax.axhline(y=0, color='black', linewidth=0.5)

        plt.tight_layout()
        path = VIS_DIR / 'spectral_embedding.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")

    # ---------- 検証 ----------

    def validate(self) -> dict:
        """全検証を実行し、結果を返す"""
        results = {}

        # 1. 64ノード
        results['64_nodes'] = self.graph.number_of_nodes() == 64

        # 2. 各ノードの次数が6
        degrees = [d for _, d in self.graph.degree()]
        results['all_degree_6'] = all(d == 6 for d in degrees)
        if not results['all_degree_6']:
            bad = [(n, d) for n, d in self.graph.degree() if d != 6]
            results['bad_degrees'] = bad

        # 3. 錯卦ペアが32組
        cuogua_pairs = self.all_cuogua_pairs()
        results['cuogua_32_pairs'] = len(cuogua_pairs) == 32

        # 4. グラフが連結
        results['is_connected'] = nx.is_connected(self.graph)

        # 5. King Wen番号⇔ビット列の基本チェック
        results['qian_bits'] = self.kw_to_bits[1] == [1, 1, 1, 1, 1, 1]
        results['kun_bits'] = self.kw_to_bits[2] == [0, 0, 0, 0, 0, 0]

        # 6. 全64ビット列がユニーク
        all_bits = [tuple(self.kw_to_bits[kw]) for kw in range(1, 65)]
        results['all_unique_bits'] = len(set(all_bits)) == 64

        # 7. yao_transitions.jsonとの整合性チェック
        transition_ok = True
        transition_errors = []
        for kw_str, tdata in self.transitions.items():
            kw = int(kw_str)
            for line_str, target in tdata['transitions'].items():
                line = int(line_str)
                expected_target = target['next_hexagram_id']
                computed_target = self.get_zhigua(kw, [line])
                if computed_target != expected_target:
                    transition_ok = False
                    transition_errors.append({
                        'hexagram': kw,
                        'line': line,
                        'expected': expected_target,
                        'computed': computed_target,
                    })
        results['transitions_consistent'] = transition_ok
        if transition_errors:
            results['transition_errors'] = transition_errors

        return results


def main():
    """メインエントリポイント: 全ステップを実行"""
    print("=" * 60)
    print("Phase 1: 64卦の状態空間モデル構築")
    print("=" * 60)

    # Step 1: データ読み込みとビット列変換テーブル構築
    print("\n[Step 1] データ読み込みとビット列変換テーブル構築...")
    model = HexagramStateSpace()
    print(f"  64卦のビット列テーブル構築完了")
    print(f"  乾為天(1) = {model.kw_to_bits[1]}")
    print(f"  坤為地(2) = {model.kw_to_bits[2]}")

    # Step 2: グラフ構築
    print("\n[Step 2] NetworkXで64ノードグラフを構築...")
    G = model.build_graph()
    print(f"  ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")

    # Step 3: 検証
    print("\n[Step 3] 検証...")
    validation = model.validate()
    all_ok = all(v for k, v in validation.items()
                 if isinstance(v, bool))
    for key, val in validation.items():
        if isinstance(val, bool):
            status = "OK" if val else "FAIL"
            print(f"  {key}: {status}")
        elif key == 'transition_errors':
            print(f"  transition_errors: {len(val)} errors")
            for err in val[:5]:
                print(f"    卦{err['hexagram']} 爻{err['line']}: "
                      f"expected={err['expected']}, computed={err['computed']}")
    if not all_ok:
        print("\n  WARNING: 一部の検証に失敗しました")
        sys.exit(1)

    # Step 4: 分析
    print("\n[Step 4] グラフの数学的性質を分析...")
    analysis = model.run_full_analysis()

    # 分析結果の表示
    print(f"\n  基本性質:")
    for k, v in analysis['basic_properties'].items():
        print(f"    {k}: {v}")
    print(f"\n  次数分布:")
    for k, v in analysis['degree_distribution'].items():
        print(f"    {k}: {v}")
    print(f"\n  コミュニティ構造:")
    print(f"    コミュニティ数: {analysis['community_structure']['num_communities']}")
    print(f"    モジュラリティ: {analysis['community_structure']['modularity']}")
    print(f"\n  スペクトル性質:")
    print(f"    最大固有値: {analysis['spectral_properties']['largest_eigenvalue']}")
    print(f"    最小固有値: {analysis['spectral_properties']['smallest_eigenvalue']}")
    print(f"    スペクトルギャップ: {analysis['spectral_properties']['spectral_gap']}")
    print(f"\n  構造的関係:")
    print(f"    錯卦ペア: {analysis['structural_relations']['cuogua_pairs_count']}")
    print(f"    綜卦ペア: {analysis['structural_relations']['zonggua_pairs_count']}")
    print(f"    自己綜卦: {analysis['structural_relations']['zonggua_self_count']}個 "
          f"{analysis['structural_relations']['zonggua_self_hexagrams']}")

    # Step 5: 分析結果の保存
    print("\n[Step 5] 分析結果をJSON出力...")
    output_path = OUTPUT_DIR / 'graph_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {output_path}")

    # Step 6: 可視化
    print("\n[Step 6] 可視化...")
    community_data = model.analyze_community_structure()

    print("  [1/4] 超立方体射影...")
    model.visualize_hypercube_projection(community_data)

    print("  [2/4] コミュニティ構造...")
    model.visualize_community_graph(community_data)

    print("  [3/4] 錯卦ペア...")
    model.visualize_cuogua_pairs()

    print("  [4/4] スペクトル埋め込み...")
    model.visualize_spectral_embedding()

    print("\n" + "=" * 60)
    print("Phase 1 完了")
    print("=" * 60)

    return model, analysis, validation


if __name__ == '__main__':
    main()
