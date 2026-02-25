#!/usr/bin/env python3
"""
Phase 2B 品質レビュー修正: 可視化ファイル生成 (M1)
====================================================

4つの可視化を生成:
  1. transition_heatmap.png     — 7x8 遷移確率ヒートマップ（度数アノテーション付き）
  2. adjusted_residuals_heatmap.png — 調整残差ヒートマップ
  3. cluster_scatter.png        — MCA Dim1 vs Dim2 クラスタリング散布図
  4. silhouette_elbow.png       — シルエットスコア + エルボー法プロット

出力先: analysis/phase2/visualizations/
"""

import json
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# --- パス設定 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PHASE2_DIR = os.path.join(PROJECT_ROOT, 'analysis', 'phase2')
VIS_DIR = os.path.join(PHASE2_DIR, 'visualizations')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'cases.jsonl')

os.makedirs(VIS_DIR, exist_ok=True)

# --- フォント設定 ---
JP_FONTS = [
    '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
    '/Library/Fonts/Arial Unicode.ttf',
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
]

jp_font = None
for f in JP_FONTS:
    if os.path.exists(f):
        jp_font = fm.FontProperties(fname=f)
        break

if jp_font is None:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [
        'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Yu Gothic', 'Arial'
    ]
else:
    plt.rcParams['font.family'] = jp_font.get_name()

plt.rcParams['axes.unicode_minus'] = False

# --- 定数 ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MCA_COLUMNS = [
    'before_state', 'trigger_type', 'action_type',
    'after_state', 'pattern_type', 'outcome', 'scale',
]


def load_transition_stats():
    """transition_stats.json を読み込む。"""
    path = os.path.join(PHASE2_DIR, 'transition_stats.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_cluster_results():
    """cluster_results.json を読み込む。"""
    path = os.path.join(PHASE2_DIR, 'cluster_results.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_font(ax):
    """日本語フォントをtick labelsに適用。"""
    if jp_font:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(jp_font)


# =============================================================================
# Plot 1: transition_heatmap.png
# =============================================================================

def plot_transition_heatmap(stats):
    """7x8 遷移確率ヒートマップ（度数アノテーション付き）。"""
    print("  [1/4] transition_heatmap.png を生成中...")

    before_states = stats['before_states']
    after_states = stats['after_states']
    count_data = stats['basic_transition']['count_matrix']
    prob_data = stats['basic_transition']['prob_matrix']

    # DataFrameに変換
    count_df = pd.DataFrame(0, index=before_states, columns=after_states)
    for bs in before_states:
        for a_s in after_states:
            count_df.loc[bs, a_s] = count_data.get(bs, {}).get(a_s, 0)

    prob_df = pd.DataFrame(0.0, index=before_states, columns=after_states)
    for bs in before_states:
        for a_s in after_states:
            prob_df.loc[bs, a_s] = prob_data.get(bs, {}).get(a_s, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # 左: 件数ヒートマップ
    ax1 = axes[0]
    sns.heatmap(
        count_df, annot=True, fmt='d', cmap='YlOrRd',
        ax=ax1, linewidths=0.5,
        xticklabels=True, yticklabels=True,
    )
    title1 = '遷移件数 (before_state x after_state)'
    ax1.set_title(title1, fontsize=14, fontproperties=jp_font if jp_font else None)
    ax1.set_xlabel('after_state', fontsize=11)
    ax1.set_ylabel('before_state', fontsize=11)
    apply_font(ax1)

    # 右: 確率ヒートマップ
    ax2 = axes[1]
    sns.heatmap(
        prob_df, annot=True, fmt='.2f', cmap='YlGnBu',
        ax=ax2, linewidths=0.5, vmin=0, vmax=1,
        xticklabels=True, yticklabels=True,
    )
    title2 = '遷移確率（行正規化）'
    ax2.set_title(title2, fontsize=14, fontproperties=jp_font if jp_font else None)
    ax2.set_xlabel('after_state', fontsize=11)
    ax2.set_ylabel('before_state', fontsize=11)
    apply_font(ax2)

    plt.tight_layout()
    fpath = os.path.join(VIS_DIR, 'transition_heatmap.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    保存: {fpath}")
    return fpath


# =============================================================================
# Plot 2: adjusted_residuals_heatmap.png
# =============================================================================

def plot_adjusted_residuals(stats):
    """調整残差ヒートマップ（有意水準ライン付き）。"""
    print("  [2/4] adjusted_residuals_heatmap.png を生成中...")

    before_states = stats['before_states']
    after_states = stats['after_states']

    # 調整残差を再構築（significant_transitions から）
    sig_pos = stats['significant_transitions']['over_represented']
    sig_neg = stats['significant_transitions']['under_represented']

    # transition_stats.json にはセル単位の調整残差は保存されていないので、
    # 度数行列から再計算する
    count_data = stats['basic_transition']['count_matrix']

    count_df = pd.DataFrame(0, index=before_states, columns=after_states)
    for bs in before_states:
        for a_s in after_states:
            count_df.loc[bs, a_s] = count_data.get(bs, {}).get(a_s, 0)

    observed = count_df.values.astype(float)
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    n = observed.sum()

    expected = row_totals * col_totals / n

    with np.errstate(divide='ignore', invalid='ignore'):
        adj_residuals = (observed - expected) / np.sqrt(
            expected * (1 - row_totals / n) * (1 - col_totals / n)
        )
    adj_residuals = np.nan_to_num(adj_residuals, nan=0.0, posinf=0.0, neginf=0.0)

    adj_res_df = pd.DataFrame(adj_residuals, index=before_states, columns=after_states)

    fig, ax = plt.subplots(figsize=(16, 10))

    # カスタムカラーマップ: 青(過少) → 白(中立) → 赤(過剰)
    sns.heatmap(
        adj_res_df, annot=True, fmt='.1f', cmap='RdBu_r',
        center=0, ax=ax, linewidths=0.5,
        vmin=-20, vmax=28,
        xticklabels=True, yticklabels=True,
    )

    title = '調整残差 (|z|>1.96: p<0.05, |z|>2.58: p<0.01)'
    ax.set_title(title, fontsize=14, fontproperties=jp_font if jp_font else None)
    ax.set_xlabel('after_state', fontsize=11)
    ax.set_ylabel('before_state', fontsize=11)
    apply_font(ax)

    # 有意水準の凡例テキスト追加
    fig.text(0.02, 0.02,
             '赤: 有意に過剰 (z>0)  |  青: 有意に過少 (z<0)  |  |z|>2.58: p<0.01',
             fontsize=9, fontproperties=jp_font if jp_font else None,
             ha='left', va='bottom')

    plt.tight_layout()
    fpath = os.path.join(VIS_DIR, 'adjusted_residuals_heatmap.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    保存: {fpath}")
    return fpath


# =============================================================================
# Plot 3: cluster_scatter.png
# =============================================================================

def plot_cluster_scatter(cluster_data):
    """MCA Dim1 vs Dim2 のクラスタリング散布図。"""
    print("  [3/4] cluster_scatter.png を生成中...")

    # MCA行座標を再計算（cluster_results.jsonにはrow coordsが含まれない）
    import prince

    records = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"    データ読み込み: {len(df)} 件")

    # MCA実行（元データ、5次元）
    df_mca = df[MCA_COLUMNS].copy()
    mca = prince.MCA(n_components=5, random_state=RANDOM_SEED)
    mca.fit(df_mca)
    row_coords = mca.row_coordinates(df_mca)

    # k=2 でクラスタリング
    km = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(row_coords.values)

    # 散布図
    fig, ax = plt.subplots(figsize=(14, 10))

    colors = ['#1f77b4', '#ff7f0e']
    cluster_names = ['Cluster 0 (メインストリーム)', 'Cluster 1 (低頻度カテゴリ)']

    for c in [0, 1]:
        mask = labels == c
        ax.scatter(
            row_coords.values[mask, 0],
            row_coords.values[mask, 1],
            c=colors[c], alpha=0.3, s=8,
            label=f'{cluster_names[c]} (n={mask.sum():,})',
        )

    # 重心をマーク
    for c in [0, 1]:
        mask = labels == c
        centroid_x = row_coords.values[mask, 0].mean()
        centroid_y = row_coords.values[mask, 1].mean()
        ax.scatter(centroid_x, centroid_y,
                   c=colors[c], edgecolors='black', s=200, linewidth=2,
                   marker='*', zorder=5)
        ax.annotate(
            f'C{c} centroid',
            (centroid_x, centroid_y),
            textcoords="offset points", xytext=(10, 10),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
        )

    title = 'MCA空間クラスタリング (k=2, Dim1 vs Dim2)'
    ax.set_title(title, fontsize=14, fontproperties=jp_font if jp_font else None)
    ax.set_xlabel(f'Dim 1 ({mca.percentage_of_variance_[0]:.1f}%)', fontsize=12)
    ax.set_ylabel(f'Dim 2 ({mca.percentage_of_variance_[1]:.1f}%)', fontsize=12)
    ax.legend(fontsize=11, loc='upper right',
              prop=jp_font if jp_font else None)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(VIS_DIR, 'cluster_scatter.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    保存: {fpath}")
    return fpath


# =============================================================================
# Plot 4: silhouette_elbow.png
# =============================================================================

def plot_silhouette_elbow(cluster_data):
    """シルエットスコア + エルボー法の2パネルプロット。"""
    print("  [4/4] silhouette_elbow.png を生成中...")

    kmeans_data = cluster_data['clustering_comparison']['kmeans']
    silhouettes = kmeans_data['silhouettes']

    # silhouettes は dict (str keys) で格納されている
    if isinstance(silhouettes, dict):
        ks = sorted([int(k) for k in silhouettes.keys()])
        sil_values = [silhouettes[str(k)] for k in ks]
    else:
        # list の場合は k=2..20 と対応
        ks = list(range(2, 2 + len(silhouettes)))
        sil_values = silhouettes

    # WCSSデータ (inertias) がcluster_results.jsonに含まれているか確認
    # 含まれていない場合はMCAから再計算
    inertias = None
    if 'inertias' in kmeans_data:
        inertias_data = kmeans_data['inertias']
        if isinstance(inertias_data, dict):
            inertias = [inertias_data[str(k)] for k in ks]
        else:
            inertias = inertias_data

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 左: シルエットスコア vs k
    ax1 = axes[0]
    ax1.plot(ks, sil_values, 'bo-', linewidth=2, markersize=8)

    # ベストkをハイライト
    best_k_idx = sil_values.index(max(sil_values))
    ax1.plot(ks[best_k_idx], sil_values[best_k_idx],
             'r*', markersize=20, zorder=5)
    ax1.annotate(
        f'Best k={ks[best_k_idx]}\n(sil={sil_values[best_k_idx]:.3f})',
        (ks[best_k_idx], sil_values[best_k_idx]),
        textcoords="offset points", xytext=(15, -15),
        fontsize=10, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='red'),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'),
    )

    title1 = 'シルエットスコア vs クラスタ数 k'
    ax1.set_title(title1, fontsize=14, fontproperties=jp_font if jp_font else None)
    ax1.set_xlabel('k (クラスタ数)', fontsize=12,
                   fontproperties=jp_font if jp_font else None)
    ax1.set_ylabel('シルエットスコア', fontsize=12,
                   fontproperties=jp_font if jp_font else None)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(ks)

    # 右: WCSS (エルボー法)
    ax2 = axes[1]
    if inertias is not None:
        ax2.plot(ks, inertias, 'go-', linewidth=2, markersize=8)
        title2 = 'WCSS (エルボー法) vs クラスタ数 k'
        ax2.set_title(title2, fontsize=14, fontproperties=jp_font if jp_font else None)
        ax2.set_xlabel('k (クラスタ数)', fontsize=12,
                       fontproperties=jp_font if jp_font else None)
        ax2.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(ks)
    else:
        # inertias が無い場合、MCAから再計算
        print("    WCSS データなし — MCAから再計算中...")
        import prince

        records = []
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
        df_mca = df[MCA_COLUMNS].copy()
        mca = prince.MCA(n_components=5, random_state=RANDOM_SEED)
        mca.fit(df_mca)
        coords = mca.row_coordinates(df_mca).values

        wcss_values = []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
            km.fit(coords)
            wcss_values.append(float(km.inertia_))

        ax2.plot(ks, wcss_values, 'go-', linewidth=2, markersize=8)
        title2 = 'WCSS (エルボー法) vs クラスタ数 k'
        ax2.set_title(title2, fontsize=14, fontproperties=jp_font if jp_font else None)
        ax2.set_xlabel('k (クラスタ数)', fontsize=12,
                       fontproperties=jp_font if jp_font else None)
        ax2.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(ks)

    plt.tight_layout()
    fpath = os.path.join(VIS_DIR, 'silhouette_elbow.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    保存: {fpath}")
    return fpath


# =============================================================================
# メイン
# =============================================================================

def main():
    print("=" * 70)
    print("Phase 2B 可視化生成 (品質レビュー修正 M1)")
    print("=" * 70)

    # データ読み込み
    stats = load_transition_stats()
    cluster_data = load_cluster_results()

    # 4つの可視化を生成
    files = []
    files.append(plot_transition_heatmap(stats))
    files.append(plot_adjusted_residuals(stats))
    files.append(plot_cluster_scatter(cluster_data))
    files.append(plot_silhouette_elbow(cluster_data))

    print("\n" + "=" * 70)
    print(f"完了: {len(files)} ファイルを生成")
    for f in files:
        size_kb = os.path.getsize(f) / 1024
        print(f"  {os.path.basename(f)}: {size_kb:.1f} KB")
    print("=" * 70)


if __name__ == '__main__':
    main()
