#!/usr/bin/env python3
"""
Phase 2B-2: MCA座標空間でのクラスタリング + 八卦タグとの事後的対応分析
=====================================================================

前提条件:
  - Phase 2Aレビューの条件1-2を最初に実行
  - 条件1: 元データ vs クリーンデータの列座標相関分析
  - 条件2: 使用データ・次元数の決定

入力: data/raw/cases.jsonl (13,060件, Read Only)
出力:
  - analysis/phase2/cluster_results.json      クラスタリング結果
  - analysis/phase2/visualizations/cluster_map.png
  - analysis/phase2/visualizations/parallel_coordinates.png

乱数シード: 42
"""

import json
import os
import warnings
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

import prince

warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    """NumPy型をJSON互換型に変換するエンコーダ。"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_numpy_types(obj):
    """再帰的にNumPy型をPython native型に変換（dictキーを含む）。"""
    if isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# --- パス設定 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'cases.jsonl')
OUTPUT_DIR = SCRIPT_DIR
VIS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')

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
    'before_state',
    'trigger_type',
    'action_type',
    'after_state',
    'pattern_type',
    'outcome',
    'scale',
]

# --- カテゴリ統合マッピング（Phase 2A mca_analysis.py から転載） ---
CATEGORY_MERGE_MAP = {
    'before_state': {
        '安定・停止': '安定・平和',
        '混乱・衰退': '混乱・カオス',
        '成長・拡大': '安定成長・成功',
        '拡大・繁栄': '安定成長・成功',
        '急成長・拡大': '安定成長・成功',
        '調和・繁栄': '安定成長・成功',
        '縮小安定・生存': '安定成長・成功',
        'V字回復・大成功': '安定成長・成功',
    },
    'trigger_type': {
        '内部矛盾・自壊': '内部崩壊',
        '自然推移・成熟': '自然推移',
        '拡大・過剰': '内部崩壊',
    },
    'action_type': {
        '拡大・攻め': '攻める・挑戦',
        '集中・拡大': '攻める・挑戦',
        '交流・発表': '対話・融合',
        '輝く・表現': '対話・融合',
        '捨てる・転換': '捨てる・撤退',
        '撤退・収縮': '捨てる・撤退',
        '撤退・縮小': '捨てる・撤退',
        '撤退・逃げる': '捨てる・撤退',
        '分散・スピンオフ': '分散・探索',
        '分散・独立': '分散・探索',
        '分散・多角化': '分散・探索',
        '分散する・独立する': '分散・探索',
        '逃げる・分散': '逃げる・放置',
        '逃げる・守る': '逃げる・放置',
    },
    'after_state': {
        'V字回復・大成功': '持続成長・大成功',
        '成長・拡大': '持続成長・大成功',
        '拡大・繁栄': '持続成長・大成功',
        '消滅・破綻': '崩壊・消滅',
        '現状維持・延命': '縮小安定・生存',
        '安定・停止': '安定・平和',
        '喜び・交流': '安定成長・成功',
        '混乱・カオス': '混乱・衰退',
        '迷走・混乱': '混乱・衰退',
        'どん底・危機': '混乱・衰退',
        '分岐・様子見': '混乱・衰退',
        '成長痛': '混乱・衰退',
    },
}


# =============================================================================
# データ読み込み
# =============================================================================

def load_data(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"読み込み完了: {len(df)} 件")
    return df


def apply_category_merge(df):
    df_clean = df.copy()
    for col, mapping in CATEGORY_MERGE_MAP.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(mapping)
    return df_clean


# =============================================================================
# 条件1: 元データ vs クリーンデータの列座標相関分析
# =============================================================================

def condition1_column_correlation(df_raw, df_clean):
    """共通カテゴリについて、元データDim1-2とクリーンデータDim1-2の列座標のPearson相関を計算。"""
    print("\n" + "=" * 70)
    print("条件1: 元データ vs クリーンデータの列座標相関分析")
    print("=" * 70)

    # 元データMCA
    df_raw_mca = df_raw[MCA_COLUMNS].copy()
    mca_raw = prince.MCA(n_components=5, random_state=RANDOM_SEED)
    mca_raw.fit(df_raw_mca)
    col_coords_raw = mca_raw.column_coordinates(df_raw_mca)

    # クリーンデータMCA
    df_clean_mca = df_clean[MCA_COLUMNS].copy()
    mca_clean = prince.MCA(n_components=5, random_state=RANDOM_SEED)
    mca_clean.fit(df_clean_mca)
    col_coords_clean = mca_clean.column_coordinates(df_clean_mca)

    # 共通カテゴリの特定
    raw_cats = set(col_coords_raw.index)
    clean_cats = set(col_coords_clean.index)
    common_cats = sorted(raw_cats & clean_cats)
    print(f"  元データのカテゴリ数: {len(raw_cats)}")
    print(f"  クリーンデータのカテゴリ数: {len(clean_cats)}")
    print(f"  共通カテゴリ数: {len(common_cats)}")

    # Pearson相関
    correlations = {}
    for dim in range(2):
        raw_coords = col_coords_raw.loc[common_cats, dim].values
        clean_coords = col_coords_clean.loc[common_cats, dim].values

        # 符号の反転チェック（MCAは符号が不定）
        r_pos, _ = scipy_stats.pearsonr(raw_coords, clean_coords)
        r_neg, _ = scipy_stats.pearsonr(raw_coords, -clean_coords)

        if abs(r_neg) > abs(r_pos):
            r = r_neg
            sign_flip = True
        else:
            r = r_pos
            sign_flip = False

        correlations[f'Dim{dim+1}'] = {
            'pearson_r': round(float(r), 6),
            'abs_r': round(abs(float(r)), 6),
            'sign_flipped': sign_flip,
            'stable': abs(r) > 0.8,
        }
        print(f"  Dim{dim+1}: r = {r:.4f} (|r| = {abs(r):.4f}, "
              f"{'STABLE' if abs(r) > 0.8 else 'UNSTABLE'}"
              f"{', sign-flipped' if sign_flip else ''})")

    all_stable = all(c['stable'] for c in correlations.values())
    print(f"\n  結論: Dim1-2は{'安定' if all_stable else '不安定'}")

    return {
        'common_categories': len(common_cats),
        'correlations': correlations,
        'all_stable': all_stable,
        'conclusion': (
            'Dim1-2 are stable across raw and clean data (|r| > 0.8 for both dimensions). '
            'Low-frequency category merging does not distort the primary structure.'
            if all_stable
            else 'Dim1-2 show instability; interpret with caution.'
        ),
    }, mca_raw, mca_clean, col_coords_raw, col_coords_clean


# =============================================================================
# 条件2: データ選択方針の明確化
# =============================================================================

def condition2_data_selection(correlation_result):
    """相関分析の結果に基づき、使用データと次元数を決定。"""
    print("\n" + "=" * 70)
    print("条件2: Phase 2Bのデータ選択方針")
    print("=" * 70)

    all_stable = correlation_result['all_stable']

    if all_stable:
        decision = {
            'data': 'clean (low-frequency categories merged)',
            'n_dimensions': 2,
            'rationale': (
                'Dim1-2 are stable (Pearson |r| > 0.8) across raw and clean data. '
                'Clean data recommended because: '
                '(1) Dim3-5 in raw data are dominated by low-frequency categories (<20 cases), '
                'likely artifacts of MCA scaling; '
                '(2) Clean data scree elbow = 2, Benzecri cum80% = 4, recommending 2 dims; '
                '(3) 2 dimensions capture the primary structure without noise from rare categories.'
            ),
        }
    else:
        decision = {
            'data': 'raw (original categories)',
            'n_dimensions': 5,
            'rationale': (
                'Dim1-2 are unstable across raw and clean data. '
                'Using raw data with 5 dimensions (scree elbow) to preserve all information.'
            ),
        }

    print(f"  使用データ: {decision['data']}")
    print(f"  次元数: {decision['n_dimensions']}")
    print(f"  根拠: {decision['rationale'][:120]}...")

    return decision


# =============================================================================
# 条件3: Benzecri 注記
# =============================================================================

def condition3_benzecri_note():
    """dimension_report.jsonにBenzecri注記を追加。"""
    print("\n" + "=" * 70)
    print("条件3: Benzecri「30次元 > 0」の注記追加")
    print("=" * 70)

    dim_report_path = os.path.join(OUTPUT_DIR, 'dimension_report.json')
    with open(dim_report_path, 'r', encoding='utf-8') as f:
        dim_report = json.load(f)

    note = (
        'The minimum eigenvalue (0.14288) exceeds the Kaiser/Benzecri threshold (1/K = 0.14286) '
        'by only 0.00002. This margin is smaller than floating-point precision concerns. '
        'Consequently, the Benzecri upper-bound criterion (dims with corrected eigenvalue > 0) '
        'has no discriminating power: it classifies all 30 computed dimensions as significant. '
        'Only the Benzecri cumulative 80% criterion (6 dims) retains utility.'
    )

    dim_report['dimension_decision']['benzecri']['minimum_eigenvalue_note'] = note
    dim_report['dimension_decision']['benzecri']['min_eigenvalue'] = 0.14288
    dim_report['dimension_decision']['benzecri']['kaiser_threshold'] = 0.14286
    dim_report['dimension_decision']['benzecri']['margin'] = 0.00002

    with open(dim_report_path, 'w', encoding='utf-8') as f:
        json.dump(dim_report, f, ensure_ascii=False, indent=2)

    print(f"  注記追加完了: {dim_report_path}")
    print(f"  内容: {note[:100]}...")

    return note


# =============================================================================
# クラスタリング
# =============================================================================

def run_mca_and_get_coordinates(df, n_dims):
    """MCAを実行し、行座標を返す。"""
    df_mca = df[MCA_COLUMNS].copy()
    mca = prince.MCA(n_components=n_dims, random_state=RANDOM_SEED)
    mca.fit(df_mca)
    row_coords = mca.row_coordinates(df_mca)
    return row_coords, mca


def kmeans_analysis(coords, k_range=range(2, 21)):
    """k-meansクラスタリング: シルエットスコア + エルボー法。"""
    inertias = []
    silhouettes = []
    best_k = None
    best_sil = -1
    best_labels = None

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(coords)
        inertias.append(float(km.inertia_))

        sil = silhouette_score(coords, labels)
        silhouettes.append(float(sil))

        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels

        if k <= 10 or k == best_k:
            print(f"    k={k:2d}: silhouette={sil:.4f}, inertia={km.inertia_:.1f}")

    return {
        'k_range': list(k_range),
        'silhouettes': silhouettes,
        'inertias': inertias,
        'best_k': best_k,
        'best_silhouette': best_sil,
        'best_labels': best_labels,
    }


def gap_statistic(coords, k_range=range(2, 21), n_references=20):
    """Gap統計量の計算。"""
    gaps = []
    gap_stds = []

    for k in k_range:
        # 実データ
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        km.fit(coords)
        log_wk = np.log(km.inertia_)

        # ランダム参照分布
        log_wk_refs = []
        for b in range(n_references):
            rng = np.random.RandomState(RANDOM_SEED + b + k * 100)
            random_data = rng.uniform(
                low=coords.min(axis=0).values,
                high=coords.max(axis=0).values,
                size=coords.shape
            )
            km_ref = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=5)
            km_ref.fit(random_data)
            log_wk_refs.append(np.log(km_ref.inertia_))

        gap = np.mean(log_wk_refs) - log_wk
        gap_std = np.std(log_wk_refs) * np.sqrt(1 + 1.0 / n_references)
        gaps.append(float(gap))
        gap_stds.append(float(gap_std))

    # Gap統計量の最適k: gap(k) >= gap(k+1) - s(k+1)
    optimal_k = None
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - gap_stds[i + 1]:
            optimal_k = list(k_range)[i]
            break

    if optimal_k is None:
        optimal_k = list(k_range)[np.argmax(gaps)]

    return {
        'k_range': list(k_range),
        'gaps': gaps,
        'gap_stds': gap_stds,
        'optimal_k': optimal_k,
    }


def dbscan_analysis(coords, eps_range=None):
    """DBSCANクラスタリング。"""
    if eps_range is None:
        # データのスケールに基づいてeps候補を決定
        distances = pdist(coords.values[:2000])  # サンプリングして計算
        p10, p50, p90 = np.percentile(distances, [10, 50, 90])
        eps_range = np.linspace(p10 * 0.5, p50, 10)

    results = []
    best_result = None
    best_sil = -1

    for eps in eps_range:
        for min_samples in [5, 10, 20]:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(coords)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()

            if n_clusters >= 2 and n_noise < len(labels) * 0.5:
                mask = labels != -1
                if mask.sum() > n_clusters:
                    sil = silhouette_score(coords[mask], labels[mask])
                else:
                    sil = -1
            else:
                sil = -1

            entry = {
                'eps': round(float(eps), 4),
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': int(n_noise),
                'noise_pct': round(float(n_noise) / len(labels) * 100, 2),
                'silhouette': round(float(sil), 4) if sil > -1 else None,
            }
            results.append(entry)

            if sil > best_sil:
                best_sil = sil
                best_result = entry
                best_result['labels'] = labels

    return {
        'parameter_search': results[:30],  # 上位30件
        'best': best_result if best_result else {'n_clusters': 0, 'note': 'No valid clustering found'},
    }


def hierarchical_analysis(coords, k_candidates=None):
    """階層的クラスタリング。"""
    # サンプルサイズが大きい場合はサンプリング
    if len(coords) > 5000:
        sample_idx = np.random.RandomState(RANDOM_SEED).choice(
            len(coords), 5000, replace=False
        )
        coords_sample = coords.iloc[sample_idx]
        is_sampled = True
    else:
        coords_sample = coords
        is_sampled = False

    Z = linkage(coords_sample.values, method='ward')

    if k_candidates is None:
        k_candidates = list(range(2, 11))

    results = {}
    best_k = None
    best_sil = -1

    for k in k_candidates:
        labels = fcluster(Z, k, criterion='maxclust')
        if len(set(labels)) >= 2:
            sil = silhouette_score(coords_sample, labels)
            results[k] = {
                'silhouette': round(float(sil), 4),
                'cluster_sizes': dict(Counter(labels)),
            }
            if sil > best_sil:
                best_sil = sil
                best_k = k

    return {
        'is_sampled': is_sampled,
        'sample_size': len(coords_sample),
        'method': 'ward',
        'results_by_k': results,
        'best_k': best_k,
        'best_silhouette': round(float(best_sil), 4),
        'linkage_matrix': Z,
    }


def interpret_clusters(df, labels, coords, n_dims):
    """各クラスタの代表的なカテゴリプロファイルを作成。"""
    df_with_labels = df.copy()
    df_with_labels['cluster'] = labels

    profiles = {}
    for cl in sorted(set(labels)):
        if cl == -1:
            continue
        mask = df_with_labels['cluster'] == cl
        cl_data = df_with_labels[mask]

        profile = {
            'size': int(mask.sum()),
            'pct': round(float(mask.sum()) / len(df) * 100, 2),
        }

        # 各変数の最頻値
        for col in MCA_COLUMNS:
            if col in cl_data.columns:
                vc = cl_data[col].value_counts(normalize=True).head(3)
                profile[col] = {str(k): round(float(v), 4) for k, v in vc.items()}

        # クラスタの重心（MCA座標）
        cl_coords = coords.iloc[mask.values] if hasattr(mask, 'values') else coords[mask]
        centroid = cl_coords.mean().tolist()
        profile['centroid'] = [round(float(c), 4) for c in centroid]

        profiles[int(cl)] = profile

    return profiles


def hexagram_association(df, labels):
    """クラスタラベルと before_hex/after_hex の分割表 + カイ二乗検定 / Cramer's V。"""
    df_with_labels = df.copy()
    df_with_labels['cluster'] = labels

    results = {}
    for hex_col in ['before_hex', 'after_hex']:
        if hex_col not in df_with_labels.columns:
            continue

        # -1 (noise) を除外
        valid = df_with_labels[df_with_labels['cluster'] != -1]
        ct = pd.crosstab(valid['cluster'], valid[hex_col])

        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p, dof, expected = scipy_stats.chi2_contingency(ct)
            n = ct.values.sum()
            k_min = min(ct.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * k_min)) if k_min > 0 else 0

            results[hex_col] = {
                'contingency_table_shape': list(ct.shape),
                'chi2': round(float(chi2), 2),
                'p_value': float(p),
                'dof': int(dof),
                'cramers_v': round(float(cramers_v), 4),
                'association_strength': (
                    'strong' if cramers_v > 0.3
                    else 'moderate' if cramers_v > 0.15
                    else 'weak' if cramers_v > 0.05
                    else 'negligible'
                ),
                'significant': p < 0.001,
                'note': (
                    'Reports association only. Isomorphism determination '
                    'is reserved for Phase 3 isomorphism-checker.'
                ),
            }

    return results


# =============================================================================
# 可視化
# =============================================================================

def plot_cluster_map(coords, labels, title='Cluster Map (MCA Dim1 vs Dim2)'):
    """クラスタリング結果の2次元散布図。"""
    fig, ax = plt.subplots(figsize=(14, 10))

    unique_labels = sorted(set(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_labels), 2)))

    for i, cl in enumerate(unique_labels):
        mask = labels == cl
        if cl == -1:
            ax.scatter(coords.iloc[mask, 0], coords.iloc[mask, 1],
                      c='gray', marker='x', s=5, alpha=0.3, label='Noise')
        else:
            ax.scatter(coords.iloc[mask, 0], coords.iloc[mask, 1],
                      c=[colors[i]], s=8, alpha=0.4, label=f'Cluster {cl}')

    ax.set_xlabel('Dim 1', fontsize=12)
    ax.set_ylabel('Dim 2', fontsize=12)
    ax.set_title(title, fontsize=14,
                 fontproperties=jp_font if jp_font else None)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(VIS_DIR, 'cluster_map.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {fname}")


def plot_silhouette_comparison(km_result, k_range_display=range(2, 16)):
    """シルエットスコア + エルボー法の比較プロット。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    k_list = km_result['k_range']
    display_idx = [i for i, k in enumerate(k_list) if k in k_range_display]
    k_display = [k_list[i] for i in display_idx]
    sil_display = [km_result['silhouettes'][i] for i in display_idx]
    inertia_display = [km_result['inertias'][i] for i in display_idx]

    # シルエットスコア
    ax1 = axes[0]
    ax1.plot(k_display, sil_display, 'bo-', markersize=6)
    ax1.axvline(x=km_result['best_k'], color='r', linestyle='--', alpha=0.7,
                label=f"Best k={km_result['best_k']}")
    ax1.set_xlabel('k', fontsize=11)
    ax1.set_ylabel('Silhouette Score', fontsize=11)
    ax1.set_title('Silhouette Score', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # エルボー法
    ax2 = axes[1]
    ax2.plot(k_display, inertia_display, 'go-', markersize=6)
    ax2.set_xlabel('k', fontsize=11)
    ax2.set_ylabel('Inertia (WSS)', fontsize=11)
    ax2.set_title('Elbow Method', fontsize=13)
    ax2.grid(True, alpha=0.3)

    # シルエットの差分（変化率）
    ax3 = axes[2]
    if len(sil_display) > 1:
        diffs = [sil_display[i+1] - sil_display[i] for i in range(len(sil_display)-1)]
        ax3.bar(k_display[1:], diffs, color='purple', alpha=0.6)
        ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_xlabel('k', fontsize=11)
    ax3.set_ylabel('Silhouette Change', fontsize=11)
    ax3.set_title('Silhouette Score Change', fontsize=13)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(VIS_DIR, 'silhouette_elbow.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {fname}")


def plot_parallel_coordinates(df, labels, sample_n=2000):
    """パラレル座標プロット。"""
    df_plot = df.copy()
    df_plot['cluster'] = labels

    # -1 (noise) を除外
    df_plot = df_plot[df_plot['cluster'] != -1]

    if len(df_plot) > sample_n:
        df_plot = df_plot.sample(sample_n, random_state=RANDOM_SEED)

    # カテゴリを数値に変換
    cols_to_plot = ['before_state', 'trigger_type', 'action_type',
                    'after_state', 'pattern_type', 'outcome']
    encodings = {}
    for col in cols_to_plot:
        cats = sorted(df_plot[col].unique())
        enc = {cat: i for i, cat in enumerate(cats)}
        encodings[col] = enc
        df_plot[col + '_num'] = df_plot[col].map(enc)

    fig, ax = plt.subplots(figsize=(16, 8))
    num_cols = [col + '_num' for col in cols_to_plot]

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(set(labels)) - (1 if -1 in labels else 0), 2)))

    for cl_idx, cl in enumerate(sorted(set(df_plot['cluster'].unique()))):
        cl_data = df_plot[df_plot['cluster'] == cl]
        for _, row in cl_data.iterrows():
            values = [row[nc] for nc in num_cols]
            ax.plot(range(len(num_cols)), values, c=colors[cl_idx], alpha=0.05, linewidth=0.5)

    # クラスタ中央値も太線で
    for cl_idx, cl in enumerate(sorted(set(df_plot['cluster'].unique()))):
        cl_data = df_plot[df_plot['cluster'] == cl]
        medians = [cl_data[nc].median() for nc in num_cols]
        ax.plot(range(len(num_cols)), medians, c=colors[cl_idx], alpha=0.9,
                linewidth=2.5, label=f'Cluster {cl} median')

    ax.set_xticks(range(len(cols_to_plot)))
    ax.set_xticklabels(cols_to_plot, fontsize=10, rotation=20,
                       fontproperties=jp_font if jp_font else None)
    ax.set_title('Parallel Coordinates by Cluster', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Y軸のカテゴリラベル
    # 最大カテゴリ数の変数でY範囲を設定
    max_cats = max(len(enc) for enc in encodings.values())
    ax.set_ylim(-0.5, max_cats - 0.5)

    plt.tight_layout()
    fname = os.path.join(VIS_DIR, 'parallel_coordinates.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {fname}")


# =============================================================================
# メイン
# =============================================================================

def main():
    print("=" * 70)
    print("Phase 2B-2: クラスタリング分析")
    print("=" * 70)

    # データ読み込み
    df_raw = load_data(DATA_PATH)
    df_clean = apply_category_merge(df_raw)

    # ===== 条件1: 列座標相関 =====
    corr_result, mca_raw, mca_clean, col_raw, col_clean = condition1_column_correlation(
        df_raw, df_clean
    )

    # ===== 条件2: データ選択方針 =====
    data_decision = condition2_data_selection(corr_result)

    # ===== 条件3: Benzecri注記 =====
    benzecri_note = condition3_benzecri_note()

    # ===== 使用データの決定 =====
    use_clean = 'clean' in data_decision['data']
    n_dims = data_decision['n_dimensions']
    df_use = df_clean if use_clean else df_raw

    print(f"\n使用データ: {'クリーン' if use_clean else '元データ'}, {n_dims}次元")

    # ===== MCA行座標の取得 =====
    print("\n--- MCA行座標の取得 ---")
    row_coords, mca_obj = run_mca_and_get_coordinates(df_use, n_dims)
    print(f"  行座標のshape: {row_coords.shape}")

    # ===== k-means =====
    print("\n--- k-means クラスタリング ---")
    km_result = kmeans_analysis(row_coords)
    print(f"  最適k: {km_result['best_k']} (silhouette: {km_result['best_silhouette']:.4f})")

    # ===== Gap統計量 =====
    print("\n--- Gap統計量 ---")
    gap_result = gap_statistic(row_coords, k_range=range(2, 16), n_references=20)
    print(f"  Gap最適k: {gap_result['optimal_k']}")

    # ===== DBSCAN =====
    print("\n--- DBSCAN クラスタリング ---")
    dbscan_result = dbscan_analysis(row_coords)
    if 'labels' in dbscan_result.get('best', {}):
        print(f"  最良パラメータ: eps={dbscan_result['best']['eps']}, "
              f"min_samples={dbscan_result['best']['min_samples']}")
        print(f"  クラスタ数: {dbscan_result['best']['n_clusters']}, "
              f"ノイズ: {dbscan_result['best']['n_noise']}件 ({dbscan_result['best']['noise_pct']}%)")
    else:
        print("  有効なクラスタリングが見つかりませんでした")

    # ===== 階層的クラスタリング =====
    print("\n--- 階層的クラスタリング ---")
    hier_result = hierarchical_analysis(row_coords, k_candidates=list(range(2, 16)))
    print(f"  最適k: {hier_result['best_k']} (silhouette: {hier_result['best_silhouette']:.4f})")

    # ===== クラスタ数の最終決定 =====
    print("\n--- クラスタ数の最終決定 ---")
    k_candidates = {
        'kmeans_silhouette': km_result['best_k'],
        'gap_statistic': gap_result['optimal_k'],
        'hierarchical_ward': hier_result['best_k'],
    }
    print(f"  各手法の推奨: {k_candidates}")

    # 多数決 + シルエット重視
    all_ks = list(k_candidates.values())
    k_counter = Counter(all_ks)
    final_k = k_counter.most_common(1)[0][0]
    print(f"  最終決定: k={final_k}")

    # 最終k-meansで再実行
    km_final = KMeans(n_clusters=final_k, random_state=RANDOM_SEED, n_init=10)
    final_labels = km_final.fit_predict(row_coords)
    final_sil = silhouette_score(row_coords, final_labels)
    print(f"  最終シルエットスコア: {final_sil:.4f}")

    # ===== クラスタの解釈 =====
    print("\n--- クラスタの解釈 ---")
    profiles = interpret_clusters(df_use, final_labels, row_coords, n_dims)
    for cl, profile in profiles.items():
        print(f"\n  Cluster {cl}: {profile['size']}件 ({profile['pct']}%)")
        for col in ['before_state', 'action_type', 'after_state', 'outcome']:
            if col in profile:
                top = list(profile[col].items())[:2]
                top_str = ', '.join([f"{k}({v:.0%})" for k, v in top])
                print(f"    {col}: {top_str}")

    # ===== 八卦タグとの事後的対応 =====
    print("\n--- 八卦タグとの事後的対応分析 ---")
    hex_assoc = hexagram_association(df_raw, final_labels)
    for hex_col, result in hex_assoc.items():
        print(f"  {hex_col}:")
        print(f"    chi2={result['chi2']:.1f}, p={result['p_value']:.2e}, "
              f"Cramer's V={result['cramers_v']:.4f} ({result['association_strength']})")

    # ===== クラスタ間の遷移パターンの違い =====
    print("\n--- クラスタ間の遷移パターンの違い ---")
    df_use_with_cl = df_use.copy()
    df_use_with_cl['cluster'] = final_labels
    transition_by_cluster = {}
    for cl in sorted(set(final_labels)):
        cl_data = df_use_with_cl[df_use_with_cl['cluster'] == cl]
        if len(cl_data) > 0:
            ct_cl = pd.crosstab(cl_data['before_state'], cl_data['after_state'])
            # 最頻遷移TOP5
            pairs = []
            for bs in ct_cl.index:
                for as_ in ct_cl.columns:
                    if ct_cl.loc[bs, as_] > 0:
                        pairs.append((bs, as_, int(ct_cl.loc[bs, as_])))
            pairs.sort(key=lambda x: -x[2])
            transition_by_cluster[int(cl)] = [
                {'before': p[0], 'after': p[1], 'count': p[2]} for p in pairs[:5]
            ]
            print(f"\n  Cluster {cl} TOP5 遷移:")
            for p in pairs[:5]:
                print(f"    {p[0]} -> {p[1]}: {p[2]}件")

    # ===== 可視化 =====
    print("\n--- 可視化 ---")
    plot_cluster_map(row_coords, final_labels)
    plot_silhouette_comparison(km_result)
    plot_parallel_coordinates(df_use, final_labels)

    # ===== 結果保存 =====
    print("\n--- 結果保存 ---")

    # linkage matrixは保存しない（大きすぎる）
    hier_save = {k: v for k, v in hier_result.items() if k != 'linkage_matrix'}

    # DBSCANのlabelsも除外
    dbscan_save = {
        'parameter_search': dbscan_result['parameter_search'],
        'best': {k: v for k, v in dbscan_result.get('best', {}).items() if k != 'labels'}
            if dbscan_result.get('best') else dbscan_result.get('best'),
    }

    output = {
        'preconditions': {
            'condition1_column_correlation': corr_result,
            'condition2_data_selection': data_decision,
            'condition3_benzecri_note': benzecri_note,
        },

        'data_used': {
            'type': 'clean' if use_clean else 'raw',
            'n_records': len(df_use),
            'n_dimensions': n_dims,
            'category_merge_applied': use_clean,
        },

        'clustering_comparison': {
            'kmeans': {
                'best_k': km_result['best_k'],
                'best_silhouette': km_result['best_silhouette'],
                'silhouettes': {k: round(s, 4) for k, s in zip(km_result['k_range'], km_result['silhouettes'])},
            },
            'gap_statistic': {
                'optimal_k': gap_result['optimal_k'],
                'gaps': {k: round(g, 4) for k, g in zip(gap_result['k_range'], gap_result['gaps'])},
            },
            'dbscan': dbscan_save,
            'hierarchical': hier_save,
        },

        'final_clustering': {
            'method': 'k-means',
            'k': final_k,
            'silhouette_score': round(float(final_sil), 4),
            'k_selection_method': 'majority vote across 3 methods (k-means silhouette, gap statistic, hierarchical ward)',
            'k_candidates': k_candidates,
        },

        'cluster_profiles': profiles,

        'hexagram_association': hex_assoc,

        'transition_by_cluster': transition_by_cluster,

        'cluster_sizes': {int(k): int(v) for k, v in Counter(final_labels).items()},
    }

    # NumPy型をPython native型に再帰的変換（キーを含む）
    output = convert_numpy_types(output)

    out_path = os.path.join(OUTPUT_DIR, 'cluster_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"保存: {out_path}")

    print("\n" + "=" * 70)
    print("Phase 2B-2 完了")
    print("=" * 70)

    return output


if __name__ == '__main__':
    main()
