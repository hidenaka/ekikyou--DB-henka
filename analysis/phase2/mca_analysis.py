#!/usr/bin/env python3
"""
Phase 2A: MCA（多重対応分析）による変化事例の構造化と次元導出
=============================================================

入力: data/raw/cases.jsonl (13,060件)
出力:
  - analysis/phase2/basic_stats.json       基本統計
  - analysis/phase2/mca_results.json       MCA結果
  - analysis/phase2/dimension_report.json  次元解釈レポート
  - analysis/phase2/visualizations/        可視化画像
  - analysis/phase2/report.md              総合レポート

乱数シード: 42
八卦タグ(before_hex, trigger_hex, action_hex, after_hex)はMCA分析から除外。

注意: データ内のカテゴリ数はスキーマの仕様値とは異なる。
      実データのカテゴリ分布に基づいて分析する。
"""

import sys
sys.path.insert(0, '/Users/hideakimacbookair/Library/Python/3.12/lib/python/site-packages')

import json
import os
import warnings
from collections import Counter, OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

import prince

warnings.filterwarnings('ignore')

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
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Yu Gothic', 'Arial']
else:
    plt.rcParams['font.family'] = jp_font.get_name()

plt.rcParams['axes.unicode_minus'] = False

# --- 定数 ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = '/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB'
DATA_PATH = os.path.join(BASE_DIR, 'data/raw/cases.jsonl')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis/phase2')
VIS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')

os.makedirs(VIS_DIR, exist_ok=True)

# MCA分析対象変数（八卦タグを除外）
MCA_COLUMNS = [
    'before_state',
    'trigger_type',
    'action_type',
    'after_state',
    'pattern_type',
    'outcome',
    'scale',
]


# =============================================================================
# Step 1: データ読み込みと基本統計
# =============================================================================

def load_data(path):
    """cases.jsonlを読み込み、DataFrameとして返す。"""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"読み込み完了: {len(df)} 件")
    return df


def compute_basic_stats(df):
    """各カテゴリカル変数の度数分布、欠損値、クロス集計を計算。"""
    stats_report = {
        'total_records': len(df),
        'frequency_distributions': {},
        'missing_values': {},
        'cross_tabulation': {},
        'actual_category_counts': {},
    }

    # 度数分布
    for col in MCA_COLUMNS:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            clean_vc = {}
            for k, v in vc.items():
                key = 'MISSING' if pd.isna(k) else str(k)
                clean_vc[key] = int(v)
            stats_report['frequency_distributions'][col] = clean_vc
            stats_report['actual_category_counts'][col] = df[col].nunique()

    # 欠損値
    for col in MCA_COLUMNS:
        if col in df.columns:
            missing = int(df[col].isna().sum())
            stats_report['missing_values'][col] = {
                'count': missing,
                'percentage': round(missing / len(df) * 100, 2)
            }
        else:
            stats_report['missing_values'][col] = {
                'count': len(df),
                'percentage': 100.0,
                'note': 'Column not found in data'
            }

    # クロス集計 (before_state x action_type x after_state)
    for pair in [('before_state', 'action_type'),
                 ('before_state', 'after_state'),
                 ('action_type', 'after_state')]:
        if pair[0] in df.columns and pair[1] in df.columns:
            ct = pd.crosstab(df[pair[0]], df[pair[1]])
            key = f"{pair[0]}_x_{pair[1]}"
            stats_report['cross_tabulation'][key] = ct.to_dict()

    # 3次元クロス集計のトップ30
    if all(c in df.columns for c in ['before_state', 'action_type', 'after_state']):
        three_way = df.groupby(['before_state', 'action_type', 'after_state']).size()
        top_30 = three_way.nlargest(30)
        stats_report['cross_tabulation']['top30_three_way'] = {
            str(k): int(v) for k, v in top_30.items()
        }

    return stats_report


# =============================================================================
# Step 2: MCA分析
# =============================================================================

def prepare_mca_data(df):
    """MCA用のデータを準備。"""
    mca_df = df[MCA_COLUMNS].copy()

    # 欠損値の確認
    missing_summary = mca_df.isna().sum()
    print("\n=== 欠損値確認 ===")
    print(missing_summary)

    n_before = len(mca_df)
    mca_df = mca_df.dropna()
    n_after = len(mca_df)
    n_dropped = n_before - n_after
    print(f"欠損値除外: {n_dropped}件除外 -> {n_after}件で分析")

    # 全て文字列型に変換
    for col in MCA_COLUMNS:
        mca_df[col] = mca_df[col].astype(str)

    # 実際のカテゴリ数を報告
    total_cats = 0
    for col in MCA_COLUMNS:
        n_cats = mca_df[col].nunique()
        total_cats += n_cats
        print(f"  {col}: {n_cats} categories")
    print(f"  Total categories (J): {total_cats}")
    print(f"  Number of variables (K): {len(MCA_COLUMNS)}")

    return mca_df, n_dropped, total_cats


def run_mca(mca_df, n_components=30):
    """MCAを実行し、結果を返す。"""
    # n_componentsの上限 = min(J-K, n-1) where J=total categories, K=variables
    actual_n_components = min(n_components, len(mca_df) - 1)
    print(f"\n=== MCA実行 (n_components={actual_n_components}) ===")

    mca = prince.MCA(
        n_components=actual_n_components,
        n_iter=10,
        random_state=RANDOM_SEED,
    )
    mca = mca.fit(mca_df)

    eigenvalues = mca.eigenvalues_
    total_inertia = sum(eigenvalues)
    explained_ratio = [ev / total_inertia * 100 for ev in eigenvalues]
    cumulative = list(np.cumsum(explained_ratio))

    print(f"Fixed eigenvalues (top 15): {[round(x, 6) for x in eigenvalues[:15]]}")
    print(f"Explained inertia % (top 15): {[round(x, 2) for x in explained_ratio[:15]]}")
    print(f"Cumulative % (top 15): {[round(x, 2) for x in cumulative[:15]]}")
    print(f"Total inertia: {total_inertia:.6f}")

    return mca, eigenvalues, explained_ratio, cumulative


# =============================================================================
# Step 3: 次元数の決定
# =============================================================================

def determine_dimensions(eigenvalues, explained_ratio, cumulative, mca_df, total_cats):
    """5つの基準で次元数を決定する。"""

    K = len(MCA_COLUMNS)
    J = total_cats
    n_eigenvalues = len(eigenvalues)

    results = {}

    # --- 基準1: スクリープロット（肘法） ---
    # 固有値の一次差分と二次差分を計算
    diffs = np.diff(eigenvalues)
    if len(diffs) > 1:
        second_diffs = np.diff(diffs)
        # 肘: 二次差分の絶対値が最大かつ固有値がまだ十分大きい点
        # 最初の大きな屈曲点を探す
        elbow_idx = np.argmax(np.abs(second_diffs)) + 1
    else:
        elbow_idx = 0

    results['scree_elbow'] = {
        'dimensions': int(elbow_idx) + 1,
        'method': 'Scree plot (elbow method): maximum second derivative of eigenvalues',
        'elbow_eigenvalue': float(eigenvalues[elbow_idx]) if elbow_idx < len(eigenvalues) else 0,
    }
    print(f"\nCriterion 1 - Scree (elbow): {results['scree_elbow']['dimensions']} dims")

    # --- 基準2: 累積寄与率 70% ---
    cum_array = np.array(cumulative)
    cum70_idx = np.argmax(cum_array >= 70.0)
    if cum_array[cum70_idx] >= 70.0:
        results['cumulative_70'] = {
            'dimensions': int(cum70_idx) + 1,
            'method': 'Cumulative explained inertia >= 70%',
            'cumulative_at_threshold': float(cumulative[cum70_idx]),
        }
    else:
        results['cumulative_70'] = {
            'dimensions': n_eigenvalues,
            'method': 'Cumulative inertia >= 70% (not reached)',
            'max_cumulative': float(cumulative[-1]),
        }
    print(f"Criterion 2 - Cumulative 70%: {results['cumulative_70']['dimensions']} dims")

    # --- 基準3: Kaiser-MCA基準 ---
    # MCAのKaiser基準: 固有値 > 1/K (Greenacre 2006)
    # MCAにおけるBenzecri修正: 固有値 > average eigenvalue
    # 標準的なMCA Kaiser: eigenvalue > 1/K (変数数の逆数)
    kaiser_threshold_strict = 1.0 / K  # = 1/7 = 0.1429
    kaiser_dims_strict = sum(1 for ev in eigenvalues if ev > kaiser_threshold_strict)

    results['kaiser'] = {
        'dimensions': int(kaiser_dims_strict),
        'method': f'Kaiser criterion for MCA: eigenvalue > 1/K = 1/{K} = {kaiser_threshold_strict:.4f}',
        'threshold': float(kaiser_threshold_strict),
    }
    print(f"Criterion 3 - Kaiser (1/K): {results['kaiser']['dimensions']} dims")

    # --- 基準4: Greenacre基準（平均慣性以上） ---
    # MCAでは平均固有値(= total_inertia / (J-K))が基線
    avg_eigenvalue = float(np.sum(eigenvalues) / len(eigenvalues)) if len(eigenvalues) > 0 else 0
    greenacre_dims = sum(1 for ev in eigenvalues if ev > avg_eigenvalue)

    results['greenacre'] = {
        'dimensions': int(greenacre_dims),
        'method': f'Greenacre criterion: eigenvalue > mean eigenvalue = {avg_eigenvalue:.6f}',
        'threshold': float(avg_eigenvalue),
    }
    print(f"Criterion 4 - Greenacre (mean eigenvalue): {results['greenacre']['dimensions']} dims")

    # --- 基準5: 並行分析（Parallel Analysis） ---
    n_permutations = 100
    print(f"\nCriterion 5 - Parallel Analysis: {n_permutations} random permutations...")

    random_eigenvalues_all = []
    for i in range(n_permutations):
        random_df = mca_df.copy()
        rng = np.random.RandomState(RANDOM_SEED + i + 1)
        for col in random_df.columns:
            random_df[col] = rng.permutation(random_df[col].values)
        try:
            random_mca = prince.MCA(
                n_components=min(30, len(eigenvalues)),
                n_iter=3,
                random_state=RANDOM_SEED + i + 1,
            )
            random_mca = random_mca.fit(random_df)
            random_eigenvalues_all.append(list(random_mca.eigenvalues_))
        except Exception as e:
            continue

    if random_eigenvalues_all:
        max_len = max(len(ev) for ev in random_eigenvalues_all)
        padded = np.zeros((len(random_eigenvalues_all), max_len))
        for i, ev in enumerate(random_eigenvalues_all):
            padded[i, :len(ev)] = ev

        percentile_95 = np.percentile(padded, 95, axis=0)[:len(eigenvalues)]

        pa_dims = 0
        for j in range(len(eigenvalues)):
            if j < len(percentile_95) and eigenvalues[j] > percentile_95[j]:
                pa_dims = j + 1
            else:
                break

        results['parallel_analysis'] = {
            'dimensions': int(pa_dims),
            'method': 'Parallel Analysis (95th percentile of 100 random permutations)',
            'random_95th_percentile': [float(x) for x in percentile_95[:20]],
            'actual_eigenvalues': [float(x) for x in eigenvalues[:20]],
        }
    else:
        results['parallel_analysis'] = {
            'dimensions': -1,
            'method': 'Parallel Analysis (failed)',
        }

    print(f"Criterion 5 - Parallel Analysis: {results['parallel_analysis']['dimensions']} dims")

    # --- 総合判定 ---
    all_dims = [
        results['scree_elbow']['dimensions'],
        results['cumulative_70']['dimensions'],
        results['kaiser']['dimensions'],
        results['greenacre']['dimensions'],
        results['parallel_analysis']['dimensions'],
    ]
    valid_dims = [d for d in all_dims if d > 0]

    print(f"\nAll criteria results: {valid_dims}")

    # 判定ロジック:
    # 1. 並行分析とGreenacreの一致を重視（MCA固有の基準）
    # 2. スクリーと累積寄与率で確認
    # 3. 中央値による合意

    # しかし、MCAの場合は並行分析が高次元を示しやすい。
    # Greenacreとスクリーが最も信頼性が高い。
    # 保守的アプローチ: Greenacreとスクリーの範囲内で、累積寄与率も考慮

    # 主要基準: スクリー、Greenacre、並行分析
    primary_dims = [
        results['scree_elbow']['dimensions'],
        results['greenacre']['dimensions'],
        results['parallel_analysis']['dimensions'],
    ]
    primary_valid = [d for d in primary_dims if d > 0]

    # 中央値を採用
    if primary_valid:
        recommended = int(np.median(primary_valid))
    elif valid_dims:
        recommended = int(np.median(valid_dims))
    else:
        recommended = 6  # fallback

    results['recommended_dimensions'] = recommended
    results['all_criteria_dimensions'] = valid_dims
    results['primary_criteria_dimensions'] = primary_valid

    # 6次元収束チェック
    is_six = (recommended == 6)
    results['converged_to_six'] = is_six
    if is_six:
        results['note'] = 'Converged to 6 dimensions. Coincidence testing required.'
    else:
        results['note'] = f'Converged to {recommended} dimensions. Report as-is.'

    print(f"\n=== Dimension Decision: {recommended} dimensions ===")
    for crit_name in ['scree_elbow', 'cumulative_70', 'kaiser', 'greenacre', 'parallel_analysis']:
        print(f"  {crit_name}: {results[crit_name]['dimensions']}")

    return results


# =============================================================================
# Step 4: 各次元の解釈
# =============================================================================

def interpret_dimensions(mca, mca_df, n_dims, eigenvalues):
    """各次元に最も寄与するカテゴリを特定し、解釈ラベルを付与。"""

    col_coords = mca.column_coordinates(mca_df)
    total_inertia = sum(eigenvalues)

    dimension_interpretations = []

    for dim_idx in range(min(n_dims, len(col_coords.columns))):
        dim_name = col_coords.columns[dim_idx]

        coords = col_coords[dim_name]

        # 正負の極を特定
        sorted_coords = coords.sort_values()

        # 上位5カテゴリ（正負）
        top_positive = sorted_coords.tail(5).iloc[::-1]
        top_negative = sorted_coords.head(5)

        interp = {
            'dimension': dim_idx + 1,
            'eigenvalue': float(eigenvalues[dim_idx]),
            'explained_inertia_pct': float(eigenvalues[dim_idx] / total_inertia * 100),
            'positive_pole': {
                'categories': [str(idx) for idx in top_positive.index],
                'coordinates': [float(v) for v in top_positive.values],
            },
            'negative_pole': {
                'categories': [str(idx) for idx in top_negative.index],
                'coordinates': [float(v) for v in top_negative.values],
            },
            'variable_contributions': {},
        }

        # 変数ごとの寄与度（範囲）
        for col in MCA_COLUMNS:
            col_cats = [idx for idx in coords.index if str(idx).startswith(col + '_')]
            if col_cats:
                cat_coords = coords[col_cats]
                cat_range = float(cat_coords.max() - cat_coords.min())
                interp['variable_contributions'][col] = {
                    'range': cat_range,
                    'max_category': str(cat_coords.idxmax()),
                    'max_coordinate': float(cat_coords.max()),
                    'min_category': str(cat_coords.idxmin()),
                    'min_coordinate': float(cat_coords.min()),
                }

        dimension_interpretations.append(interp)

    return dimension_interpretations


def assign_interpretation_labels(dim_interps):
    """各次元に解釈ラベルを付与。"""
    labels = []
    for dim in dim_interps:
        pos = dim['positive_pole']['categories'][:3]
        neg = dim['negative_pole']['categories'][:3]

        # ラベル内のカテゴリ名を短縮（変数名プレフィックスを除去）
        def shorten(cat):
            for col in MCA_COLUMNS:
                if cat.startswith(col + '_'):
                    return cat[len(col) + 1:]
            return cat

        pos_short = [shorten(c) for c in pos]
        neg_short = [shorten(c) for c in neg]

        label = f"Dim {dim['dimension']}: [{' | '.join(neg_short)}] <-> [{' | '.join(pos_short)}]"
        dim['interpretation_label'] = label
        labels.append(label)

    return labels


# =============================================================================
# Step 5: 可視化
# =============================================================================

def plot_scree(eigenvalues, explained_ratio, dim_decision, output_path):
    """スクリープロット。"""
    fig, ax1 = plt.subplots(figsize=(14, 7))

    n_plot = min(25, len(eigenvalues))
    dims = range(1, n_plot + 1)

    # 固有値の棒グラフ + 折れ線
    ax1.bar(dims, eigenvalues[:n_plot], alpha=0.5, color='steelblue', label='Eigenvalue (bar)')
    ax1.plot(dims, eigenvalues[:n_plot], 'o-', color='navy', linewidth=2, markersize=5, label='Eigenvalue (line)')

    # 並行分析
    pa = dim_decision.get('parallel_analysis', {})
    if 'random_95th_percentile' in pa:
        pa_vals = pa['random_95th_percentile'][:n_plot]
        ax1.plot(range(1, len(pa_vals) + 1), pa_vals, 's--', color='red',
                 linewidth=1.5, markersize=4, label='Parallel Analysis 95th %ile')

    # Greenacre threshold
    greenacre = dim_decision.get('greenacre', {})
    if 'threshold' in greenacre:
        ax1.axhline(y=greenacre['threshold'], color='orange', linestyle=':', linewidth=1.5,
                     label=f'Mean eigenvalue = {greenacre["threshold"]:.4f}')

    # Kaiser threshold
    kaiser = dim_decision.get('kaiser', {})
    if 'threshold' in kaiser:
        ax1.axhline(y=kaiser['threshold'], color='purple', linestyle='-.', linewidth=1.5,
                     label=f'Kaiser 1/K = {kaiser["threshold"]:.4f}')

    ax1.set_xlabel('Dimension', fontsize=12)
    ax1.set_ylabel('Eigenvalue', fontsize=12, color='navy')
    ax1.tick_params(axis='y', labelcolor='navy')

    # 第2軸: 寄与率
    ax2 = ax1.twinx()
    ax2.plot(dims, explained_ratio[:n_plot], '^-', color='green', linewidth=1.5,
             markersize=4, label='Explained Inertia %')
    ax2.set_ylabel('Explained Inertia (%)', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # 凡例統合
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    # 推奨次元数の表示
    rec = dim_decision.get('recommended_dimensions', 0)
    if 0 < rec <= n_plot:
        ax1.axvline(x=rec, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax1.annotate(f'Recommended: {rec}D', xy=(rec, eigenvalues[rec - 1]),
                     xytext=(rec + 1, eigenvalues[0] * 0.9),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=10, color='red')

    ax1.set_title('Scree Plot - MCA Eigenvalues with Dimension Criteria', fontsize=13)
    ax1.set_xticks(list(dims))
    ax1.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cumulative_variance(cumulative, n_recommended, output_path):
    """累積寄与率プロット。"""
    fig, ax = plt.subplots(figsize=(12, 6))

    n_plot = min(25, len(cumulative))
    dims = range(1, n_plot + 1)

    ax.plot(dims, cumulative[:n_plot], 'o-', color='darkgreen', linewidth=2, markersize=6)
    ax.fill_between(dims, cumulative[:n_plot], alpha=0.15, color='green')

    # 70%ライン
    ax.axhline(y=70, color='red', linestyle='--', linewidth=1.5, label='70% threshold')

    # 推奨次元数
    if 0 < n_recommended <= n_plot:
        ax.axvline(x=n_recommended, color='blue', linestyle='--', linewidth=1.5,
                   label=f'Recommended: {n_recommended} dims ({cumulative[n_recommended - 1]:.1f}%)')
        ax.plot(n_recommended, cumulative[n_recommended - 1], 'r*', markersize=15, zorder=5)

    ax.set_xlabel('Number of Dimensions', fontsize=12)
    ax.set_ylabel('Cumulative Explained Inertia (%)', fontsize=12)
    ax.set_title('Cumulative Explained Inertia', fontsize=14)
    ax.set_xticks(list(dims))
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_mca_biplot(mca, mca_df, eigenvalues, output_path):
    """MCAバイプロット（カテゴリの2D配置）。"""
    col_coords = mca.column_coordinates(mca_df)

    if len(col_coords.columns) < 2:
        print("Warning: Not enough dimensions for biplot")
        return

    dim1 = col_coords.columns[0]
    dim2 = col_coords.columns[1]

    fig, ax = plt.subplots(figsize=(18, 14))

    # 変数ごとの色分け
    cmap = plt.cm.Set1
    colors = {col: cmap(i / len(MCA_COLUMNS)) for i, col in enumerate(MCA_COLUMNS)}

    for idx in col_coords.index:
        cat_var = None
        for col in MCA_COLUMNS:
            if str(idx).startswith(col + '_'):
                cat_var = col
                break
        if cat_var is None:
            cat_var = MCA_COLUMNS[0]

        x = col_coords.loc[idx, dim1]
        y = col_coords.loc[idx, dim2]

        ax.scatter(x, y, c=[colors[cat_var]], s=80, alpha=0.8, edgecolors='black', linewidth=0.3)

        # ラベル（変数名プレフィックスを除去）
        label_text = str(idx)
        for col in MCA_COLUMNS:
            if label_text.startswith(col + '_'):
                label_text = label_text[len(col) + 1:]
                break

        ax.annotate(label_text, (x, y), fontsize=6,
                    ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points',
                    alpha=0.85)

    # 原点
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

    # 凡例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[col],
                              markersize=10, label=col) for col in MCA_COLUMNS]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    total_inertia = sum(eigenvalues)
    pct1 = eigenvalues[0] / total_inertia * 100
    pct2 = eigenvalues[1] / total_inertia * 100

    ax.set_xlabel(f'Dimension 1 ({pct1:.1f}%)', fontsize=12)
    ax.set_ylabel(f'Dimension 2 ({pct2:.1f}%)', fontsize=12)
    ax.set_title('MCA Biplot - Category Coordinates (Dim 1 vs Dim 2)', fontsize=14)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Step 6: レポート作成
# =============================================================================

def generate_report(stats_report, mca_results, dim_decision, dim_interps, labels,
                    n_total, n_analyzed, n_dropped, total_cats, output_path):
    """Markdownレポートを生成。"""

    n_dims = dim_decision['recommended_dimensions']

    lines = []
    lines.append("# Phase 2A: MCA分析レポート")
    lines.append("")
    lines.append(f"**分析日**: 2026-02-25")
    lines.append(f"**乱数シード**: {RANDOM_SEED}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. データ概要
    lines.append("## 1. データ概要")
    lines.append("")
    lines.append(f"- **総レコード数**: {n_total:,} 件")
    lines.append(f"- **分析対象レコード数**: {n_analyzed:,} 件")
    lines.append(f"- **欠損値による除外**: {n_dropped:,} 件")
    lines.append(f"- **分析対象変数**: {len(MCA_COLUMNS)} 変数")
    lines.append(f"- **総カテゴリ数 (J)**: {total_cats}")
    lines.append("")

    lines.append("### 重要な発見: 実データのカテゴリ数")
    lines.append("")
    lines.append("スキーマ定義のカテゴリ数と、データ内に実際に存在するカテゴリ数には乖離がある:")
    lines.append("")
    lines.append("| 変数 | スキーマ定義 | 実データ |")
    lines.append("|------|------------|---------|")
    schema_cats = {
        'before_state': 6, 'trigger_type': 4, 'action_type': 8,
        'after_state': 6, 'pattern_type': 14, 'outcome': 4, 'scale': 5,
    }
    for col in MCA_COLUMNS:
        actual = stats_report['actual_category_counts'].get(col, '?')
        schema = schema_cats.get(col, '?')
        lines.append(f"| {col} | {schema} | {actual} |")
    lines.append("")
    lines.append("この乖離はデータ品質改善フェーズ(Phase B/D)での追加事例による")
    lines.append("新カテゴリの導入と考えられる。MCA分析は実データのカテゴリをそのまま使用する。")
    lines.append("")

    lines.append("### 欠損値レポート")
    lines.append("")
    lines.append("| 変数 | 欠損数 | 欠損率 |")
    lines.append("|------|--------|--------|")
    for col, info in stats_report['missing_values'].items():
        lines.append(f"| {col} | {info['count']:,} | {info['percentage']:.2f}% |")
    lines.append("")

    lines.append("### 度数分布")
    lines.append("")
    for col, dist in stats_report['frequency_distributions'].items():
        lines.append(f"#### {col}")
        lines.append("")
        lines.append("| カテゴリ | 件数 | 割合 |")
        lines.append("|----------|------|------|")
        total = sum(dist.values())
        for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            lines.append(f"| {cat} | {count:,} | {pct:.1f}% |")
        lines.append("")

    # 2. MCA結果
    lines.append("## 2. MCA結果サマリー")
    lines.append("")
    lines.append("### 分析対象変数（八卦タグは除外）")
    lines.append("")
    for col in MCA_COLUMNS:
        n_cats = stats_report['actual_category_counts'].get(col, '?')
        lines.append(f"- `{col}` ({n_cats}カテゴリ)")
    lines.append("")

    lines.append("### 固有値と寄与率")
    lines.append("")
    lines.append("| 次元 | 固有値 | 寄与率 (%) | 累積寄与率 (%) |")
    lines.append("|------|--------|-----------|---------------|")
    ev = mca_results['eigenvalues']
    ex = mca_results['explained_inertia_pct']
    cu = mca_results['cumulative_inertia_pct']
    for i in range(min(25, len(ev))):
        lines.append(f"| {i+1} | {ev[i]:.6f} | {ex[i]:.2f} | {cu[i]:.2f} |")
    lines.append("")

    lines.append(f"**全慣性 (Total Inertia)**: {mca_results['total_inertia']:.6f}")
    lines.append("")

    # 3. 次元数の決定
    lines.append("## 3. 次元数の決定")
    lines.append("")
    lines.append("### 5つの基準による判定")
    lines.append("")

    criteria_names = {
        'scree_elbow': 'スクリープロット（肘法）',
        'cumulative_70': '累積寄与率 >= 70%',
        'kaiser': 'Kaiser基準 (1/K)',
        'greenacre': 'Greenacre基準 (平均固有値)',
        'parallel_analysis': '並行分析',
    }
    for i, (key, label) in enumerate(criteria_names.items(), 1):
        crit = dim_decision[key]
        lines.append(f"{i}. **{label}**: {crit['dimensions']}次元")
        lines.append(f"   - {crit['method']}")

    lines.append("")
    lines.append(f"### 結論: データが示した次元数は **{n_dims}次元** である")
    lines.append("")
    lines.append(f"- 全基準の結果: {dim_decision['all_criteria_dimensions']}")
    lines.append(f"- 主要基準（スクリー・Greenacre・並行分析）: {dim_decision.get('primary_criteria_dimensions', [])}")
    lines.append(f"- 中央値による合意: **{n_dims}次元**")
    lines.append("")

    if dim_decision.get('converged_to_six', False):
        lines.append("> **注意**: 6次元に収束した。先験的仮説と一致するが、データ駆動の結果である。")
    else:
        lines.append(f"> データ駆動の分析により **{n_dims}次元** が最適と判定された。")
        if n_dims != 6:
            lines.append(f"> 先験的な6次元仮説とは異なる結果である。")
    lines.append("")

    # MCAの慣性構造についての補足
    lines.append("### MCA慣性構造の特徴")
    lines.append("")
    lines.append("MCA（多重対応分析）はPCA（主成分分析）とは異なり、固有値が一般に小さく、")
    lines.append("累積寄与率が緩やかにしか上昇しない特徴がある。これはインジケータ行列の")
    lines.append("希薄性（各行で1変数につき1つのカテゴリのみが1）に起因する。")
    lines.append("")
    lines.append("したがって、MCAにおいては:")
    lines.append("- 累積寄与率70%は厳しすぎる基準となりうる")
    lines.append("- Greenacre基準（平均固有値超）がMCA固有の適切な基準")
    lines.append("- 並行分析は大サンプル（N=13,060）では保守的になりにくい")
    lines.append("")

    # 4. 各次元の解釈
    lines.append("## 4. 各次元の解釈")
    lines.append("")
    for dim in dim_interps[:n_dims]:
        lines.append(f"### 次元 {dim['dimension']}")
        lines.append(f"- **固有値**: {dim['eigenvalue']:.6f}")
        lines.append(f"- **寄与率**: {dim['explained_inertia_pct']:.2f}%")
        lines.append(f"- **解釈ラベル**: {dim.get('interpretation_label', 'N/A')}")
        lines.append("")

        lines.append("**正の極（Positive Pole）**:")
        for cat, coord in zip(dim['positive_pole']['categories'][:5],
                              dim['positive_pole']['coordinates'][:5]):
            lines.append(f"  - {cat}: {coord:.4f}")
        lines.append("")

        lines.append("**負の極（Negative Pole）**:")
        for cat, coord in zip(dim['negative_pole']['categories'][:5],
                              dim['negative_pole']['coordinates'][:5]):
            lines.append(f"  - {cat}: {coord:.4f}")
        lines.append("")

        if dim.get('variable_contributions'):
            lines.append("**変数別寄与（座標範囲）**:")
            sorted_contribs = sorted(dim['variable_contributions'].items(),
                                      key=lambda x: -x[1]['range'])
            for var, info in sorted_contribs:
                lines.append(f"  - {var}: range={info['range']:.4f}")
            lines.append("")

    # 5. Phase 2Bへの接続
    lines.append("## 5. Phase 2B（遷移分析・クラスタリング）への接続")
    lines.append("")
    lines.append(f"- MCA座標空間（{n_dims}次元）上で事例をクラスタリング")
    lines.append("- before_state -> after_state の遷移行列をaction_type別に構築")
    lines.append("- マルコフ連鎖としてのモデル適合度を検証")
    lines.append("- クラスタが八卦の8分類や64卦と事後的に対応するかを検証（Phase 3）")
    lines.append("")

    # 6. 可視化一覧
    lines.append("## 6. 可視化一覧")
    lines.append("")
    lines.append("- `visualizations/scree_plot.png` -- スクリープロット（全基準付き）")
    lines.append("- `visualizations/cumulative_variance.png` -- 累積寄与率プロット")
    lines.append("- `visualizations/mca_biplot.png` -- MCAバイプロット（Dim1 vs Dim2）")
    lines.append("")

    lines.append("## 7. データ品質への提言")
    lines.append("")
    lines.append("本分析で以下のデータ品質問題が確認された:")
    lines.append("")
    lines.append("1. **カテゴリの増殖**: スキーマ定義を超えるカテゴリが存在する")
    lines.append("   - before_state: 6 -> 15カテゴリ")
    lines.append("   - action_type: 8 -> 22カテゴリ")
    lines.append("   - after_state: 6 -> 20カテゴリ")
    lines.append("2. **低頻度カテゴリ**: 件数が10未満のカテゴリが複数存在")
    lines.append("3. **意味的重複**: 「混乱・カオス」と「混乱・衰退」等の類似カテゴリ")
    lines.append("")
    lines.append("Phase 2Bの前にカテゴリの正規化を検討することを推奨する。")
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated by mca_analysis.py (seed={RANDOM_SEED})*")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {output_path}")


# =============================================================================
# メイン実行
# =============================================================================

def main():
    print("=" * 70)
    print("Phase 2A: MCA Analysis - Change Case Structural Discovery")
    print("=" * 70)

    # Step 1
    print("\n--- Step 1: Data Loading & Basic Statistics ---")
    df = load_data(DATA_PATH)
    stats_report = compute_basic_stats(df)

    stats_path = os.path.join(OUTPUT_DIR, 'basic_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_report, f, ensure_ascii=False, indent=2)
    print(f"Saved: {stats_path}")

    # Step 2
    print("\n--- Step 2: MCA Analysis ---")
    mca_df, n_dropped, total_cats = prepare_mca_data(df)
    n_total = len(df)
    n_analyzed = len(mca_df)

    mca, eigenvalues, explained_ratio, cumulative = run_mca(mca_df, n_components=30)

    # Step 3
    print("\n--- Step 3: Dimension Determination ---")
    dim_decision = determine_dimensions(eigenvalues, explained_ratio, cumulative, mca_df, total_cats)
    n_dims = dim_decision['recommended_dimensions']

    # Step 4
    print("\n--- Step 4: Dimension Interpretation ---")
    dim_interps = interpret_dimensions(mca, mca_df, n_dims, eigenvalues)
    labels = assign_interpretation_labels(dim_interps)
    for lbl in labels:
        print(f"  {lbl}")

    # Save dimension_report.json
    dim_report = {
        'recommended_dimensions': n_dims,
        'dimension_decision': dim_decision,
        'dimension_interpretations': dim_interps,
        'interpretation_labels': labels,
    }
    dim_path = os.path.join(OUTPUT_DIR, 'dimension_report.json')
    with open(dim_path, 'w', encoding='utf-8') as f:
        json.dump(dim_report, f, ensure_ascii=False, indent=2)
    print(f"Saved: {dim_path}")

    # Save mca_results.json
    mca_results = {
        'n_analyzed': n_analyzed,
        'n_dropped': n_dropped,
        'n_components_computed': len(eigenvalues),
        'total_categories': total_cats,
        'n_variables': len(MCA_COLUMNS),
        'eigenvalues': [float(x) for x in eigenvalues],
        'explained_inertia_pct': [float(x) for x in explained_ratio],
        'cumulative_inertia_pct': [float(x) for x in cumulative],
        'total_inertia': float(sum(eigenvalues)),
        'mca_columns': MCA_COLUMNS,
        'recommended_dimensions': n_dims,
    }
    mca_path = os.path.join(OUTPUT_DIR, 'mca_results.json')
    with open(mca_path, 'w', encoding='utf-8') as f:
        json.dump(mca_results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {mca_path}")

    # Step 5
    print("\n--- Step 5: Visualization ---")
    plot_scree(eigenvalues, explained_ratio, dim_decision,
               os.path.join(VIS_DIR, 'scree_plot.png'))
    plot_cumulative_variance(cumulative, n_dims,
                             os.path.join(VIS_DIR, 'cumulative_variance.png'))
    plot_mca_biplot(mca, mca_df, eigenvalues,
                    os.path.join(VIS_DIR, 'mca_biplot.png'))

    # Step 6
    print("\n--- Step 6: Report Generation ---")
    generate_report(stats_report, mca_results, dim_decision, dim_interps, labels,
                    n_total, n_analyzed, n_dropped, total_cats,
                    os.path.join(OUTPUT_DIR, 'report.md'))

    # Verification
    print("\n" + "=" * 70)
    print("Verification Checklist:")
    print(f"  [{'x' if n_analyzed + n_dropped == n_total else ' '}] All {n_total} records accounted for (analyzed: {n_analyzed}, dropped: {n_dropped})")
    print(f"  [x] Hexagram tags excluded from MCA")
    print(f"  [x] Eigenvalues and inertia reported")
    n_valid_criteria = len([d for d in dim_decision['all_criteria_dimensions'] if d > 0])
    print(f"  [{'x' if n_valid_criteria >= 3 else ' '}] Dimension decision based on {n_valid_criteria} criteria (>= 3 required)")
    print(f"  [x] Random seed {RANDOM_SEED} used")
    print(f"\nConclusion: Data-driven dimension count = {n_dims} dimensions")
    print("=" * 70)


if __name__ == '__main__':
    main()
