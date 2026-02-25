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
"""

import sys
sys.path.insert(0, '/Users/hideakimacbookair/Library/Python/3.12/lib/python/site-packages')

import json
import os
import warnings
from collections import Counter, OrderedDict
from itertools import product

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import prince

warnings.filterwarnings('ignore')

# --- フォント設定 ---
# macOSの日本語フォントを探す
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
    # フォールバック: matplotlib組み込み
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

# 期待されるカテゴリ値
EXPECTED_VALUES = {
    'before_state': ['絶頂・慢心', '停滞・閉塞', '混乱・カオス', '成長痛', 'どん底・危機', '安定・平和'],
    'trigger_type': ['外部ショック', '内部崩壊', '意図的決断', '偶発・出会い'],
    'action_type': ['攻める・挑戦', '守る・維持', '捨てる・撤退', '耐える・潜伏', '対話・融合', '刷新・破壊', '逃げる・放置', '分散・スピンオフ'],
    'after_state': ['V字回復・大成功', '縮小安定・生存', '変質・新生', '現状維持・延命', '迷走・混乱', '崩壊・消滅'],
    'pattern_type': ['Shock_Recovery', 'Hubris_Collapse', 'Pivot_Success', 'Endurance', 'Slow_Decline',
                     'Steady_Growth', 'Creative_Destruction', 'Failed_Gamble', 'Forced_Adaptation',
                     'Internal_Reform', 'External_Takeover', 'Gradual_Transformation',
                     'Crisis_Innovation', 'Cyclic_Renewal'],
    'outcome': ['Success', 'PartialSuccess', 'Failure', 'Mixed'],
    'scale': ['company', 'individual', 'family', 'country', 'other'],
}

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
    }

    # 度数分布
    for col in MCA_COLUMNS:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False).to_dict()
            # NaN keyを文字列に変換
            clean_vc = {}
            for k, v in vc.items():
                key = str(k) if pd.isna(k) is False else 'MISSING'
                if isinstance(k, float) and np.isnan(k):
                    key = 'MISSING'
                clean_vc[key] = int(v)
            stats_report['frequency_distributions'][col] = clean_vc

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

    # クロス集計 (before_state × action_type × after_state)
    # 3次元は巨大なので、2次元ペアに分割
    for pair in [('before_state', 'action_type'),
                 ('before_state', 'after_state'),
                 ('action_type', 'after_state')]:
        if pair[0] in df.columns and pair[1] in df.columns:
            ct = pd.crosstab(df[pair[0]], df[pair[1]])
            key = f"{pair[0]}_x_{pair[1]}"
            stats_report['cross_tabulation'][key] = ct.to_dict()

    # 3次元クロス集計のサマリー
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
    """MCA用のデータを準備。欠損値を処理し、カテゴリカル変数を選択。"""
    mca_df = df[MCA_COLUMNS].copy()

    # 欠損値の確認と処理
    missing_summary = mca_df.isna().sum()
    print("\n=== 欠損値確認 ===")
    print(missing_summary)

    # 欠損値がある行を除外（完全ケース分析）
    n_before = len(mca_df)
    mca_df = mca_df.dropna()
    n_after = len(mca_df)
    n_dropped = n_before - n_after
    print(f"\n欠損値除外: {n_dropped}件除外 → {n_after}件で分析")

    # 全て文字列型に変換
    for col in MCA_COLUMNS:
        mca_df[col] = mca_df[col].astype(str)

    return mca_df, n_dropped


def run_mca(mca_df, n_components=20):
    """MCAを実行し、結果を返す。"""
    print(f"\n=== MCA実行 (n_components={n_components}) ===")

    mca = prince.MCA(
        n_components=n_components,
        n_iter=10,
        random_state=RANDOM_SEED,
    )
    mca = mca.fit(mca_df)

    # 固有値
    eigenvalues = mca.eigenvalues_
    print(f"固有値: {eigenvalues[:10]}")

    # 寄与率（慣性比率）
    # prince 0.16+: percentage_of_variance_ or explained_inertia_
    total_inertia = sum(eigenvalues)
    explained_ratio = [ev / total_inertia * 100 for ev in eigenvalues]
    cumulative = np.cumsum(explained_ratio)

    print(f"寄与率 (上位10): {[round(x, 2) for x in explained_ratio[:10]]}")
    print(f"累積寄与率 (上位10): {[round(x, 2) for x in cumulative[:10]]}")

    return mca, eigenvalues, explained_ratio, cumulative


# =============================================================================
# Step 3: 次元数の決定
# =============================================================================

def determine_dimensions(eigenvalues, explained_ratio, cumulative, mca_df):
    """4つの基準で次元数を決定する。"""

    n_vars = len(MCA_COLUMNS)
    total_categories = sum(len(EXPECTED_VALUES[col]) for col in MCA_COLUMNS)

    results = {}

    # --- 基準1: スクリープロット（肘法） ---
    # 固有値の差分を計算、最大の変化点を探す
    diffs = np.diff(eigenvalues)
    second_diffs = np.diff(diffs)
    # 肘 = 二次差分の絶対値が最大の点 + 1
    elbow_idx = np.argmax(np.abs(second_diffs)) + 1
    results['scree_elbow'] = {
        'dimensions': int(elbow_idx) + 1,
        'method': 'スクリープロット（肘法）: 固有値の二次差分の最大変化点',
        'elbow_eigenvalue': float(eigenvalues[elbow_idx]),
    }
    print(f"\n基準1 - スクリープロット（肘法）: {results['scree_elbow']['dimensions']}次元")

    # --- 基準2: 累積寄与率 70% ---
    cum70_idx = np.argmax(np.array(cumulative) >= 70.0)
    if cumulative[cum70_idx] >= 70.0:
        results['cumulative_70'] = {
            'dimensions': int(cum70_idx) + 1,
            'method': '累積寄与率 >= 70%',
            'cumulative_at_threshold': float(cumulative[cum70_idx]),
        }
    else:
        results['cumulative_70'] = {
            'dimensions': len(eigenvalues),
            'method': '累積寄与率 >= 70% (未到達)',
            'max_cumulative': float(cumulative[-1]),
        }
    print(f"基準2 - 累積寄与率70%: {results['cumulative_70']['dimensions']}次元")

    # --- 基準3: Kaiser基準（修正版: MCAの場合 1/変数数） ---
    # MCAにおけるKaiser基準: 固有値 > 1/K (K=変数数)
    kaiser_threshold = 1.0 / n_vars
    kaiser_dims = sum(1 for ev in eigenvalues if ev > kaiser_threshold)
    results['kaiser'] = {
        'dimensions': int(kaiser_dims),
        'method': f'Kaiser基準（固有値 > 1/{n_vars} = {kaiser_threshold:.4f}）',
        'threshold': float(kaiser_threshold),
    }
    print(f"基準3 - Kaiser基準: {results['kaiser']['dimensions']}次元")

    # --- 基準4: 並行分析（Parallel Analysis） ---
    n_permutations = 100
    n_samples = len(mca_df)

    print(f"\n基準4 - 並行分析: {n_permutations}回のランダム置換を実行中...")
    random_eigenvalues_all = []

    for i in range(n_permutations):
        # 各列を独立にシャッフル
        random_df = mca_df.copy()
        for col in random_df.columns:
            random_df[col] = np.random.permutation(random_df[col].values)

        try:
            random_mca = prince.MCA(
                n_components=min(20, len(eigenvalues)),
                n_iter=3,
                random_state=RANDOM_SEED + i,
            )
            random_mca = random_mca.fit(random_df)
            random_eigenvalues_all.append(random_mca.eigenvalues_)
        except Exception as e:
            print(f"  並行分析 {i+1}/{n_permutations} でエラー: {e}")
            continue

    if random_eigenvalues_all:
        # 95パーセンタイルを計算
        max_len = max(len(ev) for ev in random_eigenvalues_all)
        padded = np.zeros((len(random_eigenvalues_all), max_len))
        for i, ev in enumerate(random_eigenvalues_all):
            padded[i, :len(ev)] = ev

        percentile_95 = np.percentile(padded, 95, axis=0)[:len(eigenvalues)]

        # 実データの固有値 > ランダムの95パーセンタイル
        pa_dims = 0
        for j in range(len(eigenvalues)):
            if j < len(percentile_95) and eigenvalues[j] > percentile_95[j]:
                pa_dims = j + 1
            else:
                break

        results['parallel_analysis'] = {
            'dimensions': int(pa_dims),
            'method': '並行分析（100回ランダム置換の95パーセンタイル比較）',
            'random_95th_percentile': [float(x) for x in percentile_95[:min(15, len(percentile_95))]],
            'actual_eigenvalues': [float(x) for x in eigenvalues[:min(15, len(eigenvalues))]],
        }
    else:
        results['parallel_analysis'] = {
            'dimensions': -1,
            'method': '並行分析（失敗）',
        }

    print(f"基準4 - 並行分析: {results['parallel_analysis']['dimensions']}次元")

    # --- 総合判定 ---
    dimension_candidates = [
        results['scree_elbow']['dimensions'],
        results['cumulative_70']['dimensions'],
        results['kaiser']['dimensions'],
        results['parallel_analysis']['dimensions'],
    ]
    # -1（失敗）を除外
    valid_candidates = [d for d in dimension_candidates if d > 0]

    # 中央値を採用（もっとも保守的な合意点）
    if valid_candidates:
        recommended = int(np.median(valid_candidates))
    else:
        recommended = 6  # フォールバック

    results['recommended_dimensions'] = recommended
    results['all_criteria_dimensions'] = valid_candidates

    # 6次元に収束したかの判定
    is_six = (recommended == 6)
    results['converged_to_six'] = is_six
    if is_six:
        results['note'] = '6次元に収束。偶然の可能性を検定する必要あり。'
    else:
        results['note'] = f'{recommended}次元に収束。データが示す次元数をそのまま報告。'

    print(f"\n=== 次元数決定: {recommended}次元 ===")
    print(f"  スクリー: {results['scree_elbow']['dimensions']}")
    print(f"  累積70%: {results['cumulative_70']['dimensions']}")
    print(f"  Kaiser: {results['kaiser']['dimensions']}")
    print(f"  並行分析: {results['parallel_analysis']['dimensions']}")

    return results


# =============================================================================
# Step 4: 各次元の解釈
# =============================================================================

def interpret_dimensions(mca, mca_df, n_dims):
    """各次元に最も寄与するカテゴリを特定し、解釈ラベルを付与。"""

    # カテゴリ座標の取得
    col_coords = mca.column_coordinates(mca_df)

    dimension_interpretations = []

    for dim_idx in range(n_dims):
        dim_name = f"Dim_{dim_idx}"
        if dim_name not in col_coords.columns:
            # prince のバージョンによってはインデックスが異なる
            if dim_idx < len(col_coords.columns):
                dim_name = col_coords.columns[dim_idx]
            else:
                continue

        coords = col_coords[dim_name]

        # 寄与度（cos2に近似: 座標^2 / 全座標^2の合計）
        total_inertia = (col_coords ** 2).sum(axis=1)
        contributions = (coords ** 2) / total_inertia
        contributions = contributions.fillna(0)

        # 正負の極を特定
        sorted_coords = coords.sort_values()

        # 最も強い正方向のカテゴリ（上位5）
        top_positive = sorted_coords.tail(5).iloc[::-1]
        # 最も強い負方向のカテゴリ（上位5）
        top_negative = sorted_coords.head(5)

        # 解釈
        interp = {
            'dimension': dim_idx + 1,
            'eigenvalue': float(mca.eigenvalues_[dim_idx]),
            'explained_inertia_pct': float(mca.eigenvalues_[dim_idx] / sum(mca.eigenvalues_) * 100),
            'positive_pole': {
                'categories': [str(idx) for idx in top_positive.index],
                'coordinates': [float(v) for v in top_positive.values],
            },
            'negative_pole': {
                'categories': [str(idx) for idx in top_negative.index],
                'coordinates': [float(v) for v in top_negative.values],
            },
            'top_contributions': {},
        }

        # 変数ごとの寄与度
        for col in MCA_COLUMNS:
            # カテゴリ名に変数名が含まれるものを抽出
            col_cats = [idx for idx in coords.index if str(idx).startswith(col + '_')]
            if not col_cats:
                # prince のカテゴリ名フォーマットに対応
                col_cats = [idx for idx in coords.index if col in str(idx)]

            if col_cats:
                cat_coords = coords[col_cats]
                cat_range = float(cat_coords.max() - cat_coords.min())
                interp['top_contributions'][col] = {
                    'range': cat_range,
                    'categories': {str(c): float(coords[c]) for c in col_cats},
                }

        dimension_interpretations.append(interp)

    return dimension_interpretations


def assign_interpretation_labels(dim_interps):
    """各次元に人間が理解できるラベルを付与する（自動 + ヒューリスティック）。"""
    labels = []
    for dim in dim_interps:
        pos = dim['positive_pole']['categories'][:3]
        neg = dim['negative_pole']['categories'][:3]

        # ヒューリスティックルール
        pos_str = ' / '.join(pos)
        neg_str = ' / '.join(neg)

        # 簡略ラベル生成
        label = f"Axis {dim['dimension']}: [{neg_str}] ⟷ [{pos_str}]"

        dim['interpretation_label'] = label
        labels.append(label)

    return labels


# =============================================================================
# Step 5: 可視化
# =============================================================================

def plot_scree(eigenvalues, explained_ratio, pa_result, output_path):
    """スクリープロット（固有値 vs 次元番号）。"""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    n_plot = min(20, len(eigenvalues))
    dims = range(1, n_plot + 1)

    # 固有値の棒グラフ
    ax1.bar(dims, eigenvalues[:n_plot], alpha=0.6, color='steelblue', label='Eigenvalue')
    ax1.plot(dims, eigenvalues[:n_plot], 'o-', color='navy', linewidth=2, label='Eigenvalue (line)')

    # 並行分析の95パーセンタイル
    if 'random_95th_percentile' in pa_result:
        pa_vals = pa_result['random_95th_percentile'][:n_plot]
        ax1.plot(range(1, len(pa_vals) + 1), pa_vals, 's--', color='red',
                 linewidth=1.5, label='Parallel Analysis (95th percentile)')

    ax1.set_xlabel('Dimension', fontsize=12)
    ax1.set_ylabel('Eigenvalue', fontsize=12, color='navy')
    ax1.tick_params(axis='y', labelcolor='navy')

    # 寄与率の第2軸
    ax2 = ax1.twinx()
    ax2.plot(dims, explained_ratio[:n_plot], '^-', color='green', linewidth=1.5, label='Explained Inertia %')
    ax2.set_ylabel('Explained Inertia (%)', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # 凡例を統合
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    ax1.set_title('Scree Plot - MCA Eigenvalues', fontsize=14)
    ax1.set_xticks(list(dims))
    ax1.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cumulative_variance(cumulative, n_recommended, output_path):
    """累積寄与率プロット。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_plot = min(20, len(cumulative))
    dims = range(1, n_plot + 1)

    ax.plot(dims, cumulative[:n_plot], 'o-', color='darkgreen', linewidth=2, markersize=8)
    ax.fill_between(dims, cumulative[:n_plot], alpha=0.2, color='green')

    # 70%ライン
    ax.axhline(y=70, color='red', linestyle='--', linewidth=1.5, label='70% threshold')

    # 推奨次元数
    if n_recommended <= n_plot:
        ax.axvline(x=n_recommended, color='blue', linestyle='--', linewidth=1.5,
                   label=f'Recommended: {n_recommended} dims')
        ax.plot(n_recommended, cumulative[n_recommended - 1], 'r*', markersize=15)

    ax.set_xlabel('Number of Dimensions', fontsize=12)
    ax.set_ylabel('Cumulative Explained Inertia (%)', fontsize=12)
    ax.set_title('Cumulative Explained Inertia', fontsize=14)
    ax.set_xticks(list(dims))
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_mca_biplot(mca, mca_df, output_path):
    """MCAバイプロット（カテゴリの2D配置）。"""
    col_coords = mca.column_coordinates(mca_df)

    if len(col_coords.columns) < 2:
        print("Warning: Not enough dimensions for biplot")
        return

    dim1 = col_coords.columns[0]
    dim2 = col_coords.columns[1]

    fig, ax = plt.subplots(figsize=(16, 12))

    # 変数ごとに色を分ける
    colors = plt.cm.Set1(np.linspace(0, 1, len(MCA_COLUMNS)))
    color_map = {col: colors[i] for i, col in enumerate(MCA_COLUMNS)}

    for idx in col_coords.index:
        # カテゴリが属する変数を判定
        cat_var = None
        for col in MCA_COLUMNS:
            if str(idx).startswith(col + '_') or col in str(idx):
                cat_var = col
                break
        if cat_var is None:
            cat_var = MCA_COLUMNS[0]

        x = col_coords.loc[idx, dim1]
        y = col_coords.loc[idx, dim2]

        ax.scatter(x, y, c=[color_map[cat_var]], s=100, alpha=0.8, edgecolors='black', linewidth=0.5)

        # ラベル（カテゴリ名を短縮表示）
        label_text = str(idx)
        # 変数名プレフィックスを除去して短縮
        for col in MCA_COLUMNS:
            if label_text.startswith(col + '_'):
                label_text = label_text[len(col) + 1:]
                break

        ax.annotate(label_text, (x, y), fontsize=7,
                    ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points',
                    alpha=0.85)

    # 原点の十字線
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

    # 凡例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[col],
                              markersize=10, label=col) for col in MCA_COLUMNS]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    # 寄与率をラベルに含める
    total_inertia = sum(mca.eigenvalues_)
    pct1 = mca.eigenvalues_[0] / total_inertia * 100
    pct2 = mca.eigenvalues_[1] / total_inertia * 100

    ax.set_xlabel(f'Dimension 1 ({pct1:.1f}%)', fontsize=12)
    ax.set_ylabel(f'Dimension 2 ({pct2:.1f}%)', fontsize=12)
    ax.set_title('MCA Biplot - Category Coordinates', fontsize=14)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Step 6: レポート作成
# =============================================================================

def generate_report(stats_report, mca_results, dim_decision, dim_interps, labels,
                    n_total, n_analyzed, n_dropped, output_path):
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
    lines.append("### 分析対象変数（八卦タグを除外）")
    lines.append("")
    for col in MCA_COLUMNS:
        n_cats = len(EXPECTED_VALUES.get(col, []))
        lines.append(f"- `{col}` ({n_cats}カテゴリ)")
    lines.append("")

    lines.append("### 固有値と寄与率")
    lines.append("")
    lines.append("| 次元 | 固有値 | 寄与率 (%) | 累積寄与率 (%) |")
    lines.append("|------|--------|-----------|---------------|")
    eigenvalues = mca_results['eigenvalues']
    explained = mca_results['explained_inertia_pct']
    cumulative = mca_results['cumulative_inertia_pct']
    for i in range(min(20, len(eigenvalues))):
        lines.append(f"| {i+1} | {eigenvalues[i]:.6f} | {explained[i]:.2f} | {cumulative[i]:.2f} |")
    lines.append("")

    # 3. 次元数の決定
    lines.append("## 3. 次元数の決定")
    lines.append("")
    lines.append("### 4つの基準による判定")
    lines.append("")
    lines.append(f"1. **スクリープロット（肘法）**: {dim_decision['scree_elbow']['dimensions']}次元")
    lines.append(f"   - {dim_decision['scree_elbow']['method']}")
    lines.append(f"2. **累積寄与率 >= 70%**: {dim_decision['cumulative_70']['dimensions']}次元")
    lines.append(f"   - {dim_decision['cumulative_70']['method']}")
    lines.append(f"3. **Kaiser基準**: {dim_decision['kaiser']['dimensions']}次元")
    lines.append(f"   - {dim_decision['kaiser']['method']}")
    lines.append(f"4. **並行分析**: {dim_decision['parallel_analysis']['dimensions']}次元")
    lines.append(f"   - {dim_decision['parallel_analysis']['method']}")
    lines.append("")

    lines.append(f"### 結論: データが示した次元数は **{n_dims}次元** である")
    lines.append("")
    lines.append(f"各基準の結果: {dim_decision['all_criteria_dimensions']}")
    lines.append(f"中央値による合意: **{n_dims}次元**")
    lines.append("")

    if dim_decision['converged_to_six']:
        lines.append("> **注意**: 6次元に収束した。これは先験的に想定していた6次元と一致するが、")
        lines.append("> データ駆動の結果として得られたものである。偶然の一致である可能性も考慮する必要がある。")
    else:
        lines.append(f"> データ駆動の分析により、{n_dims}次元が最適と判定された。")
        lines.append(f"> これは先験的な6次元仮説とは異なる結果である。")
    lines.append("")

    # 4. 各次元の解釈
    lines.append("## 4. 各次元の解釈")
    lines.append("")
    for i, dim in enumerate(dim_interps[:n_dims]):
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

    # 5. Phase 2Bへの接続
    lines.append("## 5. Phase 2B（遷移分析・クラスタリング）への接続")
    lines.append("")
    lines.append(f"- MCA座標空間（{n_dims}次元）上での事例のクラスタリングを実施")
    lines.append("- before_state → after_state の遷移行列をaction_type別に構築")
    lines.append("- マルコフ連鎖としてのモデル適合度を検証")
    lines.append("- クラスタが八卦の8分類や64卦と事後的に対応するかを検証")
    lines.append("")

    # 6. 可視化一覧
    lines.append("## 6. 可視化一覧")
    lines.append("")
    lines.append("- `visualizations/scree_plot.png` — スクリープロット")
    lines.append("- `visualizations/cumulative_variance.png` — 累積寄与率プロット")
    lines.append("- `visualizations/mca_biplot.png` — MCAバイプロット（Dim1 vs Dim2）")
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
    print("Phase 2A: MCA分析 — 変化事例の構造化と次元導出")
    print("=" * 70)

    # Step 1: データ読み込みと基本統計
    print("\n--- Step 1: データ読み込みと基本統計 ---")
    df = load_data(DATA_PATH)
    stats_report = compute_basic_stats(df)

    # basic_stats.json保存
    stats_path = os.path.join(OUTPUT_DIR, 'basic_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_report, f, ensure_ascii=False, indent=2)
    print(f"Saved: {stats_path}")

    # Step 2: MCA実行
    print("\n--- Step 2: MCA分析 ---")
    mca_df, n_dropped = prepare_mca_data(df)
    n_total = len(df)
    n_analyzed = len(mca_df)

    mca, eigenvalues, explained_ratio, cumulative = run_mca(mca_df, n_components=20)

    # Step 3: 次元数の決定
    print("\n--- Step 3: 次元数の決定 ---")
    dim_decision = determine_dimensions(eigenvalues, explained_ratio, cumulative, mca_df)

    n_dims = dim_decision['recommended_dimensions']

    # Step 4: 各次元の解釈
    print("\n--- Step 4: 各次元の解釈 ---")
    dim_interps = interpret_dimensions(mca, mca_df, n_dims)
    labels = assign_interpretation_labels(dim_interps)
    for lbl in labels:
        print(f"  {lbl}")

    # dimension_report.json保存
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

    # mca_results.json保存
    mca_results = {
        'n_analyzed': n_analyzed,
        'n_dropped': n_dropped,
        'n_components_computed': len(eigenvalues),
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

    # Step 5: 可視化
    print("\n--- Step 5: 可視化 ---")
    plot_scree(eigenvalues, explained_ratio, dim_decision.get('parallel_analysis', {}),
               os.path.join(VIS_DIR, 'scree_plot.png'))
    plot_cumulative_variance(cumulative, n_dims,
                             os.path.join(VIS_DIR, 'cumulative_variance.png'))
    plot_mca_biplot(mca, mca_df, os.path.join(VIS_DIR, 'mca_biplot.png'))

    # Step 6: レポート作成
    print("\n--- Step 6: レポート作成 ---")
    generate_report(stats_report, mca_results, dim_decision, dim_interps, labels,
                    n_total, n_analyzed, n_dropped,
                    os.path.join(OUTPUT_DIR, 'report.md'))

    # 検証
    print("\n" + "=" * 70)
    print("検証チェックリスト:")
    print(f"  [{'x' if n_analyzed + n_dropped == n_total else ' '}] 全{n_total}件が考慮されている（分析: {n_analyzed}、除外: {n_dropped}）")
    print(f"  [x] 八卦タグがMCA分析から除外されている")
    print(f"  [x] 固有値と寄与率が報告されている")
    print(f"  [{'x' if len([d for d in dim_decision['all_criteria_dimensions'] if d > 0]) >= 3 else ' '}] 次元数の決定根拠が3つ以上の基準で示されている")
    print(f"  [x] 乱数シード42を使用")
    print(f"\n結論: データが示した次元数は {n_dims}次元 である")
    print("=" * 70)


if __name__ == '__main__':
    main()
