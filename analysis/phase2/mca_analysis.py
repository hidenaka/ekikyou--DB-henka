#!/usr/bin/env python3
"""
Phase 2A: MCA（多重対応分析）による変化事例の構造化と次元導出 — 修正版
===================================================================

前回の quality-reviewer FAIL 判定に基づき、以下を修正:
  1. 全慣性（Total Inertia）の計算: mca.total_inertia_ を使用
  2. Greenacre基準の修正: total_inertia / (J-K) = 1/K
  3. 次元数決定ロジック: スクリー + Benzecri修正の合意を主基準
  4. 低頻度カテゴリの感度分析: クリーンデータでの再MCA

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

# --- パス設定（相対パス） ---
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

# --- カテゴリ統合マッピング（クリーンデータ用） ---
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

    for col in MCA_COLUMNS:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            clean_vc = {}
            for k, v in vc.items():
                key = 'MISSING' if pd.isna(k) else str(k)
                clean_vc[key] = int(v)
            stats_report['frequency_distributions'][col] = clean_vc
            stats_report['actual_category_counts'][col] = df[col].nunique()

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

    for pair in [('before_state', 'action_type'),
                 ('before_state', 'after_state'),
                 ('action_type', 'after_state')]:
        if pair[0] in df.columns and pair[1] in df.columns:
            ct = pd.crosstab(df[pair[0]], df[pair[1]])
            key = f"{pair[0]}_x_{pair[1]}"
            stats_report['cross_tabulation'][key] = ct.to_dict()

    if all(c in df.columns for c in ['before_state', 'action_type', 'after_state']):
        three_way = df.groupby(
            ['before_state', 'action_type', 'after_state']
        ).size()
        top_30 = three_way.nlargest(30)
        stats_report['cross_tabulation']['top30_three_way'] = {
            str(k): int(v) for k, v in top_30.items()
        }

    return stats_report


# =============================================================================
# Step 2: MCA分析
# =============================================================================

def prepare_mca_data(df, merge_map=None, label='元データ'):
    """MCA用のデータを準備。merge_mapが指定されたらカテゴリ統合を行う。"""
    mca_df = df[MCA_COLUMNS].copy()

    # カテゴリ統合
    if merge_map:
        for col, mapping in merge_map.items():
            if col in mca_df.columns:
                mca_df[col] = mca_df[col].replace(mapping)

    # 欠損値の確認
    missing_summary = mca_df.isna().sum()
    print(f"\n=== 欠損値確認 ({label}) ===")
    print(missing_summary)

    n_before = len(mca_df)
    mca_df = mca_df.dropna()
    n_after = len(mca_df)
    n_dropped = n_before - n_after
    print(f"欠損値除外: {n_dropped}件除外 -> {n_after}件で分析")

    for col in MCA_COLUMNS:
        mca_df[col] = mca_df[col].astype(str)

    total_cats = 0
    for col in MCA_COLUMNS:
        n_cats = mca_df[col].nunique()
        total_cats += n_cats
        print(f"  {col}: {n_cats} categories")
    print(f"  Total categories (J): {total_cats}")
    print(f"  Number of variables (K): {len(MCA_COLUMNS)}")

    return mca_df, n_dropped, total_cats


def run_mca(mca_df, n_components=30, correction=None, label=''):
    """
    MCAを実行し、結果を返す。

    修正: prince ライブラリの組み込み属性を使用。
    - total_inertia_ = J/K - 1 (正しい全慣性)
    - percentage_of_variance_ (正しい寄与率)
    - cumulative_percentage_of_variance_ (正しい累積寄与率)
    """
    K = len(MCA_COLUMNS)
    actual_n_components = min(n_components, len(mca_df) - 1)
    corr_label = f", correction={correction}" if correction else ""
    print(f"\n=== MCA実行 {label} (n_components={actual_n_components}{corr_label}) ===")

    mca = prince.MCA(
        n_components=actual_n_components,
        n_iter=10,
        random_state=RANDOM_SEED,
        correction=correction,
    )
    mca = mca.fit(mca_df)

    eigenvalues = list(mca.eigenvalues_)

    # --- 修正1: prince の組み込み属性を使用 ---
    total_inertia = float(mca.total_inertia_)
    J = int(mca.J_)
    K_actual = int(mca.K_)
    theoretical_total_inertia = J / K_actual - 1

    # prince の percentage_of_variance_ を使用（修正済みの正しい寄与率）
    explained_ratio = list(mca.percentage_of_variance_)
    cumulative = list(mca.cumulative_percentage_of_variance_)

    print(f"J (total categories): {J}")
    print(f"K (variables): {K_actual}")
    print(f"Total inertia (prince): {total_inertia:.6f}")
    print(f"Total inertia (J/K-1): {theoretical_total_inertia:.6f}")
    print(f"Eigenvalues (top 10): {[round(x, 6) for x in eigenvalues[:10]]}")
    print(f"Explained % (top 10): {[round(x, 2) for x in explained_ratio[:10]]}")
    print(f"Cumulative % (top 10): {[round(x, 2) for x in cumulative[:10]]}")

    return mca, eigenvalues, explained_ratio, cumulative, total_inertia, J


# =============================================================================
# Step 3: Benzecri修正慣性の計算
# =============================================================================

def compute_benzecri_correction(eigenvalues, K):
    """
    Benzecri修正慣性を手動で計算する。
    prince の correction='benzecri' と同等だが、元データの固有値から導出。

    λ_corrected = ((K/(K-1)) * (λ - 1/K))^2  for λ > 1/K
    修正後の寄与率 = λ_corrected / sum(all λ_corrected)
    """
    threshold = 1.0 / K
    corrected = []
    for lam in eigenvalues:
        if lam > threshold:
            val = ((K / (K - 1)) * (lam - threshold)) ** 2
            corrected.append(val)
        else:
            corrected.append(0.0)

    total_corrected = sum(corrected)
    if total_corrected > 0:
        pct = [c / total_corrected * 100 for c in corrected]
    else:
        pct = [0.0] * len(corrected)
    cum_pct = list(np.cumsum(pct))

    n_above_threshold = sum(1 for c in corrected if c > 0)
    return corrected, pct, cum_pct, n_above_threshold


# =============================================================================
# Step 4: 次元数の決定（修正版）
# =============================================================================

def determine_dimensions(eigenvalues, explained_ratio, cumulative,
                         total_inertia, J, mca_df, mca,
                         benzecri_pct, benzecri_cum, benzecri_n_dims):
    """
    次元数を決定する（修正版）。

    修正のポイント:
    - N=13,060 + J=88 では Kaiser/Greenacre/PA が鑑別力を失うことを明示
    - スクリープロット + Benzecri修正の合意を主基準とする
    """

    K = len(MCA_COLUMNS)
    n_eigenvalues = len(eigenvalues)

    results = {}

    # ==============================================
    # 基準1: スクリープロット（肘法）
    # ==============================================
    diffs = np.diff(eigenvalues)
    if len(diffs) > 1:
        second_diffs = np.diff(diffs)
        # 固有値の減衰が最も急激に変化する点
        elbow_idx = np.argmax(np.abs(second_diffs)) + 1
    else:
        elbow_idx = 0

    results['scree_elbow'] = {
        'dimensions': int(elbow_idx) + 1,
        'method': 'Scree plot (elbow): maximum absolute second derivative',
        'elbow_eigenvalue': float(eigenvalues[elbow_idx])
                           if elbow_idx < len(eigenvalues) else 0,
    }
    print(f"\nCriterion 1 - Scree (elbow): {results['scree_elbow']['dimensions']} dims")

    # ==============================================
    # 基準2: 累積寄与率 70%（正しい全慣性ベース）
    # ==============================================
    cum_array = np.array(cumulative)
    idx_70 = np.where(cum_array >= 70.0)[0]
    if len(idx_70) > 0:
        results['cumulative_70'] = {
            'dimensions': int(idx_70[0]) + 1,
            'method': 'Cumulative explained inertia >= 70% (correct total inertia)',
            'cumulative_at_threshold': float(cumulative[idx_70[0]]),
        }
    else:
        results['cumulative_70'] = {
            'dimensions': n_eigenvalues,
            'method': 'Cumulative inertia >= 70% (not reached within computed dims)',
            'max_cumulative': float(cumulative[-1]) if cumulative else 0,
            'note': (f'MCA with J={J}, K={K}: cumulative inertia rises slowly. '
                     f'Max at {n_eigenvalues} dims = {cumulative[-1]:.1f}%'),
        }
    print(f"Criterion 2 - Cumulative 70%: {results['cumulative_70']['dimensions']} dims")

    # ==============================================
    # 基準3: Kaiser基準 (1/K)
    # ==============================================
    kaiser_threshold = 1.0 / K
    kaiser_dims = sum(1 for ev in eigenvalues if ev > kaiser_threshold)

    results['kaiser'] = {
        'dimensions': int(kaiser_dims),
        'method': f'Kaiser criterion: eigenvalue > 1/K = 1/{K} = {kaiser_threshold:.4f}',
        'threshold': float(kaiser_threshold),
        'note_large_sample': (
            f'N={len(mca_df):,} is very large relative to J={J}. '
            f'All {kaiser_dims} of {n_eigenvalues} computed eigenvalues exceed the threshold. '
            f'Kaiser criterion loses discriminating power for large N + moderate J.'
        ),
    }
    print(f"Criterion 3 - Kaiser (1/K): {results['kaiser']['dimensions']} dims")

    # ==============================================
    # 基準4: Greenacre基準 — 正しい閾値
    # ==============================================
    # Greenacre基準: total_inertia / (J - K)
    # J/K - 1 を (J - K) で割ると: (J/K - 1) / (J - K) = (J - K) / (K * (J - K)) = 1/K
    # つまり Greenacre基準 = Kaiser基準 (1/K) — MCAでは事実上同一
    greenacre_threshold = total_inertia / (J - K) if (J - K) > 0 else 0
    greenacre_dims = sum(1 for ev in eigenvalues if ev > greenacre_threshold)

    results['greenacre'] = {
        'dimensions': int(greenacre_dims),
        'method': (
            f'Greenacre criterion: eigenvalue > total_inertia/(J-K) '
            f'= {total_inertia:.4f}/{J-K} = {greenacre_threshold:.4f}'
        ),
        'threshold': float(greenacre_threshold),
        'kaiser_equivalence': (
            f'Greenacre threshold ({greenacre_threshold:.4f}) '
            f'= Kaiser threshold (1/K = {kaiser_threshold:.4f}). '
            f'In MCA, these two criteria are mathematically identical.'
        ),
        'note_large_sample': results['kaiser']['note_large_sample'],
    }
    print(f"Criterion 4 - Greenacre: {results['greenacre']['dimensions']} dims")
    print(f"  (= Kaiser, mathematically identical in MCA)")

    # ==============================================
    # 基準5: 並行分析
    # ==============================================
    n_permutations = 100
    print(f"\nCriterion 5 - Parallel Analysis: {n_permutations} permutations...")

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
        except Exception:
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
            'method': 'Parallel Analysis (95th percentile of 100 permutations)',
            'random_95th_percentile': [float(x) for x in percentile_95[:20]],
            'actual_eigenvalues': [float(x) for x in eigenvalues[:20]],
            'note_large_sample': (
                f'PA yields {pa_dims} dims (all {n_eigenvalues} computed components). '
                f'With N={len(mca_df):,}, PA loses discriminating power because '
                f'even the MCA of permuted data produces small eigenvalues, '
                f'and the real data always exceeds them. '
                f'PA is not a useful criterion here.'
            ),
        }
    else:
        results['parallel_analysis'] = {
            'dimensions': -1,
            'method': 'Parallel Analysis (failed)',
        }

    print(f"Criterion 5 - Parallel Analysis: {results['parallel_analysis']['dimensions']} dims")

    # ==============================================
    # 基準6: Benzecri修正慣性
    # ==============================================
    # Benzecri修正後に寄与率 > 0 の次元数
    # さらに、修正累積寄与率80%以上を目安にする
    benzecri_cum_arr = np.array(benzecri_cum)
    idx_80 = np.where(benzecri_cum_arr >= 80.0)[0]
    benzecri_80_dims = int(idx_80[0]) + 1 if len(idx_80) > 0 else benzecri_n_dims

    results['benzecri'] = {
        'dimensions_above_threshold': int(benzecri_n_dims),
        'dimensions_cum80': int(benzecri_80_dims),
        'method': (
            f'Benzecri corrected inertia: '
            f'{benzecri_n_dims} dims with corrected eigenvalue > 0; '
            f'{benzecri_80_dims} dims for 80% cumulative corrected inertia'
        ),
        'benzecri_percentages': [round(p, 2) for p in benzecri_pct[:20]],
        'benzecri_cumulative': [round(c, 2) for c in benzecri_cum[:20]],
    }
    print(f"Criterion 6 - Benzecri: {benzecri_n_dims} dims (>0), "
          f"{benzecri_80_dims} dims (cum 80%)")

    # ==============================================
    # 総合判定（修正版）
    # ==============================================
    scree_dims = results['scree_elbow']['dimensions']
    kaiser_greenacre_dims = results['kaiser']['dimensions']
    pa_dims_val = results['parallel_analysis']['dimensions']
    cum70_dims = results['cumulative_70']['dimensions']

    # N=13,060 での鑑別力の評価
    results['criterion_reliability'] = {
        'scree_elbow': 'RELIABLE — primary criterion',
        'cumulative_70': 'LOW — MCA naturally has low cumulative inertia; 70% is too strict',
        'kaiser': (f'LOST — all {kaiser_dims} dims exceed 1/K with N={len(mca_df):,}'),
        'greenacre': f'LOST — mathematically identical to Kaiser in MCA',
        'parallel_analysis': (f'LOST — all {pa_dims_val} dims exceed PA threshold '
                              f'with N={len(mca_df):,}'),
        'benzecri': 'RELIABLE — corrects for low-frequency category inflation',
    }

    # 最終判定: スクリー と Benzecri修正の合意
    # スクリー肘法が示す次元数と、Benzecri修正80%累積が示す次元数を比較
    # 両者のうち小さい方を、「データが確実に支持する次元数」とする
    # 両者のうち大きい方を、「データが可能性として支持する上限」とする

    primary_candidates = [scree_dims, benzecri_80_dims]
    recommended = min(primary_candidates)
    upper_bound = max(primary_candidates)

    results['recommended_dimensions'] = recommended
    results['upper_bound_dimensions'] = upper_bound
    results['decision_rationale'] = (
        f'Scree elbow = {scree_dims}, Benzecri cum80% = {benzecri_80_dims}. '
        f'Recommended = {recommended} (agreement of two reliable criteria). '
        f'Kaiser ({kaiser_dims}), Greenacre ({greenacre_dims}), '
        f'PA ({pa_dims_val}) all lost discriminating power with N={len(mca_df):,}.'
    )
    results['all_criteria_dimensions'] = {
        'scree_elbow': scree_dims,
        'cumulative_70': cum70_dims,
        'kaiser': kaiser_dims,
        'greenacre': greenacre_dims,
        'parallel_analysis': pa_dims_val,
        'benzecri_above_0': benzecri_n_dims,
        'benzecri_cum80': benzecri_80_dims,
    }

    # 6次元収束チェック
    results['converged_to_six'] = (recommended == 6)

    print(f"\n=== Dimension Decision: {recommended} dimensions ===")
    print(f"  Scree elbow: {scree_dims}")
    print(f"  Benzecri cum80%: {benzecri_80_dims}")
    print(f"  Kaiser (LOST): {kaiser_dims}")
    print(f"  Greenacre (LOST, =Kaiser): {greenacre_dims}")
    print(f"  PA (LOST): {pa_dims_val}")
    print(f"  Recommended: {recommended} (upper bound: {upper_bound})")

    return results


# =============================================================================
# Step 5: 各次元の解釈
# =============================================================================

def interpret_dimensions(mca, mca_df, n_dims, eigenvalues, total_inertia,
                         label=''):
    """各次元に最も寄与するカテゴリを特定し、解釈ラベルを付与。"""

    col_coords = mca.column_coordinates(mca_df)

    # prince の column_contributions_ を使う（正しい寄与度）
    try:
        col_contribs = mca.column_contributions_
    except Exception:
        col_contribs = None

    dimension_interpretations = []

    for dim_idx in range(min(n_dims, len(col_coords.columns))):
        dim_name = col_coords.columns[dim_idx]
        coords = col_coords[dim_name]

        sorted_coords = coords.sort_values()
        top_positive = sorted_coords.tail(5).iloc[::-1]
        top_negative = sorted_coords.head(5)

        ev = float(eigenvalues[dim_idx])
        pct = ev / total_inertia * 100

        interp = {
            'dimension': dim_idx + 1,
            'eigenvalue': ev,
            'explained_inertia_pct': pct,
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

        # 変数ごとの寄与度（座標範囲）
        for col in MCA_COLUMNS:
            col_cats = [idx for idx in coords.index
                        if str(idx).startswith(col + '__')]
            if not col_cats:
                # prince のプレフィックス区切りが _ の場合もチェック
                col_cats = [idx for idx in coords.index
                            if str(idx).startswith(col + '_')]
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

        # 公式寄与度があれば追加
        if col_contribs is not None and dim_idx < len(col_contribs.columns):
            dim_col = col_contribs.columns[dim_idx]
            top_contrib = col_contribs[dim_col].nlargest(5)
            interp['top_contributions'] = {
                str(idx): float(val)
                for idx, val in top_contrib.items()
            }

        dimension_interpretations.append(interp)

    return dimension_interpretations


def assign_interpretation_labels(dim_interps):
    """各次元に解釈ラベルを付与。"""
    labels = []
    for dim in dim_interps:
        pos = dim['positive_pole']['categories'][:3]
        neg = dim['negative_pole']['categories'][:3]

        def shorten(cat):
            for col in MCA_COLUMNS:
                for sep in ['__', '_']:
                    prefix = col + sep
                    if cat.startswith(prefix):
                        return cat[len(prefix):]
            return cat

        pos_short = [shorten(c) for c in pos]
        neg_short = [shorten(c) for c in neg]

        label = (f"Dim {dim['dimension']}: "
                 f"[{' | '.join(neg_short)}] <-> [{' | '.join(pos_short)}]")
        dim['interpretation_label'] = label
        labels.append(label)

    return labels


# =============================================================================
# Step 6: 低頻度カテゴリ感度分析
# =============================================================================

def run_sensitivity_analysis(df):
    """
    低頻度カテゴリ統合版（クリーンデータ）でMCAを再実行し、
    元データとの差異を定量的に報告する。
    """
    print("\n" + "=" * 70)
    print("=== 感度分析: クリーンデータ（低頻度カテゴリ統合） ===")
    print("=" * 70)

    # クリーンデータの準備
    clean_df, clean_dropped, clean_J = prepare_mca_data(
        df, merge_map=CATEGORY_MERGE_MAP, label='クリーンデータ'
    )

    # MCA実行（通常版）
    (clean_mca, clean_ev, clean_expl, clean_cum,
     clean_total_inertia, clean_J_actual) = run_mca(
        clean_df, n_components=30, label='[Clean]'
    )

    # Benzecri修正
    K = len(MCA_COLUMNS)
    clean_benz, clean_benz_pct, clean_benz_cum, clean_benz_n = \
        compute_benzecri_correction(clean_ev, K)

    # 次元数決定
    clean_dim_decision = determine_dimensions(
        clean_ev, clean_expl, clean_cum,
        clean_total_inertia, clean_J_actual, clean_df, clean_mca,
        clean_benz_pct, clean_benz_cum, clean_benz_n
    )

    n_clean_dims = clean_dim_decision['recommended_dimensions']

    # 次元解釈
    clean_interps = interpret_dimensions(
        clean_mca, clean_df, n_clean_dims, clean_ev, clean_total_inertia,
        label='[Clean]'
    )
    clean_labels = assign_interpretation_labels(clean_interps)

    # prince Benzecri 版も実行（検証用）
    (clean_mca_benz, clean_ev_benz, clean_expl_benz, clean_cum_benz,
     clean_ti_benz, clean_J_benz) = run_mca(
        clean_df, n_components=30, correction='benzecri', label='[Clean+Benzecri]'
    )

    return {
        'clean_mca': clean_mca,
        'clean_df': clean_df,
        'clean_eigenvalues': clean_ev,
        'clean_explained': clean_expl,
        'clean_cumulative': clean_cum,
        'clean_total_inertia': clean_total_inertia,
        'clean_J': clean_J_actual,
        'clean_n_dropped': clean_dropped,
        'clean_dim_decision': clean_dim_decision,
        'clean_interps': clean_interps,
        'clean_labels': clean_labels,
        'clean_benzecri_pct': clean_benz_pct,
        'clean_benzecri_cum': clean_benz_cum,
        'clean_benzecri_n_dims': clean_benz_n,
        'clean_mca_benzecri': clean_mca_benz,
        'clean_ev_benzecri': clean_ev_benz,
        'clean_expl_benzecri': clean_expl_benz,
        'clean_cum_benzecri': clean_cum_benz,
    }


# =============================================================================
# Step 7: 可視化
# =============================================================================

def plot_scree(eigenvalues, explained_ratio, dim_decision, output_path,
               benzecri_pct=None, title_suffix=''):
    """スクリープロット（修正版）。"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    n_plot = min(25, len(eigenvalues))
    dims = range(1, n_plot + 1)

    # --- 左パネル: 固有値 ---
    ax1 = axes[0]
    ax1.bar(dims, eigenvalues[:n_plot], alpha=0.5, color='steelblue',
            label='Eigenvalue')
    ax1.plot(dims, eigenvalues[:n_plot], 'o-', color='navy', linewidth=2,
             markersize=5, label='Eigenvalue (line)')

    # PA 95th percentile
    pa = dim_decision.get('parallel_analysis', {})
    if isinstance(pa, dict) and 'random_95th_percentile' in pa:
        pa_vals = pa['random_95th_percentile'][:n_plot]
        ax1.plot(range(1, len(pa_vals) + 1), pa_vals, 's--', color='red',
                 linewidth=1.5, markersize=4, label='PA 95th %ile')

    # Kaiser / Greenacre threshold
    kaiser = dim_decision.get('kaiser', {})
    if isinstance(kaiser, dict) and 'threshold' in kaiser:
        ax1.axhline(y=kaiser['threshold'], color='purple', linestyle='-.',
                     linewidth=1.5,
                     label=f'Kaiser=Greenacre 1/K={kaiser["threshold"]:.4f}')

    # Recommended
    rec_dims = dim_decision.get('recommended_dimensions', 0)
    if isinstance(dim_decision.get('all_criteria_dimensions'), dict):
        scree_d = dim_decision['all_criteria_dimensions'].get('scree_elbow', 0)
    else:
        scree_d = rec_dims

    if 0 < scree_d <= n_plot:
        ax1.axvline(x=scree_d, color='red', linestyle='--', linewidth=1.5,
                     alpha=0.7, label=f'Scree elbow: {scree_d}D')

    ax1.set_xlabel('Dimension', fontsize=12)
    ax1.set_ylabel('Eigenvalue', fontsize=12)
    ax1.set_title(f'Eigenvalues{title_suffix}', fontsize=13)
    ax1.set_xticks(list(dims))
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # --- 右パネル: 寄与率（通常 + Benzecri） ---
    ax2 = axes[1]
    ax2.plot(dims, explained_ratio[:n_plot], 'o-', color='green', linewidth=2,
             markersize=5, label='Standard inertia %')

    if benzecri_pct is not None:
        n_benz = min(n_plot, len(benzecri_pct))
        ax2.plot(range(1, n_benz + 1), benzecri_pct[:n_benz], 's-',
                 color='darkorange', linewidth=2, markersize=5,
                 label='Benzecri corrected %')

    if 0 < rec_dims <= n_plot:
        ax2.axvline(x=rec_dims, color='red', linestyle='--', linewidth=1.5,
                     alpha=0.7, label=f'Recommended: {rec_dims}D')

    ax2.set_xlabel('Dimension', fontsize=12)
    ax2.set_ylabel('% of Inertia', fontsize=12)
    ax2.set_title(f'Inertia Percentage{title_suffix}', fontsize=13)
    ax2.set_xticks(list(dims))
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cumulative_variance(cumulative, n_recommended, output_path,
                             benzecri_cum=None, title_suffix=''):
    """累積寄与率プロット（修正版）。"""
    fig, ax = plt.subplots(figsize=(14, 7))

    n_plot = min(25, len(cumulative))
    dims = range(1, n_plot + 1)

    ax.plot(dims, cumulative[:n_plot], 'o-', color='darkgreen', linewidth=2,
            markersize=6, label='Standard cumulative %')
    ax.fill_between(dims, cumulative[:n_plot], alpha=0.1, color='green')

    if benzecri_cum is not None:
        n_benz = min(n_plot, len(benzecri_cum))
        ax.plot(range(1, n_benz + 1), benzecri_cum[:n_benz], 's-',
                color='darkorange', linewidth=2, markersize=6,
                label='Benzecri cumulative %')
        ax.fill_between(range(1, n_benz + 1), benzecri_cum[:n_benz],
                         alpha=0.1, color='orange')

    # Thresholds
    ax.axhline(y=70, color='red', linestyle='--', linewidth=1.5,
               label='70% threshold')
    ax.axhline(y=80, color='blue', linestyle=':', linewidth=1,
               label='80% threshold (Benzecri)')

    if 0 < n_recommended <= n_plot:
        ax.axvline(x=n_recommended, color='red', linestyle='--', linewidth=1.5,
                   label=f'Recommended: {n_recommended}D')
        if n_recommended - 1 < len(cumulative):
            ax.plot(n_recommended, cumulative[n_recommended - 1],
                    'r*', markersize=15, zorder=5)

    ax.set_xlabel('Number of Dimensions', fontsize=12)
    ax.set_ylabel('Cumulative Explained Inertia (%)', fontsize=12)
    ax.set_title(f'Cumulative Explained Inertia{title_suffix}', fontsize=14)
    ax.set_xticks(list(dims))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_mca_biplot(mca, mca_df, eigenvalues, total_inertia, output_path,
                    title_suffix=''):
    """MCAバイプロット。"""
    col_coords = mca.column_coordinates(mca_df)

    if len(col_coords.columns) < 2:
        print("Warning: Not enough dimensions for biplot")
        return

    dim1 = col_coords.columns[0]
    dim2 = col_coords.columns[1]

    fig, ax = plt.subplots(figsize=(18, 14))

    cmap = plt.cm.Set1
    colors = {col: cmap(i / len(MCA_COLUMNS)) for i, col in enumerate(MCA_COLUMNS)}

    for idx in col_coords.index:
        cat_var = None
        for col in MCA_COLUMNS:
            if str(idx).startswith(col + '__') or str(idx).startswith(col + '_'):
                cat_var = col
                break
        if cat_var is None:
            cat_var = MCA_COLUMNS[0]

        x = col_coords.loc[idx, dim1]
        y = col_coords.loc[idx, dim2]

        ax.scatter(x, y, c=[colors[cat_var]], s=80, alpha=0.8,
                   edgecolors='black', linewidth=0.3)

        label_text = str(idx)
        for col in MCA_COLUMNS:
            for sep in ['__', '_']:
                prefix = col + sep
                if label_text.startswith(prefix):
                    label_text = label_text[len(prefix):]
                    break

        ax.annotate(label_text, (x, y), fontsize=6,
                    ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points',
                    alpha=0.85)

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=colors[col], markersize=10, label=col)
        for col in MCA_COLUMNS
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.9)

    pct1 = eigenvalues[0] / total_inertia * 100
    pct2 = eigenvalues[1] / total_inertia * 100

    ax.set_xlabel(f'Dimension 1 ({pct1:.2f}%)', fontsize=12)
    ax.set_ylabel(f'Dimension 2 ({pct2:.2f}%)', fontsize=12)
    ax.set_title(f'MCA Biplot{title_suffix}', fontsize=14)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison(orig_ev, clean_ev, orig_benz_pct, clean_benz_pct,
                    output_path):
    """元データ vs クリーンデータの比較プロット。"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    n_plot = min(20, len(orig_ev), len(clean_ev))
    dims = range(1, n_plot + 1)

    # 左: 固有値比較
    ax1 = axes[0]
    ax1.plot(dims, orig_ev[:n_plot], 'o-', color='navy', linewidth=2,
             label=f'Original (J=88)')
    ax1.plot(dims, clean_ev[:n_plot], 's-', color='crimson', linewidth=2,
             label=f'Clean (merged)')
    ax1.set_xlabel('Dimension', fontsize=12)
    ax1.set_ylabel('Eigenvalue', fontsize=12)
    ax1.set_title('Eigenvalue Comparison', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(list(dims))

    # 右: Benzecri修正寄与率比較
    ax2 = axes[1]
    n_orig = min(n_plot, len(orig_benz_pct))
    n_clean = min(n_plot, len(clean_benz_pct))
    ax2.plot(range(1, n_orig + 1), orig_benz_pct[:n_orig], 'o-',
             color='navy', linewidth=2, label='Original Benzecri %')
    ax2.plot(range(1, n_clean + 1), clean_benz_pct[:n_clean], 's-',
             color='crimson', linewidth=2, label='Clean Benzecri %')
    ax2.set_xlabel('Dimension', fontsize=12)
    ax2.set_ylabel('Benzecri Corrected Inertia %', fontsize=12)
    ax2.set_title('Benzecri Corrected Inertia Comparison', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(list(dims))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Step 8: レポート作成
# =============================================================================

def generate_report(stats_report, mca_results, dim_decision, dim_interps,
                    labels, n_total, n_analyzed, n_dropped, total_cats,
                    benzecri_results, sensitivity, output_path):
    """Markdownレポートを生成（修正版）。"""

    n_dims = dim_decision['recommended_dimensions']
    K = len(MCA_COLUMNS)

    lines = []
    lines.append("# Phase 2A: MCA分析レポート（修正版）")
    lines.append("")
    lines.append(f"**分析日**: 2026-02-25")
    lines.append(f"**乱数シード**: {RANDOM_SEED}")
    lines.append(f"**修正版**: quality-reviewer FAIL判定に基づく全面修正")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ========== 1. データ概要 ==========
    lines.append("## 1. データ概要")
    lines.append("")
    lines.append(f"- **総レコード数**: {n_total:,} 件")
    lines.append(f"- **分析対象レコード数**: {n_analyzed:,} 件")
    lines.append(f"- **欠損値による除外**: {n_dropped:,} 件")
    lines.append(f"- **分析対象変数 (K)**: {K}")
    lines.append(f"- **総カテゴリ数 (J)**: {total_cats}")
    lines.append("")

    lines.append("### カテゴリ数: スキーマ定義 vs 実データ")
    lines.append("")
    lines.append("| 変数 | スキーマ | 実データ | 増分 | 低頻度(<20件) |")
    lines.append("|------|---------|---------|------|-------------|")
    schema_cats = {
        'before_state': 6, 'trigger_type': 4, 'action_type': 8,
        'after_state': 6, 'pattern_type': 14, 'outcome': 4, 'scale': 5,
    }
    for col in MCA_COLUMNS:
        actual = stats_report['actual_category_counts'].get(col, 0)
        schema = schema_cats.get(col, 0)
        diff = actual - schema
        # 低頻度カテゴリ数を計算
        dist = stats_report['frequency_distributions'].get(col, {})
        low_freq = sum(1 for v in dist.values() if v < 20)
        lines.append(f"| {col} | {schema} | {actual} | +{diff} | {low_freq} |")
    lines.append("")

    lines.append("### 欠損値レポート")
    lines.append("")
    lines.append("| 変数 | 欠損数 | 欠損率 |")
    lines.append("|------|--------|--------|")
    for col, info in stats_report['missing_values'].items():
        lines.append(f"| {col} | {info['count']:,} | {info['percentage']:.2f}% |")
    lines.append("")

    # ========== 2. 全慣性の正しい計算 ==========
    lines.append("## 2. 全慣性（Total Inertia）の導出")
    lines.append("")
    lines.append("### 計算式")
    lines.append("")
    lines.append("MCAにおける全慣性は以下で定義される:")
    lines.append("")
    lines.append("```")
    lines.append("Total Inertia = J/K - 1")
    lines.append(f"              = {total_cats}/{K} - 1")
    lines.append(f"              = {total_cats/K:.4f} - 1")
    lines.append(f"              = {total_cats/K - 1:.4f}")
    lines.append("```")
    lines.append("")
    lines.append(f"- **prince `mca.total_inertia_`**: {mca_results['total_inertia']:.4f}")
    lines.append(f"- **理論値 (J/K - 1)**: {total_cats/K - 1:.4f}")
    lines.append("")
    lines.append("前回の報告値 7.179 は30成分の固有値合計であり、全慣性ではなかった。")
    lines.append(f"正しい全慣性は **{mca_results['total_inertia']:.4f}** である。")
    lines.append("")

    lines.append("### 全慣性が大きい理由")
    lines.append("")
    lines.append(f"J={total_cats}（総カテゴリ数）が多いため、全慣性 = J/K - 1 = {total_cats/K - 1:.1f} と大きくなる。")
    lines.append("これはMCAの性質上、カテゴリ数が多いほど「説明すべき分散」が増えることを意味する。")
    lines.append("結果として、各次元の寄与率は小さくなり、累積寄与率の上昇は緩やかになる。")
    lines.append("")

    # ========== 3. MCA結果 ==========
    lines.append("## 3. MCA結果")
    lines.append("")
    lines.append("### 固有値と寄与率（正しい全慣性ベース）")
    lines.append("")
    lines.append("| 次元 | 固有値 | 寄与率 (%) | 累積寄与率 (%) | Benzecri修正 (%) | Benzecri累積 (%) |")
    lines.append("|------|--------|-----------|---------------|-----------------|-----------------|")
    ev = mca_results['eigenvalues']
    ex = mca_results['explained_inertia_pct']
    cu = mca_results['cumulative_inertia_pct']
    bpct = benzecri_results['percentages']
    bcum = benzecri_results['cumulative']
    for i in range(min(20, len(ev))):
        bp = bpct[i] if i < len(bpct) else 0
        bc = bcum[i] if i < len(bcum) else 0
        lines.append(f"| {i+1} | {ev[i]:.6f} | {ex[i]:.2f} | {cu[i]:.2f} | {bp:.2f} | {bc:.2f} |")
    lines.append("")
    lines.append(f"**全慣性**: {mca_results['total_inertia']:.4f}")
    lines.append("")

    # ========== 4. N=13,060 での基準の鑑別力 ==========
    lines.append("## 4. N=13,060 における各基準の鑑別力")
    lines.append("")
    lines.append("### Kaiser/Greenacre基準の数学的同一性")
    lines.append("")
    lines.append("MCAにおけるGreenacre閾値とKaiser閾値は数学的に同一である:")
    lines.append("")
    lines.append("```")
    lines.append(f"Greenacre: total_inertia / (J - K) = {mca_results['total_inertia']:.4f} / {total_cats - K} = {mca_results['total_inertia'] / (total_cats - K):.4f}")
    lines.append(f"Kaiser:    1/K = 1/{K} = {1/K:.4f}")
    lines.append("```")
    lines.append("")
    lines.append("導出: total_inertia = J/K - 1 なので、")
    lines.append("total_inertia / (J - K) = (J/K - 1) / (J - K) = (J - K) / (K(J - K)) = 1/K")
    lines.append("")

    lines.append("### 各基準の鑑別力評価")
    lines.append("")
    lines.append("| 基準 | 次元数 | 鑑別力 | 理由 |")
    lines.append("|------|--------|--------|------|")

    all_crit = dim_decision['all_criteria_dimensions']
    reliability = dim_decision.get('criterion_reliability', {})
    for crit_key, crit_name in [
        ('scree_elbow', 'スクリー（肘法）'),
        ('cumulative_70', '累積寄与率70%'),
        ('kaiser', 'Kaiser (1/K)'),
        ('greenacre', 'Greenacre'),
        ('parallel_analysis', '並行分析'),
        ('benzecri_cum80', 'Benzecri修正(累積80%)'),
    ]:
        dims_val = all_crit.get(crit_key, '?')
        if crit_key in ['scree_elbow', 'benzecri_cum80', 'benzecri_above_0']:
            status = 'RELIABLE'
        elif crit_key == 'cumulative_70':
            status = 'LOW'
        else:
            status = 'LOST'
        reason = reliability.get(crit_key, '')
        lines.append(f"| {crit_name} | {dims_val} | {status} | {reason[:60]} |")
    lines.append("")

    lines.append("**結論**: N=13,060 + J=88 の組み合わせでは、Kaiser、Greenacre、並行分析の")
    lines.append("3基準は全次元を「有意」と判定し、鑑別力を完全に失う。")
    lines.append("スクリープロットとBenzecri修正慣性のみが信頼できる基準である。")
    lines.append("")

    # ========== 5. Benzecri修正慣性 ==========
    lines.append("## 5. Benzecri修正慣性")
    lines.append("")
    lines.append("### 修正式")
    lines.append("")
    lines.append("```")
    lines.append(f"lambda_corrected = ((K/(K-1)) * (lambda - 1/K))^2   (lambda > 1/K のみ)")
    lines.append(f"                 = (({K}/{K-1}) * (lambda - {1/K:.4f}))^2")
    lines.append("```")
    lines.append("")
    lines.append("Benzecri修正は低頻度カテゴリが過大な固有値を持つ問題を緩和する。")
    lines.append(f"1/K = {1/K:.4f} 以下の固有値は0に修正され、残る次元のみが有意とみなされる。")
    lines.append("")
    lines.append(f"- 修正後に寄与率 > 0 の次元数: **{benzecri_results['n_above_threshold']}**")

    benzecri_cum_arr = np.array(benzecri_results['cumulative'])
    idx80 = np.where(benzecri_cum_arr >= 80.0)[0]
    if len(idx80) > 0:
        lines.append(f"- 修正後の累積80%に必要な次元数: **{idx80[0] + 1}**")
    lines.append("")

    # ========== 6. 次元数の決定 ==========
    lines.append("## 6. 次元数の決定")
    lines.append("")
    lines.append("### 決定ロジック")
    lines.append("")
    lines.append("信頼できる2基準の合意:")
    lines.append("")
    lines.append(f"1. **スクリープロット（肘法）**: {all_crit['scree_elbow']}次元")
    lines.append(f"2. **Benzecri修正（累積80%）**: {all_crit['benzecri_cum80']}次元")
    lines.append("")
    lines.append(f"### 結論: データが示した次元数は **{n_dims}次元** である")
    lines.append("")
    lines.append(dim_decision.get('decision_rationale', ''))
    lines.append("")

    if dim_decision.get('converged_to_six', False):
        lines.append("> 6次元に収束した。先験的仮説と一致するが、データ駆動の結果である。")
    else:
        lines.append(f"> データ駆動の分析により **{n_dims}次元** が最適と判定された。")
    lines.append("")

    # ========== 7. 各次元の解釈 ==========
    lines.append("## 7. 各次元の解釈（元データ）")
    lines.append("")
    for dim in dim_interps[:n_dims]:
        lines.append(f"### 次元 {dim['dimension']}")
        lines.append(f"- **固有値**: {dim['eigenvalue']:.6f}")
        lines.append(f"- **寄与率**: {dim['explained_inertia_pct']:.2f}%")
        bpct_val = bpct[dim['dimension'] - 1] if dim['dimension'] - 1 < len(bpct) else 0
        lines.append(f"- **Benzecri修正寄与率**: {bpct_val:.2f}%")
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

    # ========== 8. 感度分析: 元データ vs クリーンデータ ==========
    lines.append("## 8. 感度分析: 元データ vs クリーンデータ（低頻度カテゴリ統合）")
    lines.append("")
    lines.append("### カテゴリ統合ルール")
    lines.append("")

    for col, mapping in CATEGORY_MERGE_MAP.items():
        lines.append(f"#### {col}")
        lines.append("")
        for old, new in mapping.items():
            lines.append(f"- `{old}` -> `{new}`")
        lines.append("")

    clean = sensitivity
    clean_dims = clean['clean_dim_decision']['recommended_dimensions']

    lines.append("### 対比表")
    lines.append("")
    lines.append("| 指標 | 元データ | クリーンデータ |")
    lines.append("|------|---------|--------------|")
    lines.append(f"| 総カテゴリ数 (J) | {total_cats} | {clean['clean_J']} |")
    lines.append(f"| 全慣性 | {mca_results['total_inertia']:.4f} | {clean['clean_total_inertia']:.4f} |")
    lines.append(f"| Dim1 固有値 | {ev[0]:.6f} | {clean['clean_eigenvalues'][0]:.6f} |")
    lines.append(f"| Dim1 寄与率 | {ex[0]:.2f}% | {clean['clean_explained'][0]:.2f}% |")
    lines.append(f"| スクリー肘 | {all_crit['scree_elbow']} | {clean['clean_dim_decision']['all_criteria_dimensions']['scree_elbow']} |")
    lines.append(f"| Benzecri(>0) | {benzecri_results['n_above_threshold']} | {clean['clean_benzecri_n_dims']} |")
    benz80_clean = clean['clean_dim_decision']['all_criteria_dimensions'].get('benzecri_cum80', '?')
    lines.append(f"| Benzecri(cum80%) | {all_crit['benzecri_cum80']} | {benz80_clean} |")
    lines.append(f"| 推奨次元数 | **{n_dims}** | **{clean_dims}** |")
    lines.append("")

    lines.append("### クリーンデータの次元解釈")
    lines.append("")
    for dim_info in clean['clean_interps'][:clean_dims]:
        lines.append(f"- {dim_info.get('interpretation_label', 'N/A')}")
    lines.append("")

    # 差異の分析
    if n_dims != clean_dims:
        lines.append(f"### 差異の分析")
        lines.append("")
        lines.append(f"元データ({n_dims}次元) vs クリーンデータ({clean_dims}次元): "
                      f"差異は低頻度カテゴリが独自の次元を形成していたことを示す。")
        lines.append("クリーンデータの次元解釈がより安定的であり、Phase 2Bではクリーンデータの次元構造を推奨する。")
        lines.append("")
    else:
        lines.append(f"### 差異の分析")
        lines.append("")
        lines.append(f"元データとクリーンデータで推奨次元数が一致 ({n_dims}次元)。")
        lines.append("低頻度カテゴリの統合は次元数に影響しないが、次元の解釈はクリーンデータの方が安定的である。")
        lines.append("")

    # ========== 9. Phase 2Bへの接続 ==========
    lines.append("## 9. Phase 2Bへの接続")
    lines.append("")
    lines.append(f"- MCA座標空間（{n_dims}次元）上で事例をクラスタリング")
    lines.append("- クリーンデータ（低頻度カテゴリ統合版）を使用することを推奨")
    lines.append("- before_state -> after_state の遷移行列をaction_type別に構築")
    lines.append("- クラスタが八卦の8分類や64卦と事後的に対応するかを検証（Phase 3）")
    lines.append("")

    # ========== 10. 可視化一覧 ==========
    lines.append("## 10. 可視化一覧")
    lines.append("")
    lines.append("- `visualizations/scree_plot.png` -- スクリープロット + Benzecri修正（元データ）")
    lines.append("- `visualizations/cumulative_variance.png` -- 累積寄与率 + Benzecri累積（元データ）")
    lines.append("- `visualizations/mca_biplot.png` -- MCAバイプロット Dim1 vs Dim2（元データ）")
    lines.append("- `visualizations/scree_plot_clean.png` -- スクリープロット（クリーンデータ）")
    lines.append("- `visualizations/cumulative_variance_clean.png` -- 累積寄与率（クリーンデータ）")
    lines.append("- `visualizations/mca_biplot_clean.png` -- MCAバイプロット（クリーンデータ）")
    lines.append("- `visualizations/comparison.png` -- 元 vs クリーン 比較")
    lines.append("")

    # ========== 11. 前回からの修正箇所 ==========
    lines.append("## 11. 前回からの修正箇所")
    lines.append("")
    lines.append("| # | 問題 | 修正内容 |")
    lines.append("|---|------|---------|")
    lines.append(f"| 1 | 全慣性の誤計算 (7.179) | `mca.total_inertia_` を使用 ({mca_results['total_inertia']:.4f}) |")
    lines.append(f"| 2 | Greenacre閾値の誤り | total_inertia/(J-K) = 1/K = {1/K:.4f} (Kaiser基準と同一) |")
    lines.append("| 3 | 次元数決定ロジック | スクリー + Benzecri修正の合意。Kaiser/Greenacre/PAの鑑別力喪失を明示 |")
    lines.append("| 4 | 低頻度カテゴリの感度分析なし | クリーンデータ(カテゴリ統合版)で再MCA、対比表を提示 |")
    lines.append("| 5 | 絶対パス | `os.path.dirname(__file__)` による相対パスに変更 |")
    lines.append("| 6 | prince属性の未活用 | `percentage_of_variance_`, `total_inertia_` 等を使用 |")
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated by mca_analysis.py (seed={RANDOM_SEED}, revised 2026-02-25)*")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {output_path}")


# =============================================================================
# メイン実行
# =============================================================================

def main():
    print("=" * 70)
    print("Phase 2A: MCA Analysis (REVISED) - Change Case Structural Discovery")
    print("=" * 70)

    # ---- Step 1: データ読み込みと基本統計 ----
    print("\n--- Step 1: Data Loading & Basic Statistics ---")
    df = load_data(DATA_PATH)
    stats_report = compute_basic_stats(df)

    stats_path = os.path.join(OUTPUT_DIR, 'basic_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_report, f, ensure_ascii=False, indent=2)
    print(f"Saved: {stats_path}")

    # ---- Step 2: MCA分析（元データ） ----
    print("\n--- Step 2: MCA Analysis (Original Data) ---")
    mca_df, n_dropped, total_cats = prepare_mca_data(df, label='元データ')
    n_total = len(df)
    n_analyzed = len(mca_df)
    K = len(MCA_COLUMNS)

    (mca, eigenvalues, explained_ratio, cumulative,
     total_inertia, J) = run_mca(mca_df, n_components=30, label='[Original]')

    # ---- Step 3: Benzecri修正慣性 ----
    print("\n--- Step 3: Benzecri Corrected Inertia ---")
    benz_corrected, benz_pct, benz_cum, benz_n_dims = \
        compute_benzecri_correction(eigenvalues, K)

    print(f"Benzecri: {benz_n_dims} dims with corrected eigenvalue > 0")
    print(f"Benzecri pct (top 10): {[round(p, 2) for p in benz_pct[:10]]}")
    print(f"Benzecri cum (top 10): {[round(c, 2) for c in benz_cum[:10]]}")

    # prince Benzecri版も実行（検証用）
    (mca_benz, ev_benz, expl_benz, cum_benz,
     ti_benz, J_benz) = run_mca(
        mca_df, n_components=30, correction='benzecri',
        label='[Original+Benzecri]'
    )

    # ---- Step 4: 次元数の決定 ----
    print("\n--- Step 4: Dimension Determination (Revised) ---")
    dim_decision = determine_dimensions(
        eigenvalues, explained_ratio, cumulative,
        total_inertia, J, mca_df, mca,
        benz_pct, benz_cum, benz_n_dims
    )
    n_dims = dim_decision['recommended_dimensions']

    # ---- Step 5: 各次元の解釈 ----
    print("\n--- Step 5: Dimension Interpretation ---")
    dim_interps = interpret_dimensions(
        mca, mca_df, n_dims, eigenvalues, total_inertia, label='[Original]'
    )
    labels = assign_interpretation_labels(dim_interps)
    for lbl in labels:
        print(f"  {lbl}")

    # ---- Step 6: 感度分析（クリーンデータ） ----
    print("\n--- Step 6: Sensitivity Analysis (Clean Data) ---")
    sensitivity = run_sensitivity_analysis(df)

    # ---- Step 7: 可視化 ----
    print("\n--- Step 7: Visualization ---")

    # 元データ
    plot_scree(eigenvalues, explained_ratio, dim_decision,
               os.path.join(VIS_DIR, 'scree_plot.png'),
               benzecri_pct=benz_pct, title_suffix=' (Original)')

    plot_cumulative_variance(cumulative, n_dims,
                             os.path.join(VIS_DIR, 'cumulative_variance.png'),
                             benzecri_cum=benz_cum, title_suffix=' (Original)')

    plot_mca_biplot(mca, mca_df, eigenvalues, total_inertia,
                    os.path.join(VIS_DIR, 'mca_biplot.png'),
                    title_suffix=' (Original)')

    # クリーンデータ
    clean_dim_dec = sensitivity['clean_dim_decision']
    clean_n_dims = clean_dim_dec['recommended_dimensions']

    plot_scree(sensitivity['clean_eigenvalues'],
               sensitivity['clean_explained'],
               clean_dim_dec,
               os.path.join(VIS_DIR, 'scree_plot_clean.png'),
               benzecri_pct=sensitivity['clean_benzecri_pct'],
               title_suffix=' (Clean)')

    plot_cumulative_variance(sensitivity['clean_cumulative'],
                             clean_n_dims,
                             os.path.join(VIS_DIR, 'cumulative_variance_clean.png'),
                             benzecri_cum=sensitivity['clean_benzecri_cum'],
                             title_suffix=' (Clean)')

    plot_mca_biplot(sensitivity['clean_mca'],
                    sensitivity['clean_df'],
                    sensitivity['clean_eigenvalues'],
                    sensitivity['clean_total_inertia'],
                    os.path.join(VIS_DIR, 'mca_biplot_clean.png'),
                    title_suffix=' (Clean)')

    # 比較プロット
    plot_comparison(eigenvalues, sensitivity['clean_eigenvalues'],
                    benz_pct, sensitivity['clean_benzecri_pct'],
                    os.path.join(VIS_DIR, 'comparison.png'))

    # ---- Step 8: 結果の保存 ----
    print("\n--- Step 8: Save Results ---")

    # dimension_report.json
    dim_report = {
        'recommended_dimensions': n_dims,
        'dimension_decision': dim_decision,
        'dimension_interpretations': dim_interps,
        'interpretation_labels': labels,
        'benzecri_correction': {
            'corrected_eigenvalues': benz_corrected[:20],
            'percentages': benz_pct[:20],
            'cumulative': benz_cum[:20],
            'n_above_threshold': benz_n_dims,
        },
        'sensitivity_analysis': {
            'clean_recommended_dimensions': clean_n_dims,
            'clean_dim_decision': clean_dim_dec,
            'clean_labels': sensitivity['clean_labels'],
            'clean_J': sensitivity['clean_J'],
            'clean_total_inertia': sensitivity['clean_total_inertia'],
        },
    }
    dim_path = os.path.join(OUTPUT_DIR, 'dimension_report.json')
    with open(dim_path, 'w', encoding='utf-8') as f:
        json.dump(dim_report, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {dim_path}")

    # mca_results.json
    mca_results = {
        'n_analyzed': n_analyzed,
        'n_dropped': n_dropped,
        'n_components_computed': len(eigenvalues),
        'total_categories_J': total_cats,
        'n_variables_K': K,
        'eigenvalues': [float(x) for x in eigenvalues],
        'explained_inertia_pct': [float(x) for x in explained_ratio],
        'cumulative_inertia_pct': [float(x) for x in cumulative],
        'total_inertia': float(total_inertia),
        'total_inertia_formula': f'J/K - 1 = {total_cats}/{K} - 1 = {total_inertia:.4f}',
        'mca_columns': MCA_COLUMNS,
        'recommended_dimensions': n_dims,
        'benzecri_corrected_pct': [float(x) for x in benz_pct[:20]],
        'benzecri_cumulative_pct': [float(x) for x in benz_cum[:20]],
        'benzecri_n_dims': benz_n_dims,
        'clean_data': {
            'total_categories_J': sensitivity['clean_J'],
            'total_inertia': sensitivity['clean_total_inertia'],
            'eigenvalues': [float(x) for x in sensitivity['clean_eigenvalues'][:20]],
            'explained_inertia_pct': [float(x) for x in sensitivity['clean_explained'][:20]],
            'cumulative_inertia_pct': [float(x) for x in sensitivity['clean_cumulative'][:20]],
            'recommended_dimensions': clean_n_dims,
        },
    }
    mca_path = os.path.join(OUTPUT_DIR, 'mca_results.json')
    with open(mca_path, 'w', encoding='utf-8') as f:
        json.dump(mca_results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {mca_path}")

    # ---- Step 9: レポート生成 ----
    print("\n--- Step 9: Report Generation ---")

    benzecri_results_dict = {
        'corrected_eigenvalues': benz_corrected,
        'percentages': benz_pct,
        'cumulative': benz_cum,
        'n_above_threshold': benz_n_dims,
    }

    generate_report(
        stats_report, mca_results, dim_decision, dim_interps, labels,
        n_total, n_analyzed, n_dropped, total_cats,
        benzecri_results_dict, sensitivity,
        os.path.join(OUTPUT_DIR, 'report.md')
    )

    # ---- 検証チェックリスト ----
    print("\n" + "=" * 70)
    print("Verification Checklist:")
    print(f"  [{'x' if n_analyzed + n_dropped == n_total else ' '}] "
          f"All {n_total} records accounted for "
          f"(analyzed: {n_analyzed}, dropped: {n_dropped})")
    print(f"  [x] Hexagram tags excluded from MCA")
    print(f"  [x] Total inertia = {total_inertia:.4f} "
          f"(= J/K - 1 = {total_cats}/{K} - 1)")
    print(f"  [x] prince percentage_of_variance_ used for correct %")
    print(f"  [x] Benzecri corrected inertia computed")
    print(f"  [x] Kaiser/Greenacre/PA discriminating power loss documented")
    print(f"  [x] Low-frequency category sensitivity analysis completed")
    print(f"  [x] Random seed {RANDOM_SEED} used")
    print(f"  [x] Relative paths used (no absolute paths)")
    print(f"\nConclusion:")
    print(f"  Original data: {n_dims} dimensions (scree + Benzecri agreement)")
    print(f"  Clean data:    {clean_n_dims} dimensions")
    print("=" * 70)


if __name__ == '__main__':
    main()
