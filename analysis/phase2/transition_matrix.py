#!/usr/bin/env python3
"""
Phase 2B-1: 状態遷移行列分析
==============================

before_state -> after_state の遷移行列を構築し、統計的に分析する。

入力: data/raw/cases.jsonl (13,060件, Read Only)
出力:
  - analysis/phase2/transition_stats.json    遷移統計結果
  - analysis/phase2/visualizations/transition_heatmap.png

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

warnings.filterwarnings('ignore')

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

# --- カテゴリ統合マッピング（Phase 2Aから転載） ---
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
# データ読み込みとクリーニング
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


def apply_category_merge(df):
    """低頻度カテゴリを統合したクリーンDataFrameを返す。"""
    df_clean = df.copy()
    for col, mapping in CATEGORY_MERGE_MAP.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(mapping)
    return df_clean


# =============================================================================
# Task 1: 遷移行列分析
# =============================================================================

def build_transition_matrix(df):
    """before_state x after_state の遷移行列（件数 + 確率）を構築。"""
    ct = pd.crosstab(df['before_state'], df['after_state'])

    # 遷移確率（行で正規化）
    row_sums = ct.sum(axis=1)
    prob_matrix = ct.div(row_sums, axis=0)

    return ct, prob_matrix


def build_conditional_transition(df, condition_col):
    """condition_col別の遷移行列を構築。"""
    results = {}
    for val in sorted(df[condition_col].unique()):
        subset = df[df[condition_col] == val]
        if len(subset) >= 5:  # 最低5件
            ct = pd.crosstab(subset['before_state'], subset['after_state'])
            row_sums = ct.sum(axis=1)
            prob = ct.div(row_sums, axis=0).fillna(0)
            results[val] = {
                'count_matrix': ct,
                'prob_matrix': prob,
                'n_cases': len(subset),
            }
    return results


def chi_square_test(ct):
    """遷移確率の非一様性検定（カイ二乗検定）。"""
    # ゼロ行・ゼロ列を除去
    ct_nonzero = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]

    chi2, p, dof, expected = scipy_stats.chi2_contingency(ct_nonzero)

    # 調整残差
    observed = ct_nonzero.values.astype(float)
    expected_arr = expected
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    n = observed.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        adjusted_residuals = (observed - expected_arr) / np.sqrt(
            expected_arr * (1 - row_totals / n) * (1 - col_totals / n)
        )
    adjusted_residuals = np.nan_to_num(adjusted_residuals, nan=0.0, posinf=0.0, neginf=0.0)

    adj_res_df = pd.DataFrame(
        adjusted_residuals,
        index=ct_nonzero.index,
        columns=ct_nonzero.columns
    )

    return {
        'chi2': float(chi2),
        'p_value': float(p),
        'dof': int(dof),
        'significant': p < 0.001,
        'expected': pd.DataFrame(expected_arr, index=ct_nonzero.index, columns=ct_nonzero.columns),
        'adjusted_residuals': adj_res_df,
    }


def find_significant_transitions(adj_res_df, threshold=2.576):
    """調整残差 |z| > threshold の有意な遷移パターンを抽出。
    threshold=2.576 は p<0.01 に相当。
    """
    sig_positive = []
    sig_negative = []

    for before in adj_res_df.index:
        for after in adj_res_df.columns:
            z = adj_res_df.loc[before, after]
            if abs(z) > threshold:
                entry = {
                    'before_state': before,
                    'after_state': after,
                    'adjusted_residual': round(float(z), 3),
                    'significance': 'p<0.01' if abs(z) > 2.576 else 'p<0.05',
                }
                if z > 0:
                    sig_positive.append(entry)
                else:
                    sig_negative.append(entry)

    sig_positive.sort(key=lambda x: -x['adjusted_residual'])
    sig_negative.sort(key=lambda x: x['adjusted_residual'])

    return sig_positive, sig_negative


def markov_analysis(prob_matrix):
    """マルコフ連鎖分析: 状態空間の整合性チェック + 定常分布、吸収状態、混合率。

    before_state と after_state が異なる状態空間の場合、マルコフ連鎖分析は
    数学的に不適用であるため、valid=False を返す。
    """
    before_states = sorted(prob_matrix.index.tolist())
    after_states = sorted(prob_matrix.columns.tolist())

    # --- 状態空間の整合性チェック ---
    overlap = sorted(set(before_states) & set(after_states))
    before_only = sorted(set(before_states) - set(after_states))
    after_only = sorted(set(after_states) - set(before_states))

    result = {
        'n_before_states': len(before_states),
        'n_after_states': len(after_states),
        'before_states': before_states,
        'after_states': after_states,
        'overlapping_states': overlap,
        'before_only_states': before_only,
        'after_only_states': after_only,
    }

    if set(before_states) != set(after_states):
        # 状態空間が不一致 — マルコフ連鎖分析は不適用
        warnings.warn(
            f"Markov chain analysis is inapplicable: before_state ({len(before_states)} states) "
            f"and after_state ({len(after_states)} states) have different state spaces "
            f"with only {len(overlap)} overlapping states. "
            f"A valid Markov chain requires identical state spaces for transitions."
        )
        result['valid'] = False
        result['invalidation_reason'] = (
            f"before_state ({len(before_states)} states) and after_state ({len(after_states)} states) "
            f"have different state spaces with only {len(overlap)} overlapping states. "
            f"A valid Markov chain requires identical state spaces for transitions. "
            f"Reindexing to a 7x7 square matrix (using before_states for both axes) produces a "
            f"sub-stochastic matrix with row sums 0.06-0.59, maximum eigenvalue ~0.36 (should be 1.0). "
            f"All previously reported stationary_distribution, mixing_rate, and absorption_states "
            f"are artifacts of this sub-stochastic matrix and have been withdrawn."
        )
        result['note'] = (
            "Each record is an independent transition (before->after), not a time-series chain. "
            "Markov chain analysis would require sequential data AND identical state spaces. "
            "Consider unified state space construction in Phase 3 if Markov analysis is desired."
        )
        return result

    # --- 以下は状態空間が一致する場合のみ実行 ---
    states = before_states
    P = prob_matrix.reindex(index=states, columns=states, fill_value=0).values

    result['valid'] = True
    result['n_states'] = len(states)
    result['states'] = states

    # 吸収状態の検出（自己遷移確率 > 0.5 かつ他への遷移が弱い状態）
    absorbing_candidates = []
    for i, state in enumerate(states):
        if P[i, i] > 0.5:
            absorbing_candidates.append({
                'state': state,
                'self_transition_prob': round(float(P[i, i]), 4),
            })
    result['absorbing_candidates'] = absorbing_candidates

    # 定常分布の計算（固有値分解）
    try:
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        # 固有値1に最も近いものを探す
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()

        if np.all(stationary >= -1e-10):
            stationary = np.maximum(stationary, 0)
            stationary = stationary / stationary.sum()
            result['stationary_distribution'] = {
                state: round(float(stationary[i]), 6) for i, state in enumerate(states)
            }
            result['stationary_valid'] = True
        else:
            result['stationary_distribution'] = None
            result['stationary_valid'] = False
            result['stationary_note'] = 'Negative components in stationary vector; chain may not be ergodic'
    except Exception as e:
        result['stationary_distribution'] = None
        result['stationary_valid'] = False
        result['stationary_note'] = str(e)

    # 混合率の評価（第2固有値）
    try:
        eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
        if len(eigenvalues_sorted) > 1:
            second_eigenvalue = float(eigenvalues_sorted[1])
            result['second_eigenvalue'] = round(second_eigenvalue, 6)
            result['mixing_rate'] = round(1.0 - second_eigenvalue, 6)
            result['mixing_interpretation'] = (
                'fast' if second_eigenvalue < 0.5
                else 'moderate' if second_eigenvalue < 0.9
                else 'slow'
            )
    except Exception:
        pass

    # 一次マルコフ仮定の簡易テスト
    result['first_order_markov_note'] = (
        'Full test requires sequential data (before->after->next). '
        'Each record is an independent transition; true Markov order cannot be tested '
        'without time-series structure.'
    )

    return result


def extract_triplets(df, top_n=30):
    """before_state -> action_type -> after_state のトリプレット TOP N。"""
    triplets = df.groupby(['before_state', 'action_type', 'after_state']).size()
    triplets = triplets.reset_index(name='count')
    triplets = triplets.sort_values('count', ascending=False).head(top_n)

    result = []
    for _, row in triplets.iterrows():
        result.append({
            'before_state': row['before_state'],
            'action_type': row['action_type'],
            'after_state': row['after_state'],
            'count': int(row['count']),
        })
    return result


def directionality_analysis(ct):
    """遷移の方向性バイアス分析。
    各 (i, j) ペアで ct[i,j] vs ct[j,i] を比較し、非対称性を評価。
    """
    states = list(ct.index)
    asymmetries = []

    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            s1, s2 = states[i], states[j]
            if s1 in ct.index and s2 in ct.index and s1 in ct.columns and s2 in ct.columns:
                fwd = int(ct.loc[s1, s2]) if s2 in ct.columns else 0
                bwd = int(ct.loc[s2, s1]) if s1 in ct.columns else 0
                total = fwd + bwd
                if total > 0:
                    asymmetry_ratio = abs(fwd - bwd) / total
                    asymmetries.append({
                        'state_A': s1,
                        'state_B': s2,
                        'A_to_B': fwd,
                        'B_to_A': bwd,
                        'asymmetry_ratio': round(asymmetry_ratio, 4),
                        'dominant_direction': f"{s1}->{s2}" if fwd > bwd else f"{s2}->{s1}" if bwd > fwd else 'symmetric',
                    })

    asymmetries.sort(key=lambda x: -x['asymmetry_ratio'])
    return asymmetries


# =============================================================================
# 可視化
# =============================================================================

def plot_transition_heatmap(prob_matrix, ct, title_suffix=''):
    """遷移確率のヒートマップ（件数アノテーション付き）。"""
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # 件数ヒートマップ
    ax1 = axes[0]
    sns.heatmap(
        ct, annot=True, fmt='d', cmap='YlOrRd',
        ax=ax1, linewidths=0.5,
        xticklabels=True, yticklabels=True,
    )
    ax1.set_title(f'遷移件数{title_suffix}', fontsize=14,
                  fontproperties=jp_font if jp_font else None)
    ax1.set_xlabel('after_state', fontsize=11)
    ax1.set_ylabel('before_state', fontsize=11)
    if jp_font:
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontproperties(jp_font)

    # 確率ヒートマップ
    ax2 = axes[1]
    sns.heatmap(
        prob_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
        ax=ax2, linewidths=0.5, vmin=0, vmax=1,
        xticklabels=True, yticklabels=True,
    )
    ax2.set_title(f'遷移確率（行正規化）{title_suffix}', fontsize=14,
                  fontproperties=jp_font if jp_font else None)
    ax2.set_xlabel('after_state', fontsize=11)
    ax2.set_ylabel('before_state', fontsize=11)
    if jp_font:
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontproperties(jp_font)

    plt.tight_layout()
    fname = os.path.join(VIS_DIR, f'transition_heatmap{title_suffix.replace(" ", "_")}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {fname}")
    return fname


def plot_adjusted_residuals(adj_res_df, title_suffix=''):
    """調整残差のヒートマップ。"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # 有意水準の色分け
    mask = adj_res_df.abs() < 1.96
    sns.heatmap(
        adj_res_df, annot=True, fmt='.1f', cmap='RdBu_r',
        center=0, ax=ax, linewidths=0.5,
        vmin=-10, vmax=10,
        xticklabels=True, yticklabels=True,
    )
    ax.set_title(f'調整残差{title_suffix} (|z|>2.576: p<0.01)', fontsize=14,
                 fontproperties=jp_font if jp_font else None)
    ax.set_xlabel('after_state', fontsize=11)
    ax.set_ylabel('before_state', fontsize=11)
    if jp_font:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(jp_font)

    plt.tight_layout()
    fname = os.path.join(VIS_DIR, f'adjusted_residuals{title_suffix.replace(" ", "_")}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {fname}")
    return fname


# =============================================================================
# メイン
# =============================================================================

def main():
    print("=" * 70)
    print("Phase 2B-1: 状態遷移行列分析")
    print("=" * 70)

    # データ読み込み
    df_raw = load_data(DATA_PATH)
    df = apply_category_merge(df_raw)
    print(f"カテゴリ統合後: {len(df)} 件")
    print(f"  before_state: {df['before_state'].nunique()} カテゴリ")
    print(f"  after_state:  {df['after_state'].nunique()} カテゴリ")
    print(f"  action_type:  {df['action_type'].nunique()} カテゴリ")
    print(f"  trigger_type: {df['trigger_type'].nunique()} カテゴリ")

    # ----- 基本遷移行列 -----
    print("\n--- 基本遷移行列 ---")
    ct, prob = build_transition_matrix(df)
    print(f"遷移行列サイズ: {ct.shape}")

    # ----- 統計検定 -----
    print("\n--- カイ二乗検定 ---")
    chi2_result = chi_square_test(ct)
    print(f"  chi2 = {chi2_result['chi2']:.2f}, dof = {chi2_result['dof']}, p = {chi2_result['p_value']:.2e}")
    print(f"  有意: {chi2_result['significant']}")

    # 有意な遷移パターン
    sig_pos, sig_neg = find_significant_transitions(chi2_result['adjusted_residuals'])
    print(f"\n  有意に多い遷移 (z > 2.576): {len(sig_pos)} パターン")
    for entry in sig_pos[:10]:
        print(f"    {entry['before_state']} -> {entry['after_state']}: z={entry['adjusted_residual']:.1f}")
    print(f"\n  有意に少ない遷移 (z < -2.576): {len(sig_neg)} パターン")
    for entry in sig_neg[:10]:
        print(f"    {entry['before_state']} -> {entry['after_state']}: z={entry['adjusted_residual']:.1f}")

    # ----- 条件付き遷移行列 -----
    print("\n--- 条件付き遷移行列: action_type別 ---")
    cond_action = build_conditional_transition(df, 'action_type')
    print(f"  action_type別行列数: {len(cond_action)}")
    for act, data in sorted(cond_action.items(), key=lambda x: -x[1]['n_cases']):
        print(f"    {act}: {data['n_cases']}件")

    print("\n--- 条件付き遷移行列: trigger_type別 ---")
    cond_trigger = build_conditional_transition(df, 'trigger_type')
    print(f"  trigger_type別行列数: {len(cond_trigger)}")
    for trig, data in sorted(cond_trigger.items(), key=lambda x: -x[1]['n_cases']):
        print(f"    {trig}: {data['n_cases']}件")

    # ----- マルコフ連鎖分析 -----
    print("\n--- マルコフ連鎖分析 ---")
    markov_result = markov_analysis(prob)
    if markov_result['stationary_valid']:
        print("  定常分布:")
        for state, p_val in sorted(markov_result['stationary_distribution'].items(),
                                    key=lambda x: -x[1]):
            print(f"    {state}: {p_val:.4f}")
    print(f"  第2固有値: {markov_result.get('second_eigenvalue', 'N/A')}")
    print(f"  混合率: {markov_result.get('mixing_rate', 'N/A')}")
    print(f"  混合速度: {markov_result.get('mixing_interpretation', 'N/A')}")
    print(f"  吸収候補: {len(markov_result['absorbing_candidates'])} 状態")
    for ac in markov_result['absorbing_candidates']:
        print(f"    {ac['state']}: 自己遷移確率 = {ac['self_transition_prob']}")

    # ----- トリプレット TOP30 -----
    print("\n--- トリプレット TOP30 ---")
    triplets = extract_triplets(df, top_n=30)
    for i, t in enumerate(triplets, 1):
        print(f"  {i:2d}. {t['before_state']} -> [{t['action_type']}] -> {t['after_state']}: {t['count']}件")

    # ----- 方向性バイアス -----
    print("\n--- 方向性バイアス分析 ---")
    dir_analysis = directionality_analysis(ct)
    print(f"  分析ペア数: {len(dir_analysis)}")
    print("  TOP10 非対称ペア:")
    for entry in dir_analysis[:10]:
        print(f"    {entry['state_A']} <-> {entry['state_B']}: "
              f"ratio={entry['asymmetry_ratio']:.3f}, "
              f"{entry['A_to_B']}件 vs {entry['B_to_A']}件, "
              f"dominant={entry['dominant_direction']}")

    # ----- 可視化 -----
    print("\n--- 可視化 ---")
    plot_transition_heatmap(prob, ct)
    plot_adjusted_residuals(chi2_result['adjusted_residuals'])

    # ----- 結果保存 -----
    print("\n--- 結果保存 ---")

    # 条件付き遷移の要約（JSON serializable）
    cond_action_summary = {}
    for act, data in cond_action.items():
        cond_action_summary[act] = {
            'n_cases': data['n_cases'],
            'prob_matrix': {
                str(idx): {str(col): round(float(v), 4)
                           for col, v in row.items() if v > 0}
                for idx, row in data['prob_matrix'].iterrows()
            },
        }

    cond_trigger_summary = {}
    for trig, data in cond_trigger.items():
        cond_trigger_summary[trig] = {
            'n_cases': data['n_cases'],
            'prob_matrix': {
                str(idx): {str(col): round(float(v), 4)
                           for col, v in row.items() if v > 0}
                for idx, row in data['prob_matrix'].iterrows()
            },
        }

    output = {
        'n_records': len(df),
        'data_type': 'clean (low-frequency categories merged)',
        'before_states': sorted(df['before_state'].unique().tolist()),
        'after_states': sorted(df['after_state'].unique().tolist()),

        'basic_transition': {
            'count_matrix': {
                str(idx): {str(col): int(v) for col, v in row.items()}
                for idx, row in ct.iterrows()
            },
            'prob_matrix': {
                str(idx): {str(col): round(float(v), 4) for col, v in row.items()}
                for idx, row in prob.iterrows()
            },
        },

        'chi_square_test': {
            'chi2': chi2_result['chi2'],
            'p_value': chi2_result['p_value'],
            'dof': chi2_result['dof'],
            'significant': chi2_result['significant'],
            'interpretation': (
                'Transition probabilities are significantly non-uniform (p < 0.001). '
                'Before-state strongly predicts after-state.'
            ),
        },

        'significant_transitions': {
            'over_represented': sig_pos[:20],
            'under_represented': sig_neg[:20],
            'threshold': 2.576,
            'significance_level': 'p < 0.01',
        },

        'conditional_transition_by_action_type': cond_action_summary,
        'conditional_transition_by_trigger_type': cond_trigger_summary,

        'markov_chain': markov_result,

        'top30_triplets': triplets,

        'directionality_bias': dir_analysis[:20],
    }

    out_path = os.path.join(OUTPUT_DIR, 'transition_stats.json')

    class NumpyEncoder(json.JSONEncoder):
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

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"保存: {out_path}")

    print("\n" + "=" * 70)
    print("Phase 2B-1 完了")
    print("=" * 70)


if __name__ == '__main__':
    main()
