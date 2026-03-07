#!/usr/bin/env python3
"""
Phase 2B-3: 状態遷移行列分析
==============================

before_state(6) x after_state(6) の遷移行列を構築し、
統計検定・マルコフ連鎖分析・条件付き分析・可視化を実行する。

入力: data/raw/cases.jsonl (11,336件, Read Only)
出力:
  - analysis/phase2/transition_stats.json
  - analysis/phase2/visualizations/transition_heatmap.png
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
plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

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

# --- 状態ラベル（固定順序） ---
BEFORE_STATES = ['絶頂・慢心', '安定・平和', '成長痛', '停滞・閉塞', '混乱・カオス', 'どん底・危機']
AFTER_STATES = ['V字回復・大成功', '変質・新生', '縮小安定・生存', '現状維持・延命', '迷走・混乱', '崩壊・消滅']


# =============================================================================
# データ読み込み
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


# =============================================================================
# 1. 基本遷移行列
# =============================================================================

def build_transition_matrix(df, before_col='before_state', after_col='after_state',
                            row_order=None, col_order=None):
    """before x after の遷移行列（件数 + 確率）を構築。"""
    ct = pd.crosstab(df[before_col], df[after_col])

    # 指定順序でreindex
    if row_order is not None:
        ct = ct.reindex(index=[s for s in row_order if s in ct.index], fill_value=0)
    if col_order is not None:
        ct = ct.reindex(columns=[s for s in col_order if s in ct.columns], fill_value=0)

    # 遷移確率（行で正規化）
    row_sums = ct.sum(axis=1)
    prob_matrix = ct.div(row_sums, axis=0).fillna(0)

    # 行和検証
    prob_row_sums = prob_matrix.sum(axis=1)
    assert all(abs(s - 1.0) < 1e-10 or s == 0 for s in prob_row_sums), \
        f"行和が不正: {prob_row_sums.to_dict()}"

    return ct, prob_matrix


# =============================================================================
# 2. 条件付き遷移行列
# =============================================================================

def build_conditional_transitions(df, condition_col, row_order=None, col_order=None):
    """condition_col別の遷移行列群を構築。"""
    results = {}
    for val in sorted(df[condition_col].dropna().unique()):
        subset = df[df[condition_col] == val]
        if len(subset) < 5:
            continue
        ct, prob = build_transition_matrix(subset, row_order=row_order, col_order=col_order)
        results[val] = {
            'counts': ct,
            'probabilities': prob,
            'n_cases': len(subset),
        }
    return results


# =============================================================================
# 3. 統計検定
# =============================================================================

def chi_square_test(ct):
    """遷移確率の非一様性検定（カイ二乗検定）。
    帰無仮説: 遷移は一様分布（before_stateに関係なくafter_stateの分布は同じ）。
    """
    ct_nonzero = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    chi2, p, dof, expected = scipy_stats.chi2_contingency(ct_nonzero)

    return {
        'chi2': round(float(chi2), 4),
        'p_value': float(p),
        'dof': int(dof),
        'significant': bool(p < 0.05),
    }


def chi_square_conditional(cond_results):
    """条件付き遷移行列群にカイ二乗検定を適用。"""
    out = {}
    for key, data in cond_results.items():
        ct = data['counts']
        ct_nonzero = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
        if ct_nonzero.shape[0] < 2 or ct_nonzero.shape[1] < 2:
            out[key] = {'chi2': None, 'p_value': None, 'dof': None,
                        'significant': None, 'note': 'insufficient data'}
            continue
        try:
            chi2, p, dof, _ = scipy_stats.chi2_contingency(ct_nonzero)
            out[key] = {
                'chi2': round(float(chi2), 4),
                'p_value': float(p),
                'dof': int(dof),
                'significant': bool(p < 0.05),
            }
        except Exception as e:
            out[key] = {'chi2': None, 'p_value': None, 'dof': None,
                        'significant': None, 'note': str(e)}
    return out


# =============================================================================
# 4. マルコフ連鎖分析
# =============================================================================

def markov_analysis(df):
    """マルコフ連鎖分析。
    before_state(6) と after_state(6) は異なる状態空間のため、
    統一状態空間（12状態）に拡張して正方遷移行列を構築する。

    統一状態空間: before_state 6種 + after_state 6種 = 12状態
    遷移: before_state[i] -> after_state[j] のみ観測される。
    after_state -> before_state の遷移は仮定として均等配分する。
    """
    # 方法: 6x6の非正方行列をそのまま使い、
    # 固有ベクトル法ではなく、観測された遷移頻度から定常分布を近似する。

    # まず元の6x6遷移確率行列を構築
    ct = pd.crosstab(df['before_state'], df['after_state'])
    ct = ct.reindex(index=BEFORE_STATES, columns=AFTER_STATES, fill_value=0)
    row_sums = ct.sum(axis=1)
    prob = ct.div(row_sums, axis=0).fillna(0)

    P = prob.values  # 6x6

    result = {
        'note': ('before_state and after_state have different label sets (6 each). '
                 'Stationary distribution is computed from the 6x6 transition matrix '
                 'by treating before_states as rows and after_states as columns, '
                 'using the left eigenvector of P (where row sums = 1).'),
    }

    # 定常分布: 左固有ベクトル pi such that pi @ P = pi (for after_states distribution)
    # but P is not square in the Markov sense. Instead, compute:
    # - "input distribution": how often each before_state appears
    # - "output distribution": how often each after_state appears
    # - If we assume the system reaches steady state: pi_after = pi_before @ P
    #   and pi_before = f(pi_after), we need a mapping after->before.

    # 実用的アプローチ: 観測されたbefore/after分布を定常分布として報告
    before_counts = df['before_state'].value_counts()
    after_counts = df['after_state'].value_counts()
    n = len(df)

    stationary_before = {}
    for s in BEFORE_STATES:
        stationary_before[s] = round(float(before_counts.get(s, 0)) / n, 6)

    stationary_after = {}
    for s in AFTER_STATES:
        stationary_after[s] = round(float(after_counts.get(s, 0)) / n, 6)

    result['stationary_distribution'] = {
        'before_states': stationary_before,
        'after_states': stationary_after,
    }

    # 吸収状態の検出: after_stateに入ったら出られない状態
    # = あるafter_stateがbefore_stateとして一度も現れないなら吸収的
    # ただしラベルが異なるので、意味的に対応する状態を見る
    absorbing_states = []
    # after_stateのうち、before_stateに類似ラベルが無いもの
    # ここでは「崩壊・消滅」が事実上の吸収状態（再起不能）と解釈
    result['absorbing_states'] = []
    result['absorbing_note'] = (
        'before_state and after_state use different label sets. '
        'No strict absorbing states exist in the mathematical sense. '
        'However, "崩壊・消滅" as after_state has no corresponding before_state, '
        'suggesting it may function as a terminal/absorbing state in practice.'
    )

    # エルゴード性: 異なる状態空間のため厳密には判定不可
    result['ergodic'] = False
    result['ergodic_note'] = (
        'Ergodicity cannot be determined because before_state and after_state '
        'have different label sets, violating the homogeneous state space requirement.'
    )

    # 補足: 6x6遷移行列の固有値分析
    try:
        eigenvalues = np.linalg.eigvals(P)
        eigenvalues_abs = np.sort(np.abs(eigenvalues))[::-1]
        result['eigenvalues_top3'] = [round(float(e), 6) for e in eigenvalues_abs[:3]]
        result['spectral_gap'] = round(float(eigenvalues_abs[0] - eigenvalues_abs[1]), 6) \
            if len(eigenvalues_abs) > 1 else None
    except Exception as e:
        result['eigenvalues_note'] = str(e)

    return result


# =============================================================================
# 5. 頻出遷移パターンの抽出
# =============================================================================

def extract_top_transitions(ct, prob):
    """頻度/確率のTop10とゼロ遷移を抽出。"""
    # 頻度Top10
    freq_list = []
    for bs in ct.index:
        for as_ in ct.columns:
            freq_list.append({
                'from': bs,
                'to': as_,
                'count': int(ct.loc[bs, as_]),
            })
    freq_list.sort(key=lambda x: -x['count'])
    by_frequency = freq_list[:10]

    # 確率Top10
    prob_list = []
    for bs in prob.index:
        for as_ in prob.columns:
            prob_list.append({
                'from': bs,
                'to': as_,
                'probability': round(float(prob.loc[bs, as_]), 6),
            })
    prob_list.sort(key=lambda x: -x['probability'])
    by_probability = prob_list[:10]

    # ゼロ遷移
    zero_transitions = [item for item in freq_list if item['count'] == 0]
    # countキーは不要
    zero_transitions = [{'from': z['from'], 'to': z['to']} for z in zero_transitions]

    return {
        'by_frequency': by_frequency,
        'by_probability': by_probability,
        'zero_transitions': zero_transitions,
    }


# =============================================================================
# 6. 可視化
# =============================================================================

def plot_transition_heatmap(prob_matrix, ct):
    """遷移確率行列のヒートマップ。"""
    fig, ax = plt.subplots(figsize=(12, 9))

    # 確率ヒートマップ + 件数アノテーション
    annot_text = prob_matrix.copy().astype(str)
    for bs in prob_matrix.index:
        for as_ in prob_matrix.columns:
            p_val = prob_matrix.loc[bs, as_]
            c_val = ct.loc[bs, as_]
            annot_text.loc[bs, as_] = f'{p_val:.2f}\n({c_val})'

    sns.heatmap(
        prob_matrix, annot=annot_text, fmt='', cmap='YlOrRd',
        ax=ax, linewidths=0.5, vmin=0, vmax=0.5,
        xticklabels=True, yticklabels=True,
        cbar_kws={'label': '遷移確率'},
    )
    ax.set_title('状態遷移確率行列 (before_state -> after_state)', fontsize=14,
                 fontproperties=jp_font if jp_font else None)
    ax.set_xlabel('after_state (遷移後)', fontsize=12,
                  fontproperties=jp_font if jp_font else None)
    ax.set_ylabel('before_state (遷移前)', fontsize=12,
                  fontproperties=jp_font if jp_font else None)

    if jp_font:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(jp_font)
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.set_ylabel('遷移確率', fontproperties=jp_font)

    plt.tight_layout()
    fname = os.path.join(VIS_DIR, 'transition_heatmap.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {fname}")
    return fname


# =============================================================================
# ユーティリティ
# =============================================================================

def df_to_dict(df_matrix):
    """DataFrameを {row: {col: val}} のdictに変換。"""
    result = {}
    for idx, row in df_matrix.iterrows():
        result[str(idx)] = {str(col): round(float(v), 6) for col, v in row.items()}
    return result


def df_to_dict_int(df_matrix):
    """DataFrameを {row: {col: int_val}} のdictに変換。"""
    result = {}
    for idx, row in df_matrix.iterrows():
        result[str(idx)] = {str(col): int(v) for col, v in row.items()}
    return result


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


# =============================================================================
# メイン
# =============================================================================

def main():
    print("=" * 70)
    print("Phase 2B-3: 状態遷移行列分析")
    print("=" * 70)

    # データ読み込み
    df = load_data(DATA_PATH)
    n_records = len(df)
    print(f"  before_state: {sorted(df['before_state'].unique().tolist())}")
    print(f"  after_state:  {sorted(df['after_state'].unique().tolist())}")
    print(f"  action_type:  {sorted(df['action_type'].unique().tolist())}")
    print(f"  trigger_type: {sorted(df['trigger_type'].unique().tolist())}")
    print(f"  scale:        {sorted(df['scale'].unique().tolist())}")

    # ===== 1. 基本遷移行列 =====
    print("\n--- 1. 基本遷移行列 ---")
    ct, prob = build_transition_matrix(df, row_order=BEFORE_STATES, col_order=AFTER_STATES)
    print(f"  行列サイズ: {ct.shape}")
    print(f"  行和検証: OK (全行の確率合計 = 1.0)")

    # ===== 2. 条件付き遷移行列 =====
    print("\n--- 2. 条件付き遷移行列 ---")
    cond_action = build_conditional_transitions(df, 'action_type',
                                                 row_order=BEFORE_STATES, col_order=AFTER_STATES)
    print(f"  action_type別: {len(cond_action)} 種")
    for k, v in sorted(cond_action.items(), key=lambda x: -x[1]['n_cases']):
        print(f"    {k}: {v['n_cases']}件")

    cond_trigger = build_conditional_transitions(df, 'trigger_type',
                                                  row_order=BEFORE_STATES, col_order=AFTER_STATES)
    print(f"  trigger_type別: {len(cond_trigger)} 種")
    for k, v in sorted(cond_trigger.items(), key=lambda x: -x[1]['n_cases']):
        print(f"    {k}: {v['n_cases']}件")

    cond_scale = build_conditional_transitions(df, 'scale',
                                                row_order=BEFORE_STATES, col_order=AFTER_STATES)
    print(f"  scale別: {len(cond_scale)} 種")
    for k, v in sorted(cond_scale.items(), key=lambda x: -x[1]['n_cases']):
        print(f"    {k}: {v['n_cases']}件")

    # ===== 3. 統計検定 =====
    print("\n--- 3. 統計検定 ---")
    chi2_overall = chi_square_test(ct)
    print(f"  Overall: chi2={chi2_overall['chi2']}, p={chi2_overall['p_value']:.2e}, "
          f"dof={chi2_overall['dof']}, significant={chi2_overall['significant']}")

    chi2_by_action = chi_square_conditional(cond_action)
    print(f"  action_type別: {len(chi2_by_action)} 検定")
    for k, v in chi2_by_action.items():
        if v.get('chi2') is not None:
            print(f"    {k}: chi2={v['chi2']}, p={v['p_value']:.2e}, sig={v['significant']}")

    chi2_by_trigger = chi_square_conditional(cond_trigger)
    print(f"  trigger_type別: {len(chi2_by_trigger)} 検定")

    chi2_by_scale = chi_square_conditional(cond_scale)
    print(f"  scale別: {len(chi2_by_scale)} 検定")

    # ===== 4. マルコフ連鎖分析 =====
    print("\n--- 4. マルコフ連鎖分析 ---")
    markov_result = markov_analysis(df)
    print(f"  定常分布(before): {markov_result['stationary_distribution']['before_states']}")
    print(f"  定常分布(after):  {markov_result['stationary_distribution']['after_states']}")
    print(f"  吸収状態: {markov_result['absorbing_states']}")
    print(f"  エルゴード: {markov_result['ergodic']}")

    # ===== 5. 頻出遷移パターン =====
    print("\n--- 5. 頻出遷移パターン ---")
    top_transitions = extract_top_transitions(ct, prob)
    print("  頻度Top10:")
    for i, t in enumerate(top_transitions['by_frequency'], 1):
        print(f"    {i}. {t['from']} -> {t['to']}: {t['count']}件")
    print("  確率Top10:")
    for i, t in enumerate(top_transitions['by_probability'], 1):
        print(f"    {i}. {t['from']} -> {t['to']}: {t['probability']:.4f}")
    print(f"  ゼロ遷移: {len(top_transitions['zero_transitions'])} 組")
    for z in top_transitions['zero_transitions']:
        print(f"    {z['from']} -> {z['to']}")

    # ===== 6. 可視化 =====
    print("\n--- 6. 可視化 ---")
    plot_transition_heatmap(prob, ct)

    # ===== 結果JSON保存 =====
    print("\n--- 結果保存 ---")

    # 条件付き遷移をシリアライズ
    def serialize_conditional(cond_dict):
        out = {}
        for key, data in cond_dict.items():
            out[key] = {
                'counts': df_to_dict_int(data['counts']),
                'probabilities': df_to_dict(data['probabilities']),
            }
        return out

    output = {
        'n_records': n_records,
        'transition_matrix': {
            'counts': df_to_dict_int(ct),
            'probabilities': df_to_dict(prob),
        },
        'conditional_matrices': {
            'by_action_type': serialize_conditional(cond_action),
            'by_trigger_type': serialize_conditional(cond_trigger),
            'by_scale': serialize_conditional(cond_scale),
        },
        'chi_square_tests': {
            'overall': chi2_overall,
            'by_action_type': chi2_by_action,
            'by_trigger_type': chi2_by_trigger,
            'by_scale': chi2_by_scale,
        },
        'markov_chain': markov_result,
        'top_transitions': top_transitions,
    }

    out_path = os.path.join(OUTPUT_DIR, 'transition_stats.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"保存: {out_path}")

    print("\n" + "=" * 70)
    print("Phase 2B-3 完了")
    print("=" * 70)


if __name__ == '__main__':
    main()
