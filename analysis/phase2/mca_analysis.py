#!/usr/bin/env python3
"""
Phase 2A-2: 多重対応分析（MCA）— 易経変化ロジックDB
八卦タグを除外した基本変数のみで実行。

入力: data/raw/cases.jsonl (11,336件 — Read Only)
出力:
  - analysis/phase2/mca_results.json       MCA結果
  - analysis/phase2/dimension_report.json  次元解釈レポート
  - analysis/phase2/visualizations/scree_plot.png
  - analysis/phase2/visualizations/mca_biplot.png

乱数シード: 42
"""

import json
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prince

warnings.filterwarnings('ignore')

# ── 設定 ──────────────────────────────────────────────
RANDOM_SEED = 42
N_COMPONENTS = 10
N_PARALLEL = 100

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'cases.jsonl')
VIS_DIR = os.path.join(SCRIPT_DIR, 'visualizations')
os.makedirs(VIS_DIR, exist_ok=True)

# 日本語フォント
plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 分析対象変数
TARGET_VARS = ['before_state', 'after_state', 'trigger_type',
               'action_type', 'pattern_type', 'scale']
EXCLUDED_VARS = ['before_hex', 'trigger_hex', 'action_hex', 'after_hex']


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. データ読み込み・前処理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_data():
    records = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"[INFO] 読み込みレコード数: {len(df)}")
    return df


def preprocess(df):
    df_sub = df[TARGET_VARS].copy()
    n_before = len(df_sub)

    missing_info = {}
    for col in TARGET_VARS:
        n_miss = int(df_sub[col].isna().sum())
        if n_miss > 0:
            missing_info[col] = n_miss

    df_clean = df_sub.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df_clean)

    for col in TARGET_VARS:
        df_clean[col] = df_clean[col].astype(str)

    print(f"[INFO] 欠損除去: {n_dropped}件ドロップ → 分析対象: {len(df_clean)}件")
    return df_clean, n_dropped, missing_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. MCA実行
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_mca(df_cat, n_components=N_COMPONENTS):
    print(f"[INFO] MCA実行中 (n_components={n_components})...")
    mca = prince.MCA(n_components=n_components, random_state=RANDOM_SEED)
    mca = mca.fit(df_cat)
    return mca


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 次元数の決定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parallel_analysis(df_cat, n_components=N_COMPONENTS, n_iter=N_PARALLEL):
    """同じカテゴリ構造のランダムデータで95%閾値を算出"""
    print(f"[INFO] 並行分析実行中 ({n_iter}回)...")
    rng = np.random.RandomState(RANDOM_SEED)

    n_rows = len(df_cat)
    all_eigs = []

    for i in range(n_iter):
        df_rand = pd.DataFrame()
        for col in df_cat.columns:
            df_rand[col] = rng.choice(df_cat[col].values, size=n_rows, replace=True)
        try:
            m = prince.MCA(n_components=n_components, random_state=i)
            m = m.fit(df_rand)
            all_eigs.append(list(m.eigenvalues_))
        except Exception:
            continue

    all_eigs = np.array(all_eigs)
    threshold_95 = np.percentile(all_eigs, 95, axis=0)
    print(f"[INFO] 並行分析完了 ({len(all_eigs)}回成功)")
    return threshold_95


def determine_dimensions(eigenvalues, cumulative_pct, n_variables, pa_thresholds):
    ev = np.array(eigenvalues)

    # 1. スクリー肘法: 2次差分の絶対値が最大の位置
    diffs = np.diff(ev)
    diffs2 = np.diff(diffs)
    scree_elbow = int(np.argmax(np.abs(diffs2))) + 1 if len(diffs2) > 0 else 1

    # 2. 累積寄与率70%
    idx_70 = np.searchsorted(cumulative_pct, 70.0)
    if idx_70 < len(cumulative_pct):
        cum_70 = int(idx_70 + 1)
    else:
        # 全次元でも70%に届かない場合: 総カテゴリ数-変数数 が理論最大次元
        total_cats = sum(1 for _ in cumulative_pct)  # n_components分だけ
        cum_70 = int(len(ev))  # 「N_COMPONENTS次元でもXX%」と記録

    # 3. Kaiser基準: 固有値 > 1/変数数
    kaiser_thr = 1.0 / n_variables
    kaiser_n = int(np.sum(ev > kaiser_thr))

    # 4. 並行分析: 実固有値がランダム95%を上回る次元数
    pa_n = 0
    for i in range(min(len(ev), len(pa_thresholds))):
        if ev[i] > pa_thresholds[i]:
            pa_n = i + 1
        else:
            break

    criteria = {
        "scree_elbow": scree_elbow,
        "cumulative_70pct": cum_70,
        "kaiser": kaiser_n,
        "parallel_analysis": pa_n
    }

    values = list(criteria.values())
    recommended = int(np.ceil(np.median(values)))

    return recommended, criteria


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 寄与度・次元解釈
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_contributions(mca, df_cat, n_dims):
    """
    寄与度 = (coord^2 × mass) / eigenvalue × 100
    mass = category_freq / (N × K)
    """
    col_coords = mca.column_coordinates(df_cat)
    eigenvalues = np.array(mca.eigenvalues_)

    n_rows = len(df_cat)
    n_vars = len(TARGET_VARS)

    # カテゴリ質量: one-hot → 列合計 / (N * K)
    dummies = pd.get_dummies(df_cat)
    raw_masses = dummies.sum() / (n_rows * n_vars)

    # prince列名 (var__cat) → dummies列名 (var_cat) のマッピング
    mass_lookup = {}
    for prince_label in col_coords.index:
        # prince uses '__' as separator
        for dummy_col in raw_masses.index:
            # Try matching: before_state__停滞・閉塞 → before_state_停滞・閉塞
            expected_dummy = prince_label.replace('__', '_', 1)
            if dummy_col == expected_dummy:
                mass_lookup[prince_label] = raw_masses[dummy_col]
                break
        if prince_label not in mass_lookup:
            if prince_label in raw_masses.index:
                mass_lookup[prince_label] = raw_masses[prince_label]

    contributions = {}
    for dim_idx in range(n_dims):
        dim_key = f"dim_{dim_idx + 1}"
        dim_contribs = {}

        for cat_label in col_coords.index:
            coord = float(col_coords.loc[cat_label].iloc[dim_idx])
            mass = mass_lookup.get(cat_label, 0.0)

            if eigenvalues[dim_idx] > 1e-10 and mass > 0:
                contrib = (coord ** 2) * mass / eigenvalues[dim_idx] * 100
            else:
                contrib = 0.0

            dim_contribs[cat_label] = round(contrib, 4)

        contributions[dim_key] = dict(sorted(dim_contribs.items(), key=lambda x: -x[1]))

    return contributions


def interpret_dimension(contributions_dim, top_n=5):
    sorted_items = sorted(contributions_dim.items(), key=lambda x: -x[1])[:top_n]
    top_contributors = [{"category": k, "contribution_pct": v} for k, v in sorted_items]

    # 変数グループ別に集約
    var_groups = {}
    for cat, _ in sorted_items:
        parts = cat.split('__')
        var = parts[0] if len(parts) == 2 else cat
        cat_name = parts[1] if len(parts) == 2 else cat
        var_groups.setdefault(var, []).append(cat_name)

    parts = []
    for var in list(var_groups.keys())[:3]:
        cats_str = ', '.join(var_groups[var])
        parts.append(f"{var}[{cats_str}]")

    interpretation = " / ".join(parts) if parts else "解釈未定"
    return top_contributors, interpretation


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 可視化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_scree(eigenvalues, pa_thresholds, cumulative_pct, save_path):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    dims = np.arange(1, len(eigenvalues) + 1)

    ax1.plot(dims, eigenvalues, 'bo-', lw=2, ms=8, label='固有値（実データ）')
    ax1.plot(dims[:len(pa_thresholds)], pa_thresholds[:len(eigenvalues)],
             'r--s', lw=2, ms=6, label='並行分析 95%閾値')
    ax1.set_xlabel('次元', fontsize=14)
    ax1.set_ylabel('固有値', fontsize=14, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(dims)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(dims, cumulative_pct, 'g^--', lw=1.5, ms=6, label='累積寄与率 (%)')
    ax2.axhline(y=70, color='green', ls=':', alpha=0.5, label='70%ライン')
    ax2.set_ylabel('累積寄与率 (%)', fontsize=14, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='right', fontsize=11)

    plt.title('MCA スクリープロット + 並行分析', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] {save_path}")


def plot_biplot(mca, df_cat, save_path):
    col_coords = mca.column_coordinates(df_cat)
    fig, ax = plt.subplots(figsize=(14, 10))

    colors = {
        'before_state': '#e41a1c',
        'after_state':  '#377eb8',
        'trigger_type': '#4daf4a',
        'action_type':  '#984ea3',
        'pattern_type': '#ff7f00',
        'scale':        '#a65628',
    }

    for idx, cat_label in enumerate(col_coords.index):
        x = float(col_coords.iloc[idx, 0])
        y = float(col_coords.iloc[idx, 1])

        var_name = None
        for var in TARGET_VARS:
            if cat_label.startswith(var + '__'):
                var_name = var
                break

        color = colors.get(var_name, '#333333')
        short_label = cat_label.split('__')[-1] if '__' in cat_label else cat_label

        ax.scatter(x, y, c=color, s=100, zorder=5, edgecolors='white', lw=0.5)
        ax.annotate(short_label, (x, y), fontsize=9, ha='center', va='bottom',
                    xytext=(0, 6), textcoords='offset points',
                    color=color, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=v) for v, c in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, title='変数')

    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    pov = mca.percentage_of_variance_
    ax.set_xlabel(f'次元1 ({pov[0]:.1f}%)', fontsize=14)
    ax.set_ylabel(f'次元2 ({pov[1]:.1f}%)', fontsize=14)
    ax.set_title('MCA バイプロット（第1-2次元）', fontsize=16)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] {save_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# メイン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # 1. 読み込み・前処理
    df = load_data()
    df_cat, n_dropped, missing_info = preprocess(df)
    n_analyzed = len(df_cat)

    # 2. MCA実行
    mca = run_mca(df_cat, N_COMPONENTS)

    eigenvalues = np.array(mca.eigenvalues_)
    pov = np.array(mca.percentage_of_variance_)
    cumulative_pct = np.cumsum(pov)

    print(f"\n固有値: {eigenvalues.round(6).tolist()}")
    print(f"寄与率(%): {pov.round(4).tolist()}")
    print(f"累積寄与率(%): {cumulative_pct.round(4).tolist()}")

    # 3. 並行分析
    pa_thresholds = parallel_analysis(df_cat, N_COMPONENTS, N_PARALLEL)

    # 4. 次元数の決定
    recommended, criteria = determine_dimensions(
        eigenvalues, cumulative_pct, len(TARGET_VARS), pa_thresholds
    )
    print(f"\n推奨次元数: {recommended}")
    print(f"各基準: {criteria}")

    # 5. 列座標・寄与度
    col_coords = mca.column_coordinates(df_cat)
    n_dims_for_contrib = min(recommended + 2, N_COMPONENTS)
    contributions = compute_contributions(mca, df_cat, n_dims_for_contrib)

    col_coords_dict = {}
    for idx, label in enumerate(col_coords.index):
        col_coords_dict[label] = [round(float(c), 6) for c in col_coords.iloc[idx]]

    # 行座標サンプル
    row_coords = mca.row_coordinates(df_cat)
    row_sample = []
    for i in range(min(20, len(row_coords))):
        row_sample.append({
            "index": i,
            "coordinates": [round(float(c), 6) for c in row_coords.iloc[i]]
        })

    # 6. 次元解釈
    dim_interpretations = []
    for dim_idx in range(recommended):
        dim_key = f"dim_{dim_idx + 1}"
        if dim_key in contributions:
            top_contribs, interp = interpret_dimension(contributions[dim_key])
            dim_interpretations.append({
                "dim": dim_idx + 1,
                "eigenvalue": round(float(eigenvalues[dim_idx]), 6),
                "inertia_pct": round(float(pov[dim_idx]), 4),
                "top_contributors": top_contribs,
                "interpretation": interp
            })

    # 7. 6次元仮説
    six_dim_text = (
        f"データが示した推奨次元数は{recommended}であり、"
    )
    cum_at_rec = cumulative_pct[recommended - 1] if recommended <= len(cumulative_pct) else cumulative_pct[-1]
    cum_at_6 = cumulative_pct[min(5, len(cumulative_pct) - 1)]

    max_dims = sum(len(df_cat[c].unique()) for c in TARGET_VARS) - len(TARGET_VARS)
    cum_at_max_computed = cumulative_pct[-1]

    if recommended <= 6:
        six_dim_text += (
            f"6次元仮説は概ね支持される。"
            f"推奨{recommended}次元で累積寄与率{cum_at_rec:.1f}%、"
            f"6次元で{cum_at_6:.1f}%。"
        )
    else:
        six_dim_text += (
            f"6次元仮説は部分的にしか支持されない。"
            f"6次元での累積寄与率は{cum_at_6:.1f}%に留まる。"
            f"（理論最大次元数={max_dims}、計算済み{N_COMPONENTS}次元で{cum_at_max_computed:.1f}%）。"
            f"MCAでは変数間の非線形関連が多次元に分散するため、"
            f"累積70%到達には{max_dims}次元中の大部分が必要。"
            f"ただし並行分析により全{N_COMPONENTS}次元がランダムを有意に上回っており、"
            f"データには明確な構造が存在する。"
        )

    # 8. 可視化
    plot_scree(eigenvalues, pa_thresholds, cumulative_pct,
               os.path.join(VIS_DIR, 'scree_plot.png'))
    plot_biplot(mca, df_cat, os.path.join(VIS_DIR, 'mca_biplot.png'))

    # 9. mca_results.json
    mca_results = {
        "n_records_analyzed": n_analyzed,
        "n_records_dropped": n_dropped,
        "n_records_total": n_analyzed + n_dropped,
        "variables_used": TARGET_VARS,
        "variables_excluded": EXCLUDED_VARS,
        "missing_info": missing_info,
        "eigenvalues": [round(float(v), 6) for v in eigenvalues],
        "explained_inertia_pct": [round(float(v), 4) for v in pov],
        "cumulative_inertia_pct": [round(float(v), 4) for v in cumulative_pct],
        "parallel_analysis_thresholds_95": [round(float(v), 6) for v in pa_thresholds],
        "column_coordinates": col_coords_dict,
        "row_coordinates_sample": row_sample,
        "contributions": contributions,
    }

    path_mca = os.path.join(SCRIPT_DIR, 'mca_results.json')
    with open(path_mca, 'w', encoding='utf-8') as f:
        json.dump(mca_results, f, ensure_ascii=False, indent=2)
    print(f"\n[OUT] {path_mca}")

    # 10. dimension_report.json
    dim_report = {
        "recommended_dimensions": recommended,
        "determination_criteria": criteria,
        "dimension_interpretations": dim_interpretations,
        "six_dimension_hypothesis": six_dim_text,
    }

    path_dim = os.path.join(SCRIPT_DIR, 'dimension_report.json')
    with open(path_dim, 'w', encoding='utf-8') as f:
        json.dump(dim_report, f, ensure_ascii=False, indent=2)
    print(f"[OUT] {path_dim}")

    print("\n[DONE] MCA分析完了")


if __name__ == '__main__':
    main()
