#!/usr/bin/env python3
"""
Phase 2B-4: Clustering Analysis on MCA coordinate space
易経変化ロジックDB — クラスタリング分析

Uses 6 categorical variables (before_state, after_state, trigger_type, action_type, pattern_type, scale)
to perform MCA, then clusters in 10-dim MCA space using K-Means, DBSCAN, and Agglomerative Clustering.
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import prince
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from collections import Counter

warnings.filterwarnings('ignore')
np.random.seed(42)

# --- Paths ---
BASE_DIR = "/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB"
DATA_PATH = os.path.join(BASE_DIR, "data/raw/cases.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/phase2")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

# --- Japanese font ---
plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. Load data ---
print("[1/6] Loading data...")
records = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

df = pd.DataFrame(records)
n_records = len(df)
print(f"  Loaded {n_records} records")

# --- 2. MCA on 6 variables ---
VARIABLES = ['before_state', 'after_state', 'trigger_type', 'action_type', 'pattern_type', 'scale']
N_DIMS = 10

print("[2/6] Running MCA...")
df_mca = df[VARIABLES].copy()
# Fill missing values
for col in VARIABLES:
    df_mca[col] = df_mca[col].fillna('unknown')

mca = prince.MCA(n_components=N_DIMS, random_state=42)
mca = mca.fit(df_mca)
coords = mca.row_coordinates(df_mca)
coords.columns = [f"dim_{i+1}" for i in range(N_DIMS)]
X = coords.values
print(f"  MCA coordinates shape: {X.shape}")

# Optional: scale for DBSCAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Clustering comparison ---
print("[3/6] Comparing clustering methods...")

# --- K-Means ---
k_range = range(2, 21)
kmeans_silhouettes = {}
kmeans_inertias = {}
kmeans_ch = {}

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=min(5000, n_records), random_state=42)
    ch = calinski_harabasz_score(X, labels)
    kmeans_silhouettes[f"k_{k}"] = round(sil, 4)
    kmeans_inertias[f"k_{k}"] = round(km.inertia_, 2)
    kmeans_ch[f"k_{k}"] = round(ch, 2)
    print(f"    K-Means k={k}: silhouette={sil:.4f}, CH={ch:.1f}")

# Best k by silhouette
best_k_kmeans = int(max(kmeans_silhouettes, key=kmeans_silhouettes.get).split('_')[1])
print(f"  K-Means best k (silhouette): {best_k_kmeans}")

# --- DBSCAN ---
print("  Running DBSCAN grid search...")
best_dbscan = {'eps': None, 'min_samples': None, 'n_clusters': 0, 'n_noise': 0, 'silhouette': -1}
eps_values = np.arange(0.3, 2.1, 0.1)
min_samples_values = [3, 5, 7, 10, 15, 20]

for eps in eps_values:
    for ms in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=ms)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Require 2-30 clusters for meaningful results
        if 2 <= n_clusters <= 30:
            mask = labels != -1
            if mask.sum() > n_clusters:
                try:
                    sil = silhouette_score(X_scaled[mask], labels[mask],
                                           sample_size=min(5000, int(mask.sum())), random_state=42)
                    n_noise = int((labels == -1).sum())
                    noise_ratio = n_noise / len(labels)
                    # Prefer solutions with <30% noise and good silhouette
                    if noise_ratio < 0.3 and sil > best_dbscan['silhouette']:
                        best_dbscan = {
                            'eps': round(float(eps), 2),
                            'min_samples': int(ms),
                            'n_clusters': int(n_clusters),
                            'n_noise': int(n_noise),
                            'silhouette': round(float(sil), 4)
                        }
                except:
                    pass

print(f"  DBSCAN best: eps={best_dbscan['eps']}, min_samples={best_dbscan['min_samples']}, "
      f"clusters={best_dbscan['n_clusters']}, noise={best_dbscan['n_noise']}, sil={best_dbscan['silhouette']}")

# --- Hierarchical (Agglomerative) ---
print("  Running Agglomerative Clustering...")
hier_silhouettes = {}
hier_ch = {}

for k in k_range:
    ac = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = ac.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=min(5000, n_records), random_state=42)
    ch = calinski_harabasz_score(X, labels)
    hier_silhouettes[f"k_{k}"] = round(sil, 4)
    hier_ch[f"k_{k}"] = round(ch, 2)
    print(f"    Hierarchical k={k}: silhouette={sil:.4f}, CH={ch:.1f}")

best_k_hier = int(max(hier_silhouettes, key=hier_silhouettes.get).split('_')[1])
print(f"  Hierarchical best k (silhouette): {best_k_hier}")

# --- 4. Select best method ---
print("[4/6] Selecting best method and analyzing clusters...")

candidates = {
    'kmeans': (best_k_kmeans, kmeans_silhouettes[f"k_{best_k_kmeans}"]),
    'hierarchical': (best_k_hier, hier_silhouettes[f"k_{best_k_hier}"]),
}
# Only include DBSCAN if it found 3+ meaningful clusters
if best_dbscan['silhouette'] > 0 and best_dbscan['n_clusters'] >= 3:
    candidates['dbscan'] = (best_dbscan['n_clusters'], best_dbscan['silhouette'])

best_method = max(candidates, key=lambda m: candidates[m][1])
best_k = candidates[best_method][0]
print(f"  Best method: {best_method}, k={best_k}, silhouette={candidates[best_method][1]}")

# Get final labels
if best_method == 'kmeans':
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = final_model.fit_predict(X)
elif best_method == 'hierarchical':
    final_model = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    final_labels = final_model.fit_predict(X)
else:  # dbscan
    final_model = DBSCAN(eps=best_dbscan['eps'], min_samples=best_dbscan['min_samples'])
    final_labels = final_model.fit_predict(X_scaled)

df['cluster'] = final_labels

# --- Cluster profiles ---
cluster_profiles = {}
for cl in sorted(set(final_labels)):
    if cl == -1:
        continue
    mask = df['cluster'] == cl
    subset = df[mask]
    profile = {
        'size': int(mask.sum()),
        'dominant_before_state': subset['before_state'].mode().iloc[0] if len(subset) > 0 else '',
        'dominant_after_state': subset['after_state'].mode().iloc[0] if len(subset) > 0 else '',
        'dominant_trigger_type': subset['trigger_type'].mode().iloc[0] if len(subset) > 0 else '',
        'dominant_action_type': subset['action_type'].mode().iloc[0] if len(subset) > 0 else '',
        'dominant_pattern_type': subset['pattern_type'].mode().iloc[0] if len(subset) > 0 else '',
        'dominant_scale': subset['scale'].mode().iloc[0] if len(subset) > 0 else '',
        'full_distribution': {}
    }
    for var in VARIABLES:
        counts = subset[var].value_counts().to_dict()
        profile['full_distribution'][var] = {str(k): int(v) for k, v in counts.items()}
    cluster_profiles[f"cluster_{cl}"] = profile

# --- Correspondence analysis: clusters vs trigrams/patterns ---
print("  Checking correspondence with 8 trigrams and 5 patterns...")

# vs 8 trigrams (before_hex)
if 'before_hex' in df.columns:
    valid_mask = (df['cluster'] != -1) & df['before_hex'].notna()
    ari_trigrams = adjusted_rand_score(df.loc[valid_mask, 'before_hex'], df.loc[valid_mask, 'cluster'])
    nmi_trigrams = normalized_mutual_info_score(df.loc[valid_mask, 'before_hex'], df.loc[valid_mask, 'cluster'])
    if nmi_trigrams > 0.5:
        trigram_corr = "対応あり"
    elif nmi_trigrams > 0.2:
        trigram_corr = "部分的"
    else:
        trigram_corr = "なし"
    print(f"  vs 8 trigrams: ARI={ari_trigrams:.4f}, NMI={nmi_trigrams:.4f} -> {trigram_corr}")
else:
    trigram_corr = "データなし"
    ari_trigrams = 0.0
    nmi_trigrams = 0.0

# vs 5 patterns
valid_mask2 = (df['cluster'] != -1) & df['pattern_type'].notna()
ari_patterns = adjusted_rand_score(df.loc[valid_mask2, 'pattern_type'], df.loc[valid_mask2, 'cluster'])
nmi_patterns = normalized_mutual_info_score(df.loc[valid_mask2, 'pattern_type'], df.loc[valid_mask2, 'cluster'])
if nmi_patterns > 0.5:
    pattern_corr = "対応あり"
elif nmi_patterns > 0.2:
    pattern_corr = "部分的"
else:
    pattern_corr = "なし"
print(f"  vs 5 patterns: ARI={ari_patterns:.4f}, NMI={nmi_patterns:.4f} -> {pattern_corr}")

# --- 5. Visualizations ---
print("[5/6] Creating visualizations...")

# --- cluster_map.png: MCA dim1-dim2 scatter ---
fig, ax = plt.subplots(figsize=(14, 10))
unique_labels = sorted(set(final_labels))
n_colors = max(len([l for l in unique_labels if l != -1]), 2)
colors = plt.cm.tab10(np.linspace(0, 1, n_colors))

color_idx = 0
for cl in unique_labels:
    mask = final_labels == cl
    if cl == -1:
        ax.scatter(X[mask, 0], X[mask, 1], c='gray', alpha=0.2, s=6, label="Noise")
    else:
        c = colors[color_idx]
        ax.scatter(X[mask, 0], X[mask, 1], c=[c], alpha=0.4, s=8,
                   label=f"Cluster {cl} (n={mask.sum()})")
        color_idx += 1

ax.set_xlabel('MCA Dimension 1 (9.77%)', fontsize=12)
ax.set_ylabel('MCA Dimension 2 (7.71%)', fontsize=12)
ax.set_title(f'クラスタリング結果 ({best_method.upper()}, k={best_k}) - MCA空間', fontsize=14)
ax.legend(loc='upper right', fontsize=9, markerscale=3)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'cluster_map.png'), dpi=150)
plt.close()
print("  Saved cluster_map.png")

# --- parallel_coordinates.png ---
fig, ax = plt.subplots(figsize=(16, 8))

var_labels = {
    'before_state': '初期状態',
    'after_state': '結果状態',
    'trigger_type': 'トリガー',
    'action_type': 'アクション',
    'pattern_type': 'パターン',
    'scale': 'スケール'
}

parallel_data = []
for cl in sorted(set(final_labels)):
    if cl == -1:
        continue
    mask = df['cluster'] == cl
    subset = df[mask]
    row = {'cluster': cl}
    for var in VARIABLES:
        mode_val = subset[var].mode().iloc[0]
        proportion = (subset[var] == mode_val).sum() / len(subset)
        row[f'{var}_prop'] = proportion
        row[f'{var}_label'] = mode_val
    parallel_data.append(row)

par_df = pd.DataFrame(parallel_data)

x_positions = range(len(VARIABLES))
color_idx = 0
for idx, row in par_df.iterrows():
    cl = int(row['cluster'])
    y_vals = [row[f'{v}_prop'] for v in VARIABLES]
    color = colors[color_idx]
    color_idx += 1
    ax.plot(x_positions, y_vals, 'o-', color=color, linewidth=2.5, markersize=8,
            label=f"Cluster {cl} (n={cluster_profiles[f'cluster_{cl}']['size']})", alpha=0.8)
    for xi, var in enumerate(VARIABLES):
        label_text = str(row[f'{var}_label'])
        if len(label_text) > 10:
            label_text = label_text[:10] + '..'
        offset = 0.015 * (color_idx - len(par_df)//2)
        ax.annotate(label_text, (xi, y_vals[xi] + offset),
                   fontsize=7, ha='center', va='bottom', color=color, fontweight='bold')

ax.set_xticks(list(x_positions))
ax.set_xticklabels([var_labels[v] for v in VARIABLES], fontsize=11)
ax.set_ylabel('最頻値の比率', fontsize=12)
ax.set_title(f'クラスタ別プロファイル（パラレル座標） - {best_method.upper()}, k={best_k}', fontsize=14)
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(0, 1.15)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'parallel_coordinates.png'), dpi=150)
plt.close()
print("  Saved parallel_coordinates.png")

# --- 6. Build results JSON ---
print("[6/6] Saving results...")

results = {
    "n_records": n_records,
    "mca_dimensions_used": N_DIMS,
    "methods_compared": {
        "kmeans": {
            "silhouette_scores": kmeans_silhouettes,
            "inertia": kmeans_inertias,
            "calinski_harabasz": kmeans_ch,
            "best_k": best_k_kmeans
        },
        "dbscan": {
            "best_eps": best_dbscan['eps'],
            "best_min_samples": best_dbscan['min_samples'],
            "n_clusters": best_dbscan['n_clusters'],
            "n_noise": best_dbscan['n_noise'],
            "silhouette_score": best_dbscan['silhouette']
        },
        "hierarchical": {
            "silhouette_scores": hier_silhouettes,
            "calinski_harabasz": hier_ch,
            "best_k": best_k_hier
        }
    },
    "best_method": best_method,
    "best_k": best_k,
    "best_silhouette": candidates[best_method][1],
    "cluster_profiles": cluster_profiles,
    "correspondence_analysis": {
        "vs_8_trigrams": trigram_corr,
        "vs_8_trigrams_detail": {
            "adjusted_rand_index": round(float(ari_trigrams), 4),
            "normalized_mutual_info": round(float(nmi_trigrams), 4)
        },
        "vs_5_patterns": pattern_corr,
        "vs_5_patterns_detail": {
            "adjusted_rand_index": round(float(ari_patterns), 4),
            "normalized_mutual_info": round(float(nmi_patterns), 4)
        },
        "vs_64_hexagrams": "検証対象外（Phase 3）"
    }
}

output_path = os.path.join(OUTPUT_DIR, 'cluster_results.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nDone! Results saved to {output_path}")
print(f"  Best method: {best_method}")
print(f"  Best k: {best_k}")
print(f"  Best silhouette: {candidates[best_method][1]}")
for cl_name, prof in cluster_profiles.items():
    print(f"  {cl_name}: size={prof['size']}, "
          f"pattern={prof['dominant_pattern_type']}, "
          f"before={prof['dominant_before_state']}, "
          f"after={prof['dominant_after_state']}")
