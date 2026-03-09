# Phase 3: 同型性検証レポート v2.0

**実行日時**: 2026-03-09T12:35:20.092140
**使用事例数**: 11225 (除外: 111)
**置換回数**: 1000
**乱数シード**: 42

## テスト結果

### test_a_edge_overlap
- **H0**: G_obsのQ6エッジ重複率はランダム遷移と同等
- **統計量**: 0.0727
- **p値**: 0.6100
- **効果量**: -0.324 (medium, anti)
- **備考**: overlap=8/110, null_mean=0.0752, null_std=0.0076

### test_b_hamming_6bit
- **H0**: 遷移ペアの平均ハミング距離はランダム遷移と同等
- **統計量**: 3.0684
- **p値**: 0.7220
- **効果量**: 0.560 (large, anti)
- **備考**: null_mean=3.0633, null_std=0.0092, dist_distribution={0: 197, 1: 227, 2: 2922, 3: 3314, 4: 4395, 5: 170}

### test_c_laplacian_spectrum
- **H0**: G_obsのラプラシアンスペクトルとQ6のWasserstein距離はランダムグラフと同等
- **統計量**: 0.4271
- **p値**: 0.0000
- **効果量**: -4.735 (large, pro)
- **備考**: null_mean=0.454959, null_std=0.005889, n_valid_perms=1000

### test_d_cuogua_symmetry
- **H0**: 錯卦ペアの遷移確率類似度はランダムペアと同等
- **統計量**: 0.3436
- **p値**: 0.6300
- **効果量**: -0.351 (medium, anti)
- **備考**: valid_pairs=2/32, zero_outdeg=47, null_mean=0.4176

### test_e_community_nmi
- **H0**: G_obsとQ6のコミュニティ構造のNMIはランダム遷移と同等
- **統計量**: 0.4078
- **p値**: 0.0780
- **効果量**: 1.575 (large, pro)
- **備考**: G_obs communities=36, Q6 communities=5, null_mean=0.3822

## 総合判定

- **Fisher統計量**: 1389.2172725440137
- **Fisher p値**: 0.0
- **判定レベル**: Weak
- **要約**: Fisher chi2=1389.22, p=0.000000, level=Weak. Pro方向medium+: 2/5, Pro方向small+: 2/5

### 効果量プロファイル

| テスト | 効果量 | 方向 | ラベル |
|--------|--------|------|--------|
| test_a_edge_overlap | -0.324 | anti | medium |
| test_b_hamming_6bit | 0.560 | anti | large |
| test_c_laplacian_spectrum | -4.735 | pro | large |
| test_d_cuogua_symmetry | -0.351 | anti | medium |
| test_e_community_nmi | 1.575 | pro | large |

## 判定基準

- **Strong**: Fisher p < 0.01 かつ 5テスト中4つ以上がpro方向のmedium以上効果
- **Moderate**: Fisher p < 0.05 かつ 3つ以上がpro方向のsmall以上効果
- **Weak**: Fisher p < 0.05 だが pro方向効果は2つ以下
- **None**: Fisher p >= 0.05
