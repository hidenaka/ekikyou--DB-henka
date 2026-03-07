# Phase 3: 同型性検証レポート v2.0

**実行日時**: 2026-03-07T15:48:28.688569
**使用事例数**: 9228 (除外: 2108)
**置換回数**: 1000
**乱数シード**: 42

## テスト結果

### test_a_edge_overlap
- **H0**: G_obsのQ6エッジ重複率はランダム遷移と同等
- **統計量**: 0.0515
- **p値**: 1.0000
- **効果量**: -3.985 (large, anti)
- **備考**: overlap=10/194, null_mean=0.0954, null_std=0.0110

### test_b_hamming_6bit
- **H0**: 遷移ペアの平均ハミング距離はランダム遷移と同等
- **統計量**: 2.9413
- **p値**: 0.9060
- **効果量**: 1.358 (large, anti)
- **備考**: null_mean=2.9184, null_std=0.0168, dist_distribution={0: 1113, 1: 12, 2: 3616, 3: 78, 4: 3376, 5: 38, 6: 995}

### test_c_laplacian_spectrum
- **H0**: G_obsのラプラシアンスペクトルとQ6のWasserstein距離はランダムグラフと同等
- **統計量**: 0.4314
- **p値**: 0.0000
- **効果量**: -4.034 (large, pro)
- **備考**: null_mean=0.456181, null_std=0.006130, n_valid_perms=1000

### test_d_cuogua_symmetry
- **H0**: 錯卦ペアの遷移確率類似度はランダムペアと同等
- **統計量**: 0.3661
- **p値**: 0.0010
- **効果量**: 3.881 (large, pro)
- **備考**: valid_pairs=17/32, zero_outdeg=18, null_mean=0.1741

### test_e_community_nmi
- **H0**: G_obsとQ6のコミュニティ構造のNMIはランダム遷移と同等
- **統計量**: 0.2760
- **p値**: 0.0120
- **効果量**: 2.688 (large, pro)
- **備考**: G_obs communities=16, Q6 communities=5, null_mean=0.2207

## 総合判定

- **Fisher統計量**: 1404.4096955586583
- **Fisher p値**: 0.0
- **判定レベル**: Moderate
- **要約**: Fisher chi2=1404.41, p=0.000000, level=Moderate. Pro方向medium+: 3/5, Pro方向small+: 3/5

### 効果量プロファイル

| テスト | 効果量 | 方向 | ラベル |
|--------|--------|------|--------|
| test_a_edge_overlap | -3.985 | anti | large |
| test_b_hamming_6bit | 1.358 | anti | large |
| test_c_laplacian_spectrum | -4.034 | pro | large |
| test_d_cuogua_symmetry | 3.881 | pro | large |
| test_e_community_nmi | 2.688 | pro | large |

## 判定基準

- **Strong**: Fisher p < 0.01 かつ 5テスト中4つ以上がpro方向のmedium以上効果
- **Moderate**: Fisher p < 0.05 かつ 3つ以上がpro方向のsmall以上効果
- **Weak**: Fisher p < 0.05 だが pro方向効果は2つ以下
- **None**: Fisher p >= 0.05
