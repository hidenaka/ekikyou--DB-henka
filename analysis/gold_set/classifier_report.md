# Trigram Classifier Report

**Date**: 2026-03-09
**Method**: TF-IDF (char n-gram 1-3) + Logistic Regression (balanced class weights)
**Training data**: Gold 200 annotations (160 train / 40 test)
**Applied to**: 11,336 cases (full DB)

## Test Set Performance

| Field | Accuracy | F1 (macro) | F1 (weighted) |
|-------|----------|------------|---------------|
| before_lower | 0.9000 | 0.8799 | 0.9004 |
| before_upper | 0.4750 | 0.3680 | 0.4388 |
| after_lower | 0.7500 | 0.4315 | 0.7220 |
| after_upper | 0.6750 | 0.4084 | 0.6409 |
| **Average** | **0.7000** | **0.5220** | **0.6755** |

## Agreement Rates

| Field | Gold vs Classifier |
|-------|--------------------|
| before_lower | 0.9000 |
| before_upper | 0.4750 |
| after_lower | 0.7500 |
| after_upper | 0.6750 |

## Per-Class Performance (Test Set)

### before_lower

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| 乾 | 0.80 | 0.80 | 0.80 | 5 |
| 坤 | 0.86 | 1.00 | 0.92 | 6 |
| 震 | 0.80 | 0.80 | 0.80 | 5 |
| 巽 | 0.80 | 0.80 | 0.80 | 5 |
| 坎 | 1.00 | 0.92 | 0.96 | 12 |
| 離 | 0.00 | 0.00 | 0.00 | 0 |
| 艮 | 1.00 | 1.00 | 1.00 | 7 |
| 兌 | 0.00 | 0.00 | 0.00 | 0 |

### before_upper

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| 乾 | 0.00 | 0.00 | 0.00 | 2 |
| 坤 | 0.00 | 0.00 | 0.00 | 3 |
| 震 | 0.60 | 0.33 | 0.43 | 9 |
| 巽 | 0.57 | 1.00 | 0.73 | 4 |
| 坎 | 0.57 | 0.40 | 0.47 | 10 |
| 離 | 0.25 | 0.33 | 0.29 | 3 |
| 艮 | 0.50 | 0.86 | 0.63 | 7 |
| 兌 | 0.33 | 0.50 | 0.40 | 2 |

### after_lower

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| 乾 | 0.88 | 1.00 | 0.94 | 15 |
| 坤 | 0.67 | 1.00 | 0.80 | 4 |
| 震 | 0.25 | 0.50 | 0.33 | 2 |
| 巽 | 0.00 | 0.00 | 0.00 | 1 |
| 坎 | 1.00 | 0.50 | 0.67 | 10 |
| 離 | 0.00 | 0.00 | 0.00 | 1 |
| 艮 | 0.62 | 0.83 | 0.71 | 6 |
| 兌 | 0.00 | 0.00 | 0.00 | 1 |

### after_upper

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| 乾 | 0.00 | 0.00 | 0.00 | 1 |
| 坤 | 0.00 | 0.00 | 0.00 | 2 |
| 震 | 0.38 | 0.60 | 0.46 | 5 |
| 巽 | 0.50 | 0.50 | 0.50 | 2 |
| 坎 | 0.75 | 0.50 | 0.60 | 6 |
| 離 | 0.78 | 0.93 | 0.85 | 15 |
| 艮 | 0.86 | 0.86 | 0.86 | 7 |
| 兌 | 0.00 | 0.00 | 0.00 | 2 |

## Full Dataset Application (11,336 cases)

### Trigram Distribution

| Trigram | before_lower | before_upper | after_lower | after_upper |
|---------|-------------|-------------|------------|------------|
| 乾 | 1,243 (11.0%) | 313 (2.8%) | 4,257 (37.6%) | 8 (0.1%) |
| 坤 | 2,620 (23.1%) | 688 (6.1%) | 1,226 (10.8%) | 321 (2.8%) |
| 震 | 1,427 (12.6%) | 2,655 (23.4%) | 1,044 (9.2%) | 1,668 (14.7%) |
| 巽 | 1,781 (15.7%) | 2,262 (20.0%) | 28 (0.2%) | 1,008 (8.9%) |
| 坎 | 1,545 (13.6%) | 1,865 (16.5%) | 2,005 (17.7%) | 1,203 (10.6%) |
| 離 | 0 (0.0%) | 1,129 (10.0%) | 76 (0.7%) | 5,340 (47.1%) |
| 艮 | 2,720 (24.0%) | 1,827 (16.1%) | 2,697 (23.8%) | 1,728 (15.2%) |
| 兌 | 0 (0.0%) | 597 (5.3%) | 3 (0.0%) | 60 (0.5%) |

### Hexagram Quality Metrics

| Metric | Before | After |
|--------|--------|-------|
| Unique hexagrams | 43/64 | 39/64 |
| Pure hexagram rate | 14.5% | 1.6% |

### Comparison with Prior Method (keyword-based pure hexagram)

| Metric | Old (keyword) | New (classifier) |
|--------|---------------|------------------|
| Pure before rate | 98.2% | 14.5% |
| Pure after rate | 98.2% | 1.6% |
| Unique before hexagrams | ~8 | 43 |
| Unique after hexagrams | ~8 | 39 |

### Known Issues

1. **after_lower偏り**: 乾(37.6%)と艮(23.8%)に集中。離(0.7%)、巽(0.2%)、兌(0.0%)がほぼ未使用
2. **after_upper偏り**: 離(47.1%)に集中。乾(0.1%)、兌(0.5%)が希少
3. **before_lower偏り**: 離と兌が0件。Gold 200のtraining setに離・兌のサンプルが少なかったため
4. **学習データサイズ**: 200件は8クラス分類には少ない。特に低頻度クラスの学習が不十分

### 改善方針

- Gold setを500件以上に拡大し、8卦の均等代表を確保
- 特徴量にbefore_state/after_stateの構造化ラベルを追加
- 文字n-gramに加え、単語n-gramも併用
- モデルのアンサンブル化（Random Forest, SVM等）
