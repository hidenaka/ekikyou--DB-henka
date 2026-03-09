# Dual Annotation Agreement Report

## Summary

- **Pass 1 annotations**: 660
- **Pass 2 annotations**: 800
- **Common (matched) cases**: 660
- **Full agreement (all 4 fields)**: 131 (19.8%)
- **Cases with any disagreement**: 529 (80.2%)

## Per-Field Agreement

| Field | N | Raw Agreement | Cohen's kappa | Gwet's AC1 |
|-------|---|---------------|---------------|------------|
| before_lower | 660 | 71.5% | 0.656 | 0.677 |
| before_upper | 660 | 59.7% | 0.487 | 0.546 |
| after_lower | 660 | 71.8% | 0.668 | 0.679 |
| after_upper | 660 | 51.1% | 0.432 | 0.442 |
| **Average** | - | 63.5% | 0.561 | 0.586 |

### Kappa Interpretation

| Range | Interpretation |
|-------|---------------|
| < 0.20 | Poor |
| 0.21-0.40 | Fair |
| 0.41-0.60 | Moderate |
| 0.61-0.80 | Substantial |
| 0.81-1.00 | Almost perfect |

**Quality Gate**: FAIL

## 離/兌 Inner Trigram Coverage

Critical check: do 離 and 兌 now appear as lower (inner) trigrams?

| Trigram | Pass 1 (as inner) | Pass 2 (as inner) |
|---------|-------------------|-------------------|
| 離 | 237 | 195 |
| 兌 | 97 | 92 |

離 as inner: **Fixed** (appears in annotations)
兌 as inner: **Fixed** (appears in annotations)

## Confusion Matrices (Pass1 rows x Pass2 cols)

#### before_lower

| P1 \ P2 | 乾 | 坤 | 震 | 巽 | 坎 | 離 | 艮 | 兌 | Total |
|---|---|---|---|---|---|---|---|---|---|
| **乾** | 100 | 14 | 4 | 0 | 5 | 3 | 4 | 1 | 131 |
| **坤** | 2 | 92 | 0 | 0 | 2 | 1 | 8 | 3 | 108 |
| **震** | 2 | 1 | 19 | 0 | 1 | 0 | 0 | 0 | 23 |
| **巽** | 1 | 3 | 0 | 1 | 2 | 0 | 3 | 0 | 10 |
| **坎** | 7 | 3 | 13 | 0 | 114 | 1 | 14 | 0 | 152 |
| **離** | 16 | 23 | 1 | 0 | 6 | 38 | 17 | 2 | 103 |
| **艮** | 2 | 8 | 2 | 0 | 2 | 0 | 93 | 0 | 107 |
| **兌** | 3 | 6 | 0 | 0 | 0 | 1 | 1 | 15 | 26 |
| **Total** | 133 | 150 | 39 | 1 | 132 | 44 | 140 | 21 | 660 |

#### before_upper

| P1 \ P2 | 乾 | 坤 | 震 | 巽 | 坎 | 離 | 艮 | 兌 | Total |
|---|---|---|---|---|---|---|---|---|---|
| **乾** | 6 | 3 | 5 | 0 | 0 | 11 | 0 | 7 | 32 |
| **坤** | 0 | 177 | 0 | 1 | 0 | 1 | 10 | 3 | 192 |
| **震** | 0 | 4 | 37 | 2 | 26 | 2 | 1 | 1 | 73 |
| **巽** | 0 | 2 | 1 | 5 | 5 | 0 | 0 | 0 | 13 |
| **坎** | 4 | 8 | 39 | 6 | 110 | 1 | 13 | 0 | 181 |
| **離** | 2 | 1 | 9 | 0 | 4 | 12 | 0 | 2 | 30 |
| **艮** | 2 | 41 | 4 | 0 | 30 | 1 | 34 | 1 | 113 |
| **兌** | 5 | 0 | 0 | 0 | 1 | 7 | 0 | 13 | 26 |
| **Total** | 19 | 236 | 95 | 14 | 176 | 35 | 58 | 27 | 660 |

#### after_lower

| P1 \ P2 | 乾 | 坤 | 震 | 巽 | 坎 | 離 | 艮 | 兌 | Total |
|---|---|---|---|---|---|---|---|---|---|
| **乾** | 90 | 3 | 1 | 2 | 0 | 20 | 2 | 2 | 120 |
| **坤** | 0 | 22 | 0 | 2 | 6 | 1 | 7 | 0 | 38 |
| **震** | 1 | 0 | 15 | 1 | 3 | 0 | 1 | 0 | 21 |
| **巽** | 4 | 2 | 0 | 69 | 1 | 15 | 17 | 8 | 116 |
| **坎** | 0 | 2 | 8 | 0 | 88 | 2 | 6 | 0 | 106 |
| **離** | 13 | 1 | 0 | 7 | 0 | 103 | 4 | 6 | 134 |
| **艮** | 3 | 3 | 0 | 6 | 5 | 3 | 33 | 1 | 54 |
| **兌** | 3 | 4 | 0 | 3 | 0 | 7 | 0 | 54 | 71 |
| **Total** | 114 | 37 | 24 | 90 | 103 | 151 | 70 | 71 | 660 |

#### after_upper

| P1 \ P2 | 乾 | 坤 | 震 | 巽 | 坎 | 離 | 艮 | 兌 | Total |
|---|---|---|---|---|---|---|---|---|---|
| **乾** | 97 | 3 | 1 | 7 | 1 | 32 | 1 | 33 | 175 |
| **坤** | 0 | 35 | 0 | 11 | 0 | 1 | 8 | 4 | 59 |
| **震** | 3 | 2 | 25 | 1 | 10 | 2 | 1 | 0 | 44 |
| **巽** | 1 | 6 | 1 | 33 | 4 | 12 | 4 | 3 | 64 |
| **坎** | 0 | 7 | 22 | 5 | 39 | 0 | 16 | 2 | 91 |
| **離** | 3 | 0 | 8 | 2 | 7 | 14 | 1 | 3 | 38 |
| **艮** | 0 | 13 | 0 | 4 | 7 | 0 | 39 | 1 | 64 |
| **兌** | 22 | 4 | 3 | 21 | 0 | 20 | 0 | 55 | 125 |
| **Total** | 126 | 70 | 60 | 84 | 68 | 81 | 70 | 101 | 660 |

## Most Confused Trigram Pairs

### before_lower

| Pair | Total Confusions |
|------|-----------------|
| 坤 <-> 離 | 24 |
| 乾 <-> 離 | 19 |
| 艮 <-> 離 | 17 |
| 乾 <-> 坤 | 16 |
| 坤 <-> 艮 | 16 |

### before_upper

| Pair | Total Confusions |
|------|-----------------|
| 坎 <-> 震 | 65 |
| 坤 <-> 艮 | 51 |
| 坎 <-> 艮 | 43 |
| 乾 <-> 離 | 13 |
| 乾 <-> 兌 | 12 |

### after_lower

| Pair | Total Confusions |
|------|-----------------|
| 乾 <-> 離 | 33 |
| 巽 <-> 艮 | 23 |
| 巽 <-> 離 | 22 |
| 兌 <-> 離 | 13 |
| 坎 <-> 震 | 11 |

### after_upper

| Pair | Total Confusions |
|------|-----------------|
| 乾 <-> 兌 | 55 |
| 乾 <-> 離 | 35 |
| 坎 <-> 震 | 32 |
| 兌 <-> 巽 | 24 |
| 坎 <-> 艮 | 23 |

## Trigram Distribution Comparison

### before_lower

| Trigram | Pass 1 | Pass 2 | Diff |
|---------|--------|--------|------|
| 乾 | 131 | 164 | +33 |
| 坤 | 108 | 174 | +66 |
| 震 | 23 | 45 | +22 |
| 巽 | 10 | 5 | -5 |
| 坎 | 152 | 160 | +8 |
| 離 | 103 | 67 | -36 |
| 艮 | 107 | 156 | +49 |
| 兌 | 26 | 29 | +3 |

### before_upper

| Trigram | Pass 1 | Pass 2 | Diff |
|---------|--------|--------|------|
| 乾 | 32 | 30 | -2 |
| 坤 | 192 | 284 | +92 |
| 震 | 73 | 102 | +29 |
| 巽 | 13 | 21 | +8 |
| 坎 | 181 | 210 | +29 |
| 離 | 30 | 48 | +18 |
| 艮 | 113 | 77 | -36 |
| 兌 | 26 | 28 | +2 |

### after_lower

| Trigram | Pass 1 | Pass 2 | Diff |
|---------|--------|--------|------|
| 乾 | 120 | 154 | +34 |
| 坤 | 38 | 45 | +7 |
| 震 | 21 | 34 | +13 |
| 巽 | 116 | 106 | -10 |
| 坎 | 106 | 128 | +22 |
| 離 | 134 | 166 | +32 |
| 艮 | 54 | 81 | +27 |
| 兌 | 71 | 86 | +15 |

### after_upper

| Trigram | Pass 1 | Pass 2 | Diff |
|---------|--------|--------|------|
| 乾 | 175 | 155 | -20 |
| 坤 | 59 | 85 | +26 |
| 震 | 44 | 71 | +27 |
| 巽 | 64 | 102 | +38 |
| 坎 | 91 | 84 | -7 |
| 離 | 38 | 105 | +67 |
| 艮 | 64 | 77 | +13 |
| 兌 | 125 | 121 | -4 |

## Confidence-Stratified Agreement (by Pass 1 confidence)

### before_lower

| Confidence | N | Agreement | Kappa |
|------------|---|-----------|-------|
| high | 474 | 75.1% | 0.696 |
| medium | 186 | 62.4% | 0.524 |
| low | 0 | - | - |

### before_upper

| Confidence | N | Agreement | Kappa |
|------------|---|-----------|-------|
| high | 417 | 63.3% | 0.524 |
| medium | 243 | 53.5% | 0.414 |
| low | 0 | - | - |

### after_lower

| Confidence | N | Agreement | Kappa |
|------------|---|-----------|-------|
| high | 570 | 74.9% | 0.703 |
| medium | 89 | 52.8% | 0.458 |
| low | 1 | 0.0% | 0.000 |

### after_upper

| Confidence | N | Agreement | Kappa |
|------------|---|-----------|-------|
| high | 444 | 53.4% | 0.454 |
| medium | 216 | 46.3% | 0.378 |
| low | 0 | - | - |

## Cases Needing Adjudication

Total: 529 cases with at least one field disagreement.

| Disagreements per case | Count |
|------------------------|-------|
| 1 field(s) | 237 |
| 2 field(s) | 179 |
| 3 field(s) | 84 |
| 4 field(s) | 29 |

### Sample Disagreements (first 20)

**CORP_JP_024** ()
  - before_lower: Pass1=巽 vs Pass2=坤

**CORP_JP_026** ()
  - before_lower: Pass1=坤 vs Pass2=乾
  - after_upper: Pass1=乾 vs Pass2=離

**CORP_JP_044** ()
  - after_upper: Pass1=乾 vs Pass2=離

**CORP_JP_052** ()
  - before_upper: Pass1=巽 vs Pass2=坎

**CORP_JP_071** ()
  - before_lower: Pass1=坎 vs Pass2=乾
  - after_upper: Pass1=巽 vs Pass2=離

**CORP_JP_084** ()
  - after_lower: Pass1=艮 vs Pass2=乾

**CORP_JP_085** ()
  - after_upper: Pass1=兌 vs Pass2=震

**CORP_JP_088** ()
  - after_lower: Pass1=巽 vs Pass2=離

**CORP_JP_093** ()
  - after_upper: Pass1=坤 vs Pass2=艮

**CORP_JP_104** ()
  - before_upper: Pass1=坎 vs Pass2=艮
  - after_lower: Pass1=坤 vs Pass2=坎
  - after_upper: Pass1=坎 vs Pass2=震

**CORP_JP_109** ()
  - after_lower: Pass1=巽 vs Pass2=艮
  - after_upper: Pass1=坤 vs Pass2=巽

**CORP_JP_115** ()
  - after_lower: Pass1=巽 vs Pass2=艮

**CORP_JP_1197** ()
  - after_upper: Pass1=震 vs Pass2=坤

**CORP_JP_125** ()
  - before_lower: Pass1=坎 vs Pass2=艮
  - after_upper: Pass1=巽 vs Pass2=坤

**CORP_JP_128** ()
  - before_lower: Pass1=巽 vs Pass2=艮
  - before_upper: Pass1=坎 vs Pass2=巽
  - after_upper: Pass1=巽 vs Pass2=坎

**CORP_JP_143** ()
  - before_upper: Pass1=坎 vs Pass2=震
  - after_lower: Pass1=巽 vs Pass2=艮
  - after_upper: Pass1=艮 vs Pass2=坤

**CORP_JP_144** ()
  - before_lower: Pass1=艮 vs Pass2=坎

**CORP_JP_178** ()
  - before_lower: Pass1=巽 vs Pass2=坎
  - before_upper: Pass1=坎 vs Pass2=巽

**CORP_JP_200** ()
  - before_lower: Pass1=坎 vs Pass2=乾

**CORP_JP_2177** ()
  - before_lower: Pass1=坎 vs Pass2=震
  - before_upper: Pass1=震 vs Pass2=坎
  - after_lower: Pass1=乾 vs Pass2=離
  - after_upper: Pass1=兌 vs Pass2=乾
