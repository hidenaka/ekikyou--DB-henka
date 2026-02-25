# Phase 2A: MCA分析レポート

**分析日**: 2026-02-25
**乱数シード**: 42

---

## 1. データ概要

- **総レコード数**: 13,060 件
- **分析対象レコード数**: 13,060 件
- **欠損値による除外**: 0 件
- **分析対象変数**: 7 変数
- **総カテゴリ数 (J)**: 88

### 重要な発見: 実データのカテゴリ数

スキーマ定義のカテゴリ数と、データ内に実際に存在するカテゴリ数には乖離がある:

| 変数 | スキーマ定義 | 実データ |
|------|------------|---------|
| before_state | 6 | 15 |
| trigger_type | 4 | 8 |
| action_type | 8 | 22 |
| after_state | 6 | 20 |
| pattern_type | 14 | 14 |
| outcome | 4 | 4 |
| scale | 5 | 5 |

この乖離はデータ品質改善フェーズ(Phase B/D)での追加事例による
新カテゴリの導入と考えられる。MCA分析は実データのカテゴリをそのまま使用する。

### 欠損値レポート

| 変数 | 欠損数 | 欠損率 |
|------|--------|--------|
| before_state | 0 | 0.00% |
| trigger_type | 0 | 0.00% |
| action_type | 0 | 0.00% |
| after_state | 0 | 0.00% |
| pattern_type | 0 | 0.00% |
| outcome | 0 | 0.00% |
| scale | 0 | 0.00% |

### 度数分布

#### before_state

| カテゴリ | 件数 | 割合 |
|----------|------|------|
| 安定・平和 | 2,778 | 21.3% |
| 成長痛 | 2,625 | 20.1% |
| 停滞・閉塞 | 2,520 | 19.3% |
| どん底・危機 | 1,326 | 10.2% |
| 絶頂・慢心 | 1,021 | 7.8% |
| 混乱・カオス | 823 | 6.3% |
| 混乱・衰退 | 819 | 6.3% |
| 安定・停止 | 564 | 4.3% |
| 安定成長・成功 | 331 | 2.5% |
| 拡大・繁栄 | 119 | 0.9% |
| 成長・拡大 | 84 | 0.6% |
| 調和・繁栄 | 19 | 0.1% |
| 縮小安定・生存 | 18 | 0.1% |
| V字回復・大成功 | 11 | 0.1% |
| 急成長・拡大 | 2 | 0.0% |

#### trigger_type

| カテゴリ | 件数 | 割合 |
|----------|------|------|
| 意図的決断 | 4,689 | 35.9% |
| 外部ショック | 4,633 | 35.5% |
| 内部崩壊 | 1,862 | 14.3% |
| 偶発・出会い | 925 | 7.1% |
| 自然推移 | 665 | 5.1% |
| 内部矛盾・自壊 | 259 | 2.0% |
| 自然推移・成熟 | 15 | 0.1% |
| 拡大・過剰 | 12 | 0.1% |

#### action_type

| カテゴリ | 件数 | 割合 |
|----------|------|------|
| 攻める・挑戦 | 3,117 | 23.9% |
| 耐える・潜伏 | 2,129 | 16.3% |
| 対話・融合 | 1,797 | 13.8% |
| 守る・維持 | 1,616 | 12.4% |
| 刷新・破壊 | 1,422 | 10.9% |
| 捨てる・撤退 | 964 | 7.4% |
| 分散・探索 | 499 | 3.8% |
| 捨てる・転換 | 433 | 3.3% |
| 逃げる・放置 | 369 | 2.8% |
| 交流・発表 | 195 | 1.5% |
| 分散・スピンオフ | 169 | 1.3% |
| 集中・拡大 | 140 | 1.1% |
| 拡大・攻め | 116 | 0.9% |
| 撤退・収縮 | 34 | 0.3% |
| 撤退・縮小 | 14 | 0.1% |
| 逃げる・分散 | 11 | 0.1% |
| 撤退・逃げる | 11 | 0.1% |
| 輝く・表現 | 11 | 0.1% |
| 逃げる・守る | 4 | 0.0% |
| 分散・独立 | 3 | 0.0% |
| 分散・多角化 | 3 | 0.0% |
| 分散する・独立する | 3 | 0.0% |

#### after_state

| カテゴリ | 件数 | 割合 |
|----------|------|------|
| 持続成長・大成功 | 2,205 | 16.9% |
| 崩壊・消滅 | 1,870 | 14.3% |
| 縮小安定・生存 | 1,505 | 11.5% |
| 安定・平和 | 1,110 | 8.5% |
| 安定成長・成功 | 1,091 | 8.4% |
| 停滞・閉塞 | 993 | 7.6% |
| 変質・新生 | 986 | 7.5% |
| V字回復・大成功 | 721 | 5.5% |
| 混乱・衰退 | 631 | 4.8% |
| 混乱・カオス | 378 | 2.9% |
| 安定・停止 | 363 | 2.8% |
| 拡大・繁栄 | 285 | 2.2% |
| どん底・危機 | 280 | 2.1% |
| 喜び・交流 | 220 | 1.7% |
| 迷走・混乱 | 148 | 1.1% |
| 現状維持・延命 | 126 | 1.0% |
| 成長・拡大 | 119 | 0.9% |
| 成長痛 | 26 | 0.2% |
| 分岐・様子見 | 2 | 0.0% |
| 消滅・破綻 | 1 | 0.0% |

#### pattern_type

| カテゴリ | 件数 | 割合 |
|----------|------|------|
| Steady_Growth | 1,965 | 15.0% |
| Slow_Decline | 1,440 | 11.0% |
| Shock_Recovery | 1,340 | 10.3% |
| Pivot_Success | 1,330 | 10.2% |
| Hubris_Collapse | 1,153 | 8.8% |
| Endurance | 1,129 | 8.6% |
| Breakthrough | 967 | 7.4% |
| Crisis_Pivot | 927 | 7.1% |
| Failed_Attempt | 818 | 6.3% |
| Stagnation | 691 | 5.3% |
| Managed_Decline | 409 | 3.1% |
| Exploration | 407 | 3.1% |
| Quiet_Fade | 303 | 2.3% |
| Decline | 181 | 1.4% |

#### outcome

| カテゴリ | 件数 | 割合 |
|----------|------|------|
| Success | 6,284 | 48.1% |
| Failure | 4,194 | 32.1% |
| Mixed | 2,322 | 17.8% |
| PartialSuccess | 260 | 2.0% |

#### scale

| カテゴリ | 件数 | 割合 |
|----------|------|------|
| company | 5,510 | 42.2% |
| individual | 3,217 | 24.6% |
| other | 2,165 | 16.6% |
| country | 1,381 | 10.6% |
| family | 787 | 6.0% |

## 2. MCA結果サマリー

### 分析対象変数（八卦タグは除外）

- `before_state` (15カテゴリ)
- `trigger_type` (8カテゴリ)
- `action_type` (22カテゴリ)
- `after_state` (20カテゴリ)
- `pattern_type` (14カテゴリ)
- `outcome` (4カテゴリ)
- `scale` (5カテゴリ)

### 固有値と寄与率

| 次元 | 固有値 | 寄与率 (%) | 累積寄与率 (%) |
|------|--------|-----------|---------------|
| 1 | 0.597559 | 8.32 | 8.32 |
| 2 | 0.509988 | 7.10 | 15.43 |
| 3 | 0.443109 | 6.17 | 21.60 |
| 4 | 0.393355 | 5.48 | 27.08 |
| 5 | 0.334070 | 4.65 | 31.73 |
| 6 | 0.305291 | 4.25 | 35.98 |
| 7 | 0.287282 | 4.00 | 39.99 |
| 8 | 0.283055 | 3.94 | 43.93 |
| 9 | 0.261912 | 3.65 | 47.58 |
| 10 | 0.251642 | 3.51 | 51.08 |
| 11 | 0.238821 | 3.33 | 54.41 |
| 12 | 0.227117 | 3.16 | 57.57 |
| 13 | 0.221513 | 3.09 | 60.66 |
| 14 | 0.213226 | 2.97 | 63.63 |
| 15 | 0.211753 | 2.95 | 66.58 |
| 16 | 0.191281 | 2.66 | 69.24 |
| 17 | 0.185405 | 2.58 | 71.83 |
| 18 | 0.177888 | 2.48 | 74.30 |
| 19 | 0.170226 | 2.37 | 76.67 |
| 20 | 0.167900 | 2.34 | 79.01 |
| 21 | 0.164033 | 2.28 | 81.30 |
| 22 | 0.157486 | 2.19 | 83.49 |
| 23 | 0.156958 | 2.19 | 85.68 |
| 24 | 0.152742 | 2.13 | 87.81 |
| 25 | 0.150320 | 2.09 | 89.90 |

**全慣性 (Total Inertia)**: 7.179082

## 3. 次元数の決定

### 5つの基準による判定

1. **スクリープロット（肘法）**: 5次元
   - Scree plot (elbow method): maximum second derivative of eigenvalues
2. **累積寄与率 >= 70%**: 17次元
   - Cumulative explained inertia >= 70%
3. **Kaiser基準 (1/K)**: 30次元
   - Kaiser criterion for MCA: eigenvalue > 1/K = 1/7 = 0.1429
4. **Greenacre基準 (平均固有値)**: 10次元
   - Greenacre criterion: eigenvalue > mean eigenvalue = 0.239303
5. **並行分析**: 30次元
   - Parallel Analysis (95th percentile of 100 random permutations)

### 結論: データが示した次元数は **10次元** である

- 全基準の結果: [5, 17, 30, 10, 30]
- 主要基準（スクリー・Greenacre・並行分析）: [5, 10, 30]
- 中央値による合意: **10次元**

> データ駆動の分析により **10次元** が最適と判定された。
> 先験的な6次元仮説とは異なる結果である。

### MCA慣性構造の特徴

MCA（多重対応分析）はPCA（主成分分析）とは異なり、固有値が一般に小さく、
累積寄与率が緩やかにしか上昇しない特徴がある。これはインジケータ行列の
希薄性（各行で1変数につき1つのカテゴリのみが1）に起因する。

したがって、MCAにおいては:
- 累積寄与率70%は厳しすぎる基準となりうる
- Greenacre基準（平均固有値超）がMCA固有の適切な基準
- 並行分析は大サンプル（N=13,060）では保守的になりにくい

## 4. 各次元の解釈

### 次元 1
- **固有値**: 0.597559
- **寄与率**: 8.32%
- **解釈ラベル**: Dim 1: [_逃げる・放置 | _Quiet_Fade | _崩壊・消滅] <-> [_分散・探索 | _拡大・繁栄 | _撤退・縮小]

**正の極（Positive Pole）**:
  - action_type__分散・探索: 3.1547
  - before_state__拡大・繁栄: 3.0728
  - action_type__撤退・縮小: 3.0549
  - trigger_type__自然推移: 3.0416
  - action_type__拡大・攻め: 2.9853

**負の極（Negative Pole）**:
  - action_type__逃げる・放置: -0.8105
  - pattern_type__Quiet_Fade: -0.7229
  - after_state__崩壊・消滅: -0.7207
  - pattern_type__Hubris_Collapse: -0.7139
  - trigger_type__内部崩壊: -0.7054

**変数別寄与（座標範囲）**:
  - action_type: range=3.9652
  - trigger_type: range=3.7470
  - after_state: range=3.6928
  - before_state: range=3.6652
  - pattern_type: range=3.5269
  - outcome: range=1.0812
  - scale: range=0.8407

### 次元 2
- **固有値**: 0.509988
- **寄与率**: 7.10%
- **解釈ラベル**: Dim 2: [_持続成長・大成功 | _Steady_Growth | _偶発・出会い] <-> [_撤退・縮小 | _Decline | _逃げる・放置]

**正の極（Positive Pole）**:
  - action_type__撤退・縮小: 1.9938
  - pattern_type__Decline: 1.6323
  - action_type__逃げる・放置: 1.5622
  - pattern_type__Quiet_Fade: 1.3300
  - after_state__混乱・衰退: 1.3091

**負の極（Negative Pole）**:
  - after_state__持続成長・大成功: -1.2461
  - pattern_type__Steady_Growth: -1.2264
  - trigger_type__偶発・出会い: -1.1979
  - action_type__分散・多角化: -1.1334
  - action_type__拡大・攻め: -0.9925

**変数別寄与（座標範囲）**:
  - action_type: range=3.1272
  - pattern_type: range=2.8587
  - after_state: range=2.5552
  - trigger_type: range=2.3681
  - outcome: range=1.9389
  - before_state: range=1.8781
  - scale: range=0.7250

### 次元 3
- **固有値**: 0.443109
- **寄与率**: 6.17%
- **解釈ラベル**: Dim 3: [_成長・拡大 | _拡大・攻め | _成長・拡大] <-> [_自然推移・成熟 | _調和・繁栄 | _逃げる・分散]

**正の極（Positive Pole）**:
  - trigger_type__自然推移・成熟: 7.1434
  - before_state__調和・繁栄: 6.8817
  - action_type__逃げる・分散: 4.7237
  - before_state__急成長・拡大: 3.6460
  - action_type__撤退・逃げる: 3.4234

**負の極（Negative Pole）**:
  - before_state__成長・拡大: -5.4943
  - action_type__拡大・攻め: -5.0676
  - after_state__成長・拡大: -4.9772
  - after_state__分岐・様子見: -3.1949
  - action_type__集中・拡大: -1.6612

**変数別寄与（座標範囲）**:
  - before_state: range=12.3759
  - action_type: range=9.7913
  - trigger_type: range=8.6447
  - after_state: range=8.3959
  - pattern_type: range=3.6129
  - outcome: range=0.5504
  - scale: range=0.3398

### 次元 4
- **固有値**: 0.393355
- **寄与率**: 5.48%
- **解釈ラベル**: Dim 4: [_Exploration | _分散・探索 | _混乱・衰退] <-> [_成長・拡大 | _拡大・攻め | _成長・拡大]

**正の極（Positive Pole）**:
  - before_state__成長・拡大: 8.3364
  - action_type__拡大・攻め: 7.7644
  - after_state__成長・拡大: 7.5867
  - trigger_type__自然推移・成熟: 4.7220
  - before_state__調和・繁栄: 4.4320

**負の極（Negative Pole）**:
  - pattern_type__Exploration: -2.4741
  - action_type__分散・探索: -2.1728
  - after_state__混乱・衰退: -1.7158
  - before_state__安定・停止: -1.3553
  - before_state__拡大・繁栄: -1.2179

**変数別寄与（座標範囲）**:
  - action_type: range=9.9371
  - before_state: range=9.6917
  - after_state: range=9.3025
  - trigger_type: range=4.9780
  - pattern_type: range=3.5057
  - outcome: range=0.6270
  - scale: range=0.3033

### 次元 5
- **固有値**: 0.334070
- **寄与率**: 4.65%
- **解釈ラベル**: Dim 5: [_交流・発表 | _喜び・交流 | _輝く・表現] <-> [_自然推移・成熟 | _調和・繁栄 | _逃げる・分散]

**正の極（Positive Pole）**:
  - trigger_type__自然推移・成熟: 7.8619
  - before_state__調和・繁栄: 7.3407
  - action_type__逃げる・分散: 3.1343
  - action_type__撤退・逃げる: 2.8407
  - after_state__分岐・様子見: 2.8018

**負の極（Negative Pole）**:
  - action_type__交流・発表: -6.2467
  - after_state__喜び・交流: -6.0118
  - action_type__輝く・表現: -5.3618
  - pattern_type__Crisis_Pivot: -1.3363
  - action_type__集中・拡大: -1.3090

**変数別寄与（座標範囲）**:
  - action_type: range=9.3810
  - after_state: range=8.8136
  - before_state: range=8.2914
  - trigger_type: range=7.9805
  - pattern_type: range=2.8957
  - outcome: range=0.8590
  - scale: range=0.8131

### 次元 6
- **固有値**: 0.305291
- **寄与率**: 4.25%
- **解釈ラベル**: Dim 6: [_分岐・様子見 | _分散する・独立する | _成長・拡大] <-> [_自然推移・成熟 | _調和・繁栄 | _逃げる・分散]

**正の極（Positive Pole）**:
  - trigger_type__自然推移・成熟: 12.5985
  - before_state__調和・繁栄: 11.4138
  - action_type__逃げる・分散: 5.0627
  - action_type__集中・拡大: 4.5540
  - after_state__拡大・繁栄: 3.1940

**負の極（Negative Pole）**:
  - after_state__分岐・様子見: -2.2234
  - action_type__分散する・独立する: -2.1889
  - before_state__成長・拡大: -2.0986
  - action_type__撤退・縮小: -1.9658
  - action_type__拡大・攻め: -1.9611

**変数別寄与（座標範囲）**:
  - trigger_type: range=14.1813
  - before_state: range=13.5124
  - action_type: range=7.2516
  - after_state: range=5.4174
  - pattern_type: range=2.5325
  - outcome: range=0.9348
  - scale: range=0.5186

### 次元 7
- **固有値**: 0.287282
- **寄与率**: 4.00%
- **解釈ラベル**: Dim 7: [_自然推移・成熟 | _撤退・縮小 | _調和・繁栄] <-> [_集中・拡大 | _分散する・独立する | _拡大・過剰]

**正の極（Positive Pole）**:
  - action_type__集中・拡大: 5.1837
  - action_type__分散する・独立する: 3.9741
  - trigger_type__拡大・過剰: 3.8908
  - action_type__撤退・逃げる: 3.6070
  - after_state__拡大・繁栄: 3.5533

**負の極（Negative Pole）**:
  - trigger_type__自然推移・成熟: -5.2050
  - action_type__撤退・縮小: -4.9532
  - before_state__調和・繁栄: -4.6223
  - action_type__逃げる・分散: -2.7055
  - pattern_type__Decline: -2.4607

**変数別寄与（座標範囲）**:
  - action_type: range=10.1369
  - trigger_type: range=9.0958
  - before_state: range=7.8027
  - after_state: range=5.0917
  - pattern_type: range=3.3961
  - outcome: range=0.7834
  - scale: range=0.4524

### 次元 8
- **固有値**: 0.283055
- **寄与率**: 3.94%
- **解釈ラベル**: Dim 8: [_拡大・過剰 | _分散する・独立する | _撤退・逃げる] <-> [_自然推移・成熟 | _調和・繁栄 | _逃げる・分散]

**正の極（Positive Pole）**:
  - trigger_type__自然推移・成熟: 9.7689
  - before_state__調和・繁栄: 8.5731
  - action_type__逃げる・分散: 3.8911
  - action_type__撤退・縮小: 3.7965
  - pattern_type__Decline: 2.2508

**負の極（Negative Pole）**:
  - trigger_type__拡大・過剰: -4.4833
  - action_type__分散する・独立する: -4.4727
  - action_type__撤退・逃げる: -4.0278
  - before_state__安定成長・成功: -1.6663
  - after_state__分岐・様子見: -1.5294

**変数別寄与（座標範囲）**:
  - trigger_type: range=14.2522
  - before_state: range=10.2394
  - action_type: range=8.3638
  - pattern_type: range=3.7151
  - after_state: range=2.6895
  - scale: range=1.0941
  - outcome: range=0.6962

### 次元 9
- **固有値**: 0.261912
- **寄与率**: 3.65%
- **解釈ラベル**: Dim 9: [_拡大・過剰 | _分散する・独立する | _撤退・逃げる] <-> [_自然推移・成熟 | _調和・繁栄 | _逃げる・分散]

**正の極（Positive Pole）**:
  - trigger_type__自然推移・成熟: 11.1414
  - before_state__調和・繁栄: 9.6822
  - action_type__逃げる・分散: 4.2717
  - action_type__分散・独立: 1.9264
  - action_type__交流・発表: 1.8244

**負の極（Negative Pole）**:
  - trigger_type__拡大・過剰: -6.7156
  - action_type__分散する・独立する: -6.3159
  - action_type__撤退・逃げる: -5.9020
  - before_state__安定成長・成功: -2.4131
  - action_type__撤退・縮小: -2.2109

**変数別寄与（座標範囲）**:
  - trigger_type: range=17.8570
  - before_state: range=12.0953
  - action_type: range=10.5876
  - after_state: range=2.6137
  - pattern_type: range=1.9545
  - outcome: range=0.9432
  - scale: range=0.7794

### 次元 10
- **固有値**: 0.251642
- **寄与率**: 3.51%
- **解釈ラベル**: Dim 10: [_自然推移・成熟 | _調和・繁栄 | _分岐・様子見] <-> [_撤退・縮小 | _Decline | _逃げる・守る]

**正の極（Positive Pole）**:
  - action_type__撤退・縮小: 9.5308
  - pattern_type__Decline: 4.9419
  - action_type__逃げる・守る: 3.6622
  - before_state__急成長・拡大: 2.0909
  - action_type__集中・拡大: 1.5985

**負の極（Negative Pole）**:
  - trigger_type__自然推移・成熟: -5.1172
  - before_state__調和・繁栄: -4.6110
  - after_state__分岐・様子見: -2.9016
  - action_type__撤退・逃げる: -2.6062
  - trigger_type__拡大・過剰: -2.2326

**変数別寄与（座標範囲）**:
  - action_type: range=12.1369
  - before_state: range=6.7019
  - pattern_type: range=6.4738
  - trigger_type: range=6.0202
  - after_state: range=3.8661
  - outcome: range=0.7955
  - scale: range=0.4307

## 5. Phase 2B（遷移分析・クラスタリング）への接続

- MCA座標空間（10次元）上で事例をクラスタリング
- before_state -> after_state の遷移行列をaction_type別に構築
- マルコフ連鎖としてのモデル適合度を検証
- クラスタが八卦の8分類や64卦と事後的に対応するかを検証（Phase 3）

## 6. 可視化一覧

- `visualizations/scree_plot.png` -- スクリープロット（全基準付き）
- `visualizations/cumulative_variance.png` -- 累積寄与率プロット
- `visualizations/mca_biplot.png` -- MCAバイプロット（Dim1 vs Dim2）

## 7. データ品質への提言

本分析で以下のデータ品質問題が確認された:

1. **カテゴリの増殖**: スキーマ定義を超えるカテゴリが存在する
   - before_state: 6 -> 15カテゴリ
   - action_type: 8 -> 22カテゴリ
   - after_state: 6 -> 20カテゴリ
2. **低頻度カテゴリ**: 件数が10未満のカテゴリが複数存在
3. **意味的重複**: 「混乱・カオス」と「混乱・衰退」等の類似カテゴリ

Phase 2Bの前にカテゴリの正規化を検討することを推奨する。

### MCA結果への影響

低頻度カテゴリ（件数 < 20）はMCA空間の端に極端に配置される傾向がある。
具体的に、次元3以降で「自然推移・成熟」(15件)、「調和・繁栄」(19件)、
「分散する・独立する」(3件) 等が座標値 7.0 以上を示しており、
これらの次元の解釈を歪めている可能性がある。

**推奨対応（Phase 2B前）**:
1. 低頻度カテゴリ（< 20件）を意味的に近い主要カテゴリに統合する
2. 統合後に再度MCAを実行し、結果を比較する
3. 低頻度カテゴリを「補足」としてMCAから除外し、
   主要カテゴリのみで次元構造を確認する（感度分析）

これにより、主要5次元の解釈の安定性が向上し、
次元3-10の解釈が低頻度カテゴリに支配される問題が解消される見込みである。

## 8. 分析の再現性

- **乱数シード**: 42
- **Pythonバージョン**: 3.12
- **princeバージョン**: 0.16.5
- **実行コマンド**: `python3 analysis/phase2/mca_analysis.py`
- **全13,060件が分析対象**（欠損値による除外: 0件）
- **八卦タグ（before_hex, trigger_hex, action_hex, after_hex）はMCAから除外**

---
*Generated by mca_analysis.py (seed=42)*