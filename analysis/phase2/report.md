# Phase 2A: MCA分析レポート

**分析日**: 2026-02-25
**乱数シード**: 42

---

## 1. データ概要

- **総レコード数**: 13,060 件
- **分析対象レコード数**: 13,060 件
- **欠損値による除外**: 0 件

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

### 分析対象変数（八卦タグを除外）

- `before_state` (6カテゴリ)
- `trigger_type` (4カテゴリ)
- `action_type` (8カテゴリ)
- `after_state` (6カテゴリ)
- `pattern_type` (14カテゴリ)
- `outcome` (4カテゴリ)
- `scale` (5カテゴリ)

### 固有値と寄与率

| 次元 | 固有値 | 寄与率 (%) | 累積寄与率 (%) |
|------|--------|-----------|---------------|
| 1 | 0.597559 | 10.54 | 10.54 |
| 2 | 0.509988 | 9.00 | 19.53 |
| 3 | 0.443109 | 7.82 | 27.35 |
| 4 | 0.393355 | 6.94 | 34.29 |
| 5 | 0.334070 | 5.89 | 40.18 |
| 6 | 0.305291 | 5.38 | 45.56 |
| 7 | 0.287282 | 5.07 | 50.63 |
| 8 | 0.283055 | 4.99 | 55.62 |
| 9 | 0.261912 | 4.62 | 60.24 |
| 10 | 0.251642 | 4.44 | 64.68 |
| 11 | 0.238820 | 4.21 | 68.89 |
| 12 | 0.227111 | 4.01 | 72.90 |
| 13 | 0.221504 | 3.91 | 76.81 |
| 14 | 0.213223 | 3.76 | 80.57 |
| 15 | 0.211708 | 3.73 | 84.30 |
| 16 | 0.191171 | 3.37 | 87.67 |
| 17 | 0.184957 | 3.26 | 90.94 |
| 18 | 0.176327 | 3.11 | 94.05 |
| 19 | 0.170131 | 3.00 | 97.05 |
| 20 | 0.167463 | 2.95 | 100.00 |

## 3. 次元数の決定

### 4つの基準による判定

1. **スクリープロット（肘法）**: 5次元
   - スクリープロット（肘法）: 固有値の二次差分の最大変化点
2. **累積寄与率 >= 70%**: 12次元
   - 累積寄与率 >= 70%
3. **Kaiser基準**: 20次元
   - Kaiser基準（固有値 > 1/7 = 0.1429）
4. **並行分析**: 20次元
   - 並行分析（100回ランダム置換の95パーセンタイル比較）

### 結論: データが示した次元数は **16次元** である

各基準の結果: [5, 12, 20, 20]
中央値による合意: **16次元**

> データ駆動の分析により、16次元が最適と判定された。
> これは先験的な6次元仮説とは異なる結果である。

## 4. 各次元の解釈

### 次元 1
- **固有値**: 0.597559
- **寄与率**: 10.54%
- **解釈ラベル**: Axis 1: [action_type__逃げる・放置 / pattern_type__Quiet_Fade / after_state__崩壊・消滅] ⟷ [action_type__分散・探索 / before_state__拡大・繁栄 / action_type__撤退・縮小]

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

### 次元 2
- **固有値**: 0.509988
- **寄与率**: 9.00%
- **解釈ラベル**: Axis 2: [after_state__持続成長・大成功 / pattern_type__Steady_Growth / trigger_type__偶発・出会い] ⟷ [action_type__撤退・縮小 / pattern_type__Decline / action_type__逃げる・放置]

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

### 次元 3
- **固有値**: 0.443109
- **寄与率**: 7.82%
- **解釈ラベル**: Axis 3: [before_state__成長・拡大 / action_type__拡大・攻め / after_state__成長・拡大] ⟷ [trigger_type__自然推移・成熟 / before_state__調和・繁栄 / action_type__逃げる・分散]

**正の極（Positive Pole）**:
  - trigger_type__自然推移・成熟: 7.1434
  - before_state__調和・繁栄: 6.8817
  - action_type__逃げる・分散: 4.7236
  - before_state__急成長・拡大: 3.6461
  - action_type__撤退・逃げる: 3.4235

**負の極（Negative Pole）**:
  - before_state__成長・拡大: -5.4943
  - action_type__拡大・攻め: -5.0676
  - after_state__成長・拡大: -4.9772
  - after_state__分岐・様子見: -3.1949
  - action_type__集中・拡大: -1.6612

### 次元 4
- **固有値**: 0.393355
- **寄与率**: 6.94%
- **解釈ラベル**: Axis 4: [pattern_type__Exploration / action_type__分散・探索 / after_state__混乱・衰退] ⟷ [before_state__成長・拡大 / action_type__拡大・攻め / after_state__成長・拡大]

**正の極（Positive Pole）**:
  - before_state__成長・拡大: 8.3364
  - action_type__拡大・攻め: 7.7643
  - after_state__成長・拡大: 7.5867
  - trigger_type__自然推移・成熟: 4.7219
  - before_state__調和・繁栄: 4.4320

**負の極（Negative Pole）**:
  - pattern_type__Exploration: -2.4741
  - action_type__分散・探索: -2.1728
  - after_state__混乱・衰退: -1.7158
  - before_state__安定・停止: -1.3553
  - before_state__拡大・繁栄: -1.2179

### 次元 5
- **固有値**: 0.334070
- **寄与率**: 5.89%
- **解釈ラベル**: Axis 5: [action_type__交流・発表 / after_state__喜び・交流 / action_type__輝く・表現] ⟷ [trigger_type__自然推移・成熟 / before_state__調和・繁栄 / action_type__逃げる・分散]

**正の極（Positive Pole）**:
  - trigger_type__自然推移・成熟: 7.8623
  - before_state__調和・繁栄: 7.3410
  - action_type__逃げる・分散: 3.1317
  - action_type__撤退・逃げる: 2.8407
  - after_state__分岐・様子見: 2.8039

**負の極（Negative Pole）**:
  - action_type__交流・発表: -6.2466
  - after_state__喜び・交流: -6.0117
  - action_type__輝く・表現: -5.3627
  - pattern_type__Crisis_Pivot: -1.3363
  - action_type__集中・拡大: -1.3090

### 次元 6
- **固有値**: 0.305291
- **寄与率**: 5.38%
- **解釈ラベル**: Axis 6: [after_state__分岐・様子見 / action_type__分散する・独立する / before_state__成長・拡大] ⟷ [trigger_type__自然推移・成熟 / before_state__調和・繁栄 / action_type__逃げる・分散]

**正の極（Positive Pole）**:
  - trigger_type__自然推移・成熟: 12.5988
  - before_state__調和・繁栄: 11.4140
  - action_type__逃げる・分散: 5.0599
  - action_type__集中・拡大: 4.5537
  - after_state__拡大・繁栄: 3.1942

**負の極（Negative Pole）**:
  - after_state__分岐・様子見: -2.2163
  - action_type__分散する・独立する: -2.1679
  - before_state__成長・拡大: -2.0986
  - action_type__撤退・縮小: -1.9663
  - action_type__拡大・攻め: -1.9609

### 次元 7
- **固有値**: 0.287282
- **寄与率**: 5.07%
- **解釈ラベル**: Axis 7: [trigger_type__自然推移・成熟 / action_type__撤退・縮小 / before_state__調和・繁栄] ⟷ [action_type__集中・拡大 / action_type__分散する・独立する / trigger_type__拡大・過剰]

**正の極（Positive Pole）**:
  - action_type__集中・拡大: 5.1839
  - action_type__分散する・独立する: 3.9656
  - trigger_type__拡大・過剰: 3.8897
  - action_type__撤退・逃げる: 3.6078
  - after_state__拡大・繁栄: 3.5533

**負の極（Negative Pole）**:
  - trigger_type__自然推移・成熟: -5.2039
  - action_type__撤退・縮小: -4.9519
  - before_state__調和・繁栄: -4.6212
  - action_type__逃げる・分散: -2.7137
  - pattern_type__Decline: -2.4603

### 次元 8
- **固有値**: 0.283055
- **寄与率**: 4.99%
- **解釈ラベル**: Axis 8: [trigger_type__拡大・過剰 / action_type__分散する・独立する / action_type__撤退・逃げる] ⟷ [trigger_type__自然推移・成熟 / before_state__調和・繁栄 / action_type__逃げる・分散]

**正の極（Positive Pole）**:
  - trigger_type__自然推移・成熟: 9.7697
  - before_state__調和・繁栄: 8.5739
  - action_type__逃げる・分散: 3.8852
  - action_type__撤退・縮小: 3.8005
  - pattern_type__Decline: 2.2509

**負の極（Negative Pole）**:
  - trigger_type__拡大・過剰: -4.4848
  - action_type__分散する・独立する: -4.4839
  - action_type__撤退・逃げる: -4.0280
  - before_state__安定成長・成功: -1.6660
  - after_state__分岐・様子見: -1.5216

### 次元 9
- **固有値**: 0.261912
- **寄与率**: 4.62%
- **解釈ラベル**: Axis 9: [trigger_type__拡大・過剰 / action_type__分散する・独立する / action_type__撤退・逃げる] ⟷ [trigger_type__自然推移・成熟 / before_state__調和・繁栄 / action_type__逃げる・分散]

**正の極（Positive Pole）**:
  - trigger_type__自然推移・成熟: 11.1416
  - before_state__調和・繁栄: 9.6824
  - action_type__逃げる・分散: 4.2703
  - action_type__分散・独立: 1.9105
  - action_type__交流・発表: 1.8236

**負の極（Negative Pole）**:
  - trigger_type__拡大・過剰: -6.7206
  - action_type__分散する・独立する: -6.3903
  - action_type__撤退・逃げる: -5.8878
  - before_state__安定成長・成功: -2.4126
  - action_type__撤退・縮小: -2.2187

### 次元 10
- **固有値**: 0.251642
- **寄与率**: 4.44%
- **解釈ラベル**: Axis 10: [trigger_type__自然推移・成熟 / before_state__調和・繁栄 / after_state__分岐・様子見] ⟷ [action_type__撤退・縮小 / pattern_type__Decline / action_type__逃げる・守る]

**正の極（Positive Pole）**:
  - action_type__撤退・縮小: 9.5326
  - pattern_type__Decline: 4.9410
  - action_type__逃げる・守る: 3.6544
  - before_state__急成長・拡大: 2.0300
  - action_type__集中・拡大: 1.5977

**負の極（Negative Pole）**:
  - trigger_type__自然推移・成熟: -5.1196
  - before_state__調和・繁栄: -4.6132
  - after_state__分岐・様子見: -2.9267
  - action_type__撤退・逃げる: -2.6091
  - trigger_type__拡大・過剰: -2.2322

### 次元 11
- **固有値**: 0.238820
- **寄与率**: 4.21%
- **解釈ラベル**: Axis 11: [action_type__分散・独立 / after_state__成長痛 / after_state__現状維持・延命] ⟷ [action_type__分散する・独立する / trigger_type__拡大・過剰 / pattern_type__Quiet_Fade]

**正の極（Positive Pole）**:
  - action_type__分散する・独立する: 2.8475
  - trigger_type__拡大・過剰: 2.8091
  - pattern_type__Quiet_Fade: 2.4303
  - action_type__撤退・逃げる: 2.2255
  - before_state__急成長・拡大: 1.8378

**負の極（Negative Pole）**:
  - action_type__分散・独立: -2.5706
  - after_state__成長痛: -2.3878
  - after_state__現状維持・延命: -1.9048
  - trigger_type__自然推移・成熟: -1.4781
  - before_state__調和・繁栄: -1.1432

### 次元 12
- **固有値**: 0.227111
- **寄与率**: 4.01%
- **解釈ラベル**: Axis 12: [pattern_type__Failed_Attempt / action_type__逃げる・守る / after_state__消滅・破綻] ⟷ [trigger_type__自然推移・成熟 / before_state__調和・繁栄 / action_type__逃げる・分散]

**正の極（Positive Pole）**:
  - trigger_type__自然推移・成熟: 7.1857
  - before_state__調和・繁栄: 6.1082
  - action_type__逃げる・分散: 2.0397
  - action_type__分散する・独立する: 1.9052
  - after_state__現状維持・延命: 1.3862

**負の極（Negative Pole）**:
  - pattern_type__Failed_Attempt: -2.1937
  - action_type__逃げる・守る: -1.7765
  - after_state__消滅・破綻: -1.6555
  - action_type__撤退・縮小: -1.5187
  - before_state__V字回復・大成功: -1.4592

### 次元 13
- **固有値**: 0.221504
- **寄与率**: 3.91%
- **解釈ラベル**: Axis 13: [action_type__撤退・縮小 / after_state__現状維持・延命 / action_type__撤退・収縮] ⟷ [action_type__分散する・独立する / trigger_type__拡大・過剰 / action_type__撤退・逃げる]

**正の極（Positive Pole）**:
  - action_type__分散する・独立する: 5.7217
  - trigger_type__拡大・過剰: 5.2988
  - action_type__撤退・逃げる: 5.2009
  - trigger_type__自然推移・成熟: 3.4673
  - before_state__調和・繁栄: 2.9459

**負の極（Negative Pole）**:
  - action_type__撤退・縮小: -3.0087
  - after_state__現状維持・延命: -2.2824
  - action_type__撤退・収縮: -1.9989
  - before_state__拡大・繁栄: -1.7044
  - action_type__集中・拡大: -1.6564

### 次元 14
- **固有値**: 0.213223
- **寄与率**: 3.76%
- **解釈ラベル**: Axis 14: [action_type__逃げる・守る / action_type__分散・独立 / after_state__成長痛] ⟷ [action_type__分散する・独立する / trigger_type__拡大・過剰 / action_type__撤退・逃げる]

**正の極（Positive Pole）**:
  - action_type__分散する・独立する: 11.0824
  - trigger_type__拡大・過剰: 10.9970
  - action_type__撤退・逃げる: 9.4580
  - trigger_type__自然推移・成熟: 5.6950
  - before_state__調和・繁栄: 4.9274

**負の極（Negative Pole）**:
  - action_type__逃げる・守る: -2.7368
  - action_type__分散・独立: -2.6593
  - after_state__成長痛: -2.2325
  - after_state__現状維持・延命: -1.5866
  - before_state__V字回復・大成功: -1.5647

### 次元 15
- **固有値**: 0.211708
- **寄与率**: 3.73%
- **解釈ラベル**: Axis 15: [before_state__急成長・拡大 / action_type__逃げる・守る / after_state__安定・停止] ⟷ [trigger_type__拡大・過剰 / action_type__分散する・独立する / action_type__撤退・逃げる]

**正の極（Positive Pole）**:
  - trigger_type__拡大・過剰: 11.3642
  - action_type__分散する・独立する: 10.7287
  - action_type__撤退・逃げる: 9.5449
  - trigger_type__自然推移・成熟: 8.7067
  - before_state__調和・繁栄: 7.2576

**負の極（Negative Pole）**:
  - before_state__急成長・拡大: -7.5108
  - action_type__逃げる・守る: -5.7701
  - after_state__安定・停止: -1.8600
  - trigger_type__内部矛盾・自壊: -1.5165
  - pattern_type__Stagnation: -1.2446

### 次元 16
- **固有値**: 0.191171
- **寄与率**: 3.37%
- **解釈ラベル**: Axis 16: [trigger_type__自然推移・成熟 / before_state__調和・繁栄 / action_type__分散・多角化] ⟷ [action_type__撤退・逃げる / trigger_type__拡大・過剰 / action_type__分散する・独立する]

**正の極（Positive Pole）**:
  - action_type__撤退・逃げる: 17.3638
  - trigger_type__拡大・過剰: 15.2700
  - action_type__分散する・独立する: 9.0897
  - pattern_type__Managed_Decline: 1.9610
  - action_type__逃げる・守る: 1.7737

**負の極（Negative Pole）**:
  - trigger_type__自然推移・成熟: -3.9976
  - before_state__調和・繁栄: -3.2998
  - action_type__分散・多角化: -2.8241
  - action_type__逃げる・分散: -2.0557
  - action_type__撤退・収縮: -1.5390

## 5. Phase 2B（遷移分析・クラスタリング）への接続

- MCA座標空間（16次元）上での事例のクラスタリングを実施
- before_state → after_state の遷移行列をaction_type別に構築
- マルコフ連鎖としてのモデル適合度を検証
- クラスタが八卦の8分類や64卦と事後的に対応するかを検証

## 6. 可視化一覧

- `visualizations/scree_plot.png` — スクリープロット
- `visualizations/cumulative_variance.png` — 累積寄与率プロット
- `visualizations/mca_biplot.png` — MCAバイプロット（Dim1 vs Dim2）

---
*Generated by mca_analysis.py (seed=42)*