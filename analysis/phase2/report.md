# Phase 2A: MCA分析レポート（修正版）

**分析日**: 2026-02-25
**乱数シード**: 42
**修正版**: quality-reviewer FAIL判定に基づく全面修正

---

## 1. データ概要

- **総レコード数**: 13,060 件
- **分析対象レコード数**: 13,060 件
- **欠損値による除外**: 0 件
- **分析対象変数 (K)**: 7
- **総カテゴリ数 (J)**: 88

### カテゴリ数: スキーマ定義 vs 実データ

| 変数 | スキーマ | 実データ | 増分 | 低頻度(<20件) |
|------|---------|---------|------|-------------|
| before_state | 6 | 15 | +9 | 4 |
| trigger_type | 4 | 8 | +4 | 2 |
| action_type | 8 | 22 | +14 | 8 |
| after_state | 6 | 20 | +14 | 2 |
| pattern_type | 14 | 14 | +0 | 0 |
| outcome | 4 | 4 | +0 | 0 |
| scale | 5 | 5 | +0 | 0 |

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

## 2. 全慣性（Total Inertia）の導出

### 計算式

MCAにおける全慣性は以下で定義される:

```
Total Inertia = J/K - 1
              = 88/7 - 1
              = 12.5714 - 1
              = 11.5714
```

- **prince `mca.total_inertia_`**: 11.5714
- **理論値 (J/K - 1)**: 11.5714

前回の報告値 7.179 は30成分の固有値合計であり、全慣性ではなかった。
正しい全慣性は **11.5714** である。

### 全慣性が大きい理由

J=88（総カテゴリ数）が多いため、全慣性 = J/K - 1 = 11.6 と大きくなる。
これはMCAの性質上、カテゴリ数が多いほど「説明すべき分散」が増えることを意味する。
結果として、各次元の寄与率は小さくなり、累積寄与率の上昇は緩やかになる。

## 3. MCA結果

### 固有値と寄与率（正しい全慣性ベース）

| 次元 | 固有値 | 寄与率 (%) | 累積寄与率 (%) | Benzecri修正 (%) | Benzecri累積 (%) |
|------|--------|-----------|---------------|-----------------|-----------------|
| 1 | 0.597559 | 5.16 | 5.16 | 31.14 | 31.14 |
| 2 | 0.509988 | 4.41 | 9.57 | 20.30 | 51.44 |
| 3 | 0.443109 | 3.83 | 13.40 | 13.58 | 65.02 |
| 4 | 0.393355 | 3.40 | 16.80 | 9.45 | 74.47 |
| 5 | 0.334070 | 2.89 | 19.69 | 5.51 | 79.98 |
| 6 | 0.305291 | 2.64 | 22.33 | 3.97 | 83.95 |
| 7 | 0.287282 | 2.48 | 24.81 | 3.14 | 87.10 |
| 8 | 0.283055 | 2.45 | 27.25 | 2.96 | 90.06 |
| 9 | 0.261912 | 2.26 | 29.52 | 2.13 | 92.19 |
| 10 | 0.251642 | 2.17 | 31.69 | 1.78 | 93.97 |
| 11 | 0.238821 | 2.06 | 33.76 | 1.39 | 95.36 |
| 12 | 0.227117 | 1.96 | 35.72 | 1.07 | 96.43 |
| 13 | 0.221513 | 1.91 | 37.63 | 0.93 | 97.36 |
| 14 | 0.213226 | 1.84 | 39.48 | 0.75 | 98.11 |
| 15 | 0.211753 | 1.83 | 41.31 | 0.71 | 98.82 |
| 16 | 0.191281 | 1.65 | 42.96 | 0.35 | 99.18 |
| 17 | 0.185405 | 1.60 | 44.56 | 0.27 | 99.45 |
| 18 | 0.177888 | 1.54 | 46.10 | 0.18 | 99.63 |
| 19 | 0.170226 | 1.47 | 47.57 | 0.11 | 99.75 |
| 20 | 0.167900 | 1.45 | 49.02 | 0.09 | 99.84 |

**全慣性**: 11.5714

## 4. N=13,060 における各基準の鑑別力

### Kaiser/Greenacre基準の数学的同一性

MCAにおけるGreenacre閾値とKaiser閾値は数学的に同一である:

```
Greenacre: total_inertia / (J - K) = 11.5714 / 81 = 0.1429
Kaiser:    1/K = 1/7 = 0.1429
```

導出: total_inertia = J/K - 1 なので、
total_inertia / (J - K) = (J/K - 1) / (J - K) = (J - K) / (K(J - K)) = 1/K

### 各基準の鑑別力評価

| 基準 | 次元数 | 鑑別力 | 理由 |
|------|--------|--------|------|
| スクリー（肘法） | 5 | RELIABLE | RELIABLE — primary criterion |
| 累積寄与率70% | 30 | LOW | LOW — MCA naturally has low cumulative inertia; 70% is too s |
| Kaiser (1/K) | 30 | LOST | LOST — all 30 dims exceed 1/K with N=13,060 |
| Greenacre | 30 | LOST | LOST — mathematically identical to Kaiser in MCA |
| 並行分析 | 30 | LOST | LOST — all 30 dims exceed PA threshold with N=13,060 |
| Benzecri修正(累積80%) | 6 | RELIABLE |  |

**結論**: N=13,060 + J=88 の組み合わせでは、Kaiser、Greenacre、並行分析の
3基準は全次元を「有意」と判定し、鑑別力を完全に失う。
スクリープロットとBenzecri修正慣性のみが信頼できる基準である。

## 5. Benzecri修正慣性

### 修正式

```
lambda_corrected = ((K/(K-1)) * (lambda - 1/K))^2   (lambda > 1/K のみ)
                 = ((7/6) * (lambda - 0.1429))^2
```

Benzecri修正は低頻度カテゴリが過大な固有値を持つ問題を緩和する。
1/K = 0.1429 以下の固有値は0に修正され、残る次元のみが有意とみなされる。

- 修正後に寄与率 > 0 の次元数: **30**
- 修正後の累積80%に必要な次元数: **6**

## 6. 次元数の決定

### 決定ロジック

信頼できる2基準の合意:

1. **スクリープロット（肘法）**: 5次元
2. **Benzecri修正（累積80%）**: 6次元

### 結論: データが示した次元数は **5次元** である

Scree elbow = 5, Benzecri cum80% = 6. Recommended = 5 (agreement of two reliable criteria). Kaiser (30), Greenacre (30), PA (30) all lost discriminating power with N=13,060.

> データ駆動の分析により **5次元** が最適と判定された。

## 7. 各次元の解釈（元データ）

### 次元 1
- **固有値**: 0.597559
- **寄与率**: 5.16%
- **Benzecri修正寄与率**: 31.14%
- **解釈ラベル**: Dim 1: [逃げる・放置 | Quiet_Fade | 崩壊・消滅] <-> [分散・探索 | 拡大・繁栄 | 撤退・縮小]

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
- **寄与率**: 4.41%
- **Benzecri修正寄与率**: 20.30%
- **解釈ラベル**: Dim 2: [持続成長・大成功 | Steady_Growth | 偶発・出会い] <-> [撤退・縮小 | Decline | 逃げる・放置]

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
- **寄与率**: 3.83%
- **Benzecri修正寄与率**: 13.58%
- **解釈ラベル**: Dim 3: [成長・拡大 | 拡大・攻め | 成長・拡大] <-> [自然推移・成熟 | 調和・繁栄 | 逃げる・分散]

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
- **寄与率**: 3.40%
- **Benzecri修正寄与率**: 9.45%
- **解釈ラベル**: Dim 4: [Exploration | 分散・探索 | 混乱・衰退] <-> [成長・拡大 | 拡大・攻め | 成長・拡大]

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
- **寄与率**: 2.89%
- **Benzecri修正寄与率**: 5.51%
- **解釈ラベル**: Dim 5: [交流・発表 | 喜び・交流 | 輝く・表現] <-> [自然推移・成熟 | 調和・繁栄 | 逃げる・分散]

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

## 8. 感度分析: 元データ vs クリーンデータ（低頻度カテゴリ統合）

### カテゴリ統合ルール

#### before_state

- `安定・停止` -> `安定・平和`
- `混乱・衰退` -> `混乱・カオス`
- `成長・拡大` -> `安定成長・成功`
- `拡大・繁栄` -> `安定成長・成功`
- `急成長・拡大` -> `安定成長・成功`
- `調和・繁栄` -> `安定成長・成功`
- `縮小安定・生存` -> `安定成長・成功`
- `V字回復・大成功` -> `安定成長・成功`

#### trigger_type

- `内部矛盾・自壊` -> `内部崩壊`
- `自然推移・成熟` -> `自然推移`
- `拡大・過剰` -> `内部崩壊`

#### action_type

- `拡大・攻め` -> `攻める・挑戦`
- `集中・拡大` -> `攻める・挑戦`
- `交流・発表` -> `対話・融合`
- `輝く・表現` -> `対話・融合`
- `捨てる・転換` -> `捨てる・撤退`
- `撤退・収縮` -> `捨てる・撤退`
- `撤退・縮小` -> `捨てる・撤退`
- `撤退・逃げる` -> `捨てる・撤退`
- `分散・スピンオフ` -> `分散・探索`
- `分散・独立` -> `分散・探索`
- `分散・多角化` -> `分散・探索`
- `分散する・独立する` -> `分散・探索`
- `逃げる・分散` -> `逃げる・放置`
- `逃げる・守る` -> `逃げる・放置`

#### after_state

- `V字回復・大成功` -> `持続成長・大成功`
- `成長・拡大` -> `持続成長・大成功`
- `拡大・繁栄` -> `持続成長・大成功`
- `消滅・破綻` -> `崩壊・消滅`
- `現状維持・延命` -> `縮小安定・生存`
- `安定・停止` -> `安定・平和`
- `喜び・交流` -> `安定成長・成功`
- `混乱・カオス` -> `混乱・衰退`
- `迷走・混乱` -> `混乱・衰退`
- `どん底・危機` -> `混乱・衰退`
- `分岐・様子見` -> `混乱・衰退`
- `成長痛` -> `混乱・衰退`

### 対比表

| 指標 | 元データ | クリーンデータ |
|------|---------|--------------|
| 総カテゴリ数 (J) | 88 | 51 |
| 全慣性 | 11.5714 | 6.2857 |
| Dim1 固有値 | 0.597559 | 0.500950 |
| Dim1 寄与率 | 5.16% | 7.97% |
| スクリー肘 | 5 | 2 |
| Benzecri(>0) | 30 | 16 |
| Benzecri(cum80%) | 6 | 4 |
| 推奨次元数 | **5** | **2** |

### クリーンデータの次元解釈

- Dim 1: [Breakthrough | 偶発・出会い | Steady_Growth] <-> [逃げる・放置 | Quiet_Fade | 崩壊・消滅]
- Dim 2: [崩壊・消滅 | 逃げる・放置 | Failed_Attempt] <-> [Exploration | Decline | 分散・探索]

### 差異の分析

元データ(5次元) vs クリーンデータ(2次元): 差異は低頻度カテゴリが独自の次元を形成していたことを示す。
クリーンデータの次元解釈がより安定的であり、Phase 2Bではクリーンデータの次元構造を推奨する。

## 9. Phase 2Bへの接続

- MCA座標空間（5次元）上で事例をクラスタリング
- クリーンデータ（低頻度カテゴリ統合版）を使用することを推奨
- before_state -> after_state の遷移行列をaction_type別に構築
- クラスタが八卦の8分類や64卦と事後的に対応するかを検証（Phase 3）

## 10. 可視化一覧

- `visualizations/scree_plot.png` -- スクリープロット + Benzecri修正（元データ）
- `visualizations/cumulative_variance.png` -- 累積寄与率 + Benzecri累積（元データ）
- `visualizations/mca_biplot.png` -- MCAバイプロット Dim1 vs Dim2（元データ）
- `visualizations/scree_plot_clean.png` -- スクリープロット（クリーンデータ）
- `visualizations/cumulative_variance_clean.png` -- 累積寄与率（クリーンデータ）
- `visualizations/mca_biplot_clean.png` -- MCAバイプロット（クリーンデータ）
- `visualizations/comparison.png` -- 元 vs クリーン 比較

## 11. 前回からの修正箇所

| # | 問題 | 修正内容 |
|---|------|---------|
| 1 | 全慣性の誤計算 (7.179) | `mca.total_inertia_` を使用 (11.5714) |
| 2 | Greenacre閾値の誤り | total_inertia/(J-K) = 1/K = 0.1429 (Kaiser基準と同一) |
| 3 | 次元数決定ロジック | スクリー + Benzecri修正の合意。Kaiser/Greenacre/PAの鑑別力喪失を明示 |
| 4 | 低頻度カテゴリの感度分析なし | クリーンデータ(カテゴリ統合版)で再MCA、対比表を提示 |
| 5 | 絶対パス | `os.path.dirname(__file__)` による相対パスに変更 |
| 6 | prince属性の未活用 | `percentage_of_variance_`, `total_inertia_` 等を使用 |

---
*Generated by mca_analysis.py (seed=42, revised 2026-02-25)*