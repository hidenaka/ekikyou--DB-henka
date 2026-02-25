# Phase 2 仕様書: 変化事例の構造化と次元導出

## 目的
13,060件の変化事例データから、**事前定義なし**で変化の主要軸を導出する。6次元に押し込めない — データが示す次元数に従う。

## 背景
前回のアプローチでは6次元（安定/方向/内的/外的/自覚/準備）を先験的に定義し、81%の予測精度を得たが「常識の追認」の懸念があった。今回はボトムアップで、データ自体が示す変化の構造を発見する。

## データソース
- `data/raw/cases.jsonl` — 13,060件（Read Only）
- スキーマ: `docs/schema_v4.md`

## 分析対象のカテゴリカル変数

| 変数名 | カテゴリ数 | 値 |
|--------|-----------|-----|
| `before_state` | 6 | 絶頂・慢心 / 停滞・閉塞 / 混乱・カオス / 成長痛 / どん底・危機 / 安定・平和 |
| `trigger_type` | 4 | 外部ショック / 内部崩壊 / 意図的決断 / 偶発・出会い |
| `action_type` | 8 | 攻める・挑戦 / 守る・維持 / 捨てる・撤退 / 耐える・潜伏 / 対話・融合 / 刷新・破壊 / 逃げる・放置 / 分散・スピンオフ |
| `after_state` | 6 | V字回復・大成功 / 縮小安定・生存 / 変質・新生 / 現状維持・延命 / 迷走・混乱 / 崩壊・消滅 |
| `before_hex` | 8 | 乾/坤/震/巽/坎/離/艮/兌 |
| `trigger_hex` | 8 | 同上 |
| `action_hex` | 8 | 同上 |
| `after_hex` | 8 | 同上 |
| `pattern_type` | 14 | Steady_Growth / Slow_Decline / ... |
| `outcome` | 4 | Success / PartialSuccess / Failure / Mixed |
| `scale` | 5 | company / individual / family / country / other |

## 成果物

### Phase 2A: データ構造化と探索的分析

#### 1. 基本統計 (`analysis/phase2/basic_stats.json`)
- 各カテゴリカル変数の度数分布
- クロス集計表（before_state × action_type × after_state）
- 欠損値レポート

#### 2. MCA分析 (`analysis/phase2/mca_analysis.py`)

```python
# 多重対応分析（Multiple Correspondence Analysis）
# princeライブラリを使用
import prince
import pandas as pd

# 分析手順:
# 1. カテゴリカル変数を選択（八卦タグを除外した基本変数のみ）
# 2. MCAを実行
# 3. 固有値と寄与率を計算
# 4. スクリープロットで次元数を決定
# 5. 各次元の解釈（変数ごとの寄与度から）

# 重要: 八卦タグは分析に含めない（Phase 3で事後的に検証するため）
```

**次元数の決定基準**:
- スクリープロット（肘法）
- 累積寄与率 70%以上
- Kaiser基準（固有値 > 1/変数数）
- 並行分析（Parallel Analysis）によるランダム比較

**6次元に収束した場合**: それ自体が強力な証拠。ただし偶然の可能性を帰無仮説検定で排除する。
**6次元に収束しなかった場合**: 実際の次元数を報告。それ自体が有意な結果。

### Phase 2B: 遷移パターン分析

#### 3. 状態遷移行列 (`analysis/phase2/transition_matrix.py`)

```python
# before_state × after_state の遷移行列を構築
# 条件付き: action_type別、trigger_type別

# 分析項目:
# 1. 遷移確率の非一様性検定（カイ二乗検定）
# 2. 頻出遷移パターンの抽出
# 3. マルコフ連鎖としてのモデル適合度
# 4. 定常分布の計算
# 5. 吸収状態の検出
```

#### 4. クラスタリング (`analysis/phase2/clustering.py`)

```python
# 事例をMCA座標空間上でクラスタリング
# k-means, DBSCAN, 階層的クラスタリングを比較
# クラスタ数の決定: シルエットスコア, エルボー法

# 重要: クラスタが八卦の8分類や64卦と対応するかを事後的に検証
```

### 出力ファイル

| ファイル | 内容 |
|---------|------|
| `analysis/phase2/basic_stats.json` | 基本統計 |
| `analysis/phase2/mca_analysis.py` | MCA分析スクリプト |
| `analysis/phase2/mca_results.json` | MCA結果（固有値、寄与率、座標） |
| `analysis/phase2/dimension_report.json` | 導出された次元の解釈 |
| `analysis/phase2/transition_matrix.py` | 遷移行列スクリプト |
| `analysis/phase2/transition_stats.json` | 遷移統計 |
| `analysis/phase2/clustering.py` | クラスタリングスクリプト |
| `analysis/phase2/cluster_results.json` | クラスタリング結果 |
| `analysis/phase2/visualizations/` | 可視化 |
| `analysis/phase2/report.md` | Phase 2レポート |

### 可視化

- `scree_plot.png` — スクリープロット（次元数決定）
- `mca_biplot.png` — MCAバイプロット（変数とカテゴリの配置）
- `transition_heatmap.png` — 遷移行列ヒートマップ
- `cluster_map.png` — クラスタリング結果
- `parallel_coordinates.png` — パラレル座標プロット（主要パターン）

## 検証基準
- [ ] 全13,060件が分析に含まれている（欠損値の扱いを明示）
- [ ] MCAの固有値と寄与率を報告
- [ ] 次元数の決定根拠を3つ以上の基準で示す
- [ ] 八卦タグをMCA分析から除外している（Phase 3での事後検証のため）
- [ ] 遷移行列の行和が正しい
- [ ] クラスタ数の決定根拠を明示
- [ ] 全スクリプトが再現可能（乱数シード固定）
- [ ] レポートに「データが示した次元数」を明記

## 制約
- princeライブラリが未インストールの場合: `pip install prince` を実行
- MCAが収束しない場合: SVDベースの実装にフォールバック
- メモリ制約: 13,060件は一括処理可能なサイズ
