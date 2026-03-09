# ベースライン比較設計: Q6表現の予測寄与検証

## 背景

GPT-5.2批評:
> 「『同型性』を捨て、『Q6表現が予測・説明に寄与するか』に落とせ」

Phase 3結論 (ANALOGOUS) を踏まえ、Q6表現（6bit八卦コード）が予測タスクにおいて
メタデータのみのモデルを上回るかを定量的に検証する。

## ソース偏重分析の結論（前提条件）

| 指標 | 値 |
|------|-----|
| news比率 | 88.7% (10,060/11,336) |
| 全9フィールドがχ²有意 | p < 0.001 |
| 最大効果量 | country: V=0.158 (medium) |
| after_state偏り | news→V字回復が-20.0pp少ない |
| outcome偏り | news→Successが-22.1pp少ない |

**含意**: source_typeは交絡変数。全モデルでsource_typeを特徴量に含める必要がある。

---

## 予測タスク定義

### タスク1: outcome予測
- **目的変数**: outcome (Success / Failure / Mixed / PartialSuccess)
- **意味**: 変化の結果を予測する — Q6表現は「何が起こるか」に寄与するか

### タスク2: pattern_type予測
- **目的変数**: pattern_type (Pivot_Success / Slow_Decline / Shock_Recovery / Endurance / Hubris_Collapse)
- **意味**: 変化のパターンを予測する — Q6表現は「どう変わるか」に寄与するか

---

## 4モデル比較フレームワーク

### Model A: ヌルモデル（最弱ベースライン）

**目的**: 「何も学習しない」場合の精度。これを超えないモデルは無価値。

| 項目 | 内容 |
|------|------|
| 手法 | majority class prediction（最頻クラスを常に予測） |
| タスク1 | outcome分布から最頻クラスの比率 = ベースライン精度 |
| タスク2 | pattern_type分布から最頻クラスの比率 = ベースライン精度 |
| 実装 | `sklearn.dummy.DummyClassifier(strategy='most_frequent')` |
| AUC | stratified random → 0.5 |
| 意味 | 周辺分布の偏り度合いの測定 |

### Model B: 特徴量ベースライン（メタデータのみ）

**目的**: Q6表現なしで、メタデータだけでどこまで予測できるか。

| 項目 | 内容 |
|------|------|
| 特徴量 | source_type (5値), country (N値→上位10+other), main_domain (N値→上位15+other), scale (5値) |
| エンコーディング | one-hot encoding |
| 手法1 | ロジスティック回帰 (L2正則化) — `sklearn.linear_model.LogisticRegression(max_iter=1000)` |
| 手法2 | ランダムフォレスト — `sklearn.ensemble.RandomForestClassifier(n_estimators=100)` |
| 評価指標 | Accuracy, macro-F1, weighted-AUC (OvR) |
| 交差検証 | Stratified 5-fold CV |

**選定理由**: source_type偏重を定量化する。これだけで高精度なら、outcomeはソースの関数。

### Model C: Q6表現モデル（メタデータ + 八卦6bit）

**目的**: Q6表現を追加した時の改善度 (ΔAccuracy, ΔAUC) を測定。

| 項目 | 内容 |
|------|------|
| 追加特徴量 | before_lower_trigram, before_upper_trigram, after_lower_trigram, after_upper_trigram (各8値, one-hot → 32次元追加) |
| 代替表現 | before_hex, after_hex の八卦名 → 数値ID (0-7) × 4フィールド = 4次元 |
| 手法 | Model Bと同一（LR + RF）|
| 比較 | ΔAccuracy = Model C accuracy - Model B accuracy |
| 統計検定 | paired t-test on fold-level metrics (5-fold × 2 = 10ペア) |
| 帰無仮説 | H₀: ΔAccuracy = 0 (Q6表現は寄与しない) |

**判定基準**:
- ΔAccuracy > 0 かつ p < 0.05 → Q6表現は有意に寄与
- ΔAccuracy > 0 かつ p ≥ 0.05 → 寄与不明（検出力不足の可能性）
- ΔAccuracy ≤ 0 → Q6表現は寄与しない

### Model D: テキストベースライン（TF-IDF）

**目的**: Q6表現がテキスト特徴量（人間が読める説明文）を超えるか。

| 項目 | 内容 |
|------|------|
| 特徴量 | story_summary のTF-IDF (max_features=5000, sublinear_tf=True) |
| 日本語トークナイズ | MeCab or 文字3-gram (MeCab不要の場合) |
| 手法 | ロジスティック回帰 (L2) — テキスト分類の標準ベースライン |
| 比較対象 | Model D vs Model C → テキストがQ6を超えるなら、Q6は情報圧縮として不十分 |
| 追加比較 | Model D + Q6 → テキストとQ6の相補性を検証 |

**判定基準**:
- Model D > Model C → Q6はテキストの劣化圧縮
- Model D ≈ Model C → Q6はテキストと同等の情報を保持
- Model D < Model C → Q6はテキストにない構造的情報を持つ（最も強い主張）
- Model D + Q6 > Model D → Q6はテキストと相補的

---

## 実装方針

### 使用ライブラリ
```python
# 必須
scikit-learn >= 1.3    # モデル、評価、交差検証
numpy >= 1.24          # 数値計算
scipy >= 1.11          # 統計検定

# テキスト処理
scikit-learn TfidfVectorizer  # 文字n-gram（MeCab不要）

# オプション
pandas >= 2.0          # データ整形
```

### 交差検証設計
```
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
- 層化: 目的変数の分布を各foldで保持
- 固定シード: 再現性保証
- 全モデルで同一fold分割を使用（paired comparison）

### 評価方法
```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score

scoring = {
    'accuracy': 'accuracy',
    'f1_macro': make_scorer(f1_score, average='macro'),
    'auc_ovr': make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True, average='weighted'),
}
```

### 統計的比較
```python
from scipy.stats import ttest_rel

# fold-level metricsでpaired t-test
t_stat, p_value = ttest_rel(model_c_fold_scores, model_b_fold_scores)
delta = np.mean(model_c_fold_scores) - np.mean(model_b_fold_scores)
```

---

## 検出力分析

### 前提
- N = 11,336 (全事例)
- 効果量: Cohen's d = 0.1 (small)
- 有意水準: α = 0.05 (両側)
- 検定: paired t-test (5-fold CV → 5ペア)

### 問題
5-fold CVでは自由度 = 4 しかない。小さい効果量の検出には不十分。

### 対策: Repeated Stratified K-Fold
```python
from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
# → 50ペアの比較が可能
```

### 必要サンプル数の計算
```python
from scipy.stats import norm

def required_n_paired(d=0.1, alpha=0.05, power=0.80):
    """paired t-test の必要ペア数"""
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    n = ((z_alpha + z_beta) / d) ** 2
    return int(np.ceil(n))

# d=0.1 → n = 787ペア (5-fold × 158 repeats は非現実的)
# d=0.2 → n = 198ペア (5-fold × 40 repeats)
# d=0.3 → n = 88ペア  (5-fold × 18 repeats)
# d=0.5 → n = 32ペア  (5-fold × 7 repeats)
```

### 推奨設計
- **主検定**: 5-fold × 10 repeats = 50ペア → d ≥ 0.4 を検出可能 (power=0.80)
- **補助**: permutation test (10,000回) → 分布仮定不要
- **効果量報告**: Cohen's d + 95% CI を常に報告（p値だけに依存しない）

### 注意: CV foldの非独立性
- repeated k-fold のペアは完全独立ではない（データの重複）
- Nadeau & Bengio (2003) の corrected t-test を使用:
```python
def corrected_ttest(scores_a, scores_b, n_train, n_test):
    """Corrected repeated k-fold CV t-test"""
    diff = scores_a - scores_b
    mean_diff = np.mean(diff)
    var_diff = np.var(diff, ddof=1)
    n = len(diff)
    correction = 1/n + n_test/n_train
    t = mean_diff / np.sqrt(correction * var_diff)
    p = 2 * scipy_stats.t.sf(abs(t), df=n-1)
    return t, p
```

---

## 予想される結果と解釈ガイド

| シナリオ | Model A | Model B | Model C | Model D | 解釈 |
|----------|---------|---------|---------|---------|------|
| Q6は無価値 | 40% | 55% | 55% | 65% | テキストが全てでQ6は何も追加しない |
| Q6は微小寄与 | 40% | 55% | 57% | 65% | Q6はΔ2%のみ。実用上テキストで十分 |
| Q6は有用な圧縮 | 40% | 55% | 60% | 62% | Q6は4特徴量で32次元テキストの96%をカバー |
| Q6は独自情報を持つ | 40% | 55% | 60% | 58% | Q6はテキストにない構造的情報を持つ（最強主張）|
| Q6+テキスト相補 | 40% | 55% | 60% | 62% | D+Q6=66% なら両方必要 |

---

## source_type層化の必要性

ソース偏重分析の結果、全9フィールドでsource_typeとの有意な関連が確認された。
したがって:

1. **全モデルにsource_typeを含める** (Model B以降)
2. **source_typeで層化した分析も実施**: news-only / non-news-only で別々にModel C vs Bを比較
3. **交互作用項の検討**: source_type × Q6表現の交互作用が有意なら、Q6の寄与はソースに依存

---

## 実装ステップ（次タスクで実行）

1. `scripts/baseline_comparison.py` — データ前処理 + 4モデル訓練 + 評価
2. `analysis/quality/baseline_results.json` — 数値結果
3. `analysis/quality/baseline_report.md` — 解釈付きレポート

推定所要時間: 実装2-3時間、実行10-30分（テキストTF-IDF含む）
