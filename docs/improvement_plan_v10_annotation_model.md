# 六十四卦マッピング改善計画v10 - 注釈モデル+64分布直学習版

## v9からの変更点

| v9 | v10 |
|----|-----|
| 6爻独立分解 | **64卦分布を直接学習** |
| `prior(y)`を掛ける結合式 | **注釈モデル`q(z|annotations)`を教師分布** |
| 目標値なし | **ベースライン比+停止条件を明示** |
| `w(r)=exp(β(r-3))`で集約 | **注釈者モデルで集約（混同行列学習）** |

## 設計思想

### 問題定義（維持）
> 「注釈者分布（soft label）を予測する」→ 多義性を認めた分布予測問題

### 核心の修正
> 独立分解は「相関した不確実性」を表現できない → **64卦同時分布を直接学習**

## 核心設計

### 1. 注釈モデル（ラベル生成）

**注釈者の評点から教師分布`q(z|annotations)`を推定**

```python
class AnnotationModel:
    """
    潜在変数zを置き、各注釈者の評点をモデル化
    """
    def __init__(self, n_annotators, n_hexagrams=64):
        # 各注釈者の「見え方」を混同行列で表現
        # 簡約版: 注釈者ごとの温度パラメータ + バイアス
        self.annotator_temperature = {}  # 評点のばらつき
        self.annotator_bias = {}  # 系統的バイアス

    def fit(self, annotations):
        """
        annotations: Dict[case_id, Dict[annotator_id, Dict[hexagram_id, score]]]
        EMアルゴリズムで推定:
        - E-step: P(z|annotations, θ) を計算
        - M-step: θ（温度・バイアス）を更新
        """
        pass

    def get_soft_label(self, case_id):
        """事後分布 q(z|annotations) を返す"""
        return self.posterior[case_id]  # 64次元確率ベクトル
```

**単純`w(r)`との違い**:
- 注釈者ごとの系統的バイアスを補正
- 評点の温度（集中度）を注釈者ごとに推定
- 「特定2〜3卦に集中する」相関パターンを保持

### 2. 予測モデル（64分布直学習）

**注釈モデルの事後分布を教師として64クラス交差エントロピー**

```python
class HexagramPredictor:
    """
    64卦の同時分布を直接予測
    """
    def __init__(self, encoder):
        self.encoder = encoder  # テキスト→埋め込み
        self.head = nn.Linear(hidden_dim, 64)  # 64クラス分類ヘッド

    def forward(self, x):
        h = self.encoder(x)
        logits = self.head(h)
        return F.softmax(logits, dim=-1)

    def loss(self, x, soft_label):
        """
        soft_label: 注釈モデルからの事後分布 q(z|annotations)
        交差エントロピー: -Σ q(z) log p(z|x)
        """
        pred = self.forward(x)
        return -torch.sum(soft_label * torch.log(pred + 1e-10))
```

**6爻独立との違い**:
- 「卦Aか卦B」という2峰性を直接表現可能
- 相関した不確実性を学習
- スケールによる上限がない

### 3. 構造化オプション（フォールバック用）

```
直学習64クラスが難しい場合:
  ↓
上卦/下卦の階層（8×8）
  - P(上卦|x) × P(下卦|x, 上卦)
  - 上卦を条件にすることで相関を表現
  - 独立積ではないので整合性あり
```

### 4. 意思決定（拒否判定）

**`c_u`は運用要件から固定、調整は意思決定者が行う**

```python
def decide_action(probs, c_u):
    """
    c_u: 人に聞くコスト（運用要件から固定）
    """
    # 最小期待コスト
    L_auto = compute_expected_cost(probs)

    if L_auto <= c_u:
        return 'auto', argmin_expected_cost(probs)
    else:
        return 'ask', get_top_k(probs, k=5)

def generate_coverage_risk_curve(predictions, labels, c_u_candidates):
    """
    意思決定者に提示するcoverage-risk曲線を生成
    """
    results = []
    for c_u in c_u_candidates:
        auto_count = 0
        total_risk = 0
        for pred, label in zip(predictions, labels):
            action, result = decide_action(pred, c_u)
            if action == 'auto':
                auto_count += 1
                total_risk += expected_cost_auto(pred, label)

        coverage = auto_count / len(predictions)
        risk = total_risk / auto_count if auto_count > 0 else 0
        results.append({'c_u': c_u, 'coverage': coverage, 'risk': risk})

    return results  # 意思決定者がトレードオフを見て選ぶ
```

### 5. データ分割（リーク遮断）

v9と同様だが、注釈モデルの学習も含める:

```
全300件
    ↓
[Test: 60件] ← ロック（最終評価のみ）
    ↓
[Train+Val: 240件]
    ↓
    ┌─ 注釈モデル学習（240件全体）
    │   → 教師分布 q(z|annotations) を生成
    │   → テストは60件も事前に計算しておくが使わない
    ↓
[K-fold クロスフィット (K=5)]
    Fold 1: Train 192件 → Val 48件 → OOF予測
    ...
    ↓
[OOF全体 (240件)]
    - 確率校正（温度スケーリング）
    ↓
[最終モデル]
    240件全体で再学習
    ↓
[Test評価]
    60件に1回だけ適用
```

### 6. 評価設計

**ベースラインを必須、停止条件を明示**

| ベースライン | 説明 |
|-------------|------|
| Prior only | `P(z) = 注釈分布の周辺` |
| 単純64クラス | `w(r)`正規化をそのまま使用 |
| 上卦/下卦独立 | `P(上卦|x) × P(下卦|x)` |

**停止条件**:
```python
def should_stop(metrics, baselines):
    """
    全ベースラインに勝てない場合は設計を再検討
    """
    if metrics['NLL'] > baselines['prior_only']['NLL']:
        return True, "Priorより悪い - 特徴量が効いていない"
    if metrics['NLL'] > baselines['simple_64']['NLL']:
        return True, "単純64クラスより悪い - モデル構造が不適切"
    return False, None
```

**評価指標**:
| 指標 | 定義 | 意味 |
|------|------|------|
| NLL | 負の対数尤度 | 分布予測の品質 |
| Brier Score | 確率予測の二乗誤差 | 校正品質 |
| Selective Risk | 拒否込みのリスク | 意思決定品質 |

**不確実性の報告**:
```python
def report_with_uncertainty(test_metrics, n_bootstrap=1000):
    """
    テスト60件の分散を区間で報告
    """
    bootstrap_results = []
    for _ in range(n_bootstrap):
        sample = resample(test_metrics)
        bootstrap_results.append(compute_aggregate(sample))

    return {
        'mean': np.mean(bootstrap_results),
        'ci_95': (np.percentile(bootstrap_results, 2.5),
                  np.percentile(bootstrap_results, 97.5))
    }
```

## 工数見積もり

| フェーズ | 内容 | 工数 |
|----------|------|------|
| **Phase 1: 設計** | | **40h** |
| - 注釈モデル設計 | EM推定、混同行列 | 16h |
| - 64分布モデル設計 | アーキテクチャ | 12h |
| - 評価パイプライン | ベースライン、CI | 12h |
| **Phase 2: パイロット** | | **60h** |
| - 注釈者トレーニング | 3名×4h | 12h |
| - パイロット注釈 | 50件×3名 | 20h |
| - 注釈モデル検証 | EMの収束確認 | 16h |
| - ガイド修正 | | 12h |
| **Phase 3: ゴールドセット** | | **100h** |
| - 本注釈 | 250件×3名 | 60h |
| - 注釈モデル学習 | EM実行 | 16h |
| - soft label生成 | 300件 | 8h |
| - 品質検証 | Krippendorff's α | 16h |
| **Phase 4: モデル構築** | | **60h** |
| - 64分布モデル実装 | | 20h |
| - クロスフィット実行 | OOF生成、校正 | 16h |
| - ベースライン比較 | 停止条件チェック | 12h |
| - Test評価 | CI付き報告 | 12h |
| **合計** | | **260h** |

## MVP（最小実行可能設計）

1. **注釈モデル**: 単純化版（注釈者バイアスのみ、温度は固定）
2. **予測モデル**: 64クラス分類ヘッド（構造化なし）
3. **拒否判定**: coverage-risk曲線を出し、ユーザーが選択
4. **評価**: NLL + 3ベースライン比較 + 95%CI

## フォールバック

| 条件 | 対応 |
|------|------|
| 注釈モデルのEM収束しない | 単純`w(r)`集約に戻す |
| 64クラス精度がPrior以下 | 上卦/下卦階層に変更 |
| テスト60件のCI幅が広すぎる | パイロットとして扱い、ゴールド拡張 |
| 注釈者間α<0.3 | 八卦（8クラス）に簡素化 |
