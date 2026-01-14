# 六十四卦マッピング改善計画v11 - 共同学習+構造化主設計版

## v10からの変更点

| v10 | v11 |
|----|-----|
| 注釈モデル→予測モデルの直列分離 | **共同学習（crowd-layer方式）** |
| 64×64フル混同行列 | **疎構造混同行列（近傍のみ）** |
| ゴールド/アンカーなし | **ゴールドサンプル必須** |
| 構造化はフォールバック | **上卦/下卦階層を主設計** |
| coverage-risk曲線のみ | **確率校正 + conformal prediction** |

## 設計思想

### 問題定義（維持）
> 「注釈者分布（soft label）を予測する」→ 多義性を認めた分布予測問題

### 核心の修正
> 直列分離は誤差伝播する → **注釈ノイズをモデル内に組み込み共同学習**

## 核心設計

### 1. ゴールドサンプル（識別性確保）

**EMの識別不能問題を解決するため、少量のゴールドを導入**

```python
class GoldSamples:
    """
    専門家合議で確定したアンカーサンプル
    """
    def __init__(self, n_gold=30):
        self.n_gold = n_gold  # 全300件中30件（10%）

    def select(self, cases):
        """
        ゴールド候補の選定基準:
        - 64卦が均等に出現するよう層化サンプリング
        - 各卦に最低1件（多い卦は複数）
        - 極端に難しいケースは除外
        """
        pass

    def adjudicate(self, case, annotations):
        """
        3名合議 → 意見が割れたら議論 → 最終決定
        ゴールドは単一ラベル（確率1.0）
        """
        pass
```

**識別性の保証**:
- ゴールドサンプルがあれば注釈者バイアスと真値を分離可能
- EMの収束先が一意に定まる

### 2. 疎構造混同行列（パラメータ削減）

**64×64フル行列は不可能 → 近傍混同のみモデル化**

```python
class SparseConfusionModel:
    """
    混同は「似た卦」間でしか起きないと仮定
    """
    def __init__(self, neighborhood_size=8):
        # 各卦について、混同しやすい近傍k卦のみモデル化
        # パラメータ数: 64 × 8 = 512（フル64×64=4096の1/8）
        self.neighborhood = self._compute_neighbors()

    def _compute_neighbors(self):
        """
        近傍の定義:
        - 同じ上卦を持つ卦（8卦）
        - 同じ下卦を持つ卦（8卦）
        - 1爻違いの卦（6卦）
        → 重複除去で約10-15卦
        """
        pass

    def confusion_prob(self, annotator, true_hex, observed_hex):
        """
        P(observed|true, annotator)
        近傍外は確率ε（小さな定数）
        """
        if observed_hex in self.neighborhood[true_hex]:
            return self.annotator_params[annotator][true_hex][observed_hex]
        else:
            return self.epsilon  # 遠い卦への混同は稀
```

### 3. 共同学習（Crowd-Layer方式）

**注釈ノイズをモデル内に組み込み、End-to-Endで学習**

```python
class CrowdHexagramModel(nn.Module):
    """
    予測器と注釈者ノイズを同時学習
    """
    def __init__(self, encoder, n_annotators):
        super().__init__()
        self.encoder = encoder
        self.hexagram_head = HierarchicalHead()  # 上卦/下卦階層

        # 注釈者ノイズ層（crowd-layer）
        self.annotator_confusion = nn.ParameterList([
            SparseConfusionMatrix() for _ in range(n_annotators)
        ])

    def forward(self, x, annotator_id=None):
        """
        推論時: 純粋な卦分布を出力
        学習時: 注釈者の混同を通した尤度を計算
        """
        h = self.encoder(x)
        p_hex = self.hexagram_head(h)  # P(z|x)

        if annotator_id is None:
            return p_hex  # 推論
        else:
            # 学習: 注釈を生成する確率
            confusion = self.annotator_confusion[annotator_id]
            return torch.matmul(p_hex, confusion)  # P(annot|x)

    def loss(self, x, annotations):
        """
        annotations: Dict[annotator_id, hexagram_id]
        各注釈者の観測尤度を最大化
        """
        total_loss = 0
        for annotator_id, observed_hex in annotations.items():
            p_annot = self.forward(x, annotator_id)
            total_loss -= torch.log(p_annot[observed_hex] + 1e-10)
        return total_loss

    def loss_with_gold(self, x, gold_label):
        """
        ゴールドサンプル: 直接交差エントロピー
        """
        p_hex = self.forward(x)
        return F.cross_entropy(p_hex.unsqueeze(0),
                              torch.tensor([gold_label]))
```

### 4. 上卦/下卦階層（主設計）

**構造化をフォールバックではなく主設計に**

```python
class HierarchicalHead(nn.Module):
    """
    64卦 = 上卦(8) × 下卦(8)
    P(卦|x) = P(上卦|x) × P(下卦|x, 上卦)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.upper_head = nn.Linear(hidden_dim, 8)  # 上卦
        self.lower_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 8) for _ in range(8)  # 下卦|上卦
        ])

    def forward(self, h):
        # 上卦分布
        p_upper = F.softmax(self.upper_head(h), dim=-1)  # (8,)

        # 下卦分布（上卦ごと）
        p_lower_given_upper = torch.stack([
            F.softmax(head(h), dim=-1) for head in self.lower_heads
        ])  # (8, 8)

        # 64卦の同時分布
        p_hex = p_upper.unsqueeze(-1) * p_lower_given_upper  # (8, 8)
        return p_hex.view(-1)  # (64,)
```

**利点**:
- パラメータ効率: 64クラス直接(hidden×64)より少ない
- 構造の反映: 上卦/下卦という易経の構造を活用
- 相関の表現: 独立ではなく条件付きで相関を表現

### 5. 確率校正 + Conformal Prediction

**意思決定のための信頼性保証**

```python
class CalibratedPredictor:
    def __init__(self, model, calibration_data):
        self.model = model
        self.temperature = self._calibrate(calibration_data)
        self.conformal_threshold = self._compute_conformal(calibration_data)

    def _calibrate(self, data):
        """
        温度スケーリングでECE最小化
        """
        # 校正データでグリッドサーチ
        best_temp = 1.0
        best_ece = float('inf')
        for temp in np.linspace(0.5, 2.0, 50):
            ece = compute_ece(self.model, data, temp)
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
        return best_temp

    def _compute_conformal(self, data):
        """
        Conformal prediction: 90%カバレッジを保証する閾値
        """
        scores = []
        for x, y in data:
            p = self.predict_calibrated(x)
            score = 1 - p[y]  # non-conformity score
            scores.append(score)
        # 90%点を閾値に
        return np.percentile(scores, 90)

    def predict_calibrated(self, x):
        """校正済み確率を出力"""
        logits = self.model(x)
        return F.softmax(logits / self.temperature, dim=-1)

    def predict_set(self, x, coverage=0.9):
        """
        Conformal prediction: カバレッジ保証付き集合を出力
        """
        p = self.predict_calibrated(x)
        sorted_idx = torch.argsort(p, descending=True)

        # 累積確率がカバレッジを超えるまで追加
        prediction_set = []
        cumsum = 0
        for idx in sorted_idx:
            prediction_set.append(idx.item())
            cumsum += p[idx].item()
            if cumsum >= coverage:
                break

        return prediction_set

    def decide_action(self, x, c_u):
        """
        意思決定: 校正済み確率に基づく
        """
        p = self.predict_calibrated(x)
        L_auto = compute_expected_cost(p)

        if L_auto <= c_u:
            return 'auto', p.argmax().item()
        else:
            return 'ask', self.predict_set(x)
```

### 6. 評価設計

**ベースライン、停止条件、不確実性を完備**

```python
BASELINES = {
    'prior_only': lambda x: prior_distribution,
    'simple_majority': lambda x, annot: majority_vote(annot),
    'upper_lower_independent': lambda x: p_upper(x) * p_lower(x),
    'no_crowd_layer': lambda x: direct_64class(x),  # 注釈ノイズなし
}

STOPPING_CONDITIONS = [
    ('prior_worse', lambda m, b: m['NLL'] > b['prior_only']['NLL'],
     "特徴量が効いていない"),
    ('no_crowd_benefit', lambda m, b: m['NLL'] > b['no_crowd_layer']['NLL'],
     "crowd-layerが機能していない"),
    ('calibration_failed', lambda m, _: m['ECE'] > 0.15,
     "校正が不十分"),
]

def evaluate_with_uncertainty(model, test_data, n_bootstrap=1000):
    """
    Bootstrap CIで不確実性を報告
    """
    results = {
        'NLL': {'mean': None, 'ci_95': None},
        'ECE': {'mean': None, 'ci_95': None},
        'coverage_at_risk_0.2': {'mean': None, 'ci_95': None},
    }
    # ... bootstrap計算
    return results
```

### 7. データ分割

```
全300件
    ↓
[ゴールド選定: 30件]
    - 専門家合議で確定
    - 識別性のアンカー
    ↓
[残り270件を分割]
    Test: 54件（ロック）
    Train+Val: 216件
    ↓
[K-fold クロスフィット (K=5)]
    Fold: Train 173件 + ゴールド30件 → Val 43件 → OOF予測
    ↓
[OOF全体 (216件)]
    - 確率校正
    - Conformal閾値決定
    ↓
[最終モデル]
    216件 + ゴールド30件 で再学習
    ↓
[Test評価]
    54件に1回だけ適用
```

## 工数見積もり

| フェーズ | 内容 | 工数 |
|----------|------|------|
| **Phase 1: 設計** | | **48h** |
| - ゴールド選定基準 | 層化サンプリング設計 | 8h |
| - crowd-layer設計 | 疎混同行列、階層ヘッド | 20h |
| - 評価パイプライン | ベースライン、conformal | 20h |
| **Phase 2: ゴールド作成** | | **40h** |
| - 30件の合議 | 3名×30件の議論 | 30h |
| - ゴールド検証 | 合意度確認 | 10h |
| **Phase 3: 本注釈** | | **80h** |
| - 270件注釈 | 3名×270件 | 60h |
| - 品質検証 | Krippendorff's α | 20h |
| **Phase 4: モデル構築** | | **72h** |
| - crowd-layer実装 | End-to-End学習 | 24h |
| - クロスフィット | OOF生成 | 16h |
| - 校正・conformal | 閾値決定 | 16h |
| - Test評価 | CI付き報告 | 16h |
| **合計** | | **240h** |

## MVP（最小実行可能設計）

1. **ゴールド**: 30件（層化サンプリング）
2. **注釈モデル**: 疎混同行列（近傍8卦のみ）
3. **予測モデル**: 上卦/下卦階層 + crowd-layer
4. **校正**: 温度スケーリング
5. **出力**: 校正済み確率 + conformal集合

## フォールバック

| 条件 | 対応 |
|------|------|
| crowd-layerが収束しない | 注釈モデル分離（v10方式）に戻す |
| 階層がPrior以下 | 64クラス直接に変更 |
| ゴールド30件で不足 | 追加合議（最大50件） |
| α<0.3 | 注釈ガイド改訂→再注釈 |
