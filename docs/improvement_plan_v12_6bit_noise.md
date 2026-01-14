# 六十四卦マッピング改善計画v12 - 6爻ビット構造ノイズモデル版

## v11からの変更点

| v11 | v12 |
|----|-----|
| 疎64×64混同行列（恣意的近傍8） | **6爻ビット反転ノイズ + 一様誤り** |
| ゴールド30件で64卦均等（不可能） | **識別性アンカー（構造対立例）** |
| 注釈重なり未設計 | **3名全員が全サンプル注釈** |
| 校正分割未定義 | **明示的3分割（Train/Cal/Test）** |
| 上卦/下卦階層（誤差増幅リスク） | **6爻構造化モデル（非独立）** |

## 設計思想

### 問題定義（維持）
> 「注釈者分布（soft label）を予測する」→ 多義性を認めた分布予測問題

### 核心の転換
> 64クラスではなく**6ビット表現**に基づく構造化

## 核心設計

### 1. 6爻ビット表現

**64卦 = 6ビットのバイナリ表現**

```python
HEXAGRAM_BITS = {
    1:  (1,1,1,1,1,1),  # 乾為天 111111
    2:  (0,0,0,0,0,0),  # 坤為地 000000
    3:  (1,0,0,0,1,0),  # 水雷屯 ...
    # ...64卦すべて
}

def hexagram_to_bits(hex_id):
    """64卦 → 6ビット"""
    return HEXAGRAM_BITS[hex_id]

def bits_to_hexagram(bits):
    """6ビット → 64卦"""
    return BITS_TO_HEXAGRAM[tuple(bits)]
```

### 2. 6爻ビット反転ノイズモデル

**注釈者の誤りを「爻の取り違え」としてモデル化**

```python
class BitFlipNoiseModel:
    """
    P(observed_hex | true_hex, annotator) を6ビット空間でモデル化
    """
    def __init__(self, n_annotators):
        # 各注釈者の爻ごとの反転確率
        # パラメータ数: 6 × n_annotators（v11の512/注釈者より大幅削減）
        self.flip_prob = {}  # annotator → [p1, p2, p3, p4, p5, p6]

        # 一様誤り（スパム/ランダムクリック）の確率
        self.uniform_error_prob = {}  # annotator → ε

    def confusion_prob(self, annotator, true_hex, observed_hex):
        """
        P(observed | true, annotator)
        """
        true_bits = hexagram_to_bits(true_hex)
        obs_bits = hexagram_to_bits(observed_hex)

        # 一様誤りコンポーネント
        p_uniform = self.uniform_error_prob[annotator] / 64

        # ビット反転コンポーネント
        p_flip = 1.0
        for j in range(6):
            if true_bits[j] == obs_bits[j]:
                p_flip *= (1 - self.flip_prob[annotator][j])
            else:
                p_flip *= self.flip_prob[annotator][j]

        p_structured = (1 - self.uniform_error_prob[annotator]) * p_flip

        return p_uniform + p_structured
```

**利点**:
- パラメータ数: 7 × n_annotators（flip_prob 6 + uniform 1）
- 「1爻違い」が自然に高確率
- 恣意的な「近傍」定義が不要

### 3. 6爻構造化予測モデル（非独立）

**v9の6爻独立を改良し、爻間の依存関係を入れる**

```python
class StructuredBitPredictor(nn.Module):
    """
    6ビットを自己回帰的に予測（依存関係あり）
    """
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder

        # 爻を順番に予測（下から上へ）
        # 各爻は入力 + これまでの爻に依存
        self.bit_predictors = nn.ModuleList([
            nn.Linear(hidden_dim + j, 1)  # j番目の爻は過去j爻に依存
            for j in range(6)
        ])

    def forward(self, x):
        """
        P(bits|x) = P(b1|x) × P(b2|x,b1) × ... × P(b6|x,b1,...,b5)
        """
        h = self.encoder(x)  # テキスト埋め込み
        bits_so_far = []
        log_probs = []

        for j in range(6):
            # 入力 = h + これまでの爻
            if j == 0:
                inp = h
            else:
                bits_tensor = torch.stack(bits_so_far, dim=-1)
                inp = torch.cat([h, bits_tensor], dim=-1)

            # j番目の爻の確率
            logit = self.bit_predictors[j](inp)
            p_j = torch.sigmoid(logit)

            # サンプリング（学習時）or 確率計算（推論時）
            bits_so_far.append(p_j)
            log_probs.append(p_j)

        return log_probs  # 6つの確率

    def get_64_distribution(self, x):
        """
        全64卦の確率分布を計算
        """
        probs = torch.zeros(64)
        for hex_id in range(64):
            bits = hexagram_to_bits(hex_id + 1)  # 1-indexed
            probs[hex_id] = self._prob_of_bits(x, bits)
        return probs / probs.sum()

    def _prob_of_bits(self, x, target_bits):
        """
        特定のビット列の確率を計算
        """
        h = self.encoder(x)
        prob = 1.0
        bits_so_far = []

        for j in range(6):
            if j == 0:
                inp = h
            else:
                bits_tensor = torch.tensor(bits_so_far, dtype=torch.float)
                inp = torch.cat([h, bits_tensor], dim=-1)

            logit = self.bit_predictors[j](inp)
            p_j = torch.sigmoid(logit)

            if target_bits[j] == 1:
                prob *= p_j
            else:
                prob *= (1 - p_j)

            bits_so_far.append(target_bits[j])

        return prob
```

**v9との違い**:
- v9: P(bits|x) = Π P(bj|x) 【独立】
- v12: P(bits|x) = Π P(bj|x, b1, ..., bj-1) 【依存】

### 4. ゴールドサンプル（識別性アンカー）

**「均等カバー」ではなく「構造対立例」**

```python
class IdentifiabilityAnchors:
    """
    ノイズモデルの識別性を確保するアンカー
    """
    def __init__(self, n_anchors=20):
        self.n_anchors = n_anchors

    def select(self, cases):
        """
        選定基準: 構造（爻）の対立を明示する例
        """
        anchors = []

        # 1. 各爻について、その爻が決定的なケースを選ぶ
        # 例: 「この事例は明らかに初爻が陽」
        for yao in range(6):
            # 陽が明確な例
            anchors.append(self._find_clear_yang(cases, yao))
            # 陰が明確な例
            anchors.append(self._find_clear_yin(cases, yao))

        # 2. 上卦/下卦の対立例
        # 例: 「上卦は震、下卦は坎が明確」
        anchors.extend(self._find_trigram_contrasts(cases))

        return anchors[:self.n_anchors]

    def _find_clear_yang(self, cases, yao):
        """爻がほぼ確実に陽のケースを見つける"""
        pass

    def _find_clear_yin(self, cases, yao):
        """爻がほぼ確実に陰のケースを見つける"""
        pass
```

**選定数の根拠**:
- 6爻 × 2（陽/陰）= 12件で各爻の識別性
- + 8件で上卦/下卦の対立例
- 合計20件（30件以下でも識別性確保可能）

### 5. 注釈設計（厳密化）

**crowd-layerの識別性条件を満たす**

```python
ANNOTATION_DESIGN = {
    'n_annotators': 3,
    'overlap': 'full',  # 全員が全サンプル注釈
    'min_per_annotator': 300,  # 各注釈者が全300件を担当

    # 注釈形式
    'format': {
        'primary': '最も妥当な卦を1つ選択',
        'secondary': '次に妥当な卦を0-2つ選択（任意）',
        'confidence': '確信度（1-5）'
    }
}
```

**full overlapの理由**:
- 注釈者バイアスの分離に必要
- 同一サンプルで注釈者間比較ができる

### 6. データ分割（3分割厳密化）

```
全300件
    ↓
[ゴールド選定: 20件]
    - 専門家合議（構造対立例）
    - 識別性アンカー
    ↓
[残り280件を3分割]
    Train: 168件（60%）  ← モデル学習
    Cal: 56件（20%）     ← 校正・閾値
    Test: 56件（20%）    ← 最終評価（ロック）
    ↓
[K-fold クロスフィット (K=5) on Train]
    OOFでノイズモデルパラメータ推定
    ↓
[Cal で校正]
    温度スケーリング
    ↓
[Test で最終評価]
    1回だけ
```

**conformalを外す理由**:
- Cal 56件では交換可能性の検証が困難
- 保証を掲げる資格がない
- 代わりにBootstrap CIで不確実性を報告

### 7. 評価設計

```python
BASELINES = {
    'prior_only': 'ゴールド分布をそのまま使用',
    'independent_6bit': 'v9方式（爻独立）',
    'majority_vote': '多数決',
    'random_forest_64class': '64クラス直接分類',
}

METRICS = {
    'NLL': '分布予測品質',
    'Hamming_loss': '6爻ベースの誤り',
    'Top5_accuracy': 'Top-5に正解が含まれる率',
}

STOPPING_CONDITIONS = {
    'prior_worse': ('NLL > prior_only', 'モデルが機能していない'),
    'independent_worse': ('NLL > independent_6bit', '依存構造が効いていない'),
}
```

## 工数見積もり

| フェーズ | 内容 | 工数 |
|----------|------|------|
| **Phase 1: 設計** | | **40h** |
| - 6爻ビットマッピング | 64卦↔6ビット変換 | 8h |
| - ノイズモデル設計 | ビット反転+一様 | 16h |
| - 評価パイプライン | ベースライン実装 | 16h |
| **Phase 2: ゴールド+注釈** | | **100h** |
| - ゴールド20件合議 | 構造対立例選定 | 20h |
| - 本注釈280件×3名 | Full overlap | 60h |
| - 品質検証 | Krippendorff's α | 20h |
| **Phase 3: モデル構築** | | **60h** |
| - 構造化予測器実装 | 自己回帰6爻 | 20h |
| - ノイズモデル学習 | EM推定 | 16h |
| - 校正+評価 | Cal/Test分割 | 24h |
| **合計** | | **200h** |

## MVP（最小実行可能設計）

1. **表現**: 64卦 = 6ビット
2. **ノイズ**: ビット反転確率（6×n_annotators） + 一様誤り
3. **予測**: 自己回帰6爻（依存あり）
4. **アンカー**: 構造対立例20件
5. **分割**: Train/Cal/Test 厳密3分割
6. **評価**: NLL + Hamming + Top5 + Bootstrap CI

## フォールバック

| 条件 | 対応 |
|------|------|
| 自己回帰が独立より悪い | 独立6爻に戻す |
| ノイズモデルが収束しない | 注釈者バイアス無視（単純集約） |
| α<0.3 | 6爻→上卦/下卦（2×3ビット）に簡素化 |
| CI幅が広すぎる | パイロットとして扱い、次フェーズでデータ拡張 |
