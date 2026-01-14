# 六十四卦マッピング改善計画v13 - 構造変換ノイズ+項目難易度モデル版

## v12からの変更点

| v12 | v13 |
|----|-----|
| 独立ビット反転 | **構造変換混合ノイズ（爻反転/上下取違/反転/一様）** |
| 識別性アンカー（真値未確定） | **確定ゴールド（専門家合議で真値固定）** |
| 6爻自己回帰（過剰） | **独立6ビットを基準（シンプル優先）** |
| full overlap（非効率） | **部分重複 + ゴールド** |
| 注釈者能力のみ | **GLAD/IRT系（能力 + 項目難易度）** |
| 280+20件矛盾 | **300件厳密分割** |

## 設計思想

### 問題定義（維持）
> 「注釈者分布（soft label）を予測する」→ 多義性を認めた分布予測問題

### 核心の転換
> 相関した誤りを**卦の構造に沿った変換**としてモデル化

## 核心設計

### 1. 確定ゴールド（真値を定義）

**アンカーではなく、真値を確定させる**

```python
class ConfirmedGold:
    """
    専門家合議で真値を確定させたサンプル
    """
    def __init__(self, n_gold=24):
        self.n_gold = n_gold  # 8(八卦)×3件

    def select_and_adjudicate(self, cases, experts):
        """
        選定・確定プロセス:
        1. 八卦（上卦）ごとに3件ずつ候補を選定
        2. 3名専門家が独立に判定
        3. 不一致は議論で合意形成
        4. 合意できなければ除外（別サンプルを選定）
        """
        gold = []
        for trigram in EIGHT_TRIGRAMS:
            candidates = self._filter_by_upper_trigram(cases, trigram)
            for _ in range(3):
                case = self._select_clearest(candidates)
                true_hex = self._expert_consensus(case, experts)
                if true_hex is not None:
                    gold.append((case, true_hex))
                    candidates.remove(case)
        return gold[:self.n_gold]
```

**アンカーとの違い**:
- アンカー: 「選び方」だけ、真値は未確定
- ゴールド: 専門家合議で**真値を確定**

### 2. 構造変換混合ノイズ

**卦の構造に沿った変換の混合**

```python
class StructuralTransformNoise:
    """
    P(observed | true, annotator) を構造変換の混合で表現

    変換タイプ:
    1. 正解（変換なし）
    2. 単爻反転（1ビット）
    3. 上卦取り違え（上3ビット）
    4. 下卦取り違え（下3ビット）
    5. 全体反転/錯綜
    6. 一様誤り（ランダム）
    """
    def __init__(self, n_annotators):
        # 各注釈者の変換確率（6パラメータ）
        # 合計1になるようソフトマックス
        self.transform_probs = {}  # annotator → [p_correct, p_yao, p_upper, p_lower, p_invert, p_uniform]

        # 単爻反転は6爻のどれを反転するかも確率（6パラメータ）
        self.yao_flip_probs = {}  # annotator → [p1, p2, p3, p4, p5, p6]

    def confusion_prob(self, annotator, true_hex, observed_hex):
        """
        P(observed | true, annotator)
        """
        probs = self.transform_probs[annotator]
        total = 0

        # 1. 正解
        if observed_hex == true_hex:
            total += probs[0]

        # 2. 単爻反転
        true_bits = hexagram_to_bits(true_hex)
        obs_bits = hexagram_to_bits(observed_hex)
        hamming = sum(t != o for t, o in zip(true_bits, obs_bits))
        if hamming == 1:
            flipped_yao = [i for i, (t, o) in enumerate(zip(true_bits, obs_bits)) if t != o][0]
            total += probs[1] * self.yao_flip_probs[annotator][flipped_yao]

        # 3. 上卦取り違え（下3ビット一致、上3ビット異なる）
        if true_bits[3:] == obs_bits[3:] and true_bits[:3] != obs_bits[:3]:
            total += probs[2] / 7  # 7通りの異なる上卦

        # 4. 下卦取り違え（上3ビット一致、下3ビット異なる）
        if true_bits[:3] == obs_bits[:3] and true_bits[3:] != obs_bits[3:]:
            total += probs[3] / 7  # 7通りの異なる下卦

        # 5. 全体反転/錯綜
        inverted = tuple(1 - b for b in true_bits)
        if obs_bits == inverted:
            total += probs[4]

        # 6. 一様誤り
        total += probs[5] / 64

        return total
```

**利点**:
- パラメータ数: 12 × n_annotators（変換6 + 爻確率6）
- 相関した多ビット誤りを自然に表現
- ドメイン構造（上卦/下卦）と整合

### 3. GLAD/IRT系（能力 + 難易度）

**注釈者能力だけでなく、項目難易度もモデル化**

```python
class GLADModel:
    """
    P(correct | annotator, item)
      = σ(ability[annotator] × difficulty[item])

    困難な項目は誰でも間違えやすい
    能力の高い注釈者でも難しい項目は誤る
    """
    def __init__(self, n_annotators, n_items):
        self.ability = {}  # annotator → scalar
        self.difficulty = {}  # item → scalar（正=難しい）

    def error_rate(self, annotator, item):
        """
        この注釈者がこの項目で誤る確率
        """
        return 1 - sigmoid(self.ability[annotator] * self.difficulty[item])

    def fit(self, observations, gold_items):
        """
        EMアルゴリズムで推定
        gold_itemsは真値が確定しているので、そこから能力を推定
        """
        pass
```

**GLAD vs 単純ノイズモデル**:
- 単純: 注釈者の誤り率は一定
- GLAD: 難しい項目ほど誰でも誤りやすい

### 4. 独立6ビット予測（基準モデル）

**シンプルなベースラインを主設計に**

```python
class IndependentBitPredictor(nn.Module):
    """
    P(bits|x) = Π P(bj|x)

    自己回帰は過剰。まずは独立を基準に。
    """
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.bit_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(6)
        ])

    def forward(self, x):
        h = self.encoder(x)
        probs = []
        for j in range(6):
            logit = self.bit_heads[j](h)
            probs.append(torch.sigmoid(logit))
        return probs

    def get_64_distribution(self, x):
        """独立積で64卦分布を計算"""
        bit_probs = self.forward(x)
        dist = torch.zeros(64)
        for hex_id in range(64):
            bits = hexagram_to_bits(hex_id + 1)
            p = 1.0
            for j in range(6):
                if bits[j] == 1:
                    p *= bit_probs[j]
                else:
                    p *= (1 - bit_probs[j])
            dist[hex_id] = p
        return dist
```

**自己回帰への拡張条件**:
- 独立モデルがベースラインに対して十分勝っている
- データが増えて過学習リスクが下がった
- 独立モデルの残差分析で依存構造の兆候が見えた

### 5. 部分重複注釈設計

**full overlapを捨て、効率的な重複設計**

```python
ANNOTATION_DESIGN = {
    'n_annotators': 3,
    'n_samples': 276,  # 300 - 24(ゴールド)

    # 重複設計
    'full_overlap': 60,  # 全員が注釈（同定用）
    'double_overlap': 108,  # 2名が注釈（精度向上）
    'single': 108,  # 1名のみ（カバレッジ拡大）

    # 総注釈数: 60×3 + 108×2 + 108×1 = 504件
    # full_overlap比: 504 vs 828（300×3の61%）
}

def allocate_annotators(samples, annotators, design):
    """
    効率的な割り当て
    """
    # 1. 同定に重要なサンプル（難易度が予想しにくい）はfull_overlap
    # 2. 典型的なサンプルはdouble
    # 3. 明確なサンプルはsingle
    pass
```

### 6. データ分割（厳密300件）

```
全300件
    ↓
[ゴールド選定: 24件]
    - 八卦×3件
    - 専門家合議で真値確定
    - 学習・校正・テスト全てで使用可能（真値だから）
    ↓
[残り276件を分割]
    Train: 166件（60%）
    Cal: 55件（20%）
    Test: 55件（20%）
    ↓
[注釈]
    - 全276件 × 部分重複設計
    - ゴールド24件は注釈不要（真値確定済み）
    ↓
[学習]
    Train 166件 + ゴールド24件でモデル学習
    ↓
[校正]
    Cal 55件で温度スケーリング
    ↓
[評価]
    Test 55件 + ゴールド24件の一部で最終評価

※ 166+55+55+24=300 で矛盾なし
```

### 7. 評価設計

```python
BASELINES = {
    'prior_only': '訓練データの卦分布',
    'majority_vote': '多数決',
    'simple_noise': '単純ノイズモデル（能力のみ）',
    'transform_no_difficulty': '構造変換（難易度なし）',
}

METRICS = {
    'NLL': '分布予測品質（主指標）',
    'Hamming_loss': '6ビットベースの誤り',
    'Top1_accuracy': '最頻出卦の正答率',
    'Top5_accuracy': 'Top-5に正解が含まれる率',
    'ECE': '校正誤差',
}

STOPPING_CONDITIONS = {
    'prior_worse': 'NLL > prior_only',
    'noise_no_benefit': 'NLL > simple_noise',
    'difficulty_no_benefit': 'NLL > transform_no_difficulty',
}
```

## 工数見積もり

| フェーズ | 内容 | 工数 |
|----------|------|------|
| **Phase 1: 設計** | | **48h** |
| - 構造変換ノイズ設計 | 変換定義、パラメータ | 16h |
| - GLAD統合設計 | 能力+難易度 | 16h |
| - 評価パイプライン | ベースライン実装 | 16h |
| **Phase 2: ゴールド作成** | | **32h** |
| - 24件候補選定 | 八卦×3件 | 8h |
| - 専門家合議 | 3名×24件議論 | 24h |
| **Phase 3: 本注釈** | | **72h** |
| - 276件×部分重複 | 504注釈 | 50h |
| - 品質検証 | Krippendorff's α | 12h |
| - 難易度推定 | GLAD EM | 10h |
| **Phase 4: モデル構築** | | **48h** |
| - 6ビット予測器実装 | 独立モデル | 12h |
| - ノイズモデル学習 | EM推定 | 16h |
| - 校正+評価 | Cal/Test | 20h |
| **合計** | | **200h** |

## MVP（最小実行可能設計）

1. **ゴールド**: 24件（八卦×3）確定
2. **ノイズ**: 構造変換混合（爻反転+上下取違+一様）
3. **難易度**: GLAD（能力+難易度）
4. **予測**: 独立6ビット
5. **注釈**: 部分重複（full 60 + double 108 + single 108）
6. **評価**: NLL + Hamming + Top5 + Bootstrap CI

## フォールバック

| 条件 | 対応 |
|------|------|
| 構造変換が単純ノイズに負ける | 独立ビット反転に戻す |
| GLADが収束しない | 難易度を無視（能力のみ） |
| 独立6ビットがpriorに負ける | 特徴量/エンコーダを見直し |
| α<0.3 | 上卦/下卦（2×3ビット）に簡素化 |
