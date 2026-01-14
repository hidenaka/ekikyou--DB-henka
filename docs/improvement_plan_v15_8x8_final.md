# 六十四卦マッピング改善計画v15（最終確定版）

## 11回のLLM Debate収束点

### Codexの一貫した指摘
> **「552注釈・300件で64クラスを直接扱うのは情報量的に不可能」**

### 解決策
> **64卦 → 上卦8 × 下卦8 に縮約**（2段階分類）

## 最終設計

### 1. クラス空間の縮約

**64クラス → 8×8の2段階構造**

```python
# 八卦（三爻）
EIGHT_TRIGRAMS = {
    '乾': (1,1,1),  # 天
    '兌': (1,1,0),  # 沢
    '離': (1,0,1),  # 火
    '震': (1,0,0),  # 雷
    '巽': (0,1,1),  # 風
    '坎': (0,1,0),  # 水
    '艮': (0,0,1),  # 山
    '坤': (0,0,0),  # 地
}

def hexagram_to_trigrams(hex_id):
    """64卦 → (上卦, 下卦)"""
    bits = hexagram_to_bits(hex_id)
    upper = bits[:3]  # 上卦（4-6爻）
    lower = bits[3:]  # 下卦（1-3爻）
    return (bits_to_trigram(upper), bits_to_trigram(lower))

# パラメータ削減
# 64クラスDS: 64×64×3 = 12,288セル
# 8クラスDS×2: (8×8×3)×2 = 384セル（1/32に削減）
```

### 2. 2段階Dawid-Skene

**上卦と下卦を別々に推定**

```python
class TwoStageDawidSkene:
    """
    上卦と下卦を独立にDawid-Skene
    - 上卦: 8×8混同行列 × 3注釈者 = 192パラメータ
    - 下卦: 8×8混同行列 × 3注釈者 = 192パラメータ
    - 合計: 384パラメータ（552注釈で推定可能）
    """
    def __init__(self, n_annotators=3):
        self.upper_ds = DawidSkene(n_annotators, n_classes=8)
        self.lower_ds = DawidSkene(n_annotators, n_classes=8)

    def fit(self, annotations, gold=None):
        """
        annotations: Dict[item_id, Dict[annotator, hexagram_id]]
        → 上卦・下卦に分解してそれぞれEM
        """
        upper_annot, lower_annot = self._split_trigrams(annotations)

        if gold:
            upper_gold, lower_gold = self._split_trigrams_gold(gold)
        else:
            upper_gold, lower_gold = None, None

        self.upper_ds.fit(upper_annot, upper_gold)
        self.lower_ds.fit(lower_annot, lower_gold)

    def get_soft_label(self, item_annotations):
        """
        64卦の事後分布 = 上卦事後 × 下卦事後（独立仮定）
        """
        upper_post = self.upper_ds.get_soft_label(
            self._extract_upper(item_annotations))
        lower_post = self.lower_ds.get_soft_label(
            self._extract_lower(item_annotations))

        # 64卦分布 = 外積
        hex_dist = np.outer(upper_post, lower_post).flatten()
        return hex_dist  # (64,)
```

**パラメータ効率**:
- 64クラスDS: 12,288パラメータ / 552注釈 = 22.3パラ/注釈（不可能）
- 8×8 2段階DS: 384パラメータ / 552注釈 = 0.7パラ/注釈（推定可能）

### 3. 2段階分類器

**上卦→下卦の条件付き予測**

```python
class TwoStageClassifier(nn.Module):
    """
    Stage 1: P(上卦|x)
    Stage 2: P(下卦|x, 上卦)

    独立積ではなく条件付きで相関を表現
    """
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder

        # 上卦予測（8クラス）
        self.upper_head = nn.Linear(hidden_dim, 8)

        # 下卦予測（上卦条件付き、8×8通り）
        self.lower_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 8) for _ in range(8)
        ])

    def forward(self, x):
        h = self.encoder(x)

        # 上卦分布
        p_upper = F.softmax(self.upper_head(h), dim=-1)  # (8,)

        # 下卦分布（上卦ごと）
        p_lower_given_upper = torch.stack([
            F.softmax(head(h), dim=-1) for head in self.lower_heads
        ])  # (8, 8)

        # 64卦分布 = P(上卦) × P(下卦|上卦)
        p_hex = p_upper.unsqueeze(-1) * p_lower_given_upper  # (8, 8)
        return p_hex.view(-1)  # (64,)

    def loss(self, x, soft_label_64):
        """
        2段階DSからの64卦soft labelを教師に
        """
        pred = self.forward(x)
        return -torch.sum(soft_label_64 * torch.log(pred + 1e-10))
```

### 4. 注釈設計（8クラス対応）

```python
ANNOTATION_DESIGN = {
    'n_annotators': 3,
    'n_items': 276,  # 300 - 24(ゴールド)
    'min_annotations_per_item': 2,

    # クラス被覆の保証
    'class_coverage': {
        'upper_trigrams': 8,  # 全八卦が出現
        'lower_trigrams': 8,  # 全八卦が出現
        'min_per_class': 10,  # 各八卦に最低10件
    },

    # 総注釈数: 276 × 2 = 552
}
```

### 5. ゴールド設計（テストに含める）

**Codex指摘: テストに専門家ゴールドが必須**

```python
GOLD_DESIGN = {
    'total': 40,
    'allocation': {
        'train_anchor': 24,  # 学習アンカー（八卦×3件）
        'test_gold': 16,     # テスト評価用（八卦×2件）
    }
}

# 分割
# 全300件
# → ゴールド40件（学習24 + テスト16）
# → 残り260件を分割
#    Train: 156件（60%）
#    Cal: 52件（20%）
#    Test: 52件 + ゴールドテスト16件 = 68件（ロック）
```

### 6. データ分割（厳密）

```
全300件
    ↓
[ゴールド選定: 40件]
    - 学習アンカー24件（八卦×3）
    - テストゴールド16件（八卦×2）
    - 専門家合議で真値確定
    ↓
[残り260件を分割 + 2注釈付与]
    Train: 156件（60%）
    Cal: 52件（20%）
    Test: 52件（20%）
    ↓
[学習]
    Train 156件 + 学習ゴールド24件 で2段階DS + 分類器
    ↓
[校正]
    Cal 52件で温度スケーリング
    ↓
[評価]
    Test 52件 + テストゴールド16件 = 68件で最終評価
    - テストゴールド16件は「専門家正解」として評価
    - Test 52件はDS事後を疑似正解として評価
```

### 7. 評価設計

```python
METRICS = {
    # 主指標（8クラス）
    'upper_accuracy': '上卦正答率',
    'lower_accuracy': '下卦正答率',

    # 副指標（64クラス）
    'hex_top1': '64卦Top-1正答率',
    'hex_top5': '64卦Top-5正答率',

    # 分布品質
    'NLL_upper': '上卦NLL',
    'NLL_lower': '下卦NLL',

    # 校正
    'ECE': '校正誤差',
}

BASELINES = {
    'prior': '訓練分布',
    'majority': '多数決',
    'random_forest': 'テキスト特徴+RF',
}

STOPPING_CONDITIONS = {
    'prior_worse': 'upper/lowerがpriorに負けたら撤退',
    'majority_worse': 'majorityに負けたら撤退',
}
```

## 工数見積もり

| フェーズ | 内容 | 工数 |
|----------|------|------|
| **Phase 1: 設計** | | **28h** |
| - 2段階DS実装 | 8×8×2 | 12h |
| - 2段階分類器 | 条件付き予測 | 8h |
| - 評価パイプライン | ベースライン | 8h |
| **Phase 2: ゴールド作成** | | **40h** |
| - 40件候補選定 | 八卦×5件 | 8h |
| - 専門家合議 | 3名×40件議論 | 32h |
| **Phase 3: 本注釈** | | **48h** |
| - 260件×2注釈 | 520注釈 | 36h |
| - 品質検証 | Krippendorff's α | 12h |
| **Phase 4: モデル構築** | | **32h** |
| - 2段階DS学習 | EM | 8h |
| - 分類器学習 | 訓練 | 10h |
| - 校正+評価 | Cal/Test | 14h |
| **合計** | | **148h** |

## MVP（最小実行可能設計）

1. **クラス縮約**: 64卦 → 8×8（上卦×下卦）
2. **注釈統合**: 2段階Dawid-Skene
3. **予測**: 条件付き2段階分類（P(上卦|x) × P(下卦|x,上卦)）
4. **ゴールド**: 40件（学習24 + テスト16）
5. **注釈**: 260件×2注釈
6. **評価**: 上卦/下卦accuracy + 64卦Top1/5 + Bootstrap CI

## フォールバック

| 条件 | 対応 |
|------|------|
| 2段階DSが収束しない | 単純多数決 |
| 上卦/下卦がpriorに負ける | 特徴量見直し |
| α<0.3 | 注釈ガイド改訂 |

---

## ディベート履歴（全11回）

| Ver | 主設計 | 評価 | 主な批判 |
|-----|--------|------|---------|
| v2-v5 | キーワード/階層/質問 | 不合格 | 操作的定義なし |
| v6-v8 | ソフトスコア/統合重み | 方向性OK | リーク/恣意閾値 |
| v9 | 6爻独立 | 不合格 | 結合式破綻 |
| v10 | 注釈モデル分離 | 不合格 | 識別不能 |
| v11 | crowd-layer | 不合格 | 均等カバー不可 |
| v12 | ビット反転 | 不合格 | 相関誤り |
| v13 | 構造変換+IRT | 不合格 | 単独注釈問題 |
| v14 | Dawid-Skene 64 | 不合格 | パラメータ爆発 |
| **v15** | **2段階8×8** | **収束候補** | - |

### Codexの一貫した主張
1. **64クラスを直接扱うのはデータ量的に不可能**
2. **構造（上卦/下卦）を使ってクラス空間を圧縮すべき**
3. **テストに専門家ゴールドが必須**
4. **シンプルなベースラインで勝つことが先**
