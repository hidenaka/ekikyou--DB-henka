# 六十四卦マッピング改善計画v14（最終版）

## 10回のLLM Debate総括

### 確立した設計原則

| 原則 | 根拠（Codex指摘） |
|------|------------------|
| **シンプル優先** | 複雑なモデルより堅牢なベースラインで勝つ |
| **データ設計が先** | モデルより注釈設計・分割が重要 |
| **同定可能性** | パラメータ過多は推定が信用できない |
| **テスト汚染禁止** | ゴールドのテスト使用は評価を無効化 |
| **全項目2注釈以上** | IRT/GLAD無しでも基本要件 |
| **ベースライン棄却基準** | 勝てなければ撤退 |

### 却下された設計

| 却下された設計 | 却下理由 |
|---------------|----------|
| 6爻独立分解（v9） | 相関した不確実性を表現できない |
| 64×64混同行列（v10-11） | パラメータ爆発 |
| 自己回帰6爻（v12） | データ量に対して過剰 |
| GLAD/IRT（v13） | 単独注釈があると同定不能 |
| 構造変換混合（v13） | 区別不能変換の問題 |

## 最終設計

### 1. 主設計: Dawid-Skene（注釈者混同行列）

**最もシンプルで堅牢な注釈統合モデル**

```python
class DawidSkene:
    """
    古典的な注釈者モデル
    - 各注釈者は64×64の混同行列を持つ
    - EMアルゴリズムで真値と混同行列を同時推定
    """
    def __init__(self, n_annotators, n_classes=64):
        # 混同行列: P(observed | true, annotator)
        self.confusion = {}  # annotator → (64, 64) matrix

        # クラス事前分布
        self.prior = np.ones(n_classes) / n_classes

    def fit(self, annotations, gold=None):
        """
        EMアルゴリズム
        - E-step: P(true | annotations) を計算
        - M-step: 混同行列とpriorを更新
        - gold: 確定した真値（学習アンカー）
        """
        # 初期化
        for ann in self.confusion:
            self.confusion[ann] = np.eye(64) * 0.7 + 0.3 / 64

        for iteration in range(100):
            # E-step
            posteriors = self._e_step(annotations, gold)

            # M-step
            self._m_step(annotations, posteriors)

            if self._converged():
                break

        return posteriors

    def get_soft_label(self, item_annotations):
        """
        アイテムの注釈から事後分布を計算
        """
        log_posterior = np.log(self.prior + 1e-10)
        for annotator, observed in item_annotations.items():
            log_posterior += np.log(self.confusion[annotator][:, observed] + 1e-10)
        return softmax(log_posterior)
```

**選択理由**:
- 64×64混同行列は「完全」だがスパース正則化で制御可能
- EMの収束が安定
- ゴールドをアンカーとして使用可能

### 2. 予測モデル: 64クラスSoftmax

**独立6ビットではなく、64クラス直接分類**

```python
class HexagramClassifier(nn.Module):
    """
    64クラス直接分類
    - 表現力を落とさない
    - soft labelを教師に使用
    """
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_dim, 64)

    def forward(self, x):
        h = self.encoder(x)
        return F.softmax(self.head(h), dim=-1)

    def loss(self, x, soft_label):
        """
        Dawid-Skeneの事後分布を教師として交差エントロピー
        """
        pred = self.forward(x)
        return -torch.sum(soft_label * torch.log(pred + 1e-10))
```

### 3. 注釈設計: 全項目2注釈以上

**同定可能性の基本要件**

```python
ANNOTATION_DESIGN = {
    'n_annotators': 3,
    'n_items': 300,
    'min_annotations_per_item': 2,

    # 割り当て
    'allocation': {
        # 全276件（ゴールド除く）に最低2注釈
        # 総注釈数: 276 × 2 = 552 ≤ 300 × 3 = 900（余裕あり）
        'double_overlap': 276,  # 全件2注釈
    },

    # 注釈形式
    'format': {
        'primary': '最も妥当な卦を1つ選択',
        'confidence': '確信度（1-5）',  # 任意
    }
}
```

### 4. ゴールド: 学習アンカー専用

**テストには使わない**

```python
class GoldSamples:
    """
    確定した真値のサンプル
    - 学習アンカー（Dawid-Skeneの初期化）としてのみ使用
    - テストには絶対に含めない
    """
    def __init__(self, n_gold=24):
        self.n_gold = n_gold  # 八卦×3件

    def use_as_anchor(self, ds_model):
        """
        Dawid-SkeneのE-stepでgoldは事後確率を固定
        """
        for item, true_hex in self.gold_items:
            ds_model.fix_posterior(item, true_hex)
```

### 5. データ分割: 厳密3分割

```
全300件
    ↓
[ゴールド選定: 24件]
    - 学習アンカー専用
    - テストには含めない
    ↓
[残り276件を2注釈ずつ付与]
    ↓
[分割]
    Train: 166件（60%）← Dawid-Skene + 分類器学習
    Cal: 55件（20%）← 温度スケーリング
    Test: 55件（20%）← 最終評価（ロック）
    ↓
[学習]
    Train 166件 + Gold 24件（アンカー）で学習
    ↓
[校正]
    Cal 55件で温度スケーリング
    ↓
[評価]
    Test 55件のみで最終評価
```

### 6. ベースライン棄却基準

**勝てなければ撤退**

```python
BASELINES = {
    'prior_only': '訓練データの卦分布',
    'majority_vote': '多数決',
    'random_forest': 'テキスト特徴 + RF',
}

STOPPING_CONDITIONS = {
    # 必須: priorに勝つ
    ('prior_worse', 'NLL > prior_only', 'モデルが機能していない'),

    # 必須: 多数決に勝つ
    ('majority_worse', 'Top1 < majority_vote', '注釈統合が無意味'),

    # 条件付き撤退
    ('rf_worse', 'Top1 < random_forest', '深層学習不要の可能性'),
}

def evaluate_and_decide(model, test_data, baselines):
    """
    ベースラインとの比較で続行/撤退を決定
    """
    results = {}
    for name, baseline in baselines.items():
        comparison = compare_with_ci(model, baseline, test_data)
        results[name] = comparison

    # 撤退判定
    for condition, criterion, message in STOPPING_CONDITIONS:
        if results[condition.split('_')[0]]['lost']:
            return 'STOP', message

    return 'CONTINUE', results
```

### 7. 評価指標

```python
METRICS = {
    # 主指標
    'NLL': '負の対数尤度（分布品質）',

    # 副指標
    'Top1_accuracy': '最頻出卦の正答率',
    'Top5_accuracy': 'Top-5に正解が含まれる率',
    'ECE': '校正誤差',

    # 信頼区間
    'CI_method': 'Bootstrap 95%CI',
    'n_bootstrap': 1000,
}
```

## 工数見積もり

| フェーズ | 内容 | 工数 |
|----------|------|------|
| **Phase 1: 設計** | | **32h** |
| - Dawid-Skene実装 | EM + スパース正則化 | 16h |
| - 64クラス分類器 | エンコーダ + ヘッド | 8h |
| - 評価パイプライン | ベースライン + CI | 8h |
| **Phase 2: ゴールド作成** | | **32h** |
| - 24件候補選定 | 八卦×3件 | 8h |
| - 専門家合議 | 3名×24件議論 | 24h |
| **Phase 3: 本注釈** | | **56h** |
| - 276件×2注釈 | 552注釈 | 40h |
| - 品質検証 | Krippendorff's α | 16h |
| **Phase 4: モデル構築** | | **40h** |
| - Dawid-Skene学習 | EM実行 | 8h |
| - 分類器学習 | 訓練 | 12h |
| - 校正+評価 | Cal/Test | 20h |
| **合計** | | **160h** |

## MVP（最小実行可能設計）

1. **注釈統合**: Dawid-Skene（スパース正則化）
2. **予測**: 64クラスsoftmax
3. **ゴールド**: 24件（学習アンカー専用）
4. **注釈**: 全276件×2注釈
5. **分割**: Train/Cal/Test 厳密3分割
6. **評価**: NLL + Top1/5 + Bootstrap CI
7. **棄却基準**: prior/majority_voteに負けたら撤退

## フォールバック

| 条件 | 対応 |
|------|------|
| Dawid-Skene収束しない | 単純多数決に戻す |
| 64クラスがpriorに負ける | 特徴量/エンコーダを見直し |
| α<0.3 | 注釈ガイド改訂→再注釈 |
| Test 55件でCI幅広すぎ | パイロットとして扱い、次フェーズでデータ拡張 |

---

## ディベート履歴

| バージョン | 主な設計 | Codex評価 | 主な批判 |
|-----------|---------|-----------|---------|
| v2 | キーワード分類 | 不合格 | ラベル未定義 |
| v3 | 階層分解 | 不合格 | 相互排他性なし |
| v4 | 3軸定義 | 不合格 | 操作的定義なし |
| v5 | Q1-Q4質問 | 不合格 | 判定基準なし |
| v6 | ソフトスコアリング | 方向性OK | 独立積問題 |
| v7 | 学習済み統合 | 方向性OK | データリーク |
| v8 | risk-coverage | 方向性OK | 恣意的閾値 |
| v9 | 6爻独立分解 | 不合格 | 結合式破綻 |
| v10 | 注釈モデル分離 | 不合格 | 識別不能 |
| v11 | crowd-layer | 不合格 | 均等カバー不可能 |
| v12 | ビット反転ノイズ | 不合格 | 相関誤り表現不可 |
| v13 | 構造変換+IRT | 不合格 | 単独注釈問題 |
| **v14** | **Dawid-Skene** | **評価待ち** | - |
