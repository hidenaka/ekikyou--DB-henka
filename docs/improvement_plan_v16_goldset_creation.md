# 六十四卦マッピング改善計画v16（制約下最終版）

## 発想の転換

### 従来の目標（破綻）
> 「64卦の汎化精度を上げる」
> → 300件で64クラス分類器を学習する（不可能）

### 新しい目標（成立）
> **「高品質なゴールド化データ + 曖昧例の可視化」**
> → 300件を整備して将来の拡張に備える

## 制約の確認

| 項目 | 値 | 変更可否 |
|------|-----|---------|
| サンプル数 | 300件 | 固定 |
| 注釈者数 | 3名 | 固定 |
| 対象クラス | 64卦 | 構造分解可 |

## 最終設計

### 1. 64卦の扱い

**単一カテゴリではなく、(上卦, 下卦)の2変数**

```python
# 八卦
EIGHT_TRIGRAMS = ['乾', '兌', '離', '震', '巽', '坎', '艮', '坤']

# 64卦 = 上卦 × 下卦
def hexagram_to_trigrams(hex_name):
    """64卦名 → (上卦, 下卦)"""
    return HEXAGRAM_DECOMPOSITION[hex_name]

# 例: 水雷屯 → (坎, 震)
```

**利点**:
- 部分正解を評価可能（上卦一致、下卦一致）
- 8クラス×2なのでデータ効率が良い
- 誤りパターンが解釈しやすい

### 2. 注釈フォーマット

**各サンプルで以下を記録**

```python
class Annotation:
    # 必須
    upper_trigram: str  # 上卦（8択）
    lower_trigram: str  # 下卦（8択）
    confidence: int     # 確信度（1=低, 2, 3, 4=高）

    # 任意
    hold: bool          # 保留（判断不能）
    second_upper: Optional[str]  # 第2候補の上卦
    second_lower: Optional[str]  # 第2候補の下卦

    # メタ情報
    annotator_id: str
    timestamp: datetime
    note: Optional[str]  # 判断理由のメモ
```

**強制単一選択のみは禁止**:
- 曖昧さを押し潰してノイズ化するから
- 保留と第2候補で不確実性を記録

### 3. 注釈の重複度

**全300件を3名全員が注釈**

```python
ANNOTATION_DESIGN = {
    'total_samples': 300,
    'annotators': 3,
    'overlap': 'full',  # 全員が全件
    'total_annotations': 900,  # 300 × 3
}
```

**ケチらない理由**:
- 規模が小さい以上、重複を減らす設計は自殺
- 3重複があれば多数決と裁定が成立

### 4. ゴールド（裁定）

**10-15%を裁定で確定**

```python
GOLD_DESIGN = {
    'target_ratio': 0.12,  # 12%
    'target_count': 36,    # 300 × 0.12

    # 裁定対象の選定基準
    'adjudication_criteria': [
        '全不一致（3者バラバラ）',
        '2-1分裂で確信度が低い',
        '保留が2名以上',
    ],

    # 裁定プロセス
    'process': [
        '1. 注釈者3名で議論',
        '2. 合意できれば確定',
        '3. 合意できなければ「曖昧」としてマーク',
    ],

    # 記録
    'record': [
        '最終決定（上卦, 下卦）',
        '裁定理由',
        '不一致の原因分析',
    ],
}
```

### 5. 集約モデル

**多数決 + 全不一致のみ裁定（最も頑健）**

```python
def aggregate_annotations(annotations):
    """
    3名の注釈を集約
    """
    # 上卦の多数決
    upper_votes = [a.upper_trigram for a in annotations]
    upper_majority = majority_vote(upper_votes)

    # 下卦の多数決
    lower_votes = [a.lower_trigram for a in annotations]
    lower_majority = majority_vote(lower_votes)

    # 一致度の判定
    upper_agreement = agreement_level(upper_votes)
    lower_agreement = agreement_level(lower_votes)

    return {
        'upper': upper_majority,
        'lower': lower_majority,
        'upper_agreement': upper_agreement,  # 3-0, 2-1, 1-1-1
        'lower_agreement': lower_agreement,
        'needs_adjudication': (upper_agreement == '1-1-1' or
                               lower_agreement == '1-1-1'),
    }

def agreement_level(votes):
    """3票の一致度を判定"""
    counts = Counter(votes)
    if len(counts) == 1:
        return '3-0'  # 全員一致
    elif len(counts) == 2:
        return '2-1'  # 2対1
    else:
        return '1-1-1'  # 全員バラバラ
```

### 6. 評価指標

**測定可能なもの**

```python
METRICS = {
    # 一致度
    'upper_3_0_rate': '上卦全員一致率',
    'lower_3_0_rate': '下卦全員一致率',
    'both_3_0_rate': '両方全員一致率',

    # 不一致
    'upper_1_1_1_rate': '上卦全不一致率',
    'lower_1_1_1_rate': '下卦全不一致率',

    # 裁定
    'adjudication_rate': '裁定が必要だった率',
    'adjudication_resolved_rate': '裁定で合意できた率',

    # 確信度
    'confidence_accuracy_corr': '確信度と正解の相関',

    # ゴールドに対する精度（裁定済みのみ）
    'gold_upper_accuracy': 'ゴールドに対する上卦精度',
    'gold_lower_accuracy': 'ゴールドに対する下卦精度',
    'gold_both_accuracy': 'ゴールドに対する両方精度',
}
```

**測定しても結論が出ないもの**（やらない）:
- 64クラスのmacro-F1
- クラス別精度の網羅的議論

### 7. 成果物

**300件のゴールド化データセット**

```python
class GoldDataset:
    """
    最終成果物
    """
    def __init__(self):
        self.samples = []  # 300件

    def add_sample(self, sample):
        self.samples.append({
            # 元データ
            'case_id': sample.case_id,
            'text': sample.text,

            # 注釈（3名分）
            'annotations': sample.annotations,

            # 集約結果
            'upper_trigram': sample.aggregated_upper,
            'lower_trigram': sample.aggregated_lower,
            'hexagram': sample.hexagram,  # 64卦名

            # メタ情報
            'agreement': {
                'upper': sample.upper_agreement,
                'lower': sample.lower_agreement,
            },
            'adjudicated': sample.adjudicated,
            'ambiguous': sample.ambiguous,  # 裁定でも合意できなかった

            # 品質指標
            'avg_confidence': sample.avg_confidence,
            'has_second_candidate': sample.has_second_candidate,
        })

    def export(self, path):
        """JSONL形式で出力"""
        with open(path, 'w') as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
```

### 8. 将来の拡張パス

**300件完成後の選択肢**

```
Phase 1: 300件ゴールド化（本計画）
    ↓
Phase 2: 追加データ収集
    - 高一致サンプルの類似例を追加
    - 曖昧サンプルの類似例で境界を探索
    ↓
Phase 3: 分類器学習（データが十分になったら）
    - 上卦分類器（8クラス）
    - 下卦分類器（8クラス、上卦条件付き可）
    ↓
Phase 4: 64卦分類器（さらにデータが増えたら）
```

## 工数見積もり

| フェーズ | 内容 | 工数 |
|----------|------|------|
| **Phase 1: 準備** | | **24h** |
| - 注釈ガイド作成 | 判断基準、例示 | 12h |
| - 注釈UI/フォーム | 上卦/下卦/確信度 | 8h |
| - 注釈者トレーニング | 3名×ガイド説明 | 4h |
| **Phase 2: 注釈** | | **60h** |
| - 本注釈 | 300件×3名=900注釈 | 45h |
| - 品質チェック | 一致度確認 | 8h |
| - 中間調整 | ガイド改訂 | 7h |
| **Phase 3: 裁定** | | **24h** |
| - 裁定対象選定 | 全不一致/低確信 | 4h |
| - 裁定議論 | 36件×3名 | 16h |
| - 結果記録 | | 4h |
| **Phase 4: 整備** | | **12h** |
| - データ整形 | JSONL出力 | 4h |
| - 品質レポート | 指標計算 | 8h |
| **合計** | | **120h** |

## 成功基準

| 指標 | 目標値 | 意味 |
|------|--------|------|
| 上卦3-0率 | ≥60% | 上卦の判断は比較的一致 |
| 下卦3-0率 | ≥60% | 下卦の判断は比較的一致 |
| 両方3-0率 | ≥40% | 64卦として完全一致 |
| 裁定解決率 | ≥80% | 裁定で合意できる率 |
| 曖昧率 | ≤10% | 裁定でも合意できない率 |

## フォールバック

| 条件 | 対応 |
|------|------|
| 3-0率が30%以下 | 注釈ガイド全面改訂 |
| 裁定解決率が50%以下 | 問題定義を簡素化（上卦のみ等） |
| 曖昧率が20%超 | 「本質的に多義」として受容 |

---

## 13回のディベート総括

| 教訓 | 内容 |
|------|------|
| **目標設定** | 300件で64クラス分類器学習は不可能。ゴールド化がゴール |
| **構造活用** | 64卦→(上卦8, 下卦8)に分解して効率化 |
| **不確実性記録** | 強制単一選択は禁止。保留・第2候補で曖昧さを記録 |
| **重複度** | 小規模なら全件重複。ケチらない |
| **集約** | 複雑なモデルより多数決+裁定が頑健 |
