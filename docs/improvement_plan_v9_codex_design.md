# 六十四卦マッピング改善計画v9 - Codex設計案採用版

## 設計思想の根本転換

### 従来の問題設定（破綻）
> 「真の卦を当てる」→ 単一正解を前提とした分類問題

### 新しい問題設定（Codex提案）
> **「注釈者分布（soft label）を予測する」**→ 多義性を認めた分布予測問題

## 核心設計

### 1. 6爻分解アーキテクチャ

**64クラス直当てを廃止**し、卦の構造（6爻）に分解してサンプル効率を稼ぐ。

```
64卦 = 6ビット表現
例: 乾為天(1番) = 111111, 坤為地(2番) = 000000

入力テキスト
    ↓
[6本の二値モデル]
    p_1(x) = P(爻1=陽|x)
    p_2(x) = P(爻2=陽|x)
    ...
    p_6(x) = P(爻6=陽|x)
    ↓
[64卦分布の復元]
    p(y|x) ∝ Π_j p_j(x)^{bit_j(y)} (1-p_j(x))^{1-bit_j(y)} · prior(y)
    ↓
[意思決定]
    L_auto(x) = min_ŷ Σ_y p(y|x) C(y,ŷ)
    auto iff L_auto(x) ≤ c_u
```

**利点**:
- 希少卦でも各爻は十分な出現頻度
- 未出現卦も確率ゼロにならない
- 6本の軽量モデルで十分

### 2. 目的関数の定義

```python
def cost_matrix(y_true, y_pred):
    """6爻ハミング距離によるコスト"""
    bits_true = hexagram_to_bits(y_true)  # 6ビット
    bits_pred = hexagram_to_bits(y_pred)
    hamming = sum(b1 != b2 for b1, b2 in zip(bits_true, bits_pred))
    return hamming / 6  # 0〜1に正規化

def expected_cost_auto(probs):
    """自動確定時の期待コスト"""
    # 最小期待コストの卦を選択
    min_cost = float('inf')
    for y_pred in range(64):
        cost = sum(probs[y] * cost_matrix(y, y_pred) for y in range(64))
        min_cost = min(min_cost, cost)
    return min_cost

def decide_action(probs, c_u):
    """行動決定: auto or ask"""
    L_auto = expected_cost_auto(probs)
    if L_auto <= c_u:
        return 'auto', argmin_expected_cost(probs)
    else:
        return 'ask', get_top_k(probs, k=5)
```

**c_u（ユーザー関与コスト）の決定**:
- クロスフィット内で最適化
- 恣意的な固定値（20%等）は使用しない

### 3. 出力形式

**分布出力 + 拒否（ask）を標準とする**

```python
class HexagramPrediction:
    distribution: Dict[int, float]  # 64卦の確率分布
    action: Literal['auto', 'ask']

    # autoの場合
    confirmed_hexagram: Optional[int]
    confidence: Optional[float]

    # askの場合
    top_k_candidates: Optional[List[Tuple[int, float]]]
    uncertainty_reason: Optional[str]  # "高エントロピー" or "低マージン"
```

**単一確定（auto）は高信頼域のみ**:
- 内部では常に分布を保持
- 外部に単一を出すのはautoのときだけ

### 4. soft labelの集約

**評点(1-5)を確率質量に変換**

```python
def aggregate_soft_labels(annotations, beta):
    """
    annotations: Dict[annotator_id, Dict[hexagram_id, score]]
    beta: 温度パラメータ（クロスフィット内で最適化）
    """
    # 評点→重みに変換
    def score_to_weight(r):
        return np.exp(beta * (r - 3))  # r=3が中立

    # 各卦の合算スコア
    aggregated = defaultdict(float)
    for annotator, scores in annotations.items():
        for hex_id, score in scores.items():
            aggregated[hex_id] += score_to_weight(score)

    # 正規化してsoft label化
    total = sum(aggregated.values())
    soft_label = {k: v / total for k, v in aggregated.items()}

    # 未評価の卦は0（またはスムージング）
    for hex_id in range(64):
        if hex_id not in soft_label:
            soft_label[hex_id] = 0.0

    return soft_label
```

**βの決定**:
- クロスフィット内で期待コスト最小になるβを探索
- 恣意的な固定値は使用しない

### 5. データ分割設計（リーク遮断）

```
全300件
    ↓
[Test: 60件] ← ロック（最終評価のみ）
    ↓
[Train+Val: 240件]
    ↓
[K-fold クロスフィット (K=5)]
    Fold 1: Train 192件 → Val 48件 → OOF予測
    Fold 2: Train 192件 → Val 48件 → OOF予測
    ...
    Fold 5: Train 192件 → Val 48件 → OOF予測
    ↓
[OOF全体 (240件)]
    - 統合学習（スタッキング）
    - 確率校正（温度スケーリング）
    - 閾値選択（c_uの決定）
    - β（soft label集約パラメータ）の決定
    ↓
[最終モデル]
    240件全体で再学習
    ↓
[Test評価]
    60件に1回だけ適用
```

**リーク遮断の保証**:
- 統合学習・校正・閾値選択は全てOOF上で実施
- Testデータは最終評価まで一切使用しない

### 6. 注釈一致の扱い

**alphaは合否基準にしない**

```python
def should_adjudicate(soft_label):
    """追加注釈が必要か判定"""
    probs = np.array(list(soft_label.values()))

    # 正規化エントロピー
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(64)
    normalized_entropy = entropy / max_entropy

    # マージン
    sorted_probs = np.sort(probs)[::-1]
    margin = sorted_probs[0] - sorted_probs[1]

    # 高エントロピーまたは低マージンは要アジュディケーション
    if normalized_entropy > 0.8 or margin < 0.1:
        return True, "注釈者間で意見が割れています"

    return False, None
```

**alphaの使い方**:
- プロセス監視指標としてのみ使用
- 低い場合は「カテゴリ定義・注釈ガイド」が壊れていると判断
- 合否判定には使用しない

### 7. 爻判定

64卦が確定した後、その卦の爻辞を参照して爻を判定。

```python
def determine_yao(hexagram_id, outcome_text, yao_descriptions):
    """
    hexagram_id: 確定した64卦
    outcome_text: 事例の結果記述
    yao_descriptions: その卦の6爻の説明
    """
    # 6爻分解モデルの「どの爻が変爻か」を推定
    # または、結果記述から段階を判定

    # MVP: 3段階（early/mid/late）→爻への対応
    stage = classify_stage(outcome_text)
    if stage == 'early':
        return random.choice([1, 2])  # または詳細判定
    elif stage == 'mid':
        return random.choice([3, 4])
    else:
        return random.choice([5, 6])
```

## 評価指標

### 分布品質（soft label予測）

| 指標 | 定義 | 目標値なし（データから決定） |
|------|------|--------------------------|
| NLL | 負の対数尤度 | 低いほど良い |
| Brier Score | 確率予測の二乗誤差 | 低いほど良い |
| KL Divergence | モデル分布とsoft labelの距離 | 低いほど良い |

### 意思決定品質

| 指標 | 定義 |
|------|------|
| Auto率 | 自動確定した割合 |
| Auto時精度 | 自動確定時の期待コスト |
| Ask時Top-5再現率 | Ask時にTop-5に正解が含まれる率 |

### 目標値の決定方法

**恣意的な固定値は使用しない**

```python
def determine_threshold(oof_predictions, oof_labels, c_u_candidates):
    """OOFデータから最適なc_uを決定"""
    best_c_u = None
    best_utility = float('-inf')

    for c_u in c_u_candidates:
        # c_uでのauto/ask決定をシミュレート
        total_cost = 0
        for pred, label in zip(oof_predictions, oof_labels):
            action, result = decide_action(pred, c_u)
            if action == 'auto':
                total_cost += expected_cost_auto(pred)
            else:
                total_cost += c_u

        utility = -total_cost  # コスト最小化
        if utility > best_utility:
            best_utility = utility
            best_c_u = c_u

    return best_c_u
```

## 工数見積もり

| フェーズ | 内容 | 工数 |
|----------|------|------|
| **Phase 1: 設計** | | **32h** |
| - 6爻分解モデル設計 | アーキテクチャ、入出力 | 8h |
| - soft label集約設計 | β最適化ロジック | 8h |
| - クロスフィット設計 | 分割、OOF計算 | 8h |
| - 注釈ガイド作成 | 評点基準、例示 | 8h |
| **Phase 2: パイロット** | | **60h** |
| - 注釈者トレーニング | 3名×4h | 12h |
| - パイロット注釈 | 50件×3名 | 20h |
| - アイテム単位不確実性分析 | 要アジュディケーション判定 | 12h |
| - ガイド修正・再パイロット | | 16h |
| **Phase 3: ゴールドセット** | | **100h** |
| - 本注釈 | 250件×3名 | 60h |
| - アジュディケーション | 高不確実性アイテム | 24h |
| - soft label生成 | β最適化含む | 16h |
| **Phase 4: モデル構築** | | **48h** |
| - 6爻分解モデル実装 | 6本の二値分類器 | 16h |
| - クロスフィット実行 | OOF生成、校正 | 12h |
| - 閾値最適化 | c_u決定 | 8h |
| - Test評価 | 最終評価 | 12h |
| **合計** | | **240h** |

## MVP（最小実行可能設計）

1. **学習**: 6爻分解（6本の軽量モデル）+ soft label
2. **校正・閾値**: クロスフィットOOFのみで決定
3. **出力**: 分布 + auto/ask決定
4. **UI**: askのときTop-5と根拠（不確実な爻）を提示
5. **運用**: ユーザー確定をログして次回学習に回す

## フォールバック

| 条件 | 対応 |
|------|------|
| 6爻モデルの精度が低い | 上卦/下卦（8×8）分解に変更 |
| Auto率が低すぎる（<30%） | c_uを下げる（ユーザー関与を許容） |
| Ask時Top-5再現率が低い（<70%） | Top-10に拡大 |
| 注釈者間の不確実性が高すぎる | カテゴリ定義を簡素化 |
