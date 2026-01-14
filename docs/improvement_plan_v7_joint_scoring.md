# 六十四卦マッピング改善計画v7 - 同時スコアリング方式

## Codex v6批評からの修正点

| v6の問題 | v7の修正 |
|----------|----------|
| 上卦×下卦の独立合成 | 8×8同時スコア（ペア予測） |
| LLMスコア=確率と称する | 校正（calibration）を必須化 |
| 単一正解ラベル前提 | soft label（複数妥当卦の分布） |
| 常時Top-k提示 | 不確実時のみ追加質問 |
| 上卦=外部/下卦=内部の固定 | 視点に応じた柔軟な解釈 |
| 工数80hでゴールド400件 | 注釈設計工程を現実的に |

## 設計思想

### 核心原則

1. **同時スコアリング**: 上卦・下卦を独立に推定するのではなく、(上,下)ペアの64通りを同時にスコア
2. **soft label**: 1事例に対し複数の妥当卦が並立することを前提
3. **校正された確率**: スコアを確率と呼ぶなら、校正検証を通過したもののみ
4. **条件付きユーザー関与**: 不確実時のみ追加質問、確実時は自動確定

## アーキテクチャ

```
入力: ビジネス事例テキスト
    ↓
[Stage 1] 候補生成（Retrieval）
    - 事例DBから類似事例を検索
    - 上位20-50件の卦分布を集計
    - 初期候補セットを生成
    ↓
[Stage 2] 同時スコアリング（Joint Scoring）
    - LLMで64卦の同時スコア（8×8ロジット行列）
    - 類似検索スコアと統合
    - 温度スケーリングで校正
    ↓
[Stage 3] 不確実性判定
    - エントロピー/Top-1マージンを計算
    - 確実 → 自動確定
    - 不確実 → ユーザー関与へ
    ↓
[Stage 4] 条件付きユーザー関与（不確実時のみ）
    - Top-k候補提示
    - 本質八卦フィルタ（15卦絞り込み）
    - 追加質問による判別
    ↓
[Stage 5] 64卦確定 → 爻判定
    - 確定した卦の爻辞を参照
    - 卦固有の文脈で爻を判定

出力: (hexagram, yao, confidence, reasoning, alternatives)
```

## Stage 2: 同時スコアリングの詳細

### 2.1 8×8ロジット行列

独立積（v6）:
```
Score(卦) = P(上卦) × P(下卦)  # 相関を無視、誤り
```

同時スコア（v7）:
```
Score(上i, 下j) = LogitMatrix[i][j]  # 64通りを直接スコア
```

### 2.2 LLMプロンプト設計

```
以下のビジネス事例について、64卦それぞれの該当度を評価してください。

[事例]
{before/transformation/outcome}

[出力形式]
8×8の該当度マトリクス（行=上卦、列=下卦）
各セルは0-100のスコア

      乾  坤  震  巽  坎  離  艮  兌
乾    XX  XX  XX  XX  XX  XX  XX  XX
坤    XX  XX  XX  XX  XX  XX  XX  XX
震    XX  XX  XX  XX  XX  XX  XX  XX
巽    XX  XX  XX  XX  XX  XX  XX  XX
坎    XX  XX  XX  XX  XX  XX  XX  XX
離    XX  XX  XX  XX  XX  XX  XX  XX
艮    XX  XX  XX  XX  XX  XX  XX  XX
兌    XX  XX  XX  XX  XX  XX  XX  XX

最も該当する上位5卦とその理由:
1. XX番 YY卦（上:ZZ/下:WW）: 理由...
2. ...
```

### 2.3 スコア統合

```python
def compute_final_scores(llm_logits, retrieval_scores, temperature=1.0):
    """
    llm_logits: 8×8のLLMスコア行列
    retrieval_scores: 類似検索からの64卦スコア
    temperature: 校正用温度パラメータ
    """
    # LLMスコアを64次元ベクトルに変換
    llm_flat = llm_logits.flatten()  # 64次元

    # 類似検索スコアと重み付き統合
    alpha = 0.7  # LLM重み（検証セットで最適化）
    combined = alpha * llm_flat + (1 - alpha) * retrieval_scores

    # 温度スケーリングで校正
    calibrated = combined / temperature

    # softmaxで確率化
    probs = softmax(calibrated)

    return probs
```

### 2.4 校正（Calibration）

**v6の問題**: スコアを「確率」と称したが校正なし

**v7の対策**:
1. 検証セットで温度パラメータを最適化
2. ECE（Expected Calibration Error）を測定
3. ECE ≤ 0.15 を合格基準に

```python
def calibrate_temperature(val_scores, val_labels):
    """検証セットで最適温度を探索"""
    best_temp = 1.0
    best_ece = float('inf')

    for temp in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        calibrated = softmax(val_scores / temp)
        ece = compute_ece(calibrated, val_labels)
        if ece < best_ece:
            best_ece = ece
            best_temp = temp

    return best_temp, best_ece
```

## Stage 3: 不確実性判定

### 3.1 不確実性指標

```python
def compute_uncertainty(probs):
    """確率分布から不確実性を計算"""
    # エントロピー（分布の平坦さ）
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(64)  # 一様分布のエントロピー
    normalized_entropy = entropy / max_entropy

    # Top-1マージン（1位と2位の差）
    sorted_probs = np.sort(probs)[::-1]
    margin = sorted_probs[0] - sorted_probs[1]

    return {
        'entropy': normalized_entropy,
        'margin': margin,
        'top1_prob': sorted_probs[0]
    }
```

### 3.2 判定ルール

```python
def should_involve_user(uncertainty):
    """ユーザー関与が必要か判定"""
    # 確実: Top-1確率が高く、マージンも大きい
    if uncertainty['top1_prob'] > 0.4 and uncertainty['margin'] > 0.2:
        return False  # 自動確定

    # 不確実: エントロピーが高い、またはマージンが小さい
    if uncertainty['entropy'] > 0.7 or uncertainty['margin'] < 0.1:
        return True  # ユーザー関与

    return False  # デフォルトは自動確定
```

## Stage 4: 条件付きユーザー関与

### 4.1 不確実時のみTop-k提示

```
【システム判定】この事例は複数の卦が拮抗しています。
以下の候補から最も適切なものを選択してください。

【候補1】雷水解（40番）- 確率: 18%
  解釈: 困難な状況から突破口を開く
  類似事例: ○○社の危機対応（2023年）

【候補2】水雷屯（3番）- 確率: 15%
  解釈: 困難の中での新たな始まり
  ...

【絞り込みオプション】
- 「本質は"震"だと思う」→ 震が関わる15卦に絞り込み
- 追加質問に回答 → より精密な判定
```

### 4.2 本質八卦フィルタの活用

ユーザーの「本質八卦→15卦」アイデアをUIとして実装:

```python
def filter_by_essential_trigram(candidates, essential_trigram):
    """本質八卦でフィルタリング"""
    filtered = []
    for hexagram, score in candidates:
        upper, lower = get_trigrams(hexagram)
        if upper == essential_trigram or lower == essential_trigram:
            filtered.append((hexagram, score))
    return filtered  # 最大15卦
```

## ゴールドセット設計（soft label）

### 単一正解からの脱却

**v6の問題**: 1事例=1正解卦を前提

**v7の対策**: 複数専門家の投票分布をゴールドに

### アノテーション設計

```
1事例に対して:
- 3名の注釈者が独立に「妥当な卦」を選択（複数選択可）
- 各卦に対する妥当度（1-5）を評価
- soft label = 投票分布を正規化

例:
  事例A:
    注釈者1: 雷水解(5), 水雷屯(3)
    注釈者2: 雷水解(4), 雷天大壮(2)
    注釈者3: 雷水解(5), 水雷屯(4), 火雷噬嗑(2)

  soft label:
    雷水解: 0.45, 水雷屯: 0.25, 雷天大壮: 0.08, 火雷噬嗑: 0.07, ...
```

### 評価指標（soft label対応）

| 指標 | 定義 | 目標 |
|------|------|------|
| **KL Divergence** | モデル分布とsoft labelの距離 | ≤ 1.0 |
| **Top-k Overlap** | Top-k候補とsoft label上位の重なり | ≥ 70% |
| **nDCG@10** | soft labelを正解順位として計算 | ≥ 0.6 |
| ECE | 校正誤差 | ≤ 0.15 |
| 注釈者間一致 | Fleiss' κ | ≥ 0.4 |

## 工数見積もり（現実的）

### アノテーション設計が支配的

| フェーズ | 内容 | 工数 |
|----------|------|------|
| **Phase 1: 設計** | | **40h** |
| - アーキテクチャ設計 | 同時スコアリング、校正 | 8h |
| - プロンプト設計 | 8×8マトリクス出力 | 8h |
| - ルーブリック策定 | 64卦の判定基準 | 16h |
| - パイロット設計 | 注釈フロー、ツール | 8h |
| **Phase 2: パイロット** | | **60h** |
| - 注釈者トレーニング | 3名×4h | 12h |
| - パイロット注釈 | 50件×3名 | 24h |
| - 一致度分析・調整 | 不一致解消、ルーブリック修正 | 16h |
| - 再パイロット | 修正後の検証 | 8h |
| **Phase 3: ゴールドセット** | | **120h** |
| - 本注釈 | 300件×3名 | 72h |
| - 品質監査 | サンプル検証、一貫性チェック | 16h |
| - 調停 | 高不一致事例の解決 | 24h |
| - soft label生成 | 投票分布の計算 | 8h |
| **Phase 4: モデル構築** | | **40h** |
| - 実装 | 同時スコアリング、校正 | 24h |
| - 評価・調整 | 温度最適化、閾値設定 | 16h |
| **合計** | | **260h** |

## 評価プロトコル

### データ分割

- 訓練: 60%（180件）- 温度最適化、重み調整
- 検証: 20%（60件）- 閾値決定、ハイパラ選択
- テスト: 20%（60件）- 最終評価（1回のみ使用）

### 報告指標

1. **ランキング品質**: nDCG@5, nDCG@10, MRR
2. **分布品質**: KL Divergence, Top-k Overlap
3. **校正品質**: ECE, Brier Score
4. **注釈品質**: Fleiss' κ, 注釈者間一致率
5. **運用品質**: 自動確定率, ユーザー関与時の選択率

## フォールバック

| 条件 | 対応 |
|------|------|
| Fleiss' κ < 0.3 | ルーブリック再設計、八卦レベルに簡素化 |
| nDCG@10 < 0.4 | 類似検索の重みを上げる、LLM依存を下げる |
| 自動確定率 < 50% | 不確実性閾値を緩和 |
| ECE > 0.25 | 温度スケーリング以外の校正手法を検討 |

## v6からの主な変更

| 項目 | v6 | v7 |
|------|-----|-----|
| スコアリング | 上卦×下卦独立積 | 8×8同時スコア |
| 正解定義 | 単一ラベル | soft label（複数妥当卦） |
| 確率校正 | なし | 温度スケーリング+ECE検証 |
| ユーザー関与 | 常時Top-k | 不確実時のみ |
| 評価指標 | 精度・再現率 | KL/nDCG/ECE/κ |
| 工数 | 156h | 260h |
