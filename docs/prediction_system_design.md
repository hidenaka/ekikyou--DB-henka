# 変化のロジック：予測システム設計書

## 目的

**循環論法を排除した、真の予測力を持つ易経ベースの変化予測システム**を構築する。

## 現状の問題点

1. **循環論法**: pattern_type（結果から命名）を使って予測している
2. **過学習**: 同じデータで調整と評価を行っている
3. **事後分類**: 結果を知った上での分類であり、予測ではない

## 解決アプローチ

### 1. 予測に使用する変数の厳格化

**使用可能（予測時点で判明している情報）**:
- `before_state`: 現在の状態
- `action_type`: 取ろうとしている行動
- `trigger_type`: きっかけの種類
- `scale`: 規模（企業/個人等）
- `before_hex`: 開始時の八卦

**使用不可（結果を含む情報）**:
- `pattern_type`: 結果から命名されている
- `after_state`: 結果の状態
- `outcome`: 結果そのもの
- `after_hex`: 結果の八卦

### 2. データ分割戦略

```
全データ (8449件)
├── 訓練セット (60%): モデル構築に使用
├── 検証セット (20%): ハイパーパラメータ調整
└── テストセット (20%): 最終評価（一度だけ使用）

+ 新規ケース: 継続的な予測精度の追跡
```

### 3. 予測モデルの階層

```
Level 1: ルールベース（爻位×行動マトリクス）
Level 2: 統計ベース（ベイズ推定）
Level 3: 機械学習（決定木/ランダムフォレスト）
```

---

## システムアーキテクチャ

### ディレクトリ構造

```
易経変化ロジックDB/
├── data/
│   ├── raw/
│   │   └── cases.jsonl           # 全ケース
│   ├── splits/
│   │   ├── train.jsonl           # 訓練データ
│   │   ├── validation.jsonl      # 検証データ
│   │   └── test.jsonl            # テストデータ
│   ├── predictions/
│   │   └── pending_predictions.jsonl  # 未確定の予測
│   └── models/
│       ├── rule_based_v1.json    # ルールベースモデル
│       └── stats_model_v1.json   # 統計モデル
├── scripts/
│   ├── split_data.py             # データ分割
│   ├── train_model.py            # モデル訓練
│   ├── predict.py                # 予測実行
│   ├── evaluate.py               # 評価
│   └── add_case.py               # 新規ケース追加
└── harness/
    └── prediction_workflow.py    # ワークフロー管理
```

### ワークフロー

```
[新規ケース入力]
      ↓
[before_state, action_type, trigger_type を入力]
      ↓
[爻位診断] → before_yao_position を推定
      ↓
[予測実行] → predicted_outcome を出力
      ↓
[予測を記録] → pending_predictions.jsonl に保存
      ↓
... 時間経過 ...
      ↓
[結果入力] → actual_outcome を入力
      ↓
[精度更新] → 予測精度統計を更新
```

---

## 予測ロジック詳細

### 純粋予測モデル（pattern_typeなし）

```python
def predict_outcome(before_state, action_type, trigger_type=None, scale=None):
    """
    結果を知らない状態での予測

    入力:
    - before_state: 現在の状態
    - action_type: 取ろうとしている行動
    - trigger_type: きっかけ（オプション）
    - scale: 規模（オプション）

    出力:
    - predicted_outcome: Success/PartialSuccess/Mixed/Failure
    - confidence: 確信度 (0.0-1.0)
    - reasoning: 予測理由
    """

    # Step 1: 爻位診断（before_stateのみから）
    yao_position = diagnose_yao_position(before_state)

    # Step 2: 行動適合性スコア
    compatibility = get_compatibility(yao_position, action_type)

    # Step 3: trigger_typeによる補正
    if trigger_type:
        compatibility = adjust_for_trigger(compatibility, trigger_type)

    # Step 4: スコアを予測に変換
    outcome = score_to_outcome(compatibility)

    # Step 5: 確信度計算（訓練データの分散から）
    confidence = calculate_confidence(before_state, action_type)

    return outcome, confidence, reasoning
```

### 確信度の計算

訓練データにおける同じ条件（before_state × action_type）での結果の一貫性から算出。

```python
def calculate_confidence(before_state, action_type):
    # 訓練データで同じ条件のケースを抽出
    similar_cases = get_similar_cases(before_state, action_type)

    if len(similar_cases) < 5:
        return 0.3  # データ不足で低確信度

    # 結果の分布を確認
    outcomes = Counter(c['outcome'] for c in similar_cases)
    most_common_pct = outcomes.most_common(1)[0][1] / len(similar_cases)

    # 一貫性が高いほど高確信度
    return most_common_pct
```

---

## 評価指標

### 主要指標

1. **精度 (Accuracy)**: 予測と実際の一致率
2. **近似精度**: 1段階差（Success↔PartialSuccess等）も含めた精度
3. **確信度キャリブレーション**: 確信度と実際の精度の相関

### 評価レポート

```
=== 予測精度レポート ===

【テストセット評価】
- 完全一致: XX%
- 近似一致: XX%
- 乖離: XX%

【条件別精度】
- before_state別
- action_type別
- 爻位別

【確信度別精度】
- 高確信度(>0.7): XX%
- 中確信度(0.4-0.7): XX%
- 低確信度(<0.4): XX%

【新規ケース追跡】
- 予測数: XX件
- 確定数: XX件
- 的中率: XX%
```

---

## 継続的改善サイクル

```
1. 新規ケースで予測
      ↓
2. 結果確定後に精度を更新
      ↓
3. 月次で精度レビュー
      ↓
4. 精度が低下した条件を特定
      ↓
5. 訓練データに追加して再学習
      ↓
1. に戻る
```
