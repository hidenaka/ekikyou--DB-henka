# 変化のロジック：予測ハーネス

## 概要

易経の爻位理論に基づいて、組織や個人の状態と行動から結果を予測するシステム。

## 真の予測精度

**テストセット精度: 71.3%** (95%信頼区間: 69.2% - 73.5%)

- 完全一致: 63.5%
- 近似一致: 7.8%
- 乖離: 28.7%

### 確信度別精度

| 確信度 | 精度 | サンプル数 |
|--------|------|-----------|
| 高(≥0.6) | 76.5% | 1022件 |
| 中(0.4-0.6) | 63.8% | 608件 |
| 低(<0.4) | 59.7% | 62件 |

**高確信度の予測は約77%の精度で当たる。**

---

## 使い方

### 1. 新規予測

```bash
cd 易経変化ロジックDB
python3 harness/prediction_workflow.py predict
```

対話形式で以下を入力:
- `before_state`: 現在の状態
- `action_type`: 取ろうとしている行動
- `target_name`: 対象名（オプション）

結果:
- 予測される結果（Success/PartialSuccess/Mixed/Failure）
- 確信度
- 確率分布

### 2. 未確定予測の確認

```bash
python3 harness/prediction_workflow.py pending
```

### 3. 結果の確定

結果が判明したら:

```bash
python3 harness/prediction_workflow.py confirm <予測ID> <実際の結果>
```

例:
```bash
python3 harness/prediction_workflow.py confirm abc123 Success
```

### 4. 精度統計の確認

```bash
python3 harness/prediction_workflow.py stats
```

---

## 入力値

### before_state（現在の状態）

| 値 | 爻位 | 説明 |
|----|------|------|
| 絶頂・慢心 | 6爻 | 頂点に達している、行き過ぎのリスク |
| 安定・平和 | 5爻 | 安定したリーダーポジション |
| 成長痛 | 3爻 | 成長の中で問題が発生 |
| 停滞・閉塞 | 4爻 | 膠着状態、次のステップが見えない |
| 混乱・カオス | 3爻 | 不安定、何が起きるかわからない |
| どん底・危機 | 1爻 | 最悪の状態、底からの出発 |
| 安定成長・成功 | 2爻 | 順調に成長中 |
| 成長・拡大 | 2爻 | 拡大フェーズ |

### action_type（行動タイプ）

| 値 | 説明 |
|----|------|
| 攻める・挑戦 | 投資、新規事業、拡大 |
| 守る・維持 | 防衛、コストカット、現状維持 |
| 捨てる・撤退 | 売却、撤退、閉鎖 |
| 耐える・潜伏 | 下積み、準備、表に出ない |
| 対話・融合 | 交渉、和解、統合、M&A |
| 刷新・破壊 | 大改革、ルール破壊 |
| 逃げる・放置 | 先送り、対応しない |
| 分散・スピンオフ | 分社化、副業、ポートフォリオ化 |

---

## ファイル構成

```
harness/
├── prediction_workflow.py  # メインワークフロー
└── README.md               # このファイル

data/
├── splits/
│   ├── train.jsonl         # 訓練データ (5068件)
│   ├── validation.jsonl    # 検証データ (1689件)
│   └── test.jsonl          # テストデータ (1692件)
├── models/
│   └── prediction_model_v1.json  # 訓練済みモデル
└── predictions/
    ├── pending_predictions.jsonl   # 未確定予測
    ├── confirmed_predictions.jsonl # 確定済み予測
    └── prediction_stats.json       # 精度統計
```

---

## 継続的改善

新規ケースで予測 → 結果確定 → 精度統計を追跡

精度が低下した条件を特定し、モデルを再訓練する。
