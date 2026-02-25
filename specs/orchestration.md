# オーケストレーション規約 — 易経変化ロジック再構築

## 役割分担

| 役割 | 担当 | 行動空間 |
|------|------|---------|
| オーケストレーター | メインセッション | タスク分解・委託・品質判断・方針決定 |
| state-space-modeler | Task agent | 64卦の数理モデル構築 |
| change-analyzer | Task agent | 変化事例の構造化・次元導出 |
| isomorphism-checker | Task agent | 同型性検証 |
| quality-reviewer | Task agent | 品質レビュー |

## 通信プロトコル

### エージェント間通信
- エージェント同士は直接通信しない
- 全ての情報共有はファイルシステム経由（`analysis/` ディレクトリ）
- オーケストレーターが仲介（必要な入力ファイルをプロンプトで指示）

### 成果物の命名規則
```
analysis/
├── phase1/           # state-space-modeler の出力
│   ├── state_space_model.py
│   ├── graph_analysis.json
│   ├── report.md
│   ├── review.md     # quality-reviewer の出力
│   └── visualizations/
├── phase2/           # change-analyzer の出力
│   ├── mca_analysis.py
│   ├── mca_results.json
│   ├── dimension_report.json
│   ├── transition_matrix.py
│   ├── transition_stats.json
│   ├── clustering.py
│   ├── cluster_results.json
│   ├── report.md
│   ├── review.md     # quality-reviewer の出力
│   └── visualizations/
└── phase3/           # isomorphism-checker の出力
    ├── isomorphism_test.py
    ├── statistical_tests.json
    ├── report.md
    └── review.md     # quality-reviewer の出力
```

## 実行フロー

### Step 1: Phase 1 + Phase 2A 並行実行
```
[オーケストレーター]
    ├─→ Task(state-space-modeler): Phase 1 全体
    └─→ Task(change-analyzer): Phase 2A（基本統計 + MCA）
```

### Step 2: Phase 1 レビュー + Phase 2A レビュー
```
[オーケストレーター]
    ├─→ Task(quality-reviewer): Phase 1 レビュー
    └─→ Task(quality-reviewer): Phase 2A レビュー
```

### Step 3: Phase 2B（Phase 2A PASS後）
```
[オーケストレーター]
    └─→ Task(change-analyzer): Phase 2B（遷移分析 + クラスタリング）
```

### Step 4: Phase 2B レビュー
```
[オーケストレーター]
    └─→ Task(quality-reviewer): Phase 2B レビュー
```

### Step 5: Phase 3（Phase 1 AND Phase 2 全体 PASS後）
```
[オーケストレーター]
    └─→ Task(isomorphism-checker): Phase 3
```

### Step 6: Phase 3 レビュー
```
[オーケストレーター]
    └─→ Task(quality-reviewer): Phase 3 レビュー
```

## エラーハンドリング

### レビューFAIL時
1. quality-reviewerの指摘事項を確認
2. 該当エージェントを**新しいコンテキスト**で再起動
3. 指摘事項をプロンプトに含めて再実行
4. 2回目もFAILの場合、オーケストレーターが介入

### エージェントがNOT Allowed行動を取った場合
1. 該当出力を破棄
2. エージェントを新しいコンテキストで再起動
3. NOT Allowedルールを強調してプロンプトに含める

### 予期しない結果
- 次元数が極端（2以下 or 20以上）→ 変数選択を再検討
- グラフ性質が理論値と大きく乖離 → 卦番号↔ビット列変換を確認
- 同型性検定のp値が境界的 → サンプルサイズの影響を評価

## コスト管理
- 各エージェントのTask実行は独立したトークンコスト
- Phase 1 + Phase 2Aの並行実行で時間を短縮
- quality-reviewerは比較的軽量（既存成果物のチェックのみ）
