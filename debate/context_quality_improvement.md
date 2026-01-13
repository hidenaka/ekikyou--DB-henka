# 品質改善の現状コンテキスト

## 実施内容

### Phase 1-2: 三層分類システム
- Gold/Silver/Bronze/Quarantineの4層に分類
- ドメインホワイトリスト（tier1-5）で信頼性判定
- 正規表現パターン111個でドメイン分類

### Phase 4: success_level実測化
- ベイズ平滑化（α=1.0, prior=0.5）
- Wilson score intervalで95%信頼区間
- 最低n=10で信頼性フラグ

### 成果物
- domain_rules.py: ドメイン分類ルール
- test_quality_rules.py: 34テストPASS
- 仕様書3件: success_level_spec.md, gold_dropout_analysis.md, global_rebalance_plan.md

## 現在の指標

| 指標 | 値 | 目標 | 達成率 |
|------|-----|------|--------|
| Gold | 980件 | 1,100件 | 89% |
| verified→Gold | 66.2% | 75% | 88% |
| Gold+Silver | 3,150件 | 4,000件 | 79% |
| 日本比率 | ~76% | <70% | 未達 |

## 未解決課題

1. **Gold目標未達**: 980/1,100件（89%）
2. **日本偏重**: 76%のまま改善なし
3. **企業公式バイアス**: tier4_corporateの自己呈示問題
4. **細分セルのn不足**: 卦×パターンで信頼区間が広い

## 議論したい点

1. 現在のアプローチで目標達成は可能か？
2. 根本的なアーキテクチャ変更が必要か？
3. 日本偏重解消の現実的なロードマップは？
