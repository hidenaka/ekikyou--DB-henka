# 運用フェーズ実装計画

## 背景

Phase 0-3の技術基盤が完成。次は運用フェーズとして以下4項目を実施予定。

## 現状指標

| 指標 | 現状値 | 問題点 |
|------|--------|--------|
| Unknown率 | 88.5% | 検証済みが11.5%のみ |
| 評価セット | 500件抽出済み | 未検証 |
| Wikipedia | 1,384件 | 引用降下未実装 |
| 主張タグ | 0件 | スキーマのみ |

---

## アクション1: 評価セット検証（500件二重ラベリング）

### 目的
固定評価セットの人手検証でProper Scoring Ruleの基盤構築

### 実施内容
1. evaluation_set_500.jsonlを検証者に配布
2. reviewer_1, reviewer_2が独立にラベル付け
3. 不一致箇所をadjudicatorが調停
4. 完了後にECE/Brier Score計算

### 検証フィールド
- outcome_verified: correct/incorrect/unverifiable
- source_quality: primary/secondary/tertiary/unreliable
- factual_accuracy: accurate/minor_error/major_error/unverifiable
- coi_assessment: none/potential/clear/unknown

### 工数見積
- 1件あたり15分 × 500件 × 2名 = 250時間
- adjudication（20%不一致想定）: 100件 × 30分 = 50時間
- 合計: 約300時間

### 成果物
- verified_evaluation_set.jsonl
- inter_annotator_agreement.json
- calibration_metrics.json

---

## アクション2: Wikipedia引用抽出（MediaWiki API実装）

### 目的
Wikipediaを「ポインタ」として活用し、一次/二次情報源へ降下

### 実施内容
1. MediaWiki API (`action=parse`, `prop=externallinks`) で外部リンク取得
2. 取得URLをprimary/secondary/otherに分類
3. Wikipedia単独ソースの事例に降下結果を付与
4. pointer_only → has_primary/secondary への昇格

### 技術仕様
```python
API_ENDPOINT = "https://ja.wikipedia.org/w/api.php"
params = {
    "action": "parse",
    "page": "記事タイトル",
    "prop": "externallinks",
    "format": "json",
}
```

### 対象
- pointer_only事例: 1,291件
- Wikipediaソース: 1,384件

### 工数見積
- API実装: 8時間
- 分類ロジック: 4時間
- テスト・検証: 4時間
- 合計: 約16時間

### 成果物
- scripts/quality/wikipedia_reference_extractor.py
- data/enriched/wikipedia_descended_sources.jsonl

---

## アクション3: 主張タグ付けパイロット（100件）

### 目的
Phase 3スキーマの実運用検証、Core Claims概念の妥当性確認

### 実施内容
1. evaluation_set_500からGold/Silver各50件を選定
2. 手動で主張抽出・タグ付け
3. Core Claims特定
4. 集約則（is_gold_case）の検証

### タグ付けガイドライン
- 1事例あたり3-5主張を抽出
- Core Claimsは2-3件に限定
- claim_type, evidence, coiを付与

### 工数見積
- 1件あたり30分 × 100件 = 50時間

### 成果物
- data/pilot/claims_tagged_100.jsonl
- docs/claim_tagging_guidelines.md
- analysis/core_claims_validation.md

---

## アクション4: Unknown率削減（検証済み事例拡充）

### 目的
Unknown率88.5%を50%以下に削減

### 実施内容
1. unverified事例のソース品質を評価
2. 高品質ソース（tier1/tier2）を持つunverified事例を優先検証
3. trust_level=verifiedに昇格

### 対象選定基準
```python
priority = (
    (has_primary_source * 0.4) +
    (has_secondary_source * 0.3) +
    (outcome_clear * 0.2) +
    (recent_case * 0.1)
)
```

### 目標
- 現状: verified 1,481件 (11.5%)
- 目標: verified 6,000件 (50%)
- 必要追加: 約4,500件

### 工数見積
- 1件あたり10分 × 4,500件 = 750時間

### 成果物
- data/raw/cases_verified_expanded.jsonl
- metrics/unknown_rate_progress.json

---

## 優先度と依存関係

```
アクション1 (評価セット検証)
    ↓ 完了後
アクション3 (主張タグ付けパイロット) ← アクション1の知見を活用
    ↓ 並行可能
アクション2 (Wikipedia引用抽出) ← 独立して実施可能
    ↓
アクション4 (Unknown率削減) ← 1,2,3の知見を活用
```

## 推奨実施順序

1. **アクション2** (Wikipedia引用抽出) - 自動化可能、即効性あり
2. **アクション1** (評価セット検証) - KPI基盤として必須
3. **アクション3** (主張タグ付けパイロット) - アクション1と並行可能
4. **アクション4** (Unknown率削減) - 最も工数大、最後に実施

## リスク

| リスク | 影響 | 対策 |
|--------|------|------|
| 検証者リソース不足 | アクション1,3遅延 | 優先度付けで範囲縮小 |
| Wikipedia API制限 | アクション2停滞 | レート制限対応、キャッシュ |
| 主張タグ付け品質ばらつき | アクション3無効化 | 詳細ガイドライン、IAA測定 |
| Unknown率目標未達 | 統計信頼性低下 | 中間目標設定（30%→50%） |

## 成功指標

| アクション | 完了基準 |
|------------|----------|
| 1 | IAA ≥ 0.8, ECE計算完了 |
| 2 | 降下成功率 ≥ 30% |
| 3 | 集約則検証完了、ガイドライン確定 |
| 4 | Unknown率 ≤ 50% |
