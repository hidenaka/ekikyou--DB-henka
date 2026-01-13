# 次のステップ実装計画

## 背景

v4品質分類実装後、以下の3課題を特定:
1. Bronze異常値（成功率9.5:1）
2. 統計KPIが不適切（CI幅ベース）
3. 主張単位評価への移行未着手

## Step 1: Bronze異常値の解決（短期・1週間）

### 原因分析結果

| 原因 | 件数 | 影響 |
|------|------|------|
| Wikipedia依存 | 554件 | サバイバーシップバイアス |
| unverified集中 | 976件 | 検証なしでSuccess判定 |
| google.com混入 | 20件 | 検索結果URL |

### 解決策

1. **Wikipediaの扱い見直し**
   ```python
   # 現状: tier3_specialist（Silver候補）
   # 提案: 参照情報源として別カテゴリ化
   REFERENCE_SOURCES = {
       'wikipedia.org': {
           'tier': 'reference',  # Gold/Silver対象外
           'use_case': 'context_only',  # 補足情報のみ
           'reliability': 'secondary',
       }
   }
   ```

2. **Bronze→Quarantine移動条件**
   - Wikipedia単独ソース → Quarantine
   - trust_level=None + unclassifiedドメイン → Quarantine
   - google.com/youtube.com等の非情報源 → rejected

3. **Success偏り是正**
   - Bronze事例に「要検証」フラグ追加
   - success_level計算からBronze除外オプション

### 期待効果
- Bronze: 1,873件 → ~800件（Wikipedia移動）
- 成功率正常化: 9.5:1 → ~2:1（推定）

---

## Step 2: 統計KPIの校正ベース化（中期・2週間）

### 現状の問題

| 現行KPI | 問題点 |
|---------|--------|
| CI幅 ≤ 0.3 | 強い事前分布で達成可能（ゲーム可能） |
| n ≥ 10 | 任意の閾値、統計的根拠なし |
| precision ≥ 90% | 定義が曖昧（何に対する精度か） |

### 新KPI設計

1. **校正（Calibration）指標**
   ```python
   # 予測確率と実際の成功率の一致度
   def calibration_error(predictions, actuals):
       """Expected Calibration Error (ECE)"""
       bins = np.linspace(0, 1, 11)
       ece = 0
       for i in range(len(bins) - 1):
           mask = (predictions >= bins[i]) & (predictions < bins[i+1])
           if mask.sum() > 0:
               avg_pred = predictions[mask].mean()
               avg_actual = actuals[mask].mean()
               ece += mask.sum() * abs(avg_pred - avg_actual)
       return ece / len(predictions)
   ```

2. **Proper Scoring Rule**
   ```python
   # Brier Score: 予測精度の総合評価
   def brier_score(predictions, actuals):
       return ((predictions - actuals) ** 2).mean()
   ```

3. **信頼性ダイアグラム**
   - 予測確率 vs 実績確率のプロット
   - 対角線からの乖離で校正誤差を可視化

### 新KPI一覧

| KPI | 目標 | 測定方法 |
|-----|------|----------|
| ECE (校正誤差) | ≤ 0.05 | 10bin校正 |
| Brier Score | ≤ 0.2 | 全事例 |
| AUC-ROC | ≥ 0.7 | 成功/失敗分類 |
| 有効サンプルサイズ | ≥ 5 | 階層モデル推定後 |

---

## Step 3: 主張単位評価への移行（長期・段階的）

### 移行戦略

現行の「事例単位」から「主張単位」への段階的移行:

```
Phase 0 (現状): 事例単位
  └─ 1事例 = 1 target + outcome + sources

Phase 1 (中間): 事例+主張タグ付け
  └─ 1事例 = 1 target + outcome + [主張1, 主張2, ...]
  └─ 主張 = {type, evidence, verified}

Phase 2 (最終): 主張単位
  └─ 1主張 = {claim_text, type, evidence, coi, verified}
  └─ 事例は主張の集約として再構成
```

### Phase 1 実装（事例+主張タグ）

1. **主張抽出フィールド追加**
   ```python
   case_v5 = {
       # 既存フィールド
       "target_name": "...",
       "outcome": "Success",
       "sources": [...],

       # 新規: 主張タグ
       "claims": [
           {
               "claim_id": "c001",
               "text": "2023年に売上1000億円達成",
               "type": "verifiable_fact",
               "evidence": "primary_source",
               "source_index": 0,  # sources[0]を参照
               "verified": True,
           },
           {
               "claim_id": "c002",
               "text": "業界で最も革新的な企業",
               "type": "evaluation",
               "evidence": "no_evidence",
               "verified": False,
           }
       ]
   }
   ```

2. **既存事例への後方互換**
   - claims未設定の事例は従来通り処理
   - claimsがある事例は主張単位で品質評価

3. **Gold判定の拡張**
   ```python
   # 事例単位（従来）
   Gold_legacy = verified AND high_tier AND no_coi

   # 主張単位（新）
   Gold_claim = (
       claim.type == "verifiable_fact" AND
       claim.evidence in ["primary", "secondary"] AND
       claim.verified == True AND
       claim.coi == "none"
   )

   # 事例のGold判定（主張ベース）
   Gold_case = all(Gold_claim for claim in case.claims if claim.type == "verifiable_fact")
   ```

### 移行スケジュール

```
Month 1: Phase 1設計・スキーマ確定
Month 2: 100事例パイロット（主張タグ付け）
Month 3: 主張抽出ツール開発
Month 4-6: 全事例への主張タグ付け（段階的）
Month 7+: Phase 2（主張単位評価）検討
```

---

## リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| Wikipedia除外でカバレッジ低下 | 統計精度低下 | 代替ソース収集強化 |
| 校正KPIの計算コスト | 運用負荷 | バッチ処理化 |
| 主張抽出の精度不足 | 品質劣化 | 人手確認併用 |
| 後方互換性の破壊 | 既存分析無効 | 段階移行・並行運用 |

---

## 実装優先度

```
高: Step 1 (Bronze解決) - 即座に品質数値が改善
中: Step 2 (校正KPI) - 品質評価の信頼性向上
低: Step 3 (主張単位) - 長期的な品質基盤
```

## 成功指標

| ステップ | 指標 | 目標 |
|----------|------|------|
| Step 1 | Bronze成功率 | 9.5:1 → 2:1以下 |
| Step 2 | ECE | 0.05以下 |
| Step 3 | 主張タグ付け率 | 50%以上（Phase 1完了時） |
