# 品質改善計画 v2（Codex批評を反映）

## 現行の問題点（認識済み）

1. Gold件数目標は品質指標として不適切
2. ドメイン≠信頼性（同一ドメイン内変動を無視）
3. tier4_corporateの利益相反（COI）問題
4. n=10の信頼性基準は統計的に不十分
5. 正規表現111本は運用負債

## 改善計画

### Phase A: 監査基盤構築（1週間）

**目的**: KPIを件数から精度に再定義

**実装**:
1. Gold事例から100件を無作為抽出
2. 各事例を以下の軸でラベル付け:
   - 情報種別: 一次/二次/意見
   - COI: あり/なし/不明
   - 検証可能性: URL有効/404/paywall
   - 事実正確性: 正確/軽微誤り/重大誤り
3. 不適合率を算出: 不適合 = (重大誤り + COIあり×評価主張)

**KPI変更**:
- 旧: Gold ≥ 1,100件
- 新: Gold precision ≥ 90%（不適合率 ≤ 10%）

**成果物**:
- scripts/quality/audit_sampling.py
- data/audit/gold_audit_100.jsonl
- docs/audit_methodology.md

### Phase B: 二軸化（信頼度×COI）（1週間）

**目的**: tier4_corporateの利益相反問題を解決

**実装**:
1. 新フィールド追加:
   ```python
   {
     "source_reliability": "high/medium/low",  # 情報源の信頼性
     "coi_flag": "none/self/affiliated/unknown",  # 利益相反
     "claim_type": "fact/evaluation/opinion"  # 主張の種類
   }
   ```

2. 分類ルール改訂:
   - 企業公式(tier4) + fact主張 → Gold候補
   - 企業公式(tier4) + evaluation主張 → Silver（第三者裏取り必要）
   - メディア(tier2) + any主張 → Gold候補

3. Gold判定条件:
   ```
   Gold = (reliability=high) AND
          (coi=none OR (coi=self AND claim=fact))
   ```

**成果物**:
- scripts/quality/coi_classifier.py
- domain_rules_v2.py（二軸対応）

### Phase C: 統計改善（階層ベイズ）（2週間）

**目的**: n不足による広い信頼区間を解消

**実装**:
1. 階層モデル構造:
   ```
   全体平均 μ
     ├─ 卦レベル μ_hex[i] ~ N(μ, σ_hex)
     │    └─ 爻レベル μ_hex_yao[i,j] ~ N(μ_hex[i], σ_yao)
     └─ パターンレベル μ_pattern[k] ~ N(μ, σ_pattern)
   ```

2. 部分プーリング効果:
   - n=3のセルも、同一卦の他爻から情報を借りて推定
   - 信頼区間が極端に広くならない

3. ゲート条件:
   - 旧: n ≥ 10 で信頼性フラグ
   - 新: 95% CI下限 ≥ 0.3 で表示可能

**成果物**:
- scripts/quality/hierarchical_success_level.py
- data/analysis/success_rate_hierarchical.json

### Phase D: 収集制御（2週間）

**目的**: 日本偏重76%を解消

**実装**:
1. 言語別クオータ設定:
   | 言語/地域 | 目標比率 | 現状 | 必要追加 |
   |----------|---------|------|---------|
   | 日本 | 55% | 76% | 削減不要 |
   | 米国 | 25% | 13% | +400件 |
   | 欧州 | 12% | 3% | +300件 |
   | 中国 | 5% | 3% | +70件 |
   | その他 | 3% | 5% | 現状維持 |

2. 高信頼シードリスト（言語別）:
   - 英語: SEC, FCA, Reuters, Bloomberg, FT
   - ドイツ語: BaFin, Handelsblatt, FAZ
   - 中国語: CSRC, Caixin, 新华社

3. 収集禁止ルール:
   - クオータ超過言語からの追加禁止
   - 品質基準未達の言語は追加凍結

**成果物**:
- docs/collection_quota_policy.md
- scripts/collection/quota_checker.py

### Phase E: ルール縮退（1週間）

**目的**: 正規表現111本の運用負債を削減

**実装**:
1. ドメインをメタデータ化:
   ```json
   {
     "domain": "nikkei.com",
     "tier": 2,
     "language": "ja",
     "coi": "none",
     "source_type": "news",
     "last_verified": "2026-01-01"
   }
   ```

2. 正規表現を削減:
   - 政府系: `.go.jp`, `.gov` 等は維持（安定）
   - 企業系: 個別regex → メタデータテーブル参照
   - 目標: 111本 → 30本以下

**成果物**:
- data/metadata/domain_registry.json
- domain_rules_v3.py（メタデータ参照型）

## 実装順序

```
Week 1: Phase A（監査基盤）
Week 2: Phase B（二軸化）
Week 3-4: Phase C（階層ベイズ）
Week 5-6: Phase D（収集制御）
Week 7: Phase E（ルール縮退）
```

## 新KPI

| 指標 | 目標 | 測定方法 |
|------|------|----------|
| Gold precision | ≥ 90% | 四半期監査100件抽出 |
| 日本比率 | ≤ 60% | 収集時クオータ制御 |
| CI下限平均 | ≥ 0.4 | 階層モデル推定 |
| ルール数 | ≤ 30本 | regex count |

## リスクと対策

| リスク | 対策 |
|--------|------|
| 監査コスト増 | 自動化可能な項目を先に実装 |
| 階層モデル実装難 | PyMC/Stan使用、既存実装参照 |
| 非日本語ソース不足 | シードリストから段階的拡張 |
