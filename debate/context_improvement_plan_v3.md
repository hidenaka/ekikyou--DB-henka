# 品質改善計画 v3（Codex批評4点を完全反映）

## 設計原則（v2からの根本変更）

| 項目 | v2（否定された設計） | v3（採用する設計） |
|------|---------------------|-------------------|
| 評価単位 | 記事/ドメイン | **主張（claim）単位** |
| 信頼性判定 | ドメイン固定 | **ソース単位（site×section×author_type）** |
| 階層ベイズ | 表示ゲート（CI下限≥0.3） | **推定＋監査優先度付け** |
| クオータ | 禁止ルール | **重み付けペナルティ** |
| Gold定義 | 高信頼ドメイン由来 | **検証可能な事実主張のみ** |

---

## Phase A: 主張単位の監査基盤（2週間）

### 目的
評価単位を「記事」から「主張（claim）」に移行し、ラベル体系を統一

### 設計

1. **主張抽出パイプライン**:
   ```
   記事 → 文分割 → 主張候補抽出 → 主張タイプ分類 → 根拠紐付け
   ```

2. **統一ラベル体系**（Phase A/Bの重複を解消）:
   ```python
   claim_schema = {
       "claim_type": [
           "verifiable_fact",    # 検証可能な事実（日付、数値、出来事）
           "inference",          # 推論（～と考えられる）
           "evaluation",         # 評価（良い/悪い）
           "opinion"             # 意見（～すべき）
       ],
       "evidence_type": [
           "primary_source",     # 一次資料（公式発表、決算資料）
           "secondary_source",   # 二次資料（報道、解説）
           "no_evidence"         # 根拠なし
       ],
       "verification_status": [
           "verified",           # 検証済み
           "verifiable",         # 検証可能（未検証）
           "unverifiable"        # 検証不能（paywall等）
       ]
   }
   ```

3. **Gold判定条件**:
   ```
   Gold = (claim_type = "verifiable_fact") AND
          (evidence_type IN ["primary_source", "secondary_source"]) AND
          (verification_status IN ["verified", "verifiable"]) AND
          (coi = "none" OR (coi = "self" AND claim_type = "verifiable_fact"))
   ```

4. **監査プロトコル**:
   - 判定基準書（adjudication guide）作成
   - 二重ラベリング（2名独立判定）
   - 差分調停（disagreement resolution）プロセス
   - 四半期100主張（記事ではない）抽出

### 成果物
- `scripts/quality/claim_extractor.py`
- `docs/claim_labeling_guide.md`
- `data/audit/claim_audit_protocol.md`

---

## Phase B: ソース単位レジストリ（2週間）

### 目的
「ドメイン≠信頼性」問題を解決し、同一ドメイン内変動を吸収

### 設計

1. **ソース単位の定義**:
   ```python
   source_unit = {
       "domain": "nikkei.com",
       "path_prefix": "/business/",     # セクション識別
       "source_type": "news_article",   # 記事種別
       # source_type enum:
       # - news_article: 報道記事
       # - editorial: 社説・論説
       # - contributed: 寄稿記事
       # - ugc: ユーザー投稿
       # - pr: プレスリリース転載
       # - corporate_blog: 企業ブログ
   }
   ```

2. **信頼性スコア（ドメイン固定ではなく特徴量）**:
   ```python
   reliability_features = {
       "domain_tier": 0.3,           # 重み30%
       "source_type_score": 0.3,    # 重み30%
       "author_credibility": 0.2,   # 重み20%
       "historical_accuracy": 0.2   # 重み20%（過去監査実績）
   }
   # 最終スコア = Σ(feature × weight)
   ```

3. **COI判定ロジック**:
   ```python
   def determine_coi(source_unit, target_entity):
       # ドメインではなく主体関係で判定
       if is_same_corporate_group(source_unit.owner, target_entity):
           return "self"
       if has_business_relationship(source_unit.owner, target_entity):
           return "affiliated"
       if is_paid_content(source_unit):
           return "sponsored"
       return "none"
   ```

4. **レジストリ運用**:
   - `last_verified`に加え`verification_frequency`（変動リスクに応じた再検証頻度）
   - 方針変更検知（編集方針、買収等）をRSSフィードで監視
   - セクション・著者種別の自動分類器

### 成果物
- `data/metadata/source_registry.jsonl`（ドメイン→ソース単位に拡張）
- `scripts/quality/source_classifier.py`
- `scripts/quality/coi_detector.py`

---

## Phase C: 階層ベイズの適正運用（2週間）

### 目的
「表示ゲート」から「推定＋意思決定」への転換

### 設計

1. **モデル構造**（共役Beta-Binomialで運用容易に）:
   ```python
   # 階層Beta-Binomial（部分プーリング）
   # 全体: α_0, β_0
   # 卦レベル: α_hex[i], β_hex[i] ~ f(α_0, β_0)
   # 爻レベル: α_hex_yao[i,j] ~ f(α_hex[i], β_hex[i])

   # PyMC/Stanではなく共役モデルで実装
   # → 運用・説明・検証が容易
   ```

2. **出力形式**（ゲートではなく推定値＋不確実性）:
   ```python
   output = {
       "success_rate_mean": 0.65,      # 事後平均
       "success_rate_std": 0.12,       # 事後標準偏差
       "credible_interval_80": [0.48, 0.79],
       "credible_interval_95": [0.42, 0.85],
       "effective_sample_size": 8.3,   # プーリング後の実効n
       "shrinkage_factor": 0.35        # 全体平均への縮小度
   }
   ```

3. **意思決定は損失関数で**:
   ```python
   def decision_rule(posterior, action):
       """表示/非表示をCI下限ではなく期待損失で決定"""
       if action == "display":
           # 誤情報表示の損失（精度に影響）
           loss = expected_false_positive_loss(posterior)
       elif action == "hide":
           # 有用情報非表示の損失（カバレッジに影響）
           loss = expected_false_negative_loss(posterior)
       return loss

   # 損失関数はビジネス要件から定義
   # 例: 誤情報損失 = 3 × 非表示損失
   ```

4. **監査優先度付け**:
   ```python
   audit_priority = (
       uncertainty_score * 0.4 +      # 不確実性が高い
       impact_score * 0.3 +           # 影響が大きい（参照頻度等）
       staleness_score * 0.3          # 最終検証からの経過時間
   )
   ```

### 成果物
- `scripts/quality/hierarchical_beta_binomial.py`
- `scripts/quality/decision_loss_function.py`
- `scripts/quality/audit_prioritizer.py`

---

## Phase D: 重み付け多言語化（1週間）

### 目的
硬い禁止ルールではなく、品質を落とさない多言語化

### 設計

1. **多目的最適化**:
   ```python
   total_score = (
       quality_score * w_quality +           # 品質（監査精度）
       diversity_score * w_diversity +       # 多様性（言語分布）
       recency_score * w_recency            # 鮮度（更新頻度）
   )

   # 重み例: w_quality=0.6, w_diversity=0.3, w_recency=0.1
   ```

2. **言語偏り是正**（禁止ではなくペナルティ）:
   ```python
   def language_penalty(current_ratio, target_ratio):
       """超過言語にはペナルティ、不足言語にはボーナス"""
       deviation = current_ratio - target_ratio
       if deviation > 0:
           return -penalty_coefficient * deviation  # ペナルティ
       else:
           return bonus_coefficient * abs(deviation)  # ボーナス
   ```

3. **収集判定**:
   ```python
   def should_collect(source, claim):
       """禁止ではなく総合スコアで判定"""
       base_score = quality_score(source, claim)
       lang_adjustment = language_penalty(source.language)
       final_score = base_score + lang_adjustment
       return final_score >= threshold
   ```

4. **調達・コンプライアンス**:
   - 非日本語ソースの契約・API・課金を明示的に計画
   - 転載制限の法的確認プロセス

### 成果物
- `scripts/quality/multi_objective_scorer.py`
- `docs/international_source_procurement.md`

---

## Phase E: ガバナンス体制（1週間）

### 目的
「誰がどう責任を持って更新し、どう逸脱を処理するか」を定義

### 設計

1. **役割定義**:
   | 役割 | 責任 | 頻度 |
   |------|------|------|
   | 監査責任者 | 四半期監査の実施・品質保証 | 四半期 |
   | レジストリ管理者 | ソース登録・更新・無効化 | 週次 |
   | 分類器管理者 | claim_type/COI分類の精度監視 | 月次 |
   | 統計モデル管理者 | 階層モデルの再学習・検証 | 四半期 |

2. **逸脱処理プロセス**:
   ```
   逸脱検知 → 影響評価 → 是正措置 → 根本原因分析 → 再発防止
   ```

3. **変更管理**:
   - レジストリ変更は承認フロー必須
   - 分類ルール変更はA/Bテストで検証
   - KPI変更は経営承認

### 成果物
- `docs/governance_charter.md`
- `docs/deviation_handling_process.md`

---

## 実装順序（現実的スケジュール）

```
Phase A (2週間): 主張単位の監査基盤
  - Week 1: 主張抽出器、ラベル体系設計
  - Week 2: 監査プロトコル、二重ラベリング体制

Phase B (2週間): ソース単位レジストリ
  - Week 3: レジストリスキーマ、移行スクリプト
  - Week 4: COI判定器、source_type分類器

Phase C (2週間): 階層ベイズ適正運用
  - Week 5: Beta-Binomial実装、損失関数定義
  - Week 6: 監査優先度付け、ダッシュボード

Phase D (1週間): 重み付け多言語化
  - Week 7: 多目的スコアラー、調達計画

Phase E (1週間): ガバナンス体制
  - Week 8: ガバナンス文書、運用開始
```

**合計: 8週間**（v2の7週間から1週間増だが現実的）

---

## 新KPI

| 指標 | 目標 | 測定方法 | 備考 |
|------|------|----------|------|
| Gold precision | ≥ 90% | 四半期監査100主張 | **主張単位** |
| 監査一致率 | ≥ 85% | 二重ラベリング | 再現性指標 |
| 有効CI幅平均 | ≤ 0.3 | 階層モデル | 不確実性制御 |
| 言語偏差 | ≤ 15% | 目標比率との乖離 | **禁止ではなく偏差** |
| ルール数 | ≤ 50件 | ソース単位登録 | ドメイン→ソース |

---

## v2との差分まとめ

| 変更点 | v2 | v3 | 理由 |
|--------|----|----|------|
| 評価単位 | 記事 | 主張 | 1記事にfact/opinion混在 |
| 信頼性基準 | ドメイン | ソース単位 | 同一ドメイン内変動 |
| 階層ベイズ用途 | 表示ゲート | 推定＋優先度 | 品質隠蔽防止 |
| クオータ | 禁止 | 重み付け | 運用破綻防止 |
| Gold定義 | 高信頼ドメイン | 検証可能事実 | 意見混入防止 |
| 監査 | 100件スクリプト | 二重ラベリング | 再現性確保 |
| ガバナンス | なし | 明示的定義 | 運用継続性 |
| 期間 | 7週間 | 8週間 | 現実的見積 |

---

## リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| 主張抽出の精度不足 | Gold汚染 | 保守的閾値＋人手確認 |
| 二重ラベリングコスト | 工数増 | 高不確実性のみ対象 |
| ソース単位の粒度過剰 | 管理負荷 | 優先度付けで段階的拡張 |
| 損失関数の設計困難 | 意思決定品質 | ビジネス側との合意形成 |
| 非日本語調達コスト | 予算超過 | ROI計算後に段階実施 |
