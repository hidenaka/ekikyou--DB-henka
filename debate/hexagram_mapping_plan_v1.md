# 六十四卦384爻マッピング実装計画 v1

## 目的
12,871件の事例を六十四卦・384爻で表現し、「どの状況でどのような結果になりやすいか」の傾向分析を可能にする。

## 設計原則

### 原則1: 循環論法の回避
- **卦の決定**: trigger/action/pattern_type/scale/段階から決定
- **outcomeの扱い**: 卦決定の入力には使わない。観測値として傾向分析に使用

### 原則2: 複数値・正規化
- trigger/actionは配列（複数対応）
- 正規化辞書でID管理（表記揺れ防止）

### 原則3: 根拠の保持
- 元テキストからのspan（引用）を保持
- 抽出バージョン・信頼度を記録

---

## Phase 0: スキーマ設計

### 新規フィールド構造
```json
{
  "case_id": "CORP_JP_001",
  "triggers": [
    {
      "trigger_id": "T_EXT_SHOCK",
      "label": "外部経済ショック",
      "subcategory": "金融危機",
      "span": "リーマンショックの影響で",
      "order": 1,
      "confidence": 0.95
    }
  ],
  "actions": [
    {
      "action_id": "A_RETREAT",
      "label": "事業撤退",
      "subcategory": "不採算事業",
      "span": "不採算事業の撤退",
      "order": 1,
      "confidence": 0.90
    },
    {
      "action_id": "A_FOCUS",
      "label": "事業集中",
      "subcategory": "コア事業",
      "span": "社会イノベーション事業への集中",
      "order": 2,
      "confidence": 0.88
    }
  ],
  "change_stage": "recovery",  // initiation/development/crisis/recovery/maturation
  "hexagram": {
    "number": 24,
    "name": "地雷復",
    "reading": "ちらいふく",
    "upper_trigram": "坤",
    "lower_trigram": "震",
    "yao_position": 1,
    "yao_type": "yang_in_yin",
    "mapping_basis": {
      "pattern_type": "Shock_Recovery",
      "primary_trigger": "T_EXT_SHOCK",
      "primary_action": "A_RETREAT",
      "stage": "recovery"
    }
  },
  "outcome": "Success",  // 観測値（卦決定には不使用）
  "extraction_meta": {
    "version": "v1.0",
    "extracted_at": "2026-01-14",
    "model": "claude-3-opus"
  }
}
```

---

## Phase 1: 正規化辞書作成

### trigger類型辞書（初期案）
| ID | label | subcategory例 |
|----|-------|--------------|
| T_EXT_SHOCK | 外部経済ショック | 金融危機/パンデミック/自然災害 |
| T_INT_CRISIS | 内部危機 | 不祥事/経営失敗/品質問題 |
| T_REG_CHANGE | 規制変更 | 規制強化/規制緩和/法改正 |
| T_MGMT_CHANGE | 経営者交代 | 創業者退任/外部招聘/世代交代 |
| T_MARKET_SHIFT | 市場変化 | 需要減少/競合参入/技術革新 |
| T_OPPORTUNITY | 機会発見 | 新市場/技術ブレイク/提携機会 |
| T_STAGNATION | 停滞・行き詰まり | 成長鈍化/官僚化/イノベーション欠如 |

### action類型辞書（初期案）
| ID | label | subcategory例 |
|----|-------|--------------|
| A_RETREAT | 事業撤退 | 不採算撤退/市場撤退/売却 |
| A_FOCUS | 事業集中 | コア集中/選択と集中 |
| A_REORG | 組織改革 | リストラ/分社化/統合 |
| A_INVEST | 投資・拡大 | M&A/設備投資/R&D強化 |
| A_ALLIANCE | 提携・協業 | 業務提携/JV/資本提携 |
| A_TRANSFORM | 事業転換 | ピボット/新規事業/DX |
| A_CULTURE | 意識改革 | 企業文化刷新/人材育成 |
| A_INACTION | 対応なし/遅延 | 意思決定遅延/現状維持 |

### change_stage（変化段階）
| ID | 説明 | 対応する爻 |
|----|------|----------|
| initiation | 始動・芽生え | 初爻(1) |
| development | 展開・発展 | 二爻(2)・三爻(3) |
| crisis | 危機・転換点 | 四爻(4) |
| recovery | 回復・再建 | 五爻(5) |
| maturation | 成熟・完成 | 上爻(6) |

---

## Phase 2: 卦マッピングロジック

### 八卦の意味定義
| 卦 | 象徴 | 状態 | pattern_type対応 |
|----|------|------|-----------------|
| 乾 | 天・剛健 | 強い推進力・拡大 | Steady_Growth, Breakthrough |
| 坤 | 地・柔順 | 受容・基盤固め | Endurance, Managed_Decline |
| 震 | 雷・動 | 衝撃・始動 | Shock_Recovery (初期) |
| 巽 | 風・入 | 浸透・適応 | Pivot_Success |
| 坎 | 水・険 | 困難・試練 | Crisis_Pivot, Failed_Attempt |
| 離 | 火・明 | 明晰・表出 | Breakthrough |
| 艮 | 山・止 | 停止・内省 | Stagnation, Quiet_Fade |
| 兌 | 沢・悦 | 交流・成果 | Pivot_Success (成功時) |

### 六十四卦への展開ロジック
```
上卦 = f(pattern_type, primary_action)
下卦 = f(primary_trigger, scale)
爻位 = f(change_stage)

例: 日立製作所
  pattern_type: Shock_Recovery → 震（動）要素
  trigger: 外部経済ショック → 坎（険）
  action: 撤退+集中 → 坤（受容）→ 艮（止）
  stage: recovery → 五爻

  → 下卦=震、上卦=坤 → 地雷復（24番）・五爻
```

### マッピング決定表（pattern_type → 主卦グループ）
| pattern_type | 主な卦グループ | 理由 |
|-------------|--------------|------|
| Steady_Growth | 乾為天(1)系 | 持続的上昇 |
| Slow_Decline | 坤為地(2)系 | 緩やかな下降 |
| Shock_Recovery | 地雷復(24)/雷地豫(16) | 衝撃からの回復 |
| Pivot_Success | 風雷益(42)/雷風恒(32) | 変化による成功 |
| Hubris_Collapse | 火雷噬嗑(21)/雷火豊(55) | 膨張と崩壊 |
| Crisis_Pivot | 水雷屯(3)/坎為水(29) | 困難からの脱出 |
| Failed_Attempt | 山水蒙(4)/水山蹇(39) | 困難・失敗 |
| Breakthrough | 天火同人(13)/火天大有(14) | 突破・成功 |
| Stagnation | 天地否(12)/地天泰(11) | 停滞と交流 |
| Endurance | 坤為地(2)/地山謙(15) | 忍耐・継続 |

---

## Phase 3: 実装ステップ

### Step 1: 辞書構築（100件パイロット）
1. 100件をランダム抽出
2. 人手でtrigger/action/stageをラベリング
3. 辞書を確定・拡張

### Step 2: LLM抽出（1,000件パイロット）
1. プロンプト設計（辞書ベースの選択式）
2. 1,000件で抽出実行
3. 精度評価・プロンプト改善

### Step 3: 全件適用
1. 12,871件に対してLLM抽出
2. 正規化・confidence付与
3. 不確実例（confidence < 0.7）のレビュー

### Step 4: 卦マッピング
1. マッピングルール実装
2. 全件に卦・爻を付与
3. 分布確認・偏り調整

### Step 5: 傾向分析検証
1. 卦×outcome のクロス集計
2. 「この卦のときSuccess率は何%」を算出
3. 分析価値の検証

---

## 工数見積

| Phase | 作業 | 見積時間 |
|-------|------|---------|
| 0 | スキーマ設計 | 4時間 |
| 1 | 辞書構築（100件） | 8時間 |
| 2-1 | LLM抽出プロンプト設計 | 4時間 |
| 2-2 | パイロット1,000件 | 2時間（実行） |
| 2-3 | 精度評価・改善 | 4時間 |
| 3 | 全件適用 | 4時間（実行） |
| 4 | 卦マッピング実装 | 6時間 |
| 5 | 傾向分析・検証 | 4時間 |
| **合計** | | **36時間** |

---

## リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| trigger/action抽出精度が低い | 卦マッピングが無意味に | 辞書を絞る、選択式プロンプト |
| 卦の偏り（特定卦に集中） | 分析価値が低下 | trigger×action×stageの組み合わせで分散 |
| story_summaryが短すぎる事例 | 抽出不能 | confidence低で別扱い |
| 八卦→六十四卦の規則が恣意的 | 再現性なし | 決定表を明文化、ルールベースで機械的に |

---

## 成功指標

| 指標 | 目標値 |
|------|--------|
| trigger抽出一致率（人手 vs LLM） | ≥ 80% |
| action抽出一致率 | ≥ 75% |
| 卦分布の均一性（max/min比） | ≤ 10:1 |
| 卦×outcomeの有意差 | p < 0.05 for ≥ 10卦 |
