# 重複解決ルール (Duplicate Resolution Rules)

作成日: 2026-01-15
目的: 同一性判定の基準を固定し、canonical化の土台を作る

---

## 1. 同一性判定の定義

### 1.1 完全同一（Exact Duplicate）
```
条件: target_name AND period が完全一致
対応: 1件を canonical、他を alias として統合
```

### 1.2 高類似（Near Duplicate）
```
条件: 以下のいずれかを満たす
  - target_name の編集距離 ≤ 3 AND period が重複
  - target_name の類似度 ≥ 0.8 AND story_summary の類似度 ≥ 0.7
対応: 人的レビュー後、統合または差別化
```

### 1.3 同一企業・異フェーズ（Same Entity, Different Phase）
```
条件: target_name が同一 AND period が重複しない
対応: 別事例として許容（canonical_id は別）
```

### 1.4 表記揺れ（Name Variation）
```
例:
  - 「日本航空」「JAL」「日本航空（JAL）」
  - 「ソニー」「ソニーグループ」「Sony」
対応: alias_names フィールドで統一、canonical_name を決定
```

---

## 2. Canonical選定ルール

### 2.1 優先順位（同一グループ内で1件を canonical に選定）

| 優先度 | 条件 |
|--------|------|
| 1 | credibility_rank が最高（S > A > B > C）|
| 2 | source_url が存在する |
| 3 | logic_memo が最も詳細（文字数）|
| 4 | 登録日が最新 |

### 2.2 属性衝突時の解決

| フィールド | 衝突時の対応 |
|------------|--------------|
| story_summary | canonical の値を採用 |
| before_hex, after_hex 等 | canonical の値を採用（他は metadata に保存）|
| free_tags | 全 alias から統合（重複排除）|
| source_url | 全 alias から収集（配列化）|

---

## 3. データモデル拡張

### 3.1 追加フィールド
```json
{
  "canonical_id": "CAN_001",        // クラスタの代表ID
  "is_canonical": true,             // この事例が代表か
  "alias_ids": ["CORP_JP_123", "CORP_JP_456"],  // 統合された事例ID
  "alias_names": ["日本航空", "JAL", "日本航空（JAL）"],  // 表記揺れ
  "merged_sources": [               // 統合された出典
    {"url": "...", "from_id": "CORP_JP_123"},
    {"url": "...", "from_id": "CORP_JP_456"}
  ]
}
```

### 3.2 分析時の集計ルール
```
- 薄いセル判定: canonical のみをカウント
- 統計分析: canonical のみを対象
- 検索: canonical + alias_names でヒット
```

---

## 4. 入口ゲート仕様

### 4.1 新規追加時のフロー
```
1. target_name + period で完全一致検索
   → 一致あり: 「重複です。alias として追加しますか？」

2. target_name で類似検索（閾値 0.8）
   → 類似あり: 「類似事例があります: XXX。同一ですか？」

3. story_summary で類似検索（閾値 0.7）
   → 類似あり: 「内容が類似する事例があります。確認してください」

4. 全チェック通過: 新規 canonical として追加
```

### 4.2 alias 追加時の処理
```
- 新規事例の transition_id を alias_ids に追加
- merged_sources に出典を追加
- free_tags を統合
- is_canonical = false で保存
```

---

## 5. 移行計画

### Phase 1: 完全一致重複の自動クラスタ化
- 対象: target_name + period が完全一致する 1,923 グループ
- 処理: 自動で canonical_id 付与、canonical 選定

### Phase 2: 高類似重複の半自動クラスタ化
- 対象: 類似度 ≥ 0.8 の 1,139 グループ
- 処理: 候補提示 → 人的確認 → クラスタ化

### Phase 3: 表記揺れの正規化
- 対象: 同一企業の異なる表記
- 処理: alias_names マスター作成、canonical_name 統一

---

## 6. 成功指標

| 指標 | 目標 |
|------|------|
| 完全一致重複 | 0件（全て canonical 化済み）|
| canonical 事例数 | 約 11,000件（現在 13,022 - 重複分）|
| 薄いセル（canonical ベース）| 再計算後に確定 |
| 入口ゲート通過率 | 新規追加の 95% 以上が新規 canonical |
