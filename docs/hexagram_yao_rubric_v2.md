# 卦・爻対応ルーブリック v2.0

作成日: 2026-01-15
更新日: 2026-01-15
目的: **内容品質の担保**（Codex指摘対応版）

---

## 変更履歴
- v1.0: 形式品質（重複排除、テンプレート、URL）
- v2.0: **内容品質追加**（一次テキスト根拠、反証欄、盲検レビュー、一致度計測、意味的重複）

---

## 1. 一次テキスト根拠の固定【新規】

### 1.1 必須フィールド
```json
{
  "primary_text_basis": {
    "source_type": "爻辞",           // 卦辞/爻辞/象伝/彖伝
    "original_text": "亢龍有悔",     // 原文（漢文）
    "interpretation": "高く上りすぎた龍は後悔する",  // 現代語訳
    "relevance": "WeWorkの過剰拡大と創業者追放に該当"  // 事例との対応説明
  }
}
```

### 1.2 根拠の優先順位
| 優先度 | ソース | 説明 |
|--------|--------|------|
| 1 | 爻辞 | 各爻の具体的状況記述 |
| 2 | 卦辞 | 卦全体の意味 |
| 3 | 象伝 | 象徴的解釈 |
| 4 | 彖伝 | 卦の成り立ち説明 |

### 1.3 禁止事項
- URLのみを根拠にすること
- 二次解説書のみを根拠にすること
- 原文引用なしで卦を割り当てること

---

## 2. 反証欄の義務化【新規】

### 2.1 必須フィールド
```json
{
  "hexagram_selection": {
    "selected_hexagram": 1,
    "selected_yao": 6,
    "candidate_hexagrams": [
      {
        "hexagram_id": 1,
        "yao": 6,
        "fit_score": 5,
        "reason": "亢龍有悔が過剰拡大による失敗に完全一致"
      },
      {
        "hexagram_id": 43,
        "yao": 6,
        "fit_score": 3,
        "reason": "夬（決断）の6爻も該当し得るが、決断というより慢心が主因"
      },
      {
        "hexagram_id": 23,
        "yao": 6,
        "fit_score": 2,
        "reason": "剥（崩壊）も検討したが、崩壊過程より頂点からの転落が本質"
      }
    ],
    "rejection_reasons": {
      "43": "決断の結果ではなく慢心の結果であるため",
      "23": "徐々な崩壊ではなく急激な転落であるため"
    }
  }
}
```

### 2.2 最低要件
- 候補卦: **最低3つ**を検討
- fit_score: 1-5で評価
- rejection_reasons: 不採用卦には必ず理由を記載

### 2.3 確認バイアス防止チェック
- [ ] 最初に思いついた卦以外も検討したか？
- [ ] 反対の意味を持つ卦も検討したか？
- [ ] 爻の位置は複数検討したか？

---

## 3. 盲検レビュープロセス【新規】

### 3.1 フェーズ分離
```
Phase A: 事例記述（卦情報なし）
  └── story_summary, before_state, trigger_type, action_type, after_state のみ記述

Phase B: 卦割当（事例情報のみ）
  └── Phase Aの記述を見て、独立に卦・爻を判定

Phase C: 照合・裁定
  └── Phase Bの判定結果を元記述者と照合し、不一致は裁定
```

### 3.2 運用ルール
- 記述者と判定者は**別人**が理想
- 同一人物の場合は**24時間以上**間隔を空ける
- 判定時に**元のlogic_memoを見ない**

### 3.3 裁定基準
| 一致度 | 対応 |
|--------|------|
| 卦・爻完全一致 | そのまま採用 |
| 卦一致・爻不一致 | 爻辞を再確認し裁定 |
| 卦不一致 | 第三者を交えて裁定 |

---

## 4. 一致度計測【新規】

### 4.1 計測指標
```python
# Cohen's Kappa係数
# κ = (Po - Pe) / (1 - Pe)
# Po: 観測一致率, Pe: 偶然一致率

# 目標値
卦レベル一致: κ ≥ 0.6 (substantial agreement)
爻レベル一致: κ ≥ 0.4 (moderate agreement)
```

### 4.2 記録フィールド
```json
{
  "review_metadata": {
    "primary_reviewer": "reviewer_A",
    "secondary_reviewer": "reviewer_B",
    "primary_judgment": {"hexagram": 1, "yao": 6},
    "secondary_judgment": {"hexagram": 1, "yao": 6},
    "agreement": true,
    "arbitration_needed": false,
    "final_decision": {"hexagram": 1, "yao": 6},
    "confidence": "high"
  }
}
```

### 4.3 未確定事例の扱い
- 一致率が低い事例は `confidence: "low"` としてマーク
- 低信頼事例は分析から除外可能にする

---

## 5. 意味的重複対策【新規】

### 5.1 重複の3レベル
| レベル | 定義 | 対応 |
|--------|------|------|
| 完全一致 | target_name + period が同一 | 即座に却下 |
| 高類似 | 同一企業・類似期間・類似イベント | 統合または差別化 |
| 意味的近接 | 異なる企業だが同一パターン | 粒度確認の上で許容 |

### 5.2 類似度チェック項目
```python
# 自動チェック項目
1. target_name の編集距離 (Levenshtein)
2. period の重複度
3. story_summary のコサイン類似度
4. 卦・爻の組み合わせ一致
5. pattern_type + outcome の一致
```

### 5.3 統合ルール
- 同一企業の異なるフェーズ → period で明確に分離
- 同一イベントの異なる視点 → 1件に統合し、視点をタグで区別
- 類似パターンの異なる企業 → 許容（むしろ望ましい）

---

## 6. 新テンプレート（v2対応）

### 6.1 完全版JSON構造
```json
{
  "target_name": "WeWork創業者アダム・ニューマン追放",
  "scale": "company",
  "period": "2019",
  "story_summary": "企業価値470億ドルから一転、IPO失敗後に創業者が追放された。",

  "before_state": "絶頂・慢心",
  "trigger_type": "内部崩壊",
  "action_type": "捨てる・撤退",
  "after_state": "混乱・カオス",

  "before_hex": "乾",
  "trigger_hex": "離",
  "action_hex": "艮",
  "after_hex": "坎",

  "pattern_type": "Hubris_Collapse",
  "outcome": "Failure",

  "source_type": "news",
  "credibility_rank": "A",
  "source_url": "https://www.wsj.com/...",

  "primary_text_basis": {
    "source_type": "爻辞",
    "original_text": "亢龍有悔",
    "interpretation": "高く上りすぎた龍は後悔する",
    "relevance": "過剰拡大と慢心による失敗"
  },

  "hexagram_selection": {
    "selected_hexagram": 1,
    "selected_yao": 6,
    "candidate_hexagrams": [
      {"hexagram_id": 1, "yao": 6, "fit_score": 5, "reason": "亢龍有悔が完全一致"},
      {"hexagram_id": 43, "yao": 6, "fit_score": 3, "reason": "夬も検討したが慢心が主因"},
      {"hexagram_id": 23, "yao": 6, "fit_score": 2, "reason": "剥も検討したが急激な転落"}
    ],
    "rejection_reasons": {
      "43": "決断の結果ではなく慢心の結果",
      "23": "徐々な崩壊ではなく急激な転落"
    }
  },

  "logic_memo": "【局面】...【関係性】...【選択】...【結果】...【一次テキスト根拠】亢龍有悔（乾卦上九）。【反証】夬・剥も検討したが、慢心からの転落という本質に最も合致するのは乾の6爻。",

  "yao_analysis": {
    "before_hexagram_id": 1,
    "before_yao_position": 6,
    "assigned_yao": 6
  }
}
```

---

## 7. 品質チェックリスト（v2）

| # | チェック項目 | 合格基準 |
|---|-------------|----------|
| 1 | 完全一致重複なし | 0件 |
| 2 | 一次テキスト引用あり | 100% |
| 3 | 候補卦3つ以上 | 100% |
| 4 | 排除理由記載 | 100% |
| 5 | ソースURL | 80%以上 |
| 6 | credibility A以上 | 60%以上 |
| 7 | 意味的重複チェック済 | 100% |

---

## 8. 適用範囲

- 本ルーブリックv2は2026-01-15以降の**新規収集**に適用
- 既存データへの遡及適用は**段階的に実施**
- v1準拠データは `rubric_version: 1` でマーク

