# 六十四卦マッピング実装計画 v3.1

## v3からの修正点（Codex批評対応）

| v3の問題 | v3.1での対応 |
|----------|-------------|
| カテゴリ重複（非MECE） | **8カテゴリを再定義、判定優先順位を明示** |
| 25件は統計的に不安定 | **層化サンプリング64件（各カテゴリ8件）** |
| Step5が非現実的 | **頻度ベースサンプリング（最大100件）** |
| マスク処理が無効 | **「結果前テキスト」を別フィールドとして切り出し** |
| 注釈ガイドラインなし | **Phase 0でガイドライン確定必須** |
| trigger/action別κのみ | **卦レベル一致率を主要指標に追加** |

---

## Phase 0: 注釈ガイドライン（最優先）

### trigger類型辞書（8種類・MECE）

| ID | label | 定義 | 典型例 | 反例・除外 | 判定優先順位 |
|----|-------|------|--------|-----------|-------------|
| T_EXTERNAL_FORCE | 外部強制力 | 外部から強制的に変化を迫られた | リーマンショック、法規制、パンデミック | 市場変化（強制ではない）→T_ENVIRONMENT | 1（最優先） |
| T_ENVIRONMENT | 環境変化 | 外部環境が変化したが強制ではない | 市場縮小、競合参入、技術革新 | 強制的なもの→T_EXTERNAL_FORCE | 2 |
| T_INTERNAL_CRISIS | 内部危機 | 組織内部で問題が発生 | 不祥事、経営失敗、品質問題 | 外部要因→T_EXTERNAL_FORCE | 3 |
| T_LEADERSHIP | リーダー起因 | 経営者・リーダーの決断や交代 | 創業者退任、外部招聘、方針転換 | 組織全体の問題→T_INTERNAL_CRISIS | 4 |
| T_OPPORTUNITY | 機会発見 | 新たな可能性を発見・認識 | 新市場発見、技術ブレイク、提携機会 | 外部から強制された→T_EXTERNAL_FORCE | 5 |
| T_STAGNATION | 停滞・行き詰まり | 成長が止まり、変化が必要になった | 業績停滞、イノベーション欠如 | 危機的状況→T_INTERNAL_CRISIS | 6 |
| T_GROWTH_MOMENTUM | 成長の勢い | 成長・拡大の流れの中で | 事業拡大期、好調期の施策 | 停滞中→T_STAGNATION | 7 |
| T_INTERACTION | 外部との相互作用 | 他者との協調・競争が契機 | 提携打診、業界再編、M&A | 一方的な力→T_EXTERNAL_FORCE | 8 |

### 判定フロー（trigger）
```
1. 外部から強制されたか？ → Yes → T_EXTERNAL_FORCE
2. 外部環境が変化したか？ → Yes → T_ENVIRONMENT
3. 内部で危機が発生したか？ → Yes → T_INTERNAL_CRISIS
4. リーダーの判断/交代か？ → Yes → T_LEADERSHIP
5. 新たな機会を発見したか？ → Yes → T_OPPORTUNITY
6. 停滞・行き詰まりか？ → Yes → T_STAGNATION
7. 成長の勢いの中か？ → Yes → T_GROWTH_MOMENTUM
8. 上記以外 → T_INTERACTION
```

### action類型辞書（8種類・MECE）

| ID | label | 定義 | 典型例 | 反例・除外 | 判定優先順位 |
|----|-------|------|--------|-----------|-------------|
| A_RETREAT | 撤退・縮小 | 事業・市場から退く | 事業売却、市場撤退、人員削減 | 一時停止→A_PAUSE | 1 |
| A_FOCUS | 選択と集中 | 特定領域に資源を集中 | コア事業集中、ポートフォリオ整理 | 単なる撤退→A_RETREAT | 2 |
| A_TRANSFORM | 変革・転換 | 事業モデルや組織を大きく変える | DX、ピボット、組織改革 | 小さな改善→A_ADAPT | 3 |
| A_EXPAND | 拡大・投資 | 積極的に拡大・投資する | M&A、設備投資、新規参入 | 維持的投資→A_MAINTAIN | 4 |
| A_CONNECT | 協調・提携 | 外部と協力関係を構築 | 業務提携、JV、アライアンス | 買収→A_EXPAND | 5 |
| A_ADAPT | 適応・調整 | 環境に合わせて調整する | プロセス改善、効率化、微調整 | 大変革→A_TRANSFORM | 6 |
| A_MAINTAIN | 維持・継続 | 現状を維持し耐える | 持久戦、コスト管理、品質維持 | 何もしない→A_PAUSE | 7 |
| A_PAUSE | 静観・様子見 | 行動を控え、状況を見る | 意思決定延期、戦略的静観、無策 | 維持努力→A_MAINTAIN | 8 |

### 判定フロー（action）
```
1. 事業・市場から撤退したか？ → Yes → A_RETREAT
2. 特定領域に集中したか？ → Yes → A_FOCUS
3. 大きく変革・転換したか？ → Yes → A_TRANSFORM
4. 積極的に拡大・投資したか？ → Yes → A_EXPAND
5. 外部と提携・協力したか？ → Yes → A_CONNECT
6. 環境に適応・調整したか？ → Yes → A_ADAPT
7. 現状維持・耐久したか？ → Yes → A_MAINTAIN
8. 上記以外（静観・様子見） → A_PAUSE
```

### 八卦への対応
| trigger | 下卦 | action | 上卦 |
|---------|------|--------|------|
| T_GROWTH_MOMENTUM | 乾 | A_EXPAND | 乾 |
| T_STAGNATION | 坤 | A_MAINTAIN | 坤 |
| T_EXTERNAL_FORCE | 震 | A_TRANSFORM | 震 |
| T_ENVIRONMENT | 巽 | A_ADAPT | 巽 |
| T_INTERNAL_CRISIS | 坎 | A_PAUSE | 坎 |
| T_OPPORTUNITY | 離 | A_FOCUS | 離 |
| T_LEADERSHIP | 艮 | A_RETREAT | 艮 |
| T_INTERACTION | 兌 | A_CONNECT | 兌 |

---

## Phase 1: 結果前テキストの切り出し

### 問題
story_summaryに「成功」「回復」「破綻」等の結果記述が含まれると、抽出時にリークする

### 解決策
story_summaryから「結果前の記述」を切り出し、別フィールド`pre_outcome_text`を作成

```python
def extract_pre_outcome(story_summary: str) -> str:
    """結果記述より前の部分を抽出"""
    # 結果を示す接続詞・表現で分割
    RESULT_MARKERS = [
        'の結果', 'その結果', 'により', 'を経て', 'を果たした',
        '成功', '失敗', '回復', '崩壊', '破綻', '達成', 'V字',
        '黒字化', '上場廃止', '買収成立', '和解', '勝訴', '敗訴'
    ]

    earliest_pos = len(story_summary)
    for marker in RESULT_MARKERS:
        pos = story_summary.find(marker)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos

    # 結果マーカーより前の部分を返す（最低100文字は確保）
    if earliest_pos < 100:
        return story_summary[:200]  # 短すぎる場合は200文字
    return story_summary[:earliest_pos]
```

### 出力例
```
story_summary: "リーマンショックの影響で7873億円の赤字を計上。不採算事業の撤退と社会イノベーション事業への集中を行い、奇跡的なV字回復を果たした。"

pre_outcome_text: "リーマンショックの影響で7873億円の赤字を計上。不採算事業の撤退と社会イノベーション事業への集中を行い、"
```

---

## Phase 2: 層化サンプリング＋二重ラベリング

### サンプリング設計
- **総数**: 64件（各triggerカテゴリから8件）
- **二重ラベリング**: 全64件（100%）
- **目的**: 各カテゴリの一致度を個別に測定

### 層化抽出クエリ
```python
for trigger_id in TRIGGER_IDS:
    # 既存データからtrigger候補を推定（story_summaryの特徴語で仮分類）
    candidates = estimate_trigger_candidates(cases, trigger_id)
    samples.extend(random.sample(candidates, 8))
```

### 評価指標
| 指標 | 目標値 | 不合格時の対応 |
|------|--------|----------------|
| κ（各trigger） | ≥0.6 | 該当カテゴリの定義を修正 |
| κ（各action） | ≥0.6 | 該当カテゴリの定義を修正 |
| **卦一致率** | ≥60% | ガイドライン全体を見直し |
| 混同行列の偏り | 対角≥50% | 混同するペアを統合検討 |

---

## Phase 3: LLM抽出

### プロンプト設計
```
以下の事例について、trigger（きっかけ）とaction（対応行動）を選んでください。

【事例】
{pre_outcome_text}

【triggerの選択肢】
1. T_EXTERNAL_FORCE: 外部から強制的に変化を迫られた（法規制、経済危機、災害等）
2. T_ENVIRONMENT: 外部環境が変化した（市場変化、競合、技術革新等）
3. T_INTERNAL_CRISIS: 内部で問題が発生した（不祥事、経営失敗等）
4. T_LEADERSHIP: 経営者・リーダーの決断や交代
5. T_OPPORTUNITY: 新たな機会を発見・認識した
6. T_STAGNATION: 停滞・行き詰まりから変化が必要になった
7. T_GROWTH_MOMENTUM: 成長・拡大の流れの中での施策
8. T_INTERACTION: 他者との協調・競争が契機

【actionの選択肢】
1. A_RETREAT: 撤退・縮小
2. A_FOCUS: 選択と集中
3. A_TRANSFORM: 変革・転換
4. A_EXPAND: 拡大・投資
5. A_CONNECT: 協調・提携
6. A_ADAPT: 適応・調整
7. A_MAINTAIN: 維持・継続
8. A_PAUSE: 静観・様子見

【回答形式】
trigger: T_XXX
action: A_XXX
```

### 一致率評価
- 人手64件との一致率を計算
- **目標**: trigger一致率≥70%、action一致率≥65%

---

## Phase 4: 全件マッピング

### 実行
1. 全12,871件に対してpre_outcome_text生成
2. LLMでtrigger/action抽出
3. 決定表で卦番号・爻を付与

### 品質チェック
- 抽出失敗（該当なし等）の件数
- 各卦の分布（記録のみ、品質指標ではない）

---

## Phase 5: 適合度レビュー（修正版）

### 頻度ベースサンプリング
```python
def sample_for_review(hexagram_counts: dict) -> list:
    samples = []
    for hexagram, count in hexagram_counts.items():
        if count == 0:
            continue  # 空卦は対象外
        elif count <= 10:
            samples.extend(all_cases[hexagram])  # 希少卦は全件
        elif count <= 100:
            samples.extend(random.sample(cases[hexagram], 10))  # 10件
        else:
            samples.extend(random.sample(cases[hexagram], 5))  # 頻出卦は5件
    return samples  # 最大約200件
```

### 評価方法
- **評価者**: 2名以上
- **評価項目**: 「この事例にこの卦は妥当か？」（1-5段階）
- **評価者間一致**: Krippendorff's α ≥ 0.6

### 合格基準
- 平均適合度 ≥ 3.5/5.0
- 評価者間一致 α ≥ 0.6

---

## 成功指標（v3.1）

| フェーズ | 指標 | 目標値 |
|----------|------|--------|
| Phase 2 | κ（trigger各カテゴリ） | ≥0.6 |
| Phase 2 | κ（action各カテゴリ） | ≥0.6 |
| Phase 2 | **卦一致率** | ≥60% |
| Phase 3 | LLM trigger一致率 | ≥70% |
| Phase 3 | LLM action一致率 | ≥65% |
| Phase 5 | 適合度平均 | ≥3.5/5.0 |
| Phase 5 | 評価者間一致α | ≥0.6 |

---

## 工数見積（v3.1）

| Phase | 作業 | 見積時間 |
|-------|------|---------|
| 0 | ガイドライン確定 | 4時間 |
| 1 | pre_outcome_text切り出し実装 | 3時間 |
| 2-1 | 層化サンプリング64件 | 1時間 |
| 2-2 | 二重ラベリング64件×2名 | 8時間 |
| 2-3 | κ計算・混同行列分析 | 2時間 |
| 2-4 | ガイドライン修正（必要時） | 2時間 |
| 3-1 | プロンプト設計 | 2時間 |
| 3-2 | LLM抽出500件パイロット | 2時間 |
| 3-3 | 精度評価・修正 | 2時間 |
| 4 | 全件マッピング | 4時間 |
| 5 | 適合度レビュー（~200件） | 6時間 |
| **合計** | | **36時間** |

---

## v3→v3.1の主要変更まとめ

1. ✅ **MECE化**: カテゴリ定義を明確化、判定優先順位を追加
2. ✅ **注釈ガイドライン**: 判定フロー、典型例、反例を明示
3. ✅ **結果前テキスト切り出し**: マスクではなく切り出しでリーク対策
4. ✅ **層化サンプリング64件**: 各カテゴリから8件、全件二重ラベリング
5. ✅ **卦一致率を追加**: trigger/action別だけでなく合成結果で評価
6. ✅ **頻度ベースレビュー**: 「各卦10件」固定を廃止、実データに合わせる
7. ✅ **評価者間一致**: 複数評価者のKrippendorff's αを測定
