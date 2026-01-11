---
name: add-batch-cases
description: 実例事例をバッチで調査・追加する。ニュース検索→JSON作成→DBインポートのワークフロー。
---

# バッチ事例追加

## ワークフロー

1. **調査**: WebSearchで2024年の倒産・再建・業界動向ニュースを検索
2. **重複チェック**: 既存DBで同一target_name+periodが存在しないか確認
3. **JSON作成**: `data/import/real_cases_2024_batchXX.json` に5件程度作成
4. **追加**: `python3 scripts/add_batch.py data/import/real_cases_2024_batchXX.json`
5. **確認**: 追加件数を確認

## 対象優先順位

1. 中小企業、地方企業、老舗、町工場
2. 業界全体の動向（○○業界 2024年 倒産）
3. 個人の事例（アスリート引退後、芸能人転身等）

## 避けるべき対象

- 有名大企業（すでに収録済み多数）
- 芸能人・著名人（プライバシー配慮）
- 根拠薄弱な情報

## JSON形式（必須フィールド）

```json
{
  "target_name": "企業名（説明）",
  "scale": "company/individual/other",
  "period": "YYYY-YYYY",
  "story_summary": "要約（2-3文）",
  "before_state": "絶頂・慢心/停滞・閉塞/混乱・カオス/成長痛/どん底・危機/安定・平和",
  "trigger_type": "外部ショック/内部崩壊/意図的決断/偶発・出会い",
  "action_type": "攻める・挑戦/守る・維持/捨てる・撤退/耐える・潜伏/対話・融合/刷新・破壊/逃げる・放置/分散・スピンオフ",
  "after_state": "上記と同じ選択肢",
  "before_hex": "乾/坤/震/巽/坎/離/艮/兌",
  "trigger_hex": "同上",
  "action_hex": "同上",
  "after_hex": "同上",
  "pattern_type": "Shock_Recovery/Hubris_Collapse/Pivot_Success/Endurance/Slow_Decline/Steady_Growth",
  "outcome": "Success/Failure/Mixed",
  "free_tags": ["#タグ1", "#タグ2"],
  "source_type": "news",
  "credibility_rank": "A",
  "logic_memo": "八卦変化の解説"
}
```

## 重複防止ポリシー（2026-01-02制定）

**絶対ルール**: 同一`target_name`+`period`の組み合わせは1件のみ
- ✓ OK: 「日本マクドナルド (2014-2017)」と「日本マクドナルド (2016-2019)」→異なる期間なので両方保持
- ✗ NG: 「日本マクドナルド (2014-2017)」が2件→重複削除対象

**実装**: JSON作成前に必ず既存DBをチェック
```python
# 重複チェック例
import json
existing = []
with open("data/raw/cases.jsonl") as f:
    for line in f:
        case = json.loads(line)
        existing.append((case["target_name"], case.get("period", "N/A")))

# 新規ケースが既存にないか確認
new_key = ("企業名", "2020-2024")
if new_key in existing:
    print(f"⚠ 重複: {new_key}")
```

## 八卦バランス収集戦略（2026-01-02現在）

**最優先収集対象**:
- `before_hex`: 離(不足227)、兌(不足215)、巽(不足215)
- `action_hex`: 震(不足285)、坎(不足252)、坤(不足128)
- `after_hex`: 艮(不足219)、巽(不足215)、震(不足204)

**収集抑制対象**（過剰）:
- `before_hex`: 坤(過剰427)、震(過剰112)、坎(過剰119)
- `action_hex`: 乾(過剰656)、艮(過剰136)
- `trigger_hex`: 離(過剰805)、坎(過剰378)

## 進捗追跡

現在のバッチ番号は `data/import/` 内の最新ファイルを確認。
完了後は `/changelog` で記録。
