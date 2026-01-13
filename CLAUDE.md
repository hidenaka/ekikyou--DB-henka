# CLAUDE.md

## 重要: エージェント運用ルール

**役割**: マネージャー/オーケストレーター（実装は委託）

1. **実装は絶対に自分でやらない** → 全てsubagent/Task agentに委託
2. **タスクは超細分化** → 1タスク = 1明確なアウトプット
3. **PDCAサイクル構築**:
   - Plan: TodoWriteでタスク一覧化
   - Do: Task agentに委託して実行
   - Check: 結果を確認・検証
   - Act: 問題あれば修正タスクを追加

**タスク委託テンプレート**:
```
Task agent (Bash): スクリプト実行・検証
Task agent (Explore): 調査・分析
Task agent (general-purpose): 複雑な実装
```

**中断防止**:
- 各タスク完了時にTodoWrite更新
- MCPメモリに進捗記録
- 次のタスクを明示的に開始

---

## 必須: MCPメモリ

**セッション開始時**: `mcp__memory__open_nodes({"names": ["索引"]})`
**タスク完了時**: 変更履歴をMCPメモリに記録（`/changelog`）
**方針確認**: `mcp__memory__open_nodes({"names": ["方向性ログ"]})` → status=ACTIVEのみ有効

---

## データ品質改善ワークフロー（進行中）

**コンテキスト圧縮後の再開方法**:
```bash
# 1. 状態確認スクリプト実行
python3 scripts/workflow_resume.py

# 2. MCPメモリでワークフロー確認
mcp__memory__open_nodes(["データ品質改善ワークフロー_2026-01"])
```

**フェーズ一覧**:
| Phase | 目標 | 状態 |
|-------|------|------|
| A | main_domain補完 90%以上 | ✅完了 (88.1%) |
| B | 国際事例700件以上 | 進行中 (249件) |
| C | ソースURL 80%以上 | ✅完了 (91.3%) |
| D | 歴史事例1,600件以上 | 未着手 (1,142件) |

**スクリプト**:
- `scripts/workflow_resume.py` - 状態確認
- `scripts/enrich_main_domain.py` - Phase A
- `scripts/enrich_sources.py` - Phase C

**バッチ命名規則**:
- Phase B: `batch_international_XXX_NNN.json`
- Phase D: `batch_historical_DECADE_NNN.json`

---

## プロジェクト概要

易経（八卦）を用いた変化・遷移のデータベース。企業・個人・家族・国家などの実例を、八卦（乾/坤/震/巽/坎/離/艮/兌）とパターン分類で構造化。

**目標**: 10,000事例（現在約5,500件）
**方針**: 実データのみ（ニュース・公式資料ベース）

## 基本コマンド

```bash
# 事例追加
python3 scripts/add_batch.py data/import/xxx.json

# 検証
python3 scripts/validate_cases.py

# 診断エンジン
python3 scripts/diagnostic_engine.py
```

## データ場所

- **主データ**: `data/raw/cases.jsonl`
- **インポート**: `data/import/`

## 詳細ドキュメント

- `.docs/schema-details.md` - スキーマ・Enum値
- `.docs/diagnostic-tools.md` - 診断ツール詳細
- `.docs/architecture.md` - アーキテクチャ設計
- `docs/schema_v3.md` - 公式スキーマ仕様

---

## メインハブ連携

- **司令塔**: `/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/20260109AI易経事業`
- **作業完了時**: 「メインハブで進捗報告してください」とユーザーに伝える
- **詳細**: 必要時のみ `.docs/hub_sync.md` を参照

---

## 🔍 LLM Debate（Codex批評）

**重要な判断時には、Codex (GPT-5.2) に批評を依頼すること。**

### 必須発動ケース
- 新しいアーキテクチャや設計パターンを提案する時
- 重要なビジネス戦略・方針を立案する時
- ユーザーから「批評して」「これでいいか確認して」と言われた時

### 発動しないケース
- 単純なコード修正・リファクタリング
- ファイル操作・配置変更のみ
- 質問への回答のみ

### 実行方法
```bash
node .claude/skills/llm-debate/scripts/llm-debate.js \
  --topic "{議題}" \
  --claude-opinion "{自分の提案や意見}"
```

**出力先**: `debate/llm-debate-{日付}-{議題}/`

