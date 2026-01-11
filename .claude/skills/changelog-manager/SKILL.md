---
name: changelog-manager
description: タスク完了時にMCPメモリへ変更履歴を記録する。ファイル作成・編集後に使用。
---

# 変更履歴記録

## いつ使う
- ファイル作成/編集後
- 重要な決定後
- 方向転換時

## 記録内容
エンティティ名: `変更履歴_YYYY-MM-DD_連番`
タイプ: `changelog`
必須: timestamp, action, reason, previous_state, reversible

## 方向転換時
1. 古い方向性ログに `status: ARCHIVED` を追加
2. 新しい `方向転換_YYYY-MM-DD` エンティティを作成
3. 方向性ログを更新して `status: ACTIVE` を維持

## 検索
- 索引: `mcp__memory__open_nodes({"names": ["索引"]})`
- 方針: `mcp__memory__open_nodes({"names": ["方向性ログ"]})`
