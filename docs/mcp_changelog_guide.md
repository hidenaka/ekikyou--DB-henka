# MCP変更履歴ガイド

## 基本ルール

| 状況 | 作成するもの |
|------|-------------|
| タスク完了 | `変更履歴_YYYY-MM-DD_連番` |
| 方向転換 | `方向転換_YYYY-MM-DD` + 方向性ログ更新 |
| ロールバック | `ロールバック_YYYY-MM-DD` |

## 方針の判定

**status=ACTIVE のみが現在有効**。ARCHIVEDは過去の記録。

## 検索コマンド

```python
# 索引（まずここを見る）
mcp__memory__open_nodes({"names": ["索引"]})

# 現在の方針
mcp__memory__open_nodes({"names": ["方向性ログ"]})

# 履歴検索
mcp__memory__search_nodes({"query": "変更履歴"})
```

## 方向転換の手順

1. 方向性ログに `status: ARCHIVED` を追加
2. 方向転換エンティティを作成
3. 方向性ログの内容を新方針に更新し `status: ACTIVE` を維持
