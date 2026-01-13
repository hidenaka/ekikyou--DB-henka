---
name: llm-debate
description: Claude + Codexで議論。「LLMディベート」「議論して」「debate」「批評して」で発動。
---

# LLM Debate Skill

Claude・Codex(GPT-5.2)の2者で**忖度なしの厳格な批評**を行う。

## 特徴
- **忖度禁止**: Codexはプロフェッショナルな査読者として、論理の穴・前提の誤りを遠慮なく指摘
- **断言重視**: 「〜かもしれない」ではなく「〜である」と明確に主張
- **建設的批判**: 否定だけでなく代替案・修正点を提示

## 発動キーワード
`LLMディベート` `debate` `批評して` `Codexに聞いて`

## 事前条件
このスキルを使用するには以下の条件を満たす必要があります：

1. **gitリポジトリであること** - Codex CLIはセキュリティ上、gitリポジトリ内でのみ動作します
   - 確認方法: `git status` でエラーが出ないこと
   - 未初期化の場合: `git init` を実行してください

**事前条件を満たさない場合、スキルは実行されません。**

## ワークフロー
1. Claude が意見を表明
2. Codex に**厳格な批評**を依頼
3. Claude が統合分析・まとめ

## 実行
```bash
node scripts/llm-debate.js --topic "{議題}" --claude-opinion "{意見}" --context-file "{コンテキストファイル}"
```

## オプション
| オプション | 説明 |
|-----------|------|
| `--context` `-x` | 短いコンテキスト（直接指定） |
| `--context-file` | 長いコンテキスト用ファイルパス（検索結果等を保存して渡す） |

## 出力先
`debate/llm-debate-{YYYYMMDD}-{topic}/`

## 詳細
- スクリプト仕様: `scripts/llm-debate.js` 参照
- タイムアウト: Codex 300秒
