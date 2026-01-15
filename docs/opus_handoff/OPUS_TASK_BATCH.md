# Claude Opus 用 追加指示（10ブロック運用）

## 目的
`data/reference/iching_texts_ctext_legge_ja.json` の **modern_ja を10ブロックずつ**作成し、  
最終的に64卦すべての翻訳を完了させる。

## 参照ルール（必読）
`docs/opus_handoff/refs/`
- `language_rulebook.md`（最重要）
- `killer_words_analysis.md`
- `voice_personality.md`
- `hook_patterns.md`
- `evidence_block.md`

## 「ブロック」の定義
1つの `modern_ja` が **1ブロック**。
対象は以下の全て:
- `judgment` / `tuan` / `xiang`
- `lines.*.modern_ja`
- `lines.*.xiang.modern_ja`
- `appendix[*].modern_ja`

## 進め方（10ブロックずつ）
1) `data/reference/iching_texts_ctext_legge_ja.json` を開く  
2) **未入力のmodern_ja** を **番号順に探索**  
   - 卦番号昇順 → judgment → tuan → xiang → lines(1→6→use) → line.xiang → appendix  
3) 未入力の上から **10ブロックだけ** 翻訳  
4) JSONを上書き保存  
5) 次のバッチへ

## 翻訳ルール（最重要）
- **基本トーン**: 寄り添い × 温かい言い切り
- **3層構造**: 肯定・安心 → 代弁・言語化 → 提案・選択肢  
  - 1ブロック1〜3文  
  - 長さは **40〜120字目安**
- **敬語/文体**: です/ます調  
  - 命令口調NG（「〜しなさい」「〜すべき」「〜しましょう」）
  - 疑問形で終えない
- **ペルソナ寄り**: ユウタ/ミサキ双方に届く中立語彙  
  - 「決断/迷い/タイミング/本当の自分/内なる声」など  
  - 極端にキャリアorスピリチュアルに寄せない
- **禁止語**（出力に使わない）:
  - 「易経」「卦」「爻」「変爻」「運勢」「占い」「宿命」「当たる/外れる」
- **言い換え**:
  - 「易経」→「3000年前の知恵」
  - 「卦」→「パターン」
  - 「爻」→「段階/フェーズ」
  - 「変爻」→「変化のポイント」
- **強断定は避ける**: 「必ず」「絶対」などは使わない
- **事実追加は禁止**: データ・数字・新しい比喩は追加しない（翻訳のみ）

## 出力の推奨形式（任意）
更新内容を **10ブロック分だけ** まとめて提示する場合は以下形式:
```json
[
  {"path": "hexagrams.6.judgment.modern_ja", "value": "..."},
  {"path": "hexagrams.6.lines.1.modern_ja", "value": "..."}
]
```

## 品質チェック（毎バッチ）
- 禁止語が混入していないか
- 「必ず」「絶対」「〜しましょう」が入っていないか
- 40〜120字の範囲に収まっているか
- 3層構造が崩れていないか
